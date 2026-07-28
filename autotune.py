#!/usr/bin/env python3
"""
Finds the performance flags a machine wants, by measuring rather than guessing.

    ./autotune.py                      # tune this machine, ~20-40 minutes
    ./autotune.py --cpu --quick        # a smoke test where there is no GPU
    ./autotune.py --duration 40        # longer measurements, tighter numbers

Every flag profanity2 has that affects speed and nothing else is in here. What
each is worth varies by card by more than any default can cover -- two-level
inversion is +38% on an RTX 4090 and -62% on a GTX 1070 -- so the answer for a
machine is whatever that machine measures, and this is how to get it.

How it searches
---------------
Not a grid. A grid over these six flags is thousands of runs, and almost all of
them are far from anything worth measuring. Instead each flag is hill-climbed
along a ladder of sensible values: step to a neighbour, keep going while it
improves, stop when it does not. A pass does that for every flag in turn, and
passes repeat until one changes nothing, which is a coordinate-wise local
optimum. Every measurement is cached, so the revisiting a hill climb does is
free.

What it prints
--------------
The flags to use, the worker constants to match, and a matrix showing, for each
flag, the chosen value against its neighbours on either side -- which is the
evidence that the answer is a peak and not a plateau or a slope the search
stopped early on. A neighbour that measures better by more than --threshold is
called out rather than quietly folded in: that is a real miss. One ahead by less
is reported as level with the chosen value, because a difference the search was
told not to chase is not a difference it should then complain about.

Reading the numbers
-------------------
Speed is reported in addresses per second, so a run at --variants 6 and one at
--variants 1 are directly comparable -- six addresses off one point addition is
six addresses. The statistic is a high percentile rather than a mean, because
anything else sharing the machine can only ever subtract: the fastest samples
are the ones least interfered with, and so the truest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

# The generator point, which is a valid public key and so a valid -z. Benchmark
# mode scores nothing, so which key it advances makes no difference to speed.
SEED_PUBLIC_KEY = (
    "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
    "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"
)

SPEED = re.compile(r"Total:\s*([\d.]+)\s*([KMGT]?)H/s")
UNITS = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}

# PROFANITY_SPEEDSAMPLES in Dispatcher.hpp: how many rounds the reported speed
# is averaged over, and so how many samples in the window is still filling.
SPEED_WINDOW = 20


class Config(dict):
    """One point in the search space, as the flags it stands for."""

    KEYS = ("variants", "inverse_size", "inverse_multiple", "inverse_strip",
            "inverse_group", "work", "rounds")

    def __hash__(self):
        return hash(tuple(self[k] for k in self.KEYS))

    def replace(self, key, value):
        out = Config(self)
        out[key] = value
        return out

    def argv(self, binary, cpu):
        strip, group = self.inversion()
        args = [
            binary, "--benchmark",
            "-i", str(self["inverse_size"]),
            "-I", str(self["inverse_multiple"]),
            "-S", str(strip),
            "-G", str(group),
            "-V", str(self["variants"]),
            "-R", str(self["rounds"]),
            "-z", SEED_PUBLIC_KEY,
        ]
        # Two-level inversion pins the local size to the group, so -w is only
        # a question when it is off.
        if strip == 0:
            args += ["-w", str(self["work"])]
        if cpu:
            args += ["-C"]
        return args

    def inversion(self):
        """
        The (-S, -G) this stands for.

        A strip of zero is two-level inversion switched off, and the group is
        then meaningless -- profanity2 wants both zero or neither, so the group
        is held at whatever it was and simply not passed. That keeps the two
        ladders independent without letting a group setting make an "off"
        configuration look like a different one.
        """
        strip = self["inverse_strip"]
        return (strip, self["inverse_group"] if strip else 0)

    def flags(self):
        """Just the tuning, as it would be typed on a real search."""
        strip, group = self.inversion()
        out = ["-i", str(self["inverse_size"]), "-I", str(self["inverse_multiple"])]
        if strip:
            out += ["-S", str(strip), "-G", str(group)]
        else:
            out += ["-w", str(self["work"])]
        out += ["-V", str(self["variants"])]
        if self["rounds"] != 1:
            out += ["-R", str(self["rounds"])]
        return " ".join(out)


# The ladders, each ordered so that a step is a step. A hill climb only ever
# moves between neighbours here, so what is adjacent is what this says is.
LADDERS = {
    # The lever that matters most, and the only one bounded by mathematics
    # rather than by hardware: six is the order of the curve's automorphism
    # group, so there is no seventh value to try.
    "variants": [1, 2, 3, 4, 5, 6],

    # Points a work item inverts together. Higher amortizes the one modular
    # inverse over more points but holds more of them in scratch memory.
    "inverse_size": [64, 128, 170, 255, 340, 510],

    # How many such batches run at once, which is what fills the card. Bounded
    # by VRAM: a point costs 64 bytes, so -i * -I * 64 has to fit.
    "inverse_multiple": [2048, 4096, 8192, 16384, 32768, 65536, 131072],

    # Two-level inversion, as two ladders rather than one.
    #
    # These were a single ladder of (strip, group) pairs, which was a mistake:
    # flattening two axes onto one line means a step along it is sometimes a
    # step in strip and sometimes a step in group, so a climb that is winning
    # on one axis gets offered a move on the other, loses, and stops. On 4x
    # 4090 that left it at (16, 128) without ever trying a larger strip, which
    # was the direction that had just gained 5%.
    #
    # They are not interchangeable, which is the other reason to separate them:
    # (16, 128) and (8, 256) share a batch size of 2048 points per inverse and
    # measured 9% apart. A bigger strip costs registers; a bigger group costs a
    # scan iteration, its barriers, and local memory.
    #
    # A strip of zero is two-level inversion off, and the group is then unused.
    "inverse_strip": [0, 4, 8, 16, 32, 64],
    "inverse_group": [64, 128, 256, 512],

    # OpenCL local work size, when two-level inversion is not deciding it.
    "work": [32, 64, 128, 256, 512],

    # Point additions per launch. Saves the launch and nothing else, so it is
    # here for completeness and expected to sit at 1 on anything with a launch
    # worth millions of points.
    "rounds": [1, 2, 4, 8, 16],
}

# Tried in this order within a pass: what moves the number most goes first, so
# that later flags are tuned against a configuration already close to right.
ORDER = ["variants", "inverse_strip", "inverse_group", "inverse_multiple",
         "inverse_size", "work", "rounds"]


def valid(config):
    """Whether profanity2 would accept this, so a run is not wasted finding out."""
    strip, group = config.inversion()
    if strip == 0:
        return True
    if group & (group - 1):
        return False
    # -i * -I has to divide evenly into the two-level batch, or profanity2
    # refuses to start.
    return (config["inverse_size"] * config["inverse_multiple"]) % (strip * group) == 0


def tunable(config, key):
    """
    Whether `key` means anything in `config`.

    The local work size is decided by the group when two-level inversion is on,
    and the group means nothing when it is off. Measuring either where it does
    nothing would be measuring the noise and then believing it.
    """
    if key == "work":
        return config["inverse_strip"] == 0
    if key == "inverse_group":
        return config["inverse_strip"] != 0
    return True


class Tuner:
    def __init__(self, args):
        self.args = args
        self.cache = {}
        self.why = {}
        self.runs = 0
        self.began = time.time()

    def measure(self, config, fresh=False, duration=None):
        """
        Addresses per second, as the high percentile of what a run reported.

        None if profanity2 would not run this at all -- out of memory, or a
        combination it refuses -- which a search treats as simply worse than
        whatever it already has.

        `fresh` measures again rather than answering from the cache, and
        `duration` watches for longer than usual. Both are for settling a
        difference too small for the first measurement to have resolved.
        """
        if config in self.cache and not fresh:
            return self.cache[config]

        if not valid(config):
            self.cache[config] = None
            self.why[config] = "ruled out"
            return None

        self.runs += 1
        argv = config.argv(self.args.binary, self.args.cpu)

        try:
            found = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                # profanity2 runs until it is stopped, so the timeout is how
                # long to watch it for rather than a failure condition.
                timeout=self.args.warmup + (duration or self.args.duration),
            )
            output = found.stdout + found.stderr
        except subprocess.TimeoutExpired as expired:
            output = (expired.stdout or b"").decode(errors="replace") + \
                     (expired.stderr or b"").decode(errors="replace")
        except OSError as error:
            print(f"    cannot run {self.args.binary}: {error}", file=sys.stderr)
            sys.exit(1)

        samples = [float(m.group(1)) * UNITS[m.group(2).upper()]
                   for m in SPEED.finditer(output.replace("\r", "\n"))]

        # The opening samples are start-up: the speed profanity2 prints is a
        # moving average over its last PROFANITY_SPEEDSAMPLES rounds, so until
        # that many have gone by it is averaging over a partly empty window,
        # and the card is still climbing to clock besides.
        keep = samples[max(SPEED_WINDOW, len(samples) // 10):]

        if len(keep) < 5:
            note = "no speed reported"
            for line in output.replace("\r", "\n").splitlines():
                if "error" in line.lower() or "failed" in line.lower():
                    note = line.strip()[:70]
                    break
            print(f"    {config.flags()}  --  {note}")
            self.cache[config] = None
            self.why[config] = "would not run"
            return None

        keep.sort()
        rate = keep[min(len(keep) - 1, int(len(keep) * 0.95))]
        self.cache[config] = rate
        print(f"    {config.flags():58} {human(rate)}  ({len(keep)} samples)")
        return rate

    def better(self, a, b):
        """Whether a beats b by more than the noise floor is worth believing."""
        if a is None:
            return False
        if b is None:
            return True
        return a > b * (1.0 + self.args.threshold)

    def climb(self, config, key):
        """
        Walks `key` along its ladder from where it is, while that improves.

        Both directions are tried from the starting value, so a hill climb that
        began on the far side of the peak still finds it. Only neighbours are
        ever measured: a ladder of seven values costs three runs, not seven.
        """
        ladder = LADDERS[key]
        if config[key] not in ladder:
            return config

        best, bestRate = config, self.measure(config)
        at = ladder.index(config[key])

        for step in (1, -1):
            i = at
            while 0 <= i + step < len(ladder):
                i += step
                candidate = best.replace(key, ladder[i])
                rate = self.measure(candidate)
                if not self.better(rate, bestRate):
                    break
                best, bestRate = candidate, rate

        return best

    def run(self, config):
        for sweep in range(1, self.args.passes + 1):
            print(f"\n  pass {sweep}")
            before = Config(config)

            for key in ORDER:
                if not tunable(config, key):
                    continue
                config = self.climb(config, key)

            if config == before:
                print(f"\n  settled after {sweep} pass(es)")
                break
        else:
            print(f"\n  stopped at the {self.args.passes}-pass limit, still moving")

        return config

    def confirm(self, champion, suspects):
        """
        Re-measures a challenger and the champion, for longer, and says whether
        the challenger survives it.

        A neighbour beating the chosen value by less than --threshold is what
        the threshold is there to ignore: the pass loop would have taken
        anything larger, so what surfaces here is by construction close to the
        noise. Re-running the pair for longer is what tells a real miss from a
        gust, and it costs two runs rather than a whole pass.
        """
        longer = self.args.duration * 2
        print(f"\n  re-measuring the contenders at {longer}s to see if this survives")

        best = self.measure(champion, fresh=True, duration=longer)
        print(f"    {'champion':12} {champion.flags():54} {human(best)}")

        winner, winnerRate = None, best
        for key, value, _ in suspects:
            challenger = champion.replace(key, value)
            rate = self.measure(challenger, fresh=True, duration=longer)
            print(f"    {'challenger':12} {challenger.flags():54} {human(rate)}")
            if rate is not None and rate > winnerRate:
                winner, winnerRate = challenger, rate

        if winner is None:
            print("\n  the champion held. The difference was noise, and the answer stands.")
        else:
            print(f"\n  the challenger held at {(winnerRate / best - 1) * 100:+.1f}%."
                  f" Searching again from it.")

        return winner

    def matrix(self, config):
        """
        Each flag's chosen value against the neighbours it beat.

        This is the part worth reading. A search that reports only its answer
        is asking to be trusted; this shows the two measurements on either side
        of every value it picked, which is what makes the answer a peak.
        """
        best = self.measure(config)
        rows, suspect = [], []

        for key in ORDER:
            if not tunable(config, key):
                continue

            ladder = LADDERS[key]
            at = ladder.index(config[key])

            for i in (at - 1, at, at + 1):
                if not 0 <= i < len(ladder):
                    continue

                if i == at:
                    rows.append((key, ladder[i], best, True, False))
                    continue

                neighbour = config.replace(key, ladder[i])
                rate = self.measure(neighbour)
                tied = rate is not None and rate > best and not self.better(rate, best)
                rows.append((key, ladder[i], rate, False, tied))
                if self.better(rate, best):
                    suspect.append((key, ladder[i], rate))

        print("\n" + "=" * 72)
        print("  neighbours of the chosen configuration")
        print("=" * 72)
        print(f"  {'flag':18} {'value':>12} {'addresses/s':>14}   {'vs best':>8}")
        print("  " + "-" * 68)

        previous = None
        for key, value, rate, chosen, tied in rows:
            if previous is not None and key != previous:
                print()
            previous = key

            # A neighbour ahead by less than --threshold was deliberately not
            # acted on. Said as "below threshold" rather than "within noise",
            # because at a loose --threshold the gap can be large and calling
            # a measured 25% gap noise would be a lie about the measurement
            # rather than a statement about the setting.
            mark = " <-- chosen" if chosen else (" below threshold" if tied else "")
            if rate is None:
                why = self.why.get(config.replace(key, value), "would not run")
                print(f"  {key:18} {str(value):>12} {why:>14}   {'':>8}{mark}")
            else:
                delta = (rate / best - 1.0) * 100.0
                shown = "     best" if chosen else f"{delta:+7.1f}%"
                print(f"  {key:18} {str(value):>12} {human(rate):>14}   {shown}{mark}")

        print()
        if suspect:
            print(f"  NOT A PEAK: a neighbour beat the chosen value by more than the"
                  f" {self.args.threshold * 100:.0f}% threshold --")
            for key, value, rate in suspect:
                print(f"    {key} = {value} at {human(rate)}, {(rate / best - 1) * 100:+.1f}%")
            print("  The search did not converge -- this is larger than the threshold it")
            print("  settles on, so it is a real miss rather than noise, usually because")
            print("  --passes ran out before the flags stopped moving each other.")
        else:
            ruled = sum(1 for _, _, rate, chosen, _ in rows if rate is None and not chosen)
            ties = sum(1 for _, _, _, _, tied in rows if tied)
            if ties:
                print(f"  Every neighbour that could run is slower, or ahead by less than"
                      f" the {self.args.threshold * 100:g}%")
                print("  threshold and so left alone. This is a local peak at that threshold;")
                print("  lower --threshold to resolve the ones marked below it.")
            else:
                print("  Every neighbour that could run is slower. This is a local peak.")
            if ruled:
                print(f"  {ruled} neighbour(s) are ruled out by profanity2's own arithmetic --")
                print("  -G must be a power of two and -i * -I a multiple of -S * -G -- so")
                print("  they say nothing about speed either way.")

        return rows, suspect


def human(rate):
    if rate is None:
        return "-"
    for unit in ("", "K", "M", "G", "T"):
        if rate < 1000.0:
            return f"{rate:.3f} {unit}H/s"
        rate /= 1000.0
    return f"{rate:.3f} PH/s"


def main():
    # Line-buffered even when stdout is a file, so a run redirected to a log can
    # be followed while it happens rather than only read after it ends.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(
        description="Measures the performance flags a machine wants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--binary", default="./profanity2.x64")
    parser.add_argument("--cpu", action="store_true",
                        help="tune a CPU OpenCL device, for trying this out without a GPU")
    parser.add_argument("--duration", type=int, default=25,
                        help="seconds of measurement per configuration [default: 25]")
    parser.add_argument("--warmup", type=int, default=20,
                        help="seconds allowed for compiling and seeding [default: 20]")
    parser.add_argument("--passes", type=int, default=4,
                        help="how many times to sweep every flag before giving up [default: 4]")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="fractional gain worth moving for, so noise is not chased [default: 0.01]")
    parser.add_argument("--retries", type=int, default=1,
                        help="times to re-measure and search again when a neighbour wins [default: 1]")
    parser.add_argument("--quick", action="store_true",
                        help="short measurements and small sizes, to check this script runs")
    parser.add_argument("--json", metavar="FILE",
                        help="also write the result, and every measurement taken, here")
    args = parser.parse_args()

    if args.quick:
        args.duration, args.warmup, args.passes = 8, 12, 2
        LADDERS["inverse_multiple"] = [64, 128, 256, 512]
        LADDERS["inverse_size"] = [64, 128, 255]
        LADDERS["inverse_strip"] = [0, 8, 16]
        LADDERS["inverse_group"] = [64, 128]
        LADDERS["work"] = [8, 16, 32]
        LADDERS["rounds"] = [1, 2, 4]

    found = args.binary if os.sep in args.binary else shutil.which(args.binary)
    if not found or not os.access(found, os.X_OK):
        print(f"cannot run {args.binary} -- build it first, or pass --binary", file=sys.stderr)
        return 1

    start = Config({
        "variants": 6,
        "inverse_size": LADDERS["inverse_size"][len(LADDERS["inverse_size"]) // 2],
        "inverse_multiple": LADDERS["inverse_multiple"][len(LADDERS["inverse_multiple"]) // 2],
        # Mid-ladder on both, for the same reason as the others: a hill climb
        # explores both ways from where it starts, so beginning at an end means
        # one value that happens to lose can end the climb before it reaches
        # anything good.
        "inverse_strip": LADDERS["inverse_strip"][len(LADDERS["inverse_strip"]) // 2],
        "inverse_group": LADDERS["inverse_group"][len(LADDERS["inverse_group"]) // 2],
        "work": LADDERS["work"][1] if len(LADDERS["work"]) > 1 else LADDERS["work"][0],
        "rounds": 1,
    })

    print("=" * 72)
    print("  profanity2 autotune")
    print("=" * 72)
    print(f"  measuring {args.duration}s per configuration after {args.warmup}s of warm-up")
    print(f"  starting from: {start.flags()}")

    tuner = Tuner(args)
    best = tuner.run(start)

    # A neighbour that beat the chosen value is either a real miss or noise,
    # and re-measuring the pair for longer is what tells them apart. Only a
    # challenger that survives that is worth searching from again -- taking
    # every one on faith would have the search chase its own noise, which for
    # two values within a percent of each other does not terminate.
    for attempt in range(args.retries + 1):
        rows, suspects = tuner.matrix(best)
        if not suspects:
            break
        if attempt == args.retries:
            print(f"\n  leaving it here after {args.retries} retry(ies). Raise --retries,"
                  f"\n  or --duration so the measurements can resolve it.")
            break

        winner = tuner.confirm(best, suspects)
        if winner is None:
            break
        best = tuner.run(winner)

    elapsed = time.time() - tuner.began
    print("\n" + "=" * 72)
    print("  use these")
    print("=" * 72)
    print(f"\n  profanity2 {best.flags()} <mode> -z YOUR_PUBLIC_KEY\n")

    strip, group = best.inversion()
    print("  and in profanity_worker/search.py:\n")
    print(f"    INVERSE_MULTIPLE = {best['inverse_multiple']}")
    print(f"    INVERSE_STRIP    = {strip}")
    print(f"    INVERSE_GROUP    = {group}")
    print(f"    VARIANTS         = {best['variants']}")
    print(f"    ROUNDS           = {best['rounds']}")
    if best["inverse_size"] != 255:
        print(f"\n  -i {best['inverse_size']} has no constant in the worker; it passes"
              f"\n  profanity2's default of 255. Add it to Job.argv if this holds up.")

    print(f"\n  {tuner.runs} configurations measured in {elapsed / 60:.1f} minutes")

    if args.json:
        with open(args.json, "w") as out:
            json.dump({
                "best": {k: best[k] for k in Config.KEYS},
                "flags": best.flags(),
                "rate": tuner.cache.get(best),
                "neighbours": [
                    {"flag": k, "value": list(v) if isinstance(v, tuple) else v,
                     "rate": r, "chosen": c, "within_noise": t}
                    for k, v, r, c, t in rows
                ],
                "measured": [
                    {"config": {k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in dict(c).items()},
                     "rate": r}
                    for c, r in tuner.cache.items()
                ],
            }, out, indent=2)
        print(f"  written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
