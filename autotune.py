#!/usr/bin/env python3
"""
Finds the performance flags a machine wants, by measuring rather than guessing.

    ./autotune.py                      # tune this machine, ~10-20 minutes
    ./autotune.py --target create2     # tune for salt mining, a different kernel
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

Climbing one flag at a time has a known blind spot: two flags whose product is
what matters. From (-S 16, -G 128), the winning (-S 8, -G 256) is two
coordinate moves away and each of them loses on its own, so a pass structure
alone declares a peak one ridge-step short of the top. After the passes settle,
the coupled pairs (-S, -G) and (-i, -I) are probed diagonally -- one stepped
up while the other steps down -- and if that wins, the climb starts again from
it.

How it measures
---------------
profanity2 is watched live rather than run for a fixed slice of wall clock.
The clock starts at the first speed line, so a cold kernel compile -- over a
minute for the largest strips, a couple of seconds once the driver has it
cached -- costs its own time and nothing else, instead of silently eating the
measurement window. The first seconds after that are dropped while clocks ramp
and profanity2's own speed average fills.

Then the watching lasts only as long as the answer needs. A configuration
whose every sample sits far below the incumbent is cut off early -- most
neighbours a climb probes are exactly this, and no amount of watching turns
them into winners. One that lands within noise of the incumbent is watched for
up to twice the normal duration, because that is the one measurement where the
extra seconds change the conclusion. A configuration that prints rarely --
enormous batches take whole seconds per round -- is watched for up to three
times the normal duration rather than being mistaken for one that does not run.
And one that never speaks, or falls silent -- a refused combination, a crash,
a compile or a driver that hangs -- is written off with whatever reason it
managed to print, rather than scored on the samples it produced before dying.

What it prints
--------------
The flags to use, the worker constants to match, and a matrix showing, for each
flag, the chosen value against its neighbours on either side -- which is the
evidence that the answer is a peak and not a plateau or a slope the search
stopped early on. A neighbour that measures better by more than --threshold is
called out rather than quietly folded in: that is a real miss. One ahead by less
is reported as level with the chosen value, because a difference the search was
told not to chase is not a difference it should then complain about.

Last comes the chosen flags measured against every hashing target -- account,
contract, CREATE2 -- because those are three different kernels with three
different speeds, and the number a spec sheet wants is usually not the one the
tune was aimed at. The tune itself aims at the account kernel unless --target
says otherwise; a CREATE2 search iterates salts rather than curve points,
ignores --variants and two-level inversion entirely, and so both searches and
settles faster under --target create2 than the account flags would suggest.

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
import signal
import subprocess
import sys
import threading
import time

# The generator point, which is a valid public key and so a valid -z. Benchmark
# mode scores nothing, so which key it advances makes no difference to speed.
SEED_PUBLIC_KEY = (
    "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
    "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"
)

SPEED = re.compile(r"Total:\s*([\d.]+)\s*([KMGT]?)H/s")
UNITS = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}

# PROFANITY_SPEEDSAMPLES in Dispatcher.hpp: the speed profanity2 prints is a
# moving average over its last this-many rounds, so until that many have gone
# by it is averaging over a window that is still filling -- and the rounds that
# fill it are the ones that also carry start-up in their timing. A settle
# measured in seconds does not cover it: a launch of a hundred million work
# items takes the best part of a second, so twenty of them is twenty seconds,
# and a five second settle would score a card on numbers it had not finished
# computing. Both conditions have to hold before a sample counts.
SPEED_WINDOW = 20

# What a measurement aims the kernel at. The scorer stays "benchmark" -- it
# scores nothing and prints only speed -- but the target picks the hashing
# kernel, and account, contract, and CREATE2 are three different kernels with
# three different speeds. A CREATE2 benchmark needs a factory and an init code
# hash to shape its template; which ones makes no difference to the speed, so
# these are the zero address and the keccak256 of nothing.
TARGETS = {
    "account": [],
    "contract": ["--contract"],
    "create2": ["--create2", "0" * 40,
                "--init-code-hash",
                "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"],
}


class Config(dict):
    """One point in the search space, as the flags it stands for."""

    KEYS = ("variants", "inverse_size", "inverse_multiple", "inverse_strip",
            "inverse_group", "work", "rounds")

    def canonical(self):
        """
        The values that reach profanity2, and only those.

        The group is dead weight while two-level inversion is off, and the
        local work size is pinned by the group while it is on. Two
        configurations that differ only in a field the flags never carry are
        the same run, and identity should say so rather than measure it twice.
        """
        strip, group = self.inversion()
        return (self["variants"], self["inverse_size"], self["inverse_multiple"],
                strip, group, 0 if strip else self["work"], self["rounds"])

    def __hash__(self):
        return hash(self.canonical())

    def __eq__(self, other):
        if isinstance(other, Config):
            return self.canonical() == other.canonical()
        return NotImplemented

    def replace(self, key, value):
        out = Config(self)
        out[key] = value
        return out

    def argv(self, binary, cpu, target):
        strip, group = self.inversion()
        args = [
            binary, "--benchmark",
            "-i", str(self["inverse_size"]),
            "-I", str(self["inverse_multiple"]),
            "-S", str(strip),
            "-G", str(group),
            "-V", str(self["variants"]),
            "-R", str(self["rounds"]),
        ]
        # Two-level inversion pins the local size to the group, so -w is only
        # a question when it is off.
        if strip == 0:
            args += ["-w", str(self["work"])]
        args += TARGETS[target]
        # A CREATE2 search iterates public salts rather than points from a
        # seed, so it takes no key; everything else requires one.
        if target != "create2":
            args += ["-z", SEED_PUBLIC_KEY]
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

# The pairs that trade against each other through a shared product, which is
# what the diagonal probe after each descent is for.
PAIRS = (("inverse_strip", "inverse_group"),
         ("inverse_size", "inverse_multiple"))


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


def window(samples, first, settle):
    """
    Where the real measurement starts, and the samples from there on.

    Two things have to be true before a printed speed means anything, and
    neither implies the other. The settle covers clocks ramping. The window
    covers profanity2's own moving average, which reports a partly filled
    average until SPEED_WINDOW rounds have gone by -- and a configuration
    whose launches are enormous prints once a second, so it can clear a five
    second settle while still averaging four rounds out of twenty. That
    number is not a slow speed, it is an arithmetic artefact, and it reads
    high: a CPU measured 2.3 GH/s that way.

    Returns (None, []) while either condition is still outstanding, which is
    the caller's cue to keep watching rather than to score anything.
    """
    if first is None or len(samples) <= SPEED_WINDOW:
        return None, []
    origin = max(samples[SPEED_WINDOW][0], first + settle)
    return origin, [rate for when, rate in samples if when >= origin]


def percentile(samples):
    """The p95, which is the run's answer -- see "Reading the numbers"."""
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


class Watch:
    """
    A running profanity2, its speed ticker turned into timestamped samples.

    The ticker goes to stderr as carriage-returned rewrites of a single line,
    so it is read in chunks, split at the rewrites, and parsed as it arrives.
    Both streams are also kept as text, because a run that refuses to start
    says why on one or the other: profanity2's own complaints go to stdout,
    while an OpenCL driver that rejects a kernel writes its build log to
    stderr, mixed in among the speed lines. Both pipes must be drained
    regardless -- a full pipe would stall the miner and the stall would then
    be measured as speed.
    """

    KEEP = 262144

    def __init__(self, argv):
        self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        self._lock = threading.Lock()
        self._samples = []
        self._chatter = []
        self._noise = []
        self._threads = [threading.Thread(target=self._speeds, daemon=True),
                         threading.Thread(target=self._lines, daemon=True)]
        for thread in self._threads:
            thread.start()

    def _speeds(self):
        carry = ""
        while True:
            try:
                chunk = self.proc.stderr.read1(65536)
            except (ValueError, OSError):
                return
            if not chunk:
                return
            now = time.monotonic()
            text = carry + chunk.decode(errors="replace")
            edge = max(text.rfind("\r"), text.rfind("\n"))
            if edge < 0:
                carry = text[-4096:]
                continue
            done, carry = text[:edge + 1], text[edge + 1:]
            found = [(now, float(m.group(1)) * UNITS[m.group(2).upper()])
                     for m in SPEED.finditer(done)]
            with self._lock:
                if found:
                    self._samples.extend(found)
                # Everything that is not the ticker, kept for the failure
                # path: a driver that will not build a kernel explains itself
                # here and nowhere else.
                if sum(len(c) for c in self._noise) < self.KEEP:
                    self._noise.append(SPEED.sub("", done))

    def _lines(self):
        while True:
            try:
                chunk = self.proc.stdout.read1(65536)
            except (ValueError, OSError):
                return
            if not chunk:
                return
            with self._lock:
                if sum(len(c) for c in self._chatter) < self.KEEP:
                    self._chatter.append(chunk.decode(errors="replace"))

    def samples(self):
        with self._lock:
            return list(self._samples)

    def chatter(self):
        """Everything the run said, both streams, for the failure path."""
        with self._lock:
            return "".join(self._chatter) + "".join(self._noise)

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        for thread in self._threads:
            thread.join(2)
        self.proc.stdout.close()
        self.proc.stderr.close()


class Tuner:
    def __init__(self, args):
        self.args = args
        self.cache = {}
        self.why = {}
        self.detail = {}
        self.runs = 0
        self.began = time.time()

    def measure(self, config, beat=None, fresh=False, target=None):
        """
        Addresses per second, as the high percentile of what a run reported.

        None if profanity2 would not run this at all -- out of memory, or a
        combination it refuses -- which a search treats as simply worse than
        whatever it already has.

        `beat` is the rate this measurement is up against, and is what makes
        the watching adaptive in both directions: a configuration whose every
        sample sits far under it is cut off early, and one that lands within
        noise of it is watched for up to twice as long, because that is the
        one measurement where the extra seconds change the conclusion. With no
        `beat` -- the incumbent itself, and re-measurements -- the run always
        goes full length, so anything a comparison ever leans on was measured
        properly.

        `fresh` measures again rather than answering from the cache.

        `target` aims at a different hashing kernel than the one being tuned
        -- the closing per-target table is the caller -- and the search itself
        always measures the target it was asked to tune.
        """
        target = target or self.args.target
        key = (config, target)
        if key in self.cache and not fresh:
            return self.cache[key]

        if not valid(config):
            self.cache[key] = None
            self.why[key] = "ruled out"
            return None

        self.runs += 1
        request = float(self.args.duration)
        settle = self.args.settle
        bar = None if beat is None else beat * (1.0 + self.args.threshold)

        try:
            watch = Watch(config.argv(self.args.binary, self.args.cpu, target))
        except OSError as error:
            print(f"    cannot run {self.args.binary}: {error}", file=sys.stderr)
            sys.exit(1)

        began = time.monotonic()
        first = None
        died = stalled = shortened = stretched = False
        note = ""
        want = request
        seen, freshest, gap = 0, None, 0.0

        try:
            while True:
                time.sleep(0.25)
                died = watch.proc.poll() is not None
                samples = watch.samples()
                if first is None and samples:
                    first = samples[0][0]

                if first is None:
                    if died:
                        break
                    if time.monotonic() - began > self.args.warmup:
                        note = f"no speed line within --warmup ({self.args.warmup}s)"
                        break
                    continue
                if died:
                    break

                now = time.monotonic()
                if len(samples) != seen:
                    if freshest is not None:
                        gap = max(gap, now - freshest)
                    seen, freshest = len(samples), now

                origin, kept = window(samples, first, settle)
                if origin is None:
                    # Still filling the speed window. A launch big enough to
                    # take a second means twenty of them take twenty, so this
                    # waits far longer than the settle before calling a
                    # configuration one that cannot be measured at all.
                    # Generous on purpose: a configuration with many rounds
                    # per launch can take seconds to print once, and twenty of
                    # those is a minute or two before any of it counts. Giving
                    # up early here does not save time, it loses the number.
                    if now - first > settle + 15.0 * request:
                        note = "prints too rarely to fill the speed window"
                        break
                    continue
                spent = now - origin

                # A run that stops printing has stopped mining -- a wedged
                # driver, most often -- and what it printed before wedging is
                # not a speed this machine can use. Silence is judged against
                # the run's own cadence, so a configuration that prints every
                # ten seconds because its rounds are enormous is not mistaken
                # for one that has died: only a quiet spell out of proportion
                # to anything it has done before, and never shorter than the
                # full measurement itself, is called a stall.
                if now - freshest > max(float(request), 4.0 * gap):
                    stalled = True
                    note = f"went quiet: no speed line for {now - freshest:.0f}s"
                    break

                # The hard ceiling, however rarely this prints. Below it, the
                # early cut needs every sample far under the bar -- the maximum
                # is the optimistic reading, and 8% is more climbing than any
                # sample does once the settle has passed.
                if spent >= 3.0 * request:
                    break
                if (bar is not None and len(kept) >= 5
                        and spent >= max(4.0, request / 4.0)
                        and max(kept) * 1.08 < bar):
                    shortened = True
                    break
                if spent < want:
                    continue
                # Finishing also takes a live ticker: a run that has not
                # spoken for longer than its own cadence explains is not done,
                # it is suspect, and the stall rule above -- not a scoring --
                # is what settles it.
                if now - freshest > max(2.0, 4.0 * gap):
                    continue
                if len(kept) < 8:
                    continue
                # What counts as "close" is not the threshold: at a threshold
                # deliberately below the noise floor that band collapses, and
                # the measurements that most need the extra seconds would be
                # the ones that stop getting them. A percent is the width of
                # the region where a decision is in doubt whatever the
                # threshold is set to.
                doubt = max(self.args.threshold, 0.01)
                if (bar is not None and not stretched
                        and abs(percentile(kept) - bar) <= beat * doubt):
                    stretched = True
                    want = 2.0 * request
                    continue
                break
        finally:
            watch.stop()

        _, kept = window(watch.samples(), first, settle)

        label = config.flags() if target == self.args.target \
            else f"[{target}] {config.flags()}"

        # A run that exited on its own did not run: benchmark mode only ever
        # stops by being stopped, so an exit is a refusal or a crash whatever
        # it managed to print first. A stall is the same verdict reached more
        # slowly.
        if died or stalled or len(kept) < 3:
            note = note or "no speed reported"
            said = [line.strip() for line in watch.chatter().splitlines()
                    if line.strip()]
            for line in said:
                low = line.lower()
                if ("error" in low or "failed" in low or "invalid" in low
                        or "out of" in low or "abort" in low):
                    note = line[:70]
                    break
            else:
                # Nothing named itself an error, so the last thing it managed
                # to say is the most informative thing there is. Better than
                # "no speed reported", which says only that we were watching.
                if said:
                    note = f"{note}; last said: {said[-1][:60]}"
            print(f"    {label}  --  {note}")
            self.cache[key] = None
            self.why[key] = "would not run"
            return None

        rate = percentile(kept)
        elapsed = time.monotonic() - began
        self.cache[key] = rate
        self.detail[key] = {"samples": len(kept), "seconds": round(elapsed, 1)}

        tag = f"{len(kept)} samples, {elapsed:.0f}s"
        if shortened:
            tag += ", cut short"
        print(f"    {label:58} {human(rate)}  ({tag})")
        return rate

    def reason(self, config):
        """Why a configuration has no rate under the tuned target, if it has none."""
        return self.why.get((config, self.args.target))

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

        Both directions are tried from the starting value, unless the first
        already improved: a win moving right puts the peak to the right of the
        start, and the value on the start's left is then a value the climb has
        already argued against. Rungs the arithmetic rules out are stepped
        over rather than treated as walls -- they say nothing about speed --
        but a rung that failed to run ends the walk, because past a memory
        wall is more memory.
        """
        ladder = LADDERS[key]
        if config[key] not in ladder:
            return config

        best, bestRate = config, self.measure(config)
        at = ladder.index(config[key])
        moved = False

        for step in (1, -1):
            if moved:
                break
            i = at
            while 0 <= i + step < len(ladder):
                i += step
                candidate = best.replace(key, ladder[i])
                rate = self.measure(candidate, beat=bestRate)
                if rate is None and self.reason(candidate) == "ruled out":
                    continue
                if not self.better(rate, bestRate):
                    break
                best, bestRate, moved = candidate, rate, True

        return best

    def descend(self, config):
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

    def polish(self, config):
        """
        The diagonal probe of the coupled pairs, or None if nothing won.

        A climb moves one flag at a time, and for two flags that trade
        against each other through a product that is the one move set that
        cannot follow the ridge: from (16, 128) both coordinate moves lose,
        while (8, 256) -- the same 2048 points per inverse, differently cut
        -- measured 9% better. Stepping the pair together is at most four
        runs per pair, most of them cut early or already cached, for the one
        miss the pass structure is structurally blind to.
        """
        best, bestRate = config, self.measure(config)
        won = False

        for a, b in PAIRS:
            if not (tunable(best, a) and tunable(best, b)):
                continue
            if best[a] not in LADDERS[a] or best[b] not in LADDERS[b]:
                continue
            for da, db in ((1, -1), (-1, 1)):
                ia = LADDERS[a].index(best[a]) + da
                ib = LADDERS[b].index(best[b]) + db
                if not (0 <= ia < len(LADDERS[a]) and 0 <= ib < len(LADDERS[b])):
                    continue
                candidate = best.replace(a, LADDERS[a][ia]).replace(b, LADDERS[b][ib])
                rate = self.measure(candidate, beat=bestRate)
                if self.better(rate, bestRate):
                    print(f"    the pair moved together: {a} {best[a]} -> {candidate[a]},"
                          f" {b} {best[b]} -> {candidate[b]}, {(rate / bestRate - 1) * 100:+.1f}%")
                    best, bestRate, won = candidate, rate, True

        return best if won else None

    def crossover(self, config):
        """
        The other side of the two-level inversion switch, tuned enough to be
        a fair comparison, or None if this side is still better.

        A strip of zero is not a smaller strip -- it is the feature off, and
        the flags that matter change with it: with it on the group decides
        the work group size and -w is ignored, with it off -w is the whole
        question. A ladder cannot express that, and a climb that starts in
        the middle and wins its first step upward never steps down to zero
        at all, so a card that wants the feature off would never be told so.
        That is not hypothetical: the fork's own README puts two-level
        inversion at -62% on a GTX 1070.

        Whichever side is not in use is given a climb along its own flag
        before being judged, because comparing a tuned configuration against
        an untuned one answers a question nobody asked.
        """
        best, bestRate = config, self.measure(config)

        if config["inverse_strip"] == 0:
            middle = [s for s in LADDERS["inverse_strip"] if s]
            if not middle:
                return None
            other = best.replace("inverse_strip", middle[len(middle) // 2])
            other = self.climb(other, "inverse_group")
            other = self.climb(other, "inverse_strip")
        else:
            other = best.replace("inverse_strip", 0)
            other = self.climb(other, "work")

        rate = self.measure(other)
        if not self.better(rate, bestRate):
            return None

        state = "off" if other["inverse_strip"] == 0 else f"on at -S {other['inverse_strip']}"
        print(f"    two-level inversion {state} wins by"
              f" {(rate / bestRate - 1) * 100:+.1f}%; searching again from it")
        return other

    def run(self, config):
        """
        Coordinate passes until they settle, then the ridge probe and the
        two-level inversion crossover, and the passes again if either moved.
        Twice around is rare and three unheard of, so three is where it stops.
        """
        for _ in range(3):
            config = self.descend(config)

            print("\n  ridge check: the coupled pairs, stepped together")
            moved = self.polish(config)
            if moved is not None:
                config = moved
                continue
            print("    nothing moved; the peak holds across the diagonals")

            print("\n  crossover check: two-level inversion on the other setting")
            flipped = self.crossover(config)
            if flipped is None:
                print("    the current setting holds")
                return config
            config = flipped

        print("\n  still moving after three rounds; keeping the last")
        return config

    def confirm(self, champion, suspects):
        """
        Re-measures the champion and every challenger, fresh and interleaved,
        and says whether a challenger survives it.

        A neighbour beating the chosen value by less than --threshold is what
        the threshold is there to ignore: the pass loop would have taken
        anything larger, so what surfaces here is by construction close to
        the noise. Noise is made of interference and drift, and the
        interleaving -- champion, challengers, champion, challengers -- is
        aimed at the drift: a card that has been warming for an hour is the
        card both sides measure on, so no one wins by having been measured in
        a cooler minute. Each side's score is the best of its runs, for the
        usual reason: everything else on the machine only subtracts.
        """
        print("\n  re-measuring the contenders, interleaved, to see if this survives")

        contenders = [champion] + [champion.replace(key, value)
                                   for key, value, _ in suspects]
        scores = {c: [] for c in contenders}
        for _ in range(2):
            for contender in contenders:
                rate = self.measure(contender, fresh=True)
                scores[contender].append(0.0 if rate is None else rate)
                label = "champion" if contender == champion else "challenger"
                print(f"      {label:12} {contender.flags():52} {human(rate)}")

        best = max(scores[champion])
        winner, winnerRate = None, best
        for contender in contenders[1:]:
            rate = max(scores[contender])
            if rate > winnerRate:
                winner, winnerRate = contender, rate

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
                rate = self.measure(neighbour, beat=best)
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
                why = self.reason(config.replace(key, value)) or "would not run"
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

    # A SIGTERM -- timeout(1), a supervisor, a closing terminal -- should still
    # stop the profanity2 being measured, which only happens if it unwinds as
    # an exception through measure()'s finally rather than killing this process
    # mid-instruction and orphaning a miner that runs forever.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    parser = argparse.ArgumentParser(
        description="Measures the performance flags a machine wants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--binary", default="./profanity2.x64")
    parser.add_argument("--cpu", action="store_true",
                        help="tune a CPU OpenCL device, for trying this out without a GPU")
    parser.add_argument("--target", choices=tuple(TARGETS), default="account",
                        help="the kernel to tune against; account, contract, and CREATE2"
                             " are three different kernels, and a CREATE2 search --"
                             " salts, not points -- ignores -V and two-level inversion"
                             " entirely, so it both searches and settles faster"
                             " [default: account]")
    parser.add_argument("--duration", type=int, default=20,
                        help="target seconds of clean measurement per configuration; close"
                             " calls get up to double, clear losers much less [default: 20]")
    parser.add_argument("--warmup", type=int, default=180,
                        help="seconds allowed for compiling and seeding before the first"
                             " speed line; the largest kernels compile cold for over a"
                             " minute and seed for another, and cost two seconds once"
                             " the driver has them [default: 180]")
    parser.add_argument("--settle", type=float, default=None,
                        help="seconds dropped after the first speed line, while clocks ramp"
                             " and the speed average fills [default: a quarter of"
                             " --duration, kept between 2 and 6]")
    parser.add_argument("--passes", type=int, default=4,
                        help="how many times to sweep every flag before giving up [default: 4]")
    parser.add_argument("--threshold", type=float, default=0.001,
                        help="fractional gain worth moving for. Set below the noise floor"
                             " on purpose: a gain wrongly taken costs nothing, while one"
                             " wrongly refused is performance left on the card forever."
                             " Anything within a percent of a decision is re-measured at"
                             " double length regardless, and every measurement is cached,"
                             " so a search cannot oscillate on its own noise [default:"
                             " 0.001]")
    parser.add_argument("--retries", type=int, default=1,
                        help="times to re-measure and search again when a neighbour wins [default: 1]")
    parser.add_argument("--quick", action="store_true",
                        help="short measurements and small sizes, to check this script runs")
    parser.add_argument("--json", metavar="FILE",
                        help="also write the result, and every measurement taken, here")
    args = parser.parse_args()

    if args.quick:
        args.duration, args.warmup, args.passes = 8, 45, 2
        LADDERS["inverse_multiple"] = [64, 128, 256, 512]
        LADDERS["inverse_size"] = [64, 128, 255]
        LADDERS["inverse_strip"] = [0, 8, 16]
        LADDERS["inverse_group"] = [64, 128]
        LADDERS["work"] = [8, 16, 32]
        LADDERS["rounds"] = [1, 2, 4]

    if args.target == "create2":
        # A CREATE2 search has no points: no automorphisms to take variants
        # of (the dispatcher pins them to one), no inversion and so nothing
        # for a two-level split to split, and no 64 bytes of VRAM per point
        # holding the batch size down. What is left to tune is the launch --
        # its size, its local size, and its rounds -- so the dead flags are
        # pinned rather than measured, and the batch ladder reaches the
        # larger sizes a pointless search can afford.
        LADDERS["variants"] = [1]
        LADDERS["inverse_strip"] = [0]
        if not args.quick:
            LADDERS["inverse_size"] = [255]
            LADDERS["inverse_multiple"] = LADDERS["inverse_multiple"] + [262144, 524288]

    if args.settle is None:
        args.settle = max(2.0, min(6.0, args.duration / 4.0))

    found = args.binary if os.sep in args.binary else shutil.which(args.binary)
    if not found or not os.access(found, os.X_OK):
        print(f"cannot run {args.binary} -- build it first, or pass --binary", file=sys.stderr)
        return 1

    start = Config({
        "variants": LADDERS["variants"][-1],
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
    print(f"  watching each configuration for {args.duration}s after a {args.settle:g}s"
          f" settle, from its first")
    print(f"  speed line; compiling and seeding are allowed {args.warmup}s and cost"
          f" nothing extra")
    print(f"  tuning against the {args.target} kernel")
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

    # The chosen flags against the two kernels the tune was not aimed at,
    # measured once each, full length. They are three different kernels and
    # three different numbers, and a spec table wants all three.
    todo = [target for target in TARGETS if (best, target) not in tuner.cache]
    if todo:
        print("\n  measuring the chosen flags against the other kernels")
        for target in todo:
            tuner.measure(best, target=target)
    speeds = {target: tuner.cache.get((best, target)) for target in TARGETS}

    print("\n" + "=" * 72)
    print("  use these")
    print("=" * 72)
    print(f"\n  profanity2 {best.flags()} <mode> -z YOUR_PUBLIC_KEY\n")

    strip, group = best.inversion()
    if args.target == "account":
        print("  and in profanity_worker/search.py:\n")
        print(f"    INVERSE_MULTIPLE = {best['inverse_multiple']}")
        print(f"    INVERSE_STRIP    = {strip}")
        print(f"    INVERSE_GROUP    = {group}")
        print(f"    VARIANTS         = {best['variants']}")
        print(f"    ROUNDS           = {best['rounds']}")
        if best["inverse_size"] != 255:
            print(f"\n  -i {best['inverse_size']} has no constant in the worker; it passes"
                  f"\n  profanity2's default of 255. Add it to Job.argv if this holds up.")
        print()

    print("  what these flags measure, by target:\n")
    for target, rate in speeds.items():
        note = "   <-- tuned for this" if target == args.target else ""
        print(f"    {target:10} {human(rate):>14}{note}")
    if args.target != "create2":
        print("\n  The CREATE2 row is what these flags do there, not what the card can do:")
        print("  that search ignores -V and two-level inversion and usually wants a much")
        print("  larger -I. Hunting salts seriously is worth its own --target create2 run.")
    else:
        print("\n  The account and contract rows are these flags -- one variant, no")
        print("  two-level inversion -- on kernels that profit from both, so they are")
        print("  floors, not capabilities. Their own --target runs give the real rows.")

    elapsed = time.time() - tuner.began
    print(f"\n  {tuner.runs} configurations measured in {elapsed / 60:.1f} minutes")

    if args.json:
        with open(args.json, "w") as out:
            json.dump({
                "best": {k: best[k] for k in Config.KEYS},
                "flags": best.flags(),
                "target": args.target,
                "rate": tuner.cache.get((best, args.target)),
                "speeds": speeds,
                "neighbours": [
                    {"flag": k, "value": v, "rate": r, "chosen": c, "within_noise": t}
                    for k, v, r, c, t in rows
                ],
                "measured": [
                    {"config": dict(c), "target": t, "rate": r,
                     **tuner.detail.get((c, t), {})}
                    for (c, t), r in tuner.cache.items()
                ],
            }, out, indent=2)
        print(f"  written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
