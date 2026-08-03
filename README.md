# profanity2

The absolute fastest GPU vanity address generator for Ethereum. Describe the address you want and it searches billions of candidates per second until one matches.

Tell your friends it's to optimize gas with zero bytes, but really it's to flex on the [Vanity leaderboard](https://dune.com/aurelien/vanity).

![Screenshot](/img/screenshot.png?raw=true "Wow! That's a lot of zeros!")

This repository is a performance-focused fork of [1inch/profanity2](https://github.com/1inch/profanity2), which is itself the safe rework of the original `profanity`.\
Upstream's safety model is intact, and this fork rebuilds a highly optimized search loop around what modern GPUs are actually fast at: an RTX 5090 searches **5.6 GH/s** (**3.5x** faster on the same hardware).

Supports finding leading zeros (`0x000000000000...`), arbitrary prefixes/suffixes (`0xdeadbeef...`), a number of zero bytes (gas optimized), a contract landing on a chosen prefix, and more.\
This fork also adds CREATE2 salt mining, floating pattern masks, streamed results above a score bar, and an autotuner that finds the right flags for whichever card it's running on.

## Safe by design

profanity2 never generates, holds, or prints Ethereum private keys.\
It is perfectly safe to run on hardware you don't own or trust, such as renting cheap GPUs on [vast.ai](https://vast.ai) or [Salad](https://salad.com).

The original `profanity` had a [critical vulnerability](https://blog.1inch.io/a-vulnerability-disclosed-in-profanity-an-ethereum-vanity-address-tool): weak randomness let attackers recover private keys from the addresses it made. profanity2's answer, which this fork keeps unchanged, is to take the secret out of the program entirely. You give it a **public** key with `-z`, it searches over offsets from that key, and what it prints is a scalar to **combine with your own private key**. This result is otherwise worthless to anyone who doesn't hold your half.

A CREATE2 search carries no secret: the address is determined by the factory, the init code, and a salt anyone may pick. To protect against front-running, you can restrict generated salts to be used only from your wallet.

## Best performance

All measured on an RTX 5090; see the [Benchmarks](#benchmarks) section for other cards.

- **Six addresses per EC point (+96%)** via the secp256k1 curve's automorphisms: Elliptic-Curve cryptography is slow compared to deriving additional points from an existing public key. This fork checks all automorphic groups for each point, which only need a keccak hash and scoring.
- **Two-level Montgomery batch inversion (+27%)**: a work group cooperates on one modular inverse instead of each work item computing its own, amortizing the most expensive operation in the loop. Opt-in with `-S` and `-G` as it depends the most on recent GPU architectures — see [Compromises](#compromises).
- **Inline PTX multiprecision on NVIDIA (+12%)**: OpenCL C cannot name the carry flag, so portable multiprecision code detects every carry by comparison. NVIDIA's OpenCL frontend accepts inline PTX, which can. The kernel detects the compiler and uses a PTX implementation of its two innermost routines to speedup modular multiplication.
- **Specialized keccak (+4%)**: the first and final rounds are truncated depending on what the program actually hashes (account, contract, CREATE2), as most of the state is known-zero, so generic rounds waste work proving it.
- **Fused kernels, reduced launches (+3%)**: point iteration, automorphic transforms, hashing, and scoring all run as a single kernel instead of a chain, keeping intermediate state out of global memory. You can opt-in to run several point iterations within the same kernel to further amortize launches.
- **Faster start-up (-72%)**: initial seeding is under a second even for 33M points, kernels are compiled only for the specific scorer and target actually in use, and the compiled-kernel cache is reused across identical cards, so you can start searching addresses instantly on a fresh machine.

## Features

**Pattern modes** — every mode scores candidates, and the search prints the best it has seen (or, with `--min-score`, everything above a bar you set):

| mode | finds |
| --- | --- |
| `--matching <mask>` | an exact hex pattern; non-hex characters are wildcards. A 40-character mask is anchored where written, a shorter one floats anywhere in the address |
| `--leading <c>` | as many leading repetitions of one character as possible |
| `--zeros`, `--letters`, `--numbers` | as many characters of that class as possible, anywhere |
| `--zero-bytes` | as many `00` bytes as possible — addresses that are cheap as calldata |
| `--range`, `--leading-range` | characters inside a hex range you set with `-m`/`-M` |
| `--mirror` | an address mirrored around its center |
| `--leading-doubles` | leading pairs of identical characters |

**Reporting** — by default only a result beating the best so far is printed,
which is what you want when hunting the best available. When you know exactly
what you want, `--min-score` pins the bar instead: every address at or above it
is streamed for as long as the program runs, which is what a filter of your own
(EIP-55 capitalisation, say) needs on its input.

**Targets** — every mode above works against an account address, against the
address of the account's first deployed contract (`--contract`), and against a
CREATE2 deployment (`--create2` + `--init-code-hash`, with `--caller` to pin
the salt's first 20 bytes the way front-running-proof factories require). The
CREATE2 search covers the same ground as
[create2crunch](https://github.com/0age/create2crunch), with all of the scoring
modes above on top.

**Six addresses per point** (`--variants`) — secp256k1 has exactly six
automorphisms, maps that turn one point into another with no point arithmetic:
the point itself, its negation, its two images under the curve's endomorphism,
and their negations. `--variants 6` scores all six addresses per point
addition, paying only a keccak and a cheap modular operation for each extra
one. This nearly doubles throughput on an RTX 4090 and is the single biggest
win in the fork.

**Practicalities** — multiple GPUs are used automatically (`-s` skips one);
`--cpu` runs on a CPU OpenCL runtime like PoCL for machines with no usable GPU;
a Docker image runs on rented machines with nothing to install; and
`./autotune.py` measures its way to the right performance flags for your card.

### New capabilities

- `--matching` masks that float: a mask shorter than 40 characters is looked
  for anywhere in the address (upstream masks matched only where written), and
  padding one out with wildcards anchors it — prefix, suffix, or both ends.
- `--min-score` streamed reporting, replacing upstream's `--exact` — it works
  with every scoring mode, not just masks.
- `--create2` salt mining with `--init-code-hash` and `--caller`.
- `--variants`, `-S`/`-G`, `-R` performance controls.
- `--cpu` for machines without a GPU.
- `./autotune.py`, which measures the right flags instead of guessing them.
- A test suite (`tests/`) proving the specialised kernels bit-exact against
  portable ones and a host-side big-integer reference, plus an A/B benchmark
  image (`bench/`) that runs two revisions alternately on one GPU so a change
  can be judged without trusting two rented machines to be equally fast.

## Compromises

The speed above was bought with specific trade-offs. None of them make the
program wrong on other hardware — correctness is covered by tests everywhere —
but they say where it is fast.

**It is tuned on recent NVIDIA cards.** The measuring and tuning behind this
fork happened on RTX 3060/4090/5090 machines. The inline PTX path is
NVIDIA-only by nature (other vendors take the portable path automatically, and
`make CDEFINES=-DPROFANITY_NO_PTX` forces it everywhere if a driver ever
misbehaves). AMD and Intel GPUs build and run — upstream's ROCm fixes are
included — but nobody has tuned for them, and the flags that make a 4090 fly
can make an older card slower than the defaults: two-level inversion at
`-S 8 -G 128` measured **+38%** on an RTX 4090, **−29%** on an RTX 3060 and
**−62%** on a GTX 1070. This is why it is opt-in and why `autotune.py` exists.

**Defaults are portable, not fast.** Out of the box the program runs the safe
configuration everywhere. The headline numbers need the tuned flags, and the
right ones vary by card by more than any default could cover — run
`./autotune.py` once per machine (~20–40 minutes) and use what it prints.

**Peak throughput costs start-up time.** Large inversion strips make the OpenCL
compile long: on the 4090, `-S 32` compiles in 73s cold against 34s with
two-level inversion off, and searches 3.2% faster than `-S 16` (50s). The
break-even is around thirteen minutes of searching — for long hunts take the
bigger strip, for short ones the smaller. Warm starts are ~2s regardless,
because the driver caches compiled kernels.

**Five results in six need one extra step.** Above `--variants 1`, most
results come from a transformed point, and the printed scalar needs a negation
and/or a multiplication by λ after being added to your seed key. Each result
line says which (see [Deriving the private key](#4-derive-the-final-private-key));
skip the step and you hold a valid key to the *wrong* address, so always check
the derived address against the printed one. At the default `--variants 1`
nothing changes and nothing can be gotten wrong.

**It cuts corners a general keccak cannot.** The hash kernels compute exactly
what an Ethereum address needs and not a bit more — truncated final round,
sparse first round per absorb layout. `tests/` proves them equivalent to the
portable implementations on every input the tests can think of; run those tests
on the card you intend to mine with.

## Benchmarks

| GPU | account | `--contract` | `--create2` | recommended flags |
| --- | --- | --- | --- | --- |
| RTX 5090 | 5.6 GH/s | 3.9 GH/s | 6.3 GH/s | `-I 65536 -S 32 -G 128 -V 6 -R 8` |
| RTX 4090 | 3.8 GH/s | 2.7 GH/s | 4.3 GH/s | `-I 16384 -S 32 -G 512 -V 6` |
| RTX 5080 | 2.9 GH/s | 2.0 GH/s | 3.8 GH/s | defaults |
| RTX 5070 | 876 MH/s | 610 MH/s | 2.4 GH/s | `-I 393216` |
| RX 7900 XTX | 592 MH/s | 410 MH/s | 1.6 GH/s | defaults |
| RTX 3070 | 536 MH/s | 380 MH/s | 1.5 GH/s | `-I 262144` |
| Apple M4 Max<br/>(40-core GPU) | 467 MH/s | 330 MH/s | 1.3 GH/s | defaults |
| RTX 3060 | 330 MH/s | 230 MH/s | 900 MH/s | defaults |
| RX 6700 XT | 240 MH/s | 170 MH/s | 650 MH/s | defaults |
| GTX 1070 | 225 MH/s | 160 MH/s | 620 MH/s | `-I 196608` |
| Apple M1<br/>(8-core GPU) | 60 MH/s | 42 MH/s | 170 MH/s | defaults |

Figures above are per card, and multiple GPUs are used automatically and scale near-linearly.\
For example, **45 GH/s** was measured on an eight-card RTX 5090 machine.

Keep two-level inversion (`-S`/`-G`) off on cards older than Ada: it costs an RTX 3060 -29% and a GTX 1070 -62%.\
Before a long search, run `./autotune.py` to find your optimal flags.

## Quick start

### 1. Make a seed key

Generate a fresh keypair with openssl (remove the `04` prefix from the public
key; keep the private key to yourself):

```bash
openssl ecparam -genkey -name secp256k1 -text -noout -outform DER | xxd -p -c 1000 | sed 's/41534e31204f49443a20736563703235366b310a30740201010420/Private Key: /' | sed 's/a00706052b8104000aa144034200/\'$'\nPublic Key: /'
```

Or derive the public key from a private key you already have:

```bash
openssl ec -inform DER -text -noout -in <(cat <(echo -n "302e0201010420") <(echo -n "PRIVATE_KEY_HEX") <(echo -n "a00706052b8104000a") | xxd -r -p) 2>/dev/null | tail -6 | head -5 | sed 's/[ :]//g' | tr -d '\n' && echo
```

The 128-character public key is the mandatory `-z` argument (`--create2`
searches don't want one). The examples below abbreviate it:

```bash
export PUBLIC_KEY="HEX_PUBLIC_KEY_128_CHARS_LONG"
```

### 2. Build, or use Docker

```bash
# macOS (Intel and Apple Silicon) — OpenCL ships with the system
xcode-select --install && make

# Ubuntu / Linux — see docs/BUILD_UBUNTU.md for driver setup
sudo apt install -y build-essential opencl-headers ocl-icd-opencl-dev clinfo
make

# Windows — see docs/BUILD_WINDOWS.md (MSYS2/MinGW-w64 or a vendor SDK)
```

Both produce `profanity2.x64` in the repository root. Or skip building: CI
publishes an image with the OpenCL runtime baked in, so a rented GPU machine
needs nothing installed —

```bash
docker run --rm --gpus all ghcr.io/codesensus-org/profanity2:latest --matching dead -z $PUBLIC_KEY
```

[docs/VASTAI.md](docs/VASTAI.md) walks through renting a GPU on vast.ai and
running the image there. Since the tool only ever sees your public key, the
machine does not have to be yours.

### 3. Run a search

```bash
# "dead" anywhere in the address
./profanity2.x64 --matching dead -z $PUBLIC_KEY

# Prefix: pad the mask to 40 characters to anchor it
./profanity2.x64 --matching deadXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX -z $PUBLIC_KEY

# Suffix, or both ends at once
./profanity2.x64 --matching XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX999999 -z $PUBLIC_KEY
./profanity2.x64 --matching 1111XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX2222 -z $PUBLIC_KEY

# As many leading zeros as possible, printed as the record improves
./profanity2.x64 --leading 0 -z $PUBLIC_KEY

# Every match streamed, not just record-breakers: a 4-char mask scores 4
./profanity2.x64 --matching dead --min-score 4 -z $PUBLIC_KEY

# The account whose first deployed contract gets the vanity address
./profanity2.x64 --contract --leading 0 -z $PUBLIC_KEY

# A CREATE2 salt: no key, no -z, the factory and init code decide the address
export FACTORY="0x4e59b44847b379578588920ca78fbf26c0b4956c"
export INIT_CODE_HASH="0x21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f"
./profanity2.x64 --create2 $FACTORY --init-code-hash $INIT_CODE_HASH --zero-bytes
./profanity2.x64 --create2 $FACTORY --init-code-hash $INIT_CODE_HASH \
    --caller 0xYOUR_ADDRESS --matching deadXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX --min-score 4
```

Every additional fixed character multiplies the expected search time by 16.

For CREATE2: the init code hash is the keccak256 of the **init code** — the
bytes handed to CREATE2, constructor arguments included — not of the deployed
runtime code. Get it wrong and the search runs at full speed toward addresses
that will never exist. The printed salt is the full 32 bytes to deploy with;
verifying it needs one keccak (`keccak256(0xff ++ factory ++ salt ++
init_code_hash)`, last 20 bytes) — do that before using it.

### 4. Derive the final private key

A result line prints a scalar. Add it to the private key behind your `-z`,
modulo the curve order — **never in an online calculator**:

```python
n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
"%064x" % ((SEED_PRIVATE_KEY + PRINTED_SCALAR) % n)
```

The result must be used as a **64-character** hex string — `"%064x"` pads the
leading zeros that about 1 sum in 16 needs and that `hex()` and `bc` silently
drop.

Under `--variants` the word before the scalar says which transform the sum needs. Here is the exact command for each. Pick the line whose
comment matches the printed word:

```python
n   = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
lam = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72  # λ
s   = SEED_PRIVATE_KEY + PRINTED_SCALAR

"%064x" % (s % n)                # Private
"%064x" % (-s % n)               # PrivateNegated
"%064x" % (lam * s % n)          # PrivateLambda
"%064x" % (-lam * s % n)         # PrivateLambdaNegated
"%064x" % (lam * lam * s % n)    # PrivateLambda2
"%064x" % (-lam * lam * s % n)   # PrivateLambda2Negated
```

These must run in Python (or another language with big integers and a
non-negative `%`) — the negated lines rely on Python's `%` mapping negative
values into `[0, n)`.

Skipping the transform leaves a perfectly valid key to a different address, so
always verify: import the derived key into a wallet and check the address
matches the one on the result line before sending anything to it. At the
default `--variants 1` only `Private` ever appears.

## Tuning your machine

The performance flags vary by card by more than any default can cover, so
measure rather than guess:

```bash
./autotune.py                 # ~20–40 minutes, prints the flags to use
./autotune.py --cpu --quick   # a few minutes, to see what it does
```

It hill-climbs each flag along a ladder of sensible values and ends by printing
every chosen value against its neighbours — the evidence that the answer is a
peak rather than where the search happened to stop. Speeds are in addresses
per second, so `--variants` counts compare directly.

The flags it tunes, briefly (see `./profanity2.x64 --help` for everything):

- `-V, --variants <1–6>` — addresses scored per point addition.
- `-S`/`-G` — two-level inversion: points per work item, work items sharing
  one inverse. Opt-in because it is much faster on some cards and much slower
  on others; `-S` alone sets the compile time.
- `-i`/`-I` — inverse batch size and how many run in parallel; `-i * -I` is
  the points a round covers (keep `-I` small on `--cpu`, seeding scales with it).
- `-R, --rounds` — point additions per launch; amortizes launch overhead,
  which a big launch doesn't have. Measured within noise on a 4090.

Two tuning facts worth knowing before trusting a quick manual sweep: the `-G`
axis is **not monotonic** (on the 4090, 256 loses to both 128 and 512, so a
hill climb that steps past 128 stops short of the best value — autotune walks
that ladder in full), and `-S`/`-G` are not interchangeable ways to buy the
same batch size (`(16, 128)` and `(8, 256)` share an inverse across the same
2048 points and measure 12.6% apart).

## Tests

Correctness and micro-benchmarks live in `tests/` and need an OpenCL runtime
(PoCL is enough — they run on a CPU):

```bash
cd tests && make && cd ..
./tests/test_correctness_ptx.x64      # PTX vs portable multiprecision, bit-exact
./tests/test_keccak_equiv.x64         # sparse keccak vs the generic rounds
./tests/test_scoring.x64              # every scoring mode
./tests/bench_mod_mul_ptx.x64         # what the PTX path is worth in isolation
```

On non-NVIDIA devices the PTX tests report there was nothing to compare and
pass — run them on the card you intend to mine with. For comparing two
revisions of the program itself, [bench/](bench/README.md) builds an image that
runs both alternately on the same GPU.

## Usage

`./profanity2.x64 --help` documents every flag, including the device controls
(`-s` to skip a device, `-C`/`--cpu` for CPU OpenCL, `-n` to skip the kernel
cache) and the exact semantics of each scoring mode.

If you have run an older version before, delete the `cache-opencl.*` files (or
pass `--no-cache` once) so the OpenCL program is rebuilt with the new kernels.

## Credits and license

- Original `profanity` by Johan Gustafsson.
- `profanity2`, the safe-by-design rework, by [1inch Network](https://github.com/1inch/profanity2).
- The keccak final-round truncation technique comes from
  [0age's create2crunch](https://github.com/0age/create2crunch).

Always verify that a derived private key corresponds to the printed address by
importing it into a wallet of your choice before using it. This program cuts
corners for speed by design and, like any software, may contain bugs.

Licensed under the [MIT License](LICENSE).
