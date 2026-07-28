# profanity2

Profanity is a high performance (probably the fastest!) vanity address generator for Ethereum. Create cool customized addresses that you never realized you needed! Recieve Ether in style! Wow!

![Screenshot](/img/screenshot.png?raw=true "Wow! That's a lot of zeros!")

# Important to know

A previous version of this project has a known critical issue due to a bad source of randomness. The issue enables attackers to recover private key from public key: https://blog.1inch.io/a-vulnerability-disclosed-in-profanity-an-ethereum-vanity-address-tool

This project "profanity2" was forked from the original project and modified to guarantee **safety by design**. This means source code of this project do not require any audits, but still guarantee safe usage.

Project "profanity2" is not generating key anymore, instead it adjusts user-provided public key until desired vanity address will be discovered. Users provide seed public key in form of 128-symbol hex string with `-z` parameter flag. Resulting private key should be used to be added to seed private key to achieve final private key of the desired vanity address (private keys are just 256-bit numbers). Running "profanity2" can even be outsourced to someone completely unreliable - it is still safe by design.

Note: when upgrading to a new version of profanity2, delete the `cache-opencl.*` files (or pass `--no-cache`) once so the OpenCL program is rebuilt with the new kernel.

## Getting public key for mandatory `-z` parameter

Generate private key and public key via openssl in terminal (remove prefix "04" from public key):
```bash
$ openssl ecparam -genkey -name secp256k1 -text -noout -outform DER | xxd -p -c 1000 | sed 's/41534e31204f49443a20736563703235366b310a30740201010420/Private Key: /' | sed 's/a00706052b8104000aa144034200/\'$'\nPublic Key: /'
```

Derive public key from existing private key via openssl in terminal (remove prefix "04" from public key):
```bash
$ openssl ec -inform DER -text -noout -in <(cat <(echo -n "302e0201010420") <(echo -n "PRIVATE_KEY_HEX") <(echo -n "a00706052b8104000a") | xxd -r -p) 2>/dev/null | tail -6 | head -5 | sed 's/[ :]//g' | tr -d '\n' && echo
```

## Adding private keys (never use online calculators!)

### Terminal:

Use private keys as 64-symbol hexadecimal string WITHOUT `0x` prefix:
```bash
(echo 'ibase=16;obase=10' && (echo '(PRIVATE_KEY_A + PRIVATE_KEY_B) % FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141' | tr '[:lower:]' '[:upper:]')) | bc
```

### Python

Use private keys as 64-symbol hexadecimal string WITH `0x` prefix:
```bash
$ python3
>>> "%064x" % ((PRIVATE_KEY_A + PRIVATE_KEY_B) % 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141)
```

### `--variants`: more addresses per point, and one more step

`--variants <1-6>` scores more than one address per point addition. Every point
is worth up to six that cost no point arithmetic to reach:

| `--variants` | also scores | costing |
| --- | --- | --- |
| 1 | `P`, the point itself | — |
| 2 | `-P = (x, -y)` | a modular subtraction |
| 3 | `psi(P) = (beta*x, y)` | a modular multiplication |
| 4 | `-psi(P)` | — |
| 5 | `psi^2(P) = (beta^2*x, y)` | one more multiplication |
| 6 | `-psi^2(P)` | — |

so each extra address costs a keccak where another point would cost a point
addition. Going from 1 to 6 measured **+96%** on an RTX 4090 and +145% on a CPU
device -- the gap between those two is the point of the warning below. A keccak
costs relatively more where 64-bit lanes are emulated, which is every GPU, so the
extra addresses are worth less there than a CPU measurement suggests.

**Six is the ceiling**, and a property of the curve rather than of this program.
Those six maps are the *automorphisms* of the curve — the maps `E -> E` that need
no point arithmetic — and secp256k1 is `y^2 = x^3 + 7`, so its j-invariant is
zero and its automorphism group is the sixth roots of unity, `{±1, ±w, ±w^2}`.
A group of order six has no seventh element, so there is no 7- or 8-way to find.
Anything further needs an endomorphism of degree above one, which is a point
addition by another name — and the batch inversion already amortizes those
better than this would.

Benchmark rather than assuming 6 wins. What the extra addresses are worth depends
on what a keccak costs on your card relative to the rest of the loop, and the
higher counts hold more live across the hash, where lost occupancy can take back
what the arithmetic saves.

Above 1, a search prints one of several words before its scalar, each saying what
to do to the sum after adding it to your seed key. Writing `s` for that sum:

| printed | private key of the address on the line | from |
| --- | --- | --- |
| `Private: 0x…` | `s` — the sum above, and nothing else | 1 |
| `PrivateNegated: 0x…` | `-s mod n` | 2 |
| `PrivateLambda: 0x…` | `lambda * s mod n` | 3 |
| `PrivateLambdaNegated: 0x…` | `-lambda * s mod n` | 4 |
| `PrivateLambda2: 0x…` | `lambda^2 * s mod n` | 5 |
| `PrivateLambda2Negated: 0x…` | `-lambda^2 * s mod n` | 6 |

`n` is `FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141`, the same
modulus the sum is already taken over, and `lambda` is
`5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72`. Since
`lambda^3 = 1 mod n`, `lambda^2` is also `lambda`'s inverse. In Python:

```bash
$ python3
>>> n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
>>> lam = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72
>>> s = (SEED_PRIVATE_KEY + PRINTED) % n
>>> "%064x" % (s % n)              # Private
>>> "%064x" % (-s % n)             # PrivateNegated
>>> "%064x" % (lam * s % n)        # PrivateLambda
>>> "%064x" % (-lam * s % n)       # PrivateLambdaNegated
>>> "%064x" % (lam * lam * s % n)  # PrivateLambda2
>>> "%064x" % (-lam * lam * s % n) # PrivateLambda2Negated
```

Written `-x % n` rather than `n - x` so that the one input for which they differ,
`x = 0`, gives 0 rather than `n`. Neither is a usable private key, but the first
is obviously not one. That input means the point is the identity, which cannot
have an address and which profanity2's point addition cannot represent, so it is
unreachable in both directions — this is tidiness and not a fix.

At `--variants 6` five results in six are one of the transformed kinds. Adding
the scalar and stopping, on such a line, leaves a private key that is perfectly
valid and belongs to a different address than the one printed; check the address
you derive against the one on the line before sending anything to it. At the
default of 1 the extra words never appear and there is nothing here to get wrong.

### Leading zeros

The combined private key must be used as a **64-symbol** hexadecimal string. Neither `bc` nor Python's `hex()` prints leading zeros, so whenever the sum has fewer than 64 symbols (about 1 chance in 16), pad it with leading zeros. The Python snippet above does this automatically thanks to `"%064x"`. Example:

```
PRIVATE_KEY_A =        0bc657b0af28b743c7f0d49c4de78efd47a5c8923dabfdef051fff5cdc7c30e7
PRIVATE_KEY_B =        0000f8ba428990fca1e618a252ac3614f5de19b20ff00c2ded57bfb6933830aa
sum (63 symbols):       bc7506af1b2484069d6ed3ea093c5123d83e2444d9c0a1cf277bf136fb46191
private key (padded):  0bc7506af1b2484069d6ed3ea093c5123d83e2444d9c0a1cf277bf136fb46191
```

# Tuning a machine

What the performance flags should be varies by card by more than any default can
cover, so measure rather than guess:

```bash
./autotune.py                 # ~20 minutes, prints the flags to use
./autotune.py --cpu --quick   # a few minutes, to see what it does
```

It hill-climbs each flag along a ladder of sensible values rather than sweeping a
grid, and ends by printing every chosen value against its neighbours on either
side -- which is the evidence that the answer is a peak rather than somewhere the
search happened to stop:

```
  flag                      value    addresses/s    vs best
  --------------------------------------------------------------------
  variants                      5     3.162 GH/s      -4.7%
  variants                      6     3.319 GH/s        best <-- chosen

  inverse_strip                 8     3.155 GH/s      -4.9%
  inverse_strip                16     3.319 GH/s        best <-- chosen
  inverse_strip                32       ruled out
```

A neighbour that measures better by more than `--threshold` is called out rather
than folded in: that means the search did not converge. One ahead by less is
reported as level, since a difference the search was told not to chase is not one
it should then complain about. `--duration` buys tighter measurements and
`--json` writes down every measurement it took.

Speeds are in addresses per second, so counts of `--variants` compare directly
against each other -- six addresses off one point addition really is six tried.

## Measured on an RTX 4090

Everything below was measured on rented RTX 4090 machines (Ada, driver 580),
so it is what one architecture wants and not a universal answer. Reproduce it on
yours with `./autotune.py` before believing any of it.

```
-i 255 -I 16384 -S 32 -G 512 -V 6        3.839 GH/s
```

### What the work since `main` is worth

Each row is the row above plus one change, so a step belongs to one thing:

| | addresses/s | this step | vs `main` |
| --- | --- | --- | --- |
| `main` | 1.361 GH/s | | baseline |
| + fused kernels, truncated keccak, seeding | 1.312 GH/s | -3.6% | -3.6% |
| + two-level inversion | 1.664 GH/s | +26.8% | +22.2% |
| + six addresses per point | 3.259 GH/s | +95.9% | **+139.5%** |

**2.39x overall.** Almost all of it is the endomorphism work; two-level inversion
is second. The structural row is a *loss* at one address per point: fusing put
the single-level inverse's 16 KB of private arrays into the same kernel as the
inlined keccak, and the occupancy that costs outweighs the traffic and the launch
it saves. It pays for itself only because it is what lets the rest sit where it
does.

### The two inversion axes

`-S` and `-G` are not interchangeable, and neither is well behaved. A grid at
`-i 255 -I 16384 -V 6`, against 2.723 GH/s with two-level off:

| S \ G | 64 | 128 | 256 | 512 |
| --- | --- | --- | --- | --- |
| 4 | 3.082 | 3.147 | 2.874 | 3.459 |
| 8 | 3.495 | 3.528 | 3.296 | 3.661 |
| 16 | 3.680 | 3.712 | 3.585 | 3.721 |
| **32** | 3.749 | 3.758 | 3.522 | **3.839** |
| 64 | 3.641 | 3.621 | 3.608 | n/a |

Two things to take from it.

**The group is not monotonic.** It dips at 256 and recovers at 512, at every
strip but the largest. A hill climb from 128 steps to 256, loses, and stops --
never seeing that 512 is the best of them by ~10%. `--threshold` will not save
you: the intermediate value really is worse. This is why `autotune.py` keeps the
group on a ladder short enough to walk in full.

**Batch size is not the whole story.** `(16, 128)` and `(8, 256)` both share one
inverse across 2048 points and measure 12.6% apart. A larger strip costs registers;
a larger group costs a scan iteration, its barriers, and local memory. The strip
is the cheaper way to buy the same amortization -- up to a point: it peaks at 32
and turns down at 64.

### Compile time, which is nearly all of start-up

Seeding is under a second even for the 33.4M points of `-i 255 -I 131072`.
Everything else in start-up is the OpenCL compile, and that tracks `-S` alone --
`-S 16` takes the same 50s at `-G 128` and at `-G 512`:

| `-S` | cold compile | throughput |
| --- | --- | --- |
| 0 (off) | 34s | 2.723 GH/s |
| 4 | 38s | 3.459 |
| 8 | 42s | 3.661 |
| 16 | 50s | 3.721 |
| 24 | 60s | - |
| 32 | 73s | 3.839 |
| 64 | 131s | 3.621 |

**Cold** is the word doing the work there. NVIDIA caches compiled kernels in
`~/.nv/ComputeCache` under the driver, which `-n` does not touch -- it only stops
profanity2 loading its own cache. A second run of the same flags starts in 2s
however large the strip. Measure this with the cache cleared or you will measure
nothing.

So `-S 32` searches 3.2% faster and starts 23s later than `-S 16`. They draw
level after about thirteen minutes of searching:

    T = (73 x 3.839 - 50 x 3.721) / (3.839 - 3.721) = 798s

Under that, 16 wins, and by a lot on anything short -- 31% at two minutes, and
everything at one, since 32 has not finished compiling. Over it, 32 wins by at
most 3.1% however long the run goes on. For a worker whose searches usually end
in seconds, 16 is the better bet.

### What did not matter

`--rounds` is within the noise here: 1, 2, 5, 10 and 20 span 3.319 to 3.350
GH/s, a 0.9% spread against 1.8% run-to-run variance on a single configuration.
It amortizes the kernel launch and nothing else -- a point's delta and previous
lambda still go to global memory and back every round -- and a launch of sixteen
million points has little launch overhead to amortize.

That 1.8% is worth remembering generally: it is larger than several of the
differences above, and larger than `autotune.py`'s 1% default threshold.

### On NVIDIA: the multiprecision arithmetic in inline PTX

Every carry in the multiprecision code is detected by comparison, because OpenCL
C has no way to name the flag the hardware sets for free. NVIDIA's OpenCL
frontend shares NVVM with CUDA and accepts inline PTX, which does, so on that
vendor `profanity.cl` compiles a second implementation of its two innermost
routines and uses that instead. Nothing needs to be passed to turn it on; the
kernel detects the compiler.

It is not something NVIDIA documents, so there is a way out if a driver ever
disagrees:

```bash
# Take the portable path on every device
make clean && make CDEFINES=-DPROFANITY_NO_PTX
```

Both paths are checked against each other and against a host-side big-integer
reference on every input the tests can think of:

```bash
cd tests && make && cd ..
./tests/test_correctness_ptx.x64      # bit-exact equivalence, both directions
./tests/bench_mod_mul_ptx.x64         # what it is worth, mod-mul in isolation
```

On anything that is not NVIDIA the two report that there was nothing to compare
and pass, so run them on the card you intend to mine with.

# Building

### macOS

OpenCL ships with the system, so only the Xcode Command Line Tools are needed:

```bash
xcode-select --install   # skip if already installed
make
```

This produces `profanity2.x64` in the repository root. Works on both Intel and
Apple Silicon (M1/M2/M3/M4) Macs.

### Ubuntu / Linux

See [docs/BUILD_UBUNTU.md](docs/BUILD_UBUNTU.md). In short:

```bash
sudo apt install -y build-essential opencl-headers ocl-icd-opencl-dev clinfo
make
```

plus the OpenCL runtime (driver) of your GPU vendor — the document covers
NVIDIA/AMD/Intel setup and common errors.

### Windows

See [docs/BUILD_WINDOWS.md](docs/BUILD_WINDOWS.md) — building with MSYS2/MinGW-w64
or against a vendor OpenCL SDK, plus troubleshooting for the most common errors
(`CL/cl.h: No such file or directory`, `CreateProcess(NULL, uname -s, ...) failed`,
empty device list, WSL2 limitations).

### Docker and rented GPUs

A prebuilt image with the OpenCL runtime is available, so nothing has to be installed on the machine that does the searching:

```bash
docker run --rm --gpus all ghcr.io/1inch/profanity2:latest --matching dead -z $PUBLIC_KEY
```

Since the tool only needs your public key, that machine does not have to be yours. See [docs/VASTAI.md](docs/VASTAI.md) for renting a GPU on vast.ai and running the image there with your own parameters.

# Usage
```
usage: ./profanity2 [OPTIONS]

  Mandatory args:
    -z                      Seed public key to start, add it's private key
                            to the "profanity2" resulting private key. Not
                            wanted by --create2, which searches over salts
                            rather than over keys.

  Reading a result:
    A search over keys prints the word Private before the scalar it found.
    Add it to the private key behind your -z, and that sum is the private
    key of the address on the line.

    Under --variants a search prints other words as well. Each means the
    same scalar added to the same seed key, with the sum then worked on.
    Writing s for that sum, n for the order of the curve and lambda for
    the scalar the curve's endomorphism multiplies by:

      1  Private                 s
      2  PrivateNegated         -s
      3  PrivateLambda           lambda * s
      4  PrivateLambdaNegated   -lambda * s
      5  PrivateLambda2          lambda^2 * s
      6  PrivateLambda2Negated  -lambda^2 * s

    all taken mod n, the number on the left being the lowest --variants
    that can print it. Where

      n      = 0xfffffffffffffffffffffffffffffffe
               baaedce6af48a03bbfd25e8cd0364141
      lambda = 0x5363ad4cc05c30e0a5261c028812645a
               122e22ea20816678df02967c1b23bd72

    At the default of 1 nothing but Private is ever printed, so anything
    reading these lines goes on reading them as it always has.

  Basic modes:
    --benchmark             Run without any scoring, a benchmark.
    --zeros                 Score on zeros anywhere in hash.
    --letters               Score on letters anywhere in hash.
    --numbers               Score on numbers anywhere in hash.
    --mirror                Score on mirroring from center.
    --leading-doubles       Score on hashes leading with hexadecimal pairs
    -b, --zero-bytes        Score on hashes containing the most zero bytes

  Modes with arguments:
    --leading <single hex>  Score on hashes leading with given hex character.
    --matching <hex mask>   Score on hashes matching given hex mask. Non-hex
                            characters (e.g. X) are wildcards, and the score is
                            the number of characters the mask pins down that
                            match. A mask 40 characters long is looked for where
                            it is written; a shorter one is looked for anywhere
                            in the address, so pad it out with wildcards to
                            anchor it.

  Reporting:
    -r, --min-score <score> Print every hash scoring this or better, for as
                            long as the program runs. Without it only a hash
                            that beats the best one so far is printed, and the
                            bar rises as the search goes: stumble on something
                            better than you asked for and every later address
                            that merely satisfies the request goes unreported.
                            Give this when you know what you want rather than
                            wanting the best available. A mask scores one for
                            each character it pins down, so --matching dead
                            wants --min-score 4.

  Advanced modes:
    --contract              Instead of account address, score the contract
                            address created by the account's zeroth transaction.
    --leading-range         Scores on hashes leading with characters within
                            given range.
    --range                 Scores on hashes having characters within given
                            range anywhere.

  CREATE2:
    --create2 <address>     Score the address a contract deployed with CREATE2
                            would land at, and search over salts rather than
                            over keys. Takes the address of the contract doing
                            the deploying. There is no private key anywhere in
                            such a search — what it finds is a salt, so -z is
                            neither wanted nor used, and there is nothing in
                            the run to keep secret.
    -k, --init-code-hash    The keccak256 of the init code being deployed, 64
                            hexadecimal characters. Required with --create2.
    -a, --caller <address>  Pin the salt's first 20 bytes to this address,
                            which is what a factory guarding against front-
                            running requires of whoever deploys through it.
                            Left out, those bytes are zero — which is what the
                            same factories take to mean anyone may deploy it.

    The address scored is the last 20 bytes of
      keccak256(0xff ++ create2 ++ salt ++ init-code-hash)
    and the salt printed beside every result is the whole 32 bytes to deploy
    with. Every scoring mode above works against it.

  Range:
    -m, --min <0-15>        Set range minimum (inclusive), 0 is '0' 15 is 'f'.
    -M, --max <0-15>        Set range maximum (inclusive), 0 is '0' 15 is 'f'.

  Device control:
    -s, --skip <index>      Skip device given by index.
    -n, --no-cache          Don't load cached pre-compiled version of kernel.
    -C, --cpu               Search on CPU devices instead of graphics cards,
                            which needs a CPU OpenCL runtime such as PoCL
                            installed. Orders of magnitude slower and meant for
                            machines with no usable GPU, and for trying a search
                            out before renting one; lower -I to keep start-up
                            from taking minutes.

  Tweaking:
    -V, --variants <1-6>    How many addresses to score per point addition.
                            [default = 1]

                            Every point is worth up to six addresses that
                            cost no point arithmetic to reach: itself, its
                            negation (x, -y), the two images of it under
                            the curve's endomorphism (b*x, y) and (b^2*x,
                            y), and the negations of those. A negation
                            costs a modular subtraction and an image a
                            modular multiplication, so each extra address
                            costs a keccak where another point would cost
                            a point addition.

                            Six is the ceiling, and not a limit of this
                            program: those six maps are the automorphisms
                            of the curve, and secp256k1 being y^2 = x^3 + 7
                            its automorphism group is the sixth roots of
                            unity. There is no seventh such map. Anything
                            further needs an endomorphism of degree above
                            one, which is a point addition by another name.

                            What the extra addresses are worth depends on
                            what a keccak costs on the card relative to the
                            rest of the loop, and the higher counts hold
                            more live across the hash, where lost occupancy
                            can take back what the arithmetic saves. So
                            benchmark it rather than assuming six wins.

                            Above 1, results arrive under the other words
                            listed in "Reading a result" above and want a
                            transform to reach the private key. At 6 that
                            is five results in six.
    -w, --work <size>       Set OpenCL local work size. [default = 64]
    -W, --work-max <size>   Set OpenCL maximum work size. [default = -i * -I]
    -i, --inverse-size      Set size of modular inverses to calculate in one
                            work item. [default = 255]
    -I, --inverse-multiple  Set how many above work items will run in
                            parallell. [default = 16384]
                            A --create2 search inverts nothing, but -i * -I is
                            still how many candidates a round covers.
    -S, --inverse-strip     Enable two-level inversion, with this many points
                            batched per work item. [default = 0, disabled]
    -G, --inverse-group     Work group size sharing a single inverse when
                            two-level inversion is enabled. Must be a power of
                            two. [default = 0, disabled]

  Two-level inversion:
    With -S and -G a work group cooperates on one modular inverse instead of
    each work item doing its own, which is much faster on some GPUs and much
    slower on others. Both switches must be given together, and
    -i * -I must be a multiple of -S * -G. Benchmark before using it:
      RTX 4090   -S 8 -G 128    +38% over the default
      RTX 3060   -S 8 -G 128    -29% over the default
      GTX 1070   -S 8 -G 128    -62% over the default

  Examples:
    ./profanity2 --leading f -z HEX_PUBLIC_KEY_128_CHARS_LONG
    ./profanity2 --matching dead -z HEX_PUBLIC_KEY_128_CHARS_LONG
    ./profanity2 --matching deadXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX -z HEX_PUBLIC_KEY_128_CHARS_LONG
    ./profanity2 --matching badXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXbad -z HEX_PUBLIC_KEY_128_CHARS_LONG
    ./profanity2 --matching 1337XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXc0de --min-score 8 -z HEX_PUBLIC_KEY_128_CHARS_LONG
    ./profanity2 --leading-range -m 0 -M 1 -z HEX_PUBLIC_KEY_128_CHARS_LONG
    ./profanity2 --leading-range -m 10 -M 12 -z HEX_PUBLIC_KEY_128_CHARS_LONG
    ./profanity2 --range -m 0 -M 1 -z HEX_PUBLIC_KEY_128_CHARS_LONG
    ./profanity2 --contract --leading 0 -z HEX_PUBLIC_KEY_128_CHARS_LONG
    ./profanity2 --create2 FACTORY_ADDRESS --init-code-hash INIT_CODE_HASH --zero-bytes
    ./profanity2 --create2 FACTORY_ADDRESS --init-code-hash INIT_CODE_HASH \
                 --caller YOUR_ADDRESS --matching deadXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX --min-score 4

  About:
    profanity2 is a vanity address generator for Ethereum that utilizes
    computing power from GPUs using OpenCL.

  Forked "profanity2":
    Author: 1inch Network <info@1inch.io>
    Disclaimer:
      This project "profanity2" was forked from the original project and
      modified to guarantee "SAFETY BY DESIGN". This means source code of
      this project doesn't require any audits, but still guarantee safe usage.

  From original "profanity":
    Author: Johan Gustafsson <profanity@johgu.se>
    Beer donations: 0x000dead000ae1c8e8ac27103e4ff65f42a4e9203
    Disclaimer:
      Always verify that a private key generated by this program corresponds to
      the public key printed by importing it to a wallet of your choice. This
      program like any software might contain bugs and it does by design cut
      corners to improve overall performance.
```

## Running without a GPU (`--cpu`)

profanity2 looks for graphics cards and stops if it finds none. `--cpu` points it at CPU
devices instead, which needs a CPU OpenCL runtime installed — [PoCL](http://portablecl.org) is
the usual one:

```bash
sudo apt install -y pocl-opencl-icd     # Debian/Ubuntu

# Start-up seeds -i * -I points before the first hash, so keep that small on a CPU
./profanity2.x64 --cpu --matching dead --min-score 4 -i 255 -I 64 -z $PUBLIC_KEY
```

Expect it to be slower than a graphics card by a wide margin — this is for machines that have
no usable GPU, for trying a search out before renting one, and for working on profanity2
itself. It is not a way to mine an address you actually want.

## Usage examples

All examples below require the mandatory `-z` argument: your seed public key as a 128-symbol
hex string without the `04` prefix (see
[Getting public key for mandatory `-z` parameter](#getting-public-key-for-mandatory--z-parameter)).
In the examples it is abbreviated as `$PUBLIC_KEY`:

```bash
export PUBLIC_KEY="HEX_PUBLIC_KEY_128_CHARS_LONG"
```

### Prefix (`--leading`)

Score on addresses starting with as many repetitions of a single hex character as possible.
The score is the number of leading characters, so the tool keeps running and prints better
and better results:

```bash
# 0x00000... (as many leading zeros as possible)
./profanity2.x64 --leading 0 -z $PUBLIC_KEY

# 0xaaaaa... (as many leading "a"s as possible)
./profanity2.x64 --leading a -z $PUBLIC_KEY
```

### Exact pattern (`--matching`)

`--matching` takes a hex pattern up to 40 characters long (the length of an address without `0x`).
Every position that is **not** a valid hex character (conventionally `X`) is a wildcard that
matches anything. The score is the number of matched fixed positions.

The length of the pattern says where in the address it is looked for. A pattern of the full 40
characters is looked for exactly where it is written; a shorter one is looked for at every
position it could sit at, which is much the faster search of the two — there are 41 − *length*
places for it to turn up in rather than one. Padding with wildcards is therefore how you anchor
a pattern.

**Anywhere** — a short pattern floats:

```bash
# 0x...dead... (anywhere in the address)
./profanity2.x64 --matching dead -z $PUBLIC_KEY
```

**Prefix** — pad the end of the pattern out to 40 characters to pin it to the start:

```bash
# 0xdead...
./profanity2.x64 --matching deadXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX -z $PUBLIC_KEY
```

**Suffix** — pad the beginning of a full 40-character pattern with `X` wildcards:

```bash
# 0x...999999
./profanity2.x64 --matching XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX999999 -z $PUBLIC_KEY
```

**Prefix and suffix at the same time** — fix both ends, wildcard the middle:

```bash
# 0x1111...2222
./profanity2.x64 --matching 1111XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX2222 -z $PUBLIC_KEY

# 0xbad...bad
./profanity2.x64 --matching badXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXbad -z $PUBLIC_KEY
```

**Arbitrary positions** — any mix of fixed characters and wildcards works:

```bash
# 0xXXXXcafeXXXX...XXXX (characters 5-8 are "cafe")
./profanity2.x64 --matching XXXXcafeXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX -z $PUBLIC_KEY
```

A pattern shorter than 40 characters may hold wildcards too, and floats as a whole:

```bash
# 0x...c0XXde... (anywhere in the address)
./profanity2.x64 --matching c0XXde -z $PUBLIC_KEY
```

Note: by default only results improving the best score so far are printed, so after a result
matching all fixed positions is found nothing better can appear — stop the program with
`Ctrl-C`, or see `--min-score` below to keep them coming. Keep in mind that every additional
fixed character multiplies the expected search time by 16.

### Every result at or above a score (`-r`, `--min-score`)

By default the bar rises as the search goes: each printed result becomes the new bar, and only
something better is printed after it. That is what you want when you are after the best address
the GPU can find, and wrong when you know what you are after — stumble on one address scoring
better than you asked for and every later address that merely satisfies the request goes
unreported, however many of them turn up.

`--min-score` pins the bar where you put it. Every address scoring that or better is printed,
for as long as the program runs, and nothing narrows the search. A mask scores one point for
each character it pins down, so a four-character pattern wants `--min-score 4`:

```bash
# Every address 0x1337...c0de found, not just the first one
./profanity2.x64 --matching 1337XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXc0de --min-score 8 -z $PUBLIC_KEY

# Every address with at least five leading zeros
./profanity2.x64 --matching 00000XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX --min-score 5 -z $PUBLIC_KEY

# Every address containing "dead" anywhere
./profanity2.x64 --matching dead --min-score 4 -z $PUBLIC_KEY

# Every address holding at least twelve 'a' characters
./profanity2.x64 --range -m 10 -M 10 --min-score 12 -z $PUBLIC_KEY
```

This is what you want when a filter of your own sits behind profanity2 — checking EIP-55
capitalisation, say — and needs a steady supply of candidates rather than one address and
silence. One round reports at most 40 addresses per GPU, which at the rate rounds complete is
several hundred a second; if you ever see the warning that says some were dropped, the floor is
low enough that what you asked for is not rare at all, and asking for something rarer is the
answer rather than a bigger buffer.

### Character classes anywhere (`--zeros`, `--letters`, `--numbers`)

Score on the total amount of matching characters anywhere in the address:

```bash
# As many "0" characters as possible, e.g. 0x00aa0e050bb03c0b066e00c0f70a03d0d000b0d7
./profanity2.x64 --zeros -z $PUBLIC_KEY

# Only letters (a-f), e.g. 0xffcadbfaecfcdeaddeedabadfeedfacebeefcafe
./profanity2.x64 --letters -z $PUBLIC_KEY

# Only numbers (0-9), e.g. 0x8896339129744478701529940603494328137361
./profanity2.x64 --numbers -z $PUBLIC_KEY
```

### Ranges (`--leading-range`, `--range`)

Score on characters within a given hex range, set with `-m/--min` and `-M/--max`
(0 is `0`, 15 is `f`):

```bash
# Leading characters in range 0-1, e.g. 0x0110100...
./profanity2.x64 --leading-range -m 0 -M 1 -z $PUBLIC_KEY

# Leading characters in range a-c, e.g. 0xcbabacc...
./profanity2.x64 --leading-range -m 10 -M 12 -z $PUBLIC_KEY

# Characters in range 0-1 anywhere in the address
./profanity2.x64 --range -m 0 -M 1 -z $PUBLIC_KEY
```

### Other scoring modes (`--mirror`, `--leading-doubles`, `--zero-bytes`)

```bash
# Address mirrored around its center, e.g. 0x...abccba...
./profanity2.x64 --mirror -z $PUBLIC_KEY

# Leading pairs of identical characters, e.g. 0x00fFcc55...
./profanity2.x64 --leading-doubles -z $PUBLIC_KEY

# As many zero BYTES (pairs "00" at even positions) as possible; such addresses
# save gas when used in calldata, e.g. 0x00815e00c0fd4a2d00ae00fa00e300ee00fc0034
./profanity2.x64 --zero-bytes -z $PUBLIC_KEY
```

### Vanity contract address (`--contract`)

Add `--contract` to any scoring mode to score the address of the **contract deployed by the
zeroth transaction** of the found account instead of the account address itself:

```bash
# Account whose first deployed contract gets a 0x00000... address
./profanity2.x64 --contract --leading 0 -z $PUBLIC_KEY

# Account whose first deployed contract address has the most zero bytes
./profanity2.x64 --contract --zero-bytes -z $PUBLIC_KEY
```

### Vanity CREATE2 address (`--create2`)

A contract deployed with CREATE2 lands at the last 20 bytes of

```
keccak256(0xff ++ factory ++ salt ++ keccak256(init_code))
```

and the salt is 32 bytes of the deployer's choosing. So there is no key to find
here and nothing in the search to keep secret: `--create2` looks for a **salt**,
`-z` is neither wanted nor used, and what it prints beside each address is the
salt to deploy with. It needs the address of the contract doing the deploying
and the hash of the init code being deployed:

```bash
export FACTORY="0x4e59b44847b379578588920ca78fbf26c0b4956c"
export INIT_CODE_HASH="0x21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f"

# The most zero bytes obtainable, which is what makes a contract cheap to call
./profanity2.x64 --create2 $FACTORY --init-code-hash $INIT_CODE_HASH --zero-bytes
```

```
  Time:     6s Score:  5 Salt: 0x0000…06c5f7ee Create2: 0x9489003920f95a00300000ec9621b93b8a5a007c
```

Every scoring mode above works against a CREATE2 address exactly as it does
against an account's, `--min-score` included:

```bash
# 0xdead…
./profanity2.x64 --create2 $FACTORY --init-code-hash $INIT_CODE_HASH \
    --matching deadXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX --min-score 4
```

**`--caller`** pins the salt's first 20 bytes to an address. Factories that
guard against front-running — [0age's
`ImmutableCreate2Factory`](https://github.com/0age/metamorphic) among them —
require exactly that of whoever deploys through them, so a salt found without it
is one such a factory will refuse from you:

```bash
./profanity2.x64 --create2 $FACTORY --init-code-hash $INIT_CODE_HASH \
    --caller 0xYOUR_ADDRESS --zero-bytes
```

Left out, those 20 bytes are zero, which is what the same factories take to mean
that anyone may deploy it.

The init code hash is of the **init code** — the constructor and its arguments,
the bytes actually handed to CREATE2 — and not of the deployed runtime code.
Get it wrong and the search runs at full speed on addresses that will never
exist, so it is worth checking against a deployment you have already made.

Verify a salt before you use it, which needs nothing but a keccak:

```bash
export SALT="0x…the 64 hex characters printed beside the address…"

python3 - <<'EOF'
import os
from Crypto.Hash import keccak                       # pip install pycryptodome

raw = lambda name: bytes.fromhex(os.environ[name].removeprefix("0x"))
digest = keccak.new(
    digest_bits=256,
    data=b"\xff" + raw("FACTORY") + raw("SALT") + raw("INIT_CODE_HASH"),
).digest()

print("0x" + digest[12:].hex())
EOF
```

This is the same search [create2crunch](https://github.com/0age/create2crunch)
and [create2-vanity-miner](https://github.com/beincom/create2-vanity-miner) do,
with profanity2's scoring modes over it: prefixes, suffixes, masks with
wildcards, character counts and ranges, mirroring, and zero bytes.

### Benchmark and device control

```bash
# Measure hashrate without any scoring
./profanity2.x64 --benchmark -z $PUBLIC_KEY

# Multiple GPUs are used automatically; skip a device (e.g. an integrated GPU) by index
./profanity2.x64 --leading 0 -s 1 -z $PUBLIC_KEY
```

### Benchmarks - RTX 4090

Stock clocks, default flags, `--variants 1`, `--leading 0`, nothing else
running on the card:

| search | addresses/s | keccaks per address |
| --- | --- | --- |
| account address | **1.56 GH/s** | 1 |
| `--contract` | **1.05 - 1.15 GH/s** | 2 |
| `--create2` | **4.3 GH/s** | 1, and no point arithmetic |

CREATE2 is the fastest by a wide margin because it is only a hash: no point
addition, no modular inversion, nothing to seed. The account search carries all
of that, and `--contract` hashes twice per address. That ordering — CREATE2
fastest, `--contract` slowest — is the sanity check to apply to any number this
program prints about itself.

`tests/bench_create2_state.x64` times the CREATE2 hash on its own at 4.30 GH/s,
which is within a percent of the search that wraps it, so almost nothing is lost
to scoring or to the result buffer.

What the optimizations in this fork are worth on the same card:

| change | in a search | in isolation |
| --- | --- | --- |
| inline PTX multiprecision (NVIDIA only) | +12.4% | +44% on `mp_mod_mul` |
| CREATE2 hash state written as whole lanes | — | +8% on the hash |
| contract address state written as whole lanes | +3.2% | +5.1% |

Measured against a noise floor of 0.5%, established by timing two binaries that
differ only in code the run never reaches.

### Reading the speed this program prints

Two things about that line will bite anyone scripting against it, and both cost
this fork a day of wrong conclusions:

- **The unit changes.** `Dispatcher::formatSpeed` scales through K/M/G/T, so a
  card doing more than 1000 MH/s prints `GH/s`. A `grep` for `MH/s` does not
  fail loudly; it quietly returns whichever samples happened to dip below the
  boundary, which on a fast card is a biased half of the truth or nothing at
  all. Match `[KMGT]H/s` and normalise.
- **It is a moving average over the last 20 launches**, and it starts at zero.
  Read too early and the number is still climbing — 575, 861, 957 MH/s over one
  run that settles at 1178. Give it ~30 seconds and check that consecutive
  samples agree before believing one.

And when comparing two builds, confirm nothing else is on the card. A second
process competing for it halved every figure here and left them still looking
plausible and internally consistent.

# License

This project is licensed under the [MIT License](LICENSE).

