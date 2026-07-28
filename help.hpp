#ifndef HPP_HELP
#define HPP_HELP

#include <string>

const std::string g_strHelp = R"(
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
    -R, --rounds <n>        How many point additions a launch does per point,
                            before handing back. [default = 1]

                            This saves the kernel launch and nothing else. A
                            point's delta and previous lambda still go out to
                            global memory and come back on every round, so the
                            traffic per point is what it was; only the
                            enqueue, the result read and the dispatch behind
                            them are paid once for every n rather than once
                            each. On a card whose launch is already millions
                            of points long that is a rounding error, and 1 is
                            the right answer.

                            Worth more only where a launch is short enough for
                            its overhead to show -- a small -I, a slow device,
                            a driver with an expensive enqueue. Measure before
                            raising it.

                            Costs nothing in registers. Bounded instead by how
                            long a launch may run: too high and a run answers
                            --min-score and its own cancellation late, and on
                            some drivers a kernel that runs for seconds is a
                            kernel that gets killed.

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
      corners to improve overall performance.)";

#endif /* HPP_HELP */
