/* test_scoring.cpp
 * ================
 * Correctness tests for the scoring kernels, run through the tests/harness.cl
 * wrappers over the real functions in profanity.cl.
 *
 * Part one covers `--matching`, which is what decides where in an address a mask
 * is looked for. Part two covers the three scorers that read a whole address at
 * once with bit tricks — a single-character `--range`, a `--range` spanning
 * several, and `--zero-bytes` — against the plain per-character loops they
 * replaced. See testWholeAddress and testSpanningRange.
 *
 * On `--matching`:
 *
 * A mask as long as an address is pinned where it is written; a shorter one is
 * scored at whichever of the 41 - length offsets it sits best at, so that one
 * search covers every place a pattern could appear rather than the single place
 * the caller happened to pad it to.
 *
 * Runs the real kernel from profanity.cl over masks encoded by the real
 * Mode::matching, and checks:
 *
 * 1. Every score matches a host-side reference that works from the mask and the
 *    address as hex text, rather than from the nibble arrays the kernel reads,
 *    so an error in either encoding shows up as a disagreement.
 * 2. A pattern planted at each offset in turn scores as a full match from every
 *    one of them, and a pattern planted nowhere does not.
 * 3. A mask padded out to 40 characters still only matches where it is written,
 *    which is what the prefix and suffix searches rely on.
 *
 * Build & run (see tests/Makefile), from the repository root:
 *   cd tests && make && cd .. && ./tests/test_scoring.x64 [num_random_cases]
 */

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "testutil.hpp"

#include "../Mode.hpp"

static const int ADDRESS_NIBBLES = 40;
static const int ADDRESS_BYTES = 20;

static std::mt19937_64 g_rng(0x5EED);

typedef std::vector<uint8_t> Hash;

/* ------------------------------------------------------------------------ */
/* Host-side reference                                                       */
/* ------------------------------------------------------------------------ */
/* Deliberately works from text: the address as the 40 hex characters a user
 * would read, and the mask as the characters they typed. The kernel instead
 * shifts nibbles out of packed bytes, so the two agree only if that shifting is
 * right, which is the part most easily got backwards. */

static std::string toHexString(const Hash & hash) {
	static const char digits[] = "0123456789abcdef";
	std::string out;
	for (const uint8_t byte : hash) {
		out += digits[byte >> 4];
		out += digits[byte & 0xF];
	}
	return out;
}

static bool isWildcard(const char c) {
	return std::string("0123456789abcdefABCDEF").find(c) == std::string::npos;
}

// The pinned characters matched in one run from the start of the mask, at
// whichever offset does best.
static int refScore(const std::string & mask, const Hash & hash) {
	const std::string address = toHexString(hash);
	const int length = (int)mask.size();

	int best = 0;
	for (int at = 0; at + length <= ADDRESS_NIBBLES; ++at) {
		int run = 0;
		for (int i = 0; i < length; ++i) {
			if (isWildcard(mask[i])) {
				continue;
			}
			if (std::tolower(mask[i]) != address[at + i]) {
				break;
			}
			++run;
		}
		if (run > best) {
			best = run;
		}
	}
	return best;
}

// How much a mask can score at all: its wildcards never count.
static int pinned(const std::string & mask) {
	int count = 0;
	for (const char c : mask) {
		if (!isWildcard(c)) {
			++count;
		}
	}
	return count;
}

/* ------------------------------------------------------------------------ */
/* Cases                                                                     */
/* ------------------------------------------------------------------------ */

static Hash randomHash() {
	Hash hash(ADDRESS_BYTES);
	for (uint8_t & byte : hash) {
		byte = (uint8_t)g_rng();
	}
	return hash;
}

static void setNibble(Hash & hash, const int at, const uint8_t value) {
	uint8_t & byte = hash[at / 2];
	byte = (at % 2 == 0) ? ((value << 4) | (byte & 0x0F)) : ((byte & 0xF0) | value);
}

static uint8_t hexDigit(const char c) {
	return (uint8_t)(c <= '9' ? c - '0' : (std::tolower(c) - 'a' + 10));
}

// The same hash with the mask's pinned characters written in at `at`.
static Hash plant(const Hash & hash, const std::string & mask, const int at) {
	Hash planted = hash;
	for (size_t i = 0; i < mask.size(); ++i) {
		if (!isWildcard(mask[i])) {
			setNibble(planted, at + (int)i, hexDigit(mask[i]));
		}
	}
	return planted;
}

/* ------------------------------------------------------------------------ */
/* Kernel                                                                    */
/* ------------------------------------------------------------------------ */

// One launch scores every hash against one mask, data1/data2 being per-launch.
static std::vector<int> runScoreKernel(const ClSetup & s, const std::string & mask, const std::vector<Hash> & hashes) {
	const Mode mode = Mode::matching(mask);

	std::vector<uint8_t> flat;
	for (const Hash & hash : hashes) {
		flat.insert(flat.end(), hash.begin(), hash.end());
	}
	std::vector<int> scores(hashes.size(), -1);

	cl_int err;
	cl_mem hashBuf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, flat.size(), flat.data(), &err);
	clCheck(err, "clCreateBuffer(hashes)");
	cl_mem data1Buf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(mode.data1), (void *)mode.data1, &err);
	clCheck(err, "clCreateBuffer(data1)");
	cl_mem data2Buf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(mode.data2), (void *)mode.data2, &err);
	clCheck(err, "clCreateBuffer(data2)");
	cl_mem scoreBuf = clCreateBuffer(s.context, CL_MEM_WRITE_ONLY, scores.size() * sizeof(int), NULL, &err);
	clCheck(err, "clCreateBuffer(scores)");

	cl_kernel kernel = clCreateKernel(s.program, "k_score_matching", &err);
	clCheck(err, "clCreateKernel");
	clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &hashBuf), "clSetKernelArg(0)");
	clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &data1Buf), "clSetKernelArg(1)");
	clCheck(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data2Buf), "clSetKernelArg(2)");
	clCheck(clSetKernelArg(kernel, 3, sizeof(cl_mem), &scoreBuf), "clSetKernelArg(3)");

	const size_t count = hashes.size();
	clCheck(clEnqueueNDRangeKernel(s.queue, kernel, 1, NULL, &count, NULL, 0, NULL, NULL), "clEnqueueNDRangeKernel");
	clCheck(clEnqueueReadBuffer(s.queue, scoreBuf, CL_TRUE, 0, scores.size() * sizeof(int), scores.data(), 0, NULL, NULL), "clEnqueueReadBuffer");
	clCheck(clFinish(s.queue), "clFinish");

	clReleaseKernel(kernel);
	clReleaseMemObject(hashBuf);
	clReleaseMemObject(data1Buf);
	clReleaseMemObject(data2Buf);
	clReleaseMemObject(scoreBuf);
	return scores;
}

/* ------------------------------------------------------------------------ */
/* Tests: --matching                                                         */
/* ------------------------------------------------------------------------ */

static const char * const g_masks[] = {
	"d",                                        // one nibble, 40 offsets
	"de",
	"dead",                                     // the everyday case
	"beef1337",
	"XdeadX",                                   // wildcards at both ends
	"deXXad",                                   // and in the middle
	"0000000000",                               // a run of one digit
	"deadXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", // padded: a prefix
	"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXbeef", // padded: a suffix
	"badXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXbad", // padded: both ends at once
	"0123456789abcdef0123456789abcdef01234567", // the whole address pinned
};

static size_t testAgainstReference(const ClSetup & s, const size_t numRandom) {
	size_t failures = 0;
	size_t cases = 0;

	for (const char * const mask : g_masks) {
		const std::string text(mask);

		// Random hashes rarely contain the mask, so plant it at every offset it
		// could sit at as well: without that, nothing here would ever score a
		// full match and the offsets would go untested.
		std::vector<Hash> hashes;
		for (size_t i = 0; i < numRandom; ++i) {
			hashes.push_back(randomHash());
		}
		for (int at = 0; at + (int)text.size() <= ADDRESS_NIBBLES; ++at) {
			hashes.push_back(plant(randomHash(), text, at));
		}

		const std::vector<int> scores = runScoreKernel(s, text, hashes);

		for (size_t i = 0; i < hashes.size(); ++i) {
			++cases;
			const int expect = refScore(text, hashes[i]);
			if (scores[i] != expect) {
				if (++failures <= 5) {
					std::printf("  MISMATCH mask %s address %s: got %d, expected %d\n",
						text.c_str(), toHexString(hashes[i]).c_str(), scores[i], expect);
				}
			}
		}
	}

	std::printf("score_matching vs reference: %zu cases: %s\n",
		cases, failures == 0 ? "OK" : "FAILED");
	return failures;
}

// A planted pattern must score a full match wherever it was planted, which is
// the whole point of a short mask, and the padded masks must not.
static size_t testOffsetsCovered(const ClSetup & s) {
	size_t failures = 0;

	for (const char * const mask : g_masks) {
		const std::string text(mask);
		const int full = pinned(text);
		const int offsets = ADDRESS_NIBBLES - (int)text.size() + 1;

		std::vector<Hash> hashes;
		for (int at = 0; at < offsets; ++at) {
			hashes.push_back(plant(randomHash(), text, at));
		}

		const std::vector<int> scores = runScoreKernel(s, text, hashes);
		for (int at = 0; at < offsets; ++at) {
			if (scores[at] != full) {
				if (++failures <= 5) {
					std::printf("  MISSED mask %s planted at nibble %d: scored %d of %d\n",
						text.c_str(), at, scores[at], full);
				}
			}
		}
	}

	std::printf("every offset a mask can sit at scores a full match: %s\n",
		failures == 0 ? "OK" : "FAILED");
	return failures;
}

// The padding is what anchors a search, so a mask filling all 40 nibbles must
// stay put: `deadXX...` is a prefix and has to stay one.
static size_t testAnchoring(const ClSetup & s) {
	const std::string anchored = "deadXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX";
	const std::string floating = "dead";

	std::vector<Hash> hashes;
	std::vector<int> plantedAt;
	for (int at = 0; at + 4 <= ADDRESS_NIBBLES; ++at) {
		hashes.push_back(plant(randomHash(), floating, at));
		plantedAt.push_back(at);
	}

	const std::vector<int> anchoredScores = runScoreKernel(s, anchored, hashes);
	const std::vector<int> floatingScores = runScoreKernel(s, floating, hashes);

	size_t failures = 0;
	for (size_t i = 0; i < hashes.size(); ++i) {
		const bool atStart = plantedAt[i] == 0;

		// The anchored mask has one placement: only a pattern at the start of
		// the address is a full match for it.
		if (atStart ? (anchoredScores[i] != 4) : (anchoredScores[i] == 4)) {
			if (++failures <= 5) {
				std::printf("  ANCHORING mask %s: pattern at nibble %d scored %d\n",
					anchored.c_str(), plantedAt[i], anchoredScores[i]);
			}
		}

		// The same pattern unpadded is a full match from anywhere.
		if (floatingScores[i] != 4) {
			if (++failures <= 5) {
				std::printf("  FLOATING mask %s: pattern at nibble %d scored %d\n",
					floating.c_str(), plantedAt[i], floatingScores[i]);
			}
		}
	}

	std::printf("a mask as long as an address stays where it is written: %s\n",
		failures == 0 ? "OK" : "FAILED");
	return failures;
}

/* ------------------------------------------------------------------------ */
/* Tests: the scorers that read a whole address at once                      */
/* ------------------------------------------------------------------------ */
/* A single-character --range, a --range spanning several, and --zero-bytes all
 * fold the address with bit tricks instead of walking it. Each is checked
 * against the plain loop it replaced, kept in harness.cl for exactly this.
 *
 * Random addresses are nearly useless on their own here: they hold about two
 * and a half of any given character and a zero byte once in every thirteen, so
 * nothing above a very low score would ever be reached. The counts that matter
 * are therefore built rather than waited for.
 *
 * The bench kernels do the scoring: at one iteration each returns the score of
 * the address it was handed, before the accumulator has perturbed anything, so
 * they are the wrappers already to hand. */

static std::vector<int> runScorer(const ClSetup & s, const char * const kernelName, const cl_uchar min, const cl_uchar max, const std::vector<Hash> & hashes) {
	std::vector<uint8_t> flat;
	for (const Hash & hash : hashes) {
		flat.insert(flat.end(), hash.begin(), hash.end());
	}

	cl_uchar data1[PROFANITY_MODE_DATA] = {};
	cl_uchar data2[PROFANITY_MODE_DATA] = {};
	data1[0] = min;
	data2[0] = max;

	std::vector<int> scores(hashes.size(), -1);
	const cl_uint one = 1;

	cl_int err;
	cl_mem hashBuf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, flat.size(), flat.data(), &err);
	clCheck(err, "clCreateBuffer(hashes)");
	cl_mem data1Buf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(data1), data1, &err);
	clCheck(err, "clCreateBuffer(data1)");
	cl_mem data2Buf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(data2), data2, &err);
	clCheck(err, "clCreateBuffer(data2)");
	cl_mem scoreBuf = clCreateBuffer(s.context, CL_MEM_WRITE_ONLY, scores.size() * sizeof(int), NULL, &err);
	clCheck(err, "clCreateBuffer(scores)");

	cl_kernel kernel = clCreateKernel(s.program, kernelName, &err);
	clCheck(err, "clCreateKernel");
	clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &hashBuf), "clSetKernelArg(0)");
	clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &data1Buf), "clSetKernelArg(1)");
	clCheck(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data2Buf), "clSetKernelArg(2)");
	clCheck(clSetKernelArg(kernel, 3, sizeof(cl_uint), &one), "clSetKernelArg(3)");
	clCheck(clSetKernelArg(kernel, 4, sizeof(cl_mem), &scoreBuf), "clSetKernelArg(4)");

	const size_t count = hashes.size();
	clCheck(clEnqueueNDRangeKernel(s.queue, kernel, 1, NULL, &count, NULL, 0, NULL, NULL), "clEnqueueNDRangeKernel");
	clCheck(clEnqueueReadBuffer(s.queue, scoreBuf, CL_TRUE, 0, scores.size() * sizeof(int), scores.data(), 0, NULL, NULL), "clEnqueueReadBuffer");
	clCheck(clFinish(s.queue), "clFinish");

	clReleaseKernel(kernel);
	clReleaseMemObject(hashBuf);
	clReleaseMemObject(data1Buf);
	clReleaseMemObject(data2Buf);
	clReleaseMemObject(scoreBuf);
	return scores;
}

// An address holding exactly `count` of `nibble`, the rest of it anything but.
static Hash withNibbles(const uint8_t nibble, const int count) {
	Hash hash(ADDRESS_BYTES, 0);

	for (int at = 0; at < ADDRESS_NIBBLES; ++at) {
		// Any other character will do for the remainder; stepping through them
		// keeps the filler from being a single repeated value.
		const uint8_t other = (uint8_t)((nibble + 1 + (at % 15)) & 0xF);
		setNibble(hash, at, at < count ? nibble : other);
	}

	return hash;
}

// An address whose first `count` bytes are zero and whose rest are not.
static Hash withZeroBytes(const int count) {
	Hash hash(ADDRESS_BYTES, 0);

	for (int at = 0; at < ADDRESS_BYTES; ++at) {
		hash[at] = at < count ? 0x00 : (uint8_t)(1 + (at % 255));
	}

	return hash;
}

static size_t testWholeAddress(const ClSetup & s, const size_t numRandom) {
	size_t failures = 0;

	// Counting one character, at every count of every character, so the fold
	// is exercised over its whole range rather than the two or three a random
	// address would reach.
	size_t counted = 0;
	for (uint8_t nibble = 0; nibble < 16; ++nibble) {
		std::vector<Hash> hashes;
		for (int count = 0; count <= ADDRESS_NIBBLES; ++count) {
			hashes.push_back(withNibbles(nibble, count));
		}
		for (size_t i = 0; i < numRandom; ++i) {
			hashes.push_back(randomHash());
		}

		const std::vector<int> loop = runScorer(s, "bench_score_range_loop", nibble, nibble, hashes);
		const std::vector<int> whole = runScorer(s, "bench_score_rangeequal", nibble, nibble, hashes);

		for (size_t i = 0; i < hashes.size(); ++i) {
			++counted;

			// The built cases have a count known here, so the loop is not the
			// only thing vouching for them.
			const bool built = i <= (size_t)ADDRESS_NIBBLES;
			const int expect = built ? (int)i : loop[i];

			if (whole[i] != expect || loop[i] != expect) {
				if (++failures <= 5) {
					std::printf("  COUNT of %x in %s: loop %d, whole %d, expected %d\n",
						nibble, toHexString(hashes[i]).c_str(), loop[i], whole[i], expect);
				}
			}
		}
	}

	std::printf("counting one character, whole address vs loop: %zu cases: %s\n",
		counted, failures == 0 ? "OK" : "FAILED");

	// Zero bytes, at every count of them.
	const size_t before = failures;
	std::vector<Hash> hashes;
	for (int count = 0; count <= ADDRESS_BYTES; ++count) {
		hashes.push_back(withZeroBytes(count));
	}
	for (size_t i = 0; i < numRandom; ++i) {
		hashes.push_back(randomHash());
	}

	const std::vector<int> loop = runScorer(s, "bench_score_zerobytes_loop", 0, 0, hashes);
	const std::vector<int> whole = runScorer(s, "bench_score_zerobytes", 0, 0, hashes);

	for (size_t i = 0; i < hashes.size(); ++i) {
		const bool built = i <= (size_t)ADDRESS_BYTES;
		const int expect = built ? (int)i : loop[i];

		if (whole[i] != expect || loop[i] != expect) {
			if (++failures <= 5) {
				std::printf("  ZERO BYTES in %s: loop %d, whole %d, expected %d\n",
					toHexString(hashes[i]).c_str(), loop[i], whole[i], expect);
			}
		}
	}

	std::printf("counting zero bytes, whole address vs loop: %zu cases: %s\n",
		hashes.size(), failures == before ? "OK" : "FAILED");

	return failures;
}

// A range spanning several characters, over every pair of ends there is —
// including the inverted ones, which can never hold, and the full 0 to 15, which
// always does. The guard bits are what could go wrong here: a borrow escaping
// one character corrupts its neighbour, and only the ends of the range decide
// whether one escapes, so every pair of ends is worth trying.
static size_t testSpanningRange(const ClSetup & s, const size_t numRandom) {
	std::vector<Hash> hashes;
	for (uint8_t nibble = 0; nibble < 16; ++nibble) {
		hashes.push_back(withNibbles(nibble, ADDRESS_NIBBLES));
		hashes.push_back(withNibbles(nibble, 1));
	}
	for (size_t i = 0; i < numRandom; ++i) {
		hashes.push_back(randomHash());
	}

	size_t failures = 0;
	size_t cases = 0;

	for (uint8_t min = 0; min < 16; ++min) {
		for (uint8_t max = 0; max < 16; ++max) {
			const std::vector<int> loop = runScorer(s, "bench_score_range_loop", min, max, hashes);
			const std::vector<int> guard = runScorer(s, "bench_score_range", min, max, hashes);

			for (size_t i = 0; i < hashes.size(); ++i) {
				++cases;
				if (loop[i] != guard[i]) {
					if (++failures <= 5) {
						std::printf("  RANGE %x-%x in %s: loop %d, guard bits %d\n",
							min, max, toHexString(hashes[i]).c_str(), loop[i], guard[i]);
					}
				}
			}
		}
	}

	std::printf("a range of several characters, guard bits vs loop: %zu cases: %s\n",
		cases, failures == 0 ? "OK" : "FAILED");
	return failures;
}

int main(int argc, char ** argv) {
	const size_t numRandom = argc > 1 ? (size_t)std::atoi(argv[1]) : 256;

	const ClSetup s = clSetup();
	std::printf("Random hashes per mask: %zu\n\n", numRandom);

	size_t failures = 0;
	failures += testAgainstReference(s, numRandom);
	failures += testOffsetsCovered(s);
	failures += testAnchoring(s);
	failures += testWholeAddress(s, numRandom);
	failures += testSpanningRange(s, numRandom < 64 ? numRandom : 64);

	std::printf("\n%s\n", failures == 0 ? "All tests passed." : "TESTS FAILED.");
	return failures == 0 ? 0 : 1;
}
