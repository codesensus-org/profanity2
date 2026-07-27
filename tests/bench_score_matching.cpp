/* bench_score_matching.cpp
 * ========================
 * Throughput of the scoring functions, and of the implementations they
 * replaced.
 *
 * For `--matching`:
 *   bytes  one entry per hash byte, mask and value packed together. What the
 *          mode scored with before a short mask was allowed to float, so a
 *          `contains` search had to pick one offset and pay 41 - length times
 *          the search for it.
 *   scan   one entry per mask character, wildcards included, 0xFF ending the
 *          mask. Floats a short mask, but rescans for that terminator on every
 *          hash and walks an anchored pattern through all of its padding.
 *   new    the pinned characters alone, with the count of them and the mask
 *          length in the last entry, an anchored mask taken straight out of the
 *          packed bytes, and a floating one finding the offsets worth trying
 *          rather than walking all of them.
 *
 * Scoring is a small part of the iterate kernel — the point addition and the
 * keccak permutation in front of it are the bulk — so treat these as the cost
 * of the scoring step alone and not as a throughput figure for a search. The
 * keccak measured at the end is there to say how small a part.
 *
 * Build & run (see tests/Makefile), from the repository root:
 *   cd tests && make && cd .. && ./tests/bench_score_matching.x64 [items] [iterations]
 */

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "testutil.hpp"

#include "../Mode.hpp"

static const int ADDRESS_BYTES = 20;

static std::mt19937_64 g_rng(0x5EED);

/* ------------------------------------------------------------------------ */
/* The encodings the older kernels read                                      */
/* ------------------------------------------------------------------------ */

struct MaskData {
	cl_uchar data1[PROFANITY_MODE_DATA];
	cl_uchar data2[PROFANITY_MODE_DATA];
};

static bool isWildcard(const char c) {
	return std::string("0123456789abcdefABCDEF").find(c) == std::string::npos;
}

static cl_uchar hexDigit(const char c) {
	return (cl_uchar)(c <= '9' ? c - '0' : (std::tolower(c) - 'a' + 10));
}

// Two nibbles to the byte, as Mode::matching laid it out before masks floated.
static MaskData encodeBytes(const std::string & mask) {
	MaskData m = {};

	for (size_t i = 0; i < mask.size(); i += 2) {
		const bool hiWild = isWildcard(mask[i]);
		const bool loWild = i + 1 >= mask.size() || isWildcard(mask[i + 1]);

		m.data1[i / 2] = (cl_uchar)((hiWild ? 0 : 0xF0) | (loWild ? 0 : 0x0F));
		m.data2[i / 2] = (cl_uchar)((hiWild ? 0 : hexDigit(mask[i]) << 4)
			| (loWild ? 0 : hexDigit(mask[i + 1])));
	}

	return m;
}

// One entry per mask character, 0xFF after the last of them.
static MaskData encodeScan(const std::string & mask) {
	MaskData m = {};

	for (size_t i = 0; i < mask.size(); ++i) {
		m.data1[i] = isWildcard(mask[i]) ? 0x0 : 0xF;
		m.data2[i] = isWildcard(mask[i]) ? 0x0 : hexDigit(mask[i]);
	}

	m.data1[mask.size()] = 0xFF;
	return m;
}

// What Mode::matching produces today.
static MaskData encodeCurrent(const std::string & mask) {
	const Mode mode = Mode::matching(mask);

	MaskData m = {};
	for (int i = 0; i < PROFANITY_MODE_DATA; ++i) {
		m.data1[i] = mode.data1[i];
		m.data2[i] = mode.data2[i];
	}
	return m;
}

/* ------------------------------------------------------------------------ */
/* Timing                                                                    */
/* ------------------------------------------------------------------------ */

static double runBench(const ClSetup & s, const char * const kernelName, const MaskData & mask, const std::vector<uint8_t> & hashes, const size_t items, const unsigned iterations, std::vector<int> * const captured = NULL) {
	std::vector<int> out(items, 0);

	cl_int err;
	cl_mem hashBuf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, hashes.size(), (void *)hashes.data(), &err);
	clCheck(err, "clCreateBuffer(hashes)");
	cl_mem data1Buf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(mask.data1), (void *)mask.data1, &err);
	clCheck(err, "clCreateBuffer(data1)");
	cl_mem data2Buf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(mask.data2), (void *)mask.data2, &err);
	clCheck(err, "clCreateBuffer(data2)");
	cl_mem outBuf = clCreateBuffer(s.context, CL_MEM_WRITE_ONLY, out.size() * sizeof(int), NULL, &err);
	clCheck(err, "clCreateBuffer(out)");

	cl_kernel kernel = clCreateKernel(s.program, kernelName, &err);
	clCheck(err, "clCreateKernel");
	clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &hashBuf), "clSetKernelArg(0)");
	clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &data1Buf), "clSetKernelArg(1)");
	clCheck(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data2Buf), "clSetKernelArg(2)");
	clCheck(clSetKernelArg(kernel, 3, sizeof(cl_uint), &iterations), "clSetKernelArg(3)");
	clCheck(clSetKernelArg(kernel, 4, sizeof(cl_mem), &outBuf), "clSetKernelArg(4)");

	// One launch to warm the device up, then the one that is timed.
	clCheck(clEnqueueNDRangeKernel(s.queue, kernel, 1, NULL, &items, NULL, 0, NULL, NULL), "clEnqueueNDRangeKernel(warmup)");
	clCheck(clFinish(s.queue), "clFinish(warmup)");

	const auto began = std::chrono::steady_clock::now();
	clCheck(clEnqueueNDRangeKernel(s.queue, kernel, 1, NULL, &items, NULL, 0, NULL, NULL), "clEnqueueNDRangeKernel");
	clCheck(clFinish(s.queue), "clFinish");
	const auto ended = std::chrono::steady_clock::now();

	clCheck(clEnqueueReadBuffer(s.queue, outBuf, CL_TRUE, 0, out.size() * sizeof(int), out.data(), 0, NULL, NULL), "clEnqueueReadBuffer");

	clReleaseKernel(kernel);
	clReleaseMemObject(hashBuf);
	clReleaseMemObject(data1Buf);
	clReleaseMemObject(data2Buf);
	clReleaseMemObject(outBuf);

	if (captured != NULL) {
		*captured = out;
	}

	const double seconds = std::chrono::duration<double>(ended - began).count();
	return seconds * 1e9 / ((double)items * iterations);
}

// The scorers a `count` or `zero-bytes` search runs on, against the loops they
// replaced. Correctness first: at one iteration the accumulator is the score of
// the untouched address, so the two must agree before their timings mean
// anything. tests/test_scoring.x64 is where this is checked properly.
static void benchWholeAddress(const ClSetup & s, const std::vector<uint8_t> & hashes, const size_t items, const unsigned iterations) {
	struct Pair {
		const char * what;
		const char * loop;
		const char * whole;
		cl_uchar data;
	};

	static const Pair pairs[] = {
		// Against the per-character loop both replaced, not against each other:
		// the general --range is itself no longer a loop.
		{ "count of one character", "bench_score_range_loop", "bench_score_rangeequal", 0xa },
		{ "zero bytes",             "bench_score_zerobytes_loop", "bench_score_zerobytes", 0 },
	};

	std::printf("\n  %-26s %9s %9s   %s\n", "whole-address scorer", "loop", "whole", "whole vs loop");

	for (const Pair & p : pairs) {
		MaskData m = {};
		m.data1[0] = p.data;
		m.data2[0] = p.data;

		std::vector<int> fromLoop;
		std::vector<int> fromWhole;
		runBench(s, p.loop, m, hashes, items, 1, &fromLoop);
		runBench(s, p.whole, m, hashes, items, 1, &fromWhole);

		size_t disagreed = 0;
		for (size_t i = 0; i < fromLoop.size(); ++i) {
			disagreed += fromLoop[i] != fromWhole[i];
		}

		const double loop = runBench(s, p.loop, m, hashes, items, iterations);
		const double whole = runBench(s, p.whole, m, hashes, items, iterations);

		std::printf("  %-26s %9.2f %9.2f   %+.0f%%%s\n", p.what, loop, whole,
			(loop / whole - 1.0) * 100.0,
			disagreed ? "  SCORES DISAGREE" : "");
	}
}

// Every scoring kernel, at the settings a caller would actually use, so that
// what is worth optimising can be read off rather than guessed at.
//
// Most of these stop at the first character that disappoints them — a leading
// run ends where the run ends — so they do about one step of their loop whatever
// the address, and there is nothing in them to win back. The ones that have to
// look at all forty characters every time are the ones to watch.
static void benchEveryScorer(const ClSetup & s, const std::vector<uint8_t> & hashes, const size_t items, const unsigned iterations) {
	struct Scorer {
		const char * what;
		const char * kernel;
		cl_uchar min;
		cl_uchar max;
		const char * reads;
	};

	static const Scorer scorers[] = {
		{ "benchmark (scores nothing)", "bench_score_benchmark",    0,  0, "nothing" },
		{ "leading, --leading 0",       "bench_score_leading",      0,  0, "until it differs" },
		{ "leading-range, -m 0 -M 1",   "bench_score_leadingrange", 0,  1, "until it differs" },
		{ "leading-doubles",            "bench_score_doubles",      0,  0, "until it differs" },
		{ "mirror",                     "bench_score_mirror",       0,  0, "until it differs" },
		{ "range, per-character loop",  "bench_score_range_loop",  10, 15, "all of it" },
		{ "range one char, count",      "bench_score_rangeequal",  10, 10, "all of it" },
		{ "range a span, --letters",    "bench_score_range",       10, 15, "all of it" },
		{ "zero-bytes",                 "bench_score_zerobytes",    0,  0, "all of it" },
	};

	std::printf("\n  %-30s %9s   %s\n", "scorer", "ns", "reads");

	for (const Scorer & scorer : scorers) {
		MaskData m = {};
		m.data1[0] = scorer.min;
		m.data2[0] = scorer.max;

		const double ns = runBench(s, scorer.kernel, m, hashes, items, iterations);
		std::printf("  %-30s %9.2f   %s\n", scorer.what, ns, scorer.reads);
	}
}

/* ------------------------------------------------------------------------ */

struct Case {
	const char * what;
	const char * mask;
};

static const Case g_cases[] = {
	{ "prefix, 4 of 40 pinned",  "deadXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" },
	{ "suffix, 4 of 40 pinned",  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXbeef" },
	{ "both ends, 6 of 40",      "badXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXbad" },
	{ "prefix, 8 of 40 pinned",  "deadbeefXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" },
	{ "contains, 4 floating",    "dead" },
	{ "contains, 8 floating",    "deadbeef" },
	{ "fully pinned, 40 of 40",  "0123456789abcdef0123456789abcdef01234567" },
};

int main(int argc, char ** argv) {
	const size_t items = argc > 1 ? (size_t)std::atoi(argv[1]) : 4096;
	const unsigned iterations = argc > 2 ? (unsigned)std::atoi(argv[2]) : 2000;

	const ClSetup s = clSetup();

	std::vector<uint8_t> hashes(items * ADDRESS_BYTES);
	for (uint8_t & byte : hashes) {
		byte = (uint8_t)g_rng();
	}

	std::printf("%zu work items x %u iterations, ns per scored address\n\n", items, iterations);
	std::printf("  %-26s %9s %9s %9s   %s\n", "mask", "bytes", "scan", "new", "new vs scan");

	for (const Case & c : g_cases) {
		const std::string mask(c.mask);

		const double bytes = runBench(s, "bench_score_matching_bytes", encodeBytes(mask), hashes, items, iterations);
		const double scan = runBench(s, "bench_score_matching_scan", encodeScan(mask), hashes, items, iterations);
		const double now = runBench(s, "bench_score_matching", encodeCurrent(mask), hashes, items, iterations);

		std::printf("  %-26s %9.2f %9.2f %9.2f   %+.0f%%\n",
			c.what, bytes, scan, now, (scan / now - 1.0) * 100.0);
	}

	benchWholeAddress(s, hashes, items, iterations);
	benchEveryScorer(s, hashes, items, iterations);

	std::printf("\nNote: `bytes` scores a padded mask only where it is written, so for the\n");
	std::printf("floating cases it is not doing the same search as the other two.\n");

	// What the numbers above are worth. The iterate kernel runs one keccak
	// permutation per address before it scores anything, and the elliptic curve
	// arithmetic before that again, so a scoring cost well under a keccak is a
	// scoring cost that barely shows up in a search.
	std::vector<int> out(items, 0);
	cl_int err;
	cl_mem outBuf = clCreateBuffer(s.context, CL_MEM_WRITE_ONLY, out.size() * sizeof(int), NULL, &err);
	clCheck(err, "clCreateBuffer(out)");

	cl_kernel kernel = clCreateKernel(s.program, "bench_keccak", &err);
	clCheck(err, "clCreateKernel(bench_keccak)");
	clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &outBuf), "clSetKernelArg(0)");
	clCheck(clSetKernelArg(kernel, 1, sizeof(cl_uint), &iterations), "clSetKernelArg(1)");

	clCheck(clEnqueueNDRangeKernel(s.queue, kernel, 1, NULL, &items, NULL, 0, NULL, NULL), "clEnqueueNDRangeKernel(warmup)");
	clCheck(clFinish(s.queue), "clFinish(warmup)");

	const auto began = std::chrono::steady_clock::now();
	clCheck(clEnqueueNDRangeKernel(s.queue, kernel, 1, NULL, &items, NULL, 0, NULL, NULL), "clEnqueueNDRangeKernel");
	clCheck(clFinish(s.queue), "clFinish");
	const auto ended = std::chrono::steady_clock::now();

	clReleaseKernel(kernel);
	clReleaseMemObject(outBuf);

	const double keccak = std::chrono::duration<double>(ended - began).count() * 1e9 / ((double)items * iterations);
	std::printf("\nOne keccak permutation, the same way: %.2f ns. The iterate kernel runs one\n", keccak);
	std::printf("per address (two with --contract) plus the curve arithmetic, so that is a\n");
	std::printf("lower bound on what a scored address costs before it is scored at all.\n");

	return 0;
}
