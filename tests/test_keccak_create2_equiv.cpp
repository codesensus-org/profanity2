// Direct keccak equivalence test for the CREATE2 specialisation.
//
// Runs the stock sha3_keccakf and sha3_keccakf_create2 on the SAME input state
// and asserts byte-identical 200-byte output over many random inputs that
// respect the real absorb layout: the 13 lanes the specialisation assumes zero
// are set to zero, and EVERY other lane -- including st[16], which only carries
// the pad-end bit in production -- is randomised. That is strictly stronger
// than the production call site, where lanes 0..10 hold a structured preimage
// and st[16] is exactly 0x8000000000000000.
//
// The zero lanes come from profanity.cl:1147-1149 (h.q[11..24] = 0), minus
// st[16], which sha3_keccakf* itself sets via h->d[33] ^= 0x80000000.
//
// This complements test_create2 (which pins EIP-1014 vectors end to end) by
// covering the whole 200-byte state rather than just the 20 address bytes.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "testutil.hpp"

static int g_failures = 0;

// Lane indices (st[i], ulong lanes) that sha3_keccakf_create2 RELIES UPON being
// zero for its round-0 THETA. Everything else is randomised by the test.
// NOTE: 16 is deliberately absent -- the keccak padding bit lands there.
static const int kCreate2ZeroLanes[] = {
	11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24
};

static bool isZeroLane(const int lane, const int * const zeros, const size_t n) {
	for (size_t i = 0; i < n; ++i) {
		if (zeros[i] == lane) {
			return true;
		}
	}
	return false;
}

// Pack an 88-byte (PROFANITY_CREATE2_WORDS * 4) little-endian preimage buffer
// into lanes 0..10, exactly as buildCreate2Template + profanity_create2 do.
static void packPreimage(cl_ulong * const st, const uint8_t * const preimage) {
	for (int lane = 0; lane < 11; ++lane) {
		cl_ulong v = 0;
		for (int b = 0; b < 8; ++b) {
			v |= (cl_ulong)preimage[lane * 8 + b] << (b * 8);
		}
		st[lane] = v;
	}
}

// Build N input states of 25 lanes each, zeroing the assumed-zero lanes and
// randomising the rest. The first three states pin down the exact production
// layouts so the realistic cases are always covered explicitly.
static std::vector<cl_ulong> buildInputs(const size_t n, std::mt19937_64 & rng) {
	const size_t nZeros = sizeof(kCreate2ZeroLanes) / sizeof(int);
	std::vector<cl_ulong> in(n * 25, 0);

	for (size_t item = 0; item < n; ++item) {
		cl_ulong * const st = &in[item * 25];
		for (int lane = 0; lane < 25; ++lane) {
			if (isZeroLane(lane, kCreate2ZeroLanes, nZeros)) {
				st[lane] = 0;
			} else {
				st[lane] = (cl_ulong)rng();
			}
		}

		if (item == 0) {
			// Exact production pre-padding layout: 0xff || factory[20] ||
			// salt[32] || initCodeHash[32] || 0x01 pad start, rest zero.
			uint8_t preimage[PROFANITY_CREATE2_WORDS * 4] = { 0 };
			preimage[0] = 0xff;
			for (int i = 1; i < PROFANITY_CREATE2_PREIMAGE; ++i) {
				preimage[i] = (uint8_t)(rng() & 0xFF);
			}
			preimage[PROFANITY_CREATE2_PREIMAGE] = 0x01;
			packPreimage(st, preimage);
			st[16] = 0; // pad end applied inside keccak
		} else if (item == 1) {
			// All-zero body: only the structural pad start survives.
			uint8_t preimage[PROFANITY_CREATE2_WORDS * 4] = { 0 };
			preimage[PROFANITY_CREATE2_PREIMAGE] = 0x01;
			packPreimage(st, preimage);
			for (int lane = 11; lane < 25; ++lane) {
				st[lane] = 0;
			}
		} else if (item == 2) {
			// All-ones body with the real pad start, st[16] still zero.
			uint8_t preimage[PROFANITY_CREATE2_WORDS * 4] = { 0 };
			for (int i = 0; i < PROFANITY_CREATE2_PREIMAGE; ++i) {
				preimage[i] = 0xFF;
			}
			preimage[PROFANITY_CREATE2_PREIMAGE] = 0x01;
			packPreimage(st, preimage);
			for (int lane = 11; lane < 25; ++lane) {
				st[lane] = 0;
			}
		}
	}
	return in;
}

static void runCase(const ClSetup & s, const char * const kernelName,
		const char * const label, const size_t n, std::mt19937_64 & rng) {
	const std::vector<cl_ulong> in = buildInputs(n, rng);
	std::vector<cl_ulong> stock(n * 25, 0);
	std::vector<cl_ulong> special(n * 25, 0);

	cl_int err;
	cl_mem inBuf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		in.size() * sizeof(cl_ulong), (void *)in.data(), &err);
	clCheck(err, "clCreateBuffer(in)");
	cl_mem stockBuf = clCreateBuffer(s.context, CL_MEM_WRITE_ONLY,
		stock.size() * sizeof(cl_ulong), NULL, &err);
	clCheck(err, "clCreateBuffer(stock)");
	cl_mem specialBuf = clCreateBuffer(s.context, CL_MEM_WRITE_ONLY,
		special.size() * sizeof(cl_ulong), NULL, &err);
	clCheck(err, "clCreateBuffer(special)");

	cl_kernel kernel = clCreateKernel(s.program, kernelName, &err);
	clCheck(err, "clCreateKernel");

	clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &inBuf), "arg0");
	clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &stockBuf), "arg1");
	clCheck(clSetKernelArg(kernel, 2, sizeof(cl_mem), &specialBuf), "arg2");

	const size_t globalSize = n;
	clCheck(clEnqueueNDRangeKernel(s.queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL), "enqueue");
	clCheck(clEnqueueReadBuffer(s.queue, stockBuf, CL_TRUE, 0, stock.size() * sizeof(cl_ulong), stock.data(), 0, NULL, NULL), "read stock");
	clCheck(clEnqueueReadBuffer(s.queue, specialBuf, CL_TRUE, 0, special.size() * sizeof(cl_ulong), special.data(), 0, NULL, NULL), "read special");
	clCheck(clFinish(s.queue), "finish");

	clReleaseKernel(kernel);
	clReleaseMemObject(inBuf);
	clReleaseMemObject(stockBuf);
	clReleaseMemObject(specialBuf);

	size_t mismatches = 0;
	size_t firstBad = 0;
	int firstBadLane = -1;
	// Sanity: make sure the kernel is not a degenerate no-op by tracking that
	// the stock output actually varies across items.
	bool stockVaries = false;
	for (size_t item = 0; item < n; ++item) {
		for (int lane = 0; lane < 25; ++lane) {
			const cl_ulong a = stock[item * 25 + lane];
			const cl_ulong b = special[item * 25 + lane];
			if (a != b) {
				if (mismatches == 0) {
					firstBad = item;
					firstBadLane = lane;
				}
				++mismatches;
			}
		}
		if (item > 0 && std::memcmp(&stock[item * 25], &stock[0], 25 * sizeof(cl_ulong)) != 0) {
			stockVaries = true;
		}
	}

	if (!stockVaries) {
		++g_failures;
		std::printf("FAIL  %s: stock output never varied across %zu inputs (test is not exercising keccak)\n", label, n);
		return;
	}

	if (mismatches != 0) {
		++g_failures;
		std::printf("FAIL  %s: %zu/%zu lanes differ; first at item %zu lane %d\n"
			"      stock   = 0x%016llx\n      special = 0x%016llx\n",
			label, mismatches, n * 25, firstBad, firstBadLane,
			(unsigned long long)stock[firstBad * 25 + firstBadLane],
			(unsigned long long)special[firstBad * 25 + firstBadLane]);
	} else {
		std::printf("PASS  %s: %zu inputs x 200 bytes byte-identical (stock vs specialised)\n", label, n);
	}
}

int main(int argc, char ** argv) {
	(void)argc;
	(void)argv;

	ClSetup s = clSetup();

	std::mt19937_64 rng(0x5EED0C7EA7E2BEEFULL);
	const size_t N = 200000;

	runCase(s, "k_keccak_create2_ab",
		"CREATE2 85-byte preimage (sha3_keccakf_create2)", N, rng);

	clReleaseProgram(s.program);
	clReleaseCommandQueue(s.queue);
	clReleaseContext(s.context);

	std::printf("\n%s\n", g_failures ? "FAILED" : "create2 keccak specialisation is bit-exact");
	return g_failures ? 1 : 0;
}
