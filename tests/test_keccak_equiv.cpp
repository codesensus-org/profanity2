// Direct keccak equivalence test.
//
// Runs the stock sha3_keccakf and the sparse-first-round specialisations
// (sha3_keccakf_eoa, sha3_keccakf_contract) on the SAME input state and
// asserts byte-identical 200-byte output over many random inputs that respect
// the real absorb layout (the lanes the specialisation assumes zero are set to
// zero; every other lane is randomised, which is strictly stronger than the
// production call sites since those lanes carry fixed pad constants there).
//
// This is the check the existing suite does NOT provide: roundtest/scoring use
// the same keccak on both sides (a bug cancels) and test_create2 only exercises
// the create2 keccak, never the EOA path.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "testutil.hpp"

static int g_failures = 0;

// Lane indices (st[i], ulong lanes) that each specialisation RELIES UPON being
// zero for its round-0 THETA. Everything else is randomised by the test.
static const int kEoaZeroLanes[] = {
	9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24
};
static const int kContractZeroLanes[] = {
	3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24
};

static bool isZeroLane(const int lane, const int * const zeros, const size_t n) {
	for (size_t i = 0; i < n; ++i) {
		if (zeros[i] == lane) {
			return true;
		}
	}
	return false;
}

// Build N input states of 25 lanes each, zeroing the assumed-zero lanes and
// randomising the rest. The first two states use the exact production layout so
// the realistic case is always covered explicitly.
static std::vector<cl_ulong> buildInputs(
		const size_t n, const int * const zeros, const size_t nZeros,
		const bool eoa, std::mt19937_64 & rng) {
	std::vector<cl_ulong> in(n * 25, 0);

	for (size_t item = 0; item < n; ++item) {
		cl_ulong * const st = &in[item * 25];
		for (int lane = 0; lane < 25; ++lane) {
			if (isZeroLane(lane, zeros, nZeros)) {
				st[lane] = 0;
			} else {
				st[lane] = (cl_ulong)rng();
			}
		}

		if (item == 0) {
			// Exact production pre-padding layout.
			if (eoa) {
				st[8] = 0x0000000000000001ULL; // keccak pad start (byte 64)
				st[16] = 0;                    // pad end applied inside keccak
			} else {
				// contract: st[2] high bytes carry nonce 0x80 + pad 0x01,
				// low 48 bits are address-derived (kept random here).
				st[2] = (st[2] & 0x0000ffffffffffffULL) | 0x0180000000000000ULL;
				st[16] = 0;
			}
		} else if (item == 1) {
			// All-zero message body (still respects the zero lanes).
			for (int lane = 0; lane < 25; ++lane) {
				if (!isZeroLane(lane, zeros, nZeros)) {
					st[lane] = 0;
				}
			}
			if (eoa) {
				st[8] = 0x0000000000000001ULL;
			} else {
				st[2] = 0x0180000000000000ULL;
			}
		}
	}
	return in;
}

static void runCase(const ClSetup & s, const char * const kernelName,
		const char * const label, const int * const zeros,
		const size_t nZeros, const bool eoa, const size_t n,
		std::mt19937_64 & rng) {
	const std::vector<cl_ulong> in = buildInputs(n, zeros, nZeros, eoa, rng);
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

	std::mt19937_64 rng(0xC0FFEE1234ABCDEFULL);
	const size_t N = 200000;

	runCase(s, "k_keccak_eoa_ab", "EOA 64-byte preimage (sha3_keccakf_eoa)",
		kEoaZeroLanes, sizeof(kEoaZeroLanes) / sizeof(int), true, N, rng);
	runCase(s, "k_keccak_contract_ab", "contract preimage (sha3_keccakf_contract)",
		kContractZeroLanes, sizeof(kContractZeroLanes) / sizeof(int), false, N, rng);

	clReleaseProgram(s.program);
	clReleaseCommandQueue(s.queue);
	clReleaseContext(s.context);

	std::printf("\n%s\n", g_failures ? "FAILED" : "keccak specialisations are bit-exact");
	return g_failures ? 1 : 0;
}
