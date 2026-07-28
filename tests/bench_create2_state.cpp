/* bench_create2_state.cpp
 * =======================
 * Benchmark: what it costs to build a CREATE2 candidate's hash state three
 * different ways, with the same permutation run over all three.
 *
 *   plain   -- a zeroed state, the message copied in word by word, and the
 *              counter written through the byte member of the union
 *   lanes   -- the same, but written as lanes and with the counter shifted in
 *   shipped -- the same again with every store at an index the compiler knows
 *
 * The gap between the first two is nothing and the gap to the third is large,
 * which is the point: what matters is not how many stores there are or which
 * member of the union they go through, but whether the state can live in
 * twenty-five registers for the whole permutation or has to sit in memory.
 *
 * Measured at +7% on an RTX 4090 and around a third on a CPU device.
 *
 * Each work item hashes `iterations` counters in a chain, each one folded into
 * the next, so nothing can be hoisted out of the loop. All three are
 * cross-checked: from the same inputs they must produce the same accumulator.
 *
 * Build & run (see tests/Makefile):
 *   cd tests && make && cd .. && ./tests/bench_create2_state.x64 [items] [iterations] [reps]
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "testutil.hpp"

struct BenchResult {
	double bestSeconds;
	std::vector<cl_uint> output;
};

static BenchResult benchVariant(const ClSetup & s, const char * const kernelName, cl_mem fixedBuf,
	const size_t globalSize, const cl_uint iterations, const int reps)
{
	cl_int err;
	cl_kernel kernel = clCreateKernel(s.program, kernelName, &err);
	clCheck(err, "clCreateKernel");

	cl_mem outBuf = clCreateBuffer(s.context, CL_MEM_WRITE_ONLY, globalSize * sizeof(cl_uint), NULL, &err);
	clCheck(err, "clCreateBuffer(out)");

	const cl_ulong counterBase = 0x0123456789abcdefULL;

	clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &fixedBuf), "clSetKernelArg(0)");
	clCheck(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &counterBase), "clSetKernelArg(1)");
	clCheck(clSetKernelArg(kernel, 2, sizeof(cl_uint), &iterations), "clSetKernelArg(2)");
	clCheck(clSetKernelArg(kernel, 3, sizeof(cl_mem), &outBuf), "clSetKernelArg(3)");

	BenchResult result;
	result.bestSeconds = -1;
	result.output.resize(globalSize);

	// reps timed runs plus one warmup, which also produces the output used for
	// the cross-check
	for (int rep = 0; rep <= reps; ++rep) {
		clCheck(clFinish(s.queue), "clFinish(pre)");

		const auto t0 = std::chrono::steady_clock::now();
		clCheck(clEnqueueNDRangeKernel(s.queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL), "clEnqueueNDRangeKernel");
		clCheck(clFinish(s.queue), "clFinish");
		const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

		if (rep == 0) {
			clCheck(clEnqueueReadBuffer(s.queue, outBuf, CL_TRUE, 0, globalSize * sizeof(cl_uint), result.output.data(), 0, NULL, NULL), "clEnqueueReadBuffer");
			clCheck(clFinish(s.queue), "clFinish(read)");
		} else if (result.bestSeconds < 0 || seconds < result.bestSeconds) {
			result.bestSeconds = seconds;
		}
	}

	clReleaseMemObject(outBuf);
	clReleaseKernel(kernel);
	return result;
}

int main(int argc, char * * argv) {
	const size_t globalSize = argc > 1 ? (size_t)std::atol(argv[1]) : 4096;
	const cl_uint iterations = argc > 2 ? (cl_uint)std::atol(argv[2]) : 2000;
	const int reps = argc > 3 ? std::atoi(argv[3]) : 5;

	// Something with no zero bytes anywhere, so that a lane wrongly held
	// constant carries a value rather than nothing.
	create2 fixed;
	for (int i = 0; i < 20; ++i) {
		fixed.factory[i] = (cl_uchar)(0x31 + i);
		fixed.caller[i] = (cl_uchar)(0x91 - i);
	}
	for (int i = 0; i < 32; ++i) {
		fixed.initCodeHash[i] = (cl_uchar)(0x47 + i * 3);
	}

	cl_uchar salt[32];
	for (int i = 0; i < 32; ++i) {
		salt[i] = (cl_uchar)(0x5b + i);
	}
	std::copy(fixed.caller, fixed.caller + 20, salt);

	cl_uint words[PROFANITY_CREATE2_WORDS];
	buildCreate2Template(words, fixed, salt);

	const ClSetup s = clSetup();

	cl_int err;
	cl_mem templateBuf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(words), words, &err);
	clCheck(err, "clCreateBuffer(template)");

	std::printf("Work items: %zu, chained hashes per item: %u, repetitions: %d (best time taken)\n",
		globalSize, iterations, reps);

	const BenchResult plain = benchVariant(s, "bench_create2_plain", templateBuf, globalSize, iterations, reps);
	const BenchResult lanes = benchVariant(s, "bench_create2_lanes", templateBuf, globalSize, iterations, reps);
	const BenchResult ub = benchVariant(s, "bench_create2_unrolled_bytes", templateBuf, globalSize, iterations, reps);
	const BenchResult shipped = benchVariant(s, "bench_create2_shipped", templateBuf, globalSize, iterations, reps);

	if (std::memcmp(plain.output.data(), shipped.output.data(), globalSize * sizeof(cl_uint)) != 0
		|| std::memcmp(plain.output.data(), lanes.output.data(), globalSize * sizeof(cl_uint)) != 0
		|| std::memcmp(plain.output.data(), ub.output.data(), globalSize * sizeof(cl_uint)) != 0) {
		std::printf("CROSS-CHECK FAILED: the variants did not hash the same thing\n");
		return 1;
	}
	std::printf("Cross-check: all four variants agree on every address (%zu work items x %u chained hashes).\n\n",
		globalSize, iterations);

	const double total = (double)globalSize * iterations;
	std::printf("  plain    (zeroed state, word copy, byte stores):  %9.2f ms  (%8.2f MH/s)\n",
		plain.bestSeconds * 1e3, total / plain.bestSeconds / 1e6);
	std::printf("  lanes    (lane stores, still through loops):      %9.2f ms  (%8.2f MH/s)  %+.2f%%\n",
		lanes.bestSeconds * 1e3, total / lanes.bestSeconds / 1e6, (plain.bestSeconds / lanes.bestSeconds - 1) * 100);
	std::printf("  bytes    (same shape as plain, loops written out):%9.2f ms  (%8.2f MH/s)  %+.2f%%\n",
		ub.bestSeconds * 1e3, total / ub.bestSeconds / 1e6, (plain.bestSeconds / ub.bestSeconds - 1) * 100);
	std::printf("  shipped  (lane stores at constant indices):       %9.2f ms  (%8.2f MH/s)  %+.2f%%\n",
		shipped.bestSeconds * 1e3, total / shipped.bestSeconds / 1e6, (plain.bestSeconds / shipped.bestSeconds - 1) * 100);
	std::printf("\n  dropping the union byte stores alone: %+.2f%%\n", (plain.bestSeconds / lanes.bestSeconds - 1) * 100);
	std::printf("  making every store index constant:    %+.2f%%\n", (lanes.bestSeconds / shipped.bestSeconds - 1) * 100);
	std::printf("  ... of which the lane form adds:      %+.2f%%\n", (ub.bestSeconds / shipped.bestSeconds - 1) * 100);

	clReleaseMemObject(templateBuf);
	return 0;
}
