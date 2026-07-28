/* bench_mod_mul_ptx.cpp
 * =====================
 * Benchmark: mp_mod_mul built out of the portable multiprecision routines
 * against the same thing built out of the inline-PTX ones.
 *
 * Each work item performs a dependency-chained sequence of modular multiplications
 * x = x * y (mod p) using the real kernels compiled from profanity.cl +
 * tests/harness.cl. The chained data dependency prevents the compiler from
 * eliminating work.
 *
 * Both variants are also cross-checked: starting from identical inputs, their
 * final outputs must be bit-exact identical. That is a weaker check than
 * test_correctness_ptx does — it says nothing about which inputs were exercised
 * — but a benchmark whose two sides compute different things is measuring
 * nothing, so it is worth having here too.
 *
 * Only NVIDIA's OpenCL frontend compiles the PTX routines, so on any other
 * device this prints what it found and exits without a number. There is nothing
 * to compare, not a failure.
 *
 * Build & run (see tests/Makefile):
 *   cd tests && make && ./bench_mod_mul_ptx.x64 [global_size] [iterations] [repetitions]
 */

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "testutil.hpp"

struct BenchResult {
	double bestSeconds;
	std::vector<mp_number> output;
};

static BenchResult benchVariant(const ClSetup & s, const char * const kernelName,
	const std::vector<mp_number> & x, const std::vector<mp_number> & y,
	const uint32_t iterations, const int reps)
{
	const size_t count = x.size();
	cl_int err;
	cl_kernel kernel = clCreateKernel(s.program, kernelName, &err);
	clCheck(err, "clCreateKernel");

	cl_mem yBuf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * sizeof(mp_number), (void *)y.data(), &err);
	clCheck(err, "clCreateBuffer(y)");

	BenchResult result;
	result.bestSeconds = -1;
	result.output.resize(count);

	// reps timed runs + one warmup run (which also produces the output used
	// for the cross-check)
	for (int rep = 0; rep <= reps; ++rep) {
		cl_mem xBuf = clCreateBuffer(s.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, count * sizeof(mp_number), (void *)x.data(), &err);
		clCheck(err, "clCreateBuffer(x)");
		clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &xBuf), "clSetKernelArg(0)");
		clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &yBuf), "clSetKernelArg(1)");
		clCheck(clSetKernelArg(kernel, 2, sizeof(uint32_t), &iterations), "clSetKernelArg(2)");
		clCheck(clFinish(s.queue), "clFinish(pre)");

		const auto t0 = std::chrono::steady_clock::now();
		clCheck(clEnqueueNDRangeKernel(s.queue, kernel, 1, NULL, &count, NULL, 0, NULL, NULL), "clEnqueueNDRangeKernel");
		clCheck(clFinish(s.queue), "clFinish");
		const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

		if (rep == 0) {
			clCheck(clEnqueueReadBuffer(s.queue, xBuf, CL_TRUE, 0, count * sizeof(mp_number), result.output.data(), 0, NULL, NULL), "clEnqueueReadBuffer");
			clCheck(clFinish(s.queue), "clFinish(read)");
		} else if (result.bestSeconds < 0 || seconds < result.bestSeconds) {
			result.bestSeconds = seconds;
		}

		clReleaseMemObject(xBuf);
	}

	clReleaseMemObject(yBuf);
	clReleaseKernel(kernel);
	return result;
}

int main(int argc, char * * argv) {
	const size_t globalSize = argc > 1 ? (size_t)std::atol(argv[1]) : 4096;
	const uint32_t iterations = argc > 2 ? (uint32_t)std::atol(argv[2]) : 2000;
	const int reps = argc > 3 ? std::atoi(argv[3]) : 5;

	std::mt19937_64 rng(0xBEEF);
	std::vector<mp_number> x(globalSize), y(globalSize);
	for (size_t i = 0; i < globalSize; ++i) {
		for (int j = 0; j < MP_NWORDS; ++j) {
			x[i].d[j] = (uint32_t)rng();
			y[i].d[j] = (uint32_t)rng();
		}
	}

	const ClSetup s = clSetup();

	if (!hasPtxKernels(s)) {
		std::printf("This device has no inline-PTX kernels (not an NVIDIA OpenCL device); nothing to compare.\n");
		return 0;
	}

	std::printf("Work items: %zu, chained mod-muls per item: %u, repetitions: %d (best time taken)\n",
		globalSize, iterations, reps);

	const BenchResult portableResult = benchVariant(s, "bench_mod_mul_portable", x, y, iterations, reps);
	const BenchResult ptxResult = benchVariant(s, "bench_mod_mul_ptx", x, y, iterations, reps);

	size_t mismatches = 0;
	for (size_t i = 0; i < globalSize; ++i) {
		if (!mpEqual(portableResult.output[i], ptxResult.output[i])) {
			++mismatches;
		}
	}
	if (mismatches) {
		std::printf("CROSS-CHECK FAILED: outputs differ in %zu/%zu work items\n", mismatches, globalSize);
		return 1;
	}
	std::printf("Cross-check: outputs of portable and PTX variants are bit-exact identical (%zu work items x %u chained mod-muls).\n\n",
		globalSize, iterations);

	const double totalMuls = (double)globalSize * iterations;
	std::printf("  portable:  %9.2f ms  (%8.2f Mmul/s)\n", portableResult.bestSeconds * 1e3, totalMuls / portableResult.bestSeconds / 1e6);
	std::printf("  inline PTX:%9.2f ms  (%8.2f Mmul/s)\n", ptxResult.bestSeconds * 1e3, totalMuls / ptxResult.bestSeconds / 1e6);
	std::printf("  speedup: x%.3f (%+.1f%%)\n", portableResult.bestSeconds / ptxResult.bestSeconds,
		(portableResult.bestSeconds / ptxResult.bestSeconds - 1) * 100);

	std::printf("\nThis is mp_mod_mul in isolation. What a search gains is less: point\n"
		"addition is not only multiplication, and the extra registers the PTX\n"
		"version holds live can cost occupancy in profanity_iterate. Measure\n"
		"there too, with --benchmark.\n");
	return 0;
}
