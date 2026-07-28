/* bench_address_state.cpp
 * =======================
 * Whether what made profanity_create2 faster is worth anything to the addresses
 * a search actually spends its time on.
 *
 * profanity_address builds two hash states and only one of them was worth
 * rewriting. The contract half, built under --contract, gained 3.4% on an RTX
 * 4090 from being written as whole lanes at constant indices and now is. The
 * account half gained nothing there -- three runs at -0.05%, -0.21% and -0.01%
 * -- and was left as it was.
 *
 * Three variants, so both of those stay measurable: the function as it was, the
 * function as it ships, and the function with the account half rewritten too.
 * Timed with and without --contract, since only one of them touches it.
 *
 * Correctness first: every input is hashed both ways and the addresses compared.
 * A benchmark of a faster function that computes something else is worthless,
 * and the coordinates here are not points on the curve, which does not matter to
 * a hash but does mean nothing else would catch it.
 *
 * Build & run (see tests/Makefile):
 *   cd tests && make && cd .. && ./tests/bench_address_state.x64 [items] [iterations] [reps]
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "testutil.hpp"

struct BenchResult {
	double bestSeconds;
	std::vector<cl_uint> output;
};

static BenchResult benchVariant(const ClSetup & s, const char * const kernelName, cl_mem xBuf, cl_mem yBuf,
	const cl_uchar bContract, const size_t globalSize, const cl_uint iterations, const int reps)
{
	cl_int err;
	cl_kernel kernel = clCreateKernel(s.program, kernelName, &err);
	clCheck(err, "clCreateKernel");

	cl_mem outBuf = clCreateBuffer(s.context, CL_MEM_WRITE_ONLY, globalSize * sizeof(cl_uint), NULL, &err);
	clCheck(err, "clCreateBuffer(out)");

	clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &xBuf), "clSetKernelArg(0)");
	clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &yBuf), "clSetKernelArg(1)");
	clCheck(clSetKernelArg(kernel, 2, sizeof(cl_uchar), &bContract), "clSetKernelArg(2)");
	clCheck(clSetKernelArg(kernel, 3, sizeof(cl_uint), &iterations), "clSetKernelArg(3)");
	clCheck(clSetKernelArg(kernel, 4, sizeof(cl_mem), &outBuf), "clSetKernelArg(4)");

	BenchResult result;
	result.bestSeconds = -1;
	result.output.resize(globalSize);

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

// Every input hashed both ways; the addresses must match exactly.
static int checkEquivalence(const ClSetup & s, cl_mem xBuf, cl_mem yBuf, const cl_uchar bContract, const size_t count) {
	cl_int err;
	cl_mem bufs[3];
	for (int i = 0; i < 3; ++i) {
		bufs[i] = clCreateBuffer(s.context, CL_MEM_WRITE_ONLY, count * 5 * sizeof(cl_uint), NULL, &err);
		clCheck(err, "clCreateBuffer(out)");
	}

	cl_kernel kernel = clCreateKernel(s.program, "k_address_all", &err);
	clCheck(err, "clCreateKernel(k_address_all)");
	clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &xBuf), "clSetKernelArg(0)");
	clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &yBuf), "clSetKernelArg(1)");
	clCheck(clSetKernelArg(kernel, 2, sizeof(cl_uchar), &bContract), "clSetKernelArg(2)");
	for (int i = 0; i < 3; ++i) {
		clCheck(clSetKernelArg(kernel, 3 + i, sizeof(cl_mem), &bufs[i]), "clSetKernelArg(out)");
	}

	std::vector<cl_uint> out[3];
	clCheck(clEnqueueNDRangeKernel(s.queue, kernel, 1, NULL, &count, NULL, 0, NULL, NULL), "clEnqueueNDRangeKernel");
	for (int i = 0; i < 3; ++i) {
		out[i].resize(count * 5);
		clCheck(clEnqueueReadBuffer(s.queue, bufs[i], CL_TRUE, 0, out[i].size() * sizeof(cl_uint), out[i].data(), 0, NULL, NULL), "clEnqueueReadBuffer");
	}
	clCheck(clFinish(s.queue), "clFinish");

	clReleaseKernel(kernel);
	for (int i = 0; i < 3; ++i) {
		clReleaseMemObject(bufs[i]);
	}

	int bad = 0;
	for (size_t i = 0; i < count; ++i) {
		if (std::memcmp(&out[0][i * 5], &out[1][i * 5], 5 * sizeof(cl_uint)) != 0
			|| std::memcmp(&out[0][i * 5], &out[2][i * 5], 5 * sizeof(cl_uint)) != 0) {
			++bad;
		}
	}
	return bad;
}

static void runFor(const ClSetup & s, cl_mem xBuf, cl_mem yBuf, const cl_uchar bContract,
	const size_t globalSize, const cl_uint iterations, const int reps)
{
	const char * const what = bContract ? "contract address (--contract: two hashes, the second built in a loop)"
	                                    : "account address (one hash, already built at constant indices)";
	std::printf("%s\n", what);

	const int bad = checkEquivalence(s, xBuf, yBuf, bContract, globalSize);
	if (bad) {
		std::printf("  EQUIVALENCE FAILED: %d of %zu inputs hash differently\n\n", bad, globalSize);
		return;
	}

	const BenchResult plain = benchVariant(s, "bench_address_plain", xBuf, yBuf, bContract, globalSize, iterations, reps);
	const BenchResult shipped = benchVariant(s, "bench_address_shipped", xBuf, yBuf, bContract, globalSize, iterations, reps);
	const BenchResult lanes = benchVariant(s, "bench_address_lanes", xBuf, yBuf, bContract, globalSize, iterations, reps);

	if (std::memcmp(plain.output.data(), shipped.output.data(), globalSize * sizeof(cl_uint)) != 0
		|| std::memcmp(plain.output.data(), lanes.output.data(), globalSize * sizeof(cl_uint)) != 0) {
		std::printf("  CROSS-CHECK FAILED: the chained accumulators diverged\n\n");
		return;
	}

	const double total = (double)globalSize * iterations;
	std::printf("  plain    (as it was)                    %9.2f ms  (%8.2f MH/s)\n",
		plain.bestSeconds * 1e3, total / plain.bestSeconds / 1e6);
	std::printf("  shipped  (contract half as lanes)       %9.2f ms  (%8.2f MH/s)   %+.2f%%\n",
		shipped.bestSeconds * 1e3, total / shipped.bestSeconds / 1e6,
		(plain.bestSeconds / shipped.bestSeconds - 1) * 100);
	std::printf("  lanes    (both halves as lanes)         %9.2f ms  (%8.2f MH/s)   %+.2f%%\n\n",
		lanes.bestSeconds * 1e3, total / lanes.bestSeconds / 1e6,
		(plain.bestSeconds / lanes.bestSeconds - 1) * 100);
}

int main(int argc, char * * argv) {
	const size_t globalSize = argc > 1 ? (size_t)std::atol(argv[1]) : 4096;
	const cl_uint iterations = argc > 2 ? (cl_uint)std::atol(argv[2]) : 2000;
	const int reps = argc > 3 ? std::atoi(argv[3]) : 5;

	std::mt19937_64 rng(0xADD5E55);
	std::vector<mp_number> x(globalSize), y(globalSize);
	for (size_t i = 0; i < globalSize; ++i) {
		for (int j = 0; j < MP_NWORDS; ++j) {
			x[i].d[j] = (cl_uint)rng();
			y[i].d[j] = (cl_uint)rng();
		}
	}

	const ClSetup s = clSetup();

	cl_int err;
	cl_mem xBuf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, globalSize * sizeof(mp_number), x.data(), &err);
	clCheck(err, "clCreateBuffer(x)");
	cl_mem yBuf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, globalSize * sizeof(mp_number), y.data(), &err);
	clCheck(err, "clCreateBuffer(y)");

	std::printf("Work items: %zu, chained addresses per item: %u, repetitions: %d (best time taken)\n\n",
		globalSize, iterations, reps);

	runFor(s, xBuf, yBuf, 0, globalSize, iterations, reps);
	runFor(s, xBuf, yBuf, 1, globalSize, iterations, reps);

	clReleaseMemObject(xBuf);
	clReleaseMemObject(yBuf);
	return 0;
}
