/* test_correctness_ptx.cpp
 * ========================
 * Correctness and equivalence tests for the inline-PTX multiprecision routines.
 *
 * profanity.cl carries two implementations of the two innermost multiprecision
 * routines: a portable one, and — on NVIDIA only, where the OpenCL frontend
 * accepts inline PTX — one that keeps its carries in the hardware flag. The
 * second is what the rest of the file gets on that vendor, so it has to compute
 * exactly what the first does, on every input and not merely on most.
 *
 * Checks, in order:
 *
 * 1. mp_mul_word_add_extra_portable against a host-side big-integer reference,
 *    on every device. This is the path every non-NVIDIA card takes, and it is
 *    also what tells the two checks below apart from a reference that is itself
 *    wrong.
 * 2. mp_mul_word_add_extra_ptx bit-exact against both the portable version and
 *    that reference — including the returned overflow bit, which is the part of
 *    the contract the two implementations arrive at by different routes.
 * 3. mp_mul_mod_word_sub_ptx likewise, against the portable version and against
 *    the reference already used by test_correctness_pr49.
 * 4. mp_mod_mul built out of each pair, bit-exact against each other and
 *    congruent mod p for in-domain inputs.
 *
 * On a device without the PTX kernels — anything that is not NVIDIA — checks 2
 * through 4 report SKIPPED and the run still passes; there is nothing there to
 * be wrong. Run it on an NVIDIA card before trusting the PTX path.
 *
 * Build & run (see tests/Makefile):
 *   cd tests && make && ./test_correctness_ptx.x64 [num_random_cases]
 */

#include <cstdio>
#include <random>
#include <vector>

#include "testutil.hpp"

static std::mt19937_64 g_rng(0x9E3779B9);

static uint32_t randomWord() {
	return (uint32_t)g_rng();
}

static mp_number randomNumber() {
	mp_number n;
	for (int i = 0; i < MP_NWORDS; ++i) {
		n.d[i] = randomWord();
	}
	return n;
}

/* ------------------------------------------------------------------------ */
/* mp_mul_word_add_extra                                                     */
/* ------------------------------------------------------------------------ */

struct MulAddCase {
	mp_number r;
	mp_number a;
	uint32_t w;
	uint32_t extra;
};

struct MulAddResult {
	std::vector<mp_number> r;
	std::vector<uint32_t> extra;
	std::vector<uint32_t> overflow;
};

static std::vector<MulAddCase> genMulAddCases(const size_t numRandom) {
	const mp_number zero = { { 0 } };
	const mp_number one = { { 1 } };
	const mp_number allOnes = { { 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff } };
	const mp_number highBit = { { 0, 0, 0, 0, 0, 0, 0, 0x80000000 } };
	const mp_number pattern = { { 0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0, 0xffffffff, 0 } };

	// The interesting inputs are the ones that make a carry leave the top of the
	// nine-word accumulator, which is where the two implementations differ most:
	// the portable one folds it into a single add of cM + cA, the PTX one lets
	// two separate chains each carry into word eight.
	const mp_number edge[] = { zero, one, allOnes, g_p, highBit, pattern };
	const uint32_t edgeW[] = { 0, 1, 2, 977, 0x3D1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF };
	const uint32_t edgeExtra[] = { 0, 1, 0x7FFFFFFF, 0xFFFFFFFE, 0xFFFFFFFF };

	std::vector<MulAddCase> cases;
	for (const mp_number & r : edge) {
		for (const mp_number & a : edge) {
			for (const uint32_t w : edgeW) {
				for (const uint32_t e : edgeExtra) {
					cases.push_back({ r, a, w, e });
				}
			}
		}
	}
	for (size_t i = 0; i < numRandom; ++i) {
		cases.push_back({ randomNumber(), randomNumber(), randomWord(), randomWord() });
	}
	return cases;
}

static MulAddResult runMulAddKernel(const ClSetup & s, const char * const kernelName, const std::vector<MulAddCase> & cases) {
	const size_t count = cases.size();
	MulAddResult out;
	out.r.resize(count);
	out.extra.resize(count);
	out.overflow.resize(count);

	std::vector<mp_number> a(count);
	std::vector<uint32_t> w(count);
	for (size_t i = 0; i < count; ++i) {
		out.r[i] = cases[i].r;
		out.extra[i] = cases[i].extra;
		a[i] = cases[i].a;
		w[i] = cases[i].w;
	}

	cl_int err;
	cl_mem rBuf = clCreateBuffer(s.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, count * sizeof(mp_number), out.r.data(), &err);
	clCheck(err, "clCreateBuffer(r)");
	cl_mem aBuf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * sizeof(mp_number), a.data(), &err);
	clCheck(err, "clCreateBuffer(a)");
	cl_mem wBuf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * sizeof(uint32_t), w.data(), &err);
	clCheck(err, "clCreateBuffer(w)");
	cl_mem eBuf = clCreateBuffer(s.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, count * sizeof(uint32_t), out.extra.data(), &err);
	clCheck(err, "clCreateBuffer(extra)");
	cl_mem oBuf = clCreateBuffer(s.context, CL_MEM_WRITE_ONLY, count * sizeof(uint32_t), NULL, &err);
	clCheck(err, "clCreateBuffer(overflow)");

	cl_kernel kernel = clCreateKernel(s.program, kernelName, &err);
	clCheck(err, "clCreateKernel");
	clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &rBuf), "clSetKernelArg(0)");
	clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &aBuf), "clSetKernelArg(1)");
	clCheck(clSetKernelArg(kernel, 2, sizeof(cl_mem), &wBuf), "clSetKernelArg(2)");
	clCheck(clSetKernelArg(kernel, 3, sizeof(cl_mem), &eBuf), "clSetKernelArg(3)");
	clCheck(clSetKernelArg(kernel, 4, sizeof(cl_mem), &oBuf), "clSetKernelArg(4)");

	clCheck(clEnqueueNDRangeKernel(s.queue, kernel, 1, NULL, &count, NULL, 0, NULL, NULL), "clEnqueueNDRangeKernel");
	clCheck(clEnqueueReadBuffer(s.queue, rBuf, CL_TRUE, 0, count * sizeof(mp_number), out.r.data(), 0, NULL, NULL), "clEnqueueReadBuffer(r)");
	clCheck(clEnqueueReadBuffer(s.queue, eBuf, CL_TRUE, 0, count * sizeof(uint32_t), out.extra.data(), 0, NULL, NULL), "clEnqueueReadBuffer(extra)");
	clCheck(clEnqueueReadBuffer(s.queue, oBuf, CL_TRUE, 0, count * sizeof(uint32_t), out.overflow.data(), 0, NULL, NULL), "clEnqueueReadBuffer(overflow)");
	clCheck(clFinish(s.queue), "clFinish");

	clReleaseKernel(kernel);
	clReleaseMemObject(rBuf);
	clReleaseMemObject(aBuf);
	clReleaseMemObject(wBuf);
	clReleaseMemObject(eBuf);
	clReleaseMemObject(oBuf);
	return out;
}

// Compares one kernel's output against the host reference. Returns failures.
static size_t checkMulAddAgainstReference(const char * const label, const std::vector<MulAddCase> & cases, const MulAddResult & got) {
	size_t failures = 0;
	for (size_t i = 0; i < cases.size(); ++i) {
		mp_number expectR = cases[i].r;
		uint32_t expectExtra = cases[i].extra;
		const uint32_t expectOverflow = refMulWordAddExtra(expectR, cases[i].a, cases[i].w, expectExtra);

		if (!mpEqual(got.r[i], expectR) || got.extra[i] != expectExtra || got.overflow[i] != expectOverflow) {
			if (++failures <= 5) {
				std::printf("  %s REFERENCE MISMATCH case %zu: r=%s a=%s w=%#010x extra=%#010x\n"
					"    got      r=%s extra=%#010x overflow=%u\n"
					"    expected r=%s extra=%#010x overflow=%u\n",
					label, i, mpToHex(cases[i].r).c_str(), mpToHex(cases[i].a).c_str(), cases[i].w, cases[i].extra,
					mpToHex(got.r[i]).c_str(), got.extra[i], got.overflow[i],
					mpToHex(expectR).c_str(), expectExtra, expectOverflow);
			}
		}
	}
	return failures;
}

static size_t testMulWordAddExtra(const ClSetup & s, const bool ptx, const size_t numRandom) {
	const std::vector<MulAddCase> cases = genMulAddCases(numRandom);
	const MulAddResult outPortable = runMulAddKernel(s, "k_mul_word_add_extra_portable", cases);

	size_t failures = checkMulAddAgainstReference("portable", cases, outPortable);
	std::printf("mp_mul_word_add_extra (portable): %zu cases against big-int reference: %s\n",
		cases.size(), failures == 0 ? "OK" : "FAILED");

	if (!ptx) {
		std::printf("mp_mul_word_add_extra (ptx): SKIPPED (device has no inline-PTX kernels)\n");
		return failures;
	}

	const MulAddResult outPtx = runMulAddKernel(s, "k_mul_word_add_extra_ptx", cases);

	size_t ptxFailures = 0;
	for (size_t i = 0; i < cases.size(); ++i) {
		if (!mpEqual(outPtx.r[i], outPortable.r[i]) || outPtx.extra[i] != outPortable.extra[i] || outPtx.overflow[i] != outPortable.overflow[i]) {
			if (++ptxFailures <= 5) {
				std::printf("  EQUIVALENCE MISMATCH case %zu: r=%s a=%s w=%#010x extra=%#010x\n"
					"    portable r=%s extra=%#010x overflow=%u\n"
					"    ptx      r=%s extra=%#010x overflow=%u\n",
					i, mpToHex(cases[i].r).c_str(), mpToHex(cases[i].a).c_str(), cases[i].w, cases[i].extra,
					mpToHex(outPortable.r[i]).c_str(), outPortable.extra[i], outPortable.overflow[i],
					mpToHex(outPtx.r[i]).c_str(), outPtx.extra[i], outPtx.overflow[i]);
			}
		}
	}
	ptxFailures += checkMulAddAgainstReference("ptx", cases, outPtx);

	std::printf("mp_mul_word_add_extra (ptx): %zu cases portable-vs-ptx bit-exact + big-int reference: %s\n",
		cases.size(), ptxFailures == 0 ? "OK" : "FAILED");
	return failures + ptxFailures;
}

/* ------------------------------------------------------------------------ */
/* mp_mul_mod_word_sub                                                       */
/* ------------------------------------------------------------------------ */

struct WordSubCase {
	mp_number r;
	uint32_t w;
	uint32_t withModHigher;
};

static std::vector<WordSubCase> genWordSubCases(const size_t numRandom) {
	const mp_number zero = { { 0 } };
	const mp_number one = { { 1 } };
	const mp_number allOnes = { { 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff } };
	const mp_number highBit = { { 0, 0, 0, 0, 0, 0, 0, 0x80000000 } };

	const mp_number edgeR[] = { zero, one, allOnes, g_p, highBit };
	const uint32_t edgeW[] = { 0, 1, 976, 977, 978, 0x3D1, 0x80000000, 0xFFFFFC2F, 0xFFFFFFFF };

	std::vector<WordSubCase> cases;
	for (const mp_number & r : edgeR) {
		for (const uint32_t w : edgeW) {
			for (uint32_t wmh = 0; wmh <= 1; ++wmh) {
				cases.push_back({ r, w, wmh });
			}
		}
	}
	for (size_t i = 0; i < numRandom; ++i) {
		cases.push_back({ randomNumber(), randomWord(), (uint32_t)(g_rng() & 1) });
	}
	return cases;
}

static std::vector<mp_number> runWordSubKernel(const ClSetup & s, const char * const kernelName, const std::vector<WordSubCase> & cases) {
	const size_t count = cases.size();
	std::vector<mp_number> r(count);
	std::vector<uint32_t> w(count), wmh(count);
	for (size_t i = 0; i < count; ++i) {
		r[i] = cases[i].r;
		w[i] = cases[i].w;
		wmh[i] = cases[i].withModHigher;
	}

	cl_int err;
	cl_mem rBuf = clCreateBuffer(s.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, count * sizeof(mp_number), r.data(), &err);
	clCheck(err, "clCreateBuffer(r)");
	cl_mem wBuf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * sizeof(uint32_t), w.data(), &err);
	clCheck(err, "clCreateBuffer(w)");
	cl_mem mBuf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * sizeof(uint32_t), wmh.data(), &err);
	clCheck(err, "clCreateBuffer(wmh)");

	cl_kernel kernel = clCreateKernel(s.program, kernelName, &err);
	clCheck(err, "clCreateKernel");
	clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &rBuf), "clSetKernelArg(0)");
	clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &wBuf), "clSetKernelArg(1)");
	clCheck(clSetKernelArg(kernel, 2, sizeof(cl_mem), &mBuf), "clSetKernelArg(2)");

	clCheck(clEnqueueNDRangeKernel(s.queue, kernel, 1, NULL, &count, NULL, 0, NULL, NULL), "clEnqueueNDRangeKernel");
	clCheck(clEnqueueReadBuffer(s.queue, rBuf, CL_TRUE, 0, count * sizeof(mp_number), r.data(), 0, NULL, NULL), "clEnqueueReadBuffer");
	clCheck(clFinish(s.queue), "clFinish");

	clReleaseKernel(kernel);
	clReleaseMemObject(rBuf);
	clReleaseMemObject(wBuf);
	clReleaseMemObject(mBuf);
	return r;
}

static size_t testMulModWordSub(const ClSetup & s, const bool ptx, const size_t numRandom) {
	if (!ptx) {
		std::printf("mp_mul_mod_word_sub (ptx): SKIPPED (device has no inline-PTX kernels)\n");
		return 0;
	}

	const std::vector<WordSubCase> cases = genWordSubCases(numRandom);
	const std::vector<mp_number> outPortable = runWordSubKernel(s, "k_mul_mod_word_sub_portable", cases);
	const std::vector<mp_number> outPtx = runWordSubKernel(s, "k_mul_mod_word_sub_ptx", cases);

	size_t failures = 0;
	for (size_t i = 0; i < cases.size(); ++i) {
		if (!mpEqual(outPtx[i], outPortable[i])) {
			if (++failures <= 5) {
				std::printf("  EQUIVALENCE MISMATCH case %zu: r=%s w=%#010x wmh=%u\n    portable %s\n    ptx      %s\n",
					i, mpToHex(cases[i].r).c_str(), cases[i].w, cases[i].withModHigher,
					mpToHex(outPortable[i]).c_str(), mpToHex(outPtx[i]).c_str());
			}
			continue;
		}
		const mp_number expect = refMulModWordSub(cases[i].r, cases[i].w, cases[i].withModHigher != 0);
		if (!mpEqual(outPtx[i], expect)) {
			if (++failures <= 5) {
				std::printf("  REFERENCE MISMATCH case %zu: r=%s w=%#010x wmh=%u\n    got      %s\n    expected %s\n",
					i, mpToHex(cases[i].r).c_str(), cases[i].w, cases[i].withModHigher,
					mpToHex(outPtx[i]).c_str(), mpToHex(expect).c_str());
			}
		}
	}

	std::printf("mp_mul_mod_word_sub (ptx): %zu cases portable-vs-ptx bit-exact + big-int reference: %s\n",
		cases.size(), failures == 0 ? "OK" : "FAILED");
	return failures;
}

/* ------------------------------------------------------------------------ */
/* mp_mod_mul                                                                */
/* ------------------------------------------------------------------------ */

struct ModMulCase {
	mp_number x;
	mp_number y;
};

static std::vector<ModMulCase> genModMulCases(const size_t numRandom) {
	const mp_number zero = { { 0 } };
	const mp_number one = { { 1 } };
	const mp_number two = { { 2 } };
	const mp_number allOnes = { { 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff } };
	mp_number pPlusOne = g_p; pPlusOne.d[0] += 1;
	mp_number pMinusOne = g_p; pMinusOne.d[0] -= 1;
	const mp_number highBit = { { 0, 0, 0, 0, 0, 0, 0, 0x80000000 } };
	const mp_number w977 = { { 977 } };
	const mp_number pmod = { { 0x000003D1, 1 } };

	const mp_number edge[] = { zero, one, two, pMinusOne, g_p, pPlusOne, allOnes, highBit, w977, pmod };

	std::vector<ModMulCase> cases;
	for (const mp_number & x : edge) {
		for (const mp_number & y : edge) {
			cases.push_back({ x, y });
		}
	}
	for (size_t i = 0; i < numRandom; ++i) {
		cases.push_back({ randomNumber(), randomNumber() });
	}
	return cases;
}

static std::vector<mp_number> runModMulKernel(const ClSetup & s, const char * const kernelName, const std::vector<ModMulCase> & cases) {
	const size_t count = cases.size();
	std::vector<mp_number> x(count), y(count), out(count);
	for (size_t i = 0; i < count; ++i) {
		x[i] = cases[i].x;
		y[i] = cases[i].y;
	}

	cl_int err;
	cl_mem xBuf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * sizeof(mp_number), x.data(), &err);
	clCheck(err, "clCreateBuffer(x)");
	cl_mem yBuf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * sizeof(mp_number), y.data(), &err);
	clCheck(err, "clCreateBuffer(y)");
	cl_mem rBuf = clCreateBuffer(s.context, CL_MEM_WRITE_ONLY, count * sizeof(mp_number), NULL, &err);
	clCheck(err, "clCreateBuffer(out)");

	cl_kernel kernel = clCreateKernel(s.program, kernelName, &err);
	clCheck(err, "clCreateKernel");
	clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &xBuf), "clSetKernelArg(0)");
	clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &yBuf), "clSetKernelArg(1)");
	clCheck(clSetKernelArg(kernel, 2, sizeof(cl_mem), &rBuf), "clSetKernelArg(2)");

	clCheck(clEnqueueNDRangeKernel(s.queue, kernel, 1, NULL, &count, NULL, 0, NULL, NULL), "clEnqueueNDRangeKernel");
	clCheck(clEnqueueReadBuffer(s.queue, rBuf, CL_TRUE, 0, count * sizeof(mp_number), out.data(), 0, NULL, NULL), "clEnqueueReadBuffer");
	clCheck(clFinish(s.queue), "clFinish");

	clReleaseKernel(kernel);
	clReleaseMemObject(xBuf);
	clReleaseMemObject(yBuf);
	clReleaseMemObject(rBuf);
	return out;
}

static size_t testModMul(const ClSetup & s, const bool ptx, const size_t numRandom) {
	if (!ptx) {
		std::printf("mp_mod_mul (ptx): SKIPPED (device has no inline-PTX kernels)\n");
		return 0;
	}

	const std::vector<ModMulCase> cases = genModMulCases(numRandom);
	const std::vector<mp_number> outPortable = runModMulKernel(s, "k_mod_mul_portable", cases);
	const std::vector<mp_number> outPtx = runModMulKernel(s, "k_mod_mul_ptx", cases);

	size_t failures = 0;
	size_t congruenceChecked = 0;
	for (size_t i = 0; i < cases.size(); ++i) {
		if (!mpEqual(outPtx[i], outPortable[i])) {
			if (++failures <= 5) {
				std::printf("  EQUIVALENCE MISMATCH case %zu: x=%s y=%s\n    portable %s\n    ptx      %s\n",
					i, mpToHex(cases[i].x).c_str(), mpToHex(cases[i].y).c_str(),
					mpToHex(outPortable[i]).c_str(), mpToHex(outPtx[i]).c_str());
			}
			continue;
		}

		if (!mpGte(cases[i].x, g_p) && !mpGte(cases[i].y, g_p)) {
			++congruenceChecked;
			const mp_number got = refModP256(outPtx[i]);
			const mp_number expect = refMulModP(cases[i].x, cases[i].y);
			if (!mpEqual(got, expect)) {
				if (++failures <= 5) {
					std::printf("  CONGRUENCE MISMATCH case %zu: x=%s y=%s\n    got%%p    %s\n    x*y%%p    %s\n",
						i, mpToHex(cases[i].x).c_str(), mpToHex(cases[i].y).c_str(),
						mpToHex(got).c_str(), mpToHex(expect).c_str());
				}
			}
		}
	}

	std::printf("mp_mod_mul (ptx): %zu cases portable-vs-ptx bit-exact, %zu in-domain cases congruent mod p: %s\n",
		cases.size(), congruenceChecked, failures == 0 ? "OK" : "FAILED");
	return failures;
}

int main(int argc, char * * argv) {
	const size_t numRandom = argc > 1 ? (size_t)std::atol(argv[1]) : 100000;

	const ClSetup s = clSetup();
	const bool ptx = hasPtxKernels(s);

	std::printf("Inline-PTX multiprecision routines: %s\n\n", ptx ? "present" : "ABSENT (not an NVIDIA OpenCL device)");

	size_t failures = 0;
	failures += testMulWordAddExtra(s, ptx, numRandom);
	failures += testMulModWordSub(s, ptx, numRandom);
	failures += testModMul(s, ptx, numRandom / 10 > 1000 ? numRandom / 10 : 1000);

	if (failures) {
		std::printf("\nFAILED: %zu total failures\n", failures);
		return 1;
	}

	if (!ptx) {
		std::printf("\nPassed, but the PTX path was not exercised. Run this on an NVIDIA device before trusting it.\n");
		return 0;
	}

	std::printf("\nAll tests passed.\n");
	return 0;
}
