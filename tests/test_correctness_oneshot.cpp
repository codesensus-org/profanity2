// Correctness tests for PATCH #3 (oneshot-reduction-modmul) and PATCH #6 (dedicated-mod-sqr).
//
// Assertions (mirroring the shipped harness's congruent-vs-bit-exact discipline):
//   (a) congruence : refMulModP(x,y) == refModP256(oneshot(x,y)) for ALL inputs, output < 2^256.
//   (b) bit-exact  : oneshot(x,y) == stock mp_mod_mul(x,y) for in-domain inputs (x<p && y<p).
//   (c) mod_sqr    : k_mod_sqr(x) == k_mod_mul_oneshot(x,x) bit-exact for in-domain x,
//                    and refModP256(sqr(x)) == refMulModP(x,x) congruent for ALL x.
//   (d) ptx path   : when present, *_ptx == *_portable bit-exact for ALL inputs.
//
// Note: for OUT-OF-DOMAIN operands (>= p) oneshot and stock are only congruent (they may pick
// different valid representatives) -- so bit-exactness is asserted only in-domain, exactly like
// the shipped test_correctness_ptx.cpp gates its congruence check.
#include <cstdio>
#include <random>
#include <vector>

#include "testutil.hpp"

static std::mt19937_64 g_rng(0x0DDBALL);
static uint32_t randomWord() { return (uint32_t)g_rng(); }
static mp_number randomNumber() {
	mp_number n;
	for (int i = 0; i < MP_NWORDS; ++i) n.d[i] = randomWord();
	return n;
}

static bool hasKernel(const ClSetup & s, const char * name) {
	cl_int err = CL_SUCCESS;
	cl_kernel k = clCreateKernel(s.program, name, &err);
	if (err != CL_SUCCESS) return false;
	clReleaseKernel(k);
	return true;
}

struct ModMulCase { mp_number x; mp_number y; };

static std::vector<ModMulCase> genModMulCases(const size_t numRandom) {
	const mp_number zero = { { 0 } };
	const mp_number one = { { 1 } };
	const mp_number two = { { 2 } };
	const mp_number allOnes = { { 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff } };
	mp_number pPlusOne = g_p; pPlusOne.d[0] += 1;
	mp_number pMinusOne = g_p; pMinusOne.d[0] -= 1;
	const mp_number highBit = { { 0, 0, 0, 0, 0, 0, 0, 0x80000000 } };
	const mp_number nearTop = { { 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xfffffffe } }; // 2^256-2^224-1-ish, > p
	const mp_number w977 = { { 977 } };
	const mp_number pmod = { { 0x000003D1, 1 } };   // c = 2^32 + 977
	// secp256k1 curve constants used by profanity.cl
	const mp_number Gx   = { { 0x16f81798, 0x59f2815b, 0x2dce28d9, 0x029bfcdb, 0xce870b07, 0x55a06295, 0xf9dcbbac, 0x79be667e } };
	const mp_number Gy   = { { 0xfb10d4b8, 0x9c47d08f, 0xa6855419, 0xfd17b448, 0x0e1108a8, 0x5da4fbfc, 0x26a3c465, 0x483ada77 } };
	const mp_number beta = { { 0x719501ee, 0xc1396c28, 0x12f58995, 0x9cf04975, 0xac3434e9, 0x6e64479e, 0x657c0710, 0x7ae96a2b } };

	const mp_number edge[] = { zero, one, two, pMinusOne, g_p, pPlusOne, allOnes, highBit, nearTop, w977, pmod, Gx, Gy, beta };

	std::vector<ModMulCase> cases;
	for (const mp_number & x : edge)
		for (const mp_number & y : edge)
			cases.push_back({ x, y });

	// targeted ambiguous-zone construction: (p-1)*(p-t) has true residue t; for small t that
	// residue < c so TWO valid representatives (t and t+p) exist -- this is exactly where a
	// wrong fold would diverge from stock. (p.d[0]=0xfffffc2f > 4096, so p-t only touches word 0.)
	mp_number pMinus1 = g_p; pMinus1.d[0] -= 1;
	for (uint32_t t = 1; t <= 4096; ++t) {
		mp_number y = g_p; y.d[0] -= t;   // p - t, in-domain (< p)
		cases.push_back({ pMinus1, y });
		cases.push_back({ y, pMinus1 });
	}

	for (size_t i = 0; i < numRandom; ++i)
		cases.push_back({ randomNumber(), randomNumber() });
	return cases;
}

static std::vector<mp_number> runBin(const ClSetup & s, const char * kernelName, const std::vector<mp_number> & x, const std::vector<mp_number> & y) {
	const size_t count = x.size();
	std::vector<mp_number> out(count);
	cl_int err;
	cl_mem xBuf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * sizeof(mp_number), (void*)x.data(), &err); clCheck(err, "xBuf");
	cl_mem yBuf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * sizeof(mp_number), (void*)y.data(), &err); clCheck(err, "yBuf");
	cl_mem rBuf = clCreateBuffer(s.context, CL_MEM_WRITE_ONLY, count * sizeof(mp_number), NULL, &err); clCheck(err, "rBuf");
	cl_kernel kernel = clCreateKernel(s.program, kernelName, &err); clCheck(err, "clCreateKernel");
	clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &xBuf), "arg0");
	clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &yBuf), "arg1");
	clCheck(clSetKernelArg(kernel, 2, sizeof(cl_mem), &rBuf), "arg2");
	clCheck(clEnqueueNDRangeKernel(s.queue, kernel, 1, NULL, &count, NULL, 0, NULL, NULL), "enqueue");
	clCheck(clEnqueueReadBuffer(s.queue, rBuf, CL_TRUE, 0, count * sizeof(mp_number), out.data(), 0, NULL, NULL), "read");
	clCheck(clFinish(s.queue), "finish");
	clReleaseKernel(kernel); clReleaseMemObject(xBuf); clReleaseMemObject(yBuf); clReleaseMemObject(rBuf);
	return out;
}

static std::vector<mp_number> runUn(const ClSetup & s, const char * kernelName, const std::vector<mp_number> & x) {
	const size_t count = x.size();
	std::vector<mp_number> out(count);
	cl_int err;
	cl_mem xBuf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * sizeof(mp_number), (void*)x.data(), &err); clCheck(err, "xBuf");
	cl_mem rBuf = clCreateBuffer(s.context, CL_MEM_WRITE_ONLY, count * sizeof(mp_number), NULL, &err); clCheck(err, "rBuf");
	cl_kernel kernel = clCreateKernel(s.program, kernelName, &err); clCheck(err, "clCreateKernel");
	clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &xBuf), "arg0");
	clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &rBuf), "arg1");
	clCheck(clEnqueueNDRangeKernel(s.queue, kernel, 1, NULL, &count, NULL, 0, NULL, NULL), "enqueue");
	clCheck(clEnqueueReadBuffer(s.queue, rBuf, CL_TRUE, 0, count * sizeof(mp_number), out.data(), 0, NULL, NULL), "read");
	clCheck(clFinish(s.queue), "finish");
	clReleaseKernel(kernel); clReleaseMemObject(xBuf); clReleaseMemObject(rBuf);
	return out;
}

int main(int argc, char ** argv) {
	const size_t numRandom = argc > 1 ? (size_t)std::atol(argv[1]) : 200000;

	const ClSetup s = clSetup();
	const bool ptx = hasKernel(s, "k_mod_mul_oneshot_ptx");
	std::printf("Inline-PTX oneshot kernels: %s\n\n", ptx ? "present" : "ABSENT (portable path only)");

	const std::vector<ModMulCase> cases = genModMulCases(numRandom);
	const size_t count = cases.size();
	std::vector<mp_number> x(count), y(count);
	for (size_t i = 0; i < count; ++i) { x[i] = cases[i].x; y[i] = cases[i].y; }

	size_t failures = 0;

	// ---------------- PATCH #3: oneshot mod-mul ----------------
	const std::vector<mp_number> osMul   = runBin(s, "k_mod_mul_oneshot_portable", x, y);
	const std::vector<mp_number> stockMul = runBin(s, "k_mod_mul_portable", x, y);
	{
		size_t f = 0, indom = 0, cong = 0, ndc = 0;
		for (size_t i = 0; i < count; ++i) {
			// (a) congruence for ALL inputs
			++cong;
			const mp_number got = refModP256(osMul[i]);
			const mp_number exp = refMulModP(cases[i].x, cases[i].y);
			if (!mpEqual(got, exp)) {
				if (++f <= 5) std::printf("  MUL CONGRUENCE MISMATCH case %zu: x=%s y=%s\n    got%%p %s\n    ref   %s\n",
					i, mpToHex(cases[i].x).c_str(), mpToHex(cases[i].y).c_str(), mpToHex(got).c_str(), mpToHex(exp).c_str());
				continue;
			}
			// (b) bit-exact vs stock, in-domain only
			if (!mpGte(cases[i].x, g_p) && !mpGte(cases[i].y, g_p)) {
				++indom;
				if (!mpEqual(osMul[i], stockMul[i])) {
					if (++f <= 5) std::printf("  MUL BIT-EXACT MISMATCH (in-domain) case %zu: x=%s y=%s\n    oneshot %s\n    stock   %s\n",
						i, mpToHex(cases[i].x).c_str(), mpToHex(cases[i].y).c_str(), mpToHex(osMul[i]).c_str(), mpToHex(stockMul[i]).c_str());
				}
			} else {
				++ndc;
			}
		}
		std::printf("mp_mod_mul_oneshot (portable): %zu cases, %zu congruent, %zu in-domain bit-exact vs stock, %zu out-of-domain congruent-only: %s\n",
			count, cong, indom, ndc, f == 0 ? "OK" : "FAILED");
		failures += f;
	}
	if (ptx) {
		const std::vector<mp_number> osMulPtx = runBin(s, "k_mod_mul_oneshot_ptx", x, y);
		size_t f = 0;
		for (size_t i = 0; i < count; ++i)
			if (!mpEqual(osMulPtx[i], osMul[i]))
				if (++f <= 5) std::printf("  MUL PTX!=PORTABLE case %zu: x=%s y=%s\n    portable %s\n    ptx      %s\n",
					i, mpToHex(cases[i].x).c_str(), mpToHex(cases[i].y).c_str(), mpToHex(osMul[i]).c_str(), mpToHex(osMulPtx[i]).c_str());
		std::printf("mp_mod_mul_oneshot (ptx): %zu cases ptx-vs-portable bit-exact: %s\n", count, f == 0 ? "OK" : "FAILED");
		failures += f;
	} else {
		std::printf("mp_mod_mul_oneshot (ptx): SKIPPED (portable-only device)\n");
	}

	// ---------------- PATCH #6: dedicated mod-sqr ----------------
	const std::vector<mp_number> sqr    = runUn(s, "k_mod_sqr_portable", x);
	const std::vector<mp_number> osXX   = runBin(s, "k_mod_mul_oneshot_portable", x, x);
	{
		size_t f = 0, indom = 0, cong = 0;
		for (size_t i = 0; i < count; ++i) {
			// (c) congruence vs refMulModP(x,x) for ALL
			++cong;
			const mp_number got = refModP256(sqr[i]);
			const mp_number exp = refMulModP(cases[i].x, cases[i].x);
			if (!mpEqual(got, exp)) {
				if (++f <= 5) std::printf("  SQR CONGRUENCE MISMATCH case %zu: x=%s\n    got%%p %s\n    ref   %s\n",
					i, mpToHex(cases[i].x).c_str(), mpToHex(got).c_str(), mpToHex(exp).c_str());
				continue;
			}
			// (c) bit-exact vs oneshot(x,x), in-domain only
			if (!mpGte(cases[i].x, g_p)) {
				++indom;
				if (!mpEqual(sqr[i], osXX[i])) {
					if (++f <= 5) std::printf("  SQR BIT-EXACT MISMATCH (in-domain) case %zu: x=%s\n    sqr        %s\n    oneshot^2  %s\n",
						i, mpToHex(cases[i].x).c_str(), mpToHex(sqr[i]).c_str(), mpToHex(osXX[i]).c_str());
				}
			}
		}
		std::printf("mp_mod_sqr (portable): %zu cases, %zu congruent, %zu in-domain bit-exact vs oneshot(x,x): %s\n",
			count, cong, indom, f == 0 ? "OK" : "FAILED");
		failures += f;
	}
	if (ptx) {
		const std::vector<mp_number> sqrPtx = runUn(s, "k_mod_sqr_ptx", x);
		size_t f = 0;
		for (size_t i = 0; i < count; ++i)
			if (!mpEqual(sqrPtx[i], sqr[i]))
				if (++f <= 5) std::printf("  SQR PTX!=PORTABLE case %zu: x=%s\n    portable %s\n    ptx      %s\n",
					i, mpToHex(cases[i].x).c_str(), mpToHex(sqr[i]).c_str(), mpToHex(sqrPtx[i]).c_str());
		std::printf("mp_mod_sqr (ptx): %zu cases ptx-vs-portable bit-exact: %s\n", count, f == 0 ? "OK" : "FAILED");
		failures += f;
	} else {
		std::printf("mp_mod_sqr (ptx): SKIPPED (portable-only device)\n");
	}

	if (failures) { std::printf("\nFAILED: %zu total failures\n", failures); return 1; }
	if (!ptx) std::printf("\nPassed on the portable path. Run on an NVIDIA device to exercise the PTX kernels.\n");
	else std::printf("\nAll tests passed.\n");
	return 0;
}
