/* harness.cl
 * ==========
 * Test/benchmark harness appended after the multiprecision section of profanity.cl.
 *
 * Contains:
 *   1. The pre-PR#49 ("old") implementations of mp_mul_mod_word_sub and mp_mod_mul,
 *      kept verbatim as a reference to test bit-exact equivalence against the
 *      optimized versions that now live in profanity.cl.
 *   2. Thin __kernel wrappers so the host-side tests can invoke both versions.
 *   3. Benchmark kernels that run a dependency-chained sequence of modular
 *      multiplications, so throughput of old vs new can be compared.
 */

/* ------------------------------------------------------------------------ */
/* Reference (pre-PR#49) implementations, verbatim from master              */
/* ------------------------------------------------------------------------ */

void mp_mul_mod_word_sub_old(mp_number * const r, const mp_word w, const bool withModHigher) {
	// Having these numbers declared here instead of using the global values in __constant address space seems to lead
	// to better optimizations by the compiler on my GTX 1070.
	mp_number mod = { { 0xfffffc2f, 0xfffffffe, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff} };
	mp_number modhigher = { {0x00000000, 0xfffffc2f, 0xfffffffe, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff} };

	mp_word cM = 0; // Carry for multiplication
	mp_word cS = 0; // Carry for subtraction
	mp_word tS = 0; // Temporary storage for subtraction
	mp_word tM = 0; // Temporary storage for multiplication
	mp_word cA = 0; // Carry for addition of modhigher

	for (mp_word i = 0; i < MP_WORDS; ++i) {
		tM = (mod.d[i] * w + cM);
		cM = mul_hi(mod.d[i], w) + (tM < cM);

		tM += (withModHigher ? modhigher.d[i] : 0) + cA;
		cA = tM < (withModHigher ? modhigher.d[i] : 0) ? 1 : (tM == (withModHigher ? modhigher.d[i] : 0) ? cA : 0);

		tS = r->d[i] - tM - cS;
		cS = tS > r->d[i] ? 1 : (tS == r->d[i] ? cS : 0);

		r->d[i] = tS;
	}
}

// Copy of mp_mod_mul from profanity.cl that calls the old subtraction routine.
void mp_mod_mul_old(mp_number * const r, const mp_number * const X, const mp_number * const Y) {
	mp_number Z = { {0} };
	mp_word extraWord;

	for (int i = MP_WORDS - 1; i >= 0; --i) {
		// Z = Z * 2^32
		extraWord = Z.d[7]; Z.d[7] = Z.d[6]; Z.d[6] = Z.d[5]; Z.d[5] = Z.d[4]; Z.d[4] = Z.d[3]; Z.d[3] = Z.d[2]; Z.d[2] = Z.d[1]; Z.d[1] = Z.d[0]; Z.d[0] = 0;

		// Z = Z + X * Y_i
		bool overflow = mp_mul_word_add_extra(&Z, X, Y->d[i], &extraWord);

		// Z = Z - qM
		mp_mul_mod_word_sub_old(&Z, extraWord, overflow);
	}

	*r = Z;
}

/* ------------------------------------------------------------------------ */
/* Inline-PTX multiprecision variants                                       */
/* ------------------------------------------------------------------------ */
/* On NVIDIA, profanity.cl compiles two implementations of the two innermost
 * multiprecision routines — one portable, one written in inline PTX so the
 * carries stay in the hardware flag — and the rest of the file gets the PTX
 * pair. Everywhere else only the portable pair exists.
 *
 * The kernels below reach both directly, so that on a device that has them the
 * two can be checked against each other bit for bit on the same inputs, and
 * timed against each other without rebuilding. Whether they exist at all is
 * PROFANITY_PTX_MP, which the host discovers by asking for a kernel by name and
 * seeing whether the device has one — see hasPtxKernels in testutil.hpp.
 *
 * Copies of mp_mod_mul that pin which pair they call, on the same grounds as
 * mp_mod_mul_old above: the routine under test is two levels down from the one
 * worth timing, and the loop that calls it is four lines. */

void mp_mod_mul_portable(mp_number * const r, const mp_number * const X, const mp_number * const Y) {
	mp_number Z = { {0} };
	mp_word extraWord;

	for (int i = MP_WORDS - 1; i >= 0; --i) {
		extraWord = Z.d[7]; Z.d[7] = Z.d[6]; Z.d[6] = Z.d[5]; Z.d[5] = Z.d[4]; Z.d[4] = Z.d[3]; Z.d[3] = Z.d[2]; Z.d[2] = Z.d[1]; Z.d[1] = Z.d[0]; Z.d[0] = 0;
		bool overflow = mp_mul_word_add_extra_portable(&Z, X, Y->d[i], &extraWord);
		mp_mul_mod_word_sub_portable(&Z, extraWord, overflow);
	}

	*r = Z;
}

__kernel void k_mul_word_add_extra_portable(
		__global mp_number * const R,
		__global const mp_number * const A,
		__global const uint * const W,
		__global uint * const E,
		__global uint * const OVF) {
	const size_t id = get_global_id(0);
	mp_number r = R[id];
	const mp_number a = A[id];
	mp_word e = E[id];
	OVF[id] = mp_mul_word_add_extra_portable(&r, &a, W[id], &e);
	R[id] = r;
	E[id] = e;
}

__kernel void k_mul_mod_word_sub_portable(__global mp_number * const r, __global const uint * const w, __global const uint * const withModHigher) {
	const size_t id = get_global_id(0);
	mp_number x = r[id];
	mp_mul_mod_word_sub_portable(&x, w[id], withModHigher[id] != 0);
	r[id] = x;
}

__kernel void k_mod_mul_portable(__global const mp_number * const X, __global const mp_number * const Y, __global mp_number * const R) {
	const size_t id = get_global_id(0);
	mp_number x = X[id];
	mp_number y = Y[id];
	mp_number r;
	mp_mod_mul_portable(&r, &x, &y);
	R[id] = r;
}

__kernel void bench_mod_mul_portable(__global mp_number * const X, __global const mp_number * const Y, const uint iterations) {
	const size_t id = get_global_id(0);
	mp_number x = X[id];
	const mp_number y = Y[id];

	for (uint i = 0; i < iterations; ++i) {
		mp_mod_mul_portable(&x, &x, &y);
	}

	X[id] = x;
}

#ifdef PROFANITY_PTX_MP

void mp_mod_mul_ptx(mp_number * const r, const mp_number * const X, const mp_number * const Y) {
	mp_number Z = { {0} };
	mp_word extraWord;

	for (int i = MP_WORDS - 1; i >= 0; --i) {
		extraWord = Z.d[7]; Z.d[7] = Z.d[6]; Z.d[6] = Z.d[5]; Z.d[5] = Z.d[4]; Z.d[4] = Z.d[3]; Z.d[3] = Z.d[2]; Z.d[2] = Z.d[1]; Z.d[1] = Z.d[0]; Z.d[0] = 0;
		bool overflow = mp_mul_word_add_extra_ptx(&Z, X, Y->d[i], &extraWord);
		mp_mul_mod_word_sub_ptx(&Z, extraWord, overflow);
	}

	*r = Z;
}

__kernel void k_mul_word_add_extra_ptx(
		__global mp_number * const R,
		__global const mp_number * const A,
		__global const uint * const W,
		__global uint * const E,
		__global uint * const OVF) {
	const size_t id = get_global_id(0);
	mp_number r = R[id];
	const mp_number a = A[id];
	mp_word e = E[id];
	OVF[id] = mp_mul_word_add_extra_ptx(&r, &a, W[id], &e);
	R[id] = r;
	E[id] = e;
}

__kernel void k_mul_mod_word_sub_ptx(__global mp_number * const r, __global const uint * const w, __global const uint * const withModHigher) {
	const size_t id = get_global_id(0);
	mp_number x = r[id];
	mp_mul_mod_word_sub_ptx(&x, w[id], withModHigher[id] != 0);
	r[id] = x;
}

__kernel void k_mod_mul_ptx(__global const mp_number * const X, __global const mp_number * const Y, __global mp_number * const R) {
	const size_t id = get_global_id(0);
	mp_number x = X[id];
	mp_number y = Y[id];
	mp_number r;
	mp_mod_mul_ptx(&r, &x, &y);
	R[id] = r;
}

__kernel void bench_mod_mul_ptx(__global mp_number * const X, __global const mp_number * const Y, const uint iterations) {
	const size_t id = get_global_id(0);
	mp_number x = X[id];
	const mp_number y = Y[id];

	for (uint i = 0; i < iterations; ++i) {
		mp_mod_mul_ptx(&x, &x, &y);
	}

	X[id] = x;
}

#endif /* PROFANITY_PTX_MP */

/* ------------------------------------------------------------------------ */
/* Correctness / equivalence test kernels                                   */
/* ------------------------------------------------------------------------ */

__kernel void k_mul_mod_word_sub_new(__global mp_number * const r, __global const uint * const w, __global const uint * const withModHigher) {
	const size_t id = get_global_id(0);
	mp_number x = r[id];
	mp_mul_mod_word_sub(&x, w[id], withModHigher[id] != 0);
	r[id] = x;
}

__kernel void k_mul_mod_word_sub_old(__global mp_number * const r, __global const uint * const w, __global const uint * const withModHigher) {
	const size_t id = get_global_id(0);
	mp_number x = r[id];
	mp_mul_mod_word_sub_old(&x, w[id], withModHigher[id] != 0);
	r[id] = x;
}

__kernel void k_mod_mul_new(__global const mp_number * const X, __global const mp_number * const Y, __global mp_number * const R) {
	const size_t id = get_global_id(0);
	mp_number x = X[id];
	mp_number y = Y[id];
	mp_number r;
	mp_mod_mul(&r, &x, &y);
	R[id] = r;
}

__kernel void k_mod_mul_old(__global const mp_number * const X, __global const mp_number * const Y, __global mp_number * const R) {
	const size_t id = get_global_id(0);
	mp_number x = X[id];
	mp_number y = Y[id];
	mp_number r;
	mp_mod_mul_old(&r, &x, &y);
	R[id] = r;
}

/* ------------------------------------------------------------------------ */
/* Scoring                                                                  */
/* ------------------------------------------------------------------------ */
/* The scoring functions over hashes the host hands them, rather than over the
 * ones a search would have produced, so that what they score can be tested
 * without mining for an address that happens to satisfy it.
 *
 * They read an address as the five uints the iterate kernel leaves it packed
 * into, so the flat bytes are packed the same way here — see profanity_byte for
 * the order it expects them in. */

__kernel void k_score_matching(__global const uchar * const hashes, __constant const uchar * const data1, __constant const uchar * const data2, __global int * const scores) {
	const size_t id = get_global_id(0);

	uint address[5] = { 0, 0, 0, 0, 0 };
	for (int i = 0; i < 20; ++i) {
		address[i >> 2] |= ((uint)hashes[id * 20 + i]) << ((i & 3) << 3);
	}

	scores[id] = profanity_score_fn_matching(address, data1, data2);
}

/* The implementations the current ones replaced, kept verbatim so the cost of
 * the change can be measured rather than argued about, and so the faster ones
 * have something to be checked against. Each reads its own encoding of the
 * mask, which the host builds alongside Mode::matching's. */

/* --matching, one entry per byte of the hash with mask and value packed
 * together. A mask could only ever be looked for where it was written, so a
 * `contains` search had to pick one offset out of the 41 - length it could have
 * sat at. */
inline int profanity_score_fn_matching_bytes(const uint * const address, __constant const uchar * const data1, __constant const uchar * const data2) {
	int score = 0;

	for (int i = 0; i < 20; ++i) {
		if (data1[i] > 0 && (profanity_byte(address, i) & data1[i]) == data2[i]) {
			++score;
		}
	}

	return score;
}

/* --matching, the first nibble-wise version: one entry per mask character
 * including the wildcards, 0xFF marking the end. Every hash rescans for that
 * terminator, and an anchored pattern is walked through all of its padding. */
inline int profanity_score_fn_matching_scan(const uint * const address, __constant const uchar * const data1, __constant const uchar * const data2) {
	int length = 40;
	for (int i = 0; i < 40; ++i) {
		if (data1[i] == 0xFF) {
			length = i;
			break;
		}
	}

	int score = 0;
	for (int at = 0; at + length <= 40; ++at) {
		int run = 0;

		for (int i = 0; i < length; ++i) {
			if (data1[i] == 0) {
				continue;
			}

			const int nibble = at + i;
			const uchar byte = profanity_byte(address, nibble >> 1);
			const uchar digit = (nibble & 1) ? (byte & 0x0F) : (byte >> 4);

			if (digit != data2[i]) {
				break;
			}

			++run;
		}

		score = max(score, run);
	}

	return score;
}

/* What --zero-bytes counted with before it was asked of the whole address at
 * once. The reference the shipping version is checked against: the two have to
 * agree on every address, or the faster one is just wrong faster. */
inline int profanity_score_fn_zerobytes_loop(const uint * const address, __constant const uchar * const data1, __constant const uchar * const data2) {
	int score = 0;

	for (int i = 0; i < 20; ++i) {
		if (profanity_byte(address, i) == 0) {
			score++;
		}
	}

	return score;
}

/* And what --range counted with, likewise the reference for the guard bit
 * version that replaced it. */
inline int profanity_score_fn_range_loop(const uint * const address, __constant const uchar * const data1, __constant const uchar * const data2) {
	int score = 0;

	for (int i = 0; i < 20; ++i) {
		const uchar byte = profanity_byte(address, i);
		const uchar first = (byte & 0xF0) >> 4;
		const uchar second = (byte & 0x0F);

		if (first >= data1[0] && first <= data2[0]) {
			++score;
		}

		if (second >= data1[0] && second <= data2[0]) {
			++score;
		}
	}

	return score;
}

/* Each work item scores one address `iterations` times over. The accumulator is
 * folded back into the address so no iteration can be hoisted out of the loop,
 * and every variant sees the same sequence of addresses for the same input.
 *
 * At one iteration the accumulator is just the score of the address as handed
 * in, before anything has perturbed it, which is what the correctness tests use
 * these for as well. */
#define BENCH_SCORE_KERNEL(NAME) \
__kernel void bench_score_##NAME( \
		__global const uchar * const hashes, \
		__constant const uchar * const data1, \
		__constant const uchar * const data2, \
		const uint iterations, \
		__global int * const out) { \
	const size_t id = get_global_id(0); \
	uint address[5] = { 0, 0, 0, 0, 0 }; \
	for (int i = 0; i < 20; ++i) { \
		address[i >> 2] |= ((uint)hashes[id * 20 + i]) << ((i & 3) << 3); \
	} \
	int acc = 0; \
	for (uint k = 0; k < iterations; ++k) { \
		acc += profanity_score_fn_##NAME(address, data1, data2); \
		address[0] ^= (uint)acc; \
	} \
	out[id] = acc; \
}

BENCH_SCORE_KERNEL(matching)
BENCH_SCORE_KERNEL(matching_bytes)
BENCH_SCORE_KERNEL(matching_scan)
BENCH_SCORE_KERNEL(range)
BENCH_SCORE_KERNEL(range_loop)
BENCH_SCORE_KERNEL(rangeequal)
BENCH_SCORE_KERNEL(zerobytes)
BENCH_SCORE_KERNEL(zerobytes_loop)
BENCH_SCORE_KERNEL(benchmark)
BENCH_SCORE_KERNEL(leading)
BENCH_SCORE_KERNEL(leadingrange)
BENCH_SCORE_KERNEL(mirror)
BENCH_SCORE_KERNEL(doubles)

/* A yardstick for the numbers above. Scoring is the last step of the iterate
 * kernel and the keccak permutation in front of it is the largest single thing
 * that kernel does — once per address, and twice with --contract — with the
 * point arithmetic on top of that again. So this says roughly how much of a
 * search a change to the scoring function can possibly be worth. */
__kernel void bench_keccak(__global uint * const out, const uint iterations) {
	const size_t id = get_global_id(0);

	ethhash h;
	for (int i = 0; i < 50; ++i) {
		h.d[i] = (uint)(id * 50 + i);
	}

	for (uint k = 0; k < iterations; ++k) {
		sha3_keccakf(&h);
	}

	out[id] = h.d[0];
}

/* What a modular inversion costs against a modular multiplication, which is
 * what decides whether batching inversions the way profanity_inverse does is
 * worth restructuring anything for. */
__kernel void bench_mod_inverse(__global mp_number * const X, const uint iterations) {
	const size_t id = get_global_id(0);
	mp_number x = X[id];

	for (uint i = 0; i < iterations; ++i) {
		mp_mod_inverse(&x);
		x.d[0] |= 1;
	}

	X[id] = x;
}

/* ------------------------------------------------------------------------ */
/* Benchmark kernels                                                        */
/* ------------------------------------------------------------------------ */
/* Each work item performs `iterations` chained modular multiplications:
 * x = x * y (mod p). The data dependency between iterations prevents the
 * compiler from removing or reordering the work. The final x is written back
 * so nothing is dead code (and so the host can cross-check old vs new). */

__kernel void bench_mod_mul_new(__global mp_number * const X, __global const mp_number * const Y, const uint iterations) {
	const size_t id = get_global_id(0);
	mp_number x = X[id];
	const mp_number y = Y[id];

	for (uint i = 0; i < iterations; ++i) {
		mp_mod_mul(&x, &x, &y);
	}

	X[id] = x;
}

__kernel void bench_mod_mul_old(__global mp_number * const X, __global const mp_number * const Y, const uint iterations) {
	const size_t id = get_global_id(0);
	mp_number x = X[id];
	const mp_number y = Y[id];

	for (uint i = 0; i < iterations; ++i) {
		mp_mod_mul_old(&x, &x, &y);
	}

	X[id] = x;
}
