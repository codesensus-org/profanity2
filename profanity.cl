/* profanity.cl
 * ============
 * Contains multi-precision arithmetic functions and iterative elliptical point
 * addition which is the heart of profanity.
 *
 * Terminology
 * ===========
 * 
 *
 * Cutting corners
 * ===============
 * In some instances this code will produce the incorrect results. The elliptical
 * point addition does for example not properly handle the case of two points
 * sharing the same X-coordinate. The reason the code doesn't handle it properly
 * is because it is very unlikely to ever occur and the performance penalty for
 * doing it right is too severe. In the future I'll introduce a periodic check
 * after N amount of cycles that verifies the integrity of all the points to
 * make sure that even very unlikely event are at some point rectified.
 * 
 * Currently, if any of the points in the kernels experiences the unlikely event
 * of an error then that point is forever garbage and your runtime-performance
 * will in practice be (i*I-N) / (i*I). i and I here refers to the values given
 * to the program via the -i and -I switches (default values of 255 and 16384
 * respectively) and N is the number of errornous points.
 *
 * So if a single error occurs you'll lose 1/(i*I) of your performance. That's
 * around 0.00002%. The program will still report the same hashrate of course,
 * only that some of that work is entirely wasted on this errornous point.
 *
 * Initialization of main structure
 * ================================
 *
 * Iteration
 * =========
 *
 *
 * TODO
 * ====
 *   * Update comments to reflect new optimizations and structure
 *
 */

/* ------------------------------------------------------------------------ */
/* Multiprecision functions                                                 */
/* ------------------------------------------------------------------------ */
// How many addresses one point addition is worth, in the order they are taken:
//
//   1  P            the point itself
//   2  -P           its negation, (x, -y), free but for a subtraction
//   3  ψ(P)         (βx, y), one modular multiplication
//   4  -ψ(P)
//   5  ψ²(P)        (β²x, y), one more multiplication
//   6  -ψ²(P)
//
// Six is the ceiling and not a choice of implementation. These are the
// automorphisms of the curve — the maps E → E that cost no point arithmetic —
// and secp256k1 is y² = x³ + 7, so its j-invariant is zero and its automorphism
// group is the sixth roots of unity, {±1, ±ω, ±ω²}. That group has order six, so
// there is no seventh such map to find. Anything further wants an endomorphism
// of degree above one, which is point arithmetic again by another name.
//
// The host passes the count in, and knows what to do with the private key behind
// each of them — see profanity_iterate.
#if PROFANITY_VARIANTS < 1 || PROFANITY_VARIANTS > 6
#error "PROFANITY_VARIANTS must be between 1 and 6"
#endif

// How many point additions a launch does per point before handing back. The
// state a point carries between them — its delta and its previous lambda — stays
// in the kernel across all of them, so the global memory traffic that state
// costs, and the launch it costs, are divided by this.
#if PROFANITY_ROUNDS < 1
#error "PROFANITY_ROUNDS must be at least 1"
#endif

#define MP_WORDS 8
#define MP_BITS 32
#define bswap32(n) (rotate(n & 0x00FF00FF, 24U)|(rotate(n, 8U) & 0x00FF00FF))

// The same for eight bytes, out of two of the above. Used where a value written
// big endian has to go into a state read little endian, which is a CREATE2
// search putting its counter into the salt.
static inline ulong bswap64(const ulong n) {
	return ((ulong)bswap32(((uint)n)) << 32) | (ulong)bswap32(((uint)(n >> 32)));
}

// Whether the multiprecision routines below may use the carry flag directly.
//
// Every carry in this file is detected by comparison — `c = t > a ? 1 : (t == a
// ? c : 0)` and its variants — because OpenCL C has no way to name the flag the
// hardware sets for free. That costs a setp and a selp per word per carry.
// NVIDIA's OpenCL frontend shares NVVM with CUDA and accepts inline PTX, which
// does have a way to name it, so on that one vendor the same arithmetic can be
// written in about a fifth of the instructions.
//
// __NV_CL_C_VERSION is defined by that frontend and by no other, which is what
// this has to key off rather than anything the host knows: profanity.cpp builds
// one program for every device in the context at once, so a machine with an
// NVIDIA card and an AMD one compiles this file twice from the same string and
// only the vendor's own macro tells the two compilations apart.
//
// Inline PTX in OpenCL is not something NVIDIA documents, so -D PROFANITY_NO_PTX
// turns it off and takes the portable path on every device.
#if defined(__NV_CL_C_VERSION) && !defined(PROFANITY_NO_PTX)
#define PROFANITY_PTX_MP 1
#endif

typedef uint mp_word;
typedef struct __attribute__((aligned(16))) {
	mp_word d[MP_WORDS];
} mp_number;

// mod              = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f
__constant const mp_number mod              = { {0xfffffc2f, 0xfffffffe, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff} };

// tripleNegativeGx = 0x92c4cc831269ccfaff1ed83e946adeeaf82c096e76958573f2287becbb17b196
__constant const mp_number tripleNegativeGx = { {0xbb17b196, 0xf2287bec, 0x76958573, 0xf82c096e, 0x946adeea, 0xff1ed83e, 0x1269ccfa, 0x92c4cc83 } };

// doubleNegativeGy = 0x6f8a4b11b2b8773544b60807e3ddeeae05d0976eb2f557ccc7705edf09de52bf
__constant const mp_number doubleNegativeGy = { {0x09de52bf, 0xc7705edf, 0xb2f557cc, 0x05d0976e, 0xe3ddeeae, 0x44b60807, 0xb2b87735, 0x6f8a4b11} };

// negativeGy       = 0xb7c52588d95c3b9aa25b0403f1eef75702e84bb7597aabe663b82f6f04ef2777
__constant const mp_number negativeGy       = { {0x04ef2777, 0x63b82f6f, 0x597aabe6, 0x02e84bb7, 0xf1eef757, 0xa25b0403, 0xd95c3b9a, 0xb7c52588 } };


// Multiprecision subtraction. Underflow signalled via return value.
mp_word mp_sub(mp_number * const r, const mp_number * const a, const mp_number * const b) {
	mp_word t, c = 0;

	for (mp_word i = 0; i < MP_WORDS; ++i) {
		t = a->d[i] - b->d[i] - c;
		c = t > a->d[i] ? 1 : (t == a->d[i] ? c : 0);

		r->d[i] = t;
	}

	return c;
}

// Multiprecision subtraction of the modulus saved in mod. Underflow signalled via return value.
mp_word mp_sub_mod(mp_number * const r) {
	mp_number mod = { {0xfffffc2f, 0xfffffffe, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff} };

	mp_word t, c = 0;

	for (mp_word i = 0; i < MP_WORDS; ++i) {
		t = r->d[i] - mod.d[i] - c;
		c = t > r->d[i] ? 1 : (t == r->d[i] ? c : 0);

		r->d[i] = t;
	}

	return c;
}

// Multiprecision subtraction modulo M, M = mod.
// This function is often also used for additions by subtracting a negative number. I've chosen
// to do this because:
//   1. It's easier to re-use an already existing function
//   2. A modular addition would have more overhead since it has to determine if the result of
//      the addition (r) is in the gap M <= r < 2^256. This overhead doesn't exist in a
//      subtraction. We immediately know at the end of a subtraction if we had underflow
//      or not by inspecting the carry value. M refers to the modulus saved in variable mod.
void mp_mod_sub(mp_number * const r, const mp_number * const a, const mp_number * const b) {
	mp_word i, t, c = 0;

	for (i = 0; i < MP_WORDS; ++i) {
		t = a->d[i] - b->d[i] - c;
		c = t < a->d[i] ? 0 : (t == a->d[i] ? c : 1);

		r->d[i] = t;
	}

	if (c) {
		c = 0;
		for (i = 0; i < MP_WORDS; ++i) {
			r->d[i] += mod.d[i] + c;
			c = r->d[i] < mod.d[i] ? 1 : (r->d[i] == mod.d[i] ? c : 0);
		}
	}
}

// Multiprecision subtraction modulo M from a constant number.
// I made this in the belief that using constant address space instead of private address space for any
// constant numbers would lead to increase in performance. Judges are still out on this one.
void mp_mod_sub_const(mp_number * const r, __constant const mp_number * const a, const mp_number * const b) {
	mp_word i, t, c = 0;

	for (i = 0; i < MP_WORDS; ++i) {
		t = a->d[i] - b->d[i] - c;
		c = t < a->d[i] ? 0 : (t == a->d[i] ? c : 1);

		r->d[i] = t;
	}

	if (c) {
		c = 0;
		for (i = 0; i < MP_WORDS; ++i) {
			r->d[i] += mod.d[i] + c;
			c = r->d[i] < mod.d[i] ? 1 : (r->d[i] == mod.d[i] ? c : 0);
		}
	}
}

// Multiprecision subtraction modulo M of G_x from a number.
// Specialization of mp_mod_sub in hope of performance gain.
void mp_mod_sub_gx(mp_number * const r, const mp_number * const a) {
	mp_word i, t, c = 0;

	t = a->d[0] - 0x16f81798; c = t < a->d[0] ? 0 : (t == a->d[0] ? c : 1); r->d[0] = t;
	t = a->d[1] - 0x59f2815b - c; c = t < a->d[1] ? 0 : (t == a->d[1] ? c : 1); r->d[1] = t;
	t = a->d[2] - 0x2dce28d9 - c; c = t < a->d[2] ? 0 : (t == a->d[2] ? c : 1); r->d[2] = t;
	t = a->d[3] - 0x029bfcdb - c; c = t < a->d[3] ? 0 : (t == a->d[3] ? c : 1); r->d[3] = t;
	t = a->d[4] - 0xce870b07 - c; c = t < a->d[4] ? 0 : (t == a->d[4] ? c : 1); r->d[4] = t;
	t = a->d[5] - 0x55a06295 - c; c = t < a->d[5] ? 0 : (t == a->d[5] ? c : 1); r->d[5] = t;
	t = a->d[6] - 0xf9dcbbac - c; c = t < a->d[6] ? 0 : (t == a->d[6] ? c : 1); r->d[6] = t;
	t = a->d[7] - 0x79be667e - c; c = t < a->d[7] ? 0 : (t == a->d[7] ? c : 1); r->d[7] = t;

	if (c) {
		c = 0;
		for (i = 0; i < MP_WORDS; ++i) {
			r->d[i] += mod.d[i] + c;
			c = r->d[i] < mod.d[i] ? 1 : (r->d[i] == mod.d[i] ? c : 0);
		}
	}
}

// Multiprecision subtraction modulo M of G_y from a number.
// Specialization of mp_mod_sub in hope of performance gain.
void mp_mod_sub_gy(mp_number * const r, const mp_number * const a) {
	mp_word i, t, c = 0;

	t = a->d[0] - 0xfb10d4b8; c = t < a->d[0] ? 0 : (t == a->d[0] ? c : 1); r->d[0] = t;
	t = a->d[1] - 0x9c47d08f - c; c = t < a->d[1] ? 0 : (t == a->d[1] ? c : 1); r->d[1] = t;
	t = a->d[2] - 0xa6855419 - c; c = t < a->d[2] ? 0 : (t == a->d[2] ? c : 1); r->d[2] = t;
	t = a->d[3] - 0xfd17b448 - c; c = t < a->d[3] ? 0 : (t == a->d[3] ? c : 1); r->d[3] = t;
	t = a->d[4] - 0x0e1108a8 - c; c = t < a->d[4] ? 0 : (t == a->d[4] ? c : 1); r->d[4] = t;
	t = a->d[5] - 0x5da4fbfc - c; c = t < a->d[5] ? 0 : (t == a->d[5] ? c : 1); r->d[5] = t;
	t = a->d[6] - 0x26a3c465 - c; c = t < a->d[6] ? 0 : (t == a->d[6] ? c : 1); r->d[6] = t;
	t = a->d[7] - 0x483ada77 - c; c = t < a->d[7] ? 0 : (t == a->d[7] ? c : 1); r->d[7] = t;

	if (c) {
		c = 0;
		for (i = 0; i < MP_WORDS; ++i) {
			r->d[i] += mod.d[i] + c;
			c = r->d[i] < mod.d[i] ? 1 : (r->d[i] == mod.d[i] ? c : 0);
		}
	}
}

// Multiprecision addition. Overflow signalled via return value.
mp_word mp_add(mp_number * const r, const mp_number * const a) {
	mp_word c = 0;

	for (mp_word i = 0; i < MP_WORDS; ++i) {
		r->d[i] += a->d[i] + c;
		c = r->d[i] < a->d[i] ? 1 : (r->d[i] == a->d[i] ? c : 0);
	}

	return c;
}

// Multiprecision addition of the modulus saved in mod. Overflow signalled via return value.
mp_word mp_add_mod(mp_number * const r) {
	mp_word c = 0;

	for (mp_word i = 0; i < MP_WORDS; ++i) {
		r->d[i] += mod.d[i] + c;
		c = r->d[i] < mod.d[i] ? 1 : (r->d[i] == mod.d[i] ? c : 0);
	}

	return c;
}

// Multiprecision addition of two numbers with one extra word each. Overflow signalled via return value.
mp_word mp_add_more(mp_number * const r, mp_word * const extraR, const mp_number * const a, const mp_word * const extraA) {
	const mp_word c = mp_add(r, a);
	*extraR += *extraA + c;
	return *extraR < *extraA ? 1 : (*extraR == *extraA ? c : 0);
}

// Multiprecision greater than or equal (>=) operator
mp_word mp_gte(const mp_number * const a, const mp_number * const b) {
	mp_word l = 0, g = 0;

	for (mp_word i = 0; i < MP_WORDS; ++i) {
		if (a->d[i] < b->d[i]) l |= (1 << i);
		if (a->d[i] > b->d[i]) g |= (1 << i);
	}

	return g >= l;
}

// Bit shifts a number with an extra word to the right one step
void mp_shr_extra(mp_number * const r, mp_word * const e) {
	r->d[0] = (r->d[1] << 31) | (r->d[0] >> 1);
	r->d[1] = (r->d[2] << 31) | (r->d[1] >> 1);
	r->d[2] = (r->d[3] << 31) | (r->d[2] >> 1);
	r->d[3] = (r->d[4] << 31) | (r->d[3] >> 1);
	r->d[4] = (r->d[5] << 31) | (r->d[4] >> 1);
	r->d[5] = (r->d[6] << 31) | (r->d[5] >> 1);
	r->d[6] = (r->d[7] << 31) | (r->d[6] >> 1);
	r->d[7] = (*e << 31) | (r->d[7] >> 1);
	*e >>= 1;
}

// Bit shifts a number to the right one step
void mp_shr(mp_number * const r) {
	r->d[0] = (r->d[1] << 31) | (r->d[0] >> 1);
	r->d[1] = (r->d[2] << 31) | (r->d[1] >> 1);
	r->d[2] = (r->d[3] << 31) | (r->d[2] >> 1);
	r->d[3] = (r->d[4] << 31) | (r->d[3] >> 1);
	r->d[4] = (r->d[5] << 31) | (r->d[4] >> 1);
	r->d[5] = (r->d[6] << 31) | (r->d[5] >> 1);
	r->d[6] = (r->d[7] << 31) | (r->d[6] >> 1);
	r->d[7] >>= 1;
}

// Multiplies a number with a word and adds it to an existing number with an extra word, overflow of the extra word is signalled in return value
// This is a special function only used for modular multiplication
mp_word mp_mul_word_add_extra_portable(mp_number * const r, const mp_number * const a, const mp_word w, mp_word * const extra) {
	mp_word cM = 0; // Carry for multiplication
	mp_word cA = 0; // Carry for addition
	mp_word tM = 0; // Temporary storage for multiplication

	for (mp_word i = 0; i < MP_WORDS; ++i) {
		tM = (a->d[i] * w + cM);
		cM = mul_hi(a->d[i], w) + (tM < cM);

		r->d[i] += tM + cA;
		cA = r->d[i] < tM ? 1 : (r->d[i] == tM ? cA : 0);
	}

	*extra += cM + cA;
	return *extra < cM ? 1 : (*extra == cM ? cA : 0);
}

#ifdef PROFANITY_PTX_MP
/* The same nine-word accumulation, written so the carries stay in the flag.
 *
 * What the loop above computes is (r || extra) += a * w, where the product is
 * built a word at a time and its top word falls out at the end as cM. Split by
 * where each half of a partial product lands, that is
 *
 *     (r || extra) += sum_i lo(a_i * w) << 32i      -- words 0..7
 *                   + sum_i hi(a_i * w) << 32(i+1)  -- words 1..8
 *
 * and each of those sums is one straight carry chain. mad.lo.cc/madc.lo.cc runs
 * the first and mad.hi.cc/madc.hi.cc the second, eighteen instructions against
 * the eighty or so the comparisons cost.
 *
 * Both chains have to be in one asm block. CC.CF does not survive whatever the
 * compiler decides to schedule between two of them, and nothing in the operand
 * constraints tells it not to — splitting this in half is the way to get results
 * that are wrong only sometimes, and only on some drivers.
 *
 * The overflow out of word 8 is a single bit for the same reason it is above:
 * (r || extra) is below 2^288 and a * w is below 2^288, so their sum is below
 * 2^289 and crosses 2^288 at most once, whichever chain happens to carry it.
 */
mp_word mp_mul_word_add_extra_ptx(mp_number * const r, const mp_number * const a, const mp_word w, mp_word * const extra) {
	mp_word r0 = r->d[0], r1 = r->d[1], r2 = r->d[2], r3 = r->d[3];
	mp_word r4 = r->d[4], r5 = r->d[5], r6 = r->d[6], r7 = r->d[7];

	const mp_word a0 = a->d[0], a1 = a->d[1], a2 = a->d[2], a3 = a->d[3];
	const mp_word a4 = a->d[4], a5 = a->d[5], a6 = a->d[6], a7 = a->d[7];

	mp_word e = *extra;
	mp_word overflow;

	asm volatile(
		"mad.lo.cc.u32  %0, %10, %18, %0;\n\t"
		"madc.lo.cc.u32 %1, %11, %18, %1;\n\t"
		"madc.lo.cc.u32 %2, %12, %18, %2;\n\t"
		"madc.lo.cc.u32 %3, %13, %18, %3;\n\t"
		"madc.lo.cc.u32 %4, %14, %18, %4;\n\t"
		"madc.lo.cc.u32 %5, %15, %18, %5;\n\t"
		"madc.lo.cc.u32 %6, %16, %18, %6;\n\t"
		"madc.lo.cc.u32 %7, %17, %18, %7;\n\t"
		"addc.cc.u32    %8, %8, 0;\n\t"
		"addc.u32       %9, 0, 0;\n\t"

		"mad.hi.cc.u32  %1, %10, %18, %1;\n\t"
		"madc.hi.cc.u32 %2, %11, %18, %2;\n\t"
		"madc.hi.cc.u32 %3, %12, %18, %3;\n\t"
		"madc.hi.cc.u32 %4, %13, %18, %4;\n\t"
		"madc.hi.cc.u32 %5, %14, %18, %5;\n\t"
		"madc.hi.cc.u32 %6, %15, %18, %6;\n\t"
		"madc.hi.cc.u32 %7, %16, %18, %7;\n\t"
		"madc.hi.cc.u32 %8, %17, %18, %8;\n\t"
		"addc.u32       %9, %9, 0;"
		: "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3), "+r"(r4),
		  "+r"(r5), "+r"(r6), "+r"(r7), "+r"(e), "=r"(overflow)
		: "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4),
		  "r"(a5), "r"(a6), "r"(a7), "r"(w));

	r->d[0] = r0; r->d[1] = r1; r->d[2] = r2; r->d[3] = r3;
	r->d[4] = r4; r->d[5] = r5; r->d[6] = r6; r->d[7] = r7;
	*extra = e;

	return overflow;
}
#endif /* PROFANITY_PTX_MP */

// Which of the two the rest of this file gets.
static inline mp_word mp_mul_word_add_extra(mp_number * const r, const mp_number * const a, const mp_word w, mp_word * const extra) {
#ifdef PROFANITY_PTX_MP
	return mp_mul_word_add_extra_ptx(r, a, w, extra);
#else
	return mp_mul_word_add_extra_portable(r, a, w, extra);
#endif
}

// Multiplies a number with a word, potentially adds modhigher to it, and then subtracts it from
// an existing number, no extra words, no overflow.
//
// This is a special function only used for modular multiplication.
//
// Optimization (secp256k1 fast reduction) contributed by Rodrigo Madera (@madera).
//
// The secp256k1 prime has the special form:
//
//   p = 2^256 - pmod, where pmod = 2^32 + 977 = 0x1000003D1
//
// Therefore, working modulo 2^256:
//
//   q * p = q * (2^256 - pmod) = q * 2^256 - q * pmod == -q * pmod  (mod 2^256)
//
//   (r - q * p) mod 2^256 == (r + q * pmod) mod 2^256
//
// So instead of multiplying q by the full 256-bit p and subtracting, we multiply q by the
// 33-bit pmod and add. This reduces the amount of bits used, giving us 20-35% speed improvements.
//
// Two implementations of that addition follow. They differ only in how the
// three words go into r and not in what the three words are, so what they are
// is worked out once, here.

// q * pmod, which is never more than three words wide: pmod is 33 bits, so its
// product with a single word is 65, and the modhigher term shifts one copy of
// it up by a word.
static inline void mp_mod_word_addend(const mp_word w, const bool withModHigher, mp_word * const p0, mp_word * const p1, mp_word * const p2) {
	const mp_word lo977 = 977u * w;
	const mp_word hi977 = mul_hi(977u, w);

	*p0 = lo977;
	const ulong p1_full = (ulong)w + hi977 + (withModHigher ? 0x000003D1u : 0u);
	*p1 = (mp_word)p1_full;
	*p2 = (mp_word)(p1_full >> 32) + (withModHigher ? 1u : 0u);
}

void mp_mul_mod_word_sub_portable(mp_number * const r, const mp_word w, const bool withModHigher) {
	mp_word p0, p1, p2;
	mp_mod_word_addend(w, withModHigher, &p0, &p1, &p2);

	ulong s = (ulong)r->d[0] + p0;
	r->d[0] = (mp_word)s;
	mp_word c = (mp_word)(s >> 32);

	s = (ulong)r->d[1] + p1 + c;
	r->d[1] = (mp_word)s;
	c = (mp_word)(s >> 32);

	s = (ulong)r->d[2] + p2 + c;
	r->d[2] = (mp_word)s;
	c = (mp_word)(s >> 32);

	for (mp_word i = 3; i < MP_WORDS; ++i) {
		s = (ulong)r->d[i] + c;
		r->d[i] = (mp_word)s;
		c = (mp_word)(s >> 32);
	}
}

#ifdef PROFANITY_PTX_MP
/* The same addition as one carry chain rather than eight widening adds.
 *
 * Less of a win than the multiply above — a 64-bit add already lowers to
 * add.cc/addc and the compiler does see through the pattern — but the five
 * words past p2 exist only to carry, and written this way they cost one
 * instruction each instead of an extension, an add and a shift.
 */
void mp_mul_mod_word_sub_ptx(mp_number * const r, const mp_word w, const bool withModHigher) {
	mp_word p0, p1, p2;
	mp_mod_word_addend(w, withModHigher, &p0, &p1, &p2);

	mp_word r0 = r->d[0], r1 = r->d[1], r2 = r->d[2], r3 = r->d[3];
	mp_word r4 = r->d[4], r5 = r->d[5], r6 = r->d[6], r7 = r->d[7];

	asm volatile(
		"add.cc.u32  %0, %0, %8;\n\t"
		"addc.cc.u32 %1, %1, %9;\n\t"
		"addc.cc.u32 %2, %2, %10;\n\t"
		"addc.cc.u32 %3, %3, 0;\n\t"
		"addc.cc.u32 %4, %4, 0;\n\t"
		"addc.cc.u32 %5, %5, 0;\n\t"
		"addc.cc.u32 %6, %6, 0;\n\t"
		"addc.u32    %7, %7, 0;"
		: "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3),
		  "+r"(r4), "+r"(r5), "+r"(r6), "+r"(r7)
		: "r"(p0), "r"(p1), "r"(p2));

	r->d[0] = r0; r->d[1] = r1; r->d[2] = r2; r->d[3] = r3;
	r->d[4] = r4; r->d[5] = r5; r->d[6] = r6; r->d[7] = r7;
}
#endif /* PROFANITY_PTX_MP */

// Which of the two the rest of this file gets.
static inline void mp_mul_mod_word_sub(mp_number * const r, const mp_word w, const bool withModHigher) {
#ifdef PROFANITY_PTX_MP
	mp_mul_mod_word_sub_ptx(r, w, withModHigher);
#else
	mp_mul_mod_word_sub_portable(r, w, withModHigher);
#endif
}

// Modular multiplication. Based on Algorithm 3 (and a series of hunches) from this article:
// https://www.esat.kuleuven.be/cosic/publications/article-1191.pdf
// When I first implemented it I never encountered a situation where the additional end steps
// of adding or subtracting the modulo was necessary. Maybe it's not for the particular modulo
// used in secp256k1, maybe the overflow bit can be skipped in to avoid 8 subtractions and
// trade it for the final steps? Maybe the final steps are necessary but seldom needed?
// I have no idea, for the time being I'll leave it like this, also see the comments at the
// beginning of this document under the title "Cutting corners".
void mp_mod_mul(mp_number * const r, const mp_number * const X, const mp_number * const Y) {
	mp_number Z = { {0} };
	mp_word extraWord;

	for (int i = MP_WORDS - 1; i >= 0; --i) {
		// Z = Z * 2^32
		extraWord = Z.d[7]; Z.d[7] = Z.d[6]; Z.d[6] = Z.d[5]; Z.d[5] = Z.d[4]; Z.d[4] = Z.d[3]; Z.d[3] = Z.d[2]; Z.d[2] = Z.d[1]; Z.d[1] = Z.d[0]; Z.d[0] = 0;

		// Z = Z + X * Y_i
		bool overflow = mp_mul_word_add_extra(&Z, X, Y->d[i], &extraWord);

		// Z = Z - qM
		mp_mul_mod_word_sub(&Z, extraWord, overflow);
	}

	*r = Z;
}

// Modular inversion of a number. 
void mp_mod_inverse(mp_number * const r) {
	mp_number A = { { 1 } };
	mp_number C = { { 0 } };
	mp_number v = mod;

	mp_word extraA = 0;
	mp_word extraC = 0;

	while (r->d[0] || r->d[1] || r->d[2] || r->d[3] || r->d[4] || r->d[5] || r->d[6] || r->d[7]) {
		while (!(r->d[0] & 1)) {
			mp_shr(r);
			if (A.d[0] & 1) {
				extraA += mp_add_mod(&A);
			}

			mp_shr_extra(&A, &extraA);
		}

		while (!(v.d[0] & 1)) {
			mp_shr(&v);
			if (C.d[0] & 1) {
				extraC += mp_add_mod(&C);
			}

			mp_shr_extra(&C, &extraC);
		}

		if (mp_gte(r, &v)) {
			mp_sub(r, r, &v);
			mp_add_more(&A, &extraA, &C, &extraC);
		}
		else {
			mp_sub(&v, &v, r);
			mp_add_more(&C, &extraC, &A, &extraA);
		}
	}

	while (extraC) {
		extraC -= mp_sub_mod(&C);
	}

	v = mod;
	mp_sub(r, &v, &C);
}

/* ------------------------------------------------------------------------ */
/* Elliptic point and addition (with caveats).                              */
/* ------------------------------------------------------------------------ */
typedef struct {
	mp_number x;
	mp_number y;
} point;

// Elliptical point addition
// Does not handle points sharing X coordinate, this is a deliberate design choice.
// For more information on this choice see the beginning of this file.
void point_add(point * const r, point * const p, point * const o) {
	mp_number tmp;
	mp_number newX;
	mp_number newY;

	mp_mod_sub(&tmp, &o->x, &p->x);

	mp_mod_inverse(&tmp);

	mp_mod_sub(&newX, &o->y, &p->y);
	mp_mod_mul(&tmp, &tmp, &newX);

	mp_mod_mul(&newX, &tmp, &tmp);
	mp_mod_sub(&newX, &newX, &p->x);
	mp_mod_sub(&newX, &newX, &o->x);

	mp_mod_sub(&newY, &p->x, &newX);
	mp_mod_mul(&newY, &newY, &tmp);
	mp_mod_sub(&newY, &newY, &p->y);

	r->x = newX;
	r->y = newY;
}

/* ------------------------------------------------------------------------ */
/* Profanity.                                                               */
/* ------------------------------------------------------------------------ */
// foundVariant says which of the addresses a point yields this one was, and so
// what has to be done to the private key behind it — see PROFANITY_VARIANTS and
// profanity_iterate. Zero is the point itself, whose key is the scalar the host
// prints unchanged; one is its negation, whose key is that scalar's.
//
// foundRound says which of the launch's PROFANITY_ROUNDS point additions the
// address turned up at, every one of which lands on a different scalar. Without
// it a launch doing more than one would have no way to say which.
typedef struct {
	uint found;
	uint foundId;
	uint foundRound;
	uint foundVariant;
	uchar foundHash[20];
} result;

void profanity_init_seed(__global const point * const precomp, point * const p, bool * const pIsFirst, const size_t precompOffset, const ulong seed) {
	point o;

	for (uchar i = 0; i < 8; ++i) {
		const uchar shift = i * 8;
		const uchar byte = (seed >> shift) & 0xFF;

		if (byte) {
			o = precomp[precompOffset + i * 255 + byte - 1];
			if (*pIsFirst) {
				*p = o;
				*pIsFirst = false;
			}
			else {
				point_add(p, p, &o);
			}
		}
	}
}

// The part of every work item's starting point that does not depend on the work
// item. Only the top limb of the seed carries the index, so the other three
// contribute the same point to all of them — three quarters of the scalar
// multiplication below, repeated once per point in the search and thrown away
// each time. Done once here, and read back out of pUniform by every work item.
//
// createSeed keeps those three limbs from all being zero, so there is always a
// point here — the identity is not something the affine point_add can be handed,
// and every work item below relies on starting from a real one.
__kernel void profanity_init_uniform(__global const point * const precomp, __global point * const pUniform, const ulong4 seed) {
	point p;
	bool bIsFirst = true;

	profanity_init_seed(precomp, &p, &bIsFirst, 8 * 255 * 0, seed.x);
	profanity_init_seed(precomp, &p, &bIsFirst, 8 * 255 * 1, seed.y);
	profanity_init_seed(precomp, &p, &bIsFirst, 8 * 255 * 2, seed.z);

	*pUniform = p;
}

__kernel void profanity_init(__global const point * const precomp, __global mp_number * const pDeltaX, __global mp_number * const pPrevLambda, __global result * const pResult, const ulong4 seed, const ulong4 seedX, const ulong4 seedY, __global const point * const pUniform) {
	const size_t id = get_global_id(0);
	point p = {
		.x = {.d = {
			seedX.x & 0xFFFFFFFF, seedX.x >> 32,
			seedX.y & 0xFFFFFFFF, seedX.y >> 32,
			seedX.z & 0xFFFFFFFF, seedX.z >> 32,
			seedX.w & 0xFFFFFFFF, seedX.w >> 32,
		}},
		.y = {.d = {
			seedY.x & 0xFFFFFFFF, seedY.x >> 32,
			seedY.y & 0xFFFFFFFF, seedY.y >> 32,
			seedY.z & 0xFFFFFFFF, seedY.z >> 32,
			seedY.w & 0xFFFFFFFF, seedY.w >> 32,
		}},
	};

	mp_number tmp1, tmp2;
	point tmp3;

	// Calculate k*G where k = seed.wzyx (in other words, find the point indicated
	// by the private key represented in seed). The low three limbs of the seed are
	// the same for every work item and profanity_init_uniform did them once for
	// all of them; only the top one, which carries the work item's index, is left
	// to do here. createSeed keeps those three from all being zero, so there is
	// always a point to start from.
	point p_random = *pUniform;
	bool bIsFirst = false;

	profanity_init_seed(precomp, &p_random, &bIsFirst, 8 * 255 * 3, seed.w + id);
	point_add(&p, &p, &p_random);

	// Calculate current lambda in this point
	mp_mod_sub_gx(&tmp1, &p.x);
	mp_mod_inverse(&tmp1);

	mp_mod_sub_gy(&tmp2, &p.y); 
	mp_mod_mul(&tmp1, &tmp1, &tmp2);

	// Jump to next point (precomp[0] is the generator point G)
	tmp3 = precomp[0];
	point_add(&p, &tmp3, &p);

	// pDeltaX should contain the delta (x - G_x)
	mp_mod_sub_gx(&p.x, &p.x);

	pDeltaX[id] = p.x;
	pPrevLambda[id] = tmp1;

	// One entry each rather than the whole buffer each: every work item clearing
	// all of it meant PROFANITY_MAX_SCORE writes per point to the same handful of
	// addresses, which is millions of stores contending over one cache line for
	// work that forty of them can do once. The seeding pass starts at zero, so
	// these are the first work items to run.
	if (id < PROFANITY_MAX_SCORE + 1) {
		pResult[id].found = 0;
	}
}

// This kernel calculates several modular inversions at once with just one inverse.
// It's an implementation of Algorithm 2.11 from Modern Computer Arithmetic:
// https://members.loria.fr/PZimmermann/mca/pub226.html

#if PROFANITY_INVERSE_STRIP == 0 && PROFANITY_INVERSE_GROUP == 0
// Single-level (default): invert 1 batch of PROFANITY_INVERSE_SIZE points.
#define PROFANITY_TWO_LEVEL_INVERSE 0
#elif PROFANITY_INVERSE_STRIP > 0 && PROFANITY_INVERSE_GROUP > 0
// Two-level: batch PROFANITY_INVERSE_STRIP points, and share an inverse across
// PROFANITY_INVERSE_GROUP strips. Slower on some GPUs, so is opt-in only.
#define PROFANITY_TWO_LEVEL_INVERSE 1
#else
#error "PROFANITY_INVERSE_STRIP and PROFANITY_INVERSE_GROUP must both be 0 or both be non-zero"
#endif

// Both inversion schemes are fused into the scoring kernel at the end of this
// file rather than run as a kernel of their own. An inverse is consumed by the
// point addition that immediately follows it, so writing every one of them out
// to global memory for a second kernel to read straight back in was 64 bytes a
// point of traffic and a kernel launch, for a value that never needed to leave
// registers. See PROFANITY_ROUND, which is where the fusing happens.

static inline uchar profanity_byte(const uint * const address, const int i) {
	return (uchar)(address[i >> 2] >> ((i & 3) << 3));
}

// This kernel performs en elliptical curve point addition. See:
// https://en.wikipedia.org/wiki/Elliptic_curve_point_multiplication#Point_addition
// I've made one mathematical optimization by never calculating x_r,
// instead I directly calculate the delta (x_q - x_p). It's for this
// delta we calculate the inverse and that's already been done at this
// point. By calculating and storing the next delta we don't have to
// calculate the delta in profanity_inverse_multiple which saves us
// one call to mp_mod_sub per point, but inversely we have to introduce
// an addition (or addition by subtracting a negative number) in
// profanity_end to retrieve the actual x-coordinate instead of the
// delta as that's what used for calculating the public hash.
//
// One optimization is when calculating the next y-coordinate. As
// given in the wiki the next y-coordinate is given by:
//   y_r = λ²(x_p - x_r) - y_p
// In our case the other point P is the generator point so x_p = G_x,
// a constant value. x_r is the new point which we never calculate, we
// calculate the new delta (x_q - x_p) instead. Let's denote the delta
// with d and new delta as d' and remove notation for points P and Q and
// instead refeer to x_p as G_x, y_p as G_y and x_q as x, y_q as y.
// Furthermore let's denote new x by x' and new y with y'.
//
// Then we have:
//   d = x - G_x <=> x = d + G_x
//   x' = λ² - G_x - x <=> x_r = λ² - G_x - d - G_x = λ² - 2G_x - d
//   
//   d' = x' - G_x = λ² - 2G_x - d - G_x = λ² - 3G_x - d
//
// So we see that the new delta d' can be calculated with the same
// amount of steps as the new x'; 3G_x is still just a single constant.
//
// Now for the next y-coordinate in the new notation:
//   y' =  λ(G_x - x') - G_y
//
// If we expand the expression (G_x - x') we can see that this
// subtraction can be removed! Saving us one call to mp_mod_sub!
//   G_x - x' = -(x' - G_x) = -d'
// It has the same value as the new delta but negated! We can avoid
// having to perform the negation by:
//   y' = λ * -d' - G_y = -G_y - (λ * d')
//
// We can just precalculate the constant -G_y and we get rid of one
// subtraction. Woo!
//
// But we aren't done yet! Let's expand the expression for the next
// lambda, λ'. We have:
//   λ' = (y' - G_y) / d'
//      = (-λ * d' - G_y - G_y) / d' 
//      = (-λ * d' - 2*G_y) / d' 
//      = -λ - 2*G_y / d' 
//
// So the next lambda value can be calculated from the old one. This in
// and of itself is not so interesting but the fact that the term -2 * G_y
// is a constant is! Since it's constant it'll be the same value no matter
// which point we're currently working with. This means that this factor
// can be multiplied in during the inversion, and just with one call per
// inversion instead of one call per point! This is small enough to be
// negligible and thus we've reduced our point addition from three
// multi-precision multiplications to just two! Wow. Just wow.
//
// There is additional overhead introduced by storing the previous lambda
// but it's still a net gain. To additionally decrease memory access
// overhead I never any longer store the Y coordinate. Instead I
// calculate it at the end directly from the lambda and deltaX.
// 
// In addition to this some algebraic re-ordering has been done to move
// constants into the same argument to a new function mp_mod_sub_const
// in hopes that using constant storage instead of private storage
// will aid speeds.
//
// The address a point hashes to, in private memory, where the scoring below
// grades it without a round trip through global memory. Split out of
// profanity_iterate so that one point can be hashed more than once: with
// PROFANITY_VARIANTS at two it is called for the point and for its negation.
static inline void profanity_address(const mp_number * const x, const mp_number * const y, const uchar bContract, uint * const address) {
	ethhash h = { { 0 } };

	// Initialize Keccak structure with point coordinates in big endian
	h.d[0] = bswap32(x->d[MP_WORDS - 1]);
	h.d[1] = bswap32(x->d[MP_WORDS - 2]);
	h.d[2] = bswap32(x->d[MP_WORDS - 3]);
	h.d[3] = bswap32(x->d[MP_WORDS - 4]);
	h.d[4] = bswap32(x->d[MP_WORDS - 5]);
	h.d[5] = bswap32(x->d[MP_WORDS - 6]);
	h.d[6] = bswap32(x->d[MP_WORDS - 7]);
	h.d[7] = bswap32(x->d[MP_WORDS - 8]);
	h.d[8] = bswap32(y->d[MP_WORDS - 1]);
	h.d[9] = bswap32(y->d[MP_WORDS - 2]);
	h.d[10] = bswap32(y->d[MP_WORDS - 3]);
	h.d[11] = bswap32(y->d[MP_WORDS - 4]);
	h.d[12] = bswap32(y->d[MP_WORDS - 5]);
	h.d[13] = bswap32(y->d[MP_WORDS - 6]);
	h.d[14] = bswap32(y->d[MP_WORDS - 7]);
	h.d[15] = bswap32(y->d[MP_WORDS - 8]);
	h.d[16] ^= 0x01; // length 64

	sha3_keccakf(&h);

	// The address is the low 20 bytes of the hash, words 3 through 7.
	address[0] = h.d[3];
	address[1] = h.d[4];
	address[2] = h.d[5];
	address[3] = h.d[6];
	address[4] = h.d[7];

	if (bContract) {
		ethhash c;

		// keccak(0xd6, 0x94, address, 0x80) — the RLP of this account and a
		// nonce of one, which is where the first contract it deploys lands.
		// Twenty-three bytes and the byte that ends the message after them, so
		// three lanes and twenty-two empty ones.
		//
		// Written as whole lanes at indices the compiler knows for the same
		// reason profanity_create2 is, and measured at +3.4% of a --contract
		// search on an RTX 4090 against the loop of byte stores this replaced.
		// The same rewrite is worth nothing at all to the account address above,
		// which is why only this half of the function has had it — see
		// tests/bench_address_state.cpp for both numbers.
		//
		// The address is five uints holding it little endian, so each lane takes
		// whichever of them land in it, halved where one crosses the boundary.
		c.q[0] = (ulong)0xd6                        // byte 0
			| ((ulong)0x94 << 8)                    // byte 1
			| ((ulong)address[0] << 16)             // bytes 2-5
			| ((ulong)(address[1] & 0xFFFF) << 48); // bytes 6-7
		c.q[1] = (ulong)(address[1] >> 16)          // bytes 8-9
			| ((ulong)address[2] << 16)             // bytes 10-13
			| ((ulong)(address[3] & 0xFFFF) << 48); // bytes 14-15
		c.q[2] = (ulong)(address[3] >> 16)          // bytes 16-17
			| ((ulong)address[4] << 16)             // bytes 18-21
			| ((ulong)0x80 << 48)                   // byte 22
			| ((ulong)0x01 << 56);                  // byte 23, ending the message
		c.q[3] = 0; c.q[4] = 0; c.q[5] = 0; c.q[6] = 0;
		c.q[7] = 0; c.q[8] = 0; c.q[9] = 0; c.q[10] = 0;
		c.q[11] = 0; c.q[12] = 0; c.q[13] = 0; c.q[14] = 0;
		c.q[15] = 0; c.q[16] = 0; c.q[17] = 0; c.q[18] = 0;
		c.q[19] = 0; c.q[20] = 0; c.q[21] = 0; c.q[22] = 0;
		c.q[23] = 0; c.q[24] = 0;

		sha3_keccakf(&c);

		address[0] = c.d[3];
		address[1] = c.d[4];
		address[2] = c.d[5];
		address[3] = c.d[6];
		address[4] = c.d[7];
	}
}

// After the above point addition this calculates the public address or
// addresses corresponding to the point, and writes five words for each into
// `addresses`. There are PROFANITY_VARIANTS of them.
// dX and tmp arrive by value: the caller has both in registers already, dX from
// the pass that built the prefix products and tmp as the inverse it just
// unwound, and neither has any business going out to global memory and back.
static inline void profanity_iterate(mp_number dX, mp_number tmp, __global mp_number * const pDeltaX, __global mp_number * const pPrevLambda, const size_t id, const uchar bContract, uint * const addresses) {
	// negativeGx = 0x8641998106234453aa5f9d6a3178f4f8fd640324d231d726a60d7ea3e907e497
	mp_number negativeGx = { {0xe907e497, 0xa60d7ea3, 0xd231d726, 0xfd640324, 0x3178f4f8, 0xaa5f9d6a, 0x06234453, 0x86419981 } };

	mp_number lambda = pPrevLambda[id];

	// λ' = - (2G_y) / d' - λ <=> lambda := pInversedNegativeDoubleGy[id] - pPrevLambda[id]
	mp_mod_sub(&lambda, &tmp, &lambda);

	// λ² = λ * λ <=> tmp := lambda * lambda = λ²
	mp_mod_mul(&tmp, &lambda, &lambda);

	// d' = λ² - d - 3g = (-3g) - (d - λ²) <=> x := tripleNegativeGx - (x - tmp)
	mp_mod_sub(&dX, &dX, &tmp);
	mp_mod_sub_const(&dX, &tripleNegativeGx, &dX);

	pDeltaX[id] = dX;
	pPrevLambda[id] = lambda;

	// Calculate y from dX and lambda
	// y' = (-G_Y) - λ * d' <=> p.y := negativeGy - (p.y * p.x)
	mp_mod_mul(&tmp, &lambda, &dX);
	mp_mod_sub_const(&tmp, &negativeGy, &tmp);

	// Restore X coordinate from delta value
	mp_mod_sub(&dX, &dX, &negativeGx);

	profanity_address(&dX, &tmp, bContract, addresses);

#if PROFANITY_VARIANTS > 1
	// -P = (x, -y) shares this point's x, so a second address costs one modular
	// subtraction and a second keccak rather than another point addition. Its
	// private key is this one's negated, mod the order of the curve, which is
	// why a result carries the variant it was found at.
	//
	// mod - y rather than 0 - y: y is never zero here, since that would be a
	// point of order two and this curve has none, so the subtraction can never
	// borrow and what it leaves is already reduced.
	mp_number negY;
	mp_mod_sub_const(&negY, &mod, &tmp);

	profanity_address(&dX, &negY, bContract, addresses + 5);
#endif

#if PROFANITY_VARIANTS > 2
	// secp256k1 has an endomorphism ψ(x, y) = (βx, y), and ψ(P) = λP for a λ
	// that cubes to one modulo the order of the curve. So βx is another point's
	// x coordinate for one modular multiplication, and β²x a third's for one
	// more — six addresses in all once each is taken with its negation, off a
	// single point addition.
	//
	// β² is not carried as a constant of its own: β²x is β(βx), which reuses
	// the register the first product is already in.
	//
	// Done after the two above rather than alongside them so that only one of
	// the two products is ever live at once. Everything here is held across a
	// keccak, whose state is fifty registers on its own, and what that costs in
	// occupancy is the thing most likely to take back what the arithmetic saves.
	//
	// beta = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee
	mp_number beta = { {0x719501ee, 0xc1396c28, 0x12f58995, 0x9cf04975, 0xac3434e9, 0x6e64479e, 0x657c0710, 0x7ae96a2b} };

	mp_number betaX;
	mp_mod_mul(&betaX, &dX, &beta);

	profanity_address(&betaX, &tmp, bContract, addresses + 10);
#endif

#if PROFANITY_VARIANTS > 3
	profanity_address(&betaX, &negY, bContract, addresses + 15);
#endif

#if PROFANITY_VARIANTS > 4
	mp_mod_mul(&betaX, &betaX, &beta);

	profanity_address(&betaX, &tmp, bContract, addresses + 20);
#endif

#if PROFANITY_VARIANTS > 5
	profanity_address(&betaX, &negY, bContract, addresses + 25);
#endif
}

// The i'th character of the address as it would be written out: the high nibble
// of a byte comes first, so an even index takes the top half of byte i / 2.
static inline uchar profanity_nibble(const uint * const address, const int i) {
	const uchar byte = profanity_byte(address, i >> 1);
	return (i & 1) ? (byte & 0x0F) : (byte >> 4);
}

// The same character, out of an address whose bytes have had their nibbles
// swapped: that lines the characters up in the order they are written and puts
// the i'th of them at bit 4i, which is a shift and a mask rather than the byte
// extraction and half-selection above. See profanity_score_fn_matching for when
// laying an address out that way pays for itself.
static inline uchar profanity_written_nibble(const uint * const written, const int i) {
	return (uchar)((written[i >> 3] >> ((i & 7) << 2)) & 0xF);
}

void profanity_result_update(const size_t id, const uint round, const uint variant, const uint * const address, __global result * const pResult, const uchar score, const uchar scoreMax, const uchar bAppend) {
	if (!score || score <= scoreMax) {
		return;
	}

	if (bAppend) {
		// With a score floor the bar never moves, so a round has no single best
		// hash to keep — every one that cleared the bar is wanted, and they are
		// appended rather than filed one per score. pResult[0].found counts
		// them and the entries follow it; the host resets that counter before
		// every launch, so the bounds check here also guarantees that no two
		// work items ever write the same slot. The score travels in the entry's
		// own counter, the slot index no longer standing for it.
		const uint at = atomic_inc(&pResult[0].found);

		if (at < PROFANITY_MAX_SCORE) {
			pResult[at + 1].found = score;
			pResult[at + 1].foundId = id;
			pResult[at + 1].foundRound = round;
			pResult[at + 1].foundVariant = variant;

			for (int i = 0; i < 20; ++i) {
				pResult[at + 1].foundHash[i] = profanity_byte(address, i);
			}
		}

		return;
	}

	uchar hasResult = atomic_inc(&pResult[score].found); // NOTE: If "too many" results are found it'll wrap around to 0 again and overwrite last result. Only relevant if global worksize exceeds MAX(uint).

	// Save only one result for each score, the first.
	if (hasResult == 0) {
		pResult[score].foundId = id;
		pResult[score].foundRound = round;
		pResult[score].foundVariant = variant;

		for (int i = 0; i < 20; ++i) {
			pResult[score].foundHash[i] = profanity_byte(address, i);
		}
	}
}

// Prevent the compiler from deleting the keccak behind profanity_iterate
// Scores 1 for address(0), which is unreachable, and 0 on everything else
static inline int profanity_score_fn_benchmark(const uint * const address, __constant const uchar * const data1, __constant const uchar * const data2) {
	uint sum = 0;

	for (int i = 0; i < 5; ++i) {
		sum |= address[i];
	}

	return sum == 0;
}

// A mask shorter than the address floats: it is scored wherever it sits best,
// which is what lets one search cover every offset a pattern could appear at. A
// mask pinning all 40 nibbles has one placement and so stays anchored, padding
// with wildcards being how a caller asks for that.
//
// Mode::matching hands over the pinned nibbles alone — data1 where each sits in
// the mask, data2 what it is — with the count of them and the mask's length in
// the last entry of each. Wildcards cost nothing here because they were never
// stored: an anchored four character pattern is four steps of the inner loop
// rather than the forty its padding would otherwise be walked through.
//
// The score is the pinned nibbles matched in one run from the start of the mask,
// at whichever offset does best: a full match scores every nibble the mask pins
// down and anything below that is progress towards one. Counting nibbles rather
// than bytes is what keeps that ceiling the same at every offset — by bytes a
// four nibble mask would span two of them at an even offset and three at an odd
// one, and the host's rising bar, once an odd offset had reached three, would
// shut every even one out for good.
static inline int profanity_score_fn_matching(const uint * const address, __constant const uchar * const data1, __constant const uchar * const data2) {
	const int pinned = data1[PROFANITY_MODE_DATA - 1];
	const int length = data2[PROFANITY_MODE_DATA - 1];

	// A mask of nothing but wildcards can never score, and the peeled comparison
	// below would read a first pinned character that is not there.
	if (pinned == 0) {
		return 0;
	}

	// A mask as long as an address has the one placement, so there is no scan to
	// do and no offset to beat: walk its pinned characters once, straight out of
	// the packed bytes. Laying the address out first, as the floating path below
	// does, would cost more here than the handful of reads it would save.
	if (length == 40) {
		int run = 0;
		while (run < pinned && profanity_nibble(address, data1[run]) == data2[run]) {
			++run;
		}

		return run;
	}

	// A floating mask is looked for at up to 41 - length offsets, so the address
	// is laid out in the order it is written first — three operations a word,
	// and every character then sits at bit 4i, which is what the search below
	// needs to be able to ask about all forty of them together.
	uint written[5];
	for (int i = 0; i < 5; ++i) {
		written[i] = ((address[i] & 0x0F0F0F0Fu) << 4) | ((address[i] >> 4) & 0x0F0F0F0Fu);
	}

	const int firstAt = data1[0];
	const uint firstIs = (uint)data2[0] * 0x11111111u;
	const int lastAt = 40 - length;

	// Fifteen offsets in sixteen have nothing wrong with them but the mask's
	// first character, so rather than walk every offset to turn most of them
	// away, this finds where that character actually occurs and tries only
	// those. Exclusive-or it into all forty positions at once, fold each result
	// down to whether anything was left over, and what remains marks the places
	// worth looking at — two or three of them, against the thirty-odd an offset
	// by offset scan would step through.
	int score = 0;

	for (int i = 0; i < 5; ++i) {
		uint x = written[i] ^ firstIs;

		// Leave a 1 in the low bit of every character that differed, then invert
		// to mark the ones that did not.
		x |= x >> 2;
		x |= x >> 1;

		uint found = ~x & 0x11111111u;

		while (found) {
			const uint lowest = found & (~found + 1u);
			found ^= lowest;

			// How far up the word that bit sits, which is four times the
			// character's place in it: the population count below one lone bit
			// is its position, and OpenCL 1.2 has that where it has no count of
			// trailing zeros.
			const int character = (i << 3) + (int)(popcount(lowest - 1u) >> 2);
			const int at = character - firstAt;

			// The mask would hang off one end or the other from here.
			if (at < 0 || at > lastAt) {
				continue;
			}

			// The run is its own counter: it only ever advances on a match, so
			// where it stops is both how far the loop got and what the offset
			// scored. The first character is known to match, hence starting at
			// one.
			int run = 1;
			while (run < pinned) {
				if (profanity_written_nibble(written, at + data1[run]) != data2[run]) {
					break;
				}

				++run;
			}

			score = max(score, run);
		}
	}

	return score;
}

static inline int profanity_score_fn_leading(const uint * const address, __constant const uchar * const data1, __constant const uchar * const data2) {
	int score = 0;

	for (int i = 0; i < 20; ++i) {
		const uchar byte = profanity_byte(address, i);

		if ((byte & 0xF0) >> 4 == data1[0]) {
			++score;
		}
		else {
			break;
		}

		if ((byte & 0x0F) == data1[0]) {
			++score;
		}
		else {
			break;
		}
	}

	return score;
}

// Counting the characters that fall within a range, asked of eight of them at a
// time. Comparing characters where they sit is what a packed word will not let
// you do — a borrow out of one would land in its neighbour — so each word is
// split into the high nibbles of its bytes and the low ones, leaving every
// character alone in a byte with four bits above it to spare.
//
// One of those spare bits is then a guard. Setting it before subtracting the
// bottom of the range gives 16 + character - minimum, which is between 1 and 31
// and so cannot borrow into the next byte, and whose guard bit is still standing
// exactly when the character reached the minimum. Turning the subtraction around
// asks the other half of the question, and the guards left standing in both are
// the characters that fall inside the range.
//
// Mode::range sends a range whose ends agree to profanity_score_fn_rangeequal
// instead, which has less to do than this.
static inline int profanity_score_fn_range(const uint * const address, __constant const uchar * const data1, __constant const uchar * const data2) {
	const uint spread = 0x0F0F0F0Fu;
	const uint guards = 0x10101010u;

	const uint atLeast = (uint)data1[0] * 0x01010101u;
	const uint atMost = ((uint)data2[0] * 0x01010101u) | guards;

	int score = 0;

	for (int i = 0; i < 5; ++i) {
		const uint low = address[i] & spread;
		const uint high = (address[i] >> 4) & spread;

		const uint lowIn = ((low | guards) - atLeast) & (atMost - low) & guards;
		const uint highIn = ((high | guards) - atLeast) & (atMost - high) & guards;

		// The two sets of guards sit four bits apart once one is shifted down,
		// so a single count covers both.
		score += popcount(lowIn | (highIn >> 4));
	}

	return score;
}

// A range of a single character is a count of it, which is the whole of what a
// `--range -m N -M N` search asks and what --zeros is built on. Counting does
// not care what order the characters come in, so unlike the matching kernel this
// needs no laying out: exclusive-or the character into every position at once,
// fold each result down to whether it was non-zero, and the population count of
// what is left is how many characters were not the one wanted.
//
// Mode::range picks this over profanity_score_fn_range when the two ends of the
// range agree; a range spanning several characters still goes character by
// character.
static inline int profanity_score_fn_rangeequal(const uint * const address, __constant const uchar * const data1, __constant const uchar * const data2) {
	const uint wanted = (uint)data1[0] * 0x11111111u;

	int missed = 0;

	for (int i = 0; i < 5; ++i) {
		uint x = address[i] ^ wanted;

		// Leave a 1 in the low bit of every character that was not the one.
		x |= x >> 2;
		x |= x >> 1;
		x &= 0x11111111u;

		missed += popcount(x);
	}

	return 40 - missed;
}

// Counting is not a per-character question either, so this asks it of the whole
// address at once as well: fold every byte down to whether any of its bits were
// set, and what the population count then leaves is how many were not zero.
static inline int profanity_score_fn_zerobytes(const uint * const address, __constant const uchar * const data1, __constant const uchar * const data2) {
	int missed = 0;

	for (int i = 0; i < 5; ++i) {
		uint x = address[i];

		// Leave a 1 in the low bit of every byte that held something.
		x |= x >> 4;
		x |= x >> 2;
		x |= x >> 1;
		x &= 0x01010101u;

		missed += popcount(x);
	}

	return 20 - missed;
}

static inline int profanity_score_fn_leadingrange(const uint * const address, __constant const uchar * const data1, __constant const uchar * const data2) {
	int score = 0;

	for (int i = 0; i < 20; ++i) {
		const uchar byte = profanity_byte(address, i);
		const uchar first = (byte & 0xF0) >> 4;
		const uchar second = (byte & 0x0F);

		if (first >= data1[0] && first <= data2[0]) {
			++score;
		}
		else {
			break;
		}

		if (second >= data1[0] && second <= data2[0]) {
			++score;
		}
		else {
			break;
		}
	}

	return score;
}

static inline int profanity_score_fn_mirror(const uint * const address, __constant const uchar * const data1, __constant const uchar * const data2) {
	int score = 0;

	for (int i = 0; i < 10; ++i) {
		const uchar left = profanity_byte(address, 9 - i);
		const uchar right = profanity_byte(address, 10 + i);

		const uchar leftLeft = (left & 0xF0) >> 4;
		const uchar leftRight = (left & 0x0F);

		const uchar rightLeft = (right & 0xF0) >> 4;
		const uchar rightRight = (right & 0x0F);

		if (leftRight != rightLeft) {
			break;
		}

		++score;

		if (leftLeft != rightRight) {
			break;
		}

		++score;
	}

	return score;
}

static inline int profanity_score_fn_doubles(const uint * const address, __constant const uchar * const data1, __constant const uchar * const data2) {
	int score = 0;

	for (int i = 0; i < 20; ++i) {
		const uchar byte = profanity_byte(address, i);

		if ((((byte >> 4) ^ byte) & 0x0f) == 0) {
			++score;
		}
		else {
			break;
		}
	}

	return score;
}

// Grades the variant'th of the addresses profanity_iterate wrote. Spelled out
// per variant rather than looped so that the index into `addresses` stays a
// constant and the array stays in registers.
#define PROFANITY_SCORE_ONE(NAME, V) \
	{ \
		const uint * const address = addresses + (V) * 5; \
		const int score = profanity_score_fn_##NAME(address, data1, data2); \
		profanity_result_update(id, round, (V), address, pResult, score, scoreMax, bAppend); \
	}

// Spelled out per count rather than looped, for the same reason as above: the
// index into `addresses` has to stay a constant for the array to stay in
// registers.
#if PROFANITY_VARIANTS == 1
#define PROFANITY_SCORE_VARIANTS(NAME) \
	PROFANITY_SCORE_ONE(NAME, 0)
#elif PROFANITY_VARIANTS == 2
#define PROFANITY_SCORE_VARIANTS(NAME) \
	PROFANITY_SCORE_ONE(NAME, 0) PROFANITY_SCORE_ONE(NAME, 1)
#elif PROFANITY_VARIANTS == 3
#define PROFANITY_SCORE_VARIANTS(NAME) \
	PROFANITY_SCORE_ONE(NAME, 0) PROFANITY_SCORE_ONE(NAME, 1) \
	PROFANITY_SCORE_ONE(NAME, 2)
#elif PROFANITY_VARIANTS == 4
#define PROFANITY_SCORE_VARIANTS(NAME) \
	PROFANITY_SCORE_ONE(NAME, 0) PROFANITY_SCORE_ONE(NAME, 1) \
	PROFANITY_SCORE_ONE(NAME, 2) PROFANITY_SCORE_ONE(NAME, 3)
#elif PROFANITY_VARIANTS == 5
#define PROFANITY_SCORE_VARIANTS(NAME) \
	PROFANITY_SCORE_ONE(NAME, 0) PROFANITY_SCORE_ONE(NAME, 1) \
	PROFANITY_SCORE_ONE(NAME, 2) PROFANITY_SCORE_ONE(NAME, 3) \
	PROFANITY_SCORE_ONE(NAME, 4)
#else
#define PROFANITY_SCORE_VARIANTS(NAME) \
	PROFANITY_SCORE_ONE(NAME, 0) PROFANITY_SCORE_ONE(NAME, 1) \
	PROFANITY_SCORE_ONE(NAME, 2) PROFANITY_SCORE_ONE(NAME, 3) \
	PROFANITY_SCORE_ONE(NAME, 4) PROFANITY_SCORE_ONE(NAME, 5)
#endif

// One point, from the inverse the caller unwound for it through to its score.
// Generated per scoring mode because OpenCL C has no function pointers, so the
// scoring function has to be named at compile time.
#define PROFANITY_POINT_FN(NAME) \
static inline void profanity_point_##NAME( \
		const mp_number dX, \
		const mp_number inverse, \
		__global mp_number * const pDeltaX, \
		__global mp_number * const pPrevLambda, \
		__global result * const pResult, \
		__constant const uchar * const data1, \
		__constant const uchar * const data2, \
		const uchar scoreMax, \
		const uchar bAppend, \
		const uchar bContract, \
		const size_t id, \
		const uint round) { \
	uint addresses[PROFANITY_VARIANTS * 5]; \
	profanity_iterate(dX, inverse, pDeltaX, pPrevLambda, id, bContract, addresses); \
	PROFANITY_SCORE_VARIANTS(NAME) \
}

// What every point in the round is handed to, spelled once so the two inversion
// schemes below can each stay about their own business.
#define PROFANITY_POINT(NAME, IDX, DX, INV) \
	profanity_point_##NAME((DX), (INV), pDeltaX, pPrevLambda, pResult, data1, data2, \
		scoreMax, bAppend, bContract, (IDX), round)

#if PROFANITY_TWO_LEVEL_INVERSE

// A work item multiplies PROFANITY_INVERSE_STRIP points into prefix products it
// holds in registers. Prefix and suffix scans across the work group's strip
// products then leave each item the product of every strip but its own, so a
// single inverse serves all PROFANITY_INVERSE_STRIP * PROFANITY_INVERSE_GROUP
// points, which is worth arranging: an inverse costs on the order of 190 modular
// multiplications.
//
// Each inverse is handed straight to the point addition that wanted it, so
// nothing here writes an inverse out and nothing reads one back.
#define PROFANITY_KERNEL_ATTRIBUTES __attribute__((reqd_work_group_size(PROFANITY_INVERSE_GROUP, 1, 1)))
#define PROFANITY_POINTS_PER_ITEM (PROFANITY_INVERSE_STRIP)

#define PROFANITY_ROUND(NAME) \
	{ \
		mp_number copy1, copy2, other; \
		mp_number buffer[PROFANITY_INVERSE_STRIP]; \
		\
		/* buffer[i] = pDeltaX[id] * pDeltaX[id + G] * ... * pDeltaX[id + i * G]. */ \
		/* Unrolled, or the indices stay dynamic and the array leaves registers. */ \
		buffer[0] = pDeltaX[id]; \
		_Pragma("unroll") \
		for (uint i = 1; i < PROFANITY_INVERSE_STRIP; ++i) { \
			other = pDeltaX[id + i * PROFANITY_INVERSE_GROUP]; \
			mp_mod_mul(&buffer[i], &other, &buffer[i - 1]); \
		} \
		\
		/* The round before this one read prefix and suffix after its own last */ \
		/* barrier, so nothing may overwrite them until every item is past that. */ \
		barrier(CLK_LOCAL_MEM_FENCE); \
		prefix[lid] = buffer[PROFANITY_INVERSE_STRIP - 1]; \
		suffix[lid] = buffer[PROFANITY_INVERSE_STRIP - 1]; \
		barrier(CLK_LOCAL_MEM_FENCE); \
		\
		for (uint d = 1; d < PROFANITY_INVERSE_GROUP; d <<= 1) { \
			copy1 = prefix[lid]; \
			copy2 = suffix[lid]; \
			if (lid >= d) { other = prefix[lid - d]; mp_mod_mul(&copy1, &copy1, &other); } \
			if (lid + d < PROFANITY_INVERSE_GROUP) { other = suffix[lid + d]; mp_mod_mul(&copy2, &copy2, &other); } \
			barrier(CLK_LOCAL_MEM_FENCE); \
			prefix[lid] = copy1; \
			suffix[lid] = copy2; \
			barrier(CLK_LOCAL_MEM_FENCE); \
		} \
		\
		/* Take the inverse of all x-values combined, with -2G_y multiplied in. */ \
		if (lid == 0) { \
			copy1 = prefix[PROFANITY_INVERSE_GROUP - 1]; \
			mp_mod_inverse(&copy1); \
			mp_mod_mul(&copy1, &copy1, &negativeDoubleGy); \
			groupInverse = copy1; \
		} \
		barrier(CLK_LOCAL_MEM_FENCE); \
		\
		copy1 = groupInverse; \
		if (lid > 0) { other = prefix[lid - 1]; mp_mod_mul(&copy1, &copy1, &other); } \
		if (lid + 1 < PROFANITY_INVERSE_GROUP) { other = suffix[lid + 1]; mp_mod_mul(&copy1, &copy1, &other); } \
		\
		/* Multiply out each individual inverse and spend it where it is. The */ \
		/* delta is read before the point addition overwrites it, since the */ \
		/* unwind below needs the value that went into the product. */ \
		\
		/* Deliberately not unrolled, unlike the prefix loop above. What this */ \
		/* body holds is a whole point -- the addition, PROFANITY_VARIANTS */ \
		/* keccak permutations and the scoring of each -- so unrolling it */ \
		/* inlines all of that PROFANITY_INVERSE_STRIP times over. At a strip */ \
		/* of eight that is enough code that NVIDIA's compiler stops coming */ \
		/* back: the build hangs rather than fails, at any variant count. The */ \
		/* prefix loop above is a single modular multiplication and unrolls */ \
		/* harmlessly, which is the difference. */ \
		for (uint i = PROFANITY_INVERSE_STRIP - 1; i > 0; --i) { \
			mp_mod_mul(&copy2, &copy1, &buffer[i - 1]); \
			other = pDeltaX[id + i * PROFANITY_INVERSE_GROUP]; \
			PROFANITY_POINT(NAME, id + i * PROFANITY_INVERSE_GROUP, other, copy2); \
			mp_mod_mul(&copy1, &copy1, &other); \
		} \
		\
		PROFANITY_POINT(NAME, id, buffer[0], copy1); \
	}

#define PROFANITY_ROUND_LOCALS \
	const uint lid = get_local_id(0); \
	/* Strided by the group, so the items of a group read adjacent addresses. */ \
	const size_t id = get_group_id(0) * (size_t)(PROFANITY_INVERSE_GROUP * PROFANITY_INVERSE_STRIP) + lid; \
	__local mp_number prefix[PROFANITY_INVERSE_GROUP], suffix[PROFANITY_INVERSE_GROUP], groupInverse;

#else /* !PROFANITY_TWO_LEVEL_INVERSE */

// Single-level inverse: one work item handles a whole batch of
// PROFANITY_INVERSE_SIZE points by itself, with no local memory and no barriers.
//
// My RX 480 is very sensitive to changes in the second loop and sometimes I have
// to make seemingly non-functional changes to the code to make the compiler
// generate the most optimized version.
#define PROFANITY_KERNEL_ATTRIBUTES
#define PROFANITY_POINTS_PER_ITEM (PROFANITY_INVERSE_SIZE)

#define PROFANITY_ROUND(NAME) \
	{ \
		mp_number copy1, copy2; \
		mp_number buffer[PROFANITY_INVERSE_SIZE]; \
		mp_number buffer2[PROFANITY_INVERSE_SIZE]; \
		\
		/* buffer[i] = pDeltaX[id] * ... * pDeltaX[id + i], buffer2[i] = pDeltaX[id + i] */ \
		buffer[0] = pDeltaX[id]; \
		for (uint i = 1; i < PROFANITY_INVERSE_SIZE; ++i) { \
			buffer2[i] = pDeltaX[id + i]; \
			mp_mod_mul(&buffer[i], &buffer2[i], &buffer[i - 1]); \
		} \
		\
		/* Take the inverse of all x-values combined, with -2G_y multiplied in. */ \
		copy1 = buffer[PROFANITY_INVERSE_SIZE - 1]; \
		mp_mod_inverse(&copy1); \
		mp_mod_mul(&copy1, &copy1, &negativeDoubleGy); \
		\
		/* Multiply out each individual inverse and spend it where it is. The */ \
		/* deltas are already in buffer2, which the point addition cannot touch. */ \
		for (uint i = PROFANITY_INVERSE_SIZE - 1; i > 0; --i) { \
			mp_mod_mul(&copy2, &copy1, &buffer[i - 1]); \
			mp_mod_mul(&copy1, &copy1, &buffer2[i]); \
			PROFANITY_POINT(NAME, id + i, buffer2[i], copy2); \
		} \
		\
		PROFANITY_POINT(NAME, id, buffer[0], copy1); \
	}

#define PROFANITY_ROUND_LOCALS \
	const size_t id = get_global_id(0) * PROFANITY_INVERSE_SIZE;

#endif /* PROFANITY_TWO_LEVEL_INVERSE */

// One kernel per scoring mode, each taking a batch of points from their shared
// inverse through to their scores, PROFANITY_ROUNDS times over. bContract is
// uniform across the launch and selects the second hash that turns a sender into
// the contract it deploys at nonce 0.
//
// Rounds are what a launch is worth rather than what a kernel is: the state a
// point carries between them is two mp_numbers, and at one round a launch it
// went out to global memory and came back for every one of them. Held in the
// kernel across PROFANITY_ROUNDS of them instead, that traffic and the launch
// behind it are paid once for however many rounds are asked for.
#define PROFANITY_SCORE_KERNEL(NAME) \
PROFANITY_POINT_FN(NAME) \
PROFANITY_KERNEL_ATTRIBUTES \
__kernel void profanity_iterate_score_##NAME( \
		__global mp_number * const pDeltaX, \
		__global mp_number * const pPrevLambda, \
		__global result * const pResult, \
		__constant const uchar * const data1, \
		__constant const uchar * const data2, \
		const uchar scoreMax, \
		const uchar bAppend, \
		const uchar bContract) { \
	/* negativeDoubleGy = 0x6f8a4b11b2b8773544b60807e3ddeeae05d0976eb2f557ccc7705edf09de52bf */ \
	mp_number negativeDoubleGy = { {0x09de52bf, 0xc7705edf, 0xb2f557cc, 0x05d0976e, 0xe3ddeeae, 0x44b60807, 0xb2b87735, 0x6f8a4b11 } }; \
	PROFANITY_ROUND_LOCALS \
	for (uint round = 0; round < PROFANITY_ROUNDS; ++round) { \
		PROFANITY_ROUND(NAME) \
	} \
}

// A run enqueues one scoring kernel out of the eighteen this file defines —
// nine scorers against each of the two targets — and the other seventeen are
// compiled for nothing. That is not a rounding error in the build: the iterate
// kernels carry the point arithmetic, the shared inverse and keccak through
// PROFANITY_ROUNDS of them each, and dropping the eight a run will not ask for
// takes about four fifths off the compile. What it buys is the wait before a
// search starts, on a machine that is being paid for throughout it.
//
// So the host names the one kernel it is going to create, and this builds that.
// Neither macro defined is the standalone case — the tests compile this file
// with their own harness and ask for kernels by name — and builds all of them.
// The second condition is what empties the half a run does not touch: a CREATE2
// search names a create2 scorer, and needs no iterate kernel at all.
//
// The indirection is what lets a macro be the argument. PROFANITY_SCORE_KERNEL
// pastes its parameter onto a kernel name, and a parameter next to ## is not
// expanded first, so passing the selector straight in would spell the kernel
// after the macro rather than after its value.
#define PROFANITY_SCORE_KERNEL_EXPAND(NAME) PROFANITY_SCORE_KERNEL(NAME)

#if defined(PROFANITY_ITERATE_SCORER)
PROFANITY_SCORE_KERNEL_EXPAND(PROFANITY_ITERATE_SCORER)
#elif !defined(PROFANITY_CREATE2_SCORER)
PROFANITY_SCORE_KERNEL(benchmark)
PROFANITY_SCORE_KERNEL(matching)
PROFANITY_SCORE_KERNEL(leading)
PROFANITY_SCORE_KERNEL(range)
PROFANITY_SCORE_KERNEL(rangeequal)
PROFANITY_SCORE_KERNEL(zerobytes)
PROFANITY_SCORE_KERNEL(leadingrange)
PROFANITY_SCORE_KERNEL(mirror)
PROFANITY_SCORE_KERNEL(doubles)
#endif

/* ------------------------------------------------------------------------ */
/* CREATE2                                                                  */
/* ------------------------------------------------------------------------ */
// A contract deployed with CREATE2 lands at the last twenty bytes of
//
//     keccak256(0xff ++ factory ++ salt ++ keccak256(init_code))
//
// which is eighty-five bytes in, and nothing in it is a public key. So this
// half of the program has no point arithmetic, no inverses and no seeding: the
// search runs over salts, every one of which is as good as every other, and the
// whole preimage but eight bytes of the salt stays put for the length of a run.
// The host lays those eighty-five bytes out once — see buildCreate2Template —
// and each work item copies them in and writes its own counter over the eight.
//
// Which eight is arbitrary and this takes the salt's last, leaving its first
// twenty for the caller a permissioned factory demands and four behind that for
// a nonce telling one device's search from another's. The counter goes in big
// endian so that the salt on a printed line reads as a number counting upwards,
// and because that is the order the host puts it back together in.
// What profanity_init does for the result buffer, on its own, because the rest
// of that kernel is the seeding a CREATE2 search has nothing to seed. Without a
// score floor a slot is written only by the first work item to reach it and
// read for as long as the run lasts, so it has to start at zero.
__kernel void profanity_create2_init(__global result * const pResult) {
	pResult[get_global_id(0)].found = 0;
}

// The state a candidate's hash starts from.
//
// Written out whole lanes at a time, at indices the compiler knows. Worth +7%
// on an RTX 4090 and around a third on a CPU device against what it replaced,
// which zeroed the state and copied the message in through a loop.
//
// Both halves of how it is written matter and neither on its own does anything.
// The same lane construction through a loop measures within noise of the old
// one, and so does the old one with its loops written out — see the variants
// kept in tests/harness.cl, which are there to stop this being tidied back into
// either of them. Why the cliff is exactly there is not established: the
// account address below writes 32-bit halves at constant indices and does not
// fall off it. So this is a measurement about this one site and not a rule to
// apply elsewhere without measuring again.
static inline void profanity_create2(__constant const uint * const pTemplate, const ulong counter, uint * const address) {
	ethhash h;

	// The counter goes in big endian, so the eight bytes it occupies read, as a
	// little-endian lane does, as the counter byte-reversed. It straddles two
	// lanes and reaches neither end of either, so that is a shift each way and
	// a mask over whatever the salt left there — rather than eight byte stores.
	const ulong swapped = bswap64(counter);
	const ulong keepLow = ((ulong)1 << PROFANITY_CREATE2_COUNTER_SHIFT) - 1;

	h.q[0] = (ulong)pTemplate[0] | ((ulong)pTemplate[1] << 32);
	h.q[1] = (ulong)pTemplate[2] | ((ulong)pTemplate[3] << 32);
	h.q[2] = (ulong)pTemplate[4] | ((ulong)pTemplate[5] << 32);
	h.q[3] = (ulong)pTemplate[6] | ((ulong)pTemplate[7] << 32);
	h.q[4] = (ulong)pTemplate[8] | ((ulong)pTemplate[9] << 32);
	h.q[5] = (((ulong)pTemplate[10] | ((ulong)pTemplate[11] << 32)) & keepLow) | (swapped << PROFANITY_CREATE2_COUNTER_SHIFT);
	h.q[6] = (((ulong)pTemplate[12] | ((ulong)pTemplate[13] << 32)) & ~keepLow) | (swapped >> (64 - PROFANITY_CREATE2_COUNTER_SHIFT));
	h.q[7] = (ulong)pTemplate[14] | ((ulong)pTemplate[15] << 32);
	h.q[8] = (ulong)pTemplate[16] | ((ulong)pTemplate[17] << 32);
	h.q[9] = (ulong)pTemplate[18] | ((ulong)pTemplate[19] << 32);
	h.q[10] = (ulong)pTemplate[20] | ((ulong)pTemplate[21] << 32);

	// Eighty-five bytes and a terminator reach lane ten and no further. The bit
	// that ends the block is sha3_keccakf's to set.
	h.q[11] = 0; h.q[12] = 0; h.q[13] = 0; h.q[14] = 0; h.q[15] = 0;
	h.q[16] = 0; h.q[17] = 0; h.q[18] = 0; h.q[19] = 0; h.q[20] = 0;
	h.q[21] = 0; h.q[22] = 0; h.q[23] = 0; h.q[24] = 0;

	sha3_keccakf(&h);

	address[0] = h.d[3];
	address[1] = h.d[4];
	address[2] = h.d[5];
	address[3] = h.d[6];
	address[4] = h.d[7];
}

// One kernel per scoring mode again, over the salts of a launch rather than over
// points. The host hands each launch the counter its first work item searches,
// and a work item takes PROFANITY_ROUNDS consecutive counters from there —
// blocked rather than strided, so that what a work item searched depends on
// PROFANITY_ROUNDS and its own index and on nothing else the host would have to
// agree about.
//
// Strided would want the number of work items, and get_global_size is not it:
// enqueueKernel splits a launch wider than the device's maximum into several
// NDRanges at an offset, so get_global_id runs over the whole launch while
// get_global_size is only ever the piece of it currently in flight. A blocked
// mapping never asks.
//
// What a result was found at is therefore counterBase + foundId * rounds +
// foundRound, which is what the host puts the salt back together from — see
// Dispatcher::report.
#define PROFANITY_CREATE2_KERNEL(NAME) \
__kernel void profanity_create2_score_##NAME( \
		__global result * const pResult, \
		__constant const uchar * const data1, \
		__constant const uchar * const data2, \
		const uchar scoreMax, \
		const uchar bAppend, \
		__constant const uint * const pTemplate, \
		const ulong counterBase) { \
	const size_t id = get_global_id(0); \
	const ulong counter = counterBase + (ulong)id * PROFANITY_ROUNDS; \
	uint address[5]; \
	for (uint round = 0; round < PROFANITY_ROUNDS; ++round) { \
		profanity_create2(pTemplate, counter + round, address); \
		const int score = profanity_score_fn_##NAME(address, data1, data2); \
		profanity_result_update(id, round, 0, address, pResult, score, scoreMax, bAppend); \
	} \
}

// The same selection as the iterate kernels above, and for the same reason.
#define PROFANITY_CREATE2_KERNEL_EXPAND(NAME) PROFANITY_CREATE2_KERNEL(NAME)

#if defined(PROFANITY_CREATE2_SCORER)
PROFANITY_CREATE2_KERNEL_EXPAND(PROFANITY_CREATE2_SCORER)
#elif !defined(PROFANITY_ITERATE_SCORER)
PROFANITY_CREATE2_KERNEL(benchmark)
PROFANITY_CREATE2_KERNEL(matching)
PROFANITY_CREATE2_KERNEL(leading)
PROFANITY_CREATE2_KERNEL(range)
PROFANITY_CREATE2_KERNEL(rangeequal)
PROFANITY_CREATE2_KERNEL(zerobytes)
PROFANITY_CREATE2_KERNEL(leadingrange)
PROFANITY_CREATE2_KERNEL(mirror)
PROFANITY_CREATE2_KERNEL(doubles)
#endif
