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
/* CREATE2 as the state used to be built                                    */
/* ------------------------------------------------------------------------ */
/* What profanity_create2 used to be: the message copied into a zeroed state a
 * word at a time, and the counter written over eight bytes of it through the
 * byte member of the union the state is also read from as ulongs.
 *
 * Kept verbatim for the same two reasons mp_mod_mul_old is — so the faster one
 * has something to be checked against, and so what it is worth can be measured
 * rather than argued about. The two have to agree on every counter, an address
 * being twenty-four rounds of hash away from where they differ.
 */
inline void profanity_create2_plain(__constant const uint * const pTemplate, const ulong counter, uint * const address) {
	ethhash h = { { 0 } };

	for (int i = 0; i < PROFANITY_CREATE2_WORDS; ++i) {
		h.d[i] = pTemplate[i];
	}

	for (int i = 0; i < 8; ++i) {
		h.b[PROFANITY_CREATE2_COUNTER + i] = (uchar)(counter >> ((7 - i) * 8));
	}

	sha3_keccakf(&h);

	address[0] = h.d[3];
	address[1] = h.d[4];
	address[2] = h.d[5];
	address[3] = h.d[6];
	address[4] = h.d[7];
}

/* The same message and the same permutation again, with the union byte stores
 * gone but the state still written through loops.
 *
 * Half of what profanity_create2 does, so that the two halves can be told
 * apart. It measures within noise of the plain version above — writing lanes
 * buys nothing while the index is still one the compiler has to work out.
 *
 * Left in as the standing argument against tidying profanity_create2's
 * twenty-five assignments back into the loop they obviously want to be.
 */
inline void profanity_create2_lanes(__constant const uint * const pTemplate, const ulong counter, uint * const address) {
	ethhash h;

	for (int i = 0; i < 25; ++i) {
		h.q[i] = 0;
	}

	for (int i = 0; i < PROFANITY_CREATE2_WORDS / 2; ++i) {
		h.q[i] = (ulong)pTemplate[i * 2] | ((ulong)pTemplate[i * 2 + 1] << 32);
	}

	const ulong swapped = bswap64(counter);
	// The counter spans the top 64 - SHIFT bits of one lane and the bottom SHIFT
	// bits of the next, so one mask is the complement of the other.
	const ulong keepLow = ((ulong)1 << PROFANITY_CREATE2_COUNTER_SHIFT) - 1;
	const ulong keepHigh = ~keepLow;

	h.q[PROFANITY_CREATE2_COUNTER_LANE] = (h.q[PROFANITY_CREATE2_COUNTER_LANE] & keepLow) | (swapped << PROFANITY_CREATE2_COUNTER_SHIFT);
	h.q[PROFANITY_CREATE2_COUNTER_LANE + 1] = (h.q[PROFANITY_CREATE2_COUNTER_LANE + 1] & keepHigh) | (swapped >> (64 - PROFANITY_CREATE2_COUNTER_SHIFT));

	sha3_keccakf(&h);

	address[0] = h.d[3];
	address[1] = h.d[4];
	address[2] = h.d[5];
	address[3] = h.d[6];
	address[4] = h.d[7];
}

/* The plain version again with both of its loops written out, and nothing else
 * changed: still a zeroed state, still words through h.d and the counter through
 * h.b, just at indices the compiler no longer has to work out.
 *
 * The other half, and the answer to whether the readable shape could have been
 * kept: no. This measures within noise of plain too — slightly under it — so
 * constant indices buy nothing while the stores are still 32 and 8 bits wide.
 *
 * Only the two together move anything, which is worth knowing before anywhere
 * else in this file is rewritten into shifts and masks on the strength of it.
 */
inline void profanity_create2_unrolled_bytes(__constant const uint * const pTemplate, const ulong counter, uint * const address) {
	ethhash h = { { 0 } };

	h.d[0] = pTemplate[0]; h.d[1] = pTemplate[1]; h.d[2] = pTemplate[2];
	h.d[3] = pTemplate[3]; h.d[4] = pTemplate[4]; h.d[5] = pTemplate[5];
	h.d[6] = pTemplate[6]; h.d[7] = pTemplate[7]; h.d[8] = pTemplate[8];
	h.d[9] = pTemplate[9]; h.d[10] = pTemplate[10]; h.d[11] = pTemplate[11];
	h.d[12] = pTemplate[12]; h.d[13] = pTemplate[13]; h.d[14] = pTemplate[14];
	h.d[15] = pTemplate[15]; h.d[16] = pTemplate[16]; h.d[17] = pTemplate[17];
	h.d[18] = pTemplate[18]; h.d[19] = pTemplate[19]; h.d[20] = pTemplate[20];
	h.d[21] = pTemplate[21];

	h.b[PROFANITY_CREATE2_COUNTER + 0] = (uchar)(counter >> 56);
	h.b[PROFANITY_CREATE2_COUNTER + 1] = (uchar)(counter >> 48);
	h.b[PROFANITY_CREATE2_COUNTER + 2] = (uchar)(counter >> 40);
	h.b[PROFANITY_CREATE2_COUNTER + 3] = (uchar)(counter >> 32);
	h.b[PROFANITY_CREATE2_COUNTER + 4] = (uchar)(counter >> 24);
	h.b[PROFANITY_CREATE2_COUNTER + 5] = (uchar)(counter >> 16);
	h.b[PROFANITY_CREATE2_COUNTER + 6] = (uchar)(counter >> 8);
	h.b[PROFANITY_CREATE2_COUNTER + 7] = (uchar)(counter);

	sha3_keccakf(&h);

	address[0] = h.d[3];
	address[1] = h.d[4];
	address[2] = h.d[5];
	address[3] = h.d[6];
	address[4] = h.d[7];
}

// One address per work item from each, over consecutive counters, for the host
// to compare. Two buffers rather than two launches so that nothing about how a
// launch is set up can come between them.
__kernel void k_create2_both(
		__constant const uint * const pTemplate,
		const ulong counterBase,
		__global uint * const outPlain,
		__global uint * const outShipped) {
	const size_t id = get_global_id(0);
	const ulong counter = counterBase + id;

	uint plain[5];
	uint shipped[5];

	profanity_create2_plain(pTemplate, counter, plain);
	profanity_create2(pTemplate, counter, shipped);

	for (int i = 0; i < 5; ++i) {
		outPlain[id * 5 + i] = plain[i];
		outShipped[id * 5 + i] = shipped[i];
	}
}

/* Each work item hashes `iterations` consecutive counters and folds every
 * address it gets into an accumulator, so that nothing can be hoisted out of
 * the loop and the write at the end keeps all of it alive. */
#define BENCH_CREATE2_KERNEL(NAME, CALL) \
__kernel void bench_create2_##NAME( \
		__constant const uint * const pFixed, \
		const ulong counterBase, \
		const uint iterations, \
		__global uint * const out) { \
	const size_t id = get_global_id(0); \
	uint address[5]; \
	uint acc = 0; \
	for (uint i = 0; i < iterations; ++i) { \
		CALL(pFixed, counterBase + id * (ulong)iterations + i + acc, address); \
		acc += address[0] ^ address[4]; \
	} \
	out[id] = acc; \
}

BENCH_CREATE2_KERNEL(plain, profanity_create2_plain)
BENCH_CREATE2_KERNEL(lanes, profanity_create2_lanes)
BENCH_CREATE2_KERNEL(unrolled_bytes, profanity_create2_unrolled_bytes)
BENCH_CREATE2_KERNEL(shipped, profanity_create2)

/* ------------------------------------------------------------------------ */
/* The same question asked of the EOA and contract addresses                */
/* ------------------------------------------------------------------------ */
/* profanity_address builds two hash states, and they are not alike.
 *
 * The first is already written at indices the compiler can see — h.d[0] through
 * h.d[16], no loop — so the only thing it does that profanity_create2 stopped
 * doing is write 32-bit halves into a union that sha3_keccakf reads as 64-bit
 * lanes. Whether that alone costs anything is worth knowing, because this one
 * runs for every address a search looks at, up to six per point addition.
 *
 * The second, under --contract, is the pattern that cost CREATE2 a third of its
 * throughput, at byte granularity: a loop writing c.b[i + 2] at an index the
 * compiler has to prove something about, into a union read as ulongs, out of a
 * uchar pointer aliased onto the uint[5] the first hash produced.
 *
 * The contract half of profanity_address now builds its state as whole lanes;
 * the account half does not, having measured at nothing on an RTX 4090 when it
 * did. The variant below has both, so that half of the decision stays a
 * measurement rather than becoming folklore.
 */
/* What profanity_address was before the contract half of it was rewritten:
 * both hash states built by copying into a zeroed one, the second through a
 * loop of byte stores out of a uchar pointer aliased onto the uint[5] the first
 * produced. The reference the shipping version is checked against.
 */
inline void profanity_address_plain(const mp_number * const x, const mp_number * const y, const uchar bContract, uint * const address) {
	ethhash h = { { 0 } };

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

	address[0] = h.d[3];
	address[1] = h.d[4];
	address[2] = h.d[5];
	address[3] = h.d[6];
	address[4] = h.d[7];

	if (bContract) {
		__private const uchar * const sender = (__private const uchar *)address;
		ethhash c = { { 0 } };

		// set up keccak(0xd6, 0x94, address, 0x80)
		c.b[0] = 0xd6;
		c.b[1] = 0x94;
		for (int i = 0; i < 20; ++i) {
			c.b[i + 2] = sender[i];
		}
		c.b[22] = 0x80;

		c.b[23] ^= 0x01; // length 23
		sha3_keccakf(&c);

		address[0] = c.d[3];
		address[1] = c.d[4];
		address[2] = c.d[5];
		address[3] = c.d[6];
		address[4] = c.d[7];
	}
}

inline void profanity_address_lanes(const mp_number * const x, const mp_number * const y, const uchar bContract, uint * const address) {
	ethhash h;

	// The coordinates go in big endian, a lane taking two words of each, high
	// word first — the same order the byte-swapped halves went in above.
	h.q[0] = (ulong)bswap32(x->d[7]) | ((ulong)bswap32(x->d[6]) << 32);
	h.q[1] = (ulong)bswap32(x->d[5]) | ((ulong)bswap32(x->d[4]) << 32);
	h.q[2] = (ulong)bswap32(x->d[3]) | ((ulong)bswap32(x->d[2]) << 32);
	h.q[3] = (ulong)bswap32(x->d[1]) | ((ulong)bswap32(x->d[0]) << 32);
	h.q[4] = (ulong)bswap32(y->d[7]) | ((ulong)bswap32(y->d[6]) << 32);
	h.q[5] = (ulong)bswap32(y->d[5]) | ((ulong)bswap32(y->d[4]) << 32);
	h.q[6] = (ulong)bswap32(y->d[3]) | ((ulong)bswap32(y->d[2]) << 32);
	h.q[7] = (ulong)bswap32(y->d[1]) | ((ulong)bswap32(y->d[0]) << 32);
	h.q[8] = 0x01; // sixty-four bytes of message, then the byte that ends it
	h.q[9] = 0; h.q[10] = 0; h.q[11] = 0; h.q[12] = 0;
	h.q[13] = 0; h.q[14] = 0; h.q[15] = 0; h.q[16] = 0;
	h.q[17] = 0; h.q[18] = 0; h.q[19] = 0; h.q[20] = 0;
	h.q[21] = 0; h.q[22] = 0; h.q[23] = 0; h.q[24] = 0;

	sha3_keccakf(&h);

	address[0] = h.d[3];
	address[1] = h.d[4];
	address[2] = h.d[5];
	address[3] = h.d[6];
	address[4] = h.d[7];

	if (bContract) {
		ethhash c;

		/* keccak(0xd6, 0x94, address, 0x80) — twenty-three bytes and then the
		 * byte that ends the message, which is three lanes exactly. The address
		 * arrives as five uints holding it little endian, so each lane takes
		 * what falls in it: a whole word, or the half of one either side. */
		c.q[0] = 0x94d6 | ((ulong)address[0] << 16) | ((ulong)(address[1] & 0xFFFF) << 48);
		c.q[1] = (ulong)(address[1] >> 16) | ((ulong)address[2] << 16) | ((ulong)(address[3] & 0xFFFF) << 48);
		c.q[2] = (ulong)(address[3] >> 16) | ((ulong)address[4] << 16) | 0x0180000000000000UL;
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

__kernel void k_address_all(
		__global const mp_number * const X,
		__global const mp_number * const Y,
		const uchar bContract,
		__global uint * const outPlain,
		__global uint * const outShipped,
		__global uint * const outLanes) {
	const size_t id = get_global_id(0);
	const mp_number x = X[id];
	const mp_number y = Y[id];

	uint plain[5];
	uint shipped[5];
	uint lanes[5];

	profanity_address_plain(&x, &y, bContract, plain);
	profanity_address(&x, &y, bContract, shipped);
	profanity_address_lanes(&x, &y, bContract, lanes);

	for (int i = 0; i < 5; ++i) {
		outPlain[id * 5 + i] = plain[i];
		outShipped[id * 5 + i] = shipped[i];
		outLanes[id * 5 + i] = lanes[i];
	}
}

#define BENCH_ADDRESS_KERNEL(NAME, CALL) \
__kernel void bench_address_##NAME( \
		__global const mp_number * const X, \
		__global const mp_number * const Y, \
		const uchar bContract, \
		const uint iterations, \
		__global uint * const out) { \
	const size_t id = get_global_id(0); \
	mp_number x = X[id]; \
	const mp_number y = Y[id]; \
	uint address[5]; \
	uint acc = 0; \
	for (uint i = 0; i < iterations; ++i) { \
		CALL(&x, &y, bContract, address); \
		acc += address[0] ^ address[4]; \
		x.d[0] ^= acc; \
	} \
	out[id] = acc; \
}

BENCH_ADDRESS_KERNEL(plain, profanity_address_plain)
BENCH_ADDRESS_KERNEL(shipped, profanity_address)
BENCH_ADDRESS_KERNEL(lanes, profanity_address_lanes)

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

__kernel void k_mod_mul_oneshot_portable(__global const mp_number * const X, __global const mp_number * const Y, __global mp_number * const R) {
	const size_t id = get_global_id(0);
	mp_number x = X[id];
	mp_number y = Y[id];
	mp_number r;
	mp_mod_mul_oneshot_portable(&r, &x, &y);
	R[id] = r;
}

__kernel void k_mod_sqr_portable(__global const mp_number * const X, __global mp_number * const R) {
	const size_t id = get_global_id(0);
	mp_number x = X[id];
	mp_number r;
	mp_mod_sqr_portable(&r, &x);
	R[id] = r;
}

#ifdef PROFANITY_PTX_MP
__kernel void k_mod_mul_oneshot_ptx(__global const mp_number * const X, __global const mp_number * const Y, __global mp_number * const R) {
	const size_t id = get_global_id(0);
	mp_number x = X[id];
	mp_number y = Y[id];
	mp_number r;
	mp_mod_mul_oneshot_ptx(&r, &x, &y);
	R[id] = r;
}

__kernel void k_mod_sqr_ptx(__global const mp_number * const X, __global mp_number * const R) {
	const size_t id = get_global_id(0);
	mp_number x = X[id];
	mp_number r;
	mp_mod_sqr_ptx(&r, &x);
	R[id] = r;
}
#endif

__kernel void k_mod_mul_oneshot_portable(__global const mp_number * const X, __global const mp_number * const Y, __global mp_number * const R) {
	const size_t id = get_global_id(0);
	mp_number x = X[id];
	mp_number y = Y[id];
	mp_number r;
	mp_mod_mul_oneshot_portable(&r, &x, &y);
	R[id] = r;
}

__kernel void k_mod_sqr_portable(__global const mp_number * const X, __global mp_number * const R) {
	const size_t id = get_global_id(0);
	mp_number x = X[id];
	mp_number r;
	mp_mod_sqr_portable(&r, &x);
	R[id] = r;
}

#ifdef PROFANITY_PTX_MP
__kernel void k_mod_mul_oneshot_ptx(__global const mp_number * const X, __global const mp_number * const Y, __global mp_number * const R) {
	const size_t id = get_global_id(0);
	mp_number x = X[id];
	mp_number y = Y[id];
	mp_number r;
	mp_mod_mul_oneshot_ptx(&r, &x, &y);
	R[id] = r;
}

__kernel void k_mod_sqr_ptx(__global const mp_number * const X, __global mp_number * const R) {
	const size_t id = get_global_id(0);
	mp_number x = X[id];
	mp_number r;
	mp_mod_sqr_ptx(&r, &x);
	R[id] = r;
}
#endif

// ---- sparse keccak EOA/contract equivalence kernels ----
__kernel void k_keccak_eoa_ab(
		__global const ulong * const pIn,
		__global ulong * const pStock,
		__global ulong * const pSpecial) {
	const size_t id = get_global_id(0);
	ethhash a, b;
	for (int i = 0; i < 25; ++i) {
		const ulong v = pIn[id * 25 + i];
		a.q[i] = v;
		b.q[i] = v;
	}
	sha3_keccakf(&a);
	sha3_keccakf_eoa(&b);
	for (int i = 0; i < 25; ++i) {
		pStock[id * 25 + i] = a.q[i];
		pSpecial[id * 25 + i] = b.q[i];
	}
}

__kernel void k_keccak_contract_ab(
		__global const ulong * const pIn,
		__global ulong * const pStock,
		__global ulong * const pSpecial) {
	const size_t id = get_global_id(0);
	ethhash a, b;
	for (int i = 0; i < 25; ++i) {
		const ulong v = pIn[id * 25 + i];
		a.q[i] = v;
		b.q[i] = v;
	}
	sha3_keccakf(&a);
	sha3_keccakf_contract(&b);
	for (int i = 0; i < 25; ++i) {
		pStock[id * 25 + i] = a.q[i];
		pSpecial[id * 25 + i] = b.q[i];
	}
}

// ---- CREATE2 keccak equivalence kernel ----
__kernel void k_keccak_create2_ab(
		__global const ulong * const pIn,
		__global ulong * const pStock,
		__global ulong * const pSpecial) {
	const size_t id = get_global_id(0);
	ethhash a, b;
	for (int i = 0; i < 25; ++i) {
		const ulong v = pIn[id * 25 + i];
		a.q[i] = v;
		b.q[i] = v;
	}
	sha3_keccakf(&a);
	sha3_keccakf_create2(&b);
	for (int i = 0; i < 25; ++i) {
		pStock[id * 25 + i] = a.q[i];
		pSpecial[id * 25 + i] = b.q[i];
	}
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
