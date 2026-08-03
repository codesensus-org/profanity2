/* This Keccak implementation is an amalgamation of:
 * Tiny SHA3 implementation by Markku-Juhani O. Saarinen:
 *   https://github.com/mjosaarinen/tiny_sha3
 * Keccak implementation found in xptMiner-gpu @ Github:
 *   https://github.com/llamasoft/xptMiner-gpu/blob/master/opencl/keccak.cl
 * Optimization for Ethereum addresses by truncating the last round from 0age:
 *   https://github.com/0age/create2crunch/blob/master/src/kernels/keccak256.cl
 */

typedef union {
	uchar b[200];
	ulong q[25];
	uint d[50];
} ethhash;

#define TH_ELT(t, c0, c1, c2, c3, c4, d0, d1, d2, d3, d4) \
{ \
    t = rotate((ulong)(d0 ^ d1 ^ d2 ^ d3 ^ d4), (ulong)1) ^ (c0 ^ c1 ^ c2 ^ c3 ^ c4); \
}

#define THETA(s00, s01, s02, s03, s04, \
              s10, s11, s12, s13, s14, \
              s20, s21, s22, s23, s24, \
              s30, s31, s32, s33, s34, \
              s40, s41, s42, s43, s44) \
{ \
    TH_ELT(t0, s40, s41, s42, s43, s44, s10, s11, s12, s13, s14); \
    TH_ELT(t1, s00, s01, s02, s03, s04, s20, s21, s22, s23, s24); \
    TH_ELT(t2, s10, s11, s12, s13, s14, s30, s31, s32, s33, s34); \
    TH_ELT(t3, s20, s21, s22, s23, s24, s40, s41, s42, s43, s44); \
    TH_ELT(t4, s30, s31, s32, s33, s34, s00, s01, s02, s03, s04); \
    s00 ^= t0; s01 ^= t0; s02 ^= t0; s03 ^= t0; s04 ^= t0; \
    s10 ^= t1; s11 ^= t1; s12 ^= t1; s13 ^= t1; s14 ^= t1; \
    s20 ^= t2; s21 ^= t2; s22 ^= t2; s23 ^= t2; s24 ^= t2; \
    s30 ^= t3; s31 ^= t3; s32 ^= t3; s33 ^= t3; s34 ^= t3; \
    s40 ^= t4; s41 ^= t4; s42 ^= t4; s43 ^= t4; s44 ^= t4; \
}

#define RHOPI(s00, s01, s02, s03, s04, \
              s10, s11, s12, s13, s14, \
              s20, s21, s22, s23, s24, \
              s30, s31, s32, s33, s34, \
              s40, s41, s42, s43, s44) \
{ \
	t0  = rotate(s10, (ulong) 1);  \
	s10 = rotate(s11, (ulong)44); \
	s11 = rotate(s41, (ulong)20); \
	s41 = rotate(s24, (ulong)61); \
	s24 = rotate(s42, (ulong)39); \
	s42 = rotate(s04, (ulong)18); \
	s04 = rotate(s20, (ulong)62); \
	s20 = rotate(s22, (ulong)43); \
	s22 = rotate(s32, (ulong)25); \
	s32 = rotate(s43, (ulong) 8); \
	s43 = rotate(s34, (ulong)56); \
	s34 = rotate(s03, (ulong)41); \
	s03 = rotate(s40, (ulong)27); \
	s40 = rotate(s44, (ulong)14); \
	s44 = rotate(s14, (ulong) 2); \
	s14 = rotate(s31, (ulong)55); \
	s31 = rotate(s13, (ulong)45); \
	s13 = rotate(s01, (ulong)36); \
	s01 = rotate(s30, (ulong)28); \
	s30 = rotate(s33, (ulong)21); \
	s33 = rotate(s23, (ulong)15); \
	s23 = rotate(s12, (ulong)10); \
	s12 = rotate(s21, (ulong) 6); \
	s21 = rotate(s02, (ulong) 3); \
	s02 = t0; \
}

#define KHI(s00, s01, s02, s03, s04, \
            s10, s11, s12, s13, s14, \
            s20, s21, s22, s23, s24, \
            s30, s31, s32, s33, s34, \
            s40, s41, s42, s43, s44) \
{ \
    t0 = s00 ^ (~s10 &  s20); \
    t1 = s10 ^ (~s20 &  s30); \
    t2 = s20 ^ (~s30 &  s40); \
    t3 = s30 ^ (~s40 &  s00); \
    t4 = s40 ^ (~s00 &  s10); \
    s00 = t0; s10 = t1; s20 = t2; s30 = t3; s40 = t4; \
    \
    t0 = s01 ^ (~s11 &  s21); \
    t1 = s11 ^ (~s21 &  s31); \
    t2 = s21 ^ (~s31 &  s41); \
    t3 = s31 ^ (~s41 &  s01); \
    t4 = s41 ^ (~s01 &  s11); \
    s01 = t0; s11 = t1; s21 = t2; s31 = t3; s41 = t4; \
    \
    t0 = s02 ^ (~s12 &  s22); \
    t1 = s12 ^ (~s22 &  s32); \
    t2 = s22 ^ (~s32 &  s42); \
    t3 = s32 ^ (~s42 &  s02); \
    t4 = s42 ^ (~s02 &  s12); \
    s02 = t0; s12 = t1; s22 = t2; s32 = t3; s42 = t4; \
    \
    t0 = s03 ^ (~s13 &  s23); \
    t1 = s13 ^ (~s23 &  s33); \
    t2 = s23 ^ (~s33 &  s43); \
    t3 = s33 ^ (~s43 &  s03); \
    t4 = s43 ^ (~s03 &  s13); \
    s03 = t0; s13 = t1; s23 = t2; s33 = t3; s43 = t4; \
    \
    t0 = s04 ^ (~s14 &  s24); \
    t1 = s14 ^ (~s24 &  s34); \
    t2 = s24 ^ (~s34 &  s44); \
    t3 = s34 ^ (~s44 &  s04); \
    t4 = s44 ^ (~s04 &  s14); \
    s04 = t0; s14 = t1; s24 = t2; s34 = t3; s44 = t4; \
}

#define IOTA(s00, r) { s00 ^= r; }

__constant ulong keccakf_rndc[24] = {
	0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
	0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
	0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
	0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
	0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
	0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
	0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
	0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

/* Keccak-f[1600] with the last round truncated to just the Ethereum address.
 *
 * WARNING: In the result, only h->d[3] through h->d[7] (the last 20 bytes)
 * hold correct values. Every other word is left in an intermediate state.
 */
void sha3_keccakf(ethhash * const h)
{
	ulong * const st = (ulong * const)&h->q[0];
	h->d[33] ^= 0x80000000;
	ulong t0, t1, t2, t3, t4;

	for (int i = 0; i < 23; ++i) {
		THETA(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
		RHOPI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
		KHI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
		IOTA(st[0], keccakf_rndc[i]);
	}

	// Round 24, partial
	{
		TH_ELT(t0, st[4], st[9], st[14], st[19], st[24], st[1], st[6], st[11], st[16], st[21]);
		TH_ELT(t1, st[0], st[5], st[10], st[15], st[20], st[2], st[7], st[12], st[17], st[22]);
		TH_ELT(t2, st[1], st[6], st[11], st[16], st[21], st[3], st[8], st[13], st[18], st[23]);
		TH_ELT(t3, st[2], st[7], st[12], st[17], st[22], st[4], st[9], st[14], st[19], st[24]);
		TH_ELT(t4, st[3], st[8], st[13], st[18], st[23], st[0], st[5], st[10], st[15], st[20]);

		// Theta applied only to the five lanes Rho/Pi feeds into the Khi below
		ulong s00 = st[0]  ^ t0;
		ulong s11 = st[6]  ^ t1;
		ulong s22 = st[12] ^ t2;
		ulong s33 = st[18] ^ t3;
		ulong s44 = st[24] ^ t4;

		// Rho/Pi for those same lanes (s00 is never reassigned)
		ulong s10 = rotate(s11, (ulong)44);
		ulong s20 = rotate(s22, (ulong)43);
		ulong s30 = rotate(s33, (ulong)21);
		ulong s40 = rotate(s44, (ulong)14);

		// Khi only for the three lanes that make up the address
		st[1] = s10 ^ (~s20 & s30);
		st[2] = s20 ^ (~s30 & s40);
		st[3] = s30 ^ (~s40 & s00);

		// Iota is dropped since it only touches s00
	}
}

// ==== sparse-first-round specialisations (EOA + contract), bit-exact with sha3_keccakf ====

/*
 * sha3_keccakf_eoa - keccak-f[1600] specialised for the EOA address preimage.
 *
 * Bit-exact with sha3_keccakf(); ONLY the first round's THETA is specialised
 * for the sparse absorb layout produced by profanity_address() when hashing a
 * 64-byte public key. Rounds 1..23 and the truncated final round are the stock
 * generic macros, unchanged.
 *
 * Absorbed state at entry (profanity.cl profanity_address, `ethhash h = {{0}}`):
 *   d[0..15]  set   -> lanes st[0..7]  hold the 64-byte pubkey (variable)
 *   d[16] ^= 0x01   -> st[8] = 0x0000000000000001  (keccak pad start, byte 64)
 *   nothing else set-> st[9..24] = 0
 * This function's own `h->d[33] ^= 0x80000000` then sets st[16] = 0x80..0
 * (keccak pad end, byte 135). Every other rate/capacity lane stays zero.
 *
 * Provably-zero lanes RELIED UPON for round-0 THETA (exactly these 15):
 *   st[9] st[10] st[11] st[12] st[13] st[14] st[15]
 *   st[17] st[18] st[19] st[20] st[21] st[22] st[23] st[24]
 * No assumption is made about the *value* of any nonzero lane: st[8] and st[16]
 * are read, not folded, so correctness depends only on the 15 zeros above.
 * Because the dropped THETA operands are zero, the column parities and D values
 * are identical to the generic THETA, and `lane = D` equals `0 ^ D`; hence the
 * produced state is byte-identical to sha3_keccakf().
 */
void sha3_keccakf_eoa(ethhash * const h)
{
	ulong * const st = (ulong * const)&h->q[0];
	h->d[33] ^= 0x80000000;
	ulong t0, t1, t2, t3, t4;

	/* ---- Round 0: THETA specialised for the sparse EOA absorb ---- */
	{
		/* Column parities; zero lanes (see above) dropped from each XOR. */
		const ulong c0 = st[0] ^ st[5];
		const ulong c1 = st[1] ^ st[6] ^ st[16];
		const ulong c2 = st[2] ^ st[7];
		const ulong c3 = st[3] ^ st[8];
		const ulong c4 = st[4];

		/* D[x] = C[x-1] ^ rot(C[x+1], 1) -- same as generic THETA. */
		const ulong d0 = c4 ^ rotate(c1, (ulong)1);
		const ulong d1 = c0 ^ rotate(c2, (ulong)1);
		const ulong d2 = c1 ^ rotate(c3, (ulong)1);
		const ulong d3 = c2 ^ rotate(c4, (ulong)1);
		const ulong d4 = c3 ^ rotate(c0, (ulong)1);

		/* Apply D[x] to column x. Lanes proven zero take `= D` (== 0 ^ D). */
		st[0] ^= d0; st[5] ^= d0; st[10] = d0; st[15] = d0; st[20] = d0;
		st[1] ^= d1; st[6] ^= d1; st[11] = d1; st[16] ^= d1; st[21] = d1;
		st[2] ^= d2; st[7] ^= d2; st[12] = d2; st[17] = d2; st[22] = d2;
		st[3] ^= d3; st[8] ^= d3; st[13] = d3; st[18] = d3; st[23] = d3;
		st[4] ^= d4; st[9] = d4; st[14] = d4; st[19] = d4; st[24] = d4;
	}
	RHOPI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
	KHI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
	IOTA(st[0], keccakf_rndc[0]);

	/* ---- Rounds 1..23 (generic, unchanged) ---- */
	for (int i = 1; i < 23; ++i) {
		THETA(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
		RHOPI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
		KHI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
		IOTA(st[0], keccakf_rndc[i]);
	}

	/* ---- Truncated final round (round 23), unchanged ---- */
	{
		TH_ELT(t0, st[4], st[9], st[14], st[19], st[24], st[1], st[6], st[11], st[16], st[21]);
		TH_ELT(t1, st[0], st[5], st[10], st[15], st[20], st[2], st[7], st[12], st[17], st[22]);
		TH_ELT(t2, st[1], st[6], st[11], st[16], st[21], st[3], st[8], st[13], st[18], st[23]);
		TH_ELT(t3, st[2], st[7], st[12], st[17], st[22], st[4], st[9], st[14], st[19], st[24]);
		TH_ELT(t4, st[3], st[8], st[13], st[18], st[23], st[0], st[5], st[10], st[15], st[20]);

		ulong s00 = st[0]  ^ t0;
		ulong s11 = st[6]  ^ t1;
		ulong s22 = st[12] ^ t2;
		ulong s33 = st[18] ^ t3;
		ulong s44 = st[24] ^ t4;

		ulong s10 = rotate(s11, (ulong)44);
		ulong s20 = rotate(s22, (ulong)43);
		ulong s30 = rotate(s33, (ulong)21);
		ulong s40 = rotate(s44, (ulong)14);

		st[1] = s10 ^ (~s20 & s30);
		st[2] = s20 ^ (~s30 & s40);
		st[3] = s30 ^ (~s40 & s00);
	}
}

/*
 * sha3_keccakf_contract - keccak-f[1600] specialised for the CREATE (nonce 0)
 * contract-address preimage: keccak256(0xd6 0x94 || address[20]).
 *
 * Bit-exact with sha3_keccakf(); ONLY round-0 THETA is specialised. Used for
 * the second hash in profanity_address() when bContract is set.
 *
 * Absorbed state at entry (profanity.cl, `ethhash c;` fully assigned):
 *   c.q[0], c.q[1], c.q[2] set -> st[0], st[1], st[2] (0xd6 0x94, 20-byte addr,
 *                                 nonce byte 0x80, keccak pad start 0x01)
 *   c.q[3..24] = 0             -> st[3..24] = 0
 * This function's `h->d[33] ^= 0x80000000` sets st[16] = 0x80..0 (pad end).
 *
 * Provably-zero lanes RELIED UPON for round-0 THETA (exactly these 21):
 *   st[3] st[4] st[5] st[6] st[7] st[8] st[9] st[10] st[11] st[12] st[13]
 *   st[14] st[15] st[17] st[18] st[19] st[20] st[21] st[22] st[23] st[24]
 * Two whole column parities vanish (C[3] = C[4] = 0). No assumption is made
 * about the values of st[0], st[1], st[2] or st[16].
 */
void sha3_keccakf_contract(ethhash * const h)
{
	ulong * const st = (ulong * const)&h->q[0];
	h->d[33] ^= 0x80000000;
	ulong t0, t1, t2, t3, t4;

	/* ---- Round 0: THETA specialised for the sparse contract absorb ---- */
	{
		const ulong c0 = st[0];
		const ulong c1 = st[1] ^ st[16];
		const ulong c2 = st[2];
		/* c3 == 0, c4 == 0 (whole columns 3 and 4 are zero). */

		const ulong d0 = rotate(c1, (ulong)1);          /* c4 ^ rot(c1,1), c4=0 */
		const ulong d1 = c0 ^ rotate(c2, (ulong)1);
		const ulong d2 = c1;                            /* c1 ^ rot(c3,1), c3=0 */
		const ulong d3 = c2;                            /* c2 ^ rot(c4,1), c4=0 */
		const ulong d4 = rotate(c0, (ulong)1);          /* c3 ^ rot(c0,1), c3=0 */

		st[0] ^= d0; st[5] = d0; st[10] = d0; st[15] = d0; st[20] = d0;
		st[1] ^= d1; st[6] = d1; st[11] = d1; st[16] ^= d1; st[21] = d1;
		st[2] ^= d2; st[7] = d2; st[12] = d2; st[17] = d2; st[22] = d2;
		st[3] = d3; st[8] = d3; st[13] = d3; st[18] = d3; st[23] = d3;
		st[4] = d4; st[9] = d4; st[14] = d4; st[19] = d4; st[24] = d4;
	}
	RHOPI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
	KHI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
	IOTA(st[0], keccakf_rndc[0]);

	for (int i = 1; i < 23; ++i) {
		THETA(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
		RHOPI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
		KHI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
		IOTA(st[0], keccakf_rndc[i]);
	}

	{
		TH_ELT(t0, st[4], st[9], st[14], st[19], st[24], st[1], st[6], st[11], st[16], st[21]);
		TH_ELT(t1, st[0], st[5], st[10], st[15], st[20], st[2], st[7], st[12], st[17], st[22]);
		TH_ELT(t2, st[1], st[6], st[11], st[16], st[21], st[3], st[8], st[13], st[18], st[23]);
		TH_ELT(t3, st[2], st[7], st[12], st[17], st[22], st[4], st[9], st[14], st[19], st[24]);
		TH_ELT(t4, st[3], st[8], st[13], st[18], st[23], st[0], st[5], st[10], st[15], st[20]);

		ulong s00 = st[0]  ^ t0;
		ulong s11 = st[6]  ^ t1;
		ulong s22 = st[12] ^ t2;
		ulong s33 = st[18] ^ t3;
		ulong s44 = st[24] ^ t4;

		ulong s10 = rotate(s11, (ulong)44);
		ulong s20 = rotate(s22, (ulong)43);
		ulong s30 = rotate(s33, (ulong)21);
		ulong s40 = rotate(s44, (ulong)14);

		st[1] = s10 ^ (~s20 & s30);
		st[2] = s20 ^ (~s30 & s40);
		st[3] = s30 ^ (~s40 & s00);
	}
}

// ==== CREATE2 85-byte preimage specialisation, bit-exact with sha3_keccakf ====

/*
 * sha3_keccakf_create2 - keccak-f[1600] specialised for the CREATE2 preimage.
 *
 * Bit-exact with sha3_keccakf(); ONLY the first round's THETA is specialised
 * for the sparse absorb layout produced by profanity_create2() when hashing the
 * 85-byte CREATE2 preimage 0xff || factory[20] || salt[32] || initCodeHash[32].
 * Rounds 1..22 and the truncated final round (round 23) are the stock generic
 * macros, unchanged, and the round count is unchanged (24).
 *
 * Absorbed state at entry (profanity.cl profanity_create2):
 *   h.q[0..4]   <- pTemplate[0..9]              (profanity.cl:1135-1139)
 *   h.q[5]      <- pTemplate[10..11] + counter  (profanity.cl:1140)
 *   h.q[6]      <- pTemplate[12..13] + counter  (profanity.cl:1141)
 *   h.q[7..10]  <- pTemplate[14..21]            (profanity.cl:1142-1145)
 *   h.q[11..24] = 0  explicitly                 (profanity.cl:1147-1149)
 * This function's own `h->d[33] ^= 0x80000000` then sets lane st[16] to
 * 0x8000000000000000 (the keccak pad end at preimage byte 135), so st[16] is
 * NOT zero and is treated here as a live, unknown value.
 *
 * Provably-zero lanes RELIED UPON for round-0 THETA (exactly these 13):
 *   st[11] st[12] st[13] st[14] st[15]
 *   st[17] st[18] st[19] st[20] st[21] st[22] st[23] st[24]
 * i.e. h.q[11..24] from profanity.cl:1147-1149 MINUS st[16], which the padding
 * bit makes nonzero. No assumption is made about the *value* of any other lane:
 * st[0..10] and st[16] are read, never folded, so correctness depends only on
 * the 13 zeros above.
 *
 * Because every dropped THETA operand is zero, the five column parities C[x]
 * and the five D[x] are identical to those the generic THETA computes, and for
 * a zero lane `lane = D[x]` is exactly `lane ^= D[x]`. The state handed to
 * RHOPI is therefore byte-identical to the generic path, and every subsequent
 * round is the stock code, so the whole permutation is bit-exact.
 *
 * No column parity vanishes entirely (all five columns retain >= 2 live lanes),
 * and after THETA no lane is zero any more, so -- unlike the contract variant --
 * there is nothing further to fold in round 0's RHOPI/CHI. Round 0's THETA goes
 * from 70 XOR + 5 rotate (generic macro as written) to 24 XOR + 5 rotate.
 */
void sha3_keccakf_create2(ethhash * const h)
{
	ulong * const st = (ulong * const)&h->q[0];
	h->d[33] ^= 0x80000000;
	ulong t0, t1, t2, t3, t4;

	/* ---- Round 0: THETA specialised for the sparse CREATE2 absorb ---- */
	{
		/* Column parities; the 13 zero lanes above are dropped from the XORs.
		 * st[16] carries the pad-end bit and IS included in column 1. */
		const ulong c0 = st[0] ^ st[5] ^ st[10];   /* ^ st[15] ^ st[20], both 0 */
		const ulong c1 = st[1] ^ st[6] ^ st[16];   /* ^ st[11] ^ st[21], both 0 */
		const ulong c2 = st[2] ^ st[7];            /* ^ st[12] ^ st[17] ^ st[22] */
		const ulong c3 = st[3] ^ st[8];            /* ^ st[13] ^ st[18] ^ st[23] */
		const ulong c4 = st[4] ^ st[9];            /* ^ st[14] ^ st[19] ^ st[24] */

		/* D[x] = C[x-1] ^ rot(C[x+1], 1) -- same as the generic THETA. */
		const ulong d0 = c4 ^ rotate(c1, (ulong)1);
		const ulong d1 = c0 ^ rotate(c2, (ulong)1);
		const ulong d2 = c1 ^ rotate(c3, (ulong)1);
		const ulong d3 = c2 ^ rotate(c4, (ulong)1);
		const ulong d4 = c3 ^ rotate(c0, (ulong)1);

		/* Apply D[x] to column x. Lanes proven zero take `= D` (== 0 ^ D). */
		st[0] ^= d0; st[5] ^= d0; st[10] ^= d0; st[15] = d0; st[20] = d0;
		st[1] ^= d1; st[6] ^= d1; st[11] = d1; st[16] ^= d1; st[21] = d1;
		st[2] ^= d2; st[7] ^= d2; st[12] = d2; st[17] = d2; st[22] = d2;
		st[3] ^= d3; st[8] ^= d3; st[13] = d3; st[18] = d3; st[23] = d3;
		st[4] ^= d4; st[9] ^= d4; st[14] = d4; st[19] = d4; st[24] = d4;
	}
	RHOPI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
	KHI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
	IOTA(st[0], keccakf_rndc[0]);

	/* ---- Rounds 1..22 (generic, unchanged) ---- */
	for (int i = 1; i < 23; ++i) {
		THETA(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
		RHOPI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
		KHI(st[0], st[5], st[10], st[15], st[20], st[1], st[6], st[11], st[16], st[21], st[2], st[7], st[12], st[17], st[22], st[3], st[8], st[13], st[18], st[23], st[4], st[9], st[14], st[19], st[24]);
		IOTA(st[0], keccakf_rndc[i]);
	}

	/* ---- Truncated final round (round 23), unchanged ---- */
	{
		TH_ELT(t0, st[4], st[9], st[14], st[19], st[24], st[1], st[6], st[11], st[16], st[21]);
		TH_ELT(t1, st[0], st[5], st[10], st[15], st[20], st[2], st[7], st[12], st[17], st[22]);
		TH_ELT(t2, st[1], st[6], st[11], st[16], st[21], st[3], st[8], st[13], st[18], st[23]);
		TH_ELT(t3, st[2], st[7], st[12], st[17], st[22], st[4], st[9], st[14], st[19], st[24]);
		TH_ELT(t4, st[3], st[8], st[13], st[18], st[23], st[0], st[5], st[10], st[15], st[20]);

		ulong s00 = st[0]  ^ t0;
		ulong s11 = st[6]  ^ t1;
		ulong s22 = st[12] ^ t2;
		ulong s33 = st[18] ^ t3;
		ulong s44 = st[24] ^ t4;

		ulong s10 = rotate(s11, (ulong)44);
		ulong s20 = rotate(s22, (ulong)43);
		ulong s30 = rotate(s33, (ulong)21);
		ulong s40 = rotate(s44, (ulong)14);

		st[1] = s10 ^ (~s20 & s30);
		st[2] = s20 ^ (~s30 & s40);
		st[3] = s30 ^ (~s40 & s00);
	}
}
