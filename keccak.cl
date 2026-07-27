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
