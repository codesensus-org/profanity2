#ifndef HPP_CREATE2
#define HPP_CREATE2

/* Where a contract deployed with CREATE2 lands, and how the kernels are told to
 * look for it. A single file because the layout below is an agreement between
 * three places — this header, the kernels it is compiled into, and the salt the
 * host puts back together to print — and an agreement kept in three places is
 * one that will eventually be kept in two.
 */

#include <algorithm>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/* The address is the last twenty bytes of
 *
 *     keccak256(0xff ++ factory ++ salt ++ keccak256(init_code))
 *
 * which is 1 + 20 + 32 + 32 bytes of preimage, and the byte that ends a keccak
 * message goes after it. Twenty-two words covers all of that; the kernels take
 * it a word at a time, only the salt's last eight bytes changing between one
 * candidate and the next. */
#define PROFANITY_CREATE2_PREIMAGE 85
#define PROFANITY_CREATE2_WORDS 22

/* Where in that preimage those eight bytes sit: past the 0xff, the twenty bytes
 * of the deploying contract, and the salt's first twenty-four. */
#define PROFANITY_CREATE2_COUNTER 45

/* What a CREATE2 search holds fixed. The salt is what it varies, but not all of
 * it: a factory that guards against front-running insists the first twenty
 * bytes be the address deploying through it, so the caller goes there and the
 * search has the rest. Nothing insists on it when the caller is left at zero,
 * which is also what such a factory takes to mean "anyone may deploy this". */
typedef struct {
	cl_uchar factory[20];
	cl_uchar caller[20];
	cl_uchar initCodeHash[32];
} create2;

/* The preimage as the words a kernel copies into its hash state. The 0x01 that
 * ends the message is here; the 0x80 that ends the block is the one thing
 * sha3_keccakf puts there itself. Whatever the salt holds at
 * PROFANITY_CREATE2_COUNTER is written over per work item.
 *
 * The words are packed the way a little-endian device reads its hash state back
 * as bytes, which is the same assumption the scoring makes of an address. */
inline void buildCreate2Template(cl_uint * const words, const create2 & fixed, const cl_uchar * const salt) {
	cl_uchar preimage[PROFANITY_CREATE2_WORDS * 4] = { 0 };

	preimage[0] = 0xff;
	std::copy(fixed.factory, fixed.factory + 20, preimage + 1);
	std::copy(salt, salt + 32, preimage + 21);
	std::copy(fixed.initCodeHash, fixed.initCodeHash + 32, preimage + 53);
	preimage[PROFANITY_CREATE2_PREIMAGE] = 0x01;

	for (size_t i = 0; i < PROFANITY_CREATE2_WORDS; ++i) {
		words[i] = (cl_uint) preimage[i * 4]
			| ((cl_uint) preimage[i * 4 + 1] << 8)
			| ((cl_uint) preimage[i * 4 + 2] << 16)
			| ((cl_uint) preimage[i * 4 + 3] << 24);
	}
}

/* The salt a counter stands for: the one a device is searching from, with the
 * counter written over its last eight bytes exactly as the kernel writes it.
 * Big endian, so that a salt read off a printed line counts upwards the way it
 * is written — which order it is in changes nothing about the search, one salt
 * being as good as another, but the two sides have to agree on it. */
inline void applyCreate2Counter(cl_uchar * const salt, const cl_ulong counter) {
	for (int i = 0; i < 8; ++i) {
		salt[24 + i] = (cl_uchar)(counter >> ((7 - i) * 8));
	}
}

#endif /* HPP_CREATE2 */
