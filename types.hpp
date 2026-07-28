#ifndef HPP_TYPES
#define HPP_TYPES

/* The structs declared in this file should have size/alignment hints
 * to ensure that their representation is identical to that in OpenCL.
 */
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define MP_NWORDS 8

typedef cl_uint mp_word;

typedef struct alignas(16) {
	mp_word d[MP_NWORDS];
} mp_number;

typedef struct {
    mp_number x;
    mp_number y;
} point;

// foundVariant says which of the addresses a point yields this result was, and
// so what has to be done to the private key behind it. Kept in step with the
// same struct in profanity.cl.
typedef struct {
	cl_uint found;
	cl_uint foundId;
	cl_uint foundVariant;
	cl_uchar foundHash[20];
} result;

// A point itself, whose private key is the scalar the search prints unchanged.
#define PROFANITY_VARIANT_POINT 0

// Its negation, whose private key is that scalar's negation modulo the order of
// the curve. Only ever found when the search was built to look at both.
#define PROFANITY_VARIANT_NEGATED 1

#endif /* HPP_TYPES */