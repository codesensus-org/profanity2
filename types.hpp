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

// The six addresses one point addition can be worth, and what each does to the
// scalar a search prints to reach the private key behind it. λ is the scalar
// the curve's endomorphism multiplies by, and n the order of the curve.
//
//   POINT             s          the scalar unchanged
//   NEGATED          -s
//   LAMBDA            λs
//   LAMBDA_NEGATED   -λs
//   LAMBDA2           λ²s
//   LAMBDA2_NEGATED  -λ²s
//
// all taken mod n. Which of them a search can find depends on what it was
// built to look at: one, the first; two, the first pair; six, all of them.
#define PROFANITY_VARIANT_POINT 0
#define PROFANITY_VARIANT_NEGATED 1
#define PROFANITY_VARIANT_LAMBDA 2
#define PROFANITY_VARIANT_LAMBDA_NEGATED 3
#define PROFANITY_VARIANT_LAMBDA2 4
#define PROFANITY_VARIANT_LAMBDA2_NEGATED 5

#endif /* HPP_TYPES */