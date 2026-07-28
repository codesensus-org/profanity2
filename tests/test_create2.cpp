/* test_create2.cpp
 * ================
 * That the CREATE2 kernels hash what EIP-1014 says they should.
 *
 * The vectors are the specification's own, run through the real
 * profanity_create2_score_range kernel with the real buildCreate2Template
 * behind it, so what is under test is the whole agreement between the two: the
 * preimage the host lays out, the keccak padding it leaves for the kernel, and
 * where in the salt the counter is written.
 *
 * The range scorer is chosen for scoring every address at 40 whatever it holds
 * — the point here is the address the kernel arrives at rather than what it
 * makes of it, and a scorer that returns zero would leave nothing behind to
 * read: see profanity_result_update.
 *
 * A search only ever varies the salt's last eight bytes, so a vector whose salt
 * differs anywhere else is handed over as the caller — which is the same twenty
 * bytes of preimage. That the two arrive at the same place is the point of them
 * being separate arguments at all.
 *
 * Build & run (see tests/Makefile), from the repository root:
 *   cd tests && make && cd .. && ./tests/test_create2.x64
 */

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "testutil.hpp"

static int g_failures = 0;

static std::vector<uint8_t> fromHex(const std::string & s) {
	std::vector<uint8_t> out;
	for (size_t i = 0; i + 1 < s.size(); i += 2) {
		const auto digit = [](const char c) {
			return (uint8_t)(c <= '9' ? c - '0' : (std::tolower(c) - 'a' + 10));
		};
		out.push_back((uint8_t)((digit(s[i]) << 4) | digit(s[i + 1])));
	}
	return out;
}

static std::string toHex(const uint8_t * const bytes, const size_t len) {
	static const char digits[] = "0123456789abcdef";
	std::string out;
	for (size_t i = 0; i < len; ++i) {
		out += digits[bytes[i] >> 4];
		out += digits[bytes[i] & 0xF];
	}
	return out;
}

/* ------------------------------------------------------------------------ */

struct Vector {
	const char * what;
	const char * factory;
	const char * caller;       // the salt's first 20 bytes
	cl_ulong counter;          // written over its last 8; the 4 between are zero
	const char * initCodeHash;
	const char * expected;
};

// From EIP-1014, with each salt split the way a search does: whatever is not
// the counter is the caller, and the four bytes of nonce between them are zero
// in every vector here.
static const Vector g_vectors[] = {
	{
		"zeros throughout",
		"0000000000000000000000000000000000000000",
		"0000000000000000000000000000000000000000",
		0,
		"bc36789e7a1e281436464229828f817d6612f7b477d66591ff96a9e064bcc98a", // keccak256(0x00)
		"4d1a2e2bb4f88f0250f26ffff098b0b30b26bf38",
	},
	{
		"a factory that is not zero",
		"deadbeef00000000000000000000000000000000",
		"0000000000000000000000000000000000000000",
		0,
		"bc36789e7a1e281436464229828f817d6612f7b477d66591ff96a9e064bcc98a",
		"b928f69bb1d91cd65274e3c79d8986362984fda3",
	},
	{
		"a salt whose fixed half is not zero either",
		"deadbeef00000000000000000000000000000000",
		"000000000000000000000000feed000000000000",
		0,
		"bc36789e7a1e281436464229828f817d6612f7b477d66591ff96a9e064bcc98a",
		"d04116cdd17bebe565eb2422f2497e06cc1c9833",
	},
	{
		// The one that pins where the counter goes and which way round it is
		// written: a byte out of place here lands on a different address.
		"a counter in the bytes the search varies",
		"00000000000000000000000000000000deadbeef",
		"0000000000000000000000000000000000000000",
		0x00000000cafebabeULL,
		"d4fd4e189132273036449fc9e11198c739161b4c0116a9a2dccdfa1c492006f1", // keccak256(0xdeadbeef)
		"60f3f640a8508fc6a86d45df051962668e1e8ac7",
	},
	{
		"an init code hash of nothing at all",
		"0000000000000000000000000000000000000000",
		"0000000000000000000000000000000000000000",
		0,
		"c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470", // keccak256("")
		"e33c0c7f7df4809055c3eba6c09cfe4baf1bd9e0",
	},
};

/* ------------------------------------------------------------------------ */

// What one launch of the kernel appended: which work item and which round of it
// each address turned up at, in the order the kernel wrote them.
struct Found {
	cl_uint id;
	cl_uint round;
	std::string address;
};

// Runs profanity_create2_score_range over `globalSize` work items from
// `counterBase`. Every work item takes PROFANITY_TEST_ROUNDS counters, so this
// comes back with that many entries per work item.
static std::vector<Found> runCreate2Launch(const ClSetup & s, const Vector & v, const cl_ulong counterBase, const size_t globalSize) {
	create2 fixed;
	std::memset(&fixed, 0, sizeof(fixed));

	const auto factory = fromHex(v.factory);
	const auto caller = fromHex(v.caller);
	const auto initCodeHash = fromHex(v.initCodeHash);

	if (factory.size() != 20 || caller.size() != 20 || initCodeHash.size() != 32) {
		std::fprintf(stderr, "malformed vector: %s\n", v.what);
		std::exit(1);
	}

	std::copy(factory.begin(), factory.end(), fixed.factory);
	std::copy(caller.begin(), caller.end(), fixed.caller);
	std::copy(initCodeHash.begin(), initCodeHash.end(), fixed.initCodeHash);

	// The salt as a device holds it: the caller, a nonce this test leaves at
	// zero, and room for the counter the kernel writes.
	cl_uchar salt[32] = { 0 };
	std::copy(fixed.caller, fixed.caller + 20, salt);

	cl_uint words[PROFANITY_CREATE2_WORDS];
	buildCreate2Template(words, fixed, salt);

	// Every nibble of every address falls inside 0-15, so this scores 40 and
	// the kernel keeps what it found rather than throwing it away.
	cl_uchar data1[PROFANITY_MODE_DATA] = { 0 };
	cl_uchar data2[PROFANITY_MODE_DATA] = { 0 };
	data1[0] = 0;
	data2[0] = 15;

	std::vector<result> results(PROFANITY_TEST_MAX_SCORE + 1);
	std::memset(results.data(), 0, results.size() * sizeof(result));

	cl_int err;
	cl_mem resultBuf = clCreateBuffer(s.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, results.size() * sizeof(result), results.data(), &err);
	clCheck(err, "clCreateBuffer(result)");
	cl_mem data1Buf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(data1), data1, &err);
	clCheck(err, "clCreateBuffer(data1)");
	cl_mem data2Buf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(data2), data2, &err);
	clCheck(err, "clCreateBuffer(data2)");
	cl_mem templateBuf = clCreateBuffer(s.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(words), words, &err);
	clCheck(err, "clCreateBuffer(template)");

	cl_kernel kernel = clCreateKernel(s.program, "profanity_create2_score_range", &err);
	clCheck(err, "clCreateKernel");

	const cl_uchar scoreMax = 0;
	const cl_uchar bAppend = 1;

	clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &resultBuf), "clSetKernelArg(0)");
	clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &data1Buf), "clSetKernelArg(1)");
	clCheck(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data2Buf), "clSetKernelArg(2)");
	clCheck(clSetKernelArg(kernel, 3, sizeof(cl_uchar), &scoreMax), "clSetKernelArg(3)");
	clCheck(clSetKernelArg(kernel, 4, sizeof(cl_uchar), &bAppend), "clSetKernelArg(4)");
	clCheck(clSetKernelArg(kernel, 5, sizeof(cl_mem), &templateBuf), "clSetKernelArg(5)");
	clCheck(clSetKernelArg(kernel, 6, sizeof(cl_ulong), &counterBase), "clSetKernelArg(6)");

	clCheck(clEnqueueNDRangeKernel(s.queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL), "clEnqueueNDRangeKernel");
	clCheck(clEnqueueReadBuffer(s.queue, resultBuf, CL_TRUE, 0, results.size() * sizeof(result), results.data(), 0, NULL, NULL), "clEnqueueReadBuffer");
	clCheck(clFinish(s.queue), "clFinish");

	clReleaseKernel(kernel);
	clReleaseMemObject(resultBuf);
	clReleaseMemObject(data1Buf);
	clReleaseMemObject(data2Buf);
	clReleaseMemObject(templateBuf);

	// Every work item scores every round above the floor, so a launch appends
	// one entry per counter it searched. Anything else and the loop the rounds
	// are taken in is not running the number of times it was built for.
	const cl_uint expected = (cl_uint)(globalSize * PROFANITY_TEST_ROUNDS);
	if (results[0].found != expected) {
		std::fprintf(stderr, "the kernel appended %u results, expected %u (%zu work items x %d rounds)\n",
			results[0].found, expected, globalSize, PROFANITY_TEST_ROUNDS);
		std::exit(1);
	}

	std::vector<Found> found;
	for (cl_uint i = 0; i < expected; ++i) {
		found.push_back({ results[i + 1].foundId, results[i + 1].foundRound, toHex(results[i + 1].foundHash, 20) });
	}
	return found;
}

// The address at one counter exactly: one work item from there, and the round
// it takes first is that counter itself.
static std::string runCreate2(const ClSetup & s, const Vector & v, const cl_ulong counter) {
	for (const Found & f : runCreate2Launch(s, v, counter, 1)) {
		if (f.id == 0 && f.round == 0) {
			return f.address;
		}
	}

	std::fprintf(stderr, "the kernel appended no result for round 0\n");
	std::exit(1);
}

/* ------------------------------------------------------------------------ */

// That a salt read off a printed line is the one the kernel actually hashed:
// the host has to put the counter back exactly where the kernel wrote it.
static void testSaltRoundTrip() {
	static const cl_ulong counters[] = { 0, 1, 0xcafebabeULL, 0x0123456789abcdefULL };

	for (const cl_ulong counter : counters) {
		cl_uchar salt[32] = { 0 };
		for (int i = 0; i < 24; ++i) {
			salt[i] = (cl_uchar)(0xA0 + i);
		}

		applyCreate2Counter(salt, counter);

		cl_ulong read = 0;
		for (int i = 0; i < 8; ++i) {
			read = (read << 8) | salt[24 + i];
		}

		const bool untouched = salt[23] == (cl_uchar)(0xA0 + 23);

		if (read != counter || !untouched) {
			++g_failures;
			std::printf("FAIL  the salt does not carry the counter back: wrote %llu, read %llu%s\n",
				(unsigned long long) counter, (unsigned long long) read,
				untouched ? "" : ", and it ran over the bytes in front of it");
		}
	}

	if (g_failures == 0) {
		std::printf("PASS  a counter written into a salt reads back as itself\n");
	}
}

// Which counter a work item's round stands for. The kernel gives each work item
// a block of PROFANITY_ROUNDS consecutive counters, and Dispatcher::report puts
// a salt back together from base + foundId * rounds + foundRound — so the two
// have to mean the same thing by it, or a printed salt belongs to an address
// nobody found.
//
// Checked by asking the kernel twice: once over several work items at once, and
// once for each counter that mapping claims they searched, a work item at a time
// from that counter. The addresses have to line up.
static void testCounterMapping(const ClSetup & s) {
	// The vector with the busiest preimage of the five, so that a counter going
	// astray lands somewhere visibly different.
	const Vector & v = g_vectors[3];

	const cl_ulong base = 0x00000000cafeba00ULL;
	const size_t items = 3;

	const std::vector<Found> launch = runCreate2Launch(s, v, base, items);

	// Every work item's every round, once each: a mapping that sends two of
	// them to the same counter searches fewer salts than it is credited with.
	std::vector<bool> seen(items * PROFANITY_TEST_ROUNDS, false);
	int failures = 0;

	for (const Found & f : launch) {
		if (f.id >= items || f.round >= PROFANITY_TEST_ROUNDS) {
			++failures;
			std::printf("FAIL  the kernel reported id %u round %u, outside the %zu x %d it was launched over\n",
				f.id, f.round, items, PROFANITY_TEST_ROUNDS);
			continue;
		}

		const size_t slot = f.id * PROFANITY_TEST_ROUNDS + f.round;
		if (seen[slot]) {
			++failures;
			std::printf("FAIL  id %u round %u turned up twice\n", f.id, f.round);
			continue;
		}
		seen[slot] = true;

		const cl_ulong counter = base + (cl_ulong)f.id * PROFANITY_TEST_ROUNDS + f.round;
		const std::string expected = runCreate2(s, v, counter);

		if (f.address != expected) {
			++failures;
			std::printf("FAIL  id %u round %u should be counter %llu\n      found    0x%s\n      counter  0x%s\n",
				f.id, f.round, (unsigned long long) counter, f.address.c_str(), expected.c_str());
		}
	}

	for (size_t slot = 0; slot < seen.size(); ++slot) {
		if (!seen[slot]) {
			++failures;
			std::printf("FAIL  id %zu round %zu was never searched\n", slot / PROFANITY_TEST_ROUNDS, slot % PROFANITY_TEST_ROUNDS);
		}
	}

	g_failures += failures;
	if (failures == 0) {
		std::printf("PASS  a launch searches base + id * rounds + round, once each\n");
	}
}

int main(int argc, char * * argv) {
	(void) argc;
	(void) argv;

	testSaltRoundTrip();

	ClSetup s = clSetup();

	testCounterMapping(s);

	for (const Vector & v : g_vectors) {
		const std::string got = runCreate2(s, v, v.counter);

		if (got == v.expected) {
			std::printf("PASS  %s\n", v.what);
		} else {
			++g_failures;
			std::printf("FAIL  %s\n      got      0x%s\n      expected 0x%s\n",
				v.what, got.c_str(), v.expected);
		}
	}

	clReleaseProgram(s.program);
	clReleaseCommandQueue(s.queue);
	clReleaseContext(s.context);

	std::printf("\n%s\n", g_failures ? "FAILED" : "all EIP-1014 vectors reproduced");
	return g_failures ? 1 : 0;
}
