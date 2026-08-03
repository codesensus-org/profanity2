# Builds the mp-math correctness tests and benchmarks. Mirrors the style of the
# main Makefile. Requires an OpenCL runtime; on machines without a GPU, PoCL
# (http://portablecl.org, e.g. `apt install pocl-opencl-icd`) works fine.
#
# Usage:
#   make
#   ./test_correctness_pr49.x64 [num_random_cases]
#   ./test_scoring.x64 [num_random_cases]
#   ./test_create2.x64
#   ./bench_score_matching.x64 [items] [iterations]
#   ./bench_mod_mul_pr49.x64 [global_size] [iterations] [repetitions]
#
# The ones that read profanity.cl look for it relative to the working directory,
# so run them from the repository root: ./tests/test_scoring.x64

CC=g++
CDEFINES=
EXECUTABLES=test_correctness_pr49.x64 test_correctness_ptx.x64 test_scoring.x64 test_create2.x64 test_correctness_oneshot.x64 test_keccak_equiv.x64 test_keccak_create2_equiv.x64 bench_score_matching.x64 bench_mod_mul_pr49.x64 bench_mod_mul_ptx.x64 bench_create2_state.x64 bench_address_state.x64

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	LDFLAGS=-framework OpenCL
	CFLAGS=-std=c++11 -Wall -O2
else
	LDFLAGS=-lOpenCL
	CFLAGS=-std=c++11 -Wall -O2
endif

all: $(EXECUTABLES)

test_correctness_pr49.x64: test_correctness_pr49.cpp testutil.hpp ../types.hpp ../create2.hpp ../Mode.hpp
	$(CC) $(CFLAGS) $(CDEFINES) test_correctness_pr49.cpp $(LDFLAGS) -o $@

# Both link the real Mode, so how a mask is encoded is under test as well.
test_scoring.x64: test_scoring.cpp testutil.hpp ../types.hpp ../Mode.hpp ../Mode.cpp
	$(CC) $(CFLAGS) $(CDEFINES) test_scoring.cpp ../Mode.cpp $(LDFLAGS) -o $@

# The layout the kernel and the host have to agree on lives in ../create2.hpp,
# and this reads it from there rather than restating it — a test that restated
# it would pass on the two of them disagreeing.
test_create2.x64: test_create2.cpp testutil.hpp ../types.hpp ../create2.hpp ../Mode.hpp
	$(CC) $(CFLAGS) $(CDEFINES) test_create2.cpp $(LDFLAGS) -o $@

bench_score_matching.x64: bench_score_matching.cpp testutil.hpp ../types.hpp ../Mode.hpp ../Mode.cpp
	$(CC) $(CFLAGS) $(CDEFINES) bench_score_matching.cpp ../Mode.cpp $(LDFLAGS) -o $@

bench_mod_mul_pr49.x64: bench_mod_mul_pr49.cpp testutil.hpp ../types.hpp ../create2.hpp ../Mode.hpp
	$(CC) $(CFLAGS) $(CDEFINES) bench_mod_mul_pr49.cpp $(LDFLAGS) -o $@

# The two below only have something to measure on NVIDIA, where profanity.cl
# compiles its inline-PTX multiprecision routines. They build and run anywhere;
# elsewhere they report that there was nothing to compare.
test_correctness_ptx.x64: test_correctness_ptx.cpp testutil.hpp ../types.hpp ../create2.hpp ../Mode.hpp
	$(CC) $(CFLAGS) $(CDEFINES) test_correctness_ptx.cpp $(LDFLAGS) -o $@

test_correctness_oneshot.x64: test_correctness_oneshot.cpp testutil.hpp ../types.hpp ../create2.hpp ../Mode.hpp
	$(CC) $(CFLAGS) $(CDEFINES) test_correctness_oneshot.cpp $(LDFLAGS) -o $@

test_keccak_equiv.x64: test_keccak_equiv.cpp testutil.hpp ../types.hpp
	$(CC) $(CFLAGS) $(CDEFINES) test_keccak_equiv.cpp $(LDFLAGS) -o $@

test_keccak_create2_equiv.x64: test_keccak_create2_equiv.cpp testutil.hpp ../types.hpp ../create2.hpp
	$(CC) $(CFLAGS) $(CDEFINES) test_keccak_create2_equiv.cpp $(LDFLAGS) -o $@

bench_mod_mul_ptx.x64: bench_mod_mul_ptx.cpp testutil.hpp ../types.hpp ../create2.hpp ../Mode.hpp
	$(CC) $(CFLAGS) $(CDEFINES) bench_mod_mul_ptx.cpp $(LDFLAGS) -o $@

# Why profanity_create2 writes its hash state out a lane at a time instead of
# through the loop it obviously wants to be.
bench_create2_state.x64: bench_create2_state.cpp testutil.hpp ../types.hpp ../create2.hpp ../Mode.hpp
	$(CC) $(CFLAGS) $(CDEFINES) bench_create2_state.cpp $(LDFLAGS) -o $@

# And whether the same thing is worth anything to the addresses a search that is
# not looking for a CREATE2 one spends all of its time on.
bench_address_state.x64: bench_address_state.cpp testutil.hpp ../types.hpp ../create2.hpp ../Mode.hpp
	$(CC) $(CFLAGS) $(CDEFINES) bench_address_state.cpp $(LDFLAGS) -o $@

clean:
	rm -rf $(EXECUTABLES)
