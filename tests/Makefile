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
EXECUTABLES=test_correctness_pr49.x64 test_correctness_ptx.x64 test_scoring.x64 test_create2.x64 bench_score_matching.x64 bench_mod_mul_pr49.x64 bench_mod_mul_ptx.x64

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

bench_mod_mul_ptx.x64: bench_mod_mul_ptx.cpp testutil.hpp ../types.hpp ../create2.hpp ../Mode.hpp
	$(CC) $(CFLAGS) $(CDEFINES) bench_mod_mul_ptx.cpp $(LDFLAGS) -o $@

clean:
	rm -rf $(EXECUTABLES)
