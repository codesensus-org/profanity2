#ifndef HPP_MODE
#define HPP_MODE

#include <string>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// What a mode hands the scoring kernel. Wide enough for one entry per address
// nibble, plus the one that marks where a mask shorter than an address ends.
#define PROFANITY_MODE_DATA 41

enum HashTarget {
	ADDRESS,
	CONTRACT,
	HASH_TARGET_COUNT
};

class Mode {
	private:
		Mode();

	public:
		static Mode matching(const std::string strHex);
		static Mode range(const cl_uchar min, const cl_uchar max);
		static Mode leading(const char charLeading);
		static Mode leadingRange(const cl_uchar min, const cl_uchar max);
		static Mode mirror();

		static Mode benchmark();
		static Mode zeros();
		static Mode zeroBytes();
		static Mode letters();
		static Mode numbers();
		static Mode doubles();

		std::string name;

		std::string kernel;

		HashTarget target;
		// Address, Contract, ...
		std::string transformName() const;

		cl_uchar data1[PROFANITY_MODE_DATA];
		cl_uchar data2[PROFANITY_MODE_DATA];
		cl_uchar score;
};

#endif /* HPP_MODE */
