#include <cctype>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <map>
#include <set>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h> // Included to get topology to get an actual unique identifier per device
#else
#include <CL/cl.h>
#include <CL/cl_ext.h> // Included to get topology to get an actual unique identifier per device
#endif

#define CL_DEVICE_PCI_BUS_ID_NV  0x4008
#define CL_DEVICE_PCI_SLOT_ID_NV 0x4009

#include "Dispatcher.hpp"
#include "ArgParser.hpp"
#include "Mode.hpp"
#include "help.hpp"

std::string readFile(const char * const szFilename)
{
	std::ifstream in(szFilename, std::ios::in | std::ios::binary);
	std::ostringstream contents;
	contents << in.rdbuf();
	return contents.str();
}

std::vector<cl_device_id> getAllDevices(cl_device_type deviceType = CL_DEVICE_TYPE_GPU) {
	std::vector<cl_device_id> vDevices;

	const char * const what = deviceType == CL_DEVICE_TYPE_CPU ? "CPU" : "GPU";

	cl_uint platformIdCount = 0;
	cl_int err = clGetPlatformIDs(0, NULL, &platformIdCount);
	if (err != CL_SUCCESS || platformIdCount == 0) {
		std::cerr << "warning: no OpenCL platforms found, err = " << err << std::endl;
		return vDevices;
	}

	std::vector<cl_platform_id> platformIds(platformIdCount);
	err = clGetPlatformIDs(platformIdCount, platformIds.data(), NULL);
	if (err != CL_SUCCESS) {
		std::cerr << "warning: failed to enumerate OpenCL platforms, err = " << err << std::endl;
		return vDevices;
	}

	for (auto it = platformIds.cbegin(); it != platformIds.cend(); ++it) {
		cl_uint countDevice = 0;

		err = clGetDeviceIDs(*it, deviceType, 0, NULL, &countDevice);
		if (err != CL_SUCCESS || countDevice == 0) {
			char platformName[256] = {0};
			clGetPlatformInfo(*it, CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL);
			std::cerr << "warning: skipping OpenCL platform with no usable " << what << " devices: "
			          << platformName << ", err = " << err << std::endl;
			continue;
		}

		std::vector<cl_device_id> deviceIds(countDevice);
		err = clGetDeviceIDs(*it, deviceType, countDevice, deviceIds.data(), &countDevice);
		if (err != CL_SUCCESS) {
			char platformName[256] = {0};
			clGetPlatformInfo(*it, CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL);
			std::cerr << "warning: failed to get " << what << " devices from platform: "
			          << platformName << ", err = " << err << std::endl;
			continue;
		}

		std::copy(deviceIds.begin(), deviceIds.end(), std::back_inserter(vDevices));
	}

	return vDevices;
}

template <typename T, typename U, typename V, typename W>
T clGetWrapper(U function, V param, W param2) {
	T t;
	function(param, param2, sizeof(t), &t, NULL);
	return t;
}

template <typename U, typename V, typename W>
std::string clGetWrapperString(U function, V param, W param2) {
	size_t len;
	function(param, param2, 0, NULL, &len);
	char * const szString = new char[len];
	function(param, param2, len, szString, NULL);
	std::string r(szString);
	delete[] szString;
	return r;
}

template <typename T, typename U, typename V, typename W>
std::vector<T> clGetWrapperVector(U function, V param, W param2) {
	size_t len;
	function(param, param2, 0, NULL, &len);
	len /= sizeof(T);
	std::vector<T> v;
	if (len > 0) {
		T * pArray = new T[len];
		function(param, param2, len * sizeof(T), pArray, NULL);
		for (size_t i = 0; i < len; ++i) {
			v.push_back(pArray[i]);
		}
		delete[] pArray;
	}
	return v;
}

std::vector<std::string> getBinaries(cl_program & clProgram) {
	std::vector<std::string> vReturn;
	auto vSizes = clGetWrapperVector<size_t>(clGetProgramInfo, clProgram, CL_PROGRAM_BINARY_SIZES);
	if (!vSizes.empty()) {
		unsigned char * * pBuffers = new unsigned char *[vSizes.size()];
		for (size_t i = 0; i < vSizes.size(); ++i) {
			pBuffers[i] = new unsigned char[vSizes[i]];
		}

		clGetProgramInfo(clProgram, CL_PROGRAM_BINARIES, vSizes.size() * sizeof(unsigned char *), pBuffers, NULL);
		for (size_t i = 0; i < vSizes.size(); ++i) {
			std::string strData(reinterpret_cast<char *>(pBuffers[i]), vSizes[i]);
			vReturn.push_back(strData);
			delete[] pBuffers[i];
		}

		delete[] pBuffers;
	}

	return vReturn;
}

// FNV-1a, for naming things that need to be told apart rather than kept secret:
// which sources a compiled kernel came from, and which device it was compiled
// for. Only that two different inputs are unlikely to land on the same name.
uint64_t fingerprint(const std::string & s) {
	uint64_t hash = 0xcbf29ce484222325ULL;

	for (const char c : s) {
		hash ^= (uint64_t)(unsigned char)c;
		hash *= 0x100000001b3ULL;
	}

	return hash;
}

unsigned int getUniqueDeviceIdentifier(const cl_device_id & deviceId) {
	// Where a device sits on the PCI bus is what tells it from its siblings, and
	// both of the extensions that answer that are vendor ones: a CPU device has
	// no such place, and a driver without the extension will not say. Neither
	// call is checked by clGetWrapper, so what came back on a device that has
	// neither used to be whatever was on the stack — a different answer each
	// run, and the kernel cache is filed under this. That meant recompiling
	// every start and leaving another cache file behind each time. The name is
	// the fallback: dull, but the same tomorrow.
	//
	// Recent Khronos headers define CL_DEVICE_TOPOLOGY_AMD but dropped the
	// cl_device_topology_amd struct and the TYPE_PCIE constant, hence the
	// second condition.
#if defined(CL_DEVICE_TOPOLOGY_AMD) && defined(CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD)
	cl_device_topology_amd topology;
	if (clGetDeviceInfo(deviceId, CL_DEVICE_TOPOLOGY_AMD, sizeof(topology), &topology, NULL) == CL_SUCCESS
			&& topology.raw.type == CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD) {
		return (topology.pcie.bus << 16) + (topology.pcie.device << 8) + topology.pcie.function;
	}
#endif
	cl_int bus_id = 0;
	cl_int slot_id = 0;
	if (clGetDeviceInfo(deviceId, CL_DEVICE_PCI_BUS_ID_NV, sizeof(bus_id), &bus_id, NULL) == CL_SUCCESS
			&& clGetDeviceInfo(deviceId, CL_DEVICE_PCI_SLOT_ID_NV, sizeof(slot_id), &slot_id, NULL) == CL_SUCCESS) {
		return (bus_id << 16) + slot_id;
	}

	return (unsigned int) fingerprint(clGetWrapperString(clGetDeviceInfo, deviceId, CL_DEVICE_NAME));
}

template <typename T> bool printResult(const T & t, const cl_int & err) {
	std::cout << ((t == NULL) ? toString(err) : "OK") << std::endl;
	return t == NULL;
}

bool printResult(const cl_int err) {
	std::cout << ((err != CL_SUCCESS) ? toString(err) : "OK") << std::endl;
	return err != CL_SUCCESS;
}

// A compiled kernel is only interchangeable with the source and the build
// options it came from, so both go into the name it is filed under. Keyed on the
// device and the tuning alone — as this was — a cached binary outlives the
// kernels it was built from: an upgrade that changes what the host passes a
// kernel, or what a mode packs into data1, would be answered by a binary
// compiled against the old meaning, and the search would run at full speed
// scoring the wrong thing. A name that no longer matches simply misses, and the
// kernel is compiled again, which is the whole cost of being wrong here.
std::string getDeviceCacheFilename(cl_device_id & d, const uint64_t kernelId) {
	const auto uniqueId = getUniqueDeviceIdentifier(d);

	std::ostringstream ss;
	ss << std::hex << std::setfill('0') << std::setw(16) << kernelId;

	return "cache-opencl." + ss.str() + "." + toString(uniqueId);
}

// A hex argument as the exact number of bytes it has to spell, with the 0x an
// address is usually written with allowed but not required. Nothing is written
// unless the whole of it reads, so a rejected argument leaves no half of itself
// behind for a later check to pass on.
static bool parseHex(const std::string & strHex, cl_uchar * const out, const size_t count) {
	const std::string s = (strHex.size() >= 2 && strHex[0] == '0' && (strHex[1] == 'x' || strHex[1] == 'X'))
		? strHex.substr(2)
		: strHex;

	if (s.size() != count * 2) {
		return false;
	}

	cl_uchar bytes[32];
	if (count > sizeof(bytes)) {
		return false;
	}

	static const std::string hex = "0123456789abcdef";

	for (size_t i = 0; i < count; ++i) {
		const auto indexHi = hex.find((char) std::tolower((unsigned char) s[i * 2]));
		const auto indexLo = hex.find((char) std::tolower((unsigned char) s[i * 2 + 1]));

		if (indexHi == std::string::npos || indexLo == std::string::npos) {
			return false;
		}

		bytes[i] = (cl_uchar)((indexHi << 4) | indexLo);
	}

	std::copy(bytes, bytes + count, out);
	return true;
}

int main(int argc, char * * argv) {
	// THIS LINE WILL LEAD TO A COMPILE ERROR. THIS TOOL SHOULD NOT BE USED, SEE README.

	// ^^ Commented previous line and excluded private key generation out of scope of this project,
	// now it only advances provided public key to a random offset to find vanity address

	try {
		ArgParser argp(argc, argv);
		bool bHelp = false;
		bool bModeBenchmark = false;
		bool bModeZeros = false;
		bool bModeZeroBytes = false;
		bool bModeLetters = false;
		bool bModeNumbers = false;
		std::string strModeLeading;
		std::string strModeMatching;
		std::string strPublicKey;
		bool bModeLeadingRange = false;
		bool bModeRange = false;
		bool bModeMirror = false;
		bool bModeDoubles = false;
		int rangeMin = 0;
		int rangeMax = 0;
		std::vector<size_t> vDeviceSkipIndex;
		size_t worksizeLocal = 64;
		size_t worksizeMax = 0; // Will be automatically determined later if not overriden by user
		bool bNoCache = false;
		bool bUseCpu = false;
		size_t inverseSize = 255;
		size_t inverseMultiple = 16384;
		size_t inverseStrip = 0;
		size_t inverseGroup = 0;
		bool bMineContract = false;
		size_t variants = 1;
		std::string strFactory;
		std::string strCaller;
		std::string strInitCodeHash;
		int scoreMin = 0;

		argp.addSwitch('h', "help", bHelp);
		argp.addSwitch('0', "benchmark", bModeBenchmark);
		argp.addSwitch('1', "zeros", bModeZeros);
		argp.addSwitch('2', "letters", bModeLetters);
		argp.addSwitch('3', "numbers", bModeNumbers);
		argp.addSwitch('4', "leading", strModeLeading);
		argp.addSwitch('5', "matching", strModeMatching);
		argp.addSwitch('6', "leading-range", bModeLeadingRange);
		argp.addSwitch('7', "range", bModeRange);
		argp.addSwitch('8', "mirror", bModeMirror);
		argp.addSwitch('9', "leading-doubles", bModeDoubles);
		argp.addSwitch('m', "min", rangeMin);
		argp.addSwitch('M', "max", rangeMax);
		argp.addMultiSwitch('s', "skip", vDeviceSkipIndex);
		argp.addSwitch('w', "work", worksizeLocal);
		argp.addSwitch('W', "work-max", worksizeMax);
		argp.addSwitch('n', "no-cache", bNoCache);
		argp.addSwitch('C', "cpu", bUseCpu);
		argp.addSwitch('i', "inverse-size", inverseSize);
		argp.addSwitch('I', "inverse-multiple", inverseMultiple);
		argp.addSwitch('S', "inverse-strip", inverseStrip);
		argp.addSwitch('G', "inverse-group", inverseGroup);
		argp.addSwitch('c', "contract", bMineContract);
		argp.addSwitch('x', "create2", strFactory);
		argp.addSwitch('a', "caller", strCaller);
		argp.addSwitch('k', "init-code-hash", strInitCodeHash);
		argp.addSwitch('z', "publicKey", strPublicKey);
		argp.addSwitch('b', "zero-bytes", bModeZeroBytes);
		argp.addSwitch('r', "min-score", scoreMin);
		argp.addSwitch('V', "variants", variants);

		if (!argp.parse()) {
			std::cout << "error: bad arguments, try again :<" << std::endl;
			return 1;
		}

		if (bHelp) {
			std::cout << g_strHelp << std::endl;
			return 0;
		}

		if ((inverseStrip == 0) != (inverseGroup == 0)) {
			std::cout << "error: --inverse-strip and --inverse-group must both be zero (disabled) or both be non-zero" << std::endl;
			return 1;
		}

		// Six is the order of the curve's automorphism group and so the most
		// addresses one point addition can be worth. Refused rather than clamped:
		// a run asked for more than there is would otherwise report a speed for
		// work it never did.
		if (variants < 1 || variants > 6) {
			std::cout << "error: --variants must be between 1 and 6, got " << variants << std::endl;
			std::cout << "  secp256k1 has six automorphisms and each is worth one address per point;" << std::endl;
			std::cout << "  a seventh would cost a point addition, which is what this is saving." << std::endl;
			return 1;
		}

		if (inverseStrip != 0) {
			const size_t size = inverseSize * inverseMultiple;

			if ((inverseGroup & (inverseGroup - 1)) != 0) {
				std::cout << "error: --inverse-group must be a power of two, got " << inverseGroup << std::endl;
				return 1;
			}

			if (size % (inverseStrip * inverseGroup) != 0) {
				std::cout << "error: --inverse-size * --inverse-multiple (" << size << ") must be a multiple of "
					<< "--inverse-strip * --inverse-group (" << inverseStrip * inverseGroup << ")" << std::endl;
				return 1;
			}
		}

		if (scoreMin < 0 || scoreMin > PROFANITY_MAX_SCORE) {
			std::cout << "error: --min-score must be between 1 and " << PROFANITY_MAX_SCORE << ", got " << scoreMin << std::endl;
			return 1;
		}

		Mode mode = Mode::benchmark();
		if (bModeBenchmark) {
			mode = Mode::benchmark();
		} else if (bModeZeros) {
			mode = Mode::zeros();
		} else if (bModeLetters) {
			mode = Mode::letters();
		} else if (bModeNumbers) {
			mode = Mode::numbers();
		} else if (!strModeLeading.empty()) {
			mode = Mode::leading(strModeLeading.front());
		} else if (!strModeMatching.empty()) {
			mode = Mode::matching(strModeMatching);
		} else if (bModeLeadingRange) {
			mode = Mode::leadingRange(rangeMin, rangeMax);
		} else if (bModeRange) {
			mode = Mode::range(rangeMin, rangeMax);
		} else if(bModeMirror) {
			mode = Mode::mirror();
		} else if (bModeDoubles) {
			mode = Mode::doubles();
		} else if (bModeZeroBytes) {
			mode = Mode::zeroBytes();
		} else {
			std::cout << g_strHelp << std::endl;
			return 0;
		}
		
		const bool bMineCreate2 = !strFactory.empty();

		create2 clCreate2;
		std::fill((cl_uchar *) &clCreate2, (cl_uchar *) &clCreate2 + sizeof(clCreate2), cl_uchar(0));

		if (bMineCreate2) {
			if (bMineContract) {
				std::cout << "error: --contract and --create2 are two different addresses to score, pick one" << std::endl;
				return 1;
			}

			if (!parseHex(strFactory, clCreate2.factory, sizeof(clCreate2.factory))) {
				std::cout << "error: --create2 takes the deploying contract's address, 40 hexadecimal characters" << std::endl;
				return 1;
			}

			if (strInitCodeHash.empty()) {
				std::cout << "error: --create2 needs --init-code-hash, the keccak256 of the init code being deployed" << std::endl;
				return 1;
			}

			if (!parseHex(strInitCodeHash, clCreate2.initCodeHash, sizeof(clCreate2.initCodeHash))) {
				std::cout << "error: --init-code-hash must be 64 hexadecimal characters" << std::endl;
				return 1;
			}

			// Left out, the salt's first twenty bytes stay zero, which is what
			// the factories that look at them take to mean anyone may deploy it.
			if (!strCaller.empty() && !parseHex(strCaller, clCreate2.caller, sizeof(clCreate2.caller))) {
				std::cout << "error: --caller takes an address, 40 hexadecimal characters" << std::endl;
				return 1;
			}
		} else {
			if (!strCaller.empty() || !strInitCodeHash.empty()) {
				std::cout << "error: --caller and --init-code-hash say nothing without --create2" << std::endl;
				return 1;
			}

			// A CREATE2 search has no key in it: what it varies is the salt,
			// which is public, and the addresses it finds are contracts nobody
			// holds a private key for. So there is nothing for a seed to keep
			// safe and nothing to ask for, and outsourcing such a run is safe
			// for the plainer reason that there is no secret to lose.
			if (strPublicKey.length() == 0) {
				std::cout << "error: this tool requires your public key to derive it's private key security" << std::endl;
				return 1;
			}

			if (strPublicKey.length() != 128) {
				std::cout << "error: public key must be 128 hexademical characters long" << std::endl;
				return 1;
			}
		}

		std::cout << "Mode: " << mode.name << std::endl;

		if (bMineCreate2) {
			mode.target = CREATE2;
		} else if (bMineContract) {
			mode.target = CONTRACT;
		} else {
			mode.target = ADDRESS;
		}
		std::cout << "Target: " << mode.transformName() << std:: endl;

		if (scoreMin > 0) {
			std::cout << "Reporting: every hash scoring " << scoreMin << " or more, for as long as this runs" << std::endl;
		} else {
			std::cout << "Reporting: each hash that beats the best so far (see --min-score)" << std::endl;
		}

		// Read before the devices are looked at rather than where the compiler
		// needs them, because what a cached binary may be reused for is decided
		// down there and these are what decides it.
		const std::string strKeccak = readFile("keccak.cl");
		const std::string strVanity = readFile("profanity.cl");
		const std::string strBuildOptions = "-D PROFANITY_INVERSE_SIZE=" + toString(inverseSize) + " -D PROFANITY_MAX_SCORE=" + toString(PROFANITY_MAX_SCORE)
			+ " -D PROFANITY_INVERSE_STRIP=" + toString(inverseStrip) + " -D PROFANITY_INVERSE_GROUP=" + toString(inverseGroup)
			+ " -D PROFANITY_MODE_DATA=" + toString(PROFANITY_MODE_DATA)
			+ " -D PROFANITY_CREATE2_WORDS=" + toString(PROFANITY_CREATE2_WORDS)
			+ " -D PROFANITY_CREATE2_COUNTER=" + toString(PROFANITY_CREATE2_COUNTER)
			+ " -D PROFANITY_VARIANTS=" + toString(variants);

		const uint64_t kernelId = fingerprint(strKeccak + strVanity + strBuildOptions);

		// A CPU device instead of the graphics cards, not as well as them: one
		// would contribute a rounding error's worth of hashes to a run with a
		// GPU in it while taking the cores that feed the GPU to do it.
		const cl_device_type deviceType = bUseCpu ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU;
		const char * const deviceLabel = bUseCpu ? "CPU" : "GPU";

		std::vector<cl_device_id> vFoundDevices = getAllDevices(deviceType);
		std::vector<cl_device_id> vDevices;
		std::map<cl_device_id, size_t> mDeviceIndex;

		std::vector<std::string> vDeviceBinary;
		std::vector<size_t> vDeviceBinarySize;
		cl_int errorCode;
		bool bUsedCache = false;

		std::cout << "Devices:" << std::endl;
		for (size_t i = 0; i < vFoundDevices.size(); ++i) {
			// Ignore devices in skip index
			if (std::find(vDeviceSkipIndex.begin(), vDeviceSkipIndex.end(), i) != vDeviceSkipIndex.end()) {
				continue;
			}

			cl_device_id & deviceId = vFoundDevices[i];

			const auto strName = clGetWrapperString(clGetDeviceInfo, deviceId, CL_DEVICE_NAME);
			const auto computeUnits = clGetWrapper<cl_uint>(clGetDeviceInfo, deviceId, CL_DEVICE_MAX_COMPUTE_UNITS);
			const auto globalMemSize = clGetWrapper<cl_ulong>(clGetDeviceInfo, deviceId, CL_DEVICE_GLOBAL_MEM_SIZE);
			bool precompiled = false;

			// Check if there's a prebuilt binary for this device and load it
			if(!bNoCache) {
				std::ifstream fileIn(getDeviceCacheFilename(deviceId, kernelId), std::ios::binary);
				if (fileIn.is_open()) {
					vDeviceBinary.push_back(std::string((std::istreambuf_iterator<char>(fileIn)), std::istreambuf_iterator<char>()));
					vDeviceBinarySize.push_back(vDeviceBinary.back().size());
					precompiled = true;
				}
			}

			std::cout << "  " << deviceLabel << i << ": " << strName << ", " << globalMemSize << " bytes available, " << computeUnits << " compute units (precompiled = " << (precompiled ? "yes" : "no") << ")" << std::endl;
			vDevices.push_back(vFoundDevices[i]);
			mDeviceIndex[vFoundDevices[i]] = i;
		}

		if (vDevices.empty()) {
			std::cout << "error: no OpenCL " << deviceLabel << " devices found"
				<< (bUseCpu ? ". Is a CPU OpenCL runtime such as PoCL installed?" : ". Try --cpu to run on the processor instead.") << std::endl;
			return 1;
		}

		std::cout << std::endl;
		std::cout << "Initializing OpenCL..." << std::endl;
		std::cout << "  Creating context..." << std::flush;
		auto clContext = clCreateContext( NULL, vDevices.size(), vDevices.data(), NULL, NULL, &errorCode);
		if (printResult(clContext, errorCode)) {
			return 1;
		}

		cl_program clProgram;
		if (vDeviceBinary.size() == vDevices.size()) {
			// Create program from binaries
			bUsedCache = true;

			std::cout << "  Loading kernel from binary..." << std::flush;
			const unsigned char * * pKernels = new const unsigned char *[vDevices.size()];
			for (size_t i = 0; i < vDeviceBinary.size(); ++i) {
				pKernels[i] = reinterpret_cast<const unsigned char *>(vDeviceBinary[i].data());
			}

			cl_int * pStatus = new cl_int[vDevices.size()];

			clProgram = clCreateProgramWithBinary(clContext, vDevices.size(), vDevices.data(), vDeviceBinarySize.data(), pKernels, pStatus, &errorCode);
			if(printResult(clProgram, errorCode)) {
				return 1;
			}
		} else {
			// Create a program from the kernel source
			std::cout << "  Compiling kernel..." << std::flush;
			const char * szKernels[] = { strKeccak.c_str(), strVanity.c_str() };

			clProgram = clCreateProgramWithSource(clContext, sizeof(szKernels) / sizeof(char *), szKernels, NULL, &errorCode);
			if (printResult(clProgram, errorCode)) {
				return 1;
			}
		}

		// Build the program
		std::cout << "  Building program..." << std::flush;
		if (printResult(clBuildProgram(clProgram, vDevices.size(), vDevices.data(), strBuildOptions.c_str(), NULL, NULL))) {
#ifdef PROFANITY_DEBUG
			std::cout << std::endl;
			std::cout << "build log:" << std::endl;

			size_t sizeLog;
			clGetProgramBuildInfo(clProgram, vDevices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &sizeLog);
			char * const szLog = new char[sizeLog];
			clGetProgramBuildInfo(clProgram, vDevices[0], CL_PROGRAM_BUILD_LOG, sizeLog, szLog, NULL);

			std::cout << szLog << std::endl;
			delete[] szLog;
#endif
			return 1;
		}

		// Save binary to improve future start times
		if( !bUsedCache && !bNoCache ) {
			std::cout << "  Saving program..." << std::flush;
			auto binaries = getBinaries(clProgram);
			for (size_t i = 0; i < binaries.size(); ++i) {
				std::ofstream fileOut(getDeviceCacheFilename(vDevices[i], kernelId), std::ios::binary);
				fileOut.write(binaries[i].data(), binaries[i].size());
			}
			std::cout << "OK" << std::endl;
		}

		std::cout << std::endl;

		Dispatcher d(clContext, clProgram, mode, worksizeMax == 0 ? inverseSize * inverseMultiple : worksizeMax, inverseSize, inverseMultiple, inverseStrip, inverseGroup, (cl_uchar) scoreMin, 0, strPublicKey, clCreate2, variants);
		for (auto & i : vDevices) {
			d.addDevice(i, worksizeLocal, mDeviceIndex[i]);
		}

		d.run();
		clReleaseContext(clContext);
		return 0;
	} catch (std::runtime_error & e) {
		std::cout << "std::runtime_error - " << e.what() << std::endl;
	} catch (...) {
		std::cout << "unknown exception occured" << std::endl;
	}

	return 1;
}

