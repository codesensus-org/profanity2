#include "Dispatcher.hpp"

// Includes
#include <stdexcept>
#include <iostream>
#include <thread>
#include <sstream>
#include <iomanip>
#include <random>
#include <thread>
#include <algorithm>

#if defined(__APPLE__) || defined(__MACOSX)
#include <machine/endian.h>
#elif defined(_WIN32)
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif

#include "precomp.hpp"

#ifndef htonll
#define htonll(x) ((((uint64_t)htonl(x)) << 32) | htonl((x) >> 32))
#endif

static std::string::size_type fromHex(char c) {
	if (c >= 'A' && c <= 'F') {
		c += 'a' - 'A';
	}

	const std::string hex = "0123456789abcdef";
	const std::string::size_type ret = hex.find(c);
	return ret;
}

static cl_ulong4 fromHex(const std::string & strHex) {
	uint8_t data[32];
	std::fill(data, data + sizeof(data), cl_uchar(0));

	auto index = 0;
	for(size_t i = 0; i < strHex.size(); i += 2) {
		const auto indexHi = fromHex(strHex[i]);
		const auto indexLo = i + 1 < strHex.size() ? fromHex(strHex[i+1]) : std::string::npos;

		const auto valHi = (indexHi == std::string::npos) ? 0 : indexHi << 4;
		const auto valLo = (indexLo == std::string::npos) ? 0 : indexLo;

		data[index] = valHi | valLo;
		++index;
	}

	cl_ulong4 res = {
		.s = {
			htonll(*(uint64_t *)(data + 24)),
			htonll(*(uint64_t *)(data + 16)),
			htonll(*(uint64_t *)(data + 8)),
			htonll(*(uint64_t *)(data + 0)),
		}
	};
	return res;
}

// The two halves of the seed public key. A CREATE2 search has no key anywhere
// in it and is started without one, so an absent key is zero here rather than
// an error: what would read it never runs.
static cl_ulong4 fromHexKey(const std::string & strKey, const size_t at) {
	if (strKey.size() < at + 64) {
		const cl_ulong4 zero = { { 0, 0, 0, 0 } };
		return zero;
	}

	return fromHex(strKey.substr(at, 64));
}

static std::string toHex(const uint8_t * const s, const size_t len) {
	std::string b("0123456789abcdef");
	std::string r;

	for (size_t i = 0; i < len; ++i) {
		const unsigned char h = s[i] / 16;
		const unsigned char l = s[i] % 16;

		r = r + b.substr(h, 1) + b.substr(l, 1);
	}

	return r;
}

// The private key a result was found at: the seed this device started from,
// advanced by the rounds behind it and by the work item that found it.
static std::string privateKeyFor(cl_ulong4 seed, cl_ulong round, const result & r) {
	cl_ulong carry = 0;
	cl_ulong4 seedRes;

	seedRes.s[0] = seed.s[0] + round; carry = seedRes.s[0] < round;
	seedRes.s[1] = seed.s[1] + carry; carry = carry && !seedRes.s[1];
	seedRes.s[2] = seed.s[2] + carry; carry = carry && !seedRes.s[2];
	seedRes.s[3] = seed.s[3] + carry + r.foundId;

	std::ostringstream ss;
	ss << std::hex << std::setfill('0');
	ss << std::setw(16) << seedRes.s[3] << std::setw(16) << seedRes.s[2] << std::setw(16) << seedRes.s[1] << std::setw(16) << seedRes.s[0];

	return ss.str();
}

// The salt a result was found at: this device's, with the counter the kernel
// wrote over its last eight bytes put back the same way round.
static std::string saltFor(const cl_uchar * const salt, const cl_ulong counter) {
	cl_uchar found[32];
	std::copy(salt, salt + 32, found);
	applyCreate2Counter(found, counter);

	return toHex(found, 32);
}

// `what` names the thing the finder has to keep — the private key to add to
// their seed, or the salt to deploy with — and is what the worker reads the
// line for, so it is the same word on every line of a run.
static void printResult(const std::string & what, const std::string & value, const result & r, const cl_uchar score, const std::chrono::time_point<std::chrono::steady_clock> & timeStart, const Mode & mode) {
	// Time delta
	const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - timeStart).count();

	// Print
	const std::string strVT100ClearLine = "\33[2K\r";
	std::cout << strVT100ClearLine << "  Time: " << std::setw(5) << seconds << "s";

	std::cout << " Score: " << std::setw(2) << (int) score;
	std::cout << ' ' << what << ": 0x" << value << ' ';

	std::cout << mode.transformName();
	std::cout << ": 0x" << toHex(r.foundHash, 20) << std::endl;
}

unsigned int getKernelExecutionTimeMicros(cl_event & e) {
	cl_ulong timeStart = 0, timeEnd = 0;
	clWaitForEvents(1, &e);
	clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(timeStart), &timeStart, NULL);
	clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(timeEnd), &timeEnd, NULL);
	return (timeEnd - timeStart) / 1000;
}

Dispatcher::OpenCLException::OpenCLException(const std::string s, const cl_int res) :
	std::runtime_error( s + " (res = " + toString(res) + ")"),
	m_res(res)
{

}

void Dispatcher::OpenCLException::OpenCLException::throwIfError(const std::string s, const cl_int res) {
	if (res != CL_SUCCESS) {
		throw OpenCLException(s, res);
	}
}

cl_command_queue Dispatcher::Device::createQueue(cl_context & clContext, cl_device_id & clDeviceId) {
	// nVidia CUDA Toolkit 10.1 only supports OpenCL 1.2 so we revert back to older functions for compatability
#ifdef PROFANITY_DEBUG
	cl_command_queue_properties p = CL_QUEUE_PROFILING_ENABLE;
#else
	cl_command_queue_properties p = 0;
#endif

	cl_int errorCode = CL_SUCCESS;
#ifdef CL_VERSION_2_0
	const cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, p, 0 };
	const cl_command_queue ret = clCreateCommandQueueWithProperties(clContext, clDeviceId, props, &errorCode);
#else
	const cl_command_queue ret = clCreateCommandQueue(clContext, clDeviceId, p, &errorCode);
#endif
	OpenCLException::throwIfError("failed to create command queue", errorCode);
	return ret == NULL ? throw std::runtime_error("failed to create command queue") : ret;
}

cl_kernel Dispatcher::Device::createKernel(cl_program & clProgram, const std::string s) {
	cl_kernel ret  = clCreateKernel(clProgram, s.c_str(), NULL);
	return ret == NULL ? throw std::runtime_error("failed to create kernel \"" + s + "\"") : ret;
}

cl_ulong4 Dispatcher::Device::createSeed() {
#ifdef PROFANITY_DEBUG
	cl_ulong4 r;
	r.s[0] = 1;
	r.s[1] = 1;
	r.s[2] = 1;
	r.s[3] = 1;
	return r;
#else
	// We do not need really safe crypto random here, since we inherit safety
	// of the key from the user-provided seed public key.
	// We only need this random to not repeat same job among different devices
	std::random_device rd;

	cl_ulong4 diff;
	diff.s[0] = (((uint64_t)rd()) << 32) | rd();
	diff.s[1] = (((uint64_t)rd()) << 32) | rd();
	diff.s[2] = (((uint64_t)rd()) << 32) | rd();
	diff.s[3] = (((uint64_t)rd() & 0x0000ffff) << 32) | rd(); // zeroing 2 highest bytes to prevent overflowing sum private key after adding to seed private key

	// profanity_init_uniform turns these three into the point every work item
	// starts from, and there is no such point if they are all zero. One value out
	// of 2^192 is worth excluding to let the seeding kernel assume it has one.
	if (!(diff.s[0] | diff.s[1] | diff.s[2])) {
		diff.s[0] = 1;
	}

	return diff;
#endif
}

Dispatcher::Device::Device(Dispatcher & parent, cl_context & clContext, cl_program & clProgram, cl_device_id clDeviceId, const size_t worksizeLocal, const size_t size, const size_t index, const Mode & mode, cl_ulong4 clSeedX, cl_ulong4 clSeedY, const create2 & clCreate2) :
	m_parent(parent),
	m_index(index),
	m_clDeviceId(clDeviceId),
	m_worksizeLocal(worksizeLocal),
	m_clScoreMax(parent.m_clScoreMin > 0 ? parent.m_clScoreMin - 1 : 0),
	m_clQueue(createQueue(clContext, clDeviceId) ),
	m_kernelInit( createKernel(clProgram, "profanity_init") ),
	m_kernelInitUniform( createKernel(clProgram, "profanity_init_uniform") ),
	m_kernelInverse(createKernel(clProgram, "profanity_inverse")),
	m_kernelIterate(createKernel(clProgram, mode.kernelName())),
	m_kernelCreate2Init(createKernel(clProgram, "profanity_create2_init")),
	m_memPrecomp(clContext, m_clQueue, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, sizeof(g_precomp), g_precomp),
	// The points a search keeps in flight are three quarters of a gigabyte
	// apiece at the defaults and more than twice that as the worker tunes it.
	// A CREATE2 search has no points at all, so it asks for a buffer that
	// merely exists rather than one it will never read.
	m_memPointsDeltaX(clContext, m_clQueue, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, mode.target == CREATE2 ? 1 : size, true),
	m_memInversedNegativeDoubleGy(clContext, m_clQueue, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, mode.target == CREATE2 ? 1 : size, true),
	m_memPrevLambda(clContext, m_clQueue, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, mode.target == CREATE2 ? 1 : size, true),
	// Appending resets the counter heading the result buffer from the host
	// before every round, which CL_MEM_HOST_READ_ONLY would forbid.
	m_memResult(clContext, m_clQueue, parent.m_bAppend ? CL_MEM_READ_WRITE : CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, PROFANITY_MAX_SCORE + 1),
	m_memUniform(clContext, m_clQueue, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, 1, true),
	m_memData1(clContext, m_clQueue, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, PROFANITY_MODE_DATA),
	m_memData2(clContext, m_clQueue, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, PROFANITY_MODE_DATA),
	m_memCreate2(clContext, m_clQueue, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, PROFANITY_CREATE2_WORDS),
	m_clSeed(createSeed()),
	m_clSeedX(clSeedX),
	m_clSeedY(clSeedY),
	m_round(0),
	m_counterQueued(0),
	m_counterFound(0),
	m_speed(PROFANITY_SPEEDSAMPLES),
	m_sizeInitialized(0),
	m_eventFinished(NULL)
{
	// The salt this device searches from. The caller goes where a factory that
	// guards against front-running looks for it, and the counter the kernel
	// writes takes the last eight bytes. What is left between them is a nonce:
	// every device counts its rounds from zero, so without one they would all
	// grind through the same salts and several cards would be worth one.
	//
	// createSeed's randomness is enough for the same reason it is enough there
	// — nothing here is secret, and a salt is public the moment it is used.
	const cl_ulong4 nonce = createSeed();

	std::fill(m_salt, m_salt + sizeof(m_salt), cl_uchar(0));
	std::copy(clCreate2.caller, clCreate2.caller + 20, m_salt);

	for (int i = 0; i < 4; ++i) {
		m_salt[20 + i] = (cl_uchar)(nonce.s[0] >> (i * 8));
	}

	buildCreate2Template(m_memCreate2.data(), clCreate2, m_salt);
}

Dispatcher::Device::~Device() {

}

Dispatcher::Dispatcher(cl_context & clContext, cl_program & clProgram, const Mode mode, const size_t worksizeMax, const size_t inverseSize, const size_t inverseMultiple, const size_t inverseStrip, const size_t inverseGroup, const cl_uchar clScoreMin, const cl_uchar clScoreQuit, const std::string & seedPublicKey, const create2 & clCreate2)
	: m_clContext(clContext)
	, m_clProgram(clProgram)
	, m_mode(mode)
	, m_worksizeMax(worksizeMax)
	, m_inverseSize(inverseSize)
	, m_inverseStrip(inverseStrip)
	, m_inverseGroup(inverseGroup)
	, m_size(inverseSize*inverseMultiple)
	, m_clScoreMax(clScoreMin > 0 ? clScoreMin : mode.score)
	, m_clScoreMin(clScoreMin)
	, m_clScoreQuit(clScoreQuit)
	, m_bAppend(clScoreMin > 0)
	, m_bWarnedFull(false)
	, m_eventFinished(NULL)
	, m_countPrint(0)
	, m_publicKeyX(fromHexKey(seedPublicKey, 0))
	, m_publicKeyY(fromHexKey(seedPublicKey, 64))
	, m_create2(clCreate2)
{
}

Dispatcher::~Dispatcher() {

}

void Dispatcher::addDevice(cl_device_id clDeviceId, const size_t worksizeLocal, const size_t index) {
	Device * pDevice = new Device(*this, m_clContext, m_clProgram, clDeviceId, worksizeLocal, m_size, index, m_mode, m_publicKeyX, m_publicKeyY, m_create2);
	m_vDevices.push_back(pDevice);
}

void Dispatcher::run() {
	m_eventFinished = clCreateUserEvent(m_clContext, NULL);
	timeStart = std::chrono::steady_clock::now();

	init();

	const auto timeInitialization = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - timeStart).count();
	std::cout << "Initialization time: " << timeInitialization << " seconds" << std::endl;

	m_quit = false;
	m_countRunning = m_vDevices.size();

	std::cout << "Running..." << std::endl;
	if (m_mode.target == CREATE2) {
		std::cout << "  Always verify that a salt printed here really does deploy to the address" << std::endl;
		std::cout << "  beside it, by computing the CREATE2 address yourself from the factory, the" << std::endl;
		std::cout << "  salt and the init code hash. This program like any software might contain" << std::endl;
		std::cout << "  bugs and it does by design cut corners to improve overall performance." << std::endl;
	} else {
		std::cout << "  Always verify that a private key generated by this program corresponds to the" << std::endl;
		std::cout << "  public key printed by importing it to a wallet of your choice. This program" << std::endl;
		std::cout << "  like any software might contain bugs and it does by design cut corners to" << std::endl;
		std::cout << "  improve overall performance." << std::endl;
	}
	std::cout << std::endl;

	for (auto it = m_vDevices.begin(); it != m_vDevices.end(); ++it) {
		dispatch(*(*it));
	}

	clWaitForEvents(1, &m_eventFinished);
	clReleaseEvent(m_eventFinished);
	m_eventFinished = NULL;
}

void Dispatcher::init() {
	std::cout << "Initializing devices..." << std::endl;
	if (m_mode.target == CREATE2) {
		// Nothing to seed: a salt search starts wherever its counter is told to.
		std::cout << "  A CREATE2 search has no points to seed, so this is immediate." << std::endl;
	} else {
		std::cout << "  This should take less than a minute. The number of objects initialized on each" << std::endl;
		std::cout << "  device is equal to inverse-size * inverse-multiple. To lower" << std::endl;
		std::cout << "  initialization time (and memory footprint) I suggest lowering the" << std::endl;
		std::cout << "  inverse-multiple first. You can do this via the -I switch. Do note that" << std::endl;
		std::cout << "  this might negatively impact your performance." << std::endl;
	}
	std::cout << std::endl;

	const auto deviceCount = m_vDevices.size();
	m_sizeInitTotal = m_size * deviceCount;
	m_sizeInitDone = 0;

	cl_event * const pInitEvents = new cl_event[deviceCount];

	for (size_t i = 0; i < deviceCount; ++i) {
		pInitEvents[i] = clCreateUserEvent(m_clContext, NULL);
		m_vDevices[i]->m_eventFinished = pInitEvents[i];
		initBegin(*m_vDevices[i]);
	}

	clWaitForEvents(deviceCount, pInitEvents);
	for (size_t i = 0; i < deviceCount; ++i) {
		m_vDevices[i]->m_eventFinished = NULL;
		clReleaseEvent(pInitEvents[i]);
	}

	delete[] pInitEvents;

	std::cout << std::endl;
}

void Dispatcher::initBegin(Device & d) {
	// Set mode data
	for (auto i = 0; i < PROFANITY_MODE_DATA; ++i) {
		d.m_memData1[i] = m_mode.data1[i];
		d.m_memData2[i] = m_mode.data2[i];
	}

	d.m_memData1.write(true);
	d.m_memData2.write(true);

	if (m_mode.target == CREATE2) {
		initCreate2(d);
		return;
	}

	// Write precompute table
	d.m_memPrecomp.write(true);

	// Kernel arguments - profanity_begin
	d.m_memPrecomp.setKernelArg(d.m_kernelInit, 0);
	d.m_memPointsDeltaX.setKernelArg(d.m_kernelInit, 1);
	d.m_memPrevLambda.setKernelArg(d.m_kernelInit, 2);
	d.m_memResult.setKernelArg(d.m_kernelInit, 3);
	CLMemory<cl_ulong4>::setKernelArg(d.m_kernelInit, 4, d.m_clSeed);
	CLMemory<cl_ulong4>::setKernelArg(d.m_kernelInit, 5, d.m_clSeedX);
	CLMemory<cl_ulong4>::setKernelArg(d.m_kernelInit, 6, d.m_clSeedY);
	d.m_memUniform.setKernelArg(d.m_kernelInit, 7);

	// Kernel arguments - profanity_init_uniform, and the one launch of it. The
	// queue is in order, so the seeding below reads what this leaves behind.
	d.m_memPrecomp.setKernelArg(d.m_kernelInitUniform, 0);
	d.m_memUniform.setKernelArg(d.m_kernelInitUniform, 1);
	CLMemory<cl_ulong4>::setKernelArg(d.m_kernelInitUniform, 2, d.m_clSeed);

	const size_t one = 1;
	OpenCLException::throwIfError("failed to enqueue the shared starting point",
		clEnqueueNDRangeKernel(d.m_clQueue, d.m_kernelInitUniform, 1, NULL, &one, NULL, 0, NULL, NULL));

	// Kernel arguments - profanity_inverse
	d.m_memPointsDeltaX.setKernelArg(d.m_kernelInverse, 0);
	d.m_memInversedNegativeDoubleGy.setKernelArg(d.m_kernelInverse, 1);

	// Kernel arguments - profanity_iterate_score_*
	d.m_memPointsDeltaX.setKernelArg(d.m_kernelIterate, 0);
	d.m_memInversedNegativeDoubleGy.setKernelArg(d.m_kernelIterate, 1);
	d.m_memPrevLambda.setKernelArg(d.m_kernelIterate, 2);
	d.m_memResult.setKernelArg(d.m_kernelIterate, 3);
	d.m_memData1.setKernelArg(d.m_kernelIterate, 4);
	d.m_memData2.setKernelArg(d.m_kernelIterate, 5);

	CLMemory<cl_uchar>::setKernelArg(d.m_kernelIterate, 6, d.m_clScoreMax); // Updated in handleResult(), pinned under a floor
	CLMemory<cl_uchar>::setKernelArg(d.m_kernelIterate, 7, (cl_uchar) (m_bAppend ? 1 : 0));
	CLMemory<cl_uchar>::setKernelArg(d.m_kernelIterate, 8, (cl_uchar) (m_mode.target == CONTRACT ? 1 : 0));

	// Seed device
	initContinue(d);
}

// What initBegin does instead for a CREATE2 search, which is next to nothing:
// the preimage every work item hashes, the arguments that do not change from
// one round to the next, and a pass over the result buffer to start it at zero
// — the one thing the seeding it replaces was also doing for it.
void Dispatcher::initCreate2(Device & d) {
	d.m_memCreate2.write(true);

	d.m_memResult.setKernelArg(d.m_kernelCreate2Init, 0);

	const size_t slots = PROFANITY_MAX_SCORE + 1;
	OpenCLException::throwIfError("failed to enqueue the result buffer's clearing",
		clEnqueueNDRangeKernel(d.m_clQueue, d.m_kernelCreate2Init, 1, NULL, &slots, NULL, 0, NULL, NULL));

	// Kernel arguments - profanity_create2_score_*. The counter the round
	// starts at is the one argument that moves, and dispatch() sets it.
	d.m_memResult.setKernelArg(d.m_kernelIterate, 0);
	d.m_memData1.setKernelArg(d.m_kernelIterate, 1);
	d.m_memData2.setKernelArg(d.m_kernelIterate, 2);
	CLMemory<cl_uchar>::setKernelArg(d.m_kernelIterate, 3, d.m_clScoreMax); // Updated in handleResult(), pinned under a floor
	CLMemory<cl_uchar>::setKernelArg(d.m_kernelIterate, 4, (cl_uchar) (m_bAppend ? 1 : 0));
	d.m_memCreate2.setKernelArg(d.m_kernelIterate, 5);

	OpenCLException::throwIfError("failed to clear the result buffer", clFinish(d.m_clQueue));

	const std::string strOutput = "  GPU" + toString(d.m_index) + " initialized";
	std::cout << strOutput << std::endl;

	d.m_sizeInitialized = m_size;
	m_sizeInitDone += m_size;

	clSetUserEventStatus(d.m_eventFinished, CL_COMPLETE);
}

void Dispatcher::initContinue(Device & d) {
	size_t sizeLeft = m_size - d.m_sizeInitialized;
	const size_t sizeInitLimit = m_size / 20;

	// Print progress
	const size_t percentDone = m_sizeInitDone * 100 / m_sizeInitTotal;
	std::cout << "  " << percentDone << "%\r" << std::flush;

	if (sizeLeft) {
		cl_event event;
		const size_t sizeRun = std::min(sizeInitLimit, std::min(sizeLeft, m_worksizeMax));
		const auto resEnqueue = clEnqueueNDRangeKernel(d.m_clQueue, d.m_kernelInit, 1, &d.m_sizeInitialized, &sizeRun, NULL, 0, NULL, &event);
		OpenCLException::throwIfError("kernel queueing failed during initilization", resEnqueue);

		// See: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clSetEventCallback.html
		// If an application needs to wait for completion of a routine from the above list in a callback, please use the non-blocking form of the function, and
		// assign a completion callback to it to do the remainder of your work. Note that when a callback (or other code) enqueues commands to a command-queue,
		// the commands are not required to begin execution until the queue is flushed. In standard usage, blocking enqueue calls serve this role by implicitly
		// flushing the queue. Since blocking calls are not permitted in callbacks, those callbacks that enqueue commands on a command queue should either call
		// clFlush on the queue before returning or arrange for clFlush to be called later on another thread.
		clFlush(d.m_clQueue); 

		std::lock_guard<std::mutex> lock(m_mutex);
		d.m_sizeInitialized += sizeRun;
		m_sizeInitDone += sizeRun;

		const auto resCallback = clSetEventCallback(event, CL_COMPLETE, staticCallback, &d);
		OpenCLException::throwIfError("failed to set custom callback during initialization", resCallback);
	} else {
		// Printing one whole string at once helps in avoiding garbled output when executed in parallell
		const std::string strOutput = "  GPU" + toString(d.m_index) + " initialized";
		std::cout << strOutput << std::endl;
		clSetUserEventStatus(d.m_eventFinished, CL_COMPLETE);
	}
}

void Dispatcher::enqueueKernel(cl_command_queue & clQueue, cl_kernel & clKernel, size_t worksizeGlobal, const size_t worksizeLocal, cl_event * pEvent = NULL) {
	const size_t worksizeMax = m_worksizeMax;
	size_t worksizeOffset = 0;
	while (worksizeGlobal) {
		const size_t worksizeRun = std::min(worksizeGlobal, worksizeMax);
		const size_t * const pWorksizeLocal = (worksizeLocal == 0 ? NULL : &worksizeLocal);
		const auto res = clEnqueueNDRangeKernel(clQueue, clKernel, 1, &worksizeOffset, &worksizeRun, pWorksizeLocal, 0, NULL, pEvent);
		OpenCLException::throwIfError("kernel queueing failed", res);

		worksizeGlobal -= worksizeRun;
		worksizeOffset += worksizeRun;
	}
}

void Dispatcher::enqueueInverse(Device & d, cl_event * pEvent = NULL) {
	if (m_inverseStrip == 0) {
		enqueueKernelDevice(d, d.m_kernelInverse, m_size / m_inverseSize, pEvent);
	} else {
		enqueueKernel(d.m_clQueue, d.m_kernelInverse, m_size / m_inverseStrip, m_inverseGroup, pEvent);
	}
}

void Dispatcher::enqueueKernelDevice(Device & d, cl_kernel & clKernel, size_t worksizeGlobal, cl_event * pEvent = NULL) {
	try {
		enqueueKernel(d.m_clQueue, clKernel, worksizeGlobal, d.m_worksizeLocal, pEvent);
	} catch ( OpenCLException & e ) {
		// If local work size is invalid, abandon it and let implementation decide
		if ((e.m_res == CL_INVALID_WORK_GROUP_SIZE || e.m_res == CL_INVALID_WORK_ITEM_SIZE) && d.m_worksizeLocal != 0) {
			std::cout << std::endl << "warning: local work size abandoned on GPU" << d.m_index << std::endl;
			d.m_worksizeLocal = 0;
			enqueueKernel(d.m_clQueue, clKernel, worksizeGlobal, d.m_worksizeLocal, pEvent);
		}
		else {
			throw;
		}
	}
}

void Dispatcher::dispatch(Device & d) {
	cl_event event;
	d.m_memResult.read(false, &event);

	if (m_bAppend) {
		// Reset the counter before this round's kernel appends new results. The in-order queue guarantees this write executes after
		// the read above has captured the previous round's results.
		static const cl_uint zero = 0;
		d.m_memResult.writeRegion(false, 0, sizeof(zero), &zero);
	}

	if (m_mode.target == CREATE2) {
		// The read above will bring back the round the previous call enqueued,
		// so where that round's counter started has to be kept for it. Nothing
		// has been enqueued before the first call and the buffer it reads is
		// the cleared one, so what this holds then is never looked at.
		d.m_counterFound = d.m_counterQueued;
		d.m_counterQueued = d.m_round * m_size;

		CLMemory<cl_ulong>::setKernelArg(d.m_kernelIterate, 6, d.m_counterQueued);
	}

#ifdef PROFANITY_DEBUG
	cl_event eventInverse;
	cl_event eventIterate;

	if (m_mode.target != CREATE2) {
		enqueueInverse(d, &eventInverse);
	}
	enqueueKernelDevice(d, d.m_kernelIterate, m_size, &eventIterate);
#else
	if (m_mode.target != CREATE2) {
		enqueueInverse(d);
	}
	enqueueKernelDevice(d, d.m_kernelIterate, m_size);
#endif

	clFlush(d.m_clQueue);

#ifdef PROFANITY_DEBUG
	// We're actually not allowed to call clFinish here because this function is ultimately asynchronously called by OpenCL.
	// However, this happens to work on my computer and it's not really intended for release, just something to aid me in
	// optimizations.
	clFinish(d.m_clQueue);
	if (m_mode.target == CREATE2) {
		std::cout << "Timing: profanity_create2 = " << getKernelExecutionTimeMicros(eventIterate) << "us" << std::endl;
	} else {
		std::cout << "Timing: profanity_inverse = " << getKernelExecutionTimeMicros(eventInverse) << "us, profanity_iterate = " << getKernelExecutionTimeMicros(eventIterate) << "us" << std::endl;
	}
#endif

	const auto res = clSetEventCallback(event, CL_COMPLETE, staticCallback, &d);
	OpenCLException::throwIfError("failed to set custom callback", res);
}

// Under a score floor the kernel appends every hash that cleared it: element [0]
// holds how many the last round found and the entries follow it, each carrying
// its own score, the slot index no longer standing for it. The counter is reset
// before every round (see dispatch()), so everything in the buffer is new.
//
// Nothing here raises the bar, which is the whole point of asking for a floor. A
// run left to raise its own narrows to whatever it has already found: stumble on
// one address scoring better than was asked for and every later address that
// merely satisfies the request goes unreported, however many of them turn up.
// That is right for a search looking for the best it can do, and wrong for one
// that knows what it wants — which is what the floor says.
void Dispatcher::handleFloorResult(Device & d) {
	const cl_uint count = d.m_memResult[0].found;
	if (count == 0) {
		return;
	}

	const cl_uint stored = count < PROFANITY_MAX_SCORE ? count : PROFANITY_MAX_SCORE;

	std::lock_guard<std::mutex> lock(m_mutex);

	for (cl_uint i = 0; i < stored; ++i) {
		result & r = d.m_memResult[i + 1];
		const cl_uchar score = (cl_uchar) r.found;

		if (m_clScoreQuit && score >= m_clScoreQuit) {
			m_quit = true;
		}

		report(d, r, score);
	}

	// Said once and then only counted. A floor loose enough to overrun the
	// buffer overruns it on every round of every device, and the warning
	// repeated hundreds of times a second would bury the addresses it is
	// warning about.
	//
	// Overrunning at all means the floor is far below what the search is worth
	// running for: forty a round, at ten rounds a second on each device, is
	// hundreds of addresses a second already satisfying the request. Whatever
	// is being looked for at that rate would have been found long before the
	// dropped ones mattered.
	if (count > stored && !m_bWarnedFull) {
		m_bWarnedFull = true;

		const std::string strVT100ClearLine = "\33[2K\r";
		std::cout << strVT100ClearLine << "  warning: a round found " << count << " hashes at or above --min-score but only "
			<< PROFANITY_MAX_SCORE << " fit the result buffer, so the rest were dropped. They are arriving faster than "
			<< "there is any use for; raise --min-score to ask for something rarer." << std::endl;
	}
}

// What a finder is handed depends on what was searched: a scalar to add to the
// private key behind their seed, or the salt to deploy with. Both are called
// out by name on the line, so whatever reads it does not have to know which
// search produced it before it can read it.
void Dispatcher::report(Device & d, const result & r, const cl_uchar score) {
	if (m_mode.target == CREATE2) {
		printResult("Salt", saltFor(d.m_salt, d.m_counterFound + r.foundId), r, score, timeStart, m_mode);
		return;
	}

	printResult("Private", privateKeyFor(d.m_clSeed, d.m_round, r), r, score, timeStart, m_mode);
}

void Dispatcher::handleResult(Device & d) {
	if (m_bAppend) {
		handleFloorResult(d);
		return;
	}

	for (auto i = PROFANITY_MAX_SCORE; i > m_clScoreMax; --i) {
		result & r = d.m_memResult[i];

		if (r.found > 0 && i >= d.m_clScoreMax) {
			d.m_clScoreMax = i;
			// The bar the kernel throws away everything under. The two
			// families of kernel take their arguments in different orders,
			// having different numbers of them to take.
			CLMemory<cl_uchar>::setKernelArg(d.m_kernelIterate, m_mode.target == CREATE2 ? 3 : 6, d.m_clScoreMax);

			std::lock_guard<std::mutex> lock(m_mutex);
			if (i >= m_clScoreMax) {
				m_clScoreMax = i;

				if (m_clScoreQuit && i >= m_clScoreQuit) {
					m_quit = true;
				}

				report(d, r, i);
			}

			break;
		}
	}
}

void Dispatcher::onEvent(cl_event event, cl_int status, Device & d) {
	if (status != CL_COMPLETE) {
		std::cout << "Dispatcher::onEvent - Got bad status: " << status << std::endl;
	}
	else if (d.m_eventFinished != NULL) {
		initContinue(d);
	} else {
		++d.m_round;
		handleResult(d);

		bool bDispatch = true;
		{
			std::lock_guard<std::mutex> lock(m_mutex);
			d.m_speed.sample(m_size);
			printSpeed();

			if( m_quit ) {
				bDispatch = false;
				if(--m_countRunning == 0) {
					clSetUserEventStatus(m_eventFinished, CL_COMPLETE);
				}
			}
		}

		if (bDispatch) {
			dispatch(d);
		}
	}
}

// This is run when m_mutex is held.
void Dispatcher::printSpeed() {
	++m_countPrint;
	if( m_countPrint > m_vDevices.size() ) {
		std::string strGPUs;
		double speedTotal = 0;
		unsigned int i = 0;
		for (auto & e : m_vDevices) {
			const auto curSpeed = e->m_speed.getSpeed();
			speedTotal += curSpeed;
			strGPUs += " GPU" + toString(e->m_index) + ": " + formatSpeed(curSpeed);
			++i;
		}

		const std::string strVT100ClearLine = "\33[2K\r";
		std::cerr << strVT100ClearLine << "Total: " << formatSpeed(speedTotal) << " -" << strGPUs << '\r' << std::flush;
		m_countPrint = 0;
	}
}

void CL_CALLBACK Dispatcher::staticCallback(cl_event event, cl_int event_command_exec_status, void * user_data) {
	Device * const pDevice = static_cast<Device *>(user_data);
	pDevice->m_parent.onEvent(event, event_command_exec_status, *pDevice);
	clReleaseEvent(event);
}

std::string Dispatcher::formatSpeed(double f) {
	const std::string S = " KMGT";

	unsigned int index = 0;
	while (f > 1000.0f && index < S.size()) {
		f /= 1000.0f;
		++index;
	}

	std::ostringstream ss;
	ss << std::fixed << std::setprecision(3) << (double)f << " " << S[index] << "H/s";
	return ss.str();
}
