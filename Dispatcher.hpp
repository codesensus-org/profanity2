#ifndef HPP_DISPATCHER
#define HPP_DISPATCHER

#include <stdexcept>
#include <fstream>
#include <string>
#include <vector>
#include <mutex>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "SpeedSample.hpp"
#include "CLMemory.hpp"
#include "create2.hpp"
#include "types.hpp"
#include "Mode.hpp"

#define PROFANITY_SPEEDSAMPLES 20
#define PROFANITY_MAX_SCORE 40


class Dispatcher {
	private:
		class OpenCLException : public std::runtime_error {
			public:
				OpenCLException(const std::string s, const cl_int res);

				static void throwIfError(const std::string s, const cl_int res);

				const cl_int m_res;
		};

		struct Device {
			static cl_command_queue createQueue(cl_context & clContext, cl_device_id & clDeviceId);
			static cl_kernel createKernel(cl_program & clProgram, const std::string s);
			static cl_ulong4 createSeed();

			Device(Dispatcher & parent, cl_context & clContext, cl_program & clProgram, cl_device_id clDeviceId, const size_t worksizeLocal, const size_t size, const size_t index, const Mode & mode, cl_ulong4 clSeedX, cl_ulong4 clSeedY, const create2 & clCreate2);
			~Device();

			Dispatcher & m_parent;
			const size_t m_index;

			cl_device_id m_clDeviceId;
			size_t m_worksizeLocal;
			cl_uchar m_clScoreMax;
			cl_command_queue m_clQueue;

			cl_kernel m_kernelInit;
			cl_kernel m_kernelInitUniform;
			cl_kernel m_kernelIterate;
			cl_kernel m_kernelCreate2Init;

			CLMemory<point> m_memPrecomp;
			CLMemory<mp_number> m_memPointsDeltaX;
			CLMemory<mp_number> m_memPrevLambda;
			CLMemory<result> m_memResult;

			// The starting point every work item shares. Written once by
			// profanity_init_uniform and read by every profanity_init work item;
			// the GPU is the only side that ever touches it.
			CLMemory<point> m_memUniform;

			// Data parameters used in some modes
			CLMemory<cl_uchar> m_memData1;
			CLMemory<cl_uchar> m_memData2;

			// The CREATE2 preimage every work item hashes but for the eight
			// bytes of salt it writes over, as the words the kernel copies in.
			CLMemory<cl_uint> m_memCreate2;

			// Seed and round information
			cl_ulong4 m_clSeed;
			cl_ulong4 m_clSeedX;
			cl_ulong4 m_clSeedY;
			cl_ulong m_round;

			// The salt of a CREATE2 search, whose last eight bytes stand for
			// whichever counter is being spoken about. The four before them are
			// a nonce drawn per device: every device counts from zero, so it is
			// the only thing keeping two of them off the same salts.
			cl_uchar m_salt[32];

			// Which counters a round searched. A round's results are not read
			// back until the dispatch after the one that enqueued it, so where
			// it started has to outlive the round: m_counterQueued is the round
			// being enqueued now, m_counterFound the one whose results the read
			// in flight will bring back.
			cl_ulong m_counterQueued;
			cl_ulong m_counterFound;

			// Speed sampling
			SpeedSample m_speed;

			// Initialization
			size_t m_sizeInitialized;
			cl_event m_eventFinished;
		};

	public:
		Dispatcher(cl_context & clContext, cl_program & clProgram, const Mode mode, const size_t worksizeMax, const size_t inverseSize, const size_t inverseMultiple, const size_t inverseStrip, const size_t inverseGroup, const cl_uchar clScoreMin, const cl_uchar clScoreQuit, const std::string & seedPublicKey, const create2 & clCreate2, const size_t variants, const size_t rounds);
		~Dispatcher();

		void addDevice(cl_device_id clDeviceId, const size_t worksizeLocal, const size_t index);
		void run();

	private:
		void init();
		void initBegin(Device & d);
		void initContinue(Device & d);
		void initCreate2(Device & d);

		void dispatch(Device & d);
		void enqueueKernel(cl_command_queue & clQueue, cl_kernel & clKernel, size_t worksizeGlobal, const size_t worksizeLocal, cl_event * pEvent);
		void enqueueKernelDevice(Device & d, cl_kernel & clKernel, size_t worksizeGlobal, cl_event * pEvent);
		void enqueueIterate(Device & d, cl_event * pEvent);

		void handleResult(Device & d);
		void handleFloorResult(Device & d);
		void report(Device & d, const result & r, const cl_uchar score);
		void randomizeSeed(Device & d);

		void onEvent(cl_event event, cl_int status, Device & d);

		void printSpeed();

	private:
		static void CL_CALLBACK staticCallback(cl_event event, cl_int event_command_exec_status, void * user_data);

		static std::string formatSpeed(double s);

	private: /* Instance variables */
		cl_context & m_clContext;
		cl_program & m_clProgram;
		const Mode m_mode;
		const size_t m_worksizeMax;
		const size_t m_inverseSize;
		const size_t m_inverseStrip;
		const size_t m_inverseGroup;
		const size_t m_size;

		// How many addresses a work item grades, which is how many a round is
		// worth however many points went into it. The speed a run reports is in
		// addresses and not in points, since it is the addresses that a search
		// gets through and the points are an implementation detail of reaching
		// them. One for a CREATE2 search, which hashes salts and not points.
		const size_t m_variants;

		// How many point additions a launch does per point. The state a point
		// carries between them stays in the kernel, so this divides both the
		// global memory traffic that state costs and the launches a search
		// makes. One for a CREATE2 search, which has no point to advance.
		const size_t m_rounds;

		cl_uchar m_clScoreMax;
		const cl_uchar m_clScoreMin;
		cl_uchar m_clScoreQuit;

		// Whether the kernels append every hash that clears the bar, rather than
		// keeping the best of a round. A score floor is the only thing that asks
		// for it, but the buffer flags and the per-round counter reset both turn
		// on it, so it is worth a name of its own.
		const bool m_bAppend;
		bool m_bWarnedFull;

		std::vector<Device *> m_vDevices;

		cl_event m_eventFinished;

		// Run information
		std::mutex m_mutex;
		std::chrono::time_point<std::chrono::steady_clock> timeStart;
		unsigned int m_countPrint;
		unsigned int m_countRunning;
		size_t m_sizeInitTotal;
		size_t m_sizeInitDone;
		bool m_quit;
		cl_ulong4 m_publicKeyX;
		cl_ulong4 m_publicKeyY;
		const create2 m_create2;
};

#endif /* HPP_DISPATCHER */
