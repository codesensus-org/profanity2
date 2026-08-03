#ifndef HPP_SPEEDSAMPLE
#define HPP_SPEEDSAMPLE
#include <chrono>
#include <list>
#include <utility>

class SpeedSample {
	private:
		typedef std::chrono::time_point<std::chrono::steady_clock> timepoint;

	public:
		SpeedSample(const size_t length);
		~SpeedSample();

		double getSpeed() const;
		void sample(const double V);

	private:
		static timepoint now();

	private:
		const size_t m_length;
		timepoint m_lastTime;
		// What each round did and how long it took, kept apart rather than
		// divided on the spot. See getSpeed for why the division cannot be
		// done per round and then averaged.
		std::list<std::pair<double, double> > m_lRounds;
};

#endif /* HPP_SPEEDSAMPLE */
