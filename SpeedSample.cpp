#include "SpeedSample.hpp"

SpeedSample::SpeedSample(const size_t length) :
	m_length(length),
	m_lastTime(now())
{

}

SpeedSample::~SpeedSample() {

}

// The work of the window divided by the time of the window, which is what
// throughput means.
//
// This used to divide each round by its own duration and then average those
// rates, and that is not the same number: the mean of V/t is not V/mean(t)
// unless every t is equal, and it is always the larger of the two. Rounds do
// not take equal time -- they complete through asynchronous callbacks, so
// several land together and the host times them microseconds apart -- and the
// shorter the rounds, the more uneven they are. The old average therefore read
// high, and it read highest exactly where rounds are shortest, which is to say
// on small launches. Measured through it, an RTX 4090 appeared to reach 10
// GH/s on a two-million point launch and to get faster the smaller the launch
// became, which is backwards: a tuner following that signal walks away from
// the configurations that are genuinely fast.
double SpeedSample::getSpeed() const {
	auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(now() - m_lastTime).count();
	if (delta > 5000) {
		return 0;
	}

	double work = 0;
	double time = 0;

	for (auto & r : m_lRounds) {
		work += r.first;
		time += r.second;
	}

	return time > 0 ? (1000000.0 * work) / time : 0;
}

void SpeedSample::sample(const double V) {
	const timepoint newTime = now();
	const double delta = (double) std::chrono::duration_cast<std::chrono::microseconds>(newTime - m_lastTime).count();
	m_lastTime = newTime;

	// A round the clock could not separate from the one before it carries no
	// information about speed, and dividing by it later would carry a great
	// deal of noise instead.
	if (delta <= 0) {
		return;
	}

	m_lRounds.push_back(std::make_pair(V, delta));
	if (m_lRounds.size() > m_length) {
		m_lRounds.pop_front();
	}
}

SpeedSample::timepoint SpeedSample::now() {
	return std::chrono::steady_clock::now();
}
