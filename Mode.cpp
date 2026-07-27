#include "Mode.hpp"
#include <stdexcept>

Mode::Mode() : score(0) {

}

Mode Mode::benchmark() {
	Mode r;
	r.name = "benchmark";
	r.kernel = "profanity_iterate_score_benchmark";
	return r;
}

Mode Mode::zeros() {
	Mode r = range(0, 0);
	r.name = "zeros";
	return r;
}

static std::string::size_type hexValueNoException(char c) {
	if (c >= 'A' && c <= 'F') {
		c -= 'A' - 'a';
	}

	const std::string hex = "0123456789abcdef";
	const std::string::size_type ret = hex.find(c);
	return ret;
}

static std::string::size_type hexValue(char c) {
	const std::string::size_type ret = hexValueNoException(c);
	if(ret == std::string::npos) {
		throw std::runtime_error("bad hex value");
	}

	return ret;
}

Mode Mode::matching(const std::string strHex) {
	Mode r;
	r.name = "matching";
	r.kernel = "profanity_iterate_score_matching";

	if (strHex.size() > 40) {
		throw std::runtime_error("hex mask must be at most 40 characters, got " + std::to_string(strHex.size()));
	}

	std::fill( r.data1, r.data1 + sizeof(r.data1), cl_uchar(0) );
	std::fill( r.data2, r.data2 + sizeof(r.data2), cl_uchar(0) );

	// The pinned nibbles alone, in the order they were written: data1 holds where
	// in the mask each one sits and data2 holds what it is. Wildcards are left
	// out rather than stored as gaps, so the kernel's inner loop is as long as
	// the pattern the caller actually asked for and not as long as the mask they
	// padded it into — which for an anchored search is most of 40 characters of
	// nothing. The last entry of each array carries what the list itself cannot:
	// how many pinned nibbles there are, and how long the mask was.
	//
	// The length is what decides where the mask may sit. One filling all 40
	// characters has a single placement and stays where it is written; a shorter
	// one is looked for at every offset it could sit at, so padding with
	// wildcards is how a caller anchors a pattern. That distinction is why the
	// length has to be kept even though the padding itself is dropped here.
	cl_uchar pinned = 0;

	for( size_t i = 0; i < strHex.size(); ++i ) {
		const auto index = hexValueNoException(strHex[i]);

		if (index == std::string::npos) {
			continue;
		}

		r.data1[pinned] = static_cast<cl_uchar>(i);
		r.data2[pinned] = static_cast<cl_uchar>(index);
		++pinned;
	}

	r.data1[PROFANITY_MODE_DATA - 1] = pinned;
	r.data2[PROFANITY_MODE_DATA - 1] = static_cast<cl_uchar>(strHex.size());

	return r;
}

Mode Mode::leading(const char charLeading) {

	Mode r;
	r.name = "leading";
	r.kernel = "profanity_iterate_score_leading";
	r.data1[0] = static_cast<cl_uchar>(hexValue(charLeading));
	return r;
}

Mode Mode::range(const cl_uchar min, const cl_uchar max) {
	Mode r;
	r.name = "range";
	// A range of one character is a count of it, and counting is a question the
	// whole address can be asked at once rather than character by character. A
	// range spanning several has to be asked the slow way. --zeros comes through
	// here as range(0, 0) and so takes the quick kernel too.
	r.kernel = (min == max) ? "profanity_iterate_score_rangeequal" : "profanity_iterate_score_range";
	r.data1[0] = min;
	r.data2[0] = max;
	return r;
}

Mode Mode::zeroBytes() {
	Mode r;
	r.name = "zeroBytes";
	r.kernel = "profanity_iterate_score_zerobytes";
	return r;
}

Mode Mode::letters() {
	Mode r = range(10, 15);
	r.name = "letters";
	return r;
}

Mode Mode::numbers() {
	Mode r = range(0, 9);
	r.name = "numbers";
	return r;
}

std::string Mode::transformName() const {
	switch (this->target) {
		case ADDRESS:
			return "Address";
		case CONTRACT:
			return "Contract";
		default:
			throw "No name for target";
	}
}

Mode Mode::leadingRange(const cl_uchar min, const cl_uchar max) {
	Mode r;
	r.name = "leadingrange";
	r.kernel = "profanity_iterate_score_leadingrange";
	r.data1[0] = min;
	r.data2[0] = max;
	return r;
}

Mode Mode::mirror() {
	Mode r;
	r.name = "mirror";
	r.kernel = "profanity_iterate_score_mirror";
	return r;
}

Mode Mode::doubles() {
	Mode r;
	r.name = "doubles";
	r.kernel = "profanity_iterate_score_doubles";
	return r;
}
