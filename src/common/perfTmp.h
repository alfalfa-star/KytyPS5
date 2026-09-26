// PERFTMP: temporary per-site wall-clock accumulators; remove before committing.
#ifndef KYTY_COMMON_PERFTMP_H_
#define KYTY_COMMON_PERFTMP_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace PerfTmp {

struct Site {
	const char*           name;
	std::atomic<uint64_t> ns {0};
	std::atomic<uint64_t> count {0};
	Site*                 next = nullptr;
	explicit Site(const char* n);
};

inline std::atomic<Site*>& Head() {
	static std::atomic<Site*> head {nullptr};
	return head;
}

inline Site::Site(const char* n): name(n) {
	next = Head().load();
	while (!Head().compare_exchange_weak(next, this)) {
	}
}

struct Scope {
	Site*                                 site;
	std::chrono::steady_clock::time_point start;

	explicit Scope(Site& s): site(&s), start(std::chrono::steady_clock::now()) {}
	~Scope() {
		site->ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                      std::chrono::steady_clock::now() - start)
		                                      .count());
		site->count++;
	}
};

inline void Dump(double seconds) {
	for (auto* s = Head().load(); s != nullptr; s = s->next) {
		const auto ns = s->ns.exchange(0);
		const auto c  = s->count.exchange(0);
		if (c != 0) {
			std::fprintf(stderr, "PERFTMP site %-48s %8.1f ms/s %8.0f calls/s\n", s->name,
			             static_cast<double>(ns) / 1e6 / seconds, static_cast<double>(c) / seconds);
		}
	}
}

inline std::atomic<uint32_t>& FlipCounter() {
	static std::atomic<uint32_t> flips {0};
	return flips;
}

// KYTY_TRACE_FRAMES=a,b,...: true while recording the given guest flips.
inline bool TraceFrame() {
	static const std::vector<uint32_t> frames = [] {
		std::vector<uint32_t> list;
		const char*           value = std::getenv("KYTY_TRACE_FRAMES");
		for (const char* p = value; p != nullptr && *p != 0;) {
			list.push_back(static_cast<uint32_t>(std::atoi(p)));
			p = std::strchr(p, ',');
			if (p != nullptr) {
				p++;
			}
		}
		return list;
	}();
	if (frames.empty()) {
		return false;
	}
	const auto current = FlipCounter().load(std::memory_order_relaxed);
	for (const auto frame: frames) {
		if (frame == current) {
			return true;
		}
	}
	return false;
}

} // namespace PerfTmp

#define PERFTMP_CAT2(a, b) a##b
#define PERFTMP_CAT(a, b)  PERFTMP_CAT2(a, b)
#define PERFTMP_SCOPE(name)                                                                        \
	static PerfTmp::Site PERFTMP_CAT(perftmp_site_, __LINE__) {name};                              \
	PerfTmp::Scope       PERFTMP_CAT(perftmp_scope_, __LINE__) {                                   \
		PERFTMP_CAT(perftmp_site_, __LINE__)                                                       \
	}

#endif
