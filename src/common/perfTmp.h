// PERFTMP: temporary per-site wall-clock accumulators; remove before committing.
#ifndef KYTY_COMMON_PERFTMP_H_
#define KYTY_COMMON_PERFTMP_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace PerfTmp {

inline thread_local uint64_t fault_rip = 0; // guest RIP of the fault being handled
// Signal-safe record of read-fault addresses (no allocation): written in the fault handler,
// summarized by DumpReadFaults on another thread.
inline std::array<std::atomic<uint64_t>, 8192> read_fault_ring {};
inline std::atomic<uint64_t>                   read_fault_count {0};
inline void                                    RecordReadFault(uint64_t vaddr) {
	read_fault_ring[read_fault_count.fetch_add(1, std::memory_order_relaxed) & 8191u].store(
	    vaddr, std::memory_order_relaxed);
}

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

inline void DumpReadFaults() {
	static uint64_t              last = 0;
	const auto                   now  = read_fault_count.load(std::memory_order_relaxed);
	const auto                   from = now - last > 8192u ? now - 8192u : last;
	std::map<uint64_t, uint32_t> regions;
	for (auto i = from; i < now; i++) {
		regions[read_fault_ring[i & 8191u].load(std::memory_order_relaxed) >> 16u]++;
	}
	last = now;
	std::vector<std::pair<uint32_t, uint64_t>> sorted;
	for (const auto& [region, count]: regions) {
		sorted.emplace_back(count, region);
	}
	std::sort(sorted.rbegin(), sorted.rend());
	for (size_t n = 0; n < sorted.size() && n < 8; n++) {
		std::fprintf(stderr, "PERFTMP readfault64k region=%llx0000 count=%u\n",
		             static_cast<unsigned long long>(sorted[n].second), sorted[n].first);
	}
}

inline void Dump(double seconds) {
	DumpReadFaults();
	for (auto* s = Head().load(); s != nullptr; s = s->next) {
		const auto ns = s->ns.exchange(0);
		const auto c  = s->count.exchange(0);
		if (c != 0) {
			std::fprintf(
			    stderr, "PERFTMP site %-48s %8.1f ms/s %8.0f calls/s total %.1f ms %llu calls\n",
			    s->name, static_cast<double>(ns) / 1e6 / seconds, static_cast<double>(c) / seconds,
			    static_cast<double>(ns) / 1e6, static_cast<unsigned long long>(c));
		}
	}
}

inline std::atomic<uint32_t>& FlipCounter() {
	static std::atomic<uint32_t> flips {0};
	return flips;
}

// KYTY_DUMP_SHADERS=hash,...: shaders whose bound images are dumped on traced flips.
inline bool DumpShader(uint64_t hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> list;
		for (const char* p = std::getenv("KYTY_DUMP_SHADERS"); p != nullptr && *p != 0;) {
			list.push_back(std::strtoull(p, nullptr, 16));
			p = std::strchr(p, ',');
			if (p != nullptr) {
				p++;
			}
		}
		return list;
	}();
	return std::find(hashes.begin(), hashes.end(), hash) != hashes.end();
}

// Sequence number for the next dump of this flip, or -1 once the per-flip budget is spent.
inline int NextDumpSeq() {
	static uint32_t flip = UINT32_MAX;
	static int      seq  = 0;
	if (flip != FlipCounter().load()) {
		flip = FlipCounter().load();
		seq  = 0;
	}
	return seq < 300 ? seq++ : -1;
}

inline std::atomic<uint32_t>& TriggeredFrame() {
	static std::atomic<uint32_t> frame {UINT32_MAX};
	return frame;
}

// KYTY_TRACE_TRIGGER=<file>: once the file exists, the next flip is traced and the file removed.
inline void CheckTraceTrigger() {
	static const char* path = std::getenv("KYTY_TRACE_TRIGGER");
	if (path != nullptr && std::remove(path) == 0) {
		TriggeredFrame() = FlipCounter().load() + 1;
		std::fprintf(stderr, "TRACE trigger: tracing flip %u\n", TriggeredFrame().load());
	}
}

// KYTY_TRACE_FRAMES=a,b,...: true while recording the given guest flips.
inline bool TraceFrame() {
	if (FlipCounter().load(std::memory_order_relaxed) ==
	    TriggeredFrame().load(std::memory_order_relaxed)) {
		return true;
	}
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
