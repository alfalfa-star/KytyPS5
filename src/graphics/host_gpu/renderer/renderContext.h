#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/samplerCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessHeap.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "kernel/eventQueue.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <thread>
#include <vector>

namespace Libs::VideoOut {
class VideoOutDriver;
}

namespace Libs::Graphics {

class GuestGpu;

class RenderContext {
public:
	explicit RenderContext(GraphicContext& graphics);
	~RenderContext();
	KYTY_CLASS_NO_COPY(RenderContext);

	[[nodiscard]] GraphicContext&           GetGraphics() const noexcept { return m_graphics; }
	void                                    InitializeGpu(VideoOut::VideoOutDriver* video_out);
	void                                    ShutdownGpu();
	[[nodiscard]] GuestGpu&                 GetGpu() const;
	[[nodiscard]] VideoOut::VideoOutDriver& GetVideoOut() const;

	Common::Mutex& GetMutex() { return m_mutex; }
	// Presentation takes the renderer mutex between guest GPU commands. The GPU thread re-takes
	// it for every draw, so it steps aside at command boundaries while a present is waiting
	// (YieldToPresent) instead of starving it.
	void LockForPresent() {
		m_present_waiters.fetch_add(1, std::memory_order_acq_rel);
		m_mutex.Lock();
		m_present_waiters.fetch_sub(1, std::memory_order_acq_rel);
	}
	void YieldToPresent() const {
		while (m_present_waiters.load(std::memory_order_acquire) != 0) {
			std::this_thread::yield();
		}
	}
	CommandScheduler& GetCommandScheduler() { return m_command_scheduler; }
	PipelineCache&    GetPipelineCache() { return m_pipeline_cache; }
	DescriptorHeap&   GetDescriptorHeap() { return m_descriptor_heap; }
	SamplerCache&     GetSamplerCache() { return m_sampler_cache; }
	BufferCache&      GetBufferCache() { return m_buffer_cache; }
	TextureCache&     GetTextureCache() { return m_texture_cache; }
	BindlessHeap&     GetBindlessHeap() { return m_bindless_heap; }
	RenderExecutor&   GetRenderExecutor() { return m_render_executor; }

	[[nodiscard]] bool HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept;
	[[nodiscard]] bool InvalidateMemory(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsMapped(uint64_t vaddr, uint64_t size) const noexcept;
	void               MapMemory(uint64_t vaddr, uint64_t size);
	void               UnmapMemory(uint64_t vaddr, uint64_t size);
	// Write-watches guest memory cached descriptor tables were read from; the first CPU write
	// to a watched page drops its watch and invalidates those tables. False when the range is
	// not GPU memory.
	[[nodiscard]] bool WatchDescriptorTableMemory(uint64_t vaddr, uint64_t size);
	void               PrepareBda();
	void               RunGarbageCollector();

	void AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void TriggerInterrupt(int event_id, uint32_t context_id);

private:
	struct InterruptEqRegistration {
		LibKernel::EventQueue::KernelEqueue eq       = LibKernel::EventQueue::KERNEL_EQUEUE_INVALID;
		int                                 event_id = 0;
	};

	GraphicContext&           m_graphics;
	Common::Mutex             m_mutex;
	RenderExecutor            m_render_executor;
	CommandScheduler          m_command_scheduler;
	DescriptorHeap            m_descriptor_heap;
	PipelineCache             m_pipeline_cache;
	SamplerCache              m_sampler_cache;
	PageManager               m_page_manager;
	BufferCache               m_buffer_cache;
	TextureCache              m_texture_cache;
	BindlessHeap              m_bindless_heap;
	mutable std::shared_mutex m_mapped_ranges_mutex;
	RangeSet                  m_mapped_ranges;
	std::unique_ptr<GuestGpu> m_gpu;
	VideoOut::VideoOutDriver* m_video_out             = nullptr;
	bool                      m_fault_process_pending = false;

	// Releases descriptor-table watches in [vaddr, vaddr + size).
	void ReleaseDescriptorTableWatches(uint64_t vaddr, uint64_t size);

	std::atomic<uint32_t> m_present_waiters {0};

	std::mutex         m_table_watch_mutex;
	std::set<uint64_t> m_table_watch_pages;

	Common::Mutex                        m_interrupt_mutex;
	std::vector<InterruptEqRegistration> m_interrupt_eqs;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
