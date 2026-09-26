#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BINDLESSHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BINDLESSHEAP_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class Buffer;

// Descriptor set 1 of every pipeline: textures and samplers shaders select at runtime by their
// T#/S# (ShaderRecompiler::IR::Bindless). Shaders look descriptors up in `table`; the ones they
// do not find are logged in `feedback` for the renderer to register (RenderExecutor).
class BindlessHeap {
public:
	struct Key {
		uint32_t                kind = 0;
		std::array<uint32_t, 8> dwords {};

		bool operator==(const Key& other) const = default;
	};

	struct Entry {
		uint32_t      slot = 0;
		ImageId       image;
		vk::ImageView view    = nullptr;
		vk::Sampler   sampler = nullptr;
	};

	BindlessHeap(GraphicContext& graphics, CommandScheduler& scheduler);
	~BindlessHeap();
	KYTY_CLASS_NO_COPY(BindlessHeap);

	[[nodiscard]] vk::DescriptorSetLayout Layout() const noexcept { return m_layout; }
	[[nodiscard]] vk::DescriptorSet       Set() const noexcept { return m_set; }

	// Descriptors shaders did not find since the last call.
	[[nodiscard]] std::vector<Key> TakeMisses();

	[[nodiscard]] Entry* Find(const Key& key);
	// Descriptors the renderer could not turn into a texture; never retried.
	void               Reject(const Key& key) { m_rejected.insert(key); }
	[[nodiscard]] bool Rejected(const Key& key) const { return m_rejected.contains(key); }

	// Publishes a descriptor to shaders. Returns nullptr when its array or the table is full.
	Entry* RegisterImage(const Key& key, ImageId image, vk::ImageView view);
	Entry* RegisterSampler(const Key& key, vk::Sampler sampler);
	void   UpdateImage(Entry& entry, ImageId image, vk::ImageView view);
	// Slot 0: what a lookup miss samples.
	void               SetNullImage(uint32_t kind, vk::ImageView view);
	void               SetNullSampler(vk::Sampler sampler);
	[[nodiscard]] bool HasNullDescriptors() const noexcept { return m_null_ready; }
	void               MarkNullDescriptorsReady() noexcept { m_null_ready = true; }

	template <typename Fn>
	void ForEachImage(Fn&& fn) {
		for (auto& [key, entry]: m_entries) {
			if (key.kind != ShaderRecompiler::IR::Bindless::KindSampler) {
				fn(key, entry);
			}
		}
	}

private:
	struct KeyHash {
		size_t operator()(const Key& key) const noexcept;
	};

	[[nodiscard]] bool PublishToTable(const Key& key, uint32_t slot);
	void               WriteImageDescriptor(uint32_t kind, uint32_t slot, vk::ImageView view);
	void               WriteSamplerDescriptor(uint32_t slot, vk::Sampler sampler);

	GraphicContext&                         m_graphics;
	vk::DescriptorSetLayout                 m_layout = nullptr;
	vk::DescriptorPool                      m_pool   = nullptr;
	vk::DescriptorSet                       m_set    = nullptr;
	std::unique_ptr<Buffer>                 m_table;
	std::unique_ptr<Buffer>                 m_feedback;
	std::array<uint32_t, 4>                 m_next_slot {1u, 1u, 1u, 1u}; // per array; 0 is null
	std::unordered_map<Key, Entry, KeyHash> m_entries;
	std::unordered_set<Key, KeyHash>        m_rejected;
	bool                                    m_null_ready = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BINDLESSHEAP_H_
