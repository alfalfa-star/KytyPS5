#include "graphics/host_gpu/renderer/pipeline/bindlessHeap.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <algorithm>
#include <atomic>
#include <cstdio> // PERFTMP
#include <cstring>

namespace Libs::Graphics {

namespace Bindless = ShaderRecompiler::IR::Bindless;

namespace {

constexpr uint64_t TableBytes = uint64_t {Bindless::TableEntries} * Bindless::EntryWords * 4u;
constexpr uint64_t FeedbackBytes =
    (1u + uint64_t {Bindless::FeedbackEntries} * Bindless::FeedbackWords) * 4u;

// Index of the image array of an image kind (Images2D/2DArray/3D bindings are 0..2).
uint32_t ImageArray(uint32_t kind) {
	switch (kind) {
		case Bindless::KindImage2D: return Bindless::Images2D;
		case Bindless::KindImage2DArray: return Bindless::Images2DArray;
		case Bindless::KindImage3D: return Bindless::Images3D;
		default: EXIT("bindless heap: kind %u is not an image\n", kind);
	}
}

uint32_t* Words(const Buffer& buffer) {
	return reinterpret_cast<uint32_t*>(buffer.Mapped().data());
}

} // namespace

size_t BindlessHeap::KeyHash::operator()(const Key& key) const noexcept {
	return Bindless::Hash(key.kind, key.dwords.data()) ^
	       (static_cast<size_t>(key.dwords[1]) << 32u);
}

BindlessHeap::BindlessHeap(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics) {
	const auto device = graphics.device;
	const auto images = vk::ShaderStageFlagBits::eAll;

	std::array<vk::DescriptorSetLayoutBinding, Bindless::BindingCount> bindings {};
	std::array<vk::DescriptorBindingFlags, Bindless::BindingCount>     flags {};
	const vk::DescriptorBindingFlags                                   runtime_array =
	    vk::DescriptorBindingFlagBits::ePartiallyBound |
	    vk::DescriptorBindingFlagBits::eUpdateAfterBind |
	    vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
	for (const auto binding: {Bindless::Images2D, Bindless::Images2DArray, Bindless::Images3D}) {
		bindings[binding] = {binding, vk::DescriptorType::eSampledImage, Bindless::ImageSlots,
		                     images};
		flags[binding]    = runtime_array;
	}
	bindings[Bindless::Samplers] = {Bindless::Samplers, vk::DescriptorType::eSampler,
	                                Bindless::SamplerSlots, images};
	flags[Bindless::Samplers]    = runtime_array;
	bindings[Bindless::Table] = {Bindless::Table, vk::DescriptorType::eStorageBuffer, 1u, images};
	bindings[Bindless::Feedback] = {Bindless::Feedback, vk::DescriptorType::eStorageBuffer, 1u,
	                                images};

	vk::DescriptorSetLayoutBindingFlagsCreateInfo binding_flags {};
	binding_flags.bindingCount  = static_cast<uint32_t>(flags.size());
	binding_flags.pBindingFlags = flags.data();
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.pNext        = &binding_flags;
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
	layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
	layout_info.pBindings    = bindings.data();
	EXIT_IF(device.createDescriptorSetLayout(&layout_info, nullptr, &m_layout) !=
	        vk::Result::eSuccess);

	const std::array<vk::DescriptorPoolSize, 3> sizes {{
	    {vk::DescriptorType::eSampledImage, Bindless::ImageSlots * 3u},
	    {vk::DescriptorType::eSampler, Bindless::SamplerSlots},
	    {vk::DescriptorType::eStorageBuffer, 2u},
	}};
	vk::DescriptorPoolCreateInfo                pool_info {};
	pool_info.flags         = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
	pool_info.maxSets       = 1u;
	pool_info.poolSizeCount = static_cast<uint32_t>(sizes.size());
	pool_info.pPoolSizes    = sizes.data();
	EXIT_IF(device.createDescriptorPool(&pool_info, nullptr, &m_pool) != vk::Result::eSuccess);

	vk::DescriptorSetAllocateInfo set_info {};
	set_info.descriptorPool     = m_pool;
	set_info.descriptorSetCount = 1u;
	set_info.pSetLayouts        = &m_layout;
	EXIT_IF(device.allocateDescriptorSets(&set_info, &m_set) != vk::Result::eSuccess);

	m_table =
	    std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, TableBytes);
	m_feedback = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags,
	                                      FeedbackBytes);
	EXIT_IF(m_table->Mapped().size() < TableBytes || m_feedback->Mapped().size() < FeedbackBytes);
	std::memset(m_table->Mapped().data(), 0, TableBytes);
	std::memset(m_feedback->Mapped().data(), 0, FeedbackBytes);
	m_table->Flush(0, TableBytes);
	m_feedback->Flush(0, FeedbackBytes);

	const vk::DescriptorBufferInfo        table {m_table->Handle(), 0, TableBytes};
	const vk::DescriptorBufferInfo        feedback {m_feedback->Handle(), 0, FeedbackBytes};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	writes[0].dstSet          = m_set;
	writes[0].dstBinding      = Bindless::Table;
	writes[0].descriptorCount = 1u;
	writes[0].descriptorType  = vk::DescriptorType::eStorageBuffer;
	writes[0].pBufferInfo     = &table;
	writes[1]                 = writes[0];
	writes[1].dstBinding      = Bindless::Feedback;
	writes[1].pBufferInfo     = &feedback;
	device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	graphics.bindless_set_layout = m_layout;
}

BindlessHeap::~BindlessHeap() {
	m_graphics.bindless_set_layout = nullptr;
	m_graphics.device.destroyDescriptorPool(m_pool, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_layout, nullptr);
}

std::vector<BindlessHeap::Key> BindlessHeap::TakeMisses() {
	std::vector<Key> misses;
	auto*            words = Words(*m_feedback);
	std::atomic_ref  count(words[0]);
	const auto logged = std::min(count.load(std::memory_order_acquire), Bindless::FeedbackEntries);
	if (logged != 0) { // PERFTMP
		static uint32_t reports = 0;
		if (reports++ < 20) {
			std::fprintf(stderr, "PERFTMP bindless feedback count=%u\n", count.load());
		}
	}
	for (uint32_t index = 0; index < logged; index++) {
		auto*           entry = words + 1u + index * Bindless::FeedbackWords;
		std::atomic_ref kind(entry[0]);
		const auto      value = kind.load(std::memory_order_acquire);
		if (value == Bindless::KindEmpty) {
			continue;
		}
		Key key {.kind = value};
		std::memcpy(key.dwords.data(), entry + 2u, sizeof(key.dwords));
		kind.store(Bindless::KindEmpty, std::memory_order_relaxed);
		if (!m_entries.contains(key) && !m_rejected.contains(key) &&
		    std::ranges::find(misses, key) == misses.end()) {
			misses.push_back(key);
		}
	}
	// Shaders that are still running may append behind the reset; they report the same misses
	// again on a later draw.
	count.store(0u, std::memory_order_release);
	return misses;
}

BindlessHeap::Entry* BindlessHeap::Find(const Key& key) {
	const auto found = m_entries.find(key);
	return found == m_entries.end() ? nullptr : &found->second;
}

bool BindlessHeap::PublishToTable(const Key& key, uint32_t slot) {
	auto*      words = Words(*m_table);
	const auto hash  = Bindless::Hash(key.kind, key.dwords.data());
	for (uint32_t probe = 0; probe < Bindless::Probes; probe++) {
		auto* entry =
		    words + ((hash + probe) & (Bindless::TableEntries - 1u)) * Bindless::EntryWords;
		std::atomic_ref kind(entry[0]);
		if (kind.load(std::memory_order_relaxed) != Bindless::KindEmpty) {
			continue;
		}
		entry[1] = slot;
		std::memcpy(entry + 2u, key.dwords.data(), sizeof(key.dwords));
		// Shaders take a nonzero kind as a complete entry.
		kind.store(key.kind, std::memory_order_release);
		return true;
	}
	return false;
}

void BindlessHeap::WriteImageDescriptor(uint32_t kind, uint32_t slot, vk::ImageView view) {
	const vk::DescriptorImageInfo info {nullptr, view, vk::ImageLayout::eGeneral};
	vk::WriteDescriptorSet        write {};
	write.dstSet          = m_set;
	write.dstBinding      = ImageArray(kind);
	write.dstArrayElement = slot;
	write.descriptorCount = 1u;
	write.descriptorType  = vk::DescriptorType::eSampledImage;
	write.pImageInfo      = &info;
	m_graphics.device.updateDescriptorSets(1u, &write, 0, nullptr);
}

void BindlessHeap::WriteSamplerDescriptor(uint32_t slot, vk::Sampler sampler) {
	const vk::DescriptorImageInfo info {sampler, nullptr, vk::ImageLayout::eUndefined};
	vk::WriteDescriptorSet        write {};
	write.dstSet          = m_set;
	write.dstBinding      = Bindless::Samplers;
	write.dstArrayElement = slot;
	write.descriptorCount = 1u;
	write.descriptorType  = vk::DescriptorType::eSampler;
	write.pImageInfo      = &info;
	m_graphics.device.updateDescriptorSets(1u, &write, 0, nullptr);
}

BindlessHeap::Entry* BindlessHeap::RegisterImage(const Key& key, ImageId image,
                                                 vk::ImageView view) {
	auto& next = m_next_slot[ImageArray(key.kind)];
	if (next >= Bindless::ImageSlots) {
		return nullptr;
	}
	const auto slot = next;
	WriteImageDescriptor(key.kind, slot, view);
	if (!PublishToTable(key, slot)) {
		return nullptr;
	}
	next++;
	return &(m_entries[key] = Entry {.slot = slot, .image = image, .view = view});
}

BindlessHeap::Entry* BindlessHeap::RegisterSampler(const Key& key, vk::Sampler sampler) {
	auto& next = m_next_slot[Bindless::Samplers];
	if (next >= Bindless::SamplerSlots) {
		return nullptr;
	}
	const auto slot = next;
	WriteSamplerDescriptor(slot, sampler);
	if (!PublishToTable(key, slot)) {
		return nullptr;
	}
	next++;
	return &(m_entries[key] = Entry {.slot = slot, .sampler = sampler});
}

void BindlessHeap::UpdateImage(Entry& entry, ImageId image, vk::ImageView view) {
	const auto found =
	    std::ranges::find_if(m_entries, [&](const auto& pair) { return &pair.second == &entry; });
	EXIT_IF(found == m_entries.end());
	entry.image = image;
	entry.view  = view;
	WriteImageDescriptor(found->first.kind, entry.slot, view);
}

void BindlessHeap::SetNullImage(uint32_t kind, vk::ImageView view) {
	WriteImageDescriptor(kind, 0u, view);
}

void BindlessHeap::SetNullSampler(vk::Sampler sampler) {
	WriteSamplerDescriptor(0u, sampler);
}

} // namespace Libs::Graphics
