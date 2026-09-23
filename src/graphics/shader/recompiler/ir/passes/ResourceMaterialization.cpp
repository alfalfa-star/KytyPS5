#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include "common/assert.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <functional>
#include <numeric>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr uint64_t AddressMask            = 0x0000ffffffffffffull;
constexpr uint64_t MaxIndirectImageProbes = 65536u;

struct IndirectTable {
	uint32_t                     resource = 0;
	std::vector<uint32_t>        keys;
	std::vector<uint32_t>        candidates;
	std::vector<DescriptorValue> descriptors;
};

struct MaterializedSnapshot {
	ResourceSnapshot           resources;
	std::vector<IndirectTable> indirect_images;
	std::vector<IndirectTable> indirect_buffers;
};

bool SpecializationFail(std::string_view message) {
	std::fprintf(stderr, "shader resource specialization failed: %.*s\n",
	             static_cast<int>(message.size()), message.data());
	return false;
}

Decoder::ImageDimension DescriptorDimension(const DescriptorValue&  descriptor,
                                            Decoder::ImageDimension requested) {
	const bool is_array = requested == Decoder::ImageDimension::Dim1DArray ||
	                      requested == Decoder::ImageDimension::Dim2DArray ||
	                      requested == Decoder::ImageDimension::Dim2DMsaaArray;
	switch (static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu)) {
		case Prospero::ImageType::kColor1D: return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor1DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim1DArray;
			}
			return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor3D: return Decoder::ImageDimension::Dim3D;
		case Prospero::ImageType::kCube: return Decoder::ImageDimension::Dim2DArray;
		case Prospero::ImageType::kColor2DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim2DArray;
			}
			return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaaArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim2DMsaaArray;
			}
			return Decoder::ImageDimension::Dim2DMsaa;
		case Prospero::ImageType::kColor2D: return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaa: return Decoder::ImageDimension::Dim2DMsaa;
		default: return Decoder::ImageDimension::Unknown;
	}
}

bool NullImageDescriptor(const DescriptorValue& descriptor) {
	return descriptor.dwords[0] == 0 && (descriptor.dwords[1] & 0xffu) == 0;
}

bool ValidImageDescriptor(const DescriptorValue& descriptor, bool r128 = false) {
	const auto type   = static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu);
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	if (type < Prospero::ImageType::kColor1D || format == Prospero::BufferFormat::kInvalid ||
	    format > Prospero::BufferFormat::kBc7Srgb) {
		return false;
	}
	if (r128 && type != Prospero::ImageType::kColor1D && type != Prospero::ImageType::kColor2D &&
	    type != Prospero::ImageType::kColor2DMsaa) {
		return false;
	}
	if (type == Prospero::ImageType::kColor2DMsaa ||
	    type == Prospero::ImageType::kColor2DMsaaArray) {
		const auto base_level = (descriptor.dwords[3] >> 12u) & 0xfu;
		const auto fragments  = (descriptor.dwords[3] >> 16u) & 0xfu;
		const auto max_mip    = (descriptor.dwords[5] >> 4u) & 0xfu;
		return base_level == 0 && fragments >= 1 && fragments <= 3 &&
		       (r128 || max_mip == fragments);
	}
	return true;
}

uint32_t DescriptorImageSwizzle(const DescriptorValue& descriptor) {
	return descriptor.dwords[3] & 0xfffu;
}

Prospero::BufferFormat ImageConversionFormat(Prospero::BufferFormat format) {
	return Prospero::RemapTextureFormat(format) != format ? format
	                                                      : Prospero::BufferFormat::kInvalid;
}

bool RequiresPointSampler(const ImageResource& image) {
	return image.numeric_class == Prospero::TextureNumericClass::Sint ||
	       image.conversion_format != Prospero::BufferFormat::kInvalid;
}

bool RequiresPointSampler(const ResourceSpecialization::Image& image) {
	return image.numeric_class == Prospero::TextureNumericClass::Sint ||
	       image.conversion_format != Prospero::BufferFormat::kInvalid;
}

bool DescriptorIsCube(const DescriptorValue& descriptor) {
	return static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu) ==
	       Prospero::ImageType::kCube;
}

uint32_t StorageMipCount(const ImageResource& image, const DescriptorValue& descriptor) {
	if (image.mip_mode != ImageMipMode::DynamicStorage || NullImageDescriptor(descriptor)) {
		return 1;
	}
	const auto base = (descriptor.dwords[3] >> 12u) & 0xfu;
	auto       last = (descriptor.dwords[3] >> 16u) & 0xfu;
	if (!image.r128) {
		// LastLevel may extend past MaxMip (the physically resident mip count): real hardware
		// clamps mip access to whatever is resident rather than touching levels that were never
		// allocated, so mirror that here (matching the clamp ResolveTexture applies to the actual
		// view) instead of asking for more per-level views than the image actually has.
		last = std::min(last, (descriptor.dwords[5] >> 4u) & 0xfu);
	}
	return base <= last ? last - base + 1u : 0u;
}

// Field-level sanity for an enumerated table entry: a descriptor whose base slice or base mip
// lies past its own last one is not something the texture unit could address.
bool CoherentImageDescriptor(const DescriptorValue& descriptor) {
	ShaderTextureResource texture;
	std::copy_n(descriptor.dwords.begin(), std::size(texture.fields), texture.fields);
	return texture.BaseLevel() <= texture.LastLevel() && texture.BaseArray5() <= texture.Depth();
}

bool NullBufferDescriptor(const DescriptorValue& descriptor) {
	return descriptor.dwords[0] == 0 && (descriptor.dwords[1] & 0xffffu) == 0;
}

// A V# table entry the shader could address: a typed buffer resource with a nonzero range.
bool ValidBufferDescriptor(const DescriptorValue& descriptor) {
	ShaderBufferResource buffer;
	if (descriptor.dword_count != std::size(buffer.fields)) {
		return false;
	}
	std::copy_n(descriptor.dwords.begin(), std::size(buffer.fields), buffer.fields);
	return buffer.Type() == 0 && buffer.NumRecords() != 0 && buffer.Base48() != 0;
}

bool DecodeBufferDescriptor(const DescriptorValue& descriptor, ShaderBufferResource& result) {
	if (descriptor.dword_count != std::size(result.fields)) {
		return false;
	}
	std::copy_n(descriptor.dwords.begin(), std::size(result.fields), result.fields);
	return true;
}

const DescriptorSource* Source(const ResourcePlan& program, uint32_t source) {
	if (source >= program.descriptor_sources.size()) {
		return nullptr;
	}
	return &program.descriptor_sources[source];
}

void MarkCleanFlatSlots(const ResourcePlan& program, const DescriptorSource* source,
                        std::span<uint8_t> slots) {
	if (source == nullptr) {
		return;
	}
	std::vector<Value>       pending(source->dwords.begin(),
	                                 source->dwords.begin() + source->dword_count);
	std::vector<const Inst*> visited;
	while (!pending.empty()) {
		auto value = pending.back().Resolve();
		pending.pop_back();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || std::ranges::find(visited, inst) != visited.end()) {
			continue;
		}
		visited.push_back(inst);
		if (inst->GetOpcode() == ValueOpcode::ReadConst) {
			const auto slot = inst->Arg(1).Resolve();
			if (slot.IsImmediate() && slot.GetType() == Type::U32 && slot.U32() < slots.size()) {
				slots[slot.U32()] = 1u;
				pending.push_back(program.srt_reads[slot.U32()].value);
			}
			continue;
		}
		for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
			pending.push_back(inst->Arg(arg));
		}
	}
}

bool ReadSpecializationWord(const SrtRuntime& runtime, uint64_t address, uint32_t& word) {
	return runtime.read_specialization_memory != nullptr &&
	       runtime.read_specialization_memory(runtime.userdata, address, &word);
}

bool ReadScalarBufferWord(const ShaderBufferResource& descriptor, uint32_t dynamic_offset,
                          uint32_t immediate_offset, const SrtRuntime& runtime, uint32_t& word) {
	const auto byte_offset = static_cast<uint64_t>(dynamic_offset) + immediate_offset;
	const auto aligned     = byte_offset & ~uint64_t {3};
	const auto size        = descriptor.GetSize();
	if (aligned > size || size - aligned < sizeof(uint32_t)) {
		word = 0;
		return true;
	}
	const auto base = descriptor.Base48() & ~uint64_t {3};
	if (aligned > AddressMask - base) {
		return false;
	}
	const auto address = base + aligned;
	if (!ReadSpecializationWord(runtime, address, word)) {
		return false;
	}
	return true;
}

bool MaterializeIndirectTable(const DescriptorSource::IndirectTable& indirect,
                              const DescriptorValue&                 material_value,
                              const DescriptorValue& heap_value, bool r128, uint32_t key_limit,
                              const SrtRuntime& runtime, IndirectTable& result) {
	const uint32_t entry_dwords = indirect.entry_dwords;
	const uint32_t entry_stride = indirect.entry_stride;
	const uint32_t min_stride   = indirect.via_pointer ? 8u : entry_dwords * 4u;
	if ((entry_dwords != 8u && entry_dwords != 4u) || entry_stride < min_stride ||
	    (entry_stride % sizeof(uint32_t)) != 0u) {
		return false;
	}
	// The heap is either a V# (bounds-checked like S_BUFFER_LOAD) or a raw S_LOAD base.
	ShaderBufferResource heap;
	uint64_t             heap_base    = 0;
	const bool           heap_address = heap_value.dword_count == 2u;
	if (heap_address) {
		heap_base = ((static_cast<uint64_t>(heap_value.dwords[1]) << 32u) | heap_value.dwords[0]) &
		            AddressMask;
	} else if (!DecodeBufferDescriptor(heap_value, heap)) {
		return false;
	}
	if (indirect.heap_bounded && heap_address) {
		return false;
	}
	// The byte offset of the field the key selects inside each heap record.
	const uint32_t record_offset =
	    indirect.via_pointer ? indirect.pointer_offset : indirect.heap_offset;
	if (indirect.heap_bounded) {
		// Entries at or past the V#'s end read back as zero, i.e. a null descriptor.
		const uint64_t size = heap.GetSize();
		const uint64_t entries =
		    size > record_offset ? (size - record_offset + entry_stride - 1u) / entry_stride : 0u;
		if (entries > indirect.key_count) {
			return SpecializationFail(
			    fmt::format("heap-bounded descriptor table has {} entries (limit {})", entries,
			                indirect.key_count));
		}
		key_limit = std::min<uint64_t>(key_limit, entries);
	}
	const auto ReadHeapWord = [&](uint32_t key, uint32_t dword, uint32_t& word) {
		const auto heap_offset = key * entry_stride;
		if (indirect.via_pointer) {
			uint32_t low  = 0;
			uint32_t high = 0;
			if (heap_address) {
				const auto field =
				    heap_base + static_cast<uint64_t>(heap_offset) + indirect.pointer_offset;
				if (!ReadSpecializationWord(runtime, field, low) ||
				    !ReadSpecializationWord(runtime, field + 4u, high)) {
					low  = 0;
					high = 0;
				}
			} else if (!ReadScalarBufferWord(heap, heap_offset, indirect.pointer_offset, runtime,
			                                 low) ||
			           !ReadScalarBufferWord(heap, heap_offset, indirect.pointer_offset + 4u,
			                                 runtime, high)) {
				return false;
			}
			const uint64_t base     = ((static_cast<uint64_t>(high) << 32u) | low) & AddressMask;
			const uint64_t relative = indirect.heap_offset + dword * sizeof(uint32_t);
			// A record without a pointer, or one to memory the shader cannot have reached, holds
			// no descriptor the shader could select.
			if (base == 0u || relative > AddressMask - base ||
			    !ReadSpecializationWord(runtime, base + relative, word)) {
				word = 0;
			}
			return true;
		}
		const auto immediate = indirect.heap_offset + dword * sizeof(uint32_t);
		if (!heap_address) {
			return ReadScalarBufferWord(heap, heap_offset, immediate, runtime, word);
		}
		const auto relative = static_cast<uint64_t>(heap_offset) + immediate;
		if (relative > AddressMask - heap_base) {
			return false;
		}
		if (ReadSpecializationWord(runtime, heap_base + relative, word)) {
			return true;
		}
		// A bounded key range is an upper bound on the table, not its size: an entry the game
		// never mapped is one the shader never selects, so treat it as a null descriptor.
		word = 0;
		return indirect.key_count != 0u;
	};

	std::vector<uint32_t> keys {0u};
	if (indirect.key_count != 0u) {
		// A loop bound below the static cap trims the enumeration; an empty range still needs
		// one (null) entry so the root resource exists for the code that never runs.
		keys.resize(std::max(std::min(indirect.key_count, key_limit), 1u));
		std::iota(keys.begin(), keys.end(), 0u);
	} else {
		ShaderBufferResource material;
		if (!DecodeBufferDescriptor(material_value, material) ||
		    material.Stride() != indirect.selector_stride) {
			return false;
		}

		// S_BUFFER_LOAD ignores vector-buffer swizzle/add-thread fields. The shader computes the
		// record stride explicitly; enumerate every wrapped 32-bit offset that can pass bounds.
		const auto period      = uint64_t {1} << 32u;
		const auto step        = std::gcd<uint64_t>(indirect.selector_stride, period);
		const auto residue     = static_cast<uint64_t>(indirect.selector_offset) % step;
		const auto size        = material.GetSize();
		const auto limit       = std::min<uint64_t>(UINT32_MAX, size + 3u);
		const auto probe_count = residue <= limit ? (limit - residue) / step + 1u : 0u;
		if (probe_count > MaxIndirectImageProbes) {
			return false;
		}

		std::unordered_set<uint32_t> seen {0u};
		keys.reserve(static_cast<size_t>(probe_count) + 1u);
		seen.reserve(static_cast<size_t>(probe_count) + 1u);
		for (uint64_t offset = residue; offset <= limit && probe_count != 0u; offset += step) {
			uint32_t key = 0;
			if (!ReadScalarBufferWord(material, static_cast<uint32_t>(offset),
			                          indirect.material_offset, runtime, key)) {
				return false;
			}
			if (seen.insert(key).second) {
				keys.push_back(key);
			}
			if (limit - offset < step) {
				break;
			}
		}
	}

	// A sampler table is collapsed to one S# later; keep every distinct entry for that choice.
	const size_t  max_resources = indirect.sampler     ? SIZE_MAX
	                              : entry_dwords == 8u ? ShaderInfo::MaxImages
	                                                   : ShaderInfo::MaxBuffers;
	IndirectTable next;
	next.keys = std::move(keys);
	next.candidates.reserve(next.keys.size());
	next.descriptors.reserve(std::min(next.keys.size(), max_resources));
	// A bounded key range is only an upper bound on the table: the shader never reads past
	// the entries the game filled in, which sit densely from key 0, so the first entry that is
	// not a descriptor ends the table and everything after it is unrelated memory.
	bool table_ended = false;
	for (const auto key: next.keys) {
		DescriptorValue candidate;
		candidate.dword_count = entry_dwords;
		if (table_ended) {
			candidate.dwords.fill(0);
		} else {
			for (uint32_t dword = 0; dword < candidate.dword_count; dword++) {
				if (!ReadHeapWord(key, dword, candidate.dwords[dword])) {
					return false;
				}
			}
		}
		const bool valid =
		    indirect.sampler ? !NullBufferDescriptor(candidate)
		    : entry_dwords == 8u
		        ? !NullImageDescriptor(candidate) && ValidImageDescriptor(candidate, r128) &&
		              (indirect.key_count == 0u || CoherentImageDescriptor(candidate))
		        : !NullBufferDescriptor(candidate) && ValidBufferDescriptor(candidate);
		if (!valid) {
			candidate.dwords.fill(0);
			// A heap-bounded table is sized by the heap itself and may have holes.
			table_ended = table_ended || (indirect.key_count != 0u && !indirect.heap_bounded);
		}
		const auto found = std::ranges::find(next.descriptors, candidate);
		if (found == next.descriptors.end()) {
			if (next.descriptors.size() >= max_resources) {
				return false;
			}
			next.descriptors.push_back(candidate);
			next.candidates.push_back(static_cast<uint32_t>(next.descriptors.size() - 1u));
		} else {
			next.candidates.push_back(static_cast<uint32_t>(found - next.descriptors.begin()));
		}
	}
	result = std::move(next);
	return true;
}

// Reads the table descriptors and the loop bound (when the key is a loop counter) behind an
// indirect source with memory as it is when the shader is bound.
// A select table's candidates are ordinary descriptor sources; key i names candidate i.
bool MaterializeSelectTable(const ResourcePlan&                    program,
                            const DescriptorSource::IndirectTable& indirect, bool r128,
                            const SrtRuntime& runtime, IndirectTable& table) {
	std::vector<DescriptorValue> values;
	if (!EvaluateDescriptorSources(program, indirect.candidate_sources, runtime, values)) {
		return SpecializationFail("select table candidates could not be evaluated");
	}
	IndirectTable next;
	for (uint32_t key = 0; key < values.size(); key++) {
		auto candidate        = values[key];
		candidate.dword_count = indirect.entry_dwords;
		const bool valid =
		    indirect.sampler ? true
		    : indirect.entry_dwords == 8u
		        ? !NullImageDescriptor(candidate) && ValidImageDescriptor(candidate, r128)
		        : !NullBufferDescriptor(candidate) && ValidBufferDescriptor(candidate);
		if (!valid) {
			candidate.dwords.fill(0);
		}
		next.keys.push_back(key);
		const auto found = std::ranges::find(next.descriptors, candidate);
		if (found == next.descriptors.end()) {
			next.descriptors.push_back(candidate);
			next.candidates.push_back(static_cast<uint32_t>(next.descriptors.size() - 1u));
		} else {
			next.candidates.push_back(static_cast<uint32_t>(found - next.descriptors.begin()));
		}
	}
	table = std::move(next);
	return true;
}

bool MaterializeTableSource(const ResourcePlan& program, const DescriptorSource& source, bool r128,
                            const SrtRuntime& runtime, IndirectTable& table) {
	const auto& indirect = *source.indirect_table;
	if (!indirect.candidate_sources.empty()) {
		return MaterializeSelectTable(program, indirect, r128, runtime, table);
	}
	const std::array requests {indirect.material_source, indirect.heap_source};
	SrtRuntime       clean_runtime = runtime;
	clean_runtime.read_memory      = runtime.read_specialization_memory;
	std::vector<DescriptorValue> tables;
	if (!EvaluateDescriptorSources(program, requests, clean_runtime, tables)) {
		return SpecializationFail("indirect table heap could not be evaluated");
	}
	uint32_t key_limit = UINT32_MAX;
	if (!indirect.key_bound.IsEmpty()) {
		const std::array        values {indirect.key_bound};
		std::array<uint32_t, 1> bound {};
		if (!EvaluateUniformValues(program, values, runtime, bound)) {
			return SpecializationFail("indirect table loop bound could not be evaluated");
		}
		key_limit = bound[0];
	}
	if (!MaterializeIndirectTable(indirect, tables[0], tables[1], r128, key_limit, runtime,
	                              table)) {
		return SpecializationFail(fmt::format(
		    "indirect table at heap offset 0x{:x} (stride {}, {} keys, limit {}) could not be read",
		    indirect.heap_offset, indirect.entry_stride, indirect.key_count, key_limit));
	}
	return true;
}

} // namespace

static bool MaterializeSnapshot(const ResourcePlan& program, const SrtRuntime& runtime,
                                MaterializedSnapshot& snapshot) {
	if (!program.resource_tracking_complete) {
		return SpecializationFail("resource tracking is incomplete");
	}

	if (program.requires_specialization_memory && runtime.read_specialization_memory == nullptr) {
		return SpecializationFail("descriptor tables need a specialization memory reader");
	}
	std::vector<DescriptorValue> values;
	std::vector<uint32_t>        flattened_srt;
	std::vector<uint8_t>         active_sources;
	if (!EvaluateRuntimeSources(program, program.materialization_sources, runtime, values,
	                            flattened_srt, program.clean_flat_slots, active_sources)) {
		return SpecializationFail("runtime descriptor sources could not be evaluated");
	}

	auto&                   next  = snapshot.resources;
	const auto&             fill  = program.uniform_fill;
	const auto              words = fill.fill.words;
	std::array<uint32_t, 4> stored {};
	if (words != 0 &&
	    EvaluateUniformValues(program, std::span(fill.values).first(words), runtime,
	                          std::span(stored).first(words)) &&
	    std::all_of(stored.begin(), stored.begin() + words,
	                [&](uint32_t value) { return value == stored[0]; })) {
		next.uniform_fill       = fill.fill;
		next.uniform_fill.value = stored[0];
	}
	auto cursor = values.begin();
	next.buffers.resize(program.info.buffers.size());
	for (uint32_t buffer_index = 0; buffer_index < program.info.buffers.size(); buffer_index++) {
		const auto& buffer = program.info.buffers[buffer_index];
		const auto* source = Source(program, buffer.source);
		if (source == nullptr || !source->indirect_table.has_value()) {
			next.buffers[buffer_index] = *cursor++;
			continue;
		}
		if (!active_sources[buffer.source]) {
			next.buffers[buffer_index].dword_count = 4u;
			continue;
		}
		IndirectTable table;
		if (!MaterializeTableSource(program, *source, false, runtime, table)) {
			return false;
		}
		next.buffers[buffer_index] = table.descriptors[table.candidates[0]];
		if (table.descriptors.size() > 1u) {
			table.resource = buffer_index;
			snapshot.indirect_buffers.push_back(std::move(table));
		}
	}
	next.flattened_srt = std::move(flattened_srt);
	next.images.resize(program.info.images.size());
	for (uint32_t image_index = 0; image_index < program.info.images.size(); image_index++) {
		const auto& image  = program.info.images[image_index];
		const auto* source = Source(program, image.source);
		if (source != nullptr && source->indirect_table.has_value()) {
			if (!active_sources[image.source]) {
				next.images[image_index].dword_count = 8u;
				continue;
			}
			IndirectTable table;
			if (!MaterializeTableSource(program, *source, image.r128, runtime, table)) {
				return false;
			}
			next.images[image_index] = table.descriptors[table.candidates[0]];
			if (table.descriptors.size() > 1u) {
				table.resource = image_index;
				snapshot.indirect_images.push_back(std::move(table));
			}
		} else {
			auto descriptor = *cursor++;
			if (!ValidImageDescriptor(descriptor, image.r128)) {
				descriptor.dwords.fill(0);
			}
			next.images[image_index] = descriptor;
		}
	}
	next.samplers.assign(cursor, cursor + program.info.samplers.size());
	for (uint32_t sampler_index = 0; sampler_index < program.info.samplers.size();
	     sampler_index++) {
		const auto  source_index = program.info.samplers[sampler_index].source;
		const auto* source       = Source(program, source_index);
		if (source == nullptr || !source->indirect_table.has_value()) {
			continue;
		}
		auto& sampler       = next.samplers[sampler_index];
		sampler             = {};
		sampler.dword_count = 4u;
		if (!active_sources[source_index]) {
			continue;
		}
		IndirectTable table;
		if (!MaterializeTableSource(program, *source, false, runtime, table)) {
			return false;
		}
		// Samplers bind statically: use the S# most entries share (material tables usually
		// repeat one), and say so when entries disagree.
		std::vector<uint32_t> votes(table.descriptors.size());
		for (const auto candidate: table.candidates) {
			if (!NullBufferDescriptor(table.descriptors[candidate])) {
				votes[candidate]++;
			}
		}
		const auto best = static_cast<size_t>(std::ranges::max_element(votes) - votes.begin());
		if (votes[best] != 0u) {
			sampler = table.descriptors[best];
		}
		if (std::ranges::count_if(votes, [](uint32_t count) { return count != 0u; }) > 1) {
			std::fprintf(stderr,
			             "shader resource specialization: sampler table has %zu distinct "
			             "samplers; binding the most common one\n",
			             static_cast<size_t>(std::ranges::count_if(
			                 votes, [](uint32_t count) { return count != 0u; })));
		}
	}
	next.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
	return true;
}

struct SamplerPlan {
	std::array<uint32_t, ShaderInfo::MaxSamplers> point_sampler {};
	uint32_t                                      sampler_count = 0;
};

struct ImageRemap {
	explicit ImageRemap(const ResourceSpecialization& specialization) {
		for (const auto& image: specialization.images) {
			indices.push_back(image.fmask ? UINT32_MAX : count++);
		}
	}

	template <typename T>
	void Apply(std::vector<T>& images) const {
		EXIT_IF(images.size() != indices.size());
		for (uint32_t index = 0; index < indices.size(); index++) {
			if (indices[index] != UINT32_MAX && indices[index] != index) {
				images[indices[index]] = std::move(images[index]);
			}
		}
		images.resize(count);
	}

	std::vector<uint32_t> indices;
	uint32_t              count = 0;
};

template <typename Images>
bool BuildSamplerPlan(const ShaderInfo& base, const Images& images, SamplerPlan& plan);

static bool BuildResourceSpecialization(const ResourcePlan& program, MaterializedSnapshot snapshot,
                                        ResourceSnapshot&       specialized_snapshot,
                                        ResourceSpecialization& specialization) {
	auto                   next_snapshot = std::move(snapshot.resources);
	ResourceSpecialization next_specialization;
	next_specialization.buffers.reserve(program.info.buffers.size());
	size_t image_count   = program.info.images.size();
	size_t mapping_words = 0;
	for (const auto& table: snapshot.indirect_images) {
		if (table.resource >= program.info.images.size() || table.descriptors.size() < 2u ||
		    image_count + table.descriptors.size() - 1u > ShaderInfo::MaxImages) {
			size_t total = program.info.images.size();
			for (const auto& other: snapshot.indirect_images) {
				total += other.descriptors.size() - 1u;
			}
			return SpecializationFail(fmt::format(
			    "indirect image candidates exceed the dense image resource limit ({} tables, "
			    "{} images needed in total, table has {} keys / {} candidates)",
			    snapshot.indirect_images.size(), total, table.keys.size(),
			    table.descriptors.size()));
		}
		image_count += table.descriptors.size() - 1u;
		mapping_words += 1u + table.keys.size() * 2u;
	}
	next_snapshot.images.reserve(image_count);
	next_snapshot.flattened_srt.reserve(next_snapshot.flattened_srt.size() + mapping_words);
	next_specialization.images.reserve(image_count);
	for (const auto& image: program.info.images) {
		next_specialization.images.push_back({
		    .numeric_class              = image.numeric_class,
		    .dimension                  = image.dimension,
		    .mip_count                  = image.mip_count,
		    .conversion_format          = image.conversion_format,
		    .shader_swizzle             = image.shader_swizzle,
		    .indirect_root              = image.indirect_root,
		    .indirect_mapping_offset    = image.indirect_mapping_offset,
		    .indirect_search_iterations = image.indirect_search_iterations,
		    .cube                       = image.cube,
		});
	}
	for (const auto& table: snapshot.indirect_images) {
		const auto root_image = next_specialization.images[table.resource];
		for (uint32_t candidate = 1; candidate < table.descriptors.size(); candidate++) {
			auto image          = root_image;
			image.indirect_root = table.resource;
			next_specialization.images.push_back(image);
			next_snapshot.images.push_back(table.descriptors[candidate]);
		}
		auto& root                      = next_specialization.images[table.resource];
		root.indirect_root              = table.resource;
		root.indirect_mapping_offset    = static_cast<uint32_t>(next_snapshot.flattened_srt.size());
		root.indirect_search_iterations = std::bit_width(table.keys.size());
		next_snapshot.flattened_srt.resize(next_snapshot.flattened_srt.size() + 1u +
		                                   table.keys.size() * 2u);
		std::vector<uint32_t> order(table.keys.size());
		std::iota(order.begin(), order.end(), 0u);
		std::ranges::sort(order, {}, [&](uint32_t index) { return table.keys[index]; });
		next_snapshot.flattened_srt[root.indirect_mapping_offset] =
		    static_cast<uint32_t>(table.keys.size());
		for (uint32_t entry = 0; entry < order.size(); entry++) {
			const auto source                   = order[entry];
			const auto offset                   = root.indirect_mapping_offset + 1u + entry * 2u;
			next_snapshot.flattened_srt[offset] = table.keys[source];
			next_snapshot.flattened_srt[offset + 1] = table.candidates[source];
		}
		next_snapshot.images[table.resource] = table.descriptors[0];
	}
	// Buffer tables append their extra candidates after the shader's own buffers, so the dense
	// indices resource tracking assigned stay valid.
	size_t buffer_count         = program.info.buffers.size();
	size_t buffer_mapping_words = 0;
	for (const auto& table: snapshot.indirect_buffers) {
		if (table.resource >= program.info.buffers.size() || table.descriptors.size() < 2u ||
		    buffer_count + table.descriptors.size() - 1u > ShaderInfo::MaxBuffers) {
			return SpecializationFail(fmt::format(
			    "indirect buffer candidates exceed the dense buffer resource limit ({} tables, "
			    "table at pc 0x{:x} has {} keys / {} candidates, {} buffers so far)",
			    snapshot.indirect_buffers.size(),
			    table.resource < program.info.buffers.size()
			        ? program.info.buffers[table.resource].first_use_pc
			        : 0u,
			    table.keys.size(), table.descriptors.size(), buffer_count));
		}
		buffer_count += table.descriptors.size() - 1u;
		buffer_mapping_words += 1u + table.keys.size() * 2u;
	}
	next_snapshot.buffers.reserve(buffer_count);
	next_snapshot.flattened_srt.reserve(next_snapshot.flattened_srt.size() + buffer_mapping_words);
	next_specialization.buffers.reserve(buffer_count);
	const auto specialize_buffer = [&](DescriptorValue& descriptor_value, uint32_t base_index,
	                                   ResourceSpecialization::Buffer& result) {
		ShaderBufferResource descriptor;
		if (!DecodeBufferDescriptor(descriptor_value, descriptor)) {
			return false;
		}
		if (descriptor.Type() != 0) {
			descriptor_value.dwords.fill(0);
			descriptor = {};
		}
		auto       packed_stride = descriptor.PackedStride();
		const auto stride        = packed_stride & 0x3fffu;
		const bool swizzle       = stride != 0u && ((packed_stride >> 14u) & 1u) != 0u;
		if (stride == 0u) {
			packed_stride &= ~((1u << 14u) | (3u << 16u));
		} else if (!swizzle) {
			packed_stride &= ~(3u << 16u);
		}
		const bool formatted = program.info.buffers[base_index].formatted;
		result               = {
		    .packed_stride     = packed_stride,
		    .descriptor_format = formatted ? descriptor.Format() : Prospero::BufferFormat::kInvalid,
		    .descriptor_swizzle = formatted ? descriptor.DstSelXYZW() : DstSel(4, 5, 6, 7),
		};
		return true;
	};
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		ResourceSpecialization::Buffer buffer;
		if (!specialize_buffer(next_snapshot.buffers[i], i, buffer)) {
			return SpecializationFail(fmt::format("buffer descriptor {} has invalid width", i));
		}
		next_specialization.buffers.push_back(buffer);
	}
	for (const auto& table: snapshot.indirect_buffers) {
		auto& root                      = next_specialization.buffers[table.resource];
		root.indirect_root              = table.resource;
		root.indirect_mapping_offset    = static_cast<uint32_t>(next_snapshot.flattened_srt.size());
		root.indirect_search_iterations = std::bit_width(table.keys.size());
		next_snapshot.flattened_srt.resize(next_snapshot.flattened_srt.size() + 1u +
		                                   table.keys.size() * 2u);
		std::vector<uint32_t> order(table.keys.size());
		std::iota(order.begin(), order.end(), 0u);
		std::ranges::sort(order, {}, [&](uint32_t index) { return table.keys[index]; });
		next_snapshot.flattened_srt[root.indirect_mapping_offset] =
		    static_cast<uint32_t>(table.keys.size());
		for (uint32_t entry = 0; entry < order.size(); entry++) {
			const auto source                   = order[entry];
			const auto offset                   = root.indirect_mapping_offset + 1u + entry * 2u;
			next_snapshot.flattened_srt[offset] = table.keys[source];
			next_snapshot.flattened_srt[offset + 1] = table.candidates[source];
		}
		next_snapshot.buffers[table.resource] = table.descriptors[0];
		ResourceSpecialization::Buffer exemplar;
		bool                           typed = false;
		if (!specialize_buffer(next_snapshot.buffers[table.resource], table.resource, exemplar)) {
			return SpecializationFail("indirect buffer root has an invalid descriptor");
		}
		typed = !NullBufferDescriptor(table.descriptors[0]);
		for (uint32_t candidate = 1; candidate < table.descriptors.size(); candidate++) {
			next_snapshot.buffers.push_back(table.descriptors[candidate]);
			ResourceSpecialization::Buffer buffer;
			if (!specialize_buffer(next_snapshot.buffers.back(), table.resource, buffer)) {
				return SpecializationFail("indirect buffer candidate has an invalid descriptor");
			}
			if (!typed && !NullBufferDescriptor(table.descriptors[candidate])) {
				exemplar = buffer;
				typed    = true;
			}
			buffer.indirect_root = table.resource;
			next_specialization.buffers.push_back(buffer);
		}
		// One access sequence serves every entry, so the stride, format and swizzle it was
		// compiled against must hold for each typed candidate; null entries adopt them.
		for (uint32_t index = 0; index < next_specialization.buffers.size(); index++) {
			auto& buffer = next_specialization.buffers[index];
			if (buffer.indirect_root != table.resource) {
				continue;
			}
			if (NullBufferDescriptor(next_snapshot.buffers[index])) {
				buffer.packed_stride      = exemplar.packed_stride;
				buffer.descriptor_format  = exemplar.descriptor_format;
				buffer.descriptor_swizzle = exemplar.descriptor_swizzle;
			} else if (buffer.packed_stride != exemplar.packed_stride ||
			           buffer.descriptor_format != exemplar.descriptor_format ||
			           buffer.descriptor_swizzle != exemplar.descriptor_swizzle) {
				// Tables are enumerated up to a bound, not their exact extent, so an entry the
				// shader never selects can be unrelated memory that merely decodes as a V#. One
				// access sequence serves the whole table: bind such an entry as null.
				std::fprintf(stderr,
				             "shader resource specialization: indirect buffer table at pc 0x%08x "
				             "drops an incompatible candidate (stride 0x%x/0x%x format %u/%u)\n",
				             program.info.buffers[table.resource].first_use_pc,
				             buffer.packed_stride, exemplar.packed_stride,
				             static_cast<uint32_t>(buffer.descriptor_format),
				             static_cast<uint32_t>(exemplar.descriptor_format));
				next_snapshot.buffers[index].dwords.fill(0);
				buffer.packed_stride      = exemplar.packed_stride;
				buffer.descriptor_format  = exemplar.descriptor_format;
				buffer.descriptor_swizzle = exemplar.descriptor_swizzle;
			}
		}
	}
	for (uint32_t i = 0; i < next_specialization.images.size(); i++) {
		const auto& descriptor = next_snapshot.images[i];
		auto&       image      = next_specialization.images[i];
		const auto  base_index = i < program.info.images.size() ? i : image.indirect_root;
		if (base_index >= program.info.images.size()) {
			return SpecializationFail(fmt::format("image resource {} has an invalid root", i));
		}
		const auto& base = program.info.images[base_index];
		if (base.resource_class == ImageResourceClass::None ||
		    (base.atomic && base.resource_class != ImageResourceClass::Storage)) {
			return SpecializationFail(fmt::format("image resource {} has an invalid class", i));
		}
		image.mip_count = StorageMipCount(base, descriptor);
		if (image.mip_count == 0u) {
			return SpecializationFail(
			    fmt::format("storage image descriptor {} has an invalid mip range", i));
		}
		if (NullImageDescriptor(descriptor)) {
			image.numeric_class = base.atomic ? Prospero::TextureNumericClass::Uint
			                                  : Prospero::TextureNumericClass::Float;
			image.dimension     = Decoder::ImageDimension::Dim2D;
			image.cube          = false;
			continue;
		}
		const auto descriptor_dimension = DescriptorDimension(descriptor, base.dimension);
		if (descriptor_dimension == Decoder::ImageDimension::Unknown) {
			return SpecializationFail(fmt::format(
			    "image descriptor {} has unsupported type {}: {:08x},{:08x},{:08x},{:08x},"
			    "{:08x},{:08x},{:08x},{:08x}",
			    i, (descriptor.dwords[3] >> 28u) & 0xfu, descriptor.dwords[0], descriptor.dwords[1],
			    descriptor.dwords[2], descriptor.dwords[3], descriptor.dwords[4],
			    descriptor.dwords[5], descriptor.dwords[6], descriptor.dwords[7]));
		}
		image.dimension = descriptor_dimension;
		image.cube      = DescriptorIsCube(descriptor);
		const auto format =
		    static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
		// Image atomics operate on the raw 32-bit texel regardless of the T# format; a float
		// image (IMAGE_ATOMIC_FMIN/FMAX targets) is bound through a uint view for them.
		const bool atomic_float = base.atomic && format == Prospero::BufferFormat::k32Float;
		if (base.atomic && format != Prospero::BufferFormat::k32UInt && !atomic_float) {
			return SpecializationFail(
			    fmt::format("atomic image descriptor {} uses unsupported format {}", i,
			                static_cast<uint32_t>(format)));
		}
		const bool storage = base.resource_class == ImageResourceClass::Storage;
		image.fmask        = Prospero::IsFmaskTextureFormat(format);
		if (image.fmask) {
			if (storage || base.depth_compare ||
			    image.indirect_root != ImageResource::NoIndirectImage ||
			    std::ranges::any_of(program.info.sampled_pairs,
			                        [&](const auto& pair) { return pair.image == i; })) {
				return SpecializationFail("FMASK requires a direct image load");
			}
		}
		image.conversion_format = ImageConversionFormat(format);
		if (storage || image.conversion_format != Prospero::BufferFormat::kInvalid) {
			image.shader_swizzle = DescriptorImageSwizzle(descriptor);
		}
		image.numeric_class = Prospero::SampledTextureNumericClass(format);
		// Write-only signed-integer storage goes through a uint view of the same texel width:
		// the store keeps the low bits, which is exactly the two's-complement pattern.
		const bool raw_sint_storage = storage &&
		                              image.numeric_class == Prospero::TextureNumericClass::Sint &&
		                              base.written && !base.read && !base.atomic;
		if (storage) {
			if ((!raw_sint_storage && image.numeric_class == Prospero::TextureNumericClass::Sint) ||
			    image.numeric_class == Prospero::TextureNumericClass::Unsupported) {
				return SpecializationFail(
				    fmt::format("storage image descriptor {} uses unsupported format {}", i,
				                static_cast<uint32_t>(format)));
			}
			if (raw_sint_storage || atomic_float) {
				image.numeric_class = Prospero::TextureNumericClass::Uint;
			}
		} else if (image.numeric_class == Prospero::TextureNumericClass::Unsupported ||
		           (base.depth_compare &&
		            image.numeric_class != Prospero::TextureNumericClass::Float)) {
			return SpecializationFail(
			    fmt::format("sampled image descriptor {} uses unsupported format {}", i,
			                static_cast<uint32_t>(format)));
		}
	}
	for (uint32_t root_index = 0; root_index < next_specialization.images.size(); root_index++) {
		auto& root = next_specialization.images[root_index];
		if (root.indirect_root != root_index) {
			continue;
		}
		const auto key_count = root.indirect_mapping_offset < next_snapshot.flattened_srt.size()
		                           ? next_snapshot.flattened_srt[root.indirect_mapping_offset]
		                           : 0u;
		if (root.indirect_search_iterations == 0u || key_count < 2u ||
		    static_cast<size_t>(root.indirect_mapping_offset) + 1u +
		            static_cast<size_t>(key_count) * 2u >
		        next_snapshot.flattened_srt.size()) {
			return SpecializationFail("indirect image specialization has an invalid key mapping");
		}
		uint32_t exemplar       = ImageResource::NoIndirectImage;
		uint32_t resource_count = 0;
		for (uint32_t resource = 0; resource < next_specialization.images.size(); resource++) {
			if (next_specialization.images[resource].indirect_root != root_index) {
				continue;
			}
			resource_count++;
			if (exemplar == ImageResource::NoIndirectImage &&
			    !NullImageDescriptor(next_snapshot.images[resource])) {
				exemplar = resource;
			}
		}
		if (resource_count < 2u || exemplar == ImageResource::NoIndirectImage) {
			return SpecializationFail("indirect image specialization has no typed candidate");
		}
		const auto& image_class = next_specialization.images[exemplar];
		for (uint32_t candidate = 0; candidate < next_specialization.images.size(); candidate++) {
			auto& image = next_specialization.images[candidate];
			if (image.indirect_root != root_index) {
				continue;
			}
			// A candidate that is a plain (non-array, non-cube) 2D image is address-compatible
			// with an array/cube exemplar: the sampling instruction that indexes this table is
			// compiled once, with its coordinate shape (and, for a cube exemplar, the
			// direction-to-face-index conversion) fixed by the exemplar alone, and the image view
			// built for a flat descriptor under an array-typed binding is just a 1-layer 2D-array
			// view (see TextureViewInfo). So collapse the dimension/cube gap here rather than
			// failing the whole table when only that axis differs.
			// An entry whose DST_SEL selects constant zero for every channel carries no texel
			// data either way; treat it as the null descriptor it stands in for.
			if (!NullImageDescriptor(next_snapshot.images[candidate]) &&
			    DescriptorImageSwizzle(next_snapshot.images[candidate]) == 0u &&
			    image_class.shader_swizzle != 0u) {
				next_snapshot.images[candidate].dwords.fill(0);
			}
			// A 1D texture in a table the shader samples as 2D: the texture unit follows the
			// T# type and ignores y. RDNA2 1D resources are linear, so the same memory viewed as
			// a height-1 2D (array) image is byte-identical; retype it so it joins the class.
			const bool class_is_2d = image_class.dimension == Decoder::ImageDimension::Dim2D ||
			                         image_class.dimension == Decoder::ImageDimension::Dim2DArray;
			if (class_is_2d && !NullImageDescriptor(next_snapshot.images[candidate]) &&
			    (image.dimension == Decoder::ImageDimension::Dim1D ||
			     image.dimension == Decoder::ImageDimension::Dim1DArray)) {
				auto&      dword3 = next_snapshot.images[candidate].dwords[3];
				const auto type   = static_cast<Prospero::ImageType>((dword3 >> 28u) & 0xfu);
				const auto target = type == Prospero::ImageType::kColor1DArray
				                        ? Prospero::ImageType::kColor2DArray
				                        : Prospero::ImageType::kColor2D;
				dword3            = (dword3 & 0x0fffffffu) | (static_cast<uint32_t>(target) << 28u);
				image.dimension   = target == Prospero::ImageType::kColor2DArray
				                        ? Decoder::ImageDimension::Dim2DArray
				                        : Decoder::ImageDimension::Dim2D;
			}
			const bool image_is_2d_family = image.dimension == Decoder::ImageDimension::Dim2D ||
			                                image.dimension == Decoder::ImageDimension::Dim2DArray;
			const bool class_is_2d_family =
			    image_class.dimension == Decoder::ImageDimension::Dim2D ||
			    image_class.dimension == Decoder::ImageDimension::Dim2DArray;
			if (NullImageDescriptor(next_snapshot.images[candidate]) ||
			    (image_is_2d_family && class_is_2d_family &&
			     image.numeric_class == image_class.numeric_class &&
			     image.mip_count == image_class.mip_count &&
			     image.conversion_format == image_class.conversion_format &&
			     image.shader_swizzle == image_class.shader_swizzle)) {
				image.numeric_class     = image_class.numeric_class;
				image.dimension         = image_class.dimension;
				image.mip_count         = image_class.mip_count;
				image.conversion_format = image_class.conversion_format;
				image.shader_swizzle    = image_class.shader_swizzle;
				image.cube              = image_class.cube;
			}
			if (image.numeric_class != image_class.numeric_class ||
			    image.dimension != image_class.dimension ||
			    image.mip_count != image_class.mip_count ||
			    image.conversion_format != image_class.conversion_format ||
			    image.shader_swizzle != image_class.shader_swizzle ||
			    image.cube != image_class.cube) {
				return SpecializationFail(fmt::format(
				    "indirect image table at pc 0x{:08x} has incompatible candidates: "
				    "numeric {}/{} dimension {}/{} mips {}/{} conversion {}/{} swizzle "
				    "0x{:x}/0x{:x} cube {}/{}",
				    program.info.images[root_index].first_use_pc,
				    static_cast<uint32_t>(image.numeric_class),
				    static_cast<uint32_t>(image_class.numeric_class),
				    static_cast<uint32_t>(image.dimension),
				    static_cast<uint32_t>(image_class.dimension), image.mip_count,
				    image_class.mip_count, static_cast<uint32_t>(image.conversion_format),
				    static_cast<uint32_t>(image_class.conversion_format), image.shader_swizzle,
				    image_class.shader_swizzle, image.cube, image_class.cube));
			}
		}
	}
	SamplerPlan sampler_plan;
	if (!BuildSamplerPlan(program.info, next_specialization.images, sampler_plan)) {
		return SpecializationFail("specialized sampler layout exceeds its resource limit");
	}
	for (uint32_t index = 0; index < program.info.samplers.size(); index++) {
		const auto target = sampler_plan.point_sampler[index];
		if (target != UINT32_MAX && target >= program.info.samplers.size()) {
			next_snapshot.samplers.push_back(next_snapshot.samplers[index]);
		}
	}
	ImageRemap(next_specialization).Apply(next_snapshot.images);
	specialization       = std::move(next_specialization);
	specialized_snapshot = std::move(next_snapshot);
	return true;
}

template <typename Images>
bool BuildSamplerPlan(const ShaderInfo& base, const Images& images, SamplerPlan& plan) {
	if (base.samplers.size() > plan.point_sampler.size()) {
		return false;
	}
	std::array<uint8_t, ShaderInfo::MaxSamplers> usage {};
	plan.point_sampler.fill(UINT32_MAX);
	plan.sampler_count = static_cast<uint32_t>(base.samplers.size());
	for (const auto& pair: base.sampled_pairs) {
		if (pair.image >= images.size() || pair.sampler >= base.samplers.size()) {
			return false;
		}
		usage[pair.sampler] |= RequiresPointSampler(images[pair.image]) ? 2u : 1u;
	}
	for (uint32_t index = 0; index < base.samplers.size(); index++) {
		if ((usage[index] & 2u) == 0u) {
			continue;
		}
		if ((usage[index] & 1u) == 0u) {
			plan.point_sampler[index] = index;
		} else {
			if (plan.sampler_count >= ShaderInfo::MaxSamplers) {
				return false;
			}
			plan.point_sampler[index] = plan.sampler_count++;
		}
	}
	return true;
}

static std::vector<ResourceBlock> ResourceControlFlow(const Program& program) {
	if (program.blocks.size() != program.block_info.size()) {
		return {};
	}
	// A predicate is only host-evaluable while no shader write can land before the reads
	// behind it (including on a later loop visit); other predicates just keep both successors.
	const ShaderWriteOrder                 write_order(program);
	std::unordered_map<uint32_t, uint32_t> indices;
	for (uint32_t i = 0; i < program.block_info.size(); i++) {
		if (!indices.emplace(program.block_info[i].id, i).second) {
			return {};
		}
	}
	std::vector<ResourceBlock> blocks(program.blocks.size());
	for (uint32_t i = 0; i < blocks.size(); i++) {
		auto&                 block      = blocks[i];
		const auto&           info       = program.block_info[i];
		const auto&           terminator = info.terminator;
		std::vector<uint32_t> successors;
		switch (terminator.kind) {
			case CFG::TerminatorKind::Branch: successors.push_back(terminator.true_block); break;
			case CFG::TerminatorKind::ConditionalBranch:
				successors = {terminator.true_block, terminator.false_block};
				if (ValidateRuntimeValue(program, info.condition, RuntimeValueType::Integer) &&
				    !write_order.WriteMayAffect(info.condition)) {
					block.condition = info.condition;
				}
				break;
			case CFG::TerminatorKind::IndirectBranch:
				successors = terminator.indirect_targets;
				break;
			case CFG::TerminatorKind::Return: break;
			default: return {};
		}
		for (const auto successor: successors) {
			const auto found = indices.find(successor);
			if (found == indices.end()) {
				return {};
			}
			block.successors.push_back(found->second);
		}
		for (const auto& inst: *program.blocks[i]) {
			const auto op     = inst.GetOpcode();
			const auto buffer = BufferAccessOf(op);
			const auto image  = ImageOpcodeInfoOf(op);
			if (buffer == BufferAccess::None && image.access == ImageAccess::None) {
				continue;
			}
			const auto& memory = program.memory_info.at(inst.Flags<MemoryFlags>().index);
			if (memory.planning_only) {
				continue;
			}
			if (buffer != BufferAccess::None) {
				block.sources.push_back(program.info.buffers.at(memory.resource).source);
			} else {
				block.sources.push_back(program.info.images.at(memory.resource).source);
				if (image.needs_sampler) {
					block.sources.push_back(program.info.samplers.at(memory.sampler).source);
				}
			}
		}
		std::ranges::sort(block.sources);
		block.sources.erase(std::unique(block.sources.begin(), block.sources.end()),
		                    block.sources.end());
	}
	if (std::ranges::none_of(
	        blocks, [](const ResourceBlock& block) { return !block.condition.IsEmpty(); })) {
		return {};
	}
	return blocks;
}

// Nonnegative affine coefficients for constant, local and workgroup coordinates. Reject modular
// arithmetic that could wrap; runtime coverage also bounds the largest invocation index.
static std::optional<std::array<uint64_t, 3>> FillIndex(Value value, uint32_t axis,
                                                        uint32_t depth = 0) {
	value = value.Resolve();
	if (depth > 32 || value.GetType() != Type::U32) {
		return {};
	}
	if (value.IsImmediate()) {
		return std::array<uint64_t, 3> {value.U32(), 0, 0};
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return {};
	}
	const auto op = inst->GetOpcode();
	if (op == ValueOpcode::GetBuiltin && inst->Arg(1) == Value(axis)) {
		if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId))) {
			return std::array<uint64_t, 3> {0, 1, 0};
		}
		if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::WorkgroupId))) {
			return std::array<uint64_t, 3> {0, 0, 1};
		}
	}
	if (op != ValueOpcode::IAdd32 && op != ValueOpcode::IMul32 &&
	    op != ValueOpcode::ShiftLeftLogical32) {
		return {};
	}
	auto left  = FillIndex(inst->Arg(0), axis, depth + 1);
	auto right = FillIndex(inst->Arg(1), axis, depth + 1);
	if (!left || !right) {
		return {};
	}
	if (op == ValueOpcode::IMul32 && ((*right)[1] != 0 || (*right)[2] != 0)) {
		std::swap(left, right);
	}
	if (op != ValueOpcode::IAdd32 && ((*right)[1] != 0 || (*right)[2] != 0)) {
		return {};
	}
	if (op == ValueOpcode::ShiftLeftLogical32) {
		if ((*right)[0] >= 32) return {};
		(*right)[0] = uint64_t {1} << (*right)[0];
	}
	for (uint32_t i = 0; i < left->size(); ++i) {
		(*left)[i] =
		    op == ValueOpcode::IAdd32 ? (*left)[i] + (*right)[i] : (*left)[i] * (*right)[0];
		if ((*left)[i] > UINT32_MAX) return {};
	}
	return left;
}

static UniformFillPlan AnalyzeUniformFill(const Program& program) {
	if (program.stage != ShaderType::Compute || program.blocks.empty() ||
	    program.blocks.size() != program.block_info.size() || program.info.uses_dma ||
	    !program.info.samplers.empty()) {
		return {};
	}
	std::unordered_set<uint32_t> visited;
	uint32_t                     index = 0;
	const Inst*                  store = nullptr;
	for (;;) {
		if (!visited.insert(index).second) return {};
		for (const auto& inst: *program.blocks[index]) {
			if (AddressOpcodeInfoOf(inst.GetOpcode()).access != AddressAccess::None) return {};
			if (!inst.MayHaveSideEffects()) continue;
			if (store != nullptr || (BufferAccessOf(inst.GetOpcode()) != BufferAccess::Write &&
			                         inst.GetOpcode() != ValueOpcode::ImageWrite))
				return {};
			store = &inst;
		}
		const auto& term = program.block_info[index].terminator;
		if (term.kind == CFG::TerminatorKind::Return) break;
		if (term.kind != CFG::TerminatorKind::Branch) return {};
		const auto next = std::ranges::find(program.block_info, term.true_block, &BlockInfo::id);
		if (next == program.block_info.end()) return {};
		index = static_cast<uint32_t>(next - program.block_info.begin());
	}
	if (store == nullptr || visited.size() != program.blocks.size()) return {};
	for (const auto& buffer: program.info.buffers) {
		if (buffer.read && (!buffer.scalar || buffer.written)) return {};
	}
	const auto&     memory = program.memory_info.at(store->Flags<MemoryFlags>().index);
	UniformFillPlan result;
	result.fill.resource = memory.resource;
	Value data;
	if (store->GetOpcode() == ValueOpcode::ImageWrite) {
		if (program.info.images.size() != 1 || memory.dmask != 1 || memory.data_bits != 32 ||
		    memory.image_has_mip || memory.image_sample_flags != 0 || memory.image_r128 ||
		    memory.image_dimension != Decoder::ImageDimension::Dim2DArray ||
		    store->Arg(3).Resolve() != Value(true))
			return {};
		const auto& image = program.info.images[memory.resource];
		if (image.read || image.atomic || image.mip_mode != ImageMipMode::None) return {};
		const auto* address = store->Arg(1).ResolveInstruction();
		if (address == nullptr || address->GetOpcode() != ValueOpcode::MakeImageAddress) return {};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const auto index = FillIndex(address->Arg(axis), axis);
			if (!index || (*index)[0] != 0) return {};
			if (axis < 2) {
				if ((*index)[1] != 1 || (*index)[2] == 0) return {};
			} else if ((*index)[1] != 0 || (*index)[2] != 1) {
				return {};
			}
			result.fill.group_stride[axis] = static_cast<uint32_t>((*index)[2]);
		}
		const auto* values = store->Arg(2).ResolveInstruction();
		if (values == nullptr || values->GetOpcode() != ValueOpcode::CompositeConstructU32x4)
			return {};
		result.fill.kind  = UniformFillKind::Image;
		result.fill.words = 1;
		data              = values->Arg(0);
	} else {
		if (!program.info.images.empty()) return {};
		const auto           op = store->GetOpcode();
		constexpr std::array stores {ValueOpcode::StoreBufferU32, ValueOpcode::StoreBufferU32x2,
		                             ValueOpcode::StoreBufferU32x3, ValueOpcode::StoreBufferU32x4};
		const auto           store_op = std::ranges::find(stores, op);
		if (store_op == stores.end() || store->Arg(2).Resolve() != Value(0u) ||
		    store->Arg(3).Resolve() != Value(0u) || store->Arg(5).Resolve() != Value(true))
			return {};
		if (!memory.formatted || memory.typed || !memory.idxen || memory.offen ||
		    memory.offset != 0 || memory.data_bits != 32 ||
		    memory.data_dwords != static_cast<uint32_t>(store_op - stores.begin() + 1))
			return {};
		const auto address = FillIndex(store->Arg(1), 0);
		if (!address || (*address)[0] != 0 || (*address)[1] != 1 || (*address)[2] == 0) return {};
		result.fill.kind            = UniformFillKind::Buffer;
		result.fill.group_stride[0] = static_cast<uint32_t>((*address)[2]);
		result.fill.words           = memory.data_dwords;
		data                        = store->Arg(4);
	}
	data                        = data.Resolve();
	const auto*          vector = data.TryInstruction();
	constexpr std::array composites {ValueOpcode::CompositeConstructU32x2,
	                                 ValueOpcode::CompositeConstructU32x3,
	                                 ValueOpcode::CompositeConstructU32x4};
	if (result.fill.words > 1 &&
	    (vector == nullptr || vector->GetOpcode() != composites[result.fill.words - 2]))
		return {};
	for (uint32_t i = 0; i < result.fill.words; ++i) {
		const auto word = result.fill.words == 1 ? data : vector->Arg(i);
		if (word.GetType() != Type::U32 ||
		    !ValidateRuntimeValue(program, word, RuntimeValueType::Integer))
			return {};
		result.values[i] = word;
	}
	return result;
}

ResourcePlan ExtractResourcePlan(const Program& program) {
	ResourcePlan plan;
	plan.stage                      = program.stage;
	plan.shader_hash                = program.shader_hash;
	plan.user_data_base             = program.user_data_base;
	plan.user_data_count            = program.user_data_count;
	plan.info                       = program.info;
	plan.memory_info                = program.memory_info;
	plan.srt_plan_complete          = program.srt_plan_complete;
	plan.resource_tracking_complete = program.resource_tracking_complete;

	std::unordered_map<const Inst*, Inst*> cloned;
	std::function<Value(Value)>            Clone = [&](Value value) -> Value {
		value              = value.Resolve();
		const auto* source = value.TryInstruction();
		if (source == nullptr) {
			return value;
		}
		if (source->GetOpcode() == ValueOpcode::Phi) {
			const auto invariant = ResolveInvariantPhi(program, value);
			if (!invariant.IsEmpty() && invariant != value) {
				return Clone(invariant);
			}
		}
		if (const auto found = cloned.find(source); found != cloned.end()) {
			return Value(found->second);
		}
		auto& target =
		    plan.value_storage.emplace_back(source->GetOpcode(), source->Flags<uint64_t>());
		cloned.emplace(source, &target);
		if (source->GetOpcode() == ValueOpcode::Phi) {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.AddPhiOperand(nullptr, Clone(source->Arg(index)));
			}
		} else {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.SetArg(index, Clone(source->Arg(index)));
			}
		}
		return Value(&target);
	};

	plan.descriptor_sources.reserve(program.descriptor_sources.size());
	for (const auto& source: program.descriptor_sources) {
		auto& target          = plan.descriptor_sources.emplace_back();
		target.dword_count    = source.dword_count;
		target.indirect_table = source.indirect_table;
		if (target.indirect_table.has_value() && !target.indirect_table->key_bound.IsEmpty()) {
			target.indirect_table->key_bound = Clone(target.indirect_table->key_bound);
		}
		for (uint32_t dword = 0; dword < source.dword_count; dword++) {
			target.dwords[dword] = Clone(source.dwords[dword]);
		}
	}
	plan.srt_reads.reserve(program.srt_reads.size());
	for (const auto& read: program.srt_reads) {
		plan.srt_reads.push_back({Clone(read.value), read.flat_offset});
	}
	plan.control_flow = ResourceControlFlow(program);
	for (auto& block: plan.control_flow) {
		block.condition = Clone(block.condition);
	}
	plan.uniform_fill = AnalyzeUniformFill(program);
	for (uint32_t i = 0; i < plan.uniform_fill.fill.words; ++i) {
		plan.uniform_fill.values[i] = Clone(plan.uniform_fill.values[i]);
	}
	plan.materialization_sources.reserve(plan.info.buffers.size() + plan.info.images.size() +
	                                     plan.info.samplers.size());
	for (const auto& buffer: plan.info.buffers) {
		const auto* source = Source(plan, buffer.source);
		if (source != nullptr && source->indirect_table.has_value()) {
			plan.requires_specialization_memory = true;
		} else {
			plan.materialization_sources.push_back(buffer.source);
		}
	}
	for (const auto& image: plan.info.images) {
		const auto* source = Source(plan, image.source);
		if (source != nullptr && source->indirect_table.has_value()) {
			plan.requires_specialization_memory = true;
		} else {
			plan.materialization_sources.push_back(image.source);
		}
	}
	for (const auto& sampler: plan.info.samplers) {
		// Table samplers still take a cursor slot; MaterializeSnapshot overwrites it.
		plan.materialization_sources.push_back(sampler.source);
		const auto* source = Source(plan, sampler.source);
		if (source != nullptr && source->indirect_table.has_value()) {
			plan.requires_specialization_memory = true;
		}
	}
	plan.clean_flat_slots.resize(plan.srt_reads.size());
	const auto mark_table_slots = [&](uint32_t source_index) {
		const auto* source = Source(plan, source_index);
		if (source == nullptr || !source->indirect_table.has_value() ||
		    !source->indirect_table->candidate_sources.empty()) {
			return;
		}
		MarkCleanFlatSlots(plan, Source(plan, source->indirect_table->material_source),
		                   plan.clean_flat_slots);
		MarkCleanFlatSlots(plan, Source(plan, source->indirect_table->heap_source),
		                   plan.clean_flat_slots);
	};
	for (const auto& buffer: plan.info.buffers) {
		mark_table_slots(buffer.source);
	}
	for (const auto& image: plan.info.images) {
		mark_table_slots(image.source);
	}
	for (const auto& sampler: plan.info.samplers) {
		mark_table_slots(sampler.source);
	}
	return plan;
}

bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	MaterializedSnapshot materialized;
	if (!MaterializeSnapshot(program, runtime, materialized)) {
		return false;
	}
	return BuildResourceSpecialization(program, std::move(materialized), snapshot, specialization);
}

void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization) {
	EXIT_IF(!program.resource_tracking_complete || program.shader_info_complete ||
	        program.binding_layout_complete);
	EXIT_IF(program.info.buffers.size() > specialization.buffers.size() ||
	        program.info.images.size() > specialization.images.size());

	auto buffers = program.info.buffers;
	buffers.reserve(specialization.buffers.size());
	for (size_t index = 0; index < specialization.buffers.size(); index++) {
		const auto& source = specialization.buffers[index];
		if (index >= buffers.size()) {
			EXIT_IF(source.indirect_root >= program.info.buffers.size());
			buffers.push_back(program.info.buffers[source.indirect_root]);
		}
		auto& buffer                      = buffers[index];
		buffer.packed_stride              = source.packed_stride;
		buffer.descriptor_format          = source.descriptor_format;
		buffer.descriptor_swizzle         = source.descriptor_swizzle;
		buffer.indirect_root              = source.indirect_root;
		buffer.indirect_mapping_offset    = source.indirect_mapping_offset;
		buffer.indirect_search_iterations = source.indirect_search_iterations;
		buffer.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < buffers.size(); index++) {
		const auto root = buffers[index].indirect_root;
		if (root != BufferResource::NoIndirectBuffer) {
			EXIT_IF(root >= buffers.size());
			buffers[root].indirect_resources.push_back(index);
		}
	}
	auto images = program.info.images;
	images.reserve(specialization.images.size());
	for (uint32_t index = 0; index < specialization.images.size(); index++) {
		const auto& source = specialization.images[index];
		if (index >= images.size()) {
			EXIT_IF(source.indirect_root >= program.info.images.size());
			images.push_back(program.info.images[source.indirect_root]);
		}
		auto& image                      = images[index];
		image.numeric_class              = source.numeric_class;
		image.dimension                  = source.dimension;
		image.mip_count                  = source.mip_count;
		image.conversion_format          = source.conversion_format;
		image.shader_swizzle             = source.shader_swizzle;
		image.indirect_root              = source.indirect_root;
		image.indirect_mapping_offset    = source.indirect_mapping_offset;
		image.indirect_search_iterations = source.indirect_search_iterations;
		image.cube                       = source.cube;
		image.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < images.size(); index++) {
		const auto root = images[index].indirect_root;
		if (root != ImageResource::NoIndirectImage) {
			EXIT_IF(root >= images.size());
			images[root].indirect_resources.push_back(index);
		}
	}

	SamplerPlan sampler_plan;
	EXIT_IF(!BuildSamplerPlan(program.info, images, sampler_plan));
	auto samplers      = program.info.samplers;
	auto sampled_pairs = program.info.sampled_pairs;
	samplers.reserve(sampler_plan.sampler_count);
	for (uint32_t index = 0; index < program.info.samplers.size(); index++) {
		const auto target = sampler_plan.point_sampler[index];
		if (target == UINT32_MAX) {
			continue;
		}
		if (target == index) {
			samplers[index].force_point_filtering = true;
		} else {
			EXIT_IF(target != samplers.size());
			auto sampler                  = samplers[index];
			sampler.force_point_filtering = true;
			samplers.push_back(std::move(sampler));
		}
	}
	for (auto& pair: sampled_pairs) {
		if (RequiresPointSampler(images[pair.image])) {
			EXIT_IF(sampler_plan.point_sampler[pair.sampler] == UINT32_MAX);
			pair.sampler = sampler_plan.point_sampler[pair.sampler];
		}
		samplers[pair.sampler].depth_compare |= images[pair.image].depth_compare;
	}

	auto             memory_info = program.memory_info;
	const ImageRemap image_remap(specialization);
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto&      inst         = *it;
			const auto image_opcode = ImageOpcodeInfoOf(inst.GetOpcode());
			if (image_opcode.access == ImageAccess::None) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			EXIT_IF(index >= memory_info.size());
			auto& memory = memory_info[index];
			EXIT_IF(memory.resource >= images.size());
			const auto& image = images[memory.resource];
			if (specialization.images[memory.resource].fmask) {
				EXIT_IF(inst.GetOpcode() != ValueOpcode::ImageRead || memory.data_bits != 32u);
				// Vulkan MSAA stores each sample directly; FMASK's four-bit fragment indices
				// therefore map each coverage sample to the same host sample.
				constexpr uint32_t   indices[] = {0x76543210u, 0xfedcba98u};
				std::array<Value, 2> fragments;
				for (uint32_t component = 0; component < fragments.size(); component++) {
					const auto selected =
					    block->PrependNewInst(it, ValueOpcode::SelectU32,
					                          {inst.Arg(2), Value(indices[component]), Value(0u)});
					fragments[component] = Value(&*selected);
				}
				const auto result =
				    block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x4,
				                          {fragments[0], fragments[1], Value(0u), Value(0u)});
				inst.ReplaceUsesWith(Value(&*result));
				continue;
			}
			if (image_opcode.needs_sampler && RequiresPointSampler(image) &&
			    memory.sampler < program.info.samplers.size()) {
				EXIT_IF(sampler_plan.point_sampler[memory.sampler] == UINT32_MAX);
				memory.sampler = sampler_plan.point_sampler[memory.sampler];
			}
			EXIT_IF(image.indirect_root == memory.resource &&
			        inst.GetOpcode() != ValueOpcode::ImageSampleRaw &&
			        inst.GetOpcode() != ValueOpcode::ImageRead &&
			        inst.GetOpcode() != ValueOpcode::ImageWrite);
		}
	}
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() == ValueOpcode::GetImageResource) {
				inst.SetFlags(image_remap.indices.at(inst.Flags<uint32_t>()));
			}
		}
	}
	for (auto& memory: memory_info) {
		if (memory.kind == ResourceKind::Image && !memory.planning_only) {
			memory.resource = image_remap.indices.at(memory.resource);
		}
	}
	for (auto& buffer: buffers) {
		if (buffer.image_alias != BufferResource::NoImageAlias) {
			buffer.image_alias = image_remap.indices.at(buffer.image_alias);
		}
	}
	for (auto& pair: sampled_pairs) {
		pair.image = image_remap.indices.at(pair.image);
	}
	for (auto& image: images) {
		if (image.indirect_root != ImageResource::NoIndirectImage) {
			image.indirect_root = image_remap.indices.at(image.indirect_root);
		}
		for (auto& resource: image.indirect_resources) {
			resource = image_remap.indices.at(resource);
		}
	}
	image_remap.Apply(images);
	program.info.buffers       = std::move(buffers);
	program.info.images        = std::move(images);
	program.info.samplers      = std::move(samplers);
	program.info.sampled_pairs = std::move(sampled_pairs);
	program.memory_info        = std::move(memory_info);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
