#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/Block.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.h"
#include "graphics/shader/shader.h"

#include <array>
#include <bit>
#include <list>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

enum class ResourceKind {
	None,
	ScalarBuffer,
	ScalarAddress,
	Buffer,
	Flat,
	Global,
	Scratch,
	Lds,
	Gds,
	Image,
	Sampler
};

[[nodiscard]] constexpr bool IsAddressResourceKind(ResourceKind kind) {
	return kind == ResourceKind::ScalarAddress || kind == ResourceKind::Flat ||
	       kind == ResourceKind::Global || kind == ResourceKind::Scratch;
}

struct MemoryInfo {
	ResourceKind            kind                     = ResourceKind::None;
	uint32_t                resource                 = 0;
	uint32_t                sampler                  = 0;
	uint32_t                offset                   = 0;
	uint32_t                secondary_offset         = 0;
	uint32_t                dmask                    = 0;
	uint32_t                data_dwords              = 1;
	uint32_t                data_bits                = 32;
	uint32_t                component_index          = 0;
	uint32_t                component_count          = 1;
	uint32_t                data_format              = 0;
	uint32_t                number_format            = 0;
	uint32_t                image_sample_flags       = 0;
	Decoder::ImageDimension image_dimension          = Decoder::ImageDimension::Unknown;
	uint32_t                image_address_components = 0;
	bool                    address_is_full          = false;
	bool                    data_signed              = false;
	bool                    typed                    = false;
	bool                    formatted                = false;
	// A formatted buffer access whose components travel as 16-bit values (f16 for float
	// formats, the low bits for integer ones) instead of full 32-bit dwords.
	bool d16           = false;
	bool image_has_mip = false;
	bool image_r128    = false;
	bool idxen         = false;
	bool offen         = false;
	bool planning_only = false;

	bool operator==(const MemoryInfo& other) const = default;
};

enum class ExportTargetKind { Unknown, Null, Position, Primitive, Parameter, Mrt, MrtZ };

struct ExportInfo {
	ExportTargetKind kind   = ExportTargetKind::Unknown;
	uint32_t         target = 0;
	uint32_t         index  = 0;
	uint32_t         en     = 0;
	bool             done   = false;
	bool             compr  = false;
	bool             vm     = false;

	bool operator==(const ExportInfo& other) const = default;
};

struct BufferResource {
	static constexpr uint32_t NoImageAlias     = UINT32_MAX;
	static constexpr uint32_t NoIndirectBuffer = UINT32_MAX;

	uint32_t               source             = 0;
	uint32_t               first_use_pc       = 0;
	uint32_t               max_byte_extent    = 0;
	uint32_t               packed_stride      = 0;
	Prospero::BufferFormat descriptor_format  = Prospero::BufferFormat::kInvalid;
	uint32_t               descriptor_swizzle = DstSel(4, 5, 6, 7);
	uint32_t               image_alias        = NoImageAlias;
	bool                   read               = false;
	bool                   written            = false;
	bool                   atomic             = false;
	bool                   formatted          = false;
	bool                   scalar             = false;
	// A V# the shader picks out of a descriptor table with a runtime key (see
	// DescriptorSource::IndirectTable) is specialized like an indirect image: the root keeps the
	// key mapping and every table entry the host enumerated becomes its own dense buffer.
	uint32_t              indirect_root              = NoIndirectBuffer;
	uint32_t              indirect_mapping_offset    = 0;
	uint32_t              indirect_search_iterations = 0;
	std::vector<uint32_t> indirect_resources;

	bool operator==(const BufferResource& other) const = default;
};

enum class ImageMipMode { None, DynamicStorage };

constexpr uint32_t ShaderImageIdentitySwizzle = 0x00000facu;

struct ImageResource {
	static constexpr uint32_t NoIndirectImage = UINT32_MAX;

	uint32_t                      source            = 0;
	uint32_t                      first_use_pc      = 0;
	ImageResourceClass            resource_class    = ImageResourceClass::None;
	Prospero::TextureNumericClass numeric_class     = Prospero::TextureNumericClass::Unsupported;
	Decoder::ImageDimension       dimension         = Decoder::ImageDimension::Unknown;
	ImageMipMode                  mip_mode          = ImageMipMode::None;
	uint32_t                      mip_count         = 1;
	Prospero::BufferFormat        conversion_format = Prospero::BufferFormat::kInvalid;
	uint32_t                      shader_swizzle    = ShaderImageIdentitySwizzle;
	bool                          read              = false;
	bool                          written           = false;
	bool                          atomic            = false;
	bool                          depth_compare     = false;
	bool                          cube              = false;
	bool                          r128              = false;
	uint32_t                      indirect_root     = NoIndirectImage;
	uint32_t                      indirect_mapping_offset    = 0;
	uint32_t                      indirect_search_iterations = 0;
	std::vector<uint32_t>         indirect_resources;

	bool operator==(const ImageResource& other) const = default;
};

struct SamplerResource {
	uint32_t source                = 0;
	uint32_t first_use_pc          = 0;
	bool     force_point_filtering = false;
	bool     depth_compare         = false;

	bool operator==(const SamplerResource& other) const = default;
};

struct SampledResourcePair {
	uint32_t image        = 0;
	uint32_t sampler      = 0;
	uint32_t first_use_pc = 0;

	bool operator==(const SampledResourcePair& other) const = default;
};

enum class TessellationAttribute {
	LocalOutput,
	ControlInput,
	ControlOutput,
	EvaluationInput,
	PatchOutput,
	Factor
};

enum class StageInputKind {
	VertexIndex,
	InvocationId,
	PrimitiveId,
	TessCoord,
	InstanceIndex,
	FragCoord,
	FrontFacing,
	PackedAncillary,
	Layer,
	SampleId,
	BaryCoordSmooth,
	BaryCoordSmoothCentroid,
	BaryCoordNoPerspective,
	WorkgroupId,
	LocalInvocationId,
	LocalInvocationIndex,
	GlobalInvocationId,
	Parameter,
};

enum class StageOutputKind {
	Position,
	Parameter,
	Mrt,
	Depth,
	SampleMask,
	PointSize,
	ClipDistance,
	CullDistance,
	Layer,
	ViewportIndex
};

struct PositionExportComponent {
	uint32_t clip_distance = UINT32_MAX;
	uint32_t cull_distance = UINT32_MAX;
	bool     point_size    = false;
	bool     layer         = false;
	bool     viewport      = false;
};

inline PositionExportComponent DecodePositionExportComponent(uint32_t control, uint32_t pos_index,
                                                             uint32_t component) {
	PositionExportComponent result;
	if (pos_index == 0 || component >= 4) {
		return result;
	}

	uint32_t slot   = pos_index - 1;
	uint32_t vector = 3;
	for (uint32_t i = 0; i < 3; i++) {
		if ((control & (1u << (21u + i))) != 0) {
			if (slot == 0) {
				vector = i;
				break;
			}
			slot--;
		}
	}
	if (vector == 3) {
		return result;
	}

	if (vector == 0) {
		result.point_size = component == 0 && (control & (1u << 16u)) != 0;
		result.layer      = component == 2 && (control & (1u << 18u)) != 0;
		result.viewport   = component == 2 && (control & (1u << 19u)) != 0;
		return result;
	}

	const auto scalar = (vector - 1) * 4 + component;
	const auto lower  = (1u << scalar) - 1u;
	const auto clip   = control & 0xffu;
	const auto cull   = (control >> 8u) & 0xffu;
	if ((clip & (1u << scalar)) != 0) {
		result.clip_distance = std::popcount(clip & lower);
	}
	if ((cull & (1u << scalar)) != 0) {
		result.cull_distance = std::popcount(cull & lower);
	}
	return result;
}

struct StageInput {
	StageInputKind kind            = StageInputKind::VertexIndex;
	uint32_t       location        = 0;
	uint32_t       component_count = 1;
	std::string    debug_name;
	bool           per_vertex = false;

	bool operator==(const StageInput& other) const = default;
};

struct StageOutput {
	StageOutputKind kind     = StageOutputKind::Parameter;
	uint32_t        index    = 0;
	uint32_t        location = 0;
	std::string     debug_name;

	bool operator==(const StageOutput& other) const = default;
};

inline constexpr uint32_t FirstImageBinding           = 1u;
inline constexpr uint32_t FirstComparisonImageBinding = 22u;
inline constexpr uint32_t FirstStorageImageBinding    = 29u;
inline constexpr uint32_t ImageBindingCount           = 43u;

enum class DescriptorBindingKind : uint32_t {
	Buffers  = 0u,
	Samplers = FirstImageBinding + ImageBindingCount,
	Gds,
	BdaPagetable,
	FaultBuffer,
	FlattenedSrt,
	ShaderData,
	Count,
};

static_assert(static_cast<uint32_t>(DescriptorBindingKind::Samplers) == 44u);
static_assert(static_cast<uint32_t>(DescriptorBindingKind::Count) == 50u);

struct PushData {
	static constexpr uint32_t        DwordCount         = 32;
	static constexpr uint32_t        MeshDrawDwordCount = 6;
	static constexpr uint32_t        NoStart            = UINT32_MAX;
	std::array<uint32_t, DwordCount> dwords {};

	[[nodiscard]] static constexpr bool CanFit(uint32_t start, uint32_t size) {
		return size != 0 && start <= DwordCount && size <= DwordCount - start;
	}
	[[nodiscard]] static constexpr uint32_t StartFor(uint32_t cursor, uint32_t size) {
		return CanFit(cursor, size) ? cursor : NoStart;
	}
};

static_assert(sizeof(PushData) == 128);
constexpr uint32_t NativePushConstantSize = sizeof(PushData);

[[nodiscard]] constexpr uint32_t NativeBinding(ShaderType stage, DescriptorBindingKind kind) {
	const uint32_t group = stage == ShaderType::Pixel                    ? 1u
	                       : stage == ShaderType::TessellationControl    ? 2u
	                       : stage == ShaderType::TessellationEvaluation ? 3u
	                                                                     : 0u;
	return static_cast<uint32_t>(kind) +
	       group * static_cast<uint32_t>(DescriptorBindingKind::Count);
}

[[nodiscard]] constexpr ImageResourceClass ImageBindingResourceClass(DescriptorBindingKind kind) {
	const auto value = static_cast<uint32_t>(kind);
	if (value >= FirstImageBinding && value < FirstStorageImageBinding) {
		return ImageResourceClass::Sampled;
	}
	if (value >= FirstStorageImageBinding &&
	    value < static_cast<uint32_t>(DescriptorBindingKind::Samplers)) {
		return ImageResourceClass::Storage;
	}
	return ImageResourceClass::None;
}

[[nodiscard]] constexpr uint32_t ImageBindingIndex(DescriptorBindingKind kind) {
	return static_cast<uint32_t>(kind) - FirstImageBinding;
}

[[nodiscard]] constexpr std::optional<DescriptorBindingKind>
DescriptorBindingForImage(const ImageResource& image) {
	constexpr uint32_t SampledFloatBinding = 1u;
	constexpr uint32_t SampledUintBinding  = 8u;
	constexpr uint32_t SampledSintBinding  = 15u;
	constexpr uint32_t StorageFloatBinding = FirstStorageImageBinding;
	constexpr uint32_t StorageUintBinding  = StorageFloatBinding + 5u;
	constexpr uint32_t AtomicUintBinding   = StorageUintBinding + 5u;

	uint32_t base    = 0;
	bool     sampled = false;
	if (image.resource_class == ImageResourceClass::Sampled) {
		if (image.atomic) {
			return std::nullopt;
		}
		sampled = true;
		switch (image.numeric_class) {
			case Prospero::TextureNumericClass::Float:
				base = image.depth_compare ? FirstComparisonImageBinding : SampledFloatBinding;
				break;
			case Prospero::TextureNumericClass::Uint: base = SampledUintBinding; break;
			case Prospero::TextureNumericClass::Sint: base = SampledSintBinding; break;
			case Prospero::TextureNumericClass::Unsupported: return std::nullopt;
			default: return std::nullopt;
		}
		if (image.depth_compare && image.numeric_class != Prospero::TextureNumericClass::Float) {
			return std::nullopt;
		}
	} else if (image.resource_class == ImageResourceClass::Storage) {
		if (image.atomic) {
			if (image.numeric_class != Prospero::TextureNumericClass::Uint) {
				return std::nullopt;
			}
			base = AtomicUintBinding;
		} else {
			switch (image.numeric_class) {
				case Prospero::TextureNumericClass::Float: base = StorageFloatBinding; break;
				case Prospero::TextureNumericClass::Uint: base = StorageUintBinding; break;
				case Prospero::TextureNumericClass::Sint:
				case Prospero::TextureNumericClass::Unsupported: return std::nullopt;
				default: return std::nullopt;
			}
		}
	} else {
		return std::nullopt;
	}

	uint32_t dimension = 0;
	switch (image.dimension) {
		case Decoder::ImageDimension::Dim1D: break;
		case Decoder::ImageDimension::Dim1DArray: dimension = 1u; break;
		case Decoder::ImageDimension::Dim2D: dimension = 2u; break;
		case Decoder::ImageDimension::Dim2DArray: dimension = 3u; break;
		case Decoder::ImageDimension::Dim2DMsaa:
			if (!sampled) {
				return std::nullopt;
			}
			dimension = 4u;
			break;
		case Decoder::ImageDimension::Dim2DMsaaArray:
			if (!sampled) {
				return std::nullopt;
			}
			dimension = 5u;
			break;
		case Decoder::ImageDimension::Dim3D: dimension = sampled ? 6u : 4u; break;
		case Decoder::ImageDimension::Unknown: return std::nullopt;
		default: return std::nullopt;
	}
	return static_cast<DescriptorBindingKind>(base + dimension);
}

struct DescriptorBinding {
	DescriptorBindingKind kind = DescriptorBindingKind::Buffers;
	std::vector<uint32_t> resources;

	bool operator==(const DescriptorBinding& other) const = default;
};

struct BindingLayout {
	uint32_t                       push_data_start_dword = PushData::NoStart;
	uint32_t                       memory_offset_dword   = 0;
	uint32_t                       memory_offset_count   = 0;
	std::vector<uint32_t>          user_data_registers;
	std::vector<DescriptorBinding> descriptors;

	[[nodiscard]] uint32_t ShaderDataDwords() const {
		return memory_offset_dword + (memory_offset_count + 3u) / 4u;
	}
	[[nodiscard]] bool UsesPushData() const { return push_data_start_dword != PushData::NoStart; }
	void               AdvancePushData(uint32_t& cursor) const {
		if (UsesPushData()) {
			cursor = push_data_start_dword + ShaderDataDwords();
		}
	}

	bool operator==(const BindingLayout& other) const = default;
};

struct ShaderInfo {
	static constexpr uint32_t MaxBuffers      = 1024;
	static constexpr uint32_t MaxImages       = 1024;
	static constexpr uint32_t MaxSamplers     = 32;
	static constexpr uint32_t MaxSampledPairs = 64;

	std::vector<BufferResource>      buffers;
	std::vector<ImageResource>       images;
	std::vector<SamplerResource>     samplers;
	std::vector<SampledResourcePair> sampled_pairs;
	std::vector<StageInput>          inputs;
	std::vector<StageOutput>         outputs;
	std::array<uint8_t, 32>          vertex_fetch_components {};
	int32_t                          vertex_offset_sgpr   = -1;
	int32_t                          instance_offset_sgpr = -1;
	bool                             has_bitwise_xor      = false;
	bool                             uses_dma             = false;

	bool operator==(const ShaderInfo& other) const = default;
};

struct BlockInfo {
	uint32_t        id       = 0;
	uint32_t        start_pc = 0;
	uint32_t        end_pc   = 0;
	CFG::Terminator terminator;
	Value           condition;
	Value           indirect_target;
};

struct DescriptorSource {
	// A descriptor the shader fetches from a table of T#s (entry_dwords 8) or V#s (entry_dwords 4)
	// with a key it only knows at runtime. The host enumerates the entries the key can select when
	// the shader is bound and the emitted code picks among them by key.
	struct IndirectTable {
		uint32_t material_source = 0;
		uint32_t heap_source     = 0;
		uint32_t selector_stride = 0;
		uint32_t selector_offset = 0;
		uint32_t key_arg         = 0;
		// Fixed byte offset of the key field within one material record, for materials that
		// store their bindless heap index somewhere other than the record's first dword.
		uint32_t material_offset = 0;
		// Nonzero when the key is a value the shader computes with a provably small range
		// (e.g. the bit index of a wave-reduced mask driving a waterfall loop) rather than a
		// material-record read: the candidates are then heap entries [0, key_count) and
		// material_source is unused. The heap is a 4-dword V# or a 2-dword raw address.
		uint32_t key_count = 0;
		// Immediate byte offset of the descriptor table within the heap, added to
		// key * entry_stride.
		uint32_t heap_offset  = 0;
		uint32_t entry_dwords = 8;
		// Byte distance between consecutive table entries: the descriptor's own size for a
		// packed table, larger when the descriptor heads a bigger per-entry record.
		uint32_t entry_stride = 32;
		// A loop counter key is bounded by the loop's exit comparison rather than by its shape:
		// this host-evaluable value caps the enumerated keys below key_count.
		Value key_bound;
		// The key is arbitrary shader data, but the heap is a V# read with S_BUFFER_LOAD, which
		// returns zero past NumRecords: only keys whose entry lies inside the heap can reach a
		// descriptor, so the heap's own size bounds the enumeration.
		bool heap_bounded = false;
		// The table entry is not in the heap itself: the heap record holds a 64-bit address at
		// pointer_offset, and the descriptor sits heap_offset bytes past that address.
		bool     via_pointer    = false;
		uint32_t pointer_offset = 0;
		// An S# table: the host collapses it to one sampler, since samplers bind statically.
		bool sampler = false;
		// A descriptor waterfall over a per-lane choice among fixed descriptors: the candidates
		// are these runtime-evaluable sources and the key is the index of the chosen one.
		std::vector<uint32_t> candidate_sources;

		bool operator==(const IndirectTable& other) const = default;
	};

	std::array<Value, 8>         dwords {};
	uint32_t                     dword_count = 0;
	std::optional<IndirectTable> indirect_table;

	bool operator==(const DescriptorSource& other) const = default;
};

struct SrtRead {
	Value    value;
	uint32_t flat_offset = 0;

	bool operator==(const SrtRead& other) const = default;
};

struct ResourceBlock {
	// Conditional successors are ordered true, false; an empty condition follows every edge.
	Value                 condition;
	std::vector<uint32_t> successors;
	std::vector<uint32_t> sources;
};

// Stable shader metadata consumed by the renderer after native IR has been discarded.
struct CompiledShaderInfo {
	ShaderType    stage             = ShaderType::Unknown;
	uint64_t      shader_hash       = 0;
	uint32_t      wave_size         = 64;
	uint32_t      user_data_base    = 0;
	uint32_t      user_data_count   = 64;
	uint32_t      scratch_dwords    = 0;
	uint32_t      param_export_mask = 0;
	ShaderInfo    info;
	BindingLayout bindings;
};

struct UniformFillPlan {
	UniformFill          fill;
	std::array<Value, 4> values;
};

// Immutable runtime resource analysis retained by the shader cache. It owns descriptor/SRT,
// uniform condition and fill values without retaining translated blocks.
struct ResourcePlan {
	ResourcePlan() = default;
	~ResourcePlan();

	ResourcePlan(const ResourcePlan&)            = delete;
	ResourcePlan& operator=(const ResourcePlan&) = delete;
	ResourcePlan(ResourcePlan&&) noexcept        = default;
	ResourcePlan& operator=(ResourcePlan&& other) noexcept;

	ShaderType                    stage           = ShaderType::Unknown;
	uint64_t                      shader_hash     = 0;
	uint32_t                      user_data_base  = 0;
	uint32_t                      user_data_count = 64;
	std::list<Inst>               value_storage;
	std::vector<MemoryInfo>       memory_info;
	std::vector<DescriptorSource> descriptor_sources;
	std::vector<ResourceBlock>    control_flow;
	std::vector<uint32_t>         materialization_sources;
	std::vector<SrtRead>          srt_reads;
	std::vector<uint8_t>          clean_flat_slots;
	bool                          requires_specialization_memory = false;
	bool                          srt_plan_complete              = false;
	bool                          resource_tracking_complete     = false;
	ShaderInfo                    info;
	UniformFillPlan               uniform_fill;
};

struct Program: ResourcePlan {
	Program() = default;
	~Program();

	Program(const Program&)            = delete;
	Program& operator=(const Program&) = delete;
	Program(Program&&) noexcept        = default;
	Program&           operator=(Program&& other) noexcept;
	CompiledShaderInfo TakeCompiledInfo() &&;

	std::vector<std::unique_ptr<Block>> block_storage;
	BlockList                           blocks;
	uint32_t                            wave_size           = 64;
	uint32_t                            scratch_dwords      = 0;
	bool                                dispatcher_fallback = false;
	CFG::FailureKind                    cfg_failure_kind    = CFG::FailureKind::None;
	std::string                         fallback_reason;
	std::vector<BlockInfo>              block_info;
	// Typed memory and export instructions reference shader-local metadata by dense index.
	// Decoder-only details (such as NSA register numbers) have already become IR operands.
	std::vector<ExportInfo> export_info;
	std::vector<Value>      dynamic_reads;
	bool                    shader_info_complete = false;
	BindingLayout           bindings;
	bool                    binding_layout_complete = false;
};

std::string ProgramToString(const Program& program);
bool        IsMemoryWriteOpcode(ValueOpcode op);
bool        IsMemoryReadOpcode(ValueOpcode op);
bool        HasShaderMemoryWrites(const Program& program);

// The host evaluates uniform values (branch predicates, descriptor dwords) from memory as it is
// when the shader is bound, so a value is only trustworthy when no write in this shader can
// execute before a memory read it depends on. Program order here is same-block-earlier or any
// block reachable from a writing block, which covers loop back-edges and barrier-ordered
// cross-invocation writes alike; a write through the same descriptor the read uses counts in
// any order, since other waves run the store concurrently.
class ShaderWriteOrder {
public:
	explicit ShaderWriteOrder(const Program& program);

	[[nodiscard]] bool HasWrites() const { return !m_first_write.empty(); }
	[[nodiscard]] bool WriteMayPrecede(const Inst& read) const;
	[[nodiscard]] bool WriteMayAffect(Value value) const;

private:
	const Program&                                m_program;
	std::unordered_map<const Block*, const Inst*> m_first_write;
	std::unordered_set<const Block*>              m_after_write;
	// Descriptor handles any write goes through: a read of the same resource is stale as soon
	// as another wave stores to it, whatever this wave's program order says.
	std::vector<const Inst*> m_written_handles;
};

void  ValidateProgram(const Program& program, bool require_ssa);
void  ResolveControlFlowIdentities(Program& program);
bool  EquivalentValue(const ResourcePlan& program, Value left, Value right);
Value ResolveInvariantPhi(const ResourcePlan& program, Value value);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_ */
