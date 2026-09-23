#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

uint32_t EmitShaderDataDwordLoad(EmitterState& state, uint32_t dword_index) {
	if (state.program.bindings.UsesPushData()) {
		dword_index += state.program.bindings.push_data_start_dword;
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypePushConstantElementPointer(state),
		                          pointer, state.push_constant_variable, ConstantU32(state, 0),
		                          ConstantU32(state, dword_index));
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
		return value;
	}
	if (state.shader_data_storage_variable != 0) {
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
		                          pointer, state.shader_data_storage_variable,
		                          ConstantU32(state, 0), ConstantU32(state, dword_index));
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
		return value;
	}
	return ConstantU32(state, 0);
}

uint32_t EmitBinaryU32(EmitterState& state, spv::Op opcode, uint32_t lhs, uint32_t rhs) {
	const auto ret = state.builder.AllocateId();
	state.builder.AddFunction(opcode, TypeU32(state), ret, lhs, rhs);
	return ret;
}

uint32_t StorageBufferPackedStride(const EmitterState& state, const IR::MemoryInfo& mem) {
	if (mem.resource >= state.program.info.buffers.size()) {
		ExitDescriptorBindingFailure(state, IR::DescriptorBindingKind::Buffers, mem.resource,
		                             "buffer specialization is missing");
	}
	return state.program.info.buffers[mem.resource].packed_stride;
}

Prospero::BufferFormat StorageBufferFormat(const EmitterState& state, const IR::MemoryInfo& mem) {
	if (mem.resource >= state.program.info.buffers.size()) {
		ExitDescriptorBindingFailure(state, IR::DescriptorBindingKind::Buffers, mem.resource,
		                             "buffer specialization is missing");
	}
	return state.program.info.buffers[mem.resource].descriptor_format;
}

void EmitMemoryOffsets(EmitterState& state) {
	for (uint32_t i = 0; i < state.program.bindings.memory_offset_count; i++) {
		const auto word =
		    EmitShaderDataDwordLoad(state, state.program.bindings.memory_offset_dword + i / 4u);
		const auto shift             = ConstantU32(state, (i % 4u) * 8u);
		state.memory_byte_offsets[i] = EmitBinaryU32(
		    state, spv::OpBitwiseAnd, EmitBinaryU32(state, spv::OpShiftRightLogical, word, shift),
		    ConstantU32(state, 0xffu));
	}
}

uint32_t LdsDwordCount(const EmitterState& state) {
	const auto* workgroup = ShaderWorkgroupInput(state.program.stage, state.input_info);
	return workgroup != nullptr ? workgroup->lds_size_dwords : 8192u;
}

static void EnsureLdsStorage(EmitterState& state) {
	if (state.lds_variable != 0) {
		return;
	}
	if (ShaderWorkgroupInput(state.program.stage, state.input_info) == nullptr) {
		EXIT("function LDS was not prepared before SPIR-V function emission\n");
	}
	state.lds_variable = state.builder.DefineGlobalVariable(
	    TypeU32ArrayPointer(state, spv::StorageClassWorkgroup, LdsDwordCount(state)),
	    spv::StorageClassWorkgroup);
	state.builder.AddName(state.lds_variable, "lds_dwords");
}

MemoryResourceAccess PrepareStorageBufferResourceAccess(EmitterState&         state,
                                                        const IR::MemoryInfo& mem,
                                                        uint32_t variable, uint32_t pointer_type) {
	if (variable == 0) {
		ExitDescriptorBindingFailure(state, IR::DescriptorBindingKind::Buffers, mem.resource,
		                             "storage buffer descriptor array was not emitted");
	}
	MemoryResourceAccess access {.kind = mem.kind};
	access.object_pointer = state.builder.AllocateId();
	if (state.indirect_buffer.resource == mem.resource) {
		// A table entry chosen by key: the array index is dynamically uniform (the key is an
		// SGPR value), so plain dynamic indexing of the descriptor array is enough.
		state.builder.AddFunction(spv::OpAccessChain, pointer_type, access.object_pointer, variable,
		                          state.indirect_buffer.array_index);
		access.byte_offset = state.indirect_buffer.byte_offset;
	} else {
		const auto array_index =
		    ResourceForDescriptor(state, IR::DescriptorBindingKind::Buffers, mem.resource);
		state.builder.AddFunction(spv::OpAccessChain, pointer_type, access.object_pointer, variable,
		                          ConstantU32(state, array_index));
		access.byte_offset = state.memory_byte_offsets[array_index];
	}
	access.length = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpArrayLength, TypeU32(state), access.length,
	                          access.object_pointer, 0);
	return access;
}

uint32_t EmitIndirectTableSelection(EmitterState& state, uint32_t key, uint32_t mapping_offset,
                                    uint32_t search_iterations) {
	const auto LoadMapping = [&](uint32_t index) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
		                          pointer, state.flattened_srt_variable, ConstantU32(state, 0),
		                          index);
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
		return value;
	};
	const auto mapping  = ConstantU32(state, mapping_offset);
	auto       low      = ConstantU32(state, 0u);
	auto       high     = LoadMapping(mapping);
	auto       selected = ConstantU32(state, 0u);
	for (uint32_t iteration = 0; iteration < search_iterations; iteration++) {
		const auto active = Binary(state, spv::OpULessThan, TypeBool(state), low, high);
		const auto mid =
		    Binary(state, spv::OpShiftRightLogical, TypeU32(state),
		           Binary(state, spv::OpIAdd, TypeU32(state), low, high), ConstantU32(state, 1u));
		const auto probe = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), probe, active, mid,
		                          ConstantU32(state, 0u));
		const auto entry      = Binary(state, spv::OpIAdd, TypeU32(state), mapping,
		                               Binary(state, spv::OpIAdd, TypeU32(state),
		                                      Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
		                                             probe, ConstantU32(state, 1u)),
		                                      ConstantU32(state, 1u)));
		const auto mapped_key = LoadMapping(entry);
		const auto candidate =
		    LoadMapping(Binary(state, spv::OpIAdd, TypeU32(state), entry, ConstantU32(state, 1u)));
		const auto equal         = Binary(state, spv::OpIEqual, TypeBool(state), mapped_key, key);
		const auto match         = Binary(state, spv::OpLogicalAnd, TypeBool(state), active, equal);
		const auto next_selected = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), next_selected, match, candidate,
		                          selected);
		selected              = next_selected;
		const auto less       = Binary(state, spv::OpULessThan, TypeBool(state), mapped_key, key);
		const auto take_upper = Binary(state, spv::OpLogicalAnd, TypeBool(state), active, less);
		const auto take_lower = Binary(state, spv::OpLogicalAnd, TypeBool(state), active,
		                               Unary(state, spv::OpLogicalNot, TypeBool(state), less));
		const auto next_low   = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpSelect, TypeU32(state), next_low, take_upper,
		    Binary(state, spv::OpIAdd, TypeU32(state), mid, ConstantU32(state, 1u)), low);
		low                  = next_low;
		const auto next_high = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), next_high, take_lower, mid, high);
		high = next_high;
	}
	return selected;
}

IndirectBufferScope::IndirectBufferScope(ValueEmitContext& ctx, const IR::Inst& inst)
    : m_state(ctx.state) {
	m_state.indirect_buffer = {};
	if (IR::BufferAccessOf(inst.GetOpcode()) == IR::BufferAccess::None || inst.NumArgs() == 0u) {
		return;
	}
	const auto& mem = ctx.Memory(inst);
	if (mem.planning_only || mem.resource >= m_state.program.info.buffers.size()) {
		return;
	}
	const auto& buffer = m_state.program.info.buffers[mem.resource];
	if (buffer.indirect_root != mem.resource || buffer.indirect_resources.size() < 2u) {
		return;
	}
	const auto* handle = inst.Arg(0).ResolveInstruction();
	const auto* source = buffer.source < m_state.program.descriptor_sources.size()
	                         ? &m_state.program.descriptor_sources[buffer.source]
	                         : nullptr;
	if (handle == nullptr || source == nullptr || !source->indirect_table.has_value() ||
	    source->indirect_table->key_arg >= handle->NumArgs()) {
		ctx.Fail(inst, "has invalid indirect buffer key provenance");
	}
	if (m_state.flattened_srt_variable == 0 || buffer.indirect_search_iterations == 0u) {
		ctx.Fail(inst, "has no indirect buffer runtime mapping");
	}
	const auto key      = ctx.Def(handle->Arg(source->indirect_table->key_arg));
	const auto selected = EmitIndirectTableSelection(m_state, key, buffer.indirect_mapping_offset,
	                                                 buffer.indirect_search_iterations);
	// Resolve the candidate to its dense array slot and byte offset with a select chain; the
	// tables are small and this keeps the access itself identical to a direct buffer.
	auto array_index =
	    ConstantU32(m_state, ResourceForDescriptor(m_state, IR::DescriptorBindingKind::Buffers,
	                                               buffer.indirect_resources[0]));
	auto byte_offset = m_state.memory_byte_offsets[ResourceForDescriptor(
	    m_state, IR::DescriptorBindingKind::Buffers, buffer.indirect_resources[0])];
	for (uint32_t candidate = 1; candidate < buffer.indirect_resources.size(); candidate++) {
		const auto slot  = ResourceForDescriptor(m_state, IR::DescriptorBindingKind::Buffers,
		                                         buffer.indirect_resources[candidate]);
		const auto match = Binary(m_state, spv::OpIEqual, TypeBool(m_state), selected,
		                          ConstantU32(m_state, candidate));
		array_index =
		    Select(m_state, TypeU32(m_state), match, ConstantU32(m_state, slot), array_index);
		byte_offset = Select(m_state, TypeU32(m_state), match, m_state.memory_byte_offsets[slot],
		                     byte_offset);
	}
	m_state.indirect_buffer = {
	    .resource = mem.resource, .array_index = array_index, .byte_offset = byte_offset};
}

IndirectBufferScope::~IndirectBufferScope() {
	m_state.indirect_buffer = {};
}

MemoryResourceAccess PrepareMemoryResourceAccess(EmitterState& state, const IR::MemoryInfo& mem) {
	MemoryResourceAccess access {.kind = mem.kind};
	switch (mem.kind) {
		case IR::ResourceKind::Lds:
			EnsureLdsStorage(state);
			access.object_pointer = state.lds_variable;
			access.length         = ConstantU32(state, LdsDwordCount(state));
			return access;
		case IR::ResourceKind::Gds:
			if (state.gds_variable == 0) {
				ExitDescriptorBindingFailure(state, IR::DescriptorBindingKind::Gds, mem.resource,
				                             "GDS binding was not emitted");
			}
			access.object_pointer = state.gds_variable;
			if (state.gds_length == 0) {
				EXIT("GDS length was not prepared at function entry\n");
			}
			access.length = state.gds_length;
			return access;
		case IR::ResourceKind::Scratch:
			if (state.scratch_variable[state.lane_half] == 0) {
				EXIT("scratch storage was not prepared before SPIR-V function emission\n");
			}
			access.object_pointer = state.scratch_variable[state.lane_half];
			access.length         = ConstantU32(state, state.program.scratch_dwords);
			return access;
		case IR::ResourceKind::ScalarAddress:
		case IR::ResourceKind::Flat:
		case IR::ResourceKind::Global: EXIT("physical address memory must use the BDA emitter\n");
		case IR::ResourceKind::ScalarBuffer:
		case IR::ResourceKind::Buffer: {
			access = PrepareStorageBufferResourceAccess(state, mem, state.storage_buffer_variable,
			                                            TypeStorageBufferPointer(state));
			access.index_offset = EmitBinaryU32(state, spv::OpShiftRightLogical, access.byte_offset,
			                                    ConstantU32(state, 2u));
			access.add_index_offset = true;
			return access;
		}
		default: EXIT("unsupported memory resource kind: %u\n", static_cast<unsigned>(mem.kind));
	}
	access.length = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpArrayLength, TypeU32(state), access.length,
	                          access.object_pointer, 0);
	return access;
}

uint32_t EmitMemoryElementIndex(EmitterState& state, const MemoryResourceAccess& access,
                                uint32_t raw_index) {
	return access.add_index_offset ? EmitAddU32(state, raw_index, access.index_offset) : raw_index;
}

uint32_t EmitMemoryElementInBounds(EmitterState& state, const MemoryResourceAccess& access,
                                   uint32_t index) {
	const auto in_bounds = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpULessThan, TypeBool(state), in_bounds, index, access.length);
	return in_bounds;
}

uint32_t EmitMemoryElementPointer(EmitterState& state, const MemoryResourceAccess& access,
                                  uint32_t index) {
	if (access.kind == IR::ResourceKind::Lds || access.kind == IR::ResourceKind::Scratch) {
		const auto pointer = state.builder.AllocateId();
		const auto storage_class =
		    access.kind == IR::ResourceKind::Scratch ? spv::StorageClassFunction
		    : ShaderWorkgroupInput(state.program.stage, state.input_info) != nullptr
		        ? spv::StorageClassWorkgroup
		        : spv::StorageClassFunction;
		state.builder.AddFunction(spv::OpAccessChain, TypeU32ElementPointer(state, storage_class),
		                          pointer, access.object_pointer, index);
		return pointer;
	}
	return EmitStorageBufferElementPointer(state, access, index,
	                                       TypeStorageBufferElementPointer(state));
}

uint32_t EmitStorageBufferElementPointer(EmitterState& state, const MemoryResourceAccess& access,
                                         uint32_t index, uint32_t pointer_type) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, pointer_type, pointer, access.object_pointer,
	                          ConstantU32(state, 0), index);
	return pointer;
}

uint32_t EmitTBufferBitcastU32ToI32(EmitterState& state, uint32_t value) {
	const auto ret = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitcast, TypeI32(state), ret, value);
	return ret;
}

bool IsSignedFormatComponent(Format::ComponentType type) {
	return type == Format::ComponentType::Sint || type == Format::ComponentType::Snorm ||
	       type == Format::ComponentType::Sscaled;
}

uint32_t EmitUFloatToF32Bits(EmitterState& state, uint32_t raw, uint32_t bits) {
	const auto mantissa_bits = bits == 11u ? 6u : 5u;
	const auto mantissa_mask = (1u << mantissa_bits) - 1u;
	const auto mantissa =
	    EmitBinaryU32(state, spv::OpBitwiseAnd, raw, ConstantU32(state, mantissa_mask));
	const auto exponent = EmitBinaryU32(
	    state, spv::OpBitwiseAnd,
	    EmitBinaryU32(state, spv::OpShiftRightLogical, raw, ConstantU32(state, mantissa_bits)),
	    ConstantU32(state, 0x1fu));

	const auto exponent_32 = EmitAddU32(state, exponent, ConstantU32(state, 127u - 15u));
	const auto exponent_bits =
	    EmitBinaryU32(state, spv::OpShiftLeftLogical, exponent_32, ConstantU32(state, 23));
	const auto mantissa_bits_32 = EmitBinaryU32(state, spv::OpShiftLeftLogical, mantissa,
	                                            ConstantU32(state, 23u - mantissa_bits));
	const auto normal_bits =
	    EmitBinaryU32(state, spv::OpBitwiseOr, exponent_bits, mantissa_bits_32);
	const auto normal = EmitBitcastU32ToF32(state, normal_bits);

	const auto special_bits =
	    EmitBinaryU32(state, spv::OpBitwiseOr, ConstantU32(state, 0x7f800000u), mantissa_bits_32);
	const auto special = EmitBitcastU32ToF32(state, special_bits);

	const auto mantissa_f32 = state.builder.AllocateId();
	const auto subnormal    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), mantissa_f32, mantissa);
	state.builder.AddFunction(
	    spv::OpFMul, TypeF32(state), subnormal, mantissa_f32,
	    ConstantF32Value(state, std::ldexp(1.0f, 1 - 15 - static_cast<int>(mantissa_bits))));

	const auto zero_exp    = EmitCompareU32Constant(state, spv::OpIEqual, exponent, 0);
	const auto special_exp = EmitCompareU32Constant(state, spv::OpIEqual, exponent, 31);
	const auto finite      = EmitTBufferSelectF32(state, zero_exp, subnormal, normal);
	const auto result      = EmitTBufferSelectF32(state, special_exp, special, finite);
	return EmitBitcastF32ToU32(state, result);
}

uint32_t NormalizeFormatComponent(EmitterState& state, const Format::BufferFormatInfo& info,
                                  uint32_t component, uint32_t raw) {
	const auto bits = info.component_bits[component];
	switch (info.type) {
		case Format::ComponentType::Uint:
		case Format::ComponentType::Sint: return raw;
		case Format::ComponentType::Uscaled: {
			const auto value = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), value, raw);
			return EmitBitcastF32ToU32(state, value);
		}
		case Format::ComponentType::Sscaled: {
			const auto signed_raw = EmitTBufferBitcastU32ToI32(state, raw);
			const auto value      = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertSToF, TypeF32(state), value, signed_raw);
			return EmitBitcastF32ToU32(state, value);
		}
		case Format::ComponentType::Unorm: {
			const auto value      = state.builder.AllocateId();
			const auto normalized = state.builder.AllocateId();
			const auto max_value  = static_cast<float>((1u << bits) - 1u);
			state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), value, raw);
			state.builder.AddFunction(spv::OpFDiv, TypeF32(state), normalized, value,
			                          ConstantF32Value(state, max_value));
			return EmitBitcastF32ToU32(state, normalized);
		}
		case Format::ComponentType::Snorm: {
			const auto signed_raw = EmitTBufferBitcastU32ToI32(state, raw);
			const auto value      = state.builder.AllocateId();
			const auto normalized = state.builder.AllocateId();
			const auto clamped    = state.builder.AllocateId();
			const auto max_value  = static_cast<float>((1u << (bits - 1u)) - 1u);
			state.builder.AddFunction(spv::OpConvertSToF, TypeF32(state), value, signed_raw);
			state.builder.AddFunction(spv::OpFDiv, TypeF32(state), normalized, value,
			                          ConstantF32Value(state, max_value));
			state.builder.AddFunction(spv::OpExtInst, TypeF32(state), clamped, GlslStd450(state),
			                          GLSLstd450FMax, normalized, ConstantF32Value(state, -1.0f));
			return EmitBitcastF32ToU32(state, clamped);
		}
		case Format::ComponentType::Float:
			if (bits == 32u) {
				return raw;
			}
			if (bits == 16u) {
				return EmitBitcastF32ToU32(state, EmitF16BitsToF32(state, raw));
			}
			return EmitUFloatToF32Bits(state, raw, bits);
		default: return raw;
	}
}

void EmitDeviceAtomicMemoryBarrier(EmitterState& state) {
	const auto semantics =
	    spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask;
	state.builder.AddFunction(spv::OpMemoryBarrier, ConstantU32(state, spv::ScopeDevice),
	                          ConstantU32(state, semantics));
}

uint32_t EmitFloatAtomicReplacement(EmitterState& state, uint32_t old, uint32_t source,
                                    bool max_value) {
	struct OrderedBits {
		uint32_t nan;
		uint32_t zero;
		uint32_t key;
	};
	const auto classify = [&](uint32_t bits) {
		const auto cls      = EmitClassifyF32Bits(state, bits);
		const auto negative = EmitCompareU32Constant(state, spv::OpINotEqual,
		                                             EmitAndConstant(state, bits, 0x80000000u), 0u);
		const auto negative_key = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpNot, TypeU32(state), negative_key, bits);
		const auto positive_key =
		    EmitBinaryU32(state, spv::OpBitwiseXor, bits, ConstantU32(state, 0x80000000u));
		return OrderedBits {cls.nan, cls.zero,
		                    EmitSelectValueU32(state, negative, negative_key, positive_key)};
	};
	const auto source_class = classify(source);
	const auto old_class    = classify(old);
	const auto unordered =
	    EmitLogicalOrBool(state, EmitLogicalOrBool(state, source_class.nan, old_class.nan),
	                      EmitLogicalAndBool(state, source_class.zero, old_class.zero));
	const auto compare = state.builder.AllocateId();
	state.builder.AddFunction(max_value ? spv::OpUGreaterThan : spv::OpULessThan, TypeBool(state),
	                          compare, source_class.key, old_class.key);
	return EmitSelectValueU32(
	    state, EmitLogicalAndBool(state, EmitLogicalNotBool(state, unordered), compare), source,
	    old);
}

uint32_t EmitDsSwizzleTargetLane(EmitterState& state, uint32_t subid, uint32_t control) {
	if ((control & 0xc000u) == 0xc000u) {
		const uint32_t mask         = control & 0x1fu;
		const uint32_t rotate       = (control >> 5u) & 0x1fu;
		const uint32_t rotate_delta = (control & 0x400u) != 0u ? ((32u - rotate) & 0x1fu) : rotate;
		const auto     lane         = EmitAndConstant(state, subid, 31);
		const auto     rotated_sum  = EmitAddU32(state, lane, ConstantU32(state, rotate_delta));
		const auto     rotated      = EmitAndConstant(state, rotated_sum, 31);
		const auto     kept         = EmitAndConstant(state, lane, mask);
		const auto     moved        = EmitAndConstant(state, rotated, (~mask) & 31u);
		const auto     combined     = EmitOrU32(state, kept, moved);
		const auto     base         = EmitAndConstant(state, subid, 0xffffffe0u);
		return EmitOrU32(state, base, combined);
	}

	if ((control & 0x8000u) != 0) {
		const auto lane2  = state.builder.AllocateId();
		const auto shift  = state.builder.AllocateId();
		const auto perm0  = state.builder.AllocateId();
		const auto perm   = state.builder.AllocateId();
		const auto base   = state.builder.AllocateId();
		const auto target = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane2, subid,
		                          ConstantU32(state, 3));
		state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU32(state), shift, lane2,
		                          ConstantU32(state, 1));
		state.builder.AddFunction(spv::OpShiftRightLogical, TypeU32(state), perm0,
		                          ConstantU32(state, control), shift);
		state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), perm, perm0,
		                          ConstantU32(state, 3));
		state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), base, subid,
		                          ConstantU32(state, 0xfffffffcu));
		state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(state), target, base, perm);
		return target;
	}

	const auto lane   = state.builder.AllocateId();
	const auto masked = state.builder.AllocateId();
	const auto ored   = state.builder.AllocateId();
	const auto xored  = state.builder.AllocateId();
	const auto base   = state.builder.AllocateId();
	const auto target = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane, subid,
	                          ConstantU32(state, 31));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), masked, lane,
	                          ConstantU32(state, control & 0x1fu));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(state), ored, masked,
	                          ConstantU32(state, (control >> 5u) & 0x1fu));
	state.builder.AddFunction(spv::OpBitwiseXor, TypeU32(state), xored, ored,
	                          ConstantU32(state, (control >> 10u) & 0x1fu));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), base, subid,
	                          ConstantU32(state, 0xffffffe0u));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(state), target, base, xored);
	return target;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
