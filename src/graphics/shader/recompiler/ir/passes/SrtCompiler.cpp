// Flattens a ResourcePlan's SRT expression graph once per shader (Compiler) and evaluates it per
// draw on demand (CompiledSrtEvaluator). Each case mirrors SrtWalker.cpp's Evaluator: opcodes it
// does not evaluate compile to AlwaysFails. Idea and compiler structure from EmK530's KytyPS5
// renderer-optimizations branch; evaluation stays lazy here because our memory reader can wait
// for the GPU, so reads on untaken paths must not happen.
#include "graphics/shader/recompiler/ir/passes/SrtCompiler.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <deque>
#include <unordered_map>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

// Sentinel stored in m_slot_for_inst while an Inst's operands are still being compiled, so a
// genuine cycle (should be impossible post-Phi-elimination, but never trust that blindly) fails
// closed as AlwaysFails instead of infinitely recursing.
constexpr uint32_t kInProgressSlot = UINT32_MAX - 1;

bool IsRawReadOpcode(const ResourcePlan& program, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= program.memory_info.size()) {
		return false;
	}
	const auto kind = program.memory_info[index].kind;
	return (op == ValueOpcode::LoadAddressU32 && kind == ResourceKind::ScalarAddress) ||
	       (op == ValueOpcode::ReadConstBuffer && kind == ResourceKind::ScalarBuffer);
}

class Compiler {
public:
	explicit Compiler(const ResourcePlan& program): m_program(program) {}

	CompiledSrtProgram Run() {
		CompiledSrtProgram result;
		result.descriptor_source_slots.resize(m_program.descriptor_sources.size());
		for (size_t i = 0; i < m_program.descriptor_sources.size(); i++) {
			const auto& source = m_program.descriptor_sources[i];
			for (uint32_t d = 0; d < source.dword_count; d++) {
				result.descriptor_source_slots[i][d] = CompileValue(source.dwords[d]);
			}
		}
		result.srt_read_slots.resize(m_program.srt_reads.size());
		for (size_t i = 0; i < m_program.srt_reads.size(); i++) {
			result.srt_read_slots[i] = CompileValue(m_program.srt_reads[i].value);
		}
		result.control_flow_condition_slots.resize(m_program.control_flow.size());
		for (size_t i = 0; i < m_program.control_flow.size(); i++) {
			const auto& condition = m_program.control_flow[i].condition;
			result.control_flow_condition_slots[i] =
			    condition.IsEmpty() ? kInvalidSlot : CompileValue(condition);
		}
		result.ops = std::move(m_ops);
		return result;
	}

private:
	static CompiledOp AlwaysFailsOp() {
		CompiledOp op;
		op.kind = CompiledOpKind::AlwaysFails;
		return op;
	}

	// Post-order: every operand this node references is compiled (and therefore already has a
	// lower slot index in m_ops) before this node's own CompiledOp is appended -- the invariant
	// ExecuteSrtProgram's single forward pass depends on.
	uint32_t CompileValue(Value value) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			return EmitConstant(value);
		}
		auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			return EmitOp(AlwaysFailsOp());
		}
		if (const auto it = m_slot_for_inst.find(inst); it != m_slot_for_inst.end()) {
			if (it->second == kInProgressSlot) {
				return EmitOp(AlwaysFailsOp());
			}
			return it->second;
		}
		m_slot_for_inst.emplace(inst, kInProgressSlot);
		auto       op         = CompileInst(*inst);
		const auto slot       = EmitOp(std::move(op));
		m_slot_for_inst[inst] = slot;
		return slot;
	}

	uint32_t EmitOp(CompiledOp op) {
		m_ops.push_back(std::move(op));
		return static_cast<uint32_t>(m_ops.size() - 1);
	}

	uint32_t EmitConstant(Value value) {
		CompiledOp op;
		op.kind = CompiledOpKind::Constant;
		switch (value.GetType()) {
			case Type::U1: op.immediate = value.U1(); break;
			case Type::U8: op.immediate = value.U8(); break;
			case Type::U16: op.immediate = value.U16(); break;
			case Type::U32: op.immediate = value.U32(); break;
			case Type::U64: op.immediate = value.U64(); break;
			case Type::F32: op.immediate = std::bit_cast<uint32_t>(value.F32Value()); break;
			default: return EmitOp(AlwaysFailsOp());
		}
		return EmitOp(std::move(op));
	}

	// Mirrors Evaluator::EvaluateInst case-by-case -- see this file's header comment.
	CompiledOp CompileInst(const Inst& inst) {
		const auto opcode = inst.GetOpcode();
		switch (opcode) {
			case ValueOpcode::GetUserData: {
				if (inst.NumArgs() != 1 || inst.Arg(0).GetType() != Type::ScalarReg) {
					return AlwaysFailsOp();
				}
				const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
				if (reg < m_program.user_data_base) {
					return AlwaysFailsOp();
				}
				CompiledOp op;
				op.kind      = CompiledOpKind::GetUserData;
				op.immediate = reg - m_program.user_data_base;
				return op;
			}
			case ValueOpcode::GetShaderBase: {
				CompiledOp op;
				op.kind = CompiledOpKind::GetShaderBase;
				return op;
			}
			case ValueOpcode::Phi: {
				// Evaluator::EvaluatePhi: only a Phi with one invariant value evaluates.
				const auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
				if (value.IsEmpty()) {
					return AlwaysFailsOp();
				}
				CompiledOp op;
				op.kind         = CompiledOpKind::ExtractPassthrough;
				op.num_operands = 1;
				op.operands[0]  = CompileValue(value);
				return op;
			}
			case ValueOpcode::ReadFirstLane: {
				if (inst.NumArgs() != 2) {
					return AlwaysFailsOp();
				}
				CompiledOp op;
				op.kind         = CompiledOpKind::ReadFirstLane;
				op.num_operands = 2;
				op.operands[0]  = CompileValue(inst.Arg(0));
				op.operands[1]  = CompileValue(inst.Arg(1));
				return op;
			}
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32: return GenericUnary(inst, opcode);
			case ValueOpcode::CompositeExtractU64:
			case ValueOpcode::CompositeExtractU32x2: return CompileExtract(inst, opcode);
			case ValueOpcode::CompositeConstructU64: return GenericBinary(inst, opcode);
			case ValueOpcode::ReadConst: {
				if (inst.NumArgs() != 2) {
					return AlwaysFailsOp();
				}
				const auto slot = inst.Arg(1).Resolve();
				if (!slot.IsImmediate() || slot.GetType() != Type::U32) {
					return AlwaysFailsOp();
				}
				const auto slot_index = slot.U32();
				if (slot_index >= m_program.srt_reads.size()) {
					return AlwaysFailsOp();
				}
				CompiledOp op;
				op.kind         = CompiledOpKind::ReadConst;
				op.component    = slot_index;
				op.num_operands = 1;
				op.operands[0]  = CompileValue(m_program.srt_reads[slot_index].value);
				return op;
			}
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::ReadConstBuffer:
				if (IsRawReadOpcode(m_program, inst)) {
					return CompileRawRead(inst, opcode);
				}
				return AlwaysFailsOp();
			case ValueOpcode::IAdd32:
			case ValueOpcode::IAdd64:
			case ValueOpcode::ISub32:
			case ValueOpcode::ISub64:
			case ValueOpcode::IMul32:
			case ValueOpcode::IMul64:
			case ValueOpcode::UMin32:
			case ValueOpcode::FPMul32:
			case ValueOpcode::FPOrdLessThanEqual32:
			case ValueOpcode::FPOrdGreaterThanEqual32:
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::BitwiseAnd64:
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::BitwiseXor32:
			case ValueOpcode::ShiftLeftLogical32:
			case ValueOpcode::ShiftLeftLogical64:
			case ValueOpcode::ShiftRightLogical32:
			case ValueOpcode::ShiftRightLogical64:
			case ValueOpcode::ShiftRightArithmetic32:
			case ValueOpcode::ShiftRightArithmetic64:
			case ValueOpcode::IEqual32:
			case ValueOpcode::INotEqual32:
			case ValueOpcode::ULessThan32:
			case ValueOpcode::UGreaterThan32:
			case ValueOpcode::LogicalAnd:
			case ValueOpcode::LogicalOr:
			case ValueOpcode::LogicalXor: return GenericBinary(inst, opcode);
			case ValueOpcode::ConvertF32U32:
			case ValueOpcode::ConvertU32F32:
			case ValueOpcode::FPTrunc32:
			case ValueOpcode::FPIsNan32:
			case ValueOpcode::BitwiseNot32:
			case ValueOpcode::LogicalNot: return GenericUnary(inst, opcode);
			case ValueOpcode::BitFieldUExtract:
			case ValueOpcode::BitFieldSExtract: return GenericTernary(inst, opcode);
			case ValueOpcode::BitFieldInsert: return GenericQuaternary(inst, opcode);
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectF32: return CompileSelect(inst);
			case ValueOpcode::UndefU1:
			case ValueOpcode::UndefU8:
			case ValueOpcode::UndefU16:
			case ValueOpcode::UndefU32:
			case ValueOpcode::UndefU64:
			default: return AlwaysFailsOp();
		}
	}

	CompiledOp GenericUnary(const Inst& inst, ValueOpcode opcode) {
		if (inst.NumArgs() < 1) {
			return AlwaysFailsOp();
		}
		CompiledOp op;
		op.kind         = CompiledOpKind::Generic;
		op.opcode       = opcode;
		op.num_operands = 1;
		op.operands[0]  = CompileValue(inst.Arg(0));
		return op;
	}

	CompiledOp GenericBinary(const Inst& inst, ValueOpcode opcode) {
		if (inst.NumArgs() < 2) {
			return AlwaysFailsOp();
		}
		CompiledOp op;
		op.kind         = CompiledOpKind::Generic;
		op.opcode       = opcode;
		op.num_operands = 2;
		op.operands[0]  = CompileValue(inst.Arg(0));
		op.operands[1]  = CompileValue(inst.Arg(1));
		return op;
	}

	CompiledOp GenericTernary(const Inst& inst, ValueOpcode opcode) {
		if (inst.NumArgs() < 3) {
			return AlwaysFailsOp();
		}
		CompiledOp op;
		op.kind         = CompiledOpKind::Generic;
		op.opcode       = opcode;
		op.num_operands = 3;
		op.operands[0]  = CompileValue(inst.Arg(0));
		op.operands[1]  = CompileValue(inst.Arg(1));
		op.operands[2]  = CompileValue(inst.Arg(2));
		return op;
	}

	CompiledOp GenericQuaternary(const Inst& inst, ValueOpcode opcode) {
		if (inst.NumArgs() < 4) {
			return AlwaysFailsOp();
		}
		CompiledOp op;
		op.kind         = CompiledOpKind::Generic;
		op.opcode       = opcode;
		op.num_operands = 4;
		op.operands[0]  = CompileValue(inst.Arg(0));
		op.operands[1]  = CompileValue(inst.Arg(1));
		op.operands[2]  = CompileValue(inst.Arg(2));
		op.operands[3]  = CompileValue(inst.Arg(3));
		return op;
	}

	// Mirrors EvaluateInst's Select case: the predicate is resolved through the clean pass (see
	// ExecutePass), and only the branch it selects is ever compiled to require a value -- but
	// both branches still need their own slots compiled here so the flat array stays complete for
	// anyone else referencing the same nodes.
	CompiledOp CompileSelect(const Inst& inst) {
		if (inst.NumArgs() != 3) {
			return AlwaysFailsOp();
		}
		CompiledOp op;
		op.kind         = CompiledOpKind::Select;
		op.num_operands = 3;
		op.operands[0]  = CompileValue(inst.Arg(0));
		op.operands[1]  = CompileValue(inst.Arg(1));
		op.operands[2]  = CompileValue(inst.Arg(2));
		return op;
	}

	// Mirrors Evaluator::EvaluateExtract.
	CompiledOp CompileExtract(const Inst& inst, ValueOpcode opcode) {
		if (inst.NumArgs() != 2) {
			return AlwaysFailsOp();
		}
		const auto index = inst.Arg(1).Resolve();
		if (!index.IsImmediate() || index.GetType() != Type::U32) {
			return AlwaysFailsOp();
		}
		const auto component = index.U32();
		if (component >= 2u) {
			return AlwaysFailsOp();
		}
		if (opcode == ValueOpcode::CompositeExtractU64) {
			CompiledOp op;
			op.kind         = CompiledOpKind::ExtractU64;
			op.component    = component;
			op.num_operands = 1;
			op.operands[0]  = CompileValue(inst.Arg(0));
			return op;
		}
		const auto* source = inst.Arg(0).ResolveInstruction();
		if (source == nullptr) {
			return AlwaysFailsOp();
		}
		const auto source_opcode = source->GetOpcode();
		if (source_opcode == ValueOpcode::CompositeConstructU32x2) {
			// EvaluateExtract never evaluates the construct node itself -- it reaches straight
			// through to one of its arguments. Guard against a construct with too few args
			// (source->Arg(component) would otherwise assert inside Inst::Arg).
			if (component >= source->NumArgs()) {
				return AlwaysFailsOp();
			}
			CompiledOp op;
			op.kind         = CompiledOpKind::ExtractPassthrough;
			op.num_operands = 1;
			op.operands[0]  = CompileValue(source->Arg(component));
			return op;
		}
		if (source_opcode == ValueOpcode::IAddCarry32) {
			if (source->NumArgs() < 2) {
				return AlwaysFailsOp();
			}
			CompiledOp op;
			op.kind         = CompiledOpKind::ExtractCarryHalf;
			op.component    = component;
			op.num_operands = 2;
			op.operands[0]  = CompileValue(source->Arg(0));
			op.operands[1]  = CompileValue(source->Arg(1));
			return op;
		}
		return AlwaysFailsOp();
	}

	// Mirrors Evaluator::EvaluateRawRead's *structural* validation -- everything here is static
	// (memory_info, handle shape, immediate offset), so it's fully resolved once, at compile
	// time. Only the address arithmetic and the actual guest read remain for execute time.
	CompiledOp CompileRawRead(const Inst& inst, ValueOpcode opcode) {
		const auto flags = inst.Flags<MemoryFlags>();
		if (flags.index >= m_program.memory_info.size()) {
			return AlwaysFailsOp();
		}
		const auto& mem = m_program.memory_info[flags.index];
		if (inst.NumArgs() < 2) {
			return AlwaysFailsOp();
		}
		const auto* handle = inst.Arg(0).ResolveInstruction();
		if (handle == nullptr) {
			return AlwaysFailsOp();
		}
		const bool is_const_buffer_read = opcode == ValueOpcode::ReadConstBuffer;
		if (is_const_buffer_read && handle->NumArgs() != 4u) {
			return AlwaysFailsOp();
		}
		if (handle->NumArgs() < 2) {
			return AlwaysFailsOp();
		}
		CompiledOp op;
		op.kind                 = CompiledOpKind::RawRead;
		op.is_const_buffer_read = is_const_buffer_read;
		op.immediate =
		    static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(mem.offset)));
		op.num_operands = is_const_buffer_read ? 5 : 3;
		op.operands[0]  = CompileValue(handle->Arg(0)); // low
		op.operands[1]  = CompileValue(handle->Arg(1)); // high
		op.operands[2]  = CompileValue(inst.Arg(1));    // offset
		if (is_const_buffer_read) {
			op.operands[3] = CompileValue(handle->Arg(2)); // records
			op.operands[4] = CompileValue(handle->Arg(3)); // word3
		}
		return op;
	}

	const ResourcePlan&                       m_program;
	std::vector<CompiledOp>                   m_ops;
	std::unordered_map<const Inst*, uint32_t> m_slot_for_inst;
};

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

float Float32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}
uint64_t Float32Bits(float value) {
	return std::bit_cast<uint32_t>(value);
}

SrtMemoryReader CleanReader(const SrtRuntime& runtime) {
	return runtime.read_specialization_memory != nullptr
	           ? runtime.read_specialization_memory
	           : +[](void*, uint64_t, uint32_t*) { return false; };
}

// Storage for nested (ReadFirstLane) evaluators, reused per nesting depth. A deque keeps the
// outer levels' storage in place while a deeper level grows the pool.
struct StoragePool {
	std::deque<std::vector<uint64_t>> results;
	std::deque<std::vector<uint8_t>>  states;
	uint32_t                          depth = 0;
};

thread_local StoragePool g_storage_pool;

} // namespace

CompiledSrtProgram CompileSrtProgram(const ResourcePlan& program) {
	return Compiler(program).Run();
}

CompiledSrtEvaluator::CompiledSrtEvaluator(const CompiledSrtProgram& compiled,
                                           const SrtRuntime&         runtime,
                                           std::span<const uint8_t>  clean_flat_slots,
                                           CompiledSrtEvaluator* clean, uint32_t active_mask_slot)
    : m_compiled(compiled), m_runtime(runtime), m_clean_flat_slots(clean_flat_slots),
      m_clean(clean), m_active_mask_slot(active_mask_slot) {
	auto& pool = g_storage_pool;
	if (pool.states.size() <= pool.depth) {
		pool.results.emplace_back();
		pool.states.emplace_back();
	}
	m_results = &pool.results[pool.depth];
	m_states  = &pool.states[pool.depth];
	pool.depth++;
	m_results->resize(std::max(m_results->size(), compiled.ops.size()));
	m_states->assign(compiled.ops.size(), 0u);
}

CompiledSrtEvaluator::~CompiledSrtEvaluator() {
	g_storage_pool.depth--;
}

bool CompiledSrtEvaluator::Evaluate(uint32_t slot, uint64_t& result) {
	if (slot >= m_compiled.ops.size()) {
		return false;
	}
	const auto& op = m_compiled.ops[slot];
	// Inside a ReadFirstLane, a select on that lane mask is true for the lane being read.
	if (op.kind == CompiledOpKind::Select && m_active_mask_slot != kInvalidSlot &&
	    op.operands[0] == m_active_mask_slot) {
		return Evaluate(op.operands[1], result);
	}
	auto& state = (*m_states)[slot];
	if (state == 1u) {
		result = (*m_results)[slot];
		return true;
	}
	if (state == 2u) {
		return false;
	}
	uint64_t value = 0;
	if (!Compute(op, value)) {
		// Ops only reference earlier slots, so a failure cannot be caused by a cycle and is the
		// same on every later request.
		(*m_states)[slot] = 2u;
		return false;
	}
	(*m_states)[slot]  = 1u;
	(*m_results)[slot] = value;
	result             = value;
	return true;
}

bool CompiledSrtEvaluator::RawReadAddress(uint32_t slot, uint64_t& address, bool& zero) {
	zero = false;
	if (slot >= m_compiled.ops.size()) {
		return false;
	}
	const auto& op = m_compiled.ops[slot];
	if (op.kind != CompiledOpKind::RawRead) {
		return false;
	}
	uint64_t low    = 0;
	uint64_t high   = 0;
	uint64_t offset = 0;
	if (!Evaluate(op.operands[0], low) || !Evaluate(op.operands[1], high) ||
	    !Evaluate(op.operands[2], offset)) {
		return false;
	}
	const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	const auto immediate = static_cast<int64_t>(op.immediate);
	if (op.is_const_buffer_read) {
		uint64_t records = 0;
		uint64_t word3   = 0;
		if (!Evaluate(op.operands[3], records) || !Evaluate(op.operands[4], word3)) {
			return false;
		}
		if (immediate < 0) {
			return false;
		}
		const auto byte_offset = static_cast<uint64_t>(immediate) + static_cast<uint32_t>(offset);
		const auto aligned     = byte_offset & ~uint64_t {3};
		const auto stride      = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		const auto size = stride == 0u
		                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (aligned > size || size - aligned < sizeof(uint32_t)) {
			// S_BUFFER_LOAD past NumRecords returns zero.
			zero = true;
			return true;
		}
		address = ((base & ~uint64_t {3}) + byte_offset) & ~uint64_t {3};
		return true;
	}
	const auto relative =
	    (immediate & ~int64_t {3}) + static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
	return AddSignedAddress(base & ~uint64_t {3}, relative, address);
}

bool CompiledSrtEvaluator::Compute(const CompiledOp& op, uint64_t& result) {
	const auto arg = [&](uint32_t index, uint64_t& value) {
		return Evaluate(op.operands[index], value);
	};
	switch (op.kind) {
		case CompiledOpKind::AlwaysFails: return false;
		case CompiledOpKind::Constant: result = op.immediate; return true;
		case CompiledOpKind::GetUserData:
			if (op.immediate >= m_runtime.user_data.size()) {
				return false;
			}
			result = m_runtime.user_data[op.immediate];
			return true;
		case CompiledOpKind::GetShaderBase: result = m_runtime.shader_base; return true;
		case CompiledOpKind::ReadFirstLane: {
			auto clean_runtime        = m_runtime;
			clean_runtime.read_memory = CleanReader(m_runtime);
			CompiledSrtEvaluator clean_active(m_compiled, clean_runtime, {}, nullptr,
			                                  op.operands[1]);
			CompiledSrtEvaluator active(m_compiled, m_runtime, m_clean_flat_slots, &clean_active,
			                            op.operands[1]);
			return active.Evaluate(op.operands[0], result);
		}
		case CompiledOpKind::ReadConst: {
			const auto slot = op.component;
			if (slot < m_clean_flat_slots.size() && m_clean_flat_slots[slot] != 0u &&
			    m_clean != nullptr) {
				return m_clean->Evaluate(op.operands[0], result);
			}
			return arg(0, result);
		}
		case CompiledOpKind::ExtractU64: {
			uint64_t packed = 0;
			if (!arg(0, packed)) {
				return false;
			}
			result = static_cast<uint32_t>(packed >> (op.component * 32u));
			return true;
		}
		case CompiledOpKind::ExtractPassthrough: return arg(0, result);
		case CompiledOpKind::ExtractCarryHalf: {
			uint64_t lhs = 0;
			uint64_t rhs = 0;
			if (!arg(0, lhs) || !arg(1, rhs)) {
				return false;
			}
			const auto sum =
			    static_cast<uint64_t>(static_cast<uint32_t>(lhs)) + static_cast<uint32_t>(rhs);
			result =
			    op.component == 0u ? static_cast<uint32_t>(sum) : static_cast<uint32_t>(sum >> 32u);
			return true;
		}
		case CompiledOpKind::RawRead: {
			// RawReadAddress needs this op's slot; recover it from the op's position.
			const auto slot    = static_cast<uint32_t>(&op - m_compiled.ops.data());
			uint64_t   address = 0;
			bool       zero    = false;
			if (!RawReadAddress(slot, address, zero)) {
				return false;
			}
			if (zero) {
				result = 0;
				return true;
			}
			uint32_t word = 0;
			if (m_runtime.read_memory != nullptr) {
				if (!m_runtime.read_memory(m_runtime.userdata, address, &word)) {
					return false;
				}
			} else {
				std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
			}
			result = word;
			return true;
		}
		case CompiledOpKind::Select: {
			auto&    predicate = m_clean != nullptr ? *m_clean : *this;
			uint64_t value     = 0;
			if (!predicate.Evaluate(op.operands[0], value)) {
				return false;
			}
			return arg(value != 0u ? 1u : 2u, result);
		}
		case CompiledOpKind::Generic: break;
	}

	uint64_t a = 0;
	uint64_t b = 0;
	uint64_t c = 0;
	uint64_t d = 0;
	if ((op.num_operands > 0 && !arg(0, a)) || (op.num_operands > 1 && !arg(1, b)) ||
	    (op.num_operands > 2 && !arg(2, c)) || (op.num_operands > 3 && !arg(3, d))) {
		return false;
	}
	switch (op.opcode) {
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32: result = a; return true;
		case ValueOpcode::CompositeConstructU64:
			result =
			    static_cast<uint32_t>(a) | (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case ValueOpcode::IAdd32: result = static_cast<uint32_t>(a + b); return true;
		case ValueOpcode::IAdd64: result = a + b; return true;
		case ValueOpcode::ISub32: result = static_cast<uint32_t>(a - b); return true;
		case ValueOpcode::ISub64: result = a - b; return true;
		case ValueOpcode::IMul32: result = static_cast<uint32_t>(a * b); return true;
		case ValueOpcode::IMul64: result = a * b; return true;
		case ValueOpcode::UMin32:
			result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
			return true;
		case ValueOpcode::ConvertF32U32:
			result = Float32Bits(static_cast<float>(static_cast<uint32_t>(a)));
			return true;
		case ValueOpcode::ConvertU32F32: {
			const auto value = Float32(a);
			if (!std::isfinite(value) || value < 0.0f || static_cast<double>(value) > UINT32_MAX) {
				return false;
			}
			result = static_cast<uint32_t>(value);
			return true;
		}
		case ValueOpcode::FPMul32: result = Float32Bits(Float32(a) * Float32(b)); return true;
		case ValueOpcode::FPTrunc32: result = Float32Bits(std::trunc(Float32(a))); return true;
		case ValueOpcode::FPIsNan32: result = std::isnan(Float32(a)); return true;
		case ValueOpcode::FPOrdLessThanEqual32: result = Float32(a) <= Float32(b); return true;
		case ValueOpcode::FPOrdGreaterThanEqual32: result = Float32(a) >= Float32(b); return true;
		case ValueOpcode::BitwiseAnd32: result = static_cast<uint32_t>(a & b); return true;
		case ValueOpcode::BitwiseAnd64: result = a & b; return true;
		case ValueOpcode::BitwiseOr32: result = static_cast<uint32_t>(a | b); return true;
		case ValueOpcode::BitwiseXor32: result = static_cast<uint32_t>(a ^ b); return true;
		case ValueOpcode::BitwiseNot32: result = ~static_cast<uint32_t>(a); return true;
		case ValueOpcode::ShiftLeftLogical32:
			result = static_cast<uint32_t>(a) << (b & 31u);
			return true;
		case ValueOpcode::ShiftLeftLogical64: result = a << (b & 63u); return true;
		case ValueOpcode::ShiftRightLogical32:
			result = static_cast<uint32_t>(a) >> (b & 31u);
			return true;
		case ValueOpcode::ShiftRightLogical64: result = a >> (b & 63u); return true;
		case ValueOpcode::ShiftRightArithmetic32:
			result = static_cast<uint32_t>(std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >>
			                               (b & 31u));
			return true;
		case ValueOpcode::ShiftRightArithmetic64:
			result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
			return true;
		case ValueOpcode::BitFieldUExtract: {
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			const auto mask = width == 32u  ? UINT32_MAX
			                  : width == 0u ? 0u
			                                : (uint32_t {1} << width) - 1u;
			result          = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
			return true;
		}
		case ValueOpcode::BitFieldSExtract: {
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = 0;
				return true;
			}
			const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
			auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
			if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
				bits |= ~mask;
			}
			result = bits;
			return true;
		}
		case ValueOpcode::BitFieldInsert: {
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask = width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result =
			    (static_cast<uint32_t>(a) & ~mask) | ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case ValueOpcode::IEqual32:
			result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::INotEqual32:
			result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::ULessThan32:
			result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::UGreaterThan32:
			result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::LogicalAnd: result = (a != 0u) && (b != 0u); return true;
		case ValueOpcode::LogicalOr: result = (a != 0u) || (b != 0u); return true;
		case ValueOpcode::LogicalXor: result = (a != 0u) != (b != 0u); return true;
		case ValueOpcode::LogicalNot: result = a == 0u; return true;
		default: return false;
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
