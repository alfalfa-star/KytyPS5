#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <algorithm>
#include <bit>
#include <fmt/format.h>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr uint32_t SamplerBorderClampMask = (1u << 2u) | (1u << 5u) | (1u << 8u);
// Entries a loop-counter key may select at most; the loop bound trims this when it is smaller.
constexpr uint32_t MaxLoopTableKeys = 32u;
// Upper bound on entries of a heap-size-bounded table; the host checks the actual V# size.
constexpr uint32_t MaxHeapTableKeys          = 4096u;
constexpr uint32_t SamplerDword3ReservedMask = 0x3ffff000u;

uint32_t PossibleU32Bits(Value value) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		return value.GetType() == Type::U32 ? value.U32() : UINT32_MAX;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return UINT32_MAX;
	}
	switch (inst->GetOpcode()) {
		case ValueOpcode::BitwiseAnd32:
			return PossibleU32Bits(inst->Arg(0)) & PossibleU32Bits(inst->Arg(1));
		case ValueOpcode::BitwiseOr32:
			return PossibleU32Bits(inst->Arg(0)) | PossibleU32Bits(inst->Arg(1));
		case ValueOpcode::ShiftLeftLogical32: {
			const auto shift = inst->Arg(1).Resolve();
			return shift.IsImmediate() && shift.GetType() == Type::U32
			           ? PossibleU32Bits(inst->Arg(0)) << (shift.U32() & 31u)
			           : UINT32_MAX;
		}
		default: return UINT32_MAX;
	}
}

Value CanonicalizeSampleAdjustDword3(Value value) {
	for (;;) {
		value            = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::BitwiseOr32) {
			return value;
		}
		const auto left           = inst->Arg(0).Resolve();
		const auto right          = inst->Arg(1).Resolve();
		const bool left_reserved  = (PossibleU32Bits(left) & ~SamplerDword3ReservedMask) == 0;
		const bool right_reserved = (PossibleU32Bits(right) & ~SamplerDword3ReservedMask) == 0;
		if (left_reserved && right_reserved) {
			return Value(0u);
		}
		if (left_reserved) {
			value = right;
		} else if (right_reserved) {
			value = left;
		} else {
			return value;
		}
	}
}

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "vertex";
		case ShaderType::Pixel: return "pixel";
		case ShaderType::Fetch: return "fetch";
		case ShaderType::Compute: return "compute";
		default: return "unknown";
	}
}

uint32_t ByteExtent(const MemoryInfo& memory) {
	const auto bytes = std::max((memory.data_bits + 7u) / 8u, 1u);
	const auto count = std::max(memory.data_dwords, 1u);
	const auto end   = static_cast<uint64_t>(memory.offset) + static_cast<uint64_t>(bytes) * count;
	return end > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(end);
}

class Tracker {
public:
	explicit Tracker(Program& program)
	    : m_program(program), m_info(program.info), m_write_order(program) {
		m_info.buffers.clear();
		m_info.images.clear();
		m_info.samplers.clear();
		m_info.sampled_pairs.clear();
		m_info.uses_dma = false;
	}

	void Run() {
		if (m_program.resource_tracking_complete) {
			Fail(0, "resources already tracked");
		}
		if (!m_program.srt_plan_complete) {
			Fail(0, "SRT plan is not ready");
		}
		PlanIndirectTables();
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				Collect(inst);
			}
		}
		LinkImageAliases();
		for (const auto& patch: m_handle_patches) {
			patch.handle->SetFlags<uint32_t>(patch.resource);
		}
		for (const auto& patch: m_memory_patches) {
			auto& memory    = m_program.memory_info[patch.index];
			memory.resource = patch.resource;
			if (patch.has_sampler) {
				memory.sampler = patch.sampler;
			}
		}
		for (const auto& plan: m_indirect_tables) {
			// The handle now names the runtime key. An image handle also carries the heap
			// descriptor in its spare operands so the values stay live past dead-code
			// elimination; a V# handle has no spare operands and pins them with references.
			plan.handle->SetArg(0, plan.key);
			if (plan.width == 8u) {
				for (uint32_t dword = 0; dword < 4u; dword++) {
					plan.handle->SetArg(dword + 1u, plan.roots[dword + 4u]);
				}
				for (uint32_t dword = 5u; dword < plan.roots.size(); dword++) {
					plan.handle->SetArg(dword, plan.key);
				}
			} else {
				for (uint32_t dword = 1u; dword < plan.width; dword++) {
					plan.handle->SetArg(dword, plan.key);
				}
				for (uint32_t dword = 4u; dword < plan.roots.size(); dword++) {
					if (plan.roots[dword].Resolve().TryInstruction() != nullptr) {
						plan.handle->Parent()->AppendNewInst(ValueOpcode::ReferenceU32,
						                                     {plan.roots[dword]});
					}
				}
			}
			// A candidate may be defined in one branch arm only; pin it where it is defined.
			for (const auto& pin: plan.pins) {
				auto* defining = pin.Resolve().TryInstruction();
				auto* block    = defining != nullptr ? defining->Parent() : plan.handle->Parent();
				block->AppendNewInst(ValueOpcode::ReferenceU32, {pin});
			}
			if (plan.select) {
				continue;
			}
			for (uint32_t dword = 0; dword < plan.width; dword++) {
				m_program.memory_info[plan.memory[dword]].planning_only = true;
			}
		}
		std::erase_if(m_program.dynamic_reads, [&](Value value) {
			const auto* inst = value.Resolve().TryInstruction();
			return std::any_of(m_indirect_tables.begin(), m_indirect_tables.end(),
			                   [&](const IndirectTablePlan& plan) {
				                   return std::ranges::find(plan.reads, inst) != plan.reads.end();
			                   });
		});
		m_program.descriptor_sources         = std::move(m_sources);
		m_program.info                       = std::move(m_info);
		m_program.resource_tracking_complete = true;
	}

private:
	struct HandlePatch {
		Inst*    handle   = nullptr;
		uint32_t resource = 0;
	};

	struct MemoryPatch {
		uint32_t index       = 0;
		uint32_t resource    = 0;
		uint32_t sampler     = 0;
		bool     has_sampler = false;
	};

	struct IndirectTablePlan {
		Inst*                      handle = nullptr;
		uint32_t                   source = 0;
		uint32_t                   width  = 8;
		Value                      key;
		std::array<Value, 8>       roots {};
		std::array<uint32_t, 8>    memory {};
		std::array<const Inst*, 8> reads {};
		// A select table reads no heap: nothing becomes planning-only, and its candidate
		// descriptor values are pinned so they outlive dead-code elimination.
		bool               select = false;
		std::vector<Value> pins;
	};

	[[noreturn]] void Fail(uint32_t pc, const std::string& reason) const {
		const auto message =
		    fmt::format("shader resource tracking: hash=0x{:016x} stage={} pc=0x{:08x} {}",
		                m_program.shader_hash, StageName(m_program.stage), pc, reason);
		EXIT("%s", message.c_str());
		std::abort();
	}

	struct SelectingBranch {
		Value    condition;
		uint32_t true_arm = 0;
	};

	const Block* BlockById(uint32_t id) const {
		for (size_t index = 0; index < m_program.block_info.size(); index++) {
			if (m_program.block_info[index].id == id) {
				return m_program.blocks[index];
			}
		}
		return nullptr;
	}

	// Blocks reachable from `from` without entering `stop`. Fails when `avoid` is reached.
	static bool Reachable(const Block* from, const Block* stop, const Block* avoid,
	                      std::unordered_set<const Block*>& reached) {
		std::vector<const Block*> pending;
		if (from != stop && reached.insert(from).second) {
			pending.push_back(from);
		}
		while (!pending.empty()) {
			const auto* block = pending.back();
			pending.pop_back();
			if (block == avoid) {
				return false;
			}
			for (const auto* successor: block->ImmSuccessors()) {
				if (successor != stop && reached.insert(successor).second) {
					pending.push_back(successor);
				}
			}
		}
		return true;
	}

	// The conditional branch whose outcome alone decides which of `incoming` enters `merge`:
	// every path to the merge passes it, its two successor regions (up to the merge) are
	// disjoint and never lead back to it, and each incoming block sits in a different region.
	// The arms may contain arbitrary control flow of their own.
	std::optional<SelectingBranch>
	FindSelectingBranch(const Block* merge, const std::array<const Block*, 2>& incoming) {
		if (const auto cached = m_selecting_branches.find(merge);
		    cached != m_selecting_branches.end()) {
			return cached->second;
		}
		const auto found            = SearchSelectingBranch(merge, incoming);
		m_selecting_branches[merge] = found;
		return found;
	}

	std::optional<SelectingBranch>
	SearchSelectingBranch(const Block* merge, const std::array<const Block*, 2>& incoming) const {
		for (size_t index = 0; index < m_program.blocks.size(); index++) {
			const auto* branch = m_program.blocks[index];
			const auto& info   = m_program.block_info[index];
			if (branch == merge || branch->ImmSuccessors().size() != 2u ||
			    info.terminator.kind != CFG::TerminatorKind::ConditionalBranch) {
				continue;
			}
			const std::array successors {BlockById(info.terminator.true_block),
			                             BlockById(info.terminator.false_block)};
			if (successors[0] == nullptr || successors[1] == nullptr ||
			    successors[0] == successors[1]) {
				continue;
			}
			std::unordered_set<const Block*> bypass;
			if (branch != m_program.blocks.front() &&
			    (!Reachable(m_program.blocks.front(), branch, merge, bypass) ||
			     bypass.contains(merge))) {
				continue;
			}
			std::array<std::unordered_set<const Block*>, 2> regions;
			if (!Reachable(successors[0], merge, branch, regions[0]) ||
			    !Reachable(successors[1], merge, branch, regions[1]) ||
			    std::ranges::any_of(
			        regions[0], [&](const Block* block) { return regions[1].contains(block); })) {
				continue;
			}
			std::array<uint32_t, 2> sides {};
			bool                    placed = true;
			for (uint32_t arm = 0; arm < 2 && placed; arm++) {
				if (incoming[arm] == branch) {
					placed     = successors[0] == merge || successors[1] == merge;
					sides[arm] = successors[0] == merge ? 0u : 1u;
				} else if (regions[0].contains(incoming[arm])) {
					sides[arm] = 0u;
				} else if (regions[1].contains(incoming[arm])) {
					sides[arm] = 1u;
				} else {
					placed = false;
				}
			}
			if (!placed || sides[0] == sides[1]) {
				continue;
			}
			return SelectingBranch {info.condition, sides[0] == 0u ? 0u : 1u};
		}
		return {};
	}

	// Rewrites a two-way Phi that merges a host-evaluable predicate's arms into a host Select on
	// that predicate. Applies recursively to the predicate and the arms, so a predicate the
	// compiler materialized as a chain of boolean Phis (one per nested branch) lowers as well.
	Value LowerDescriptorPhi(Value value, RuntimeValueType type = RuntimeValueType::Any,
	                         uint32_t depth = 0) {
		value           = value.Resolve();
		const auto* phi = value.TryInstruction();
		if (depth > 8u || phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi ||
		    phi->NumArgs() != 2u || phi->NumPhiBlocks() != 2u ||
		    (phi->GetType() != Type::U32 && phi->GetType() != Type::U1) ||
		    m_program.blocks.size() != m_program.block_info.size()) {
			return value;
		}
		if (const auto known = m_descriptor_selections.find(phi);
		    known != m_descriptor_selections.end()) {
			return known->second;
		}
		// Until proven otherwise the Phi stays as is; this also stops re-entry through a
		// predicate that depends on the value being lowered.
		m_descriptor_selections.emplace(phi, value);
		const auto*                       merge = phi->Parent();
		const std::array<const Block*, 2> incoming {phi->PhiBlock(0), phi->PhiBlock(1)};
		if (merge == nullptr || incoming[0] == nullptr || incoming[1] == nullptr ||
		    incoming[0] == incoming[1] || incoming[0] == merge || incoming[1] == merge ||
		    merge->ImmPredecessors().size() != 2u ||
		    !std::ranges::is_permutation(merge->ImmPredecessors(), incoming)) {
			return value;
		}
		const auto branch = FindSelectingBranch(merge, incoming);
		if (!branch.has_value()) {
			return value;
		}
		const auto condition =
		    LowerDescriptorPhi(branch->condition, RuntimeValueType::Integer, depth + 1u);
		const auto on_true  = LowerDescriptorPhi(phi->Arg(branch->true_arm), type, depth + 1u);
		const auto on_false = LowerDescriptorPhi(phi->Arg(branch->true_arm ^ 1u), type, depth + 1u);
		if (!ValidateRuntimeValue(m_program, condition, RuntimeValueType::Integer)) {
			// A divergent predicate (typically a ballot deciding whether any lane needs the
			// resource) cannot be decided on the host. When the register was never written on
			// one path, that arm carries the SSA placeholder zero rather than a real descriptor:
			// the resource is only ever used on the other path, so its descriptor is the one.
			if (type == RuntimeValueType::Any && phi->GetType() == Type::U32) {
				const bool true_unset  = on_true == Value(0u);
				const bool false_unset = on_false == Value(0u);
				const auto defined     = true_unset ? on_false : on_true;
				if (true_unset != false_unset && ValidateRuntimeValue(m_program, defined, type) &&
				    !m_write_order.WriteMayAffect(defined)) {
					m_descriptor_selections[phi] = defined;
					return defined;
				}
			}
			return value;
		}
		if (!ValidateRuntimeValue(m_program, on_true, type) ||
		    !ValidateRuntimeValue(m_program, on_false, type)) {
			return value;
		}
		// The host picks the arm from memory as bound, which only matches the GPU when no write
		// in this shader can land before the reads behind the predicate or either arm.
		if (m_write_order.WriteMayAffect(condition) || m_write_order.WriteMayAffect(on_true) ||
		    m_write_order.WriteMayAffect(on_false)) {
			return value;
		}
		// Retain a host expression; replacing the GPU Phi would break SSA dominance.
		auto& selected = m_program.value_storage.emplace_back(
		    phi->GetType() == Type::U1 ? ValueOpcode::SelectU1 : ValueOpcode::SelectU32);
		selected.SetArg(0, condition);
		selected.SetArg(1, on_true);
		selected.SetArg(2, on_false);
		m_descriptor_selections[phi] = Value(&selected);
		return Value(&selected);
	}

	void MakeSource(const Inst& handle, uint32_t width, bool sampler, bool sample_adjust,
	                DescriptorSource& descriptor, uint32_t pc) {
		if (handle.NumArgs() != width) {
			Fail(pc, fmt::format("{} has {} descriptor dwords, expected {}",
			                     ValueOpcodeName(handle.GetOpcode()), handle.NumArgs(), width));
		}
		descriptor.dword_count = width;
		for (uint32_t i = 0; i < width; i++) {
			descriptor.dwords[i] = LowerDescriptorPhi(handle.Arg(i));
		}
		if (sample_adjust) {
			descriptor.dwords[3] = CanonicalizeSampleAdjustDword3(descriptor.dwords[3]);
		}
		const auto dword0 = descriptor.dwords[0].Resolve();
		if (sampler && dword0.IsImmediate() && dword0.GetType() == Type::U32 &&
		    (dword0.U32() & SamplerBorderClampMask) == 0) {
			// Border color and its table index are unused unless a clamp axis selects border mode.
			descriptor.dwords[3] = Value(0u);
		}
	}

	bool ValidateSource(const DescriptorSource& descriptor, uint32_t& bad_dword) const {
		for (uint32_t i = 0; i < descriptor.dword_count; i++) {
			bad_dword = i;
			if (descriptor.dwords[i].Resolve().GetType() != Type::U32) {
				return false;
			}
			if (!ValidateRuntimeValue(m_program, descriptor.dwords[i])) {
				return false;
			}
		}
		return true;
	}

	uint32_t InternSource(const DescriptorSource& descriptor) {
		for (uint32_t candidate = 0; candidate < m_sources.size(); candidate++) {
			const auto& current = m_sources[candidate];
			if (current.dword_count != descriptor.dword_count ||
			    current.indirect_table != descriptor.indirect_table) {
				continue;
			}
			bool same = true;
			for (uint32_t i = 0; i < descriptor.dword_count; i++) {
				same = same && EquivalentValue(m_program, current.dwords[i], descriptor.dwords[i]);
			}
			if (same) {
				return candidate;
			}
		}
		m_sources.push_back(descriptor);
		return static_cast<uint32_t>(m_sources.size() - 1);
	}

	static bool ImmediateU32(Value value, uint32_t& result) {
		value = value.Resolve();
		if (!value.IsImmediate() || value.GetType() != Type::U32) {
			return false;
		}
		result = value.U32();
		return true;
	}

	// A 32-bit scalar add as translation shapes it: S_ADD_I32/S_LSHLn_ADD_U32 give a plain
	// IAdd32 while S_ADD_U32 keeps its carry (IAddCarry32 pairs, the second adding a zero
	// carry-in). Peels either form down to its two operands.
	// `value` scales a key to a byte offset: `key << n` or `key * imm`. Returns the stride.
	static bool MatchEntryStride(const Inst& value, uint32_t& stride) {
		uint32_t immediate = 0;
		if (value.NumArgs() != 2u || !ImmediateU32(value.Arg(1), immediate)) {
			return false;
		}
		if (value.GetOpcode() == ValueOpcode::ShiftLeftLogical32) {
			if (immediate >= 32u) {
				return false;
			}
			stride = 1u << immediate;
			return true;
		}
		if (value.GetOpcode() == ValueOpcode::IMul32) {
			if (immediate == 0u || (immediate % sizeof(uint32_t)) != 0u) {
				return false;
			}
			stride = immediate;
			return true;
		}
		return false;
	}

	static bool MatchAdd(Value value, Value& lhs, Value& rhs) {
		for (uint32_t depth = 0; depth < 4u; depth++) {
			const auto* inst = value.Resolve().TryInstruction();
			if (inst == nullptr) {
				return false;
			}
			if (inst->GetOpcode() == ValueOpcode::IAdd32 && inst->NumArgs() == 2u) {
				lhs = inst->Arg(0).Resolve();
				rhs = inst->Arg(1).Resolve();
				return true;
			}
			uint32_t component = 0;
			if (inst->GetOpcode() != ValueOpcode::CompositeExtractU32x2 || inst->NumArgs() != 2u ||
			    !ImmediateU32(inst->Arg(1), component) || component != 0u) {
				return false;
			}
			const auto* add = inst->Arg(0).Resolve().TryInstruction();
			if (add == nullptr || add->GetOpcode() != ValueOpcode::IAddCarry32 ||
			    add->NumArgs() != 2u) {
				return false;
			}
			uint32_t zero = 0;
			if (ImmediateU32(add->Arg(1), zero) && zero == 0u) {
				value = add->Arg(0);
				continue;
			}
			if (ImmediateU32(add->Arg(0), zero) && zero == 0u) {
				value = add->Arg(1);
				continue;
			}
			lhs = add->Arg(0).Resolve();
			rhs = add->Arg(1).Resolve();
			return true;
		}
		return false;
	}

	// A register that an inner loop carries unchanged gets a Phi at that loop's header whose
	// arms all name the same outer value; see through such forwarding Phis to that value.
	static Value ResolveForwardingPhis(Value value) {
		for (uint32_t depth = 0; depth < 8u; depth++) {
			value           = value.Resolve();
			const auto* phi = value.TryInstruction();
			if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi || phi->NumArgs() == 0u) {
				return value;
			}
			Value forwarded;
			for (size_t index = 0; index < phi->NumArgs(); index++) {
				const auto arm = phi->Arg(index).Resolve();
				if (arm == value) {
					continue;
				}
				if (forwarded.IsEmpty()) {
					forwarded = arm;
				} else if (forwarded != arm) {
					return value;
				}
			}
			if (forwarded.IsEmpty()) {
				return value;
			}
			value = forwarded;
		}
		return value;
	}

	// `value` is `key + 1` in any add shape.
	static bool IsIncrementOf(Value value, Value key) {
		Value    lhs;
		Value    rhs;
		uint32_t step = 0;
		if (!MatchAdd(value, lhs, rhs)) {
			return false;
		}
		key = ResolveForwardingPhis(key);
		return (ResolveForwardingPhis(lhs) == key && ImmediateU32(rhs, step) && step == 1u) ||
		       (ResolveForwardingPhis(rhs) == key && ImmediateU32(lhs, step) && step == 1u);
	}

	// `value` reaches `root` only through the arithmetic an add shape introduces (see MatchAdd),
	// with no other consumer along the way.
	static bool FlowsOnlyInto(const Inst& value, const Inst& root, uint32_t depth = 0) {
		if (depth > 8u || value.Uses().empty()) {
			return false;
		}
		return std::ranges::all_of(value.Uses(), [&](const Use& use) {
			const auto* user = use.user;
			if (user == &root) {
				return true;
			}
			const auto op = user->GetOpcode();
			if (op != ValueOpcode::IAddCarry32 && op != ValueOpcode::CompositeExtractU32x2 &&
			    op != ValueOpcode::IAdd32) {
				return false;
			}
			return FlowsOnlyInto(*user, root, depth + 1u);
		});
	}

	static bool UsesOnly(const Inst& value, std::span<const Inst* const> users) {
		return !value.Uses().empty() && std::ranges::all_of(value.Uses(), [&](const Use& use) {
			return std::ranges::find(users, use.user) != users.end();
		});
	}

	// A single-dword scalar load, either through a V# (S_BUFFER_LOAD) or a raw address (S_LOAD).
	const MemoryInfo* ScalarReadMemory(const Inst& read, uint32_t& index) const {
		const auto op           = read.GetOpcode();
		uint32_t   zero         = 0;
		const bool buffer_read  = op == ValueOpcode::ReadConstBuffer && read.NumArgs() == 2u;
		const bool address_read = op == ValueOpcode::LoadAddressU32 && read.NumArgs() == 4u &&
		                          ImmediateU32(read.Arg(2), zero) && zero == 0u;
		if (!buffer_read && !address_read) {
			return nullptr;
		}
		index = read.Flags<MemoryFlags>().index;
		if (index >= m_program.memory_info.size()) {
			return nullptr;
		}
		const auto& memory = m_program.memory_info[index];
		const auto  kind   = buffer_read ? ResourceKind::ScalarBuffer : ResourceKind::ScalarAddress;
		return memory.kind == kind && memory.data_bits == 32u && memory.data_dwords == 1u ? &memory
		                                                                                  : nullptr;
	}

	// The shader-computed key of a bindless table lookup when its value provably stays below a
	// small bound, so the host can enumerate every entry the shader might pick. Keys are bit
	// indices of wave masks, masked or extracted fields, or wave-uniform selections among small
	// constants (a V_CNDMASK/S_CSELECT tree feeding V_READFIRSTLANE).
	// `active` is the lane mask a V_READFIRSTLANE above reads under: a VALU write merged with
	// that same mask (Select(exec, new, old)) only contributes its active-lane value.
	static bool BoundedKeyRange(Value key, uint32_t& count, uint32_t depth = 0,
	                            Value active = Value {}) {
		constexpr uint32_t MaxRange = 32u;
		key                         = key.Resolve();
		if (depth > 12u) {
			return false;
		}
		uint32_t immediate = 0;
		if (ImmediateU32(key, immediate)) {
			count = immediate + 1u;
			return immediate < MaxRange;
		}
		const auto* inst = key.TryInstruction();
		if (inst == nullptr) {
			return false;
		}
		const auto operand_bound = [&](size_t index, uint32_t& bound) {
			return index < inst->NumArgs() &&
			       BoundedKeyRange(inst->Arg(index), bound, depth + 1u, active);
		};
		switch (inst->GetOpcode()) {
			case ValueOpcode::FindILsb32:
				// S_FF1 over a 32-bit mask: a bit index, or -1 for an empty mask that no
				// waterfall loop ever looks up.
				count = 32u;
				return inst->NumArgs() == 1u;
			case ValueOpcode::BitwiseAnd32: {
				uint32_t mask = 0;
				if (inst->NumArgs() != 2u ||
				    !(ImmediateU32(inst->Arg(1), mask) || ImmediateU32(inst->Arg(0), mask)) ||
				    mask == 0u || mask >= MaxRange) {
					return false;
				}
				count = std::bit_ceil(mask + 1u);
				return true;
			}
			case ValueOpcode::BitFieldUExtract: {
				uint32_t width = 0;
				if (inst->NumArgs() != 3u || !ImmediateU32(inst->Arg(2), width) || width == 0u ||
				    (1u << width) > MaxRange) {
					return false;
				}
				count = 1u << width;
				return true;
			}
			case ValueOpcode::ReadFirstLane:
				if (inst->NumArgs() != 2u) {
					return false;
				}
				return BoundedKeyRange(inst->Arg(0), count, depth + 1u, inst->Arg(1).Resolve());
			case ValueOpcode::Phi: {
				// A register assigned on several paths (typically a default before a branch and
				// a selection inside it); a loop-carried value fails on its own recursion.
				count = 0;
				if (inst->NumArgs() == 0u) {
					return false;
				}
				for (size_t index = 0; index < inst->NumArgs(); index++) {
					uint32_t bound = 0;
					if (!operand_bound(index, bound)) {
						return false;
					}
					count = std::max(count, bound);
				}
				return true;
			}
			case ValueOpcode::SelectU32: {
				if (inst->NumArgs() != 3u) {
					return false;
				}
				if (!active.IsEmpty() && inst->Arg(0).Resolve() == active) {
					return operand_bound(1, count);
				}
				uint32_t on_true  = 0;
				uint32_t on_false = 0;
				if (!operand_bound(1, on_true) || !operand_bound(2, on_false)) {
					return false;
				}
				count = std::max(on_true, on_false);
				return true;
			}
			case ValueOpcode::BitwiseOr32: {
				uint32_t left  = 0;
				uint32_t right = 0;
				if (!operand_bound(0, left) || !operand_bound(1, right)) {
					return false;
				}
				count = std::bit_ceil(std::max(left, right));
				return count <= MaxRange;
			}
			case ValueOpcode::UMin32: {
				uint32_t   left          = 0;
				uint32_t   right         = 0;
				const bool left_bounded  = operand_bound(0, left);
				const bool right_bounded = operand_bound(1, right);
				if (!left_bounded && !right_bounded) {
					return false;
				}
				count = left_bounded && right_bounded ? std::min(left, right)
				        : left_bounded                ? left
				                                      : right;
				return true;
			}
			default: return false;
		}
	}

	// A comparison the shader branches on. Loop exits are S_CMP/V_CMP results feeding SCC or a
	// block terminator, so any other use of the compare is not evidence of a bound.
	// A comparison the shader branches on, directly or through the boolean logic that combines
	// it with other conditions (S_CSELECT of exec/0 on SCC, S_AND of vcc masks and so on).
	bool IsBranchCondition(const Inst& compare, uint32_t depth = 0) const {
		for (const auto& info: m_program.block_info) {
			if (info.condition.Resolve() == Value(&compare)) {
				return true;
			}
		}
		return std::ranges::any_of(compare.Uses(), [&](const Use& use) {
			switch (use.user->GetOpcode()) {
				case ValueOpcode::SetScc:
				case ValueOpcode::SetVcc:
				case ValueOpcode::SetExec:
				case ValueOpcode::Reference: return true;
				case ValueOpcode::LogicalNot:
				case ValueOpcode::LogicalAnd:
				case ValueOpcode::LogicalOr: break;
				case ValueOpcode::SelectU1:
					if (use.operand != 0u) {
						return false;
					}
					break;
				default: return false;
			}
			return depth < 4u && IsBranchCondition(*use.user, depth + 1u);
		});
	}

	// The shader-computed key of a table lookup when it is a loop counter: a Phi that starts at
	// an immediate and steps by one, with the loop leaving once a comparison against a
	// host-evaluable bound fails. The host then enumerates keys [0, bound), so the exact form
	// of the comparison only matters for the bound it names; an inclusive or post-increment
	// compare over-approximates by one entry, which the dense-prefix rule absorbs.
	bool LoopBoundedKey(Value key, uint32_t& count, Value& bound) {
		key             = ResolveForwardingPhis(key);
		const auto* phi = key.TryInstruction();
		if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi || phi->NumArgs() != 2u ||
		    phi->GetType() != Type::U32) {
			return false;
		}
		bool counter = false;
		for (uint32_t arm = 0; arm < 2u && !counter; arm++) {
			uint32_t init = 0;
			counter = ImmediateU32(phi->Arg(arm), init) && IsIncrementOf(phi->Arg(arm ^ 1u), key);
		}
		if (!counter) {
			return false;
		}
		const auto is_counter = [&](Value value) {
			return ResolveForwardingPhis(value) == key || IsIncrementOf(value, key);
		};
		for (const auto* block: m_program.blocks) {
			for (const auto& inst: *block) {
				bool inclusive = false;
				switch (inst.GetOpcode()) {
					case ValueOpcode::ULessThan32:
					case ValueOpcode::SLessThan32:
					case ValueOpcode::UGreaterThan32:
					case ValueOpcode::SGreaterThan32:
					case ValueOpcode::IEqual32:
					case ValueOpcode::INotEqual32: break;
					case ValueOpcode::ULessThanEqual32:
					case ValueOpcode::SLessThanEqual32:
					case ValueOpcode::UGreaterThanEqual32:
					case ValueOpcode::SGreaterThanEqual32: inclusive = true; break;
					default: continue;
				}
				if (inst.NumArgs() != 2u) {
					continue;
				}
				const bool first  = is_counter(inst.Arg(0));
				const bool second = is_counter(inst.Arg(1));
				if (first == second) {
					continue;
				}
				// The bound only sizes the host's enumeration, and the table entries it guards
				// are read without write-order proof too, so a loop-body store (which the order
				// analysis assumes may precede any later iteration's reads) does not disqualify it.
				const auto limit = inst.Arg(first ? 1u : 0u).Resolve();
				if (!ValidateRuntimeValue(m_program, limit, RuntimeValueType::Integer) ||
				    !IsBranchCondition(inst)) {
					continue;
				}
				bound = limit;
				if (inclusive) {
					auto& plus_one = m_program.value_storage.emplace_back(ValueOpcode::IAdd32);
					plus_one.SetArg(0, limit);
					plus_one.SetArg(1, Value(1u));
					bound = Value(&plus_one);
				}
				count = MaxLoopTableKeys;
				return true;
			}
		}
		return false;
	}

	bool MemoryIndexBelongsTo(uint32_t index, const Inst& owner) const {
		for (const auto* block: m_program.blocks) {
			for (const auto& inst: *block) {
				const auto op = inst.GetOpcode();
				if ((BufferAccessOf(op) == BufferAccess::None &&
				     AddressOpcodeInfoOf(op).access == AddressAccess::None &&
				     ImageOpcodeInfoOf(op).access == ImageAccess::None) ||
				    &inst == &owner) {
					continue;
				}
				if (inst.Flags<MemoryFlags>().index == index) {
					return false;
				}
			}
		}
		return true;
	}

	bool MakeRuntimeBufferSource(const Inst& handle, uint32_t pc, uint32_t& source,
	                             DescriptorSource& descriptor) {
		if (handle.GetOpcode() != ValueOpcode::GetBufferResource) {
			return false;
		}
		MakeSource(handle, 4u, false, false, descriptor, pc);
		uint32_t bad_dword = 0;
		if (!ValidateSource(descriptor, bad_dword)) {
			return false;
		}
		source = InternSource(descriptor);
		return true;
	}

	bool MakeRuntimeHeapSource(const Inst& handle, uint32_t pc, uint32_t& source,
	                           DescriptorSource& descriptor) {
		if (handle.GetOpcode() != ValueOpcode::GetAddressResource) {
			return MakeRuntimeBufferSource(handle, pc, source, descriptor);
		}
		MakeSource(handle, 2u, false, false, descriptor, pc);
		uint32_t bad_dword = 0;
		if (!ValidateSource(descriptor, bad_dword)) {
			return false;
		}
		source = InternSource(descriptor);
		return true;
	}

	bool MatchMaterialOffset(Value value, Value& selector, uint32_t& stride,
	                         uint32_t& offset) const {
		value           = value.Resolve();
		offset          = 0;
		auto* candidate = value.TryInstruction();
		if (candidate != nullptr && candidate->GetOpcode() == ValueOpcode::IAdd32 &&
		    candidate->NumArgs() == 2u) {
			uint32_t immediate = 0;
			if (ImmediateU32(candidate->Arg(0), immediate)) {
				value = candidate->Arg(1).Resolve();
			} else if (ImmediateU32(candidate->Arg(1), immediate)) {
				value = candidate->Arg(0).Resolve();
			} else {
				return false;
			}
			offset = immediate;
		}
		const auto* multiply = value.TryInstruction();
		if (multiply == nullptr || multiply->GetOpcode() != ValueOpcode::IMul32 ||
		    multiply->NumArgs() != 2u) {
			return false;
		}
		if (ImmediateU32(multiply->Arg(0), stride)) {
			selector = multiply->Arg(1).Resolve();
		} else if (ImmediateU32(multiply->Arg(1), stride)) {
			selector = multiply->Arg(0).Resolve();
		} else {
			return false;
		}
		const auto* selector_inst = selector.TryInstruction();
		return stride != 0u && selector_inst != nullptr &&
		       selector_inst->GetOpcode() == ValueOpcode::ReadFirstLane;
	}

	// Matches `handle` against a descriptor-table fetch: every dword is a scalar read of the same
	// heap at `table_offset + (key << log2(entry_bytes)) + dword * 4`. The key is either a
	// material-record lookup or a shader-computed value with a provable bound.
	bool EquivalentHandle(const Inst& lhs, const Inst& rhs) const {
		if (lhs.GetOpcode() != rhs.GetOpcode() || lhs.NumArgs() != rhs.NumArgs()) {
			return false;
		}
		for (size_t arg = 0; arg < lhs.NumArgs(); arg++) {
			if (!EquivalentValue(m_program, lhs.Arg(arg), rhs.Arg(arg))) {
				return false;
			}
		}
		return true;
	}

	// A descriptor waterfall over a per-lane choice among a few fixed descriptors (a cascade or
	// LOD pick): every dword is V_READFIRSTLANE of a value each lane selects, through EXEC merges
	// and control-flow phis, from the same small set of runtime-evaluable descriptors. Those are
	// the candidates; a parallel phi/select tree yields each lane's candidate index, and the key
	// is that index in the first active lane.
	bool TryMakeSelectTable(Inst& handle, uint32_t width, uint32_t pc, IndirectTablePlan& plan) {
		constexpr uint32_t MaxSelectCandidates = 16u;
		// The same shape without the waterfall: a scalar descriptor chosen by control flow the
		// host cannot evaluate (phis over fixed descriptors). The index is then used directly.
		std::array<Value, 8> lanes {};
		Value                mask;
		bool                 waterfall = true;
		for (uint32_t dword = 0; dword < width && waterfall; dword++) {
			const auto* read = handle.Arg(dword).Resolve().TryInstruction();
			if (read == nullptr || read->GetOpcode() != ValueOpcode::ReadFirstLane ||
			    read->NumArgs() != 2u || (dword != 0u && read->Arg(1).Resolve() != mask)) {
				waterfall = false;
				break;
			}
			if (dword == 0u) {
				mask = read->Arg(1).Resolve();
			}
			lanes[dword] = read->Arg(0).Resolve();
		}
		if (!waterfall) {
			bool evaluable = true;
			for (uint32_t dword = 0; dword < width; dword++) {
				lanes[dword] = handle.Arg(dword).Resolve();
				evaluable    = evaluable && (lanes[dword].IsImmediate() ||
				                             ValidateRuntimeValue(m_program, lanes[dword]));
			}
			if (evaluable) {
				return false;
			}
		}
		const auto opcode_of = [](const Value& value) {
			const auto* inst = value.TryInstruction();
			return inst != nullptr ? inst->GetOpcode() : ValueOpcode::Void;
		};
		const auto branches = [&](const Value& value) {
			const auto op = opcode_of(value);
			return op == ValueOpcode::Phi || op == ValueOpcode::SelectU32;
		};
		std::vector<std::array<Value, 8>>      leaves;
		std::unordered_map<const Inst*, Value> built;
		bool                                   ok = true;
		// Mirrors the value tree of all dwords at once. A dword that is the same in every
		// branch stays a plain value while the others split.
		const auto build = [&](const auto& self, const std::array<Value, 8>& tuple,
		                       uint32_t depth) -> Value {
			if (!ok || depth > 64u) {
				ok = false;
				return Value(0u);
			}
			const auto driver = std::find_if(tuple.begin(), tuple.begin() + width, branches);
			if (driver == tuple.begin() + width) {
				for (uint32_t index = 0; index < leaves.size(); index++) {
					bool same = true;
					for (uint32_t dword = 0; dword < width && same; dword++) {
						same = EquivalentValue(m_program, leaves[index][dword], tuple[dword]);
					}
					if (same) {
						return Value(index);
					}
				}
				for (uint32_t dword = 0; dword < width; dword++) {
					if (!tuple[dword].IsImmediate() &&
					    !ValidateRuntimeValue(m_program, tuple[dword])) {
						ok = false;
						return Value(0u);
					}
				}
				if (leaves.size() >= MaxSelectCandidates) {
					ok = false;
					return Value(0u);
				}
				leaves.push_back(tuple);
				return Value(static_cast<uint32_t>(leaves.size() - 1u));
			}
			auto*      node = driver->TryInstruction();
			const auto op   = node->GetOpcode();
			if (const auto found = built.find(node); found != built.end()) {
				return found->second;
			}
			// Every splitting dword must split at the same point.
			for (uint32_t dword = 0; dword < width; dword++) {
				if (!branches(tuple[dword])) {
					continue;
				}
				const auto* other = tuple[dword].TryInstruction();
				if (other->GetOpcode() != op || other->NumArgs() != node->NumArgs() ||
				    other->Parent() != node->Parent() ||
				    (op == ValueOpcode::SelectU32 &&
				     other->Arg(0).Resolve() != node->Arg(0).Resolve())) {
					ok = false;
					return Value(0u);
				}
				for (size_t arg = 0; op == ValueOpcode::Phi && arg < node->NumArgs(); arg++) {
					if (other->PhiBlock(arg) != node->PhiBlock(arg)) {
						ok = false;
						return Value(0u);
					}
				}
			}
			const auto sub = [&](size_t arg) {
				std::array<Value, 8> next = tuple;
				for (uint32_t dword = 0; dword < width; dword++) {
					if (branches(tuple[dword])) {
						next[dword] = tuple[dword].TryInstruction()->Arg(arg).Resolve();
					}
				}
				return next;
			};
			auto* block = node->Parent();
			if (op == ValueOpcode::Phi) {
				auto& phi = *block->PrependNewInst(block->begin(), ValueOpcode::Phi);
				phi.SetFlags(Type::U32);
				built.emplace(node, Value(&phi));
				for (size_t arg = 0; arg < node->NumArgs(); arg++) {
					phi.AddPhiOperand(node->PhiBlock(arg), self(self, sub(arg), depth + 1u));
				}
				return Value(&phi);
			}
			const auto chosen = self(self, sub(1), depth + 1u);
			const auto other  = self(self, sub(2), depth + 1u);
			const auto where  = std::ranges::find_if(
			    block->Instructions(), [&](const Inst& candidate) { return &candidate == node; });
			auto& select = *block->PrependNewInst(where, ValueOpcode::SelectU32,
			                                      {node->Arg(0), chosen, other});
			built.emplace(node, Value(&select));
			return Value(&select);
		};
		const auto index = build(build, lanes, 0u);
		if (!ok || leaves.size() < 2u) {
			// The phis already built are unused and fall to dead-code elimination.
			return false;
		}
		auto*      block = handle.Parent();
		const auto where = std::ranges::find_if(
		    block->Instructions(), [&](const Inst& candidate) { return &candidate == &handle; });
		const auto key =
		    waterfall
		        ? Value(&*block->PrependNewInst(where, ValueOpcode::ReadFirstLane, {index, mask}))
		        : index;

		std::vector<uint32_t> candidates;
		for (const auto& leaf: leaves) {
			DescriptorSource candidate;
			candidate.dword_count = width;
			candidate.dwords.fill(Value(0u));
			for (uint32_t dword = 0; dword < width; dword++) {
				candidate.dwords[dword] = leaf[dword];
				if (!leaf[dword].IsImmediate()) {
					plan.pins.push_back(leaf[dword]);
				}
			}
			candidates.push_back(InternSource(candidate));
		}
		DescriptorSource table_source;
		table_source.dword_count = 8u;
		table_source.dwords.fill(Value(0u));
		table_source.indirect_table = DescriptorSource::IndirectTable {
		    .key_count         = static_cast<uint32_t>(leaves.size()),
		    .entry_dwords      = width,
		    .sampler           = handle.GetOpcode() == ValueOpcode::GetSamplerResource,
		    .candidate_sources = std::move(candidates),
		};
		plan.handle = &handle;
		plan.width  = width;
		plan.source = InternSource(table_source);
		plan.key    = key;
		plan.roots  = table_source.dwords;
		plan.select = true;
		(void)pc;
		return true;
	}

	bool TryMakeIndirectTable(Inst& handle, uint32_t pc, IndirectTablePlan& plan) {
		uint32_t width = 0;
		if (handle.GetOpcode() == ValueOpcode::GetImageResource && handle.NumArgs() == 8u) {
			width = 8u;
		} else if ((handle.GetOpcode() == ValueOpcode::GetBufferResource ||
		            handle.GetOpcode() == ValueOpcode::GetSamplerResource) &&
		           handle.NumArgs() == 4u) {
			width = 4u;
		} else {
			return false;
		}

		if (TryMakeSelectTable(handle, width, pc, plan)) {
			return true;
		}
		std::array<Inst*, 8> heap_reads {};
		Inst*                heap_handle = nullptr;
		Value                heap_offset;
		uint32_t             table_offset = 0;
		for (uint32_t dword = 0; dword < width; dword++) {
			heap_reads[dword] = handle.Arg(dword).Resolve().TryInstruction();
			if (heap_reads[dword] == nullptr) {
				return false;
			}
			uint32_t    memory_index = 0;
			const auto* memory       = ScalarReadMemory(*heap_reads[dword], memory_index);
			// The table may sit at a fixed offset inside the heap (e.g. an S_LOAD immediate).
			if (dword == 0u && memory != nullptr) {
				table_offset = memory->offset;
			}
			if (memory == nullptr || (table_offset % sizeof(uint32_t)) != 0u ||
			    memory->offset != table_offset + dword * sizeof(uint32_t) ||
			    !MemoryIndexBelongsTo(memory_index, *heap_reads[dword])) {
				return false;
			}
			// Split loads of one entry each carry their own (equivalent) heap handle.
			auto* current_handle = heap_reads[dword]->Arg(0).Resolve().TryInstruction();
			if (current_handle == nullptr ||
			    (heap_handle != nullptr && current_handle != heap_handle &&
			     !EquivalentHandle(*current_handle, *heap_handle))) {
				return false;
			}
			heap_handle = current_handle;
			if (dword == 0u) {
				heap_offset = heap_reads[dword]->Arg(1).Resolve();
			} else if (!EquivalentValue(m_program, heap_offset, heap_reads[dword]->Arg(1))) {
				return false;
			}
			plan.memory[dword] = memory_index;
			plan.reads[dword]  = heap_reads[dword];
		}

		// Pointer hop: the descriptors are S_LOADed at fixed offsets through an address that the
		// shader itself fetched from a keyed record (S_BUFFER_LOAD of two dwords at
		// key * stride). The record table is then the heap and the address field is followed
		// per key on the host.
		bool                 via_pointer    = false;
		uint32_t             pointer_offset = 0;
		std::array<Inst*, 2> pointer_reads {};
		if (uint32_t inner = 0; heap_handle->GetOpcode() == ValueOpcode::GetAddressResource &&
		                        heap_handle->NumArgs() == 2u && ImmediateU32(heap_offset, inner) &&
		                        (inner % sizeof(uint32_t)) == 0u) {
			Inst* outer_handle = nullptr;
			Value outer_offset;
			bool  matched = true;
			for (uint32_t half = 0; half < 2u && matched; half++) {
				auto*    read         = heap_handle->Arg(half).Resolve().TryInstruction();
				uint32_t memory_index = 0;
				// The record table is a V# (S_BUFFER_LOAD) or a raw address (S_LOAD).
				const auto* memory =
				    read != nullptr ? ScalarReadMemory(*read, memory_index) : nullptr;
				if (half == 0u && memory != nullptr) {
					pointer_offset = memory->offset;
				}
				auto* read_handle =
				    memory != nullptr ? read->Arg(0).Resolve().TryInstruction() : nullptr;
				matched = memory != nullptr && read_handle != nullptr &&
				          (read_handle->GetOpcode() == ValueOpcode::GetBufferResource ||
				           read_handle->GetOpcode() == ValueOpcode::GetAddressResource) &&
				          memory->offset == pointer_offset + half * sizeof(uint32_t) &&
				          (half == 0u || (read_handle == outer_handle &&
				                          EquivalentValue(m_program, outer_offset, read->Arg(1))));
				if (matched && half == 0u) {
					outer_handle = read_handle;
					outer_offset = read->Arg(1).Resolve();
				}
				pointer_reads[half] = read;
			}
			if (matched) {
				via_pointer = true;
				table_offset += inner;
				heap_handle = outer_handle;
				heap_offset = outer_offset;
			}
		}
		// The dynamic offset is `key * entry_stride` (a shift for a packed table, a multiply
		// when the descriptor heads a larger per-entry record), optionally plus an immediate
		// table base the compiler folded into the address rather than the load
		// (S_LSHL4_ADD_U32 and co).
		const auto* shift        = heap_offset.TryInstruction();
		const Inst* offset_root  = shift;
		uint32_t    entry_stride = 0;
		uint32_t&   folded       = via_pointer ? pointer_offset : table_offset;
		Value       add_lhs;
		Value       add_rhs;
		if (MatchAdd(heap_offset, add_lhs, add_rhs)) {
			uint32_t immediate = 0;
			if (ImmediateU32(add_rhs, immediate)) {
				shift = add_lhs.TryInstruction();
			} else if (ImmediateU32(add_lhs, immediate)) {
				shift = add_rhs.TryInstruction();
			} else {
				return false;
			}
			if ((immediate % sizeof(uint32_t)) != 0u) {
				return false;
			}
			folded += immediate;
		}
		const uint32_t min_stride = via_pointer ? 2u * sizeof(uint32_t) : width * sizeof(uint32_t);
		if (shift == nullptr || shift->NumArgs() != 2u || !MatchEntryStride(*shift, entry_stride) ||
		    entry_stride < min_stride) {
			return false;
		}
		auto* material_read = shift->Arg(0).Resolve().TryInstruction();
		if (material_read == nullptr) {
			return false;
		}
		std::array<const Inst*, 10> heap_users {};
		std::copy(heap_reads.begin(), heap_reads.end(), heap_users.begin());
		heap_users[8] = pointer_reads[0];
		heap_users[9] = pointer_reads[1];
		const std::array<const Inst*, 1> table_users {&handle};
		uint32_t                         key_count = 0;
		Value                            key_bound;
		bool bounded      = BoundedKeyRange(Value(material_read), key_count) ||
		                    LoopBoundedKey(Value(material_read), key_count, key_bound);
		bool heap_bounded = false;
		if (!bounded && heap_handle->GetOpcode() == ValueOpcode::GetBufferResource) {
			// Neither a small computed range nor a material-record read (that form is matched
			// below): fall back to the heap V#'s size, which S_BUFFER_LOAD bounds-checks.
			uint32_t memory_index = 0;
			if (ScalarReadMemory(*material_read, memory_index) == nullptr) {
				bounded      = true;
				heap_bounded = true;
				key_count    = MaxHeapTableKeys;
			}
		}
		if (bounded) {
			// A waterfall over a small shader-computed index (typically S_FF1 over a wave mask
			// of per-lane texture indices) or a loop counter rather than a material-record
			// lookup: enumerate the table entries the key can reach instead of probing material
			// records. The key may have other uses (loop bookkeeping, lane matching), only the
			// offset must be ours. The offset register is often carried around the loop by a
			// Phi as well; that is SSA bookkeeping, not another consumer of the table entry.
			// When the descriptor heads a larger record, the same offset also addresses the
			// record's other fields; those are ordinary scalar data loads.
			// Offset arithmetic (record base + another field's immediate) that only addresses
			// further scalar reads of the record counts as the same kind of use.
			const auto offset_is_ours = [&](const Inst& value) {
				const auto is_ours = [&](const auto& self, const Inst& current,
				                         uint32_t depth) -> bool {
					return !current.Uses().empty() &&
					       std::ranges::all_of(current.Uses(), [&](const Use& use) {
						       uint32_t   memory_index = 0;
						       const auto op           = use.user->GetOpcode();
						       return op == ValueOpcode::Phi ||
						              std::ranges::find(heap_users, use.user) != heap_users.end() ||
						              (use.operand == 1u &&
						               ScalarReadMemory(*use.user, memory_index) != nullptr) ||
						              (depth < 2u && op == ValueOpcode::IAdd32 &&
						               self(self, *use.user, depth + 1u));
					       });
				};
				return is_ours(is_ours, value, 0u);
			};
			// The scaled key itself may also address sibling tables of the same record.
			if (!offset_is_ours(*offset_root) ||
			    (offset_root != shift && !FlowsOnlyInto(*shift, *offset_root) &&
			     !offset_is_ours(*shift))) {
				return false;
			}
			// One fetched entry may feed several accesses (e.g. a loop body sampling the
			// selected texture more than once); each handle becomes a plan over the same table.
			for (uint32_t dword = 0; dword < width; dword++) {
				const auto& read = *heap_reads[dword];
				if (read.Uses().empty() || !std::ranges::all_of(read.Uses(), [&](const Use& use) {
					    return use.user->GetOpcode() == handle.GetOpcode() &&
					           use.user->NumArgs() == width;
				    })) {
					return false;
				}
			}
			DescriptorSource heap_source;
			uint32_t         heap_source_index = 0;
			if (!MakeRuntimeHeapSource(*heap_handle, pc, heap_source_index, heap_source)) {
				return false;
			}
			DescriptorSource table_source;
			table_source.dword_count = 8u;
			table_source.dwords.fill(Value(0u));
			std::copy(heap_source.dwords.begin(),
			          heap_source.dwords.begin() + heap_source.dword_count,
			          table_source.dwords.begin() + 4u);
			table_source.indirect_table = DescriptorSource::IndirectTable {
			    .material_source = heap_source_index,
			    .heap_source     = heap_source_index,
			    .key_count       = key_count,
			    .heap_offset     = table_offset,
			    .entry_dwords    = width,
			    .entry_stride    = entry_stride,
			    .key_bound       = key_bound,
			    .heap_bounded    = heap_bounded,
			    .via_pointer     = via_pointer,
			    .pointer_offset  = pointer_offset,
			    .sampler         = handle.GetOpcode() == ValueOpcode::GetSamplerResource,
			};
			plan.handle = &handle;
			plan.width  = width;
			plan.source = InternSource(table_source);
			plan.key    = Value(material_read);
			plan.roots  = table_source.dwords;
			return true;
		}
		if (offset_root != shift || entry_stride != (width * sizeof(uint32_t)) || via_pointer ||
		    handle.GetOpcode() == ValueOpcode::GetSamplerResource) {
			// Material-record keys index a packed heap directly; a folded immediate or a wider
			// record belongs to the bounded forms only.
			return false;
		}
		uint32_t    material_memory_index = 0;
		const auto* material_memory       = ScalarReadMemory(*material_read, material_memory_index);
		if (material_memory == nullptr ||
		    !MemoryIndexBelongsTo(material_memory_index, *material_read)) {
			return false;
		}
		// The bindless key does not have to be the first dword of the material record; some
		// games (e.g. Demon's Souls) keep several texture-heap indices at different fixed
		// offsets within the same record.
		const uint32_t material_offset = material_memory->offset;
		auto*          material_handle = material_read->Arg(0).Resolve().TryInstruction();
		if (material_handle == nullptr) {
			return false;
		}

		Value    selector;
		uint32_t selector_stride = 0;
		uint32_t selector_offset = 0;
		if (!MatchMaterialOffset(material_read->Arg(1), selector, selector_stride,
		                         selector_offset)) {
			return false;
		}

		const std::array<const Inst*, 1> material_users {shift};
		if (!UsesOnly(*material_read, material_users) || !UsesOnly(*shift, heap_users)) {
			return false;
		}
		for (uint32_t dword = 0; dword < width; dword++) {
			if (!UsesOnly(*heap_reads[dword], table_users)) {
				return false;
			}
		}

		DescriptorSource material_source;
		DescriptorSource heap_source;
		uint32_t         material_source_index = 0;
		uint32_t         heap_source_index     = 0;
		if (!MakeRuntimeBufferSource(*material_handle, pc, material_source_index,
		                             material_source) ||
		    !MakeRuntimeHeapSource(*heap_handle, pc, heap_source_index, heap_source)) {
			return false;
		}

		DescriptorSource table_source;
		table_source.dword_count = 8u;
		table_source.dwords.fill(Value(0u));
		std::copy(material_source.dwords.begin(), material_source.dwords.begin() + 4u,
		          table_source.dwords.begin());
		std::copy(heap_source.dwords.begin(), heap_source.dwords.begin() + heap_source.dword_count,
		          table_source.dwords.begin() + 4u);
		table_source.indirect_table = DescriptorSource::IndirectTable {
		    .material_source = material_source_index,
		    .heap_source     = heap_source_index,
		    .selector_stride = selector_stride,
		    .selector_offset = selector_offset,
		    .material_offset = material_offset,
		    .heap_offset     = table_offset,
		    .entry_dwords    = width,
		    .entry_stride    = entry_stride,
		};

		plan.handle = &handle;
		plan.width  = width;
		plan.source = InternSource(table_source);
		plan.key    = Value(material_read);
		plan.roots  = table_source.dwords;
		return true;
	}

	const IndirectTablePlan* FindIndirectTable(const Inst& handle) const {
		const auto found =
		    std::find_if(m_indirect_tables.begin(), m_indirect_tables.end(),
		                 [&](const IndirectTablePlan& plan) { return plan.handle == &handle; });
		return found == m_indirect_tables.end() ? nullptr : &*found;
	}

	bool IsIndirectPlanningMemory(uint32_t index) const {
		return std::any_of(
		    m_indirect_tables.begin(), m_indirect_tables.end(), [&](const IndirectTablePlan& plan) {
			    return !plan.select &&
			           std::find(plan.memory.begin(), plan.memory.begin() + plan.width, index) !=
			               plan.memory.begin() + plan.width;
		    });
	}

	void PlanIndirectTables() {
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				const auto op = inst.GetOpcode();
				if ((ImageOpcodeInfoOf(op).access == ImageAccess::None &&
				     BufferAccessOf(op) == BufferAccess::None) ||
				    inst.NumArgs() == 0u) {
					continue;
				}
				const auto pc = inst.Flags<MemoryFlags>().pc;
				if (ImageOpcodeInfoOf(op).needs_sampler && inst.NumArgs() >= 2u) {
					auto*             sampler = inst.Arg(1).Resolve().TryInstruction();
					IndirectTablePlan sampler_plan;
					if (sampler != nullptr && FindIndirectTable(*sampler) == nullptr &&
					    TryMakeIndirectTable(*sampler, pc, sampler_plan)) {
						m_indirect_tables.push_back(std::move(sampler_plan));
					}
				}
				auto* handle = inst.Arg(0).Resolve().TryInstruction();
				if (handle == nullptr || FindIndirectTable(*handle) != nullptr) {
					continue;
				}
				// The emitter switches over the table's candidates only for these image
				// operations; leave every other user to the plain (single-descriptor) path.
				if (handle->GetOpcode() == ValueOpcode::GetImageResource &&
				    !std::ranges::all_of(handle->Uses(), [](const Use& use) {
					    const auto user = use.user->GetOpcode();
					    return user == ValueOpcode::ImageSampleRaw ||
					           user == ValueOpcode::ImageRead || user == ValueOpcode::ImageWrite;
				    })) {
					continue;
				}
				IndirectTablePlan plan;
				if (TryMakeIndirectTable(*handle, pc, plan)) {
					m_indirect_tables.push_back(std::move(plan));
				}
			}
		}
	}

	void GetHandle(Value value, ValueOpcode expected, uint32_t width, uint32_t pc, Inst*& handle,
	               uint32_t& source, bool sampler = false, bool sample_adjust = false) {
		handle = value.Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != expected) {
			Fail(pc, fmt::format("memory operation requires {}", ValueOpcodeName(expected)));
		}
		DescriptorSource descriptor;
		MakeSource(*handle, width, sampler, sample_adjust, descriptor, pc);
		uint32_t bad_dword = 0;
		if (expected == ValueOpcode::GetImageResource) {
			for (; bad_dword < descriptor.dword_count; bad_dword++) {
				const auto* value = descriptor.dwords[bad_dword].Resolve().TryInstruction();
				if (value != nullptr && value->GetOpcode() == ValueOpcode::ReadConstBuffer) {
					Fail(pc, fmt::format("{} dword {} is not a valid runtime value",
					                     ValueOpcodeName(expected), bad_dword));
				}
			}
			bad_dword = 0;
		}
		if (!ValidateSource(descriptor, bad_dword)) {
			Fail(pc, fmt::format("{} dword {} is not a valid runtime value ({})",
			                     ValueOpcodeName(expected), bad_dword,
			                     DescribeInvalidValue(descriptor.dwords[bad_dword], 0u)));
		}
		source = InternSource(descriptor);
	}

	// Names the instruction that makes a value host-unevaluable: the first one along the chain
	// that fails validation although every operand passes.
	std::string DescribeInvalidValue(Value value, uint32_t depth) const {
		value            = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			return "non-instruction value";
		}
		if (depth < 24u) {
			for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
				const auto operand = inst->Arg(arg).Resolve();
				if (operand.TryInstruction() != nullptr &&
				    operand.TryInstruction()->GetType() != Type::Void &&
				    !ValidateRuntimeValue(m_program, operand)) {
					return DescribeInvalidValue(operand, depth + 1u);
				}
			}
		}
		return fmt::format("unevaluable {} with {} operands", ValueOpcodeName(inst->GetOpcode()),
		                   inst->NumArgs());
	}

	void ValidateAddressHandle(Value value, uint32_t pc) const {
		const auto* handle = value.Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetAddressResource) {
			Fail(pc, "address operation requires GetAddressResource");
		}
		if (handle->NumArgs() != 2) {
			Fail(pc, "GetAddressResource must have two address dwords");
		}
	}

	uint32_t AddBuffer(uint32_t source, const MemoryInfo& memory, ValueOpcode op, uint32_t pc) {
		for (uint32_t i = 0; i < m_info.buffers.size(); i++) {
			if (m_info.buffers[i].source == source) {
				Merge(m_info.buffers[i], memory, op, pc);
				return i;
			}
		}
		if (m_info.buffers.size() >= ShaderInfo::MaxBuffers) {
			return UINT32_MAX;
		}
		BufferResource resource;
		resource.source       = source;
		resource.first_use_pc = pc;
		Merge(resource, memory, op, pc);
		m_info.buffers.push_back(resource);
		return static_cast<uint32_t>(m_info.buffers.size() - 1);
	}

	static void Merge(BufferResource& resource, const MemoryInfo& memory, ValueOpcode op,
	                  uint32_t pc) {
		const auto access        = BufferAccessOf(op);
		const bool atomic        = access == BufferAccess::Atomic;
		const bool write         = access == BufferAccess::Write || atomic;
		resource.first_use_pc    = std::min(resource.first_use_pc, pc);
		resource.max_byte_extent = std::max(resource.max_byte_extent, ByteExtent(memory));
		resource.read            = resource.read || !write || atomic;
		resource.written         = resource.written || write;
		resource.atomic          = resource.atomic || atomic;
		resource.formatted       = resource.formatted || memory.formatted;
		resource.scalar          = resource.scalar || op == ValueOpcode::ReadConstBuffer ||
		                           memory.kind == ResourceKind::ScalarBuffer;
	}

	uint32_t AddImage(uint32_t source, const MemoryInfo& memory, ValueOpcode op, uint32_t pc) {
		const auto resource_class = ImageOpcodeInfoOf(op).resource_class;
		const auto mip   = resource_class == ImageResourceClass::Storage && memory.image_has_mip
		                       ? ImageMipMode::DynamicStorage
		                       : ImageMipMode::None;
		const bool depth = (memory.image_sample_flags & Decoder::ImageSampleFlagCompare) != 0;
		for (uint32_t i = 0; i < m_info.images.size(); i++) {
			auto& image = m_info.images[i];
			if (image.source == source && image.resource_class == resource_class &&
			    image.dimension == memory.image_dimension && image.mip_mode == mip &&
			    image.depth_compare == depth && image.r128 == memory.image_r128) {
				Merge(image, op, pc);
				return i;
			}
		}
		if (m_info.images.size() >= ShaderInfo::MaxImages) {
			return UINT32_MAX;
		}
		ImageResource image;
		image.source         = source;
		image.first_use_pc   = pc;
		image.resource_class = resource_class;
		image.dimension      = memory.image_dimension;
		image.mip_mode       = mip;
		image.depth_compare  = depth;
		image.r128           = memory.image_r128;
		Merge(image, op, pc);
		m_info.images.push_back(image);
		return static_cast<uint32_t>(m_info.images.size() - 1);
	}

	static void Merge(ImageResource& image, ValueOpcode op, uint32_t pc) {
		const auto access  = ImageOpcodeInfoOf(op).access;
		const bool atomic  = access == ImageAccess::Atomic;
		const bool write   = access == ImageAccess::Write || atomic;
		image.first_use_pc = std::min(image.first_use_pc, pc);
		image.read         = image.read || !write || atomic;
		image.written      = image.written || write;
		image.atomic       = image.atomic || atomic;
	}

	uint32_t AddSampler(uint32_t source, uint32_t pc) {
		for (uint32_t i = 0; i < m_info.samplers.size(); i++) {
			if (m_info.samplers[i].source == source) {
				m_info.samplers[i].first_use_pc = std::min(m_info.samplers[i].first_use_pc, pc);
				return i;
			}
		}
		if (m_info.samplers.size() >= ShaderInfo::MaxSamplers) {
			return UINT32_MAX;
		}
		m_info.samplers.push_back({source, pc});
		return static_cast<uint32_t>(m_info.samplers.size() - 1);
	}

	void AddSampledPair(uint32_t image, uint32_t sampler, uint32_t pc) {
		for (auto& pair: m_info.sampled_pairs) {
			if (pair.image == image && pair.sampler == sampler) {
				pair.first_use_pc = std::min(pair.first_use_pc, pc);
				return;
			}
		}
		if (m_info.sampled_pairs.size() >= ShaderInfo::MaxSampledPairs) {
			Fail(pc, "sampled image/sampler pair limit exceeded");
		}
		m_info.sampled_pairs.push_back({image, sampler, pc});
	}

	void AddHandlePatch(Inst* handle, uint32_t resource, uint32_t pc) {
		for (const auto& patch: m_handle_patches) {
			if (patch.handle == handle) {
				if (patch.resource != resource) {
					Fail(pc, fmt::format("{} is reused with incompatible resource classes",
					                     ValueOpcodeName(handle->GetOpcode())));
				}
				return;
			}
		}
		m_handle_patches.push_back({handle, resource});
	}

	void AddMemoryPatch(uint32_t index, uint32_t resource, uint32_t sampler, bool has_sampler,
	                    uint32_t pc) {
		for (auto& patch: m_memory_patches) {
			if (patch.index != index) {
				continue;
			}
			if (patch.resource != resource ||
			    (has_sampler && patch.has_sampler && patch.sampler != sampler)) {
				Fail(pc, "memory metadata is reused with incompatible resources");
			}
			if (has_sampler) {
				patch.sampler     = sampler;
				patch.has_sampler = true;
			}
			return;
		}
		m_memory_patches.push_back({index, resource, sampler, has_sampler});
	}

	void Collect(Inst& inst) {
		const auto op           = inst.GetOpcode();
		const auto buffer       = BufferAccessOf(op);
		const auto address_info = AddressOpcodeInfoOf(op);
		const auto image_info   = ImageOpcodeInfoOf(op);
		if (buffer == BufferAccess::None && address_info.access == AddressAccess::None &&
		    image_info.access == ImageAccess::None) {
			return;
		}
		const auto flags = inst.Flags<MemoryFlags>();
		if (flags.index >= m_program.memory_info.size()) {
			Fail(flags.pc, fmt::format("memory metadata index {} is out of range", flags.index));
		}
		if (inst.NumArgs() == 0) {
			Fail(flags.pc, "memory operation has no resource handle");
		}
		const auto& memory = m_program.memory_info[flags.index];
		if (memory.planning_only || IsIndirectPlanningMemory(flags.index)) {
			return;
		}
		Inst*    handle   = nullptr;
		uint32_t source   = 0;
		uint32_t resource = 0;

		if (buffer != BufferAccess::None) {
			handle               = inst.Arg(0).Resolve().TryInstruction();
			const auto* indirect = handle != nullptr ? FindIndirectTable(*handle) : nullptr;
			if (indirect != nullptr) {
				// Formatted accesses are fine: specialization makes every candidate share the
				// exemplar's stride, format and swizzle, which is all the access sequence reads.
				source = indirect->source;
			} else {
				GetHandle(inst.Arg(0), ValueOpcode::GetBufferResource, 4, flags.pc, handle, source);
			}
			resource = AddBuffer(source, memory, op, flags.pc);
			if (resource == UINT32_MAX) {
				Fail(flags.pc, "buffer resource limit exceeded");
			}
			AddHandlePatch(handle, resource, flags.pc);
			AddMemoryPatch(flags.index, resource, 0, false, flags.pc);
			return;
		}
		if (address_info.access != AddressAccess::None) {
			if (!IsAddressResourceKind(memory.kind)) {
				Fail(flags.pc, "address operation has invalid resource kind");
			}
			if (memory.kind == ResourceKind::Scratch) {
				handle = inst.Arg(0).Resolve().TryInstruction();
				if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetScratchResource ||
				    handle->NumArgs() != 0) {
					Fail(flags.pc, "scratch operation requires GetScratchResource");
				}
				if (m_program.scratch_dwords == 0) {
					Fail(flags.pc, "scratch operation requires a nonzero AGC per-thread size");
				}
				return;
			}
			ValidateAddressHandle(inst.Arg(0), flags.pc);
			m_info.uses_dma = true;
			return;
		}

		if (memory.kind != ResourceKind::Image ||
		    image_info.resource_class == ImageResourceClass::None) {
			Fail(flags.pc, "image operation has invalid resource kind");
		}
		handle               = inst.Arg(0).Resolve().TryInstruction();
		const auto* indirect = handle != nullptr ? FindIndirectTable(*handle) : nullptr;
		if (indirect != nullptr) {
			source = indirect->source;
		} else {
			GetHandle(inst.Arg(0), ValueOpcode::GetImageResource, 8, flags.pc, handle, source);
		}
		resource = AddImage(source, memory, op, flags.pc);
		if (resource == UINT32_MAX) {
			Fail(flags.pc, "image resource limit exceeded");
		}
		AddHandlePatch(handle, resource, flags.pc);
		uint32_t sampler = 0;
		if (image_info.needs_sampler) {
			if (inst.NumArgs() < 2) {
				Fail(flags.pc, "sampled image operation has no sampler handle");
			}
			Inst*      sampler_handle = nullptr;
			uint32_t   sampler_source = 0;
			const bool sample_adjust =
			    (memory.image_sample_flags & Decoder::ImageSampleFlagAdjust) != 0;
			sampler_handle = inst.Arg(1).Resolve().TryInstruction();
			const auto* sampler_table =
			    sampler_handle != nullptr ? FindIndirectTable(*sampler_handle) : nullptr;
			if (sampler_table != nullptr) {
				sampler_source = sampler_table->source;
			} else {
				GetHandle(inst.Arg(1), ValueOpcode::GetSamplerResource, 4, flags.pc, sampler_handle,
				          sampler_source, true, sample_adjust);
			}
			sampler = AddSampler(sampler_source, flags.pc);
			if (sampler == UINT32_MAX) {
				Fail(flags.pc, "sampler resource limit exceeded");
			}
			AddHandlePatch(sampler_handle, sampler, flags.pc);
			AddSampledPair(resource, sampler, flags.pc);
		}
		AddMemoryPatch(flags.index, resource, sampler, image_info.needs_sampler, flags.pc);
	}

	const DescriptorSource* Source(uint32_t source) const {
		return source < m_sources.size() ? &m_sources[source] : nullptr;
	}

	void LinkImageAliases() {
		for (auto& buffer: m_info.buffers) {
			const auto* buffer_source = Source(buffer.source);
			if (buffer_source == nullptr || buffer_source->dword_count != 4 ||
			    buffer_source->indirect_table.has_value()) {
				continue;
			}
			for (uint32_t image = 0; image < m_info.images.size(); image++) {
				const auto* image_source = Source(m_info.images[image].source);
				if (image_source == nullptr || image_source->dword_count != 8 ||
				    image_source->indirect_table.has_value()) {
					continue;
				}
				bool alias = true;
				for (uint32_t dword = 0; dword < 4; dword++) {
					alias = alias && EquivalentValue(m_program, buffer_source->dwords[dword],
					                                 image_source->dwords[dword]);
				}
				if (alias) {
					buffer.image_alias = image;
					break;
				}
			}
		}
	}

	Program&                                                         m_program;
	ShaderInfo                                                       m_info;
	std::vector<DescriptorSource>                                    m_sources;
	std::vector<HandlePatch>                                         m_handle_patches;
	std::vector<MemoryPatch>                                         m_memory_patches;
	std::vector<IndirectTablePlan>                                   m_indirect_tables;
	std::unordered_map<const Inst*, Value>                           m_descriptor_selections;
	std::unordered_map<const Block*, std::optional<SelectingBranch>> m_selecting_branches;
	ShaderWriteOrder                                                 m_write_order;
};

} // namespace

void TrackResources(Program& program) {
	Tracker(program).Run();
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
