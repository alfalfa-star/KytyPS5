#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTCOMPILER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTCOMPILER_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

// A ResourcePlan's reachable SRT expression graph flattened into an array of ops whose operands
// are indices of earlier ops. Built once per shader plan; each draw evaluates it on demand with
// CompiledSrtEvaluator instead of re-walking the Inst graph.
enum class CompiledOpKind : uint8_t {
	AlwaysFails,
	Constant,
	GetUserData,
	GetShaderBase,
	Generic,
	ExtractU64,
	ExtractPassthrough,
	ExtractCarryHalf,
	ReadFirstLane,
	ReadConst,
	RawRead,
	Select,
};

inline constexpr uint32_t kInvalidSlot = UINT32_MAX;

struct CompiledOp {
	CompiledOpKind kind         = CompiledOpKind::AlwaysFails;
	ValueOpcode    opcode       = ValueOpcode::Void; // Generic only
	uint8_t        num_operands = 0;
	// RawRead of a constant buffer uses all five: handle low, high, offset, records, word3.
	std::array<uint32_t, 5> operands {kInvalidSlot, kInvalidSlot, kInvalidSlot, kInvalidSlot,
	                                  kInvalidSlot};
	uint64_t immediate = 0; // Constant: value. GetUserData: user data index. RawRead: offset.
	uint32_t component = 0; // Extract: component. ReadConst: SRT slot.
	bool     is_const_buffer_read = false;
};

struct CompiledSrtProgram {
	std::vector<CompiledOp>              ops;
	std::vector<std::array<uint32_t, 8>> descriptor_source_slots; // per descriptor_sources dword
	std::vector<uint32_t>                srt_read_slots;          // per srt_reads entry
	std::vector<uint32_t> control_flow_condition_slots; // per control_flow; kInvalidSlot if none
};

CompiledSrtProgram CompileSrtProgram(const ResourcePlan& program);

// Evaluates slots of a CompiledSrtProgram with the same semantics as SrtWalker's Evaluator:
// values are computed only when asked for and cached for the evaluator's lifetime, Select reads
// its predicate from the clean evaluator and evaluates only the chosen side, and ReadFirstLane
// evaluates its value in fresh evaluators scoped to its lane mask.
class CompiledSrtEvaluator {
public:
	CompiledSrtEvaluator(const CompiledSrtProgram& compiled, const SrtRuntime& runtime,
	                     std::span<const uint8_t> clean_flat_slots = {},
	                     CompiledSrtEvaluator*    clean            = nullptr,
	                     uint32_t                 active_mask_slot = kInvalidSlot);
	~CompiledSrtEvaluator();
	CompiledSrtEvaluator(const CompiledSrtEvaluator&)            = delete;
	CompiledSrtEvaluator& operator=(const CompiledSrtEvaluator&) = delete;

	bool Evaluate(uint32_t slot, uint64_t& result);
	// The dword address a RawRead op loads; zero when it reads past a buffer and returns zero.
	bool RawReadAddress(uint32_t slot, uint64_t& address, bool& zero);

private:
	bool Compute(const CompiledOp& op, uint64_t& result);

	const CompiledSrtProgram& m_compiled;
	SrtRuntime                m_runtime;
	std::span<const uint8_t>  m_clean_flat_slots;
	CompiledSrtEvaluator*     m_clean            = nullptr;
	uint32_t                  m_active_mask_slot = kInvalidSlot;
	std::vector<uint64_t>*    m_results          = nullptr;
	std::vector<uint8_t>*     m_states           = nullptr; // 0 unknown, 1 computed, 2 failed
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTCOMPILER_H_
