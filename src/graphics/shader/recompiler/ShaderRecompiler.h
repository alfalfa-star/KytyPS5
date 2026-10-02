#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {

struct CompileOptions {
	ShaderType                stage                 = ShaderType::Compute;
	uint32_t                  wave_size             = 64;
	uint32_t                  user_data_base        = 0;
	uint64_t                  shader_hash           = 0;
	bool                      dump_ir               = true;
	bool                      early_dump            = false;
	const char*               dump_label            = nullptr;
	bool                      maximal_reconvergence = false;
	std::span<const uint32_t> user_data;
	std::span<const uint32_t> back_code;
	ShaderStageInputInfo      input_info;
};

struct TranslateResult {
	IR::Program program;
	std::string decoded_dump;
	std::string cfg_dump;
};

struct CompileResult {
	std::vector<uint32_t> spirv;
	std::string           decoded_dump;
	std::string           ir_dump;
	IR::Program           program;
};

// Decoded and structured control flow of a shader. It depends only on the code (and the stage
// and back code of fused stages), so every permutation of a program can share it.
struct PreparedProgram {
	std::vector<uint32_t> code; // decoded.code points here
	Decoder::Program      decoded;
	CFG::Graph            cfg;
	std::string           decoded_dump;
};

[[nodiscard]] std::shared_ptr<const PreparedProgram> PrepareProgram(std::span<const uint32_t> code,
                                                                    const CompileOptions& options);
[[nodiscard]] TranslateResult TranslateProgram(const PreparedProgram& prepared,
                                               const CompileOptions&  options);
[[nodiscard]] TranslateResult TranslateProgram(std::span<const uint32_t> code,
                                               const CompileOptions&     options);
[[nodiscard]] CompileResult   CompileProgram(TranslateResult                   translated,
                                             const CompileOptions&             options,
                                             const IR::ResourceSpecialization& specialization,
                                             uint32_t push_data_start_dword = 0);

} // namespace Libs::Graphics::ShaderRecompiler

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_ */
