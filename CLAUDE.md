# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

KytyPS5 is a free, open-source PlayStation 5 emulator written in C++20, based on a heavily
modified fork of [Kyty](https://github.com/InoriRus/Kyty). It targets Windows (primary), Linux,
and experimental macOS (x86-64 under Rosetta 2). The project is early-stage: focus is on boot
reliability and compatibility.

The PS5 GPU is AMD RDNA 2. When touching shader decoding/recompilation, use AMD's RDNA 2
Instruction Set Architecture Reference Guide (document 70648) as the primary instruction-encoding
reference. The host renderer targets Vulkan 1.3 — keep shader changes consistent with both RDNA 2
ISA semantics and SPIR-V validation rules.

## Build

Clang is required on every platform; MSVC's `cl.exe` and Homebrew's arm64-only Qt are rejected.
Always initialize submodules before configuring:

```bash
git submodule update --init --recursive
```

Linux:
```bash
cmake -S . -B _Build/linux -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_PREFIX_PATH="$Qt6_DIR"
cmake --build _Build/linux --target launcher --parallel
cmake --install _Build/linux --prefix _Build/linux/install
```

Windows uses `clang-cl` instead of `clang`/`clang++` (see README for the full toolchain list:
Qt 6 MSVC 2022 64-bit, glslang, Ninja). On NixOS, `nix-shell`/`nix develop` exports
`CMAKE_PREFIX_PATH`/`QT_PLUGIN_PATH` so `-DCMAKE_PREFIX_PATH` can be omitted.

Build just the emulator binary (no Qt launcher needed) with target `kyty_emulator` instead of
`launcher`.

## Tests

Tests are plain executables with their own `main()`; no test framework is used.
Each test file in `tests/` builds as its own `EXCLUDE_FROM_ALL` executable registered as a CTest
test; there is no single "tests" binary.

Build and run everything:
```bash
cmake --build _Build/linux --target kyty_tests
ctest --test-dir _Build/linux --output-on-failure
```

Run a single test target/executable, e.g. the shader recompiler tests:
```bash
cmake --build _Build/linux --target shader_recompiler_compute_tests
ctest --test-dir _Build/linux -R shader_recompiler_compute --output-on-failure
```

CTest names (`add_test(NAME ...)` in `CMakeLists.txt`) don't always match the source file name
one-to-one — check `CMakeLists.txt` for the exact `NAME`/target pair before assuming one.

## Formatting

Formatting is enforced via `clang-format` (config at `src/.clang-format`) through a pre-commit
hook, scoped to staged `.cpp`/`.h`/`.inc` files under `src`:
```bash
python -m pre_commit install --install-hooks
```

## Architecture

Data flow, guest → host:

`loader` (ELF/self loading, dynamic linking, guest syscalls) → `libs` (HLE reimplementations of
PS5 system libraries, e.g. `libKernel`, `libPad`, `libAudio`, `libSaveData`) → `kernel` (guest
memory/thread/sync primitives: `memory`, `pthread`, `eventQueue`, `syncOnAddress`) →
`graphics/guest_gpu` (PM4 command-buffer parsing and hardware-context emulation as the guest GPU
sees it) → `graphics/shader` (recompiles guest RDNA2 shader binaries to SPIR-V) →
`graphics/host_gpu` (Vulkan backend: memory tracking, page-fault-based dirty tracking, resource
caches, pipeline/image management).

Key directories:

- `src/loader` — ELF/PS5 executable loading, runtime linking/symbol resolution, x64 instruction
  emulation for unsupported guest instructions, game-specific patches.
- `src/libs` — one file per emulated PS5 system module (`lib*.cpp`); this is where guest syscalls
  land as HLE implementations.
- `src/kernel` — guest-facing OS primitives (memory address space, threads, semaphores, event
  queues/flags, `syncOnAddress`).
- `src/graphics/guest_gpu/command_processor` — PM4 packet dispatch (`pm4Dispatch.*`) and opcode
  handlers (`pm4Handlers.cpp`) that drive the emulated hardware context.
- `src/graphics/shader/recompiler` — the shader recompiler pipeline:
  - `frontend/decode` — binary decode of RDNA2 instruction encodings into opcode tables
    (`OpcodeTable.h`, `ScalarAluOps`, `VectorAluOps`, `ImageOps`, `MemoryOps`, `ExportOps`).
  - `frontend/cfg` — control-flow graph reconstruction from decoded instructions.
  - `frontend/translate` — turns decoded instructions into SSA IR, one file per instruction
    category (`Vector.cpp`, `Scalar.cpp`, `Float.cpp`, `Integer.cpp`, `Memory.cpp`, `Compare.cpp`,
    `Convert.cpp`, `Control.cpp`, `Dispatch.cpp`, `Attribute.cpp`).
  - `ir` — the SSA IR itself (`Value`, `Block`, `Program`, `Type`, `IREmitter`) plus
    `ir/opcodes` (IR opcode definitions) and `ir/passes` (constant propagation, dead code
    elimination, SSA rewriting, resource tracking/materialization, binding layout, SRT walking).
  - `backend/spirv` — lowers the IR to SPIR-V (`SpirvBuilder`, `SpirvEmitter*` split by concern:
    ALU, image, memory, flow, mesh, tessellation).
  - Adding a new shader instruction typically touches: a decode entry in `ShaderDecoder.h` +
    the relevant `*Ops.cpp` decode file, a translation case in the matching `frontend/translate`
    file, and a regression test in `tests/ShaderRecompilerComputeTests.cpp` (see recent commits
    like "shader: V_CMP_NLT_F16" for the shape of such a change).
- `src/graphics/host_gpu` — Vulkan host backend: `vulkanCommon`, VMA-based allocation (`vma.cpp`),
  guest memory tracking via page protection (`pageManager`, `memoryTracker`), and
  `renderer/{cache,image,pipeline}` for resource/texture/pipeline caching.
- `src/launcher` — the Qt 6 GUI (game list, settings); builds separately from the core emulator
  and is Windows/Linux/macOS with a `.app` bundle produced on macOS.
- `tests/` — focused regression tests per subsystem (shader recompiler/CFG, memory tracker/page
  manager, resource tracking/materialization, kernel filesystem, audio, IME dialog, etc.), not an
  exhaustive suite — new tests are added alongside the feature/fix they cover.

## Contribution constraints

- Windows is the primary target: a change to shared code must not regress Windows; changes
  confined to one platform's own code paths only need to build there.
- AI-assisted contributions are allowed for research/RE/development, but PR descriptions, code
  comments, and issue comments must come from the human contributor, not an autonomous agent, and
  the human must understand, review, and test all submitted code.
