cmake_minimum_required(VERSION 3.18)
project(orbitwars_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Use the venv interpreter; nanobind + Torch cmake dirs come in via CMAKE_PREFIX_PATH
# (set by build.cmd from `python -c "import nanobind,torch;..."`).
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Link LibTorch from the installed wheel directly (no find_package(Torch) -> no CUDA toolkit).
if(NOT TORCH_INSTALL_DIR)
  message(FATAL_ERROR "Set -DTORCH_INSTALL_DIR=<.../site-packages/torch>")
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/LinkTorch.cmake)

nanobind_add_module(orbitwars_native STABLE_ABI NB_STATIC
  bindings/module.cpp
)

target_include_directories(orbitwars_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(orbitwars_native PRIVATE torch_local)
# /Zc:__cplusplus so __cplusplus reports the real standard; /bigobj for heavy torch headers.
target_compile_options(orbitwars_native PRIVATE /EHsc /bigobj /Zc:__cplusplus)

# Drop the built .pyd next to the Python package so it's importable in-place.
set_target_properties(orbitwars_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../orbit_wars_rl/_native"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../orbit_wars_rl/_native"
)

# --- GRPO (continuous actor): agent-model + algorithm sources (hpp/cpp split) -----------
# SquashedGaussian is header-only (distribution.hpp); the rest of the policy is here.
set(OW_RL_SRC
  rl/model/policy_net.cpp
  rl/model/agent.cpp
  rl/algo/grpo.cpp
)

# GRPO trainer (grouped episodic rollout + group-relative update). The Python-free dev loop:
#   edit -> native\build.cmd -> native\run.cmd ow_train_grpo --init-from scratch ...
add_executable(ow_train_grpo apps/train_grpo.cpp ${OW_RL_SRC} rl/rollout.cpp rl/grpo_trainer.cpp)
target_include_directories(ow_train_grpo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ow_train_grpo PRIVATE torch_local)
target_compile_options(ow_train_grpo PRIVATE /EHsc /bigobj /Zc:__cplusplus)

# --- GPU-batched simulator (gpu_env) + its parity oracle vs the reference C++ ow::step ---------
add_executable(ow_parity_gpu apps/parity_gpu.cpp rl/gpu_env.cpp)
target_include_directories(ow_parity_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ow_parity_gpu PRIVATE torch_local)
target_compile_options(ow_parity_gpu PRIVATE /EHsc /bigobj /Zc:__cplusplus)
