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

# --- Native executables: Python-free train/eval loop -----------------------
# Same headers as the module, but driven by a C++ main() so the dev loop is
# edit -> build -> native\run.cmd ow_train ... with no Python / no .pyd reimport.
# They load-time-link the wheel's torch DLLs, so run via native\run.cmd (puts
# torch\lib on PATH). Executables land in the build dir (e.g. _bld\ow_train.exe).
foreach(_app ow_train ow_eval)
  if(_app STREQUAL ow_train)
    add_executable(${_app} apps/train.cpp)
  else()
    add_executable(${_app} apps/eval.cpp)
  endif()
  target_include_directories(${_app} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${_app} PRIVATE torch_local)
  target_compile_options(${_app} PRIVATE /EHsc /bigobj /Zc:__cplusplus)
endforeach()

# --- GRPO refactor: agent-model + algorithm sources (hpp/cpp split) ---------
set(OW_RL_SRC
  rl/model/policy_net.cpp
  rl/model/distribution.cpp
  rl/model/agent.cpp
  rl/algo/grpo.cpp
)
# Parity gate: new PolicyNet/Agent vs the old EntityPolicy on a checkpoint.
add_executable(ow_test_model apps/test_model.cpp ${OW_RL_SRC})
target_include_directories(ow_test_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ow_test_model PRIVATE torch_local)
target_compile_options(ow_test_model PRIVATE /EHsc /bigobj /Zc:__cplusplus)

# GRPO trainer (grouped episodic rollout + KL-anchored update).
add_executable(ow_train_grpo apps/train_grpo.cpp ${OW_RL_SRC} rl/rollout.cpp rl/grpo_trainer.cpp)
target_include_directories(ow_train_grpo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ow_train_grpo PRIVATE torch_local)
target_compile_options(ow_train_grpo PRIVATE /EHsc /bigobj /Zc:__cplusplus)
