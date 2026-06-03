# Link LibTorch directly from the installed `torch` wheel, bypassing find_package(Torch).
#
# Why: find_package(Torch) for a CUDA-enabled LibTorch pulls in Caffe2's CUDA detection,
# which requires the full CUDA toolkit (nvcc). We never compile .cu files -- we only use
# tensor.to(kCUDA), which needs LibTorch headers + the wheel's torch_cuda.lib at link time
# and the bundled CUDA runtime DLLs at run time. So we link the import libs by hand.
#
# Input:  TORCH_INSTALL_DIR (root of the installed torch package)
# Output: INTERFACE target `torch_local`

if(NOT TORCH_INSTALL_DIR)
  message(FATAL_ERROR "TORCH_INSTALL_DIR not set")
endif()

set(_torch_inc "${TORCH_INSTALL_DIR}/include")
set(_torch_api_inc "${TORCH_INSTALL_DIR}/include/torch/csrc/api/include")
set(_torch_lib "${TORCH_INSTALL_DIR}/lib")

add_library(torch_local INTERFACE)
target_include_directories(torch_local INTERFACE "${_torch_inc}" "${_torch_api_inc}")
target_link_directories(torch_local INTERFACE "${_torch_lib}")

# Core LibTorch import libs (CPU + CUDA). torch.lib is the umbrella; the rest are explicit
# so the linker resolves c10/cuda symbols regardless of link order.
target_link_libraries(torch_local INTERFACE
  "${_torch_lib}/torch.lib"
  "${_torch_lib}/torch_cpu.lib"
  "${_torch_lib}/torch_cuda.lib"
  "${_torch_lib}/c10.lib"
  "${_torch_lib}/c10_cuda.lib"
)

# Definitions LibTorch expects on MSVC.
target_compile_definitions(torch_local INTERFACE NOMINMAX)
# torch_cuda is a DLL (torch_cuda.dll); its static initializers register CUDA kernels on
# load, so no /INCLUDE force-load hack is needed (that's only for static LibTorch).
