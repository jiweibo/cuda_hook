import re
import subprocess
import sys
from typing import Dict, List

from clang.cindex import Config, CursorKind, Index

LIB_CUDA_SO = "/usr/lib/x86_64-linux-gnu/libcuda.so"
CUDA_HEADER = "/usr/local/cuda/include/cuda.h"
CUDA_EGL_HEADER = "/usr/local/cuda/targets/x86_64-linux/include/cudaEGL.h"
CUDA_GL_HEADER = "/usr/local/cuda/targets/x86_64-linux/include/cudaGL.h"
CUDA_PROFILER_HEADER = "/usr/local/cuda/targets/x86_64-linux/include/cudaProfiler.h"
# CUDA_VDPAU_HEADER = "/usr/local/cuda/targets/x86_64-linux/include/cudaVDPAU.h"
# CUDA_DBG_HEADER = "/usr/local/cuda/extras/Debugger/include/cudadebugger.h"


def _canonical_func_name(func_name: str) -> str:
    """
    map cuMemcpyHtoDAsync_v2 / _v2_ptsz / _v3_xxx to cuMemcpyHtoDAsync.
    """
    _VERSION_RE = re.compile(r"_v\d+.*$")
    _STREAM_SUFFIX_RE = re.compile(r"_(ptsz|ptds)$")

    if "_v" in func_name:
        return _VERSION_RE.sub("", func_name)
    return _STREAM_SUFFIX_RE.sub("", func_name)


def _canonical_func_name_for_stem(func_name: str) -> str:
    """
    map cuMemcpyHtoDAsync_v2_ptsz to cuMemcpyHtoDAsync_v2.
    """
    _STREAM_SUFFIX_RE = re.compile(r"_(ptsz|ptds)$")
    return _STREAM_SUFFIX_RE.sub("", func_name)


def _extract_signatures(filename) -> List[Dict]:
    """
    extrace signatures from file.

    return [
        {'name': 'cuCheckpointProcessLock', 'return_type': 'CUresult', 'params': [{'name': 'pid', 'type': 'int'}]},
        ...
    ]
    """
    index = Index.create()
    tu = index.parse(filename, args=["-x", "c", "--target=x86_64-unknown-linux-gnu"])
    signatures = []
    for cursor in tu.cursor.walk_preorder():
        if cursor.kind == CursorKind.FUNCTION_DECL:
            func_name = cursor.spelling
            if not func_name.startswith("cu"):
                continue

            ret_type = cursor.result_type.spelling
            args = []
            params = []
            for arg in cursor.get_arguments():
                arg_type = arg.type.spelling
                arg_name = arg.spelling
                params.append({"name": arg_name, "type": arg_type})
            func_obj = {
                "name": func_name,
                "return_type": ret_type,
                "params": params,
            }
            signatures.append(func_obj)

            canonical_func_name = _canonical_func_name_for_stem(func_name)

            if canonical_func_name != func_name:
                canonical_func_obj = {
                    "name": canonical_func_name,
                    "return_type": ret_type,
                    "params": params,
                }
                signatures.append(canonical_func_obj)
            else:
                # handle for xx_v2 / xx_v2_ptsz.
                canonical_func_name = _canonical_func_name(func_name)
                if canonical_func_name != func_name:
                    canonical_func_obj = {
                        "name": canonical_func_name,
                        "return_type": ret_type,
                        "params": params,
                    }
                    signatures.append(canonical_func_obj)

            # if func_name.startswith("cuStreamGetCaptureInfo"):
            #     print(
            #         f"func_name {func_name}, canonical_func_name {canonical_func_name}, {canonical_func_obj}"
            #     )

    return signatures


def _get_funcs_from_lib():
    try:
        cmd = "nm -D {SO} | grep -i cu | awk '{{print $3}}'".format(SO=LIB_CUDA_SO)
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=True
        )
        funcs = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing nm command: {e}")
        exit(-1)
    return funcs.strip().split("\n")


def _get_sig_from_headers():
    sigs = []
    for header in [
        CUDA_HEADER,
        CUDA_EGL_HEADER,
        CUDA_GL_HEADER,
        CUDA_PROFILER_HEADER,
        # CUDA_VDPAU_HEADER,
        # CUDA_DBG_HEADER,
    ]:
        sigs.extend(_extract_signatures(header))
    dic = {}
    for sig in sigs:
        if not sig["name"].startswith("cu"):
            continue

        if sig["name"] == "cuOccupancyMaxPotentialBlockSize":
            print(sig)
        if sig["name"] in dic:
            continue
        dic[sig["name"]] = sig
    return dic


def _gen_cpp_code_for_func(func_name, all_sigs):
    template = r"""
HOOK_C_API HOOK_DECL_EXPORT {FUNC_RET_TYPE} {FUNC_NAME}({FUNC_ARGS}) {{
  using func_ptr = {FUNC_RET_TYPE} (*)({FUNC_TYPES});
  static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("{FUNC_NAME}"));
  return func_entry({FUNC_ARGUMENTS});
}}
"""
    ori_func_name = func_name
    if func_name not in all_sigs:
        func_name = _canonical_func_name_for_stem(func_name)

    if func_name not in all_sigs:
        func_name = _canonical_func_name(func_name)

    if func_name not in all_sigs:
        print(f"{func_name} cpp code not generate. Not found sig in header file.")
        return None

    func_sig = all_sigs[func_name]
    func_ret_type = func_sig["return_type"]
    func_args = []
    func_types = []
    func_arguments = []
    for param in func_sig["params"]:
        func_args.append(param["type"] + " " + param["name"])
        func_types.append(param["type"])
        func_arguments.append(param["name"])
    return template.format(
        FUNC_NAME=ori_func_name,
        FUNC_ARGS=",".join(func_args),
        FUNC_TYPES=",".join(func_types),
        FUNC_ARGUMENTS=",".join(func_arguments),
        FUNC_RET_TYPE=func_ret_type,
    )


def main():
    cpp_code = r"""
#include <cuda.h>
#include <cudaEGL.h>
#include <cudaGL.h>
#include <cudaProfiler.h>
#include "cuda_hook.h"

#define HOOK_C_API extern "C"
#define HOOK_DECL_EXPORT __attribute__((visibility("default")))

constexpr const char* LIB_CUDA_SO{"/usr/lib/x86_64-linux-gnu/libcuda.so"};

#undef cuDeviceTotalMem
#undef cuCtxCreate
#undef cuCtxCreate_v3
#undef cuCtxCreate_v4
#undef cuModuleGetGlobal
#undef cuMemGetInfo
#undef cuMemAlloc
#undef cuMemAllocPitch
#undef cuMemFree
#undef cuMemGetAddressRange
#undef cuMemAllocHost
#undef cuMemHostGetDevicePointer
#undef cuMemcpyHtoD
#undef cuMemcpyDtoH
#undef cuMemcpyDtoD
#undef cuMemcpyDtoA
#undef cuMemcpyAtoD
#undef cuMemcpyHtoA
#undef cuMemcpyAtoH
#undef cuMemcpyAtoA
#undef cuMemcpyHtoAAsync
#undef cuMemcpyAtoHAsync
#undef cuMemcpy2D
#undef cuMemcpy2DUnaligned
#undef cuMemcpy3D
#undef cuMemcpyHtoDAsync
#undef cuMemcpyDtoHAsync
#undef cuMemcpyDtoDAsync
#undef cuMemcpy2DAsync
#undef cuMemcpy3DAsync
#undef cuMemcpyBatchAsync
#undef cuMemcpy3DBatchAsync
#undef cuMemsetD8
#undef cuMemsetD16
#undef cuMemsetD32
#undef cuMemsetD2D8
#undef cuMemsetD2D16
#undef cuMemsetD2D32
#undef cuArrayCreate
#undef cuArrayGetDescriptor
#undef cuArray3DCreate
#undef cuArray3DGetDescriptor
#undef cuTexRefSetAddress
#undef cuTexRefGetAddress
#undef cuGraphicsResourceGetMappedPointer
#undef cuCtxDestroy
#undef cuCtxPopCurrent
#undef cuCtxPushCurrent
#undef cuStreamDestroy
#undef cuEventDestroy
#undef cuTexRefSetAddress2D
#undef cuLinkCreate
#undef cuLinkAddData
#undef cuLinkAddFile
#undef cuMemHostRegister
#undef cuGraphicsResourceSetMapFlags
#undef cuStreamBeginCapture
#undef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxReset
#undef cuDevicePrimaryCtxSetFlags
#undef cuDeviceGetUuid_v2
#undef cuIpcOpenMemHandle
#undef cuGraphInstantiate
#undef cuGraphExecUpdate
#undef cuGetProcAddress
#undef cuGraphAddKernelNode
#undef cuGraphKernelNodeGetParams
#undef cuGraphKernelNodeSetParams
#undef cuGraphExecKernelNodeSetParams
#undef cuStreamWriteValue32
#undef cuStreamWaitValue32
#undef cuStreamWriteValue64
#undef cuStreamWaitValue64
#undef cuStreamBatchMemOp
#undef cuStreamGetCaptureInfo
#undef cuStreamGetCaptureInfo_v2

#undef cuGLCtxCreate
#undef cuGLGetDevices
#undef cuGLMapBufferObject
#undef cuGLMapBufferObjectAsync

class HookSingleton {
public:
  static HookSingleton& GetInstance() {
    static auto* single = new HookSingleton();
    return *single;
  }

  void* GetSymbol(const char* name) { return lib_.SymbolAddress(name); }

private:
  HookSingleton() : lib_(LIB_CUDA_SO) {}
  DynamicLibrary lib_;
};

#define HOOK_CUDA_SYMBOL(func_name) HookSingleton::GetInstance().GetSymbol(func_name)
"""

    all_sigs = _get_sig_from_headers()
    funcs = _get_funcs_from_lib()
    for func in funcs:
        code = _gen_cpp_code_for_func(func, all_sigs)
        if not code:
            continue
        cpp_code += code
        cpp_code += "\n"

    with open("cuda_hook.gen.cc", "w") as f:
        f.write(cpp_code)


if __name__ == "__main__":
    main()
