# cuda_hook
A simple hook method for cuda.


1. `python3 gen_code.py` generate cpp hook code `cuda_hook.gen.cc`.

2. Add hook code. For example, add callback for each kernel that has finished running on the device side.

```cpp
void CUDA_CB hostCallback(void* userData) {
  const char* msg = (const char*)userData;
    printf("%s\n", msg);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuLaunchKernel(CUfunction f,unsigned int gridDimX,unsigned int gridDimY,unsigned int gridDimZ,unsigned int blockDimX,unsigned int blockDimY,unsigned int blockDimZ,unsigned int sharedMemBytes,CUstream hStream,void **kernelParams,void **extra) {
  using func_ptr = CUresult (*)(CUfunction,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,CUstream,void **,void **);
  static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLaunchKernel"));
  auto status = func_entry(f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,hStream,kernelParams,extra);
  const char* msg = "Hello, World";
  cuLaunchHostFunc(hStream, hostCallback, (void*)msg);
  return status;
}
```

3. `mkdir build && cd build && cmake .. && make`

4. Link the hooked `libcuda.so.1` instead of standard `/usr/lib/x86_64-linux-gnu/libcuda.so.1`.