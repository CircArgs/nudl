import nimcuda/[cuda_runtime_api, driver_types, nimcuda, vector_types]
import sequtils, sugar

type GpuArray[T] = object
  data: ref[ptr T]
  len: int

{.emit: """
#define NUDLGLOBAL __global__
#define NUDLDEVICE __device__
#define NUDLHOST __host__
#define NUDLNOINLINE __noinline__
#define NUDLFORCEINLINE __forceinline__
#define NUDLCONSTANT __constant__
#define NUDLSHARED __shared__
#define NUDLRESTRICT __restrict__
""".}

let threadIdx {.importc, used, nodecl.} : uint3
let blockIdx  {.importc, used, nodecl.} : uint3
let gridDim   {.importc, used, nodecl.} : dim3
let blockDim  {.importc, used, nodecl.} : dim3
let warpSize  {.importc, used, nodecl.} : cint

template syncthreads(): untyped =
  {.emit: "__syncthreads()".}

template threadfence(): untyped =
  {.emit: "__threadfence()".}

template threadfence_block(): untyped =
  {.emit: "__threadfence_block()".}

template threadfence_system(): untyped =
  {.emit: "__threadfence_system()".}


{.emit:"""
NUDLGLOBAL
void __nudlglobal__square(float* d_out, float* d_in);
void cuda_square(int bpg, int tpb, float * d_out, float * d_in){
    __nudlglobal__square<<<bpg,tpb>>>(d_out, d_in);
}
""".}

proc cuda_square(bpg, tpb: cint, y: ptr cfloat, x: ptr cfloat) {.importc.}

template cuda(body) =
  {.push stackTrace: off, checks: off, optimization: speed, exportc, used, cdecl.}
  body
  {.pop.}

import macros
dumpTree:
  {.push stackTrace: off, checks: off, optimization: speed.}
  proc square*(d_out, d_in: ptr cfloat){.exportc: "__nudlglobal__$1", cdecl.} =
    let idx = threadIdx.x
    let offset = cast[uint](idx)*cast[uint](sizeof(cfloat))
    let in_addr = cast[ptr[cfloat]](cast[uint](d_in) + offset)
    let out_addr = cast[ptr[cfloat]](cast[uint](d_out) + offset)
    let f: cfloat = in_addr[]
    out_addr[] = f * f
  {.pop.}

proc cudaMalloc[T](size: int): ptr T {.noSideEffect.} =
  let s = size * sizeof(T)
  check cudaMalloc(cast[ptr pointer](addr result), s)

proc deallocCuda[T](p: ref[ptr T]) {.noSideEffect.} =
  if not p[].isNil:
    check cudaFree(p[])

proc newGpuArray[T](len: int): GpuArray[T] {.noSideEffect.} =
  new(result.data, deallocCuda)
  result.len = len
  result.data[] = cudaMalloc[T](result.len)

proc cuda[T](s: seq[T]): GpuArray[T] {.noSideEffect.} =
  result = newGpuArray[T](s.len)

  let size = result.len * sizeof(T)

  check cudaMemCpy(result.data[],
                   unsafeAddr s[0],
                   size,
                   cudaMemcpyHostToDevice)

proc cpu[T](g: GpuArray[T]): seq[T] {.noSideEffect.} =
  result = newSeq[T](g.len)

  let size = result.len * sizeof(T)

  check cudaMemCpy(addr result[0],
                   g.data[],
                   size,
                   cudaMemcpyDeviceToHost)


proc main() =
  let a = newSeq[cfloat](64)

  let b = toSeq(0..63).map(x => x.cfloat)

  echo a
  echo b

  var u = a.cuda
  let v = b.cuda

  cuda_square(1.cint, 64.cint, u.data[], v.data[])

  check cudaDeviceSynchronize()

  let z = u.cpu
  echo z

main()
## Output:

# @[0.0, 0.0, 0.0, 0.0, 0.0, ...]
# @[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, ...]
# @[0.0, 1.0, 4.0, 9.0, 16.0, 25.0, ...]
