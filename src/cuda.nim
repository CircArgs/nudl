import nimcuda/[cuda_runtime_api, driver_types, nimcuda, vector_types]
import sequtils, sugar

type GpuArray[T] = object
  data: ref[ptr T]
  len: int
    # __global__ void square(float * d_out, float * d_in){
    #     int idx = threadIdx.x;
    #     float f = d_in[idx];
    #     d_out[idx] = f * f;
    # }
  # __global__ void __nimcudaglobal__square(float* d_out, float* d_in);
{.emit: """

void cuda_square(int bpg, int tpb, float * d_out, float * d_in){
    __nimcudaglobal__square<<<bpg,tpb>>>(d_out, d_in);
}
""".}



# proc cuda_square(bpg, tpb: cint, y: ptr cfloat, x: ptr cfloat) {.importc.}
## Compute the square of x and store it in y
## bpg: BlocksPerGrid
## tpb: ThreadsPerBlock
##
##
# type
#   CudaIndex {.importc: "struct {x: unsigned int; y: unsigned int; z: unsigned int}", nodecl.} = object
#     x, y, z: cuint

proc cuda_square(bpg, tpb: cint, y: ptr cfloat, x: ptr cfloat) {.importc.}

let threadIdx {.exportc, used, nodecl.} = default(uint3)
let blockIdx {.exportc, used, nodecl.} = default(uint3)
let gridDim {.exportc, used, nodecl.} = default(dim3)
let blockDim {.exportc, used, nodecl.} = default(dim3)
let warpSize {.exportc, used, nodecl.} = default(cint)

template cuda(body) =
  {.push stackTrace: off, exportc, used, cdecl.}
  body
  {.pop.}

# import macros
# dumpTree:
proc square*(d_out, d_in: ptr cfloat){.exportc: "__nudlglobal__$1", used.} =
  let idx = threadIdx.x
  let offset = cast[uint](idx)*cast[uint](sizeof(cfloat))
  let in_addr = cast[ptr[cfloat]](cast[uint](d_in) + offset)
  let out_addr = cast[ptr[cfloat]](cast[uint](d_out) + offset)
  let f: cfloat = in_addr[]
  out_addr[] = f * f

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
