import nimcuda/[cuda_runtime_api, driver_types, nimcuda, vector_types]
import sequtils, sugar

type GpuArray[T] = object
  data: ref[ptr T]
  len: int

{.emit: """
        __global__ void cuda_square(float * d_out, float * d_in){
            int idx = threadIdx.x;
            float f = d_in[idx];
            d_out[idx] = f * f;
        }
        """.}

proc cuda_square(y: ptr cfloat, x: ptr cfloat) {.importc, nodecl.}
## Compute the square of x and store it in y
## bpg: BlocksPerGrid
## tpb: ThreadsPerBlock

proc cudaMalloc[T](size: int): ptr T {.noSideEffect.}=
  let s = size * sizeof(T)
  check cudaMalloc(cast[ptr pointer](addr result), s)

proc deallocCuda[T](p: ref[ptr T]) {.noSideEffect.}=
  if not p[].isNil:
    check cudaFree(p[])

proc newGpuArray[T](len: int): GpuArray[T] {.noSideEffect.}=
  new(result.data, deallocCuda)
  result.len = len
  result.data[] = cudaMalloc[T](result.len)

proc cuda[T](s: seq[T]): GpuArray[T] {.noSideEffect.}=
  result = newGpuArray[T](s.len)
  
  let size = result.len * sizeof(T)
  
  check cudaMemCpy(result.data[],
                   unsafeAddr s[0],
                   size,
                   cudaMemcpyHostToDevice)

proc cpu[T](g: GpuArray[T]): seq[T] {.noSideEffect.}=
  result = newSeq[T](g.len)
  
  let size = result.len * sizeof(T)
  
  check cudaMemCpy(addr result[0],
                   g.data[],
                   size,
                   cudaMemcpyDeviceToHost)


proc main() =
  let a = newSeq[float32](64)
  
  let b = toSeq(0..63).map(x => x.float32)
  
  echo a
  echo b
  
  var u = a.cuda
  let v = b.cuda
  let args = [u.data[], v.data[]]
  check cudaLaunchKernel(cuda_square, dim3(x: 1, y: 1, z: 1), dim3(x: 64, y: 1, z: 1), cast[ptr pointer](unsafeAddr args), 0, nil)
  
  check cudaDeviceSynchronize()
  
  let z = u.cpu
  echo z

main()
## Output:

# @[0.0, 0.0, 0.0, 0.0, 0.0, ...]
# @[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, ...]
# @[0.0, 1.0, 4.0, 9.0, 16.0, 25.0, ...]