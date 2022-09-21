## This is an example
## 
import nimcuda/[cuda_runtime_api, driver_types, nimcuda]
import sequtils, sugar
import nudl

## square is a kernel hence `global`
## cuda: global is the same as __global__ pragma in c
proc square*(d_out, d_in: GpuPointer[cfloat]){.cuda: global.} =
  let idx = threadIdx.x
  d_out[idx]= d_in[idx]*d_in[idx]

proc main() =
  let a = newSeq[cfloat](64)

  let b = toSeq(0..63).map(x => x.cfloat)

  echo a
  echo b

  var u = a.cuda
  var v = b.cuda
  # akin to calling square<<<1, 64>>> in c 
  invoke 1, 64, square(u, v)

  check cudaDeviceSynchronize()

  let z = u.cpu
  echo z

main()
## Output:

# @[0.0, 0.0, 0.0, 0.0, 0.0, ...]
# @[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, ...]
# @[0.0, 1.0, 4.0, 9.0, 16.0, 25.0, ...]
