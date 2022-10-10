## This is an example
## 
import nimcuda/[cuda_runtime_api, driver_types, vector_types, nimcuda]
import sequtils, sugar
import nudl
# {.checks: off.}
# proc square_t*(d_out, d_in: ptr cint){.cuda: global.} =
#   let idx = threadIdx.x
#   d_out[idx]= d_in[idx]*d_in[idx]

proc square*(d_out, d_in: ptr cfloat){.cuda: global.} =
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
  let nb = dim3(x: 1, y: 1, z: 1)
  launch nb, dim3(x: 64, y: 1, z: 1), square(u, v)
  check cudaDeviceSynchronize()

  let z = u.cpu
  echo z

main()
## Output:

# @[0.0, 0.0, 0.0, 0.0, 0.0, ...]
# @[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, ...]
# @[0.0, 1.0, 4.0, 9.0, 16.0, 25.0, ...]
