## This is an example
## 
import nimcuda/[cuda_runtime_api, driver_types, nimcuda, vector_types]
import sequtils, sugar
import nudl

## square is a kernel hence `global`
## cuda: global is the same as __global__ pragma in c
proc square*(d_out, d_in: ptr cfloat){.cuda: global.} =
  let idx = threadIdx.x
  let offset = cast[uint](idx)*cast[uint](sizeof(cfloat))
  let in_addr = cast[ptr[cfloat]](cast[uint](d_in) + offset)
  let out_addr = cast[ptr[cfloat]](cast[uint](d_out) + offset)
  let f: cfloat = in_addr[]
  out_addr[] = f * f

proc main() =
  let a = newSeq[cfloat](64)

  let b = toSeq(0..63).map(x => x.cfloat)

  echo a
  echo b

  var u = a.cuda
  let v = b.cuda
  # akin to calling square<<<1, 64>>> in c 
  invoke 1, 64, square(u.data[], v.data[])

  check cudaDeviceSynchronize()

  let z = u.cpu
  echo z

main()
## Output:

# @[0.0, 0.0, 0.0, 0.0, 0.0, ...]
# @[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, ...]
# @[0.0, 1.0, 4.0, 9.0, 16.0, 25.0, ...]
