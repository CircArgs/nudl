import nimcuda/[cuda_runtime_api, driver_types, nimcuda, vector_types]
import sequtils, sugar
import macros, strformat
import fusion/matching
{.experimental: "caseStmtMacros".}

type GpuArray*[T] = object
  data: ref[ptr T]
  len: int

type
  CudaDecl* = enum
    host
    global
    device
    noinline
    forceinline

template nudl_announce*: untyped =
  nnkPragma.newTree(
      newColonExpr(
        newIdentNode("emit"),
        newLit("//nudl was here!!!")
    ))

macro cuda*(prefix: CudaDecl, val: untyped): untyped =
  val.addPragma(newColonExpr(ident"exportc", newLit(fmt"__nudl{prefix}__{val.name}")))
  val.addPragma(ident"cdecl")
  nnkStmtList.newTree(nudl_announce, val)

macro invoke*(numBlocks, blockSize: uint, val: untyped): untyped =
  case val:
    of Call[Ident(strVal: @name), .._]:
      result = nnkStmtList.newTree(
        nudl_announce,
        nnkPragma.newTree(
        newColonExpr(
          newIdentNode("emit"),
          newLit(fmt"//nudlinvoke __nudlglobal__{name} __nudlglobal__{name}<<<{numBlocks.repr}, {blockSize.repr}>>>")
        )), val)
    else:
      error("Cannot invoke this")

# {.emit: """
# #ifndef __NUDL__
# #define __NUDL__
# #define NUDLGLOBAL __global__
# #define NUDLDEVICE __device__
# #define NUDLHOST __host__
# #define NUDLNOINLINE __noinline__
# #define NUDLFORCEINLINE __forceinline__
# #define NUDLCONSTANT __constant__
# #define NUDLSHARED __shared__
# #define NUDLRESTRICT __restrict__
# #endif
# """.}

let threadIdx* {.importc, used, nodecl.} : uint3
let blockIdx*  {.importc, used, nodecl.} : uint3
let gridDim*   {.importc, used, nodecl.} : dim3
let blockDim*  {.importc, used, nodecl.} : dim3
let warpSize*  {.importc, used, nodecl.} : cint

template syncthreads*(): untyped =
  {.emit: "__syncthreads()".}

template threadfence*(): untyped =
  {.emit: "__threadfence()".}

template threadfence_block*(): untyped =
  {.emit: "__threadfence_block()".}

template threadfence_system*(): untyped =
  {.emit: "__threadfence_system()".}

proc cudaMalloc*[T](size: int): ptr T {.noSideEffect.} =
  let s = size * sizeof(T)
  check cudaMalloc(cast[ptr pointer](addr result), s)

proc deallocCuda*[T](p: ref[ptr T]) {.noSideEffect.} =
  if not p[].isNil:
    check cudaFree(p[])

proc newGpuArray*[T](len: int): GpuArray[T] {.noSideEffect.} =
  new(result.data, deallocCuda)
  result.len = len
  result.data[] = cudaMalloc[T](result.len)

proc cuda*[T](s: seq[T]): GpuArray[T] {.noSideEffect.} =
  result = newGpuArray[T](s.len)

  let size = result.len * sizeof(T)

  check cudaMemCpy(result.data[],
                   unsafeAddr s[0],
                   size,
                   cudaMemcpyHostToDevice)

proc cpu*[T](g: GpuArray[T]): seq[T] {.noSideEffect.} =
  result = newSeq[T](g.len)

  let size = result.len * sizeof(T)

  check cudaMemCpy(addr result[0],
                   g.data[],
                   size,
                   cudaMemcpyDeviceToHost)

