import nimcuda/[cuda_runtime_api, nimcuda, vector_types]
import macros, strformat
import fusion/matching
{.experimental: "caseStmtMacros".}


type
  CudaDecl* = enum
    host
    global
    device
    noinline
    forceinline

template nudl_announce: untyped =
  nnkPragma.newTree(
      newColonExpr(
        newIdentNode("emit"),
        newLit("//nudl was here!!!")
    ))

macro cuda*(prefix: CudaDecl, val: untyped): untyped =
  val.addPragma(ident"exportc")
  val.addPragma(newColonExpr(ident"codegenDecl", newLit(
      fmt"__{prefix}__ $# $#$#")))
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
          newLit(fmt"void invoke_{name} {{name}<<<{numBlocks.repr}, {blockSize.repr}>>>};")
        )), val)
    else:
      error("Cannot invoke this")

let threadIdx* {.importc, used, nodecl.}: uint3
let blockIdx* {.importc, used, nodecl.}: uint3
let gridDim* {.importc, used, nodecl.}: dim3
let blockDim* {.importc, used, nodecl.}: dim3
let warpSize* {.importc, used, nodecl.}: cint

template syncthreads*(): untyped =
  {.emit: "__syncthreads()".}

template threadfence*(): untyped =
  {.emit: "__threadfence()".}

template threadfence_block*(): untyped =
  {.emit: "__threadfence_block()".}

template threadfence_system*(): untyped =
  {.emit: "__threadfence_system()".}

type 

  GpuPointer*[T] =  ptr[T]
  GpuArray*[T] = object
    data*: ref[GpuPointer[T]]
    len*: int


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

proc `[]`*[T](g: GpuArray[T]): GpuPointer[T] =
  g.data[]

proc `[]`*[T](g: GpuPointer[T], i: uint): T {.exportc: "__nudldevice__GpuPointer_getitem",  inline.}=
  {.push checks: off.}
  let offset = cast[uint](i)*cast[uint](sizeof(cfloat))
  let offset_addr = cast[ptr[cfloat]](cast[uint](g) + offset)
  result = offset_addr[]
  {.pop.}

proc `[]=`*[T](g: GpuPointer[T], i: uint, y: T) {.exportc: "__nudldevice__GpuPointer_setitem", inline.} =
  {.push checks: off.}
  let offset = cast[uint](i)*cast[uint](sizeof(cfloat))
  let offset_addr = cast[ptr[cfloat]](cast[uint](g) + offset)
  offset_addr[]= y
  {.pop.}

converter `ptr[T]`*[T](g: GpuArray[T]): GpuPointer[T] =
  g[]