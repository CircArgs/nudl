import nimcuda/[cuda_runtime_api, nimcuda, vector_types]
import macros, strformat, tables, strutils
import fusion/matching
{.experimental: "caseStmtMacros".}


type
  CudaDecl* = enum
    host
    global
    device
    noinline
    forceinline

var nudlNumBlocks* {.exportc: "__nudlNumBlocks".}: dim3
var nudlBlockSize* {.exportc: "__nudlBlockSize".}: dim3


proc rep_string(s: string): string =
  s.multiReplace(("`", "BackTick"), ("+", "Add"), ("-", "Sub"), ("*", "Mul"), ("/", "Div"), ("=",
      "Eq"), (
    "[", "LeftSquareBracket"), ("]", "RightSquareBracket"), ("(", "LeftParen"), (
    ")", "RightParen"), ("{", "LeftCurlyBracket"), ("}", "RightCurlyBracket"))

macro cuda*(prefix: CudaDecl, val: untyped): untyped =
  case val:
    of ProcDef[@name, _, @generic, FormalParams[any @params], .._]:
      var new_name = name
      var str_name = ""
      case name:
        of Postfix[@exp, @name]:
          str_name = rep_string(fmt"__nudl{name.repr}")
          new_name = nnkPostfix.newTree(exp, ident(rep_string(
              fmt"launch_{name.repr}")))
        else:
          str_name = rep_string(fmt"__nudl{name.repr}")
          new_name = ident(rep_string(fmt"launch_{name.repr}"))
      val.addPragma(newColonExpr(ident"exportc", newLit(fmt"{str_name}")))
      val.addPragma(newColonExpr(ident"codegenDecl", newLit(
          fmt"__{prefix}__ $# {str_name}$3")))
      # val.addPragma(ident"cdecl")
      if prefix.repr != "global":
        return val
      result = nnkStmtList.newTree(
        val,
        newProc(
          new_name,
          params,
          newEmptyNode(),
          nnkProcDef,
          nnkPragma.newTree(
            newColonExpr(
              ident"importc",
              newLit(fmt"{str_name}<<<__nudlNumBlocks, __nudlBlockSize>>>")
        ),
        ident"nodecl")
      ))
    else:
      error(fmt"Cannot declare this as {prefix}")
  echo result.repr


macro launch*(numBlocks, blockSize: dim3, val: untyped): untyped =
  case val:
    of Call[Ident(strVal: @name), any @args]:
      result = nnkStmtList.newTree(
          nnkAsgn.newTree(
            newIdentNode("nudlNumBlocks"),
            numBlocks
        ),
          nnkAsgn.newTree(
            newIdentNode("nudlBlockSize"),
            blockSize
        ),
        newCall(ident(rep_string(fmt"launch_{name}")), args)
      )
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

  GpuPointer*[T] = ptr[T]
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

proc `[]`*[T](g: GpuPointer[T], i: uint): T {.cuda: device, inline.} =
  {.push checks: off.}
  let offset = cast[uint](i)*cast[uint](sizeof(cfloat))
  let offset_addr = cast[ptr[cfloat]](cast[uint](g) + offset)
  result = offset_addr[]
  {.pop.}

proc `[]=`*[T](g: GpuPointer[T], i: uint, y: T) {.cuda: device, inline.} =
  {.push checks: off.}
  let offset = cast[uint](i)*cast[uint](sizeof(cfloat))
  let offset_addr = cast[ptr[cfloat]](cast[uint](g) + offset)
  offset_addr[] = y
  {.pop.}

converter `ptr[T]`*[T](g: GpuArray[T]): GpuPointer[T] =
  g[]