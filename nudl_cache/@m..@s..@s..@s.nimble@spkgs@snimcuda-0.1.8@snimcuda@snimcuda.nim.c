/* Generated by Nim Compiler v1.6.6 */
/* Compiled for: Linux, amd64, clang */
/* Command for C compiler:
   /home/nick/Projects/nudl/nudlcc -c -w -ferror-limit=3 -O3   -I/home/nick/.choosenim/toolchains/nim-1.6.6/lib -I/home/nick/Projects/nudl/src -o /home/nick/Projects/nudl/nudl_cache/@m..@s..@s..@s.nimble@spkgs@snimcuda-0.1.8@snimcuda@snimcuda.nim.c.o /home/nick/Projects/nudl/nudl_cache/@m..@s..@s..@s.nimble@spkgs@snimcuda-0.1.8@snimcuda@snimcuda.nim.c */
#define NIM_INTBITS 64

#include "nimbase.h"
#undef LANGUAGE_C
#undef MIPSEB
#undef MIPSEL
#undef PPC
#undef R3000
#undef R4000
#undef i386
#undef linux
#undef mips
#undef near
#undef far
#undef powerpc
#undef unix
#define nimfr_(x, y)
#define nimln_(x, y)
typedef struct TNimType TNimType;
typedef struct TNimNode TNimNode;
typedef struct tyObject_CudaError__uH4At6Xfefy9bj0ZsCuY3DQ tyObject_CudaError__uH4At6Xfefy9bj0ZsCuY3DQ;
typedef struct tyObject_IOError__iLZrPn9anoh9ad1MmO0RczFw tyObject_IOError__iLZrPn9anoh9ad1MmO0RczFw;
typedef struct tyObject_CatchableError__qrLSDoe2oBoAqNtJ9badtnA tyObject_CatchableError__qrLSDoe2oBoAqNtJ9badtnA;
typedef struct Exception Exception;
typedef struct RootObj RootObj;
typedef struct NimStringDesc NimStringDesc;
typedef struct TGenericSeq TGenericSeq;
typedef struct tySequence__uB9b75OUPRENsBAu4AnoePA tySequence__uB9b75OUPRENsBAu4AnoePA;
typedef struct tyObject_StackTraceEntry__oLyohQ7O2XOvGnflOss8EA tyObject_StackTraceEntry__oLyohQ7O2XOvGnflOss8EA;
typedef NU8 tyEnum_TNimKind__jIBKr1ejBgsfM33Kxw4j7A;
typedef NU8 tySet_tyEnum_TNimTypeFlag__v8QUszD1sWlSIWZz7mC4bQ;
typedef N_NIMCALL_PTR(void, tyProc__ojoeKfW4VYIm36I9cpDTQIg) (void* p, NI op);
typedef N_NIMCALL_PTR(void*, tyProc__WSm2xU5ARYv9aAR4l0z9c9auQ) (void* p);
struct TNimType {
NI size;
NI align;
tyEnum_TNimKind__jIBKr1ejBgsfM33Kxw4j7A kind;
tySet_tyEnum_TNimTypeFlag__v8QUszD1sWlSIWZz7mC4bQ flags;
TNimType* base;
TNimNode* node;
void* finalizer;
tyProc__ojoeKfW4VYIm36I9cpDTQIg marker;
tyProc__WSm2xU5ARYv9aAR4l0z9c9auQ deepcopy;
};
typedef NU8 tyEnum_TNimNodeKind__unfNsxrcATrufDZmpBq4HQ;
struct TNimNode {
tyEnum_TNimNodeKind__unfNsxrcATrufDZmpBq4HQ kind;
NI offset;
TNimType* typ;
NCSTRING name;
NI len;
TNimNode** sons;
};
struct RootObj {
TNimType* m_type;
};
struct TGenericSeq {
NI len;
NI reserved;
};
struct NimStringDesc {
  TGenericSeq Sup;
NIM_CHAR data[SEQ_DECL_SIZE];
};
struct Exception {
  RootObj Sup;
Exception* parent;
NCSTRING name;
NimStringDesc* message;
tySequence__uB9b75OUPRENsBAu4AnoePA* trace;
Exception* up;
};
struct tyObject_CatchableError__qrLSDoe2oBoAqNtJ9badtnA {
  Exception Sup;
};
struct tyObject_IOError__iLZrPn9anoh9ad1MmO0RczFw {
  tyObject_CatchableError__qrLSDoe2oBoAqNtJ9badtnA Sup;
};
struct tyObject_CudaError__uH4At6Xfefy9bj0ZsCuY3DQ {
  tyObject_IOError__iLZrPn9anoh9ad1MmO0RczFw Sup;
};
struct tyObject_StackTraceEntry__oLyohQ7O2XOvGnflOss8EA {
NCSTRING procname;
NI line;
NCSTRING filename;
};
struct tySequence__uB9b75OUPRENsBAu4AnoePA {
  TGenericSeq Sup;
  tyObject_StackTraceEntry__oLyohQ7O2XOvGnflOss8EA data[SEQ_DECL_SIZE];
};
N_LIB_PRIVATE N_NIMCALL(void, nimGCvisit)(void* d, NI op);
static N_NIMCALL(void, Marker_tyRef__L9aLneJnpBIJFkSwgYQgzvA)(void* p, NI op);
extern TNimType NTIioerror__iLZrPn9anoh9ad1MmO0RczFw_;
N_LIB_PRIVATE TNimType NTIcudaerror__uH4At6Xfefy9bj0ZsCuY3DQ_;
N_LIB_PRIVATE TNimType NTIrefcudaerror__L9aLneJnpBIJFkSwgYQgzvA_;
static N_NIMCALL(void, Marker_tyRef__L9aLneJnpBIJFkSwgYQgzvA)(void* p, NI op) {
	tyObject_CudaError__uH4At6Xfefy9bj0ZsCuY3DQ* a;
	a = (tyObject_CudaError__uH4At6Xfefy9bj0ZsCuY3DQ*)p;
	nimGCvisit((void*)(*a).Sup.Sup.Sup.parent, op);
	nimGCvisit((void*)(*a).Sup.Sup.Sup.message, op);
	nimGCvisit((void*)(*a).Sup.Sup.Sup.trace, op);
	nimGCvisit((void*)(*a).Sup.Sup.Sup.up, op);
}
N_LIB_PRIVATE N_NIMCALL(void, nimcuda_nimcudaDatInit000)(void) {
static TNimNode TM__hun83zXz9cQYt9a9bQFeYIzHA_0[1];
NTIcudaerror__uH4At6Xfefy9bj0ZsCuY3DQ_.size = sizeof(tyObject_CudaError__uH4At6Xfefy9bj0ZsCuY3DQ);
NTIcudaerror__uH4At6Xfefy9bj0ZsCuY3DQ_.align = NIM_ALIGNOF(tyObject_CudaError__uH4At6Xfefy9bj0ZsCuY3DQ);
NTIcudaerror__uH4At6Xfefy9bj0ZsCuY3DQ_.kind = 17;
NTIcudaerror__uH4At6Xfefy9bj0ZsCuY3DQ_.base = (&NTIioerror__iLZrPn9anoh9ad1MmO0RczFw_);
TM__hun83zXz9cQYt9a9bQFeYIzHA_0[0].len = 0; TM__hun83zXz9cQYt9a9bQFeYIzHA_0[0].kind = 2;
NTIcudaerror__uH4At6Xfefy9bj0ZsCuY3DQ_.node = &TM__hun83zXz9cQYt9a9bQFeYIzHA_0[0];
NTIrefcudaerror__L9aLneJnpBIJFkSwgYQgzvA_.size = sizeof(tyObject_CudaError__uH4At6Xfefy9bj0ZsCuY3DQ*);
NTIrefcudaerror__L9aLneJnpBIJFkSwgYQgzvA_.align = NIM_ALIGNOF(tyObject_CudaError__uH4At6Xfefy9bj0ZsCuY3DQ*);
NTIrefcudaerror__L9aLneJnpBIJFkSwgYQgzvA_.kind = 22;
NTIrefcudaerror__L9aLneJnpBIJFkSwgYQgzvA_.base = (&NTIcudaerror__uH4At6Xfefy9bj0ZsCuY3DQ_);
NTIrefcudaerror__L9aLneJnpBIJFkSwgYQgzvA_.marker = Marker_tyRef__L9aLneJnpBIJFkSwgYQgzvA;
}
