/* Generated by Nim Compiler v1.6.6 */
/* Compiled for: Linux, amd64, clang */
/* Command for C compiler:
   /home/nick/Projects/nudl/nudlcc -c -w -ferror-limit=3 -O3   -I/home/nick/.choosenim/toolchains/nim-1.6.6/lib -I/home/nick/Projects/nudl/src -o /home/nick/Projects/nudl/nudl_cache/stdlib_formatfloat.nim.c.o /home/nick/Projects/nudl/nudl_cache/stdlib_formatfloat.nim.c */
#define NIM_INTBITS 64

#include "nimbase.h"
#include <stdio.h>
#include <string.h>
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
typedef struct NimStringDesc NimStringDesc;
typedef struct TGenericSeq TGenericSeq;
struct TGenericSeq {
NI len;
NI reserved;
};
struct NimStringDesc {
  TGenericSeq Sup;
NIM_CHAR data[SEQ_DECL_SIZE];
};
typedef NIM_CHAR tyArray__eVNFTutn6un5gcq48fQLdg[65];
N_LIB_PRIVATE N_NIMCALL(NI, writeFloatToBufferSprintf__systemZformatfloat_59)(NIM_CHAR* buf, NF value);
N_LIB_PRIVATE N_NIMCALL(void, writeToBuffer__systemZformatfloat_51)(NIM_CHAR* buf, NCSTRING value);
N_LIB_PRIVATE N_NIMCALL(void, addCstringN__systemZformatfloat_5)(NimStringDesc** result, NCSTRING buf, NI buflen);
N_LIB_PRIVATE N_NIMCALL(NimStringDesc*, setLengthStr)(NimStringDesc* s, NI newLen);
N_LIB_PRIVATE N_NIMCALL(void, unsureAsgnRef)(void** dest, void* src);
N_LIB_PRIVATE N_NIMCALL(void, writeToBuffer__systemZformatfloat_51)(NIM_CHAR* buf, NCSTRING value) {
	NI i;
	i = ((NI) 0);
	{
		while (1) {
			if (!!(((NU8)(value[i]) == (NU8)(0)))) goto LA2;
			buf[(i)- 0] = value[i];
			i += ((NI) 1);
		} LA2: ;
	}
}
N_LIB_PRIVATE N_NIMCALL(NI, writeFloatToBufferSprintf__systemZformatfloat_59)(NIM_CHAR* buf, NF value) {
	NI result;
	NI n;
	int T1_;
	NIM_BOOL hasDot;
	result = (NI)0;
	T1_ = (int)0;
	T1_ = sprintf(((NCSTRING) (buf)), "%.16g", value);
	n = ((NI) (T1_));
	hasDot = NIM_FALSE;
	{
		NI i;
		NI colontmp_;
		NI res;
		i = (NI)0;
		colontmp_ = (NI)0;
		colontmp_ = (NI)(n - ((NI) 1));
		res = ((NI) 0);
		{
			while (1) {
				if (!(res <= colontmp_)) goto LA4;
				i = res;
				{
					if (!((NU8)(buf[(i)- 0]) == (NU8)(44))) goto LA7_;
					buf[(i)- 0] = 46;
					hasDot = NIM_TRUE;
				}
				goto LA5_;
				LA7_: ;
				{
					if (!(((NU8)(buf[(i)- 0])) >= ((NU8)(97)) && ((NU8)(buf[(i)- 0])) <= ((NU8)(122)) || ((NU8)(buf[(i)- 0])) >= ((NU8)(65)) && ((NU8)(buf[(i)- 0])) <= ((NU8)(90)) || ((NU8)(buf[(i)- 0])) == ((NU8)(46)))) goto LA10_;
					hasDot = NIM_TRUE;
				}
				goto LA5_;
				LA10_: ;
				LA5_: ;
				res += ((NI) 1);
			} LA4: ;
		}
	}
	{
		if (!!(hasDot)) goto LA14_;
		buf[(n)- 0] = 46;
		buf[((NI)(n + ((NI) 1)))- 0] = 48;
		buf[((NI)(n + ((NI) 2)))- 0] = 0;
		result = (NI)(n + ((NI) 2));
	}
	goto LA12_;
	LA14_: ;
	{
		result = n;
	}
	LA12_: ;
	{
		if (!(((NU8)(buf[((NI)(n - ((NI) 1)))- 0])) == ((NU8)(110)) || ((NU8)(buf[((NI)(n - ((NI) 1)))- 0])) == ((NU8)(78)) || ((NU8)(buf[((NI)(n - ((NI) 1)))- 0])) == ((NU8)(68)) || ((NU8)(buf[((NI)(n - ((NI) 1)))- 0])) == ((NU8)(100)) || ((NU8)(buf[((NI)(n - ((NI) 1)))- 0])) == ((NU8)(41)))) goto LA19_;
		writeToBuffer__systemZformatfloat_51(buf, "nan");
		result = ((NI) 3);
	}
	goto LA17_;
	LA19_: ;
	{
		if (!((NU8)(buf[((NI)(n - ((NI) 1)))- 0]) == (NU8)(70))) goto LA22_;
		{
			if (!((NU8)(buf[(((NI) 0))- 0]) == (NU8)(45))) goto LA26_;
			writeToBuffer__systemZformatfloat_51(buf, "-inf");
			result = ((NI) 4);
		}
		goto LA24_;
		LA26_: ;
		{
			writeToBuffer__systemZformatfloat_51(buf, "inf");
			result = ((NI) 3);
		}
		LA24_: ;
	}
	goto LA17_;
	LA22_: ;
	LA17_: ;
	return result;
}
N_LIB_PRIVATE N_NIMCALL(void, addCstringN__systemZformatfloat_5)(NimStringDesc** result, NCSTRING buf, NI buflen) {
	NI oldLen;
	NI newLen;
	void* T1_;
	oldLen = ((*result) ? (*result)->Sup.len : 0);
	newLen = (NI)(oldLen + buflen);
	unsureAsgnRef((void**) (&(*result)), setLengthStr((*result), ((NI) (newLen))));
	T1_ = (void*)0;
	T1_ = memcpy(((void*) ((&(*result)->data[oldLen]))), ((void*) (buf)), ((size_t) (buflen)));
}
N_LIB_PRIVATE N_NIMCALL(void, addFloatSprintf__systemZformatfloat_99)(NimStringDesc** result, NF x_0) {
	tyArray__eVNFTutn6un5gcq48fQLdg buffer;
	NI n;
	n = writeFloatToBufferSprintf__systemZformatfloat_59(buffer, x_0);
	addCstringN__systemZformatfloat_5(result, ((NCSTRING) ((&buffer[(((NI) 0))- 0]))), n);
}
