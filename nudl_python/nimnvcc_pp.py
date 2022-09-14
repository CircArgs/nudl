import re

def log(*args):
    with open(".nudl.log", "a") as f:
        for s in args:
            f.write(s + " ")
        f.write("\n")

device_decl = re.compile(r"^.*?N_CDECL.*?__nudl(?P<kind>(global|host|device)+).*$", re.M)

decls = ["global", "device", "host", "noinline", "forceinline"]


def gen_decl(kind):
    decl = []
    for d in decls:
        if d in kind:
            decl.append(f"NUDL{d.upper()}")
    return " ".join(decl)


def make_decls(code):
    matches = device_decl.finditer(code)
    ret=[]
    pos=0
    for m in matches:
        kind = m.group("kind")
        decl=gen_decl(kind)
        s, e = m.span()
        s, e = s-pos, e-pos
        pos=e
        ret.append(code[:s])
        current = code[s:e]
        rep = decl+'\n'+current
        ret.append(rep)
        code = code[e:]
    return ''.join(ret)+code


# del_headers=['vector_types']
# re_del_headers=[re.compile(fr"#include.*?{h}\.h.*?") for h in del_headers]


# def replace_headers(code):
#     for r in re_del_headers:
#         matches=r.finditer(code)
#         for match in matches:
#             code=code.replace(code[slice(*match.span())], "")

#     return code

def pp(infile):
    with open(infile, encoding="utf-8") as f:
        code=f.read()

    # code = replace_headers(code)
    code = make_decls(code)

    with open(infile, "w", encoding="utf-8") as f:
        f.write(code)
