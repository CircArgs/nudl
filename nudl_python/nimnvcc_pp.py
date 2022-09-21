import re

def log(*args):
    with open(".nudl.log", "a") as f:
        for s in args:
            f.write(s + " ")
        f.write("\n")

device_decl = re.compile(r"^.*?(N_CDECL|N_INLINE).*?__nudl(?P<kind>(global|host|device)+).*$", re.M)

decls = ["global", "device", "host", "noinline", "forceinline"]


def gen_decl(kind):
    decl = []
    for d in decls:
        if d in kind:
            decl.append(f"__{d}__")
    return " ".join(decl)


def make_decls(code):
    match = device_decl.search(code)
    ret = []
    while match:
        s, e = match.span()
        ret.append(code[:s])
        kind = match.group('kind')
        to = gen_decl(kind)
        ret.append(f"\n{to}\n{code[s:e]}")
        code = code[e:]
        match = device_decl.search(code)
    return ''.join(ret)+code

invocation = re.compile(r"^.*?//nudlinvoke (?P<from>.*?) (?P<to>.*?)$", re.M)

def make_invocations(code):
    match = invocation.search(code)
    ret = []
    while match:
        s, e = match.span()
        from_, to = match.group('from'), match.group('to')
        ret.append(code[:e])
        code=re.sub(from_, to, code[e:], count=1)
        match = invocation.search(code)
    return ''.join(ret)+code


def is_nudl_file(path):
    with open(path, encoding="utf-8") as f:
        f = f.read()
        if "//nudl was here!!!" in f:
            return f
    return False

def pp(infile):
    code = is_nudl_file(infile)
    if code:
        
        # code = replace_headers(code)
        code = make_decls(code)
        code = make_invocations(code)
        with open(infile, "w", encoding="utf-8") as f:
            f.write(code)
