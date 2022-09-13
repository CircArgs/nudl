import re

device_decl = re.compile(r".*?__nudl(?P<kind>(global|host|device)+).*")

decls = ["global", "device", "host", "noinline", "forceinline"]


def gen_decl(kind):
    decl = []
    for d in decls:
        if d in kind:
            decl.append("__" + d + "__")
    return " ".join(decl)


def make_decl(code):
    matches = device_decl.finditer(code)
    for m in matches:
        kind = m.group("kind")
        current = code[slice(*m.span())]
        code = code.replace(current, gen_decl(kind) + " " + current)
    return code


def pp(infile):
    with open(infile) as f:
        pp = make_decl(f.read())

    with open(infile, "w") as f:
        f.write(pp)
