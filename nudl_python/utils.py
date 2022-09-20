import subprocess
import sys
import os
import re
from pathlib import Path
import json
from nimnvcc_pp import pp
import warnings

arg_splitter = re.compile(r'\s"?.*?"?\s?')

def get_nim_c_files(cmd):
    return [p for p in arg_splitter.split(cmd) if p.strip().endswith('.nim.c') and os.path.exists(p)]



DEBUG = False

def catch_run(cmd):
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout, stderr = proc.communicate()
    try:
        return (
            proc.returncode,
            stdout.decode("utf-8").strip(),
            stderr.decode("utf-8").strip(),
        )
    except:
        log("ERR", str(stderr))
        return (
            proc.returncode,
            str(stdout),#.decode("utf-8").strip(),
            str(stderr),#.decode("utf-8").strip(),
        )



def log(*args):
    if DEBUG:
        with open(".nudl.log", "a") as f:
            for s in args:
                f.write(s + " ")
            f.write("\n")


def run(cmd):
    log(cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

    stdout = p.communicate()

    return stdout[0].decode("utf-8").strip()


unknown_option = re.compile(r".*?Unknown option\s+'(?P<option>.*)'.*")


def try_command(cmd):
    """
    recursively tries calling `cmd` while replacing unknown flags

    Note: only checks for incorrect flags according to the `unknown_option` pattern
        Any other exceptions will still error
    """
    code, out, err = catch_run(cmd)
    if code != 0:
        match = unknown_option.search(err)
        if match is not None:
            option = match.group("option")
            warnings.warn(f"{option} is unsupported by nvcc. Attempting without this.")
            try_command(cmd.replace(option, ""))
        else:
            print(err, file=sys.stderr)
            sys.exit(1)
    else:
        log(cmd)
        print(out)


