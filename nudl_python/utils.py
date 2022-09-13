import subprocess
import sys
import re
from pathlib import Path
from nimnvcc_pp import pp


DEBUG = True


def catch_run(cmd):
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout, stderr = proc.communicate()

    return (
        proc.returncode,
        stdout.decode("utf-8").strip(),
        stderr.decode("utf-8").strip(),
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
    log(cmd)
    code, out, err = catch_run(cmd)
    if code != 0:
        match = unknown_option.search(err)
        if match is not None:
            option = match.group("option")
            try_command(cmd.replace(option, ""))
        else:
            print(err, file=sys.stderr)
            sys.exit(1)
    else:
        print(out)


nudlcache = re.compile(r".*?--nudlcache\s(?P<nudlcache>.*?)\s.*")
nudlfile = re.compile(r".*?--nudlfile\s(?P<nudlfile>.*?)\s.*")


def get_nudl_arg(cmd, arg):
    assert arg in ("nudlcache", "nudlfile")
    if arg == "nudlfile":
        match = nudlfile.search(cmd)
    else:
        match = nudlcache.search(cmd)
    return match.group(arg)


def intercept(cmd, pattern):
    return re.match(fr".*?{pattern}\.o.*?{pattern}.*", cmd) is not None
