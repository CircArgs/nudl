"""
wrapper called bu nimnvcc to clean up flags
"""

DEBUG = True


import subprocess
import sys
import re


def run(cmd):
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout, stderr = proc.communicate()

    return (
        proc.returncode,
        stdout.decode("utf-8").strip(),
        stderr.decode("utf-8").strip(),
    )


unknown_option = re.compile(r".*?Unknown option\s+'(?P<option>.*)'.*")


def write_command(cmd):
    if DEBUG:
        with open("log.txt", "a") as f:
            f.write(command)
            f.write(" -> ")
            f.write(cmd)
            f.write("\n")


def try_command(cmd):
    """
    recursively tries calling `cmd` while replacing unknown flags

    Note: only checks for incorrect flags according to the `unknown_option` pattern
        Any other exceptions will still error
    """
    write_command(cmd)
    code, out, err = run(cmd)
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


command = (
    sys.stdin.read().strip().replace("-c", "-c --x cu")
)  # .replace("-fmax-errors=3", "").replace("-fno-strict-aliasing", "").replace("-fno-ident", "")


try_command(command)
