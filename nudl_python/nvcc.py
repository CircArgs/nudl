"""
wrapper called bu nimnvcc to clean up flags
"""

from utils import *

command = sys.stdin.read().strip()

nim_file = Path(get_nudl_arg(command, "nudlfile"))
nim_cache = Path(get_nudl_arg(command, "nudlcache"))
file_in_cache = str(nim_cache / nim_file.name)
file_pattern = str(nim_cache / nim_file.name.replace(".", "\."))

# check if command is working with our input file
if intercept(command, file_pattern):
    # preprocess c file nim made with c preprocessor
    run(f"nvcc -E {file_in_cache} > {file_in_cache}")
    # make our mods with nudlpp
    pp(file_in_cache)

try_command(command)
