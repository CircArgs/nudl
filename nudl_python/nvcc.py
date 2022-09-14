"""
wrapper called bu nimnvcc to clean up flags
"""

import shutil
import os
from pathlib import Path
from utils import *

env=read_nudl_cache_env()

command = sys.stdin.read().strip()
command = command.replace("nvcc -c", "nvcc -c --x cu")


nim_file = Path(env["nim_input_file"])
nim_cache = Path(env["nim_cache_dir"])
file_in_cache = str(nim_cache / ("@m"+nim_file.name+".c"))
temp_file_in_cache=file_in_cache+".temp"
file_pattern = file_in_cache.replace(".", "\.")

# check if command is working with our input file
if intercept(command, file_pattern):
#     # put modified header file in
#     shutil.copy2(Path(__file__).parent/"nimbase.h", nim_cache)
#     # preprocess c file nim made with c preprocessor
#     run(f"nvcc -E {file_in_cache} >> {temp_file_in_cache}")
#     os.remove(file_in_cache)
#     os.rename(temp_file_in_cache, file_in_cache)
#     # make our mods with nudlpp
    pp(file_in_cache)

try_command(command)
