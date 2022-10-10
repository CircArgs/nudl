"""
wrapper called by nimnvcc to clean up flags
"""

import shutil
import os
from pathlib import Path
from utils import *

command = sys.stdin.read().strip()
rm=["-std=gnu++14"]
for r in rm:
    command=command.replace(r, "-std=gnu++11")
# rm=["-std=gnu++14"]
# for r in rm:
#     command=command.replace(r, "")
# nim_files = get_nim_c_files(command)

# for f in nim_files:
#     pp(f)

try_command(command)
