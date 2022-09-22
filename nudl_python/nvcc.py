"""
wrapper called by nimnvcc to clean up flags
"""

import shutil
import os
from pathlib import Path
from utils import *

command = sys.stdin.read().strip()
command = command.replace("nvcc -c", "nvcc -c --x cu")

# nim_files = get_nim_c_files(command)

# for f in nim_files:
#     pp(f)

try_command(command)
