'''
wrapper called bu nimnvcc to clean up flags
'''
import os
import sys
command=sys.stdin.read().strip().replace("-fmax-errors=3", "--x cu")
with open('commands.txt', 'a') as f:
    f.write(command)
    f.write("\n")
os.system(command)
