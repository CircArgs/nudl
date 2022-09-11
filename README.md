# nudl
poc cuda in pure nim
Inspired by https://forum.nim-lang.org/t/3171

# Build
use the `build` script in the root of the repo
Ensure you have cuda installed. Tested in the dockerfile as well as `nvcc 11.7` on Manjaro 21.3.7


# Current oddities
Taking note here of some oddities I have encountered thus far:
I am bundling a `nimbase.h` currently with a couple of small mods. See comments there of `// NVCC COMPLAINS HERE`
The modifications looks to be some checks to ensure nim's expected sizes are correct and these should probably exist but I've ditched them during experimenting since with them builds failed.