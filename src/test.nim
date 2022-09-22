import system
let 
  a = 0.cfloat
  b: cfloat = 0
  temp = cast[ptr pointer](alloc(2*sizeof(ptr pointer)))

temp[] = unsafeAddr a
cast[ptr pointer](cast[uint](temp)+cast[uint](sizeof(pointer)))[] = unsafeAddr b
echo temp[0]