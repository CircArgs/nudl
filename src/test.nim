import macros, strutils, os

macro magicWord(statments: untyped): untyped =
    ## Designed by Steve Kellock
    result = statments
    for st in statments:
          for node in st:
                if node.kind == nnkStrLit:
                     node.strVal = node.strVal & ", Please."

macro rpt(ix: untyped, cnt: int, statements: untyped)=
  quote do:
    for `ix` in 1..`cnt`:
      `statements`

rpt j, 10:
  magicWord:
     echo j, "- Give me some bear"
     echo "Now"