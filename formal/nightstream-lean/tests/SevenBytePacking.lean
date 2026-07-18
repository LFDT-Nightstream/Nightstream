import Nightstream.Implementation.R1CS.Core.SevenBytePacking

/-! Focused interface gate for shared seven-byte packing. -/

open Nightstream.Implementation.R1CS.SevenBytePacking

#check packSevenAt
#check packBytesAsNats
#check packBytesAsNats_length

#guard packBytesAsNats [110, 101, 111] = [3, 7300462]
