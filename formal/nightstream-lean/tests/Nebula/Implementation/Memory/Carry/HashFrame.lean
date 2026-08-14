import Nightstream.Implementation.Nebula.Memory.Carry.HashBinding

/-! Focused regressions for the exact V2 memory-carry hash frame. -/

namespace NightstreamTests.NebulaMemoryCarryHashFrame

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.MemoryCarryHashFrame
open Nightstream.Implementation.Nebula.MemoryCarryHashBinding
open Nightstream.Protocol.Nebula

example :
    MemoryWireGeometry.carryBits = 3433 ∧ paddedBitCount = 3456 ∧
      highPaddingBitCount = 23 :=
  exact_geometry

example :
    framePrefix = [0x4e534d43, 1, 2, 2, 1, 1, 3433, 32, 108] :=
  framePrefix_exact

example (block : MemoryCarryParser.Block) : (frame block).length = 117 :=
  frame_length block

example {HashDigest : Type} [DecidableEq MemoryCarryParser.Block]
    (hash : Hash HashDigest)
    {headers : FPrime.ChainHeaders Digest.Value}
    {leftBlock rightBlock : MemoryCarryParser.Block}
    {leftValue rightValue : MemoryCarryCodec.Value}
    (leftAccepted : MemoryCarryParser.parse headers leftBlock = some leftValue)
    (rightAccepted : MemoryCarryParser.parse headers rightBlock = some rightValue)
    (equal : digest hash leftBlock = digest hash rightBlock) :
    leftValue = rightValue ∨ Collision hash :=
  parsed_value_eq_or_collision hash leftAccepted rightAccepted equal

end NightstreamTests.NebulaMemoryCarryHashFrame
