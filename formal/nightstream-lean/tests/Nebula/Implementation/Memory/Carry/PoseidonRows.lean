import Nightstream.Implementation.Nebula.Memory.Carry.PoseidonBinding

/-! Focused regressions for the exact V2 carry Poseidon2 relation. -/

namespace NightstreamTests.NebulaMemoryCarryPoseidonRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.MemoryCarryPoseidonRows
open Nightstream.Implementation.Nebula.MemoryCarryPoseidonBinding
open Nightstream.Protocol.Nebula

example :
    expectedSchedule.length = 31 ∧
      (expectedSchedule.filter (· = .absorb 4)).length = 29 ∧
      (expectedSchedule.filter (· = .absorb 1)).length = 1 ∧
      (expectedSchedule.filter (· = .pad)).length = 1 :=
  expectedSchedule_exact

example {layout : Layout} (valid : layout.Valid) :
    layout.trace.rounds.length = 31 :=
  valid.round_count_exact

example {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat} {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed
      layout.frame.packing.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment) :
    ∀ lane : Fin 4,
      assignment (layout.trace.outputColumns.getD lane.val 0) =
        carryDigest block lane :=
  output_columns_eq_carryDigest valid canonical one placed holds

example [DecidableEq MemoryCarryParser.Block]
    {headers : FPrime.ChainHeaders Digest.Value}
    {leftBlock rightBlock : MemoryCarryParser.Block}
    {leftValue rightValue : MemoryCarryCodec.Value}
    (leftAccepted : MemoryCarryParser.parse headers leftBlock = some leftValue)
    (rightAccepted : MemoryCarryParser.parse headers rightBlock = some rightValue)
    (equal : carryDigest leftBlock = carryDigest rightBlock) :
    leftValue = rightValue ∨ PoseidonCollision :=
  parsed_value_eq_or_poseidon_collision leftAccepted rightAccepted equal

end NightstreamTests.NebulaMemoryCarryPoseidonRows
