import Nightstream.Implementation.NebulaV2.FPrime.State.OutputPoseidonBinding

/-! Focused regressions for the mandatory two-stage V2 state hash. -/

namespace NightstreamTests.NebulaV2StateOutputPoseidonBinding

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.StateOutputFrameRows
open Nightstream.Implementation.NebulaV2.StateOutputPoseidonRows
open Nightstream.Implementation.NebulaV2.StateOutputPoseidonBinding
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgram

example :
    StateOutputPoseidonRows.expectedSchedule.length = 9 ∧
      (StateOutputPoseidonRows.expectedSchedule.filter
        (· = .absorb 4)).length = 8 ∧
      (StateOutputPoseidonRows.expectedSchedule.filter (· = .pad)).length = 1 :=
  StateOutputPoseidonRows.expectedSchedule_exact

example (semanticPresent nebulaPresent : Bool) :
    canonical semanticPresent nebulaPresent = canonical true true ↔
      semanticPresent = true ∧ nebulaPresent = true :=
  canonical_shape_eq_v2_iff semanticPresent nebulaPresent

example : cost (canonical true true) = 32 :=
  v2_source_program_cost

example [DecidableEq MemoryCarryParser.Block]
    {layout : MemoryCarryStateOutputRows.Layout} (valid : layout.Valid)
    {leftAssignment rightAssignment : Nat → Nat}
    {leftBlock rightBlock : MemoryCarryParser.Block}
    (leftCanonical : ∀ column, leftAssignment column < goldilocksP)
    (rightCanonical : ∀ column, rightAssignment column < goldilocksP)
    (leftOne : leftAssignment 0 = 1)
    (rightOne : rightAssignment 0 = 1)
    (leftPlaced : PublicBitBlock.Placed
      layout.carry.frame.packing.publicBits leftAssignment leftBlock)
    (rightPlaced : PublicBitBlock.Placed
      layout.carry.frame.packing.publicBits rightAssignment rightBlock)
    (leftHolds : Satisfies
      (MemoryCarryStateOutputRows.rows layout) leftAssignment)
    (rightHolds : Satisfies
      (MemoryCarryStateOutputRows.rows layout) rightAssignment)
    (equalOutputs : ∀ lane : Fin 4,
      leftAssignment
          (layout.stateOutput.trace.outputColumns.getD lane.val 0) =
        rightAssignment
          (layout.stateOutput.trace.outputColumns.getD lane.val 0)) :
    (stateFrame layout.stateOutput.frame leftAssignment leftBlock =
        stateFrame layout.stateOutput.frame rightAssignment rightBlock ∧
      leftBlock = rightBlock) ∨
      OuterCollision ∨ MemoryCarryPoseidonBinding.PoseidonCollision :=
  satisfying_rows_bind_authority_or_collision valid leftCanonical rightCanonical
    leftOne rightOne leftPlaced rightPlaced leftHolds rightHolds equalOutputs

end NightstreamTests.NebulaV2StateOutputPoseidonBinding
