import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.SamplerLayout

/-!
Public structural regressions for the active PiRLC sampler layout facade.

| Surface | Expected fixed formula |
|---|---|
| scalar-local columns/rows | affine in `rho : Fin 15` |
| canonical lanes | affine in `rho : Fin 15` and two `Fin 4` indices |
| selected outputs | exact `ChallengeWiring` columns |
-/

namespace NightstreamTests.FPrimeRecursivePiRlcChallengeSamplerLayout

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.SamplerLayout

private def block0 : Fin digestBlockCount := ⟨0, by decide⟩
private def block1 : Fin digestBlockCount := ⟨1, by decide⟩
private def block2 : Fin digestBlockCount := ⟨2, by decide⟩
private def block3 : Fin digestBlockCount := ⟨3, by decide⟩

private def lane0 : Fin lanesPerBlock := ⟨0, by decide⟩
private def lane1 : Fin lanesPerBlock := ⟨1, by decide⟩
private def lane2 : Fin lanesPerBlock := ⟨2, by decide⟩
private def lane3 : Fin lanesPerBlock := ⟨3, by decide⟩

#check initialCountColumn
#check initializationRow
#check selectionZeroColumn
#check selectionZeroRow
#check fieldColumn
#check bitStart
#check canonicalRow
#check laneResidualRow
#check tailFirstAllocated
#check tailRow
#check outputColumn

example (rho : Fin 15) :
    tailBitStarts rho =
      [bitStart rho block0 lane0, bitStart rho block0 lane1,
       bitStart rho block0 lane2, bitStart rho block0 lane3,
       bitStart rho block1 lane0, bitStart rho block1 lane1,
       bitStart rho block1 lane2, bitStart rho block1 lane3,
       bitStart rho block2 lane0, bitStart rho block2 lane1,
       bitStart rho block2 lane2, bitStart rho block2 lane3,
       bitStart rho block3 lane0, bitStart rho block3 lane1,
       bitStart rho block3 lane2, bitStart rho block3 lane3] := by
  rfl

example (rho : Fin 15) :
    predecessorColumn rho block0 lane0 = initialCountColumn rho := by
  rfl

example (rho : Fin 15) :
    predecessorColumn rho block0 lane1 = bitStart rho block0 lane0 + 157 := by
  rfl

example (rho : Fin 15) :
    predecessorColumn rho block1 lane0 = bitStart rho block0 lane3 + 157 := by
  rfl

example (rho : Fin 15) (block lane : Fin 4) :
    laneResidualRow rho block lane =
      canonicalRow rho block lane + canonicalRowCount := by
  rfl

example : StructureValid := structure_check

example (rho : Fin scalarCount) :
    initialCountColumn rho ≠ selectionZeroColumn rho :=
  zero_columns_distinct rho

example (rho : Fin scalarCount) :
    selectionZeroColumn rho = tailFirstAllocated rho + 5 :=
  selection_zero_column_eq_tail_first_allocated rho

example (rho : Fin scalarCount) :
    selectionZeroRow rho = tailRow rho + 6 :=
  selection_zero_row_eq_tail_row rho

example : outputColumns =
    Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.ChallengeWiring.samplerOutputColumns :=
  output_columns_match_challenge_wiring

end NightstreamTests.FPrimeRecursivePiRlcChallengeSamplerLayout
