import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra

/-!
Focused regressions for the first typed Phi81 `PiDEC.Algebra` slice.

| Stage path | Regression |
|---|---|
| `nifs.pi_dec.verify.radix.parameters` | production radix, child count, and combined bound are fixed |
| `nifs.pi_dec.verify.radix.scalar.digits` | positive and negative thirteen use the same short bit positions |
| `nifs.pi_dec.verify.radix.scalar.total` | the out-of-bound fallback is exact and visibly not a shortness claim |
| `nifs.pi_dec.verify.radix.recompose` | scalar and complete-assignment recomposition theorems are exported |
| `nifs.pi_dec.verify.radix.split_norm` | strict combined-bound inputs split into strict fresh-bound children |
| `nifs.pi_dec.verify.radix.recompose_norm` | arbitrary strict fresh-bound children recompose below the combined bound |
-/

namespace tests.Phi81PiDECAlgebra

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra

#check Radix.production_parameters
#check Radix.magnitudeDigit_lt_two
#check Radix.recomposeScalar
#check Radix.splitScalar_recompose
#check Radix.split_recompose
#check Radix.split_norm
#check Radix.recompose_norm

private def positiveThirteen : F := Radix.fieldOfNat 13
private def negativeThirteen : F := -(Radix.fieldOfNat 13)

example : Radix.splitScalar positiveThirteen ⟨0, by decide⟩ = 1 := by
  decide

example : Radix.splitScalar positiveThirteen ⟨1, by decide⟩ = 0 := by
  decide

example : Radix.splitScalar positiveThirteen ⟨2, by decide⟩ = 1 := by
  decide

example : Radix.splitScalar positiveThirteen ⟨3, by decide⟩ = 1 := by
  decide

example : Radix.splitScalar positiveThirteen ⟨4, by decide⟩ = 0 := by
  decide

example : Radix.splitScalar negativeThirteen ⟨0, by decide⟩ = -(1 : F) := by
  decide

example : Radix.splitScalar negativeThirteen ⟨1, by decide⟩ = 0 := by
  decide

example : Radix.splitScalar negativeThirteen ⟨2, by decide⟩ = -(1 : F) := by
  decide

example : Radix.splitScalar negativeThirteen ⟨3, by decide⟩ = -(1 : F) := by
  decide

example : Radix.splitScalar negativeThirteen ⟨4, by decide⟩ = 0 := by
  decide

/- At the strict boundary, the total fallback retains the value in child zero
and is not used by `split_norm`. -/
example :
    Radix.splitScalar (Radix.fieldOfNat Radix.combinedBound)
        ⟨0, by decide⟩ =
      Radix.fieldOfNat Radix.combinedBound := by
  decide

example :
    Radix.splitScalar (Radix.fieldOfNat Radix.combinedBound)
        ⟨1, by decide⟩ = 0 := by
  decide

end tests.Phi81PiDECAlgebra
