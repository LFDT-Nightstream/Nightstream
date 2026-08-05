import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources

/-!
Necessity of separating the row cube from the Phi81 assignment carrier.

Protocol: SuperNeo `Pi_CCS` and the Phi81 coefficient embedding.
Phase: row-domain indexing versus completed assignment-carrier indexing.
Constraint family: semantic shape ownership only; this file emits no rows.

Owns: a kernel-checked width obstruction for the paper-model `ColumnLayout`.
The paper permits a complete 54-lane Phi81 carrier when it fits in the Boolean
row cube and pads the remaining rows with zero. It rejects only a carrier that
is wider than that cube.

Does not own: production cube-width derivation, Rust refinement, R1CS lowering,
row removal, or constraint counts.

Emits constraints: no.

Authority boundary: this theorem checks the exact paper inequality
`n_F <= 2^ell`. It does not claim that a non-power-of-two carrier is invalid.
The production model still gives rows and columns distinct typed domains and
separately proves their encodings.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| paper `Pi_CCS` | padded source domain | `ColumnLayout` | assignment columns fit in `2^variables` rows |
| Phi81 embedding | complete carrier shape | 54-lane blocks | every carrier width is divisible by three |
| shape assurance | arithmetic separation | powers of two | no power of two is divisible by three |
| model necessity | row versus column ownership | complete carrier | no layout exists when `2^variables < carrierWidth logicalWidth` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.DomainSeparation

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Phi81CarrierLayout
open UnifiedSources

/-- Powers of two never vanish modulo three. The proof is arithmetic and does
not appeal to a runtime decision procedure. -/
theorem twoPow_mod_three_ne_zero (variables : Nat) :
    2 ^ variables % 3 ≠ 0 := by
  induction variables with
  | zero => decide
  | succ variables inductionHypothesis =>
      rw [Nat.pow_succ, Nat.mul_mod]
      have remainderLt : 2 ^ variables % 3 < 3 :=
        Nat.mod_lt _ (by decide)
      have remainderPositive : 0 < 2 ^ variables % 3 :=
        Nat.pos_of_ne_zero inductionHypothesis
      omega

/-- Every completed Phi81 carrier is a multiple of 54 and therefore vanishes
modulo three. -/
theorem carrierWidth_mod_three_eq_zero (logicalWidth : Nat) :
    carrierWidth logicalWidth % 3 = 0 := by
  change (Phi81ColumnLayout.blockCount logicalWidth * 54) % 3 = 0
  omega

/-- A complete Phi81 carrier width cannot equal the cardinality of any
Boolean cube. -/
theorem carrierWidth_ne_twoPow (logicalWidth variables : Nat) :
    carrierWidth logicalWidth ≠ 2 ^ variables := by
  intro equal
  have remainders := congrArg (fun value => value % 3) equal
  change carrierWidth logicalWidth % 3 = 2 ^ variables % 3 at remainders
  rw [carrierWidth_mod_three_eq_zero] at remainders
  exact twoPow_mod_three_ne_zero variables remainders.symm

/-- Inclusion-necessity result: the paper's padded `ColumnLayout` cannot serve
as a layout when the complete Phi81 carrier is wider than its Boolean row
cube. Non-power-of-two widths are valid when this inequality is reversed. -/
theorem no_columnLayout_for_completeCarrier
    (logicalWidth variables : Nat)
    (tooWide : 2 ^ variables < carrierWidth logicalWidth) :
    ¬ Nonempty (ColumnLayout variables (carrierWidth logicalWidth)) := by
  rintro ⟨layout⟩
  have fits := layout.columns_le_twoPow
  omega

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.DomainSeparation
