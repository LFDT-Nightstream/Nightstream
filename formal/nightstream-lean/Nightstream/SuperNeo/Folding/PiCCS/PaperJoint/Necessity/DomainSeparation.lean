import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources

/-!
Necessity of separating the row cube from the Phi81 assignment carrier.

Protocol: SuperNeo `Pi_CCS` and the Phi81 coefficient embedding.
Phase: row-domain indexing versus completed assignment-carrier indexing.
Constraint family: semantic shape ownership only; this file emits no rows.

Owns: a kernel-checked incompatibility proof between the existing paper-model
`ColumnLayout` and every complete 54-lane Phi81 carrier. A two-sided
`ColumnLayout` forces its column count to be a power of two, while every
complete Phi81 carrier width is divisible by three. No caller can construct
both shapes at once.

Does not own: the replacement two-domain semantics, FE/NC SplitNc soundness,
production `ell_n`/`ell_m` derivation, Rust refinement, R1CS lowering, row
removal, or constraint counts.

Emits constraints: no.

Authority boundary: this theorem invalidates reuse of the square paper
row/column bijection as the production carrier contract. It does not claim a
production bug. The next semantic model must give rows and columns distinct
typed domains and separately prove their production encodings.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| paper `Pi_CCS` | square source domain | `ColumnLayout` | two-sided inversion forces `columns = 2^variables` |
| Phi81 embedding | complete carrier shape | 54-lane blocks | every carrier width is divisible by three |
| shape assurance | arithmetic separation | powers of two | no power of two is divisible by three |
| model necessity | row versus column ownership | complete carrier | no `ColumnLayout variables (carrierWidth logicalWidth)` exists |
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

/-- Inclusion-necessity result: the existing square-domain `ColumnLayout`
cannot serve as a layout for the complete Phi81 carrier, for any logical width
or row-cube dimension. Rows and carrier columns must be modeled separately. -/
theorem no_columnLayout_for_completeCarrier
    (logicalWidth variables : Nat) :
    ¬ Nonempty (ColumnLayout variables (carrierWidth logicalWidth)) := by
  rintro ⟨layout⟩
  exact carrierWidth_ne_twoPow logicalWidth variables
    layout.columns_eq_twoPow

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.DomainSeparation
