import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Padding.Semantics

/-!
Contract: obligation-level necessity of each typed public-padding zero check
when a verifier accepts an arbitrary raw carrier assignment.

Owns: for every padding coordinate, a constant-one candidate with that one
coordinate set to one and all other public-padding coordinates set to zero.

Does not own: isolation of the corresponding column from every other emitted
selective row. Therefore this is not yet permission to retain or delete a
Rust row. It proves necessity only for the raw-input `FixedPublicPadding`
obligation. For the canonical constructor the same property is computed, as
proved by `canonicalAssignment_complete`.

Emits constraints: no.

| Stage path | Invalid raw carrier admitted after check removal | Lean witness | Guarantee |
|---|---|---|---|
| `f_prime.selective_ccs.padding.zero_pin[i]` | coefficient `257+i` equals one | `oneAtPadding` | `eachRawPaddingCheck_necessary` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Necessity

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Semantics

def AllOtherZeroPinsHold (dimensions : Dimensions)
    (omitted : Fin fixedPaddingWidth)
    (candidate : Assignment dimensions.shape) : Prop :=
  ∀ offset, offset ≠ omitted → ZeroPinHolds dimensions candidate offset

/-- Raw-carrier countermodel for one omitted zero check. -/
def oneAtPadding (dimensions : Dimensions)
    (target : Fin fixedPaddingWidth) : Assignment dimensions.shape :=
  fun column =>
    if column = constantColumn dimensions then 1
    else if column = paddingCarrierColumn dimensions target then 1
    else 0

theorem oneAtPadding_constant_one (dimensions : Dimensions)
    (target : Fin fixedPaddingWidth) :
    oneAtPadding dimensions target (constantColumn dimensions) = 1 := by
  simp [oneAtPadding]

theorem oneAtPadding_target_one (dimensions : Dimensions)
    (target : Fin fixedPaddingWidth) :
    oneAtPadding dimensions target
      (paddingCarrierColumn dimensions target) = 1 := by
  simp [oneAtPadding]

theorem oneAtPadding_other_zero (dimensions : Dimensions)
    (target offset : Fin fixedPaddingWidth) (different : offset ≠ target) :
    oneAtPadding dimensions target
      (paddingCarrierColumn dimensions offset) = 0 := by
  have notConstant :
      paddingCarrierColumn dimensions offset ≠ constantColumn dimensions :=
    (constantColumn_ne_padding dimensions offset).symm
  have notTarget :
      paddingCarrierColumn dimensions offset ≠
        paddingCarrierColumn dimensions target := by
    intro equal
    exact different (paddingCarrierColumn_injective dimensions equal)
  simp [oneAtPadding, notConstant, notTarget]

theorem oneAtPadding_target_fails (dimensions : Dimensions)
    (target : Fin fixedPaddingWidth) :
    ¬ ZeroPinHolds dimensions (oneAtPadding dimensions target) target := by
  rw [zeroPinHolds_iff dimensions _
    (oneAtPadding_constant_one dimensions target) target]
  rw [oneAtPadding_target_one]
  decide

theorem oneAtPadding_others_hold (dimensions : Dimensions)
    (target : Fin fixedPaddingWidth) :
    AllOtherZeroPinsHold dimensions target (oneAtPadding dimensions target) := by
  intro offset different
  rw [zeroPinHolds_iff dimensions _
    (oneAtPadding_constant_one dimensions target) offset]
  exact oneAtPadding_other_zero dimensions target offset different

theorem oneAtPadding_violates_fixedPublicPadding (dimensions : Dimensions)
    (target : Fin fixedPaddingWidth) :
    ¬ FixedPublicPadding dimensions (oneAtPadding dimensions target) := by
  intro fixed
  have targetZero := fixed target
  rw [oneAtPadding_target_one dimensions target] at targetZero
  have oneNeZero : (1 : F) ≠ 0 := by decide
  exact oneNeZero targetZero

/-- Every padding check is inclusion-necessary when the carrier is accepted as
raw input. This does not establish necessity after canonical construction. -/
theorem eachRawPaddingCheck_necessary (dimensions : Dimensions)
    (target : Fin fixedPaddingWidth) :
    ∃ candidate,
      candidate (constantColumn dimensions) = 1 ∧
        AllOtherZeroPinsHold dimensions target candidate ∧
        ¬ FixedPublicPadding dimensions candidate :=
  ⟨oneAtPadding dimensions target,
    oneAtPadding_constant_one dimensions target,
    oneAtPadding_others_hold dimensions target,
    oneAtPadding_violates_fixedPublicPadding dimensions target⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Necessity
