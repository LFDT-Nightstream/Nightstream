import NightstreamFPrime.Export.Stage1.Wide.Stage1Plan
import tests.AxiomAudit

/-! Regressions for all deleted sampler intervals and the required map proof. -/

namespace NightstreamFPrimeTests.WideRetainedSupport

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Export.Stage1
open Wide ProductionRelation

def Removed (source : Nat) : Prop :=
  (113904174 ≤ source ∧ source < 114443652) ∨
  (116589018 ≤ source ∧ source < 119147994) ∨
  (140293745 ≤ source ∧ source < 149282257)

theorem removed_not_live (program : RetainedLayout.Program) (source : Nat) (removed : Removed source) :
    ¬ RetainedLayout.Live program source := by
  obtain ⟨hash, sharedStart, sharedEnd, application⟩ := RetainedLayout.boundaries program
  unfold RetainedLayout.Live
  rw [hash, sharedStart, sharedEnd, application]
  unfold Removed at removed
  omega

theorem removed_unmapped (program : RetainedLayout.Program) (source : Nat) (removed : Removed source) :
    RetainedLayout.column? program source = none := by
  cases mapped : RetainedLayout.column? program source with
  | none => rfl
  | some target =>
    exact False.elim (removed_not_live program source removed
      ((RetainedLayout.live_iff_mapped program source).mpr (by rw [mapped]; rfl)))

/-- Even a stored zero coefficient cannot hide a reference to a deleted column. -/
theorem removed_entry_rejected (program : RetainedLayout.Program)
    (source : Fin (PerApplicationFixedPoint.logicalWidth program)) (coefficient : F)
    (removed : Removed source.val) :
    ¬ ReadSupport.Form program (SparseForm.singleton source coefficient) := by
  intro supported
  exact removed_not_live program source.val removed
    (supported ⟨source, coefficient⟩ List.mem_cons_self)

/-- The old unchecked call must no longer type-check, including at valid inputs. -/
example (program : RetainedLayout.Program)
    (_source : Fin (PerApplicationFixedPoint.logicalWidth program)) : True := by
  fail_if_success
    have unchecked : Fin (RetainedLayout.logicalWidth program) := RetainedLayout.column program _source
  trivial

/-- The constant-one column is retained by its own proof, not used as a fallback. -/
theorem constant_stays_zero (program : RetainedLayout.Program) :
    (Stage1Plan.piRlcInterface program).oneColumn.val = 0 := by
  exact RetainedLayout.column_of_some program
    (ApplicationRetainedGeometry.oneColumn (Stage1Plan.referenceGeometry program))
    (ReadSupport.one program _ rfl) 0 (RetainedLayout.publicColumn program 0 (by decide))

end NightstreamFPrimeTests.WideRetainedSupport

#audit_axioms NightstreamFPrimeTests.WideRetainedSupport.removed_not_live
#audit_axioms NightstreamFPrimeTests.WideRetainedSupport.removed_unmapped
#audit_axioms NightstreamFPrimeTests.WideRetainedSupport.removed_entry_rejected
#audit_axioms NightstreamFPrimeTests.WideRetainedSupport.constant_stays_zero
