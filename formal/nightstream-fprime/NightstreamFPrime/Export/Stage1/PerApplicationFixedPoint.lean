import NightstreamFPrime.Export.Stage1.DirectApplicationPrefixPlan

/-!
Owns the finite recursive relation construction for one Lean-authored
application. The final logical width is the exact retained width. The final
SuperNeo relation is derived from the exact ordered Stage 1 matrix plan.

The zero relation below is construction scaffolding only. It can break the
value-level cycle because Stage 1 verifier rows depend on the relation shape,
not on caller-selected matrix entries. The fixed-point theorem must prove that
assembling with the derived relation gives the same plan.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def logicalWidth (application : Lifecycle.Stage1.Application.Program) : Nat :=
  ApplicationRetainedGeometry.completeLogicalWidth application

def publicFits (application : Lifecycle.Stage1.Application.Program) :
    ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth (logicalWidth application) := by
  apply Nat.le_trans (m := logicalWidth application)
  · unfold logicalWidth
    rw [ApplicationRetainedGeometry.completeLogicalWidth_eq]
    norm_num [ringDegree, publicRingColumns]
    omega
  · exact Phi81CarrierLayout.logicalWidth_le_carrierWidth _

/-- Exact finite obligations supplied by one concrete application package. -/
structure FitsTwoPow28
    (application : Lifecycle.Stage1.Application.Program) : Prop where
  package : PerApplicationPackage.FitsTwoPow28 application
  carrier : Phi81CarrierLayout.carrierWidth (logicalWidth application) ≤
    2 ^ Lifecycle.cubeVariables

/-- Construct every final domain obligation from the three small numerical
bounds owned by one concrete application. -/
def fitsTwoPow28OfApplicationBounds
    (application : Lifecycle.Stage1.Application.Program)
    (rows : (PerApplicationPackage.applicationPlan application).rowCount ≤
      239217427)
    (columns : PerApplicationPackage.addedPrivateColumnCount application ≤
      239098731)
    (carrierWords : application.witnessWordCount +
      ApplicationRetainedBlocks.localCount application ≤ 298023) :
    FitsTwoPow28 application where
  package := PerApplicationPackage.fitsTwoPow28OfApplicationBounds application
    rows columns
  carrier := by
    apply (ApplicationRetainedGeometry.carrierWidth_le_twoPow28_iff
      application).2
    exact carrierWords

def geometry (application : Lifecycle.Stage1.Application.Program) :
    ApplicationRetainedGeometry.Geometry application
      (logicalWidth application) where
  completeFits := Nat.le_refl _

/-- Shape-correct seed. Its matrices are not semantic authority. -/
def seedRelation (application : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 application) :
    ProductionKey.LogicalRelation (logicalWidth application)
      (publicFits application) where
  matrices := fun _ _ _ => 0
  cubeFits := fits.carrier

/-- Exact Lean-assembled plan used to derive the recursive relation. -/
def structuralPlan (application : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 application) :
    ProductionRelation.Plan (logicalWidth application) :=
  DirectApplicationPrefixPlan.plan (seedRelation application fits)
    fits.package (geometry application)

/-- The only key-facing relation for this application plan. -/
def relation (application : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 application) :
    ProductionKey.LogicalRelation (logicalWidth application)
      (publicFits application) :=
  (structuralPlan application fits).logicalRelation fits.carrier

/-- Reassembling Stage 1 against its derived relation is the same matrix plan. -/
theorem plan_fixedPoint (application : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 application) :
    DirectApplicationPrefixPlan.plan (relation application fits)
        fits.package (geometry application) =
      structuralPlan application fits := by
  unfold structuralPlan
  exact DirectApplicationPrefixPlan.plan_eq_of_same_shape
    (relation application fits) (seedRelation application fits)
    fits.package (geometry application)

/-- The key-facing matrices are exactly the matrices selected by the complete
ordered application plan. -/
@[simp] theorem relation_matrices
    (application : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 application) :
    (relation application fits).matrices =
      (structuralPlan application fits).matrix := by
  rfl

/-- Exact live-row count of the recursive plan through the selected
application. -/
@[simp] theorem structuralPlan_rowCount
    (application : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 application) :
    (structuralPlan application fits).rowCount =
      6369850 + (PerApplicationPackage.applicationPlan application).rowCount +
        9 := by
  unfold structuralPlan
  exact DirectApplicationPrefixPlan.plan_rowCount _ fits.package
    (geometry application)

theorem structuralPlan_rowCount_le
    (application : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 application) :
    (structuralPlan application fits).rowCount ≤
      2 ^ Lifecycle.cubeVariables :=
  (structuralPlan application fits).rowCount_le

/-- The final row cube and retained carrier fit one common `2^28` domain. -/
theorem jointDomain_le_twoPow28
    (application : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 application) :
    max (structuralPlan application fits).rowCount
        (Phi81CarrierLayout.carrierWidth (logicalWidth application)) ≤
      2 ^ Lifecycle.cubeVariables := by
  exact Nat.max_le.mpr
    ⟨structuralPlan_rowCount_le application fits, fits.carrier⟩

/-- Any assignment accepted by the self-derived matrix plan satisfies the
complete direct prefix and the exact verifier-owned application transition. -/
theorem rowsZero_implies_semantics
    (application : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 application)
    (assignment : Assignment F (logicalWidth application))
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment
      (ApplicationRetainedGeometry.oneColumn (geometry application)) = 1)
    (encodes : DirectApplicationPrefixPlan.Encodes (geometry application)
      assignment base groupValue products)
    (accepted : (structuralPlan application fits).RowsZero assignment) :
    DirectApplicationPrefixPlan.Semantics (relation application fits)
      (geometry application) assignment base groupValue products
        := by
  apply DirectApplicationPrefixPlan.rowsZero_implies_semantics
    (relation application fits) fits.package (geometry application) assignment
    base groupValue products one encodes
  rw [plan_fixedPoint application fits]
  exact accepted

end NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint
