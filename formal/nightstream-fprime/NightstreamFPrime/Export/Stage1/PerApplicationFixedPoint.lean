import NightstreamFPrime.Export.Stage1.DirectApplicationPrefixPlan

/-!
Owns the finite recursive relation construction for one Lean-authored
application.
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

theorem publicFits (application : Lifecycle.Stage1.Application.Program) :
    ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth (logicalWidth application) := by
  apply Nat.le_trans (m := logicalWidth application)
  · unfold logicalWidth
    rw [ApplicationRetainedGeometry.completeLogicalWidth_eq]
    norm_num [ringDegree, publicRingColumns]
    omega
  · exact Phi81CarrierLayout.logicalWidth_le_carrierWidth _

structure FitsTwoPow28
    (application : Lifecycle.Stage1.Application.Program) : Prop where
  package : PerApplicationPackage.FitsTwoPow28 application
  carrier : Phi81CarrierLayout.carrierWidth (logicalWidth application) ≤
    2 ^ Lifecycle.cubeVariables

theorem fitsTwoPow28OfApplicationBounds
    (application : Lifecycle.Stage1.Application.Program)
    (rows : (PerApplicationPackage.applicationPlan application).rowCount ≤
      257057539)
    (columns : PerApplicationPackage.addedPrivateColumnCount application ≤
      256970581)
    (carrierWords : application.witnessWordCount +
      ApplicationRetainedBlocks.localCount application ≤ 5480402) :
    FitsTwoPow28 application where
  package := PerApplicationPackage.fitsTwoPow28OfApplicationBounds application
    rows columns
  carrier := by
    apply (ApplicationRetainedGeometry.carrierWidth_le_twoPow28_iff
      application).2
    exact carrierWords

theorem geometry (application : Lifecycle.Stage1.Application.Program) :
    ApplicationRetainedGeometry.Geometry application
      (logicalWidth application) where
  completeFits := Nat.le_refl _

def seedRelation (application : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 application) :
    ProductionKey.LogicalRelation (logicalWidth application)
      (publicFits application) where
  matrices := fun _ _ _ => 0
  cubeFits := fits.carrier

def structuralPlan (application : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 application) :
    ProductionRelation.Plan (logicalWidth application) :=
  DirectApplicationPrefixPlan.plan (seedRelation application fits)
    fits.package (geometry application)

def relation (application : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 application) :
    ProductionKey.LogicalRelation (logicalWidth application)
      (publicFits application) :=
  (structuralPlan application fits).logicalRelation fits.carrier

theorem plan_fixedPoint (application : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 application) :
    DirectApplicationPrefixPlan.plan (relation application fits)
        fits.package (geometry application) =
      structuralPlan application fits := by
  unfold structuralPlan
  exact DirectApplicationPrefixPlan.plan_eq_of_same_shape
    (relation application fits) (seedRelation application fits)
    fits.package (geometry application)

@[simp] theorem relation_matrices
    (application : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 application) :
    (relation application fits).matrices =
      (structuralPlan application fits).matrix := by
  rfl

@[simp] theorem structuralPlan_rowCount
    (application : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 application) :
    (structuralPlan application fits).rowCount =
      998638 + (PerApplicationPackage.applicationPlan application).rowCount +
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

theorem jointDomain_le_twoPow28
    (application : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 application) :
    max (structuralPlan application fits).rowCount
        (Phi81CarrierLayout.carrierWidth (logicalWidth application)) ≤
      2 ^ Lifecycle.cubeVariables := by
  exact Nat.max_le.mpr
    ⟨structuralPlan_rowCount_le application fits, fits.carrier⟩

theorem rowsZero_implies_semantics
    (application : Lifecycle.Stage1.Application.Program)
    (fits : FitsTwoPow28 application)
    (assignment : Assignment F (logicalWidth application))
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (one : assignment
      (ApplicationRetainedGeometry.oneColumn (geometry application)) = 1)
    (encodes : DirectApplicationPrefixPlan.Encodes (geometry application)
      assignment base groupValue)
    (accepted : (structuralPlan application fits).RowsZero assignment) :
    DirectApplicationPrefixPlan.Semantics (relation application fits)
      (geometry application) assignment base groupValue
        := by
  apply DirectApplicationPrefixPlan.rowsZero_implies_semantics
    (relation application fits) fits.package (geometry application) assignment
    base groupValue one encodes
  rw [plan_fixedPoint application fits]
  exact accepted

end NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint
