import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransportExecution
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalNorm
import NightstreamFPrime.Layout.Stage1.PiRLCInputBounds
import NightstreamFPrime.Layout.PiRLC.v1_1.Preservation

/-!
Owns the source copy from a completed Spartan prefix and the application-owned
private suffix to the existing canonical assignment input. Prefix columns use
the existing per-application shift. Recovery is pointwise on the declared
source domain; values outside that domain have no equality contract.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PerApplicationSourceAssignment

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PerApplicationAssignmentTransportExecution

/-- Copy the completed base package, insert precisely the application-owned
private interval, and shift the old constant/public suffix by its declared
width. The output is the existing bounded physical-source function. -/
def ofCompleted (application : Lifecycle.Stage1.Application.Program)
    (target : Env)
    (applicationPrivate : Fin (PerApplicationPackage.addedPrivateColumnCount
      application) → F) : BaseValues application :=
  fun column =>
    if before : column.val < PerApplicationPackage.basePackage.layout.constantColumn then
      target column.val
    else if inside : column.val <
        PerApplicationPackage.basePackage.layout.constantColumn +
          PerApplicationPackage.addedPrivateColumnCount application then
      applicationPrivate ⟨column.val -
        PerApplicationPackage.basePackage.layout.constantColumn, by omega⟩
    else
      target (column.val - PerApplicationPackage.addedPrivateColumnCount application)

private theorem baseConstant_eq_private :
    PerApplicationPackage.basePackage.layout.constantColumn =
      Spartan.privateColumnCount := by
  rw [Spartan.privateColumnCount_eq]
  exact Package.circuitPackage_layout_values.2.2.1

private theorem baseTotal_eq_spartan :
    PerApplicationPackage.basePackage.layout.totalColumnCount =
      Spartan.spartanColumnCount := by
  rw [PerApplicationPackage.basePackage_totalColumnCount_eq,
    Spartan.spartanColumnCount_eq]

private theorem applicationColumn_lt
    (application : Lifecycle.Stage1.Application.Program)
    (index : Fin (PerApplicationPackage.addedPrivateColumnCount application)) :
    Spartan.privateColumnCount + index.val <
      PiRLCProductPlan.baseSourceWidth application := by
  rw [PiRLCProductPlan.baseSourceWidth,
    PerApplicationPackage.package_totalColumnCount, baseTotal_eq_spartan]
  have privateBound : Spartan.privateColumnCount < Spartan.spartanColumnCount := by
    rw [Spartan.privateColumnCount_eq, Spartan.spartanColumnCount_eq]
    decide
  have indexBound := index.isLt
  omega

/-- Each application-private source slot contains exactly the supplied
application witness, local, or lowering value at its existing offset. -/
theorem application_ofCompleted
    (application : Lifecycle.Stage1.Application.Program)
    (target : Env)
    (applicationPrivate : Fin (PerApplicationPackage.addedPrivateColumnCount
      application) → F)
    (index : Fin (PerApplicationPackage.addedPrivateColumnCount application)) :
    ofCompleted application target applicationPrivate
      ⟨Spartan.privateColumnCount + index.val,
        applicationColumn_lt application index⟩ = applicationPrivate index := by
  have before : ¬(Spartan.privateColumnCount + index.val <
      PerApplicationPackage.basePackage.layout.constantColumn) := by
    rw [baseConstant_eq_private]
    omega
  have inside : Spartan.privateColumnCount + index.val <
      PerApplicationPackage.basePackage.layout.constantColumn +
        PerApplicationPackage.addedPrivateColumnCount application := by
    rw [baseConstant_eq_private]
    exact Nat.add_lt_add_left index.isLt _
  unfold ofCompleted
  rw [dif_neg before, dif_pos inside]
  apply congrArg applicationPrivate
  apply Fin.ext
  change Spartan.privateColumnCount + index.val -
    PerApplicationPackage.basePackage.layout.constantColumn = index.val
  rw [baseConstant_eq_private, Nat.add_sub_cancel_left]

private theorem shifted_ofCompleted
    (application : Lifecycle.Stage1.Application.Program)
    (target : Env)
    (applicationPrivate : Fin (PerApplicationPackage.addedPrivateColumnCount
      application) → F)
    (column : Nat)
    (bound : column < PiRLCProductPlan.basePackage.layout.totalColumnCount) :
    ofCompleted application target applicationPrivate
      (PiRLCProductPlan.shiftedPackageColumn application column bound) =
        target column := by
  unfold ofCompleted PiRLCProductPlan.shiftedPackageColumn
  dsimp only
  by_cases before : column < PerApplicationPackage.basePackage.layout.constantColumn
  · simp only [PerApplicationPackage.shiftColumn, if_pos before, dif_pos before]
  · simp only [PerApplicationPackage.shiftColumn, if_neg before]
    rw [dif_neg (by omega), dif_neg (by omega)]
    rw [Nat.add_sub_cancel_right]

/-- The canonical physical-source packet recovers the completed prefix on
its full declared logical-source domain. The proof uses the existing Spartan
map and package shift; it assumes no equality between caller environments. -/
theorem source_ofCompleted
    (application : Lifecycle.Stage1.Application.Program)
    (target : Env)
    (applicationPrivate : Fin (PerApplicationPackage.addedPrivateColumnCount
      application) → F)
    (column : Nat) (bound : column < Spartan.SourceColumnCount) :
    PiRLCFirst54DirectPlan.baseEnv application
      (ofCompleted application target applicationPrivate) column =
        Spartan.pullback target column := by
  have mappedBound : Spartan.sourceToSpartan column <
      PiRLCProductPlan.basePackage.layout.totalColumnCount := by
    change Spartan.sourceToSpartan column <
      PerApplicationPackage.basePackage.layout.totalColumnCount
    rw [baseTotal_eq_spartan]
    exact Spartan.sourceToSpartan_lt column bound
  change SourceCompiler.sourceEnv (ofCompleted application target applicationPrivate)
      (PerApplicationPackage.shiftColumn application (Spartan.sourceToSpartan column)) =
    target (Spartan.sourceToSpartan column)
  change SourceCompiler.sourceEnv (ofCompleted application target applicationPrivate)
      (PiRLCProductPlan.shiftedPackageColumn application
        (Spartan.sourceToSpartan column) mappedBound).val = _
  rw [SourceCompiler.sourceEnv_at]
  exact shifted_ofCompleted application target applicationPrivate _ mappedBound

/-- Actual PiRLC rows in the completed Spartan prefix imply the strict norm
bound on the complete canonical carrier after the source copy. The phase's
existing soundness and completeness interfaces establish its row scope. The
fixed PiRLC input assumptions follow from `PiRLCInputBounds.assumptions` for
the copied source. No bit, scope, source-equality, or cryptographic premise
is supplied; the existing Ajtai key is used by the phase interfaces. -/
theorem completeAssignment_norm_of_completedRows
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (target : Env)
    (applicationPrivate : Fin (PerApplicationPackage.addedPrivateColumnCount
      application) → F)
    (rows : holdsFlat (Spartan.pullback target)
      (Formal.opsAt (PerApplicationFixedPoint.relation application fits)
        PiRLCInputs.interface PiRLCInputs.phaseOffset))
    (column : Fin (Phi81CarrierLayout.carrierWidth
      (PerApplicationFixedPoint.logicalWidth application))) :
    centeredMagnitude
      ((canonicalRawValues application
        (ofCompleted application target applicationPrivate)).completeAssignment column) < 2 := by
  let raw := canonicalRawValues application
    (ofCompleted application target applicationPrivate)
  have assumptions := PiRLCInputBounds.assumptions
    (PerApplicationFixedPoint.relation application fits) (Spartan.pullback target)
  have phase := (Formal.circuit
    (PerApplicationFixedPoint.relation application fits) ajtai PiRLCInputs.interface).soundness
      (Spartan.pullback target) PiRLCInputs.phaseOffset assumptions
      (holdsFlat_implies_holds _ _ rows)
  have phaseScope := NightstreamFPrime.Layout.PiRLC.v1_1.plan_constraints_varsBelow_of_phase
    (PerApplicationFixedPoint.relation application fits) ajtai PiRLCInputs.interface
    PiRLCInputs.phaseOffset (Spartan.pullback target) assumptions phase
  change ∀ expression ∈ flatConstraints
      (Formal.opsAt (PerApplicationFixedPoint.relation application fits)
        PiRLCInputs.interface PiRLCInputs.phaseOffset),
    expression.VarsBelow (PiRLCInputs.phaseOffset + Formal.logicalPrivateCount) at phaseScope
  have endpoint : PiRLCInputs.phaseOffset + Formal.logicalPrivateCount ≤
      Spartan.SourceColumnCount := by
    rw [Spartan.sourceColumnCount_eq]
    norm_num [PiRLCInputs.phaseOffset, Formal.logicalPrivateCount]
  have scope : ∀ expression ∈ flatConstraints
      (Formal.opsAt (PerApplicationFixedPoint.relation application fits)
        PiRLCInputs.interface PiRLCInputs.phaseOffset),
      expression.VarsBelow Spartan.SourceColumnCount := by
    intro expression member
    exact Expr.VarsBelow.mono expression (phaseScope expression member) endpoint
  have copiedRows : holdsFlat (PiRLCFirst54DirectPlan.baseEnv application raw.base)
      (Formal.opsAt (PerApplicationFixedPoint.relation application fits)
        PiRLCInputs.interface PiRLCInputs.phaseOffset) := by
    apply constraintsHold_of_agree_below (Spartan.pullback target)
      (PiRLCFirst54DirectPlan.baseEnv application raw.base)
      _ Spartan.SourceColumnCount scope _ rows
    intro index bound
    exact source_ofCompleted application target applicationPrivate index bound
  exact PerApplicationCanonicalAssignment.completeAssignment_norm_of_piRlcRows
    fits raw (PiRLCInputBounds.assumptions
      (PerApplicationFixedPoint.relation application fits) _) copiedRows column

end NightstreamFPrime.Export.Stage1.PerApplicationSourceAssignment
