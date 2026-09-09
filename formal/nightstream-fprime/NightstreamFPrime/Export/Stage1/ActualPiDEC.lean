import NightstreamFPrime.Export.Stage1.PiDECMatrixProgramSubstitution
import NightstreamFPrime.Export.Stage1.PiCCSAssignmentSoundness
import NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint
import NightstreamFPrime.Layout.Stage1.PiDECInputBounds

/-!
Owns PiDEC values and its opaque phase contract on arbitrary accepted rows.
PiDEC-owned forms supply its arithmetic values. Other reads retain the PiCCS
parent decoder, including the shared evaluation point. No raw packet or
semantic representation is a premise. The NIFS parent connection is separate.
-/

namespace NightstreamFPrime.Export.Stage1.ActualPiDEC

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

variable {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
  {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

def decodedEnv
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) : Env := fun column =>
  match PiDECDirectPlan.classifyTarget column with
  | some decoded => (decoded.location.form geometry).eval assignment
  | none => PiCCSAssignmentSoundness.decodedEnv
      (PiDECRetainedGeometry.prefixGeometry geometry) assignment column

/-- Every column used by a PiDEC row has its own declared form value. -/
theorem decodedEnv_preserves_target
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (column : Fin Spartan.spartanColumnCount)
    (support : PiDECSourceSupport.Target column.val) :
    ((PiDECDirectPlan.sourceMap geometry).form column).eval assignment =
      decodedEnv geometry assignment column.val := by
  rcases PiDECDirectPlan.classifyTarget_complete support with
    ⟨decoded, found, _mapped⟩
  simp only [PiDECDirectPlan.sourceMap, decodedEnv, found]

/-- Each PiDEC value resolves to the existing physical owner's logical form. -/
theorem decodedEnv_location
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (location : PiDECDirectPlan.Location) :
    (Spartan.pullback (decodedEnv geometry assignment)) location.sourceColumn =
      (location.form geometry).eval assignment := by
  let column : Fin Spartan.spartanColumnCount :=
    ⟨Spartan.sourceToSpartan location.sourceColumn,
      Spartan.sourceToSpartan_lt _ location.sourceColumn_lt⟩
  have support : PiDECSourceSupport.Target column.val :=
    ⟨location.sourceColumn, location.sourceSupport, rfl⟩
  have mapped := PiDECMatrixProgram.substitution_agrees_on_target
    geometry column support
  have selected := PiDECMatrixProgram.substitution_location_form? geometry location
  have same : (PiDECDirectPlan.sourceMap geometry).form column =
      location.form geometry := Option.some.inj (mapped.symm.trans selected)
  exact (decodedEnv_preserves_target geometry assignment column support).symm.trans
    (congrArg (fun form => form.eval assignment) same)

private theorem location_afterPiCCS (location : PiDECDirectPlan.Location) :
    PiRLCInputs.phaseOffset ≤ location.sourceColumn := by
  cases location <;>
    simp only [PiDECDirectPlan.Location.sourceColumn,
      PiDECSourceSupport.parentCommitmentStart_eq,
      PiDECSourceSupport.parentPublicInputStart_eq,
      PiDECSourceSupport.parentEvalKStart_eq, PiDECSourceSupport.parentEvalAStart_eq] <;>
    norm_num [PiRLCInputs.phaseOffset, PiDECInputs.proofInputStart,
      PiDECStarts.phaseLogicalStart, PiDECStarts.phaseFreshStart,
      PiDECInputs.phaseOffset, PiDECInputs.proofInputColumnCount_eq,
      Lifecycle.PiDEC.v1_1.Formal.logicalPrivateCount] <;> omega

private theorem decodedEnv_beforePiRLC
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Nat)
    (before : source < PiRLCInputs.phaseOffset)
    (bounded : source < Spartan.SourceColumnCount) :
    (Spartan.pullback (decodedEnv geometry assignment)) source =
      (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
        (PiDECRetainedGeometry.prefixGeometry geometry) assignment)) source := by
  have outside : PiDECDirectPlan.classifyTarget
      (Spartan.sourceToSpartan source) = none := by
    unfold PiDECDirectPlan.classifyTarget
    rw [Spartan.spartanToSource_sourceToSpartan source bounded]
    cases found : PiDECDirectPlan.classifySource source with
    | none => simp only [found]
    | some located =>
        have after := location_afterPiCCS located.location
        rw [located.owns] at after
        omega
  change decodedEnv geometry assignment (Spartan.sourceToSpartan source) = _
  simp only [decodedEnv, outside, Spartan.pullback]

private theorem pointWord_eq_piCcs
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (coordinate : Fin productionShape.cubeVariables) (component : Fin 2) :
    (Spartan.pullback (decodedEnv geometry assignment))
        (PiCCSTranscriptOutputForms.pointSource coordinate component) =
      (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
        (PiDECRetainedGeometry.prefixGeometry geometry) assignment))
        (PiCCSTranscriptOutputForms.pointSource coordinate component) := by
  have before : PiCCSTranscriptOutputForms.pointSource coordinate component <
      PiRLCInputs.phaseOffset := by
    have bound : (PiCCSTranscriptOutputForms.pointInvocation coordinate component).val <
        718 := by
      simpa only [PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq] using
        (PiCCSTranscriptOutputForms.pointInvocation coordinate component).isLt
    rw [PiCCSTranscriptOutputForms.pointSource_eq_transcriptSource]
    norm_num [PiCCSTranscriptOutputForms.transcriptSource,
      PiCCSTranscriptOutputForms.transcriptSourceStart, PiCCSInputs.phaseOffset_eq,
      PiRLCInputs.phaseOffset]
    omega
  apply decodedEnv_beforePiRLC geometry assignment _ before
  apply Nat.lt_of_lt_of_le before
  rw [Spartan.sourceColumnCount_eq]
  norm_num [PiRLCInputs.phaseOffset]

/-- PiDEC retains the complete evaluation point from its PiCCS parent. Its
arithmetic decoder cannot replace any point word with a zero or local value. -/
theorem evalPoint_eq_piCcs
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) :
    Lifecycle.PiDEC.v1_1.InputBinding.evalPoint
        ((PiDECInputs.interface relationLogicalWidth relationPublicFits).point
          PiDECInputs.phaseOffset)
        (Spartan.pullback (decodedEnv geometry assignment)) =
      Lifecycle.PiDEC.v1_1.InputBinding.evalPoint
        ((PiDECInputs.interface relationLogicalWidth relationPublicFits).point
          PiDECInputs.phaseOffset)
        (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
          (PiDECRetainedGeometry.prefixGeometry geometry) assignment)) := by
  unfold Lifecycle.PiDEC.v1_1.InputBinding.evalPoint
    Lifecycle.PiRLC.v1_1.InputBinding.evalPoint
    Lifecycle.PiCCS.v1_1.StatementAbsorption.evalPoint
  congr 2
  funext coordinate
  change ((RunningTransitionInputs.recursiveRunningExpr
    relationLogicalWidth relationPublicFits).point coordinate).eval _ =
    ((RunningTransitionInputs.recursiveRunningExpr
      relationLogicalWidth relationPublicFits).point coordinate).eval _
  rw [RunningTransitionInputs.recursivePoint_eq_direct]
  simp only [RunningTransitionInputs.directRoundPoint, KExpr.eval]
  apply congrArg₂ K.mk
  · simpa only [RunningTransitionInputs.directRoundPoint, KExpr.eval, Expr.eval,
      PiCCSTranscriptOutputForms.pointSource_c0] using
      pointWord_eq_piCcs geometry assignment coordinate 0
  · simpa only [RunningTransitionInputs.directRoundPoint, KExpr.eval, Expr.eval,
      PiCCSTranscriptOutputForms.pointSource_c1] using
      pointWord_eq_piCcs geometry assignment coordinate 1

private theorem compiledRowsZero_implies_sourceRows
    {sourceRows : List R1CS.Row}
    (source : PiDECDirectPlan.SupportedProgram sourceRows)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiDECRetainedGeometry.oneColumn geometry) = 1)
    (rows : (source.toProgram.compile
      (PiDECDirectPlan.inputs source.toProgram geometry)).toPlan.RowsZero assignment) :
    R1CS.RowsHold (decodedEnv geometry assignment) sourceRows := by
  have preserves : ∀ index, OrdinarySourcePlan.SourceMap.PreservesRow
      ((PiDECDirectPlan.inputs source.toProgram geometry).sourceMap index)
      assignment (decodedEnv geometry assignment)
      (source.toProgram.row index) (source.toProgram.bounded index) := by
    intro index
    have supported := source.supported index
    refine ⟨?_, ?_, ?_⟩
    · intro term member
      exact decodedEnv_preserves_target geometry assignment
        ⟨term.1, (source.toProgram.bounded index).1 term member⟩
        (supported.1 term member)
    · intro term member
      exact decodedEnv_preserves_target geometry assignment
        ⟨term.1, (source.toProgram.bounded index).2.1 term member⟩
        (supported.2.1 term member)
    · intro term member
      exact decodedEnv_preserves_target geometry assignment
        ⟨term.1, (source.toProgram.bounded index).2.2 term member⟩
        (supported.2.2 term member)
  have sourceHolds := (OrdinarySourcePlan.Program.rowsZero_iff source.toProgram
    (PiDECDirectPlan.inputs source.toProgram geometry) assignment
    (decodedEnv geometry assignment) one preserves).mp rows
  rw [← source.exactRows]
  exact List.forall_mem_ofFn_iff.mpr sourceHolds

/-- The four compiled packets imply their exact canonical source rows in one
decoded environment. The proof composes indexed contracts, not row data. -/
theorem rowsZero_implies_sourceRows
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiDECRetainedGeometry.oneColumn geometry) = 1)
    (rows : (PiDECDirectPlan.plan relation geometry).RowsZero assignment) :
    R1CS.RowsHold (decodedEnv geometry assignment)
      (PiDECOrdinaryDirectSource.sourceRows relationLogicalWidth relationPublicFits) := by
  simp only [PiDECDirectPlan.plan, PiDECDirectPlan.recompositionPlan,
    PiDECDirectPlan.evaluationPlan, ProductionRelation.Plan.append_rowsZero_iff] at rows
  have publicRows := compiledRowsZero_implies_sourceRows
    (PiDECDirectPlan.publicSource relation) geometry assignment one rows.1
  have commitmentRows := compiledRowsZero_implies_sourceRows
    (PiDECDirectPlan.commitmentSource relation) geometry assignment one rows.2.1
  have evalKRows := compiledRowsZero_implies_sourceRows
    (PiDECDirectPlan.evalKSource relation) geometry assignment one rows.2.2.1
  have evalARows := compiledRowsZero_implies_sourceRows
    (PiDECDirectPlan.evalASource relation) geometry assignment one rows.2.2.2
  simp only [PiDECOrdinaryDirectSource.sourceRows, R1CS.rowsHold_append]
  exact ⟨⟨⟨publicRows, commitmentRows⟩, evalKRows⟩, evalARows⟩

/-- The same arbitrary assignment satisfies the canonical physical PiDEC
layout after the proved source permutation. -/
theorem rowsZero_implies_physical
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiDECRetainedGeometry.oneColumn geometry) = 1)
    (rows : (PiDECDirectPlan.plan relation geometry).RowsZero assignment) :
    Layout.PiDEC.v1_1.PhysicalHolds relation
      (PiDECArithmetic.phaseInterface relationLogicalWidth relationPublicFits)
      PiDECInputs.phaseOffset (Spartan.pullback (decodedEnv geometry assignment)) := by
  have sourceRows := rowsZero_implies_sourceRows relation geometry assignment one rows
  rw [PiDECOrdinaryDirectSource.sourceRows_eq_canonical] at sourceRows
  have exactRows := PiDECArithmetic.Plan.rows_to_layout
    (PiDECArithmetic.canonicalPlan relationLogicalWidth relationPublicFits)
    (PiDECArithmetic.canonicalLayoutPlan relation)
    (PiDECArithmetic.canonicalPlan_matches relation)
  rw [exactRows] at sourceRows
  exact (Spartan.remapRows_hold (decodedEnv geometry assignment)
    (PiDECArithmetic.canonicalLayoutPlan relation).rows).mp sourceRows

/-- The canonical input bounds and opaque physical contract derive the exact
PiDEC output predicate, with no caller-supplied representation or scope fact. -/
theorem rowsZero_implies_phaseHolds
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (ajtai : AjtaiKey (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits))
    (geometry : PiDECRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiDECRetainedGeometry.oneColumn geometry) = 1)
    (rows : (PiDECDirectPlan.plan relation geometry).RowsZero assignment) :
    Lifecycle.PiDEC.v1_1.Semantics.PhaseHolds relation ajtai
      (PiDECArithmetic.phaseInterface relationLogicalWidth relationPublicFits)
      PiDECInputs.phaseOffset (Spartan.pullback (decodedEnv geometry assignment)) :=
  Layout.PiDEC.v1_1.physical_implies_phaseHolds relation ajtai
    (PiDECArithmetic.phaseInterface relationLogicalWidth relationPublicFits)
    PiDECInputs.phaseOffset (Spartan.pullback (decodedEnv geometry assignment))
    (PiDECInputs.assumptions relation _)
    (rowsZero_implies_physical relation geometry assignment one rows)

def selectedGeometry (application : Lifecycle.Stage1.Application.Program) :
    PiDECRetainedGeometry.Geometry application
      (PerApplicationFixedPoint.logicalWidth application) :=
  DirectApplicationPrefixPlan.piDecGeometry (PerApplicationFixedPoint.geometry application)

/-- The sole selected Stage 1 plan and actual public boundary imply PiDEC in
its decoded environment. The public marker supplies the one coordinate. -/
theorem selectedRowsAndPublic_imply_phaseHolds
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (digest : Digest)
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest)
    (rows : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment) :
    Lifecycle.PiDEC.v1_1.Semantics.PhaseHolds
      (PerApplicationFixedPoint.relation application fits) ajtai
      (PiDECArithmetic.phaseInterface (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      PiDECInputs.phaseOffset
      (Spartan.pullback (decodedEnv (selectedGeometry application) assignment)) := by
  let relation := PerApplicationFixedPoint.relation application fits
  let geometry := PerApplicationFixedPoint.geometry application
  have publicBound : RecursivePublicOutputPlan.publicInput geometry assignment =
      encHash (publicFits := RecursivePublicOutputPlan.carrierPublicFits geometry) digest := by
    rw [RecursivePublicOutputPlan.publicInput_eq_projectPublicInput]
    exact publicEqual
  have one := RecursivePublicOutputPlan.publicEqual_implies_one
    geometry assignment digest publicBound
  have selected : (DirectApplicationPrefixPlan.plan relation fits.package geometry
      ).RowsZero assignment := by
    rw [PerApplicationFixedPoint.plan_fixedPoint]
    exact rows
  have children := (DirectApplicationPrefixPlan.rowsZero_iff relation
    fits.package geometry assignment).mp selected
  have prefixRows := (DirectPiRLCSamplerCompletePrefixPlan.rowsZero_iff relation
    (DirectApplicationPrefixPlan.prefixGeometry geometry) assignment).mp children.1.1.1
  exact rowsZero_implies_phaseHolds relation ajtai (selectedGeometry application)
    assignment one prefixRows.2.2.2.1

end NightstreamFPrime.Export.Stage1.ActualPiDEC
