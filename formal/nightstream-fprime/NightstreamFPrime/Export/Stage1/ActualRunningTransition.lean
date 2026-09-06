import NightstreamFPrime.Export.Stage1.RunningTransitionMatrixProgramSubstitution
import NightstreamFPrime.Export.Stage1.ActualPreimageFraming
import NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint

/-!
Owns the running-transition contract decoded from arbitrary accepted logical
rows. The decoder evaluates the existing source map. No raw packet or encoding
premise supplies its values. Parent preimage wiring is proved separately.
-/

namespace NightstreamFPrime.Export.Stage1.ActualRunningTransition

open NightstreamFPrime.Circuit
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
    (geometry : RunningTransitionRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) : Env :=
  SourceCompiler.sourceEnv fun column =>
    ((RunningTransitionDirectPlan.sourceMap geometry).form column).eval assignment

theorem decodedEnv_preserves
    (geometry : RunningTransitionRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) :
    (RunningTransitionDirectPlan.sourceMap geometry).Preserves assignment
      (decodedEnv geometry assignment) := by
  intro column
  exact (SourceCompiler.sourceEnv_at
    (fun column => ((RunningTransitionDirectPlan.sourceMap geometry).form column).eval
      assignment) column).symm

/-- Every accepted ordinary transition row holds in the environment decoded
from that same assignment. The source map supplies all preservation facts. -/
theorem rowsZero_implies_physical
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : RunningTransitionRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (RunningTransitionRetainedGeometry.oneColumn geometry) = 1)
    (rows : (RunningTransitionDirectPlan.plan relation geometry).RowsZero assignment) :
    RunningTransitionLayout.PhysicalHolds relationLogicalWidth relationPublicFits
      (Spartan.pullback (decodedEnv geometry assignment)) := by
  apply (RunningTransitionDirectSource.program_holds_iff_physical relation
    (decodedEnv geometry assignment)).mp
  apply (OrdinarySourcePlan.Program.rowsZero_iff
    (RunningTransitionDirectSource.program relation)
    (RunningTransitionDirectPlan.inputs relation geometry) assignment
    (decodedEnv geometry assignment) one ?_).mp rows
  intro index
  refine ⟨?_, ?_, ?_⟩ <;> intro term member <;>
    exact decodedEnv_preserves geometry assignment _

/-- The existing opaque physical contract applies to every accepted logical
assignment; no `Encodes`, `RawValues`, or generated witness is assumed. -/
theorem rowsZero_implies_specHolds
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : RunningTransitionRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (RunningTransitionRetainedGeometry.oneColumn geometry) = 1)
    (rows : (RunningTransitionDirectPlan.plan relation geometry).RowsZero assignment) :
    Lifecycle.Stage1.RunningTransition.SpecHolds
      (RunningTransitionInputs.interface relationLogicalWidth relationPublicFits)
      RunningTransitionInputs.phaseOffset
      (Spartan.pullback (decodedEnv geometry assignment)) :=
  RunningTransitionLayout.physical_implies_specHolds relation _
    (rowsZero_implies_physical relation geometry assignment one rows)

private theorem decodedEnv_location
    (geometry : RunningTransitionRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (location : RunningTransitionDirectPlan.Location)
    (support : RunningTransitionSourceSupport.Source location.sourceColumn) :
    (Spartan.pullback (decodedEnv geometry assignment)) location.sourceColumn =
      (location.form geometry).eval assignment := by
  let column : Fin Spartan.spartanColumnCount :=
    ⟨Spartan.sourceToSpartan location.sourceColumn,
      Spartan.sourceToSpartan_lt _ location.sourceColumn_lt⟩
  have target : RunningTransitionSourceSupport.Target column.val :=
    ⟨location.sourceColumn, support, rfl⟩
  have mapped := RunningTransitionMatrixProgram.substitution_agrees_on_target
    geometry column target
  have selected := RunningTransitionMatrixProgram.substitution_location_form?
    geometry location
  have same : (RunningTransitionDirectPlan.sourceMap geometry).form column =
      location.form geometry := Option.some.inj (mapped.symm.trans selected)
  change SourceCompiler.sourceEnv
    (fun column => ((RunningTransitionDirectPlan.sourceMap geometry).form column).eval
      assignment) column.val = _
  rw [SourceCompiler.sourceEnv_at, same]

/-- The transition's state input is read from its declared owned form. -/
theorem stateWord_eq_form
    (geometry : RunningTransitionRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (index : Fin RunningTransitionSourceSupport.stateCount) :
    (Spartan.pullback (decodedEnv geometry assignment))
        (RunningTransitionSourceSupport.stateStart + index.val) =
      ((RunningTransitionDirectPlan.Location.state index).form geometry).eval assignment := by
  apply decodedEnv_location geometry assignment (.state index)
  apply Or.inl
  apply Or.inl
  change RunningTransitionSourceSupport.stateStart ≤
      RunningTransitionSourceSupport.stateStart + index.val ∧
    RunningTransitionSourceSupport.stateStart + index.val <
      RunningTransitionSourceSupport.stateStart + RunningTransitionSourceSupport.stateCount
  have bound := index.isLt
  omega

/-- The transition's complete output uses its declared owned preimage forms. -/
theorem outputWord_eq_form
    (geometry : RunningTransitionRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (index : Fin RunningTransitionSourceSupport.outputCount) :
    (Spartan.pullback (decodedEnv geometry assignment))
        (RunningTransitionSourceSupport.outputStart + index.val) =
      ((RunningTransitionDirectPlan.Location.output index).form geometry).eval assignment := by
  apply decodedEnv_location geometry assignment (.output index)
  apply Or.inl
  apply Or.inr
  apply Or.inl
  change RunningTransitionSourceSupport.outputStart ≤
      RunningTransitionSourceSupport.outputStart + index.val ∧
    RunningTransitionSourceSupport.outputStart + index.val <
      RunningTransitionSourceSupport.outputStart + RunningTransitionSourceSupport.outputCount
  have bound := index.isLt
  omega

def selectedGeometry (application : Lifecycle.Stage1.Application.Program) :
    RunningTransitionRetainedGeometry.Geometry application
      (PerApplicationFixedPoint.logicalWidth application) :=
  DirectPiDECPrefixPlan.runningGeometry
    (DirectApplicationPrefixPlan.piDecGeometry
      (PerApplicationFixedPoint.geometry application))

private theorem selectedStateWord_eq_prior
    (application : Lifecycle.Stage1.Application.Program)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (index : Fin RunningTransitionSourceSupport.stateCount) :
    (Spartan.pullback (decodedEnv (selectedGeometry application) assignment))
        (RunningTransitionSourceSupport.stateStart + index.val) =
      ActualPreimageFraming.priorState
        (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
          (PerApplicationFixedPoint.geometry application)) assignment (28 + index.val) := by
  rw [stateWord_eq_form, RunningTransitionDirectPlan.Location.state_form_eq_pilot]
  have bounded : 28 + index.val < PilotProduction.stateHashWords := by
    have bound := index.isLt
    change index.val < 11 at bound
    rw [PilotProduction.stateHashWords_eq]
    omega
  rw [ActualPreimageFraming.priorState, dif_pos bounded]
  rfl

private theorem selectedOutputWord_eq_next
    (application : Lifecycle.Stage1.Application.Program)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (index : Fin PilotProduction.stateHashWords) :
    (Spartan.pullback (decodedEnv (selectedGeometry application) assignment))
        (PilotProduction.outputPreimageStart + index.val) =
      ActualPreimageFraming.outputState
        (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
          (PerApplicationFixedPoint.geometry application)) assignment index.val := by
  change (Spartan.pullback (decodedEnv (selectedGeometry application) assignment))
    (RunningTransitionSourceSupport.outputStart + index.val) = _
  rw [outputWord_eq_form, RunningTransitionDirectPlan.Location.output_form_eq_pilot]
  rw [ActualPreimageFraming.outputState, dif_pos index.isLt]
  rfl

/-- The full typed running output is the output preimage used by the pilot.
Every point, commitment, public input, and evaluation uses the same forms. -/
theorem selectedOutputRunning_eq_running
    (application : Lifecycle.Stage1.Application.Program)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application)) :
    PiCCS.v1_1.StatementAbsorption.evalRunning
        (RunningTransitionInputs.outputRunningExpr
          (PerApplicationFixedPoint.logicalWidth application)
          (PerApplicationFixedPoint.publicFits application))
        (Spartan.pullback (decodedEnv (selectedGeometry application) assignment)) =
      StateDecoder.running (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application)
        (ActualPreimageFraming.outputState
          (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
            (PerApplicationFixedPoint.geometry application)) assignment) := by
  rw [StateDecoder.evalOutputRunning_eq_running]
  unfold StateDecoder.running
  apply congrArg (PiCCSInputs.decodedRunning
    (PerApplicationFixedPoint.logicalWidth application)
    (PerApplicationFixedPoint.publicFits application))
  unfold StateDecoder.externalValues
  congr 1
  funext word
  exact selectedOutputWord_eq_next application assignment word

/-- The sole complete selected row plan contains the running-transition rows.
This projection preserves the arbitrary assignment and the existing layout. -/
theorem selectedRowsZero_implies_specHolds
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (one : assignment (ApplicationRetainedGeometry.oneColumn
      (PerApplicationFixedPoint.geometry application)) = 1)
    (rows : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero
      assignment) :
    Lifecycle.Stage1.RunningTransition.SpecHolds
      (RunningTransitionInputs.interface
        (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      RunningTransitionInputs.phaseOffset
      (Spartan.pullback (decodedEnv (selectedGeometry application) assignment)) := by
  let relation := PerApplicationFixedPoint.relation application fits
  let geometry := PerApplicationFixedPoint.geometry application
  have selected : (DirectApplicationPrefixPlan.plan relation fits.package geometry
      ).RowsZero assignment := by
    rw [PerApplicationFixedPoint.plan_fixedPoint]
    exact rows
  have children := (DirectApplicationPrefixPlan.rowsZero_iff relation
    fits.package geometry assignment).mp selected
  have prefixRows := (DirectPiRLCSamplerCompletePrefixPlan.rowsZero_iff relation
    (DirectApplicationPrefixPlan.prefixGeometry geometry) assignment).mp children.1.1.1
  exact rowsZero_implies_specHolds relation (selectedGeometry application)
    assignment one prefixRows.2.2.2.2

/-- HyperNova's base branch follows from the actual public boundary and
accepted rows: the initial/current states agree and the hashed running output
is the complete default value. This is one branch of the full step target. -/
theorem selectedRowsAndPublic_imply_baseState
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (digest : Digest)
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest)
    (rows : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment)
    (iterationZero : StateDecoder.iteration
      (ActualPreimageFraming.priorState
        (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
          (PerApplicationFixedPoint.geometry application)) assignment) = 0) :
    let geometry := PerApplicationFixedPoint.geometry application
    let prior := ActualPreimageFraming.priorState
      (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment
    let next := ActualPreimageFraming.outputState
      (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment
    StateDecoder.initialState prior = StateDecoder.currentState prior ∧
      StateDecoder.running (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application) next =
      defaultRunning (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application) := by
  let geometry := PerApplicationFixedPoint.geometry application
  let prior := ActualPreimageFraming.priorState
    (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment
  let env := Spartan.pullback (decodedEnv (selectedGeometry application) assignment)
  have publicBound : RecursivePublicOutputPlan.publicInput geometry assignment =
      encHash (publicFits := RecursivePublicOutputPlan.carrierPublicFits geometry) digest := by
    rw [RecursivePublicOutputPlan.publicInput_eq_projectPublicInput]
    exact publicEqual
  have one := RecursivePublicOutputPlan.publicEqual_implies_one
    geometry assignment digest publicBound
  have specification := selectedRowsZero_implies_specHolds application fits
    assignment one rows
  have fieldZero : prior 28 = 0 := by
    simpa only [StateDecoder.iteration, StateDecoder.natWord_val] using
      congrArg natWord iterationZero
  have iterationRead : Lifecycle.Stage1.RunningTransition.iterationValue
      (RunningTransitionInputs.interface
        (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      RunningTransitionInputs.phaseOffset env = prior 28 := by
    simpa only [Lifecycle.Stage1.RunningTransition.iterationValue,
      RunningTransitionInputs.interface, RunningTransitionInputs.iterationExpr,
      RunningTransitionInputs.iterationWordIndex, Expr.eval, Nat.add_zero,
      RunningTransitionSourceSupport.stateStart_eq] using
      selectedStateWord_eq_prior application assignment ⟨0, by decide⟩
  have transitionZero := iterationRead.trans fieldZero
  constructor
  · change StateDecoder.initialState prior = StateDecoder.currentState prior
    unfold StateDecoder.initialState StateDecoder.currentState StateDecoder.slice
    apply congrArg List.ofFn
    funext index
    have bound : index.val < 4 := index.isLt
    have initial := selectedStateWord_eq_prior application assignment
      ⟨2 + index.val, by change 2 + index.val < 11; omega⟩
    have current := selectedStateWord_eq_prior application assignment
      ⟨7 + index.val, by change 7 + index.val < 11; omega⟩
    change env (28 + (2 + index.val)) = prior (28 + (2 + index.val)) at initial
    change env (28 + (7 + index.val)) = prior (28 + (7 + index.val)) at current
    rw [show 28 + (2 + index.val) = 30 + index.val by omega] at initial
    rw [show 28 + (7 + index.val) = 35 + index.val by omega] at current
    have native := specification.initialState transitionZero index
    change env (30 + index.val) = env (35 + index.val) at native
    exact initial.symm.trans (native.trans current)
  · have base := RunningTransitionInputs.spec_typed_base specification transitionZero
    rw [selectedOutputRunning_eq_running] at base
    exact base

end NightstreamFPrime.Export.Stage1.ActualRunningTransition
