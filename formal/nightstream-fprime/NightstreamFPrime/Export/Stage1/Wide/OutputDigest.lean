import NightstreamFPrime.Export.Stage1.Wide.CommonSchedule
import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransportExpressions

/-! The emitted output-digest recipe reads the four canonical public source
values. Its equality follows from checked relocation and the source assignment. -/

namespace NightstreamFPrime.Export.Stage1.Wide.OutputDigest

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle ProductionRelation
open AssignmentTransport AssignmentTransportExecution AssignmentTransportSemantics

private theorem output_column (lane : Fin 4) :
    Layout.Stage1.Spartan.sourceToSpartan (PilotProduction.outputDigestStart + lane.val) =
      28785011 + lane.val := by
  have bounded : lane.val < 4 := lane.isLt
  change Layout.Stage1.Spartan.sourceToSpartan (99056 + lane.val) = _
  unfold Layout.Stage1.Spartan.sourceToSpartan
  rw [if_pos (by change 99056 + lane.val < 14722512; omega)]
  unfold PilotSpartan.sourceToSpartan
  rw [if_neg (by change ¬99056 + lane.val < 49393; omega),
    if_neg (by change ¬99056 + lane.val < 49663; omega),
    if_neg (by change ¬99056 + lane.val < 99056; omega),
    if_pos (by change 99056 + lane.val < 99060; omega),
    PilotSpartan.secondPublicStart_value, PilotSpartan.outputDigestStart_value,
    Nat.add_sub_cancel_left]
  unfold Layout.Stage1.Spartan.liftPilotColumn
  rw [if_neg (by change ¬14722509 + lane.val < 98786; omega),
    if_neg (by change ¬14722509 + lane.val < 14722238; omega),
    Layout.Stage1.Spartan.privateColumnCount_eq]
  change 28784740 + (14722509 + lane.val - 14722238) = _
  omega

private def outputSource (program : Program) (lane : Fin 4) : Nat :=
  28785011 + lane.val + PerApplicationPackage.addedPrivateColumnCount program

private theorem output_source_lt (program : Program) (lane : Fin 4) :
    outputSource program lane < PiRLCProductPlan.baseSourceWidth program := by
  unfold outputSource PiRLCProductPlan.baseSourceWidth
  rw [PerApplicationPackage.package_totalColumnCount,
    PerApplicationPackage.basePackage_totalColumnCount_eq]
  have bounded := lane.isLt
  omega

private theorem output_expression (program : Program) (lane : Fin 4) :
    PerApplicationAssignmentTransport.outputDigestExpression program lane =
      .var (outputSource program lane) := by
  have outside : PermutationOutput.Readout.decode PiCCSTranscriptReadout.phaseStart
      Layout.Stage1.PiCCSOrdinarySourceSupport.transcriptInvocationCount
      (28785011 + lane.val) = none := by
    rw [PiCCSTranscriptReadout.phaseStart_eq,
      Layout.Stage1.PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq]
    unfold PermutationOutput.Readout.decode
    rw [dif_pos (by omega)]
    dsimp only
    rw [dif_neg (by omega)]
  simp only [PerApplicationAssignmentTransport.outputDigestExpression,
    PerApplicationAssignmentTransport.physicalExpr, PilotProduction.outputInterface,
    PilotProduction.makeOutputInterface, PilotProduction.outputDigest,
    CompactRows.renameExpr, output_column, PermutationOutput.Readout.rewriteExpr,
    PermutationOutput.Readout.variableExpr, outside]
  change Expr.var (PerApplicationPackage.shiftColumn program (28785011 + lane.val)) = _
  unfold PerApplicationPackage.shiftColumn
  rw [if_neg (by
    have layout : PerApplicationPackage.basePackage.layout.constantColumn = 28784740 :=
      Package.circuitPackage_layout_values.2.2.1
    rw [layout]
    omega)]
  rfl

private theorem mapped_output_value (program : Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (lane : Fin 4) (moved : Expr)
    (emitted : finalMap.expression
      (PerApplicationAssignmentTransport.outputDigestExpression program lane) = .ok moved) :
    moved.eval (physicalValues program env application) =
      (PerApplicationAssignmentTransport.outputDigestExpression program lane).eval
        (SourceCompiler.sourceEnv (SourceAssignment.raw program env application).retainedSource) := by
  rw [output_expression] at emitted ⊢
  cases mapped : finalColumn (outputSource program lane) with
  | error message =>
    simp [finalMap, PhysicalRelabel.Map.expression, mapped, Bind.bind, Except.bind] at emitted
  | ok target =>
    have same : Expr.var target = moved := by
      simpa only [finalMap, PhysicalRelabel.Map.expression, mapped, Bind.bind,
        Except.bind, Pure.pure, Except.pure, Except.ok.injEq] using emitted
    subst moved
    let source : Fin (PiRLCProductPlan.baseSourceWidth program) :=
      ⟨outputSource program lane, output_source_lt program lane⟩
    let raw := SourceAssignment.raw program env application
    have baseValue := AssignmentTransportCommonSource.base_value program env application source target mapped
    have retained := SourceCompiler.sourceEnv_at raw.retainedSource
      (PiRLCRetainedPreservation.baseSourceColumn program source)
    have sameBase := PiRLCRetainedPreservation.sourceAssignment_base program
      raw.base raw.groupValue raw.products source
    exact baseValue.trans (retained.trans sameBase).symm

private theorem list_map_eq {α β γ : Type} (relation : α → β → Prop)
    (left : α → γ) (right : β → γ) (before : List α) (after : List β)
    (pairs : List.Forall₂ relation before after)
    (agrees : ∀ original ∈ before, ∀ moved, relation original moved → right moved = left original) :
    after.map right = before.map left := by
  revert agrees
  induction pairs with
  | nil => intro _; rfl
  | @cons original moved before after related pairs inductionHypothesis =>
    intro agrees
    simp only [List.map_cons]
    rw [agrees original (List.mem_cons_self ..) moved related]
    exact congrArg (List.cons (left original))
      (inductionHypothesis (fun expression member other related =>
        agrees expression (List.mem_cons_of_mem _ member) other related))

private theorem mapped_values (program : Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (before after : List Expr)
    (pairs : List.Forall₂ (fun original moved => finalMap.expression original = .ok moved) before after)
    (members : ∀ expression ∈ before,
      expression ∈ PerApplicationAssignmentTransport.outputDigestExpressions program) :
    after.map (fun expression => expression.eval (physicalValues program env application)) =
      before.map (fun expression => expression.eval
        (SourceCompiler.sourceEnv (SourceAssignment.raw program env application).retainedSource)) := by
  apply list_map_eq _ _ _ before after pairs
  intro original member moved emitted
  obtain ⟨lane, rfl⟩ := List.mem_ofFn.mp (members original member)
  exact mapped_output_value program env application lane moved emitted

/-- A successful emitted plan computes the reference public output digest.
The equality requires no row, witness, or source-read premise. -/
theorem value (program : Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (plan : AssignmentTransport.Plan)
    (emitted : AssignmentTransport.plan program (CommonSchedule.physicalWidth program) = .ok plan) :
    AssignmentTransportExecution.outputDigest plan (physicalValues program env application) =
      (SourceAssignment.raw program env application).outputDigest := by
  obtain ⟨_, _, _, _, _, _, _, _, expressions⟩ :=
    emitted_parts program (CommonSchedule.physicalWidth program) plan emitted
  rw [AssignmentTransportExecution.outputDigest,
    mapped_values program env application _ _
      (PhysicalRelabel.mapM_pairs _ _ _ expressions) (fun _ member => member)]
  let raw := SourceAssignment.raw program env application
  calc
    _ = List.ofFn (fun lane : Fin PilotProduction.digestWords => raw.outputDigest.getD lane.val 0) := by
      rw [PerApplicationAssignmentTransport.outputDigestExpressions, List.map_ofFn]
      apply congrArg List.ofFn
      funext lane
      simpa only [PerApplicationAssignmentTransportExpressions.outputDigestExpressions_getD] using
        PerApplicationAssignmentTransportExpressions.outputDigestExpression_eval program raw lane
    _ = raw.outputDigest := by
      unfold PerApplicationCanonicalAssignment.RawValues.outputDigest
      apply congrArg List.ofFn
      funext lane
      exact PriorStateHash.ofFn_getD _ lane 0

end NightstreamFPrime.Export.Stage1.Wide.OutputDigest
