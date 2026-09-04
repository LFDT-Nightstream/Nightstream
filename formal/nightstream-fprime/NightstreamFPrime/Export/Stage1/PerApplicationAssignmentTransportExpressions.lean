import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransport

/-!
Owns expression-level value preservation for the sealed assignment transport.
It proves that final physical-column renaming preserves source evaluation, then
specializes that result to the PiCCS payload and Pilot output-digest recipes.

This module does not execute the transport or derive product-family values.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransportExpressions

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransport
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PerApplicationCanonicalAssignment

abbrev Program := Lifecycle.Stage1.Application.Program

/-- Renaming an original Stage 1 expression to its final physical package
columns preserves evaluation in the corresponding retained-source view. -/
theorem physicalExpr_eval_sourceEnv (program : Program)
    (source : Fin (PiCCSActionPayloadBlock.prefixSourceWidth program) → F)
    (expression : Expr) :
    (physicalExpr program expression).eval (SourceCompiler.sourceEnv source) =
      expression.eval (PiCCSActionPayloadBlock.packageEnv program source) := by
  rw [physicalExpr, CompactRows.renameExpr_eval]
  rfl

/-- Pointwise lookup of one mapped PiCCS payload expression does not expand
the complete payload list. -/
@[simp] theorem payloadExpressions_getD (program : Program)
    (index : Fin PiCCSActionPayloadBlock.payloadCount) :
    (payloadExpressions program).getD index.val 0 =
      physicalExpr program (PiCCSActionPayloadBlock.payloadExpression index) := by
  unfold payloadExpressions
  exact NightstreamFPrime.Lifecycle.PriorStateHash.ofFn_getD _ index 0

/-- Every mapped PiCCS payload expression computes the exact value appended
to the retained source assignment. -/
theorem payloadExpression_eval (program : Program)
    (raw : RawValues program)
    (index : Fin PiCCSActionPayloadBlock.payloadCount) :
    ((payloadExpressions program).getD index.val 0).eval
        (SourceCompiler.sourceEnv raw.retainedSource) =
      PiCCSActionPayloadBlock.payloadValue program raw.retainedSource index := by
  rw [payloadExpressions_getD]
  exact physicalExpr_eval_sourceEnv program raw.retainedSource
    (PiCCSActionPayloadBlock.payloadExpression index)

private theorem outputDigest_source_lt
    (lane : Fin PilotProduction.digestWords) :
    PilotProduction.outputDigestStart + lane.val <
      NightstreamFPrime.Layout.Stage1.Spartan.SourceColumnCount := by
  have laneBound := lane.isLt
  rw [NightstreamFPrime.Layout.Stage1.Spartan.sourceColumnCount_eq]
  norm_num [PilotProduction.outputDigestStart,
    PilotProduction.outputPreimageStart, PilotProduction.priorPublicInputStart,
    PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
    PilotProduction.digestWords, PilotValues.digestWords,
    PriorStateHash.publicWidth_eq] at laneBound ⊢
  omega

private theorem outputDigest_pilot_lt
    (lane : Fin PilotProduction.digestWords) :
    PilotProduction.outputDigestStart + lane.val <
      NightstreamFPrime.Layout.Stage1.Spartan.pilotSourceColumnCount := by
  have laneBound := lane.isLt
  norm_num [NightstreamFPrime.Layout.Stage1.Spartan.pilotSourceColumnCount,
    PilotProduction.outputDigestStart, PilotProduction.outputPreimageStart,
    PilotProduction.priorPublicInputStart, PilotProduction.priorPreimageStart,
    PilotProduction.stateHashWords_eq, PilotProduction.digestWords,
    PilotValues.digestWords, PriorStateHash.publicWidth_eq] at laneBound ⊢
  omega

/-- At each Pilot output-digest source, the retained-source package view and
the original Pilot pullback read the same package-bound base value. -/
private theorem packageEnv_outputDigest_eq_pilot
    (program : Program) (raw : RawValues program)
    (lane : Fin PilotProduction.digestWords) :
    PiCCSActionPayloadBlock.packageEnv program raw.retainedSource
        (PilotProduction.outputDigestStart + lane.val) =
      PilotSpartan.pullback
        (PilotOrdinaryDirectPlan.pilotEnv program raw.base)
        (PilotProduction.outputDigestStart + lane.val) := by
  let column := PilotProduction.outputDigestStart + lane.val
  have sourceBound : column <
      NightstreamFPrime.Layout.Stage1.Spartan.SourceColumnCount :=
    outputDigest_source_lt lane
  have pilotBound : column <
      NightstreamFPrime.Layout.Stage1.Spartan.pilotSourceColumnCount :=
    outputDigest_pilot_lt lane
  change PiCCSActionPayloadBlock.packageEnv program
      (PiRLCRetainedPreservation.sourceAssignment program raw.base
        raw.groupValue raw.products) column = _
  rw [PiCCSPoseidonPreservation.packageEnv_sourceAssignment program raw.base
    raw.groupValue raw.products column sourceBound]
  unfold PilotSpartan.pullback PilotOrdinaryDirectPlan.pilotEnv
  have sourceMap :
      NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan column =
        NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn
          (PilotSpartan.sourceToSpartan column) := by
    unfold NightstreamFPrime.Layout.Stage1.Spartan.sourceToSpartan
    rw [if_pos pilotBound]
  rw [← sourceMap]
  unfold RunningTransitionDirectPlan.transitionEnv
  rw [dif_pos (PiCCSPoseidonPreservation.sourceToSpartan_lt_basePackage
    column sourceBound)]
  apply congrArg raw.base
  apply Fin.ext
  rfl

/-- Pointwise lookup of one mapped output-digest expression does not expand
the four-expression list. -/
@[simp] theorem outputDigestExpressions_getD (program : Program)
    (lane : Fin PilotProduction.digestWords) :
    (outputDigestExpressions program).getD lane.val 0 =
      outputDigestExpression program lane := by
  unfold outputDigestExpressions
  exact NightstreamFPrime.Lifecycle.PriorStateHash.ofFn_getD _ lane 0

/-- Every mapped output-digest expression computes the matching digest word
derived from the constrained raw Pilot source. -/
theorem outputDigestExpression_eval (program : Program)
    (raw : RawValues program) (lane : Fin PilotProduction.digestWords) :
    ((outputDigestExpressions program).getD lane.val 0).eval
        (SourceCompiler.sourceEnv raw.retainedSource) =
      raw.outputDigest.getD lane.val 0 := by
  rw [outputDigestExpressions_getD]
  unfold outputDigestExpression
  calc
    _ = (PilotProduction.outputInterface.digest
          (Lifecycle.Pilot.outputOffset PilotProduction.interface
            PilotProduction.witnessOffset) lane).eval
          (PiCCSActionPayloadBlock.packageEnv program raw.retainedSource) :=
      physicalExpr_eval_sourceEnv program raw.retainedSource _
    _ = raw.outputDigest.getD lane.val 0 := by
      unfold RawValues.outputDigest
      rw [NightstreamFPrime.Lifecycle.PriorStateHash.ofFn_getD]
      simpa [PilotProduction.outputInterface,
        PilotProduction.makeOutputInterface, PilotProduction.outputDigest,
        Expr.eval] using packageEnv_outputDigest_eq_pilot program raw lane

end NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransportExpressions
