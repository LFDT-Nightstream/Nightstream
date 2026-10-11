import NightstreamFPrime.Export.Stage1.PiCCSPayloadWiring
import NightstreamFPrime.Export.Stage1.PiCCSPoseidonPreservation
import NightstreamFPrime.Export.Stage1.PiCCSTranscriptDirectSemantics

/-!
Owns the arbitrary-assignment PiCCS transcript link. Accepted Poseidon rows
and parent-owned payload forms imply the indexed transcript in the same
decoded environment as the arithmetic leaves. No raw encoding is assumed.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSDecodedTranscript

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open PiCCSPoseidonPreservation

variable {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}

private abbrev decoded
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) : Env :=
  Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv geometry assignment)

private theorem payload_eval
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (current : Fin PiCCSPoseidonPlan.invocationCount)
    (lane : Fin Spec.Poseidon2.width) :
    (PiCCSPoseidonPlan.payloadForm (PiCCSPayloadWiring.form geometry)
      current lane).eval assignment =
      (Hash.evalList (decoded geometry assignment)
        (PiCCSActionPayloadBlock.selectedBlock current)).getD lane.val 0 := by
  unfold PiCCSPoseidonPlan.payloadForm
  by_cases rateLane : lane.val < Spec.Poseidon2.rate
  · rw [dif_pos rateLane, PiCCSPayloadWiring.form_eval geometry _ assignment one,
      PiCCSActionPayloadBlock.payloadExpression_encode]
    unfold PiCCSActionPayloadBlock.payloadExpr
    have zeroEval : (0 : Expr).eval (decoded geometry assignment) = (0 : F) := by
      apply Fin.ext
      norm_num [Expr.eval, goldilocksModulus]
    unfold Hash.evalList
    rw [← zeroEval]
    exact (List.getD_map (n := lane.val)
      (PiCCSActionPayloadBlock.selectedBlock current) (0 : Expr)
      (Expr.eval (decoded geometry assignment))).symm
  · rw [dif_neg rateLane, SparseForm.empty_eval]
    have lengthBound : (PiCCSActionPayloadBlock.selectedBlock current).length ≤
        Spec.Poseidon2.rate := by
      have wellFormed := PiCCSActionPayloadBlock.kindAt_wellFormed current
      cases found : PiCCSActionPayloadBlock.kindAt current with
      | absorb block =>
          simpa [PiCCSActionPayloadBlock.selectedBlock,
            PiCCSActionPayloadBlock.selectedBlockForKind, found] using! wellFormed
    apply Eq.symm
    apply List.getD_eq_default
    simp only [Hash.evalList, List.length_map]
    omega

private theorem absorb_input
    (ordinary : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (poseidon : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn ordinary) = 1)
    (current : Fin PiCCSPoseidonPlan.invocationCount) (block : List Expr)
    (found : PiCCSActionPayloadBlock.kindAt current = .absorb block) :
    SparseLayer.evalState assignment
        (PiCCSPoseidonPlan.inputState (PiCCSPayloadWiring.form ordinary)
          poseidon current) =
      Hash.absorbF (previousValue poseidon assignment current)
        (Hash.evalList (decoded ordinary assignment) block) := by
  funext lane
  simp only [PiCCSPoseidonPlan.inputState, found, SparseLayer.evalState,
    SparseForm.add_eval]
  rw [show (PiCCSPoseidonPlan.previousOutput poseidon current lane).eval
      assignment = previousValue poseidon assignment current lane from
    congrFun (previousOutput_eval poseidon assignment current) lane]
  rw [payload_eval ordinary assignment one current lane]
  simp only [PiCCSActionPayloadBlock.selectedBlock, found,
    PiCCSActionPayloadBlock.selectedBlockForKind, Hash.absorbF]

/-- The actual parent forms and arbitrary accepted Poseidon rows force the
complete indexed transcript. The only value premise is the enforced constant
coordinate; no retained encoding or semantic representation is supplied. -/
theorem rowsZero_implies_indexedSemantics
    (ordinary : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (poseidon : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiCCSOrdinaryRetainedGeometry.oneColumn ordinary) = 1)
    (rows : (PiCCSPoseidonPlan.plan (PiCCSPayloadWiring.form ordinary)
      poseidon).RowsZero assignment) :
    PoseidonActionSemantics.IndexedSemantics
      (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv ordinary assignment))
      Spec.Poseidon2.zeroState PiCCSActionPayloadBlock.kindAt
      (valueState poseidon assignment) := by
  have sameOne : PiCCSPoseidonPlan.oneColumn poseidon =
      PiCCSOrdinaryRetainedGeometry.oneColumn ordinary := by
    apply Fin.ext
    rfl
  have semantics := PiCCSPoseidonPlan.rowsZero_implies_semantics
    (PiCCSPayloadWiring.form ordinary) poseidon assignment
    (by rw [sameOne]; exact one) rows
  have step : ∀ current,
      valueState poseidon assignment current =
        PoseidonActionSemantics.runKind (decoded ordinary assignment)
          (PoseidonActionSemantics.previousState Spec.Poseidon2.zeroState
            (valueState poseidon assignment) current)
          (PiCCSActionPayloadBlock.kindAt current) := by
    intro current
    rw [previousState_eq_previousValue poseidon assignment current]
    have invocation : List.ofFn (outputValue poseidon assignment current) =
        Spec.Poseidon2.permute (List.ofFn (SparseLayer.evalState assignment
          (PiCCSPoseidonPlan.inputState (PiCCSPayloadWiring.form ordinary)
            poseidon current))) := semantics.invocation current
    cases found : PiCCSActionPayloadBlock.kindAt current with
    | absorb block =>
        rw [absorb_input ordinary poseidon assignment one current block found]
          at invocation
        simpa only [valueState, PoseidonActionSemantics.runKind,
          Spec.Poseidon2.absorbBlock, Hash.absorbF_input_eq_reference]
          using invocation
  exact ⟨step⟩

end NightstreamFPrime.Export.Stage1.PiCCSDecodedTranscript
