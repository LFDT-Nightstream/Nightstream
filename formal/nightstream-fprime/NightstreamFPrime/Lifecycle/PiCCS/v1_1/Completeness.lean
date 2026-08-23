import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness.Transcript
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness.Evaluation
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness.Terminal

/-!
Completes the opaque-child PiCCS v1_1 assembler. This file owns only the
three-group composition and the sole parent `FormalCircuit`. It adds no
protocol predicate or alternate path.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.SumCheck
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open CompletenessSupport

/-- One canonical child operation exposes only its proved specification to
the logical parent. -/
theorem childOp_holds_of_spec (name : String) (child : FormalCircuit)
    (offset : Nat) (env : Env) (specification : child.spec offset env) :
    (childOp name child offset).holds env := by
  change child.assumptions offset env → child.spec offset env
  intro _
  exact specification

/-- The 12 child specifications are exactly the opaque logical meaning of
the ordered parent operation list. -/
theorem specHolds_implies_holds
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (env : Env) (offset : Nat)
    (specification : SpecHolds relation interface offset env) :
    holds env (opsAt relation interface offset) := by
  simp only [opsAt, holds_cons, holds_nil, and_true]
  exact ⟨
    childOp_holds_of_spec _ _ _ _ specification.statementBinding,
    childOp_holds_of_spec _ _ _ _ specification.statementAbsorption,
    childOp_holds_of_spec _ _ _ _ specification.challenge,
    childOp_holds_of_spec _ _ _ _ specification.roundTranscript,
    childOp_holds_of_spec _ _ _ _ specification.initialClaim,
    childOp_holds_of_spec _ _ _ _ specification.sumcheck,
    childOp_holds_of_spec _ _ _ _ specification.eval_K,
    childOp_holds_of_spec _ _ _ _ specification.eval_A,
    childOp_holds_of_spec _ _ _ _ specification.ccs,
    childOp_holds_of_spec _ _ _ _ specification.norm,
    childOp_holds_of_spec _ _ _ _ specification.finalIdentity,
    childOp_holds_of_spec _ _ _ _ specification.outputBinding⟩

/-- Exact private symbolic-variable count of the complete PiCCS assembler. -/
def privateCount (degreeBound : Nat) : Nat :=
  14499140 + 24 * RoundTranscript.perRoundRecipeCount degreeBound

/-- Exact flattened logical-row count of the complete PiCCS assembler. -/
def rowCount (degreeBound : Nat) : Nat :=
  14499190 + 24 * RoundTranscript.perRoundRecipeCount degreeBound

private theorem transcriptPrefix_localLength_eq
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    localLength (transcriptPrefixOps interface offset) =
      10342832 + 24 * RoundTranscript.perRoundRecipeCount degreeBound := by
  simp only [transcriptPrefixOps, localLength, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, Nat.add_zero, childOp_privateCount]
  unfold statementBindingCircuit statementAbsorptionCircuit
    challengeCircuit roundTranscriptCircuit
  simp only [FormalCircuit.withConstantFootprint_privateCount]
  omega

private theorem evaluationPrefix_localLength_eq
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    localLength (evaluationPrefixOps interface offset) = 52022 := by
  simp only [evaluationPrefixOps, localLength, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, Nat.add_zero, childOp_privateCount]
  unfold initialClaimCircuit sumcheckCircuit evalKCircuit
    evalACircuit
  simp only [FormalCircuit.withConstantFootprint_privateCount]

private theorem terminalPrefix_localLength_eq
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    localLength (terminalPrefixOps relation interface offset) = 4104286 := by
  simp only [terminalPrefixOps, localLength, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, Nat.add_zero, childOp_privateCount]
  unfold ccsCircuit normCircuit finalIdentityCircuit
    outputBindingCircuit
  simp only [FormalCircuit.withConstantFootprint_privateCount]

private theorem transcriptPrefix_rowCount_eq
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    NightstreamFPrime.Circuit.rowCount
      (transcriptPrefixOps interface offset) =
      10342832 + 24 * RoundTranscript.perRoundRecipeCount degreeBound := by
  simp only [transcriptPrefixOps, NightstreamFPrime.Circuit.rowCount,
    List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero,
    childOp_rowCount]
  unfold statementBindingCircuit statementAbsorptionCircuit challengeCircuit
    roundTranscriptCircuit
  simp only [FormalCircuit.withConstantFootprint_rowCount]
  omega

private theorem evaluationPrefix_rowCount_eq
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    NightstreamFPrime.Circuit.rowCount
      (evaluationPrefixOps interface offset) = 52070 := by
  simp only [evaluationPrefixOps, NightstreamFPrime.Circuit.rowCount,
    List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero,
    childOp_rowCount]
  unfold initialClaimCircuit sumcheckCircuit evalKCircuit evalACircuit
  simp only [FormalCircuit.withConstantFootprint_rowCount]

private theorem terminalPrefix_rowCount_eq
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    NightstreamFPrime.Circuit.rowCount
      (terminalPrefixOps relation interface offset) = 4104288 := by
  simp only [terminalPrefixOps, NightstreamFPrime.Circuit.rowCount,
    List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero,
    childOp_rowCount]
  unfold ccsCircuit normCircuit finalIdentityCircuit outputBindingCircuit
  simp only [FormalCircuit.withConstantFootprint_rowCount]

private theorem opsAt_eq_prefixes
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    opsAt relation interface offset =
      transcriptPrefixOps interface offset ++
        evaluationPrefixOps interface offset ++
          terminalPrefixOps relation interface offset := by
  rfl

theorem localLength_eq
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    localLength (Circuit.ops (main relation interface) offset) =
      privateCount degreeBound := by
  rw [main_ops]
  rw [opsAt_eq_prefixes]
  rw [Sequence.localLength_append, Sequence.localLength_append,
    transcriptPrefix_localLength_eq, evaluationPrefix_localLength_eq,
    terminalPrefix_localLength_eq]
  unfold privateCount
  omega

theorem flatConstraints_length_eq
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    (flatConstraints (Circuit.ops (main relation interface) offset)).length =
      rowCount degreeBound := by
  rw [main_ops, NightstreamFPrime.Circuit.flatConstraints_length_eq_rowCount]
  rw [opsAt_eq_prefixes, NightstreamFPrime.Circuit.rowCount_append,
    NightstreamFPrime.Circuit.rowCount_append,
    transcriptPrefix_rowCount_eq, evaluationPrefix_rowCount_eq,
    terminalPrefix_rowCount_eq]
  unfold rowCount
  omega

theorem privateCount_eq_of_degreeBound_eq_four (degreeBound : Nat)
    (degreeEq : degreeBound = 4) : privateCount degreeBound = 14584388 := by
  rw [degreeEq]
  norm_num [privateCount, RoundTranscript.perRoundRecipeCount]

theorem rowCount_eq_of_degreeBound_eq_four (degreeBound : Nat)
    (degreeEq : degreeBound = 4) : rowCount degreeBound = 14584438 := by
  rw [degreeEq]
  norm_num [rowCount, RoundTranscript.perRoundRecipeCount]

theorem completePrefix
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (specification : PhaseHolds relation ajtai interface offset env template) :
    ∃ completed : Sequence.Prefix env offset,
      completed.operations = opsAt relation interface offset := by
  rcases completeTranscriptPrefix relation interface env offset assumptions with
    ⟨p4, o4, s4, statementSpecP4, challengeSpecP4, roundSpecP4⟩
  rcases completeEvaluationPrefix relation ajtai interface env offset template
      assumptions specification.accepted p4 statementSpecP4 challengeSpecP4
        roundSpecP4 s4 with
    ⟨p8, o8, s8, _p4to8, evidenceP8⟩
  rcases completeTerminalPrefix relation ajtai interface env offset template
      assumptions p8 evidenceP8 s8 with
    ⟨p12, o12, _s12, _p8to12⟩
  have operationsEq : p12.operations = opsAt relation interface offset := by
    rw [o12, o8, o4]
    simp [transcriptPrefixOps, evaluationPrefixOps, terminalPrefixOps, opsAt]
  exact ⟨p12, operationsEq⟩

theorem completeness
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (specification : PhaseHolds relation ajtai interface offset env template) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main relation interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main relation interface) offset) := by
  rcases completePrefix relation ajtai interface template env offset assumptions
      specification with ⟨completed, operationsEq⟩
  refine ⟨completed.current, ?_, ?_⟩
  · change AgreesOutside env completed.current offset
      (localLength (opsAt relation interface offset))
    rw [← operationsEq]
    exact completed.agrees
  · change holdsFlat completed.current (opsAt relation interface offset)
    rw [← operationsEq]
    exact completed.rows

def circuit
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (template : Proof (ProductionKey.degreeBound relation)) : FormalCircuit where
  main := main relation interface
  assumptions := Assumptions relation interface
  spec := fun offset env =>
    PhaseHolds relation ajtai interface offset env template
  soundness := fun env offset assumptions rows =>
    spec_implies_phaseHolds relation ajtai interface offset env template
      (soundness relation interface env offset assumptions rows)
  completeness := completeness relation ajtai interface template

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal
