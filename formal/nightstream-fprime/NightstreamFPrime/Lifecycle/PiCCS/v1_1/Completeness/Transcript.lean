import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness.Core

/-!
Proof-only PiCCS v1_1 assembler group. It composes opaque leaf contracts and
adds no protocol predicate, circuit row, or alternate path.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.CompletenessSupport

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.SumCheck
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def transcriptPrefixOps
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List Op :=
  let shared := atOffset interface offset
  [childOp "piccs.v1_1.statement_binding"
      (statementBindingCircuit shared) offset,
    childOp "piccs.v1_1.statement_absorption"
      (statementAbsorptionCircuit shared)
        (statementAbsorptionOffset interface offset),
    childOp "piccs.v1_1.challenge_derivation" (challengeCircuit interface offset)
      (challengeOffset interface offset),
    childOp "piccs.v1_1.round_transcript" (roundTranscriptCircuit shared)
      (roundTranscriptOffset interface offset)]

private theorem appendStatementBinding
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (before : Sequence.Prefix env offset)
    (startEq : offset + localLength before.operations = offset) :
    ∃ after : Sequence.Prefix env offset,
      after.operations = before.operations ++
        [childOp "piccs.v1_1.statement_binding"
          (statementBindingCircuit (atOffset interface offset)) offset] ∧
      offset + localLength after.operations =
        statementAbsorptionOffset interface offset ∧
      Sequence.PreservesPrefix before after := by
  let shared := atOffset interface offset
  have childAssumptions :=
    (assumptionsAt assumptions before.current).statementBinding
  have childScope := StatementBinding.flatConstraints_varsBelow
    (statementBindingInterface shared) offset
  have childSpec :
      (statementBindingCircuit shared).spec offset before.current := by
    exact ⟨fun _ => rfl, fun _ => rfl, fun _ => rfl⟩
  rcases appendAt before "piccs.v1_1.statement_binding"
      (statementBindingCircuit shared) offset startEq childScope childAssumptions
      childSpec with
    ⟨after, operationsEq, nextEq, preserves, _childHolds⟩
  refine ⟨after, ?_, ?_, preserves⟩
  · simpa [shared] using operationsEq
  · simpa [shared, statementAbsorptionOffset] using nextEq

private theorem appendStatementAbsorption
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (before : Sequence.Prefix env offset)
    (startEq : offset + localLength before.operations =
      statementAbsorptionOffset interface offset) :
    ∃ after : Sequence.Prefix env offset,
      after.operations = before.operations ++
        [childOp "piccs.v1_1.statement_absorption"
          (statementAbsorptionCircuit (atOffset interface offset))
            (statementAbsorptionOffset interface offset)] ∧
      offset + localLength after.operations =
        challengeOffset interface offset ∧
      Sequence.PreservesPrefix before after ∧
      StatementAbsorption.SpecHolds
        (statementAbsorptionInterface (atOffset interface offset))
          (statementAbsorptionOffset interface offset) after.current := by
  let shared := atOffset interface offset
  have childAssumptions :=
    (assumptionsAt assumptions before.current).statementAbsorption
  have childScope := StatementAbsorption.flatConstraints_varsBelow
    (statementAbsorptionInterface shared)
      (statementAbsorptionOffset interface offset) before.current
        childAssumptions
  rcases StatementAbsorption.build (statementAbsorptionInterface shared)
      before.current (statementAbsorptionOffset interface offset)
      childAssumptions with ⟨built, childAgrees, childRows⟩
  rcases appendBuiltAt before "piccs.v1_1.statement_absorption"
      (statementAbsorptionCircuit shared)
      (statementAbsorptionOffset interface offset) startEq childScope
      built childAgrees childRows with
    ⟨after, operationsEq, nextEq, preserves, childHolds⟩
  have afterSpec := (statementAbsorptionCircuit shared).soundness
    after.current (statementAbsorptionOffset interface offset)
      (assumptionsAt assumptions after.current).statementAbsorption
        (holdsFlat_implies_holds after.current _ childHolds)
  refine ⟨after, ?_, ?_, preserves, ?_⟩
  · simpa [shared] using operationsEq
  · simpa [shared, challengeOffset] using nextEq
  · simpa [shared] using afterSpec

private theorem appendChallengeDerivation
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (before : Sequence.Prefix env offset)
    (startEq : offset + localLength before.operations =
      challengeOffset interface offset) :
    ∃ after : Sequence.Prefix env offset,
      after.operations = before.operations ++
        [childOp "piccs.v1_1.challenge_derivation"
          (challengeCircuit interface offset)
            (challengeOffset interface offset)] ∧
      offset + localLength after.operations =
        roundTranscriptOffset interface offset ∧
      Sequence.PreservesPrefix before after ∧
      ChallengeDerivation.SpecHolds
        (challengeInterface (atOffset interface offset) offset)
          (challengeOffset interface offset) after.current := by
  have childAssumptions :=
    (assumptionsAt assumptions before.current).challenge
  have childScope := ChallengeDerivation.flatConstraints_varsBelow
    (challengeInterface (atOffset interface offset) offset)
      (challengeOffset interface offset) before.current childAssumptions
  rcases ChallengeDerivation.build
      (challengeInterface (atOffset interface offset) offset)
      before.current (challengeOffset interface offset) childAssumptions with
    ⟨built, childAgrees, childRows⟩
  rcases appendBuiltAt before "piccs.v1_1.challenge_derivation"
      (challengeCircuit interface offset) (challengeOffset interface offset)
      startEq childScope built childAgrees childRows with
    ⟨after, operationsEq, nextEq, preserves, childHolds⟩
  have childSpec := (challengeCircuit interface offset).soundness
    after.current (challengeOffset interface offset)
      (assumptionsAt assumptions after.current).challenge
        (holdsFlat_implies_holds after.current _ childHolds)
  exact ⟨after, operationsEq, nextEq, preserves, childSpec⟩

private theorem appendRoundTranscript
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (before : Sequence.Prefix env offset)
    (challengeSpec : ChallengeDerivation.SpecHolds
      (challengeInterface (atOffset interface offset) offset)
        (challengeOffset interface offset) before.current)
    (startEq : offset + localLength before.operations =
      roundTranscriptOffset interface offset) :
    ∃ after : Sequence.Prefix env offset,
      after.operations = before.operations ++
        [childOp "piccs.v1_1.round_transcript"
          (roundTranscriptCircuit (atOffset interface offset))
            (roundTranscriptOffset interface offset)] ∧
      offset + localLength after.operations =
        initialClaimOffset interface offset ∧
      Sequence.PreservesPrefix before after ∧
      ChallengeDerivation.SpecHolds
        (challengeInterface (atOffset interface offset) offset)
          (challengeOffset interface offset) after.current ∧
      RoundTranscript.SpecHolds
        (roundTranscriptInterface (atOffset interface offset))
          (roundTranscriptOffset interface offset) after.current := by
  let shared := atOffset interface offset
  have childAssumptions :=
    (assumptionsAt assumptions before.current).roundTranscript
  have childScope := RoundTranscript.flatConstraints_varsBelow
    (roundTranscriptInterface shared) (roundTranscriptOffset interface offset)
      before.current childAssumptions
  rcases RoundTranscript.build (roundTranscriptInterface shared)
      before.current (roundTranscriptOffset interface offset)
      childAssumptions with ⟨built, childAgrees, childRows⟩
  rcases appendBuiltAt before "piccs.v1_1.round_transcript"
      (roundTranscriptCircuit shared)
      (roundTranscriptOffset interface offset) startEq childScope
      built childAgrees childRows with
    ⟨after, operationsEq, nextEq, preserves, childHolds⟩
  have roundSpecAfter := (roundTranscriptCircuit shared).soundness
    after.current (roundTranscriptOffset interface offset)
      (assumptionsAt assumptions after.current).roundTranscript
        (holdsFlat_implies_holds after.current _ childHolds)
  have challengeAgreement : ∀ index,
      index < challengeOffset interface offset +
        (ChallengeDerivation.program
          (challengeInterface (atOffset interface offset) offset)
            (challengeOffset interface offset)).recipes.length →
      after.current index = before.current index := by
    intro index below
    apply preserves.values index
    rw [startEq]
    simpa only [roundTranscriptOffset_eq,
      ChallengeDerivation.program_recipes_length] using below
  have challengeSpecAfter := ChallengeDerivation.specHolds_of_agree_below
    (challengeInterface (atOffset interface offset) offset)
      (challengeOffset interface offset) before.current after.current
      (assumptionsAt assumptions before.current).challenge
        challengeAgreement challengeSpec
  refine ⟨after, ?_, ?_, preserves, challengeSpecAfter, roundSpecAfter⟩
  · simpa [shared] using operationsEq
  · simpa [shared, initialClaimOffset] using nextEq

theorem completeTranscriptPrefix
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env) :
    ∃ completed : Sequence.Prefix env offset,
      completed.operations = transcriptPrefixOps interface offset ∧
      offset + localLength completed.operations =
        initialClaimOffset interface offset ∧
      StatementAbsorption.SpecHolds
        (statementAbsorptionInterface (atOffset interface offset))
          (statementAbsorptionOffset interface offset) completed.current ∧
      ChallengeDerivation.SpecHolds
        (challengeInterface (atOffset interface offset) offset)
          (challengeOffset interface offset) completed.current ∧
      RoundTranscript.SpecHolds
        (roundTranscriptInterface (atOffset interface offset))
          (roundTranscriptOffset interface offset) completed.current := by
  let p0 := Sequence.empty env offset
  have s0 : offset + localLength p0.operations = offset := by rfl
  rcases appendStatementBinding relation interface env offset assumptions
      p0 s0 with
    ⟨p1, o1, n1, _p0to1⟩
  rcases appendStatementAbsorption relation interface env offset assumptions
      p1 n1 with
    ⟨p2, o2, n2, _p1to2, statementSpecP2⟩
  rcases appendChallengeDerivation relation interface env offset assumptions
      p2 n2 with
    ⟨p3, o3, n3, p2to3, challengeSpecP3⟩
  rcases appendRoundTranscript relation interface env offset assumptions
      p3 challengeSpecP3 n3 with
    ⟨p4, o4, n4, p3to4, challengeSpecP4, roundSpecP4⟩
  have p2to4 := p2to3.trans p3to4
  have statementAgreement : ∀ index,
      index < statementAbsorptionOffset interface offset +
        (StatementAbsorption.program
          (statementAbsorptionInterface (atOffset interface offset))
            (statementAbsorptionOffset interface offset)).recipes.length →
      p4.current index = p2.current index := by
    intro index below
    apply p2to4.values index
    rw [n2]
    simpa using below
  have statementSpecP4 := StatementAbsorption.specHolds_of_agree_below
    (statementAbsorptionInterface (atOffset interface offset))
      (statementAbsorptionOffset interface offset) p2.current p4.current
      (assumptionsAt assumptions p2.current).statementAbsorption
        statementAgreement statementSpecP2
  refine ⟨p4, ?_, n4, statementSpecP4, challengeSpecP4, roundSpecP4⟩
  rw [o4, o3, o2, o1]
  simp [p0, Sequence.empty, transcriptPrefixOps]

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.CompletenessSupport
