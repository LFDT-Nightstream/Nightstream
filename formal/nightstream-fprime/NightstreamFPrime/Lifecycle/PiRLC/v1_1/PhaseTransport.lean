import NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics

/-!
Owns semantic transport for PiRLC across environments with explicitly equal
evaluated inputs and child-owned outputs. It adds no verifier predicate,
challenge, row, or layout assumption.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

namespace Sampler.RelationHolds

/-- Transport one sampler relation through exact equality of its evaluated
initial state and its two child-owned output families. -/
theorem of_eval_eq (interface : Sampler.Interface) (coordinate offset : Nat)
    (left right : Env)
    (initialEq : Sampler.evalInitialState interface offset left =
      Sampler.evalInitialState interface offset right)
    (coefficientsEq : Sampler.outputCoefficients left offset =
      Sampler.outputCoefficients right offset)
    (outputStateEq : Sampler.evalState left
        (Sampler.outputState interface coordinate offset) =
      Sampler.evalState right
        (Sampler.outputState interface coordinate offset))
    (relation : Sampler.RelationHolds interface coordinate offset left) :
    Sampler.RelationHolds interface coordinate offset right := by
  rcases relation with ⟨coefficients, sampled, output, state⟩
  refine ⟨coefficients, ?_, ?_, ?_⟩
  · simpa [Sampler.productionCandidates, Sampler.productionSource, initialEq]
      using sampled
  · rw [← coefficientsEq]
    exact output
  · calc
      Sampler.evalState right
          (Sampler.outputState interface coordinate offset) =
          Sampler.evalState left
            (Sampler.outputState interface coordinate offset) :=
        outputStateEq.symm
      _ = (Sampler.productionSource interface coordinate offset left).nextState :=
        state
      _ = (Sampler.productionSource interface coordinate offset right).nextState := by
        simp [Sampler.productionSource, initialEq]

/-- Cross-offset form of `of_eval_eq`. All layout-specific relocation stays
outside this semantic theorem and enters only as exact evaluated equalities. -/
theorem of_cross_eval_eq
    (leftInterface rightInterface : Sampler.Interface)
    (coordinate leftOffset rightOffset : Nat) (left right : Env)
    (initialEq : Sampler.evalInitialState leftInterface leftOffset left =
      Sampler.evalInitialState rightInterface rightOffset right)
    (coefficientsEq : Sampler.outputCoefficients left leftOffset =
      Sampler.outputCoefficients right rightOffset)
    (outputStateEq : Sampler.evalState left
        (Sampler.outputState leftInterface coordinate leftOffset) =
      Sampler.evalState right
        (Sampler.outputState rightInterface coordinate rightOffset))
    (relation : Sampler.RelationHolds leftInterface coordinate leftOffset left) :
    Sampler.RelationHolds rightInterface coordinate rightOffset right := by
  rcases relation with ⟨coefficients, sampled, output, state⟩
  refine ⟨coefficients, ?_, ?_, ?_⟩
  · simpa [Sampler.productionCandidates, Sampler.productionSource, initialEq]
      using sampled
  · rw [← coefficientsEq]
    exact output
  · calc
      Sampler.evalState right
          (Sampler.outputState rightInterface coordinate rightOffset) =
          Sampler.evalState left
            (Sampler.outputState leftInterface coordinate leftOffset) :=
        outputStateEq.symm
      _ = (Sampler.productionSource leftInterface coordinate leftOffset
          left).nextState := state
      _ = (Sampler.productionSource rightInterface coordinate rightOffset
          right).nextState := by
        simp [Sampler.productionSource, initialEq]

end Sampler.RelationHolds

namespace SamplerChain.RelationHolds

/-- Transport the complete 17-source sampler chain from exact evaluated
state, selector-output, and challenge equalities. -/
theorem of_eval_eq (interface : SamplerChain.Interface) (offset : Nat)
    (left right : Env)
    (stateAtEq : ∀ count, count ≤ SamplerChain.sourceCount →
      SamplerChain.evalStateAt interface offset left count =
        SamplerChain.evalStateAt interface offset right count)
    (coefficientsEq : ∀ source : Fin SamplerChain.sourceCount,
      Sampler.outputCoefficients left
          (SamplerChain.sourceOffset offset source.val) =
        Sampler.outputCoefficients right
          (SamplerChain.sourceOffset offset source.val))
    (outputStateEq : ∀ source : Fin SamplerChain.sourceCount,
      Sampler.evalState left
          (Sampler.outputState
            (SamplerChain.childInterface interface offset source.val)
            source.val (SamplerChain.sourceOffset offset source.val)) =
        Sampler.evalState right
          (Sampler.outputState
            (SamplerChain.childInterface interface offset source.val)
            source.val (SamplerChain.sourceOffset offset source.val)))
    (challengesEq : SamplerChain.evalChallenges interface offset left =
      SamplerChain.evalChallenges interface offset right)
    (relation : SamplerChain.RelationHolds interface offset left) :
    SamplerChain.RelationHolds interface offset right := by
  have initialEq : SamplerChain.evalInitialState interface offset left =
      SamplerChain.evalInitialState interface offset right :=
    stateAtEq 0 (by omega)
  refine {
    child := ?_
    response := ?_
    finalState := ?_ }
  · intro source
    apply Sampler.RelationHolds.of_eval_eq
    · simpa [Sampler.evalInitialState, SamplerChain.childInterface,
        SamplerChain.evalStateAt] using
          stateAtEq source.val (Nat.le_of_lt source.isLt)
    · exact coefficientsEq source
    · exact outputStateEq source
    · exact relation.child source
  · calc
      NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.piRlcChallenges
          (SamplerChain.evalInitialState interface offset right)
          SamplerChain.sourceCount =
        NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.piRlcChallenges
          (SamplerChain.evalInitialState interface offset left)
          SamplerChain.sourceCount := by
        exact congrArg (fun state =>
          NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.piRlcChallenges
            state SamplerChain.sourceCount) initialEq.symm
      _ = some (SamplerChain.evalChallenges interface offset left) :=
        relation.response
      _ = some (SamplerChain.evalChallenges interface offset right) := by
        rw [challengesEq]
  · calc
      SamplerChain.evalFinalState interface offset right =
          SamplerChain.evalFinalState interface offset left := by
        unfold SamplerChain.evalFinalState
        rw [stateAtEq SamplerChain.sourceCount (Nat.le_refl _)]
      _ = NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.stateAt
          NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.specification
          (SamplerChain.evalInitialState interface offset left)
          SamplerChain.sourceCount := relation.finalState
      _ = NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.stateAt
          NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.specification
          (SamplerChain.evalInitialState interface offset right)
          SamplerChain.sourceCount := by
        exact congrArg (fun state =>
          NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.stateAt
            NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.specification
            state SamplerChain.sourceCount) initialEq

/-- Cross-offset transport of the complete 17-source sampler chain. -/
theorem of_cross_eval_eq
    (leftInterface rightInterface : SamplerChain.Interface)
    (leftOffset rightOffset : Nat) (left right : Env)
    (stateAtEq : ∀ count, count ≤ SamplerChain.sourceCount →
      SamplerChain.evalStateAt leftInterface leftOffset left count =
        SamplerChain.evalStateAt rightInterface rightOffset right count)
    (coefficientsEq : ∀ source : Fin SamplerChain.sourceCount,
      Sampler.outputCoefficients left
          (SamplerChain.sourceOffset leftOffset source.val) =
        Sampler.outputCoefficients right
          (SamplerChain.sourceOffset rightOffset source.val))
    (outputStateEq : ∀ source : Fin SamplerChain.sourceCount,
      Sampler.evalState left
          (Sampler.outputState
            (SamplerChain.childInterface leftInterface leftOffset source.val)
            source.val (SamplerChain.sourceOffset leftOffset source.val)) =
        Sampler.evalState right
          (Sampler.outputState
            (SamplerChain.childInterface rightInterface rightOffset source.val)
            source.val (SamplerChain.sourceOffset rightOffset source.val)))
    (challengesEq : SamplerChain.evalChallenges leftInterface leftOffset left =
      SamplerChain.evalChallenges rightInterface rightOffset right)
    (relation : SamplerChain.RelationHolds leftInterface leftOffset left) :
    SamplerChain.RelationHolds rightInterface rightOffset right := by
  have initialEq : SamplerChain.evalInitialState leftInterface leftOffset left =
      SamplerChain.evalInitialState rightInterface rightOffset right :=
    stateAtEq 0 (by omega)
  refine {
    child := ?_
    response := ?_
    finalState := ?_ }
  · intro source
    apply Sampler.RelationHolds.of_cross_eval_eq
      (SamplerChain.childInterface leftInterface leftOffset source.val)
      (SamplerChain.childInterface rightInterface rightOffset source.val)
      source.val (SamplerChain.sourceOffset leftOffset source.val)
      (SamplerChain.sourceOffset rightOffset source.val) left right
    · simpa [Sampler.evalInitialState, SamplerChain.childInterface,
        SamplerChain.evalStateAt] using
          stateAtEq source.val (Nat.le_of_lt source.isLt)
    · exact coefficientsEq source
    · exact outputStateEq source
    · exact relation.child source
  · calc
      NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.piRlcChallenges
          (SamplerChain.evalInitialState rightInterface rightOffset right)
          SamplerChain.sourceCount =
        NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.piRlcChallenges
          (SamplerChain.evalInitialState leftInterface leftOffset left)
          SamplerChain.sourceCount := by
        exact congrArg (fun state =>
          NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.piRlcChallenges
            state SamplerChain.sourceCount) initialEq.symm
      _ = some (SamplerChain.evalChallenges leftInterface leftOffset left) :=
        relation.response
      _ = some (SamplerChain.evalChallenges rightInterface rightOffset right) := by
        rw [challengesEq]
  · calc
      SamplerChain.evalFinalState rightInterface rightOffset right =
          SamplerChain.evalFinalState leftInterface leftOffset left := by
        unfold SamplerChain.evalFinalState
        rw [stateAtEq SamplerChain.sourceCount (Nat.le_refl _)]
      _ = NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.stateAt
          NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.specification
          (SamplerChain.evalInitialState leftInterface leftOffset left)
          SamplerChain.sourceCount := relation.finalState
      _ = NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.stateAt
          NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.specification
          (SamplerChain.evalInitialState rightInterface rightOffset right)
          SamplerChain.sourceCount := by
        exact congrArg (fun state =>
          NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.stateAt
            NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.specification
            state SamplerChain.sourceCount) initialEq

end SamplerChain.RelationHolds

namespace Semantics.PhaseHolds

/-- Transport a complete PiRLC phase after independently establishing the
sampler relation and equality of the canonical public attempt. -/
theorem of_attempt_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (left right : Env)
    (sampler : SamplerChain.RelationHolds
      (Formal.samplerInterface (Formal.atOffset interface offset))
      (Formal.samplerOffset offset) right)
    (attemptEq : Semantics.attempt relation interface offset left =
      Semantics.attempt relation interface offset right)
    (phase : Semantics.PhaseHolds relation ajtai interface offset left) :
    Semantics.PhaseHolds relation ajtai interface offset right := by
  refine ⟨sampler, ?_⟩
  rw [← attemptEq]
  exact phase.accepted

/-- Cross-interface form used by the compact Stage 1 assembler. -/
theorem of_cross_attempt_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (leftInterface rightInterface : Formal.Interface logicalWidth publicFits)
    (leftOffset rightOffset : Nat) (left right : Env)
    (sampler : SamplerChain.RelationHolds
      (Formal.samplerInterface (Formal.atOffset rightInterface rightOffset))
      (Formal.samplerOffset rightOffset) right)
    (attemptEq : Semantics.attempt relation leftInterface leftOffset left =
      Semantics.attempt relation rightInterface rightOffset right)
    (phase : Semantics.PhaseHolds relation ajtai leftInterface leftOffset left) :
    Semantics.PhaseHolds relation ajtai rightInterface rightOffset right := by
  refine ⟨sampler, ?_⟩
  rw [← attemptEq]
  exact phase.accepted

end Semantics.PhaseHolds

end NightstreamFPrime.Lifecycle.PiRLC.v1_1
