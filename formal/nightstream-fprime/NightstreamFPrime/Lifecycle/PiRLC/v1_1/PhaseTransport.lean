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

namespace SamplerChain.SpecHolds

/-- Exact initial state, checked words, and final state transport the same
sampler specification across environments and offsets. -/
theorem of_cross_eval_eq
    (leftInterface rightInterface : SamplerChain.Interface)
    (leftOffset rightOffset : Nat) (left right : Env)
    (initialEq : SamplerChain.evalInitialState leftInterface leftOffset left =
      SamplerChain.evalInitialState rightInterface rightOffset right)
    (wordsEq : ∀ source : Fin SamplerChain.sourceCount, ∀ position : Fin ringDegree,
      (Sampler.outputWord (SamplerChain.sourceOffset leftOffset source.val) position).eval left =
        (Sampler.outputWord (SamplerChain.sourceOffset rightOffset source.val) position).eval right)
    (outputStateEq : Sampler.evalState left (SamplerChain.outputState leftInterface leftOffset) =
      Sampler.evalState right (SamplerChain.outputState rightInterface rightOffset))
    (specification : SamplerChain.SpecHolds leftInterface leftOffset left) :
    SamplerChain.SpecHolds rightInterface rightOffset right := by
  unfold SamplerChain.evalInitialState at initialEq
  constructor
  · rw [← outputStateEq, specification.state, initialEq]
  · intro source position
    rw [← wordsEq source position, ← initialEq]
    exact specification.digits source position

theorem of_eval_eq (interface : SamplerChain.Interface) (offset : Nat) (left right : Env)
    (initialEq : SamplerChain.evalInitialState interface offset left =
      SamplerChain.evalInitialState interface offset right)
    (wordsEq : ∀ source : Fin SamplerChain.sourceCount, ∀ position : Fin ringDegree,
      (Sampler.outputWord (SamplerChain.sourceOffset offset source.val) position).eval left =
        (Sampler.outputWord (SamplerChain.sourceOffset offset source.val) position).eval right)
    (outputStateEq : Sampler.evalState left (SamplerChain.outputState interface offset) =
      Sampler.evalState right (SamplerChain.outputState interface offset))
    (specification : SamplerChain.SpecHolds interface offset left) :
    SamplerChain.SpecHolds interface offset right :=
  of_cross_eval_eq interface interface offset offset left right initialEq wordsEq outputStateEq specification

end SamplerChain.SpecHolds

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
    (sampler : SamplerChain.SpecHolds
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
    (sampler : SamplerChain.SpecHolds
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
