import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Types
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

/-!
Finite-support obstruction for the production Split-NC challenge carrier.

Assurance tier: model-level obstruction.

Owns: exact denominator/support correspondence, its positivity consequence,
and a zero-denominator countermodel that changes only the unrestricted
`challengeSetSize` field of the production verifier-context carrier.

Does not own: an abstract replacement support, a claim that a selected future
production constructor sets zero, challenge sampling, Fiat--Shamir, Rust/R1CS,
encoding, or rows.

Emits constraints: no.

| Boundary | Owned equation | Excluded boundary |
|---|---|---|
| denominator alignment | `challengeSetSize = alphabet.cardinality` | invented production alphabet |
| carrier countermodel | only `challengeSetSize` changes | claim about a future refined constructor |
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
  {arity : BatchArity productionGlobalParams}

/-- Exact correspondence required between the production theorem's
`challengeSetSize` denominator and an actually sampled finite-uniform support.
The support itself carries `Nodup` and nonemptiness proofs. -/
def ChallengeSupportAligned
    (input : Context shape State publicRingColumns publicFits verifierRows arity)
    (alphabet : Support K) : Prop :=
  input.challengeSetSize = alphabet.cardinality

/-- Any exact denominator/support correspondence forces the production
denominator to be positive. -/
theorem challengeSetSize_pos_of_aligned
    (input : Context shape State publicRingColumns publicFits verifierRows arity)
    (alphabet : Support K)
    (aligned : ChallengeSupportAligned input alphabet) :
    0 < input.challengeSetSize := by
  rw [aligned]
  exact alphabet.cardinality_pos

/-- Change only the unrestricted denominator on the current production
context carrier. The complete statement input, transcript schedule, and
schedule-derived coins are preserved. -/
def withChallengeSetSize
    (input : Context shape State publicRingColumns publicFits verifierRows arity)
    (challengeSetSize : Nat) :
    Context shape State publicRingColumns publicFits verifierRows arity :=
  { input with challengeSetSize := challengeSetSize }

@[simp] theorem withChallengeSetSize_input
    (input : Context shape State publicRingColumns publicFits verifierRows arity)
    (challengeSetSize : Nat) :
    (withChallengeSetSize input challengeSetSize).input = input.input := by
  rfl

@[simp] theorem withChallengeSetSize_schedule
    (input : Context shape State publicRingColumns publicFits verifierRows arity)
    (challengeSetSize : Nat) :
    (withChallengeSetSize input challengeSetSize).piCcsSchedule =
      input.piCcsSchedule := by
  rfl

@[simp] theorem withChallengeSetSize_statement
    (input : Context shape State publicRingColumns publicFits verifierRows arity)
    (challengeSetSize : Nat) :
    (withChallengeSetSize input challengeSetSize).piCcsStatement =
      input.piCcsStatement := by
  rfl

@[simp] theorem withChallengeSetSize_priorState
    (input : Context shape State publicRingColumns publicFits verifierRows arity)
    (challengeSetSize : Nat) :
    (withChallengeSetSize input challengeSetSize).priorState =
      input.priorState := by
  rfl

@[simp] theorem withChallengeSetSize_value
    (input : Context shape State publicRingColumns publicFits verifierRows arity)
    (challengeSetSize : Nat) :
    (withChallengeSetSize input challengeSetSize).challengeSetSize =
      challengeSetSize := by
  rfl

@[simp] theorem withChallengeSetSize_feCoins
    (input : Context shape State publicRingColumns publicFits verifierRows arity)
    (challengeSetSize : Nat) :
    (withChallengeSetSize input challengeSetSize).feCoins =
      input.feCoins := by
  rfl

@[simp] theorem withChallengeSetSize_ncCoins
    (input : Context shape State publicRingColumns publicFits verifierRows arity)
    (challengeSetSize : Nat) :
    (withChallengeSetSize input challengeSetSize).ncCoins =
      input.ncCoins := by
  rfl

/-- Exact support obstruction at the current carrier boundary. The carrier
admits denominator zero without changing the complete statement, prior state,
transcript schedule, FE coins, or NC coins, whereas every admissible support
has positive cardinality. This does not obstruct a refined production
constructor that supplies and aligns an explicit support. -/
theorem zeroChallengeSetSize_has_no_aligned_support
    (input : Context shape State publicRingColumns publicFits verifierRows arity) :
    ¬ ∃ alphabet : Support K,
      ChallengeSupportAligned (withChallengeSetSize input 0) alphabet := by
  rintro ⟨alphabet, aligned⟩
  have positive : 0 < 0 := by
    simpa only [withChallengeSetSize_value] using
      challengeSetSize_pos_of_aligned
        (withChallengeSetSize input 0) alphabet aligned
  exact (Nat.lt_irrefl 0) positive

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionMixingBoundary
