import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerTotalizedOutputLaw

/-!
Owns a comparison-only copy of the selected NIFS verifier with scalarwise
totalized PiRLC sampling. Actual success preserves the complete sampler
batch and the exact running output. The production key and decoder are
unchanged. No failed-trace, work, distribution, or Fiat--Shamir claim is made.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.TotalizedComparison

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
open ProductionStrongSet
open NightstreamFPrime.Lifecycle.PaperAlgebra

/-- The comparison runs the same fixed state schedule and totalizes each
scalar separately. Its fallback is a complete typed production scalar. -/
def totalizedBatch (fallback : Scalar) (initial : Transcript.State) (count : Nat) :
    Transcript.PiRlcSampler.Batch count where
  challenges := fun index => Phi81StrongSet.embedScalar
    (Lifecycle.PiRLC.v1_1.SamplerTotalizedOutputLaw.totalizedFieldBatch fallback
      (fun index : Fin count => Lifecycle.PiRLC.v1_1.SamplerFieldShortfall.fieldWindow initial index.val)
      index)
  finalState := stateAt Transcript.PiRlcSampler.specification initial count

private theorem batch_ext {count : Nat} (left right : Transcript.PiRlcSampler.Batch count)
    (values : left.challenges = right.challenges)
    (state : left.finalState = right.finalState) : left = right := by
  cases left
  cases right
  cases values
  cases state
  rfl

/-- Actual success agrees with the comparison in every ordered challenge
and the complete returned sampler state, for every initial state and count. -/
theorem sampleBatch_success_eq (fallback : Scalar) (initial : Transcript.State) (count : Nat)
    (batch : Transcript.PiRlcSampler.Batch count)
    (success : Transcript.PiRlcSampler.sampleBatch initial count = some batch) :
    batch = totalizedBatch fallback initial count := by
  apply batch_ext
  · apply List.ofFn_injective
    simpa only [totalizedBatch, List.map_ofFn, Function.comp_def] using
      Lifecycle.PiRLC.v1_1.SamplerTotalizedOutputLaw.sampleBatch_success_ring_list_eq
        fallback initial count batch success
  · exact Transcript.PiRlcSampler.piRlcChallengesWithState_finalState success


private def totalizedResponse (fallback : Scalar) (initial : Transcript.State) :
    Option (Fin Nifs.PaperProfile.arity.total → RingF) :=
  some (totalizedBatch fallback initial Nifs.PaperProfile.arity.total).challenges

private theorem totalizedResponse_valid (fallback : Scalar) (initial : Transcript.State)
    (response : Fin Nifs.PaperProfile.arity.total → RingF)
    (success : totalizedResponse fallback initial = some response)
    (index : Fin Nifs.PaperProfile.arity.total) :
    Phi81Relation.PiRLCAlgebra.Challenge.challengeValid (response index) := by
  have same :
      (totalizedBatch fallback initial Nifs.PaperProfile.arity.total).challenges = response :=
    Option.some.inj success
  rw [← same]
  exact Phi81Relation.PiRLCAlgebra.Challenge.embedScalar_valid _

section SymbolicKey

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (key : ProductionKey.KeyType relation)
  (response : Transcript.State → Option (Fin key.arity.total → RingF))
  (valid : ∀ state values, response state = some values →
    ∀ index, key.piRlcAlgebra.challengeValid (values index))

/-- Keep the key symbolic while proving the composition identities. The
selected instantiation below changes only this response and its proof. -/
private def replaceResponse : ProductionKey.KeyType relation :=
  { key with piRlcResponse := response, piRlcResponseValid := valid }

variable
  (running : Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  (proof : Lifecycle.Proof (ProductionKey.degreeBound relation))

private theorem replaceResponse_execution_eq :
    (replaceResponse relation key response valid).piCcsExecution running fresh proof =
      key.piCcsExecution running fresh proof := by
  rfl

private theorem replaceResponse_verify_eq
    (same : response (key.piCcsExecution running fresh proof).outgoingState =
      key.piRlcChallenges running fresh proof) :
    Nifs.PaperNonInteractive.verify (replaceResponse relation key response valid)
        running fresh proof =
      Nifs.PaperNonInteractive.verify key running fresh proof := by
  have ccs :
      Nifs.PaperNonInteractive.piCcsCheck (replaceResponse relation key response valid)
          running fresh proof =
        Nifs.PaperNonInteractive.piCcsCheck key running fresh proof := by
    rfl
  have challenges :
      (replaceResponse relation key response valid).piRlcChallenges running fresh proof =
        key.piRlcChallenges running fresh proof := by
    change response
      ((replaceResponse relation key response valid).piCcsExecution running fresh proof).outgoingState =
        key.piRlcChallenges running fresh proof
    rw [replaceResponse_execution_eq]
    exact same
  have parentFunction :
      (replaceResponse relation key response valid).parentForChallenges running fresh proof =
        key.parentForChallenges running fresh proof := by
    rfl
  have parent :
      (replaceResponse relation key response valid).parent running fresh proof =
        key.parent running fresh proof :=
    congrArg₂ (fun f values => Option.map f values) parentFunction challenges
  have attemptFunction :
      (replaceResponse relation key response valid).piDecAttemptForParent proof =
        key.piDecAttemptForParent proof := by
    rfl
  have attempt :
      (replaceResponse relation key response valid).piDecAttempt running fresh proof =
        key.piDecAttempt running fresh proof :=
    congrArg₂ (fun f values => Option.map f values) attemptFunction parent
  have dec :
      Nifs.PaperNonInteractive.piDecCheck (replaceResponse relation key response valid)
          running fresh proof =
        Nifs.PaperNonInteractive.piDecCheck key running fresh proof := by
    unfold Nifs.PaperNonInteractive.piDecCheck
    rw [attempt]
    cases key.piDecAttempt running fresh proof with
    | none => rfl
    | some _ => exact decide_eq_decide.mpr Iff.rfl
  have output :
      (replaceResponse relation key response valid).output running fresh proof =
        key.output running fresh proof := by
    unfold Nifs.PaperNonInteractive.Key.output
    rw [attempt]
    rfl
  unfold Nifs.PaperNonInteractive.verify
  rw [ccs, dec, output]

end SymbolicKey

section SelectedKey

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (fallback : Scalar)
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- Only the comparison response and its membership proof differ from the
production key. All C/D checks, public inputs, relation and transcript fields
remain the existing selected owners. -/
noncomputable def comparisonKey : ProductionKey.KeyType relation :=
  replaceResponse relation (ProductionKey.key relation ajtai)
    (totalizedResponse fallback) (totalizedResponse_valid fallback)

variable
  (running : Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  (proof : Lifecycle.Proof (ProductionKey.degreeBound relation))

/-- Reuse the complete NIFS verifier with the comparison response. This is
not installed in the production setup or any runtime entrypoint. -/
noncomputable def verify :
    Option (Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits)) :=
  Nifs.PaperNonInteractive.verify (comparisonKey fallback relation ajtai) running fresh proof

/-- Once actual sampling succeeds, both complete verifier results agree,
including rejection by C or D. No hypothesis on the child messages is used. -/
theorem verify_eq_of_sampleBatch_success
    (batch : Transcript.PiRlcSampler.Batch Nifs.PaperProfile.arity.total)
    (success : Transcript.PiRlcSampler.sampleBatch
      ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).outgoingState
      Nifs.PaperProfile.arity.total = some batch) :
    verify fallback relation ajtai running fresh proof =
      Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai) running fresh proof := by
  have same := sampleBatch_success_eq fallback
    ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).outgoingState
    Nifs.PaperProfile.arity.total batch success
  have response :
      totalizedResponse fallback
        ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).outgoingState =
        (ProductionKey.key relation ajtai).piRlcChallenges running fresh proof := by
    change some (totalizedBatch fallback
        ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).outgoingState
        Nifs.PaperProfile.arity.total).challenges =
      (Transcript.PiRlcSampler.sampleBatch
        ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).outgoingState
        Nifs.PaperProfile.arity.total).map Transcript.PiRlcSampler.Batch.challenges
    rw [success, Option.map_some, same]
  exact replaceResponse_verify_eq relation (ProductionKey.key relation ajtai)
    (totalizedResponse fallback) (totalizedResponse_valid fallback) running fresh proof response

/-- Acceptance of the actual verifier gives the same comparison output
and the exact complete actual sampler batch. Actual acceptance is impossible
on shortfall; no equivalence of failed traces is claimed. -/
theorem accepted_actual_implies_comparison
    (output : Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (accepted : Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
      running fresh proof = some output) :
    verify fallback relation ajtai running fresh proof = some output ∧
      Transcript.PiRlcSampler.sampleBatch
        ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).outgoingState
        Nifs.PaperProfile.arity.total =
        some (totalizedBatch fallback
          ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).outgoingState
          Nifs.PaperProfile.arity.total) := by
  cases sampled : Transcript.PiRlcSampler.sampleBatch
      ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).outgoingState
      Nifs.PaperProfile.arity.total with
  | none =>
      have failure : (ProductionKey.key relation ajtai).piRlcChallenges running fresh proof = none := by
        change (Transcript.PiRlcSampler.sampleBatch
          ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).outgoingState
          Nifs.PaperProfile.arity.total).map Transcript.PiRlcSampler.Batch.challenges = none
        rw [sampled, Option.map_none]
      have rejected := Nifs.PaperNonInteractive.verify_eq_none_of_piRlcFailure
        (ProductionKey.key relation ajtai) running fresh proof failure
      rw [accepted] at rejected
      cases rejected
  | some batch =>
      constructor
      · rw [verify_eq_of_sampleBatch_success fallback relation ajtai running fresh proof batch sampled]
        exact accepted
      · exact congrArg some (sampleBatch_success_eq fallback
          ((ProductionKey.key relation ajtai).piCcsExecution running fresh proof).outgoingState
          Nifs.PaperProfile.arity.total batch sampled)

end SelectedKey

end NightstreamFPrime.Lifecycle.Nifs.TotalizedComparison
