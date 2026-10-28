import NightstreamFPrime.Lifecycle.TranscriptCoverage
import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1

/-! Checks for the transcript coverage contract. -/

namespace NightstreamFPrime.Tests.TranscriptCoverageChecks

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.TranscriptCoverage

section Pins

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  {degree : Nat}
  (fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  (proof proof' : Proof degree)

example (coordinate : Fin productionShape.cubeVariables) :
    AgreeOnAbsorbed fresh fresh' proof proof' (.alpha coordinate) ↔ fresh = fresh' :=
  Iff.rfl

example : AgreeOnAbsorbed fresh fresh' proof proof' .gamma ↔ fresh = fresh' :=
  Iff.rfl

example (round : Fin productionShape.cubeVariables) :
    AgreeOnAbsorbed fresh fresh' proof proof' (.round round) ↔
      fresh = fresh' ∧ ∀ earlier : Fin productionShape.cubeVariables,
        earlier.val ≤ round.val → proof.piCcsRounds earlier = proof'.piCcsRounds earlier :=
  Iff.rfl

example (index : Fin (Nifs.PaperProfile.arity).total) :
    AgreeOnAbsorbed fresh fresh' proof proof' (.rho index) ↔
      fresh = fresh' ∧
        { proof with
          piDecCommitments := proof'.piDecCommitments
          piDecEvaluations := proof'.piDecEvaluations } = proof' :=
  Iff.rfl

end Pins

section TagsAndLabels

example : Transcript.foldDomainBytes =
    [78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47,
      83, 117, 112, 101, 114, 78, 101, 111, 47, 102, 111, 108, 100, 47, 118, 50] :=
  rfl

example : Lifecycle.stateDomainBytes =
    [72, 121, 112, 101, 114, 78, 111, 118, 97, 47, 78, 73, 86, 67,
      47, 115, 116, 97, 116, 101, 47, 118, 50] :=
  rfl

example : Lifecycle.Stage1.Poseidon2HashChainV1.domainTagBytes =
    [78, 105, 103, 104, 116, 115, 116, 114, 101, 97,
      109, 47, 83, 116, 97, 103, 101, 49, 47, 80,
      111, 115, 101, 105, 100, 111, 110, 50, 72, 97,
      115, 104, 67, 104, 97, 105, 110, 47, 118, 49] :=
  rfl

example (coordinate : Fin productionShape.cubeVariables) :
    Transcript.coinPosition (.alpha coordinate) = coordinate.val :=
  rfl

example : Transcript.coinPosition .gamma = productionShape.cubeVariables :=
  rfl

example (round : Fin productionShape.cubeVariables) :
    Transcript.coinPosition (.sumcheck round) = 0 :=
  rfl

example (state : Transcript.State) (position : Nat) :
    (Transcript.squeezeAt state position).1 =
      ⟨state.getD (2 * (position % 6)) 0, state.getD (2 * (position % 6) + 1) 0⟩ :=
  rfl

example (state : Transcript.State) (coordinate : Nat) :
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.enter state coordinate =
      Poseidon2.absorbBlock state [natWord 4, natWord coordinate] :=
  rfl

example (state : Transcript.State) (round : Fin productionShape.cubeVariables)
    (message : SumCheck.Finite.Message K) :
    Transcript.piCcsOracle.transcript.absorbRound state round message =
      Transcript.absorb state (Transcript.serializeMessage message) :=
  rfl

example {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (ProductionKey.key relation ajtai).publicInputState running fresh =
      Transcript.absorb (Transcript.absorb Transcript.initialState Transcript.foldDomainChunk)
        (ProductionKey.publicInputBlocks fresh).flatten :=
  rfl

example (state : Transcript.State) (label : FiatShamir.ChallengeLabel productionShape) :
    Transcript.piCcsOracle.transcript.squeeze state label =
      Transcript.squeezeAt state (Transcript.coinPosition label) :=
  rfl

end TagsAndLabels

section Refutations

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

private def droppedCommitmentCalls
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) : List Call :=
  absorbCalls Transcript.foldDomainChunk ++
    absorbCalls (ProductionKey.priorDigest fresh ++
      (List.finRange productionShape.freshCount).flatMap fun index =>
        serializePublicInput (fresh.publicInputs index))

private def freshWith (value : F) :
    Fresh (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  { commitments := fun _ _ _ => value, publicInputs := fun _ _ => 0 }

example :
    ¬ ∀ fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits),
      droppedCommitmentCalls fresh = droppedCommitmentCalls fresh' → fresh = fresh' := by
  intro coverage
  have same := coverage (freshWith 0) (freshWith 1) rfl
  have entry := congrFun (congrFun (congrFun
    (congrArg Nifs.PaperNonInteractive.Fresh.commitments same) ⟨0, by decide⟩)
      ⟨0, by decide⟩) ⟨0, by decide⟩
  change (0 : F) = 1 at entry
  exact absurd entry (by decide)

end Refutations

private def padOnlyWords (output : FullOutputCoordinates.FullOutput K productionShape) :
    List F :=
  (List.finRange productionShape.sourceCount).flatMap fun source =>
    (List.finRange productionShape.coefficientCount).flatMap fun coefficient =>
      serializeK (output.padCoordinate source coefficient)

private def outputWith (value : F) : FullOutputCoordinates.FullOutput K productionShape :=
  { padCoordinate := fun _ _ => K.zero, matrixCoordinate := fun _ _ _ => ⟨value, 0⟩ }

example :
    ¬ ∀ output output' : FullOutputCoordinates.FullOutput K productionShape,
      padOnlyWords output = padOnlyWords output' → output = output' := by
  intro coverage
  have same := coverage (outputWith 0) (outputWith 1) rfl
  have entry := congrArg K.c0 (congrFun (congrFun (congrFun
    (congrArg FullOutputCoordinates.FullOutput.matrixCoordinate same) ⟨0, by decide⟩)
      ⟨0, by decide⟩) ⟨0, by decide⟩)
  exact absurd entry (by decide)

end NightstreamFPrime.Tests.TranscriptCoverageChecks
