import NightstreamFPrime.Lifecycle.TranscriptCoverage

/-!
Checks for the transcript coverage contract.

The pins restate the dependency specification `AgreeOnAbsorbed` by `Iff.rfl`,
and the PiCCS domain tag, the challenge labels `[1, c]`, `[2]`, `[3, r]`,
`[4, i]`, the key's round index and the block length prefix by `rfl`, so a
change to any of them must also change this file. The two refutations take
copies of the schedule, one without the fresh commitment and one with `y′`
words without the matrix coordinates, and show that the identify property
fails for them. They show only that the property is not vacuous;
`challenge_seal` and the identify theorems tie it to the key.
-/

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

example : Transcript.piCcsDigestDomainTagBytes =
    "Nightstream/SuperNeo/PiCCS/digest-only/v1_1".toList.map Char.toNat :=
  rfl

example (coordinate : Fin productionShape.cubeVariables) :
    Transcript.labelWord (.alpha coordinate) = [natWord 1, natWord coordinate.val] :=
  rfl

example : Transcript.labelWord .gamma = [natWord 2] :=
  rfl

example (round : Fin productionShape.cubeVariables) :
    Transcript.labelWord (.sumcheck round) = [natWord 3, natWord round.val] :=
  rfl

example (state : Transcript.State) (coordinate : Nat) :
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.enter state coordinate =
      Poseidon2.absorbBlock state [natWord 4, natWord coordinate] :=
  rfl

example (state : Transcript.State) (round : Fin productionShape.cubeVariables)
    (message : SumCheck.Finite.Message K) :
    Transcript.piCcsOracle.transcript.absorbRound state round message =
      Transcript.absorb state
        (natWord (Transcript.serializeMessage message).length.succ ::
          natWord round.val :: Transcript.serializeMessage message) :=
  rfl

end TagsAndLabels

section Refutations

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The key's statement calls with the fresh commitment removed. -/
private def droppedCommitmentCalls
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) : List Call :=
  absorbCalls Transcript.piCcsDigestDomainTag ++
    ([ProductionKey.priorDigest fresh] ++
      (List.finRange productionShape.freshCount).flatMap fun index =>
        [serializePublicInput (fresh.publicInputs index)]).flatMap fun words =>
      absorbCalls (block words)

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

/-- `y′` words with the matrix coordinates removed. -/
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
