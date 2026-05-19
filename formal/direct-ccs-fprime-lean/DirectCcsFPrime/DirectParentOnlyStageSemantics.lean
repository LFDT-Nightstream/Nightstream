import DirectCcsFPrime.DirectStageSemantics
import DirectCcsFPrime.Poseidon2ParentCEBHash

/-!
Verified SuperNeo stage semantics for the parent-only direct CCS F' handle.

This module connects the parent-only terminal theorem to the existing verified
`Pi_CCS`/`Pi_RLC` stage package. The public accumulator still carries only the
parent `CE(B)` source; the verified stage package is applied to the private
child-carrying handle built inside the parent-only latest step.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyStageSemantics

/--
The verified stage object induced by a contextual reused SuperNeo stage package.

This is the parent-only analogue of the existing direct stage contextual path:
the caller supplies deterministic contextual `Pi_CCS`/`Pi_RLC` computations and
the reused SuperNeo authorities; this abbreviation exposes the verified stage
relations consumed by the parent-only terminal theorem.
-/
abbrev verifiedStageOfContextual
    {Digest : Type}
    {n : Nat}
    (stage : DirectStageSemantics.ContextualReusedStageComputations Digest n) :
    DirectStageSemantics.VerifiedStageComputations
      Digest
      DirectStageSemantics.ContextualPiCCSOut
      n :=
  DirectStageSemantics.ReusedStageComputations.toVerified
    (DirectStageSemantics.ContextualReusedStageComputations.toReused stage)

/--
An accepted parent-only `Pi_CCS -> Pi_RLC` stage relation for a verified stage
computes exactly the deterministic parent source owned by that stage package.
-/
theorem parentSourceFrom_verified_stage_eq_compute
    {Digest PiCCSOut : Type}
    {n : Nat}
    (stage :
      DirectStageSemantics.VerifiedStageComputations Digest PiCCSOut n)
    {i : Nat}
    {prior :
      ParentOnlyAccumulatorStep.AccumulatorHandle
        (DigestParentBinding.Source Digest)}
    {priorInputs : DecDigitUniqueness.ColumnDigits n}
    {source : DigestParentBinding.Source Digest}
    (hSource :
      ParentOnlyAccumulatorStep.ParentSourceFromPiStages
        (n := n)
        (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS stage)
        (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC stage)
        i
        prior
        priorInputs
        source) :
    source =
      stage.computePiRLC
        i
        (stage.computePiCCS
          i
          { parentSource := prior.parentSource
            nextPiCCSInputs := priorInputs }) := by
  rcases hSource with ⟨out, hPiCCS, hPiRLC⟩
  have hOut :
      out =
        stage.computePiCCS
          i
          { parentSource := prior.parentSource
            nextPiCCSInputs := priorInputs } :=
    hPiCCS.1
  subst out
  exact hPiRLC.1

/--
Parent-only terminal soundness for verified deterministic SuperNeo stages and
an arbitrary sound prior-authority predicate.
-/
theorem terminal_soundness_of_verified_stage_program_and_msis_of_prior_authority_sound
    {Digest Boundary PiCCSOut Authority : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data : DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : DirectStageSemantics.VerifiedStageComputations Digest PiCCSOut n)
    {hashEncoded : List Nat → Digest}
    {computeBoundary : Nat → Boundary → Boundary}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (DirectParentOnlyTerminalSoundness.AccHandle Digest)}
    {AuthorityAccepts :
      Nat →
        Authority →
        Construction2DirectFPrime.PublicImage
          Digest
          Boundary
          (DirectParentOnlyTerminalSoundness.AccHandle Digest) →
          Prop}
    {priorSteps : Nat}
    {priorAuthority : Authority}
    {proof : Unit}
    (hInitialStep : initial.step = 0)
    (hInitialWellFormed : Construction2DirectFPrime.WellFormed initial)
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (hAuthority :
      FPrimeInduction.PriorAuthoritySound
        (DirectParentOnlyTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS stage)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC stage)
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        AuthorityAccepts)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        AuthorityAccepts
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority := Authority)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectParentOnlyTerminalSoundness.AccumulatorStep
            (params := params)
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS stage)
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC stage)
            hashEncoded
            data.ce
            (ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)))
        priorSteps
        priorAuthority
        priorImage
        nextImage
        proof)
    (hAlt :
      Construction2DirectFPrime.Transition
        (DirectProgramStep.ComputedBoundaryStep computeBoundary)
        (DirectParentOnlyTerminalSoundness.AccumulatorStep
          (params := params)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS stage)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC stage)
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (DirectParentOnlyTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS stage)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC stage)
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage.accumulator.parentSource =
        altNext.accumulator.parentSource ∧
      (∃ priorInputs,
        ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
          (n := n)
          (hashEncoded := hashEncoded)
          (params := params)
          (ce := data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)
          priorImage.accumulator.parentSource
          priorInputs ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS stage)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC stage)
          priorSteps
          priorImage.accumulator
          priorInputs
          nextImage.accumulator.parentSource ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS stage)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC stage)
          priorSteps
          priorImage.accumulator
          priorInputs
          altNext.accumulator.parentSource) ∧
      nextImage.step = priorSteps + 1 ∧
      initial.vkDigest = nextImage.vkDigest ∧
      initial.initialBoundary = nextImage.initialBoundary ∧
      Construction2DirectFPrime.WellFormed nextImage :=
  DirectParentOnlyTerminalSoundness.terminal_soundness_of_prior_authority_sound_and_msis
    data
    (DirectStageSemantics.VerifiedStageComputations.piCCSFunctional stage)
    (DirectStageSemantics.VerifiedStageComputations.piRLCFunctional stage)
    hInitialStep
    hInitialWellFormed
    hDigest
    hRed
    hMsis
    hAuthority
    hAccepted
    hAlt

/--
Parent-only terminal soundness for verified deterministic SuperNeo stages and
a compressed prior verifier packaged as `SoundVerifier`.
-/
theorem terminal_soundness_of_verified_stage_program_and_msis_of_sound_verifier
    {Digest Boundary PiCCSOut PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data : DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : DirectStageSemantics.VerifiedStageComputations Digest PiCCSOut n)
    {hashEncoded : List Nat → Digest}
    {computeBoundary : Nat → Boundary → Boundary}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (DirectParentOnlyTerminalSoundness.AccHandle Digest)}
    (verifier :
      CompressedFPrimeAuthority.SoundVerifier
        (Image :=
          Construction2DirectFPrime.PublicImage
            Digest
            Boundary
            (DirectParentOnlyTerminalSoundness.AccHandle Digest))
        (Proof := PriorProof)
        (DirectParentOnlyTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS stage)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC stage)
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    (hInitialStep : initial.step = 0)
    (hInitialWellFormed : Construction2DirectFPrime.WellFormed initial)
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.SoundVerifier.Accepts verifier)
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority := PriorProof)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectParentOnlyTerminalSoundness.AccumulatorStep
            (params := params)
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS stage)
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC stage)
            hashEncoded
            data.ce
            (ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)))
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      Construction2DirectFPrime.Transition
        (DirectProgramStep.ComputedBoundaryStep computeBoundary)
        (DirectParentOnlyTerminalSoundness.AccumulatorStep
          (params := params)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS stage)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC stage)
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (DirectParentOnlyTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS stage)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC stage)
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage.accumulator.parentSource =
        altNext.accumulator.parentSource ∧
      (∃ priorInputs,
        ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
          (n := n)
          (hashEncoded := hashEncoded)
          (params := params)
          (ce := data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)
          priorImage.accumulator.parentSource
          priorInputs ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS stage)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC stage)
          priorSteps
          priorImage.accumulator
          priorInputs
          nextImage.accumulator.parentSource ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS stage)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC stage)
          priorSteps
          priorImage.accumulator
          priorInputs
          altNext.accumulator.parentSource) ∧
      nextImage.step = priorSteps + 1 ∧
      initial.vkDigest = nextImage.vkDigest ∧
      initial.initialBoundary = nextImage.initialBoundary ∧
      Construction2DirectFPrime.WellFormed nextImage :=
  terminal_soundness_of_verified_stage_program_and_msis_of_prior_authority_sound
    data
    stage
    hInitialStep
    hInitialWellFormed
    hDigest
    hRed
    hMsis
    (CompressedFPrimeAuthority.sound_verifier_prior_authority_sound verifier)
    hAccepted
    hAlt

/--
Parent-only terminal soundness for contextual reused SuperNeo stages and a
compressed prior verifier packaged as `SoundVerifier`.

This is the production-shaped stage theorem before choosing the concrete parent
hash. It consumes the real contextual reused stage object instead of bare
`Pi_CCS`/`Pi_RLC` relations.
-/
theorem terminal_soundness_of_contextual_reused_stage_program_and_msis_of_sound_verifier
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data : DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : DirectStageSemantics.ContextualReusedStageComputations Digest n)
    {hashEncoded : List Nat → Digest}
    {computeBoundary : Nat → Boundary → Boundary}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (DirectParentOnlyTerminalSoundness.AccHandle Digest)}
    (verifier :
      CompressedFPrimeAuthority.SoundVerifier
        (Image :=
          Construction2DirectFPrime.PublicImage
            Digest
            Boundary
            (DirectParentOnlyTerminalSoundness.AccHandle Digest))
        (Proof := PriorProof)
        (DirectParentOnlyTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (verifiedStageOfContextual stage))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (verifiedStageOfContextual stage))
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    (hInitialStep : initial.step = 0)
    (hInitialWellFormed : Construction2DirectFPrime.WellFormed initial)
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.SoundVerifier.Accepts verifier)
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority := PriorProof)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectParentOnlyTerminalSoundness.AccumulatorStep
            (params := params)
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
              (verifiedStageOfContextual stage))
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
              (verifiedStageOfContextual stage))
            hashEncoded
            data.ce
            (ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)))
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      Construction2DirectFPrime.Transition
        (DirectProgramStep.ComputedBoundaryStep computeBoundary)
        (DirectParentOnlyTerminalSoundness.AccumulatorStep
          (params := params)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (verifiedStageOfContextual stage))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (verifiedStageOfContextual stage))
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (DirectParentOnlyTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (verifiedStageOfContextual stage))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (verifiedStageOfContextual stage))
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage.accumulator.parentSource =
        altNext.accumulator.parentSource ∧
      (∃ priorInputs,
        ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
          (n := n)
          (hashEncoded := hashEncoded)
          (params := params)
          (ce := data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)
          priorImage.accumulator.parentSource
          priorInputs ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (verifiedStageOfContextual stage))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (verifiedStageOfContextual stage))
          priorSteps
          priorImage.accumulator
          priorInputs
          nextImage.accumulator.parentSource ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (verifiedStageOfContextual stage))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (verifiedStageOfContextual stage))
          priorSteps
          priorImage.accumulator
          priorInputs
          altNext.accumulator.parentSource) ∧
      nextImage.step = priorSteps + 1 ∧
      initial.vkDigest = nextImage.vkDigest ∧
      initial.initialBoundary = nextImage.initialBoundary ∧
      Construction2DirectFPrime.WellFormed nextImage :=
  terminal_soundness_of_verified_stage_program_and_msis_of_sound_verifier
    data
    (verifiedStageOfContextual stage)
    verifier
    hInitialStep
    hInitialWellFormed
    hDigest
    hRed
    hMsis
    hAccepted
    hAlt

/--
Parent-only terminal soundness for the contextual reused SuperNeo stage package,
the canonical parent `CE(B)` hash-binding object, and a sound compressed prior
verifier.

This is the hash-boundary production surface before choosing a concrete
implementation hash. Callers supply the structured parent-handle binding object
directly; the theorem does not accept a loose digest-binding proof detached
from the hash function it protects.
-/
theorem terminal_soundness_of_parent_hash_contextual_reused_stage_program_and_msis_of_sound_verifier
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (parentHash : ParentCEBHashBinding.ParentCEBHash Digest)
    (data : DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : DirectStageSemantics.ContextualReusedStageComputations Digest n)
    {computeBoundary : Nat → Boundary → Boundary}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (DirectParentOnlyTerminalSoundness.AccHandle Digest)}
    (verifier :
      CompressedFPrimeAuthority.SoundVerifier
        (Image :=
          Construction2DirectFPrime.PublicImage
            Digest
            Boundary
            (DirectParentOnlyTerminalSoundness.AccHandle Digest))
        (Proof := PriorProof)
        (DirectParentOnlyTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (verifiedStageOfContextual stage))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (verifiedStageOfContextual stage))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    (hInitialStep : initial.step = 0)
    (hInitialWellFormed : Construction2DirectFPrime.WellFormed initial)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.SoundVerifier.Accepts verifier)
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority := PriorProof)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectParentOnlyTerminalSoundness.AccumulatorStep
            (params := params)
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
              (verifiedStageOfContextual stage))
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
              (verifiedStageOfContextual stage))
            parentHash.hashEncoded
            data.ce
            (ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)))
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      Construction2DirectFPrime.Transition
        (DirectProgramStep.ComputedBoundaryStep computeBoundary)
        (DirectParentOnlyTerminalSoundness.AccumulatorStep
          (params := params)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (verifiedStageOfContextual stage))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (verifiedStageOfContextual stage))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (DirectParentOnlyTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (verifiedStageOfContextual stage))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (verifiedStageOfContextual stage))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage.accumulator.parentSource =
        altNext.accumulator.parentSource ∧
      (∃ priorInputs,
        ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
          (n := n)
          (hashEncoded := parentHash.hashEncoded)
          (params := params)
          (ce := data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)
          priorImage.accumulator.parentSource
          priorInputs ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (verifiedStageOfContextual stage))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (verifiedStageOfContextual stage))
          priorSteps
          priorImage.accumulator
          priorInputs
          nextImage.accumulator.parentSource ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (verifiedStageOfContextual stage))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (verifiedStageOfContextual stage))
          priorSteps
          priorImage.accumulator
          priorInputs
          altNext.accumulator.parentSource) ∧
      nextImage.step = priorSteps + 1 ∧
      initial.vkDigest = nextImage.vkDigest ∧
      initial.initialBoundary = nextImage.initialBoundary ∧
      Construction2DirectFPrime.WellFormed nextImage :=
  terminal_soundness_of_contextual_reused_stage_program_and_msis_of_sound_verifier
    data
    stage
    verifier
    hInitialStep
    hInitialWellFormed
    (ParentCEBHashBinding.encodedParentCEBDigestBinding parentHash)
    hRed
    hMsis
    hAccepted
    hAlt

/--
Parent-only terminal soundness for the contextual reused SuperNeo stage package,
the implementation-facing Poseidon2 parent hash boundary, and a sound compressed
prior verifier.

This theorem is the compact implementation-facing surface for the optimized
path: the public accumulator carries only the parent `CE(B)` source, the
post-DEC children remain private, and the theorem extracts the same pointwise
authorized child table for the accepted and alternate latest steps.
-/
theorem terminal_soundness_of_poseidon2_parent_hash_contextual_reused_stage_program_and_msis_of_sound_verifier
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (parentHash : Poseidon2ParentCEBHash.Hash Digest)
    (data : DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : DirectStageSemantics.ContextualReusedStageComputations Digest n)
    {computeBoundary : Nat → Boundary → Boundary}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (DirectParentOnlyTerminalSoundness.AccHandle Digest)}
    (verifier :
      CompressedFPrimeAuthority.SoundVerifier
        (Image :=
          Construction2DirectFPrime.PublicImage
            Digest
            Boundary
            (DirectParentOnlyTerminalSoundness.AccHandle Digest))
        (Proof := PriorProof)
        (DirectParentOnlyTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (verifiedStageOfContextual stage))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (verifiedStageOfContextual stage))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    (hInitialStep : initial.step = 0)
    (hInitialWellFormed : Construction2DirectFPrime.WellFormed initial)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.SoundVerifier.Accepts verifier)
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority := PriorProof)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectParentOnlyTerminalSoundness.AccumulatorStep
            (params := params)
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
              (verifiedStageOfContextual stage))
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
              (verifiedStageOfContextual stage))
            parentHash.hashEncoded
            data.ce
            (ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)))
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      Construction2DirectFPrime.Transition
        (DirectProgramStep.ComputedBoundaryStep computeBoundary)
        (DirectParentOnlyTerminalSoundness.AccumulatorStep
          (params := params)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (verifiedStageOfContextual stage))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (verifiedStageOfContextual stage))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (DirectParentOnlyTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (verifiedStageOfContextual stage))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (verifiedStageOfContextual stage))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage.accumulator.parentSource =
        altNext.accumulator.parentSource ∧
      (∃ priorInputs,
        ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
          (n := n)
          (hashEncoded := parentHash.hashEncoded)
          (params := params)
          (ce := data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent)
          priorImage.accumulator.parentSource
          priorInputs ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (verifiedStageOfContextual stage))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (verifiedStageOfContextual stage))
          priorSteps
          priorImage.accumulator
          priorInputs
          nextImage.accumulator.parentSource ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (verifiedStageOfContextual stage))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (verifiedStageOfContextual stage))
          priorSteps
          priorImage.accumulator
          priorInputs
          altNext.accumulator.parentSource) ∧
      nextImage.step = priorSteps + 1 ∧
      initial.vkDigest = nextImage.vkDigest ∧
      initial.initialBoundary = nextImage.initialBoundary ∧
      Construction2DirectFPrime.WellFormed nextImage :=
  terminal_soundness_of_contextual_reused_stage_program_and_msis_of_sound_verifier
    data
    stage
    verifier
    hInitialStep
    hInitialWellFormed
    (Poseidon2ParentCEBHash.encodedParentCEBDigestBinding parentHash)
    hRed
    hMsis
    hAccepted
    hAlt

end DirectParentOnlyStageSemantics

end DirectCcsFPrime
