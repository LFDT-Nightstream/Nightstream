import DirectCcsFPrime.DirectStageSemanticsContextual
import DirectCcsFPrime.ParentCEBHashBinding

/-!
Reduced-handle terminal composition.

This module owns the terminal theorem shape for the reduced parent-handle path:
the parent hash is supplied as a `ParentCEBHashBinding.ParentCEBHash`, so the
terminal proof consumes the exact parent `CE(B)` binding boundary
instead of a raw encoded-digest premise.
-/

namespace DirectCcsFPrime

namespace ReducedHandleTerminal

/--
Terminal soundness for the reduced parent-handle path with an arbitrary sound
prior-authority predicate.

This is the essential theorem boundary: the terminal proof consumes one
structured parent `CE(B)` hash, verified SuperNeo stage context, and a
prior authority whose acceptance implies actual F' reachability.
-/
theorem prior_authority_soundness
    {Digest Boundary Authority : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (parentHash : ParentCEBHashBinding.ParentCEBHash Digest)
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : DirectStageSemantics.ContextualReusedStageComputations Digest n)
    {computeBoundary : Nat → Boundary → Boundary}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (DirectTerminalSoundness.AccHandle Digest n)}
    {AuthorityAccepts :
      Nat →
        Authority →
        Construction2DirectFPrime.PublicImage
          Digest
          Boundary
          (DirectTerminalSoundness.AccHandle Digest n) →
          Prop}
    {priorSteps : Nat}
    {priorAuthority : Authority}
    {proof : Unit}
    (hInitialStep : initial.step = 0)
    (hInitialWellFormed : Construction2DirectFPrime.WellFormed initial)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (hAuthority :
      FPrimeInduction.PriorAuthoritySound
        (DirectTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        AuthorityAccepts)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        AuthorityAccepts
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority :=
            Authority)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectTerminalSoundness.AccumulatorStep
            (params := params)
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
              (DirectStageSemantics.ReusedStageComputations.toVerified
                (DirectStageSemantics.ContextualReusedStageComputations.toReused
                  stage)))
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
              (DirectStageSemantics.ReusedStageComputations.toVerified
                (DirectStageSemantics.ContextualReusedStageComputations.toReused
                  stage)))
            parentHash.hashEncoded
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
        (DirectTerminalSoundness.AccumulatorStep
          (params := params)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (DirectTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage = altNext ∧
      nextImage.step = priorSteps + 1 ∧
      initial.vkDigest = nextImage.vkDigest ∧
      initial.initialBoundary = nextImage.initialBoundary ∧
      Construction2DirectFPrime.WellFormed nextImage :=
  DirectStageSemantics.terminal_soundness_of_contextual_reused_stage_program_and_msis_of_prior_authority_sound_with_public_image_invariants
    data
    stage
    hInitialStep
    hInitialWellFormed
    (ParentCEBHashBinding.encodedParentCEBDigestBinding parentHash)
    hRed
    hMsis
    hAuthority
    hAccepted
    hAlt

/--
An accepted unreachable prior image cannot be valid reduced-handle induction
authority.

This is the local guardrail against self-consistent digest chains: for the
exact reduced-handle F' transition below, a prior-authority predicate is sound
only if every accepted prior image is reachable from the initial image.
-/
theorem accepted_unreachable_prior_is_not_sound_authority
    {Digest Boundary Authority : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (parentHash : ParentCEBHashBinding.ParentCEBHash Digest)
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : DirectStageSemantics.ContextualReusedStageComputations Digest n)
    {computeBoundary : Nat → Boundary → Boundary}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {initial image :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (DirectTerminalSoundness.AccHandle Digest n)}
    {AuthorityAccepts :
      Nat →
        Authority →
        Construction2DirectFPrime.PublicImage
          Digest
          Boundary
          (DirectTerminalSoundness.AccHandle Digest n) →
          Prop}
    {steps : Nat}
    {authority : Authority}
    (hAccept :
      AuthorityAccepts steps authority image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
          (DirectTerminalSoundness.Transition
            (params := params)
            (DirectProgramStep.ComputedBoundaryStep computeBoundary)
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
              (DirectStageSemantics.ReusedStageComputations.toVerified
                (DirectStageSemantics.ContextualReusedStageComputations.toReused
                  stage)))
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
              (DirectStageSemantics.ReusedStageComputations.toVerified
                (DirectStageSemantics.ContextualReusedStageComputations.toReused
                  stage)))
            parentHash.hashEncoded
            data.ce
            (ParentOpeningAuthorization.StatementEncodesByCommitment
              commitmentOfParent))
          initial
          steps
          image) :
    ¬ FPrimeInduction.PriorAuthoritySound
        (DirectTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        AuthorityAccepts :=
  FPrimeInduction.digest_only_acceptance_not_sound_when_it_accepts_unreachable
    hAccept
    hUnreachable

/--
Compressed-prior terminal soundness for the reduced parent-handle path.

The parent hash context supplies the encoded parent `CE(B)` binding
premise consumed by the lower terminal theorem.
-/
theorem compressed_prior_soundness
    {Digest Boundary : Type}
    {PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (parentHash : ParentCEBHashBinding.ParentCEBHash Digest)
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : DirectStageSemantics.ContextualReusedStageComputations Digest n)
    {computeBoundary : Nat → Boundary → Boundary}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (DirectTerminalSoundness.AccHandle Digest n)}
    {VerifyPrior :
      Nat →
        PriorProof →
        Construction2DirectFPrime.PublicImage
          Digest
          Boundary
          (DirectTerminalSoundness.AccHandle Digest n) →
          Prop}
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    (hInitialStep : initial.step = 0)
    (hInitialWellFormed : Construction2DirectFPrime.WellFormed initial)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (hPriorVerifier :
      CompressedFPrimeAuthority.VerifierSound
        (DirectTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        VerifyPrior)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts VerifyPrior)
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority :=
            PriorProof)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectTerminalSoundness.AccumulatorStep
            (params := params)
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
              (DirectStageSemantics.ReusedStageComputations.toVerified
                (DirectStageSemantics.ContextualReusedStageComputations.toReused
                  stage)))
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
              (DirectStageSemantics.ReusedStageComputations.toVerified
                (DirectStageSemantics.ContextualReusedStageComputations.toReused
                  stage)))
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
        (DirectTerminalSoundness.AccumulatorStep
          (params := params)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (DirectTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage = altNext ∧
      nextImage.step = priorSteps + 1 ∧
      initial.vkDigest = nextImage.vkDigest ∧
      initial.initialBoundary = nextImage.initialBoundary ∧
      Construction2DirectFPrime.WellFormed nextImage :=
  prior_authority_soundness
    parentHash
    data
    stage
    hInitialStep
    hInitialWellFormed
    hRed
    hMsis
    (CompressedFPrimeAuthority.accepts_sound_of_verifier_sound
      hPriorVerifier)
    hAccepted
    hAlt

/--
Compressed-prior terminal soundness when every accepted compressed proof opens
to proof-carrying folded F' authority.

This is the production-shaped specialization of `compressed_prior_soundness`:
callers do not need to provide `VerifierSound` directly, but must prove that
the compressed verifier's accepted proof represents a theorem-level folded
authority for the same `(steps, image)`.
-/
theorem compressed_prior_soundness_of_opens_to_folded_authority
    {Digest Boundary : Type}
    {PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (parentHash : ParentCEBHashBinding.ParentCEBHash Digest)
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : DirectStageSemantics.ContextualReusedStageComputations Digest n)
    {computeBoundary : Nat → Boundary → Boundary}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (DirectTerminalSoundness.AccHandle Digest n)}
    {VerifyPrior :
      Nat →
        PriorProof →
        Construction2DirectFPrime.PublicImage
          Digest
          Boundary
          (DirectTerminalSoundness.AccHandle Digest n) →
          Prop}
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    (hInitialStep : initial.step = 0)
    (hInitialWellFormed : Construction2DirectFPrime.WellFormed initial)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (hOpens :
      ∀ steps proof image,
        VerifyPrior steps proof image →
          ∃ authority :
            FoldedFPrimeAuthority.Authority
              (DirectTerminalSoundness.Transition
                (params := params)
                (DirectProgramStep.ComputedBoundaryStep computeBoundary)
                (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
                  (DirectStageSemantics.ReusedStageComputations.toVerified
                    (DirectStageSemantics.ContextualReusedStageComputations.toReused
                      stage)))
                (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
                  (DirectStageSemantics.ReusedStageComputations.toVerified
                    (DirectStageSemantics.ContextualReusedStageComputations.toReused
                      stage)))
                parentHash.hashEncoded
                data.ce
                (ParentOpeningAuthorization.StatementEncodesByCommitment
                  commitmentOfParent))
              initial,
            FoldedFPrimeAuthority.Accepts
              (Transition :=
                DirectTerminalSoundness.Transition
                  (params := params)
                  (DirectProgramStep.ComputedBoundaryStep computeBoundary)
                  (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
                    (DirectStageSemantics.ReusedStageComputations.toVerified
                      (DirectStageSemantics.ContextualReusedStageComputations.toReused
                        stage)))
                  (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
                    (DirectStageSemantics.ReusedStageComputations.toVerified
                      (DirectStageSemantics.ContextualReusedStageComputations.toReused
                        stage)))
                  parentHash.hashEncoded
                  data.ce
                  (ParentOpeningAuthorization.StatementEncodesByCommitment
                    commitmentOfParent))
              (initial := initial)
              steps
              authority
              image)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts VerifyPrior)
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority :=
            PriorProof)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectTerminalSoundness.AccumulatorStep
            (params := params)
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
              (DirectStageSemantics.ReusedStageComputations.toVerified
                (DirectStageSemantics.ContextualReusedStageComputations.toReused
                  stage)))
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
              (DirectStageSemantics.ReusedStageComputations.toVerified
                (DirectStageSemantics.ContextualReusedStageComputations.toReused
                  stage)))
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
        (DirectTerminalSoundness.AccumulatorStep
          (params := params)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (DirectTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage = altNext ∧
      nextImage.step = priorSteps + 1 ∧
      initial.vkDigest = nextImage.vkDigest ∧
      initial.initialBoundary = nextImage.initialBoundary ∧
      Construction2DirectFPrime.WellFormed nextImage :=
  compressed_prior_soundness
    parentHash
    data
    stage
    hInitialStep
    hInitialWellFormed
    hRed
    hMsis
    (CompressedFPrimeAuthority.verifier_sound_of_opens_to_folded_authority
      hOpens)
    hAccepted
    hAlt

/--
Compressed-prior terminal soundness through a verifier object that carries its
own opening theorem to proof-carrying folded F' authority.
-/
theorem compressed_prior_soundness_of_sound_verifier
    {Digest Boundary : Type}
    {PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (parentHash : ParentCEBHashBinding.ParentCEBHash Digest)
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : DirectStageSemantics.ContextualReusedStageComputations Digest n)
    {computeBoundary : Nat → Boundary → Boundary}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (DirectTerminalSoundness.AccHandle Digest n)}
    (verifier :
      CompressedFPrimeAuthority.SoundVerifier
        (Image :=
          Construction2DirectFPrime.PublicImage
            Digest
            Boundary
            (DirectTerminalSoundness.AccHandle Digest n))
        (Proof := PriorProof)
        (DirectTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
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
          (Authority :=
            PriorProof)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectTerminalSoundness.AccumulatorStep
            (params := params)
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
              (DirectStageSemantics.ReusedStageComputations.toVerified
                (DirectStageSemantics.ContextualReusedStageComputations.toReused
                  stage)))
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
              (DirectStageSemantics.ReusedStageComputations.toVerified
                (DirectStageSemantics.ContextualReusedStageComputations.toReused
                  stage)))
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
        (DirectTerminalSoundness.AccumulatorStep
          (params := params)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (DirectTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage = altNext ∧
      nextImage.step = priorSteps + 1 ∧
      initial.vkDigest = nextImage.vkDigest ∧
      initial.initialBoundary = nextImage.initialBoundary ∧
      Construction2DirectFPrime.WellFormed nextImage :=
  compressed_prior_soundness_of_opens_to_folded_authority
    parentHash
    data
    stage
    hInitialStep
    hInitialWellFormed
    hRed
    hMsis
    verifier.opensToFoldedAuthority
    hAccepted
    hAlt

/--
Proof-carrying terminal soundness for the reduced parent-handle path.

This is the non-compressed induction-authority variant. It is useful as the
theorem-level reference while the concrete compressed verifier soundness theorem
is instantiated.
-/
theorem proof_carrying_soundness
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (parentHash : ParentCEBHashBinding.ParentCEBHash Digest)
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : DirectStageSemantics.ContextualReusedStageComputations Digest n)
    {computeBoundary : Nat → Boundary → Boundary}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {initial priorImage nextImage altNext :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (DirectTerminalSoundness.AccHandle Digest n)}
    {priorSteps : Nat}
    {priorAuthority :
      DirectTerminalSoundness.Authority
        (params := params)
        (DirectProgramStep.ComputedBoundaryStep computeBoundary)
        (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
          (DirectStageSemantics.ReusedStageComputations.toVerified
            (DirectStageSemantics.ContextualReusedStageComputations.toReused
              stage)))
        (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
          (DirectStageSemantics.ReusedStageComputations.toVerified
            (DirectStageSemantics.ContextualReusedStageComputations.toReused
              stage)))
        parentHash.hashEncoded
        data.ce
        (ParentOpeningAuthorization.StatementEncodesByCommitment
          commitmentOfParent)
        initial}
    {proof : Unit}
    (hInitialStep : initial.step = 0)
    (hInitialWellFormed : Construction2DirectFPrime.WellFormed initial)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (FoldedFPrimeAuthority.Accepts
          (Transition :=
            DirectTerminalSoundness.Transition
              (params := params)
              (DirectProgramStep.ComputedBoundaryStep computeBoundary)
              (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
                (DirectStageSemantics.ReusedStageComputations.toVerified
                  (DirectStageSemantics.ContextualReusedStageComputations.toReused
                    stage)))
              (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
                (DirectStageSemantics.ReusedStageComputations.toVerified
                  (DirectStageSemantics.ContextualReusedStageComputations.toReused
                    stage)))
              parentHash.hashEncoded
              data.ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent))
          (initial := initial))
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority :=
            DirectTerminalSoundness.Authority
              (params := params)
              (DirectProgramStep.ComputedBoundaryStep computeBoundary)
              (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
                (DirectStageSemantics.ReusedStageComputations.toVerified
                  (DirectStageSemantics.ContextualReusedStageComputations.toReused
                    stage)))
              (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
                (DirectStageSemantics.ReusedStageComputations.toVerified
                  (DirectStageSemantics.ContextualReusedStageComputations.toReused
                    stage)))
              parentHash.hashEncoded
              data.ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent)
              initial)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectTerminalSoundness.AccumulatorStep
            (params := params)
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
              (DirectStageSemantics.ReusedStageComputations.toVerified
                (DirectStageSemantics.ContextualReusedStageComputations.toReused
                  stage)))
            (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
              (DirectStageSemantics.ReusedStageComputations.toVerified
                (DirectStageSemantics.ContextualReusedStageComputations.toReused
                  stage)))
            parentHash.hashEncoded
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
        (DirectTerminalSoundness.AccumulatorStep
          (params := params)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        priorSteps
        priorImage
        altNext) :
    FPrimeInduction.Reachable
        (DirectTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
            (DirectStageSemantics.ReusedStageComputations.toVerified
              (DirectStageSemantics.ContextualReusedStageComputations.toReused
                stage)))
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage = altNext ∧
      nextImage.step = priorSteps + 1 ∧
      initial.vkDigest = nextImage.vkDigest ∧
      initial.initialBoundary = nextImage.initialBoundary ∧
      Construction2DirectFPrime.WellFormed nextImage :=
  prior_authority_soundness
    parentHash
    data
    stage
    hInitialStep
    hInitialWellFormed
    hRed
    hMsis
    FoldedFPrimeAuthority.accepts_sound
    hAccepted
    hAlt

end ReducedHandleTerminal

end DirectCcsFPrime
