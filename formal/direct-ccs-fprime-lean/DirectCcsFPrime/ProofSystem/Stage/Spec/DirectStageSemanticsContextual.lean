import DirectCcsFPrime.ProofSystem.Stage.Spec.DirectStageSemantics
import DirectCcsFPrime.Commitment.Parent.Security.ParentCEBHashBinding

/-!
Context-carried terminal entry points for direct CCS F'.

This module owns only the thin terminal wrappers for
`ContextualReusedStageComputations`. The core stage semantics stay in
`DirectStageSemantics`.
-/

namespace DirectCcsFPrime

namespace DirectStageSemantics

/--
Terminal theorem for context-carried reused stages with an arbitrary sound
prior-authority predicate.

This is the exact Construction-2 induction boundary: the latest direct F' step
may be checked at the terminal boundary only when the accepted prior authority
implies reachability under the same direct F' transition.
-/
theorem terminal_soundness_of_contextual_reused_stage_program_and_msis_of_prior_authority_sound_with_public_image_invariants
    {Digest Boundary Authority : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : ContextualReusedStageComputations Digest n)
    {hashEncoded : List Nat → Digest}
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
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (hAuthority :
      FPrimeInduction.PriorAuthoritySound
        (DirectTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (VerifiedStageComputations.VerifiedPiCCS
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
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
          (Authority :=
            Authority)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectTerminalSoundness.AccumulatorStep
            (params := params)
            (VerifiedStageComputations.VerifiedPiCCS
              (ReusedStageComputations.toVerified
                (ContextualReusedStageComputations.toReused stage)))
            (VerifiedStageComputations.VerifiedPiRLC
              (ReusedStageComputations.toVerified
                (ContextualReusedStageComputations.toReused stage)))
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
        (DirectTerminalSoundness.AccumulatorStep
          (params := params)
          (VerifiedStageComputations.VerifiedPiCCS
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          hashEncoded
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
          (VerifiedStageComputations.VerifiedPiCCS
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          hashEncoded
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
      Construction2DirectFPrime.WellFormed nextImage := by
  let hSound :=
    terminal_soundness_of_verified_stage_program_and_msis_of_prior_authority_sound
      data
      (ReusedStageComputations.toVerified
        (ContextualReusedStageComputations.toReused stage))
      hDigest
      hRed
      hMsis
      hAuthority
      hAccepted
      hAlt
  let hPublic :=
    Construction2DirectFPrime.terminal_direct_fprime_public_image_invariants
      hInitialStep
      hInitialWellFormed
      hAuthority
      hAccepted
  rcases hSound with ⟨hReach, hUnique⟩
  rcases hPublic with ⟨hStep, hVk, hInitialBoundary, hWellFormed⟩
  exact
    ⟨hReach, hUnique, hStep, hVk, hInitialBoundary, hWellFormed⟩

/--
Compressed-prior terminal theorem for context-carried reused stages.

This is the strict direct-stage entry point: the frontend supplies stage
computations whose accepted Pi_CCS/Pi_RLC contexts are carried by the computed
objects themselves, so the output/source bridge predicates are definitional
equalities.
-/
theorem terminal_soundness_of_contextual_reused_stage_program_and_msis_of_compressed_prior_verifier_sound_with_public_image_invariants
    {Digest Boundary : Type}
    {PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : ContextualReusedStageComputations Digest n)
    {hashEncoded : List Nat → Digest}
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
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (hPriorVerifier :
      CompressedFPrimeAuthority.VerifierSound
        (DirectTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (VerifiedStageComputations.VerifiedPiCCS
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          hashEncoded
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
            (VerifiedStageComputations.VerifiedPiCCS
              (ReusedStageComputations.toVerified
                (ContextualReusedStageComputations.toReused stage)))
            (VerifiedStageComputations.VerifiedPiRLC
              (ReusedStageComputations.toVerified
                (ContextualReusedStageComputations.toReused stage)))
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
        (DirectTerminalSoundness.AccumulatorStep
          (params := params)
          (VerifiedStageComputations.VerifiedPiCCS
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          hashEncoded
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
          (VerifiedStageComputations.VerifiedPiCCS
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          hashEncoded
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
  terminal_soundness_of_reused_stage_program_and_msis_of_compressed_prior_verifier_sound_with_public_image_invariants
    data
    (ContextualReusedStageComputations.toReused stage)
    hInitialStep
    hInitialWellFormed
    hDigest
    hRed
    hMsis
    hPriorVerifier
    hAccepted
    hAlt

/--
Context-carried reused-stage terminal theorem through a sound compressed prior
verifier object.

This is the strict production entry point for the direct stage layer: the
compressed prior proof system is accepted only through a `SoundVerifier`, whose
accepted proofs open to proof-carrying folded F' authority.
-/
theorem terminal_soundness_of_contextual_reused_stage_program_and_msis_of_sound_verifier_with_public_image_invariants
    {Digest Boundary : Type}
    {PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : ContextualReusedStageComputations Digest n)
    {hashEncoded : List Nat → Digest}
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
          (VerifiedStageComputations.VerifiedPiCCS
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
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
          (Authority :=
            PriorProof)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectTerminalSoundness.AccumulatorStep
            (params := params)
            (VerifiedStageComputations.VerifiedPiCCS
              (ReusedStageComputations.toVerified
                (ContextualReusedStageComputations.toReused stage)))
            (VerifiedStageComputations.VerifiedPiRLC
              (ReusedStageComputations.toVerified
                (ContextualReusedStageComputations.toReused stage)))
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
        (DirectTerminalSoundness.AccumulatorStep
          (params := params)
          (VerifiedStageComputations.VerifiedPiCCS
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          hashEncoded
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
          (VerifiedStageComputations.VerifiedPiCCS
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          hashEncoded
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
  terminal_soundness_of_contextual_reused_stage_program_and_msis_of_compressed_prior_verifier_sound_with_public_image_invariants
    data
    stage
    hInitialStep
    hInitialWellFormed
    hDigest
    hRed
    hMsis
    (CompressedFPrimeAuthority.verifier_sound_of_sound_verifier verifier)
    hAccepted
    hAlt

/--
Context-carried reused-stage terminal theorem through the exact parent `CE(B)`
hash-binding object and a sound compressed prior verifier.

This is the narrow production theorem surface for the reduced-handle direct CCS
path: the parent hash assumption is supplied only as `ParentCEBHash`, and the
prior compressed verifier is supplied only as `SoundVerifier`.
-/
theorem terminal_soundness_of_contextual_reused_stage_program_and_msis_of_parent_hash_and_sound_verifier_with_public_image_invariants
    {Digest Boundary : Type}
    {PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (parentHash : ParentCEBHashBinding.ParentCEBHash Digest)
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : ContextualReusedStageComputations Digest n)
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
          (VerifiedStageComputations.VerifiedPiCCS
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
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
            (VerifiedStageComputations.VerifiedPiCCS
              (ReusedStageComputations.toVerified
                (ContextualReusedStageComputations.toReused stage)))
            (VerifiedStageComputations.VerifiedPiRLC
              (ReusedStageComputations.toVerified
                (ContextualReusedStageComputations.toReused stage)))
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
          (VerifiedStageComputations.VerifiedPiCCS
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
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
          (VerifiedStageComputations.VerifiedPiCCS
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
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
  terminal_soundness_of_contextual_reused_stage_program_and_msis_of_sound_verifier_with_public_image_invariants
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
Proof-carrying terminal theorem for context-carried reused stages.

This is the non-compressed counterpart of the strict direct-stage entry point:
the prior authority carries reachability evidence, and the latest Pi_CCS/Pi_RLC
contexts are carried by the computed stage data.
-/
theorem terminal_soundness_of_contextual_reused_stage_program_and_msis_with_public_image_invariants
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : ContextualReusedStageComputations Digest n)
    {hashEncoded : List Nat → Digest}
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
        (VerifiedStageComputations.VerifiedPiCCS
          (ReusedStageComputations.toVerified
            (ContextualReusedStageComputations.toReused stage)))
        (VerifiedStageComputations.VerifiedPiRLC
          (ReusedStageComputations.toVerified
            (ContextualReusedStageComputations.toReused stage)))
        hashEncoded
        data.ce
        (ParentOpeningAuthorization.StatementEncodesByCommitment
          commitmentOfParent)
        initial}
    {proof : Unit}
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
        (FoldedFPrimeAuthority.Accepts
          (Transition :=
            DirectTerminalSoundness.Transition
              (params := params)
              (DirectProgramStep.ComputedBoundaryStep computeBoundary)
              (VerifiedStageComputations.VerifiedPiCCS
                (ReusedStageComputations.toVerified
                  (ContextualReusedStageComputations.toReused stage)))
              (VerifiedStageComputations.VerifiedPiRLC
                (ReusedStageComputations.toVerified
                  (ContextualReusedStageComputations.toReused stage)))
              hashEncoded
              data.ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent))
          (initial := initial))
        (Construction2DirectFPrime.VerifyLatestStep
          (Authority :=
            DirectTerminalSoundness.Authority
              (params := params)
              (DirectProgramStep.ComputedBoundaryStep computeBoundary)
              (VerifiedStageComputations.VerifiedPiCCS
                (ReusedStageComputations.toVerified
                  (ContextualReusedStageComputations.toReused stage)))
              (VerifiedStageComputations.VerifiedPiRLC
                (ReusedStageComputations.toVerified
                  (ContextualReusedStageComputations.toReused stage)))
              hashEncoded
              data.ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent)
              initial)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectTerminalSoundness.AccumulatorStep
            (params := params)
            (VerifiedStageComputations.VerifiedPiCCS
              (ReusedStageComputations.toVerified
                (ContextualReusedStageComputations.toReused stage)))
            (VerifiedStageComputations.VerifiedPiRLC
              (ReusedStageComputations.toVerified
                (ContextualReusedStageComputations.toReused stage)))
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
        (DirectTerminalSoundness.AccumulatorStep
          (params := params)
          (VerifiedStageComputations.VerifiedPiCCS
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          hashEncoded
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
          (VerifiedStageComputations.VerifiedPiCCS
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified
              (ContextualReusedStageComputations.toReused stage)))
          hashEncoded
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
  terminal_soundness_of_reused_stage_program_and_msis_with_public_image_invariants
    data
    (ContextualReusedStageComputations.toReused stage)
    hInitialStep
    hInitialWellFormed
    hDigest
    hRed
    hMsis
    hAccepted
    hAlt

end DirectStageSemantics

end DirectCcsFPrime
