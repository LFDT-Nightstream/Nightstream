import DirectCcsFPrime.DirectParentOnlyStageSemantics

/-!
Production-facing soundness surface for the parent-only direct CCS F' path.

This module owns the compact theorem context for the optimized terminal path.
It does not add a new hash model or a new protocol abstraction. It packages the
real implementation obligations that remain after the parent-only theorem:
canonical concrete CE data, contextual reused SuperNeo stages, a Poseidon2
parent `CE(B)` binding object, deterministic boundary and parent-commitment
functions, MSIS assumptions, and a sound compressed prior verifier.
-/

namespace DirectCcsFPrime

namespace DirectParentOnlyProductionSoundness

/-- Public accumulator handle for the optimized parent-only terminal path. -/
abbrev AccHandle (Digest : Type) :=
  DirectParentOnlyTerminalSoundness.AccHandle Digest

/-- Public image for the optimized parent-only terminal path. -/
abbrev PublicImage (Digest Boundary : Type) :=
  Construction2DirectFPrime.PublicImage
    Digest
    Boundary
    (AccHandle Digest)

/--
Static verifier context for one optimized parent-only terminal instance.

The fields are deliberately the real proof requirements. The context does not
claim that Poseidon2 is implemented in Lean; it requires the implementation
hash object whose only theorem obligation is canonical parent-encoding binding.
The stage field is the contextual reused SuperNeo package, not bare stage
relations.
-/
structure Context
    (Digest Boundary : Type)
    (n : Nat)
    (params : SuperNeo.ProofSystem.AjtaiParams) where
  parentHash : Poseidon2ParentCEBHash.Hash Digest
  data : DirectConcreteInstantiation.ConcreteCEData n params
  stage : DirectStageSemantics.ContextualReusedStageComputations Digest n
  computeBoundary : Nat → Boundary → Boundary
  commitmentOfParent :
    ParentEncoding.SomeParentCEB →
      SuperNeo.ProofSystem.Commitment
  initial : PublicImage Digest Boundary
  initialStep : initial.step = 0
  initialWellFormed : Construction2DirectFPrime.WellFormed initial
  msisReduction : SuperNeo.ProofSystem.MSISToAjtaiReductions params
  msisHardness : SuperNeo.ProofSystem.MSISHardnessAssumption params

/-- Verified stage object induced by the context's contextual reused stages. -/
abbrev verifiedStage
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params) :
    DirectStageSemantics.VerifiedStageComputations
      Digest
      DirectStageSemantics.ContextualPiCCSOut
      n :=
  DirectParentOnlyStageSemantics.verifiedStageOfContextual ctx.stage

/-- Parent-only accumulator step fixed by a production context. -/
abbrev AccumulatorStep
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params) :
    Nat →
      AccHandle Digest →
      AccHandle Digest →
        Prop :=
  DirectParentOnlyTerminalSoundness.AccumulatorStep
    (params := params)
    (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
      (verifiedStage ctx))
    (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
      (verifiedStage ctx))
    ctx.parentHash.hashEncoded
    ctx.data.ce
    (ParentOpeningAuthorization.StatementEncodesByCommitment
      ctx.commitmentOfParent)

/-- Construction-2 transition fixed by a production context. -/
abbrev Transition
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params) :
    Nat →
      PublicImage Digest Boundary →
      PublicImage Digest Boundary →
        Prop :=
  DirectParentOnlyTerminalSoundness.Transition
    (params := params)
    (DirectProgramStep.ComputedBoundaryStep ctx.computeBoundary)
    (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
      (verifiedStage ctx))
    (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
      (verifiedStage ctx))
    ctx.parentHash.hashEncoded
    ctx.data.ce
    (ParentOpeningAuthorization.StatementEncodesByCommitment
      ctx.commitmentOfParent)

/-- Latest-step verifier fixed by a production context. -/
abbrev VerifyLatestStep
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params) :
    Nat →
      PriorProof →
      PublicImage Digest Boundary →
      PublicImage Digest Boundary →
      Unit →
        Prop :=
  Construction2DirectFPrime.VerifyLatestStep
    (Authority := PriorProof)
    (DirectProgramStep.ComputedBoundaryStep ctx.computeBoundary)
    (AccumulatorStep ctx)

/-- Sound compressed prior verifier for the context's exact transition. -/
abbrev SoundPriorVerifier
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params) :=
  CompressedFPrimeAuthority.SoundVerifier
    (Image := PublicImage Digest Boundary)
    (Proof := PriorProof)
    (Transition ctx)
    ctx.initial

/--
Proof-carrying prior proof for the context's exact transition.

This is the non-compressed baseline: the prior proof itself carries folded `F'`
reachability. Concrete compressed verifiers must prove that accepted proofs open
to this authority shape.
-/
abbrev ProofCarryingPriorProof
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params) :=
  FoldedFPrimeAuthority.Authority
    (Transition ctx)
    ctx.initial

/--
Sound prior verifier induced by proof-carrying folded `F'` authority.

This supplies the canonical baseline instance of `SoundPriorVerifier`: accepting
a prior proof is exactly accepting the proof-carrying folded authority for the
same `(steps, image)`.
-/
def proofCarryingPriorVerifier
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params) :
    SoundPriorVerifier
      (PriorProof := ProofCarryingPriorProof ctx)
      ctx where
  verify := fun steps proof image =>
    FoldedFPrimeAuthority.Accepts
      (Transition := Transition ctx)
      (initial := ctx.initial)
      steps
      proof
      image
  opensToFoldedAuthority := by
    intro steps proof image hVerify
    exact ⟨proof, hVerify⟩

/-- A proof-carrying prior proof is accepted for its own `(steps, image)`. -/
theorem proofCarryingPriorVerifier_accepts
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    (proof : ProofCarryingPriorProof ctx) :
    (proofCarryingPriorVerifier ctx).verify
      proof.steps
      proof
      proof.image := by
  exact ⟨rfl, rfl⟩

/--
Opening obligation for a concrete compressed prior verifier.

This is the implementation-facing compressed-prior requirement: every accepted
opaque prior proof must open to proof-carrying folded `F'` authority for the
same `(steps, image)` under the context's exact transition and initial image.
-/
def OpensToProofCarryingPriorAuthority
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    (VerifyPrior :
      Nat →
        PriorProof →
        PublicImage Digest Boundary →
          Prop) : Prop :=
  ∀ steps proof image,
    VerifyPrior steps proof image →
      ∃ authority : ProofCarryingPriorProof ctx,
        FoldedFPrimeAuthority.Accepts
          (Transition := Transition ctx)
          (initial := ctx.initial)
          steps
          authority
          image

/--
Build the production sound prior-verifier object from a concrete verifier and
its opening theorem.
-/
def soundPriorVerifier_of_opens_to_proof_carrying_prior_authority
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    (VerifyPrior :
      Nat →
        PriorProof →
        PublicImage Digest Boundary →
          Prop)
    (hOpens :
      OpensToProofCarryingPriorAuthority ctx VerifyPrior) :
    SoundPriorVerifier (PriorProof := PriorProof) ctx where
  verify := VerifyPrior
  opensToFoldedAuthority := hOpens

/--
A concrete prior verifier with the opening theorem reaches every prior image it
accepts.
-/
theorem prior_verifier_reaches_prior_of_opening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        PublicImage Digest Boundary →
          Prop}
    (hOpens :
      OpensToProofCarryingPriorAuthority ctx VerifyPrior)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : VerifyPrior steps proof image) :
    FPrimeInduction.Reachable
      (Transition ctx)
      ctx.initial
      steps
      image :=
  CompressedFPrimeAuthority.verifier_sound_of_opens_to_folded_authority
    hOpens
    steps
    proof
    image
    hVerify

/--
A concrete prior verifier with the opening theorem cannot accept an unreachable
prior image.
-/
theorem prior_verifier_cannot_accept_unreachable_of_opening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        PublicImage Digest Boundary →
          Prop}
    (hOpens :
      OpensToProofCarryingPriorAuthority ctx VerifyPrior)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : VerifyPrior steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (Transition ctx)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (prior_verifier_reaches_prior_of_opening
      ctx
      hOpens
      hVerify)

/--
Any accepted sound prior proof reaches the claimed prior image.

This is the compressed-verifier obligation specialized to the production
context. A digest-only or self-consistent prior proof cannot satisfy
`SoundPriorVerifier` if it accepts an unreachable prior image.
-/
theorem soundPriorVerifier_reaches_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    (verifier : SoundPriorVerifier (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : verifier.verify steps proof image) :
    FPrimeInduction.Reachable
      (Transition ctx)
      ctx.initial
      steps
      image :=
  CompressedFPrimeAuthority.verifier_sound_of_sound_verifier
    verifier
    steps
    proof
    image
    hVerify

/-- A sound prior verifier cannot accept an unreachable prior image. -/
theorem soundPriorVerifier_cannot_accept_unreachable_prior
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    (verifier : SoundPriorVerifier (PriorProof := PriorProof) ctx)
    {steps : Nat}
    {proof : PriorProof}
    {image : PublicImage Digest Boundary}
    (hVerify : verifier.verify steps proof image)
    (hUnreachable :
      ¬ FPrimeInduction.Reachable
        (Transition ctx)
        ctx.initial
        steps
        image) :
    False :=
  hUnreachable
    (soundPriorVerifier_reaches_prior
      ctx
      verifier
      hVerify)

/-- Terminal acceptance for the context's exact prior and latest verifiers. -/
def AcceptedTerminal
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    (verifier : SoundPriorVerifier (PriorProof := PriorProof) ctx)
    (priorSteps : Nat)
    (priorProof : PriorProof)
    (priorImage nextImage : PublicImage Digest Boundary)
    (latestProof : Unit) : Prop :=
  FPrimeInduction.TerminalCompressionAccepted
    (CompressedFPrimeAuthority.SoundVerifier.Accepts verifier)
    (VerifyLatestStep ctx)
    priorSteps
    priorProof
    priorImage
    nextImage
    latestProof

/-- Terminal acceptance with proof-carrying folded prior authority. -/
def AcceptedProofCarryingTerminal
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    (priorSteps : Nat)
    (priorProof : ProofCarryingPriorProof ctx)
    (priorImage nextImage : PublicImage Digest Boundary)
    (latestProof : Unit) : Prop :=
  AcceptedTerminal
    ctx
    (proofCarryingPriorVerifier ctx)
    priorSteps
    priorProof
    priorImage
    nextImage
    latestProof

/--
Alternate latest transition checked against the same prior image.

This is the adversarial comparison point: an alternate accepted latest step may
choose different private advice syntactically, but it must satisfy the same
context-fixed transition relation.
-/
def AlternateLatestStep
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage altNext : PublicImage Digest Boundary) : Prop :=
  Construction2DirectFPrime.Transition
    (DirectProgramStep.ComputedBoundaryStep ctx.computeBoundary)
    (AccumulatorStep ctx)
    priorSteps
    priorImage
    altNext

/--
Soundness conclusion for the optimized parent-only terminal path.

The existential child table is the important non-public witness: both the
accepted latest step and the alternate latest step are forced to use the same
pointwise-authorized private DEC children in the `Pi_CCS -> Pi_RLC` stage path.
-/
def TerminalSoundness
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    (priorSteps : Nat)
    (priorImage nextImage altNext : PublicImage Digest Boundary) : Prop :=
  FPrimeInduction.Reachable
      (Transition ctx)
      ctx.initial
      (priorSteps + 1)
      nextImage ∧
    nextImage = altNext ∧
    nextImage.accumulator.parentSource =
      altNext.accumulator.parentSource ∧
    (∃ priorInputs,
      ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := ctx.parentHash.hashEncoded)
        (params := params)
        (ce := ctx.data.ce)
        (StatementEncodes :=
          ParentOpeningAuthorization.StatementEncodesByCommitment
            ctx.commitmentOfParent)
        priorImage.accumulator.parentSource
        priorInputs ∧
      ParentOnlyAccumulatorStep.ParentSourceFromPiStages
        (n := n)
        (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
          (verifiedStage ctx))
        (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
          (verifiedStage ctx))
        priorSteps
        priorImage.accumulator
        priorInputs
        nextImage.accumulator.parentSource ∧
      ParentOnlyAccumulatorStep.ParentSourceFromPiStages
        (n := n)
        (DirectStageSemantics.VerifiedStageComputations.VerifiedPiCCS
          (verifiedStage ctx))
        (DirectStageSemantics.VerifiedStageComputations.VerifiedPiRLC
          (verifiedStage ctx))
        priorSteps
        priorImage.accumulator
        priorInputs
        altNext.accumulator.parentSource) ∧
    nextImage.currentBoundary =
      ctx.computeBoundary priorSteps priorImage.currentBoundary ∧
    altNext.currentBoundary =
      ctx.computeBoundary priorSteps priorImage.currentBoundary ∧
    nextImage.step = priorSteps + 1 ∧
    ctx.initial.vkDigest = nextImage.vkDigest ∧
    ctx.initial.initialBoundary = nextImage.initialBoundary ∧
    Construction2DirectFPrime.WellFormed nextImage

/--
Exact parent source computed by the context's deterministic `Pi_CCS -> Pi_RLC`
stage functions from a pointwise-authorized private child table.
-/
def ComputedParentSource
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    (i : Nat)
    (prior : AccHandle Digest)
    (priorInputs : DecDigitUniqueness.ColumnDigits n) :
    DigestParentBinding.Source Digest :=
  (verifiedStage ctx).computePiRLC
    i
    ((verifiedStage ctx).computePiCCS
      i
      { parentSource := prior.parentSource
        nextPiCCSInputs := priorInputs })

/--
Canonical latest public image computed by the production context from the prior
image and the shared pointwise-authorized private child table.
-/
def ComputedNextImage
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    (i : Nat)
    (priorImage : PublicImage Digest Boundary)
    (priorInputs : DecDigitUniqueness.ColumnDigits n) :
    PublicImage Digest Boundary where
  vkDigest := priorImage.vkDigest
  step := i + 1
  initialBoundary := priorImage.initialBoundary
  currentBoundary :=
    ctx.computeBoundary i priorImage.currentBoundary
  accumulator :=
    { parentSource :=
        ComputedParentSource
          ctx
          i
          priorImage.accumulator
          priorInputs }
  pc := 1

/--
Any latest transition whose parent source is the deterministic one is exactly
the context-computed latest public image.
-/
theorem latest_eq_computed_next_image_of_parent_source
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    {i : Nat}
    {priorImage image : PublicImage Digest Boundary}
    {priorInputs : DecDigitUniqueness.ColumnDigits n}
    (hParentSource :
      image.accumulator.parentSource =
        ComputedParentSource
          ctx
          i
          priorImage.accumulator
          priorInputs)
    (hLatest :
      AlternateLatestStep
        ctx
        i
        priorImage
        image) :
    image =
      ComputedNextImage
        ctx
        i
        priorImage
        priorInputs := by
  rcases hLatest with
    ⟨_hPrior, hNext, hVk, hInitial, _hPriorPc,
      hNextPc, hBoundary, _hAcc⟩
  cases image with
  | mk vk step initialBoundary currentBoundary accumulator pc =>
      cases accumulator with
      | mk parentSource =>
          simp only
            [ComputedNextImage,
              Construction2DirectFPrime.PublicImage.mk.injEq,
              ParentOnlyAccumulatorStep.AccumulatorHandle.mk.injEq]
          exact
            ⟨hVk.symm,
              hNext,
              hInitial.symm,
              hBoundary,
              hParentSource,
              hNextPc⟩

/--
Production-facing optimized terminal soundness.

For any accepted compressed terminal proof and any alternate latest transition
from the same prior image, the final image is reachable, the parent `CE(B)`
source is fixed, public-image invariants are preserved, and the two latest
steps share one pointwise-authorized private DEC child table.
-/
theorem terminal_soundness
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    (verifier : SoundPriorVerifier (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext : PublicImage Digest Boundary}
    (hAccepted :
      AcceptedTerminal
        ctx
        verifier
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    TerminalSoundness
      ctx
      priorSteps
      priorImage
      nextImage
      altNext :=
  by
    have hBase :=
      DirectParentOnlyStageSemantics.terminal_soundness_of_poseidon2_parent_hash_contextual_reused_stage_program_and_msis_of_sound_verifier
        ctx.parentHash
        ctx.data
        ctx.stage
        verifier
        ctx.initialStep
        ctx.initialWellFormed
        ctx.msisReduction
        ctx.msisHardness
        hAccepted
        hAlt
    rcases hBase with
      ⟨hReach,
        hParentSource,
        hShared,
        hStep,
        hVk,
        hInitialBoundary,
        hWellFormed⟩
    exact
      ⟨hReach,
        DirectParentOnlyTerminalSoundness.latest_publicImage_functional_of_parentSource
          hParentSource
          hAccepted.latestAccepted
          hAlt,
        hParentSource,
        hShared,
        DirectProgramStep.latest_currentBoundary_eq_compute
          hAccepted.latestAccepted,
        DirectProgramStep.latest_currentBoundary_eq_compute
          hAlt,
        hStep,
        hVk,
        hInitialBoundary,
        hWellFormed⟩

/--
Production-facing terminal soundness with the exact deterministic parent-source
update exposed.

The accepted latest step and the alternate latest step share one
pointwise-authorized private DEC child table, and both parent sources are the
context's computed `Pi_RLC` source for the `Pi_CCS` output over that table.
-/
theorem terminal_soundness_with_computed_parent_sources
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    (verifier : SoundPriorVerifier (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext : PublicImage Digest Boundary}
    (hAccepted :
      AcceptedTerminal
        ctx
        verifier
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    TerminalSoundness
        ctx
        priorSteps
        priorImage
        nextImage
        altNext ∧
      ∃ priorInputs,
        ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
          (n := n)
          (hashEncoded := ctx.parentHash.hashEncoded)
          (params := params)
          (ce := ctx.data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              ctx.commitmentOfParent)
          priorImage.accumulator.parentSource
          priorInputs ∧
        nextImage.accumulator.parentSource =
          ComputedParentSource
            ctx
            priorSteps
            priorImage.accumulator
            priorInputs ∧
        altNext.accumulator.parentSource =
          ComputedParentSource
            ctx
            priorSteps
            priorImage.accumulator
            priorInputs := by
  have hSound :
      TerminalSoundness
        ctx
        priorSteps
        priorImage
        nextImage
        altNext :=
    terminal_soundness ctx verifier hAccepted hAlt
  rcases hSound with
    ⟨hReach,
      hImageEq,
      hParentSource,
      hShared,
      hNextBoundary,
      hAltBoundary,
      hStep,
      hVk,
      hInitialBoundary,
      hWellFormed⟩
  rcases hShared with ⟨priorInputs, hPointwise, hNextSource, hAltSource⟩
  have hNextComputed :
      nextImage.accumulator.parentSource =
        ComputedParentSource
          ctx
          priorSteps
          priorImage.accumulator
          priorInputs :=
    DirectParentOnlyStageSemantics.parentSourceFrom_verified_stage_eq_compute
      (verifiedStage ctx)
      hNextSource
  have hAltComputed :
      altNext.accumulator.parentSource =
        ComputedParentSource
          ctx
          priorSteps
          priorImage.accumulator
          priorInputs :=
    DirectParentOnlyStageSemantics.parentSourceFrom_verified_stage_eq_compute
      (verifiedStage ctx)
      hAltSource
  exact
    ⟨⟨hReach,
        hImageEq,
        hParentSource,
        ⟨priorInputs, hPointwise, hNextSource, hAltSource⟩,
        hNextBoundary,
        hAltBoundary,
        hStep,
        hVk,
        hInitialBoundary,
        hWellFormed⟩,
      priorInputs,
      hPointwise,
      hNextComputed,
      hAltComputed⟩

/--
Production-facing terminal soundness with the exact deterministic latest public
image exposed.

Both the accepted latest image and any alternate latest image from the same
context equal the record computed from the prior image, the deterministic
boundary update, and the deterministic `Pi_CCS -> Pi_RLC` parent source over
the shared pointwise-authorized private child table.
-/
theorem terminal_soundness_with_computed_next_images
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    (verifier : SoundPriorVerifier (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext : PublicImage Digest Boundary}
    (hAccepted :
      AcceptedTerminal
        ctx
        verifier
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    TerminalSoundness
        ctx
        priorSteps
        priorImage
        nextImage
        altNext ∧
      ∃ priorInputs,
        ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
          (n := n)
          (hashEncoded := ctx.parentHash.hashEncoded)
          (params := params)
          (ce := ctx.data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              ctx.commitmentOfParent)
          priorImage.accumulator.parentSource
          priorInputs ∧
        nextImage =
          ComputedNextImage
            ctx
            priorSteps
            priorImage
            priorInputs ∧
        altNext =
          ComputedNextImage
            ctx
            priorSteps
            priorImage
            priorInputs := by
  rcases
      terminal_soundness_with_computed_parent_sources
        ctx
        verifier
        hAccepted
        hAlt with
    ⟨hSound, priorInputs, hPointwise, hNextComputed, hAltComputed⟩
  exact
    ⟨hSound,
      priorInputs,
      hPointwise,
      latest_eq_computed_next_image_of_parent_source
        ctx
        hNextComputed
        hAccepted.latestAccepted,
      latest_eq_computed_next_image_of_parent_source
        ctx
        hAltComputed
        hAlt⟩

/--
In a production context, pointwise private-DEC requirements are unique for one
compact parent source.

This is the context-level anti-substitution theorem: assuming Poseidon2 parent
encoding binding, MSIS-backed Ajtai binding, and the concrete CE opening data
from `ctx`, an attacker cannot present a different pointwise-valid private DEC
child table for the same parent source.
-/
theorem pointwise_private_dec_inputs_unique_of_context
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    {source : DigestParentBinding.Source Digest}
    {inputsA inputsB : DecDigitUniqueness.ColumnDigits n}
    (hA :
      ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := ctx.parentHash.hashEncoded)
        (params := params)
        (ce := ctx.data.ce)
        (StatementEncodes :=
          ParentOpeningAuthorization.StatementEncodesByCommitment
            ctx.commitmentOfParent)
        source
        inputsA)
    (hB :
      ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
        (n := n)
        (hashEncoded := ctx.parentHash.hashEncoded)
        (params := params)
        (ce := ctx.data.ce)
        (StatementEncodes :=
          ParentOpeningAuthorization.StatementEncodesByCommitment
            ctx.commitmentOfParent)
        source
        inputsB) :
    inputsA = inputsB :=
  ParentOnlyAccumulatorStep.pointwise_private_dec_requirements_functional_of_statementCommitment_and_ajtaiCEOpening
    (Poseidon2ParentCEBHash.encodedParentCEBDigestBinding ctx.parentHash)
    (AjtaiResidueBinding.noAjtaiBindingCollision_of_msis
      ctx.msisReduction
      ctx.msisHardness)
    (AjtaiResidueBinding.ceOpeningAdapter_of_assignmentOpeningAdapter
      (AjtaiResidueBinding.assignmentOpeningAdapter_of_ajtaiBackedCommitMap
        ctx.data.ajtaiBackedCommitMap))
    hA
    hB

/--
Production-facing terminal soundness with a unique pointwise private child
table.

The computed-image theorem already extracts one private child table shared by
the accepted and alternate latest steps. This theorem also proves that any
other pointwise-valid private DEC child table for the same parent source is
definitionally the same table.
-/
theorem terminal_soundness_with_unique_pointwise_children
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    (verifier : SoundPriorVerifier (PriorProof := PriorProof) ctx)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext : PublicImage Digest Boundary}
    (hAccepted :
      AcceptedTerminal
        ctx
        verifier
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    TerminalSoundness
        ctx
        priorSteps
        priorImage
        nextImage
        altNext ∧
      ∃ priorInputs,
        ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
          (n := n)
          (hashEncoded := ctx.parentHash.hashEncoded)
          (params := params)
          (ce := ctx.data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              ctx.commitmentOfParent)
          priorImage.accumulator.parentSource
          priorInputs ∧
        nextImage =
          ComputedNextImage
            ctx
            priorSteps
            priorImage
            priorInputs ∧
        altNext =
          ComputedNextImage
            ctx
            priorSteps
            priorImage
            priorInputs ∧
        (∀ otherInputs,
          ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
            (n := n)
            (hashEncoded := ctx.parentHash.hashEncoded)
            (params := params)
            (ce := ctx.data.ce)
            (StatementEncodes :=
              ParentOpeningAuthorization.StatementEncodesByCommitment
                ctx.commitmentOfParent)
            priorImage.accumulator.parentSource
            otherInputs →
              otherInputs = priorInputs) := by
  rcases
      terminal_soundness_with_computed_next_images
        ctx
        verifier
        hAccepted
        hAlt with
    ⟨hSound, priorInputs, hPointwise, hNextComputed, hAltComputed⟩
  refine
    ⟨hSound,
      priorInputs,
      hPointwise,
      hNextComputed,
      hAltComputed,
      ?_⟩
  intro otherInputs hOther
  exact
    pointwise_private_dec_inputs_unique_of_context
      ctx
      hOther
      hPointwise

/--
Production-facing terminal soundness with proof-carrying folded prior authority.

This closes the theorem-level baseline for the optimized path. The compressed
production path must instantiate `SoundPriorVerifier`; the proof-carrying path
gets that verifier canonically from folded reachability evidence.
-/
theorem terminal_soundness_of_proof_carrying_prior_authority
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    {priorSteps : Nat}
    {priorProof : ProofCarryingPriorProof ctx}
    {latestProof : Unit}
    {priorImage nextImage altNext : PublicImage Digest Boundary}
    (hAccepted :
      AcceptedProofCarryingTerminal
        ctx
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    TerminalSoundness
      ctx
      priorSteps
      priorImage
      nextImage
      altNext :=
  terminal_soundness
    ctx
    (proofCarryingPriorVerifier ctx)
    hAccepted
    hAlt

/--
Proof-carrying folded-prior variant with the exact deterministic parent-source
update exposed.
-/
theorem terminal_soundness_with_computed_parent_sources_of_proof_carrying_prior_authority
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    {priorSteps : Nat}
    {priorProof : ProofCarryingPriorProof ctx}
    {latestProof : Unit}
    {priorImage nextImage altNext : PublicImage Digest Boundary}
    (hAccepted :
      AcceptedProofCarryingTerminal
        ctx
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    TerminalSoundness
        ctx
        priorSteps
        priorImage
        nextImage
        altNext ∧
      ∃ priorInputs,
        ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
          (n := n)
          (hashEncoded := ctx.parentHash.hashEncoded)
          (params := params)
          (ce := ctx.data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              ctx.commitmentOfParent)
          priorImage.accumulator.parentSource
          priorInputs ∧
        nextImage.accumulator.parentSource =
          ComputedParentSource
            ctx
            priorSteps
            priorImage.accumulator
            priorInputs ∧
        altNext.accumulator.parentSource =
          ComputedParentSource
            ctx
            priorSteps
            priorImage.accumulator
            priorInputs :=
  terminal_soundness_with_computed_parent_sources
    ctx
    (proofCarryingPriorVerifier ctx)
    hAccepted
    hAlt

/--
Proof-carrying folded-prior variant with the exact deterministic latest public
image exposed.
-/
theorem terminal_soundness_with_computed_next_images_of_proof_carrying_prior_authority
    {Digest Boundary : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    {priorSteps : Nat}
    {priorProof : ProofCarryingPriorProof ctx}
    {latestProof : Unit}
    {priorImage nextImage altNext : PublicImage Digest Boundary}
    (hAccepted :
      AcceptedProofCarryingTerminal
        ctx
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    TerminalSoundness
        ctx
        priorSteps
        priorImage
        nextImage
        altNext ∧
      ∃ priorInputs,
        ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
          (n := n)
          (hashEncoded := ctx.parentHash.hashEncoded)
          (params := params)
          (ce := ctx.data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              ctx.commitmentOfParent)
          priorImage.accumulator.parentSource
          priorInputs ∧
        nextImage =
          ComputedNextImage
            ctx
            priorSteps
            priorImage
            priorInputs ∧
        altNext =
          ComputedNextImage
            ctx
            priorSteps
            priorImage
            priorInputs :=
  terminal_soundness_with_computed_next_images
    ctx
    (proofCarryingPriorVerifier ctx)
    hAccepted
    hAlt

/--
Production-facing terminal soundness from a concrete compressed prior verifier
and its opening theorem.

This is the direct implementation adapter for the compressed production path:
callers may keep the prior proof opaque, but they must prove verifier
acceptance opens to proof-carrying folded `F'` authority under this exact
context. The conclusion is the strongest computed-latest-image form.
-/
theorem terminal_soundness_with_computed_next_images_of_prior_verifier_opening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        PublicImage Digest Boundary →
          Prop}
    (hOpens :
      OpensToProofCarryingPriorAuthority ctx VerifyPrior)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext : PublicImage Digest Boundary}
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts VerifyPrior)
        (VerifyLatestStep ctx)
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    TerminalSoundness
        ctx
        priorSteps
        priorImage
        nextImage
        altNext ∧
      ∃ priorInputs,
        ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
          (n := n)
          (hashEncoded := ctx.parentHash.hashEncoded)
          (params := params)
          (ce := ctx.data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              ctx.commitmentOfParent)
          priorImage.accumulator.parentSource
          priorInputs ∧
        nextImage =
          ComputedNextImage
            ctx
            priorSteps
            priorImage
            priorInputs ∧
        altNext =
          ComputedNextImage
            ctx
            priorSteps
            priorImage
            priorInputs := by
  let verifier :
      SoundPriorVerifier (PriorProof := PriorProof) ctx :=
    soundPriorVerifier_of_opens_to_proof_carrying_prior_authority
      ctx
      VerifyPrior
      hOpens
  have hAcceptedSound :
      AcceptedTerminal
        ctx
        verifier
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof := by
    constructor
    · exact hAccepted.priorAccepted
    · exact hAccepted.latestAccepted
  exact
    terminal_soundness_with_computed_next_images
      ctx
      verifier
      hAcceptedSound
      hAlt

/--
Compressed-prior raw-verifier adapter with unique pointwise private children.

This is the strongest production-facing conclusion for the optimized path: a
raw compressed prior verifier may remain opaque, but after its opening theorem
is supplied, the terminal proof reaches the claimed image, computes both latest
images from one deterministic private child table, and rules out every distinct
pointwise-valid private DEC table for the same parent source.
-/
theorem terminal_soundness_with_unique_pointwise_children_of_prior_verifier_opening
    {Digest Boundary PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (ctx : Context Digest Boundary n params)
    {VerifyPrior :
      Nat →
        PriorProof →
        PublicImage Digest Boundary →
          Prop}
    (hOpens :
      OpensToProofCarryingPriorAuthority ctx VerifyPrior)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    {priorImage nextImage altNext : PublicImage Digest Boundary}
    (hAccepted :
      FPrimeInduction.TerminalCompressionAccepted
        (CompressedFPrimeAuthority.Accepts VerifyPrior)
        (VerifyLatestStep ctx)
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof)
    (hAlt :
      AlternateLatestStep
        ctx
        priorSteps
        priorImage
        altNext) :
    TerminalSoundness
        ctx
        priorSteps
        priorImage
        nextImage
        altNext ∧
      ∃ priorInputs,
        ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
          (n := n)
          (hashEncoded := ctx.parentHash.hashEncoded)
          (params := params)
          (ce := ctx.data.ce)
          (StatementEncodes :=
            ParentOpeningAuthorization.StatementEncodesByCommitment
              ctx.commitmentOfParent)
          priorImage.accumulator.parentSource
          priorInputs ∧
        nextImage =
          ComputedNextImage
            ctx
            priorSteps
            priorImage
            priorInputs ∧
        altNext =
          ComputedNextImage
            ctx
            priorSteps
            priorImage
            priorInputs ∧
        (∀ otherInputs,
          ParentOnlyAccumulatorStep.PointwisePrivateDecRequirements
            (n := n)
            (hashEncoded := ctx.parentHash.hashEncoded)
            (params := params)
            (ce := ctx.data.ce)
            (StatementEncodes :=
              ParentOpeningAuthorization.StatementEncodesByCommitment
                ctx.commitmentOfParent)
            priorImage.accumulator.parentSource
            otherInputs →
              otherInputs = priorInputs) := by
  let verifier :
      SoundPriorVerifier (PriorProof := PriorProof) ctx :=
    soundPriorVerifier_of_opens_to_proof_carrying_prior_authority
      ctx
      VerifyPrior
      hOpens
  have hAcceptedSound :
      AcceptedTerminal
        ctx
        verifier
        priorSteps
        priorProof
        priorImage
        nextImage
        latestProof := by
    constructor
    · exact hAccepted.priorAccepted
    · exact hAccepted.latestAccepted
  exact
    terminal_soundness_with_unique_pointwise_children
      ctx
      verifier
      hAcceptedSound
      hAlt

end DirectParentOnlyProductionSoundness

end DirectCcsFPrime
