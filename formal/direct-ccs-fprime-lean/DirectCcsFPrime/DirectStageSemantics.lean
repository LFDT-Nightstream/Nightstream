import DirectCcsFPrime.DirectProgramStep
import DirectCcsFPrime.CompressedFPrimeAuthority
import DirectCcsFPrime.SuperNeoBridge
import SuperNeo.FoldingProtocol.PiCCSInterface
import SuperNeo.FoldingProtocol.PiRLCInterface

/-!
Verified SuperNeo stage computations for direct CCS F'.

This module removes the bare-stage-computation boundary from the strongest
direct terminal theorem. A caller must package `computePiCCS` and `computePiRLC`
with the imported SuperNeo theorem statements and explicit predicates tying
their outputs to the corresponding paper-stage context.

The output/source bridge predicates are frontend-specific because the direct
CCS/R1CS frontend owns the concrete encoding of `PiCCSOut` and the reduced
parent source. The imported SuperNeo facts are not optional: they are part of
the verified stage object.
-/

namespace DirectCcsFPrime

namespace DirectStageSemantics

/--
Verified deterministic stage computations for one direct CCS F' relation.

`computePiCCS` and `computePiRLC` are the deterministic stage outputs consumed
by the direct terminal theorem. The remaining fields bind those computations to
SuperNeo theorem contexts:

* `piCCSStrongStatement_of_compute` gives the imported `Π_CCS` theorem surface
  for the step/prior accumulator.
* `piRLCWeakStatement_of_compute` gives the imported `Π_RLC` theorem surface
  for the step/`Π_CCS` output.
* `piCCSOutputSound` and `piRLCSourceSound` are the frontend's exact bridge from
  its encoded output/source values to those theorem contexts.
-/
structure VerifiedStageComputations
    (Digest PiCCSOut : Type)
    (n : Nat) where
  computePiCCS :
    Nat →
      DirectTerminalSoundness.AccHandle Digest n →
      PiCCSOut
  computePiRLC :
    Nat →
      PiCCSOut →
      DigestParentBinding.Source Digest
  piCCSOutputSound :
    SuperNeo.ProtocolTargetContext →
      PiCCSOut →
      Prop
  piRLCSourceSound :
    SuperNeo.ProtocolTargetContext →
      PiCCSOut →
      DigestParentBinding.Source Digest →
      Prop
  piCCSCtx :
    Nat →
      DirectTerminalSoundness.AccHandle Digest n →
      SuperNeo.ProtocolTargetContext
  piCCSStrongStatement_of_compute :
    ∀ i prior,
      SuperNeo.PiCCSInterface.piCCSStrongStatement
        (piCCSCtx i prior)
  piCCSOutputSound_of_compute :
    ∀ i prior,
      piCCSOutputSound
        (piCCSCtx i prior)
        (computePiCCS i prior)
  piRLCCtx :
    Nat →
      PiCCSOut →
      SuperNeo.ProtocolTargetContext
  piRLCWeakStatement_of_compute :
    ∀ i out,
      SuperNeo.PiRLCInterface.piRLCWeakStatement
        (piRLCCtx i out)
  piRLCSourceSound_of_compute :
    ∀ i out,
      piRLCSourceSound
        (piRLCCtx i out)
        out
        (computePiRLC i out)

/--
Verified stage computations instantiated through the existing SuperNeo
`ceRelation` authority.

This is the direct reuse path for the already-formalized SuperNeo stage layer:
each direct stage carries a `ReusedStageAuthority`, and that authority derives
the imported `Π_CCS`/`Π_RLC` theorem statements used by
`VerifiedStageComputations`.
-/
structure ReusedStageComputations
    (Digest PiCCSOut : Type)
    (n : Nat) where
  computePiCCS :
    Nat →
      DirectTerminalSoundness.AccHandle Digest n →
      PiCCSOut
  computePiRLC :
    Nat →
      PiCCSOut →
      DigestParentBinding.Source Digest
  piCCSOutputSound :
    SuperNeo.ProtocolTargetContext →
      PiCCSOut →
      Prop
  piRLCSourceSound :
    SuperNeo.ProtocolTargetContext →
      PiCCSOut →
      DigestParentBinding.Source Digest →
      Prop
  piCCSCtx :
    Nat →
      DirectTerminalSoundness.AccHandle Digest n →
      SuperNeo.ProtocolTargetContext
  piCCSAuthority :
    ∀ i prior,
      SuperNeoBridge.ReusedStageAuthority
        (piCCSCtx i prior)
  piCCSOutputSound_of_compute :
    ∀ i prior,
      piCCSOutputSound
        (piCCSCtx i prior)
        (computePiCCS i prior)
  piRLCCtx :
    Nat →
      PiCCSOut →
      SuperNeo.ProtocolTargetContext
  piRLCAuthority :
    ∀ i out,
      SuperNeoBridge.ReusedStageAuthority
        (piRLCCtx i out)
  piRLCSourceSound_of_compute :
    ∀ i out,
      piRLCSourceSound
        (piRLCCtx i out)
        out
        (computePiRLC i out)

namespace ReusedStageComputations

/--
Convert a reused-CE-relation stage package into the general verified-stage
package consumed by the terminal theorem.
-/
def toVerified
    {Digest PiCCSOut : Type}
    {n : Nat}
    (stage : ReusedStageComputations Digest PiCCSOut n) :
    VerifiedStageComputations Digest PiCCSOut n where
  computePiCCS := stage.computePiCCS
  computePiRLC := stage.computePiRLC
  piCCSOutputSound := stage.piCCSOutputSound
  piRLCSourceSound := stage.piRLCSourceSound
  piCCSCtx := stage.piCCSCtx
  piCCSStrongStatement_of_compute := by
    intro i prior
    exact SuperNeoBridge.ReusedStageAuthority.piCCSStrong
      (stage.piCCSAuthority i prior)
  piCCSOutputSound_of_compute := stage.piCCSOutputSound_of_compute
  piRLCCtx := stage.piRLCCtx
  piRLCWeakStatement_of_compute := by
    intro i out
    exact SuperNeoBridge.ReusedStageAuthority.piRLCWeak
      (stage.piRLCAuthority i out)
  piRLCSourceSound_of_compute := stage.piRLCSourceSound_of_compute

end ReusedStageComputations

/--
Pi_CCS output shape for frontends that carry the theorem context with the
computed stage output.

The `step` field lets later stage code reconstruct the fold index from the
output object. The context field is the imported SuperNeo target that the
Pi_CCS theorem statement is about.
-/
structure ContextualPiCCSOut where
  step : Nat
  ctx : SuperNeo.ProtocolTargetContext

/--
Reused-stage computations whose bridge data is carried by the computed outputs.

This removes the arbitrary output/source-soundness predicate choice from the
frontend path. The accepted Pi_CCS output is accepted for exactly its carried
context, and the accepted Pi_RLC source is accepted for exactly the context
computed from the carried Pi_CCS output and source.
-/
structure ContextualReusedStageComputations
    (Digest : Type)
    (n : Nat) where
  computePiCCS :
    Nat →
      DirectTerminalSoundness.AccHandle Digest n →
      ContextualPiCCSOut
  computePiCCS_step :
    ∀ i prior,
      (computePiCCS i prior).step = i
  computePiRLC :
    Nat →
      ContextualPiCCSOut →
      DigestParentBinding.Source Digest
  piRLCContext :
    ContextualPiCCSOut →
      DigestParentBinding.Source Digest →
      SuperNeo.ProtocolTargetContext
  piCCSAuthority :
    ∀ i prior,
      SuperNeoBridge.ReusedStageAuthority
        ((computePiCCS i prior).ctx)
  piRLCAuthority :
    ∀ i out,
      SuperNeoBridge.ReusedStageAuthority
        (piRLCContext out (computePiRLC i out))

namespace ContextualReusedStageComputations

/--
Forget the context-carried frontend shape into the general reused-stage package.

The resulting bridge predicates are definitional equalities:

* Pi_CCS output context equals the context carried by the output.
* Pi_RLC source context equals the context computed from the output/source.
-/
def toReused
    {Digest : Type}
    {n : Nat}
    (stage : ContextualReusedStageComputations Digest n) :
    ReusedStageComputations Digest ContextualPiCCSOut n where
  computePiCCS := stage.computePiCCS
  computePiRLC := stage.computePiRLC
  piCCSOutputSound := fun ctx out => ctx = out.ctx
  piRLCSourceSound := fun ctx out source =>
    ctx = stage.piRLCContext out source
  piCCSCtx := fun i prior => (stage.computePiCCS i prior).ctx
  piCCSAuthority := stage.piCCSAuthority
  piCCSOutputSound_of_compute := by
    intro i prior
    rfl
  piRLCCtx := fun i out =>
    stage.piRLCContext out (stage.computePiRLC i out)
  piRLCAuthority := stage.piRLCAuthority
  piRLCSourceSound_of_compute := by
    intro i out
    rfl

end ContextualReusedStageComputations

namespace VerifiedStageComputations

/-- The packaged `Π_CCS` context satisfies the imported strong theorem. -/
theorem piCCSStrong
    {Digest PiCCSOut : Type}
    {n : Nat}
    (stage : VerifiedStageComputations Digest PiCCSOut n)
    (i : Nat)
    (prior : DirectTerminalSoundness.AccHandle Digest n) :
    SuperNeo.PiCCSInterface.piCCSStrongStatement
      (stage.piCCSCtx i prior) :=
  stage.piCCSStrongStatement_of_compute i prior

/-- The packaged `Π_RLC` context satisfies the imported weak theorem. -/
theorem piRLCWeak
    {Digest PiCCSOut : Type}
    {n : Nat}
    (stage : VerifiedStageComputations Digest PiCCSOut n)
    (i : Nat)
    (out : PiCCSOut) :
    SuperNeo.PiRLCInterface.piRLCWeakStatement
      (stage.piRLCCtx i out) :=
  stage.piRLCWeakStatement_of_compute i out

/--
Accepted direct `Π_CCS` output relation for a verified stage package.

Acceptance requires the deterministic output, the frontend output-soundness
bridge, and the imported SuperNeo strong `Π_CCS` theorem for the packaged
context.
-/
def VerifiedPiCCS
    {Digest PiCCSOut : Type}
    {n : Nat}
    (stage : VerifiedStageComputations Digest PiCCSOut n)
    (i : Nat)
    (prior : DirectTerminalSoundness.AccHandle Digest n)
    (out : PiCCSOut) : Prop :=
  out = stage.computePiCCS i prior ∧
    stage.piCCSOutputSound (stage.piCCSCtx i prior) out ∧
    SuperNeo.PiCCSInterface.piCCSStrongStatement
      (stage.piCCSCtx i prior)

/--
Accepted direct `Π_RLC` parent-source relation for a verified stage package.

Acceptance requires the deterministic source, the frontend source-soundness
bridge, and the imported SuperNeo weak `Π_RLC` theorem for the packaged context.
-/
def VerifiedPiRLC
    {Digest PiCCSOut : Type}
    {n : Nat}
    (stage : VerifiedStageComputations Digest PiCCSOut n)
    (i : Nat)
    (out : PiCCSOut)
    (source : DigestParentBinding.Source Digest) : Prop :=
  source = stage.computePiRLC i out ∧
    stage.piRLCSourceSound (stage.piRLCCtx i out) out source ∧
    SuperNeo.PiRLCInterface.piRLCWeakStatement
      (stage.piRLCCtx i out)

/-- The deterministic `Π_CCS` output is accepted by the verified relation. -/
theorem verifiedPiCCS_of_compute
    {Digest PiCCSOut : Type}
    {n : Nat}
    (stage : VerifiedStageComputations Digest PiCCSOut n)
    (i : Nat)
    (prior : DirectTerminalSoundness.AccHandle Digest n) :
    VerifiedPiCCS
      stage
      i
      prior
      (stage.computePiCCS i prior) := by
  exact
    ⟨rfl,
      stage.piCCSOutputSound_of_compute i prior,
      stage.piCCSStrong i prior⟩

/-- The deterministic `Π_RLC` source is accepted by the verified relation. -/
theorem verifiedPiRLC_of_compute
    {Digest PiCCSOut : Type}
    {n : Nat}
    (stage : VerifiedStageComputations Digest PiCCSOut n)
    (i : Nat)
    (out : PiCCSOut) :
    VerifiedPiRLC
      stage
      i
      out
      (stage.computePiRLC i out) := by
  exact
    ⟨rfl,
      stage.piRLCSourceSound_of_compute i out,
      stage.piRLCWeak i out⟩

/-- Verified stage computations induce a functional accepted `Π_CCS` relation. -/
theorem piCCSFunctional
    {Digest PiCCSOut : Type}
    {n : Nat}
    (stage : VerifiedStageComputations Digest PiCCSOut n) :
    ParentSourceStep.PiCCSFunctional
      (VerifiedPiCCS stage) := by
  intro i prior outA outB hA hB
  exact hA.1.trans hB.1.symm

/-- Verified stage computations induce a functional accepted `Π_RLC` relation. -/
theorem piRLCFunctional
    {Digest PiCCSOut : Type}
    {n : Nat}
    (stage : VerifiedStageComputations Digest PiCCSOut n) :
    ParentSourceStep.PiRLCFunctional
      (VerifiedPiRLC stage) := by
  intro i out sourceA sourceB hA hB
  exact hA.1.trans hB.1.symm

end VerifiedStageComputations

/--
Stage-verified direct-program terminal theorem with an arbitrary sound prior
authority.

This is the theorem shape needed by a standalone terminal proof. The prior
folded `F'` authority need not be a Lean proof-carrying object, but its
acceptance predicate must satisfy `PriorAuthoritySound`: accepting it for a
given `(steps, image)` implies actual reachability under the direct `F'`
transition relation.
-/
theorem terminal_soundness_of_verified_stage_program_and_msis_of_prior_authority_sound
    {Digest Boundary PiCCSOut : Type}
    {Authority : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : VerifiedStageComputations Digest PiCCSOut n)
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
          (VerifiedStageComputations.VerifiedPiCCS stage)
          (VerifiedStageComputations.VerifiedPiRLC stage)
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
            (VerifiedStageComputations.VerifiedPiCCS stage)
            (VerifiedStageComputations.VerifiedPiRLC stage)
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
          (VerifiedStageComputations.VerifiedPiCCS stage)
          (VerifiedStageComputations.VerifiedPiRLC stage)
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
          (VerifiedStageComputations.VerifiedPiCCS stage)
          (VerifiedStageComputations.VerifiedPiRLC stage)
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage = altNext := by
  have hReach :
      FPrimeInduction.Reachable
        (DirectTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (VerifiedStageComputations.VerifiedPiCCS stage)
          (VerifiedStageComputations.VerifiedPiRLC stage)
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage :=
    FPrimeInduction.terminal_compression_reaches_final
      hAuthority
      Construction2DirectFPrime.latest_step_sound
      hAccepted
  have hAcc :
      nextImage.accumulator.parentSource =
          altNext.accumulator.parentSource ∧
        nextImage.accumulator.nextPiCCSInputs =
          altNext.accumulator.nextPiCCSInputs :=
    ParentSourceStep.transition_accumulator_fields_functional_of_stages_and_ajtaiCEOpening
      (VerifiedStageComputations.piCCSFunctional stage)
      (VerifiedStageComputations.piRLCFunctional stage)
      hDigest
      ParentOpeningAuthorization.statementEncodesByCommitment_functional
      (AjtaiResidueBinding.noAjtaiBindingCollision_of_msis hRed hMsis)
      (AjtaiResidueBinding.ceOpeningAdapter_of_assignmentOpeningAdapter
        (AjtaiResidueBinding.assignmentOpeningAdapter_of_ajtaiBackedCommitMap
          data.ajtaiBackedCommitMap))
      hAccepted.latestAccepted
      hAlt
  constructor
  · exact hReach
  · exact
      DirectProgramStep.latest_publicImage_functional_of_accumulator_fields
        hAcc
        hAccepted.latestAccepted
        hAlt

/--
Stage-verified direct-program terminal theorem specialized to a compressed
prior-F' verifier.

This is the production-shaped counterpart of the arbitrary-authority theorem.
The prior proof may be opaque, but its verifier must satisfy
`CompressedFPrimeAuthority.VerifierSound`: verifier acceptance implies actual
reachability of the prior public image under the same direct `F'` transition.
-/
theorem terminal_soundness_of_verified_stage_program_and_msis_of_compressed_prior_verifier_sound
    {Digest Boundary PiCCSOut : Type}
    {PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : VerifiedStageComputations Digest PiCCSOut n)
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
          (VerifiedStageComputations.VerifiedPiCCS stage)
          (VerifiedStageComputations.VerifiedPiRLC stage)
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
            (VerifiedStageComputations.VerifiedPiCCS stage)
            (VerifiedStageComputations.VerifiedPiRLC stage)
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
          (VerifiedStageComputations.VerifiedPiCCS stage)
          (VerifiedStageComputations.VerifiedPiRLC stage)
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
          (VerifiedStageComputations.VerifiedPiCCS stage)
          (VerifiedStageComputations.VerifiedPiRLC stage)
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage = altNext :=
  terminal_soundness_of_verified_stage_program_and_msis_of_prior_authority_sound
    data
    stage
    hDigest
    hRed
    hMsis
    (CompressedFPrimeAuthority.accepts_sound_of_verifier_sound
      hPriorVerifier)
    hAccepted
    hAlt

/--
Stage-verified compressed-prior theorem with final public-image invariants.

This is the strict terminal statement for implementations that expose a compact
Construction-2 public image. In addition to reachability and latest-step
uniqueness, the accepted compressed proof forces the public step counter,
verifier-key digest, initial boundary, and fixed `pc = 1` invariant inherited
from the base image.
-/
theorem terminal_soundness_of_verified_stage_program_and_msis_of_compressed_prior_verifier_sound_with_public_image_invariants
    {Digest Boundary PiCCSOut : Type}
    {PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : VerifiedStageComputations Digest PiCCSOut n)
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
          (VerifiedStageComputations.VerifiedPiCCS stage)
          (VerifiedStageComputations.VerifiedPiRLC stage)
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
            (VerifiedStageComputations.VerifiedPiCCS stage)
            (VerifiedStageComputations.VerifiedPiRLC stage)
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
          (VerifiedStageComputations.VerifiedPiCCS stage)
          (VerifiedStageComputations.VerifiedPiRLC stage)
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
          (VerifiedStageComputations.VerifiedPiCCS stage)
          (VerifiedStageComputations.VerifiedPiRLC stage)
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
    terminal_soundness_of_verified_stage_program_and_msis_of_compressed_prior_verifier_sound
      data
      stage
      hDigest
      hRed
      hMsis
      hPriorVerifier
      hAccepted
      hAlt
  let hPublic :=
    Construction2DirectFPrime.terminal_direct_fprime_public_image_invariants
      hInitialStep
      hInitialWellFormed
      (CompressedFPrimeAuthority.accepts_sound_of_verifier_sound
        hPriorVerifier)
      hAccepted
  rcases hSound with ⟨hReach, hUnique⟩
  rcases hPublic with ⟨hStep, hVk, hInitialBoundary, hWellFormed⟩
  exact
    ⟨hReach, hUnique, hStep, hVk, hInitialBoundary, hWellFormed⟩

/--
Compressed-prior terminal theorem for stages that reuse the existing SuperNeo
CE-relation authority.

This is the strongest production-shaped theorem over the direct reuse adapter:
the direct frontend supplies deterministic stage computations and output/source
bridges, while the imported SuperNeo `ceRelation` surface derives the
`Π_CCS` and `Π_RLC` theorem statements consumed by the terminal proof.
-/
theorem terminal_soundness_of_reused_stage_program_and_msis_of_compressed_prior_verifier_sound_with_public_image_invariants
    {Digest Boundary PiCCSOut : Type}
    {PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : ReusedStageComputations Digest PiCCSOut n)
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
            (ReusedStageComputations.toVerified stage))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified stage))
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
              (ReusedStageComputations.toVerified stage))
            (VerifiedStageComputations.VerifiedPiRLC
              (ReusedStageComputations.toVerified stage))
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
            (ReusedStageComputations.toVerified stage))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified stage))
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
            (ReusedStageComputations.toVerified stage))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified stage))
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
  terminal_soundness_of_verified_stage_program_and_msis_of_compressed_prior_verifier_sound_with_public_image_invariants
    data
    (ReusedStageComputations.toVerified stage)
    hInitialStep
    hInitialWellFormed
    hDigest
    hRed
    hMsis
    hPriorVerifier
    hAccepted
    hAlt

/--
Stage-verified direct-program terminal theorem specialized to proof-carrying
folded authority.

This is a convenience specialization of the arbitrary-authority theorem above.
Production compressed-proof integrations should instead instantiate
`PriorAuthoritySound` for their verifier and use
`terminal_soundness_of_verified_stage_program_and_msis_of_prior_authority_sound`.
-/
theorem terminal_soundness_of_verified_stage_program_and_msis
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : VerifiedStageComputations Digest PiCCSOut n)
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
        (VerifiedStageComputations.VerifiedPiCCS stage)
        (VerifiedStageComputations.VerifiedPiRLC stage)
        hashEncoded
        data.ce
        (ParentOpeningAuthorization.StatementEncodesByCommitment
          commitmentOfParent)
        initial}
    {proof : Unit}
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
              (VerifiedStageComputations.VerifiedPiCCS stage)
              (VerifiedStageComputations.VerifiedPiRLC stage)
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
              (VerifiedStageComputations.VerifiedPiCCS stage)
              (VerifiedStageComputations.VerifiedPiRLC stage)
              hashEncoded
              data.ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent)
              initial)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectTerminalSoundness.AccumulatorStep
            (params := params)
            (VerifiedStageComputations.VerifiedPiCCS stage)
            (VerifiedStageComputations.VerifiedPiRLC stage)
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
          (VerifiedStageComputations.VerifiedPiCCS stage)
          (VerifiedStageComputations.VerifiedPiRLC stage)
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
          (VerifiedStageComputations.VerifiedPiCCS stage)
          (VerifiedStageComputations.VerifiedPiRLC stage)
          hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial
        (priorSteps + 1)
        nextImage ∧
      nextImage = altNext :=
  terminal_soundness_of_verified_stage_program_and_msis_of_prior_authority_sound
    data
    stage
    hDigest
    hRed
    hMsis
    FoldedFPrimeAuthority.accepts_sound
    hAccepted
    hAlt

/--
Stage-verified proof-carrying terminal theorem with final public-image
invariants.

This is the strict theorem for the non-compressed authority path: the prior
folded `F'` object itself carries reachability evidence, the latest step is
checked at the terminal boundary, and the exposed Construction-2 image inherits
the public step counter, verifier-key digest, initial boundary, and fixed
`pc = 1` invariant from the base image.
-/
theorem terminal_soundness_of_verified_stage_program_and_msis_with_public_image_invariants
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : VerifiedStageComputations Digest PiCCSOut n)
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
        (VerifiedStageComputations.VerifiedPiCCS stage)
        (VerifiedStageComputations.VerifiedPiRLC stage)
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
              (VerifiedStageComputations.VerifiedPiCCS stage)
              (VerifiedStageComputations.VerifiedPiRLC stage)
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
              (VerifiedStageComputations.VerifiedPiCCS stage)
              (VerifiedStageComputations.VerifiedPiRLC stage)
              hashEncoded
              data.ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent)
              initial)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectTerminalSoundness.AccumulatorStep
            (params := params)
            (VerifiedStageComputations.VerifiedPiCCS stage)
            (VerifiedStageComputations.VerifiedPiRLC stage)
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
          (VerifiedStageComputations.VerifiedPiCCS stage)
          (VerifiedStageComputations.VerifiedPiRLC stage)
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
          (VerifiedStageComputations.VerifiedPiCCS stage)
          (VerifiedStageComputations.VerifiedPiRLC stage)
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
    terminal_soundness_of_verified_stage_program_and_msis
      data
      stage
      hDigest
      hRed
      hMsis
      hAccepted
      hAlt
  let hPublic :=
    Construction2DirectFPrime.terminal_direct_fprime_public_image_invariants
      hInitialStep
      hInitialWellFormed
      FoldedFPrimeAuthority.accepts_sound
      hAccepted
  rcases hSound with ⟨hReach, hUnique⟩
  rcases hPublic with ⟨hStep, hVk, hInitialBoundary, hWellFormed⟩
  exact
    ⟨hReach, hUnique, hStep, hVk, hInitialBoundary, hWellFormed⟩

/--
Proof-carrying terminal theorem for stages that reuse the existing SuperNeo
CE-relation authority.

This closes the theorem-level non-compressed induction path for direct CCS:
the prior folded `F'` authority is proof-carrying, while the latest direct
stage consumes imported `Π_CCS`/`Π_RLC` theorem statements derived from
SuperNeo `ceRelation` evidence.
-/
theorem terminal_soundness_of_reused_stage_program_and_msis_with_public_image_invariants
    {Digest Boundary PiCCSOut : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (data :
      DirectConcreteInstantiation.ConcreteCEData n params)
    (stage : ReusedStageComputations Digest PiCCSOut n)
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
          (ReusedStageComputations.toVerified stage))
        (VerifiedStageComputations.VerifiedPiRLC
          (ReusedStageComputations.toVerified stage))
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
                (ReusedStageComputations.toVerified stage))
              (VerifiedStageComputations.VerifiedPiRLC
                (ReusedStageComputations.toVerified stage))
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
                (ReusedStageComputations.toVerified stage))
              (VerifiedStageComputations.VerifiedPiRLC
                (ReusedStageComputations.toVerified stage))
              hashEncoded
              data.ce
              (ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent)
              initial)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectTerminalSoundness.AccumulatorStep
            (params := params)
            (VerifiedStageComputations.VerifiedPiCCS
              (ReusedStageComputations.toVerified stage))
            (VerifiedStageComputations.VerifiedPiRLC
              (ReusedStageComputations.toVerified stage))
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
            (ReusedStageComputations.toVerified stage))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified stage))
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
            (ReusedStageComputations.toVerified stage))
          (VerifiedStageComputations.VerifiedPiRLC
            (ReusedStageComputations.toVerified stage))
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
  terminal_soundness_of_verified_stage_program_and_msis_with_public_image_invariants
    data
    (ReusedStageComputations.toVerified stage)
    hInitialStep
    hInitialWellFormed
    hDigest
    hRed
    hMsis
    hAccepted
    hAlt

end DirectStageSemantics

end DirectCcsFPrime
