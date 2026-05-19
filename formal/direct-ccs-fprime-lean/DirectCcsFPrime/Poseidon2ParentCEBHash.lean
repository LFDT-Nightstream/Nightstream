import DirectCcsFPrime.DirectParentOnlyTerminalSoundness
import DirectCcsFPrime.ParentCEBHashBinding

/-!
Poseidon2 parent `CE(B)` hash boundary.

This module does not implement Poseidon2. It names the implementation-facing
assumption: the Poseidon2 field-list hash used for parent `CE(B)` handles binds
the canonical `ParentEncoding.encodeSomeParentCEB` preimage. From that single
assumption it constructs the structured `ParentCEBHashBinding.ParentCEBHash`
object consumed by terminal direct CCS theorems.
-/

namespace DirectCcsFPrime

namespace Poseidon2ParentCEBHash

/--
The exact Poseidon2 binding assumption needed by the parent-handle path.

The hash implementation is represented by `hashEncoded`; Lean assumes only
that equal digests over canonical parent `CE(B)` encodings imply equal
canonical encodings.
-/
def BindingAssumption
    {Digest : Type}
    (hashEncoded : List Nat → Digest) : Prop :=
  ParentEncoding.EncodedParentCEBDigestBinding hashEncoded

/--
Implementation-facing Poseidon2 parent-hash object.

`hashEncoded` is the verifier's Poseidon2 field-list hash for parent handles.
`binding` is the cryptographic assumption for that exact function, restricted
to canonical encoded parent `CE(B)` handles.
-/
structure Hash (Digest : Type) where
  hashEncoded : List Nat → Digest
  binding : BindingAssumption hashEncoded

/-- Convert the Poseidon2-assumed hash object to the canonical parent hash. -/
def toParentCEBHash
    {Digest : Type}
    (ctx : Hash Digest) :
    ParentCEBHashBinding.ParentCEBHash Digest where
  hashEncoded := ctx.hashEncoded
  encodedBinding := ctx.binding

/-- The Poseidon2 object supplies the canonical parent-encoding binding premise. -/
theorem encodedParentCEBDigestBinding
    {Digest : Type}
    (ctx : Hash Digest) :
    ParentEncoding.EncodedParentCEBDigestBinding ctx.hashEncoded :=
  ctx.binding

/-- Digest of a canonical parent `CE(B)` handle with the Poseidon2 parent hash. -/
def digest
    {Digest : Type}
    (ctx : Hash Digest)
    (parent : ParentEncoding.SomeParentCEB) : Digest :=
  ParentCEBHashBinding.digest (toParentCEBHash ctx) parent

/-- Digest source induced by a canonical parent `CE(B)` handle. -/
def source
    {Digest : Type}
    (ctx : Hash Digest)
    (parent : ParentEncoding.SomeParentCEB) :
    DigestParentBinding.Source Digest :=
  ParentCEBHashBinding.source (toParentCEBHash ctx) parent

/--
Equal Poseidon2 parent-handle digests recover the exact parent `CE(B)` handle.
-/
theorem same_parentCEB_of_digest_eq
    {Digest : Type}
    (ctx : Hash Digest)
    {parentA parentB : ParentEncoding.SomeParentCEB}
    (hDigest : digest ctx parentA = digest ctx parentB) :
    parentA = parentB :=
  ParentCEBHashBinding.same_parentCEB_of_digest_eq
    (toParentCEBHash ctx)
    hDigest

/--
The Poseidon2 parent-handle source functionally binds every deterministic DEC
residue projection from an opened parent `CE(B)` handle.
-/
theorem projected_residue_source_functional
    {n : Nat}
    {Digest : Type}
    (ctx : Hash Digest)
    (project : ParentEncoding.SomeParentCEB → (Fin n → Nat)) :
    GoldilocksChildTableAuthorization.SourceBindsParentFunctionally
      (ParentEncoding.BindsProjectedParentCEBResidues
        (n := n)
        ctx.hashEncoded
        project) :=
  ParentCEBHashBinding.projected_residue_source_functional
    (toParentCEBHash ctx)
    project

/--
Parent-only terminal soundness with arbitrary sound prior authority and the
implementation-facing Poseidon2 parent hash object.
-/
theorem terminal_soundness_of_poseidon2_parent_hash_prior_authority_sound_and_msis
    {Digest Boundary PiCCSOut Authority : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (parentHash : Hash Digest)
    (data : DirectConcreteInstantiation.ConcreteCEData n params)
    {computeBoundary : Nat → Boundary → Boundary}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle
          (DigestParentBinding.Source Digest)
          n →
        PiCCSOut →
          Prop}
    {PiRLC :
      Nat →
        PiCCSOut →
        DigestParentBinding.Source Digest →
          Prop}
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
    (hPiCCS : ParentSourceStep.PiCCSFunctional PiCCS)
    (hPiRLC : ParentSourceStep.PiRLCFunctional PiRLC)
    (hInitialStep : initial.step = 0)
    (hInitialWellFormed : Construction2DirectFPrime.WellFormed initial)
    (hRed :
      SuperNeo.ProofSystem.MSISToAjtaiReductions params)
    (hMsis :
      SuperNeo.ProofSystem.MSISHardnessAssumption params)
    (hAuthority :
      FPrimeInduction.PriorAuthoritySound
        (DirectParentOnlyTerminalSoundness.Transition
          (params := params)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          PiCCS
          PiRLC
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
          (Authority := Authority)
          (DirectProgramStep.ComputedBoundaryStep computeBoundary)
          (DirectParentOnlyTerminalSoundness.AccumulatorStep
            (params := params)
            PiCCS
            PiRLC
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
        (DirectParentOnlyTerminalSoundness.AccumulatorStep
          (params := params)
          PiCCS
          PiRLC
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
          PiCCS
          PiRLC
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
          PiCCS
          PiRLC
          priorSteps
          priorImage.accumulator
          priorInputs
          nextImage.accumulator.parentSource ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC
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
    hPiCCS
    hPiRLC
    hInitialStep
    hInitialWellFormed
    (encodedParentCEBDigestBinding parentHash)
    hRed
    hMsis
    hAuthority
    hAccepted
    hAlt

/--
Parent-only terminal soundness with a compressed prior verifier and the
implementation-facing Poseidon2 parent hash object.
-/
theorem terminal_soundness_of_poseidon2_parent_hash_sound_verifier_and_msis
    {Digest Boundary PiCCSOut PriorProof : Type}
    {n : Nat}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    (parentHash : Hash Digest)
    (data : DirectConcreteInstantiation.ConcreteCEData n params)
    {computeBoundary : Nat → Boundary → Boundary}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {PiCCS :
      Nat →
        ReducedAccumulatorStep.AccumulatorHandle
          (DigestParentBinding.Source Digest)
          n →
        PiCCSOut →
          Prop}
    {PiRLC :
      Nat →
        PiCCSOut →
        DigestParentBinding.Source Digest →
          Prop}
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
          PiCCS
          PiRLC
          parentHash.hashEncoded
          data.ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        initial)
    {priorSteps : Nat}
    {priorProof : PriorProof}
    {latestProof : Unit}
    (hPiCCS : ParentSourceStep.PiCCSFunctional PiCCS)
    (hPiRLC : ParentSourceStep.PiRLCFunctional PiRLC)
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
            PiCCS
            PiRLC
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
          PiCCS
          PiRLC
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
          PiCCS
          PiRLC
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
          PiCCS
          PiRLC
          priorSteps
          priorImage.accumulator
          priorInputs
          nextImage.accumulator.parentSource ∧
        ParentOnlyAccumulatorStep.ParentSourceFromPiStages
          (n := n)
          PiCCS
          PiRLC
          priorSteps
          priorImage.accumulator
          priorInputs
          altNext.accumulator.parentSource) ∧
      nextImage.step = priorSteps + 1 ∧
      initial.vkDigest = nextImage.vkDigest ∧
      initial.initialBoundary = nextImage.initialBoundary ∧
      Construction2DirectFPrime.WellFormed nextImage :=
  terminal_soundness_of_poseidon2_parent_hash_prior_authority_sound_and_msis
    parentHash
    data
    hPiCCS
    hPiRLC
    hInitialStep
    hInitialWellFormed
    hRed
    hMsis
    (CompressedFPrimeAuthority.sound_verifier_prior_authority_sound verifier)
    hAccepted
    hAlt

end Poseidon2ParentCEBHash

end DirectCcsFPrime
