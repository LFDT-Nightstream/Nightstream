import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
import NightstreamFPrime.Layout.Stage1.PiCCSSecurity
import NightstreamFPrime.Spec.Folding.Nifs.PaperSecurityComposition
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.ForkStrongSet

/-!
Owns the deterministic binding reduction for one verifier-selected canonical
Stage 1 application package.

The success branch identifies the complete circuit-and-matrix envelope and
all four raw verifier-authority word lists. The failure branches name only
Poseidon2 collisions. This module assigns no probability to those events and
does not authorize a proof backend.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationSecurity

open NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
open NightstreamFPrime.Layout.Stage1.PiCCSSecurity
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold

abbrev Program := Lifecycle.Stage1.Application.Program

abbrev FitsTwoPow28 (program : Program) :=
  PerApplicationFixedPoint.FitsTwoPow28 program

abbrev CommitmentKey (program : Program) :=
  AjtaiKey
    (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
    (publicFits := PerApplicationFixedPoint.publicFits program)

abbrev CanonicalKey (program : Program) (fits : FitsTwoPow28 program) :=
  ProductionKey.KeyType (PerApplicationFixedPoint.relation program fits)

noncomputable def canonicalKey {program : Program}
    (fits : FitsTwoPow28 program) (ajtai : CommitmentKey program) :
    CanonicalKey program fits :=
  ProductionKey.key (PerApplicationFixedPoint.relation program fits) ajtai

abbrev StepInput (program : Program) (fits : FitsTwoPow28 program) :=
  Input KeyDigest AppState AppWitness
    (Running
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program))
    (Fresh
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program))
    (Proof (ProductionKey.degreeBound
      (PerApplicationFixedPoint.relation program fits))) slotCount

abbrev StepOutput (program : Program) :=
  Output Digest AppState
    (Running
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program)) slotCount

def selectedRunning {program : Program} {fits : FitsTwoPow28 program}
    (input : StepInput program fits) :=
  input.running functionIndex

def verificationKeyDigest {program : Program}
    (fits : FitsTwoPow28 program) (ajtai : CommitmentKey program) : KeyDigest :=
  (verificationKeyBinding fits ajtai).digest

/-- Exact prior-state hash preimage selected by this package and the actual
HyperNova input. -/
noncomputable def replayState {program : Program}
    (fits : FitsTwoPow28 program) (ajtai : CommitmentKey program)
    (input : StepInput program fits) :
    HashPreimage
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program) :=
  priorHashPreimage
    (Lifecycle.setup (PerApplicationFixedPoint.relation program fits) ajtai
      (verificationKeyDigest fits ajtai)) input

/-- Exact PiCCS statement and round-message replay selected by the canonical
package key and the actual NIFS proof. No transcript field is caller-owned. -/
noncomputable def replayInput {program : Program}
    (fits : FitsTwoPow28 program) (ajtai : CommitmentKey program)
    (input : StepInput program fits) :
    Spec.Folding.PiCCS.TranscriptReplay.ReplayInput K Transcript.State
      productionShape :=
  let key := ProductionKey.key
    (PerApplicationFixedPoint.relation program fits) ajtai
  let running := selectedRunning input
  {
    statement := {
      priorState := key.publicInputState running input.fresh
      input := (key.statement running input.fresh).verifierInput key.lift
    }
    rounds := {
      rounds := fun round => (input.nifsProof.piCcsRounds round).toMessage
    }
  }

/-- The replay view derives exactly the coins used by the production NIFS
key. -/
theorem replayInput_derive_eq_piCcsExecution {program : Program}
    (fits : FitsTwoPow28 program) (ajtai : CommitmentKey program)
    (input : StepInput program fits) :
    (replayInput fits ajtai input).derive
        (ProductionKey.key
          (PerApplicationFixedPoint.relation program fits) ajtai).oracle =
      ((ProductionKey.key
        (PerApplicationFixedPoint.relation program fits) ajtai).piCcsExecution
          (selectedRunning input) input.fresh input.nifsProof).coins := by
  rfl

/-- Two different complete PiCCS outputs produce one post-output transcript
state when absorbed from the same causal pre-output state. -/
def PiCcsOutputAbsorptionCollision {program : Program}
    (fits : FitsTwoPow28 program) (ajtai : CommitmentKey program)
    (state : Transcript.State)
    (left right : FullOutputCoordinates.FullOutput K productionShape) : Prop :=
  left ≠ right ∧
    (canonicalKey fits ajtai).absorbPiCcsOutput state left =
      (canonicalKey fits ajtai).absorbPiCcsOutput state right

/-- Convert the isolated production low-norm theorem into the exact strong-set
unit record used by one concrete extraction algebra. -/
noncomputable def productionStrongSet {program : Program}
    (fits : FitsTwoPow28 program) (ajtai : CommitmentKey program)
    (laws : Spec.Folding.PiRLC.PaperForkExtraction.ExtractionAlgebra
      (canonicalKey fits ajtai).piRlcSemantics
      (canonicalKey fits ajtai).params
      (canonicalKey fits ajtai).piRlcAlgebra)
    (ringExact : laws.ring =
      Spec.Phi81Relation.PiRLCAlgebra.ForkStrongSet.ring)
    (theorem8 : Spec.Phi81StrongSet.LowNormInvertibility) :
    Spec.Folding.PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      (canonicalKey fits ajtai).piRlcAlgebra.challengeValid := by
  rw [ringExact]
  exact Spec.Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits theorem8

/-- Package-derived committed-statement reduction. The left authority, state,
statement, fresh input, and round messages are all computed from one canonical
package and one actual HyperNova input. The right side is the alleged replay
being compared with it. -/
theorem packageReplay_identifies_claim_or_failure {program : Program}
    (fits : FitsTwoPow28 program) (ajtai : CommitmentKey program)
    (input : StepInput program fits)
    (claimedAuthority : VerifierContext.Authority)
    (claimedState : HashPreimage
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program))
    (claimedReplay : Spec.Folding.PiCCS.TranscriptReplay.ReplayInput K
      Transcript.State productionShape)
    (stateWellFormed :
      NightstreamFPrime.Layout.Stage1.StateEncoding.WellFormed
        (replayState fits ajtai input))
    (claimedStateWellFormed :
      NightstreamFPrime.Layout.Stage1.StateEncoding.WellFormed claimedState)
    (contextDigestEqual :
      (VerifierContext.descriptor (authority fits ajtai)).digest4 =
        (VerifierContext.descriptor claimedAuthority).digest4)
    (stateDigestEqual :
      stateHash (publicFits := PerApplicationFixedPoint.publicFits program)
          (replayState fits ajtai input) =
        stateHash (publicFits := PerApplicationFixedPoint.publicFits program)
          claimedState)
    (alphaEqual :
      ((replayInput fits ajtai input).derive
        (ProductionKey.key
          (PerApplicationFixedPoint.relation program fits) ajtai).oracle).alpha =
      (claimedReplay.derive
        (ProductionKey.key
          (PerApplicationFixedPoint.relation program fits) ajtai).oracle).alpha)
    (gammaEqual :
      ((replayInput fits ajtai input).derive
        (ProductionKey.key
          (PerApplicationFixedPoint.relation program fits) ajtai).oracle).gamma =
      (claimedReplay.derive
        (ProductionKey.key
          (PerApplicationFixedPoint.relation program fits) ajtai).oracle).gamma)
    (roundPointEqual :
      ((replayInput fits ajtai input).derive
        (ProductionKey.key
          (PerApplicationFixedPoint.relation program fits) ajtai).oracle
        ).roundPoint =
      (claimedReplay.derive
        (ProductionKey.key
          (PerApplicationFixedPoint.relation program fits) ajtai).oracle
        ).roundPoint) :
    (authority fits ajtai = claimedAuthority ∧
      replayState fits ajtai input = claimedState ∧
      replayInput fits ajtai input = claimedReplay) ∨
      AuthorityComponentDigestCollision (authority fits ajtai)
        claimedAuthority ∨
      ContextDigestCollision (VerifierContext.descriptor (authority fits ajtai))
        (VerifierContext.descriptor claimedAuthority) ∨
      StateHashCollision (replayState fits ajtai input) claimedState ∨
      Spec.Folding.PiCCS.TranscriptReplay.TranscriptReplayCollision
        (ProductionKey.key
          (PerApplicationFixedPoint.relation program fits) ajtai).oracle
        (replayInput fits ajtai input) claimedReplay := by
  exact committed_authority_statement_challenges_identify_or_failure
    (ProductionKey.key
      (PerApplicationFixedPoint.relation program fits) ajtai).oracle
    (authority fits ajtai) claimedAuthority
    (replayState fits ajtai input) claimedState
    (replayInput fits ajtai input) claimedReplay
    stateWellFormed claimedStateWellFormed contextDigestEqual stateDigestEqual
    alphaEqual gammaEqual roundPointEqual

/-- Package-derived committed-statement and complete-output reduction. The
success branch identifies the actual NIFS proof's PiCCS output as well as its
authority, state, statement, fresh input, and round messages. -/
theorem packageReplayAndOutput_identifies_claim_or_failure {program : Program}
    (fits : FitsTwoPow28 program) (ajtai : CommitmentKey program)
    (input : StepInput program fits)
    (claimedAuthority : VerifierContext.Authority)
    (claimedState : HashPreimage
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program))
    (claimedReplay : Spec.Folding.PiCCS.TranscriptReplay.ReplayInput K
      Transcript.State productionShape)
    (claimedOutput : FullOutputCoordinates.FullOutput K productionShape)
    (stateWellFormed :
      NightstreamFPrime.Layout.Stage1.StateEncoding.WellFormed
        (replayState fits ajtai input))
    (claimedStateWellFormed :
      NightstreamFPrime.Layout.Stage1.StateEncoding.WellFormed claimedState)
    (contextDigestEqual :
      (VerifierContext.descriptor (authority fits ajtai)).digest4 =
        (VerifierContext.descriptor claimedAuthority).digest4)
    (stateDigestEqual :
      stateHash (publicFits := PerApplicationFixedPoint.publicFits program)
          (replayState fits ajtai input) =
        stateHash (publicFits := PerApplicationFixedPoint.publicFits program)
          claimedState)
    (alphaEqual :
      ((replayInput fits ajtai input).derive
        (canonicalKey fits ajtai).oracle).alpha =
      (claimedReplay.derive (canonicalKey fits ajtai).oracle).alpha)
    (gammaEqual :
      ((replayInput fits ajtai input).derive
        (canonicalKey fits ajtai).oracle).gamma =
      (claimedReplay.derive (canonicalKey fits ajtai).oracle).gamma)
    (roundPointEqual :
      ((replayInput fits ajtai input).derive
        (canonicalKey fits ajtai).oracle).roundPoint =
      (claimedReplay.derive (canonicalKey fits ajtai).oracle).roundPoint)
    (outgoingStateEqual :
      ((canonicalKey fits ajtai).piCcsExecution (selectedRunning input)
        input.fresh input.nifsProof).outgoingState =
      (canonicalKey fits ajtai).absorbPiCcsOutput
        (claimedReplay.derive (canonicalKey fits ajtai).oracle).finalState
        claimedOutput) :
    (authority fits ajtai = claimedAuthority ∧
      replayState fits ajtai input = claimedState ∧
      replayInput fits ajtai input = claimedReplay ∧
      input.nifsProof.piCcsOutput = claimedOutput) ∨
      AuthorityComponentDigestCollision (authority fits ajtai)
        claimedAuthority ∨
      ContextDigestCollision (VerifierContext.descriptor (authority fits ajtai))
        (VerifierContext.descriptor claimedAuthority) ∨
      StateHashCollision (replayState fits ajtai input) claimedState ∨
      Spec.Folding.PiCCS.TranscriptReplay.TranscriptReplayCollision
        (canonicalKey fits ajtai).oracle (replayInput fits ajtai input)
        claimedReplay ∨
      PiCcsOutputAbsorptionCollision fits ajtai
        ((replayInput fits ajtai input).derive
          (canonicalKey fits ajtai).oracle).finalState
        input.nifsProof.piCcsOutput claimedOutput := by
  rcases packageReplay_identifies_claim_or_failure fits ajtai input
      claimedAuthority claimedState claimedReplay stateWellFormed
      claimedStateWellFormed contextDigestEqual stateDigestEqual alphaEqual
      gammaEqual roundPointEqual with
    identified | componentFailure | contextFailure | stateFailure |
      transcriptFailure
  · rcases identified with ⟨authoritySame, stateSame, replaySame⟩
    by_cases outputSame : input.nifsProof.piCcsOutput = claimedOutput
    · exact Or.inl ⟨authoritySame, stateSame, replaySame, outputSame⟩
    · apply Or.inr
      apply Or.inr
      apply Or.inr
      apply Or.inr
      apply Or.inr
      refine ⟨outputSame, ?_⟩
      calc
        (canonicalKey fits ajtai).absorbPiCcsOutput
            ((replayInput fits ajtai input).derive
              (canonicalKey fits ajtai).oracle).finalState
            input.nifsProof.piCcsOutput =
            ((canonicalKey fits ajtai).piCcsExecution
              (selectedRunning input) input.fresh input.nifsProof
            ).outgoingState := by
              rfl
        _ = (canonicalKey fits ajtai).absorbPiCcsOutput
            (claimedReplay.derive (canonicalKey fits ajtai).oracle).finalState
            claimedOutput := outgoingStateEqual
        _ = (canonicalKey fits ajtai).absorbPiCcsOutput
            ((replayInput fits ajtai input).derive
              (canonicalKey fits ajtai).oracle).finalState claimedOutput := by
              rw [replaySame]
  · exact Or.inr (Or.inl componentFailure)
  · exact Or.inr (Or.inr (Or.inl contextFailure))
  · exact Or.inr (Or.inr (Or.inr (Or.inl stateFailure)))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl transcriptFailure))))

private theorem selectedIndex_eq_functionIndex
    {programCounter : Nat} (valid : InRange slotCount programCounter) :
    selectedIndex valid = functionIndex := by
  apply Fin.ext
  have bound := (selectedIndex valid).isLt
  change (selectedIndex valid).val < 1 at bound
  change (selectedIndex valid).val = 0
  omega

/-- Complete per-application security boundary for one exact HyperNova step.
The base branch performs no NIFS extraction. The recursive branch uses the
same relation, Ajtai key, proof, running input, and output already selected by
`StepHoldsFor`. -/
theorem stepHoldsFor_implies_base_or_securityOutcome {program : Program}
    (fits : FitsTwoPow28 program) (ajtai : CommitmentKey program)
    (input : StepInput program fits) (output : StepOutput program)
    (laws : Spec.Folding.PiRLC.PaperForkExtraction.ExtractionAlgebra
      (canonicalKey fits ajtai).piRlcSemantics
      (canonicalKey fits ajtai).params
      (canonicalKey fits ajtai).piRlcAlgebra)
    (ringExact : laws.ring =
      Spec.Phi81Relation.PiRLCAlgebra.ForkStrongSet.ring)
    (theorem8 : Spec.Phi81StrongSet.LowNormInvertibility)
    (step : Lifecycle.StepHoldsFor
      (PerApplicationFixedPoint.relation program fits) ajtai
      (verificationKeyDigest fits ajtai) program input output) :
    Lifecycle.StepHoldsFor
        (PerApplicationFixedPoint.relation program fits) ajtai
        (verificationKeyDigest fits ajtai) program input output /\
      (input.iteration = 0 \/
        (0 < input.iteration /\
          Spec.Folding.Nifs.PaperSecurityComposition.SecurityOutcome
            (canonicalKey fits ajtai) (selectedRunning input) input.fresh
            input.nifsProof laws
            (productionStrongSet fits ajtai laws ringExact theorem8))) := by
  refine ⟨step, ?_⟩
  change FixedAugmentedTransition
    (Lifecycle.setup (PerApplicationFixedPoint.relation program fits) ajtai
      (verificationKeyDigest fits ajtai))
    (Lifecycle.machineFor (PerApplicationFixedPoint.publicFits program) program)
    functionIndex input output at step
  rcases step.2.2.2 with base | recursive
  · exact Or.inl base.1
  · rcases recursive with
      ⟨priorPcValid, iterationPositive, _priorPublic, selectedNifs, _unchanged⟩
    have selectedEq : selectedIndex priorPcValid = functionIndex :=
      selectedIndex_eq_functionIndex priorPcValid
    rw [selectedEq] at selectedNifs
    have accepted :
        Spec.Folding.Nifs.PaperNonInteractive.verify
            (canonicalKey fits ajtai) (selectedRunning input) input.fresh
            input.nifsProof =
          some (output.runningNext functionIndex) := by
      simpa [Accepts, Lifecycle.setup,
        Lifecycle.nifsVerifier, canonicalKey, selectedRunning] using selectedNifs
    exact Or.inr ⟨iterationPositive,
      Spec.Folding.Nifs.PaperSecurityComposition.accepted_implies_securityOutcome
        (canonicalKey fits ajtai) (selectedRunning input) input.fresh
        input.nifsProof (output.runningNext functionIndex) laws
        (productionStrongSet fits ajtai laws ringExact theorem8) accepted⟩

/-- Two different canonical circuit-and-matrix envelopes produce one
structural package digest. -/
def StructuralPackageCollision
    (leftProgram rightProgram : Program)
    (leftFits : FitsTwoPow28 leftProgram)
    (rightFits : FitsTwoPow28 rightProgram) : Prop :=
  sealedPackageValue leftProgram leftFits ≠
      sealedPackageValue rightProgram rightFits ∧
    structuralPackageIdentity leftProgram leftFits =
      structuralPackageIdentity rightProgram rightFits

/-- Two different package-and-authority preimages produce one final package
identity. -/
def FinalPackageBindingCollision
    {leftProgram rightProgram : Program}
    (leftFits : FitsTwoPow28 leftProgram)
    (rightFits : FitsTwoPow28 rightProgram)
    (leftKey : CommitmentKey leftProgram)
    (rightKey : CommitmentKey rightProgram) : Prop :=
  packageIdentityPreimage leftFits leftKey ≠
      packageIdentityPreimage rightFits rightKey ∧
    packageIdentity leftFits leftKey = packageIdentity rightFits rightKey

/-- Equal final package identities identify the exact canonical circuit,
matrix program, and every raw verifier-authority word list unless one named
Poseidon2 binding layer collides. -/
theorem packageIdentity_identifies_package_authority_or_collision
    {leftProgram rightProgram : Program}
    (leftFits : FitsTwoPow28 leftProgram)
    (rightFits : FitsTwoPow28 rightProgram)
    (leftKey : CommitmentKey leftProgram)
    (rightKey : CommitmentKey rightProgram)
    (identityEqual : packageIdentity leftFits leftKey =
      packageIdentity rightFits rightKey) :
    (sealedPackageValue leftProgram leftFits =
        sealedPackageValue rightProgram rightFits ∧
      authority leftFits leftKey = authority rightFits rightKey) ∨
      StructuralPackageCollision leftProgram rightProgram leftFits rightFits ∨
      AuthorityComponentDigestCollision
        (authority leftFits leftKey) (authority rightFits rightKey) ∨
      FinalPackageBindingCollision leftFits rightFits leftKey rightKey := by
  by_cases preimageSame : packageIdentityPreimage leftFits leftKey =
      packageIdentityPreimage rightFits rightKey
  · have components := packageIdentityPreimage_components leftFits rightFits
      leftKey rightKey preimageSame
    rcases descriptor_identifies_authority_or_component_collision
        (authority leftFits leftKey) (authority rightFits rightKey)
        components.2 with authoritySame | componentCollision
    · by_cases packageSame : sealedPackageValue leftProgram leftFits =
          sealedPackageValue rightProgram rightFits
      · exact Or.inl ⟨packageSame, authoritySame⟩
      · exact Or.inr (Or.inl ⟨packageSame, components.1⟩)
    · exact Or.inr (Or.inr (Or.inl componentCollision))
  · exact Or.inr (Or.inr (Or.inr ⟨preimageSame, identityEqual⟩))

/-- Equality of verifier-key bindings has the same exact package-and-authority
reduction. The prover cannot use binding equality to select another circuit,
matrix program, application, or key without one named collision event. -/
theorem verificationKeyBinding_identifies_package_authority_or_collision
    {leftProgram rightProgram : Program}
    (leftFits : FitsTwoPow28 leftProgram)
    (rightFits : FitsTwoPow28 rightProgram)
    (leftKey : CommitmentKey leftProgram)
    (rightKey : CommitmentKey rightProgram)
    (bindingEqual : verificationKeyBinding leftFits leftKey =
      verificationKeyBinding rightFits rightKey) :
    (sealedPackageValue leftProgram leftFits =
        sealedPackageValue rightProgram rightFits ∧
      authority leftFits leftKey = authority rightFits rightKey) ∨
      StructuralPackageCollision leftProgram rightProgram leftFits rightFits ∨
      AuthorityComponentDigestCollision
        (authority leftFits leftKey) (authority rightFits rightKey) ∨
      FinalPackageBindingCollision leftFits rightFits leftKey rightKey := by
  apply packageIdentity_identifies_package_authority_or_collision
    leftFits rightFits leftKey rightKey
  have packageIdentityEqual := congrArg
    Lifecycle.Stage1.VerificationKey.Binding.packageIdentity bindingEqual
  simpa only [verificationKeyBinding_packageIdentity] using
    packageIdentityEqual

end NightstreamFPrime.Export.Stage1.PerApplicationSecurity
