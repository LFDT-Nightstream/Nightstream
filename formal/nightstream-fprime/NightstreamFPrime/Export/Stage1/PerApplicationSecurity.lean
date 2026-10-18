import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
import NightstreamFPrime.Export.Stage1.PerApplicationFixedPointSoundness
import NightstreamFPrime.Lifecycle.PaperExtractionAlgebra
import NightstreamFPrime.Layout.Stage1.PiCCSSecurity
import NightstreamFPrime.Spec.Folding.Nifs.PaperSecurityComposition
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.ForkStrongSet
import NightstreamFPrime.Spec.Phi81StrongSet.LowNormInvertibility

/-!
Owns the deterministic binding reduction for one verifier-selected canonical
Stage 1 application package.
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

abbrev CommitmentSetup (program : Program) :=
  PerApplicationCanonicalPackage.CommitmentSetup program

abbrev CanonicalKey (program : Program) (fits : FitsTwoPow28 program) :=
  ProductionKey.KeyType (PerApplicationFixedPoint.relation program fits)

noncomputable def canonicalKey {program : Program}
    (fits : FitsTwoPow28 program)
    (commitmentSetup : CommitmentSetup program) :
    CanonicalKey program fits :=
  ProductionKey.key (PerApplicationFixedPoint.relation program fits)
    (commitmentKey commitmentSetup)

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

def verifierContextDigest {program : Program}
    (fits : FitsTwoPow28 program)
    (commitmentSetup : CommitmentSetup program) : KeyDigest :=
  PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup

noncomputable def replayState {program : Program}
    (fits : FitsTwoPow28 program)
    (commitmentSetup : CommitmentSetup program)
    (input : StepInput program fits) :
    HashPreimage
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program) :=
  priorHashPreimage
    (Lifecycle.setup (PerApplicationFixedPoint.relation program fits)
      (commitmentKey commitmentSetup)
      (verifierContextDigest fits commitmentSetup)) input

noncomputable def replayInput {program : Program}
    (fits : FitsTwoPow28 program)
    (commitmentSetup : CommitmentSetup program)
    (input : StepInput program fits) :
    Spec.Folding.PiCCS.TranscriptReplay.ReplayInput K Transcript.State
      productionShape :=
  let key := ProductionKey.key
    (PerApplicationFixedPoint.relation program fits)
    (commitmentKey commitmentSetup)
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

theorem replayInput_statement_input {program : Program}
    (fits : FitsTwoPow28 program)
    (commitmentSetup : CommitmentSetup program)
    (input : StepInput program fits) :
    (replayInput fits commitmentSetup input).statement.input =
      ((canonicalKey fits commitmentSetup).statement
        (selectedRunning input) input.fresh).verifierInput
          (canonicalKey fits commitmentSetup).lift := by
  rfl

theorem replayInput_derive_eq_piCcsExecution {program : Program}
    (fits : FitsTwoPow28 program)
    (commitmentSetup : CommitmentSetup program)
    (input : StepInput program fits) :
    (replayInput fits commitmentSetup input).derive
        (ProductionKey.key
          (PerApplicationFixedPoint.relation program fits)
          (commitmentKey commitmentSetup)).oracle =
      ((ProductionKey.key
        (PerApplicationFixedPoint.relation program fits)
        (commitmentKey commitmentSetup)).piCcsExecution
          (selectedRunning input) input.fresh input.nifsProof).coins := by
  rfl

def PiCcsOutputAbsorptionCollision {program : Program}
    (fits : FitsTwoPow28 program)
    (commitmentSetup : CommitmentSetup program)
    (state : Transcript.State)
    (left right : FullOutputCoordinates.FullOutput K productionShape) : Prop :=
  left ≠ right ∧
    (canonicalKey fits commitmentSetup).absorbPiCcsOutput state left =
      (canonicalKey fits commitmentSetup).absorbPiCcsOutput state right

noncomputable def productionExtractionAlgebra {program : Program}
    (fits : FitsTwoPow28 program)
    (commitmentSetup : CommitmentSetup program) :
    Spec.Folding.PiRLC.PaperForkExtraction.ExtractionAlgebra
      (canonicalKey fits commitmentSetup).piRlcSemantics
      (canonicalKey fits commitmentSetup).params
      (canonicalKey fits commitmentSetup).piRlcAlgebra :=
  Lifecycle.PaperExtractionAlgebra.extractionAlgebra
    (commitmentKey commitmentSetup)

noncomputable def productionStrongSet {program : Program}
    (fits : FitsTwoPow28 program)
    (commitmentSetup : CommitmentSetup program) :
    Spec.Folding.PiRLC.PaperForkExtraction.StrongSetUnits
      (productionExtractionAlgebra fits commitmentSetup).ring
      (canonicalKey fits commitmentSetup).piRlcAlgebra.challengeValid := by
  exact Spec.Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits
    Spec.Phi81StrongSet.lowNormInvertibility

theorem packageReplay_identifies_claim_or_failure {program : Program}
    (fits : FitsTwoPow28 program)
    (commitmentSetup : CommitmentSetup program)
    (input : StepInput program fits)
    (claimedAuthority : VerifierContext.Authority)
    (claimedState : HashPreimage
      (logicalWidth := PerApplicationFixedPoint.logicalWidth program)
      (publicFits := PerApplicationFixedPoint.publicFits program))
    (claimedReplay : Spec.Folding.PiCCS.TranscriptReplay.ReplayInput K
      Transcript.State productionShape)
    (stateWellFormed :
      NightstreamFPrime.Layout.Stage1.StateEncoding.WellFormed
        (replayState fits commitmentSetup input))
    (claimedStateWellFormed :
      NightstreamFPrime.Layout.Stage1.StateEncoding.WellFormed claimedState)
    (contextDigestEqual :
      (VerifierContext.descriptor (authority fits commitmentSetup)).digest4 =
        (VerifierContext.descriptor claimedAuthority).digest4)
    (stateDigestEqual :
      stateHash (publicFits := PerApplicationFixedPoint.publicFits program)
          (replayState fits commitmentSetup input) =
        stateHash (publicFits := PerApplicationFixedPoint.publicFits program)
          claimedState)
    (alphaEqual :
      ((replayInput fits commitmentSetup input).derive
        (ProductionKey.key
          (PerApplicationFixedPoint.relation program fits)
          (commitmentKey commitmentSetup)).oracle).alpha =
      (claimedReplay.derive
        (ProductionKey.key
          (PerApplicationFixedPoint.relation program fits)
          (commitmentKey commitmentSetup)).oracle).alpha)
    (gammaEqual :
      ((replayInput fits commitmentSetup input).derive
        (ProductionKey.key
          (PerApplicationFixedPoint.relation program fits)
          (commitmentKey commitmentSetup)).oracle).gamma =
      (claimedReplay.derive
        (ProductionKey.key
          (PerApplicationFixedPoint.relation program fits)
          (commitmentKey commitmentSetup)).oracle).gamma)
    (roundPointEqual :
      ((replayInput fits commitmentSetup input).derive
        (ProductionKey.key
          (PerApplicationFixedPoint.relation program fits)
          (commitmentKey commitmentSetup)).oracle
        ).roundPoint =
      (claimedReplay.derive
        (ProductionKey.key
          (PerApplicationFixedPoint.relation program fits)
          (commitmentKey commitmentSetup)).oracle
        ).roundPoint) :
    (authority fits commitmentSetup = claimedAuthority ∧
      replayState fits commitmentSetup input = claimedState ∧
      (replayInput fits commitmentSetup input).authority =
        claimedReplay.authority) ∨
      AuthorityComponentDigestCollision (authority fits commitmentSetup)
        claimedAuthority ∨
      ContextDigestCollision
        (VerifierContext.descriptor (authority fits commitmentSetup))
        (VerifierContext.descriptor claimedAuthority) ∨
      StateHashCollision (replayState fits commitmentSetup input) claimedState ∨
      Spec.Folding.PiCCS.TranscriptReplay.TranscriptReplayCollision
        (ProductionKey.key
          (PerApplicationFixedPoint.relation program fits)
          (commitmentKey commitmentSetup)).oracle
        (replayInput fits commitmentSetup input) claimedReplay := by
  exact committed_authority_statement_challenges_identify_or_failure
    (ProductionKey.key
      (PerApplicationFixedPoint.relation program fits)
      (commitmentKey commitmentSetup)).oracle
    (authority fits commitmentSetup) claimedAuthority
    (replayState fits commitmentSetup input) claimedState
    (replayInput fits commitmentSetup input) claimedReplay
    stateWellFormed claimedStateWellFormed contextDigestEqual stateDigestEqual
    alphaEqual gammaEqual roundPointEqual

theorem packageReplayAndOutput_identifies_claim_or_failure {program : Program}
    (fits : FitsTwoPow28 program)
    (commitmentSetup : CommitmentSetup program)
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
        (replayState fits commitmentSetup input))
    (claimedStateWellFormed :
      NightstreamFPrime.Layout.Stage1.StateEncoding.WellFormed claimedState)
    (contextDigestEqual :
      (VerifierContext.descriptor (authority fits commitmentSetup)).digest4 =
        (VerifierContext.descriptor claimedAuthority).digest4)
    (stateDigestEqual :
      stateHash (publicFits := PerApplicationFixedPoint.publicFits program)
          (replayState fits commitmentSetup input) =
        stateHash (publicFits := PerApplicationFixedPoint.publicFits program)
          claimedState)
    (alphaEqual :
      ((replayInput fits commitmentSetup input).derive
        (canonicalKey fits commitmentSetup).oracle).alpha =
      (claimedReplay.derive
        (canonicalKey fits commitmentSetup).oracle).alpha)
    (gammaEqual :
      ((replayInput fits commitmentSetup input).derive
        (canonicalKey fits commitmentSetup).oracle).gamma =
      (claimedReplay.derive
        (canonicalKey fits commitmentSetup).oracle).gamma)
    (roundPointEqual :
      ((replayInput fits commitmentSetup input).derive
        (canonicalKey fits commitmentSetup).oracle).roundPoint =
      (claimedReplay.derive
        (canonicalKey fits commitmentSetup).oracle).roundPoint)
    (outgoingStateEqual :
      ((canonicalKey fits commitmentSetup).piCcsExecution
        (selectedRunning input) input.fresh input.nifsProof).outgoingState =
      (canonicalKey fits commitmentSetup).absorbPiCcsOutput
        (claimedReplay.derive
          (canonicalKey fits commitmentSetup).oracle).finalState
        claimedOutput) :
    (authority fits commitmentSetup = claimedAuthority ∧
      replayState fits commitmentSetup input = claimedState ∧
      (replayInput fits commitmentSetup input).authority =
        claimedReplay.authority ∧
      input.nifsProof.piCcsOutput = claimedOutput) ∨
      AuthorityComponentDigestCollision (authority fits commitmentSetup)
        claimedAuthority ∨
      ContextDigestCollision
        (VerifierContext.descriptor (authority fits commitmentSetup))
        (VerifierContext.descriptor claimedAuthority) ∨
      StateHashCollision (replayState fits commitmentSetup input) claimedState ∨
      Spec.Folding.PiCCS.TranscriptReplay.TranscriptReplayCollision
        (canonicalKey fits commitmentSetup).oracle
        (replayInput fits commitmentSetup input)
        claimedReplay ∨
      PiCcsOutputAbsorptionCollision fits commitmentSetup
        ((replayInput fits commitmentSetup input).derive
          (canonicalKey fits commitmentSetup).oracle).finalState
        input.nifsProof.piCcsOutput claimedOutput := by
  rcases packageReplay_identifies_claim_or_failure fits commitmentSetup input
      claimedAuthority claimedState claimedReplay stateWellFormed
      claimedStateWellFormed contextDigestEqual stateDigestEqual alphaEqual
      gammaEqual roundPointEqual with
    identified | componentFailure | contextFailure | stateFailure |
      transcriptFailure
  · rcases identified with ⟨authoritySame, stateSame, replaySame⟩
    have replayDerivedSame :=
      Spec.Folding.PiCCS.TranscriptReplay.ReplayInput.derive_eq_of_authority_eq
        (canonicalKey fits commitmentSetup).oracle
        (replayInput fits commitmentSetup input) claimedReplay replaySame
    have replayExecution :
        (replayInput fits commitmentSetup input).derive
            (canonicalKey fits commitmentSetup).oracle =
          ((canonicalKey fits commitmentSetup).piCcsExecution
            (selectedRunning input) input.fresh input.nifsProof).coins := by
      simpa [canonicalKey] using
        replayInput_derive_eq_piCcsExecution fits commitmentSetup input
    by_cases outputSame : input.nifsProof.piCcsOutput = claimedOutput
    · exact Or.inl ⟨authoritySame, stateSame, replaySame, outputSame⟩
    · apply Or.inr
      apply Or.inr
      apply Or.inr
      apply Or.inr
      apply Or.inr
      refine ⟨outputSame, ?_⟩
      calc
        (canonicalKey fits commitmentSetup).absorbPiCcsOutput
            ((replayInput fits commitmentSetup input).derive
              (canonicalKey fits commitmentSetup).oracle).finalState
            input.nifsProof.piCcsOutput =
            (canonicalKey fits commitmentSetup).absorbPiCcsOutput
              ((canonicalKey fits commitmentSetup).piCcsExecution
                (selectedRunning input) input.fresh input.nifsProof).coins.finalState
              input.nifsProof.piCcsOutput := by
                exact congrArg
                  (fun coins => (canonicalKey fits commitmentSetup
                    ).absorbPiCcsOutput coins.finalState
                      input.nifsProof.piCcsOutput)
                  replayExecution
        _ =
            ((canonicalKey fits commitmentSetup).piCcsExecution
              (selectedRunning input) input.fresh input.nifsProof
            ).outgoingState := by
              rfl
        _ = (canonicalKey fits commitmentSetup).absorbPiCcsOutput
            (claimedReplay.derive
              (canonicalKey fits commitmentSetup).oracle).finalState
            claimedOutput := outgoingStateEqual
        _ = (canonicalKey fits commitmentSetup).absorbPiCcsOutput
            ((replayInput fits commitmentSetup input).derive
              (canonicalKey fits commitmentSetup).oracle).finalState
            claimedOutput := by
              rw [replayDerivedSame]
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

theorem stepHoldsFor_implies_base_or_securityOutcome {program : Program}
    (fits : FitsTwoPow28 program)
    (commitmentSetup : CommitmentSetup program)
    (input : StepInput program fits) (output : StepOutput program)
    (step : Lifecycle.StepHoldsFor
      (PerApplicationFixedPoint.relation program fits)
      (commitmentKey commitmentSetup)
      (verifierContextDigest fits commitmentSetup) program input output) :
    Lifecycle.StepHoldsFor
        (PerApplicationFixedPoint.relation program fits)
        (commitmentKey commitmentSetup)
        (verifierContextDigest fits commitmentSetup) program input output /\
      (input.iteration = 0 \/
        (0 < input.iteration /\
          Spec.Folding.Nifs.PaperSecurityComposition.SecurityOutcome
            (canonicalKey fits commitmentSetup) (selectedRunning input)
            input.fresh input.nifsProof
            (productionExtractionAlgebra fits commitmentSetup)
            (productionStrongSet fits commitmentSetup))) := by
  refine ⟨step, ?_⟩
  change FixedAugmentedTransition
    (Lifecycle.setup (PerApplicationFixedPoint.relation program fits)
      (commitmentKey commitmentSetup)
      (verifierContextDigest fits commitmentSetup))
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
            (canonicalKey fits commitmentSetup) (selectedRunning input)
            input.fresh input.nifsProof =
          some (output.runningNext functionIndex) := by
      simpa [Accepts, Lifecycle.setup,
        Lifecycle.nifsVerifier, canonicalKey, selectedRunning] using selectedNifs
    exact Or.inr ⟨iterationPositive,
      Spec.Folding.Nifs.PaperSecurityComposition.accepted_implies_securityOutcome
        (canonicalKey fits commitmentSetup) (selectedRunning input) input.fresh
        input.nifsProof (output.runningNext functionIndex)
        (productionExtractionAlgebra fits commitmentSetup)
        (productionStrongSet fits commitmentSetup) accepted⟩

theorem verifierBoundRowsZero_implies_base_or_securityOutcome
    {program : Program} (fits : FitsTwoPow28 program)
    (commitmentSetup : CommitmentSetup program)
    (raw : PerApplicationCanonicalAssignment.RawValues program)
    (accepted : (PerApplicationFixedPoint.structuralPlan program fits
      ).RowsZero
        (PerApplicationVerifierBoundAssignment.bind fits commitmentSetup raw
          ).assignment) :
    let bound := PerApplicationVerifierBoundAssignment.bind fits
      commitmentSetup raw
    let input := PerApplicationDecodedIO.input program fits bound
    let output := PerApplicationDecodedIO.output program bound
    Lifecycle.StepHoldsFor
        (PerApplicationFixedPoint.relation program fits)
        (commitmentKey commitmentSetup)
        (verifierContextDigest fits commitmentSetup) program input output ∧
      (input.iteration = 0 ∨
        (0 < input.iteration ∧
          Spec.Folding.Nifs.PaperSecurityComposition.SecurityOutcome
            (canonicalKey fits commitmentSetup) (selectedRunning input)
            input.fresh input.nifsProof
            (productionExtractionAlgebra fits commitmentSetup)
            (productionStrongSet fits commitmentSetup))) := by
  dsimp only
  apply stepHoldsFor_implies_base_or_securityOutcome fits commitmentSetup _ _
  exact PerApplicationFixedPointSoundness.verifierBoundRowsZero_implies_stepHoldsFor
    program fits commitmentSetup raw accepted

def StructuralPackageCollision
    (leftProgram rightProgram : Program)
    (leftFits : FitsTwoPow28 leftProgram)
    (rightFits : FitsTwoPow28 rightProgram) : Prop :=
  sealedPackageValue leftProgram leftFits ≠
      sealedPackageValue rightProgram rightFits ∧
    structuralPackageIdentity leftProgram leftFits =
      structuralPackageIdentity rightProgram rightFits

def FinalPackageBindingCollision
    {leftProgram rightProgram : Program}
    (leftFits : FitsTwoPow28 leftProgram)
    (rightFits : FitsTwoPow28 rightProgram)
    (leftSetup : CommitmentSetup leftProgram)
    (rightSetup : CommitmentSetup rightProgram) : Prop :=
  packageIdentityPreimage leftFits leftSetup ≠
      packageIdentityPreimage rightFits rightSetup ∧
    packageIdentity leftFits leftSetup =
      packageIdentity rightFits rightSetup

theorem packageIdentity_identifies_package_authority_or_collision
    {leftProgram rightProgram : Program}
    (leftFits : FitsTwoPow28 leftProgram)
    (rightFits : FitsTwoPow28 rightProgram)
    (leftSetup : CommitmentSetup leftProgram)
    (rightSetup : CommitmentSetup rightProgram)
    (identityEqual : packageIdentity leftFits leftSetup =
      packageIdentity rightFits rightSetup) :
    (sealedPackageValue leftProgram leftFits =
        sealedPackageValue rightProgram rightFits ∧
      authority leftFits leftSetup = authority rightFits rightSetup) ∨
      StructuralPackageCollision leftProgram rightProgram leftFits rightFits ∨
      AuthorityComponentDigestCollision
        (authority leftFits leftSetup) (authority rightFits rightSetup) ∨
      FinalPackageBindingCollision leftFits rightFits leftSetup rightSetup := by
  by_cases preimageSame : packageIdentityPreimage leftFits leftSetup =
      packageIdentityPreimage rightFits rightSetup
  · have components := packageIdentityPreimage_components leftFits rightFits
      leftSetup rightSetup preimageSame
    rcases descriptor_identifies_authority_or_component_collision
        (authority leftFits leftSetup) (authority rightFits rightSetup)
        components.2 with authoritySame | componentCollision
    · by_cases packageSame : sealedPackageValue leftProgram leftFits =
          sealedPackageValue rightProgram rightFits
      · exact Or.inl ⟨packageSame, authoritySame⟩
      · exact Or.inr (Or.inl ⟨packageSame, components.1⟩)
    · exact Or.inr (Or.inr (Or.inl componentCollision))
  · exact Or.inr (Or.inr (Or.inr ⟨preimageSame, identityEqual⟩))

theorem verificationKeyBinding_identifies_package_authority_or_collision
    {leftProgram rightProgram : Program}
    (leftFits : FitsTwoPow28 leftProgram)
    (rightFits : FitsTwoPow28 rightProgram)
    (leftSetup : CommitmentSetup leftProgram)
    (rightSetup : CommitmentSetup rightProgram)
    (bindingEqual : verificationKeyBinding leftFits leftSetup =
      verificationKeyBinding rightFits rightSetup) :
    (sealedPackageValue leftProgram leftFits =
        sealedPackageValue rightProgram rightFits ∧
      authority leftFits leftSetup = authority rightFits rightSetup) ∨
      StructuralPackageCollision leftProgram rightProgram leftFits rightFits ∨
      AuthorityComponentDigestCollision
        (authority leftFits leftSetup) (authority rightFits rightSetup) ∨
      FinalPackageBindingCollision leftFits rightFits leftSetup rightSetup := by
  apply packageIdentity_identifies_package_authority_or_collision
    leftFits rightFits leftSetup rightSetup
  have packageIdentityEqual := congrArg
    Lifecycle.Stage1.VerificationKey.Binding.packageIdentity bindingEqual
  simpa only [verificationKeyBinding_packageIdentity] using
    packageIdentityEqual

theorem verificationKeyBindingAndRowsZero_implies_securityOrCollision
    {expectedProgram claimedProgram : Program}
    (expectedFits : FitsTwoPow28 expectedProgram)
    (claimedFits : FitsTwoPow28 claimedProgram)
    (expectedSetup : CommitmentSetup expectedProgram)
    (claimedSetup : CommitmentSetup claimedProgram)
    (raw : PerApplicationCanonicalAssignment.RawValues claimedProgram)
    (bindingEqual : verificationKeyBinding expectedFits expectedSetup =
      verificationKeyBinding claimedFits claimedSetup)
    (accepted : (PerApplicationFixedPoint.structuralPlan claimedProgram
      claimedFits).RowsZero
        (PerApplicationVerifierBoundAssignment.bind
          claimedFits claimedSetup raw).assignment) :
    ((sealedPackageValue expectedProgram expectedFits =
          sealedPackageValue claimedProgram claimedFits ∧
        authority expectedFits expectedSetup =
          authority claimedFits claimedSetup) ∧
      (let bound := PerApplicationVerifierBoundAssignment.bind
          claimedFits claimedSetup raw
       let input := PerApplicationDecodedIO.input
          claimedProgram claimedFits bound
       let output := PerApplicationDecodedIO.output claimedProgram bound
       Lifecycle.StepHoldsFor
            (PerApplicationFixedPoint.relation claimedProgram claimedFits)
            (commitmentKey claimedSetup)
            (verifierContextDigest claimedFits claimedSetup)
            claimedProgram input output ∧
          (input.iteration = 0 ∨
            (0 < input.iteration ∧
              Spec.Folding.Nifs.PaperSecurityComposition.SecurityOutcome
                (canonicalKey claimedFits claimedSetup)
                (selectedRunning input) input.fresh input.nifsProof
                (productionExtractionAlgebra claimedFits claimedSetup)
                (productionStrongSet claimedFits claimedSetup))))) ∨
      StructuralPackageCollision expectedProgram claimedProgram
        expectedFits claimedFits ∨
      AuthorityComponentDigestCollision
        (authority expectedFits expectedSetup)
        (authority claimedFits claimedSetup) ∨
      FinalPackageBindingCollision expectedFits claimedFits
        expectedSetup claimedSetup := by
  rcases verificationKeyBinding_identifies_package_authority_or_collision
      expectedFits claimedFits expectedSetup claimedSetup bindingEqual with
    identified | structuralCollision | componentCollision | finalCollision
  · exact Or.inl ⟨identified,
      verifierBoundRowsZero_implies_base_or_securityOutcome
        claimedFits claimedSetup raw accepted⟩
  · exact Or.inr (Or.inl structuralCollision)
  · exact Or.inr (Or.inr (Or.inl componentCollision))
  · exact Or.inr (Or.inr (Or.inr finalCollision))

end NightstreamFPrime.Export.Stage1.PerApplicationSecurity
