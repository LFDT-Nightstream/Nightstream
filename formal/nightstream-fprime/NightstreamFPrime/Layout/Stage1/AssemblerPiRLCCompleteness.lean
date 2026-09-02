import NightstreamFPrime.Layout.Stage1.AssemblerCompleteness
import NightstreamFPrime.Layout.Stage1.AssemblerSoundness
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.GeneratedSupport
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.PhaseDeterminism

/-!
Owns the exact PiCCS-to-PiRLC semantic transport and the next opaque-child
append for the compact Stage 1 assembler. It adds no row, challenge, or
alternate verifier predicate.
-/

namespace NightstreamFPrime.Layout.Stage1.AssemblerPiRLCCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem piRlcPhase_of_piCcs
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation))
    (left right : Env)
    (externalAgrees : ∀ index, PiCCSOrdinarySourceSupport.External index →
      left index = right index)
    (localAgrees : ∀ index, AssemblerInputs.piRlcOffset program ≤ index →
      left index = right index)
    (piCcsLeft : PiCCS.v1_1.Formal.PhaseHolds relation ajtai
      (AssemblerInputs.piCcsInterface program)
      (AssemblerInputs.piCcsOffset program) left template)
    (piCcsRight : PiCCS.v1_1.Formal.PhaseHolds relation ajtai
      (AssemblerInputs.piCcsInterface program)
      (AssemblerInputs.piCcsOffset program) right template)
    (piRlcLeft : PiRLC.v1_1.Semantics.PhaseHolds relation ajtai
      (AssemblerInputs.piRlcInterface relation program)
      (AssemblerInputs.piRlcOffset program) left) :
    PiRLC.v1_1.Semantics.PhaseHolds relation ajtai
        (AssemblerInputs.piRlcInterface relation program)
        (AssemblerInputs.piRlcOffset program) right ∧
      PiRLC.v1_1.Semantics.attempt relation
          (AssemblerInputs.piRlcInterface relation program)
          (AssemblerInputs.piRlcOffset program) left =
        PiRLC.v1_1.Semantics.attempt relation
          (AssemblerInputs.piRlcInterface relation program)
          (AssemblerInputs.piRlcOffset program) right := by
  let piCcsInterface := AssemblerInputs.piCcsInterface
    (logicalWidth := logicalWidth) (publicFits := publicFits) program
  let piCcsOffset := AssemblerInputs.piCcsOffset program
  let support := AssemblerCompleteness.piCcsExternalSupport
    (logicalWidth := logicalWidth) (publicFits := publicFits) program
  have runningEq :=
    PiCCS.v1_1.Formal.PhaseTransport.evalRunning_eq_of_agree_satisfy
      piCcsInterface piCcsOffset PiCCSOrdinarySourceSupport.External
      left right support externalAgrees
  have freshEq :=
    PiCCS.v1_1.Formal.PhaseTransport.evalFresh_eq_of_agree_satisfy
      piCcsInterface piCcsOffset PiCCSOrdinarySourceSupport.External
      left right support externalAgrees
  have proofEq :=
    PiCCS.v1_1.Formal.PhaseTransport.evalProof_eq_of_agree_satisfy
      relation piCcsInterface piCcsOffset
      PiCCSOrdinarySourceSupport.External left right template support
      externalAgrees
  let stageInterface := AssemblerInputs.interface relation program
  let leftProof := AssemblerSoundness.nifsProofValue stageInterface template
    piCcsOffset (AssemblerInputs.piDecOffset program) left
  let rightProof := AssemblerSoundness.nifsProofValue stageInterface template
    piCcsOffset (AssemblerInputs.piDecOffset program) right
  have roundsEq : leftProof.piCcsRounds = rightProof.piCcsRounds := by
    simpa [leftProof, rightProof, stageInterface,
      AssemblerSoundness.nifsProofValue] using
        congrArg (fun proof => proof.piCcsRounds) proofEq
  have outputEq : leftProof.piCcsOutput = rightProof.piCcsOutput := by
    simpa [leftProof, rightProof, stageInterface,
      AssemblerSoundness.nifsProofValue] using
        congrArg (fun proof => proof.piCcsOutput) proofEq
  let key := ProductionKey.key relation ajtai
  have executionEq :
      key.piCcsExecution
          (PiCCS.v1_1.Formal.evalRunning piCcsInterface piCcsOffset left)
          (PiCCS.v1_1.Formal.evalFresh piCcsInterface piCcsOffset left)
          leftProof =
        key.piCcsExecution
          (PiCCS.v1_1.Formal.evalRunning piCcsInterface piCcsOffset right)
          (PiCCS.v1_1.Formal.evalFresh piCcsInterface piCcsOffset right)
          rightProof := by
    rw [runningEq, freshEq]
    unfold Key.piCcsExecution Key.piCcsCertificate
    rw [roundsEq, outputEq]
  have outputsEq :
      key.piCcsOutputs
          (PiCCS.v1_1.Formal.evalRunning piCcsInterface piCcsOffset left)
          (PiCCS.v1_1.Formal.evalFresh piCcsInterface piCcsOffset left)
          leftProof =
        key.piCcsOutputs
          (PiCCS.v1_1.Formal.evalRunning piCcsInterface piCcsOffset right)
          (PiCCS.v1_1.Formal.evalFresh piCcsInterface piCcsOffset right)
          rightProof := by
    rw [runningEq, freshEq]
    unfold Key.piCcsOutputs Key.piCcsProbe Key.piCcsExecution
      Key.piCcsCertificate
    rw [roundsEq, outputEq]
  have inputsEq : PiRLC.v1_1.Semantics.evalInputs relation
      (AssemblerInputs.piRlcInterface relation program)
      (AssemblerInputs.piRlcOffset program) left =
    PiRLC.v1_1.Semantics.evalInputs relation
      (AssemblerInputs.piRlcInterface relation program)
      (AssemblerInputs.piRlcOffset program) right := by
    calc
      _ = key.piCcsOutputs
          (PiCCS.v1_1.Formal.evalRunning piCcsInterface piCcsOffset left)
          (PiCCS.v1_1.Formal.evalFresh piCcsInterface piCcsOffset left)
          leftProof := by
        simpa [piCcsInterface, piCcsOffset, stageInterface, leftProof, key] using
          AssemblerSoundness.compactPiRlcInputs_eq_keyOutputs relation ajtai
            program left template piCcsLeft
      _ = key.piCcsOutputs
          (PiCCS.v1_1.Formal.evalRunning piCcsInterface piCcsOffset right)
          (PiCCS.v1_1.Formal.evalFresh piCcsInterface piCcsOffset right)
          rightProof := outputsEq
      _ = _ := by
        symm
        simpa [piCcsInterface, piCcsOffset, stageInterface, rightProof, key] using
          AssemblerSoundness.compactPiRlcInputs_eq_keyOutputs relation ajtai
            program right template piCcsRight
  have initialStateEq : PiRLC.v1_1.SamplerChain.evalInitialState
      (PiRLC.v1_1.Formal.samplerInterface
        (PiRLC.v1_1.Formal.atOffset
          (AssemblerInputs.piRlcInterface relation program)
          (AssemblerInputs.piRlcOffset program)))
      (PiRLC.v1_1.Formal.samplerOffset
        (AssemblerInputs.piRlcOffset program)) left =
    PiRLC.v1_1.SamplerChain.evalInitialState
      (PiRLC.v1_1.Formal.samplerInterface
        (PiRLC.v1_1.Formal.atOffset
          (AssemblerInputs.piRlcInterface relation program)
          (AssemblerInputs.piRlcOffset program)))
      (PiRLC.v1_1.Formal.samplerOffset
        (AssemblerInputs.piRlcOffset program)) right := by
    calc
      _ = (key.piCcsExecution
          (PiCCS.v1_1.Formal.evalRunning piCcsInterface piCcsOffset left)
          (PiCCS.v1_1.Formal.evalFresh piCcsInterface piCcsOffset left)
          leftProof).outgoingState := by
        simpa [piCcsInterface, piCcsOffset, stageInterface, leftProof, key] using
          AssemblerSoundness.compactPiRlcInitialState_eq_key relation ajtai
            program left template piCcsLeft
      _ = (key.piCcsExecution
          (PiCCS.v1_1.Formal.evalRunning piCcsInterface piCcsOffset right)
          (PiCCS.v1_1.Formal.evalFresh piCcsInterface piCcsOffset right)
          rightProof).outgoingState := congrArg
            (fun execution => execution.outgoingState) executionEq
      _ = _ := by
        symm
        simpa [piCcsInterface, piCcsOffset, stageInterface, rightProof, key] using
          AssemblerSoundness.compactPiRlcInitialState_eq_key relation ajtai
            program right template piCcsRight
  have roundPointEq :
      PiCCS.v1_1.RoundTranscript.evalRoundPoint
          (PiCCS.v1_1.Formal.roundTranscriptInterface
            (PiCCS.v1_1.Formal.atOffset piCcsInterface piCcsOffset))
          (PiCCS.v1_1.Formal.roundTranscriptOffset piCcsInterface piCcsOffset)
          left =
        PiCCS.v1_1.RoundTranscript.evalRoundPoint
          (PiCCS.v1_1.Formal.roundTranscriptInterface
            (PiCCS.v1_1.Formal.atOffset piCcsInterface piCcsOffset))
          (PiCCS.v1_1.Formal.roundTranscriptOffset piCcsInterface piCcsOffset)
          right := by
    calc
      _ = (key.piCcsExecution
          (PiCCS.v1_1.Formal.evalRunning piCcsInterface piCcsOffset left)
          (PiCCS.v1_1.Formal.evalFresh piCcsInterface piCcsOffset left)
          leftProof).coins.roundPoint := by
        simpa [piCcsInterface, piCcsOffset, stageInterface, leftProof, key,
          AssemblerSoundness.nifsProofValue] using piCcsLeft.roundPoint
      _ = (key.piCcsExecution
          (PiCCS.v1_1.Formal.evalRunning piCcsInterface piCcsOffset right)
          (PiCCS.v1_1.Formal.evalFresh piCcsInterface piCcsOffset right)
          rightProof).coins.roundPoint := congrArg
            (fun execution => execution.coins.roundPoint) executionEq
      _ = _ := by
        symm
        simpa [piCcsInterface, piCcsOffset, stageInterface, rightProof, key,
          AssemblerSoundness.nifsProofValue] using piCcsRight.roundPoint
  have pointEq : PiCCS.v1_1.StatementAbsorption.evalPoint
      (AssemblerInputs.piCcsRoundPoint
        (logicalWidth := logicalWidth) (publicFits := publicFits) program) left =
    PiCCS.v1_1.StatementAbsorption.evalPoint
      (AssemblerInputs.piCcsRoundPoint
        (logicalWidth := logicalWidth) (publicFits := publicFits) program) right := by
    calc
      _ = PiCCS.v1_1.RoundTranscript.evalRoundPoint
          (PiCCS.v1_1.Formal.roundTranscriptInterface
            (PiCCS.v1_1.Formal.atOffset piCcsInterface piCcsOffset))
          (PiCCS.v1_1.Formal.roundTranscriptOffset piCcsInterface piCcsOffset)
          left := by
        simpa [piCcsInterface, piCcsOffset] using
          AssemblerSoundness.compactPoint_eq_roundTranscript
            (logicalWidth := logicalWidth) (publicFits := publicFits)
            program left
      _ = _ := roundPointEq
      _ = PiCCS.v1_1.StatementAbsorption.evalPoint
          (AssemblerInputs.piCcsRoundPoint
            (logicalWidth := logicalWidth) (publicFits := publicFits) program) right := by
        symm
        simpa [piCcsInterface, piCcsOffset] using
          AssemblerSoundness.compactPoint_eq_roundTranscript
            (logicalWidth := logicalWidth) (publicFits := publicFits)
            program right
  let piRlcInterface := AssemblerInputs.piRlcInterface relation program
  let piRlcOffset := AssemblerInputs.piRlcOffset program
  let samplerInterface := PiRLC.v1_1.Formal.samplerInterface
    (PiRLC.v1_1.Formal.atOffset piRlcInterface piRlcOffset)
  have samplerRight :=
    PiRLC.v1_1.SamplerChain.relationHolds_of_initial_and_agree_from
      samplerInterface (PiRLC.v1_1.Formal.samplerOffset piRlcOffset)
      left right (by simpa [samplerInterface, piRlcInterface, piRlcOffset]
        using initialStateEq) (by
          intro index bounded
          apply localAgrees index
          simpa [piRlcOffset, PiRLC.v1_1.Formal.samplerOffset] using bounded)
      (by simpa [samplerInterface, piRlcInterface, piRlcOffset] using
        piRlcLeft.sampler)
  have challengesEq :=
    PiRLC.v1_1.SamplerChain.evalChallenges_eq_of_agree_from
      samplerInterface (PiRLC.v1_1.Formal.samplerOffset piRlcOffset)
      left right (by
        intro index bounded
        apply localAgrees index
        simpa [piRlcOffset, PiRLC.v1_1.Formal.samplerOffset] using bounded)
  have outputValueEq := PiRLC.v1_1.Semantics.evalOutput_eq_of_point_and_agree_from
    relation piRlcInterface piRlcOffset left right (by
      simpa [piRlcInterface, AssemblerInputs.piRlcInterface] using pointEq)
    localAgrees
  have attemptEq := PiRLC.v1_1.Semantics.attempt_eq_of_components relation
    piRlcInterface piRlcOffset left right inputsEq (by
      simpa [PiRLC.v1_1.Semantics.evalChallenges, piRlcInterface, piRlcOffset,
        samplerInterface] using challengesEq) outputValueEq
  exact ⟨PiRLC.v1_1.Semantics.PhaseHolds.of_attempt_eq relation ajtai
    piRlcInterface piRlcOffset left right samplerRight attemptEq piRlcLeft,
    attemptEq⟩

/-- Honest completion through the PiRLC parent child. The parent keeps PiRLC
opaque while the canonical PiRLC builder completes its seven internal
children. -/
theorem completePiRlcPrefix
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation))
    (env : Env)
    (specification : Lifecycle.Stage1.SpecHolds relation ajtai program
      (AssemblerInputs.interface relation program) template
      (AssemblerInputs.rootOffset program) env) :
    ∃ completed : Sequence.Prefix env (AssemblerInputs.rootOffset program),
      completed.operations =
        [Lifecycle.Stage1.childOp "stage1.prior_state_hash"
          (Lifecycle.Stage1.priorChild relation program
            (AssemblerInputs.interface relation program))
          (AssemblerInputs.priorOffset program),
        Lifecycle.Stage1.childOp "stage1.output_hash"
          (Lifecycle.Stage1.outputHashChild relation program
            (AssemblerInputs.interface relation program))
          (AssemblerInputs.outputHashOffset program),
        Lifecycle.Stage1.childOp "stage1.piccs.v1_1"
          (Lifecycle.Stage1.piCcsChild relation ajtai program
            (AssemblerInputs.interface relation program) template)
          (AssemblerInputs.piCcsOffset program),
        Lifecycle.Stage1.childOp "stage1.pirlc.v1_1"
          (Lifecycle.Stage1.piRlcChild relation ajtai program
            (AssemblerInputs.interface relation program))
          (AssemblerInputs.piRlcOffset program)] ∧
      AssemblerInputs.rootOffset program +
          localLength completed.operations =
        AssemblerInputs.piDecOffset program ∧
      PiRLC.v1_1.Semantics.attempt relation
          (AssemblerInputs.piRlcInterface relation program)
          (AssemblerInputs.piRlcOffset program) env =
        PiRLC.v1_1.Semantics.attempt relation
          (AssemblerInputs.piRlcInterface relation program)
          (AssemblerInputs.piRlcOffset program) completed.current := by
  rcases AssemblerCompleteness.completePiCcsPrefix relation ajtai program
      template env specification with
    ⟨p3, p3Operations, p3End⟩
  have piCcsInitial := specification.piCcs
  change PiCCS.v1_1.Formal.PhaseHolds relation ajtai
    (AssemblerInputs.piCcsInterface program)
    (Lifecycle.Stage1.piCcsOffset relation program
      (AssemblerInputs.interface relation program)
      (AssemblerInputs.rootOffset program)) env template at piCcsInitial
  rw [AssemblerInputs.parent_piCcsOffset_eq relation program] at piCcsInitial
  have p3Holds := holdsFlat_implies_holds p3.current p3.operations p3.rows
  have piCcsCurrent : PiCCS.v1_1.Formal.PhaseHolds relation ajtai
      (AssemblerInputs.piCcsInterface program)
      (AssemblerInputs.piCcsOffset program) p3.current template := by
    have selected := p3Holds
      (Lifecycle.Stage1.childOp "stage1.piccs.v1_1"
        (Lifecycle.Stage1.piCcsChild relation ajtai program
          (AssemblerInputs.interface relation program) template)
        (AssemblerInputs.piCcsOffset program))
      (by rw [p3Operations]; simp)
    change (Lifecycle.Stage1.piCcsChild relation ajtai program
        (AssemblerInputs.interface relation program) template).assumptions
          (AssemblerInputs.piCcsOffset program) p3.current →
      (Lifecycle.Stage1.piCcsChild relation ajtai program
        (AssemblerInputs.interface relation program) template).spec
          (AssemblerInputs.piCcsOffset program) p3.current at selected
    exact selected (AssemblerBounds.piCcsAssumptions relation program p3.current)
  have piRlcInitial := specification.piRlc
  change PiRLC.v1_1.Semantics.PhaseHolds relation ajtai
    (AssemblerInputs.piRlcInterface relation program)
    (Lifecycle.Stage1.piRlcOffset relation ajtai program
      (AssemblerInputs.interface relation program) template
      (AssemblerInputs.rootOffset program)) env at piRlcInitial
  rw [AssemblerInputs.parent_piRlcOffset_eq relation ajtai program template]
    at piRlcInitial
  have externalAgrees : ∀ index,
      PiCCSOrdinarySourceSupport.External index →
        env index = p3.current index := by
    intro index external
    symm
    apply p3.agrees index
    apply Or.inl
    have sourceBound :=
      PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount
        (PiCCSOrdinarySourceSupport.external_source index external)
    rw [Spartan.sourceColumnCount_eq] at sourceBound
    rw [AssemblerPilotBounds.rootOffset_eq]
    omega
  have localAgrees : ∀ index, AssemblerInputs.piRlcOffset program ≤ index →
      env index = p3.current index := by
    intro index generated
    symm
    apply p3.agrees index
    apply Or.inr
    rw [p3End]
    exact generated
  rcases piRlcPhase_of_piCcs relation ajtai program template env
      p3.current externalAgrees localAgrees piCcsInitial piCcsCurrent
      piRlcInitial with
    ⟨phase, envToP3Attempt⟩
  have assumptions := AssemblerBounds.piRlcAssumptions relation program p3.current
  rcases PiRLC.v1_1.Formal.completePrefix relation ajtai
      (AssemblerInputs.piRlcInterface relation program) p3.current
      (AssemblerInputs.piRlcOffset program) assumptions phase with
    ⟨built, builtOperations⟩
  let child := Lifecycle.Stage1.piRlcChild relation ajtai program
    (AssemblerInputs.interface relation program)
  have childMain : child.main = PiRLC.v1_1.Formal.main relation
      (AssemblerInputs.piRlcInterface relation program) := by
    rfl
  have childOperations : built.operations = Circuit.ops child.main
      (AssemblerInputs.piRlcOffset program) := by
    rw [childMain, PiRLC.v1_1.Formal.main_ops]
    exact builtOperations
  have childScope : ∀ expression ∈ flatConstraints
      (Circuit.ops child.main (AssemblerInputs.piRlcOffset program)),
      expression.VarsBelow
        (AssemblerInputs.piRlcOffset program + localLength
          (Circuit.ops child.main (AssemblerInputs.piRlcOffset program))) := by
    rw [← childOperations]
    exact built.scope
  have childAgrees : AgreesOutside p3.current built.current
      (AssemblerInputs.piRlcOffset program)
      (localLength
        (Circuit.ops child.main (AssemblerInputs.piRlcOffset program))) := by
    rw [← childOperations]
    exact built.agrees
  have childRows : holdsFlat built.current
      (Circuit.ops child.main (AssemblerInputs.piRlcOffset program)) := by
    rw [← childOperations]
    exact built.rows
  rcases Sequence.appendBuiltAt p3 "stage1.pirlc.v1_1" child
      (AssemblerInputs.piRlcOffset program) p3End childScope built.current
      childAgrees childRows with
    ⟨p4, p4Operations, p4End, p3to4, piRlcRows⟩
  have p4Phase : PiRLC.v1_1.Semantics.PhaseHolds relation ajtai
      (AssemblerInputs.piRlcInterface relation program)
      (AssemblerInputs.piRlcOffset program) p4.current := by
    change child.spec (AssemblerInputs.piRlcOffset program) p4.current
    exact child.soundness p4.current (AssemblerInputs.piRlcOffset program)
      (AssemblerBounds.piRlcAssumptions relation program p4.current)
      (holdsFlat_implies_holds p4.current _ piRlcRows)
  have belowAgrees : ∀ index,
      index < AssemblerInputs.piRlcOffset program →
        p3.current index = p4.current index := by
    intro index below
    symm
    apply p3to4.values index
    rw [p3End]
    exact below
  have inputsP3P4 := AssemblerBounds.piRlcInputs_eq_of_agree_below
    relation program p3.current p4.current belowAgrees
  have initialStateP3P4 :=
    AssemblerBounds.piRlcInitialState_eq_of_agree_below relation program
      p3.current p4.current belowAgrees
  have pointP3P4 := AssemblerBounds.piRlcPoint_eq_of_agree_below
    relation program p3.current p4.current belowAgrees
  have outputPointP3P4 :
      (PiRLC.v1_1.Semantics.evalOutput relation
          (AssemblerInputs.piRlcInterface relation program)
          (AssemblerInputs.piRlcOffset program) p3.current).point =
        (PiRLC.v1_1.Semantics.evalOutput relation
          (AssemblerInputs.piRlcInterface relation program)
          (AssemblerInputs.piRlcOffset program) p4.current).point := by
    simpa [PiRLC.v1_1.Semantics.evalOutput,
      PiRLC.v1_1.OutputBinding.evalOutput,
      PiRLC.v1_1.Formal.outputBindingInterface,
      PiRLC.v1_1.Formal.atOffset, AssemblerInputs.piRlcInterface] using
        pointP3P4
  have challengesP3P4 :=
    PiRLC.v1_1.Semantics.PhaseHolds.challenges_eq_of_initialState_eq
      phase p4Phase initialStateP3P4
  have outputP3P4 :=
    PiRLC.v1_1.Semantics.PhaseHolds.evalOutput_eq_of_shared relation ajtai
      (AssemblerInputs.piRlcInterface relation program)
      (AssemblerInputs.piRlcOffset program) p3.current p4.current phase p4Phase
      inputsP3P4 initialStateP3P4 outputPointP3P4
  have attemptP3P4 := PiRLC.v1_1.Semantics.attempt_eq_of_components relation
    (AssemblerInputs.piRlcInterface relation program)
    (AssemblerInputs.piRlcOffset program) p3.current p4.current inputsP3P4
    challengesP3P4 outputP3P4
  refine ⟨p4, ?_, ?_, envToP3Attempt.trans attemptP3P4⟩
  · rw [p4Operations, p3Operations]
    rfl
  · calc
      _ = AssemblerInputs.piRlcOffset program + localLength
          (Circuit.ops child.main (AssemblerInputs.piRlcOffset program)) :=
        p4End
      _ = Lifecycle.Stage1.piDecOffset relation ajtai program
          (AssemblerInputs.interface relation program) template
          (AssemblerInputs.rootOffset program) := by
        unfold Lifecycle.Stage1.piDecOffset
        rw [AssemblerInputs.parent_piRlcOffset_eq relation ajtai program template]
        rw [child.privateCount_eq]
      _ = AssemblerInputs.piDecOffset program :=
        AssemblerInputs.parent_piDecOffset_eq relation ajtai program template

end NightstreamFPrime.Layout.Stage1.AssemblerPiRLCCompleteness
