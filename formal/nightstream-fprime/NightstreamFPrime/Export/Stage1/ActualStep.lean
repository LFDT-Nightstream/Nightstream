import NightstreamFPrime.Export.Stage1.ActualApplicationStep
import NightstreamFPrime.Export.Stage1.ActualRunningTransition
import NightstreamFPrime.Export.Stage1.ActualHashSlots
import NightstreamFPrime.Export.Stage1.ActualPiCCSInputs
import NightstreamFPrime.Lifecycle.Relation

/-!
Owns typed step data and composition on arbitrary accepted assignments.
The immediate context key is decoded from the actual prior preimage. Its
connection to the canonical verifier context remains a separate obligation.
The base theorem permits any fresh claim and NIFS proof because that branch
does not use them. It does not claim the recursive NIFS connection.
-/

namespace NightstreamFPrime.Export.Stage1.ActualStep

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper

def priorState (application : Lifecycle.Stage1.Application.Program)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application)) :
    Nat → F :=
  ActualPreimageFraming.priorState
    (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
      (PerApplicationFixedPoint.geometry application)) assignment

def outputState (application : Lifecycle.Stage1.Application.Program)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application)) :
    Nat → F :=
  ActualPreimageFraming.outputState
    (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
      (PerApplicationFixedPoint.geometry application)) assignment

def contextKey (application : Lifecycle.Stage1.Application.Program)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application)) :
    KeyDigest := StateDecoder.keyDigest (priorState application assignment)

/-- The fresh claim uses exactly the PiCCS leaf's decoded input forms. -/
def decodedFresh (application : Lifecycle.Stage1.Application.Program)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application)) :
    Fresh (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application) :=
  PiCCS.v1_1.Formal.evalFresh
    (PiCCSInvocations.parentInterface (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application))
    PiCCSInputs.phaseOffset
    (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
      (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
        (PerApplicationFixedPoint.geometry application)) assignment))

/-- Read the PiCCS-owned proof fields from their actual forms. The PiDEC
fields remain template data until their separate decoding proof is supplied. -/
def withDecodedPiCCS (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (template : Proof (ProductionKey.degreeBound
      (PerApplicationFixedPoint.relation application fits))) :
    Proof (ProductionKey.degreeBound (PerApplicationFixedPoint.relation application fits)) :=
  PiCCS.v1_1.Formal.evalProof (PerApplicationFixedPoint.relation application fits)
    (PiCCSInvocations.parentInterface (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application))
    PiCCSInputs.phaseOffset
    (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
      (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
        (PerApplicationFixedPoint.geometry application)) assignment)) template

/-- State and application advice are read from the actual assignment.
Fresh and proof are protocol data supplied by the phase decoder. -/
def input (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (fresh : Fresh (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (proof : Proof (ProductionKey.degreeBound
      (PerApplicationFixedPoint.relation application fits))) :
    Input KeyDigest AppState AppWitness
      (Running (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Fresh (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Proof (ProductionKey.degreeBound
        (PerApplicationFixedPoint.relation application fits))) slotCount where
  iteration := StateDecoder.iteration (priorState application assignment)
  z0 := StateDecoder.initialState (priorState application assignment)
  zi := StateDecoder.currentState (priorState application assignment)
  running := fun _ => StateDecoder.running
    (PerApplicationFixedPoint.logicalWidth application)
    (PerApplicationFixedPoint.publicFits application) (priorState application assignment)
  fresh := fresh
  priorPc := 1
  witness := ActualApplicationStep.witness
    (PerApplicationFixedPoint.geometry application) assignment
  nifsProof := proof

def output (application : Lifecycle.Stage1.Application.Program)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (digest : Digest) :
    Output Digest AppState
      (Running (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application)) slotCount where
  zNext := StateDecoder.currentState (outputState application assignment)
  runningNext := fun _ => StateDecoder.running
    (PerApplicationFixedPoint.logicalWidth application)
    (PerApplicationFixedPoint.publicFits application) (outputState application assignment)
  pcNext := functionIndex
  x := digest

/-- The typed recursive input hashes the same prior preimage as the pilot. -/
theorem priorHashPreimage_eq_prior
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (fresh : Fresh (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (proof : Proof (ProductionKey.degreeBound
      (PerApplicationFixedPoint.relation application fits))) :
    priorHashPreimage
      (setup (PerApplicationFixedPoint.relation application fits) ajtai
        (contextKey application assignment))
      (input application fits assignment fresh proof) =
      StateDecoder.preimage (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application)
        (priorState application assignment) := by
  rfl

/-- Typed next-step construction uses exactly the preimage whose hash is
derived from the selected rows, including the prior counter plus one. -/
theorem nextHashPreimage_eq_next
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (fresh : Fresh (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (proof : Proof (ProductionKey.degreeBound
      (PerApplicationFixedPoint.relation application fits)))
    (digest : Digest) :
    nextHashPreimage
      (setup (PerApplicationFixedPoint.relation application fits) ajtai
        (contextKey application assignment))
      (input application fits assignment fresh proof)
      (output application assignment digest) =
      ActualHashSlots.nextPreimage (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application)
        (priorState application assignment) (outputState application assignment) := by
  rfl

/-- The exact decoded base step holds for every accepted assignment and
every value of its unused fresh/proof advice. No representation, generated
assignment, application-correctness, or NIFS-correctness premise is assumed. -/
theorem selectedRowsAndPublic_imply_baseStep
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (fresh : Fresh (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (proof : Proof (ProductionKey.degreeBound
      (PerApplicationFixedPoint.relation application fits)))
    (digest : Digest) (fixed : digest.length = 4)
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest)
    (rows : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment)
    (iterationZero : StateDecoder.iteration (priorState application assignment) = 0) :
    StepHoldsFor (PerApplicationFixedPoint.relation application fits) ajtai
      (contextKey application assignment) application
      (input application fits assignment fresh proof)
      (output application assignment digest) := by
  let geometry := PerApplicationFixedPoint.geometry application
  have publicBound : RecursivePublicOutputPlan.publicInput geometry assignment =
      encHash (publicFits := RecursivePublicOutputPlan.carrierPublicFits geometry) digest := by
    rw [RecursivePublicOutputPlan.publicInput_eq_projectPublicInput]
    exact publicEqual
  have one := RecursivePublicOutputPlan.publicEqual_implies_one
    geometry assignment digest publicBound
  have baseState := ActualRunningTransition.selectedRowsAndPublic_imply_baseState
    application fits assignment digest publicEqual rows iterationZero
  have applicationStep := ActualApplicationStep.selectedRowsZero_implies_decodedStep
    application fits assignment one rows
  have hash := ActualHashSlots.selectedRowsAndPublic_imply_outputHash
    application fits assignment digest fixed publicEqual rows
  change FixedAugmentedTransition
    (setup (PerApplicationFixedPoint.relation application fits) ajtai
      (contextKey application assignment))
    (machineFor (PerApplicationFixedPoint.publicFits application) application)
    functionIndex (input application fits assignment fresh proof)
    (output application assignment digest)
  refine ⟨rfl, applicationStep, ?_, Or.inl ⟨iterationZero, baseState.1, ?_⟩⟩
  · change digest = stateHash
      (nextHashPreimage
        (setup (PerApplicationFixedPoint.relation application fits) ajtai
          (contextKey application assignment))
        (input application fits assignment fresh proof)
        (output application assignment digest))
    rw [nextHashPreimage_eq_next]
    exact hash
  · funext slot
    exact baseState.2

/-- Accepted rows discharge the HyperNova application, hash and state envelope.
The remaining recursive condition is the concrete verifier equation on these
same decoded values. This theorem does not derive that equation from rows or
claim that arbitrary proof advice is accepted. -/
theorem selectedRowsAndPublic_step_iff_baseOrNifs
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (proof : Proof (ProductionKey.degreeBound
      (PerApplicationFixedPoint.relation application fits)))
    (digest : Digest) (fixed : digest.length = 4)
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest)
    (rows : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment) :
    StepHoldsFor (PerApplicationFixedPoint.relation application fits) ajtai
      (contextKey application assignment) application
      (input application fits assignment (decodedFresh application assignment) proof)
      (output application assignment digest) ↔
      StateDecoder.iteration (priorState application assignment) = 0 ∨
        Nifs.PaperNonInteractive.verify
          (ProductionKey.key (PerApplicationFixedPoint.relation application fits) ajtai)
          (StateDecoder.running (PerApplicationFixedPoint.logicalWidth application)
            (PerApplicationFixedPoint.publicFits application)
            (priorState application assignment))
          (decodedFresh application assignment) proof =
          some (StateDecoder.running (PerApplicationFixedPoint.logicalWidth application)
            (PerApplicationFixedPoint.publicFits application)
            (outputState application assignment)) := by
  constructor
  · intro step
    rcases step.2.2.2 with base | ⟨_valid, _positive, _priorHash, nifs, _unchanged⟩
    · exact Or.inl base.1
    · dsimp only [HyperNova.NonInteractiveMultiFold.Accepts, setup, nifsVerifier,
        input, output] at nifs
      exact Or.inr nifs
  · intro branch
    by_cases iterationZero : StateDecoder.iteration (priorState application assignment) = 0
    · exact selectedRowsAndPublic_imply_baseStep application fits ajtai assignment
        (decodedFresh application assignment) proof digest fixed publicEqual rows iterationZero
    have nifs := branch.resolve_left iterationZero
    let geometry := PerApplicationFixedPoint.geometry application
    have publicBound : RecursivePublicOutputPlan.publicInput geometry assignment =
        encHash (publicFits := RecursivePublicOutputPlan.carrierPublicFits geometry) digest := by
      rw [RecursivePublicOutputPlan.publicInput_eq_projectPublicInput]
      exact publicEqual
    have one := RecursivePublicOutputPlan.publicEqual_implies_one
      geometry assignment digest publicBound
    have applicationStep := ActualApplicationStep.selectedRowsZero_implies_decodedStep
      application fits assignment one rows
    have hash := ActualHashSlots.selectedRowsAndPublic_imply_outputHash
      application fits assignment digest fixed publicEqual rows
    change FixedAugmentedTransition
      (setup (PerApplicationFixedPoint.relation application fits) ajtai
        (contextKey application assignment))
      (machineFor (PerApplicationFixedPoint.publicFits application) application)
      functionIndex
      (input application fits assignment (decodedFresh application assignment) proof)
      (output application assignment digest)
    refine ⟨rfl, applicationStep, ?_,
      Or.inr ⟨by change 1 ≤ (1 : Nat) ∧ 1 ≤ 1; exact ⟨le_rfl, le_rfl⟩,
        ?_, ?_, ?_, ?_⟩⟩
    · change digest = stateHash
        (nextHashPreimage
          (setup (PerApplicationFixedPoint.relation application fits) ajtai
            (contextKey application assignment))
          (input application fits assignment (decodedFresh application assignment) proof)
          (output application assignment digest))
      rw [nextHashPreimage_eq_next]
      exact hash
    · exact Nat.pos_of_ne_zero iterationZero
    · change (decodedFresh application assignment).publicInputs ⟨0, by decide⟩ =
        encHash (stateHash
          (priorHashPreimage
            (setup (PerApplicationFixedPoint.relation application fits) ajtai
              (contextKey application assignment))
            (input application fits assignment (decodedFresh application assignment) proof)))
      rw [priorHashPreimage_eq_prior]
      exact ActualPiCCSInputs.selectedRowsZero_implies_freshPublicHash
        application fits assignment one rows ⟨0, by decide⟩
    · dsimp only [HyperNova.NonInteractiveMultiFold.Accepts, setup, nifsVerifier,
        input, output]
      exact nifs
    · intro slot different
      apply False.elim
      apply different
      apply Fin.ext
      have bounded := slot.isLt
      change slot.val < 1 at bounded
      change slot.val = 0
      omega

/-- The concrete NIFS verifier's PiCCS check succeeds on the actual decoded
step inputs and PiCCS proof fields. Later message fields are still arbitrary;
their acceptance and the final running output remain separate obligations. -/
theorem selectedRowsAndPublic_imply_piCcsCheck
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (template : Proof (ProductionKey.degreeBound
      (PerApplicationFixedPoint.relation application fits)))
    (digest : Digest) (fixed : digest.length = 4)
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest)
    (rows : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment) :
    Nifs.PaperNonInteractive.piCcsCheck
      (ProductionKey.key (PerApplicationFixedPoint.relation application fits) ajtai)
      (StateDecoder.running (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application)
        (priorState application assignment))
      (decodedFresh application assignment)
      (withDecodedPiCCS application fits assignment template) = true := by
  have observations := ActualPiCCSInputs.selectedRowsAndPublic_imply_phaseAndHashes
    application fits ajtai template assignment digest fixed publicEqual rows
  have accepted := observations.1.accepted
  have agreement := congrArg
    (fun running => Nifs.PaperNonInteractive.piCcsCheck
      (ProductionKey.key (PerApplicationFixedPoint.relation application fits) ajtai)
      running (decodedFresh application assignment)
      (withDecodedPiCCS application fits assignment template)) observations.2.1
  exact agreement.symm.trans accepted

/-- With the PiCCS proof fields decoded from accepted rows, the recursive
step's unresolved conditions are exactly the concrete PiDEC check over the
verifier-computed PiRLC parent and equality of the computed running output.
Neither condition is assumed or discharged by this equivalence. -/
theorem selectedRowsAndPublic_step_iff_baseOrPiDec
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (template : Proof (ProductionKey.degreeBound
      (PerApplicationFixedPoint.relation application fits)))
    (digest : Digest) (fixed : digest.length = 4)
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest)
    (rows : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment) :
    let key := ProductionKey.key (PerApplicationFixedPoint.relation application fits) ajtai
    let running := StateDecoder.running (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application) (priorState application assignment)
    let fresh := decodedFresh application assignment
    let proof := withDecodedPiCCS application fits assignment template
    let next := StateDecoder.running (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application) (outputState application assignment)
    StepHoldsFor (PerApplicationFixedPoint.relation application fits) ajtai
      (contextKey application assignment) application
      (input application fits assignment fresh proof) (output application assignment digest) ↔
      StateDecoder.iteration (priorState application assignment) = 0 ∨
        (Nifs.PaperNonInteractive.piDecCheck key running fresh proof = true ∧
          key.output running fresh proof = some next) := by
  apply (selectedRowsAndPublic_step_iff_baseOrNifs application fits ajtai assignment
    (withDecodedPiCCS application fits assignment template) digest fixed publicEqual rows).trans
  apply or_congr Iff.rfl
  rw [Nifs.PaperNonInteractive.verify_eq_some_iff]
  exact and_iff_right (selectedRowsAndPublic_imply_piCcsCheck
    application fits ajtai assignment template digest fixed publicEqual rows)

end NightstreamFPrime.Export.Stage1.ActualStep
