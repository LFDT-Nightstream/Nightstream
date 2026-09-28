import NightstreamFPrime.Export.Stage1.ActualStep
import NightstreamFPrime.Export.Stage1.ActualPiDEC
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
import NightstreamFPrime.Layout.Stage1.AssemblerInputs
import tests.AxiomAudit

/-!
Review artifact: a conditional proof plan for the selected hash-chain package.
This file changes no production definition or circuit. Its open premises are
the exact recursive PiDEC check, computed output, and selected-context equality.
It is not a completed production soundness theorem.
-/

namespace FPrimeProve2Review

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev application := Poseidon2HashChainV1Package.application
abbrev fits := Poseidon2HashChainV1Package.fits
abbrev width := PerApplicationFixedPoint.logicalWidth application
abbrev publicFits := PerApplicationFixedPoint.publicFits application
abbrev relation := PerApplicationFixedPoint.relation application fits
abbrev ArbitraryAssignment := Assignment F width

def ccsEnv (assignment : ArbitraryAssignment) : Env :=
  Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
    (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
      (PerApplicationFixedPoint.geometry application)) assignment)

def decAttempt (assignment : ArbitraryAssignment) :=
  PiDEC.v1_1.Semantics.inputAttempt relation
    (PiDECArithmetic.phaseInterface width publicFits) PiDECInputs.phaseOffset
    (Spartan.pullback (ActualPiDEC.decodedEnv
      (ActualPiDEC.selectedGeometry application) assignment))

/-- Every proof field is read from the existing assignment-derived owners.
There is no raw packet, encoding premise, or caller-supplied proof template. -/
def decodedProof (assignment : ArbitraryAssignment) :
    Proof (ProductionKey.degreeBound relation) where
  piCcsRounds := fun round =>
    ((PiCCSInvocations.parentInterface width publicFits).round
      PiCCSInputs.phaseOffset round).semanticPolynomial (ccsEnv assignment)
  piCcsOutput := PiCCS.v1_1.Formal.evalOutput
    (PiCCSInvocations.parentInterface width publicFits)
    PiCCSInputs.phaseOffset (ccsEnv assignment)
  piDecCommitments := fun child =>
    ((decAttempt assignment).messages (AssemblerInputs.childOfRunning child)).commitment
  piDecEvaluations := fun child =>
    ((decAttempt assignment).messages (AssemblerInputs.childOfRunning child)
      ).evaluations.getD 0 evaluationZero

theorem decodedProof_piCcs_fixed (assignment : ArbitraryAssignment) :
    ActualStep.withDecodedPiCCS application fits assignment
      (decodedProof assignment) = decodedProof assignment := by
  rfl

def decodedRunning (assignment : ArbitraryAssignment) :=
  StateDecoder.running width publicFits (ActualStep.priorState application assignment)

def decodedNext (assignment : ArbitraryAssignment) :=
  StateDecoder.running width publicFits (ActualStep.outputState application assignment)

def decodedInput (assignment : ArbitraryAssignment) :=
  ActualStep.input application fits assignment
    (ActualStep.decodedFresh application assignment) (decodedProof assignment)

def decodedOutput (assignment : ArbitraryAssignment) (digest : Digest) :=
  ActualStep.output application assignment digest

noncomputable def selectedContext : KeyDigest :=
  PerApplicationCanonicalPackage.verifierContextDigest fits
    Poseidon2HashChainV1Setup.productionSetup

noncomputable def selectedKey :=
  ProductionKey.key relation Poseidon2HashChainV1Setup.productionAjtaiKey

/-- These are exact missing claims, not new authority records. Each still
needs a proof from the selected rows and the actual acceptance boundary. -/
noncomputable def PiDECCheck (assignment : ArbitraryAssignment) : Prop :=
  Nifs.PaperNonInteractive.piDecCheck selectedKey (decodedRunning assignment)
    (ActualStep.decodedFresh application assignment) (decodedProof assignment) = true

noncomputable def ComputedOutput (assignment : ArbitraryAssignment) : Prop :=
  selectedKey.output (decodedRunning assignment)
    (ActualStep.decodedFresh application assignment) (decodedProof assignment) =
      some (decodedNext assignment)

noncomputable def SelectedContext (assignment : ArbitraryAssignment) : Prop :=
  ActualStep.contextKey application assignment = selectedContext

/-- A checked composition plan. The three named premises remain OPEN.
The theorem proves their sufficiency for one full decoded step, not their
truth for accepted rows, package conformance, or terminal-chain security. -/
theorem rootSketch
    (assignment : ArbitraryAssignment) (digest : Digest)
    (fixed : digest.length = 4)
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape width publicFits)
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := publicFits) digest)
    (rows : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment)
    (context : SelectedContext assignment)
    (piDec : StateDecoder.iteration (ActualStep.priorState application assignment) ≠ 0 →
      PiDECCheck assignment)
    (computed : StateDecoder.iteration (ActualStep.priorState application assignment) ≠ 0 →
      ComputedOutput assignment) :
    StepHoldsFor relation Poseidon2HashChainV1Setup.productionAjtaiKey
      selectedContext application (decodedInput assignment)
      (decodedOutput assignment digest) := by
  have reduction := ActualStep.selectedRowsAndPublic_step_iff_baseOrPiDec
    application fits Poseidon2HashChainV1Setup.productionAjtaiKey assignment
    (decodedProof assignment) digest fixed publicEqual rows
  dsimp only at reduction
  rw [decodedProof_piCcs_fixed] at reduction
  have branch : StateDecoder.iteration (ActualStep.priorState application assignment) = 0 ∨
      (PiDECCheck assignment ∧ ComputedOutput assignment) := by
    by_cases zero : StateDecoder.iteration (ActualStep.priorState application assignment) = 0
    · exact Or.inl zero
    · exact Or.inr ⟨piDec zero, computed zero⟩
  have step := reduction.mpr branch
  change ActualStep.contextKey application assignment = selectedContext at context
  rw [context] at step
  exact step

#audit_axioms decodedProof_piCcs_fixed
#audit_axioms rootSketch

end FPrimeProve2Review
