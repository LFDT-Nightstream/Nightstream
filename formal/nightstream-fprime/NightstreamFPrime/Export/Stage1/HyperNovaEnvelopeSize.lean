import NightstreamFPrime.Export.Stage1.PerApplicationTerminal

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaEnvelopeSize

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper

abbrev Program := Lifecycle.Stage1.Application.Program

private def assignmentWords (application : Program)
    (_value : PaperAlgebra.Assignment
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application)) : Nat :=
  Fintype.card (Fin (Phi81CarrierLayout.carrierWidth
    (PerApplicationFixedPoint.logicalWidth application)))

def wordCount (application : Program)
    (proof : PerApplicationTerminal.ProofEnvelope application) : Nat :=
  match proof with
  | .bottom => 1
  | .recursive payload =>
      2 + ((List.finRange slotCount).map fun slot =>
        (serializeRunningFields
          (publicFits := PerApplicationFixedPoint.publicFits application)
          (payload.running slot)).length +
        (serializeChildPublicInputs
          (publicFits := PerApplicationFixedPoint.publicFits application)
          (payload.running slot)).length +
        ((List.finRange productionShape.runningCount).map fun source =>
          assignmentWords application (payload.runningWitness slot source)).sum).sum +
      ((List.finRange productionShape.freshCount).map fun source =>
        (block (serializeCommitment (payload.fresh.commitments source))).length +
        (block (serializePublicInput
          (publicFits := PerApplicationFixedPoint.publicFits application)
          (payload.fresh.publicInputs source))).length).sum +
      assignmentWords application payload.freshWitness

def fixedWordBound (application : Program) : Nat :=
  let width := Phi81CarrierLayout.carrierWidth
    (PerApplicationFixedPoint.logicalWidth application)
  2 + slotCount * (27704 + 4320 + productionShape.runningCount * width) +
    productionShape.freshCount *
      (productionProfile.commitmentWidth * ringDegree + 1 +
        ringDegree * publicRingColumns + 1) + width

private theorem wordCount_le (application : Program)
    (proof : PerApplicationTerminal.ProofEnvelope application) :
    wordCount application proof ≤ fixedWordBound application := by
  cases proof with
  | bottom =>
      change 1 ≤ fixedWordBound application
      dsimp only [fixedWordBound]
      omega
  | recursive payload =>
      apply Nat.le_of_eq
      simp [wordCount, assignmentWords, fixedWordBound, serializeRunningFields_length,
        serializeChildPublicInputs_length, fullShape, Phi81Relation.Shape.publicWidth,
        Nat.add_assoc]

theorem accepted_wordCount_le
    (application : Program)
    (fits : PerApplicationTerminal.FitsTwoPow28 application)
    (commitmentSetup : PerApplicationTerminal.CommitmentSetup application)
    (statement : TerminalStatement AppState)
    (proof : PerApplicationTerminal.ProofEnvelope application)
    (accepted : PerApplicationTerminal.Holds application fits commitmentSetup
      statement proof) :
    wordCount application proof ≤ fixedWordBound application ∧
      (match proof with
      | .bottom => True
      | .recursive payload => payload.pc = 1) := by
  refine ⟨wordCount_le application proof, ?_⟩
  cases proof with
  | bottom => trivial
  | recursive payload =>
      obtain ⟨validPc, _⟩ :=
        ((PerApplicationTerminal.holds_recursive_iff application fits
          commitmentSetup statement payload).mp accepted).2.2
      change 1 ≤ payload.pc ∧ payload.pc ≤ 1 at validPc
      exact Nat.le_antisymm validPc.2 validPc.1

end NightstreamFPrime.Export.Stage1.HyperNovaEnvelopeSize
