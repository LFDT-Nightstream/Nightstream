import NightstreamFPrime.Export.Stage1.CheckedReplayStep
import NightstreamFPrime.Export.Stage1.HyperNovaAcceptedNext
import NightstreamFPrime.Export.Stage1.PerApplicationFixedPointSoundness

/-! Accept the exact canonical successor packet after a checked local replay.
Input/context custody are ABI links to the actual packet. Its output and digest
are derived from row soundness and deterministic transition uniqueness. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.CheckedReplaySuccessor

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Layout.Stage1
open PerApplicationCanonicalAssignment
open Poseidon2HashChainV1Package (application fits)
open Poseidon2HashChainV1Setup (productionSetup productionAjtaiKey)

private theorem output_ext
    {Digest State Running : Type} {slots : Nat}
    (left right : Output Digest State Running slots)
    (state : left.zNext = right.zNext)
    (running : left.runningNext = right.runningNext)
    (pc : left.pcNext = right.pcNext) (digest : left.x = right.x) : left = right := by
  cases left
  cases right
  cases state
  cases running
  cases pc
  cases digest
  rfl

/-- Determinism uses the actual NIFS verifier function. No hash injectivity,
source extraction, or probabilistic assumption is involved. -/
theorem output_unique
    {Key Digest State Witness Running Fresh Proof Encoded : Type} {slots : Nat}
    (configuration : Setup Key Running Fresh Proof slots)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slots)
    (index : Fin slots) (input : Input Key State Witness Running Fresh Proof slots)
    (left right : Output Digest State Running slots)
    (leftStep : FixedAugmentedTransition configuration machine index input left)
    (rightStep : FixedAugmentedTransition configuration machine index input right) :
    left = right := by
  have pc : left.pcNext = right.pcNext := leftStep.1.trans rightStep.1.symm
  have state : left.zNext = right.zNext := leftStep.2.1.trans rightStep.2.1.symm
  have running : left.runningNext = right.runningNext := by
    rcases leftStep.2.2.2 with baseLeft | ⟨validLeft, positiveLeft, _, nifsLeft, unchangedLeft⟩
    · rcases rightStep.2.2.2 with baseRight | ⟨_, positiveRight, _⟩
      · exact baseLeft.2.2.trans baseRight.2.2.symm
      · omega
    · rcases rightStep.2.2.2 with baseRight | ⟨validRight, _, _, nifsRight, unchangedRight⟩
      · omega
      · have selected : selectedIndex validLeft = selectedIndex validRight := rfl
        rw [← selected] at nifsRight unchangedRight
        have folded : left.runningNext (selectedIndex validLeft) =
            right.runningNext (selectedIndex validLeft) :=
          Option.some.inj (nifsLeft.symm.trans nifsRight)
        funext slot
        by_cases same : slot = selectedIndex validLeft
        · simpa only [same] using folded
        · exact (unchangedLeft slot same).trans (unchangedRight slot same).symm
  have preimage : nextHashPreimage configuration input left =
      nextHashPreimage configuration input right := by
    simp only [nextHashPreimage, state, running, pc]
  exact output_ext left right state running pc
    (leftStep.2.2.1.trans ((congrArg machine.hash preimage).trans rightStep.2.2.1.symm))

def nextStatement (statement : PerApplicationTerminal.Statement) (advice : AppWitness) :
    PerApplicationTerminal.Statement where
  iteration := statement.iteration + 1
  z0 := statement.z0
  zi := application.step statement.zi advice

/-- One exact successor: the sixteen returned child witnesses and the same
canonical fresh carrier used by the commitment and public input. -/
noncomputable def payload
    (result : Running (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (children : Stage1.Terminal.RunningWitness
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (raw : RawValues application) : PerApplicationTerminal.Payload
        Poseidon2HashChainV1Package.application where
  running := fun _ => result
  runningWitness := fun _ => children
  fresh := {
    commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit
      productionAjtaiKey raw.completeAssignment
    publicInputs := fun _ => encHash raw.outputDigest }
  freshWitness := raw.completeAssignment
  pc := 1

/-- The checked local replay and the actual canonical row packet yield an
accepted exact successor. Rows and norm are natural assignment premises.
Child CE membership is supplied by the source/R/D opening kernels, never by
public D acceptance alone. Input and context custody are explicit ABI links;
no output equality or accepted successor is a premise. -/
theorem accepted_of_checked_rows
    (statement : PerApplicationTerminal.Statement)
    (input : PiCCSInputCheck.Input) (batch : PiRLCParent.Batch)
    (parent : PiRLCParent.Values) (messages : PiDECInputCheck.Messages)
    (runningWitness : Stage1.Terminal.RunningWitness
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (freshWitness : Stage1.Terminal.FreshWitness
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (advice : AppWitness) (raw : RawValues application)
    (children : Stage1.Terminal.RunningWitness
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (sampled : PiRLCInputCheck.sampled input = some batch)
    (returned : PiRLCParent.computedParent input batch = some parent)
    (checked : PiDECInputCheck.accepted parent messages = true)
    (accepted : PerApplicationTerminal.Holds application fits productionSetup
      statement (.recursive (CheckedReplayStep.prior input runningWitness freshWitness)))
    (nonwrap : statement.iteration + 1 < goldilocksModulus)
    (rows : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero raw.assignment)
    (bounded : ∀ column, centeredMagnitude (raw.completeAssignment column) < 2)
    (inputCustody : PerApplicationDecodedIO.input application fits raw =
      HyperNovaStepData.input statement (CheckedReplayStep.prior input runningWitness freshWitness)
        advice (CheckedReplayNifs.proof input messages))
    (contextCustody : PerApplicationDecodedIO.contextKey raw =
      PerApplicationCanonicalPackage.verifierContextDigest fits productionSetup)
    (childOpenings : ∀ child,
      CE.Holds (semantics productionAjtaiKey) productionGlobalParams
        (Lifecycle.runningStatement (PerApplicationFixedPoint.relation application fits)
          (PiCCSInputCheck.runningFromInput messages) child) (children child)) :
    PerApplicationTerminal.Holds application fits productionSetup
      (nextStatement statement advice)
      (.recursive (payload (PiCCSInputCheck.runningFromInput messages) children raw)) := by
  have selectedStep := (CheckedReplayStep.checked_step statement input batch parent messages
    runningWitness freshWitness advice sampled returned checked accepted nonwrap).1
  have rawStep := PerApplicationFixedPointSoundness.rowsZero_implies_stepHoldsFor
    application fits productionAjtaiKey raw rows
  rw [inputCustody, contextCustody] at rawStep
  have outputEq := output_unique
    (setup (PerApplicationFixedPoint.relation application fits) productionAjtaiKey
      (PerApplicationCanonicalPackage.verifierContextDigest fits productionSetup))
    (machineFor (PerApplicationFixedPoint.publicFits application) application) functionIndex
    (HyperNovaStepData.input statement (CheckedReplayStep.prior input runningWitness freshWitness)
      advice (CheckedReplayNifs.proof input messages))
    (PerApplicationDecodedIO.output application raw)
    (HyperNovaStepData.output statement advice (PiCCSInputCheck.runningFromInput messages))
    rawStep selectedStep
  have digest : raw.outputDigest =
      (HyperNovaStepData.output statement advice (PiCCSInputCheck.runningFromInput messages)).x :=
    congrArg (fun output => output.x) outputEq
  have priorValid := ((PerApplicationTerminal.holds_recursive_iff application fits productionSetup
    statement (CheckedReplayStep.prior input runningWitness freshWitness)).mp accepted).1
  have valid : Stage1.Terminal.StatementValid (nextStatement statement advice) :=
    ⟨nonwrap, priorValid.2.1, Stage1.Poseidon2HashChainV1.step_output_length statement.zi advice⟩
  exact HyperNovaAcceptedNext.terminal_of_memberships application fits productionSetup
    (nextStatement statement advice) (PiCCSInputCheck.runningFromInput messages) children raw
    valid (Nat.zero_lt_succ _) digest childOpenings
    (HyperNovaAcceptedNext.freshHolds_of_rows application fits productionAjtaiKey raw rows bounded)

end NightstreamFPrime.Export.Stage1.CheckedReplaySuccessor
