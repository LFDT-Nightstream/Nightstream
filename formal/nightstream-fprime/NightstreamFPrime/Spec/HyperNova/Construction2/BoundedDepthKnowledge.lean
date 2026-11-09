import NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth

/-! The reverse extraction of HyperNova Lemma 17 (Appendix H.3) for option B, without
probabilities. `StepKnowledge` is Definition 7 for one fold: from an output witness at norm `β`,
the input claim has a witness at `Schedule.extractedBefore step β` and the fresh instance has a
witness, or the fold's failure event holds. The light and refresh security theorems give this
from forks with the Π_CCS bad events as the failure; the random-oracle layer bounds the failure
probability, as HyperNova errata Assumption 1 does for Stage 1. The contract is true for a
failure such as "the input has no witness", so these theorems say something only for a failure
event whose probability that layer bounds.

Run backward over the folds a slot receives, the contracts give every running claim a witness at
its recovered bound, at most `B + cap · T · (b − 1)`, or they name a failed fold. The relaxed
binding of `Phi81Relation.PiRLCAlgebra.Binding.chainCollision_to_shortKernel` covers openings at
these bounds. The history is an input here; reading it from the step witnesses is the
probabilistic layer's work. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthKnowledge

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold
open NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepth
open NightstreamFPrime.Spec.Folding.Schedule (Step extractedBefore recoveredBound recoveredBounds)

universe uKey uClaim uFresh uProof uClaimWitness uFreshWitness

variable {Key : Type uKey} {Claim : Type uClaim} {Fresh : Type uFresh} {Proof : Type uProof}
  {ClaimWitness : Type uClaimWitness} {FreshWitness : Type uFreshWitness}

/-- HyperNova Definition 7 for one fold, without probabilities. -/
def StepKnowledge (p : GlobalParams) (folds : Folds Key Claim Fresh Proof)
    (Holds : Nat → Claim → ClaimWitness → Prop) (FreshHolds : Fresh → FreshWitness → Prop)
    (Failure : Step → Key → Claim → Fresh → Proof → ClaimWitness → Prop) : Prop :=
  ∀ step key input fresh proof output outputWitness bound,
    folds step key input fresh proof = some output → Holds bound output outputWitness →
      (∃ inputWitness freshWitness, Holds (extractedBefore p step bound) input inputWitness ∧
          FreshHolds fresh freshWitness) ∨
        Failure step key input fresh proof outputWitness

/-- The folds a slot receives, from running claim `start` to `final`. Each fold reads the claim
the previous fold left. -/
def Runs (cap : Nat) (folds : Folds Key Claim Fresh Proof) :
    Running Claim → List (Fold Key Claim Fresh Proof) → Running Claim → Prop
  | start, [], final => final = start
  | start, fold :: rest, final =>
      fold.input = start.1 ∧
        Accepts (verifier cap folds fold.circuit) fold.key start fold.fresh fold.proof
          fold.output ∧
        Runs cap folds fold.output rest final

theorem Runs.acceptedFrom {cap : Nat} {folds : Folds Key Claim Fresh Proof} :
    ∀ (history : List (Fold Key Claim Fresh Proof)) (start final : Running Claim),
      Runs cap folds start history final → AcceptedFrom cap folds start.2 history
  | [], _, _, _ => trivial
  | fold :: rest, start, final, ⟨same, accepted, later⟩ => by
      refine ⟨?_, Runs.acceptedFrom rest fold.output final later⟩
      rw [same]
      exact accepted

/-- Every running claim of the history has a witness at its recovered bound, and every fresh
instance has a witness. -/
def ValidHistory (p : GlobalParams) (Holds : Nat → Claim → ClaimWitness → Prop)
    (FreshHolds : Fresh → FreshWitness → Prop) :
    Claim → List (Fold Key Claim Fresh Proof) → Nat → Prop
  | claim, [], bound => ∃ witness, Holds bound claim witness
  | claim, fold :: rest, bound =>
      (∃ witness, Holds (recoveredBound p bound (steps (fold :: rest))) claim witness) ∧
        (∃ freshWitness, FreshHolds fold.fresh freshWitness) ∧
        ValidHistory p Holds FreshHolds fold.output.1 rest bound

theorem ValidHistory.head {p : GlobalParams} {Holds : Nat → Claim → ClaimWitness → Prop}
    {FreshHolds : Fresh → FreshWitness → Prop} :
    ∀ (history : List (Fold Key Claim Fresh Proof)) (claim : Claim) (bound : Nat),
      ValidHistory p Holds FreshHolds claim history bound →
        ∃ witness, Holds (recoveredBound p bound (steps history)) claim witness
  | [], _, _, valid => valid
  | _ :: _, _, _, valid => valid.1

/-- Some fold's failure event holds at an output witness that is valid at the fold's recovered
output bound. -/
def Failed (p : GlobalParams) (Holds : Nat → Claim → ClaimWitness → Prop)
    (Failure : Step → Key → Claim → Fresh → Proof → ClaimWitness → Prop) (bound : Nat) :
    List (Fold Key Claim Fresh Proof) → Prop
  | [] => False
  | fold :: rest => Failed p Holds Failure bound rest ∨
      ∃ outputWitness, Holds (recoveredBound p bound (steps rest)) fold.output.1 outputWitness ∧
        Failure (circuitStep fold.circuit) fold.key fold.input fold.fresh fold.proof outputWitness

variable {p : GlobalParams} {cap : Nat} {folds : Folds Key Claim Fresh Proof}
  {Holds : Nat → Claim → ClaimWitness → Prop} {FreshHolds : Fresh → FreshWitness → Prop}
  {Failure : Step → Key → Claim → Fresh → Proof → ClaimWitness → Prop}

/-- From a witness of the final claim at `bound`, the step contracts give a valid history or a
failed fold. -/
theorem reverse_extracts (knowledge : StepKnowledge p folds Holds FreshHolds Failure)
    (bound : Nat) :
    ∀ (history : List (Fold Key Claim Fresh Proof)) (start final : Running Claim)
      (finalWitness : ClaimWitness),
      Runs cap folds start history final → Holds bound final.1 finalWitness →
        ValidHistory p Holds FreshHolds start.1 history bound ∨
          Failed p Holds Failure bound history
  | [], start, final, finalWitness, runs, holds => by
      left
      simp only [Runs] at runs
      subst runs
      exact ⟨finalWitness, holds⟩
  | fold :: rest, start, final, finalWitness, ⟨same, accepted, later⟩, holds => by
      rcases reverse_extracts knowledge bound rest fold.output final finalWitness later holds with
        valid | failed
      · obtain ⟨outputWitness, outputHolds⟩ := ValidHistory.head rest fold.output.1 bound valid
        obtain ⟨_, folded, _⟩ := (accepts_iff cap folds fold.circuit fold.key start.1 start.2
          fold.fresh fold.proof fold.output).mp accepted
        rcases knowledge _ _ _ _ _ _ _ _ folded outputHolds with
          ⟨inputWitness, freshWitness, inputHolds, freshHolds⟩ | failure
        · exact Or.inl ⟨⟨inputWitness, inputHolds⟩, ⟨freshWitness, freshHolds⟩, valid⟩
        · refine Or.inr (Or.inr ⟨outputWitness, outputHolds, ?_⟩)
          rw [same]
          exact failure
      · exact Or.inr (Or.inl failed)

/-- With the final claim checked below `B`, every recovered bound of a valid history is at most
`B + cap · T · (b − 1)`. -/
theorem reverse_extracts_within (knowledge : StepKnowledge p folds Holds FreshHolds Failure)
    (history : List (Fold Key Claim Fresh Proof)) (start final : Running Claim) (bound : Nat)
    (finalWitness : ClaimWitness) (runs : Runs cap folds start history final)
    (holds : Holds bound final.1 finalWitness) (finalAtMostB : bound ≤ p.bigB) :
    (ValidHistory p Holds FreshHolds start.1 history bound ∧
        ∀ recovered ∈ recoveredBounds p bound (steps history),
          recovered ≤ p.bigB + cap * (p.expansionT * (p.b - 1))) ∨
      Failed p Holds Failure bound history := by
  have admissible := admissible_of_accepted cap folds history start.2
    (Runs.acceptedFrom history start final runs)
  rcases reverse_extracts knowledge bound history start final finalWitness runs holds with
    valid | failed
  · exact Or.inl ⟨valid,
      Folding.Schedule.recoveredBounds_le p finalAtMostB (steps history) start.2 admissible⟩
  · exact Or.inr failed

section TwoSlot

variable {Digest State StepWitness Encoded : Type}

/-- One recursive step of the two-slot machine. -/
structure MachineStep (Key : Type uKey) (Claim : Type uClaim) (Fresh : Type uFresh)
    (Proof : Type uProof) (Digest State StepWitness : Type) where
  functionIndex : Fin 2
  input : Paper.Input Key State StepWitness (Running Claim) Fresh Proof 2
  output : Paper.Output Digest State (Running Claim) 2

/-- Recursive steps of the two-slot machine from running claims `start` to `final`, each
reading the running claims the previous step left. -/
def Execution (setup : PerCircuit.Setup Key (Running Claim) Fresh Proof 2)
    (machine : Paper.Machine Key Digest State StepWitness (Running Claim) Fresh Encoded 2) :
    (Fin 2 → Running Claim) →
      List (MachineStep Key Claim Fresh Proof Digest State StepWitness) →
      (Fin 2 → Running Claim) → Prop
  | start, [], final => final = start
  | start, step :: rest, final =>
      step.input.running = start ∧ 0 < step.input.iteration ∧
        PerCircuit.Transition setup machine step.functionIndex step.input step.output ∧
        Execution setup machine step.output.runningNext rest final

/-- The fold a step makes on `slot` when its prior program counter selects that slot. -/
def slotFold (verifierKeys : Fin 2 → Key) (slot : Fin 2)
    (step : MachineStep Key Claim Fresh Proof Digest State StepWitness) :
    Fold Key Claim Fresh Proof where
  circuit := step.functionIndex
  key := verifierKeys slot
  input := (step.input.running slot).1
  fresh := step.input.fresh
  proof := step.input.nifsProof
  output := step.output.runningNext slot

/-- The folds `slot` receives: one for each step whose prior program counter selects it. -/
def slotHistory (verifierKeys : Fin 2 → Key) (slot : Fin 2) :
    List (MachineStep Key Claim Fresh Proof Digest State StepWitness) →
      List (Fold Key Claim Fresh Proof)
  | [] => []
  | step :: rest =>
      if step.input.priorPc = slot.val + 1 then
        slotFold verifierKeys slot step :: slotHistory verifierKeys slot rest
      else slotHistory verifierKeys slot rest

/-- Each slot of an accepted execution receives an accepted run of folds (HyperNova
Construction 2, step 4: the other slots do not change). -/
theorem slot_runs (verifierKeys : Fin 2 → Key) (defaultClaim : Claim)
    (machine : Paper.Machine Key Digest State StepWitness (Running Claim) Fresh Encoded 2)
    (slot : Fin 2) :
    ∀ (trace : List (MachineStep Key Claim Fresh Proof Digest State StepWitness))
      (start final : Fin 2 → Running Claim),
      Execution (setup cap folds verifierKeys defaultClaim) machine start trace final →
        Runs cap folds (start slot) (slotHistory verifierKeys slot trace) (final slot)
  | [], start, final, execution => by
      simp only [Execution] at execution
      subst execution
      rfl
  | step :: rest, start, final, ⟨runningEq, recursive, holds, later⟩ => by
      have tail := slot_runs verifierKeys defaultClaim machine slot rest step.output.runningNext
        final later
      obtain ⟨priorPcValid, accepted, unchanged⟩ := PerCircuit.recursive_selected
        (setup cap folds verifierKeys defaultClaim) machine step.functionIndex step.input
        step.output holds recursive
      have lower := priorPcValid.1
      by_cases selects : step.input.priorPc = slot.val + 1
      · have same : Paper.selectedIndex priorPcValid = slot := by
          apply Fin.ext
          simp only [Paper.selectedIndex]
          omega
        rw [same] at accepted
        simp only [slotHistory, selects, ↓reduceIte]
        refine ⟨?_, ?_, tail⟩
        · simp only [slotFold, runningEq]
        · rw [← runningEq]
          exact accepted
      · have other : Paper.selectedIndex priorPcValid ≠ slot := by
          intro equal
          apply selects
          have value := congrArg Fin.val equal
          simp only [Paper.selectedIndex] at value
          omega
        have kept : step.output.runningNext slot = start slot :=
          (unchanged slot (Ne.symm other)).trans (congrFun runningEq slot)
        simp only [slotHistory, selects, ↓reduceIte]
        rw [kept] at tail
        exact tail

/-- HyperNova Lemma 17's reverse extraction for option B on one slot of an accepted execution:
from a witness of the slot's final claim below `B`, every claim the slot held has a witness at a
bound of at most `B + cap · T · (b − 1)` and every fresh instance it folded has a witness, or a
fold of the slot failed. -/
theorem execution_extracts (knowledge : StepKnowledge p folds Holds FreshHolds Failure)
    (verifierKeys : Fin 2 → Key) (defaultClaim : Claim)
    (machine : Paper.Machine Key Digest State StepWitness (Running Claim) Fresh Encoded 2)
    (trace : List (MachineStep Key Claim Fresh Proof Digest State StepWitness))
    (start final : Fin 2 → Running Claim)
    (execution : Execution (setup cap folds verifierKeys defaultClaim) machine start trace final)
    (slot : Fin 2) (bound : Nat) (finalWitness : ClaimWitness)
    (holds : Holds bound (final slot).1 finalWitness) (finalAtMostB : bound ≤ p.bigB) :
    let history := slotHistory verifierKeys slot trace
    (ValidHistory p Holds FreshHolds (start slot).1 history bound ∧
        ∀ recovered ∈ recoveredBounds p bound (steps history),
          recovered ≤ p.bigB + cap * (p.expansionT * (p.b - 1))) ∨
      Failed p Holds Failure bound history :=
  reverse_extracts_within knowledge _ (start slot) (final slot) bound finalWitness
    (slot_runs verifierKeys defaultClaim machine slot trace start final execution) holds
    finalAtMostB

end TwoSlot

end NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthKnowledge
