import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityConstruction2Encoding
import Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne
import Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs
import Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne

/-!
Contract: fixed-one HyperNova Construction 2 shell for the selected
`PaddedRowIdentity` SuperNeo relation.

Owns: the sole control slot, the concrete paper SuperNeo NIFS setup, the exact
270-field `encHash`, the selected fresh-public projection, exact base
evaluation without a fold, exact recursive evaluation with one NIFS call, and
the concrete terminal checker without a fold.

Does not own: the selected application state and witness types, the application
transition, the Poseidon2 state-hash preimage encoding, or the generated
application compiler. These two functions remain explicit until their source
artifacts exist; this module does not replace them with acceptance premises.

Emits constraints: no.

Assurance tier: concrete protocol shell. The setup, instance encoding, NIFS,
and terminal relations are fixed. The application and state hash are the two
remaining application-owned functions.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConstruction2

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Protocol.FPrime
open Nightstream.SuperNeo.Concrete
open Nightstream.HyperNova.NonInteractiveMultiFold
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityHyperNova

namespace InstanceEncoding
export
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConstruction2Encoding
    (Digest freshPublic encHash encHash_injective)
end InstanceEncoding

namespace Compatibility
export
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityNIVCCompatibility
    (Parameters Structure construction2Setup statementId)
end Compatibility

namespace FixedOne
export Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne
  (Input selected outputFor eval Accepts priorHashPreimage
    accepts_iff_transition)
end FixedOne

namespace FixedOneTerminal
export Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne
  (Proof Accepts eval accepts_iff_transition selected_counter_in_range
    fin_eq_selected)
end FixedOneTerminal

namespace PaperNifs
export Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive (verify)
end PaperNifs

abbrev Digest := InstanceEncoding.Digest
abbrev Key := PaddedRowIdentityHyperNova.VerifierKey
abbrev Running := PaddedRowIdentityHyperNova.PublicRunning
abbrev Fresh := PaddedRowIdentityHyperNova.PublicFresh
abbrev NifsProof := PaddedRowIdentityHyperNova.NifsProof
abbrev RunningWitness := PaddedRowIdentityHyperNova.RunningWitness
abbrev FreshWitness := PaddedRowIdentityHyperNova.Assignment
abbrev Encoded := PaddedRowIdentityConcreteAlgebra.PublicInput
abbrev Parameters := Compatibility.Parameters
abbrev Structure := Compatibility.Structure

abbrev Input (State Witness : Type) :=
  FixedOne.Input State Witness Running Fresh NifsProof

abbrev Output (State : Type) :=
  Nightstream.HyperNova.Construction2.Paper.Output Digest State Running 1

abbrev TerminalProof :=
  FixedOneTerminal.Proof Running RunningWitness Fresh FreshWitness

/-- The one-slot selected setup. Its verifier is the paper SuperNeo NIFS and
its key is bound to the compact compiler description. -/
noncomputable def setup (parameters : Parameters) (system : Structure) :
    Setup Key Running Fresh NifsProof 1 :=
  Compatibility.construction2Setup
    (fun _ => parameters) (fun _ => system)

/-- The fixed protocol part of the selected machine. Only the application
transition and the complete state hash remain application-owned. -/
def machine
    {State Witness : Type}
    (applicationStep : State -> Witness -> State)
    (stateHash : HashPreimage Key State Running 1 -> Digest) :
    Machine Key Digest State Witness Running Fresh Encoded 1 where
  control := fun _ _ => FixedOne.selected
  step := fun _ state witness => applicationStep state witness
  freshPublic := InstanceEncoding.freshPublic
  encodeInstance := InstanceEncoding.encHash
  hash := stateHash

@[simp] theorem machine_control
    {State Witness : Type}
    (applicationStep : State -> Witness -> State)
    (stateHash : HashPreimage Key State Running 1 -> Digest)
    (state : State) (witness : Witness) :
    (machine applicationStep stateHash).control state witness =
      FixedOne.selected :=
  rfl

@[simp] theorem machine_step
    {State Witness : Type}
    (applicationStep : State -> Witness -> State)
    (stateHash : HashPreimage Key State Running 1 -> Digest)
    (state : State) (witness : Witness) :
    (machine applicationStep stateHash).step FixedOne.selected state witness =
      applicationStep state witness :=
  rfl

@[simp] theorem machine_freshPublic
    {State Witness : Type}
    (applicationStep : State -> Witness -> State)
    (stateHash : HashPreimage Key State Running 1 -> Digest)
    (fresh : Fresh) :
    (machine applicationStep stateHash).freshPublic fresh =
      InstanceEncoding.freshPublic fresh :=
  rfl

@[simp] theorem machine_encodeInstance
    {State Witness : Type}
    (applicationStep : State -> Witness -> State)
    (stateHash : HashPreimage Key State Running 1 -> Digest)
    (digest : Digest) :
    (machine applicationStep stateHash).encodeInstance digest =
      InstanceEncoding.encHash digest :=
  rfl

@[simp] theorem setup_verifierKey
    (parameters : Parameters) (system : Structure) :
    (setup parameters system).verifierKeys FixedOne.selected =
      PaddedRowIdentityConcreteNifs.key
        (Compatibility.statementId parameters system)
        parameters.ajtaiKey
        (PaddedRowIdentityCompilerDescription.matrices system) := by
  rfl

/-- The selected setup invokes the exact paper SuperNeo verifier. -/
@[simp] theorem setup_nifs_verify
    (parameters : Parameters) (system : Structure)
    (key : Key) (running : Running) (fresh : Fresh)
    (proof : NifsProof) :
    (setup parameters system).nifs.verify key running fresh proof =
      PaperNifs.verify key running fresh proof :=
  rfl

/-! ## State recovery boundary -/

/-- Exact collision event for the Construction 2 state hash. When the
application supplies the selected Poseidon2 state hash, this is its collision
event; it is not assumed impossible in this model. -/
def StateHashCollision
    {State : Type}
    (stateHash : HashPreimage Key State Running 1 -> Digest) : Prop :=
  exists left right,
    left ≠ right ∧ stateHash left = stateHash right

/-- Equality of the complete 270-field public encodings recovers the full
state-hash preimage or gives an explicit collision in that hash. -/
theorem preimage_eq_or_stateHashCollision
    {State : Type}
    (stateHash : HashPreimage Key State Running 1 -> Digest)
    (left right : HashPreimage Key State Running 1)
    (samePublic :
      InstanceEncoding.encHash (stateHash left) =
        InstanceEncoding.encHash (stateHash right)) :
    left = right ∨ StateHashCollision stateHash := by
  have sameHash : stateHash left = stateHash right :=
    InstanceEncoding.encHash_injective samePublic
  by_cases recovered : left = right
  · exact Or.inl recovered
  · exact Or.inr ⟨left, right, recovered, sameHash⟩

/-- One accepted public link cannot refer to two different prior preimages
unless the state hash has a collision. -/
theorem linked_preimage_eq_or_stateHashCollision
    {State : Type}
    (stateHash : HashPreimage Key State Running 1 -> Digest)
    (fresh : Fresh)
    (left right : HashPreimage Key State Running 1)
    (leftLinked :
      InstanceEncoding.freshPublic fresh =
        InstanceEncoding.encHash (stateHash left))
    (rightLinked :
      InstanceEncoding.freshPublic fresh =
        InstanceEncoding.encHash (stateHash right)) :
    left = right ∨ StateHashCollision stateHash := by
  apply preimage_eq_or_stateHashCollision stateHash left right
  exact leftLinked.symm.trans rightLinked

/-- Pointwise recovery for every fixed constant number of Construction 2
links. The theorem is finite for each caller-selected constant and does not
claim a uniform polynomial-length extractor. -/
theorem constantStep_preimages_eq_or_stateHashCollision
    {State : Type}
    (stateHash : HashPreimage Key State Running 1 -> Digest)
    (stepCount : Nat)
    (left right : Fin stepCount -> HashPreimage Key State Running 1)
    (samePublic : forall step,
      InstanceEncoding.encHash (stateHash (left step)) =
        InstanceEncoding.encHash (stateHash (right step))) :
    left = right ∨ StateHashCollision stateHash := by
  classical
  have sameHash : forall step, stateHash (left step) = stateHash (right step) :=
    fun step => InstanceEncoding.encHash_injective (samePublic step)
  by_cases recovered : left = right
  · exact Or.inl recovered
  · right
    by_contra noCollision
    apply recovered
    funext step
    by_contra different
    exact noCollision ⟨left step, right step, different, sameHash step⟩

/-- The selected executable evaluator accepts exactly the frozen paper
Construction 2 transition. The right side has one fixed slot and no
implementation-only branch. -/
theorem eval_eq_some_iff_transition
    {State Witness : Type}
    [DecidableEq State]
    (parameters : Parameters)
    (system : Structure)
    (applicationStep : State -> Witness -> State)
    (stateHash : HashPreimage Key State Running 1 -> Digest)
    (input : Input State Witness)
    (output : Output State) :
    FixedOne.eval (setup parameters system)
          (machine applicationStep stateHash) input = some output <->
      Transition (setup parameters system) (machine applicationStep stateHash)
        FixedOne.selected
        (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric
          (Key := Key) input)
        output := by
  change
    FixedOne.Accepts (setup parameters system)
          (machine applicationStep stateHash) input output <-> _
  exact FixedOne.accepts_iff_transition
    (setup parameters system) (machine applicationStep stateHash) input output

/-- Exact selected one-step soundness. An accepted fixed-one Construction 2
step satisfies the independent paper NIFS transition, or it exposes one of
the paper SuperNeo NIFS bad events. No caller-supplied NIFS relation or
correctness premise appears in this theorem. -/
theorem accepts_implies_paperTransition_or_nifsBadEvent
    {State Witness : Type}
    [DecidableEq State]
    (parameters : Parameters)
    (system : Structure)
    (applicationStep : State -> Witness -> State)
    (stateHash : HashPreimage Key State Running 1 -> Digest)
    (input : Input State Witness)
    (output : Output State)
    (accepted : FixedOne.Accepts (setup parameters system)
      (machine applicationStep stateHash) input output) :
    Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.SemanticTransition
        (setup parameters system) (machine applicationStep stateHash)
        Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Transition
        FixedOne.selected
        (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric
          (Key := Key) input)
        output ∨
      Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.SelectedNifsBadEvent
        (setup parameters system)
        Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.BadEvent
        (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric
          (Key := Key) input)
        output := by
  apply
    Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs.canonicalFprime_accepts_implies_paperTransition_or_nifsBadEvent
      rfl
      (fun _ => PaddedRowIdentityConcreteNifs.key
        (Compatibility.statementId parameters system)
        parameters.ajtaiKey
        (PaddedRowIdentityCompilerDescription.matrices system))
      PaddedRowIdentityHyperNova.defaultRunning
      (machine applicationStep stateHash)
      FixedOne.selected
      (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric
        (Key := Key) input)
      output
  exact
    (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.accepts_iff_generic
      (setup parameters system) (machine applicationStep stateHash)
      input output).mp accepted

/-- Pointwise composition for any fixed caller-selected step count. Every
accepted step has the independent paper NIFS transition, or at least one
step exposes the selected SuperNeo NIFS bad event. This theorem does not
claim a uniform polynomial-length extractor. -/
theorem constantStep_accepts_implies_paperTransitions_or_nifsBadEvent
    {State Witness : Type}
    [DecidableEq State]
    (parameters : Parameters)
    (system : Structure)
    (applicationStep : State -> Witness -> State)
    (stateHash : HashPreimage Key State Running 1 -> Digest)
    (stepCount : Nat)
    (inputs : Fin stepCount -> Input State Witness)
    (outputs : Fin stepCount -> Output State)
    (accepted : forall step,
      FixedOne.Accepts (setup parameters system)
        (machine applicationStep stateHash) (inputs step) (outputs step)) :
    (forall step,
      Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.SemanticTransition
        (setup parameters system) (machine applicationStep stateHash)
        Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Transition
        FixedOne.selected
        (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric
          (Key := Key) (inputs step))
        (outputs step)) ∨
      exists step,
        Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.SelectedNifsBadEvent
          (setup parameters system)
          Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.BadEvent
          (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric
            (Key := Key) (inputs step))
          (outputs step) := by
  classical
  by_cases everyTransition : forall step,
      Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.SemanticTransition
        (setup parameters system) (machine applicationStep stateHash)
        Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Transition
        FixedOne.selected
        (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric
          (Key := Key) (inputs step))
        (outputs step)
  · exact Or.inl everyTransition
  · right
    push Not at everyTransition
    rcases everyTransition with ⟨step, notTransition⟩
    rcases accepts_implies_paperTransition_or_nifsBadEvent
        parameters system applicationStep stateHash (inputs step)
        (outputs step) (accepted step) with transition | badEvent
    · exact False.elim (notTransition transition)
    · exact ⟨step, badEvent⟩

/-- Fixed-step semantic soundness combined with state-preimage recovery. A
successful trace has all paper NIFS transitions and one recovered preimage
sequence, or it exposes a selected NIFS bad event or a state-hash collision.
The missing application compiler and witness extractor are not hidden in this
result. -/
theorem constantStep_semanticSoundness_and_stateRecovery
    {State Witness : Type}
    [DecidableEq State]
    (parameters : Parameters)
    (system : Structure)
    (applicationStep : State -> Witness -> State)
    (stateHash : HashPreimage Key State Running 1 -> Digest)
    (stepCount : Nat)
    (inputs : Fin stepCount -> Input State Witness)
    (outputs : Fin stepCount -> Output State)
    (accepted : forall step,
      FixedOne.Accepts (setup parameters system)
        (machine applicationStep stateHash) (inputs step) (outputs step))
    (left right : Fin stepCount -> HashPreimage Key State Running 1)
    (samePublic : forall step,
      InstanceEncoding.encHash (stateHash (left step)) =
        InstanceEncoding.encHash (stateHash (right step))) :
    ((forall step,
        Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.SemanticTransition
          (setup parameters system) (machine applicationStep stateHash)
          Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Transition
          FixedOne.selected
          (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric
            (Key := Key) (inputs step))
          (outputs step)) /\ left = right) ∨
      (exists step,
        Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.SelectedNifsBadEvent
          (setup parameters system)
          Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.BadEvent
          (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input.toGeneric
            (Key := Key) (inputs step))
          (outputs step)) ∨
      StateHashCollision stateHash := by
  rcases constantStep_accepts_implies_paperTransitions_or_nifsBadEvent
      parameters system applicationStep stateHash stepCount inputs outputs
      accepted with everyTransition | badEvent
  · rcases constantStep_preimages_eq_or_stateHashCollision
      stateHash stepCount left right samePublic with recovered | collision
    · exact Or.inl ⟨everyTransition, recovered⟩
    · exact Or.inr (Or.inr collision)
  · exact Or.inr (Or.inl badEvent)

/-- Construction 2 base completeness. The result contains only the
application step, default running value, and next state hash. The NIFS
verifier is not evaluated. -/
theorem base_complete_without_fold
    {State Witness : Type}
    [DecidableEq State]
    (parameters : Parameters)
    (system : Structure)
    (applicationStep : State -> Witness -> State)
    (stateHash : HashPreimage Key State Running 1 -> Digest)
    (input : Input State Witness)
    (iterationZero : input.iteration = 0)
    (initialState : input.z0 = input.zi) :
    FixedOne.eval (setup parameters system)
        (machine applicationStep stateHash) input =
      some (FixedOne.outputFor (setup parameters system)
        (machine applicationStep stateHash) input
        (fun _ => (setup parameters system).defaultRunning)) := by
  classical
  simp [FixedOne.eval, iterationZero, initialState]

/-- Construction 2 recursive completeness for one supplied paper NIFS
message. The fixed-one evaluator executes this one verifier call and installs
its unique result. -/
theorem recursive_complete_with_one_fold
    {State Witness : Type}
    [DecidableEq State]
    (parameters : Parameters)
    (system : Structure)
    (applicationStep : State -> Witness -> State)
    (stateHash : HashPreimage Key State Running 1 -> Digest)
    (input : Input State Witness)
    (nextRunning : Running)
    (iterationNonzero : input.iteration ≠ 0)
    (priorLinked :
      InstanceEncoding.freshPublic input.fresh =
        InstanceEncoding.encHash
          (stateHash
            (FixedOne.priorHashPreimage (setup parameters system) input)))
    (foldAccepted :
      PaperNifs.verify
          ((setup parameters system).verifierKeys FixedOne.selected)
          (input.running FixedOne.selected) input.fresh input.nifsProof =
        some nextRunning) :
    FixedOne.eval (setup parameters system)
        (machine applicationStep stateHash) input =
      some (FixedOne.outputFor (setup parameters system)
        (machine applicationStep stateHash) input
        (fun _ => nextRunning)) := by
  classical
  unfold FixedOne.eval
  rw [if_neg iterationNonzero]
  have linked :
      (machine applicationStep stateHash).freshPublic input.fresh =
        (machine applicationStep stateHash).encodeInstance
          ((machine applicationStep stateHash).hash
            (FixedOne.priorHashPreimage (setup parameters system) input)) := by
    exact priorLinked
  rw [if_pos linked]
  rw [setup_nifs_verify, foldAccepted]

/-- The exact selected terminal evaluator. It checks the prior link and the
CE/CCS relations on positive iterations, and it never invokes NIFS. -/
noncomputable def terminalAccepts
    {State Witness : Type}
    [DecidableEq State]
    (parameters : Parameters)
    (system : Structure)
    (applicationStep : State -> Witness -> State)
    (stateHash : HashPreimage Key State Running 1 -> Digest)
    (statement : TerminalStatement State)
    (proof : TerminalProof) : Prop :=
  FixedOneTerminal.Accepts (setup parameters system)
    (machine applicationStep stateHash)
    (terminalRelations (slotCount := 1))
    (terminalChecks (slotCount := 1)) statement proof

/-- The selected terminal checker accepts exactly the frozen paper terminal
transition. This equation also shows that terminal verification has no NIFS
call. -/
theorem terminalAccepts_iff_transition
    {State Witness : Type}
    [DecidableEq State]
    (parameters : Parameters)
    (system : Structure)
    (applicationStep : State -> Witness -> State)
    (stateHash : HashPreimage Key State Running 1 -> Digest)
    (statement : TerminalStatement State)
    (proof : TerminalProof) :
    terminalAccepts parameters system applicationStep stateHash statement
        proof <->
      TerminalTransition (setup parameters system)
        (machine applicationStep stateHash)
        (terminalRelations (slotCount := 1)) statement
        (Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Proof.toGeneric
          proof) := by
  unfold terminalAccepts
  exact FixedOneTerminal.accepts_iff_transition
    (setup parameters system) (machine applicationStep stateHash)
    (terminalRelations (slotCount := 1))
    (terminalChecks (slotCount := 1)) statement proof

/-- Terminal base acceptance checks only equality with the initial state and
performs no fold or relation check. -/
theorem terminal_base_accepts_without_fold
    {State Witness : Type}
    [DecidableEq State]
    (parameters : Parameters)
    (system : Structure)
    (applicationStep : State -> Witness -> State)
    (stateHash : HashPreimage Key State Running 1 -> Digest)
    (statement : TerminalStatement State)
    (proof : TerminalProof)
    (iterationZero : statement.iteration = 0)
    (initialState : statement.zi = statement.z0) :
    terminalAccepts parameters system applicationStep stateHash statement
      proof := by
  classical
  unfold terminalAccepts FixedOneTerminal.Accepts
  simp [FixedOneTerminal.eval, iterationZero, initialState]

/-- Positive terminal acceptance from the exact prior link and the selected
running and fresh relation witnesses. The terminal verifier performs no
additional NIFS call. -/
theorem terminal_recursive_accepts_without_fold
    {State Witness : Type}
    [DecidableEq State]
    (parameters : Parameters)
    (system : Structure)
    (applicationStep : State -> Witness -> State)
    (stateHash : HashPreimage Key State Running 1 -> Digest)
    (statement : TerminalStatement State)
    (proof : TerminalProof)
    (iterationPositive : 0 < statement.iteration)
    (priorLinked :
      InstanceEncoding.freshPublic proof.fresh =
        InstanceEncoding.encHash (stateHash {
          verifierKeys := (setup parameters system).verifierKeys
          iteration := statement.iteration
          z0 := statement.z0
          current := statement.zi
          running := fun _ => proof.running
          pc := 1
        }))
    (runningValid :
      TerminalRunningHolds
        ((setup parameters system).verifierKeys FixedOne.selected)
        proof.running proof.runningWitness)
    (freshValid :
      TerminalFreshHolds
        ((setup parameters system).verifierKeys FixedOne.selected)
        proof.fresh proof.freshWitness) :
    terminalAccepts parameters system applicationStep stateHash statement
      proof := by
  classical
  unfold terminalAccepts
  apply (FixedOneTerminal.accepts_iff_transition
    (setup parameters system) (machine applicationStep stateHash)
    (terminalRelations (slotCount := 1))
    (terminalChecks (slotCount := 1)) statement proof).2
  right
  refine ⟨FixedOneTerminal.selected_counter_in_range, iterationPositive,
    ?_, ?_, ?_⟩
  · exact priorLinked
  · intro slot
    rw [FixedOneTerminal.fin_eq_selected slot]
    exact runningValid
  · exact freshValid

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConstruction2
