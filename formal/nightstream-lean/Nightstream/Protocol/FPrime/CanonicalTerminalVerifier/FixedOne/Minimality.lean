import Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne
import Nightstream.SuperNeo.CheckPlan

/-!
Inclusion-minimality for the paper-only fixed-one terminal verifier.

Assurance tier: model-level.

Owns: an explicit typed one-slot terminal model, four ordinary removal
counterexamples, and inclusion-minimality of the four obligations retained by
the payload-minimal terminal normal form.

Does not own: a global lower bound across arithmetizations, SuperNeo concrete
relations, Rust, R1CS, lowering, costs, Poseidon2/Ajtai internals, or physical
row removal.

Emits constraints: no.

The scope is deliberately the typed fixed-one terminal normal form.  Its base
branch retains the endpoint equality.  Its recursive branch retains the prior
public link, the sole running relation, and the selected fresh relation.  The
one-based counter and generic all-slot conjunction are derived from `Fin 1`
and are recorded as eliminated structural facts, not retained checks.

| Stage path | Mathematical obligation | Status | Lean owner |
| `fprime.terminal.fixed_one.base_endpoint` | base endpoint agrees with `z₀` | retained | `Family.baseEndpoint` |
| `fprime.terminal.fixed_one.prior_link` | recursive fresh public input has the exact prior preimage | retained | `Family.priorPublicLink` |
| `fprime.terminal.fixed_one.running` | sole running instance/witness relation holds | retained | `Family.runningRelation` |
| `fprime.terminal.fixed_one.fresh` | selected fresh instance/witness relation holds | retained | `Family.freshRelation` |
| `fprime.terminal.fixed_one.counter` | `pc = 1` is derived from the profile | eliminated | `oneBasedCounter_derived` |
| `fprime.terminal.fixed_one.all_slots` | generic finite conjunction reduces to the sole slot | eliminated | `allRunningSlots_derived` |
| `fprime.terminal.fixed_one.minimality` | every retained leaf has a concrete non-transition removal witness | model-level | `inclusionMinimalSound` |
-/

namespace Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Minimality

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.SuperNeo

namespace Model

/-- A small, fully typed paper model.  It is a countermodel fixture, not an
implementation or an arithmetization. -/
def setup : Setup Unit Unit Bool Unit 1 where
  verifierKeys := fun _ => ()
  nifs := { verify := fun _ _ _ _ => some () }
  defaultRunning := ()

/-- The public link exposes the fresh Boolean directly, while the expected
digest is always `false`.  This makes a broken link observable without using
any concrete hash implementation. -/
def machine : Machine Unit Bool Bool Unit Unit Bool Bool 1 where
  control := fun _ _ => selected
  step := fun _ state _ => state
  freshPublic := fun fresh => fresh
  encodeInstance := fun digest => digest
  hash := fun _ => false

/-- Relation truth is controlled independently by its witness bit. -/
def relations : TerminalRelations Unit Unit Bool Bool Bool 1 where
  runningHolds := fun _ _ _ witness => witness = false
  freshHolds := fun _ _ _ witness => witness = false

/-- Boolean checkers exact for the two independently stated relation
predicates. -/
def relationChecks : RelationChecks relations where
  runningCheck := fun _ _ _ witness => decide (witness = false)
  freshCheck := fun _ _ _ witness => decide (witness = false)
  runningCheck_iff := by
    intro slot key value witness
    simp [relations]
  freshCheck_iff := by
    intro slot key value witness
    simp [relations]

/-- One complete typed terminal candidate. -/
structure Candidate where
  statement : TerminalStatement Bool
  proof : Proof Unit Bool Bool Bool

/-- Build a candidate while making each countermodel mutation visible at its
owned field. -/
def candidate
    (iteration : Nat) (z0 zi fresh runningWitness freshWitness : Bool) :
    Candidate where
  statement := { iteration := iteration, z0 := z0, zi := zi }
  proof := {
    running := ()
    runningWitness := runningWitness
    fresh := fresh
    freshWitness := freshWitness
  }

/-- Base endpoint countermodel: only the base endpoint is wrong. -/
def baseEndpointWitness : Candidate :=
  candidate 0 false true false false false

/-- Recursive countermodel: only the fresh public value is wrong. -/
def priorPublicLinkWitness : Candidate :=
  candidate 1 false false true false false

/-- Recursive countermodel: only the sole running witness is wrong. -/
def runningRelationWitness : Candidate :=
  candidate 1 false false false true false

/-- Recursive countermodel: only the selected fresh witness is wrong. -/
def freshRelationWitness : Candidate :=
  candidate 1 false false false false true

end Model

/-- The complete fixed-one terminal review vocabulary.  The final two entries
are structural facts rather than retained terminal checks. -/
inductive Family where
  | baseEndpoint
  | priorPublicLink
  | runningRelation
  | freshRelation
  | oneBasedCounter
  | allRunningSlots
  deriving DecidableEq

/-- The exact four semantic terminal obligations retained by the fixed-one
normal form. -/
def checks : List Family :=
  [.baseEndpoint, .priorPublicLink, .runningRelation, .freshRelation]

/-- Structural facts that the profile derives, so neither is a removable
terminal verifier obligation. -/
def eliminated : List Family := [.oneBasedCounter, .allRunningSlots]

/-- Every review-family is classified exactly once as retained or derived. -/
theorem family_classified (family : Family) :
    family ∈ checks ∨ family ∈ eliminated := by
  cases family <;> simp [checks, eliminated]

/-- No review-family is simultaneously retained and eliminated. -/
theorem family_classification_disjoint (family : Family) :
    ¬ (family ∈ checks ∧ family ∈ eliminated) := by
  cases family <;> simp [checks, eliminated]

/-- One named family owns one semantic predicate on a complete terminal
candidate.  The branch condition is computed by the verifier; it is not a
fifth caller-supplied check. -/
def semantics : Family -> Model.Candidate -> Prop
  | .baseEndpoint, candidate =>
      candidate.statement.iteration = 0 ->
        candidate.statement.zi = candidate.statement.z0
  | .priorPublicLink, candidate =>
      candidate.statement.iteration ≠ 0 ->
        Model.machine.freshPublic candidate.proof.fresh =
          Model.machine.encodeInstance (Model.machine.hash {
            verifierKeys := Model.setup.verifierKeys
            iteration := candidate.statement.iteration
            z0 := candidate.statement.z0
            current := candidate.statement.zi
            running := fun _ => candidate.proof.running
            pc := oneBased selected
          })
  | .runningRelation, candidate =>
      candidate.statement.iteration = 0 ∨
        Model.relations.runningHolds selected (Model.setup.verifierKeys selected)
          candidate.proof.running candidate.proof.runningWitness
  | .freshRelation, candidate =>
      candidate.statement.iteration = 0 ∨
        Model.relations.freshHolds selected (Model.setup.verifierKeys selected)
          candidate.proof.fresh candidate.proof.freshWitness
  | .oneBasedCounter, candidate =>
      candidate.proof.toGeneric.pc = 1
  | .allRunningSlots, candidate =>
      CanonicalTerminalVerifier.allRunningAccepted Model.relationChecks
        Model.setup.verifierKeys candidate.proof.toGeneric =
        runningAccepted Model.relationChecks Model.setup.verifierKeys
          candidate.proof

/-- The independent paper transition that the four retained checks must
recover. -/
def target (candidate : Model.Candidate) : Prop :=
  TerminalTransition Model.setup Model.machine Model.relations
    candidate.statement candidate.proof.toGeneric

/-- The profile computes the only generic counter, rather than accepting a
prover-supplied counter. -/
theorem oneBasedCounter_derived (candidate : Model.Candidate) :
    semantics .oneBasedCounter candidate := by
  rfl

/-- The generic finite running conjunction is exactly the direct sole-slot
check for every fixed-one candidate. -/
theorem allRunningSlots_derived (candidate : Model.Candidate) :
    semantics .allRunningSlots candidate := by
  exact allRunningAccepted_eq_runningAccepted Model.relationChecks
    Model.setup.verifierKeys candidate.proof

/-- The four retained predicates are exactly the independent paper terminal
transition on the fixed-one carrier. -/
theorem accepts_iff_transition (candidate : Model.Candidate) :
    CheckPlan.Accepts semantics checks candidate ↔ target candidate := by
  constructor
  · intro accepted
    by_cases iterationZero : candidate.statement.iteration = 0
    · exact Or.inl ⟨iterationZero,
        accepted .baseEndpoint (by simp [checks]) iterationZero⟩
    · refine Or.inr ⟨by
        simpa [Proof.toGeneric] using selected_counter_in_range,
        Nat.pos_of_ne_zero iterationZero, ?_, ?_, ?_⟩
      · have prior := accepted .priorPublicLink (by simp [checks])
        simpa [Proof.toGeneric] using prior iterationZero
      · intro slot
        have running := accepted .runningRelation (by simp [checks])
        rw [fin_eq_selected slot]
        simpa [Proof.toGeneric] using running.resolve_left iterationZero
      · have fresh := accepted .freshRelation (by simp [checks])
        simpa [Proof.toGeneric] using fresh.resolve_left iterationZero
  · intro transition family _member
    rcases transition with base | recursive
    · rcases base with ⟨iterationZero, endpoint⟩
      cases family with
      | baseEndpoint =>
          intro _
          exact endpoint
      | priorPublicLink =>
          intro notZero
          exact (notZero iterationZero).elim
      | runningRelation => exact Or.inl iterationZero
      | freshRelation => exact Or.inl iterationZero
      | oneBasedCounter => exact oneBasedCounter_derived candidate
      | allRunningSlots => exact allRunningSlots_derived candidate
    · rcases recursive with
        ⟨pcValid, iterationPositive, priorPublicInput, runningValid,
          freshValid⟩
      cases family with
      | baseEndpoint =>
          intro iterationZero
          exact (Nat.ne_of_gt iterationPositive iterationZero).elim
      | priorPublicLink =>
          intro _
          simpa [Proof.toGeneric] using priorPublicInput
      | runningRelation =>
          right
          simpa [Proof.toGeneric] using runningValid selected
      | freshRelation =>
          right
          simpa [Proof.toGeneric] using freshValid
      | oneBasedCounter => exact oneBasedCounter_derived candidate
      | allRunningSlots => exact allRunningSlots_derived candidate

/-- The semantic check plan and the direct executable fixed-one checker agree
extensionally; the counterexamples below are therefore invalid accepted
terminal transitions, not merely invalid abstract records. -/
theorem accepts_iff_fixedOne_eval (candidate : Model.Candidate) :
    CheckPlan.Accepts semantics checks candidate ↔
      Accepts Model.setup Model.machine Model.relations Model.relationChecks
        candidate.statement candidate.proof := by
  exact (accepts_iff_transition candidate).trans
    (FixedOne.accepts_iff_transition Model.setup Model.machine Model.relations
      Model.relationChecks candidate.statement candidate.proof).symm

/-- Omitting the base endpoint equality accepts the explicit unequal endpoint
at iteration zero, which is outside the independent paper transition. -/
theorem baseEndpoint_necessary :
    CheckPlan.NecessaryForSoundness semantics target checks .baseEndpoint := by
  refine ⟨Model.baseEndpointWitness, ?_, ?_⟩
  · intro family member
    have inChecks := (CheckPlan.mem_without_iff.mp member).1
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases family with
    | baseEndpoint => exact (retained rfl).elim
    | priorPublicLink => simp [semantics, Model.baseEndpointWitness, Model.candidate]
    | runningRelation => simp [semantics, Model.baseEndpointWitness, Model.candidate]
    | freshRelation => simp [semantics, Model.baseEndpointWitness, Model.candidate]
    | oneBasedCounter => simp [checks] at inChecks
    | allRunningSlots => simp [checks] at inChecks
  · intro transition
    rcases transition with ⟨iterationZero, endpoint⟩ |
      ⟨pcValid, iterationPositive, priorPublicInput, runningValid, freshValid⟩
    · exact (by decide : true ≠ false) (by
        simpa [Model.baseEndpointWitness, Model.candidate] using endpoint)
    · exact (Nat.lt_irrefl 0) (by
        simpa [Model.baseEndpointWitness, Model.candidate] using iterationPositive)

/-- Omitting the recursive prior-public link accepts a fresh value whose
relation witness remains valid but whose verifier-owned prior image differs. -/
theorem priorPublicLink_necessary :
    CheckPlan.NecessaryForSoundness semantics target checks .priorPublicLink := by
  refine ⟨Model.priorPublicLinkWitness, ?_, ?_⟩
  · intro family member
    have inChecks := (CheckPlan.mem_without_iff.mp member).1
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases family with
    | baseEndpoint => simp [semantics, Model.priorPublicLinkWitness, Model.candidate]
    | priorPublicLink => exact (retained rfl).elim
    | runningRelation => simp [semantics, Model.priorPublicLinkWitness, Model.candidate,
        Model.relations]
    | freshRelation => simp [semantics, Model.priorPublicLinkWitness, Model.candidate,
        Model.relations]
    | oneBasedCounter => simp [checks] at inChecks
    | allRunningSlots => simp [checks] at inChecks
  · intro transition
    rcases transition with ⟨iterationZero, endpoint⟩ |
      ⟨pcValid, iterationPositive, priorPublicInput, runningValid, freshValid⟩
    · exact Nat.one_ne_zero (by
        simpa [Model.priorPublicLinkWitness, Model.candidate] using iterationZero)
    · exact (by decide : true ≠ false) (by
        simpa [Model.priorPublicLinkWitness, Model.candidate, Model.machine,
          Proof.toGeneric] using priorPublicInput)

/-- Omitting the sole running relation accepts an otherwise valid recursive
candidate with exactly that relation witness flipped. -/
theorem runningRelation_necessary :
    CheckPlan.NecessaryForSoundness semantics target checks .runningRelation := by
  refine ⟨Model.runningRelationWitness, ?_, ?_⟩
  · intro family member
    have inChecks := (CheckPlan.mem_without_iff.mp member).1
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases family with
    | baseEndpoint => simp [semantics, Model.runningRelationWitness, Model.candidate]
    | priorPublicLink => simp [semantics, Model.runningRelationWitness, Model.candidate,
        Model.machine]
    | runningRelation => exact (retained rfl).elim
    | freshRelation => simp [semantics, Model.runningRelationWitness, Model.candidate,
        Model.relations]
    | oneBasedCounter => simp [checks] at inChecks
    | allRunningSlots => simp [checks] at inChecks
  · intro transition
    rcases transition with ⟨iterationZero, endpoint⟩ |
      ⟨pcValid, iterationPositive, priorPublicInput, runningValid, freshValid⟩
    · exact Nat.one_ne_zero (by
        simpa [Model.runningRelationWitness, Model.candidate] using iterationZero)
    · have invalid := runningValid selected
      exact (by decide : true ≠ false) (by
        simpa [Model.runningRelationWitness, Model.candidate, Model.relations,
          Proof.toGeneric] using invalid)

/-- Omitting the selected fresh relation accepts an otherwise valid recursive
candidate with exactly that relation witness flipped. -/
theorem freshRelation_necessary :
    CheckPlan.NecessaryForSoundness semantics target checks .freshRelation := by
  refine ⟨Model.freshRelationWitness, ?_, ?_⟩
  · intro family member
    have inChecks := (CheckPlan.mem_without_iff.mp member).1
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases family with
    | baseEndpoint => simp [semantics, Model.freshRelationWitness, Model.candidate]
    | priorPublicLink => simp [semantics, Model.freshRelationWitness, Model.candidate,
        Model.machine]
    | runningRelation => simp [semantics, Model.freshRelationWitness, Model.candidate,
        Model.relations]
    | freshRelation => exact (retained rfl).elim
    | oneBasedCounter => simp [checks] at inChecks
    | allRunningSlots => simp [checks] at inChecks
  · intro transition
    rcases transition with ⟨iterationZero, endpoint⟩ |
      ⟨pcValid, iterationPositive, priorPublicInput, runningValid, freshValid⟩
    · exact Nat.one_ne_zero (by
        simpa [Model.freshRelationWitness, Model.candidate] using iterationZero)
    · exact (by decide : true ≠ false) (by
        simpa [Model.freshRelationWitness, Model.candidate, Model.relations,
          Proof.toGeneric] using freshValid)

/-- Every retained terminal family has an ordinary kernel-checked removal
counterexample. -/
theorem retained_necessary
    (family : Family)
    (member : family ∈ checks) :
    CheckPlan.NecessaryForSoundness semantics target checks family := by
  cases family with
  | baseEndpoint => exact baseEndpoint_necessary
  | priorPublicLink => exact priorPublicLink_necessary
  | runningRelation => exact runningRelation_necessary
  | freshRelation => exact freshRelation_necessary
  | oneBasedCounter => exact (by simp [checks] at member)
  | allRunningSlots => exact (by simp [checks] at member)

/-- The exact four-family fixed-one terminal plan is inclusion-minimal for
the independent paper transition.  This is a model-level inclusion result for
the stated normal form, never a claim about all possible R1CS encodings. -/
theorem inclusionMinimalSound :
    CheckPlan.InclusionMinimalSound semantics target checks := by
  apply CheckPlan.inclusionMinimalSound_of_witnesses
  · intro candidate accepted
    exact (accepts_iff_transition candidate).1 accepted
  · exact retained_necessary

end Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Minimality
