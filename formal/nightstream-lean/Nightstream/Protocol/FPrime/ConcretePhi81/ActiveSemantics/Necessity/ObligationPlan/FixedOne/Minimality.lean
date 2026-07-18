import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.FixedOne.Global
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs

/-!
Closed model-level inclusion-minimality fixture for the canonical fixed-one
active obligation plan.

Owns: one explicit outer machine; three canonical inputs differing only in
the outer iteration; proof that those mutations preserve the complete NIFS
context; consumption of the baseline's explicitly sampler-backed honest NIFS
result; concrete removal witnesses for the three retained families; and
inclusion-minimality of the global canonical plan.

Does not own: production decoding, Poseidon2 refinement, Rust, R1CS, physical
row ownership, the bounded-sampler construction already owned by
`HonestBaseline.Context`, costs, global gate-count minimality, or row removal.
It also does not justify replacing the full running-child carrier with a
parent-only Construction-2 handle.

Emits constraints: no.

Authority boundary: this is a degenerate model fixture over the independently
proved 270-coordinate source relation. The machine is explicit and shared by
all witnesses. The iteration and prior-link countermodels mutate only an
outer field ignored by the verifier-owned context callbacks; the complete
NIFS context equality is proved. The honest result is obtained from
`HonestBaseline.Context.exists_resultTransition`, whose centered-zero
54-of-64 batch is explicitly constructed rather than assumed.
The inclusion-minimality result is relative to this rich carrier, in which
the complete child vector is already present; it is not a compression or
cross-step binding theorem.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.fixed_one.minimality.machine` | one outer machine distinguishes iteration two without changing NIFS context | explicit model fixture | `machine` |
| `fprime.fixed_one.minimality.context` | iteration-only outer mutations preserve the complete selected NIFS context | exact dataflow | `contextAt_inputAt` |
| `fprime.fixed_one.minimality.baseline` | independent honest sources plus the explicit centered-zero sampler construct one semantic selected result | imported derivation | `HonestBaseline.Context.exists_resultTransition`, `exists_honestNext` |
| `fprime.fixed_one.minimality.iteration` | removing positivity admits iteration zero | removal witness | `iteration_necessary` |
| `fprime.fixed_one.minimality.prior_link` | removing the prior link admits the iteration-two digest mismatch | removal witness | `priorPublicInput_necessary` |
| `fprime.fixed_one.minimality.nifs` | removing NIFS admits a changed parent with unchanged checked children | removal witness | `selectedNifs_necessary` |
| `fprime.fixed_one.minimality.ledger` | every family is retained or eliminated, never both | exact classification | `family_classified`, `family_classification_disjoint` |
| `fprime.fixed_one.minimality.closed` | the exact three-family global plan is inclusion-minimal for soundness | model-level theorem | `inclusionMinimalSound` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.FixedOne.Minimality

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

namespace Baseline

abbrev ModelInput :=
  FixedOneCanonical.Input Unit Unit Unit HonestBaseline.Sources.shape
    HonestBaseline.Context.publicRingColumns HonestBaseline.Context.publicFits
    HonestBaseline.Context.verifierRows

/-- Canonical semantic carrier obtained by erasing the two verifier-computed
fields from the independently validated raw baseline. -/
def input : ModelInput :=
  FixedOneCanonical.Input.erase HonestBaseline.Context.input

/-- Change only the outer iteration. Fresh/running NIFS authority is shared
definitionally across all three countermodels. -/
def inputAt (iteration : Nat) : ModelInput :=
  { input with iteration := iteration }

def iterationInput : ModelInput := inputAt 0

def honestInput : ModelInput := inputAt 1

def priorLinkInput : ModelInput := inputAt 2

/-- One public carrier point used only to prove that the shifted public input
is visibly distinct. -/
def firstPublicColumn :
    Fin (RelationShape HonestBaseline.Sources.shape
      HonestBaseline.Context.publicRingColumns
      HonestBaseline.Context.publicFits).publicWidth :=
  ⟨0, by decide⟩

/-- A deterministic value distinct from the baseline fresh public input. -/
def shiftedPublicInput :
    RelationPublicInput HonestBaseline.Sources.shape
      HonestBaseline.Context.publicRingColumns
      HonestBaseline.Context.publicFits :=
  fun column =>
    if input.fresh.publicInput column = 0 then 1 else 0

theorem publicInput_ne_shifted :
    input.fresh.publicInput ≠ shiftedPublicInput := by
  intro equal
  have atColumn := congrFun equal firstPublicColumn
  by_cases isZero : input.fresh.publicInput firstPublicColumn = 0
  · have shiftedAt : shiftedPublicInput firstPublicColumn = 1 := by
      simp only [shiftedPublicInput, isZero, if_true]
    have inputEqOne : input.fresh.publicInput firstPublicColumn = 1 :=
      atColumn.trans shiftedAt
    have zeroEqOne : (0 : F) = 1 := isZero.symm.trans inputEqOne
    exact (by decide : (0 : F) ≠ 1) zeroEqOne
  · have shiftedAt : shiftedPublicInput firstPublicColumn = 0 := by
      simp only [shiftedPublicInput, isZero, if_false]
    have isZero' : input.fresh.publicInput firstPublicColumn = 0 :=
      atColumn.trans shiftedAt
    exact isZero isZero'

/-- Shared outer machine. Hashing distinguishes only iteration two; both
iteration zero and the honest iteration one encode the observed baseline
public input. -/
def machine :
    Machine Unit Bool Unit Unit HonestBaseline.Sources.shape
      HonestBaseline.Context.publicRingColumns HonestBaseline.Context.publicFits
      HonestBaseline.Context.verifierRows 1 where
  control := fun _ _ => Paper.ProgramCounter.ofIndex FixedOneCanonical.selected
  step := fun _ _ _ => ()
  defaultRunning := fun _ child => HonestBaseline.Context.children child
  hash := fun preimage => decide (preimage.iteration = 2)
  encodeInstance := fun digest =>
    if digest then shiftedPublicInput else input.fresh.publicInput

/-- All iteration-only mutations reconstruct the same complete NIFS context
as the validated raw baseline. -/
theorem contextAt_inputAt (iteration : Nat) :
    contextAt HonestBaseline.Context.setup
        ((inputAt iteration).toActive HonestBaseline.Context.setup)
        FixedOneCanonical.selected =
      HonestBaseline.Context.context := by
  rfl

/-- Transport one semantic NIFS transition across an iteration-only outer
mutation using the proved context equality. -/
theorem transition_inputAt
    (iteration : Nat)
    {selectedNext :
      Slot HonestBaseline.Sources.shape
        HonestBaseline.Context.publicRingColumns
        HonestBaseline.Context.publicFits
        HonestBaseline.Context.verifierRows}
    (transition :
      FixedActive.ResultTransition HonestBaseline.Context.context selectedNext) :
    FixedActive.ResultTransition
      (contextAt HonestBaseline.Context.setup
        ((inputAt iteration).toActive HonestBaseline.Context.setup)
        FixedOneCanonical.selected)
      selectedNext := by
  rw [contextAt_inputAt]
  exact transition

theorem iterationInput_priorPublicInput :
    (iterationInput.toActive HonestBaseline.Context.setup).fresh.publicInput =
      machine.encodeInstance
        (machine.hash (Paper.priorHashPreimage
          (iterationInput.toActive HonestBaseline.Context.setup).toPaper)) := by
  rfl

theorem honestInput_priorPublicInput :
    (honestInput.toActive HonestBaseline.Context.setup).fresh.publicInput =
      machine.encodeInstance
        (machine.hash (Paper.priorHashPreimage
          (honestInput.toActive HonestBaseline.Context.setup).toPaper)) := by
  rfl

theorem priorLinkInput_not_priorPublicInput :
    (priorLinkInput.toActive HonestBaseline.Context.setup).fresh.publicInput ≠
      machine.encodeInstance
        (machine.hash (Paper.priorHashPreimage
          (priorLinkInput.toActive HonestBaseline.Context.setup).toPaper)) := by
  simpa [priorLinkInput, inputAt, machine] using publicInput_ne_shifted

/-- One independently constructed semantic result for the shared NIFS
context. No accepted result or sampler-success bit is supplied by a caller. -/
theorem exists_honestNext :
    ∃ selectedNext :
        Slot HonestBaseline.Sources.shape
          HonestBaseline.Context.publicRingColumns
          HonestBaseline.Context.publicFits
          HonestBaseline.Context.verifierRows,
      FixedActive.ResultTransition HonestBaseline.Context.context selectedNext := by
  rcases HonestBaseline.Context.exists_resultTransition with
    ⟨certificate, _accepted, transition⟩
  exact ⟨FixedActive.resultOf HonestBaseline.Context.context certificate,
    transition⟩

end Baseline

/-- Removing positivity admits the otherwise honest iteration-zero case. The
selected NIFS transition is transported across the proved context equality,
so no NIFS premise is changed by this outer-only mutation. -/
theorem iteration_necessary :
    CheckPlan.NecessaryForSoundness
      (Global.semantics HonestBaseline.Context.setup Baseline.machine
        FixedOneCanonical.selected)
      (Global.target HonestBaseline.Context.setup Baseline.machine)
      Canonical.checks .iterationPositive := by
  apply Global.lift_local_necessary
    HonestBaseline.Context.setup Baseline.machine FixedOneCanonical.selected
      Baseline.iterationInput .iterationPositive
  rcases Baseline.exists_honestNext with ⟨selectedNext, transition⟩
  refine ⟨selectedNext, ?_, ?_⟩
  · intro family member
    have inChecks := (CheckPlan.mem_without_iff.mp member).1
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases family with
    | iterationPositive =>
        exact (retained rfl).elim
    | priorSlot =>
        exact (Canonical.priorSlot_not_mem inChecks).elim
    | priorPublicInput =>
        simpa [Canonical.semantics, ObligationPlan.semantics] using
          Baseline.iterationInput_priorPublicInput
    | expectedStructure =>
        exact (Canonical.expectedStructure_not_mem inChecks).elim
    | selectedNifs =>
        simpa [Canonical.semantics, ObligationPlan.semantics] using
          Baseline.transition_inputAt 0 transition
    | dispatch =>
        exact (Canonical.dispatch_not_mem inChecks).elim
  · intro obligations
    simpa [Baseline.iterationInput, Baseline.inputAt] using
      obligations.iterationPositive

/-- Removing the prior-public-input link admits iteration two, for which the
explicit machine maps the prior digest to the shifted public input. The
selected NIFS context and result remain exactly the honest baseline. -/
theorem priorPublicInput_necessary :
    CheckPlan.NecessaryForSoundness
      (Global.semantics HonestBaseline.Context.setup Baseline.machine
        FixedOneCanonical.selected)
      (Global.target HonestBaseline.Context.setup Baseline.machine)
      Canonical.checks .priorPublicInput := by
  apply Global.lift_local_necessary
    HonestBaseline.Context.setup Baseline.machine FixedOneCanonical.selected
      Baseline.priorLinkInput .priorPublicInput
  rcases Baseline.exists_honestNext with ⟨selectedNext, transition⟩
  refine ⟨selectedNext, ?_, ?_⟩
  · intro family member
    have inChecks := (CheckPlan.mem_without_iff.mp member).1
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases family with
    | iterationPositive =>
        change 0 < Baseline.priorLinkInput.iteration
        change 0 < 2
        decide
    | priorSlot =>
        exact (Canonical.priorSlot_not_mem inChecks).elim
    | priorPublicInput =>
        exact (retained rfl).elim
    | expectedStructure =>
        exact (Canonical.expectedStructure_not_mem inChecks).elim
    | selectedNifs =>
        simpa [Canonical.semantics, ObligationPlan.semantics] using
          Baseline.transition_inputAt 2 transition
    | dispatch =>
        exact (Canonical.dispatch_not_mem inChecks).elim
  · intro obligations
    exact Baseline.priorLinkInput_not_priorPublicInput
      obligations.priorPublicInput

/-- Removing semantic NIFS authority admits a changed parent norm stage while
preserving the complete checked child accumulator. The honest accepted result
comes from the explicit sampler-backed baseline, and parent uniqueness rejects
the mutation. -/
theorem selectedNifs_necessary :
    CheckPlan.NecessaryForSoundness
      (Global.semantics HonestBaseline.Context.setup Baseline.machine
        FixedOneCanonical.selected)
      (Global.target HonestBaseline.Context.setup Baseline.machine)
      Canonical.checks .selectedNifs := by
  apply Global.lift_local_necessary
    HonestBaseline.Context.setup Baseline.machine FixedOneCanonical.selected
      Baseline.honestInput .selectedNifs
  rcases Baseline.exists_honestNext with ⟨selectedNext, transition⟩
  let accepted :
      FixedOneCanonical.Obligations HonestBaseline.Context.setup
        Baseline.machine Baseline.honestInput selectedNext := {
    iterationPositive := by
      simp [Baseline.honestInput, Baseline.inputAt]
    priorPublicInput := Baseline.honestInput_priorPublicInput
    selectedNifs := Baseline.transition_inputAt 1 transition
  }
  let realization :
      SelectedNifs.Realization HonestBaseline.Context.setup Baseline.machine
        FixedOneCanonical.selected
        (Baseline.honestInput.toActive HonestBaseline.Context.setup) := {
    selected := FixedOneCanonical.selected
    acceptedNext := selectedNext
    accepted := accepted.toActive FixedOneCanonical.selected
  }
  refine ⟨realization.forgedNext, ?_, ?_⟩
  · intro family member
    have inChecks := (CheckPlan.mem_without_iff.mp member).1
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases family with
    | iterationPositive =>
        simpa [Canonical.semantics, ObligationPlan.semantics] using
          accepted.iterationPositive
    | priorSlot =>
        exact (Canonical.priorSlot_not_mem inChecks).elim
    | priorPublicInput =>
        simpa [Canonical.semantics, ObligationPlan.semantics] using
          accepted.priorPublicInput
    | expectedStructure =>
        exact (Canonical.expectedStructure_not_mem inChecks).elim
    | selectedNifs =>
        exact (retained rfl).elim
    | dispatch =>
        exact (Canonical.dispatch_not_mem inChecks).elim
  · intro obligations
    exact realization.forged_not_transition obligations.selectedNifs

/-- Exact exhaustive and disjoint ledger for the canonical family universe. -/
theorem family_ledger (family : Family) :
    (family ∈ Canonical.checks ∨ family ∈ Canonical.eliminated) ∧
      ¬(family ∈ Canonical.checks ∧ family ∈ Canonical.eliminated) :=
  ⟨Canonical.classified family, Canonical.classification_disjoint family⟩

/-- Every retained family has its concrete model-level removal witness. -/
theorem retained_necessary
    (family : Family)
    (member : family ∈ Canonical.checks) :
    CheckPlan.NecessaryForSoundness
      (Global.semantics HonestBaseline.Context.setup Baseline.machine
        FixedOneCanonical.selected)
      (Global.target HonestBaseline.Context.setup Baseline.machine)
      Canonical.checks family := by
  cases family with
  | iterationPositive =>
      exact iteration_necessary
  | priorSlot =>
      exact (Canonical.priorSlot_not_mem member).elim
  | priorPublicInput =>
      exact priorPublicInput_necessary
  | expectedStructure =>
      exact (Canonical.expectedStructure_not_mem member).elim
  | selectedNifs =>
      exact selectedNifs_necessary
  | dispatch =>
      exact (Canonical.dispatch_not_mem member).elim

/-- The global three-family canonical fixed-one plan is sound and every
retained leaf is inclusion-necessary relative to the independent canonical
obligations. This is model-level inclusion minimality, not a gate-count lower
bound or production row-removal authorization. -/
theorem inclusionMinimalSound :
    CheckPlan.InclusionMinimalSound
      (Global.semantics HonestBaseline.Context.setup Baseline.machine
        FixedOneCanonical.selected)
      (Global.target HonestBaseline.Context.setup Baseline.machine)
      Canonical.checks := by
  apply CheckPlan.inclusionMinimalSound_of_witnesses
  · intro case accepted
    exact
      (Global.accepts_iff_obligations HonestBaseline.Context.setup
        Baseline.machine FixedOneCanonical.selected case).1 accepted
  · exact retained_necessary

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.FixedOne.Minimality
