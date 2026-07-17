import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.FixedOneCanonical
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan

/-!
Fixed-one-slot specialization of the active F-prime obligation plan.

Owns: the approved model-level profile `slotCount = 1`; an exact five-family
plan over the existing raw input carrier; and an exact three-family plan over
the canonical semantic carrier.

Does not own: the sole slot, fixed dispatch theorem, canonical carrier,
canonical obligations, selection of this profile by Rust, decoding of a
production input into the canonical carrier, construction of removal
witnesses, sampler success, R1CS, costs, or row removal. Those semantic
objects are owned by `ActiveSemantics.FixedOneCanonical`.

Emits constraints: no.

Authority boundary: `slotCount = 1` is a protocol-profile specialization, not
an inference from implementation behavior. On the raw carrier, `priorPc` and
the fresh relation structure remain caller supplied, so both checks remain.
The canonical semantic carrier computes both values through verifier-owned
setup. Production may use the three-family plan only after a separate
refinement proves that its decoder constructs exactly that carrier.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.fixed_one.raw` | iteration, raw prior-counter, prior-link, structure, and NIFS obligations; fixed dispatch is derived | exact five-family model | `Raw.checks`, `Raw.eliminated`, `Raw.exact` |
| `fprime.fixed_one.canonical.carrier` | prior counter and fresh structure are verifier-computed | imported semantic authority | `FixedOneCanonical.Input` |
| `fprime.fixed_one.canonical.eliminated` | prior-slot, expected-structure, and dispatch equations hold by canonical construction | derived/eliminated | `Canonical.eliminated_hold` |
| `fprime.fixed_one.canonical` | iteration, prior link, and NIFS are exact for `FixedOneCanonical.Obligations` | exact three-family model | `Canonical.checks`, `Canonical.eliminated`, `Canonical.exact` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.FixedOne

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uOuterKey uAppState uWitness uDigest uTranscriptState

section

variable {OuterKey : Type uOuterKey}
variable {Digest : Type uDigest}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {TranscriptState : Type uTranscriptState}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

namespace Raw

/-- Exact retained checks on the existing raw carrier. `priorSlot` and
`expectedStructure` remain because both underlying values are caller-owned
fields on that carrier. -/
def checks : List Family :=
  [.iterationPositive, .priorSlot, .priorPublicInput, .expectedStructure,
    .selectedNifs]

/-- Raw-carrier obligations eliminated by proof rather than checked. -/
def eliminated : List Family := [.dispatch]

@[simp] theorem dispatch_not_mem : Family.dispatch ∉ checks := by
  simp [checks]

/-- Every obligation family is recorded as either retained or eliminated in
the raw fixed-one profile. -/
theorem classified (family : Family) :
    family ∈ checks ∨ family ∈ eliminated := by
  cases family <;> simp [checks, eliminated]

/-- No raw fixed-one family is both retained and eliminated. -/
theorem classification_disjoint (family : Family) :
    ¬(family ∈ checks ∧ family ∈ eliminated) := by
  cases family <;> simp [checks, eliminated]

/-- Removing dispatch from the raw one-slot plan preserves exactness because
the checked counter codomain has only one value. -/
theorem accepts_iff_obligations
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      ActiveSemantics.Input OuterKey AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (candidate :
      ObligationPlan.Candidate shape publicRingColumns publicFits verifierRows
        1) :
    CheckPlan.Accepts
        (ObligationPlan.semantics setup machine functionIndex input)
        checks candidate ↔
      ObligationPlan.target setup machine functionIndex input candidate := by
  constructor
  · intro accepted
    exact {
      iterationPositive :=
        accepted .iterationPositive (by simp [checks])
      priorSlot :=
        accepted .priorSlot (by simp [checks])
      priorPublicInput :=
        accepted .priorPublicInput (by simp [checks])
      expectedStructure :=
        accepted .expectedStructure (by simp [checks])
      selectedNifs :=
        accepted .selectedNifs (by simp [checks])
      dispatch := by
        simpa [FixedOneCanonical.Input.erase] using
          FixedOneCanonical.dispatch_derived machine functionIndex
            (FixedOneCanonical.Input.erase input)
    }
  · intro obligations family member
    cases family with
    | iterationPositive =>
        exact obligations.iterationPositive
    | priorSlot =>
        exact obligations.priorSlot
    | priorPublicInput =>
        exact obligations.priorPublicInput
    | expectedStructure =>
        exact obligations.expectedStructure
    | selectedNifs =>
        exact obligations.selectedNifs
    | dispatch =>
        exact (dispatch_not_mem member).elim

/-- The five retained raw-carrier leaves are exact for the complete active
obligations in the fixed-one profile. -/
theorem exact
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      ActiveSemantics.Input OuterKey AppState Witness shape publicRingColumns
        publicFits verifierRows 1) :
    CheckPlan.Exact
      (ObligationPlan.semantics setup machine functionIndex input)
      (ObligationPlan.target setup machine functionIndex input)
      checks := by
  intro candidate
  exact accepts_iff_obligations setup machine functionIndex input candidate

end Raw

namespace Canonical

/-- Exact retained family list after the canonical semantic carrier computes
the prior counter and fresh relation structure and the fixed-one codomain
derives dispatch. -/
def checks : List Family :=
  [.iterationPositive, .priorPublicInput, .selectedNifs]

/-- Canonical-carrier obligations eliminated by construction or proof. -/
def eliminated : List Family := [.priorSlot, .expectedStructure, .dispatch]

@[simp] theorem priorSlot_not_mem : Family.priorSlot ∉ checks := by
  simp [checks]

@[simp] theorem expectedStructure_not_mem :
    Family.expectedStructure ∉ checks := by
  simp [checks]

@[simp] theorem dispatch_not_mem : Family.dispatch ∉ checks := by
  simp [checks]

/-- Every obligation family is recorded as either retained or eliminated in
the canonical fixed-one profile. -/
theorem classified (family : Family) :
    family ∈ checks ∨ family ∈ eliminated := by
  cases family <;> simp [checks, eliminated]

/-- No canonical fixed-one family is both retained and eliminated. -/
theorem classification_disjoint (family : Family) :
    ¬(family ∈ checks ∧ family ∈ eliminated) := by
  cases family <;> simp [checks, eliminated]

/-- Interpret every named family through the general obligation semantics,
using only the canonical semantic input and the selected next slot. This is
an adapter, not a second carrier or a second semantic target. -/
def semantics
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      FixedOneCanonical.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows) :
    Family ->
      Slot shape publicRingColumns publicFits verifierRows -> Prop :=
  fun family selectedNext =>
    ObligationPlan.semantics setup machine functionIndex
      (input.toActive setup) family {
        selected := FixedOneCanonical.selected
        selectedNext := selectedNext
      }

/-- Each family in the eliminated ledger is discharged by the canonical
semantic carrier or fixed-one program-counter codomain. -/
theorem eliminated_hold
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      FixedOneCanonical.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows)
    (family : Family)
    (member : family ∈ eliminated) :
    semantics setup machine functionIndex input family selectedNext := by
  cases family with
  | iterationPositive => simp [eliminated] at member
  | priorSlot =>
      simpa [semantics, ObligationPlan.semantics] using
        FixedOneCanonical.Input.priorSlot_derived setup input
          FixedOneCanonical.selected
  | priorPublicInput => simp [eliminated] at member
  | expectedStructure =>
      simpa [semantics, ObligationPlan.semantics] using
        FixedOneCanonical.Input.expectedStructure_derived setup input
          FixedOneCanonical.selected
  | selectedNifs => simp [eliminated] at member
  | dispatch =>
      simpa [semantics, ObligationPlan.semantics,
        FixedOneCanonical.Input.toActive] using
          FixedOneCanonical.dispatch_derived machine functionIndex input

/-- The three retained leaves are exactly the canonical obligations already
owned by `ActiveSemantics.FixedOneCanonical`. -/
theorem accepts_iff_obligations
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      FixedOneCanonical.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (selectedNext :
      Slot shape publicRingColumns publicFits verifierRows) :
    CheckPlan.Accepts
        (semantics setup machine functionIndex input) checks selectedNext ↔
      FixedOneCanonical.Obligations setup machine input selectedNext := by
  constructor
  · intro accepted
    exact {
      iterationPositive :=
        accepted .iterationPositive (by simp [checks])
      priorPublicInput :=
        accepted .priorPublicInput (by simp [checks])
      selectedNifs :=
        accepted .selectedNifs (by simp [checks])
    }
  · intro obligations family member
    cases family with
    | iterationPositive =>
        exact obligations.iterationPositive
    | priorSlot =>
        exact (priorSlot_not_mem member).elim
    | priorPublicInput =>
        exact obligations.priorPublicInput
    | expectedStructure =>
        exact (expectedStructure_not_mem member).elim
    | selectedNifs =>
        exact obligations.selectedNifs
    | dispatch =>
        exact (dispatch_not_mem member).elim

/-- Exactness of the three-family canonical fixed-one plan. This is a
model-level theorem; it does not establish production decoding refinement. -/
theorem exact
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      FixedOneCanonical.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows) :
    CheckPlan.Exact
      (semantics setup machine functionIndex input)
      (FixedOneCanonical.Obligations setup machine input)
      checks := by
  intro selectedNext
  exact accepts_iff_obligations setup machine functionIndex input selectedNext

end Canonical

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.FixedOne
