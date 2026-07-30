import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplex
import Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck

/-!
Contract: semantic refinement of the symbolic Poseidon2 duplex planner.

`SymbolicDuplex` is an encoding-time builder.  This module evaluates its
linear-combination state under an assignment and proves that satisfaction of
the emitted permutation rows makes every builder operation agree with the
value-level `Poseidon2Duplex`.

The distinction is load-bearing.  Merely using output-port expressions as
challenges prevents a second challenge column, but does not by itself show
that those ports contain the transcript permutation.  `valid_of_satisfied`
and the operation lemmas below close that gap without accepting a transcript
value or verifier conclusion as a premise.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.ProjectionProgram

/-- Evaluate a symbolic lane state under one physical assignment. -/
def evalState (assignment : Nat → Nat) (state : State) : Values :=
  fun lane => lcEval assignment (state lane)

theorem lcEval_lt (assignment : Nat → Nat) (value : LinCombNormal.LinComb) :
    lcEval assignment value < goldilocksP := by
  unfold lcEval
  exact Nat.mod_lt _ (by decide)

/-- The value-level duplex state denoted by a symbolic builder. -/
def decodedBuilder (assignment : Nat → Nat)
    (builder : SymbolicDuplex.Builder) : Poseidon2Duplex.State where
  lanes := evalState assignment builder.lanes
  absorbed := builder.absorbed

/-- One emitted entry computes the selected permutation. -/
def EntryValid (base : Nat) (constants : Constants)
    (assignment : Nat → Nat) (entry : SymbolicDuplex.Entry) : Prop :=
  ∀ lane : Fin width,
    lcEval assignment (SymbolicDuplex.outputState base entry.call lane) =
      referencePermutation constants (evalState assignment entry.state) lane

/-- Every entry accumulated by a builder has its row-level meaning. -/
def Valid (base : Nat) (constants : Constants)
    (assignment : Nat → Nat) (builder : SymbolicDuplex.Builder) : Prop :=
  ∀ entry ∈ builder.entries, EntryValid base constants assignment entry

theorem outputState_eval
    (base call : Nat) (assignment : Nat → Nat)
    (residues : ∀ column, assignment column < goldilocksP)
    (lane : Fin width) :
    lcEval assignment (SymbolicDuplex.outputState base call lane) =
      assignment ((SymbolicDuplex.layoutAt base call).outputPort lane) := by
  unfold SymbolicDuplex.outputState
  rw [KMul.lcEval_singleton_col,
    Nat.mod_eq_of_lt (residues _)]

/-- Satisfaction of the emitted row list is the sole semantic authority for
all permutation entries. -/
theorem valid_of_satisfied
    (base : Nat) (constants : Constants)
    (builder : SymbolicDuplex.Builder) (assignment : Nat → Nat)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies (SymbolicDuplex.rows base constants builder) assignment) :
    Valid base constants assignment builder := by
  intro entry member lane
  rw [outputState_eval base entry.call assignment residues lane]
  exact SymbolicDuplex.call_computes_reference base constants builder
    assignment residues constantWire satisfied entry member
    (evalState assignment entry.state) (fun _ => rfl) lane

/-- Builder extension is physical entry inclusion, not a supplied receipt. -/
def Extends (older newer : SymbolicDuplex.Builder) : Prop :=
  ∀ entry ∈ older.entries, entry ∈ newer.entries

theorem Extends.refl (builder : SymbolicDuplex.Builder) :
    Extends builder builder :=
  fun _ member => member

theorem Extends.trans
    {first second third : SymbolicDuplex.Builder}
    (left : Extends first second) (right : Extends second third) :
    Extends first third :=
  fun entry member => right entry (left entry member)

theorem Valid.of_extends
    {base : Nat} {constants : Constants} {assignment : Nat → Nat}
    {older newer : SymbolicDuplex.Builder}
    (valid : Valid base constants assignment newer)
    (extension : Extends older newer) :
    Valid base constants assignment older :=
  fun entry member => valid entry (extension entry member)

theorem permute_extends (base : Nat) (builder : SymbolicDuplex.Builder) :
    Extends builder (SymbolicDuplex.permute base builder) := by
  intro entry member
  exact List.mem_append_left _ member

theorem guarded_extends (base : Nat) (builder : SymbolicDuplex.Builder) :
    Extends builder (SymbolicDuplex.guarded base builder) := by
  unfold SymbolicDuplex.guarded
  split
  · exact permute_extends base builder
  · exact Extends.refl builder

theorem absorb_extends (base : Nat) (value : LinCombNormal.LinComb)
    (builder : SymbolicDuplex.Builder) :
    Extends builder (SymbolicDuplex.absorb base value builder) := by
  exact guarded_extends base builder

theorem absorbMany_extends (base : Nat) :
    ∀ (values : List LinCombNormal.LinComb)
      (builder : SymbolicDuplex.Builder),
      Extends builder (SymbolicDuplex.absorbMany base values builder)
  | [], builder => Extends.refl builder
  | value :: rest, builder =>
      (absorb_extends base value builder).trans
        (absorbMany_extends base rest
          (SymbolicDuplex.absorb base value builder))

theorem gate_extends (base : Nat) (builder : SymbolicDuplex.Builder) :
    Extends builder (SymbolicDuplex.gate base builder) :=
  (absorb_extends base SymbolicDuplex.one builder).trans
    (permute_extends base
      (SymbolicDuplex.absorb base SymbolicDuplex.one builder))

theorem squeezeK_extends (base : Nat) (builder : SymbolicDuplex.Builder) :
    Extends builder (SymbolicDuplex.squeezeK base builder).2 :=
  gate_extends base builder

private theorem appendedEntry_valid
    (base : Nat) (constants : Constants)
    (assignment : Nat → Nat) (builder : SymbolicDuplex.Builder)
    (valid :
      Valid base constants assignment
        (SymbolicDuplex.permute base builder)) :
    EntryValid base constants assignment
      { call := builder.entries.length, state := builder.lanes } := by
  apply valid
  simp [SymbolicDuplex.permute]

/-- A forced symbolic permutation denotes the value-level permutation. -/
theorem decodedBuilder_permute
    (base : Nat) (constants : Constants)
    (assignment : Nat → Nat) (builder : SymbolicDuplex.Builder)
    (valid :
      Valid base constants assignment
        (SymbolicDuplex.permute base builder)) :
    decodedBuilder assignment (SymbolicDuplex.permute base builder) =
      Poseidon2Duplex.permute constants (decodedBuilder assignment builder) := by
  rw [Poseidon2Duplex.State.mk.injEq]
  refine ⟨?_, rfl⟩
  funext lane
  exact appendedEntry_valid base constants assignment builder valid lane

/-- A guarded symbolic state denotes the value-level guarded state. -/
theorem decodedBuilder_guarded
    (base : Nat) (constants : Constants)
    (assignment : Nat → Nat) (builder : SymbolicDuplex.Builder)
    (valid :
      Valid base constants assignment
        (SymbolicDuplex.guarded base builder)) :
    decodedBuilder assignment (SymbolicDuplex.guarded base builder) =
      Poseidon2Duplex.guarded constants (decodedBuilder assignment builder) := by
  unfold SymbolicDuplex.guarded Poseidon2Duplex.guarded
  by_cases full : Poseidon2Sponge.rate ≤ builder.absorbed
  · rw [if_pos full, if_pos (by
      simpa only [decodedBuilder] using full)]
    apply decodedBuilder_permute base constants assignment builder
    simpa only [SymbolicDuplex.guarded, if_pos full] using valid
  · rw [if_neg full, if_neg (by
      simpa only [decodedBuilder] using full)]

/-- One symbolic overwrite denotes one value-level overwrite. -/
theorem decodedBuilder_absorb
    (base : Nat) (constants : Constants)
    (assignment : Nat → Nat) (value : LinCombNormal.LinComb)
    (builder : SymbolicDuplex.Builder)
    (valid :
      Valid base constants assignment
        (SymbolicDuplex.absorb base value builder)) :
    decodedBuilder assignment (SymbolicDuplex.absorb base value builder) =
      Poseidon2Duplex.absorbElem constants (lcEval assignment value)
        (decodedBuilder assignment builder) := by
  have guardedValid :
      Valid base constants assignment
        (SymbolicDuplex.guarded base builder) := by
    intro entry member
    exact valid entry member
  have guardedEq :=
    decodedBuilder_guarded base constants assignment builder guardedValid
  unfold SymbolicDuplex.absorb Poseidon2Duplex.absorbElem
  simp only
  rw [← guardedEq]
  rw [Poseidon2Duplex.State.mk.injEq]
  refine ⟨?_, rfl⟩
  funext lane
  simp only [decodedBuilder, evalState]
  split <;> rename_i same
  · simp only [lcEval, Nat.mod_mod]
  · rfl

/-- A list of symbolic overwrites denotes value-level left-to-right
absorption of the evaluated field list. -/
theorem decodedBuilder_absorbMany
    (base : Nat) (constants : Constants)
    (assignment : Nat → Nat) :
    ∀ (values : List LinCombNormal.LinComb)
      (builder : SymbolicDuplex.Builder),
      Valid base constants assignment
          (SymbolicDuplex.absorbMany base values builder) →
      decodedBuilder assignment
          (SymbolicDuplex.absorbMany base values builder) =
        Poseidon2Duplex.absorbList constants
          (values.map (lcEval assignment))
          (decodedBuilder assignment builder)
  | [], _, _ => rfl
  | value :: rest, builder, valid => by
      rw [SymbolicDuplex.absorbMany, List.map_cons,
        Poseidon2Duplex.absorbList]
      rw [decodedBuilder_absorbMany base constants assignment rest
        (SymbolicDuplex.absorb base value builder) valid]
      congr 1
      apply decodedBuilder_absorb base constants assignment value builder
      exact valid.of_extends
        (absorbMany_extends base rest
          (SymbolicDuplex.absorb base value builder))

/-- The constant-wire expression denotes the value-level gate marker. -/
theorem one_eval (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1) :
    lcEval assignment SymbolicDuplex.one = 1 := by
  unfold SymbolicDuplex.one
  rw [KMul.lcEval_singleton_col, constantWire]
  decide

/-- The symbolic gate denotes the value-level pre-squeeze gate. -/
theorem decodedBuilder_gate
    (base : Nat) (constants : Constants)
    (assignment : Nat → Nat) (builder : SymbolicDuplex.Builder)
    (constantWire : assignment 0 = 1)
    (valid :
      Valid base constants assignment
        (SymbolicDuplex.gate base builder)) :
    decodedBuilder assignment (SymbolicDuplex.gate base builder) =
      Poseidon2Duplex.gate constants (decodedBuilder assignment builder) := by
  unfold SymbolicDuplex.gate Poseidon2Duplex.gate
  rw [decodedBuilder_permute base constants assignment
    (SymbolicDuplex.absorb base SymbolicDuplex.one builder) valid]
  apply congrArg
  rw [decodedBuilder_absorb base constants assignment
    SymbolicDuplex.one builder
    (valid.of_extends
      (permute_extends base
        (SymbolicDuplex.absorb base SymbolicDuplex.one builder))),
    one_eval assignment constantWire]

/-! ## Extension challenge decoding -/

/-- The two freshly permuted rate lanes interpreted as one quadratic
extension value.  Reduction is explicit so this remains total even for an
arbitrary value-level state. -/
def challengeValue (state : Poseidon2Duplex.State) : K where
  c0 := ⟨state.lanes ⟨0, by decide⟩ % goldilocksP,
    Nat.mod_lt _ (by decide)⟩
  c1 := ⟨state.lanes ⟨1, by decide⟩ % goldilocksP,
    Nat.mod_lt _ (by decide)⟩

/-- One value-level extension squeeze: gate, then read lanes zero and one. -/
def squeezeKValue (constants : Constants) (state : Poseidon2Duplex.State) :
    K × Poseidon2Duplex.State :=
  let next := Poseidon2Duplex.gate constants state
  (challengeValue next, next)

/-- The carried expression returned by the symbolic planner decodes to the
two rate lanes returned by the value-level duplex. -/
theorem decoded_squeezeK
    (base : Nat) (constants : Constants)
    (assignment : Nat → Nat) (builder : SymbolicDuplex.Builder)
    (constantWire : assignment 0 = 1)
    (valid :
      Valid base constants assignment
        (SymbolicDuplex.squeezeK base builder).2) :
    (KFixedPhaseSumCheck.decodeCarried assignment
          (SymbolicDuplex.squeezeK base builder).1,
        decodedBuilder assignment
          (SymbolicDuplex.squeezeK base builder).2) =
      squeezeKValue constants (decodedBuilder assignment builder) := by
  have gateEq :=
    decodedBuilder_gate base constants assignment builder constantWire valid
  apply Prod.ext
  · apply KBridge.toPair_injective
    rw [KFixedPhaseSumCheck.toPair_decodeCarried]
    unfold SymbolicDuplex.squeezeK squeezeKValue challengeValue
    simp only [KBridge.toPair, carriedValue, Pair.mk.injEq]
    refine ⟨?_, ?_⟩
    · have laneEq := congrArg
        (fun state => state.lanes ⟨0, by decide⟩) gateEq
      simp only [decodedBuilder, evalState] at laneEq
      change lcEval assignment
          ((SymbolicDuplex.gate base builder).lanes ⟨0, by decide⟩) =
        (Poseidon2Duplex.gate constants
          (decodedBuilder assignment builder)).lanes ⟨0, by decide⟩ %
            goldilocksP
      have laneEq' :
          lcEval assignment
              ((SymbolicDuplex.gate base builder).lanes ⟨0, by decide⟩) =
            (Poseidon2Duplex.gate constants
              (decodedBuilder assignment builder)).lanes ⟨0, by decide⟩ := by
        simpa only [decodedBuilder] using laneEq
      rw [← laneEq', Nat.mod_eq_of_lt
        (lcEval_lt assignment _)]
    · have laneEq := congrArg
        (fun state => state.lanes ⟨1, by decide⟩) gateEq
      simp only [decodedBuilder, evalState] at laneEq
      change lcEval assignment
          ((SymbolicDuplex.gate base builder).lanes ⟨1, by decide⟩) =
        (Poseidon2Duplex.gate constants
          (decodedBuilder assignment builder)).lanes ⟨1, by decide⟩ %
            goldilocksP
      have laneEq' :
          lcEval assignment
              ((SymbolicDuplex.gate base builder).lanes ⟨1, by decide⟩) =
            (Poseidon2Duplex.gate constants
              (decodedBuilder assignment builder)).lanes ⟨1, by decide⟩ := by
        simpa only [decodedBuilder] using laneEq
      rw [← laneEq', Nat.mod_eq_of_lt
        (lcEval_lt assignment _)]
  · exact gateEq

end Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics
