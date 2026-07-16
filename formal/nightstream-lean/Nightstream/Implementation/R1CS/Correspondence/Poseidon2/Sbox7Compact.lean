import Nightstream.Implementation.R1CS.Core.LinearSubstitution

/-!
Contract: model-level equivalence of the compact selector-gated Poseidon2
S-box row and the four-row topological multiplication schedule.

Owns: decoded selector/input/output linear combinations, the active relation
`out = input^7`, the source intermediates `x2`, `x4`, `x6`, their unique
canonical materializer, and the exact degree/role contract.

Does not own: production column discovery, centered-slot allocation, generated
layout or Rust conformance, selector authority, or authorization to remove
source rows.

Emits constraints: no. This file compares two mathematical relations.

Authority boundary: the compact row implies `out = input^7` only when its
selector LC is verifier-bound to one. A zero selector makes the emitted row
vacuous.

Assurance tier: model-level.

| Predicate/theorem | Mathematical obligation | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `emittedRowHolds_iff_activeHolds` | active compact row | Selector-gated degree-eight row is exactly `out = input^7` | decoded selector equals one | no |
| `activeHolds_iff_exists_topological` | S-box semantics | Compact relation iff four topological multiplication equations have a witness | decoded input/output | no |
| `activeHolds_iff_existsUnique_canonical` | witness construction | The topological witness exists uniquely and is canonical | compact relation | no |
| `emittedRow_transport_iff` | outer lowering | Sparse LC substitution preserves the emitted row exactly | exact column expansion | no |
| `exact_degree_role_contract` | gate vocabulary | Local schema roles are selector/input/output; active degree is 7 and gated degree is 8 | fixed polynomial | no |
| `selectorAuthority_necessary` | selector authority | A zero selector accepts a concrete invalid input/output pair | explicit witness | no |
| `compactRow_necessary` | check necessity | Omitting the row accepts a concrete invalid S-box output | explicit witness | no |
-/

namespace Nightstream.Implementation.R1CS.Poseidon2Sbox7Compact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.LinearSubstitution

/-- Local schema roles consumed by one compact S-box row. These are not
generated production-matrix columns. -/
inductive Role where
  | selector
  | input
  | output
deriving DecidableEq, Repr

/-- Local schema order only; these are not generated production-matrix
indices. -/
def roleIndex : Role → Nat
  | .selector => 0
  | .input => 1
  | .output => 2

/-- Degree of `input^7 = output` after fixing the selector to one. -/
def activeDegree : Nat := 7

/-- Degree of `selector * (input^7 - output) = 0`. -/
def gatedDegree : Nat := 8

/-- The two selector-gated monomials have degrees eight and two. -/
def inputMonomialDegree : Nat := 1 + 7
def outputMonomialDegree : Nat := 1 + 1

theorem exact_degree_role_contract :
    roleIndex .selector = 0 ∧
      roleIndex .input = 1 ∧
      roleIndex .output = 2 ∧
      activeDegree = 7 ∧
      inputMonomialDegree = 8 ∧
      outputMonomialDegree = 2 ∧
      gatedDegree = 8 := by
  decide

/-- Three decoded sparse LCs in the exact emitted row. -/
structure Gate where
  selector : List (Nat × Nat)
  input : List (Nat × Nat)
  output : List (Nat × Nat)
deriving DecidableEq, Repr

def selectorValue (assignment : Nat → Nat) (gate : Gate) : Nat :=
  lcEval assignment gate.selector

def inputValue (assignment : Nat → Nat) (gate : Gate) : Nat :=
  lcEval assignment gate.input

def outputValue (assignment : Nat → Nat) (gate : Gate) : Nat :=
  lcEval assignment gate.output

/-- Goldilocks seventh power used by Poseidon2. -/
def pow7 (input : Nat) : Nat :=
  input ^ 7 % goldilocksP

/-- Exact active S-box relation. -/
def ActiveHolds (assignment : Nat → Nat) (gate : Gate) : Prop :=
  outputValue assignment gate = pow7 (inputValue assignment gate)

instance (assignment : Nat → Nat) (gate : Gate) :
    Decidable (ActiveHolds assignment gate) := by
  unfold ActiveHolds
  infer_instance

/-- Exact emitted selector-gated row:
`selector * input^7 = selector * output`. -/
def EmittedRowHolds (assignment : Nat → Nat) (gate : Gate) : Prop :=
  selectorValue assignment gate * pow7 (inputValue assignment gate) %
      goldilocksP =
    selectorValue assignment gate * outputValue assignment gate %
      goldilocksP

instance (assignment : Nat → Nat) (gate : Gate) :
    Decidable (EmittedRowHolds assignment gate) := by
  unfold EmittedRowHolds
  infer_instance

/-- Selector authority turns the emitted gated equation into the active
seventh-power relation. -/
theorem emittedRowHolds_iff_activeHolds
    (assignment : Nat → Nat) (gate : Gate)
    (selectorOne : selectorValue assignment gate = 1) :
    EmittedRowHolds assignment gate ↔ ActiveHolds assignment gate := by
  have outputCanonical : outputValue assignment gate < goldilocksP := by
    unfold outputValue lcEval
    exact Nat.mod_lt _ (by decide)
  simp [EmittedRowHolds, ActiveHolds, selectorOne, pow7,
    Nat.mod_eq_of_lt outputCanonical, eq_comm]

/-- Translate all three decoded LCs through the same sparse outer image. -/
def Gate.translate (expansion : ColumnExpansion) (gate : Gate) : Gate where
  selector := terms expansion gate.selector
  input := terms expansion gate.input
  output := terms expansion gate.output

theorem selectorValue_translate
    (expansion : ColumnExpansion) (encoded : Nat → Nat) (gate : Gate) :
    selectorValue encoded (gate.translate expansion) =
      selectorValue (assignment expansion encoded) gate := by
  exact lcEval_terms expansion encoded gate.selector

theorem inputValue_translate
    (expansion : ColumnExpansion) (encoded : Nat → Nat) (gate : Gate) :
    inputValue encoded (gate.translate expansion) =
      inputValue (assignment expansion encoded) gate := by
  exact lcEval_terms expansion encoded gate.input

theorem outputValue_translate
    (expansion : ColumnExpansion) (encoded : Nat → Nat) (gate : Gate) :
    outputValue encoded (gate.translate expansion) =
      outputValue (assignment expansion encoded) gate := by
  exact lcEval_terms expansion encoded gate.output

/-- The exact compact row commutes with arbitrary sparse LC substitution. -/
theorem emittedRow_transport_iff
    (expansion : ColumnExpansion) (encoded : Nat → Nat) (gate : Gate) :
    EmittedRowHolds encoded (gate.translate expansion) ↔
      EmittedRowHolds (assignment expansion encoded) gate := by
  simp only [EmittedRowHolds, selectorValue_translate, inputValue_translate,
    outputValue_translate]

theorem active_transport_iff
    (expansion : ColumnExpansion) (encoded : Nat → Nat) (gate : Gate) :
    ActiveHolds encoded (gate.translate expansion) ↔
      ActiveHolds (assignment expansion encoded) gate := by
  simp only [ActiveHolds, inputValue_translate, outputValue_translate]

/-- The three source-only intermediates allocated before the S-box output. -/
structure Intermediates where
  x2 : Nat
  x4 : Nat
  x6 : Nat
deriving DecidableEq, Repr

def Intermediates.Canonical (witness : Intermediates) : Prop :=
  witness.x2 < goldilocksP ∧
    witness.x4 < goldilocksP ∧
    witness.x6 < goldilocksP

instance (witness : Intermediates) : Decidable witness.Canonical := by
  unfold Intermediates.Canonical
  infer_instance

/-- Exact four-row topological source relation:

`x2=x*x`, `x4=x2*x2`, `x6=x2*x4`, `out=x6*x`.
-/
def TopologicalHolds (input output : Nat)
    (witness : Intermediates) : Prop :=
  witness.x2 = input * input % goldilocksP ∧
    witness.x4 = witness.x2 * witness.x2 % goldilocksP ∧
    witness.x6 = witness.x2 * witness.x4 % goldilocksP ∧
    output = witness.x6 * input % goldilocksP

instance (input output : Nat) (witness : Intermediates) :
    Decidable (TopologicalHolds input output witness) := by
  unfold TopologicalHolds
  infer_instance

/-- Canonical topological witness determined solely by the input. -/
def materialize (input : Nat) : Intermediates :=
  let x2 := input * input % goldilocksP
  let x4 := x2 * x2 % goldilocksP
  let x6 := x2 * x4 % goldilocksP
  ⟨x2, x4, x6⟩

theorem materialize_canonical (input : Nat) :
    (materialize input).Canonical := by
  refine ⟨?_, ?_, ?_⟩ <;>
    simp [materialize] <;>
    exact Nat.mod_lt _ (by decide)

private theorem x2_eq_pow (input : Nat) :
    input * input % goldilocksP = input ^ 2 % goldilocksP := by
  simp [Nat.pow_succ, Nat.mul_comm]

private theorem x4_eq_pow (input : Nat) :
    (input * input % goldilocksP) *
        (input * input % goldilocksP) % goldilocksP =
      input ^ 4 % goldilocksP := by
  rw [Nat.mod_mul_mod, Nat.mul_mod_mod]
  simp [Nat.pow_succ, Nat.mul_assoc]

private theorem x6_eq_pow (input : Nat) :
    (input * input % goldilocksP) *
        ((input * input % goldilocksP) *
          (input * input % goldilocksP) % goldilocksP) %
        goldilocksP =
      input ^ 6 % goldilocksP := by
  rw [x4_eq_pow, x2_eq_pow, Nat.mod_mul_mod, Nat.mul_mod_mod]
  rw [← Nat.pow_add]

private theorem materialized_output_eq_pow7 (input : Nat) :
    (materialize input).x6 * input % goldilocksP = pow7 input := by
  simp only [materialize]
  rw [x6_eq_pow]
  simp [pow7, Nat.pow_succ]

/-- Four source rows imply the compact seventh-power relation. -/
theorem topological_sound
    {input output : Nat} {witness : Intermediates}
    (holds : TopologicalHolds input output witness) :
    output = pow7 input := by
  rcases holds with ⟨x2, x4, x6, outputRow⟩
  rw [outputRow, x6, x4, x2]
  exact materialized_output_eq_pow7 input

/-- The canonical materializer satisfies all four source equations whenever
the compact relation holds. -/
theorem topological_complete
    {input output : Nat} (compact : output = pow7 input) :
    TopologicalHolds input output (materialize input) := by
  refine ⟨rfl, rfl, rfl, ?_⟩
  rw [compact]
  exact (materialized_output_eq_pow7 input).symm

/-- Topological source witnesses are sequentially determined by the input. -/
theorem topological_unique
    {input output : Nat} {witness : Intermediates}
    (holds : TopologicalHolds input output witness) :
    witness = materialize input := by
  cases witness with
  | mk witnessX2 witnessX4 witnessX6 =>
    simp only [TopologicalHolds] at holds
    rcases holds with ⟨x2, x4, x6, _⟩
    simp only [materialize]
    subst witnessX2
    subst witnessX4
    subst witnessX6
    rfl

theorem topological_canonical
    {input output : Nat} {witness : Intermediates}
    (holds : TopologicalHolds input output witness) :
    witness.Canonical := by
  rw [topological_unique holds]
  exact materialize_canonical input

/-- Value-level semantic equivalence used by both decoded gate and source
relation instantiations. -/
theorem pow7_iff_exists_topological (input output : Nat) :
    output = pow7 input ↔
      ∃ witness, TopologicalHolds input output witness := by
  constructor
  · intro compact
    exact ⟨materialize input, topological_complete compact⟩
  · rintro ⟨witness, holds⟩
    exact topological_sound holds

/-- Explicit unique-existence relation for the canonical topological witness. -/
def HasUniqueCanonicalWitness (input output : Nat) : Prop :=
  ∃ witness : Intermediates,
    witness.Canonical ∧
      TopologicalHolds input output witness ∧
      ∀ other : Intermediates,
        other.Canonical ∧ TopologicalHolds input output other →
          other = witness

/-- The compact row admits exactly one canonical topological witness. -/
theorem pow7_iff_existsUnique_canonical (input output : Nat) :
    output = pow7 input ↔
      HasUniqueCanonicalWitness input output := by
  constructor
  · intro compact
    refine ⟨materialize input,
      materialize_canonical input, topological_complete compact, ?_⟩
    intro other otherHolds
    exact topological_unique otherHolds.2
  · rintro ⟨witness, _, witnessHolds, _⟩
    exact topological_sound witnessHolds

theorem activeHolds_iff_exists_topological
    (assignment : Nat → Nat) (gate : Gate) :
    ActiveHolds assignment gate ↔
      ∃ witness,
        TopologicalHolds (inputValue assignment gate)
          (outputValue assignment gate) witness := by
  exact pow7_iff_exists_topological
    (inputValue assignment gate) (outputValue assignment gate)

theorem activeHolds_iff_existsUnique_canonical
    (assignment : Nat → Nat) (gate : Gate) :
    ActiveHolds assignment gate ↔
      HasUniqueCanonicalWitness (inputValue assignment gate)
        (outputValue assignment gate) := by
  exact pow7_iff_existsUnique_canonical
    (inputValue assignment gate) (outputValue assignment gate)

theorem emittedRowHolds_iff_exists_topological
    (assignment : Nat → Nat) (gate : Gate)
    (selectorOne : selectorValue assignment gate = 1) :
    EmittedRowHolds assignment gate ↔
      ∃ witness,
        TopologicalHolds (inputValue assignment gate)
          (outputValue assignment gate) witness := by
  rw [emittedRowHolds_iff_activeHolds assignment gate selectorOne,
    activeHolds_iff_exists_topological]

/-- Relation left after omitting the only compact S-box row. -/
def WithoutCompactRow (_assignment : Nat → Nat) (_gate : Gate) : Prop :=
  True

instance (assignment : Nat → Nat) (gate : Gate) :
    Decidable (WithoutCompactRow assignment gate) := by
  unfold WithoutCompactRow
  infer_instance

private def necessityGate : Gate where
  selector := [(0, 1)]
  input := [(1, 1)]
  output := [(2, 1)]

private def necessityAssignment : Nat → Nat
  | 0 => 1
  | 1 => 2
  | _ => 0

private def zeroSelectorAssignment : Nat → Nat
  | 0 => 0
  | 1 => 2
  | _ => 0

/-- Selector-one authority is necessary: with selector zero, input two and
output zero satisfy the emitted row while violating the active relation. -/
theorem selectorAuthority_necessary :
    selectorValue zeroSelectorAssignment necessityGate = 0 ∧
      ¬ ActiveHolds zeroSelectorAssignment necessityGate ∧
      EmittedRowHolds zeroSelectorAssignment necessityGate := by
  native_decide

/-- The compact row is inclusion-necessary: its omission accepts input two
with output zero, while the exact degree-seven relation rejects it. -/
theorem compactRow_necessary :
    WithoutCompactRow necessityAssignment necessityGate ∧
      selectorValue necessityAssignment necessityGate = 1 ∧
      ¬ EmittedRowHolds necessityAssignment necessityGate := by
  native_decide

end Nightstream.Implementation.R1CS.Poseidon2Sbox7Compact
