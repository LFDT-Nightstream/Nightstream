import Nightstream.Implementation.R1CS.Core.ProjectionLengths

/-!
Contract: the minimal Lean-owned coefficient trace checked by the canonical
PiRLC quotient-identity program.

The historical `ProjectionProgram.ProjectionTrace` also stores a power ladder,
materialized evaluations, and multiplication traces.  Those fields describe a
different physical encoding and are not inputs to the canonical program.  This
module owns the smaller boundary that program actually consumes:

- one verifier-selected extension-field challenge;
- the challenge/input coefficient columns of every product;
- output and quotient coefficient columns; and
- the fixed representation widths.

`ofLegacy` is a compatibility projection only.  No theorem in the selected
canonical program derives authority from the discarded legacy fields.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KProjectionTrace

open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.SuperNeo.ProjectionCheck

/-- Coefficient columns for one `rho * input` summand. -/
structure PairColumns where
  rho : List Nat
  input : List Nat
deriving DecidableEq, Repr

/-- The exact static data consumed by one canonical quotient-identity check. -/
structure Trace where
  beta : KColumns
  pairs : List PairColumns
  output : List Nat
  quotient : List Nat
  maxDegree : Nat
deriving DecidableEq, Repr

/-- Every coefficient column read by one trace is placed below `base`.
The verifier challenge columns are tracked separately because they are read by
the Horner program but do not occur in `Identity.Exact`. -/
def Trace.CoefficientsBelow (trace : Trace) (base : Nat) : Prop :=
  (∀ pair ∈ trace.pairs,
      (∀ column ∈ pair.rho, column < base) ∧
      (∀ column ∈ pair.input, column < base)) ∧
    (∀ column ∈ trace.output, column < base) ∧
    (∀ column ∈ trace.quotient, column < base)

/-- Fixed-width eligibility for the selected Phi81 identity. -/
def Trace.Valid (trace : Trace) : Prop :=
  trace.pairs ≠ [] ∧
  (∀ pair ∈ trace.pairs,
    pair.rho.length = 54 ∧ pair.input.length = 54) ∧
  trace.output.length = 54 ∧
  trace.quotient.length = 53 ∧
  trace.maxDegree = 106

instance (trace : Trace) : Decidable trace.Valid := by
  unfold Trace.Valid
  infer_instance

def PairColumns.productPolynomial (pair : PairColumns)
    (assignment : Nat → Nat) : List K :=
  Polynomial.mul
    (basePolynomial assignment pair.rho)
    (basePolynomial assignment pair.input)

/-- The exact frozen coefficient identity represented by a canonical trace. -/
def Trace.identity (trace : Trace) (assignment : Nat → Nat) : Identity K where
  lhs := Polynomial.sum
    (trace.pairs.map fun pair => pair.productPolynomial assignment)
  rhs := Polynomial.add
    (Polynomial.mul
      (basePolynomial assignment trace.quotient) Polynomial.phi81)
    (Polynomial.padRight (trace.maxDegree + 1)
      (basePolynomial assignment trace.output))
  beta := trace.beta.value assignment
  maxDegree := trace.maxDegree

def BatchIdentity (traces : List Trace) (assignment : Nat → Nat) :
    List (Identity K) :=
  traces.map fun trace => trace.identity assignment

private theorem basePolynomial_congr_below
    (left right : Nat → Nat) (columns : List Nat) (base : Nat)
    (below : ∀ column ∈ columns, column < base)
    (agree : ∀ column, column < base → left column = right column) :
    basePolynomial left columns = basePolynomial right columns := by
  unfold basePolynomial
  apply List.map_congr_left
  intro column member
  unfold baseAt
  rw [agree column (below column member)]

/-- Coefficient exactness is stable under any witness extension that preserves
the trace's source columns.  This is the batch-completeness transport: later
auxiliary blocks may change, but they cannot change an already selected
identity. -/
theorem Trace.exact_congr_below
    (trace : Trace) (left right : Nat → Nat) (base : Nat)
    (below : trace.CoefficientsBelow base)
    (agree : ∀ column, column < base → left column = right column) :
    (trace.identity left).Exact ↔ (trace.identity right).Exact := by
  have pairProducts :
      trace.pairs.map (fun pair => pair.productPolynomial left) =
        trace.pairs.map (fun pair => pair.productPolynomial right) := by
    apply List.map_congr_left
    intro pair member
    unfold PairColumns.productPolynomial
    rw [basePolynomial_congr_below left right pair.rho base
        (below.1 pair member).1 agree,
      basePolynomial_congr_below left right pair.input base
        (below.1 pair member).2 agree]
  have output :
      basePolynomial left trace.output =
        basePolynomial right trace.output :=
    basePolynomial_congr_below left right trace.output base below.2.1 agree
  have quotient :
      basePolynomial left trace.quotient =
        basePolynomial right trace.quotient :=
    basePolynomial_congr_below left right trace.quotient base below.2.2 agree
  unfold Identity.Exact Trace.identity
  rw [pairProducts, output, quotient]

theorem PairColumns.productPolynomial_length
    (pair : PairColumns) (assignment : Nat → Nat)
    (rhoLength : pair.rho.length = 54)
    (inputLength : pair.input.length = 54) :
    (pair.productPolynomial assignment).length = 107 := by
  unfold PairColumns.productPolynomial
  rw [Polynomial.length_mul]
  · simp [rhoLength, inputLength]
  · intro empty
    have lengthEq := congrArg List.length empty
    simp [rhoLength] at lengthEq
  · intro empty
    have lengthEq := congrArg List.length empty
    simp [inputLength] at lengthEq

/-- Valid canonical traces produce exactly the frozen fixed-width identity. -/
theorem Trace.identity_wellFormed
    (trace : Trace) (assignment : Nat → Nat) (valid : trace.Valid) :
    (trace.identity assignment).WellFormed := by
  rcases valid with
    ⟨pairsNonempty, pairWidths, outputLength, quotientLength, maxDegree⟩
  have mappedNonempty :
      (trace.pairs.map fun pair => pair.productPolynomial assignment) ≠ [] := by
    simpa using pairsNonempty
  have productWidths : ∀ polynomial ∈
      (trace.pairs.map fun pair => pair.productPolynomial assignment),
      polynomial.length = 107 := by
    intro polynomial member
    rcases List.mem_map.mp member with ⟨pair, pairMember, rfl⟩
    rcases pairWidths pair pairMember with ⟨rhoLength, inputLength⟩
    exact pair.productPolynomial_length assignment rhoLength inputLength
  have lhsLength : (Polynomial.sum (trace.pairs.map fun pair =>
      pair.productPolynomial assignment)).length = 107 :=
    Polynomial.length_sum_eq mappedNonempty productWidths
  have phiLength : Polynomial.phi81.length = 55 := by decide
  have quotientNonempty :
      basePolynomial assignment trace.quotient ≠ [] := by
    intro empty
    have lengthEq := congrArg List.length empty
    simp [quotientLength] at lengthEq
  have phiNonempty : Polynomial.phi81 ≠ [] := by decide
  have quotientProductLength :
      (Polynomial.mul
        (basePolynomial assignment trace.quotient)
        Polynomial.phi81).length = 107 := by
    rw [Polynomial.length_mul quotientNonempty phiNonempty,
      basePolynomial_length, quotientLength, phiLength]
  have outputPadLength :
      (Polynomial.padRight (trace.maxDegree + 1)
        (basePolynomial assignment trace.output)).length = 107 := by
    have within :
        (basePolynomial assignment trace.output).length ≤
          trace.maxDegree + 1 := by
      simp [outputLength, maxDegree]
    rw [Polynomial.length_padRight within, maxDegree]
  unfold Trace.identity Identity.WellFormed
  simp only
  rw [lhsLength, Polynomial.length_add, quotientProductLength,
    outputPadLength, Nat.max_self, maxDegree]
  decide

/-! ## Compatibility with the historical materialized trace -/

def PairColumns.ofLegacy (pair : PairTrace) : PairColumns :=
  ⟨pair.rhoColumns, pair.inputColumns⟩

/-- Forget the historical ladder/evaluation gadgets and retain exactly the
columns consumed by the canonical quotient program. -/
def Trace.ofLegacy (trace : ProjectionTrace) : Trace where
  beta := trace.ladder.beta
  pairs := trace.pairs.map PairColumns.ofLegacy
  output := trace.outputColumns
  quotient := trace.quotientColumns
  maxDegree := trace.maxDegree

theorem PairColumns.productPolynomial_ofLegacy
    (pair : PairTrace) (assignment : Nat → Nat) :
    (PairColumns.ofLegacy pair).productPolynomial assignment =
      pair.productPolynomial assignment :=
  rfl

/-- Forgetting the legacy physical trace does not change the frozen identity. -/
theorem Trace.identity_ofLegacy
    (trace : ProjectionTrace) (assignment : Nat → Nat) :
    (Trace.ofLegacy trace).identity assignment = trace.identity assignment := by
  have products :
      (trace.pairs.map PairColumns.ofLegacy).map
          (fun pair => pair.productPolynomial assignment) =
        trace.pairs.map (fun pair => pair.productPolynomial assignment) := by
    rw [List.map_map]
    exact List.map_congr_left (fun pair _ =>
      PairColumns.productPolynomial_ofLegacy pair assignment)
  unfold Trace.identity ProjectionTrace.identity Trace.ofLegacy
  rw [products]

/-- Existing historical width evidence can populate the smaller canonical
boundary, but discarded evaluation metadata is never consulted afterward. -/
theorem Trace.valid_ofLegacy
    (trace : ProjectionTrace)
    (layout : trace.LayoutValid)
    (pairsNonempty : trace.pairs ≠ [])
    (pairWidths : ∀ pair ∈ trace.pairs,
      pair.rhoColumns.length = 54 ∧ pair.inputColumns.length = 54) :
    (Trace.ofLegacy trace).Valid := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simpa [Trace.ofLegacy] using pairsNonempty
  · intro pair member
    rcases List.mem_map.mp member with ⟨legacy, legacyMember, rfl⟩
    exact pairWidths legacy legacyMember
  · exact layout.2.2.2.2.2.2.2.2.2.2.2.2.1
  · exact layout.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  · exact layout.2.2.2.2.2.2.2.2.2.2.2.2.2.2

end Nightstream.Implementation.R1CS.Canonical.KProjectionTrace
