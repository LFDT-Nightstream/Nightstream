import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Types

/-!
Finite executable equality for concrete Phi81 verifier carriers.

Protocol: SuperNeo NIFS.
Phase: shared finite comparison substrate for incoming authority, challenge
binding, and outgoing `Pi_DEC` recomposition.
Constraint family: semantic comparisons only; this file emits no rows.

Owns: explicit finite equality for fields, extension elements, fixed-size
functions, `RingF`, `RingK`, commitments, public inputs, points, and arrays of
evaluations, together with exact Boolean/equality theorems.

Does not own: relation-structure equality, transcript execution, protocol
acceptance, Rust/R1CS lowering, physical rows, costs, or row removal.

Emits constraints: no.

Authority boundary: every comparison enumerates the complete typed carrier.
No digest, truncated prefix, default array read, or noncomputable equality
decision may replace those coordinates. Relation structures are deliberately
excluded because their current semantic-function representation is not an
executable public encoding.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.concrete.equality.function` | compare every coordinate of one typed finite function | computed | `functionEqual`, `functionEqual_eq_true_iff` |
| `nifs.concrete.equality.ring_f` | compare all 54 base-ring coefficients | computed | `ringFEqual`, `ringFEqual_eq_true_iff` |
| `nifs.concrete.equality.ring_k` | compare all 54 extension-ring coefficients | computed | `ringKEqual`, `ringKEqual_eq_true_iff` |
| `nifs.concrete.equality.commitment` | compare every verifier row and every base-ring coefficient | computed | `commitmentEqual`, `commitmentEqual_eq_true_iff` |
| `nifs.concrete.equality.public_input` | compare the complete aligned public carrier | computed | `publicInputEqual`, `publicInputEqual_eq_true_iff` |
| `nifs.concrete.equality.point` | compare the complete typed row-cube point | computed | `pointEqual`, `pointEqual_eq_true_iff` |
| `nifs.concrete.equality.evaluations` | compare array length, matrix order, and every `RingK` lane | computed | `evaluationsEqual`, `evaluationsEqual_eq_true_iff` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CarrierEquality

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

universe u

/-- Exact finite comparison of two functions over the same typed domain. -/
def functionEqual {count : Nat} {Value : Type u}
    (valueEqual : Value -> Value -> Bool)
    (left right : Fin count -> Value) : Bool :=
  (List.finRange count).all fun coordinate =>
    valueEqual (left coordinate) (right coordinate)

/-- Complete finite-function comparison is extensional equality whenever the
leaf comparison is exact. -/
theorem functionEqual_eq_true_iff
    {count : Nat} {Value : Type u}
    (valueEqual : Value -> Value -> Bool)
    (valueEqualExact : forall left right,
      valueEqual left right = true <-> left = right)
    (left right : Fin count -> Value) :
    functionEqual valueEqual left right = true <-> left = right := by
  constructor
  · intro equal
    funext coordinate
    exact (valueEqualExact _ _).mp
      ((List.all_eq_true.mp equal) coordinate (by simp))
  · intro equal
    subst right
    apply List.all_eq_true.mpr
    intro coordinate _member
    exact (valueEqualExact _ _).mpr rfl

/-- Canonical equality for one Goldilocks field value. -/
def fieldEqual (left right : F) : Bool :=
  decide (left = right)

theorem fieldEqual_eq_true_iff (left right : F) :
    fieldEqual left right = true <-> left = right := by
  exact decide_eq_true_iff

/-- Canonical equality for one quadratic-extension value. -/
def extensionEqual (left right : K) : Bool :=
  decide (left = right)

theorem extensionEqual_eq_true_iff (left right : K) :
    extensionEqual left right = true <-> left = right := by
  exact decide_eq_true_iff

/-- Compare all coefficients of one base-field Phi81 ring element. -/
def ringFEqual (left right : RingF) : Bool :=
  functionEqual fieldEqual left right

theorem ringFEqual_eq_true_iff (left right : RingF) :
    ringFEqual left right = true <-> left = right := by
  exact functionEqual_eq_true_iff fieldEqual fieldEqual_eq_true_iff left right

/-- Compare all coefficients of one extension-field Phi81 ring element. -/
def ringKEqual (left right : RingK) : Bool :=
  functionEqual extensionEqual left right

theorem ringKEqual_eq_true_iff (left right : RingK) :
    ringKEqual left right = true <-> left = right := by
  exact
    functionEqual_eq_true_iff extensionEqual extensionEqual_eq_true_iff
      left right

/-- Compare every row of one typed public Ajtai commitment. -/
def commitmentEqual {verifierRows : Nat}
    (left right : CommitmentValue verifierRows) : Bool :=
  functionEqual ringFEqual left right

theorem commitmentEqual_eq_true_iff {verifierRows : Nat}
    (left right : CommitmentValue verifierRows) :
    commitmentEqual left right = true <-> left = right := by
  exact
    functionEqual_eq_true_iff ringFEqual ringFEqual_eq_true_iff left right

/-- Compare every coordinate of one exact aligned public input. -/
def publicInputEqual {shape : Shape}
    (left right : PublicInput shape) : Bool :=
  functionEqual fieldEqual left right

theorem publicInputEqual_eq_true_iff {shape : Shape}
    (left right : PublicInput shape) :
    publicInputEqual left right = true <-> left = right := by
  exact functionEqual_eq_true_iff fieldEqual fieldEqual_eq_true_iff left right

/-- Length-sensitive list comparison parameterized by an exact leaf
comparison. -/
def listEqual {Value : Type u}
    (valueEqual : Value -> Value -> Bool) : List Value -> List Value -> Bool
  | [], [] => true
  | left :: leftRest, right :: rightRest =>
      valueEqual left right && listEqual valueEqual leftRest rightRest
  | _, _ => false

theorem listEqual_eq_true_iff
    {Value : Type u}
    (valueEqual : Value -> Value -> Bool)
    (valueEqualExact : forall left right,
      valueEqual left right = true <-> left = right)
    (left right : List Value) :
    listEqual valueEqual left right = true <-> left = right := by
  induction left generalizing right with
  | nil =>
      cases right <;> simp [listEqual]
  | cons leftHead leftTail inductionHypothesis =>
      cases right with
      | nil => simp [listEqual]
      | cons rightHead rightTail =>
          simp [listEqual, valueEqualExact, inductionHypothesis]

/-- Compare the complete coordinate list of one dimension-checked evaluation
point. The dimension proof is proof-irrelevant and carries no runtime data. -/
def pointEqual {shape : Shape}
    (left right : Point shape) : Bool :=
  listEqual extensionEqual left.coordinates right.coordinates

theorem pointEqual_eq_true_iff {shape : Shape}
    (left right : Point shape) :
    pointEqual left right = true <-> left = right := by
  rw [pointEqual,
    listEqual_eq_true_iff extensionEqual extensionEqual_eq_true_iff]
  constructor
  · intro coordinates
    cases left with
    | mk leftCoordinates leftDimension =>
        cases right with
        | mk rightCoordinates rightDimension =>
            dsimp at coordinates
            subst rightCoordinates
            rfl
  · intro equal
    exact congrArg (fun point : Point shape => point.coordinates) equal

/-- Exact array comparison without requiring decidable equality for the
function-valued element carrier. -/
def arrayEqual {Value : Type u}
    (valueEqual : Value -> Value -> Bool)
    (left right : Array Value) : Bool :=
  listEqual valueEqual left.toList right.toList

theorem arrayEqual_eq_true_iff
    {Value : Type u}
    (valueEqual : Value -> Value -> Bool)
    (valueEqualExact : forall left right,
      valueEqual left right = true <-> left = right)
    (left right : Array Value) :
    arrayEqual valueEqual left right = true <-> left = right := by
  rw [arrayEqual, listEqual_eq_true_iff valueEqual valueEqualExact]
  exact Array.toList_inj

/-- Compare matrix-array length, ordering, and every RingK coefficient. -/
def evaluationsEqual (left right : Array Evaluation) : Bool :=
  arrayEqual ringKEqual left right

theorem evaluationsEqual_eq_true_iff (left right : Array Evaluation) :
    evaluationsEqual left right = true <-> left = right := by
  exact arrayEqual_eq_true_iff ringKEqual ringKEqual_eq_true_iff left right

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CarrierEquality
