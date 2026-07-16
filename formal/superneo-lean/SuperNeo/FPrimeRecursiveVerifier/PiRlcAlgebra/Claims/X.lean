import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Refinement.ExactMaterialization

/-!
Owns: exact combination semantics for the five active public-X columns and
the model-level canonical-zero predicate for every inactive full-X column.

Does not own: derivation of the full column count from `m_in`, the concrete
flat Rust matrix bridge, transcript binding, one-point projection security,
or low-level ring multiplication.

Emits constraints: no.

Authority boundary: active parent columns are bound by the exact combination;
every inactive input and parent column is explicitly zero in
`FullXCombination`. Input/rho authority and the full width remain upstream.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `XCombination` | `identities.x` | Five active degree-54 ring combinations | Fixed active prefix | No — Rust refinement open |
| `InactiveXZero` | `padding.x` | Every coefficient in every inactive column is zero | Caller-supplied full width | No — Rust refinement open |
| `inactiveXZero_iff_coefficients_zero` | `padding.x` | Column equality is exactly coefficientwise zeroing | Same carrier on both sides | No — Rust refinement open |
| `FullXCombination` | `identities.x`, `padding.x` | Active combination plus input/parent zero tails | `activeXColumns <= columnCount` | No — Rust refinement open |
| `fullXCombinationWithIntermediates_iff_direct` | `identities.x`, `padding.x` | Exact active intermediates preserve the full relation | Exact coefficient relation | No — Rust refinement open |

No theorem here yet proves that Rust's row-major `D * m_in` matrices instantiate
`FullXValue`, or that its verifier-derived active width is exactly this prefix.
That production `padding.x` refinement remains open.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra

abbrev ActiveXValue := Fin activeXColumns → RingCoefficients
abbrev FullXValue (columnCount : Nat) := Fin columnCount → RingCoefficients

/-- Project the fixed active prefix from a full-X carrier. -/
def activeXPrefix {columnCount : Nat}
    (hWidth : activeXColumns ≤ columnCount)
    (value : FullXValue columnCount) : ActiveXValue :=
  fun column => value ⟨column.1, Nat.lt_of_lt_of_le column.2 hWidth⟩

/-- Every column outside the fixed active prefix is canonically zero. -/
def InactiveXZero {columnCount : Nat} (value : FullXValue columnCount) : Prop :=
  ∀ column, activeXColumns ≤ column.1 → value column = 0

/-- Full-column zeroing is exactly zeroing each of its ring coefficients. -/
theorem inactiveXZero_iff_coefficients_zero
    {columnCount : Nat} (value : FullXValue columnCount) :
    InactiveXZero value ↔
      ∀ column coefficient, activeXColumns ≤ column.1 →
        value column coefficient = 0 := by
  constructor
  · intro hZero column coefficient hInactive
    exact congrFun (hZero column hInactive) coefficient
  · intro hZero column hInactive
    funext coefficient
    exact hZero column coefficient hInactive

def XCombination
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → ActiveXValue)
    (parent : ActiveXValue) : Prop :=
  ∀ column,
    DirectRingCombination rhos (fun inputIndex => inputs inputIndex column) (parent column)

def XCombinationWithIntermediates
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → ActiveXValue)
    (parent : ActiveXValue) : Prop :=
  ∀ column,
    IntermediateRingCombination rhos (fun inputIndex => inputs inputIndex column) (parent column)

theorem xCombinationWithIntermediates_iff_direct
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → ActiveXValue)
    (parent : ActiveXValue) :
    XCombinationWithIntermediates rhos inputs parent ↔
      XCombination rhos inputs parent := by
  simp only [XCombinationWithIntermediates, XCombination,
    intermediateRingCombination_iff_direct]

/-- Active public-X equations together with canonical inactive input/output columns. -/
def FullXCombination {columnCount : Nat}
    (hWidth : activeXColumns ≤ columnCount)
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → FullXValue columnCount)
    (parent : FullXValue columnCount) : Prop :=
  XCombination rhos
      (fun inputIndex => activeXPrefix hWidth (inputs inputIndex))
      (activeXPrefix hWidth parent) ∧
    (∀ inputIndex, InactiveXZero (inputs inputIndex)) ∧
    InactiveXZero parent

def FullXCombinationWithIntermediates {columnCount : Nat}
    (hWidth : activeXColumns ≤ columnCount)
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → FullXValue columnCount)
    (parent : FullXValue columnCount) : Prop :=
  XCombinationWithIntermediates rhos
      (fun inputIndex => activeXPrefix hWidth (inputs inputIndex))
      (activeXPrefix hWidth parent) ∧
    (∀ inputIndex, InactiveXZero (inputs inputIndex)) ∧
    InactiveXZero parent

theorem fullXCombinationWithIntermediates_iff_direct
    {columnCount : Nat}
    (hWidth : activeXColumns ≤ columnCount)
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → FullXValue columnCount)
    (parent : FullXValue columnCount) :
    FullXCombinationWithIntermediates hWidth rhos inputs parent ↔
      FullXCombination hWidth rhos inputs parent := by
  simp only [FullXCombinationWithIntermediates, FullXCombination,
    xCombinationWithIntermediates_iff_direct]

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra
