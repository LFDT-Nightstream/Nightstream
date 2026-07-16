import Nightstream.SuperNeo.Relations

/-!
Countermodels for insufficient `Pi_DEC` digit authorization.

Owns: small arithmetic witnesses showing that signed low norm does not make
radix-two recomposition injective, that untyped variable arity admits
leading-zero aliases, and that modular recomposition without an integer range
admits wraparound aliases.

Does not own: a canonical child alphabet, the production child assignment
layout, Goldilocks encoding, concrete `Pi_DEC` verifier acceptance, Rust/R1CS
refinement, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: these are model-level necessity witnesses. Production may
use them only after an exact refinement proves which child digits, arity, and
residue equation its rows enforce. In particular, this file does not propose
binary digits as the protocol alphabet; binary examples are merely a subset
that isolates the missing arity and range obligations.

| Protocol | Phase | Omitted family | Kernel-checked invalid ambiguity |
|---|---|---|---|
| `Pi_DEC` | child opening | canonical signed representation | `(1,0)` and `(-1,1)` are low norm and recompose to the same parent |
| `Pi_DEC` | child shape | fixed arity | `[]` and `[0]` are binary and recompose equally |
| `Pi_DEC` | field boundary | no-wrap range | two fixed-length binary lists differ but recompose equally modulo two |
-/

namespace Nightstream.SuperNeo.Folding.PiDEC.Necessity.DigitAuthorization

/-- Two-digit, least-significant-first radix-two recomposition over integers. -/
def recomposeSigned2 (digits : Int × Int) : Int :=
  digits.1 + 2 * digits.2

/-- The strict magnitude-two window expressed without choosing a canonical
signed representation. -/
def signedLowNorm2 (digits : Int × Int) : Prop :=
  -1 ≤ digits.1 /\ digits.1 ≤ 1 /\
    -1 ≤ digits.2 /\ digits.2 ≤ 1

/-- Low norm plus recomposition does not uniquely authorize signed children:
`1 + 2*0 = -1 + 2*1`. -/
theorem signed_low_norm_base2_not_unique :
    recomposeSigned2 (1, 0) = recomposeSigned2 (-1, 1) /\
      signedLowNorm2 (1, 0) /\
      signedLowNorm2 (-1, 1) /\
      (1, 0) ≠ (-1, 1) := by
  constructor
  · decide
  constructor
  · unfold signedLowNorm2
    omega
  constructor
  · unfold signedLowNorm2
    omega
  · intro equal
    injection equal with first _second
    omega

/-- Recompose least-significant-first radix-two natural digits. -/
def recomposeNatDigits : List Nat -> Nat
  | [] => 0
  | digit :: rest => digit + 2 * recomposeNatDigits rest

/-- Every digit lies in the binary subset `{0,1}`. -/
def Binary (digits : List Nat) : Prop :=
  forall digit, digit ∈ digits -> digit < 2

/-- Even the binary subset is not unique if the carrier does not enforce exact
arity: a leading zero aliases the empty list. In the typed active model, `Fin k`
makes this obligation intrinsic; a raw decoder must prove that refinement. -/
theorem binary_recomposition_not_unique_without_length :
    recomposeNatDigits [] = recomposeNatDigits [0] /\
      Binary [] /\ Binary [0] /\ ([] : List Nat) ≠ [0] := by
  exact ⟨rfl, by simp [Binary], by simp [Binary], by decide⟩

/-- Fixed length and binary range still do not give uniqueness if equality is
checked only modulo a modulus and the recomposed integer range is omitted. -/
theorem fixed_length_binary_mod_recomposition_not_unique_without_range :
    ([0, 0] : List Nat).length = ([0, 1] : List Nat).length /\
      Binary [0, 0] /\ Binary [0, 1] /\
      recomposeNatDigits [0, 0] % 2 = recomposeNatDigits [0, 1] % 2 /\
      ([0, 0] : List Nat) ≠ [0, 1] := by
  exact ⟨rfl, by simp [Binary], by simp [Binary], rfl, by decide⟩

end Nightstream.SuperNeo.Folding.PiDEC.Necessity.DigitAuthorization
