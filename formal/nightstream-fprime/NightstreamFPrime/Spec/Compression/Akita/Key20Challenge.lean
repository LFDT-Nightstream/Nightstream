import NightstreamFPrime.Spec.Compression.Akita.ChallengeFamily
import NightstreamFPrime.Spec.Compression.Akita.Key20Tower
import NightstreamFPrime.Spec.Compression.Akita.TrigBuild

/-! The challenge families of the schedule file `crates/ns-compress/artifacts/key20.aks` (Akita,
ePrint 2026/1983, revision of 2026-11-23, Section 3.2 and Definition 9.2, E1).

The Akita code (revision `d50f700f`, `crates/akita-challenges/src/config.rs`) has the family
`(31, 10)` for the degree 64 and `(31, 0)` for the degree 128. The levels with the Euclidean
route have a filter on the base family `(31, 11)` for the degree 64 and `(31, 0)` for the degree
128.

- `units_of_mem` and `sparse_of_mem`: the challenges of a decoder into a sparse family
  `(n₁, n₂)` of a degree `d = 2^{n+1}` have units as differences, and the norms
  `ω = n₁ + 2·n₂` and `c_max`.
- `Kept` states that `2^128` members or more of each filtered family pass the filter of the
  code, as Akita Table 12 certifies.
- `rootMass` and `rootPeak` are the norms of the families of the three groups of the root. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Key20

open NightstreamFPrime.Spec.Compression.Akita.Extract

attribute [local instance low] Classical.propDecidable

section Sparse

variable {Slot : Type} {d ones twos : ℕ} [NeZero d] {decode : Slot → Option (Rq d)}

/-- The challenges of a decoder into a sparse family of a degree `2^{n+1}` have units as
differences (Akita Definition 9.2, E1). -/
theorem units_of_mem (n : ℕ) (power : 2 ^ (n + 1) = d)
    (mem : ∀ {slot challenge}, decode slot = some challenge →
      challenge ∈ SparseFamily d ones twos) :
    UnitDifferences {challenge | ∃ slot, decode slot = some challenge} :=
  fun c ⟨_, decoded⟩ c' ⟨_, decoded'⟩ =>
    SparseFamily.unitDifferences n power ones twos c (mem decoded) c' (mem decoded')

/-- The challenges of a decoder into a sparse family have `‖c‖_1 = n₁ + 2·n₂` and
`‖c‖_∞ = c_max`. -/
theorem sparse_of_mem
    (mem : ∀ {slot challenge}, decode slot = some challenge →
      challenge ∈ SparseFamily d ones twos)
    (challenge : Rq d) (decoded : ∃ slot, decode slot = some challenge) :
    Sparse (ones + 2 * twos) (peak ones twos) challenge :=
  SparseFamily.sparse (mem decoded.choose_spec)

end Sparse

/-- The filtered base families of the Akita code have `2^128` members or more that pass the
filter of the code: `(31, 11)` of the degree 64 for `Γ` = 18, and `(31, 0)` of the degree 128 for
`Γ` = 13. Akita Table 12 certifies these counts with the lower bound `|C_0|·p_0 ≥ 2^128`. -/
def Kept : Prop :=
  2 ^ 128 ≤ Nat.card {c ∈ SparseFamily 64 31 11 | TrigTable.codeAccepts 64 18 c} ∧
    2 ^ 128 ≤ Nat.card {c ∈ SparseFamily 128 31 0 | TrigTable.codeAccepts 128 13 c}

/-- The count of the coefficients `±2` of the family of each group of the root: the key group
has the degree 128, and the groups of `W` and `V` have the degree 64. -/
def rootTwos : Fin 3 → ℕ := ![0, 10, 10]

/-- `ω` of the family of each group of the root: 31, 51 and 51. -/
def rootMass (g : Fin 3) : ℕ := 31 + 2 * rootTwos g

/-- `c_max` of the family of each group of the root: 1, 2 and 2. -/
def rootPeak (g : Fin 3) : ℕ := peak 31 (rootTwos g)

end NightstreamFPrime.Spec.Compression.Akita.Key20
