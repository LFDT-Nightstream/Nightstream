import Mathlib.Analysis.SpecialFunctions.Log.Base
import NightstreamFPrime.Spec.Compression.Akita.Extract.Recursion

/-! The Fiat–Shamir budget of the opening suffix of Akita (ePrint 2026/1983, revision of
2026-10-15, the condition (195), Corollary 10.48 with (214) and (215), and the sum (206)).

- `effectiveStrength` is `λ^eff_FS` of (214). `budget_iff` states that the condition (195) is
  `λ_FS ≤ λ^eff_FS`, and `effectiveStrength_le` is the ceiling (215).
- `LevelError` has the numbers of the stage errors of one level, and `interactiveError` is their
  sum for the levels, in the form of the sum that `Tier.Suffix` carries. `foldLevel_value` gives
  the numbers of a level with the checks `foldCheck`.
- `budget_of_levels` gives the condition (195) from one inequality of natural numbers.
- `Tier.Suffix.extraction_le_strength` is `Tier.Suffix.extraction_le` under the condition (195). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open scoped BigOperators

/-- `λ^eff_FS(sch; Q_max) = −log_2((Q_max + 1)·ε_int(sch))` (Akita (214)). -/
noncomputable def effectiveStrength (queryMax : ℕ) (error : ℝ) : ℝ :=
  -Real.logb 2 ((queryMax + 1) * error)

/-- Akita Corollary 10.48, the first claim: the condition `(Q_max + 1)·ε_int ≤ 2^{−λ_FS}` of
(195) is `λ_FS ≤ λ^eff_FS`. -/
theorem budget_iff {queryMax : ℕ} {error : ℝ} (positive : 0 < error) (strength : ℝ) :
    (queryMax + 1) * error ≤ (2 : ℝ) ^ (-strength) ↔
      strength ≤ effectiveStrength queryMax error := by
  have product : 0 < ((queryMax : ℝ) + 1) * error := mul_pos (by positivity) positive
  unfold effectiveStrength
  rw [le_neg, Real.logb_le_iff_le_rpow (by norm_num) product]

/-- Akita Corollary 10.48, (215): when the interactive error has a term `c/|E|` with `c > 0`,
`λ^eff_FS ≤ log_2|E| − log_2 c − log_2(Q_max + 1)`. -/
theorem effectiveStrength_le {queryMax : ℕ} {error count field : ℝ}
    (term : count / field ≤ error) (nonzero : 0 < count) (size : 0 < field) :
    effectiveStrength queryMax error ≤
      Real.logb 2 field - Real.logb 2 count - Real.logb 2 (queryMax + 1) := by
  have queries : (0 : ℝ) < queryMax + 1 := by positivity
  have lower : 0 < ((queryMax : ℝ) + 1) * (count / field) :=
    mul_pos queries (div_pos nonzero size)
  have mono := Real.logb_le_logb_of_le (by norm_num : (1 : ℝ) < 2) lower
    (mul_le_mul_of_nonneg_left term queries.le)
  rw [Real.logb_mul queries.ne' (div_pos nonzero size).ne',
    Real.logb_div nonzero.ne' size.ne'] at mono
  unfold effectiveStrength
  linarith

namespace Extract

open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

attribute [local instance low] Classical.propDecidable

/-- The numbers of the stage errors of one level of the opening suffix: the number `r` of blocks,
the numerator `c` of the sum `c/|E|` of the bounds of the draws of the checks, and the largest
probability `p` of one fold challenge. -/
structure LevelError where
  blocks : ℕ
  numerator : ℕ
  chance : ℝ

/-- The stage errors `r·p + c/|E|` of one level, for `|E| = field`. -/
noncomputable def LevelError.value (field : ℕ) (level : LevelError) : ℝ :=
  level.blocks * level.chance + level.numerator / field

/-- The sum of the stage errors of the levels (Akita (206)), in the form of the sum of
`Tier.suffix_push` over `Tier.suffix_last`. -/
noncomputable def interactiveError (field : ℕ) : List LevelError → ℝ
  | [] => 0
  | level :: rest => level.value field + interactiveError field rest

/-- The numbers of a level with the checks `foldCheck` (Akita (189) after `ε_cw,h`). -/
def foldLevel (blocks : ℕ) (chance : ℝ) (badBound vars batch base : ℕ) : LevelError where
  blocks := blocks
  numerator := badBound + batch + vars + vars + vars * (base / 2 + 1) + 1 + 1 + 3 * vars
  chance := chance

/-- The stage errors `r·p + ε_fold` of a level with the checks `foldCheck`, as
`Tier.suffix_push` and `Tier.suffix_last` have them, are the value of `foldLevel`. -/
theorem foldLevel_value {E : Type} [Field E] [Fintype E] [Algebra Zq E] (R : Reduction E)
    (blocks base : ℕ) (chance : ℝ) :
    blocks * chance + foldError R base =
      (foldLevel blocks chance R.badBound R.vars R.batch base).value (Fintype.card E) := by
  rw [foldError_eq]
  rfl

/-- The sum of the stage errors for fold challenges with a probability of at most `2^{−k}` each
and a field with `2^e` elements or more. -/
theorem interactiveError_le {field challengeBits fieldBits : ℕ} {levels : List LevelError}
    (chances : ∀ level ∈ levels, level.chance ≤ 1 / 2 ^ challengeBits)
    (large : 2 ^ fieldBits ≤ field) :
    interactiveError field levels ≤
      (((levels.map (·.blocks)).sum : ℕ) : ℝ) / 2 ^ challengeBits +
        (((levels.map (·.numerator)).sum : ℕ) : ℝ) / 2 ^ fieldBits := by
  induction levels with
  | nil => simp [interactiveError]
  | cons level rest induction =>
      have tail := induction fun other member => chances other (List.mem_cons_of_mem _ member)
      have chance := mul_le_mul_of_nonneg_left (chances level List.mem_cons_self)
        (Nat.cast_nonneg level.blocks)
      have term : (level.numerator : ℝ) / field ≤ level.numerator / 2 ^ fieldBits :=
        div_le_div_of_nonneg_left (Nat.cast_nonneg _) (by positivity) (by exact_mod_cast large)
      rw [mul_one_div] at chance
      simp only [interactiveError, List.map_cons, List.sum_cons, LevelError.value, Nat.cast_add,
        add_div]
      linarith

/-- The condition (195) from the numbers of the levels (Akita Corollary 10.48 for a suffix):
`(Q_max + 1)·ε_int ≤ 2^{−λ}` when each fold challenge has a probability of at most `2^{−k}`, the
field has `2^e` elements or more, and
`(Q_max + 1)·(Σ r·2^e + Σ c·2^k)·2^λ ≤ 2^{k + e}` for the sums over the levels. -/
theorem budget_of_levels {field challengeBits fieldBits queryMax strength : ℕ}
    {levels : List LevelError}
    (chances : ∀ level ∈ levels, level.chance ≤ 1 / 2 ^ challengeBits)
    (large : 2 ^ fieldBits ≤ field)
    (explicit : (queryMax + 1) * ((levels.map (·.blocks)).sum * 2 ^ fieldBits +
      (levels.map (·.numerator)).sum * 2 ^ challengeBits) * 2 ^ strength ≤
        2 ^ (challengeBits + fieldBits)) :
    ((queryMax : ℝ) + 1) * interactiveError field levels ≤ (2 : ℝ) ^ (-(strength : ℝ)) := by
  have cast : ((queryMax : ℝ) + 1) * ((((levels.map (·.blocks)).sum : ℕ) : ℝ) * 2 ^ fieldBits +
      (((levels.map (·.numerator)).sum : ℕ) : ℝ) * 2 ^ challengeBits) * 2 ^ strength ≤
        2 ^ (challengeBits + fieldBits) := by
    exact_mod_cast explicit
  have same : ((queryMax : ℝ) + 1) * ((((levels.map (·.blocks)).sum : ℕ) : ℝ) / 2 ^ challengeBits +
      (((levels.map (·.numerator)).sum : ℕ) : ℝ) / 2 ^ fieldBits) * 2 ^ strength =
        ((queryMax : ℝ) + 1) * ((((levels.map (·.blocks)).sum : ℕ) : ℝ) * 2 ^ fieldBits +
          (((levels.map (·.numerator)).sum : ℕ) : ℝ) * 2 ^ challengeBits) * 2 ^ strength /
            2 ^ (challengeBits + fieldBits) := by
    rw [pow_add]
    field_simp
  refine (mul_le_mul_of_nonneg_left (interactiveError_le chances large)
    (by positivity)).trans ?_
  rw [Real.rpow_neg (by norm_num), Real.rpow_natCast, ← one_div, le_div_iff₀ (by positivity),
    same, div_le_one (by positivity)]
  exact cast

namespace Tier

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor}
  {E Wire Coord Slot Result Source : Type} [CommRing E] [Module Zq E]
  {T : Tier O fold E Wire Coord Slot Result Source} {total : ℕ}
  [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]
  {Inner BreakT CandT KeyT : Type} {accept : Prefix Wire (Coord → Slot) total → Prop}
  {body : Structure (Prefix Wire (Coord → Slot) total) Coord Slot (ℕ → Wire) Inner BreakT CandT
    KeyT (Option (Weak O))}
  {embed : T.Break → BreakT} {error : ℝ} {loops tree : ℕ}

theorem Suffix.error_nonnegative (suffix : T.Suffix accept body embed error loops tree)
    (root : T.start = 0) : 0 ≤ error := by
  rw [← suffix.errorSum]
  exact Finset.sum_nonneg fun level _ => body.charge_nonnegative (suffix.lawful root) level

variable [Finite Wire]

/-- Akita Theorem 10.17 for a suffix from the stage 0 under the condition (195) of the schedule:
when `(Q_max + 1)·ε_int ≤ 2^{−λ}` for the sum `error` of the stage errors, each prover with at
most `Q ≤ Q_max` queries makes the verifier accept with no output of the extractor with a
probability of at most `2^{−λ} + (Q + 1)·loops/budget`. -/
theorem Suffix.extraction_le_strength (suffix : T.Suffix accept body embed error loops tree)
    (root : T.start = 0)
    (prover : OracleComp (Prefix Wire (Coord → Slot) total) (Coord → Slot) (ℕ → Wire))
    {queries queryMax : ℕ} (bounded : prover.QueryBound queries) (within : queries ≤ queryMax)
    {strength : ℝ} (budgeted : (queryMax + 1) * error ≤ (2 : ℝ) ^ (-strength)) {budget : ℕ}
    (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape (Prefix Wire (Coord → Slot) total) Coord Slot budget
          body.protocol.stages,
        (if body.protocol.accept (body.protocol.final prover table) ∧
          Checkpoint.extract body prover budget table tape = none then (1 : ℝ) else 0) ≤
      (2 : ℝ) ^ (-strength) + (queries + 1) * loops / budget := by
  refine (suffix.extraction_le root prover bounded positive).trans (add_le_add ?_ le_rfl)
  exact (mul_le_mul_of_nonneg_right (add_le_add (Nat.cast_le.mpr within) le_rfl)
    (suffix.error_nonnegative root)).trans budgeted

end Tier

end Extract

end NightstreamFPrime.Spec.Compression.Akita
