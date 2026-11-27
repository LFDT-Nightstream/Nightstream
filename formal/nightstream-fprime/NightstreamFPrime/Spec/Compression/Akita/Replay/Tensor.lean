import NightstreamFPrime.Spec.Compression.Akita.Replay.Range
import NightstreamFPrime.Spec.Compression.Akita.Replay.Relation
import NightstreamFPrime.Spec.Compression.Akita.Tensor

/-! The tensor reduction of one claim of the pinned Akita verifier on words (Akita revision
`d50f700`, `akita-verifier/src/stages/opening_claims/extension_claim.rs`; ePrint 2026/1983,
Section 3.7), and its relation to `tensorCheck` of the specification.

- `tensorRun` checks that the opening is the opening of the column partials,
  `v = (1 − r_0)·S_0 + r_0·S_1` (`derive_tensor_extension_opening_claim_from_partials`), runs
  the plain sum-check for `(1 − η)·row_0 + η·row_1` of the row partials
  (`tensor_row_partials_from_columns`, `tensor_reduction_claim_from_rows`), and checks that the
  final claim `θ·f̄(ρ)` of the prover is the end of the sum-check (`final_claims`,
  `expected_batched_final`). `none` is a rejection.
- `tensorRun_runs`: `tensorCheck` on the path of the messages and the challenges of the code ends
  in the point and the final claim of the code, with the factor `θ = A_η(ρ)` of the
  specification.
- `tensorFactor` is `θ` as the code computes it (`tensor_equality_factor_eval_at_point` of
  `akita-types/src/extension_opening_reduction.rs`): a state of the two coordinates of
  `ẽq(tail, x)` in the basis `(1, u)`, one step `factorStep` for each pair of a coordinate of the
  tail and of `ρ`, then `(1 − η)·s_0 + η·s_1`. `tensorFactor_into`: it is the factor
  `mle ρ (tensorWeight tail η)` of `tensorCheck`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

/-- `(1 − η)·row_0 + η·row_1` for the rows `(S_0.c0, S_1.c0)` and `(S_0.c1, S_1.c1)`. -/
def tensorInput (p0 p1 η : Ext) : Ext :=
  (Ext.one - η) * ⟨p0.c0, p1.c0⟩ + η * ⟨p0.c1, p1.c1⟩

def tensorRun (ν : ℕ) (point : List Ext) (v p0 p1 η : Ext) (rounds : List (List Ext × Ext))
    (final : Ext) : Option Ended :=
  match point with
  | [] => none
  | head :: tail =>
      if tail.length = ν then
        if v.same ((Ext.one - head) * p0 + head * p1) then
          if (plainRun (tensorInput p0 p1 η) rounds).claim.same final then
            some { plainRun (tensorInput p0 p1 η) rounds with claim := final }
          else none
        else none
      else none

section Spec

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- The path of `tensorCheck`: the column partials and, when the opening check passes, `η`, the
rounds and the final claim. -/
noncomputable def tensorPath (E : Type) [Field E] [Algebra Zq E] [Degree2 E] (ν : ℕ)
    (point : List Ext) (v p0 p1 η : Ext) (rounds : List (List Ext × Ext)) (final : Ext) :
    List Move :=
  match point with
  | [] => []
  | head :: tail =>
      if tail.length = ν then
        Move.recv ((p0.into, p1.into) : E × E) ::
          if v.same ((Ext.one - head) * p0 + head * p1) then
            Move.draw (η.into : E) ::
              (plainPath (Ext.into (E := E)) 1 (tensorInput p0 p1 η) rounds ++
                [Move.recv (final.into : E)])
          else []
      else []

omit [Fintype E] [DecidableEq E] in
theorem into_pair (a b : Fq) : ((⟨a, b⟩ : Ext).into : E) = Degree2.pair a.toZq b.toZq := rfl

omit [Fintype E] [DecidableEq E] in
theorem into_tensorInput (p0 p1 η : Ext) :
    ((tensorInput p0 p1 η).into : E) =
      (1 - η.into) * tensorRow ((p0.into, p1.into) : E × E) false +
        η.into * tensorRow ((p0.into, p1.into) : E × E) true := by
  simp only [tensorInput, Ext.into_add, Ext.into_mul, Ext.into_sub, Ext.into_one, tensorRow,
    re_into, im_into, into_pair, Bool.false_eq_true, ite_false, ite_true]

/-- `tensorCheck` of the specification, on the path of the messages and the challenges of the
code, ends in the result of `tensorRun`: `none` when the opening check or the check of the final
claim fails, and otherwise the point `ρ`, the factor `θ = A_η(ρ)` and the final claim. -/
theorem tensorRun_runs (ν : ℕ) (w : Table) (point : List Ext) (v p0 p1 η : Ext)
    (rounds : List (List Ext × Ext)) (final : Ext) (count : rounds.length = ν)
    (lengths : ∀ x ∈ rounds, x.1.length = 2) :
    (tensorCheck ν w (point.map (Ext.into (E := E))) v.into).Runs
      (tensorPath E ν point v p0 p1 η rounds final)
      ((tensorRun ν point v p0 p1 η rounds final).map fun out =>
        (out.point.map Ext.into,
          mle (out.point.map Ext.into) (tensorWeight (point.tail.map Ext.into) η.into),
          out.claim.into)) := by
  cases point with
  | nil => simp [tensorCheck, tensorPath, tensorRun, Runs]
  | cons head tail =>
      simp only [List.map_cons, tensorCheck, tensorPath, tensorRun, List.length_map,
        List.tail_cons]
      by_cases long : tail.length = ν
      · simp only [ite_eq_left long]
        refine runs_recv ?_
        have value : (((Ext.one - head) * p0 + head * p1).into : E) =
            (1 - head.into) * p0.into + head.into * p1.into := by
          simp only [Ext.into_add, Ext.into_mul, Ext.into_sub, Ext.into_one]
        by_cases passes : v.same ((Ext.one - head) * p0 + head * p1) = true
        · have holds : (v.into : E) = (1 - head.into) * ((p0.into, p1.into) : E × E).1 +
              head.into * ((p0.into, p1.into) : E × E).2 :=
            ((same_degree2 (E := E) _ _).mp passes).trans value
          rw [ite_eq_left passes, ite_eq_left passes, ite_eq_left holds]
          refine runs_draw ?_
          have rounds_run := plainRun_runs (degree2_extHom (E := E)) 1
            (combRound ν (tensorTables w (tail.map Ext.into) (η.into : E)) productComb) rounds []
            (tensorInput p0 p1 η) lengths
          rw [count] at rounds_run
          rw [into_tensorInput (E := E) p0 p1 η] at rounds_run
          refine runs_bind rounds_run (runs_recv ?_)
          by_cases ends : (plainRun (tensorInput p0 p1 η) rounds).claim.same final = true
          · have equal : ((plainRun (tensorInput p0 p1 η) rounds).claim.into : E) = final.into :=
              (same_degree2 (E := E) _ _).mp ends
            rw [ite_eq_left equal, ite_eq_left ends]
            rfl
          · have differ : ¬ ((plainRun (tensorInput p0 p1 η) rounds).claim.into : E) =
                final.into := fun equal => ends ((same_degree2 (E := E) _ _).mpr equal)
            rw [ite_eq_right differ, ite_eq_right ends]
            rfl
        · have fails : ¬ (v.into : E) = (1 - head.into) * ((p0.into, p1.into) : E × E).1 +
              head.into * ((p0.into, p1.into) : E × E).2 := fun equal =>
            passes ((same_degree2 (E := E) _ _).mpr (equal.trans value.symm))
          rw [ite_eq_right passes, ite_eq_right passes, ite_eq_right fails]
          rfl
      · simp only [ite_eq_right long]
        rfl

/-- One pair `(t, r)` of a coordinate of the tail and of `ρ` on the state `s` of the two
coordinates: `s_dst ← Σ_src s_src·((1 − r)·[β_src·(1 − t)]_dst + r·[β_src·t]_dst)` for the basis
`(β_0, β_1) = (1, u)`, with `u·(a + b·u) = 2·b + a·u`. -/
def factorStep (s : Ext × Ext) (pair : Ext × Ext) : Ext × Ext :=
  let low := (Ext.one - pair.2) * Ext.ofBase (Ext.one - pair.1).c0 + pair.2 * Ext.ofBase pair.1.c0
  let high := (Ext.one - pair.2) * Ext.ofBase (Ext.one - pair.1).c1 +
    pair.2 * Ext.ofBase pair.1.c1
  (s.1 * low + s.2 * (Ext.ofBase (Fq.ofNat 2) * high), s.1 * high + s.2 * low)

/-- The factor `θ = A_η(ρ)` of the tensor reduction of one claim. -/
def tensorFactor (tail : List Ext) (η : Ext) (ρ : List Ext) : Ext :=
  let s := (tail.zip ρ).foldl factorStep (Ext.one, Ext.zero)
  (Ext.one - η) * s.1 + η * s.2

section Factor

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

/-- `Σ_x ẽq(ρ, x)·((1 − η)·[z·ẽq(tail, x)]_0 + η·[z·ẽq(tail, x)]_1)`. -/
noncomputable def factorSum (tail ρ : List E) (η z : E) : E :=
  mle ρ fun x => (1 - η) * algebraMap Zq E (Degree2.re (z * eqAt tail x)) +
    η * algebraMap Zq E (Degree2.im (z * eqAt tail x))

theorem factorSum_linear (tail ρ : List E) (η w : E) :
    factorSum tail ρ η w = algebraMap Zq E (Degree2.re w) * factorSum tail ρ η 1 +
      algebraMap Zq E (Degree2.im w) * factorSum tail ρ η (Degree2.pair 0 1) := by
  unfold factorSum
  rw [← mle_mul_left, ← mle_mul_left, ← mle_add]
  congr 1
  funext x
  simp only [one_mul, Degree2.re_mul, Degree2.im_mul, Degree2.re_pair, Degree2.im_pair, map_add,
    map_mul, map_one, map_ofNat, zero_mul, zero_add]
  ring

theorem factorSum_cons (t r : E) (tail ρ : List E) (η z : E) :
    factorSum (t :: tail) (r :: ρ) η z =
      (1 - r) * factorSum tail ρ η (z * (1 - t)) + r * factorSum tail ρ η (z * t) := by
  unfold factorSum
  rw [mle_cons, ← mle_mul_left, ← mle_mul_left, ← mle_add]
  congr 1
  funext x
  have even : 2 * x % 2 = 0 := by omega
  have odd : (2 * x + 1) % 2 = 1 := by omega
  have half : 2 * x / 2 = x := by omega
  have half' : (2 * x + 1) / 2 = x := by omega
  simp only [bindLow, eqAt, even, odd, half, half', zero_ne_one, ite_false, ite_true, ← mul_assoc]

theorem factorSum_nil (η z : E) :
    factorSum [] [] η z =
      (1 - η) * algebraMap Zq E (Degree2.re z) + η * algebraMap Zq E (Degree2.im z) := by
  simp [factorSum, mle, fold, eqAt]

theorem re_sub (a b : E) : Degree2.re (a - b) = Degree2.re a - Degree2.re b := by
  simp [Degree2.re]

theorem im_sub (a b : E) : Degree2.im (a - b) = Degree2.im a - Degree2.im b := by
  simp [Degree2.im]

theorem factor_fold (tail ρ : List Ext) (length : ρ.length = tail.length) (η : Ext)
    (s : Ext × Ext) :
    (1 - η.into) * (((tail.zip ρ).foldl factorStep s).1.into : E) +
        η.into * ((tail.zip ρ).foldl factorStep s).2.into =
      s.1.into * factorSum (tail.map Ext.into) (ρ.map Ext.into) η.into 1 +
        s.2.into * factorSum (tail.map Ext.into) (ρ.map Ext.into) η.into (Degree2.pair 0 1) := by
  have one_re : Degree2.re (1 : E) = 1 := by simpa using Degree2.re_algebraMap (E := E) 1
  have one_im : Degree2.im (1 : E) = 0 := by simpa using Degree2.im_algebraMap (E := E) 1
  induction tail generalizing ρ s with
  | nil =>
      rw [List.length_nil, List.length_eq_zero_iff] at length
      subst length
      simp only [List.zip_nil_left, List.foldl_nil, List.map_nil, factorSum_nil, one_re, one_im,
        Degree2.re_pair, Degree2.im_pair, map_one, map_zero]
      ring
  | cons t tail ih =>
      obtain ⟨r, ρ, rfl⟩ := List.exists_cons_of_length_eq_add_one length
      rw [List.zip_cons_cons, List.foldl_cons, ih ρ (by simpa using length), List.map_cons,
        List.map_cons, factorSum_cons, factorSum_cons, factorSum_linear _ _ _ (1 * (1 - _)),
        factorSum_linear _ _ _ (1 * _), factorSum_linear _ _ _ (Degree2.pair 0 1 * (1 - _)),
        factorSum_linear _ _ _ (Degree2.pair 0 1 * _)]
      have c0 : (Ext.one - t).c0.toZq = 1 - t.c0.toZq := by
        rw [show (Ext.one - t).c0 = Fq.one.sub t.c0 from rfl, Fq.toZq_sub, Fq.toZq_one]
      have c1 : (Ext.one - t).c1.toZq = -t.c1.toZq := by
        rw [show (Ext.one - t).c1 = Fq.zero.sub t.c1 from rfl, Fq.toZq_sub, Fq.toZq_zero, zero_sub]
      simp only [factorStep, Ext.into_add, Ext.into_mul, Ext.into_sub, Ext.into_one,
        Ext.into_ofBase, c0, c1, Fq.toZq_ofNat, Nat.cast_ofNat, one_mul, Degree2.re_mul,
        Degree2.im_mul, re_sub, im_sub, one_re, one_im, re_into, im_into, Degree2.re_pair,
        Degree2.im_pair, zero_mul, zero_add, zero_sub, map_sub, map_neg, map_mul, map_one,
        map_ofNat]
      ring

/-- The factor of the code is the factor `θ = mle ρ (tensorWeight tail η)` of `tensorCheck`. -/
theorem tensorFactor_into (tail ρ : List Ext) (length : ρ.length = tail.length) (η : Ext) :
    ((tensorFactor tail η ρ).into : E) =
      mle (ρ.map Ext.into) (tensorWeight (tail.map Ext.into) η.into) := by
  have one_re : Degree2.re (1 : E) = 1 := by simpa using Degree2.re_algebraMap (E := E) 1
  have one_im : Degree2.im (1 : E) = 0 := by simpa using Degree2.im_algebraMap (E := E) 1
  have start := factor_fold (E := E) tail ρ length η (Ext.one, Ext.zero)
  simp only [Ext.into_one, Ext.into_zero, one_mul, zero_mul, add_zero] at start
  rw [tensorFactor, Ext.into_add, Ext.into_mul, Ext.into_mul, Ext.into_sub, Ext.into_one, start,
    factorSum]
  exact congrArg _ (funext fun x => by simp [tensorWeight])

end Factor

end Spec

end NightstreamFPrime.Spec.Compression.Akita.Replay
