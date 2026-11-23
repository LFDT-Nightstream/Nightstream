import NightstreamFPrime.Spec.Compression.Akita.Replay.Range
import NightstreamFPrime.Spec.Compression.Akita.Replay.Relation
import NightstreamFPrime.Spec.Compression.Akita.Tensor

/-! The tensor reduction of one claim of the pinned Akita verifier on words (Akita revision
`d50f700`, `akita-verifier/src/stages/opening_claims/extension_claim.rs`; ePrint 2026/1983,
Section 3.7), and its relation to `tensorCheck` of the specification.

- `tensorRun` checks that the opening is the opening of the column partials,
  `v = (1 − r_0)·S_0 + r_0·S_1` (`derive_tensor_extension_opening_claim_from_partials`), and runs
  the plain sum-check for `(1 − η)·row_0 + η·row_1` of the row partials
  (`tensor_row_partials_from_columns`, `tensor_reduction_claim_from_rows`). `none` is a rejection.
- `tensorRun_runs`: `tensorCheck` on the path of the messages and the challenges of the code ends
  in the point and the end claim of the code, with the factor `θ = A_η(ρ)` of the specification.
  The code receives then the final claim `f̄(ρ)·θ` as a message and compares it with the end claim
  (`tensor.end`); the specification has no message there. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

/-- `(1 − η)·row_0 + η·row_1` for the rows `(S_0.c0, S_1.c0)` and `(S_0.c1, S_1.c1)`. -/
def tensorInput (p0 p1 η : Ext) : Ext :=
  (Ext.one - η) * ⟨p0.c0, p1.c0⟩ + η * ⟨p0.c1, p1.c1⟩

def tensorRun (ν : ℕ) (point : List Ext) (v p0 p1 η : Ext) (rounds : List (List Ext × Ext)) :
    Option Ended :=
  match point with
  | [] => none
  | head :: tail =>
      if tail.length = ν then
        if v.same ((Ext.one - head) * p0 + head * p1) then
          some (plainRun (tensorInput p0 p1 η) rounds)
        else none
      else none

section Spec

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- The path of `tensorCheck`: the column partials and, when the opening check passes, `η` and
the rounds. -/
noncomputable def tensorPath (E : Type) [Field E] [Algebra Zq E] [Degree2 E] (ν : ℕ)
    (point : List Ext) (v p0 p1 η : Ext) (rounds : List (List Ext × Ext)) : List Move :=
  match point with
  | [] => []
  | head :: tail =>
      if tail.length = ν then
        Move.recv ((p0.into, p1.into) : E × E) ::
          if v.same ((Ext.one - head) * p0 + head * p1) then
            Move.draw (η.into : E) :: plainPath (Ext.into (E := E)) 1 (tensorInput p0 p1 η) rounds
          else []
      else []

omit [Fintype E] [DecidableEq E] in
theorem into_pair (a b : Fq) : ((⟨a, b⟩ : Ext).into : E) = Degree2.pair a.toZq b.toZq := rfl

omit [Fintype E] [DecidableEq E] in
theorem re_into (z : Ext) : Degree2.re (z.into : E) = z.c0.toZq := by
  simp [Ext.into, Degree2.re, Ext.toE]

omit [Fintype E] [DecidableEq E] in
theorem im_into (z : Ext) : Degree2.im (z.into : E) = z.c1.toZq := by
  simp [Ext.into, Degree2.im, Ext.toE]

omit [Fintype E] [DecidableEq E] in
theorem into_tensorInput (p0 p1 η : Ext) :
    ((tensorInput p0 p1 η).into : E) =
      (1 - η.into) * tensorRow ((p0.into, p1.into) : E × E) false +
        η.into * tensorRow ((p0.into, p1.into) : E × E) true := by
  simp only [tensorInput, Ext.into_add, Ext.into_mul, Ext.into_sub, Ext.into_one, tensorRow,
    re_into, im_into, into_pair, Bool.false_eq_true, ite_false, ite_true]

/-- `tensorCheck` of the specification, on the path of the messages and the challenges of the
code, ends in the result of `tensorRun`: `none` when the opening check fails, and otherwise the
point `ρ`, the factor `θ = A_η(ρ)` and the end claim. -/
theorem tensorRun_runs (ν : ℕ) (w : Table) (point : List Ext) (v p0 p1 η : Ext)
    (rounds : List (List Ext × Ext)) (count : rounds.length = ν)
    (lengths : ∀ x ∈ rounds, x.1.length = 2) :
    (tensorCheck ν w (point.map (Ext.into (E := E))) v.into).Runs
      (tensorPath E ν point v p0 p1 η rounds)
      ((tensorRun ν point v p0 p1 η rounds).map fun out =>
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
          have input := into_tensorInput (E := E) p0 p1 η
          unfold Ext.into at input
          rw [input] at rounds_run
          exact runs_bind_end rounds_run rfl
        · have fails : ¬ (v.into : E) = (1 - head.into) * ((p0.into, p1.into) : E × E).1 +
              head.into * ((p0.into, p1.into) : E × E).2 := fun equal =>
            passes ((same_degree2 (E := E) _ _).mpr (equal.trans value.symm))
          rw [ite_eq_right passes, ite_eq_right passes, ite_eq_right fails]
          rfl
      · simp only [ite_eq_right long]
        rfl

end Spec

end NightstreamFPrime.Spec.Compression.Akita.Replay
