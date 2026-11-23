import NightstreamFPrime.Spec.Compression.Akita.Replay.Field
import NightstreamFPrime.Spec.Compression.Akita.EqSumCheck
import NightstreamFPrime.Spec.Compression.Game.Path

/-! The rounds of a sum-check of the pinned Akita verifier on words (Akita revision `d50f700`,
`akita-sumcheck/src/proof_stream.rs`: `verify_sumcheck_rounds`,
`verify_eq_factored_sumcheck_rounds`; `single.rs`; ePrint 2026/1983, revision of 2026-11-23,
Section 3.4, Figure 2, and Section 3.5), and its relation to the sum-check games of the
specification (`docs/akita-work/match-design.md` section 5.2, T1).

- `sumcheckRun` is the claim chain of the code. In the plain form (tensor, norm, fused, Stage 3)
  a message is the round polynomial `p` without its linear coefficient, which `plainStep` solves
  from `p(0) + p(1) = claim` (`CompressedPoly::eval_from_hint`). In the equality-factored form
  (the stages of the range tree) a message is the inner polynomial `q` without its constant
  coefficient, which `eqStep` solves from the normalized claim `c = (1 − τ)·q(0) + τ·q(1)`
  (`advance_eq_factored_claim`); `factor` is `Π_j eq(τ_j, r_j)`. No round has a check. `none` is
  the shape check of the code (`validate_instance`, an empty equality point), which is not a
  rejection of a proof. The values are those of the code; the order of the field operations is
  not.
- `ExtHom φ` states that `φ` maps the operations of `Ext` to those of a ring: `Ext.toE` to
  `Quadratic` (`toE_extHom`), and `Ext.into` to a ring with `Degree2` (`degree2_extHom` of
  `Replay/Relation.lean`).
- `plainPath` and `eqCodePath` are the paths of the games of the specification for the messages
  of the code. `plainPoly` is the round polynomial with the solved linear coefficient, and
  `listFn` gives the message of the code to `eqSumcheck` with no change.
  `sumcheckRun_plain` and `sumcheckRun_plain_with`: `Game.sumcheck` and `sumcheckWith` end in the
  point and the end claim of the code. `sumcheckRun_eqSumcheck`: `eqSumcheck`, the stage of the
  range tree, ends in the point and the end claim of the code, with no condition.
- `eqPath` is the path of the full-value games `Game.sumcheck` and `sumcheckWith` for the
  equality-factored messages, with the round polynomial `fullPoly` (`E·eq(τ, X)·q(X)`).
  `sumcheckRun_equality` and `sumcheckRun_equality_with`: they end in the point and `E_n·c_n`
  with `E_n = ẽq(τ, r)`, and `equality_end` states that for `E_n ≠ 0` the full-value end check
  `E_n·c_n = ẽq(τ, r)·S` holds exactly when the normalized end check `c_n = S` holds.
- `plainCode` is the message of the code for a message of the specification.
  `plainCode_coefficients` and `coefficients_plainCode`: the messages of the code are in
  bijection with the messages of the specification that pass the round check, and
  `coefficients_plainPoly_check` states that each image passes it. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial
open NightstreamFPrime.Spec.Compression (Game)

/-- `Σ_i m_i`. -/
def total : List Ext → Ext
  | [] => Ext.zero
  | c :: m => c + total m

/-- `Σ_i m_i·r^i`. -/
def horner (r : Ext) : List Ext → Ext
  | [] => Ext.zero
  | c :: m => c + r * horner r m

/-- One plain round: the message is `(p_0, p_2, …, p_d)`, `p_1 = claim − 2·p_0 − Σ_{i ≥ 2} p_i`,
and the next claim is `p(r)`. -/
def plainStep (claim : Ext) (message : List Ext) (r : Ext) : Ext :=
  message.headD Ext.zero +
    r * (claim - message.headD Ext.zero - message.headD Ext.zero - total message.tail +
      r * horner r message.tail)

/-- One equality-factored round: the message is `(q_1, …, q_D)`, `q_0 = claim − τ·Σ_i q_i`, and
the next normalized claim is `q(r)`. -/
def eqStep (claim τ : Ext) (message : List Ext) (r : Ext) : Ext :=
  claim - τ * total message + r * horner r message

/-- `eq(τ, r) = τ·r + (1 − τ)·(1 − r)`. -/
def eqVal (τ r : Ext) : Ext := τ * r + (Ext.one - τ) * (Ext.one - r)

/-- The message form of a sum-check: plain, or equality-factored with the point before. -/
inductive Form
  | plain
  | equality (before : List Ext)

/-- The end of the rounds: the point, the end claim, and the accumulated equality factor (1 in
the plain form). -/
structure Ended where
  point : List Ext
  claim : Ext
  factor : Ext
deriving DecidableEq, Repr

def plainRun : Ext → List (List Ext × Ext) → Ended
  | claim, [] => ⟨[], claim, Ext.one⟩
  | claim, (message, r) :: rounds =>
      let ended := plainRun (plainStep claim message r) rounds
      { ended with point := r :: ended.point }

def eqRun : Ext → Ext → List Ext → List (List Ext × Ext) → Option Ended
  | claim, factor, [], [] => some ⟨[], claim, factor⟩
  | claim, factor, τ :: before, (message, r) :: rounds =>
      (eqRun (eqStep claim τ message r) (factor * eqVal τ r) before rounds).map fun ended =>
        { ended with point := r :: ended.point }
  | _, _, _, _ => none

/-- The rounds of a sum-check of the code from its input claim, with each round's message words
and challenge. -/
def sumcheckRun : Form → Ext → List (List Ext × Ext) → Option Ended
  | .plain, claim, rounds => some (plainRun claim rounds)
  | .equality [], _, _ => none
  | .equality (τ :: before), claim, rounds => eqRun claim Ext.one (τ :: before) rounds

theorem eqRun_none : ∀ (before : List Ext) (rounds : List (List Ext × Ext)) (claim factor : Ext),
    eqRun claim factor before rounds = none ↔ before.length ≠ rounds.length
  | [], [], _, _ => by simp [eqRun]
  | [], _ :: _, _, _ => by simp [eqRun]
  | _ :: _, [], _, _ => by simp [eqRun]
  | τ :: before, (message, r) :: rounds, claim, factor => by
      simp only [eqRun, Option.map_eq_none_iff, List.length_cons, ne_eq, Nat.add_right_cancel_iff]
      exact eqRun_none before rounds _ _

/-- The shape check of the code is the only `none` of `sumcheckRun`. -/
theorem sumcheckRun_none (form : Form) (claim : Ext) (rounds : List (List Ext × Ext)) :
    sumcheckRun form claim rounds = none ↔
      ∃ before, form = .equality before ∧ (before = [] ∨ before.length ≠ rounds.length) := by
  cases form with
  | plain => simp [sumcheckRun]
  | equality before =>
      cases before with
      | nil => simp [sumcheckRun]
      | cons τ before => simp [sumcheckRun, eqRun_none]

theorem sumcheckRun_equality_eqRun {before : List Ext} {claim : Ext}
    {rounds : List (List Ext × Ext)} {out : Ended}
    (ran : sumcheckRun (.equality before) claim rounds = some out) :
    eqRun claim Ext.one before rounds = some out := by
  cases before with
  | nil => simp [sumcheckRun] at ran
  | cons τ before => exact ran

/-- The point of the plain rounds is the list of their challenges. -/
theorem plainRun_point : ∀ (claim : Ext) (rounds : List (List Ext × Ext)),
    (plainRun claim rounds).point = rounds.map (·.2)
  | _, [] => rfl
  | claim, (message, r) :: rounds => by
      simp only [plainRun, List.map_cons]
      rw [plainRun_point]

/-- The point of the equality-factored rounds is the list of their challenges. -/
theorem eqRun_point : ∀ (claim factor : Ext) (before : List Ext) (rounds : List (List Ext × Ext))
    {e : Ended}, eqRun claim factor before rounds = some e → e.point = rounds.map (·.2)
  | _, _, [], [], e, ran => by
      simp only [eqRun, Option.some.injEq] at ran
      subst ran
      rfl
  | claim, factor, τ :: before, (message, r) :: rounds, e, ran => by
      simp only [eqRun, Option.map_eq_some_iff] at ran
      obtain ⟨e', ran', rfl⟩ := ran
      simp only [List.map_cons]
      rw [eqRun_point _ _ _ _ ran']
  | _, _, [], _ :: _, _, ran => by simp [eqRun] at ran
  | _, _, _ :: _, [], _, ran => by simp [eqRun] at ran

/-- The point of the end of a sum-check is the list of the challenges of its rounds. -/
theorem sumcheckRun_point {form : Form} {claim : Ext} {rounds : List (List Ext × Ext)}
    {e : Ended} (ran : sumcheckRun form claim rounds = some e) : e.point = rounds.map (·.2) := by
  rcases form with _ | ⟨_ | ⟨τ, before⟩⟩
  · simp only [sumcheckRun, Option.some.injEq] at ran
    rw [← ran, plainRun_point]
  · simp [sumcheckRun] at ran
  · exact eqRun_point _ _ _ _ ran

section Spec

variable {K : Type}

/-- `φ` maps the operations of `Ext` to those of the ring `K`. -/
structure ExtHom [CommRing K] (φ : Ext → K) : Prop where
  zero : φ Ext.zero = 0
  one : φ Ext.one = 1
  add : ∀ a b, φ (a + b) = φ a + φ b
  sub : ∀ a b, φ (a - b) = φ a - φ b
  mul : ∀ a b, φ (a * b) = φ a * φ b

theorem toE_extHom : ExtHom Ext.toE :=
  ⟨Ext.toE_zero, Ext.toE_one, Ext.toE_add, Ext.toE_sub, Ext.toE_mul⟩

variable [Field K]

/-- `Σ_i C(m_i)·X^i`. -/
noncomputable def listPoly : List K → K[X]
  | [] => 0
  | c :: m => C c + X * listPoly m

/-- The message of `Game.sumcheck` for a polynomial: its first `n + 1` coefficients. -/
noncomputable def coefficients (n : ℕ) (p : K[X]) : Fin (n + 1) → K := fun i => p.coeff i

/-- The round polynomial of a plain message `(p_0, p_2, …, p_d)` for the claim `claim`. -/
noncomputable def plainPoly (claim : K) (message : List K) : K[X] :=
  listPoly (message.headD 0 ::
    (claim - message.headD 0 - message.headD 0 - message.tail.sum) :: message.tail)

theorem listPoly_eval_zero : ∀ m : List K, (listPoly m).eval 0 = m.headD 0
  | [] => by simp [listPoly]
  | c :: m => by simp [listPoly]

theorem listPoly_eval_one : ∀ m : List K, (listPoly m).eval 1 = m.sum
  | [] => by simp [listPoly]
  | c :: m => by simp [listPoly, listPoly_eval_one m]

theorem listPoly_coeff : ∀ (m : List K) (i : ℕ), (listPoly m).coeff i = m.getD i 0
  | [], i => by simp [listPoly]
  | c :: m, 0 => by simp [listPoly]
  | c :: m, i + 1 => by
      simp [listPoly, coeff_X_mul, listPoly_coeff m i]

theorem listPoly_natDegree : ∀ (m : List K) (n : ℕ), m.length ≤ n + 1 →
    (listPoly m).natDegree ≤ n
  | [], n, _ => by simp [listPoly]
  | [c], n, _ => by simp [listPoly]
  | c :: c' :: m, n, short => by
      obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by simp at short; omega⟩
      refine (natDegree_add_le _ _).trans (max_le (by simp) ?_)
      refine natDegree_mul_le.trans ?_
      have := listPoly_natDegree (c' :: m) k (by simp at short ⊢; omega)
      have one : (X : K[X]).natDegree ≤ 1 := natDegree_X_le
      omega

theorem coeffPoly_coefficients {n : ℕ} {p : K[X]} (degree : p.natDegree ≤ n) :
    coeffPoly (coefficients n p) = p :=
  coeffPoly_coeff p degree

theorem plainPoly_check (claim : K) (message : List K) :
    (plainPoly claim message).eval 0 + (plainPoly claim message).eval 1 = claim := by
  rw [plainPoly, listPoly_eval_zero, listPoly_eval_one]
  simp only [List.headD_cons, List.sum_cons]
  ring

theorem plainPoly_natDegree {d : ℕ} (claim : K) {message : List K}
    (length : message.length = d + 1) : (plainPoly claim message).natDegree ≤ d + 1 :=
  listPoly_natDegree _ _ (by cases message <;> simp_all)

/-- A list as a message of `D` entries, with zeros after its end. -/
def listFn (D : ℕ) (l : List K) : Fin D → K := fun i => l.getD i 0

theorem coeffPoly_listFn {D : ℕ} {l : List K} (length : l.length = D) :
    coeffPoly (listFn D l) = listPoly l := by
  subst length
  induction l with
  | nil => simp [coeffPoly, listPoly]
  | cons c l step =>
      rw [listPoly, ← step]
      simp only [coeffPoly, listFn, List.length_cons, Fin.sum_univ_succ, Fin.val_zero,
        List.getD_cons_zero, pow_zero, mul_one, Fin.val_succ, List.getD_cons_succ, Finset.mul_sum,
        pow_succ]
      congr 1
      exact Finset.sum_congr rfl fun i _ => by ring

theorem sum_listFn {D : ℕ} {l : List K} (length : l.length = D) :
    ∑ i, listFn D l i = l.sum := by
  subst length
  induction l with
  | nil => simp [listFn]
  | cons c l step =>
      rw [List.sum_cons, ← step]
      simp only [listFn, List.length_cons, Fin.sum_univ_succ, Fin.val_zero, List.getD_cons_zero,
        Fin.val_succ, List.getD_cons_succ]

section Transfer

variable {φ : Ext → K} (hom : ExtHom φ)
include hom

theorem map_total : ∀ m : List Ext, φ (total m) = (m.map φ).sum
  | [] => hom.zero
  | c :: m => by rw [total, hom.add, map_total m, List.map_cons, List.sum_cons]

theorem map_horner (r : Ext) : ∀ m : List Ext,
    φ (horner r m) = (listPoly (m.map φ)).eval (φ r)
  | [] => by simp [horner, listPoly, hom.zero]
  | c :: m => by
      rw [horner, hom.add, hom.mul, map_horner r m]
      simp [listPoly]

theorem map_headD (m : List Ext) : (m.map φ).headD 0 = φ (m.headD Ext.zero) := by
  cases m <;> simp [hom.zero]

theorem map_plainStep (claim : Ext) (message : List Ext) (r : Ext) :
    φ (plainStep claim message r) =
      (plainPoly (φ claim) (message.map φ)).eval (φ r) := by
  rw [plainStep, plainPoly, hom.add, hom.mul, hom.add, hom.sub, hom.sub, hom.sub, hom.mul,
    map_total hom, map_horner hom, ← map_headD hom, List.map_tail]
  simp [listPoly]

theorem map_eqStep {D : ℕ} (claim τ : Ext) (message : List Ext) (r : Ext)
    (length : message.length = D) :
    φ (eqStep claim τ message r) =
      (innerPoly (φ claim) (φ τ) (listFn D (message.map φ))).eval (φ r) := by
  have mapped : (message.map φ).length = D := by simpa using length
  rw [eqStep, innerPoly_eval, coeffPoly_listFn mapped, sum_listFn mapped, hom.add, hom.sub,
    hom.mul, hom.mul, map_total hom, map_horner hom]

theorem map_eqVal (τ r : Ext) : φ (eqVal τ r) = (eqLine (φ τ)).eval (φ r) := by
  rw [eqVal, eqLine_eval, hom.add, hom.mul, hom.mul, hom.sub, hom.sub, hom.one]
  ring

theorem eqRun_factor : ∀ (before : List Ext) (rounds : List (List Ext × Ext))
    (claim factor : Ext) (out : Ended), eqRun claim factor before rounds = some out →
      out.point.length = before.length ∧
        φ out.factor = φ factor * eqEval (before.map φ) (out.point.map φ)
  | [], [], claim, factor, out, ran => by
      simp only [eqRun, Option.some.injEq] at ran
      subst ran
      simp [eqEval]
  | [], _ :: _, _, _, _, ran => by simp [eqRun] at ran
  | _ :: _, [], _, _, _, ran => by simp [eqRun] at ran
  | τ :: before, (message, r) :: rounds, claim, factor, out, ran => by
      simp only [eqRun, Option.map_eq_some_iff] at ran
      obtain ⟨ended, ran, rfl⟩ := ran
      obtain ⟨length, value⟩ := eqRun_factor before rounds _ _ ended ran
      refine ⟨by simp [length], ?_⟩
      rw [value, hom.mul, map_eqVal hom, eqLine_eval]
      simp only [List.map_cons, eqEval]
      ring

theorem sumcheckRun_equality_factor {before : List Ext} {claim : Ext}
    {rounds : List (List Ext × Ext)} {out : Ended}
    (ran : sumcheckRun (.equality before) claim rounds = some out) :
    out.point.length = before.length ∧
      φ out.factor = mle (out.point.map φ) (eqAt (before.map φ)) := by
  obtain ⟨length, value⟩ := eqRun_factor hom before rounds claim Ext.one out
    (sumcheckRun_equality_eqRun ran)
  refine ⟨length, ?_⟩
  rw [value, hom.one, one_mul, mle_eqAt _ _ (by simpa using length)]

/-- T1 (a): with a nonzero equality factor `E_n = ẽq(τ, r)`, the end check of the specification
`E_n·c_n = ẽq(τ, r)·S` holds exactly when the end check `c_n = S` of the code holds. -/
theorem equality_end {before : List Ext} {claim : Ext} {rounds : List (List Ext × Ext)}
    {out : Ended} (ran : sumcheckRun (.equality before) claim rounds = some out)
    (nonzero : φ out.factor ≠ 0) (S : K) :
    mle (out.point.map φ) (eqAt (before.map φ)) * φ out.claim =
        mle (out.point.map φ) (eqAt (before.map φ)) * S ↔ φ out.claim = S := by
  rw [← (sumcheckRun_equality_factor hom ran).2]
  exact mul_right_inj' nonzero

end Transfer

/-- The path of the plain games for the messages and the challenges of the code. -/
noncomputable def plainPath (φ : Ext → K) (d : ℕ) : Ext → List (List Ext × Ext) → List Move
  | _, [] => []
  | claim, (message, r) :: rounds =>
      Move.recv (coefficients (d + 1) (plainPoly (φ claim) (message.map φ))) ::
        Move.draw (φ r) :: plainPath φ d (plainStep claim message r) rounds

/-- The path of the full-value games for the equality-factored messages and the challenges of the
code. -/
noncomputable def eqPath (φ : Ext → K) (D : ℕ) :
    Ext → Ext → List Ext → List (List Ext × Ext) → List Move
  | factor, claim, τ :: before, (message, r) :: rounds =>
      Move.recv (coefficients (D + 1)
        (fullPoly (φ factor) (φ claim) (φ τ) (listFn D (message.map φ)))) ::
        Move.draw (φ r) :: eqPath φ D (factor * eqVal τ r) (eqStep claim τ message r) before rounds
  | _, _, _, _ => []

/-- The path of `eqSumcheck` for the equality-factored messages and the challenges of the code:
the messages of the code are the messages of the specification. -/
noncomputable def eqCodePath (φ : Ext → K) (D : ℕ) : List (List Ext × Ext) → List Move
  | [] => []
  | (message, r) :: rounds =>
      Move.recv (listFn D (message.map φ)) :: Move.draw (φ r) :: eqCodePath φ D rounds

theorem runs_recv {α M : Type} [Nonempty M] {next : M → Game α} {message : M}
    {path : List Move} {result : α} (runs : (next message).Runs path result) :
    (Game.recv M next).Runs (Move.recv message :: path) result :=
  Exists.intro rfl runs

theorem runs_draw {α C : Type} [Fintype C] [Nonempty C] {bad : Finset C} {next : C → Game α}
    {challenge : C} {path : List Move} {result : α} (runs : (next challenge).Runs path result) :
    (Game.draw C bad next).Runs (Move.draw challenge :: path) result :=
  Exists.intro rfl runs

variable [Fintype K] [DecidableEq K] {φ : Ext → K} (hom : ExtHom φ)
include hom

theorem plainRun_runs (d : ℕ) (round : List K → K[X]) :
    ∀ (rounds : List (List Ext × Ext)) (point : List K) (claim : Ext),
      (∀ x ∈ rounds, x.1.length = d + 1) →
      (Game.sumcheck (d + 1) round rounds.length point (φ claim)).Runs (plainPath φ d claim rounds)
        (some (point ++ (plainRun claim rounds).point.map φ, φ (plainRun claim rounds).claim))
  | [], point, claim, _ => by simp [Game.sumcheck, plainPath, plainRun, Runs]
  | (message, r) :: rounds, point, claim, lengths => by
      have degree : (plainPoly (φ claim) (message.map φ)).natDegree ≤ d + 1 :=
        plainPoly_natDegree _ (by simpa using lengths _ List.mem_cons_self)
      refine runs_recv ?_
      simp only [coeffPoly_coefficients degree, plainPoly_check, ite_true]
      refine runs_draw ?_
      rw [← map_plainStep hom]
      have next := plainRun_runs d round rounds (point ++ [φ r]) (plainStep claim message r)
        fun x member => lengths x (List.mem_cons_of_mem _ member)
      simpa [plainRun] using next

theorem plainRun_runs_with (d : ℕ) (round : List K → K[X]) (extra : List K → Finset K) :
    ∀ (rounds : List (List Ext × Ext)) (point : List K) (claim : Ext),
      (∀ x ∈ rounds, x.1.length = d + 1) →
      (sumcheckWith (d + 1) round extra rounds.length point (φ claim)).Runs
        (plainPath φ d claim rounds)
        (some (point ++ (plainRun claim rounds).point.map φ, φ (plainRun claim rounds).claim))
  | [], point, claim, _ => by simp [sumcheckWith, plainPath, plainRun, Runs]
  | (message, r) :: rounds, point, claim, lengths => by
      have degree : (plainPoly (φ claim) (message.map φ)).natDegree ≤ d + 1 :=
        plainPoly_natDegree _ (by simpa using lengths _ List.mem_cons_self)
      refine runs_recv ?_
      simp only [coeffPoly_coefficients degree, plainPoly_check, ite_true]
      refine runs_draw ?_
      rw [← map_plainStep hom]
      have next := plainRun_runs_with d round extra rounds (point ++ [φ r])
        (plainStep claim message r) fun x member => lengths x (List.mem_cons_of_mem _ member)
      simpa [plainRun] using next

theorem eqRun_runs (D : ℕ) (round : List K → K[X]) :
    ∀ (before : List Ext) (rounds : List (List Ext × Ext)) (point : List K) (claim factor : Ext)
      (out : Ended), (∀ x ∈ rounds, x.1.length = D) → eqRun claim factor before rounds = some out →
      (Game.sumcheck (D + 1) round rounds.length point (φ factor * φ claim)).Runs
        (eqPath φ D factor claim before rounds)
        (some (point ++ out.point.map φ, φ out.factor * φ out.claim))
  | [], [], point, claim, factor, out, _, ran => by
      simp only [eqRun, Option.some.injEq] at ran
      subst ran
      simp [Game.sumcheck, eqPath, Runs]
  | [], _ :: _, _, _, _, _, _, ran => by simp [eqRun] at ran
  | _ :: _, [], _, _, _, _, _, ran => by simp [eqRun] at ran
  | τ :: before, (message, r) :: rounds, point, claim, factor, out, lengths, ran => by
      simp only [eqRun, Option.map_eq_some_iff] at ran
      obtain ⟨ended, ran, rfl⟩ := ran
      have degree := fullPoly_natDegree (φ factor) (φ claim) (φ τ) (listFn D (message.map φ))
      refine runs_recv ?_
      simp only [coeffPoly_coefficients degree, fullPoly_check, ite_true]
      refine runs_draw ?_
      have value : (fullPoly (φ factor) (φ claim) (φ τ) (listFn D (message.map φ))).eval (φ r) =
          φ (factor * eqVal τ r) * φ (eqStep claim τ message r) := by
        rw [hom.mul, map_eqVal hom, map_eqStep hom claim τ message r (lengths _ List.mem_cons_self),
          fullPoly_eval]
      rw [value]
      have next := eqRun_runs D round before rounds (point ++ [φ r]) (eqStep claim τ message r)
        (factor * eqVal τ r) ended (fun x member => lengths x (List.mem_cons_of_mem _ member)) ran
      simpa using next

theorem eqRun_runs_with (D : ℕ) (round : List K → K[X]) (extra : List K → Finset K) :
    ∀ (before : List Ext) (rounds : List (List Ext × Ext)) (point : List K) (claim factor : Ext)
      (out : Ended), (∀ x ∈ rounds, x.1.length = D) → eqRun claim factor before rounds = some out →
      (sumcheckWith (D + 1) round extra rounds.length point (φ factor * φ claim)).Runs
        (eqPath φ D factor claim before rounds)
        (some (point ++ out.point.map φ, φ out.factor * φ out.claim))
  | [], [], point, claim, factor, out, _, ran => by
      simp only [eqRun, Option.some.injEq] at ran
      subst ran
      simp [sumcheckWith, eqPath, Runs]
  | [], _ :: _, _, _, _, _, _, ran => by simp [eqRun] at ran
  | _ :: _, [], _, _, _, _, _, ran => by simp [eqRun] at ran
  | τ :: before, (message, r) :: rounds, point, claim, factor, out, lengths, ran => by
      simp only [eqRun, Option.map_eq_some_iff] at ran
      obtain ⟨ended, ran, rfl⟩ := ran
      have degree := fullPoly_natDegree (φ factor) (φ claim) (φ τ) (listFn D (message.map φ))
      refine runs_recv ?_
      simp only [coeffPoly_coefficients degree, fullPoly_check, ite_true]
      refine runs_draw ?_
      have value : (fullPoly (φ factor) (φ claim) (φ τ) (listFn D (message.map φ))).eval (φ r) =
          φ (factor * eqVal τ r) * φ (eqStep claim τ message r) := by
        rw [hom.mul, map_eqVal hom, map_eqStep hom claim τ message r (lengths _ List.mem_cons_self),
          fullPoly_eval]
      rw [value]
      have next := eqRun_runs_with D round extra before rounds (point ++ [φ r])
        (eqStep claim τ message r) (factor * eqVal τ r) ended
        (fun x member => lengths x (List.mem_cons_of_mem _ member)) ran
      simpa using next

theorem eqRun_eqSumcheck (D : ℕ) (round : List K → K[X]) (extra : List K → Finset K)
    (τ : List K) :
    ∀ (before : List Ext) (rounds : List (List Ext × Ext)) (point : List K) (claim factor : Ext)
      (out : Ended), (∀ x ∈ rounds, x.1.length = D) → eqRun claim factor before rounds = some out →
      τ.drop point.length = before.map φ →
      (eqSumcheck D round extra τ rounds.length point (φ claim)).Runs (eqCodePath φ D rounds)
        (point ++ out.point.map φ, φ out.claim)
  | [], [], point, claim, factor, out, _, ran, _ => by
      simp only [eqRun, Option.some.injEq] at ran
      subst ran
      simp [eqSumcheck, eqCodePath, Runs]
  | [], _ :: _, _, _, _, _, _, ran, _ => by simp [eqRun] at ran
  | _ :: _, [], _, _, _, _, _, ran, _ => by simp [eqRun] at ran
  | t :: before, (message, r) :: rounds, point, claim, factor, out, lengths, ran, dropped => by
      simp only [eqRun, Option.map_eq_some_iff] at ran
      obtain ⟨ended, ran, rfl⟩ := ran
      have coordinate : τ.getD point.length 0 = φ t := by
        rw [List.getD_eq_getElem?_getD, ← List.head?_drop, dropped]
        rfl
      refine runs_recv (runs_draw ?_)
      simp only [coordinate]
      rw [← map_eqStep hom claim t message r (lengths _ List.mem_cons_self)]
      have next := eqRun_eqSumcheck D round extra τ before rounds (point ++ [φ r])
        (eqStep claim t message r) (factor * eqVal t r) ended
        (fun x member => lengths x (List.mem_cons_of_mem _ member)) ran
        (by rw [List.length_append, List.length_singleton, ← List.tail_drop, dropped]; rfl)
      simpa using next

/-- The equality-factored form: `eqSumcheck` with the point before `τ`, the stage of the range
tree of the specification, on the path of the code ends in the point and the end claim of the
code. The end check of the specification and of the code is then the same check. -/
theorem sumcheckRun_eqSumcheck (D : ℕ) (round : List K → K[X]) (extra : List K → Finset K)
    {before : List Ext} {claim : Ext} {rounds : List (List Ext × Ext)} {out : Ended}
    (ran : sumcheckRun (.equality before) claim rounds = some out)
    (lengths : ∀ x ∈ rounds, x.1.length = D) :
    (eqSumcheck D round extra (before.map φ) rounds.length [] (φ claim)).Runs
      (eqCodePath φ D rounds) (out.point.map φ, φ out.claim) := by
  simpa using eqRun_eqSumcheck hom D round extra (before.map φ) before rounds [] claim Ext.one out
    lengths (sumcheckRun_equality_eqRun ran) rfl

/-- The plain form: `Game.sumcheck` on the path of the code ends in the point and the end claim
of the code. -/
theorem sumcheckRun_plain (d : ℕ) (round : List K → K[X]) {claim : Ext}
    {rounds : List (List Ext × Ext)} {out : Ended}
    (ran : sumcheckRun .plain claim rounds = some out)
    (lengths : ∀ x ∈ rounds, x.1.length = d + 1) :
    (Game.sumcheck (d + 1) round rounds.length [] (φ claim)).Runs (plainPath φ d claim rounds)
      (some (out.point.map φ, φ out.claim)) := by
  simp only [sumcheckRun, Option.some.injEq] at ran
  subst ran
  simpa using plainRun_runs hom d round rounds [] claim lengths

/-- `sumcheckRun_plain` for `sumcheckWith`. -/
theorem sumcheckRun_plain_with (d : ℕ) (round : List K → K[X]) (extra : List K → Finset K)
    {claim : Ext} {rounds : List (List Ext × Ext)} {out : Ended}
    (ran : sumcheckRun .plain claim rounds = some out)
    (lengths : ∀ x ∈ rounds, x.1.length = d + 1) :
    (sumcheckWith (d + 1) round extra rounds.length [] (φ claim)).Runs
      (plainPath φ d claim rounds) (some (out.point.map φ, φ out.claim)) := by
  simp only [sumcheckRun, Option.some.injEq] at ran
  subst ran
  simpa using plainRun_runs_with hom d round extra rounds [] claim lengths

/-- The equality-factored form: `Game.sumcheck` on the path of the code ends in the point of the
code and `E_n·c_n`, for `E_n = ẽq(τ, r)` and the end claim `c_n` of the code. -/
theorem sumcheckRun_equality (D : ℕ) (round : List K → K[X]) {before : List Ext} {claim : Ext}
    {rounds : List (List Ext × Ext)} {out : Ended}
    (ran : sumcheckRun (.equality before) claim rounds = some out)
    (lengths : ∀ x ∈ rounds, x.1.length = D) :
    (Game.sumcheck (D + 1) round rounds.length [] (φ claim)).Runs
      (eqPath φ D Ext.one claim before rounds)
      (some (out.point.map φ, mle (out.point.map φ) (eqAt (before.map φ)) * φ out.claim)) := by
  have runs := eqRun_runs hom D round before rounds [] claim Ext.one out lengths
    (sumcheckRun_equality_eqRun ran)
  rw [hom.one, one_mul, (sumcheckRun_equality_factor hom ran).2] at runs
  simpa using runs

/-- `sumcheckRun_equality` for `sumcheckWith`. -/
theorem sumcheckRun_equality_with (D : ℕ) (round : List K → K[X]) (extra : List K → Finset K)
    {before : List Ext} {claim : Ext} {rounds : List (List Ext × Ext)} {out : Ended}
    (ran : sumcheckRun (.equality before) claim rounds = some out)
    (lengths : ∀ x ∈ rounds, x.1.length = D) :
    (sumcheckWith (D + 1) round extra rounds.length [] (φ claim)).Runs
      (eqPath φ D Ext.one claim before rounds)
      (some (out.point.map φ, mle (out.point.map φ) (eqAt (before.map φ)) * φ out.claim)) := by
  have runs := eqRun_runs_with hom D round extra before rounds [] claim Ext.one out lengths
    (sumcheckRun_equality_eqRun ran)
  rw [hom.one, one_mul, (sumcheckRun_equality_factor hom ran).2] at runs
  simpa using runs

end Spec

section Code

variable {K : Type} [Field K]

/-- The message of the code for a message of `Game.sumcheck (d + 1)`: its coefficients without
the linear one. -/
def plainCode {d : ℕ} (message : Fin (d + 2) → K) : List K :=
  message 0 :: List.ofFn fun i : Fin d => message i.succ.succ

theorem coefficients_plainPoly_check {d : ℕ} (claim : K) {message : List K}
    (length : message.length = d + 1) :
    (coeffPoly (coefficients (d + 1) (plainPoly claim message))).eval 0 +
      (coeffPoly (coefficients (d + 1) (plainPoly claim message))).eval 1 = claim := by
  rw [coeffPoly_coefficients (plainPoly_natDegree claim length), plainPoly_check]

theorem plainCode_coefficients {d : ℕ} (claim : K) {message : List K}
    (length : message.length = d + 1) :
    plainCode (coefficients (d + 1) (plainPoly claim message)) = message := by
  obtain ⟨c, rest, rfl⟩ := List.exists_cons_of_length_eq_add_one length
  simp only [List.length_cons, Nat.add_right_cancel_iff] at length
  subst length
  simp only [plainCode, coefficients, plainPoly, listPoly_coeff, List.headD_cons, List.tail_cons]
  refine congrArg _ (List.ext_getElem (by simp) fun i _ inside => ?_)
  simp [List.getD_eq_getElem?_getD]

theorem coefficients_plainCode {d : ℕ} (claim : K) (message : Fin (d + 2) → K)
    (checked : (coeffPoly message).eval 0 + (coeffPoly message).eval 1 = claim) :
    coefficients (d + 1) (plainPoly claim (plainCode message)) = message := by
  rw [coeffPoly_check, Fin.sum_univ_succ, Fin.sum_univ_succ] at checked
  funext i
  simp only [coefficients, plainPoly, plainCode, listPoly_coeff, List.headD_cons, List.tail_cons]
  rcases i with ⟨_ | _ | i, inside⟩
  · rfl
  · simp only [List.getD_cons_succ, List.getD_cons_zero, List.sum_ofFn]
    rw [← checked]
    simp only [Fin.succ_zero_eq_one]
    ring_nf
    rfl
  · have small : i < d := by omega
    simp [List.getD_eq_getElem?_getD, small]

end Code

end NightstreamFPrime.Spec.Compression.Akita.Replay
