import NightstreamFPrime.Spec.Compression.Verifier.Sound

/-! The verifier game in two parts, with the Akita openings as data (SPEC.md section 6).

`game` tests the three openings against the tables that the prover names (`Accepts`). Here the
game ends in the three opening claims instead, and the test of the claims is outside the game.

- `Checked` has the conditions of `Accepts` that read no table, and `Opened` states that three
  tables have the claimed values at the points of the claims `Finals` (`accepts_iff`).
- `front` is the game after the commitment of `W`: the draw of the lookup challenges. `back` is
  the game after the commitment of `V`: the draws and the messages through the key sum-check. It
  ends in the opening claims when the conditions `Checked` hold.
- `afterTables_eq`: the game after `W` is `front`, the message `V`, `back` and the test `Opened`.
- `front_sound`: outside the bad set, if some `V` has no error then `Relation` holds for `W`.
  `front_avoids` is the form for a false `W`. `back_avoids`: on each path of `back` that avoids
  the bad sets, true opening claims give no error for `(W, V)`.
- `split_avoids` is `game_avoids` from these two parts. -/

namespace NightstreamFPrime.Spec.Compression

theorem Game.bind_assoc {α β γ : Type} (game : Game α) (after : α → Game β)
    (last : β → Game γ) :
    (game.bind after).bind last = game.bind fun result => (after result).bind last := by
  induction game with
  | pure result => rfl
  | draw C bad next induction => exact congrArg (Game.draw C bad) (funext induction)
  | recv M next induction => exact congrArg (Game.recv M) (funext induction)

namespace Verifier

open Finset NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Algebra
open NightstreamFPrime.Spec.Compression.Game Polynomial

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
  [Algebra F K]

local notation "ι" => algebraMap F K

/-- The three opening claims at the end of the verifier (SPEC.md Algorithm 10, the Akita
claims): the point and the claimed value of the key polynomial, of `W` and of `V`. -/
structure Finals (K : Type) where
  key : List K × K
  w : List K × K
  v : List K × K

omit [Fintype K] [DecidableEq K] in
/-- The three tables have the claimed values at the points of the claims. -/
def Opened (key w v : ℕ → K) (claims : Finals K) : Prop :=
  claims.key.2 = mle claims.key.1 key ∧ claims.w.2 = mle claims.w.1 w ∧
    claims.v.2 = mle claims.v.1 v

/-- The lookup challenges are not values of a table (Rust `verify`, the check of `τ`). -/
def Apart (P : Params F K) (τ : Fin P.lookups → F) : Prop :=
  ∀ c, (P.shape c).lookup = true → ∀ k, ∀ v < 2 * (P.shape c).bound + 1,
    τ k ≠ tableValue (P.shape c).bound v

variable {P : Params F K} (vf : VerifierData P)

/-- The conditions of `Accepts` that read no table. -/
def Checked (st : Statement P) (τ : Fin P.lookups → F) (ch : Challenges P) (ρ : List K)
    (value : K) (vals : Values K) (kp : List K) (kfinal kv : K) : Prop :=
  Shaped P st ∧ Apart P τ ∧ (∀ i < P.rowVars, 1 + ch.omega ^ 2 ^ i ≠ 0) ∧
  kfinal = rowScale P.rowVars ch.omega * kv *
    weightAt P.lanes P.laneVars (laneForm P ch ρ) (uses P ch ρ) kp ∧
  value = mainSummand (fun k => ι (τ k)) ch.psi vals.1
    (vals.2.2 + vf.matrixPart st ch ρ + closedPart P st τ ch ρ)
    (checkAt P ch ρ) (lowAt P ρ * vals.2.1) (sumAt P ch ρ)

/-- The opening claims of a transcript of the verifier. -/
def finals (ch : Challenges P) (ρ : List K) (vals : Values K) (kp : List K) (kv : K) :
    Finals K :=
  ⟨(kp ++ rowPoint P.rowVars ch.omega, kv), (ρ, vals.1),
    (ρ.take P.spanVars ++ List.replicate vf.zeros 0, vals.2.1)⟩

/-- The key polynomial of `Params.key` as a table. -/
def keyCells (P : Params F K) : ℕ → K :=
  keyTable P.lanes P.keyBlockVars P.rows fun i j h => ι (P.key i j h)

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem accepts_iff (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F) (ch : Challenges P)
    (ρ : List K) (value : K) (vals : Values K) (kp : List K) (kfinal kv : K) :
    Accepts vf st W V τ ch ρ value vals kp kfinal kv ↔
      Checked vf st τ ch ρ value vals kp kfinal kv ∧
        Opened (keyCells P) (fun x => ι (W x)) (fun x => ι (V x))
          (finals vf ch ρ vals kp kv) := by
  simp only [Accepts, Checked, Apart, Opened, finals, keyCells, and_assoc]

/-- The draw of the lookup challenges, after the commitment of `W`. -/
noncomputable def front (P : Params F K) (W : ℕ → F) : Game (Fin P.lookups → F) :=
  .draw (Fin P.lookups → F) (tauBad P W) .pure

/-- `back` after the key sum-check. -/
noncomputable def backKey (st : Statement P) (τ : Fin P.lookups → F) (ch : Challenges P)
    (ρ : List K) (value : K) (vals : Values K) :
    Option (List K × K) → Game (Option (Finals K))
  | none => .pure none
  | some (kp, kfinal) => .recv K fun kv => by
      classical
      exact .pure (if Checked vf st τ ch ρ value vals kp kfinal kv then
        some (finals vf ch ρ vals kp kv) else none)

/-- `back` after the first sum-check. -/
noncomputable def backMain (st : Statement P) (τ : Fin P.lookups → F)
    (ch : Challenges P) : Option (List K × K) → Game (Option (Finals K))
  | none => .pure none
  | some (ρ, value) =>
      .recv (Values K) fun vals =>
        bind (sumcheck 2 (combRound (P.laneVars + P.keyBlockVars) (keyTables ch ρ)
          productComb) (P.laneVars + P.keyBlockVars) [] vals.2.2)
          (backKey vf st τ ch ρ value vals)

/-- The verifier after the commitment of `V`, for the lookup challenges `τ`. It ends in the
opening claims, or in `none` when a check fails. The tables are in the bad sets only. -/
noncomputable def back (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F) :
    Game (Option (Finals K)) :=
  .draw (Challenges P) (challengeBad P st W V τ) fun ch =>
    bind (sumcheck (P.lookups + 2)
      (combRound P.witnessVars (mainTables P st W V τ ch) (mainComb P τ ch))
      P.witnessVars [] (claimedSum P st ch)) (backMain vf st τ ch)

/-- The result of `game` for a result of `back`: the test of the ideal openings. -/
noncomputable def verdict (st : Statement P) (W V : ℕ → F) (result : Option (Finals K)) :
    Option (Statement P × (ℕ → F)) := by
  classical
  exact result.bind fun claims =>
    if Opened (keyCells P) (fun x => ι (W x)) (fun x => ι (V x)) claims then some (st, W)
    else none

/-- The game after `W` is `front`, the message `V`, `back` and the test of the openings. -/
theorem afterTables_eq (st : Statement P) (W : ℕ → F) :
    afterTables vf st W = bind (front P W) fun τ => .recv (ℕ → F) fun V =>
      bind (back vf st W V τ) fun result => .pure (verdict st W V result) := by
  refine congrArg (Game.draw _ _) (funext fun τ => congrArg (Game.recv _) (funext fun V =>
    congrArg (Game.draw _ _) (funext fun ch => ?_)))
  rw [Game.bind_assoc]
  refine congrArg _ (funext fun first => ?_)
  rcases first with _ | ⟨ρ, value⟩
  · rfl
  refine congrArg (Game.recv _) (funext fun vals => ?_)
  rw [Game.bind_assoc]
  refine congrArg _ (funext fun second => ?_)
  rcases second with _ | ⟨kp, kfinal⟩
  · rfl
  refine congrArg (Game.recv _) (funext fun kv => ?_)
  show Game.pure _ = Game.pure _
  have iff := accepts_iff vf st W V τ ch ρ value vals kp kfinal kv
  by_cases checked : Checked vf st τ ch ρ value vals kp kfinal kv
  · by_cases opened : Opened (keyCells P) (fun x => ι (W x)) (fun x => ι (V x))
        (finals vf ch ρ vals kp kv)
    · simp [verdict, checked, opened, iff.mpr ⟨checked, opened⟩]
    · have rejected : ¬ Accepts vf st W V τ ch ρ value vals kp kfinal kv :=
        fun accepted => opened (iff.mp accepted).2
      simp [verdict, checked, opened, rejected]
  · have rejected : ¬ Accepts vf st W V τ ch ρ value vals kp kfinal kv :=
      fun accepted => checked (iff.mp accepted).1
    simp [verdict, checked, rejected]

omit [Fintype K] [DecidableEq K] in
/-- Outside the bad set of `front`, lookup challenges that are no table values and a table `V`
with no error give `Relation` for `W`. The table `V` is free: it can be named after the draw. -/
theorem front_sound (counts : CountsNonzero P) (st : Statement P) (W : ℕ → F) :
    (front P W).Avoids fun τ => ∀ V, Apart P τ → ¬ Fails P st W V τ → Relation P st W :=
  fun τ good V apart holds => relation_of_good st W V τ counts apart good holds

omit [Fintype K] [DecidableEq K] in
/-- For a table `W` that does not satisfy `Relation`, on each path of `front` that avoids the
bad set no table `V` is without error. -/
theorem front_avoids (counts : CountsNonzero P) (st : Statement P) (W : ℕ → F)
    (refuted : ¬ Relation P st W) :
    (front P W).Avoids fun τ => ¬ (Apart P τ ∧ ∃ V, ¬ Fails P st W V τ) :=
  (front_sound counts st W).mono fun _ sound passed =>
    refuted (sound passed.2.choose passed.1 passed.2.choose_spec)

omit [Fintype F] [DecidableEq F] in
/-- On each path of `back` that avoids the bad sets, opening claims that are true for the key
polynomial, `W` and `V` give lookup challenges that are no table values and no error for
`(W, V)`. -/
theorem back_avoids (wf : WellFormed P) (matrix : MatrixSpec vf) (rows : P.rows ≤ 2 ^ P.rowVars)
    (key_fits : ∀ c, (P.shape c).witness.vars - P.laneVars ≤ P.keyBlockVars)
    (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F) :
    (back vf st W V τ).Avoids fun result => ∀ claims, result = some claims →
      Opened (keyCells P) (fun x => ι (W x)) (fun x => ι (V x)) claims →
        Apart P τ ∧ ¬ Fails P st W V τ := by
  intro ch ch_good
  have main_truth := combRound_truth P.witnessVars (P.lookups + 2) (mainTables P st W V τ ch)
    (mainComb P τ ch) (mainComb_degree P τ ch)
  show (bind _ _).Avoids _
  rw [avoids_bind]
  refine ((sumcheck_avoids (P.lookups + 2) P.witnessVars _ _ main_truth P.witnessVars []
    (claimedSum P st ch) (by simp)).and_ends
    (sumcheck_ends (P.lookups + 2) _ P.witnessVars [] _)).mono ?_
  rintro first ⟨sound₁, ends₁⟩
  rcases first with _ | ⟨ρ, value⟩
  · intro claims none_eq
    cases none_eq
  have ρ_length : ρ.length = P.witnessVars := by
    obtain ⟨cs, eq, len⟩ := ends₁ ρ value rfl
    rw [eq, List.nil_append, len]
  intro vals
  have key_truth := combRound_truth (P.laneVars + P.keyBlockVars) 2 (keyTables ch ρ) productComb
    productComb_degree
  show (bind _ _).Avoids _
  rw [avoids_bind]
  refine ((sumcheck_avoids 2 (P.laneVars + P.keyBlockVars) _ _ key_truth _ [] vals.2.2
    (by simp)).and_ends (sumcheck_ends 2 _ _ [] _)).mono ?_
  rintro second ⟨sound₂, ends₂⟩
  rcases second with _ | ⟨kp, kfinal⟩
  · intro claims none_eq
    cases none_eq
  have kp_length : kp.length = P.laneVars + P.keyBlockVars := by
    obtain ⟨ks, eq, len⟩ := ends₂ kp kfinal rfl
    rw [eq, List.nil_append, len]
  intro kv claims equal opened
  split at equal
  · rename_i checked
    cases equal
    obtain ⟨shaped, τ_ok, regular, key_check, main_check⟩ := checked
    have kv_open : kv = mle (kp ++ rowPoint P.rowVars ch.omega)
        (keyTable P.lanes P.keyBlockVars P.rows fun i j h => ι (P.key i j h)) := opened.1
    have wv_open : vals.1 = mle ρ (fun x => ι (W x)) := opened.2.1
    have v_open : vals.2.1 =
        mle (ρ.take P.spanVars ++ List.replicate vf.zeros 0) (fun x => ι (V x)) := opened.2.2
    rw [mle_V_opening] at v_open
    have point_length (c : Fin P.claims) :
        (uses P ch ρ c).point.length = (P.shape c).witness.vars - P.laneVars := by
      have := (wf.witness_fits c).2.2
      simp [uses, ρ_length, this]
    have block_fits (c : Fin P.claims) : (uses P ch ρ c).blocks ≤ 2 ^ P.keyBlockVars :=
      (shaped.blocks_le c).trans (Nat.pow_le_pow_right two_pos (key_fits c))
    have kw : vals.2.2 = keyTrue P ch ρ := by
      by_contra differ
      have truth_nil : combTruth (P.laneVars + P.keyBlockVars) (keyTables ch ρ) productComb [] =
          keyTrue P ch ρ := by
        rw [product_truth_nil, cubeSum_eq_sum_range, show 2 ^ (P.laneVars + P.keyBlockVars) =
          P.lanes * 2 ^ P.keyBlockVars by rw [pow_add]; rfl]
        exact key_sum P.lanes P.keyBlockVars _ _ _ block_fits (pow_pos two_pos _)
      apply sound₂ kp kfinal rfl (by rw [truth_nil]; exact differ)
      rw [← kp_length, product_truth_final]
      simp only [keyTables, Matrix.cons_val_zero, Matrix.cons_val_one, Params.lanes]
      rw [weightAt_spec P.laneVars P.keyBlockVars _ _ kp kp_length fun c =>
        ⟨by rw [point_length]; exact key_fits c, by rw [point_length]; exact shaped.blocks_le c⟩,
        folded_at (2 ^ P.laneVars) P.keyBlockVars P.rows P.rowVars _ ch.omega kp
          (by rw [kp_length, pow_add]) rows regular]
      rw [key_check, kv_open]
      simp only [Params.lanes]
      ring
    have claimed : claimedSum P st ch = ∑ x ∈ range (2 ^ P.witnessVars),
        mainSummand (fun k => ι (τ k)) ch.psi (ι (W x)) (weightTable P st τ ch x)
          (checkTable P ch x) (inverseTable P V x) (sumTable P ch x) := by
      by_contra differ
      apply sound₁ ρ value rfl (by rw [main_truth_nil]; exact differ)
      rw [main_truth_final st W V τ ch ρ ρ_length, main_check, wv_open, kw,
        matrix st ch ρ ρ_length, mle_weightTable st τ ch ρ wf shaped ρ_length,
        mle_checkTable ch ρ wf ρ_length, mle_sumTable ch ρ wf ρ_length,
        mle_inverseTable V ρ wf ρ_length, v_open]
    have zero := sum_sub_claimed st W V τ ch wf
    rw [← claimed, sub_self] at zero
    refine ⟨τ_ok, fun fails => ch_good ?_⟩
    simp only [challengeBad, fails, ite_true, mem_filter, mem_univ, true_and]
    exact zero.symm
  · cases equal

/-- `game_avoids` from the two parts: on each path of `game` that avoids the bad sets, an
accepted proof gives `Relation` for the committed `W`. -/
theorem split_avoids (wf : WellFormed P) (counts : CountsNonzero P) (matrix : MatrixSpec vf) (rows : P.rows ≤ 2 ^ P.rowVars)
    (key_fits : ∀ c, (P.shape c).witness.vars - P.laneVars ≤ P.keyBlockVars) :
    (game vf).Avoids fun r => ∀ o, r = some o → Relation P o.1 o.2 := by
  intro st W
  show (afterTables vf st W).Avoids _
  rw [afterTables_eq, avoids_bind]
  refine (front_sound counts st W).mono fun τ sound V => ?_
  show (bind _ _).Avoids _
  rw [avoids_bind]
  refine (back_avoids vf wf matrix rows key_fits st W V τ).mono fun result passes o
    equal => ?_
  rcases result with _ | claims
  · cases equal
  · simp only [verdict, Option.bind_some] at equal
    split at equal
    · rename_i opened
      cases equal
      exact sound V (passes claims rfl opened).1 (passes claims rfl opened).2
    · cases equal

end Verifier

end NightstreamFPrime.Spec.Compression
