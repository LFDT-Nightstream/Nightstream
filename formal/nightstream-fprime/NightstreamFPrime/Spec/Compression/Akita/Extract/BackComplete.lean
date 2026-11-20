import NightstreamFPrime.Spec.Compression.Verifier.Honest
import NightstreamFPrime.Spec.Compression.Akita.Extract.CheckComplete
import NightstreamFPrime.Spec.Compression.Akita.Extract.Reject
import NightstreamFPrime.Spec.Compression.Akita.Extract.SplitGame

/-! The honest prover of the two parts `led` and `back` of the verifier game (SPEC.md
Algorithm 9, steps 6 to 11 and Rejects; Algorithm 10).

- `Rejects` has the two failures of the honest prover: a lookup challenge is a table value, or
  `ω` has no row point.
- `back_ends`: the points of the opening claims of `back` have `laneVars + keyBlockVars +
  rowVars`, `witnessVars` and `spanVars + columnVars + zeros` coordinates.
- `rejects_sample`: the condition on an answer of the oracle that decodes to no challenge in
  `Rejects`.
- `rejects_mass_le`: an answer of the oracle decodes to a challenge of `Rejects` with
  probability at most `tableError + omegaError`.
- `honestInverses_zero`: the honest inverses are 0 after the columns of the lookup challenges.
- `led_wins`: an honest prover of `lead` is an honest prover of `led`, and the lookup challenges
  at its end are no table values.
- `back_wins`: for a statement and a table `W` with `Relation`, honest counts and the honest
  inverses `V`, the prover has a message for each draw sequence outside `Rejects` such that
  `back` ends in opening claims that are true for the key polynomial, `W` and `V`. The prover
  sends the column values of `V` below `2^columnVars` only (`shortValues`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Verifier

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Algebra
open NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression.Akita.Extract
open NightstreamFPrime.Spec.RandomOracle (mass)
open scoped BigOperators

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
  [Algebra F K] {P : Params F K} (vf : VerifierData P)

local notation "ι" => algebraMap F K

/-- The challenges for which the honest prover of the argument fails (SPEC.md Algorithm 9,
Rejects): lookup challenges with a table value, and challenges after `V` whose `ω` has no row
point. -/
def Rejects (P : Params F K) (C : Type) (challenge : C) : Prop :=
  Hit {τ | τ ∈ tableHit P} C challenge ∨ Hit {ch | ch ∈ omegaBad P} C challenge

omit [Algebra F K] in
/-- An answer of the oracle that decodes to lookup challenges with no table value and to
challenges after `V` whose `ω` has a row point decodes to no challenge in `Rejects`. -/
theorem rejects_sample {Reply : Type} [Fintype Reply] {reply : Reply}
    (calm : sample Reply (Fin P.lookups → F) reply ∉ tableHit P)
    (regular : sample Reply (Challenges P) reply ∉ omegaBad P) (C : Type) [finite : Fintype C]
    [Nonempty C] : ¬ Rejects P C (sample Reply C reply) := by
  rintro (⟨same, member⟩ | ⟨same, member⟩)
  · subst same
    have found : sample Reply (Fin P.lookups → F) reply ∈ tableHit P := member
    refine calm ?_
    convert found
  · subst same
    have found : sample Reply (Challenges P) reply ∈ omegaBad P := member
    refine regular ?_
    convert found

omit [Algebra F K] in
/-- An answer of the oracle decodes to a challenge of `Rejects` with probability at most
`tableError + omegaError`, the bound of SPEC.md Algorithm 9, Rejects. `lookups` and `challenges`:
an answer gives uniform lookup challenges and uniform challenges after `V`. -/
theorem rejects_mass_le {Reply : Type} [Fintype Reply]
    (lookups : ∃ decode : Reply → Fin P.lookups → F, Fair decode)
    (challenges : ∃ decode : Reply → Challenges P, Fair decode) :
    mass {reply : Reply | sample Reply (Fin P.lookups → F) reply ∈ tableHit P ∨
      sample Reply (Challenges P) reply ∈ omegaBad P} ≤ tableError P + omegaError P := by
  have union : mass {reply : Reply | sample Reply (Fin P.lookups → F) reply ∈ tableHit P ∨
      sample Reply (Challenges P) reply ∈ omegaBad P} ≤
      mass {reply : Reply | sample Reply (Fin P.lookups → F) reply ∈ tableHit P} +
        mass {reply : Reply | sample Reply (Challenges P) reply ∈ omegaBad P} := by
    unfold mass
    rw [← Finset.expect_add_distrib]
    refine Finset.expect_le_expect fun reply _ => ?_
    by_cases left : sample Reply (Fin P.lookups → F) reply ∈ tableHit P <;>
      by_cases right : sample Reply (Challenges P) reply ∈ omegaBad P <;> simp [left, right]
  exact union.trans (add_le_add ((sample_fair lookups _).trans density_tableHit_le)
    ((sample_fair challenges _).trans (le_of_eq density_omegaBad)))

omit [Field K] [Fintype K] [DecidableEq K] [Algebra F K] in
theorem led_settled {Lead : Type} {lead : Game (Option Lead)} (settled : Settled lead)
    (W : ℕ → F) : Settled (led P lead W) := by
  cases lead with
  | pure result => cases result <;> exact trivial
  | draw C bad next => exact trivial
  | recv M next => exact settled.elim

omit [Fintype F] [DecidableEq F] in
/-- The opening claims at an end of `back` have points of `laneVars + keyBlockVars + rowVars`,
`witnessVars` and `spanVars + columnVars + zeros` coordinates. -/
theorem back_ends (wf : WellFormed P) (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F) :
    (back vf st W V τ).Ends fun result => ∀ claims, result = some claims →
      claims.key.1.length = P.laneVars + P.keyBlockVars + P.rowVars ∧
        claims.w.1.length = P.witnessVars ∧
        claims.v.1.length = P.spanVars + vf.columnVars + vf.zeros := by
  intro ch
  show (bind _ _).Ends _
  rw [ends_bind]
  refine (sumcheck_ends 3 _ P.witnessVars [] _).mono ?_
  intro first ends₁
  rcases first with _ | ⟨ρ, value⟩
  · intro claims none_eq
    cases none_eq
  have ρ_length : ρ.length = P.witnessVars := by
    obtain ⟨cs, eq, len⟩ := ends₁ ρ value rfl
    rw [eq, List.nil_append, len]
  intro vals col
  show (bind _ _).Ends _
  rw [ends_bind]
  refine (sumcheck_ends 2 _ _ [] _).mono ?_
  intro second ends₂
  rcases second with _ | ⟨kp, kfinal⟩
  · intro claims none_eq
    cases none_eq
  have kp_length : kp.length = P.laneVars + P.keyBlockVars := by
    obtain ⟨ks, eq, len⟩ := ends₂ kp kfinal rfl
    rw [eq, List.nil_append, len]
  intro kv claims equal
  split at equal
  · cases equal
    refine ⟨?_, ρ_length, ?_⟩
    · show (kp ++ rowPoint P.rowVars ch.omega).length = _
      rw [List.length_append, kp_length, rowPoint_length]
    · show (ρ.take P.spanVars ++ List.ofFn col ++ List.replicate vf.zeros 0).length = _
      rw [List.length_append, List.length_append, List.length_take, List.length_ofFn,
        List.length_replicate, ρ_length, Nat.min_eq_left wf.span_le]
  · cases equal

omit [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] [Algebra F K] in
/-- The honest inverses are 0 from the cell `2^(spanVars + n)`, when `2^n` columns hold the
lookup challenges. -/
theorem honestInverses_zero (W : ℕ → F) (τ : Fin P.lookups → F) {n cell : ℕ}
    (columns : P.lookups ≤ 2 ^ n) (large : 2 ^ (P.spanVars + n) ≤ cell) :
    honestInverses P W τ cell = 0 := by
  have high : 2 ^ n ≤ cell / 2 ^ P.spanVars :=
    (Nat.le_div_iff_mul_le (Nat.two_pow_pos _)).mpr (by rwa [pow_add, Nat.mul_comm] at large)
  exact ite_eq_right fun inside => absurd (lt_of_lt_of_le inside.1 columns) (not_lt.mpr high)

variable {sendable : (M : Type) → M → Prop} {slots : ℕ}

omit [Algebra F K] in
/-- An honest prover of `lead` is an honest prover of `led`: the game ends in the value of
`lead` and lookup challenges that are no table values. -/
theorem led_wins {Lead : Type} {lead : Game (Option Lead)} {draws : ℕ} {goal : Lead → Prop}
    (W : ℕ → F)
    (leads : WinsOff (Rejects P) sendable slots lead draws 0 fun result _ _ =>
      ∃ value, result = some value ∧ goal value) :
    WinsOff (Rejects P) sendable slots (led P lead W) (draws + 1) 0 fun result _ _ =>
      ∃ value τ, result = some (value, τ) ∧ goal value ∧ Apart P τ := by
  refine (winsOff_bind _ _ _ _ _ _ _ _).mpr ((leads.more 1).mono ?_)
  rintro result rest place ⟨spare, value, rfl, good⟩
  obtain ⟨more, rfl⟩ := Nat.exists_eq_add_of_le spare
  rw [Nat.add_comm]
  intro τ allowed
  exact ⟨value, τ, rfl, good, fun c table k v lt equal => allowed (Or.inl (hit_self.mpr
    (Finset.mem_filter.mpr ⟨Finset.mem_univ _, c, table, k, v, lt, equal⟩)))⟩

/-- `W(ρ)`, the values `V_k` below `2^columnVars` and `K_w` of SPEC.md Algorithm 9, step 9. The
verifier reads no other value `V_k`. -/
noncomputable def shortValues (W V : ℕ → F) (ch : Challenges P) (ρ : List K) : Values K :=
  (mle ρ fun x => ι (W x), fun k => if k < 2 ^ vf.columnVars then columnAt P V ρ k else 0,
    keyTrue P ch ρ)

/-- The honest prover of `back` (SPEC.md Algorithm 9, steps 7 to 11). For a statement and a
table `W` with `Relation`, honest counts, lookup challenges that are no table values and the
honest inverses, the prover has a message for each draw sequence outside `Rejects` such that
`back` ends in opening claims that are true for the key polynomial, `W` and the inverses.
`sends`, `wide`, `room` and `short`: a round polynomial, an element of `K` and the values
`shortValues` are messages of the prover. -/
theorem back_wins {width cap : ℕ} (sends : Sends sendable K width cap) (wide : 4 ≤ width)
    (room : 0 < slots)
    (short : ∀ values : Values K, (∀ k, 2 ^ vf.columnVars ≤ k → values.2.1 k = 0) →
      sendable (Values K) values)
    (wf : WellFormed P) (matrix : MatrixSpec vf) (columns : P.lookups ≤ 2 ^ vf.columnVars)
    (rows : P.rows ≤ 2 ^ P.rowVars)
    (key_fits : ∀ c, (P.shape c).witness.vars - P.laneVars ≤ P.keyBlockVars)
    (st : Statement P) (W : ℕ → F) (τ : Fin P.lookups → F) (shaped : Shaped P st)
    (relation : Relation P st W) (honest : HonestCounts P W) (distinct : TablesDistinct P)
    (apart : Apart P τ) :
    WinsOff (Rejects P) sendable slots (back vf st W (honestInverses P W τ) τ)
      (backCosts vf).length 0 fun result _ _ => ∃ claims, result = some claims ∧
        Opened (keyCells P) (fun x => ι (W x)) (fun x => ι (honestInverses P W τ x)) claims := by
  have count : (backCosts vf).length =
      0 + (P.laneVars + P.keyBlockVars) + 1 + P.witnessVars + 1 := by
    simp only [backCosts, List.length_cons, List.length_append, List.length_replicate]
    omega
  rw [count]
  intro ch allowed
  have regular : ∀ i < P.rowVars, 1 + ch.omega ^ 2 ^ i ≠ 0 := fun i lt zero =>
    allowed (Or.inr (hit_self.mpr (Finset.mem_filter.mpr ⟨Finset.mem_univ _, i, lt, zero⟩)))
  have inverses := honestInverses_spec W τ wf
  have main_truth := combRound_truth P.witnessVars 3
    (mainTables P st W (honestInverses P W τ) τ ch) mainComb mainComb_degree
  refine (winsOff_bind _ _ _ _ _ _ _ _).mpr ?_
  refine (sumcheck_winsAt main_truth (fun _ => sends.vector 4 _ wide)
    (0 + (P.laneVars + P.keyBlockVars) + 1) _ P.witnessVars [] (claimedSum P st ch) 0 (by simp)
    (by
      rw [main_truth_nil]
      exact (honest_sum st W _ τ ch wf relation honest inverses distinct apart).symm) room
    ?_).off
  intro ρ ρ_sized place early
  have ρ_length : ρ.length = P.witnessVars := by simpa using ρ_sized
  refine ⟨lt_of_le_of_lt early room, shortValues vf W (honestInverses P W τ) ch ρ,
    short _ fun k large => ite_eq_right (not_lt.mpr large), ?_⟩
  intro col _
  have point_length (c : Fin P.claims) :
      (uses P ch ρ c).point.length = (P.shape c).witness.vars - P.laneVars := by
    have := (wf.witness_fits c).2.2
    simp [uses, ρ_length, this]
  have block_fits (c : Fin P.claims) : (uses P ch ρ c).blocks ≤ 2 ^ P.keyBlockVars :=
    (shaped.blocks_le c).trans (Nat.pow_le_pow_right two_pos (key_fits c))
  have key_truth := combRound_truth (P.laneVars + P.keyBlockVars) 2 (keyTables ch ρ) productComb
    productComb_degree
  refine (winsOff_bind _ _ _ _ _ _ _ _).mpr ?_
  refine (sumcheck_winsAt key_truth (fun _ => sends.vector 3 _ (by omega)) 0 _
    (P.laneVars + P.keyBlockVars) [] (keyTrue P ch ρ) 0 (by simp)
    (by
      rw [product_truth_nil, cubeSum_eq_sum_range, show 2 ^ (P.laneVars + P.keyBlockVars) =
        P.lanes * 2 ^ P.keyBlockVars by rw [pow_add]; rfl]
      exact (key_sum P.lanes P.keyBlockVars _ _ _ block_fits (pow_pos two_pos _)).symm) room
    ?_).off
  intro kp kp_sized last late
  have kp_length : kp.length = P.laneVars + P.keyBlockVars := by simpa using kp_sized
  refine ⟨lt_of_le_of_lt late room, honestKey ch kp, sends.scalar _, ?_⟩
  have cut : ∀ k : Fin P.lookups,
      (shortValues vf W (honestInverses P W τ) ch ρ).2.1 k =
        columnAt P (honestInverses P W τ) ρ k :=
    fun k => ite_eq_left (lt_of_lt_of_le k.isLt columns)
  have checked : Checked vf st τ ch ρ
      (combTruth P.witnessVars (mainTables P st W (honestInverses P W τ) τ ch) mainComb ρ)
      (shortValues vf W (honestInverses P W τ) ch ρ) kp
      (combTruth (P.laneVars + P.keyBlockVars) (keyTables ch ρ) productComb kp)
      (honestKey ch kp) := by
    refine ⟨shaped, apart, regular, ?_, ?_⟩
    · rw [← kp_length, product_truth_final]
      exact key_check_honest ch ρ kp wf st shaped ρ_length kp_length rows regular key_fits
    · have main := main_check_honest vf st W (honestInverses P W τ) τ ch ρ wf shaped ρ_length
        matrix
      simp only [← cut] at main
      exact main
  refine ⟨_, ite_eq_left checked, rfl, rfl, ?_⟩
  exact (mle_congr _ fun k small => ite_eq_left (by rwa [List.length_ofFn] at small)).trans
    (mle_V_opening _ ρ wf ρ_length _ _).symm

end NightstreamFPrime.Spec.Compression.Verifier
