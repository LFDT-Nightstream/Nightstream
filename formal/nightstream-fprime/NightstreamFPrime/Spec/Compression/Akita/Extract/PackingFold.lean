import NightstreamFPrime.Spec.Compression.Akita.FlatFold
import NightstreamFPrime.Spec.Compression.Akita.Extract.Bound

/-! The rewinding extraction of one packing fold of Akita with the checks `foldCheck`
(ePrint 2026/1983, revision of 2026-10-15, Theorem 10.17 with Corollary 10.32 and the terms of
(189) after the fold challenge).

- `foldCosts` has the bounds of the draws of `foldCheck` in their order, `foldCosts_sum` states
  that their sum is `foldError`, and `foldCheck_staged` states that each draw has its bound when an
  answer of the oracle has a slot for each coordinate of a draw.
- `packingFold` is the `Level` of one packing fold: the checks are `foldCheck` for the flat
  witness of the digit vectors of the proof and the quotient-free reduction, and the verifier
  accepts when the final claim `(r_2, v_W)` is true for this flat witness.
- `packingFold_extract_sound`, `packingFold_extraction_le` and `packingFold_runs_le` are the
  extraction theorem of one packing fold. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

section Rounds

variable {K : Type} [Field K] [Fintype K] [DecidableEq K] {Reply : Type} [Fintype Reply]

/-- Each draw of `sumcheckWith` has the bound `(d + e)/|K|`, and an accepted end has the entries
`rest` after the bounds of its rounds. -/
theorem sumcheckWith_staged (fair : ∃ decode : Reply → K, Fair decode) (d bound e : ℕ)
    (round : List K → K[X])
    (degree : ∀ point : List K, point.length < bound → (round point).natDegree ≤ d)
    (extra : List K → Finset K) (small : ∀ point, (extra point).card ≤ e) (rest : List ℝ)
    (tail : Option (List K × K) → List ℝ → Prop) (rejected : ∀ costs, tail none costs)
    (ended : ∀ final, tail (some final) rest) :
    ∀ (rounds : ℕ) (point : List K) (claim : K), point.length + rounds ≤ bound →
      StagedTo Reply (sumcheckWith d round extra rounds point claim)
        (List.replicate rounds (((d + e : ℕ) : ℝ) / Fintype.card K) ++ rest) tail := by
  intro rounds
  induction rounds with
  | zero =>
      intro point claim _
      exact ended _
  | succ rounds induction =>
      intro point claim fits message
      dsimp only
      split
      · rw [List.replicate_succ, List.cons_append]
        exact stagedTo_draw fair (density_le _ _ ((Finset.card_union_le _ _).trans
          (Nat.add_le_add (zeros_card_le_of_natDegree_le ((natDegree_sub_le _ _).trans
            (max_le (coeffPoly_natDegree message) (degree point (by omega))))) (small point))))
          fun c => induction (point ++ [c]) _ (by simp; omega)
      · exact rejected _

/-- Each draw of `Game.sumcheck` has the bound `d/|K|`, and an accepted end has the entries
`rest` after the bounds of its rounds. -/
theorem sumcheck_staged (fair : ∃ decode : Reply → K, Fair decode) (d bound : ℕ)
    (round : List K → K[X])
    (degree : ∀ point : List K, point.length < bound → (round point).natDegree ≤ d)
    (rest : List ℝ) (tail : Option (List K × K) → List ℝ → Prop)
    (rejected : ∀ costs, tail none costs) (ended : ∀ final, tail (some final) rest) :
    ∀ (rounds : ℕ) (point : List K) (claim : K), point.length + rounds ≤ bound →
      StagedTo Reply (sumcheck d round rounds point claim)
        (List.replicate rounds ((d : ℝ) / Fintype.card K) ++ rest) tail := by
  intro rounds
  induction rounds with
  | zero =>
      intro point claim _
      exact ended _
  | succ rounds induction =>
      intro point claim fits message
      dsimp only
      split
      · rw [List.replicate_succ, List.cons_append]
        exact stagedTo_draw fair (density_le _ _ (zeros_card_le_of_natDegree_le
          ((natDegree_sub_le _ _).trans (max_le (coeffPoly_natDegree message)
            (degree point (by omega)))))) fun c => induction (point ++ [c]) _ (by simp; omega)
      · exact rejected _

end Rounds

section Costs

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- The bounds of the draws of `foldCheck`, in the order of the draws: `α`, `τ_0`, `τ_1`, the
rounds of the range sum-check, `ζ_bin`, `γ` and the rounds of the fused sum-check
(Akita (189)). -/
noncomputable def foldCosts (E : Type) [Fintype E] (badBound vars batch base : ℕ) : List ℝ :=
  ((badBound : ℝ) / Fintype.card E) :: ((vars : ℝ) / Fintype.card E) ::
    ((batch : ℝ) / Fintype.card E) ::
      (List.replicate vars (((base / 2 + 1 + 1 : ℕ) : ℝ) / Fintype.card E) ++
        (((1 : ℕ) : ℝ) / Fintype.card E) :: (((1 : ℕ) : ℝ) / Fintype.card E) ::
          List.replicate vars (((3 : ℕ) : ℝ) / Fintype.card E))

omit [DecidableEq E] in
/-- The sum of the bounds of the draws is `foldError`. -/
theorem foldCosts_sum (R : Reduction E) (base : ℕ) :
    (foldCosts E R.badBound R.vars R.batch base).sum = foldError R base := by
  simp only [foldCosts, foldError, List.sum_cons, List.sum_append, List.sum_replicate,
    nsmul_eq_mul]

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem foldCosts_nonnegative (badBound vars batch base : ℕ) :
    ∀ cost ∈ foldCosts E badBound vars batch base, 0 ≤ cost := by
  intro cost member
  simp only [foldCosts, List.mem_cons, List.mem_append, List.mem_replicate] at member
  rcases member with rfl | rfl | rfl | ⟨_, rfl⟩ | rfl | rfl | ⟨_, rfl⟩ <;> positivity

variable [Nonempty Coord]

theorem fusedCheck_staged (μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (weight : ℕ → E) (target : E) (r : List E) (sClaim ζ γ : E) :
    Staged (Coord → E) (fusedCheck μ support w weight target r sClaim ζ γ)
      (List.replicate μ (((3 : ℕ) : ℝ) / Fintype.card E)) := by
  have rounds := sumcheck_staged (Reply := Coord → E) exists_fair_slot 3 μ _
    (fused_roundTruth μ w weight (virtWeight support r γ ζ)).degree []
    (fun result rest => StagedTo (Coord → E) (match result with
      | none => Game.pure none
      | some (r2, value) => Game.recv E fun v =>
          if value = v * mle r2 weight + mle r2 (virtWeight support r γ ζ) * (v * (v + 1)) then
            Game.pure (some (r2, v))
          else Game.pure none) rest fun _ _ => True)
    (fun _ => trivial) (fun final v => by
      dsimp only
      split <;> trivial) μ [] (target + γ * sClaim) (by simp)
  rw [List.append_nil] at rounds
  exact (stagedTo_bind _ _ _ _).mpr rounds

/-- Each draw of `foldCheck` has its bound of `foldCosts`, when an answer of the oracle has `μ'`
slots or more and `s_rel` slots or more. -/
theorem foldCheck_staged (R : Reduction E) (base : ℕ) (support : ℕ → Prop)
    [DecidablePred support] (w : Table) (varsFit : R.vars ≤ Fintype.card Coord)
    (batchFit : R.batch ≤ Fintype.card Coord) :
    Staged (Coord → E) (foldCheck R base support w)
      (foldCosts E R.badBound R.vars R.batch base) := by
  have field : ∃ decode : (Coord → E) → E, Fair decode := exists_fair_slot
  unfold Staged foldCheck foldCosts
  refine stagedTo_draw field (density_le (C := E) (R.bad w) _ (R.bad_card w)) fun α => ?_
  refine stagedTo_draw (exists_fair_slots varsFit) (density_rangeBad_le (E := E) base R.vars w)
    fun τ0 => ?_
  refine stagedTo_draw (exists_fair_slots batchFit)
    ((R.rows α).density_batchBad_le R.vars (R.fits α) w) fun τ1 => ?_
  refine (stagedTo_bind _ _ _ _).mpr (sumcheckWith_staged field (base / 2 + 1) R.vars 1 _
    (range_roundTruth base R.vars (List.ofFn τ0) w).degree _ (fun _ => bindBad_card_le _ _) _ _
    (fun _ => trivial) ?_ R.vars [] 0 (by simp))
  rintro ⟨r, value⟩ sClaim
  dsimp only
  split
  · exact stagedTo_draw field (density_le _ 1
      (zeros_card_le_of_natDegree_le (linePoly_natDegree _ _))) fun ζ =>
        stagedTo_draw field (density_le _ 1
          (zeros_card_le_of_natDegree_le (linePoly_natDegree _ _))) fun γ =>
            fusedCheck_staged R.vars support w _ _ r sClaim ζ γ
  · trivial

end Costs

section Fold

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E] [Fintype E]
  [DecidableEq E] [Algebra Zq E] {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord]
  (β : Module.Basis (Fin O.planes) Zq E) (st : Statement O E) (μ : ℕ)
  (fits : O.cells fold < 2 ^ μ) (batch : ℕ) (rowsFit : O.rowCount + 1 ≤ 2 ^ batch)
  (block : Fin S.blocks → Coord) (decode : E → Option (Rq O.carrierDegree))
  (witnessOf : Wire → Witness O fold) (read : Reader Wire) (weight : ℕ)

/-- One packing fold for one claim as a `Level`. A slot of an answer of the oracle is one uniform
element of `E`. The checks are `foldCheck` for the quotient-free reduction of `flatRelation` and
the flat witness of the digit vectors of the proof. The verifier accepts a result `(r_2, v_W)`
when `v_W = w̃(r_2)` for this flat witness: the proof has the digit vectors in the clear, so the
verifier computes the evaluation. -/
noncomputable def packingFold : Level O fold E Wire Coord E (Option (List E × E)) where
  basis := β
  statement := st
  block := block
  decode := decode
  witnessOf := witnessOf
  read := read
  check challenge candidate := foldCheck (flatReduction β st challenge μ fits batch rowsFit)
    S.certifiedBase (O.binarySupport fold) candidate.flat
  passes candidate result := ∃ r2 v, result = some (r2, v) ∧ v = mle r2 (lifted E candidate.flat)
  costs := foldCosts E (O.largestDegree - 1) μ batch S.certifiedBase
  weight := weight

variable {β st μ fits batch rowsFit block decode witnessOf read weight}

/-- One packing fold satisfies the conditions of `Level.Valid`. The hypotheses are: the blocks
have different coordinates, the challenge family has units as differences (Akita Definition 9.2,
E1) and members with `‖c‖_1 = ω`, `b_z ≥ 2`, `b⋆ < q`, and an answer of the oracle has `μ'` slots
or more and `s_rel` slots or more. -/
theorem packingFold_valid [Nonempty Coord] {peak : ℕ} (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord) :
    (packingFold β st μ fits batch rowsFit block decode witnessOf read weight).Valid peak where
  injective := injective
  units := units
  sparse := sparse
  base := base
  sound challenge candidate :=
    (flatFold_satisfies β st challenge μ fits small batch rowsFit candidate).mono
      fun _ implies passed => by
        obtain ⟨r2, v, ended, opened⟩ := passed
        exact implies r2 v ended opened
  staged challenge candidate :=
    foldCheck_staged (flatReduction β st challenge μ fits batch rowsFit) S.certifiedBase
      (O.binarySupport fold) candidate.flat varsFit batchFit
  nonnegative := foldCosts_nonnegative _ _ _ _
  draws := by simp [packingFold, foldCosts]

/-- An output of the extractor for one packing fold: the verifier accepts the transcript of the
prover, and the output is a weak opening of the statement with `κ̄_1 = 2ω` and `β̄_∞ = Δ^cert_f`
(Akita Definition 10.22), or a solution of Module-SIS for `B` or `D` with the bound `b⋆ − 1` or for
a compression map with the bound 1 (Akita Corollary 10.32). -/
theorem packingFold_extract_sound [Nonempty Coord] {peak : ℕ}
    (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (prover : OracleComp
      (packingFold β st μ fits batch rowsFit block decode witnessOf read weight).Stage
      (Coord → E) (ℕ → Wire)) (budget : ℕ)
    (table : (packingFold β st μ fits batch rowsFit block decode witnessOf read weight).Stage →
      Coord → E)
    (tape : Tape (packingFold β st μ fits batch rowsFit block decode witnessOf read weight).Stage
      Coord E budget
      (packingFold β st μ fits batch rowsFit block decode witnessOf read
        weight).extraction.protocol.stages)
    (out : (packingFold β st μ fits batch rowsFit block decode witnessOf read weight).Cert ⊕
      Option (OuterBreak st.key ⊕ OpenBreak st.openKey))
    (result : Checkpoint.extract
      (packingFold β st μ fits batch rowsFit block decode witnessOf read weight).extraction prover
      budget table tape = some out) :
    (packingFold β st μ fits batch rowsFit block decode witnessOf read weight).Accepts
        ((packingFold β st μ fits batch rowsFit block decode witnessOf read
          weight).extraction.protocol.final prover table) ∧
      Sum.elim
        (fun certificate => ∃ ω, certificate.opening = some ω ∧
          ω.Opens β (2 * weight) (fold.sourceBound S) st (S.decode ω.source))
        (fun solution => solution.isSome) out :=
  (packingFold β st μ fits batch rowsFit block decode witnessOf read weight).extract_sound
    (packingFold_valid injective units sparse base small varsFit batchFit) prover budget table
    tape out result

variable [Finite Wire]

/-- Akita Theorem 10.17 for one packing fold, with the reduction of Akita Corollary 10.32 at the
fold challenge. For each prover with at most `Q` queries to the oracle, the probability that the
verifier accepts and the extractor returns no output is at most
`(Q + 1)·(r·p + ε_fold) + (Q + 1)·(r + 1)/budget`. `r` is the number of blocks, `p` is the largest
probability of one fold challenge (`Level.chance`), `ε_fold` is `foldError` of the quotient-free
reduction, which is the same for each `challenge`, and `budget` is the limit of the attempts of a
retry loop of the extractor. By `packingFold_extract_sound`, an output is a weak opening or a
solution of Module-SIS. -/
theorem packingFold_extraction_le [Nonempty Coord] {peak : ℕ}
    (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (challenge : Fin S.blocks → Rq O.carrierDegree)
    (prover : OracleComp
      (packingFold β st μ fits batch rowsFit block decode witnessOf read weight).Stage
      (Coord → E) (ℕ → Wire)) {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ}
    (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape
        (packingFold β st μ fits batch rowsFit block decode witnessOf read weight).Stage Coord E
        budget (packingFold β st μ fits batch rowsFit block decode witnessOf read
          weight).extraction.protocol.stages,
        (if (packingFold β st μ fits batch rowsFit block decode witnessOf read weight).Accepts
            ((packingFold β st μ fits batch rowsFit block decode witnessOf read
              weight).extraction.protocol.final prover table) ∧
          Checkpoint.extract
            (packingFold β st μ fits batch rowsFit block decode witnessOf read weight).extraction
            prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (S.blocks *
          (packingFold β st μ fits batch rowsFit block decode witnessOf read weight).chance +
          foldError (flatReduction β st challenge μ fits batch rowsFit) S.certifiedBase) +
        (queries + 1) * (S.blocks + 1) / budget := by
  rw [← foldCosts_sum]
  exact (packingFold β st μ fits batch rowsFit block decode witnessOf read weight).extraction_le
    (packingFold_valid injective units sparse base small varsFit batchFit) prover bounded positive

/-- The expected number of runs of the prover in the extraction of one packing fold is at most
`K̂ + (Q + 1)·(K̂ − 1)` with `K̂ = 2·(r + 1)` for `r` blocks (Akita Theorem 10.17, the cost). -/
theorem packingFold_runs_le
    (prover : OracleComp
      (packingFold β st μ fits batch rowsFit block decode witnessOf read weight).Stage
      (Coord → E) (ℕ → Wire)) {queries : ℕ} (bounded : prover.QueryBound queries) (budget : ℕ) :
    𝔼 table, 𝔼 tape : Tape
        (packingFold β st μ fits batch rowsFit block decode witnessOf read weight).Stage Coord E
        budget (packingFold β st μ fits batch rowsFit block decode witnessOf read
          weight).extraction.protocol.stages,
        (extractRuns
          (packingFold β st μ fits batch rowsFit block decode witnessOf read weight).extraction
          prover budget table tape : ℝ) ≤
      2 * (S.blocks + 1) + (queries + 1) * (2 * (S.blocks + 1) - 1) :=
  (packingFold β st μ fits batch rowsFit block decode witnessOf read weight).runs_le
    (by simp [packingFold, foldCosts]) prover bounded budget

end Fold

end NightstreamFPrime.Spec.Compression.Akita.Extract
