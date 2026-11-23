import NightstreamFPrime.Spec.Compression.Akita.NormCheck
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceRoute

/-! A committed trace level of the opening suffix of Akita on the Euclidean route, with the direct
norm check in the checks of the fold (ePrint 2026/1983, revision of 2026-11-23, Sections 6.2 and
6.3 with (81) to (93), Lemma 6.3, Lemma 10.34 with (184), Lemma 10.37 and Proposition 10.44).

- `Shape.normKernel` is the kernel `K(u, x)` of the integer coordinates `z^Z_u` of (81) in the
  flat witness of a trace level, and `normTable_normKernel` states that `normTable` with it is the
  sequence of the coefficients of `z = G_{b_z,m_A}·ẑ`. `sum_sq_cellsOf` gives the sum (85) of
  Lemma 6.3 from the sum of the cube, and `Shape.directLimit` is `U_dir` of (84) for the plane
  bounds `b⋆/2` of the digit range check.
- `foldCostsSq` has the bounds of the draws of `foldCheckSq`, and `foldCheckSq_stagedTo` and
  `foldCheckSq_bare` are its bounds and its verifier without the witness.
- `committedTraceSq` is the `Tier` of a committed trace fold on the Euclidean route: the method
  `traceSq` and the checks `foldCheckSq`. `committedTraceSq_sound` is the soundness of its checks
  with the no-wraparound condition `max{U_dir, S_max} < q` of Lemma 6.3,
  `committedTraceSq_valid` states the conditions of `Tier.Valid`, and `committedTraceSq_uniform`
  states that its verifier does not read the table. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial
open NightstreamFPrime.Spec.Compression (Game)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

attribute [local instance low] Classical.propDecidable

section Kernel

variable {d m : ℕ} [NeZero d]

/-- The sum of the squares of the cells of the cube of `ν` variables is the sum of the squares of
the coefficients of the vector, when the cube has room for its `m·d` coefficients. -/
theorem sum_sq_cellsOf {ι : Type} [Fintype ι] (code : ι ≃ Fin m) (v : ι → Rq d) {ν : ℕ}
    (room : m * d ≤ 2 ^ ν) :
    ∑ u ∈ Finset.range (2 ^ ν), cellsOf code v u ^ 2 =
      ∑ j, ∑ i ∈ Finset.range d, coeff (v j) i ^ 2 := by
  rw [← Finset.sum_range_add_sum_Ico _ room, Finset.sum_eq_zero (s := Finset.Ico _ _)
    fun u member => by rw [cellsOf, sequence_of_le _ (Finset.mem_Ico.mp member).1]; ring,
    add_zero, Finset.sum_range, sum_fin_mul m d fun t => cellsOf code v t ^ 2]
  refine Eq.trans ?_ (code.symm.sum_comp fun j => ∑ i ∈ Finset.range d, coeff (v j) i ^ 2)
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [Finset.sum_range]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [show (k : ℕ) + d * c = c * d + k by ring, cellsOf, sequence_apply _ _ k.isLt]

variable (S : Shape) (fold : Descriptor)

/-- The index `δ_com·x + u_com` of the coordinate `(x, u_com)` of the recomposed response. -/
def Shape.sourceCode : Fin S.blockLength × Fin S.sourceDigits ≃
    Fin (S.blockLength * S.sourceDigits) :=
  finProdFinEquiv

/-- The kernel `K(u, x)` of the integer coordinates of Akita (81) in the flat witness: for the
coefficient `k = u mod d_A` of the coordinate `u / d_A` of `z`, the cell of digit `a` of the
response has the factor `b_z^a`. -/
noncomputable def Shape.normKernel (u x : ℕ) : Zq :=
  if inside : u / S.innerDegree < S.blockLength * S.sourceDigits then
    ∑ a : Fin fold.responseDigits,
      if x = S.responseCode fold (S.sourceCode.symm ⟨u / S.innerDegree, inside⟩, a) *
          S.innerDegree + u % S.innerDegree
      then (fold.responseBase : Zq) ^ (a : ℕ) else 0
  else 0

/-- `U_dir = N_A·(Σ_{a<δ_f} b_z^a·b⋆/2)²` of Akita (84) for a trace level, with the plane bounds
`B_dig,a = b⋆/2` of the digit range check. -/
def Shape.directLimit : ℕ :=
  directBound fold.responseBase (fun _ : Fin fold.responseDigits => S.certifiedBase / 2)
    (Fintype.card (Fin S.blockLength × Fin S.sourceDigits) * S.innerDegree)

variable {S fold}

/-- The cells of `ẑ` are the first cells of the flat witness. -/
theorem OpeningShape.responseCells_le_cells (O : OpeningShape S) (fold : Descriptor) :
    S.responseCells fold ≤ O.cells fold := by
  unfold OpeningShape.cells OpeningShape.openSecondStart OpeningShape.secondStart
    OpeningShape.openFirstStart OpeningShape.chainStart OpeningShape.innerStart
  omega

/-- The integer coordinates of a trace level are the coefficients of `z = G_{b_z,m_A}·ẑ` for the
response digits `ẑ` of the table, in the order of `Shape.sourceCode`, when the cube of `μ`
variables has the cells of `ẑ` (Akita (81), in `Z_q`). -/
theorem normTable_normKernel {μ : ℕ} (room : S.responseCells fold ≤ 2 ^ μ) (table : Table) :
    normTable μ (S.normKernel fold) table = cellsOf S.sourceCode
      (gadget fold.responseBase (vectorOf (S.responseCode fold) table : S.Response fold)) := by
  funext u
  have small : u % S.innerDegree < S.innerDegree := Nat.mod_lt _ (NeZero.pos _)
  unfold normTable Shape.normKernel cellsOf sequence
  by_cases inside : u / S.innerDegree < S.blockLength * S.sourceDigits
  · simp only [inside, ↓reduceDIte, gadget, coeff_sum, coeff_natCast_pow_mul, mul_ite, mul_zero,
      Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun a _ => ?_
    rw [Finset.sum_ite_eq', ite_eq_left (Finset.mem_range.mpr ((cell_lt _ small).trans_le room)),
      vectorOf, coeff_ofCoeffs _ small, mul_comm]
  · simp only [inside, ↓reduceDIte, mul_zero, Finset.sum_const_zero]

end Kernel

namespace Extract

section Check

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]

/-- The bounds of the draws of the range tree on the Euclidean route: `treeCosts`, the merge
coefficient `ζ_norm`, and the `μ'` rounds of the last stage with the binary-support point
(Akita (187), (90) and Lemma 10.34). -/
noncomputable def rangeCostsSq (E : Type) [Fintype E] (μ b : ℕ) : List ℝ :=
  treeCosts E μ (rangeArities b) 1 ++ (((1 : ℕ) : ℝ) / Fintype.card E) ::
    List.replicate μ (((max (rangeLeaf b + 1) 2 + 1 : ℕ) : ℝ) / Fintype.card E)

/-- The sum of the numerators of `rangeCostsSq`. -/
def rangeNumeratorSq (μ b : ℕ) : ℕ :=
  treeNumerator μ (rangeArities b) 1 + 1 + μ * (max (rangeLeaf b + 1) 2 + 1)

/-- The bounds of the draws of `foldCheckSq`, in the order of the draws: `α`, `τ_0`, `τ_1`, the
draws of the range tree with `ζ_norm`, then `η_norm`, `ζ_bin` at a level with compression
digits, `γ` and the rounds of the fused sum-check (Akita (189) with `N^norm = 2`). -/
noncomputable def foldCostsSq (E : Type) [Fintype E] (badBound vars batch base : ℕ)
    (bin : Bool) : List ℝ :=
  ((badBound : ℝ) / Fintype.card E) :: ((vars : ℝ) / Fintype.card E) ::
    ((batch : ℝ) / Fintype.card E) ::
      (rangeCostsSq E vars base ++
        (((1 : ℕ) : ℝ) / Fintype.card E) :: (binCosts E bin ++
          (((1 : ℕ) : ℝ) / Fintype.card E) ::
            List.replicate vars (((3 : ℕ) : ℝ) / Fintype.card E)))

omit [Field E] [DecidableEq E] [Algebra Zq E] in
/-- The sum of the bounds of the draws of `foldCheckSq`. It is the sum of `foldCosts` plus
`2/|E|` at a base with leaf polynomials of one factor or more. -/
theorem foldCostsSq_sum (badBound vars batch base : ℕ) (bin : Bool) :
    (foldCostsSq E badBound vars batch base bin).sum =
      ((badBound + vars + batch + rangeNumeratorSq vars base + 1 + binDraws bin + 1 +
        vars * 3 : ℕ) : ℝ) / Fintype.card E := by
  simp only [foldCostsSq, rangeCostsSq, rangeNumeratorSq, List.sum_cons, List.sum_append,
    List.sum_replicate, nsmul_eq_mul, treeCosts_sum, binCosts_sum]
  push_cast
  ring

/-- The losses of the sites of the draws of `foldCheckSq` in the Akita code, in the order of
`foldCostsSq`: `RingSwitchAlpha`, `Tau0Point`, `Tau1Point`, the Stage 1 sites of the product levels,
`L2NormMerge` (1), the `μ'` rounds of the norm sum-check `PhysicalL2` (`leafDegree + 1`),
`L2VirtualBatch` (1 for the direct norm proof), `CompressionBinary`, `Stage2Batch` (1) and the
Stage 2 rounds (3) (`transcript_grinding/plan.rs`, `append_nonterminal`). -/
def foldLossesSq (alpha vars batch base : ℕ) (bin : Bool) : List ℕ :=
  alpha :: vars :: batch ::
    ((treeLosses vars (rangeArities base) 1 ++ 1 :: List.replicate vars (leafDegree base + 1)) ++
      1 :: (binLosses bin ++ 1 :: List.replicate vars 3))

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem foldLossesSq_length (alpha badBound vars batch base : ℕ) (bin : Bool) :
    (foldLossesSq alpha vars batch base bin).length =
      (foldCostsSq E badBound vars batch base bin).length := by
  simp only [foldLossesSq, foldCostsSq, rangeCostsSq, List.length_cons, List.length_append,
    List.length_replicate, treeLosses_length (E := E), binLosses_length, binCosts_length]

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem foldCostsSq_nonnegative (badBound vars batch base : ℕ) (bin : Bool) :
    ∀ cost ∈ foldCostsSq E badBound vars batch base bin, 0 ≤ cost := by
  intro cost member
  simp only [foldCostsSq, rangeCostsSq, List.mem_cons, List.mem_append, List.mem_replicate]
    at member
  rcases member with rfl | rfl | rfl | (tree | rfl | ⟨_, rfl⟩) | rfl | binned | rfl | ⟨_, rfl⟩
  · positivity
  · positivity
  · positivity
  · exact treeCosts_nonnegative vars _ _ cost tree
  · positivity
  · positivity
  · positivity
  · exact binCosts_nonnegative bin cost binned
  · positivity
  · positivity

/-- Each draw of `foldCheckSq` has its bound of `foldCostsSq`, and an accepted end has the
entries `rest` after these bounds, when an answer of the oracle has `μ'` slots or more and
`s_rel` slots or more. -/
theorem foldCheckSq_stagedTo {Coord : Type} [Fintype Coord] [DecidableEq Coord] [Nonempty Coord]
    (R : Reduction E) (base : ℕ) (support : ℕ → Prop) [DecidablePred support] (bin : Bool)
    (limit : ℕ) (kernel : ℕ → ℕ → Zq) (w : Table) (varsFit : R.vars ≤ Fintype.card Coord)
    (batchFit : R.batch ≤ Fintype.card Coord) (rest : List ℝ)
    (tail : Option (List E × E) → List ℝ → Prop) (rejected : ∀ costs, tail none costs)
    (ended : ∀ final, tail (some final) rest) :
    StagedTo (Coord → E) (foldCheckSq R base support bin limit kernel w)
      (foldCostsSq E R.badBound R.vars R.batch base bin ++ rest) tail := by
  have field : ∃ decode : (Coord → E) → E, Fair decode := exists_fair_slot
  have line (a b : E) : ((zeros (linePoly a b)).card : ℝ) / Fintype.card E ≤
      ((1 : ℕ) : ℝ) / Fintype.card E :=
    density_le _ 1 (zeros_card_le_of_natDegree_le (linePoly_natDegree _ _))
  unfold foldCheckSq foldCostsSq rangeCostsSq
  simp only [List.cons_append, List.append_assoc]
  refine stagedTo_draw field (density_le (C := E) (R.bad w) _ (R.bad_card w)) fun α => ?_
  refine stagedTo_draw (exists_fair_slots varsFit) (density_rangeBad_le (E := E) R.low base R.vars w)
    fun τ0 => ?_
  refine stagedTo_draw (exists_fair_slots batchFit)
    ((R.rows α).density_batchBad_le R.vars (R.fits α) w) fun τ1 => ?_
  refine (stagedTo_bind _ _ _ _).mpr (rangeTree_stagedTo (rangeLeaf base) R.vars w
    (normLeafStage (rangeLeaf base) R.vars limit kernel support w) _ _
    (fun _ => rejected _) (fun nodes weights point claim => ?_) _ _ _ _ _)
  intro claimed
  dsimp only
  split
  swap
  · exact rejected _
  refine stagedTo_draw field (line _ _) fun ζn => ?_
  refine (stagedTo_bind _ _ _ _).mpr (sumcheckWith_staged (Reply := Coord → E) field
    (max (rangeLeaf base + 1) 2) R.vars 1 _
    (normLeaf_roundTruth (rangeLeaf base) nodes R.vars weights ζn kernel point w).degree _
    (fun _ => bindBad_card_le _ _) _ _ ?_ ?_ R.vars [] _ (by simp))
  · exact fun _ => rejected _
  rintro ⟨r, value⟩ values
  dsimp only
  split
  · exact stagedTo_draw field (line _ _) fun η =>
      drawBin_stagedTo bin (zeros_card_le_of_natDegree_le (linePoly_natDegree _ _)) fun ζ =>
      stagedTo_draw field (line _ _) fun γ =>
        fusedCheck_stagedTo R.vars support w _ _ r values.2 ζ γ rest tail rejected ended
  · exact rejected _

/-- The verifier of the last stage of the range tree on the Euclidean route is the same for each
witness. -/
theorem normLeafStage_bare (leaf μ limit : ℕ) (kernel : ℕ → ℕ → Zq) (support : ℕ → Prop)
    [DecidablePred support] (w other : Table) (nodes : ℕ) (weights : ℕ → E) (point : List E)
    (claim : E) :
    bare (normLeafStage leaf μ limit kernel support w nodes weights point claim) =
      bare (normLeafStage leaf μ limit kernel support other nodes weights point claim) := by
  unfold normLeafStage
  refine bare_recv_congr fun claimed => ?_
  by_cases small : claimed ≤ limit
  swap
  · rw [ite_eq_right small, ite_eq_right small]
  rw [ite_eq_left small, ite_eq_left small]
  refine bare_draw_congr fun ζ => bare_bind_congr (sumcheckWith_bare _ _ _ _ _ μ [] _)
    fun result => ?_
  rcases result with _ | ⟨r, value⟩ <;> rfl

/-- The verifier of `foldCheckSq` is the same for each witness. -/
theorem foldCheckSq_bare (R : Reduction E) (b : ℕ) (support : ℕ → Prop) [DecidablePred support]
    (bin : Bool) (limit : ℕ) (kernel : ℕ → ℕ → Zq) (w other : Table) :
    bare (foldCheckSq R b support bin limit kernel w) =
      bare (foldCheckSq R b support bin limit kernel other) := by
  unfold foldCheckSq
  refine bare_draw_congr fun α => bare_draw_congr fun τ0 => bare_draw_congr fun τ1 =>
    bare_bind_congr (rangeTree_bare (rangeLeaf b) R.vars w other _ _
      (fun nodes weights point claim => normLeafStage_bare (rangeLeaf b) R.vars limit kernel
        support w other nodes weights point claim) _ _ _ _ _) fun result => ?_
  rcases result with _ | ⟨r, sClaim, y⟩
  · rfl
  · exact bare_draw_congr fun η => drawBin_bare_congr bin fun ζ => bare_draw_congr fun γ =>
      bare_bind_congr (sumcheck_bare 3 _ _ R.vars [] _) fun _ => rfl

end Check

section Level

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E] [Fintype E]
  [DecidableEq E] [Algebra Zq E] {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- A trace fold for one claim on the Euclidean route whose flat witness the next level commits
to, as a `Tier` with the first stage `start`. The method is `traceSq` with the operator bound `Γ`
and the cap `S_max`. The checks are `foldCheckSq` for the quotient-free reduction of
`traceRelation`, with the kernel `Shape.normKernel` of the integer coordinates of the
response. A final claim `(r_2, v_W)` is true for a table `w` when `v_W = w̃(r_2)`. -/
noncomputable def committedTraceSq (key : Key S) (openKey : OpeningKey O)
    (claim : Wire → TraceClaim O E) (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch) (block : Fin S.blocks → Coord)
    (decode : E → Option (Rq S.innerDegree)) (read : Reader Wire)
    (weight operator limit start : ℕ) :
    Tier (Fin S.blocks) (fun _ => Rq S.innerDegree) (TraceClaim O E)
      (LevelBreakSq key openKey (64 * operator ^ 2 * limit)) (Witness O fold) (Weak O)
        Wire Coord E
      (Option (List E × E)) Table where
  method := traceSq key openKey weight operator limit
  claim := claim
  block b := some (block b)
  decode := fun _ => decode
  read := read
  digits := O.ofFlat fold
  check wire challenge table :=
    foldCheckSq (traceReduction ((claim wire).statement key openKey) challenge μ fits batch
      rowsFit) S.certifiedBase (O.binarySupport fold) (O.compressed fold) limit (S.normKernel fold)
        table
  passes table result := ∃ r2 v, result = some (r2, v) ∧ v = mle r2 (lifted E table)
  costs := foldCostsSq E (O.largestDegree - 1) μ batch S.certifiedBase (O.compressed fold)
  bits := (foldLossesSq (2 * S.innerDegree - 1) μ batch S.certifiedBase (O.compressed fold)).map
    workBits
  foldBits := foldWorkBits S.blocks (setBits true S.innerDegree)
  start := start

variable {key : Key S} {openKey : OpeningKey O} {claim : Wire → TraceClaim O E} {μ : ℕ}
  {fits : O.cells fold ≤ 2 ^ μ} {batch : ℕ} {rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch}
  {block : Fin S.blocks → Coord} {decode : E → Option (Rq S.innerDegree)}
  {read : Reader Wire} {weight operator limit start : ℕ}

omit [Fintype Coord] [DecidableEq Coord] in
/-- Akita Lemma 10.42 (i) on the Euclidean route: on a path of the checks of a committed trace
fold that avoids the bad sets, a final claim that is true for the table gives the exact branch
relation of `traceSq` for the digit vectors `ofFlat` of the table: `Witness.TraceSatisfies`, and
`‖G_{b_z,m_A}·ẑ‖_2² ≤ S_max` from the sum (85) with Akita Lemma 6.3. The cube of the `μ'`
variables of the witness has room for the `N_A` coordinates of `z`, and
`max{U_dir, S_max} < q` for `U_dir` of `Shape.directLimit`. -/
theorem committedTraceSq_sound (small : S.certifiedBase < fieldModulus)
    (room : S.blockLength * S.sourceDigits * S.innerDegree ≤ 2 ^ μ)
    (noWrap : max (S.directLimit fold) limit < fieldModulus)
    (wire : Wire) (challenge : Fin S.blocks → Rq S.innerDegree) (table : Table) :
    ((committedTraceSq key openKey claim μ fits batch rowsFit block decode read weight operator
      limit start).check wire challenge table).Avoids fun result =>
      (committedTraceSq key openKey claim μ fits batch rowsFit block decode read weight
        operator limit start).passes table result →
      (committedTraceSq key openKey claim μ fits batch rowsFit block decode read weight
        operator limit start).method.Satisfies (O.ofFlat fold table) (claim wire) challenge := by
  have cells : S.responseCells fold ≤ 2 ^ μ :=
    (O.responseCells_le_cells fold).trans fits
  refine (foldCheckSq_avoids (traceReduction ((claim wire).statement key openKey) challenge
    μ fits batch rowsFit) S.certifiedBase (O.binarySupport fold) (O.compressed fold)
      (O.binarySupport_empty fold) limit (S.normKernel fold)
    table).mono fun _ implies ⟨r2, v, ended, opening⟩ => ?_
  obtain ⟨checked, claimed, accepted, energy⟩ := implies r2 v ended opening
  have satisfies := traceSatisfies_of_table ((claim wire).statement key openKey) challenge μ fits
    small batch rowsFit checked.1 checked.2.1 checked.2.2
  have energy : ∑ u ∈ Finset.range (2 ^ μ), normTable μ (S.normKernel fold) table u ^ 2 =
      (claimed : Zq) := energy
  rw [normTable_normKernel cells, sum_sq_cellsOf S.sourceCode _ room] at energy
  exact ⟨satisfies, vecNormSq_gadget_le_of_direct
    (PlaneBounded.of_balanced satisfies.responseDigits) noWrap accepted energy⟩

variable [Nonempty Coord]

/-- Each draw of the checks of a committed trace fold on the Euclidean route has its bound of
`costs`, and an accepted end has the entries `rest` after these bounds, when an answer of the
oracle has `μ'` slots or more and `s_rel` slots or more. -/
theorem committedTraceSq_stagedTo (varsFit : μ ≤ Fintype.card Coord)
    (batchFit : batch ≤ Fintype.card Coord) (wire : Wire)
    (challenge : Fin S.blocks → Rq S.innerDegree) (table : Table) (rest : List ℝ)
    (tail : Option (List E × E) → List ℝ → Prop) (rejected : ∀ costs, tail none costs)
    (ended : ∀ final, tail (some final) rest) :
    StagedTo (Coord → E) ((committedTraceSq key openKey claim μ fits batch rowsFit block decode
      read weight operator limit start).check wire challenge table)
      ((committedTraceSq key openKey claim μ fits batch rowsFit block decode read weight
        operator limit start).costs ++ rest) tail :=
  foldCheckSq_stagedTo _ _ _ _ limit _ table varsFit batchFit rest tail rejected ended

/-- A committed trace fold on the Euclidean route satisfies the conditions of `Tier.Valid`. The
hypotheses are those of `committedTrace_valid`, the operator bound `‖c‖_mul,2 ≤ Γ` of each
challenge of the family (`bounded`), and the conditions `room` and `noWrap` of
`committedTraceSq_sound`. -/
theorem committedTraceSq_valid {peak : ℕ} (injective : Function.Injective block)
    (units : UnitDifferences {challenge | ∃ slot, decode slot = some challenge})
    (sparse : ∀ challenge, (∃ slot, decode slot = some challenge) → Sparse weight peak challenge)
    (bounded : ∀ challenge, (∃ slot, decode slot = some challenge) →
      MulBound operator challenge)
    (base : 2 ≤ fold.responseBase) (small : S.certifiedBase < fieldModulus)
    (varsFit : μ ≤ Fintype.card Coord) (batchFit : batch ≤ Fintype.card Coord)
    (room : S.blockLength * S.sourceDigits * S.innerDegree ≤ 2 ^ μ)
    (noWrap : max (S.directLimit fold) limit < fieldModulus) :
    (committedTraceSq key openKey claim μ fits batch rowsFit block decode read weight operator
      limit start).Valid where
  injective := (Option.some_injective _).comp injective
  avoid _ := Option.some_ne_none _
  merges := traceSq_sound units sparse bounded base
  sound := committedTraceSq_sound small room noWrap
  staged wire challenge table := by
    have staged := committedTraceSq_stagedTo (key := key) (openKey := openKey) (claim := claim)
      (fits := fits) (rowsFit := rowsFit) (block := block) (decode := decode)
      (read := read) (weight := weight) (operator := operator) (limit := limit) (start := start)
      varsFit batchFit wire challenge table [] (fun _ _ => True) (fun _ => trivial)
      (fun _ => trivial)
    rwa [List.append_nil] at staged
  nonnegative cost member := foldCostsSq_nonnegative _ _ _ _ _ cost member
  draws := by simp [committedTraceSq, foldCostsSq]
  aligned := (List.length_map ..).trans (foldLossesSq_length ..)

omit [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] in
/-- The verifier of the checks of a committed trace fold on the Euclidean route does not read the
table. -/
theorem committedTraceSq_uniform (wire : Wire) (challenge : Fin S.blocks → Rq S.innerDegree)
    (table other : Table) :
    bare ((committedTraceSq key openKey claim μ fits batch rowsFit block decode read weight
        operator limit start).check wire challenge table) =
      bare ((committedTraceSq key openKey claim μ fits batch rowsFit block decode read weight
        operator limit start).check wire challenge other) :=
  foldCheckSq_bare _ _ _ _ limit _ table other

end Level

end Extract

end NightstreamFPrime.Spec.Compression.Akita
