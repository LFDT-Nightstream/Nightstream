import NightstreamFPrime.Spec.Compression.Akita.NormCheck
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceRoute

/-! A committed trace level of the opening suffix of Akita on the Euclidean route, with the direct
norm check after the checks of the fold (ePrint 2026/1983, revision of 2026-10-15, Section 6.2
with (81) to (85), Lemma 6.3, Lemma 10.34 with (184), Lemma 10.37 and Proposition 10.44).

- `normCosts` has the bounds of the draws of `normCheck`, and `normCheck_stagedTo` and
  `normCheck_bare` are its bounds and its verifier without the witness.
- `Shape.normKernel` is the kernel `K(u, x)` of the integer coordinates `z^Z_u` of (81) in the
  flat witness of a trace level, and `normTable_normKernel` states that `normTable` with it is the
  sequence of the coefficients of `z = G_{b_z,m_A}·ẑ`. `sum_sq_cellsOf` gives the sum (85) of
  Lemma 6.3 from the sum of the cube, and `Shape.directLimit` is `U_dir` of (84) for the plane
  bounds `b⋆/2` of the digit range check.
- `committedTraceSq` is the `Tier` of a committed trace fold on the Euclidean route: the method
  `traceSq` and the checks `foldCheck` followed by `normCheck`. `committedTraceSq_sound` is the
  soundness of its checks with the no-wraparound condition `max{U_dir, S_max} < q` of Lemma 6.3,
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

/-- The bounds of the draws of `normCheck`, in their order: the `ν` rounds of degree 2 of the
sum-check of (85), the binding coefficient `η`, and the `μ` rounds of degree 2 of the binding
sum-check. -/
noncomputable def normCosts (E : Type) [Fintype E] (ν μ : ℕ) : List ℝ :=
  List.replicate ν (((2 : ℕ) : ℝ) / Fintype.card E) ++
    (((1 : ℕ) : ℝ) / Fintype.card E) :: List.replicate μ (((2 : ℕ) : ℝ) / Fintype.card E)

omit [Field E] [DecidableEq E] [Algebra Zq E] in
/-- The sum of the bounds of the draws of `normCheck` is `(2·ν + 1 + 2·μ)/|E|`. -/
theorem normCosts_sum (ν μ : ℕ) :
    (normCosts E ν μ).sum = ((2 * ν + 1 + 2 * μ : ℕ) : ℝ) / Fintype.card E := by
  simp only [normCosts, List.sum_append, List.sum_cons, List.sum_replicate, nsmul_eq_mul]
  push_cast
  ring

omit [Field E] [DecidableEq E] [Algebra Zq E] in
theorem normCosts_nonnegative (ν μ : ℕ) : ∀ cost ∈ normCosts E ν μ, 0 ≤ cost := by
  intro cost member
  simp only [normCosts, List.mem_append, List.mem_cons, List.mem_replicate] at member
  rcases member with ⟨_, rfl⟩ | rfl | ⟨_, rfl⟩ <;> positivity

/-- Each draw of `normCheck` has its bound of `normCosts`, and an accepted end has the entries
`rest` after these bounds, when an answer of the oracle has a slot. -/
theorem normCheck_stagedTo {Coord : Type} [Fintype Coord] [DecidableEq Coord] [Nonempty Coord]
    (ν μ limit : ℕ) (kernel : ℕ → ℕ → Zq) (w : Table) (r2 : List E) (v : E) (rest : List ℝ)
    (tail : Option (List E × E) → List ℝ → Prop) (rejected : ∀ costs, tail none costs)
    (ended : ∀ final, tail (some final) rest) :
    StagedTo (Coord → E) (normCheck ν μ limit kernel w r2 v) (normCosts E ν μ ++ rest) tail := by
  have field : ∃ decode : (Coord → E) → E, Fair decode := exists_fair_slot
  unfold normCheck normCosts
  split
  swap
  · exact rejected _
  intro claimed
  dsimp only
  split
  swap
  · exact rejected _
  rw [List.append_assoc, List.cons_append]
  refine (stagedTo_bind _ _ _ _).mpr (sumcheck_staged field 2 ν _
    (combRound_truth ν 2 (normTables μ kernel w) productComb productComb_degree).degree _ _ ?_ ?_
    ν [] _ (by simp))
  · exact fun _ => rejected _
  rintro ⟨ru, value⟩ y
  dsimp only
  split
  swap
  · exact rejected _
  refine stagedTo_draw field (density_le _ 1
    (zeros_card_le_of_natDegree_le (linePoly_natDegree _ _))) fun η => ?_
  refine (stagedTo_bind _ _ _ _).mpr (sumcheck_staged field 2 μ _
    (combRound_truth μ 2 (bindTables ν kernel w r2 ru η) productComb productComb_degree).degree
    rest _ ?_ ?_ μ [] _ (by simp))
  · exact fun _ => rejected _
  rintro ⟨r3, final⟩ v3
  dsimp only
  split
  · exact ended _
  · exact rejected _

/-- The verifier of `normCheck` is the same for each witness. -/
theorem normCheck_bare (ν μ limit : ℕ) (kernel : ℕ → ℕ → Zq) (w other : Table) (r2 : List E)
    (v : E) :
    bare (normCheck ν μ limit kernel w r2 v) = bare (normCheck ν μ limit kernel other r2 v) := by
  unfold normCheck
  by_cases length : r2.length = μ
  swap
  · rw [ite_eq_right length, ite_eq_right length]
  rw [ite_eq_left length, ite_eq_left length]
  refine bare_recv_congr fun claimed => ?_
  by_cases small : claimed ≤ limit
  swap
  · rw [ite_eq_right small, ite_eq_right small]
  rw [ite_eq_left small, ite_eq_left small]
  refine bare_bind_congr (sumcheck_bare 2 _ _ ν [] _) fun result => ?_
  rcases result with _ | ⟨ru, value⟩
  · rfl
  refine bare_recv_congr fun y => ?_
  by_cases squared : value = y * y
  swap
  · rw [ite_eq_right squared, ite_eq_right squared]
  rw [ite_eq_left squared, ite_eq_left squared]
  exact bare_draw_congr fun η =>
    bare_bind_congr (sumcheck_bare 2 _ _ μ [] _) fun _ => rfl

end Check

section Level

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E] [Fintype E]
  [DecidableEq E] [Algebra Zq E] {Wire Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- A trace fold for one claim on the Euclidean route whose flat witness the next level commits
to, as a `Tier` with the first stage `start`. The method is `traceSq` with the operator bound `Γ`
and the cap `S_max`. The checks are `foldCheck` for the quotient-free reduction of
`lockedTraceRelation`, and then `normCheck` with the kernel `Shape.normKernel` on `ν` variables
for the claim `(r_2, v_W)` of `foldCheck`. A final claim `(r_3, v)` is true for a table `w` when
`v = w̃(r_3)`. -/
noncomputable def committedTraceSq (key : Key S) (openKey : OpeningKey O)
    (claim : Wire → TraceClaim O E) (μ : ℕ) (fits : O.cells fold < 2 ^ μ) (batch : ℕ)
    (rowsFit : O.traceRowCount + 1 + 1 ≤ 2 ^ batch) (ν : ℕ) (block : Fin S.blocks → Coord)
    (decode : E → Option (Rq S.innerDegree)) (read : Reader Wire)
    (weight operator limit start : ℕ) :
    Tier (Fin S.blocks) (fun _ => Rq S.innerDegree) (TraceClaim O E)
      (LevelBreakSq key openKey (64 * operator ^ 2 * limit)) (Witness O fold) (Weak O)
        Wire Coord E
      (Option (List E × E)) Table where
  method := traceSq key openKey weight operator limit
  claim := claim
  block := block
  decode := fun _ => decode
  read := read
  digits := O.ofFlat fold
  check wire challenge table :=
    (foldCheck (lockedTraceReduction ((claim wire).statement key openKey) challenge μ fits batch
      rowsFit) S.certifiedBase (O.binarySupport fold) table).bind fun
      | none => .pure none
      | some (r2, v) => normCheck ν μ limit (S.normKernel fold) table r2 v
  passes table result := ∃ r3 v, result = some (r3, v) ∧ v = mle r3 (lifted E table)
  costs := foldCosts E (O.largestDegree - 1) μ batch S.certifiedBase ++ normCosts E ν μ
  start := start

variable {key : Key S} {openKey : OpeningKey O} {claim : Wire → TraceClaim O E} {μ : ℕ}
  {fits : O.cells fold < 2 ^ μ} {batch : ℕ} {rowsFit : O.traceRowCount + 1 + 1 ≤ 2 ^ batch}
  {ν : ℕ} {block : Fin S.blocks → Coord} {decode : E → Option (Rq S.innerDegree)}
  {read : Reader Wire} {weight operator limit start : ℕ}

omit [Fintype Coord] [DecidableEq Coord] in
/-- Akita Lemma 10.42 (i) on the Euclidean route: on a path of the checks of a committed trace
fold that avoids the bad sets, a final claim that is true for the table gives the exact branch
relation of `traceSq` for the digit vectors `ofFlat` of the table: `Witness.TraceSatisfies` from
`foldCheck`, and `‖G_{b_z,m_A}·ẑ‖_2² ≤ S_max` from `normCheck` with Akita Lemma 6.3. The cube of
`ν` variables has room for the `N_A` coordinates of `z`, and `max{U_dir, S_max} < q` for `U_dir`
of `Shape.directLimit`. -/
theorem committedTraceSq_sound (small : S.certifiedBase < fieldModulus)
    (room : S.blockLength * S.sourceDigits * S.innerDegree ≤ 2 ^ ν)
    (noWrap : max (S.directLimit fold) limit < fieldModulus)
    (wire : Wire) (challenge : Fin S.blocks → Rq S.innerDegree) (table : Table) :
    ((committedTraceSq key openKey claim μ fits batch rowsFit ν block decode read weight operator
      limit start).check wire challenge table).Avoids fun result =>
      (committedTraceSq key openKey claim μ fits batch rowsFit ν block decode read weight
        operator limit start).passes table result →
      (committedTraceSq key openKey claim μ fits batch rowsFit ν block decode read weight
        operator limit start).method.Satisfies (O.ofFlat fold table) (claim wire) challenge := by
  have cells : S.responseCells fold ≤ 2 ^ μ :=
    Nat.le_of_lt (Nat.lt_of_le_of_lt (O.responseCells_le_cells fold) fits)
  refine (avoids_bind _ _ _).mpr ((lockedTraceFold_satisfies ((claim wire).statement key openKey)
    challenge μ fits small batch rowsFit table).mono fun result implies => ?_)
  rcases result with _ | ⟨r2, v⟩
  · rintro ⟨_, _, ended, _⟩
    simp at ended
  refine (normCheck_avoids ν μ limit (S.normKernel fold) table r2 v).mono
    fun _ opened ⟨r3, v3, ended, opening⟩ => ?_
  obtain ⟨first, claimed, accepted, energy⟩ := opened r3 v3 ended opening
  have satisfies := implies r2 v rfl first
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
    StagedTo (Coord → E) ((committedTraceSq key openKey claim μ fits batch rowsFit ν block decode
      read weight operator limit start).check wire challenge table)
      ((committedTraceSq key openKey claim μ fits batch rowsFit ν block decode read weight
        operator limit start).costs ++ rest) tail := by
  rw [show (committedTraceSq key openKey claim μ fits batch rowsFit ν block decode read weight
      operator limit start).costs ++ rest =
    foldCosts E (O.largestDegree - 1) μ batch S.certifiedBase ++ (normCosts E ν μ ++ rest) from
      List.append_assoc _ _ _]
  refine (stagedTo_bind _ _ _ _).mpr (foldCheck_stagedTo _ _ _ table varsFit batchFit _ _
    (fun _ => rejected _) fun final => ?_)
  exact normCheck_stagedTo ν μ limit _ table final.1 final.2 rest tail rejected ended

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
    (room : S.blockLength * S.sourceDigits * S.innerDegree ≤ 2 ^ ν)
    (noWrap : max (S.directLimit fold) limit < fieldModulus) :
    (committedTraceSq key openKey claim μ fits batch rowsFit ν block decode read weight operator
      limit start).Valid where
  injective := injective
  merges := traceSq_sound units sparse bounded base
  sound := committedTraceSq_sound small room noWrap
  staged wire challenge table := by
    have staged := committedTraceSq_stagedTo (key := key) (openKey := openKey) (claim := claim)
      (fits := fits) (rowsFit := rowsFit) (ν := ν) (block := block) (decode := decode)
      (read := read) (weight := weight) (operator := operator) (limit := limit) (start := start)
      varsFit batchFit wire challenge table [] (fun _ _ => True) (fun _ => trivial)
      (fun _ => trivial)
    rwa [List.append_nil] at staged
  nonnegative cost member := (List.mem_append.mp member).elim
    (foldCosts_nonnegative _ _ _ _ cost) (normCosts_nonnegative (E := E) ν μ cost)
  draws := by simp [committedTraceSq, foldCosts]

omit [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] in
/-- The verifier of the checks of a committed trace fold on the Euclidean route does not read the
table. -/
theorem committedTraceSq_uniform (wire : Wire) (challenge : Fin S.blocks → Rq S.innerDegree)
    (table other : Table) :
    bare ((committedTraceSq key openKey claim μ fits batch rowsFit ν block decode read weight
        operator limit start).check wire challenge table) =
      bare ((committedTraceSq key openKey claim μ fits batch rowsFit ν block decode read weight
        operator limit start).check wire challenge other) :=
  bare_bind_congr (foldCheck_bare _ _ _ table other) fun result => by
    rcases result with _ | ⟨r2, v⟩
    · rfl
    · exact normCheck_bare ν μ limit _ table other r2 v

end Level

end Extract

end NightstreamFPrime.Spec.Compression.Akita
