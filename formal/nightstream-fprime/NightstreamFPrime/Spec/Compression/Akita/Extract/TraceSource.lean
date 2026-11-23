import NightstreamFPrime.Spec.Compression.Akita.Extract.FlatSource

/-! The relation of one fold of Akita with the evaluation-trace opening method on the flat
witness, for one claim in one group (ePrint 2026/1983, revision of 2026-11-23, Section 7.1 with
(98), (101) and (103), Section 7.2 with Figure 5, Section 7.3 with (108) to (110), and the
hypothesis of Corollary 10.32).

A trace fold has the digit vectors of a packing fold, so the flat witness is `Witness.flat` and
the decoder of a committed table is `OpeningShape.ofFlat`.

- `traceEvaluationRow` is the row (103) in `R_{q,d_A}`, and `traceInnerRow` is a row of (104) with
  the challenges `c_i ∈ R_{q,d_A}`. `carriedRows` has the rows of Akita Figure 5 without a fold
  challenge: (105), (106) and (47) or (48). `traceScalarWeight` has the weights of the scalar
  row (101).
- `traceRelation` is the `FlatRelation` of one trace fold, and `traceRelation_holds_iff` states
  that `Witness.flat` satisfies it when the digit vectors satisfy the rows of
  `Witness.TraceSatisfies`, and only then. `Witness.traceSatisfies_of_flat` is the premise
  `encodes` of `foldCheck_traceSatisfies`, and `Witness.TraceSatisfies.flat` is the honest
  direction.
- `traceReduction` is the quotient-free reduction of `traceRelation`, and `traceFold_satisfies` is
  `foldCheck_traceSatisfies` for `Witness.flat`.
- `traceSatisfies_of_table` and `traceTableFold_satisfies` give the exact branch relation of the
  digit vectors `ofFlat` of a table that the next level commits to. The rows read only cells
  before `OpeningShape.cells`, so the table can have each value from this cell.

The quotient-lift route is not in this file. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E] [Algebra Zq E]

section Rows

variable (O) (fold)

/-- The left side `Σ_i c_i·e_i` of the row (103): the occurrence `(i, u)` has the multiplier
`c_i·b_1^u` and reads digit `u` of the `d_A` coordinates of partial `i` (Akita (66)). -/
noncomputable def tracePartialForm (challenge : Fin S.blocks → Rq S.innerDegree) :
    Form S.innerDegree where
  Occurrence := Fin S.blocks × Fin S.openingDigits
  multiplier entry := challenge entry.1 * (S.openingBase : Rq S.innerDegree) ^ (entry.2 : ℕ)
  address entry j := O.partialPosition fold entry.1 entry.2 j

/-- The right side `a^⊤·G_{b,M}·G_{b_z,m_A}·ẑ` of the row (103): the occurrence
`(x, u_com, u_fold)` has the multiplier `a_x·b^{u_com}·b_z^{u_fold}` and reads this element of
`ẑ`. -/
noncomputable def traceContractForm (subfield : E →+* Rq S.innerDegree)
    (position : Fin S.blockLength → E) : Form S.innerDegree where
  Occurrence := (Fin S.blockLength × Fin S.sourceDigits) × Fin fold.responseDigits
  multiplier col := subfield (position col.1.1) *
    ((S.sourceBase : Rq S.innerDegree) ^ (col.1.2 : ℕ) *
      (fold.responseBase : Rq S.innerDegree) ^ (col.2 : ℕ))
  address col := S.responseCell fold col

/-- The row (103) of Akita Figure 5, in `R_{q,d_A}`. -/
noncomputable def traceEvaluationRow (st : TraceStatement O E)
    (challenge : Fin S.blocks → Rq S.innerDegree) : RingRow :=
  ((tracePartialForm O fold challenge).sub
    (traceContractForm fold st.subfield st.position)).row 0

/-- Row `a` of `Σ_i c_i·G_{b_1,n_A}·t̂_i`: the occurrence `(i, u)` has the multiplier `c_i·b_1^u`
and reads digit `u` of row `a` of `t̂_i`. -/
noncomputable def traceImageForm (challenge : Fin S.blocks → Rq S.innerDegree)
    (a : Fin S.innerRank) : Form S.innerDegree where
  Occurrence := Fin S.blocks × Fin S.openingDigits
  multiplier entry := challenge entry.1 * (S.openingBase : Rq S.innerDegree) ^ (entry.2 : ℕ)
  address entry := O.innerCell fold entry.1 a entry.2

/-- Row `a` of the inner row (104) of Akita Figure 5 with `c^A_i = c_i`, in `R_{q,d_A}`. -/
noncomputable def traceInnerRow (key : Key S) (challenge : Fin S.blocks → Rq S.innerDegree)
    (a : Fin S.innerRank) : RingRow :=
  ((responseForm fold key a).sub (traceImageForm O fold challenge a)).row 0

/-- The weight of the cell `x` in the scalar row (101) of Akita Figure 5: the sum of
`θ·χ_blk(i)·b_1^u·T_{ρ_pack}(X^j)` over the digits `u` of the coordinates `j` of the partials `i`
that are at `x`. -/
noncomputable def traceScalarWeight (st : TraceStatement O E) (x : ℕ) : E :=
  ∑ entry : Fin S.blocks × Fin S.innerDegree × Fin S.openingDigits,
    if O.partialPosition fold entry.1 entry.2.2 entry.2.1 = some x then
      st.scale * (st.block entry.1 * ((S.openingBase : Zq) ^ (entry.2.2 : ℕ) •
        st.functional (AdjoinRoot.root (modulus S.innerDegree) ^ (entry.2.1 : ℕ))))
    else 0

/-- The number of the ring rows of one fold that have no fold challenge. -/
def OpeningShape.carriedCount : ℕ :=
  S.slices * S.outerRank + (1 * O.openRank +
    (firstRank S.chain + (firstRank O.chain + (secondRank S.chain + secondRank O.chain))))

/-- The ring rows of one fold that have no fold challenge, in the order of Akita Figure 5: the
rows (105) of `B` for each slice, the rows (106) of `D`, the rows of `F_1`, the rows of `H_1`, the
rows of `F_2` and the rows of `H_2` (47). -/
noncomputable def carriedRows (key : Key S) (openKey : OpeningKey O) (payload : S.Payload)
    (openPayload : O.Payload) : Fin O.carriedCount → RingRow :=
  Fin.append (imageRow S.chain (outerForm O fold key) (O.firstCell fold) payload)
    (Fin.append (imageRow O.chain (openForm O fold openKey) (O.openFirstCell fold) openPayload)
      (Fin.append (firstRow S.chain key.chain (O.firstCell fold) (O.secondCell fold))
        (Fin.append (firstRow O.chain openKey.chain (O.openFirstCell fold) (O.openSecondCell fold))
          (Fin.append (secondRow S.chain key.chain (O.secondCell fold) payload)
            (secondRow O.chain openKey.chain (O.openSecondCell fold) openPayload)))))

theorem carriedRows_below (key : Key S) (openKey : OpeningKey O) (payload : S.Payload)
    (openPayload : O.Payload) (r : Fin O.carriedCount) :
    (carriedRows O fold key openKey payload openPayload r).Below (O.cells fold) := by
  revert r
  refine (forall_append (RingRow.Below · (O.cells fold)) _ _).mpr ⟨?_,
    (forall_append (RingRow.Below · (O.cells fold)) _ _).mpr ⟨?_,
    (forall_append (RingRow.Below · (O.cells fold)) _ _).mpr ⟨?_,
    (forall_append (RingRow.Below · (O.cells fold)) _ _).mpr ⟨?_,
    (forall_append (RingRow.Below · (O.cells fold)) _ _).mpr ⟨?_, ?_⟩⟩⟩⟩⟩
  · exact imageRow_below S.chain _ _ _
      (fun _ => Form.ofMatrix_below _ (fun _ _ _ => O.sliceCell_lt fold _ _ _) _)
      (fun _ _ => O.firstCell_lt fold _)
  · exact imageRow_below O.chain _ _ _
      (fun _ => Form.ofMatrix_below _ (fun _ _ _ => O.partialCell_lt fold _ _) _)
      (fun _ _ => O.openFirstCell_lt fold _)
  · exact firstRow_below S.chain _ _ _ (fun _ _ => O.firstCell_lt fold _)
      (fun _ _ => O.secondCell_lt fold _)
  · exact firstRow_below O.chain _ _ _ (fun _ _ => O.openFirstCell_lt fold _)
      (fun _ _ => O.openSecondCell_lt fold _)
  · exact secondRow_below S.chain _ _ _ (fun _ _ => O.secondCell_lt fold _)
  · exact secondRow_below O.chain _ _ _ (fun _ _ => O.openSecondCell_lt fold _)

theorem carriedRows_degree_le (key : Key S) (openKey : OpeningKey O) (payload : S.Payload)
    (openPayload : O.Payload) (r : Fin O.carriedCount) :
    (carriedRows O fold key openKey payload openPayload r).degree ≤ O.largestDegree := by
  revert r
  refine (forall_append (RingRow.degree · ≤ O.largestDegree) _ _).mpr ⟨fun j => ?_,
    (forall_append (RingRow.degree · ≤ O.largestDegree) _ _).mpr ⟨fun j => ?_,
    (forall_append (RingRow.degree · ≤ O.largestDegree) _ _).mpr ⟨fun r => ?_,
    (forall_append (RingRow.degree · ≤ O.largestDegree) _ _).mpr ⟨fun r => ?_,
    (forall_append (RingRow.degree · ≤ O.largestDegree) _ _).mpr
      ⟨fun r => ?_, fun r => ?_⟩⟩⟩⟩⟩
  · rw [imageRow_degree]
    exact le_max_of_le_right (le_max_of_le_left (le_max_left _ _))
  · rw [imageRow_degree]
    exact le_max_of_le_right (le_max_of_le_left (le_max_right _ _))
  · exact (firstRow_degree_le S.chain _ _ _ r).trans
      (le_max_of_le_right (le_max_of_le_right (le_max_left _ _)))
  · exact (firstRow_degree_le O.chain _ _ _ r).trans
      (le_max_of_le_right (le_max_of_le_right (le_max_right _ _)))
  · exact (secondRow_degree_le S.chain _ _ _ r).trans
      (le_max_of_le_right (le_max_of_le_right (le_max_left _ _)))
  · exact (secondRow_degree_le O.chain _ _ _ r).trans
      (le_max_of_le_right (le_max_of_le_right (le_max_right _ _)))

/-- The number of ring rows of one trace fold. -/
def OpeningShape.traceRowCount : ℕ := 1 + (S.innerRank + O.carriedCount)

/-- The ring rows of one trace fold in the order of Akita Figure 5: the row (103), the rows (104)
of `A`, and the rows `carriedRows`. -/
noncomputable def traceRows (st : TraceStatement O E)
    (challenge : Fin S.blocks → Rq S.innerDegree) : Fin O.traceRowCount → RingRow :=
  Fin.append (fun _ : Fin 1 => traceEvaluationRow O fold st challenge)
    (Fin.append (traceInnerRow O fold st.key challenge)
      (carriedRows O fold st.key st.openKey st.payload st.openPayload))

theorem traceRows_below (st : TraceStatement O E) (challenge : Fin S.blocks → Rq S.innerDegree)
    (r : Fin O.traceRowCount) : (traceRows O fold st challenge r).Below (O.cells fold) := by
  revert r
  refine (forall_append (RingRow.Below · (O.cells fold)) _ _).mpr ⟨fun _ => ?_,
    (forall_append (RingRow.Below · (O.cells fold)) _ _).mpr ⟨fun a => ?_,
      carriedRows_below O fold st.key st.openKey st.payload st.openPayload⟩⟩
  · exact (Form.Below.sub (f := tracePartialForm O fold challenge)
      (g := traceContractForm fold st.subfield st.position)
      (fun _ _ _ => O.partialPosition_lt fold _ _ _) (fun _ _ _ => O.responseCell_lt fold _ _)).row
        0
  · exact (Form.Below.sub (f := responseForm fold st.key a)
      (g := traceImageForm O fold challenge a)
      (fun _ _ _ => O.responseCell_lt fold _ _) (fun _ _ _ => O.innerCell_lt fold _ _ _ _)).row 0

theorem traceRows_degree_le (st : TraceStatement O E)
    (challenge : Fin S.blocks → Rq S.innerDegree) (r : Fin O.traceRowCount) :
    (traceRows O fold st challenge r).degree ≤ O.largestDegree := by
  revert r
  exact (forall_append (RingRow.degree · ≤ O.largestDegree) _ _).mpr
    ⟨fun _ => le_max_of_le_left (le_max_right O.carrierDegree S.innerDegree),
      (forall_append (RingRow.degree · ≤ O.largestDegree) _ _).mpr
        ⟨fun _ => le_max_of_le_left (le_max_right O.carrierDegree S.innerDegree),
          carriedRows_degree_le O fold st.key st.openKey st.payload st.openPayload⟩⟩

/-- The relation of one trace fold on the flat witness `Witness.flat`, for the fold challenges
`c_i ∈ R_{q,d_A}` (Akita Figure 5 for one group and one claim, on the quotient-free route): the
ring rows `traceRows` and the scalar row (101). The cube has `μ` variables, with
`cells < 2^μ`. -/
noncomputable def traceRelation (st : TraceStatement O E)
    (challenge : Fin S.blocks → Rq S.innerDegree) (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ) :
    FlatRelation E where
  vars := μ
  low := O.low
  ring := O.traceRowCount
  row := traceRows O fold st challenge
  inside r c j _ _ read := (traceRows_below O fold st challenge r c j _ read).trans_le fits
  join := Join.plain E O.traceRowCount
  independent := Join.plain_independent (algebraMap Zq E).injective _
  weight := traceScalarWeight O fold st
  value := st.value

end Rows

section Values

variable (w : Witness O fold)

theorem tracePartialForm_value (challenge : Fin S.blocks → Rq S.innerDegree) :
    (tracePartialForm O fold challenge).value w.flat =
      ∑ i, challenge i * O.tracePartial w.partials i := by
  have partials : ∀ i, O.tracePartial w.partials i = ∑ u : Fin S.openingDigits,
      (S.openingBase : Rq S.innerDegree) ^ (u : ℕ) *
        ofCoeffs fun j => (O.partialPosition fold i u j).elim 0 w.flat :=
    fun i => ext_coeff fun j small => by
      rw [OpeningShape.tracePartial, coeff_ofCoeffs _ small, ← w.sum_flat_partialPosition i j,
        coeff_sum]
      exact Finset.sum_congr rfl fun u _ => by
        rw [coeff_natCast_pow_mul, coeff_ofCoeffs _ small]
  show (∑ entry : Fin S.blocks × Fin S.openingDigits,
    challenge entry.1 * (S.openingBase : Rq S.innerDegree) ^ (entry.2 : ℕ) *
      ofCoeffs fun j => (O.partialPosition fold entry.1 entry.2 j).elim 0 w.flat) = _
  rw [Fintype.sum_prod_type]
  simp only [partials, Finset.mul_sum, mul_assoc]

omit [Algebra Zq E] in
theorem traceContractForm_value (subfield : E →+* Rq S.innerDegree)
    (position : Fin S.blockLength → E) :
    (traceContractForm fold subfield position).value w.flat =
      traceContract subfield position
        (gadget S.sourceBase (gadget fold.responseBase w.response)) := by
  rw [traceContractForm, Form.value_mk _ _ w.flat fun col =>
    ofCoeffs_eq fun _ small => w.flat_responseCell col small]
  simp only [traceContract, gadget, Finset.mul_sum, Fintype.sum_prod_type, mul_assoc]

theorem traceEvaluationRow_holds (st : TraceStatement O E)
    (challenge : Fin S.blocks → Rq S.innerDegree) :
    (traceEvaluationRow O fold st challenge).Holds w.flat ↔
      ∑ i, challenge i * O.tracePartial w.partials i =
        st.contract (gadget S.sourceBase (gadget fold.responseBase w.response)) := by
  rw [traceEvaluationRow, Form.row_holds, Form.value_sub, sub_eq_zero, tracePartialForm_value,
    traceContractForm_value]
  rfl

theorem traceImageForm_value (challenge : Fin S.blocks → Rq S.innerDegree)
    (a : Fin S.innerRank) :
    (traceImageForm O fold challenge a).value w.flat =
      (∑ i, challenge i • gadget S.openingBase (w.inner i)) a := by
  rw [traceImageForm, Form.value_mk _ _ w.flat
    (x := fun entry : Fin S.blocks × Fin S.openingDigits => w.inner entry.1 (a, entry.2))
    fun entry : Fin S.blocks × Fin S.openingDigits =>
      ofCoeffs_eq fun _ small => w.flat_innerCell entry.1 a entry.2 small]
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, gadget, Finset.mul_sum,
    Fintype.sum_prod_type, mul_assoc]

theorem traceInnerRow_holds (key : Key S) (challenge : Fin S.blocks → Rq S.innerDegree)
    (a : Fin S.innerRank) :
    (traceInnerRow O fold key challenge a).Holds w.flat ↔
      key.inner.mulVec (gadget fold.responseBase w.response) a =
        (∑ i, challenge i • gadget S.openingBase (w.inner i)) a := by
  rw [traceInnerRow, Form.row_holds, Form.value_sub, sub_eq_zero, responseForm_value,
    traceImageForm_value]

/-- The rows without a fold challenge hold for the flat witness when the digit vectors satisfy
the rows (105) and (106) with their rows (47) or (48), and only then. -/
theorem carriedRows_holds_iff (key : Key S) (openKey : OpeningKey O) (payload : S.Payload)
    (openPayload : O.Payload) :
    (∀ r, (carriedRows O fold key openKey payload openPayload r).Holds w.flat) ↔
      Rows S.chain key.chain (stack key.outer (S.sliceInput w.inner)) w.chain payload ∧
        OpeningRows openKey w.partials w.openChain openPayload := by
  have outer := rows_iff S.chain key.chain w.chain payload (outerForm O fold key)
    (stack key.outer (S.sliceInput w.inner)) (O.firstCell fold) (O.secondCell fold) w.flat
    (outerForm_value w key) w.flat_firstCell w.flat_secondCell
  have opening := rows_iff O.chain openKey.chain w.openChain openPayload
    (openForm O fold openKey) (stack openKey.matrix fun _ : Fin 1 => w.partials)
    (O.openFirstCell fold) (O.openSecondCell fold) w.flat (openForm_value w openKey)
    w.flat_openFirstCell w.flat_openSecondCell
  have rows : (∀ r, (carriedRows O fold key openKey payload openPayload r).Holds w.flat) ↔
      (∀ j, (imageRow S.chain (outerForm O fold key) (O.firstCell fold) payload j).Holds
        w.flat) ∧
      (∀ j, (imageRow O.chain (openForm O fold openKey) (O.openFirstCell fold)
        openPayload j).Holds w.flat) ∧
      (∀ r, (firstRow S.chain key.chain (O.firstCell fold) (O.secondCell fold) r).Holds
        w.flat) ∧
      (∀ r, (firstRow O.chain openKey.chain (O.openFirstCell fold) (O.openSecondCell fold)
        r).Holds w.flat) ∧
      (∀ r, (secondRow S.chain key.chain (O.secondCell fold) payload r).Holds w.flat) ∧
      ∀ r, (secondRow O.chain openKey.chain (O.openSecondCell fold) openPayload r).Holds
        w.flat :=
    (forall_append (RingRow.Holds · w.flat) _ _).trans (and_congr_right'
      ((forall_append (RingRow.Holds · w.flat) _ _).trans (and_congr_right'
        ((forall_append (RingRow.Holds · w.flat) _ _).trans (and_congr_right'
          ((forall_append (RingRow.Holds · w.flat) _ _).trans (and_congr_right'
            (forall_append (RingRow.Holds · w.flat) _ _))))))))
  constructor
  · intro all
    obtain ⟨outerImage, openImage, outerFirst, openFirst, outerSecond, openSecond⟩ :=
      rows.mp all
    exact ⟨outer.mp ⟨outerImage, outerFirst, outerSecond⟩,
      opening.mp ⟨openImage, openFirst, openSecond⟩⟩
  · rintro ⟨outerRows, openRows⟩
    obtain ⟨outerImage, outerFirst, outerSecond⟩ := outer.mpr outerRows
    obtain ⟨openImage, openFirst, openSecond⟩ := opening.mpr openRows
    exact rows.mpr ⟨outerImage, openImage, outerFirst, openFirst, outerSecond, openSecond⟩

/-- The scalar row of the flat witness is the left side of the scalar row (101) of Akita. -/
theorem traceScalarWeight_sum (st : TraceStatement O E) (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ) :
    ∑ x ∈ Finset.range (2 ^ μ), algebraMap Zq E (w.flat x) * traceScalarWeight O fold st x =
      st.scalar (O.tracePartial w.partials) := by
  have functional : ∀ i, st.functional (O.tracePartial w.partials i) =
      ∑ j : Fin S.innerDegree, ∑ u : Fin S.openingDigits,
        ((S.openingBase : Zq) ^ (u : ℕ) * (O.partialPosition fold i u j).elim 0 w.flat) •
          st.functional (AdjoinRoot.root (modulus S.innerDegree) ^ (j : ℕ)) := fun i => by
    rw [OpeningShape.tracePartial, ofCoeffs_eq_sum, map_sum, Finset.sum_range]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [map_smul, ← w.sum_flat_partialPosition i j, Finset.sum_smul]
  unfold traceScalarWeight
  rw [sum_mul_indicator _ _ _ _ fun entry _ read => (O.partialPosition_lt fold _ _ _ read).trans_le
    fits]
  simp only [elim_algebraMap, Fintype.sum_prod_type, TraceStatement.scalar, functional,
    Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ =>
    Finset.sum_congr rfl fun u _ => ?_
  simp only [Algebra.smul_def, map_mul]
  ring

end Values

section Link

variable (st : TraceStatement O E) (challenge : Fin S.blocks → Rq S.innerDegree) (μ : ℕ)
  (fits : O.cells fold ≤ 2 ^ μ)

/-- The flat witness satisfies the relation of one trace fold when the digit vectors satisfy the
rows (103), (104), (105) and (106) with their rows (47) or (48), and the scalar row (101), of
Akita Figure 5, and only then. -/
theorem traceRelation_holds_iff (w : Witness O fold) :
    (traceRelation O fold st challenge μ fits).Holds w.flat ↔
      ∑ i, challenge i * O.tracePartial w.partials i =
        st.contract (gadget S.sourceBase (gadget fold.responseBase w.response)) ∧
      st.key.inner.mulVec (gadget fold.responseBase w.response) =
        ∑ i, challenge i • gadget S.openingBase (w.inner i) ∧
      Rows S.chain st.key.chain (stack st.key.outer (S.sliceInput w.inner)) w.chain st.payload ∧
      OpeningRows st.openKey w.partials w.openChain st.openPayload ∧
      st.scalar (O.tracePartial w.partials) = st.value := by
  have rows : (∀ r, (traceRows O fold st challenge r).Holds w.flat) ↔
      (Fin 1 → (traceEvaluationRow O fold st challenge).Holds w.flat) ∧
      (∀ a, (traceInnerRow O fold st.key challenge a).Holds w.flat) ∧
      ∀ r, (carriedRows O fold st.key st.openKey st.payload st.openPayload r).Holds w.flat :=
    (forall_append (RingRow.Holds · w.flat) _ _).trans (and_congr_right'
      (forall_append (RingRow.Holds · w.flat) _ _))
  have carried := carriedRows_holds_iff w st.key st.openKey st.payload st.openPayload
  have value := traceScalarWeight_sum w st μ fits
  constructor
  · rintro ⟨ring, scalarRow⟩
    obtain ⟨evaluation, inner, rest⟩ := rows.mp ring
    obtain ⟨outerRows, openRows⟩ := carried.mp rest
    exact ⟨(traceEvaluationRow_holds w st challenge).mp (evaluation 0),
      funext fun a => (traceInnerRow_holds w st.key challenge a).mp (inner a),
      outerRows, openRows, value.symm.trans scalarRow⟩
  · rintro ⟨evaluation, inner, outerRows, openRows, scalarRow⟩
    exact ⟨rows.mpr ⟨fun _ => (traceEvaluationRow_holds w st challenge).mpr evaluation,
      fun a => (traceInnerRow_holds w st.key challenge a).mpr (congrFun inner a),
      carried.mpr ⟨outerRows, openRows⟩⟩, value.trans scalarRow⟩

variable {st challenge}

/-- The premise `encodes` of `foldCheck_traceSatisfies` for `traceRelation`: a witness with the
certified alphabets whose flat witness satisfies the relation of one trace fold satisfies the
exact branch relation `Witness.TraceSatisfies` (the hypothesis of Akita Corollary 10.32). -/
theorem Witness.traceSatisfies_of_flat {w : Witness O fold} (alphabets : w.Alphabets)
    (holds : (traceRelation O fold st challenge μ fits).Holds w.flat) :
    w.TraceSatisfies st challenge := by
  obtain ⟨evaluation, inner, outerRows, openRows, value⟩ :=
    (traceRelation_holds_iff st challenge μ fits w).mp holds
  exact ⟨⟨outerRows, openRows, value, alphabets.innerDigits, alphabets.chainDigits,
    alphabets.partialDigits, alphabets.openDigits⟩, evaluation, inner, alphabets.responseDigits⟩

/-- The honest direction: the flat witness of digit vectors that satisfy the exact branch
relation satisfies the relation of one trace fold (Akita Theorem 7.4, item 1, for the rows). -/
theorem Witness.TraceSatisfies.flat {w : Witness O fold}
    (satisfies : w.TraceSatisfies st challenge) :
    (traceRelation O fold st challenge μ fits).Holds w.flat :=
  (traceRelation_holds_iff st challenge μ fits w).mpr ⟨satisfies.evaluation, satisfies.innerRows,
    satisfies.carried.rows, satisfies.carried.openRows, satisfies.carried.value⟩

variable (O) (fold) (st) (challenge)

/-- The weight of the scalar row (101) is 0 from the offset `chainStart`. -/
theorem traceScalarWeight_of_le {x : ℕ} (large : O.chainStart fold ≤ x) :
    traceScalarWeight O fold st x = 0 :=
  Finset.sum_eq_zero fun entry _ => ite_eq_right fun same => by
    have below := O.partialPosition_lt_innerStart fold entry.1 entry.2.2 entry.2.1 same
    unfold OpeningShape.chainStart at large
    omega

variable {O fold st challenge μ fits}

/-- Two tables with the same cells before `cells` satisfy `traceRelation` together. -/
theorem traceRelation_holds_congr {table other : Table}
    (same : ∀ x < O.cells fold, table x = other x)
    (holds : (traceRelation O fold st challenge μ fits).Holds table) :
    (traceRelation O fold st challenge μ fits).Holds other := by
  refine ⟨fun r => RingRow.holds_congr (traceRows_below O fold st challenge r) same (holds.1 r),
    ?_⟩
  rw [← holds.2]
  refine Finset.sum_congr rfl fun u _ => ?_
  show algebraMap Zq E (other u) * traceScalarWeight O fold st u =
    algebraMap Zq E (table u) * traceScalarWeight O fold st u
  by_cases inside : u < O.cells fold
  · rw [same u inside]
  · rw [traceScalarWeight_of_le O fold st
      ((O.chainStart_le_cells fold).trans (Nat.le_of_not_lt inside)), mul_zero, mul_zero]

variable [Fintype E] (st challenge μ fits)

/-- The quotient-free reduction of the relation of one trace fold (Akita Section 7.3 and
Theorem 7.4), with `2^batch` places for the rows. Its bad set has at most `d_max − 1` points. -/
noncomputable def traceReduction (batch : ℕ) (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch) :
    Reduction E :=
  (traceRelation O fold st challenge μ fits).free O.largestDegree
    batch (traceRows_degree_le O fold st challenge) rowsFit

variable [DecidableEq E]

/-- `foldCheck_traceSatisfies` for the flat witness `Witness.flat` and the relation
`traceRelation`, with no layout and no premise `encodes`: on a path of `foldCheck` that avoids the
bad sets, an accepted claim that is true for the flat witness gives the exact branch relation
`Witness.TraceSatisfies` for the fold challenges (the hypothesis of Akita Corollary 10.32 for one
child). -/
theorem traceFold_satisfies (small : S.certifiedBase < fieldModulus) (batch : ℕ)
    (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch) (w : Witness O fold) :
    (foldCheck (traceReduction st challenge μ fits batch rowsFit) S.certifiedBase
      (O.binarySupport fold) (O.compressed fold) w.flat).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E w.flat) →
        w.TraceSatisfies st challenge :=
  foldCheck_traceSatisfies (traceReduction st challenge μ fits batch rowsFit)
    (O.flatLayout fold μ fits) small st challenge
    (fun _ alphabets holds => Witness.traceSatisfies_of_flat μ fits alphabets holds) w

omit [DecidableEq E] in
/-- The exact branch relation `Witness.TraceSatisfies` of the digit vectors `ofFlat` of a table
with the relation of `traceReduction`, the digit range statement and the binary statement. The
table can have each value from the cell `cells`. -/
theorem traceSatisfies_of_table (small : S.certifiedBase < fieldModulus) (batch : ℕ)
    (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch) {table : Table}
    (relation : (traceReduction (E := E) st challenge μ fits batch rowsFit).Holds table)
    (range : RangeHolds S.certifiedBase μ table)
    (binary : BinaryOn (O.binarySupport fold) μ table) :
    (O.ofFlat fold table).TraceSatisfies st challenge := by
  have same : ∀ x < O.cells fold, table x = (O.ofFlat fold table).flat x :=
    fun x inside => (O.flat_ofFlat fold table inside).symm
  have ranged : RangeHolds S.certifiedBase μ (O.ofFlat fold table).flat := fun x inside => by
    by_cases below : x < O.cells fold
    · rw [← same x below]
      exact range x inside
    · rw [Witness.flat_of_le _ (Nat.le_of_not_lt below), zero_mul]
      exact range.zero
  have binaried : BinaryOn (O.binarySupport fold) μ (O.ofFlat fold table).flat :=
    fun x inside member => by
      rw [← same x member.2]
      exact binary x inside member
  exact Witness.traceSatisfies_of_flat μ fits
    ((O.flatLayout fold μ fits).alphabets small _ ranged binaried)
    (traceRelation_holds_congr same relation)

/-- Akita Lemma 10.42 (i) for the table of a successor source of a trace fold, with
`foldCheck_avoids`: on a path of `foldCheck` that avoids the bad sets, a final claim that is true
for the table gives the exact branch relation `Witness.TraceSatisfies` of the digit vectors
`ofFlat` of the table (the hypothesis of Akita Corollary 10.32 for one child). -/
theorem traceTableFold_satisfies (small : S.certifiedBase < fieldModulus) (batch : ℕ)
    (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch) (table : Table) :
    (foldCheck (traceReduction st challenge μ fits batch rowsFit) S.certifiedBase
      (O.binarySupport fold) (O.compressed fold) table).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E table) →
        (O.ofFlat fold table).TraceSatisfies st challenge :=
  (foldCheck_avoids (traceReduction st challenge μ fits batch rowsFit)
    S.certifiedBase (O.binarySupport fold) (O.compressed fold) (O.binarySupport_empty fold)
      table).mono fun _ checked r2 v equal opening =>
      traceSatisfies_of_table st challenge μ fits small batch rowsFit
        (checked r2 v equal opening).1 (checked r2 v equal opening).2.1
        (checked r2 v equal opening).2.2

end Link

end NightstreamFPrime.Spec.Compression.Akita
