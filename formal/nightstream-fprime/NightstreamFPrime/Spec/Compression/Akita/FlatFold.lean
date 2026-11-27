import NightstreamFPrime.Spec.Compression.Akita.FlatWitness

/-! The relation of one fold of Akita with coefficient packing on the flat witness, for one claim
in one group, and its link to the exact branch relation (ePrint 2026/1983, revision of 2026-11-23,
Section 7.2 with Figure 5, Section 7.3 with (108) to (110), and the hypothesis of
Corollary 10.32).

- `Form` is a sum `Σ_c A_c·W_c` of (110): for each occurrence a public multiplier and an address
  map. `Form.value` is its value for a flat witness, `Form.sub` is a difference of two sums, and
  `Form.row` is the ring row (110) with a target.
- `evaluationRow` is plane `t` of the carrier row (102), `innerRow` is a row of (104),
  `imageRow` is a row of (105) or (106) with the first row of (47) or the row (48), and `firstRow`
  and `secondRow` are the other rows of (47). `scalarWeight` has the weights of the scalar
  row (100).
- `flatRows` has the ring rows in the order of Akita Figure 5, and `flatRelation` is the
  `FlatRelation` of one fold. `flatRelation_holds_iff` states that `Witness.flat` satisfies it
  when the digit vectors satisfy the rows of `Witness.Satisfies`, and only then.
- `Witness.satisfies_of_flat` is the premise `encodes` of `foldCheck_satisfies`, and
  `Witness.Satisfies.flat` is the honest direction.
- `flatReduction` is the quotient-free reduction of `flatRelation`. `flatFold_satisfies` and
  `flatFold_satisfies_error` are `foldCheck_satisfies` and `foldCheck_satisfies_error` for
  `Witness.flat`, with no layout and no premise `encodes`.

The order of the occurrences in a row is not given, because (110) is a sum. The quotient-lift
route, with its quotient digits in the witness, is not in this file. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

/-- A property holds for each entry of two joined families when it holds for each entry of each
family, and only then. -/
theorem forall_append {α : Type} {m n : ℕ} (P : α → Prop) (a : Fin m → α) (b : Fin n → α) :
    (∀ r, P (Fin.append a b r)) ↔ (∀ i, P (a i)) ∧ ∀ j, P (b j) := by
  constructor
  · intro all
    refine ⟨fun i => ?_, fun j => ?_⟩
    · have entry := all (Fin.castAdd n i)
      rwa [Fin.append_left] at entry
    · have entry := all (Fin.natAdd m j)
      rwa [Fin.append_right] at entry
  · rintro ⟨left, right⟩ r
    refine Fin.addCases (fun i => ?_) (fun j => ?_) r
    · rw [Fin.append_left]
      exact left i
    · rw [Fin.append_right]
      exact right j

/-- A sum over the entries of two joined families is the sum of the sums over each family. -/
theorem sum_append {α M : Type} [AddCommMonoid M] {m n : ℕ} (f : α → M) (a : Fin m → α)
    (b : Fin n → α) : ∑ r, f (Fin.append a b r) = ∑ i, f (a i) + ∑ j, f (b j) := by
  rw [Fin.sum_univ_add]
  simp only [Fin.append_left, Fin.append_right]

/-- A sum over the cells of a table, with weights that are sums of the coefficients of the
occurrences at each cell, is the sum over the occurrences. An occurrence with no cell adds 0. -/
theorem sum_mul_indicator {E κ : Type} [Semiring E] [Fintype κ] (n : ℕ) (address : κ → Option ℕ)
    (coefficient : κ → E) (f : ℕ → E) (inside : ∀ c x, address c = some x → x < n) :
    ∑ x ∈ Finset.range n, f x * ∑ c, (if address c = some x then coefficient c else 0) =
      ∑ c, (address c).elim 0 f * coefficient c := by
  simp only [Finset.mul_sum, mul_ite, mul_zero]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun c _ => ?_
  cases read : address c with
  | none => simp
  | some x =>
      simp only [Option.some.injEq, Option.elim_some]
      rw [Finset.sum_ite_eq, ite_eq_left (Finset.mem_range.mpr (inside c x read))]

theorem elim_algebraMap {E : Type} [CommSemiring E] [Algebra Zq E] (o : Option ℕ) (w : Table) :
    o.elim 0 (fun x => algebraMap Zq E (w x)) = algebraMap Zq E (o.elim 0 w) := by
  cases o <;> simp

theorem plane_natCast_pow_mul {d dCar : ℕ} [NeZero d] [NeZero dCar] (stride a b u : ℕ) (x : Rq d) :
    plane dCar stride a ((b : Rq d) ^ u * x) = (b : Rq dCar) ^ u * plane dCar stride a x := by
  rw [← Nat.cast_pow, ← Nat.cast_pow, ← nsmul_eq_mul, ← nsmul_eq_mul, map_nsmul]

/-- Each address of the row is below `bound`. -/
def RingRow.Below (row : RingRow) (bound : ℕ) : Prop :=
  ∀ c j x, row.address c j = some x → x < bound

/-- A sum `Σ_c A_c·W_c` of Akita (110) in `R_{q,d}`: for each occurrence `c` a public multiplier
`A_c` and an address map (108), with `none` for a coefficient 0. -/
structure Form (d : ℕ) where
  Occurrence : Type
  [finite : Fintype Occurrence]
  multiplier : Occurrence → Rq d
  address : Occurrence → ℕ → Option ℕ

attribute [instance] Form.finite

namespace Form

section Parts

variable {d : ℕ} (f g : Form d)

/-- The difference of two sums: the occurrences of `f`, then the occurrences of `g` with the
negations of their multipliers. -/
noncomputable def sub : Form d where
  Occurrence := f.Occurrence ⊕ g.Occurrence
  multiplier := Sum.elim f.multiplier fun c => -g.multiplier c
  address := Sum.elim f.address g.address

/-- Each address of the sum is below `bound`. -/
def Below (bound : ℕ) : Prop := ∀ c j x, f.address c j = some x → x < bound

theorem Below.sub {f g : Form d} {bound : ℕ} (left : f.Below bound) (right : g.Below bound) :
    (f.sub g).Below bound :=
  fun c => match c with
    | .inl c => left c
    | .inr c => right c

/-- `Σ_c A_c·W_c` for the flat witness `w`, with `W_c` of Akita (109). -/
noncomputable def value (w : Table) : Rq d :=
  ∑ c, f.multiplier c * ofCoeffs fun j => (f.address c j).elim 0 w

theorem value_mk {κ : Type} [Fintype κ] (multiplier : κ → Rq d) (address : κ → ℕ → Option ℕ)
    (w : Table) {x : κ → Rq d} (reads : ∀ c, ofCoeffs (fun j => (address c j).elim 0 w) = x c) :
    (Form.mk κ multiplier address).value w = ∑ c, multiplier c * x c :=
  Finset.sum_congr rfl fun c _ => by rw [← reads c]

theorem value_sub (w : Table) : (f.sub g).value w = f.value w - g.value w := by
  show (∑ c : f.Occurrence ⊕ g.Occurrence, Sum.elim f.multiplier (fun c => -g.multiplier c) c *
    ofCoeffs fun j => (Sum.elim f.address g.address c j).elim 0 w) = _
  rw [Fintype.sum_sum_type]
  simp only [Sum.elim_inl, Sum.elim_inr, neg_mul, Finset.sum_neg_distrib, value, sub_eq_add_neg]

/-- `Σ_u 2^u·ξ_{v + s·u}` for the coefficients `v = entry·d + k` of one element of an image of
`n` elements, with `s = n·d` (Akita Appendix L.1, binary digit placement). `cell` gives the
position of an entry of the digit sequence `ξ`. -/
noncomputable def recompose (d n : ℕ) (cell : ℕ → Option ℕ) (entry : Fin n) : Form d where
  Occurrence := Fin binaryDigits
  multiplier u := ((2 : ℕ) : Rq d) ^ (u : ℕ)
  address u k := cell (entry * d + k + n * d * u)

theorem recompose_below {n bound : ℕ} {cell : ℕ → Option ℕ}
    (below : ∀ position x, cell position = some x → x < bound) (entry : Fin n) :
    (recompose d n cell entry).Below bound :=
  fun _ _ => below _

/-- Row `r` of a product `M·x`: the entries of the row of `M` as multipliers, and the address
map of each element of `x`. -/
def ofMatrix {n : ℕ} {cols : Type} [Fintype cols] (M : Matrix (Fin n) cols (Rq d))
    (address : cols → ℕ → Option ℕ) (r : Fin n) : Form d where
  Occurrence := cols
  multiplier c := M r c
  address := address

theorem ofMatrix_below {n bound : ℕ} {cols : Type} [Fintype cols] (M : Matrix (Fin n) cols (Rq d))
    {address : cols → ℕ → Option ℕ} (below : ∀ c k x, address c k = some x → x < bound)
    (r : Fin n) :
    (ofMatrix M address r).Below bound :=
  below

theorem value_ofMatrix {n : ℕ} {cols : Type} [Fintype cols] (M : Matrix (Fin n) cols (Rq d))
    (address : cols → ℕ → Option ℕ) (w : Table) {x : cols → Rq d}
    (reads : ∀ c, ofCoeffs (fun k => (address c k).elim 0 w) = x c) (r : Fin n) :
    (ofMatrix M address r).value w = M.mulVec x r :=
  value_mk (fun c => M r c) address w reads

end Parts

variable {d : ℕ} [NeZero d] (f g : Form d)

theorem value_recompose {n width inputDegree : ℕ} [NeZero inputDegree]
    (digits : Fin width → Rq inputDegree) (w : Table) (cell : ℕ → Option ℕ)
    (reads : ∀ position, (cell position).elim 0 w = sequence digits position) (entry : Fin n) :
    (recompose d n cell entry).value w = binaryRecompose d n digits entry := by
  refine ext_coeff fun k small => ?_
  show coeff (∑ u : Fin binaryDigits, ((2 : ℕ) : Rq d) ^ (u : ℕ) *
    ofCoeffs fun k => (cell (entry * d + k + n * d * u)).elim 0 w) k = _
  rw [coeff_sum, binaryRecompose, coeff_ofCoeffs _ small,
    ← Fin.sum_univ_eq_sum_range (fun u => 2 ^ u * sequence digits (entry * d + k + n * d * u))]
  exact Finset.sum_congr rfl fun u _ => by
    rw [coeff_natCast_pow_mul, coeff_ofCoeffs _ small, reads, Nat.cast_ofNat]

/-- The ring row `Σ_c A_c·W_c = Y` of Akita (110). The order of its occurrences is not given. -/
noncomputable def row (target : Rq d) : RingRow where
  degree := d
  occurrences := Fintype.card f.Occurrence
  multiplier c := f.multiplier ((Fintype.equivFin f.Occurrence).symm c)
  address c := f.address ((Fintype.equivFin f.Occurrence).symm c)
  target := target

theorem row_holds (target : Rq d) (w : Table) : (f.row target).Holds w ↔ f.value w = target := by
  show (∑ c : Fin (Fintype.card f.Occurrence),
    f.multiplier ((Fintype.equivFin f.Occurrence).symm c) *
      (ofCoeffs fun j => (f.address ((Fintype.equivFin f.Occurrence).symm c) j).elim 0 w :
        Rq d)) - target = 0 ↔ _
  rw [sub_eq_zero]
  exact Eq.congr_left ((Fintype.equivFin f.Occurrence).symm.sum_comp fun c =>
    f.multiplier c * ofCoeffs fun j => (f.address c j).elim 0 w)

theorem Below.row {f : Form d} {bound : ℕ} (below : f.Below bound) (target : Rq d) :
    (f.row target).Below bound :=
  fun _ j => below _ j

end Form

section Path

/-- The rank of the first compression map, and 0 without compression maps. -/
def firstRank : Option ChainShape → ℕ
  | none => 0
  | some c => c.firstRank

/-- The rank of the second compression map, and 0 without compression maps. -/
def secondRank : Option ChainShape → ℕ
  | none => 0
  | some c => c.secondRank

/-- The larger ring degree of the two compression maps, and 0 without compression maps. -/
def chainDegree : Option ChainShape → ℕ
  | none => 0
  | some c => max c.firstDegree c.secondDegree

/-- The ring degree of the first compression map, and 0 without compression maps. -/
def firstDegree : Option ChainShape → ℕ
  | none => 0
  | some c => c.firstDegree

/-- The ring degree of the second compression map, and 0 without compression maps. -/
def secondDegree : Option ChainShape → ℕ
  | none => 0
  | some c => c.secondDegree

variable {d n : ℕ}

/-- Row `r` of `K_2·ξ̂_2 = p` of Akita (47). `second` gives the position of an entry of the digit
sequence `ξ̂_2`. -/
noncomputable def secondRow : (chain : Option ChainShape) → PathKey chain → (second : ℕ → Option ℕ)
  →
    Payload d n chain → Fin (secondRank chain) → RingRow
  | none, _, _, _, r => r.elim0
  | some c, key, second, payload, r =>
      (Form.ofMatrix (ChainKey.second key) (fun col k => second (col * c.secondDegree + k))
        r).row (payload r)

theorem secondRow_below {bound : ℕ} : (chain : Option ChainShape) → (key : PathKey chain) →
    (second : ℕ → Option ℕ) → (payload : Payload d n chain) → (∀ position x, second position = some
      x → x < bound) →
    ∀ r, (secondRow chain key second payload r).Below bound
  | none, _, _, _, _, r => r.elim0
  | some _, _, _, _, below, _ => (Form.ofMatrix_below _ (fun _ _ => below _) _).row _

theorem secondRow_degree_le : (chain : Option ChainShape) → (key : PathKey chain) →
    (second : ℕ → Option ℕ) → (payload : Payload d n chain) →
    ∀ r, (secondRow chain key second payload r).degree ≤ chainDegree chain
  | none, _, _, _, r => r.elim0
  | some c, _, _, _, _ => le_max_right c.firstDegree c.secondDegree

theorem secondRow_degree : (chain : Option ChainShape) → (key : PathKey chain) →
    (second : ℕ → Option ℕ) → (payload : Payload d n chain) →
    ∀ r, (secondRow chain key second payload r).degree = secondDegree chain
  | none, _, _, _, r => r.elim0
  | some _, _, _, _, _ => rfl

/-- Row `r` of `K_1·ξ̂_1 = G·ξ̂_2` of Akita (47). -/
noncomputable def firstRow : (chain : Option ChainShape) → PathKey chain →
    (first second : ℕ → Option ℕ) → Fin (firstRank chain) → RingRow
  | none, _, _, _, r => r.elim0
  | some c, key, first, second, r =>
      ((Form.ofMatrix (ChainKey.first key) (fun col k => first (col * c.firstDegree + k)) r).sub
        (Form.recompose c.firstDegree c.firstRank second r)).row 0

theorem firstRow_below {bound : ℕ} : (chain : Option ChainShape) → (key : PathKey chain) →
    (first second : ℕ → Option ℕ) → (∀ position x, first position = some x → x < bound) →
    (∀ position x, second position = some x → x < bound) → ∀ r, (firstRow chain key first second
      r).Below bound
  | none, _, _, _, _, _, r => r.elim0
  | some _, _, _, _, firsts, seconds, _ =>
      ((Form.ofMatrix_below _ (fun _ _ => firsts _) _).sub (Form.recompose_below seconds _)).row _

theorem firstRow_degree_le : (chain : Option ChainShape) → (key : PathKey chain) →
    (first second : ℕ → Option ℕ) → ∀ r, (firstRow chain key first second r).degree ≤ chainDegree
      chain
  | none, _, _, _, r => r.elim0
  | some c, _, _, _, _ => le_max_left c.firstDegree c.secondDegree

theorem firstRow_degree : (chain : Option ChainShape) → (key : PathKey chain) →
    (first second : ℕ → Option ℕ) → ∀ r, (firstRow chain key first second r).degree = firstDegree
      chain
  | none, _, _, _, r => r.elim0
  | some _, _, _, _, _ => rfl

variable [NeZero d]

/-- Row `j` of the image `M·x̂` of a path: `M·x̂ = p` of Akita (48) without compression maps, and
`M·x̂ = G·ξ̂_1` of Akita (47) with them. `image j` is entry `j` of `M·x̂` as a sum (110). -/
noncomputable def imageRow : (chain : Option ChainShape) → (Fin n → Form d) → (first : ℕ → Option
  ℕ) →
    Payload d n chain → Fin n → RingRow
  | none, image, _, payload, j => (image j).row (payload j)
  | some _, image, first, _, j => ((image j).sub (Form.recompose d n first j)).row 0

theorem imageRow_below {bound : ℕ} : (chain : Option ChainShape) → (image : Fin n → Form d) →
    (first : ℕ → Option ℕ) → (payload : Payload d n chain) → (∀ j, (image j).Below bound) →
    (∀ position x, first position = some x → x < bound) → ∀ j, (imageRow chain image first payload
      j).Below bound
  | none, _, _, _, images, _, j => (images j).row _
  | some _, _, _, _, images, firsts, j => ((images j).sub (Form.recompose_below firsts j)).row _

theorem imageRow_degree : (chain : Option ChainShape) → (image : Fin n → Form d) →
    (first : ℕ → Option ℕ) → (payload : Payload d n chain) → ∀ j,
    (imageRow chain image first payload j).degree = d
  | none, _, _, _, _ => rfl
  | some _, _, _, _, _ => rfl

/-- The rows of a path hold for a flat witness that has the two digit sequences of the
compression maps at the positions `first` and `second`, when the rows (47), or the row (48), hold
for the digit vectors, and only then. `value j` is the value of the sum `image j`. -/
theorem rows_iff : (chain : Option ChainShape) → (key : PathKey chain) →
    (digits : PathDigits chain) → (payload : Payload d n chain) → (image : Fin n → Form d) →
    (value : Fin n → Rq d) → (first second : ℕ → Option ℕ) → (w : Table) →
    (∀ j, (image j).value w = value j) →
    (∀ position, (first position).elim 0 w = PathDigits.firstTable chain digits position) →
    (∀ position, (second position).elim 0 w = PathDigits.secondTable chain digits position) →
    ((∀ j, (imageRow chain image first payload j).Holds w) ∧
      (∀ r, (firstRow chain key first second r).Holds w) ∧
        (∀ r, (secondRow chain key second payload r).Holds w) ↔
      Rows chain key value digits payload)
  | none, _, _, payload, image, value, _, _, w, values, _, _ => by
      constructor
      · rintro ⟨images, -, -⟩
        exact funext fun j => (values j).symm.trans ((Form.row_holds _ _ _).mp (images j))
      · intro rows
        exact ⟨fun j => (Form.row_holds _ _ _).mpr ((values j).trans (congrFun rows j)),
          fun r => r.elim0, fun r => r.elim0⟩
  | some c, key, digits, payload, image, value, first, second, w, values, firsts, seconds => by
      have firstReads : ∀ col : Fin c.firstWidth, ofCoeffs (fun k => (first (col * c.firstDegree +
        k)).elim 0 w) =
          ChainDigits.first digits col := fun col =>
        ofCoeffs_eq fun k small => (firsts _).trans (sequence_apply _ col small)
      have secondReads : ∀ col : Fin c.secondWidth, ofCoeffs (fun k => (second (col *
        c.secondDegree + k)).elim 0 w) =
          ChainDigits.second digits col := fun col =>
        ofCoeffs_eq fun k small => (seconds _).trans (sequence_apply _ col small)
      have imageIff : ∀ j, (imageRow (some c) image first payload j).Holds w ↔
          value j = binaryRecompose d n (ChainDigits.first digits) j := fun j => by
        rw [imageRow, Form.row_holds, Form.value_sub, sub_eq_zero, values j,
          Form.value_recompose (ChainDigits.first digits) w first firsts j]
      have firstIff : ∀ r, (firstRow (some c) key first second r).Holds w ↔
          (ChainKey.first key).mulVec (ChainDigits.first digits) r =
            binaryRecompose c.firstDegree c.firstRank (ChainDigits.second digits) r := fun r => by
        rw [firstRow, Form.row_holds, Form.value_sub, sub_eq_zero]
        exact Eq.congr (Form.value_ofMatrix _ _ w firstReads r)
          (Form.value_recompose (ChainDigits.second digits) w second seconds r)
      have secondIff : ∀ r, (secondRow (some c) key second payload r).Holds w ↔
          (ChainKey.second key).mulVec (ChainDigits.second digits) r = payload r := fun r => by
        rw [secondRow, Form.row_holds]
        exact Eq.congr_left (Form.value_ofMatrix _ _ w secondReads r)
      exact and_congr ((forall_congr' imageIff).trans funext_iff.symm)
        (and_congr ((forall_congr' firstIff).trans funext_iff.symm)
          ((forall_congr' secondIff).trans funext_iff.symm))

end Path

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E : Type} [Field E] [Algebra Zq E]

section Rows

variable (O) (fold)

/-- The left side `Σ_i c_i·e_{i,t}` of plane `t` of the carrier row (102): the occurrence
`(i, u)` has the multiplier `c_i·b_1^u` and reads digit `u` of the coordinates of plane `t` of
partial `i` (Akita (58)). -/
noncomputable def partialForm (challenge : Fin S.blocks → Rq O.carrierDegree) (t : Fin O.planes) :
    Form O.carrierDegree where
  Occurrence := Fin S.blocks × Fin S.openingDigits
  multiplier entry := challenge entry.1 * (S.openingBase : Rq O.carrierDegree) ^ (entry.2 : ℕ)
  address entry j := O.partialPosition fold entry.1 entry.2 (t * O.carrierDegree + j)

/-- The right side `L_t(G_{b,M}·G_{b_z,m_A}·ẑ)` of plane `t` of the carrier row (102): the
occurrence `(x, a, u_com, u_fold)` has the multiplier
`[ẽq(r_pos, x)·ẽq(r_pack, a)]_t·b^{u_com}·b_z^{u_fold}` and reads the coefficients
`a + k·h_car·j` of the element `(x, u_com, u_fold)` of `ẑ` (Akita (57)). -/
noncomputable def contractForm (β : Module.Basis (Fin O.planes) Zq E) (point : Point O E)
    (t : Fin O.planes) : Form O.carrierDegree where
  Occurrence := Fin S.blockLength × Fin O.stride × Fin S.sourceDigits × Fin fold.responseDigits
  multiplier entry :=
    algebraMap Zq (Rq O.carrierDegree) (β.repr (point.position entry.1 * point.pack entry.2.1) t) *
      ((S.sourceBase : Rq O.carrierDegree) ^ (entry.2.2.1 : ℕ) *
        (fold.responseBase : Rq O.carrierDegree) ^ (entry.2.2.2 : ℕ))
  address entry j :=
    S.responseCell fold ((entry.1, entry.2.2.1), entry.2.2.2) (entry.2.1 + O.stride * j)

/-- Plane `t` of the carrier row (102) of Akita Figure 5, in `S = R_{q,d_car}`. -/
noncomputable def evaluationRow (β : Module.Basis (Fin O.planes) Zq E) (point : Point O E)
    (challenge : Fin S.blocks → Rq O.carrierDegree) (t : Fin O.planes) : RingRow :=
  ((partialForm O fold challenge t).sub (contractForm O fold β point t)).row 0

/-- Row `a` of `A·G_{b_z,m_A}·ẑ`: the occurrence `(x, u_com, u_fold)` has the multiplier
`A_{a,(x,u_com)}·b_z^{u_fold}` and reads this element of `ẑ`. -/
noncomputable def responseForm (key : Key S) (a : Fin S.innerRank) : Form S.innerDegree where
  Occurrence := (Fin S.blockLength × Fin S.sourceDigits) × Fin fold.responseDigits
  multiplier col := key.inner a col.1 * (fold.responseBase : Rq S.innerDegree) ^ (col.2 : ℕ)
  address col := S.responseCell fold col

/-- Row `a` of `Σ_i ι(c_i)·G_{b_1,n_A}·t̂_i`: the occurrence `(i, u)` has the multiplier
`ι(c_i)·b_1^u` and reads digit `u` of row `a` of `t̂_i`. -/
noncomputable def imageForm (challenge : Fin S.blocks → Rq O.carrierDegree) (a : Fin S.innerRank) :
    Form S.innerDegree where
  Occurrence := Fin S.blocks × Fin S.openingDigits
  multiplier entry := embed O.stride O.stride_tower (challenge entry.1) *
    (S.openingBase : Rq S.innerDegree) ^ (entry.2 : ℕ)
  address entry := O.innerCell fold entry.1 a entry.2

/-- Row `a` of the inner row (104) of Akita Figure 5, in `R_{q,d_A}`, as its left side minus its
right side (110), the sign of the pinned code. -/
noncomputable def innerRow (key : Key S) (challenge : Fin S.blocks → Rq O.carrierDegree)
    (a : Fin S.innerRank) : RingRow :=
  ((imageForm O fold challenge a).sub (responseForm fold key a)).row 0

/-- Entry `j` of the stacked image of `B` (Akita (45) and (105)): the row of `B` on the input of
its slice, in the `R_{q,d_B}` view of `t̂`. -/
def outerForm (key : Key S) (j : Fin (S.slices * S.outerRank)) : Form S.outerDegree :=
  Form.ofMatrix key.outer (O.sliceCell fold (finProdFinEquiv.symm j).1) (finProdFinEquiv.symm j).2

/-- Entry `j` of `D·ê` (Akita (106)). -/
def openForm (key : OpeningKey O) (j : Fin (1 * O.openRank)) : Form O.openDegree :=
  Form.ofMatrix key.matrix (O.partialCell fold) (finProdFinEquiv.symm j).2

/-- The weight of the cell `x` in the scalar row (100) of Akita Figure 5: the sum of
`ẽq(r_blk, i)·ẽq(r_tail, j)·b_1^u·β_t` over the digits `u` of the coordinates `(t, j)` of the
partials `i` that are at `x`. -/
noncomputable def scalarWeight (β : Module.Basis (Fin O.planes) Zq E) (point : Point O E)
    (x : ℕ) : E :=
  ∑ entry : Fin S.blocks × Fin O.carrierDegree × Fin O.planes × Fin S.openingDigits,
    if O.partialPosition fold entry.1 entry.2.2.2
        (entry.2.2.1 * O.carrierDegree + entry.2.1) = some x then
      point.block entry.1 * (point.tail entry.2.1 *
        ((S.openingBase : Zq) ^ (entry.2.2.2 : ℕ) • β entry.2.2.1))
    else 0

/-- The weight of the scalar row is 0 from the offset `chainStart`: the scalar row reads only
the segment `ê`. -/
theorem scalarWeight_of_le (β : Module.Basis (Fin O.planes) Zq E) (point : Point O E) {x : ℕ}
    (large : O.chainStart fold ≤ x) : scalarWeight O fold β point x = 0 :=
  Finset.sum_eq_zero fun entry _ => ite_eq_right fun same => by
    have below := O.partialPosition_lt_innerStart fold entry.1 entry.2.2.2
      (entry.2.2.1 * O.carrierDegree + entry.2.1) same
    unfold OpeningShape.chainStart at large
    omega

/-- The number of ring rows of one fold after the carrier row. -/
def OpeningShape.tailCount : ℕ :=
  S.innerRank + (S.slices * S.outerRank + (1 * O.openRank +
    (firstRank S.chain + (firstRank O.chain + (secondRank S.chain + secondRank O.chain)))))

/-- The number of ring rows over `Z_q` of one fold: each plane of the carrier row is one row. -/
def OpeningShape.planeRows : ℕ := O.planes + O.tailCount

/-- The number of rows over `E` of one fold before the scalar row: the carrier row (102) is one
row. This is the number of rows of the relation matrix of the pinned code
(`relation_matrix_row_count` in `crates/akita-params/src/layout/params.rs` of the Akita code). -/
def OpeningShape.rowCount : ℕ := 1 + O.tailCount

/-- `d_max`, the largest ring degree of a row (Akita Theorem 7.4). -/
def OpeningShape.largestDegree : ℕ :=
  max (max O.carrierDegree S.innerDegree)
    (max (max S.outerDegree O.openDegree) (max (chainDegree S.chain) (chainDegree O.chain)))

/-- The ring rows over `Z_q` of one fold in the order of Akita Figure 5: the planes of the carrier
row (102), the rows (104) of `A`, the rows (105) of `B` for each slice, the rows (106) of `D`, the
rows of `F_1`, the rows of `H_1`, the rows of `F_2` and the rows of `H_2` (47). -/
noncomputable def flatRows (β : Module.Basis (Fin O.planes) Zq E) (st : Statement O E)
    (challenge : Fin S.blocks → Rq O.carrierDegree) : Fin O.planeRows → RingRow :=
  Fin.append (evaluationRow O fold β st.point challenge)
    (Fin.append (innerRow O fold st.key challenge)
      (Fin.append (imageRow S.chain (outerForm O fold st.key) (O.firstCell fold) st.payload)
        (Fin.append
          (imageRow O.chain (openForm O fold st.openKey) (O.openFirstCell fold) st.openPayload)
          (Fin.append (firstRow S.chain st.key.chain (O.firstCell fold) (O.secondCell fold))
            (Fin.append
              (firstRow O.chain st.openKey.chain (O.openFirstCell fold) (O.openSecondCell fold))
              (Fin.append (secondRow S.chain st.key.chain (O.secondCell fold) st.payload)
                (secondRow O.chain st.openKey.chain (O.openSecondCell fold)
                  st.openPayload)))))))

theorem flatRows_below (β : Module.Basis (Fin O.planes) Zq E) (st : Statement O E)
    (challenge : Fin S.blocks → Rq O.carrierDegree) (r : Fin O.planeRows) :
    (flatRows O fold β st challenge r).Below (O.cells fold) := by
  revert r
  refine (forall_append (RingRow.Below · (O.cells fold)) _ _).mpr ⟨fun t => ?_,
    (forall_append (RingRow.Below · (O.cells fold)) _ _).mpr ⟨fun a => ?_,
    (forall_append (RingRow.Below · (O.cells fold)) _ _).mpr ⟨?_,
    (forall_append (RingRow.Below · (O.cells fold)) _ _).mpr ⟨?_,
    (forall_append (RingRow.Below · (O.cells fold)) _ _).mpr ⟨?_,
    (forall_append (RingRow.Below · (O.cells fold)) _ _).mpr ⟨?_,
    (forall_append (RingRow.Below · (O.cells fold)) _ _).mpr ⟨?_, ?_⟩⟩⟩⟩⟩⟩⟩
  · exact (Form.Below.sub (f := partialForm O fold challenge t)
      (g := contractForm O fold β st.point t)
      (fun _ _ _ => O.partialPosition_lt fold _ _ _) (fun _ _ _ => O.responseCell_lt fold _ _)).row
        0
  · exact (Form.Below.sub (f := imageForm O fold challenge a) (g := responseForm fold st.key a)
      (fun _ _ _ => O.innerCell_lt fold _ _ _ _) (fun _ _ _ => O.responseCell_lt fold _ _)).row 0
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

theorem flatRows_degree_le (β : Module.Basis (Fin O.planes) Zq E) (st : Statement O E)
    (challenge : Fin S.blocks → Rq O.carrierDegree) (r : Fin O.planeRows) :
    (flatRows O fold β st challenge r).degree ≤ O.largestDegree := by
  revert r
  refine (forall_append (RingRow.degree · ≤ O.largestDegree) _ _).mpr ⟨fun t => ?_,
    (forall_append (RingRow.degree · ≤ O.largestDegree) _ _).mpr ⟨fun a => ?_,
    (forall_append (RingRow.degree · ≤ O.largestDegree) _ _).mpr ⟨fun j => ?_,
    (forall_append (RingRow.degree · ≤ O.largestDegree) _ _).mpr ⟨fun j => ?_,
    (forall_append (RingRow.degree · ≤ O.largestDegree) _ _).mpr ⟨fun r => ?_,
    (forall_append (RingRow.degree · ≤ O.largestDegree) _ _).mpr ⟨fun r => ?_,
    (forall_append (RingRow.degree · ≤ O.largestDegree) _ _).mpr
      ⟨fun r => ?_, fun r => ?_⟩⟩⟩⟩⟩⟩⟩
  · exact le_max_of_le_left (le_max_left O.carrierDegree S.innerDegree)
  · exact le_max_of_le_left (le_max_right O.carrierDegree S.innerDegree)
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

/-- The sum of the ring degrees of the rows of one fold, in the order of `flatRows`. -/
def OpeningShape.degreeSum : ℕ :=
  O.planes * O.carrierDegree + (S.innerRank * S.innerDegree +
    (S.slices * S.outerRank * S.outerDegree + (1 * O.openRank * O.openDegree +
      (firstRank S.chain * firstDegree S.chain + (firstRank O.chain * firstDegree O.chain +
        (secondRank S.chain * secondDegree S.chain +
          secondRank O.chain * secondDegree O.chain))))))

theorem flatRows_degree_sum (β : Module.Basis (Fin O.planes) Zq E) (st : Statement O E)
    (challenge : Fin S.blocks → Rq O.carrierDegree) :
    ∑ r, (flatRows O fold β st challenge r).degree = O.degreeSum := by
  have planes : ∀ t, (evaluationRow O fold β st.point challenge t).degree = O.carrierDegree :=
    fun _ => rfl
  have inner : ∀ a, (innerRow O fold st.key challenge a).degree = S.innerDegree := fun _ => rfl
  unfold flatRows OpeningShape.degreeSum OpeningShape.planeRows OpeningShape.tailCount
  simp only [sum_append RingRow.degree, planes, inner, imageRow_degree, firstRow_degree,
    secondRow_degree, Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]

/-- The relation of one fold with coefficient packing on the flat witness `Witness.flat`, for
the fold challenges `c_i` (Akita Figure 5 for one group and one claim, on the quotient-free
route): the ring rows `flatRows`, with the planes of the carrier row joined into one row over `E`
by the basis `β`, and the scalar row (100). The cube has `μ` variables, with
`cells < 2^μ`. -/
noncomputable def flatRelation (β : Module.Basis (Fin O.planes) Zq E) (st : Statement O E)
    (challenge : Fin S.blocks → Rq O.carrierDegree) (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ) :
    FlatRelation E where
  vars := μ
  low := O.low
  ring := O.planeRows
  row := flatRows O fold β st challenge
  inside r c j _ _ read := (flatRows_below O fold β st challenge r c j _ read).trans_le fits
  join := (Join.all β).append (Join.plain E O.tailCount)
  independent := Join.append_independent (Join.all_independent β.linearIndependent)
    (Join.plain_independent (algebraMap Zq E).injective _)
  weight := scalarWeight O fold β st.point
  value := st.value

end Rows

section Values

variable (w : Witness O fold)

theorem partialForm_value (challenge : Fin S.blocks → Rq O.carrierDegree) (t : Fin O.planes) :
    (partialForm O fold challenge t).value w.flat =
      ∑ i, challenge i * O.partialPlane w.partials i t := by
  have plane : ∀ i, O.partialPlane w.partials i t = ∑ u : Fin S.openingDigits,
      (S.openingBase : Rq O.carrierDegree) ^ (u : ℕ) *
        ofCoeffs fun j => (O.partialPosition fold i u (t * O.carrierDegree + j)).elim 0 w.flat :=
    fun i => ext_coeff fun j small => by
      rw [w.coeff_partialPlane i t ⟨j, small⟩, coeff_sum]
      exact Finset.sum_congr rfl fun u _ => by
        rw [coeff_natCast_pow_mul, coeff_ofCoeffs _ small]
  show (∑ entry : Fin S.blocks × Fin S.openingDigits,
    challenge entry.1 * (S.openingBase : Rq O.carrierDegree) ^ (entry.2 : ℕ) *
      ofCoeffs fun j => (O.partialPosition fold entry.1 entry.2
        (t * O.carrierDegree + j)).elim 0 w.flat) = _
  rw [Fintype.sum_prod_type]
  simp only [plane, Finset.mul_sum, mul_assoc]

theorem contractForm_value (β : Module.Basis (Fin O.planes) Zq E) (point : Point O E)
    (t : Fin O.planes) :
    (contractForm O fold β point t).value w.flat =
      contract β point t (gadget S.sourceBase (gadget fold.responseBase w.response)) := by
  have reads : ∀ entry :
      Fin S.blockLength × Fin O.stride × Fin S.sourceDigits × Fin fold.responseDigits,
      ofCoeffs (fun j => (S.responseCell fold ((entry.1, entry.2.2.1), entry.2.2.2)
        (entry.2.1 + O.stride * j)).elim 0 w.flat) =
        plane O.carrierDegree O.stride entry.2.1
          (w.response ((entry.1, entry.2.2.1), entry.2.2.2)) :=
    fun entry => ofCoeffs_eq fun j small => by
      rw [coeff_plane _ _ _ small,
        w.flat_responseCell _ (plane_address_lt O.stride_tower entry.2.1.isLt small)]
  rw [contractForm, Form.value_mk _ _ w.flat reads]
  simp only [contract_apply, gadget, map_sum, plane_natCast_pow_mul,
    Algebra.smul_def, Finset.mul_sum, Fintype.sum_prod_type, mul_assoc]

theorem evaluationRow_holds (β : Module.Basis (Fin O.planes) Zq E) (point : Point O E)
    (challenge : Fin S.blocks → Rq O.carrierDegree) (t : Fin O.planes) :
    (evaluationRow O fold β point challenge t).Holds w.flat ↔
      ∑ i, challenge i * O.partialPlane w.partials i t =
        contract β point t (gadget S.sourceBase (gadget fold.responseBase w.response)) := by
  rw [evaluationRow, Form.row_holds, Form.value_sub, sub_eq_zero, partialForm_value,
    contractForm_value]

theorem responseForm_value (key : Key S) (a : Fin S.innerRank) :
    (responseForm fold key a).value w.flat =
      key.inner.mulVec (gadget fold.responseBase w.response) a := by
  rw [responseForm, Form.value_mk _ _ w.flat fun col =>
    ofCoeffs_eq fun _ small => w.flat_responseCell col small]
  simp only [Matrix.mulVec, dotProduct, gadget, Finset.mul_sum,
    Fintype.sum_prod_type, mul_assoc]

theorem imageForm_value (challenge : Fin S.blocks → Rq O.carrierDegree) (a : Fin S.innerRank) :
    (imageForm O fold challenge a).value w.flat =
      (∑ i, embed O.stride O.stride_tower (challenge i) • gadget S.openingBase (w.inner i)) a := by
  rw [imageForm, Form.value_mk _ _ w.flat
    (x := fun entry : Fin S.blocks × Fin S.openingDigits => w.inner entry.1 (a, entry.2))
    fun entry : Fin S.blocks × Fin S.openingDigits =>
      ofCoeffs_eq fun _ small => w.flat_innerCell entry.1 a entry.2 small]
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, gadget, Finset.mul_sum,
    Fintype.sum_prod_type, mul_assoc]

theorem innerRow_holds (key : Key S) (challenge : Fin S.blocks → Rq O.carrierDegree)
    (a : Fin S.innerRank) :
    (innerRow O fold key challenge a).Holds w.flat ↔
      key.inner.mulVec (gadget fold.responseBase w.response) a =
        (∑ i, embed O.stride O.stride_tower (challenge i) • gadget S.openingBase (w.inner i)) a := by
  rw [innerRow, Form.row_holds, Form.value_sub, sub_eq_zero, responseForm_value, imageForm_value]
  exact eq_comm

theorem outerForm_value (key : Key S) (j : Fin (S.slices * S.outerRank)) :
    (outerForm O fold key j).value w.flat = stack key.outer (S.sliceInput w.inner) j :=
  Form.value_ofMatrix _ _ _ (fun entry => w.read_sliceCell _ entry) _

theorem openForm_value (key : OpeningKey O) (j : Fin (1 * O.openRank)) :
    (openForm O fold key j).value w.flat = stack key.matrix (fun _ : Fin 1 => w.partials) j :=
  Form.value_ofMatrix _ _ _ (fun col => ofCoeffs_eq fun _ small => w.flat_partialCell col small) _

/-- The scalar row of the flat witness is the left side of the scalar opening (59) of Akita. -/
theorem scalarWeight_sum (β : Module.Basis (Fin O.planes) Zq E) (point : Point O E) (μ : ℕ)
    (fits : O.cells fold ≤ 2 ^ μ) :
    ∑ x ∈ Finset.range (2 ^ μ), algebraMap Zq E (w.flat x) * scalarWeight O fold β point x =
      scalar β point (O.partialPlane w.partials) := by
  unfold scalarWeight
  rw [sum_mul_indicator _ _ _ _ fun entry _ read => (O.partialPosition_lt fold _ _ _ read).trans_le
    fits]
  simp only [elim_algebraMap, Fintype.sum_prod_type, scalar, Finset.mul_sum, w.coeff_partialPlane,
    Finset.sum_smul]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ =>
    Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun u _ => ?_
  simp only [Algebra.smul_def, map_mul]
  ring

end Values

section Link

variable (β : Module.Basis (Fin O.planes) Zq E) (st : Statement O E)
  (challenge : Fin S.blocks → Rq O.carrierDegree) (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ)

/-- The flat witness satisfies the relation of one fold when the digit vectors satisfy the rows
(102), (104), (105) and (106) with their rows (47) or (48), and the scalar row (100), of Akita
Figure 5, and only then. -/
theorem flatRelation_holds_iff (w : Witness O fold) :
    (flatRelation O fold β st challenge μ fits).Holds w.flat ↔
      (∀ t, ∑ i, challenge i * O.partialPlane w.partials i t =
        contract β st.point t (gadget S.sourceBase (gadget fold.responseBase w.response))) ∧
      st.key.inner.mulVec (gadget fold.responseBase w.response) =
        ∑ i, embed O.stride O.stride_tower (challenge i) • gadget S.openingBase (w.inner i) ∧
      Rows S.chain st.key.chain (stack st.key.outer (S.sliceInput w.inner)) w.chain st.payload ∧
      OpeningRows st.openKey w.partials w.openChain st.openPayload ∧
      scalar β st.point (O.partialPlane w.partials) = st.value := by
  have outer := rows_iff S.chain st.key.chain w.chain st.payload (outerForm O fold st.key)
    (stack st.key.outer (S.sliceInput w.inner)) (O.firstCell fold) (O.secondCell fold) w.flat
    (outerForm_value w st.key) w.flat_firstCell w.flat_secondCell
  have opening := rows_iff O.chain st.openKey.chain w.openChain st.openPayload
    (openForm O fold st.openKey) (stack st.openKey.matrix fun _ : Fin 1 => w.partials)
    (O.openFirstCell fold) (O.openSecondCell fold) w.flat (openForm_value w st.openKey)
    w.flat_openFirstCell w.flat_openSecondCell
  have rows : (∀ r, (flatRows O fold β st challenge r).Holds w.flat) ↔
      (∀ t, (evaluationRow O fold β st.point challenge t).Holds w.flat) ∧
      (∀ a, (innerRow O fold st.key challenge a).Holds w.flat) ∧
      (∀ j, (imageRow S.chain (outerForm O fold st.key) (O.firstCell fold) st.payload j).Holds
        w.flat) ∧
      (∀ j, (imageRow O.chain (openForm O fold st.openKey) (O.openFirstCell fold)
        st.openPayload j).Holds w.flat) ∧
      (∀ r, (firstRow S.chain st.key.chain (O.firstCell fold) (O.secondCell fold) r).Holds
        w.flat) ∧
      (∀ r, (firstRow O.chain st.openKey.chain (O.openFirstCell fold) (O.openSecondCell fold)
        r).Holds w.flat) ∧
      (∀ r, (secondRow S.chain st.key.chain (O.secondCell fold) st.payload r).Holds w.flat) ∧
      ∀ r, (secondRow O.chain st.openKey.chain (O.openSecondCell fold) st.openPayload r).Holds
        w.flat :=
    (forall_append (RingRow.Holds · w.flat) _ _).trans (and_congr_right'
      ((forall_append (RingRow.Holds · w.flat) _ _).trans (and_congr_right'
        ((forall_append (RingRow.Holds · w.flat) _ _).trans (and_congr_right'
          ((forall_append (RingRow.Holds · w.flat) _ _).trans (and_congr_right'
            ((forall_append (RingRow.Holds · w.flat) _ _).trans (and_congr_right'
              ((forall_append (RingRow.Holds · w.flat) _ _).trans (and_congr_right'
                (forall_append (RingRow.Holds · w.flat) _ _))))))))))))
  have value := scalarWeight_sum w β st.point μ fits
  constructor
  · rintro ⟨ring, scalarRow⟩
    obtain ⟨evaluation, inner, outerImage, openImage, outerFirst, openFirst, outerSecond,
      openSecond⟩ := rows.mp ring
    exact ⟨fun t => (evaluationRow_holds w β st.point challenge t).mp (evaluation t),
      funext fun a => (innerRow_holds w st.key challenge a).mp (inner a),
      outer.mp ⟨outerImage, outerFirst, outerSecond⟩,
      opening.mp ⟨openImage, openFirst, openSecond⟩, value.symm.trans scalarRow⟩
  · rintro ⟨evaluation, inner, outerRows, openRows, scalarRow⟩
    obtain ⟨outerImage, outerFirst, outerSecond⟩ := outer.mpr outerRows
    obtain ⟨openImage, openFirst, openSecond⟩ := opening.mpr openRows
    exact ⟨rows.mpr ⟨fun t => (evaluationRow_holds w β st.point challenge t).mpr (evaluation t),
      fun a => (innerRow_holds w st.key challenge a).mpr (congrFun inner a),
      outerImage, openImage, outerFirst, openFirst, outerSecond, openSecond⟩,
      value.trans scalarRow⟩

variable {β st challenge}

/-- The premise `encodes` of `foldCheck_satisfies` for `flatRelation`: a witness with the
certified alphabets whose flat witness satisfies the relation of one fold satisfies the exact
branch relation `Witness.Satisfies` (the hypothesis of Akita Corollary 10.32). -/
theorem Witness.satisfies_of_flat {w : Witness O fold} (alphabets : w.Alphabets)
    (holds : (flatRelation O fold β st challenge μ fits).Holds w.flat) :
    w.Satisfies β st challenge := by
  obtain ⟨evaluation, inner, outerRows, openRows, value⟩ :=
    (flatRelation_holds_iff β st challenge μ fits w).mp holds
  exact ⟨⟨outerRows, openRows, value, alphabets.innerDigits, alphabets.chainDigits,
    alphabets.partialDigits, alphabets.openDigits⟩, evaluation, inner, alphabets.responseDigits⟩

/-- The honest direction: the flat witness of digit vectors that satisfy the exact branch
relation satisfies the relation of one fold (Akita Theorem 7.4, item 1, for the rows). -/
theorem Witness.Satisfies.flat {w : Witness O fold} (satisfies : w.Satisfies β st challenge) :
    (flatRelation O fold β st challenge μ fits).Holds w.flat :=
  (flatRelation_holds_iff β st challenge μ fits w).mpr ⟨satisfies.evaluation, satisfies.innerRows,
    satisfies.carried.rows, satisfies.carried.openRows, satisfies.carried.value⟩

variable [Fintype E] (β st challenge)

/-- The quotient-free reduction of the relation of one fold (Akita Section 7.3 and
Theorem 7.4), with `2^batch` places for the rows. Its bad set has at most `d_max − 1` points. -/
noncomputable def flatReduction (batch : ℕ) (rowsFit : O.rowCount + 1 ≤ 2 ^ batch) :
    Reduction E :=
  (flatRelation O fold β st challenge μ fits).free O.largestDegree
    batch (flatRows_degree_le O fold β st challenge) rowsFit

variable [DecidableEq E]

/-- `foldCheck_satisfies` for the flat witness `Witness.flat` and the relation `flatRelation`,
with no layout and no premise `encodes`: on a path of `foldCheck` that avoids the bad sets, an
accepted claim that is true for the flat witness gives the exact branch relation
`Witness.Satisfies` for the fold challenges (the hypothesis of Akita Corollary 10.32 for one
child). -/
theorem flatFold_satisfies (small : S.certifiedBase < fieldModulus) (batch : ℕ)
    (rowsFit : O.rowCount + 1 ≤ 2 ^ batch) (w : Witness O fold) :
    (foldCheck (flatReduction β st challenge μ fits batch rowsFit) S.certifiedBase
      (O.binarySupport fold) (O.compressed fold) w.flat).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E w.flat) →
        w.Satisfies β st challenge :=
  foldCheck_satisfies (flatReduction β st challenge μ fits batch rowsFit)
    (O.flatLayout fold μ fits) small β st challenge
    (fun _ alphabets holds => Witness.satisfies_of_flat μ fits alphabets holds) w

/-- For each prover, the probability that the checks of a packing fold end in a true claim for a
flat witness whose digit vectors do not satisfy the exact branch relation is at most `foldError`
of the quotient-free reduction. -/
theorem flatFold_satisfies_error (small : S.certifiedBase < fieldModulus) (batch : ℕ)
    (rowsFit : O.rowCount + 1 ≤ 2 ^ batch) (w : Witness O fold)
    (s : Strategy (Option (List E × E))) :
    prob s (foldCheck (flatReduction β st challenge μ fits batch rowsFit) S.certifiedBase
      (O.binarySupport fold) (O.compressed fold) w.flat) (fun result =>
      ¬ ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E w.flat) →
        w.Satisfies β st challenge) ≤
      foldError (flatReduction β st challenge μ fits batch rowsFit) S.certifiedBase
        (O.compressed fold) :=
  foldCheck_satisfies_error (flatReduction β st challenge μ fits batch rowsFit)
    (O.flatLayout fold μ fits) small β st challenge
    (fun _ alphabets holds => Witness.satisfies_of_flat μ fits alphabets holds) w s

end Link

end NightstreamFPrime.Spec.Compression.Akita
