import NightstreamFPrime.Spec.Compression.Akita.Binding
import NightstreamFPrime.Spec.Compression.Akita.Challenge
import NightstreamFPrime.Spec.Compression.Akita.Opening

/-! One fold of Akita with direct coefficient packing, for one claim in one group, on the
coefficient-norm route (ePrint 2026/1983, revision of 2026-10-15, Sections 5.4, 5.5 and 7.2, and
Definition 10.22, Lemma 10.23 and Corollary 10.32).

- `vecNormInf_gadget_le_of_balanced` and `vecNormInf_gadget_sub_le` are the envelopes (77) and
  (78) of a recomposition of digits in a certified alphabet, and `Descriptor.sourceBound_eq` states
  that (78) is `Descriptor.sourceBound`.
- `response` is the folded response (73). `Statement` has the public data of one fold, `Carried`
  has the digit vectors that do not depend on the fold challenge, and `Witness` adds the response
  digits `ẑ`. `Witness.Satisfies` is the exact branch relation of Akita Section 10 for one
  challenge vector: the rows (102), (104), (105), (106) and (100) of Akita Figure 5 with the
  alphabets.
- `Weak` and `Weak.Opens` are the weak opening of Akita Definition 10.22, conditions (i) to (v).
- `weakCollision` and `weakCollision_isSome` are Akita Lemma 10.23 with the bound (174), and
  `weakCollision_isSome_of_bounds` has the bound (176).
- `Fork` has two accepting children for each block, `extract` is the weak opening of the proof of
  Akita Theorem 10.31, and `extract_opens_or_collision` is Akita Corollary 10.32 for one claim. -/

namespace NightstreamFPrime.Spec.Compression.Akita

section Envelope

variable {d : ℕ} [NeZero d] {ι : Type} {δ : ℕ}

theorem magnitude_sum_le {κ : Type} (s : Finset κ) (f : κ → Zq) :
    magnitude (∑ u ∈ s, f u) ≤ ∑ u ∈ s, magnitude (f u) :=
  Finset.le_sum_of_subadditive magnitude magnitude_zero.le magnitude_add_le s f

theorem magnitude_natCast_pow_le (b u : ℕ) : magnitude ((b : Zq) ^ u) ≤ b ^ u := by
  have cast : (b : Zq) ^ u = (((b ^ u : ℕ) : ℤ) : Zq) := by
    rw [Int.cast_natCast, Nat.cast_pow]
  rw [cast]
  exact (magnitude_intCast_le _).trans_eq (Int.natAbs_natCast _)

omit [NeZero d] in
theorem gadget_sub (b : ℕ) (digits digits' : ι × Fin δ → Rq d) :
    gadget b (digits - digits') = gadget b digits - gadget b digits' := by
  funext j
  simp only [gadget, Pi.sub_apply, mul_sub, Finset.sum_sub_distrib]

omit [NeZero d] in
theorem gadget_smul (b : ℕ) (c : Rq d) (digits : ι × Fin δ → Rq d) :
    gadget b (c • digits) = c • gadget b digits := by
  funext j
  simp only [gadget, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
  exact Finset.sum_congr rfl fun u _ => mul_left_comm _ _ _

omit [NeZero d] in
theorem gadget_sum {κ : Type} (b : ℕ) (s : Finset κ) (f : κ → ι × Fin δ → Rq d) :
    gadget b (∑ i ∈ s, f i) = ∑ i ∈ s, gadget b (f i) := by
  funext j
  simp only [gadget, Finset.sum_apply, Finset.mul_sum]
  exact Finset.sum_comm

/-- The recomposition `G_{b,n} ẑ` of a digit vector with `‖ẑ‖_∞ ≤ β` has
`‖G_{b,n} ẑ‖_∞ ≤ β · Σ_{u<δ} b^u`. The bound has no condition on `q`. -/
theorem vecNormInf_gadget_le [Fintype ι] (b : ℕ) {bound : ℕ} {digits : ι × Fin δ → Rq d}
    (short : vecNormInf digits ≤ bound) :
    vecNormInf (gadget b digits) ≤ bound * ∑ u ∈ Finset.range δ, b ^ u := by
  refine (vecNormInf_le_iff _ _).mpr fun j => (normInf_le_iff _ _).mpr fun i small => ?_
  rw [gadget, coeff_sum]
  simp only [coeff_natCast_pow_mul]
  refine (magnitude_sum_le _ _).trans ?_
  rw [Finset.mul_sum, ← Fin.sum_univ_eq_sum_range (fun u => bound * b ^ u) δ]
  refine Finset.sum_le_sum fun u _ => ?_
  rw [Nat.mul_comm]
  exact (magnitude_mul_le _ _).trans (Nat.mul_le_mul (magnitude_natCast_pow_le b u)
    ((normInf_le_iff _ _).mp ((normInf_le_vecNormInf digits (j, u)).trans short) i small))

/-- Akita (77): the recomposition in base `b` of `δ` digit planes of the certified alphabet
`A_{b⋆}` has the coefficient bound `β^cert_fold = (b⋆/2) · Σ_{u<δ} b^u`. -/
theorem vecNormInf_gadget_le_of_balanced [Fintype ι] (b : ℕ) {certified : ℕ}
    {digits : ι × Fin δ → Rq d} (balanced : Balanced certified digits) :
    vecNormInf (gadget b digits) ≤ certified / 2 * ∑ u ∈ Finset.range δ, b ^ u :=
  vecNormInf_gadget_le b balanced.vecNormInf_le

/-- Akita (78): two recompositions in base `b` of `δ` digit planes of the certified alphabet
`A_{b⋆}` differ by at most `(b⋆ − 1) · Σ_{u<δ} b^u` in each coefficient. -/
theorem vecNormInf_gadget_sub_le [Fintype ι] (b : ℕ) {certified : ℕ}
    {digits digits' : ι × Fin δ → Rq d} (left : Balanced certified digits)
    (right : Balanced certified digits') :
    vecNormInf (gadget b digits - gadget b digits') ≤
      (certified - 1) * ∑ u ∈ Finset.range δ, b ^ u := by
  rw [← gadget_sub]
  exact vecNormInf_gadget_le b ((vecNormInf_sub_le_of_digits left right).trans (by omega))

/-- The bound of Akita (78) is `Δ^cert_f` of Akita (50) for a response base `b_z ≥ 2`. -/
theorem Descriptor.sourceBound_eq (fold : Descriptor) (S : Shape) (two : 2 ≤ fold.responseBase) :
    fold.sourceBound S =
      (S.certifiedBase - 1) * ∑ u ∈ Finset.range fold.responseDigits, fold.responseBase ^ u := by
  rw [Descriptor.sourceBound, Nat.geomSum_eq two]

/-- Two sums with the coefficients `a` and `a'`, which agree outside the position `i`, differ by
the term of `i` (the cancellation of Akita Theorem 10.31, "Coordinate siblings"). -/
theorem sum_smul_sub_sum_smul {κ A M : Type} [Fintype κ] [Ring A] [AddCommGroup M] [Module A M]
    (a a' : κ → A) (v : κ → M) (i : κ) (agree : ∀ j, j ≠ i → a j = a' j) :
    ∑ j, a j • v j - ∑ j, a' j • v j = (a i - a' i) • v i := by
  rw [← Finset.sum_sub_distrib, Finset.sum_eq_single i]
  · rw [sub_smul]
  · intro j _ other
    rw [agree j other, sub_self]
  · intro absent
    exact absurd (Finset.mem_univ i) absent

end Envelope

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor}

/-- The digits `ẑ ∈ R_{q,d_A}^{m_A·δ_f}` of a folded response, with `m_A = M·δ_com`
(Akita (74)). -/
abbrev Shape.Response (S : Shape) (fold : Descriptor) : Type :=
  (Fin S.blockLength × Fin S.sourceDigits) × Fin fold.responseDigits → Rq S.innerDegree

/-- The folded response `z = Σ_i ι(c_i)·s_i` of one claim (Akita (73)). -/
noncomputable def response (O : OpeningShape S) (challenge : Fin S.blocks → Rq O.carrierDegree)
    (source : Fin S.blocks → S.Source) : S.Source :=
  ∑ i, embed O.stride O.stride_tower (challenge i) • source i

/-- The public data of one fold for one claim (Akita Definition 10.22, "The fixed folding
statement"): the matrices of the commitment and of the opening path, the payloads `u_pub` and
`v_pub`, the weights of the opening point and the claimed value `v`. -/
structure Statement (O : OpeningShape S) (E : Type) where
  key : Key S
  openKey : OpeningKey O
  payload : S.Payload
  openPayload : O.Payload
  point : Point O E
  value : E

/-- The digit vectors of one fold that do not depend on the fold challenge: the digits `t̂_i` of
the inner images, the digits of `F_1`, `F_2`, the digits `ê` of the partials and the digits of
`H_1`, `H_2` (Akita (75)). -/
@[ext] structure Carried (O : OpeningShape S) where
  inner : Fin S.blocks → S.InnerDigits
  chain : PathDigits S.chain
  partials : O.Partials
  openChain : PathDigits O.chain

/-- The digit vectors `ẑ`, `ê`, `t̂` and the chain digits of one fold (Akita (75)). -/
structure Witness (O : OpeningShape S) (fold : Descriptor) extends Carried O where
  response : S.Response fold

/-- A solution of Module-SIS for `B` with the bound `b⋆ − 1`, or for `F_1` or `F_2` with the
bound 1 (Akita Table 3). -/
abbrev OuterBreak (key : Key S) : Type :=
  ShortKernel key.outer (S.certifiedBase - 1) ⊕ PathKernel S.chain key.chain

/-- A solution of Module-SIS for `D` with the bound `η_D = b⋆ − 1`, or for `H_1` or `H_2` with the
bound 1 (Akita Table 3). -/
abbrev OpenBreak (key : OpeningKey O) : Type :=
  ShortKernel key.matrix (S.certifiedBase - 1) ⊕ PathKernel O.chain key.chain

/-- The reduction of Akita Lemma 4.3 for the path of `B`, `F_1` and `F_2`. -/
noncomputable def outerCollision (key : Key S) (inner inner' : Fin S.blocks → S.InnerDigits)
    (chain chain' : PathDigits S.chain) : Option (OuterBreak key) :=
  chainCollision S.chain key.chain key.outer (S.certifiedBase - 1) (S.sliceInput inner)
    (S.sliceInput inner') chain chain'

theorem outerCollision_isSome {key : Key S} {inner inner' : Fin S.blocks → S.InnerDigits}
    {chain chain' : PathDigits S.chain} {payload : S.Payload}
    (rows : Rows S.chain key.chain (stack key.outer (S.sliceInput inner)) chain payload)
    (rows' : Rows S.chain key.chain (stack key.outer (S.sliceInput inner')) chain' payload)
    (digits : ∀ i, Balanced S.certifiedBase (inner i))
    (digits' : ∀ i, Balanced S.certifiedBase (inner' i))
    (binary : PathDigits.Binary S.chain chain) (binary' : PathDigits.Binary S.chain chain')
    (different : inner ≠ inner' ∨ chain ≠ chain') :
    (outerCollision key inner inner' chain chain').isSome :=
  chainCollision_isSome rows rows'
    (S.sliceInput_sub_le fun i => S.outerView_sub_le (digits i) (digits' i)) binary binary'
    (different.imp_left fun differs same => differs (S.sliceInput_injective same))

/-- The reduction of Akita Lemma 4.3 for the path of `D`, `H_1` and `H_2`, which has one slice. -/
noncomputable def openCollision (key : OpeningKey O) (partials partials' : O.Partials)
    (chain chain' : PathDigits O.chain) : Option (OpenBreak key) :=
  chainCollision O.chain key.chain key.matrix (S.certifiedBase - 1) (fun _ : Fin 1 => partials)
    (fun _ => partials') chain chain'

theorem openCollision_isSome {key : OpeningKey O} {partials partials' : O.Partials}
    {chain chain' : PathDigits O.chain} {payload : O.Payload}
    (rows : OpeningRows key partials chain payload)
    (rows' : OpeningRows key partials' chain' payload)
    (digits : Balanced S.certifiedBase partials) (digits' : Balanced S.certifiedBase partials')
    (binary : PathDigits.Binary O.chain chain) (binary' : PathDigits.Binary O.chain chain')
    (different : partials ≠ partials' ∨ chain ≠ chain') :
    (openCollision key partials partials' chain chain').isSome :=
  chainCollision_isSome rows rows'
    (fun _ => (vecNormInf_sub_le_of_digits digits digits').trans (by omega)) binary binary'
    (different.imp_left fun differs same => differs (congrFun same 0))

/-- The reduction of Akita Lemma 10.23, "Digit paths": a solution for the path of `B`, or else for
the path of `D`, from two families of carried digits. -/
noncomputable def carriedCollision {E : Type} (st : Statement O E) (digits digits' : Carried O) :
    Option (OuterBreak st.key ⊕ OpenBreak st.openKey) :=
  ((outerCollision st.key digits.inner digits'.inner digits.chain digits'.chain).map Sum.inl).or
    ((openCollision st.openKey digits.partials digits'.partials digits.openChain
      digits'.openChain).map Sum.inr)

variable {E : Type} [CommRing E] [Module Zq E]

/-- The rows of Akita Figure 5 that have no fold challenge, with their alphabets: the rows (105)
of `B` with the rows (47) or (48) to `u_pub`, the row (106) of `D` with the rows (47) or (48) to
`v_pub`, the scalar row (100) for one claim, which is Akita (59), the alphabet `A_{b⋆}` of `t̂` and
`ê`, and the alphabet `{−1, 0}` of the chain digits. -/
structure Carried.Opens (β : Module.Basis (Fin O.planes) Zq E) (digits : Carried O)
    (st : Statement O E) : Prop where
  rows : Rows S.chain st.key.chain (stack st.key.outer (S.sliceInput digits.inner)) digits.chain
    st.payload
  openRows : OpeningRows st.openKey digits.partials digits.openChain st.openPayload
  value : scalar β st.point (O.partialPlane digits.partials) = st.value
  innerDigits : ∀ i, Balanced S.certifiedBase (digits.inner i)
  chainDigits : PathDigits.Binary S.chain digits.chain
  partialDigits : Balanced S.certifiedBase digits.partials
  openDigits : PathDigits.Binary O.chain digits.openChain

/-- Akita Lemma 10.23, "Digit paths": two families of carried digits that satisfy the rows of one
statement and that differ give a solution of Module-SIS for `B` or `D` with the bound `b⋆ − 1`,
or for a compression map with the bound 1. -/
theorem carriedCollision_isSome {β : Module.Basis (Fin O.planes) Zq E} {st : Statement O E}
    {digits digits' : Carried O} (opens : digits.Opens β st) (opens' : digits'.Opens β st)
    (different : digits ≠ digits') : (carriedCollision st digits digits').isSome := by
  refine isSome_or_map.mpr ?_
  by_cases outer : digits.inner ≠ digits'.inner ∨ digits.chain ≠ digits'.chain
  · exact Or.inl (outerCollision_isSome opens.rows opens'.rows opens.innerDigits
      opens'.innerDigits opens.chainDigits opens'.chainDigits outer)
  · refine Or.inr (openCollision_isSome opens.openRows opens'.openRows opens.partialDigits
      opens'.partialDigits opens.openDigits opens'.openDigits ?_)
    by_contra same
    obtain ⟨inner, chain⟩ := not_or.mp outer
    obtain ⟨partials, openChain⟩ := not_or.mp same
    exact different (Carried.ext (not_not.mp inner) (not_not.mp chain) (not_not.mp partials)
      (not_not.mp openChain))

/-- The exact branch relation of one fold for the challenges `c_i` (Akita Section 10, before
Corollary 10.32), for one claim with coefficient packing. `carried` has the rows without a
challenge. `evaluation` is the row (102) on each plane `t`:
`Σ_i c_i·e_{i,t} = L_t(G_{b,M}·G_{b_z,m_A}·ẑ)` in `S`. `innerRows` is the row (104) with (64):
`A·G_{b_z,m_A}·ẑ = Σ_i ι(c_i)·G_{b_1,n_A}·t̂_i`. `responseDigits` is the alphabet `A_{b⋆}` of
`ẑ`. -/
structure Witness.Satisfies (β : Module.Basis (Fin O.planes) Zq E) (w : Witness O fold)
    (st : Statement O E) (challenge : Fin S.blocks → Rq O.carrierDegree) : Prop where
  carried : w.toCarried.Opens β st
  evaluation : ∀ t, ∑ i, challenge i * O.partialPlane w.partials i t =
    contract β st.point t (gadget S.sourceBase (gadget fold.responseBase w.response))
  innerRows : st.key.inner.mulVec (gadget fold.responseBase w.response) =
    ∑ i, embed O.stride O.stride_tower (challenge i) • gadget S.openingBase (w.inner i)
  responseDigits : Balanced S.certifiedBase w.response

/-- The parts of a weak opening of one fold for one claim (Akita Definition 10.22): a certified
decommitment, which has `s_i`, `t̂_i`, the digits of `F_1`, `F_2` and the factors `ι(c̄_i)`, with
the digits `ê` of the partials and the digits of `H_1`, `H_2`. -/
structure Weak (O : OpeningShape S) extends Certified S where
  partials : O.Partials
  openChain : PathDigits O.chain

/-- The carried digits of a weak opening. -/
def Weak.carried (ω : Weak O) : Carried O := ⟨ω.inner, ω.chain, ω.partials, ω.openChain⟩

/-- Akita Definition 10.22 for one claim on the coefficient-norm route: `ω` is a weak opening of
the statement, and its source blocks decode to the table. `certified` is the conditions (i) and
(ii), and `subring` is the part of (i) that each factor is `ι(c̄_i)` for a unit `c̄_i` of `S`.
`openRows` is (iii). `planes` and `value` are (iv): `e_{i,t} = L_t(G_{b,M}·s_i)` on each plane,
and the scalar row (59). `partialDigits` and `openDigits` are the part of (v) for the opening
path. -/
structure Weak.Opens (β : Module.Basis (Fin O.planes) Zq E) (factorBound sourceBound : ℕ)
    (ω : Weak O) (st : Statement O E) (table : Table) : Prop where
  certified : ω.toCertified.Opens factorBound sourceBound st.key st.payload table
  subring : ∀ i, ∃ c : Rq O.carrierDegree, IsUnit c ∧ ω.factor i = embed O.stride O.stride_tower c
  openRows : OpeningRows st.openKey ω.partials ω.openChain st.openPayload
  planes : ∀ i t, O.partialPlane ω.partials i t =
    contract β st.point t (gadget S.sourceBase (ω.source i))
  value : scalar β st.point (O.partialPlane ω.partials) = st.value
  partialDigits : Balanced S.certifiedBase ω.partials
  openDigits : PathDigits.Binary O.chain ω.openChain

section Binding

variable {β : Module.Basis (Fin O.planes) Zq E} {st : Statement O E} {ω ω' : Weak O}
  {factorBound sourceBound factorBound' sourceBound' : ℕ} {table table' : Table}

theorem Weak.Opens.carried (opens : ω.Opens β factorBound sourceBound st table) :
    ω.carried.Opens β st :=
  ⟨opens.certified.source.rows, opens.openRows, opens.value, opens.certified.source.innerDigits,
    opens.certified.source.chainDigits, opens.partialDigits, opens.openDigits⟩

/-- The value equation of a weak opening on its source blocks: the scalar (59) of the contracted
source blocks `L(G_{b,M}·s_i)` is the claimed value (Akita Definition 10.22 (iv)). -/
theorem Weak.Opens.value_source (opens : ω.Opens β factorBound sourceBound st table) :
    scalar β st.point (fun i t => contract β st.point t (gadget S.sourceBase (ω.source i))) =
      st.value := by
  have same : O.partialPlane ω.partials =
      fun i t => contract β st.point t (gadget S.sourceBase (ω.source i)) :=
    funext fun i => funext fun t => opens.planes i t
  rw [← same]
  exact opens.value

/-- Two weak openings to different tables have different source blocks. -/
theorem Weak.Opens.source_ne (opens : ω.Opens β factorBound sourceBound st table)
    (opens' : ω'.Opens β factorBound' sourceBound' st table') (different : table ≠ table') :
    ω.source ≠ ω'.source :=
  fun same => different
    (by rw [opens.certified.source.decodes, opens'.certified.source.decodes, same])

/-- The vector `z_A = ι(c̄'_i)·(ι(c̄_i)·s_i) − ι(c̄_i)·(ι(c̄'_i)·s'_i)` of the proof of Akita
Lemma 10.23, for the first block where it is a solution for `A`. -/
noncomputable def sourceCollision (key : Key S) (innerBound : ℕ) (st st' : Certified S) :
    Option (ShortKernel key.inner innerBound) :=
  (List.finRange S.blocks).findSome? fun i =>
    ShortKernel.ofCollision? key.inner innerBound (st'.factor i • (st.factor i • st.source i))
      (st.factor i • (st'.factor i • st'.source i))

/-- Akita Lemma 10.23, "Source blocks", with the bound (174), which is `E_∞(ν, ν')` of Akita
(164): two certified decommitments of one payload with the same digits `t̂` and different source
blocks give a solution of Module-SIS for `A` with each bound `η_A ≥ κ̄'_1·β̄_∞ + κ̄_1·β̄'_∞`. -/
theorem sourceCollision_isSome {key : Key S} {innerBound : ℕ} {st st' : Certified S}
    {payload : S.Payload}
    (opens : st.Opens factorBound sourceBound key payload table)
    (opens' : st'.Opens factorBound' sourceBound' key payload table')
    (covers : factorBound' * sourceBound + factorBound * sourceBound' ≤ innerBound)
    (inner : st.inner = st'.inner) (different : st.source ≠ st'.source) :
    (sourceCollision key innerBound st st').isSome := by
  obtain ⟨i, differs⟩ := Function.ne_iff.mp different
  refine List.findSome?_isSome_iff.mpr ⟨i, List.mem_finRange i,
    ShortKernel.ofCollision?_isSome ⟨?_, ?_, ?_⟩⟩
  · intro same
    rw [smul_smul, smul_smul, mul_comm] at same
    exact differs (((opens.unit i).mul (opens'.unit i)).smul_left_cancel.mp same)
  · simp only [Matrix.mulVec_smul]
    rw [opens.source.innerRows, opens'.source.innerRows, inner, smul_comm]
  · exact ((vecNormInf_sub_le _ _).trans (Nat.add_le_add
      ((vecNormInf_smul_le _ _).trans (Nat.mul_le_mul (opens'.factorNorm i) (opens.sourceNorm i)))
      ((vecNormInf_smul_le _ _).trans
        (Nat.mul_le_mul (opens.factorNorm i) (opens'.sourceNorm i))))).trans covers

/-- A solution of Module-SIS for one matrix of a fold (Akita Table 3): for `A` with the bound
`innerBound`, for `B` or `D` with the bound `b⋆ − 1`, or for `F_1`, `F_2`, `H_1` or `H_2` with the
bound 1. -/
abbrev FoldBreak (st : Statement O E) (innerBound : ℕ) : Type :=
  ShortKernel st.key.inner innerBound ⊕ (OuterBreak st.key ⊕ OpenBreak st.openKey)

/-- The reduction of Akita Lemma 10.23 for two weak openings: the solution of the digit paths, or
else the solution `z_A` for `A`. -/
noncomputable def weakCollision (st : Statement O E) (innerBound : ℕ) (ω ω' : Weak O) :
    Option (FoldBreak st innerBound) :=
  ((carriedCollision st ω.carried ω'.carried).map Sum.inr).or
    ((sourceCollision st.key innerBound ω.toCertified ω'.toCertified).map Sum.inl)

/-- Akita Lemma 10.23 (weak binding of one fold) with the bound (174): two weak openings of one
statement that differ on a source block or on a carried digit vector give a solution of
Module-SIS: for `A` with each bound `η_A ≥ κ̄'_1·β̄_∞ + κ̄_1·β̄'_∞`, for `B` or `D` with the bound
`b⋆ − 1`, or for a compression map with the bound 1. -/
theorem weakCollision_isSome {innerBound : ℕ}
    (opens : ω.Opens β factorBound sourceBound st table)
    (opens' : ω'.Opens β factorBound' sourceBound' st table')
    (covers : factorBound' * sourceBound + factorBound * sourceBound' ≤ innerBound)
    (different : ω.source ≠ ω'.source ∨ ω.carried ≠ ω'.carried) :
    (weakCollision st innerBound ω ω').isSome := by
  refine isSome_or_map.mpr ?_
  by_cases digits : ω.carried = ω'.carried
  · exact Or.inr (sourceCollision_isSome opens.certified opens'.certified covers
      (congrArg Carried.inner digits) (different.resolve_right (not_not.mpr digits)))
  · exact Or.inl (carriedCollision_isSome opens.carried opens'.carried digits)

/-- Akita Lemma 10.23 with the bound (176): for two weak openings with the same bounds `κ̄_1` and
`β̄_∞`, the solution for `A` has the bound `2·κ̄_1·β̄_∞`, which is `η_A` of Akita (50) for
`β̄_∞ = Δ^cert_f`. -/
theorem weakCollision_isSome_of_bounds (opens : ω.Opens β factorBound sourceBound st table)
    (opens' : ω'.Opens β factorBound sourceBound st table')
    (different : ω.source ≠ ω'.source ∨ ω.carried ≠ ω'.carried) :
    (weakCollision st (2 * factorBound * sourceBound) ω ω').isSome :=
  weakCollision_isSome opens opens' (le_of_eq (by ring)) different

end Binding

section Extraction

/-- One accepting extension of a transcript prefix through the fold challenge (a child of Akita
Definition 10.30): its challenges `c_i` and the digit vectors of its branch relation. -/
structure Child (O : OpeningShape S) (fold : Descriptor) where
  challenge : Fin S.blocks → Rq O.carrierDegree
  witness : Witness O fold

/-- Two children for each block `i` (Akita Definition 10.30 (ii), with `Q` the set of blocks). The
pairs need not share a child. -/
structure Fork (O : OpeningShape S) (fold : Descriptor) where
  first : Fin S.blocks → Child O fold
  second : Fin S.blocks → Child O fold

variable {β : Module.Basis (Fin O.planes) Zq E} {st : Statement O E} {fork : Fork O fold}
  {family : Set (Rq O.carrierDegree)} {digits : Carried O}

/-- The coordinate-wise structure of Akita Definition 10.30 (ii) with the hypothesis of Akita
Corollary 10.32: the two children of block `i` have challenges of the family at `i` that differ,
their challenges agree at each other block, and the digit vectors of each child satisfy its exact
branch relation for the statement. -/
structure Fork.Accepts (β : Module.Basis (Fin O.planes) Zq E) (family : Set (Rq O.carrierDegree))
    (fork : Fork O fold) (st : Statement O E) : Prop where
  firstMember : ∀ i, (fork.first i).challenge i ∈ family
  secondMember : ∀ i, (fork.second i).challenge i ∈ family
  different : ∀ i, (fork.first i).challenge i ≠ (fork.second i).challenge i
  agree : ∀ i j, j ≠ i → (fork.first i).challenge j = (fork.second i).challenge j
  firstSatisfies : ∀ i, (fork.first i).witness.Satisfies β st (fork.first i).challenge
  secondSatisfies : ∀ i, (fork.second i).witness.Satisfies β st (fork.second i).challenge

/-- `c̄_i = c_i^{(i,1)} − c_i^{(i,0)}` (Akita Theorem 10.31, "Coordinate siblings"). -/
noncomputable def Fork.difference (fork : Fork O fold) (i : Fin S.blocks) : Rq O.carrierDegree :=
  (fork.second i).challenge i - (fork.first i).challenge i

/-- `z^{(i,1)} − z^{(i,0)}` for the recomposed responses `z = G_{b_z,m_A}·ẑ` of the two children
of block `i`. -/
noncomputable def Fork.responseDifference (fork : Fork O fold) (i : Fin S.blocks) : S.Source :=
  gadget fold.responseBase (fork.second i).witness.response -
    gadget fold.responseBase (fork.first i).witness.response

/-- The weak opening of the proof of Akita Theorem 10.31: the factor `ι(c̄_i)`, the source block
`s_i = ι(c̄_i)^{-1}·(z^{(i,1)} − z^{(i,0)})`, and the carried digits `digits` of the children. -/
noncomputable def extract (fork : Fork O fold) (digits : Carried O) : Weak O where
  source i := Ring.inverse (embed O.stride O.stride_tower (fork.difference i)) •
    fork.responseDifference i
  inner := digits.inner
  chain := digits.chain
  factor i := embed O.stride O.stride_tower (fork.difference i)
  partials := digits.partials
  openChain := digits.openChain

/-- The reduction of Akita Corollary 10.32, "Digit paths": the solution of `carriedCollision` for
the first child whose carried digits are not `digits`. -/
noncomputable def forkCollision (st : Statement O E) (fork : Fork O fold) (digits : Carried O) :
    Option (OuterBreak st.key ⊕ OpenBreak st.openKey) :=
  (List.finRange S.blocks).findSome? fun i =>
    (carriedCollision st (fork.first i).witness.toCarried digits).or
      (carriedCollision st (fork.second i).witness.toCarried digits)

theorem Fork.Accepts.isUnit_difference (accepts : fork.Accepts β family st)
    (units : UnitDifferences family) (i : Fin S.blocks) : IsUnit (fork.difference i) :=
  units _ (accepts.secondMember i) _ (accepts.firstMember i) (accepts.different i).symm

/-- `ι(c̄_i)·s_i = z^{(i,1)} − z^{(i,0)}` for the extracted source block. -/
theorem extract_factor_smul_source (fork : Fork O fold) (digits : Carried O) (i : Fin S.blocks)
    (unit : IsUnit (fork.difference i)) :
    (extract fork digits).factor i • (extract fork digits).source i =
      fork.responseDifference i := by
  show embed O.stride O.stride_tower (fork.difference i) •
    (Ring.inverse (embed O.stride O.stride_tower (fork.difference i)) •
      fork.responseDifference i) = fork.responseDifference i
  rw [smul_smul, Ring.mul_inverse_cancel _ (isUnit_embed O.stride_tower unit), one_smul]

/-- The difference of the rows (104) of the two children of block `i`:
`A·(z^{(i,1)} − z^{(i,0)}) = ι(c̄_i)·G_{b_1,n_A}·t̂_i`. -/
theorem Fork.Accepts.inner_difference (accepts : fork.Accepts β family st)
    (firstDigits : ∀ i, (fork.first i).witness.toCarried = digits)
    (secondDigits : ∀ i, (fork.second i).witness.toCarried = digits) (i : Fin S.blocks) :
    st.key.inner.mulVec (fork.responseDifference i) =
      embed O.stride O.stride_tower (fork.difference i) •
        gadget S.openingBase (digits.inner i) := by
  have first := (accepts.firstSatisfies i).innerRows
  have second := (accepts.secondSatisfies i).innerRows
  rw [firstDigits i] at first
  rw [secondDigits i] at second
  rw [Fork.responseDifference, Matrix.mulVec_sub, first, second, Fork.difference, map_sub]
  exact sum_smul_sub_sum_smul
    (fun j => embed O.stride O.stride_tower ((fork.second i).challenge j))
    (fun j => embed O.stride O.stride_tower ((fork.first i).challenge j))
    (fun j => gadget S.openingBase (digits.inner j)) i
    (fun j other => congrArg _ (accepts.agree i j other).symm)

/-- The difference of the rows (102) of the two children of block `i` on the plane `t`:
`c̄_i·e_{i,t} = L_t(G_{b,M}·(z^{(i,1)} − z^{(i,0)}))`. -/
theorem Fork.Accepts.plane_difference (accepts : fork.Accepts β family st)
    (firstDigits : ∀ i, (fork.first i).witness.toCarried = digits)
    (secondDigits : ∀ i, (fork.second i).witness.toCarried = digits) (i : Fin S.blocks)
    (t : Fin O.planes) :
    fork.difference i * O.partialPlane digits.partials i t =
      contract β st.point t (gadget S.sourceBase (fork.responseDifference i)) := by
  have first := (accepts.firstSatisfies i).evaluation t
  have second := (accepts.secondSatisfies i).evaluation t
  rw [firstDigits i] at first
  rw [secondDigits i] at second
  rw [Fork.responseDifference, gadget_sub, map_sub, ← first, ← second]
  exact (sum_smul_sub_sum_smul (fun j => (fork.second i).challenge j)
    (fun j => (fork.first i).challenge j) (fun j => O.partialPlane digits.partials j t) i
    (fun j other => (accepts.agree i j other).symm)).symm

/-- Akita Theorem 10.31, "Coordinate siblings" and "Assembly", for one claim: when each child has
the carried digits `digits`, the extracted `extract fork digits` is a weak opening of the
statement to the table that its source blocks decode to, with `κ̄_1 = 2ω` and
`β̄_∞ = Δ^cert_f`. The family has units as differences (`units`) and members of the norm
`‖c‖_1 = ω` (`sparse`), and `b_z ≥ 2`. -/
theorem extract_opens {mass peak : ℕ} (units : UnitDifferences family)
    (sparse : ∀ c ∈ family, Sparse mass peak c) (base : 2 ≤ fold.responseBase)
    (accepts : fork.Accepts β family st) (opens : digits.Opens β st)
    (firstDigits : ∀ i, (fork.first i).witness.toCarried = digits)
    (secondDigits : ∀ i, (fork.second i).witness.toCarried = digits) :
    (extract fork digits).Opens β (2 * mass) (fold.sourceBound S) st
      (S.decode (extract fork digits).source) := by
  have unit := fun i => accepts.isUnit_difference units i
  have cancel := fun i => extract_factor_smul_source fork digits i (unit i)
  refine {
    certified := {
      source := {
        decodes := rfl
        innerRows := fun i => ?_
        rows := opens.rows
        innerDigits := opens.innerDigits
        chainDigits := opens.chainDigits }
      unit := fun i => isUnit_embed O.stride_tower (unit i)
      factorNorm := fun i => ?_
      sourceNorm := fun i => ?_ }
    subring := fun i => ⟨fork.difference i, unit i, rfl⟩
    openRows := opens.openRows
    planes := fun i t => ?_
    value := opens.value
    partialDigits := opens.partialDigits
    openDigits := opens.openDigits }
  · show st.key.inner.mulVec (Ring.inverse (embed O.stride O.stride_tower (fork.difference i)) •
      fork.responseDifference i) = gadget S.openingBase (digits.inner i)
    rw [Matrix.mulVec_smul, accepts.inner_difference firstDigits secondDigits i, smul_smul,
      Ring.inverse_mul_cancel _ (isUnit_embed O.stride_tower (unit i)), one_smul]
  · show normOne (embed O.stride O.stride_tower (fork.difference i)) ≤ 2 * mass
    rw [normOne_embed]
    exact (sparse _ (accepts.secondMember i)).normOne_sub_le (sparse _ (accepts.firstMember i))
  · rw [cancel i, Descriptor.sourceBound_eq fold S base]
    exact vecNormInf_gadget_sub_le _ (accepts.secondSatisfies i).responseDigits
      (accepts.firstSatisfies i).responseDigits
  · refine (unit i).mul_left_cancel ?_
    show fork.difference i * O.partialPlane digits.partials i t =
      fork.difference i *
        contract β st.point t (gadget S.sourceBase ((extract fork digits).source i))
    rw [accepts.plane_difference firstDigits secondDigits i t, ← cancel i, gadget_smul]
    exact contract_embed_smul β st.point t _ _

/-- Akita Corollary 10.32 for one claim with coefficient packing, on the coefficient-norm route.
`fork` has two children for each block that satisfy their exact branch relations for one
statement, with challenges of a family with unit differences and `‖c‖_1 = ω`. `digits` is a family
of carried digits that satisfies the rows of the statement, for example that of a child. Then a
child with other carried digits gives a solution of Module-SIS for `B` or `D` with the bound
`b⋆ − 1` or for a compression map with the bound 1, or `extract fork digits` is a weak opening
with `κ̄_1 = 2ω` and `β̄_∞ = Δ^cert_f`. -/
theorem extract_opens_or_collision {mass peak : ℕ} (units : UnitDifferences family)
    (sparse : ∀ c ∈ family, Sparse mass peak c) (base : 2 ≤ fold.responseBase)
    (accepts : fork.Accepts β family st) (opens : digits.Opens β st) :
    (forkCollision st fork digits).isSome ∨
      (extract fork digits).Opens β (2 * mass) (fold.sourceBound S) st
        (S.decode (extract fork digits).source) := by
  by_cases same : (∀ i, (fork.first i).witness.toCarried = digits) ∧
      ∀ i, (fork.second i).witness.toCarried = digits
  · exact Or.inr (extract_opens units sparse base accepts opens same.1 same.2)
  · have other : ∃ i, (fork.first i).witness.toCarried ≠ digits ∨
        (fork.second i).witness.toCarried ≠ digits := by
      by_contra none
      exact same ⟨fun i => not_not.mp fun differs => none ⟨i, Or.inl differs⟩,
        fun i => not_not.mp fun differs => none ⟨i, Or.inr differs⟩⟩
    obtain ⟨i, differs⟩ := other
    refine Or.inl (List.findSome?_isSome_iff.mpr ⟨i, List.mem_finRange i, ?_⟩)
    rw [Option.isSome_or, Bool.or_eq_true]
    exact differs.imp
      (carriedCollision_isSome (accepts.firstSatisfies i).carried opens)
      (carriedCollision_isSome (accepts.secondSatisfies i).carried opens)

end Extraction

end NightstreamFPrime.Spec.Compression.Akita
