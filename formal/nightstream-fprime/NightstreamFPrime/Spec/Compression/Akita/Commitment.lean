import Mathlib.Logic.Equiv.Fin.Basic
import NightstreamFPrime.Spec.Compression.Akita.Ring

/-! The Akita commitment of one polynomial (ePrint 2026/1983, revision of 2026-11-23, Section 4.5)
and its decommitment relation (Akita Definitions 10.21 and 10.22).

- `Shape` has the numbers of the commitment shape (Akita (28)): the ring degrees, the ranks, the
  block geometry, the digit bases and the optional compression maps `F_1`, `F_2` (`ChainShape`).
  `Key` has the matrices `A`, `B` and the compression maps.
- `gadget` is `G_{b,n}` (Akita Section 3.3). `Shape.outerView` and `binaryRecompose` are the fixed
  layouts of Akita Appendix L.1. `Shape.sliceInput` is Akita (44), and `stack` is Akita (45).
- `Rows` has the rows (47), or the row (48) without compression maps.
- `Decommitment.Opens` is Akita Definition 10.21 for one polynomial, with `Shape.decode` as the
  decoder `Dec` under coefficient packing.
- `Certified.Opens` adds the weak certificate of Akita Definition 10.22 (i) for the coefficient
  norm on each source block. Definition 10.21 has no norm bound on the source blocks, so it does
  not bind a commitment to one table.
- `Descriptor` has the numbers of an opening descriptor that fix the bounds of that certificate.
  `Descriptor.factorBound`, `Descriptor.sourceBound` and `Descriptor.innerBound` are `κ̄_1`,
  `Δ^cert_f` and `η_A` of Akita (50). -/

namespace NightstreamFPrime.Spec.Compression.Akita

section Digits

variable {d : ℕ} [NeZero d] {ι : Type}

/-- Each coefficient of each entry of `v` has its representative in `(−q/2, q/2]` in the interval
`[lo, hi]`. -/
def Digits (lo hi : ℤ) (v : ι → Rq d) : Prop :=
  ∀ j, ∀ i < d, lo ≤ (coeff (v j) i).valMinAbs ∧ (coeff (v j) i).valMinAbs ≤ hi

/-- Each coefficient of `v` is in the balanced alphabet `A_b = {−b/2, …, b/2 − 1}`
(Akita Section 3.3). -/
def Balanced (b : ℕ) (v : ι → Rq d) : Prop := Digits (-((b / 2 : ℕ) : ℤ)) (((b / 2 : ℕ) : ℤ) - 1) v

/-- Each coefficient of `v` is in the alphabet `{−1, 0}` of a compression map
(Akita Section 4.5). -/
def NegativeBinary (v : ι → Rq d) : Prop := Digits (-1) 0 v

/-- Two values of one interval `[lo, hi]` differ by at most `hi − lo`. -/
theorem magnitude_sub_le_of_interval {lo hi : ℤ} {a b : Zq}
    (left : lo ≤ a.valMinAbs ∧ a.valMinAbs ≤ hi) (right : lo ≤ b.valMinAbs ∧ b.valMinAbs ≤ hi) :
    magnitude (a - b) ≤ (hi - lo).toNat := by
  have difference : a - b = ((a.valMinAbs - b.valMinAbs : ℤ) : Zq) := by
    rw [Int.cast_sub, ZMod.coe_valMinAbs, ZMod.coe_valMinAbs]
  rw [difference]
  refine (magnitude_intCast_le _).trans ?_
  omega

theorem vecNormInf_sub_le_of_digits [Fintype ι] {lo hi : ℤ} {v w : ι → Rq d}
    (left : Digits lo hi v) (right : Digits lo hi w) : vecNormInf (v - w) ≤ (hi - lo).toNat := by
  refine (vecNormInf_le_iff _ _).mpr fun j => (normInf_le_iff _ _).mpr fun i small => ?_
  rw [Pi.sub_apply, coeff_sub]
  exact magnitude_sub_le_of_interval (left j i small) (right j i small)

/-- Two digit vectors of the alphabet `{−1, 0}` differ by at most 1 in each coefficient. -/
theorem vecNormInf_sub_le_one [Fintype ι] {v w : ι → Rq d} (left : NegativeBinary v)
    (right : NegativeBinary w) : vecNormInf (v - w) ≤ 1 :=
  vecNormInf_sub_le_of_digits left right

/-- `G_{b,n} = I_n ⊗ (1, b, …, b^{δ−1})` on a vector of `δ` digits for each of its `n` entries
(Akita Section 3.3). -/
noncomputable def gadget (b : ℕ) {δ : ℕ} (digits : ι × Fin δ → Rq d) : ι → Rq d :=
  fun j => ∑ u : Fin δ, (b : Rq d) ^ (u : ℕ) * digits (j, u)

end Digits

section Layout

/-- The coefficients of a vector of ring elements in one sequence: position `j·d + i` has
coefficient `i` of entry `j`, and each position after the last entry has 0. -/
noncomputable def sequence {d m : ℕ} [NeZero d] (v : Fin m → Rq d) (position : ℕ) : Zq :=
  if inside : position / d < m then coeff (v ⟨position / d, inside⟩) (position % d) else 0

/-- `δ = ⌈log_2 q⌉`, the number of binary digits of a coefficient (Akita Section 4.5). -/
def binaryDigits : ℕ := Nat.clog 2 fieldModulus

/-- The gadget map `G_{A_F,·}` of a compression stage (Akita Appendix L.1, binary digit
placement): for an image of `n` elements of `R_{q,d}`, which has `s = n·d` coefficients, digit `u`
of coefficient `v` is at position `v + s·u` of the digit sequence, and coefficient `v` of the
image is `Σ_u 2^u ξ_{v + s·u}`. -/
noncomputable def binaryRecompose {width inputDegree : ℕ} [NeZero inputDegree] (d n : ℕ) [NeZero d]
    (digits : Fin width → Rq inputDegree) : Fin n → Rq d :=
  fun entry => ofCoeffs fun k =>
    ∑ u ∈ Finset.range binaryDigits, 2 ^ u * sequence digits (entry * d + k + n * d * u)

/-- The stacked image `u = (u_0 ‖ … ‖ u_{S−1})` with `u_s = M x_s` (Akita (45)), ordered by slice
and then by row. One slice is the map `M` without slices. -/
noncomputable def stack {d n slices : ℕ} {cols : Type} [Fintype cols] (M : Matrix (Fin n) cols (Rq d))
    (x : Fin slices → cols → Rq d) : Fin (slices * n) → Rq d :=
  fun j => M.mulVec (x (finProdFinEquiv.symm j).1) (finProdFinEquiv.symm j).2

theorem mulVec_eq_of_stack_eq {d n slices : ℕ} {cols : Type} [Fintype cols]
    {M : Matrix (Fin n) cols (Rq d)} {x x' : Fin slices → cols → Rq d}
    (same : stack M x = stack M x') (s : Fin slices) : M.mulVec (x s) = M.mulVec (x' s) := by
  funext row
  have entry := congrFun same (finProdFinEquiv (s, row))
  simpa only [stack, Equiv.symm_apply_apply] using entry

end Layout

/-- The shapes of the compression maps `F_1` and `F_2` of a commitment with `L_F = 2`
(Akita Section 4.5): ring degree, rank and width of each map. -/
structure ChainShape where
  firstDegree : ℕ
  firstRank : ℕ
  firstWidth : ℕ
  secondDegree : ℕ
  secondRank : ℕ
  secondWidth : ℕ
  [firstPositive : NeZero firstDegree]
  [secondPositive : NeZero secondDegree]

attribute [instance] ChainShape.firstPositive ChainShape.secondPositive

/-- The compression maps `F_1` and `F_2`. -/
structure ChainKey (c : ChainShape) where
  first : Matrix (Fin c.firstRank) (Fin c.firstWidth) (Rq c.firstDegree)
  second : Matrix (Fin c.secondRank) (Fin c.secondWidth) (Rq c.secondDegree)

/-- The digit vectors `û_1` and `û_2` of the compression maps. -/
@[ext] structure ChainDigits (c : ChainShape) where
  first : Fin c.firstWidth → Rq c.firstDegree
  second : Fin c.secondWidth → Rq c.secondDegree

/-- The compression maps of a path to the payload: none for `L_F = 0`, and `F_1`, `F_2` for
`L_F = 2`. -/
def PathKey : Option ChainShape → Type
  | none => PUnit
  | some c => ChainKey c

/-- The digit vectors of the compression maps of a path. -/
def PathDigits : Option ChainShape → Type
  | none => PUnit
  | some c => ChainDigits c

/-- The payload `u_pub` (Akita (46)) of an image of `n` elements of `R_{q,d}`: the image for
`L_F = 0`, and the image of `F_2` for `L_F = 2`. -/
def Payload (d n : ℕ) : Option ChainShape → Type
  | none => Fin n → Rq d
  | some c => Fin c.secondRank → Rq c.secondDegree

/-- The rows (47) of Akita from the image `M x̂` to the payload `p`, or the row (48) for `L = 0`:
`M x̂ = G ξ̂_1`, `K_1 ξ̂_1 = G ξ̂_2` and `K_2 ξ̂_2 = p`, or `M x̂ = p`. -/
def Rows {d n : ℕ} [NeZero d] : (chain : Option ChainShape) → PathKey chain → (Fin n → Rq d) →
    PathDigits chain → Payload d n chain → Prop
  | none, _, image, _, payload => image = payload
  | some c, key, image, digits, payload =>
      image = binaryRecompose d n (ChainDigits.first digits) ∧
      (ChainKey.first key).mulVec (ChainDigits.first digits) =
        binaryRecompose c.firstDegree c.firstRank (ChainDigits.second digits) ∧
      (ChainKey.second key).mulVec (ChainDigits.second digits) = payload

/-- Each coefficient of each digit vector of the compression maps is in `{−1, 0}`. -/
def PathDigits.Binary : (chain : Option ChainShape) → PathDigits chain → Prop
  | none, _ => True
  | some _, digits =>
      NegativeBinary (ChainDigits.first digits) ∧ NegativeBinary (ChainDigits.second digits)

/-- The shape of the commitment of one polynomial (Akita (28)): the ring degrees `d_A` and `d_B`
with `d_B ∣ d_A`, the number `B` of blocks and the block length `M`, the base `b` and the digit
count `δ_com` of a source block, the rank `n_A`, the base `b_1` and the digit count `δ_open` of an
inner image, the certified base `b⋆` of its digits (Akita Section 4.6), the rank `n_B`, the slice
count `S_B`, and the compression maps. -/
structure Shape where
  innerDegree : ℕ
  outerDegree : ℕ
  [innerPositive : NeZero innerDegree]
  [outerPositive : NeZero outerDegree]
  divides : outerDegree ∣ innerDegree
  blocks : ℕ
  blockLength : ℕ
  sourceBase : ℕ
  sourceDigits : ℕ
  innerRank : ℕ
  openingBase : ℕ
  openingDigits : ℕ
  certifiedBase : ℕ
  outerRank : ℕ
  slices : ℕ
  [slicesPositive : NeZero slices]
  chain : Option ChainShape

attribute [instance] Shape.innerPositive Shape.outerPositive Shape.slicesPositive

/-- A table of base-field cells, 0 after its last cell. -/
abbrev Table : Type := ℕ → Zq

namespace Shape

variable (S : Shape)

/-- A source block `s_i ∈ R_{q,d_A}^{M·δ_com}`. -/
abbrev Source : Type := Fin S.blockLength × Fin S.sourceDigits → Rq S.innerDegree

/-- The digits `t̂_i ∈ R_{q,d_A}^{n_A·δ_open}` of an inner image. -/
abbrev InnerDigits : Type := Fin S.innerRank × Fin S.openingDigits → Rq S.innerDegree

/-- `d_A / d_B`. -/
def ratio : ℕ := S.innerDegree / S.outerDegree

/-- The `g_B = n_A · (d_A / d_B) · δ_open` positions of one block in the input of `B`, in the
order `A` row, `R_{q,d_B}` subcolumn, digit (Akita Section 4.5). -/
abbrev OuterColumn : Type := Fin S.innerRank × Fin S.ratio × Fin S.openingDigits

/-- `L^blk_B = ⌈B / S_B⌉`, the number of blocks in the input of one slice. -/
def sliceLength : ℕ := (S.blocks + S.slices - 1) / S.slices

/-- The input `x_{B,s} ∈ R_{q,d_B}^{m_B}` of one slice, with `m_B = L^blk_B · g_B`. -/
abbrev OuterInput : Type := Fin S.sliceLength × S.OuterColumn → Rq S.outerDegree

/-- The payload `u_pub` of the commitment. -/
abbrev Payload : Type := Akita.Payload S.outerDegree (S.slices * S.outerRank) S.chain

/-- The view of the digits of an inner image in `R_{q,d_B}` (Akita Appendix L.1): coefficient `k`
of the element at `(a, y, u)` is coefficient `y·d_B + k` of digit `u` of row `a`. -/
noncomputable def outerView (digits : S.InnerDigits) : S.OuterColumn → Rq S.outerDegree :=
  fun column => ofCoeffs fun k =>
    coeff (digits (column.1, column.2.2)) (column.2.1 * S.outerDegree + k)

/-- `⌊s·B / S_B⌋`, the first block of slice `s` (Akita (43)). -/
def sliceStart (s : ℕ) : ℕ := s * S.blocks / S.slices

theorem sliceStart_le {s : ℕ} (inside : s ≤ S.slices) : S.sliceStart s ≤ S.blocks :=
  Nat.div_le_of_le_mul (Nat.mul_le_mul_right _ inside)

/-- The input of slice `s` (Akita (44) for one polynomial): the views of the blocks of the range
`I_{B,s} = [⌊s·B / S_B⌋, ⌊(s+1)·B / S_B⌋)`, then zeros. -/
noncomputable def sliceInput (inner : Fin S.blocks → S.InnerDigits) (s : Fin S.slices) :
    S.OuterInput :=
  fun position =>
    if inside : S.sliceStart s + position.1 < S.sliceStart ((s : ℕ) + 1) then
      S.outerView (inner ⟨S.sliceStart s + position.1,
        inside.trans_le (S.sliceStart_le (Nat.succ_le_of_lt s.isLt))⟩) position.2
    else 0

theorem address_lt (y : Fin S.ratio) {k : ℕ} (small : k < S.outerDegree) :
    y * S.outerDegree + k < S.innerDegree :=
  calc y * S.outerDegree + k < (y + 1) * S.outerDegree := by rw [Nat.add_mul, one_mul]; omega
    _ ≤ S.ratio * S.outerDegree := Nat.mul_le_mul_right _ y.isLt
    _ = S.innerDegree := Nat.div_mul_cancel S.divides

theorem outerView_injective : Function.Injective S.outerView := by
  intro digits digits' same
  funext entry
  refine ext_coeff fun j small => ?_
  have positive := NeZero.pos S.outerDegree
  have column := congrArg (fun x => coeff x (j % S.outerDegree)) (congrFun same
    (entry.1, ⟨j / S.outerDegree, Nat.div_lt_div_of_lt_of_dvd S.divides small⟩, entry.2))
  simp only [outerView, coeff_ofCoeffs _ (Nat.mod_lt j positive), Nat.div_add_mod'] at column
  exact column

/-- The views of two digit vectors of the alphabet `A_b` differ by at most `b − 1` in each
coefficient (Akita (49)). -/
theorem outerView_sub_le {b : ℕ} {digits digits' : S.InnerDigits} (left : Balanced b digits)
    (right : Balanced b digits') :
    vecNormInf (S.outerView digits - S.outerView digits') ≤ b - 1 := by
  refine (vecNormInf_le_iff _ _).mpr fun column => (normInf_le_iff _ _).mpr fun k small => ?_
  rw [Pi.sub_apply, coeff_sub, outerView, outerView, coeff_ofCoeffs _ small,
    coeff_ofCoeffs _ small]
  refine (magnitude_sub_le_of_interval (left _ _ (S.address_lt column.2.1 small))
    (right _ _ (S.address_lt column.2.1 small))).trans ?_
  omega

theorem exists_slice {i : ℕ} (inside : i < S.blocks) :
    ∃ s < S.slices, S.sliceStart s ≤ i ∧ i < S.sliceStart (s + 1) := by
  have search : ∀ n, i < S.sliceStart n → ∃ s < n, S.sliceStart s ≤ i ∧ i < S.sliceStart (s + 1) := by
    intro n
    induction n with
    | zero =>
        intro below
        simp [sliceStart] at below
    | succ n step =>
        intro below
        by_cases earlier : i < S.sliceStart n
        · obtain ⟨s, small, range⟩ := step earlier
          exact ⟨s, Nat.lt_succ_of_lt small, range⟩
        · exact ⟨n, Nat.lt_succ_self n, Nat.le_of_not_lt earlier, below⟩
  refine search S.slices ?_
  rwa [sliceStart, Nat.mul_div_cancel_left _ (NeZero.pos S.slices)]

/-- A slice has at most `L^blk_B` blocks. -/
theorem sliceStart_succ_le (s : ℕ) : S.sliceStart (s + 1) ≤ S.sliceStart s + S.sliceLength := by
  have positive := NeZero.pos S.slices
  have start := Nat.div_add_mod (s * S.blocks) S.slices
  have startRest := Nat.mod_lt (s * S.blocks) positive
  have length := Nat.div_add_mod (S.blocks + S.slices - 1) S.slices
  have lengthRest := Nat.mod_lt (S.blocks + S.slices - 1) positive
  unfold sliceStart sliceLength
  rw [← Nat.lt_succ_iff, Nat.div_lt_iff_lt_mul positive, Nat.succ_mul, Nat.succ_mul, Nat.add_mul,
    Nat.mul_comm (s * S.blocks / S.slices), Nat.mul_comm ((S.blocks + S.slices - 1) / S.slices)]
  omega

theorem sliceInput_apply (inner : Fin S.blocks → S.InnerDigits) (s : Fin S.slices)
    (position : Fin S.sliceLength) (column : S.OuterColumn) (i : Fin S.blocks)
    (address : S.sliceStart s + position = i) (inside : (i : ℕ) < S.sliceStart ((s : ℕ) + 1)) :
    S.sliceInput inner s (position, column) = S.outerView (inner i) column := by
  have block : (⟨S.sliceStart s + position, address ▸ i.isLt⟩ : Fin S.blocks) = i :=
    Fin.ext address
  have range : S.sliceStart s + position < S.sliceStart ((s : ℕ) + 1) := address ▸ inside
  simp only [sliceInput, range, ↓reduceDIte, block]

theorem sliceInput_injective : Function.Injective S.sliceInput := by
  intro inner inner' same
  funext i
  obtain ⟨s, small, low, high⟩ := S.exists_slice i.isLt
  have width := S.sliceStart_succ_le s
  refine S.outerView_injective (funext fun column => ?_)
  have entry := congrFun (congrFun same ⟨s, small⟩)
    ((⟨i - S.sliceStart s, by omega⟩ : Fin S.sliceLength), column)
  rwa [S.sliceInput_apply inner ⟨s, small⟩ _ column i (by simp; omega) high,
    S.sliceInput_apply inner' ⟨s, small⟩ _ column i (by simp; omega) high] at entry

/-- The inputs of one slice for two digit families differ by at most `bound` in each coefficient
when the views of each block do. -/
theorem sliceInput_sub_le {bound : ℕ} {inner inner' : Fin S.blocks → S.InnerDigits}
    (close : ∀ i, vecNormInf (S.outerView (inner i) - S.outerView (inner' i)) ≤ bound)
    (s : Fin S.slices) : vecNormInf (S.sliceInput inner s - S.sliceInput inner' s) ≤ bound := by
  refine (vecNormInf_le_iff _ _).mpr fun position => ?_
  rw [Pi.sub_apply]
  unfold sliceInput
  split
  · exact (normInf_le_vecNormInf (S.outerView (inner _) - S.outerView (inner' _)) position.2).trans
      (close _)
  · rw [sub_self, normInf_zero]
    exact Nat.zero_le _

/-- The decoder `Dec` for coefficient packing (Akita Section 4.5): cell `(i·M + x)·d_A + k` of the
table is coefficient `k` of entry `x` of `G_{b,M} s_i`, and each cell after block `B − 1` is 0. -/
noncomputable def decode (source : Fin S.blocks → S.Source) : Table :=
  fun cell =>
    if inside : cell / (S.blockLength * S.innerDegree) < S.blocks then
      sequence (gadget S.sourceBase (source ⟨cell / (S.blockLength * S.innerDegree), inside⟩))
        (cell % (S.blockLength * S.innerDegree))
    else 0

end Shape

/-- The matrices of a commitment: the inner matrix `A ∈ R_{q,d_A}^{n_A × M·δ_com}`, the outer
matrix `B ∈ R_{q,d_B}^{n_B × m_B}`, and the compression maps (Akita Section 4.3). -/
structure Key (S : Shape) where
  inner : Matrix (Fin S.innerRank) (Fin S.blockLength × Fin S.sourceDigits) (Rq S.innerDegree)
  outer : Matrix (Fin S.outerRank) (Fin S.sliceLength × S.OuterColumn) (Rq S.outerDegree)
  chain : PathKey S.chain

/-- The parts of a decommitment (Akita Definition 10.21): the source blocks `s_i`, the digits
`t̂_i` of the inner images, and the digit vectors of the compression maps. -/
structure Decommitment (S : Shape) where
  source : Fin S.blocks → S.Source
  inner : Fin S.blocks → S.InnerDigits
  chain : PathDigits S.chain

/-- Akita Definition 10.21 for one polynomial: `st` is an algebraic decommitment of the payload to
the table. (i) The source blocks decode to the table, and `A s_i = G_{b_1,n_A} t̂_i`. (ii) The
slice inputs (44) of the digits and the compression maps satisfy the rows (47) or (48) and end at
the payload. (iii) The digits `t̂_i` are in the certified alphabet `A_{b⋆}`, and the digits of the
compression maps are in `{−1, 0}`. -/
structure Decommitment.Opens {S : Shape} (st : Decommitment S) (key : Key S) (payload : S.Payload)
    (table : Table) : Prop where
  decodes : table = S.decode st.source
  innerRows : ∀ i, key.inner.mulVec (st.source i) = gadget S.openingBase (st.inner i)
  rows : Rows S.chain key.chain (stack key.outer (S.sliceInput st.inner)) st.chain payload
  innerDigits : ∀ i, Balanced S.certifiedBase (st.inner i)
  chainDigits : PathDigits.Binary S.chain st.chain

/-- A decommitment with a factor `ι(c̄_i) ∈ R_{q,d_A}` for each source block
(Akita Definition 10.22). -/
structure Certified (S : Shape) extends Decommitment S where
  factor : Fin S.blocks → Rq S.innerDegree

/-- Akita Definition 10.21 with the weak certificate of Definition 10.22 (i) for the coefficient
norm: each factor is a unit with `‖ι(c̄_i)‖_1 ≤ κ̄_1` and `‖ι(c̄_i)·s_i‖_∞ ≤ β̄_∞`. -/
structure Certified.Opens {S : Shape} (factorBound sourceBound : ℕ) (st : Certified S)
    (key : Key S) (payload : S.Payload) (table : Table) : Prop where
  source : st.toDecommitment.Opens key payload table
  unit : ∀ i, IsUnit (st.factor i)
  factorNorm : ∀ i, normOne (st.factor i) ≤ factorBound
  sourceNorm : ∀ i, vecNormInf (st.factor i • st.source i) ≤ sourceBound

/-- The numbers of the opening descriptor of a group that fix the bounds of the certificate of its
source blocks (Akita (50) and (53)): the norm `κ_1 = ‖c‖_1` of a fold challenge, the response base
`b_z` and the response depth `δ_f`. -/
structure Descriptor where
  challengeMass : ℕ
  responseBase : ℕ
  responseDigits : ℕ

namespace Descriptor

variable (O : Descriptor) (S : Shape)

/-- `κ̄_1 = 2·κ_1`, the bound on `‖c − c'‖_1` for two fold challenges. -/
def factorBound : ℕ := 2 * O.challengeMass

/-- `Δ^cert_f = (b⋆ − 1)·(b_z^{δ_f} − 1)/(b_z − 1)` (Akita (50)), the bound `β̄_∞`. -/
def sourceBound : ℕ :=
  (S.certifiedBase - 1) * ((O.responseBase ^ O.responseDigits - 1) / (O.responseBase - 1))

/-- `η_A = 2·κ̄_1·Δ^cert_f` (Akita (50)), the bound of the Module-SIS instance of `A`. -/
def innerBound : ℕ := 2 * O.factorBound * O.sourceBound S

/-- The bound `β̄_∞ + κ̄_1·ρ` of Akita (164) for a reference occurrence with `ρ ≤ Δ^cert_f` is at
most `η_A`. -/
theorem mixed_le {referenceBound : ℕ} (mass : 0 < O.challengeMass)
    (reference : referenceBound ≤ O.sourceBound S) :
    O.sourceBound S + O.factorBound * referenceBound ≤ O.innerBound S := by
  have factor : 0 < O.factorBound := Nat.mul_pos Nat.two_pos mass
  calc O.sourceBound S + O.factorBound * referenceBound
      ≤ O.factorBound * O.sourceBound S + O.factorBound * O.sourceBound S :=
        Nat.add_le_add (Nat.le_mul_of_pos_left _ factor) (Nat.mul_le_mul_left _ reference)
    _ = O.innerBound S := by rw [innerBound, Nat.mul_assoc, Nat.two_mul]

end Descriptor

end NightstreamFPrime.Spec.Compression.Akita
