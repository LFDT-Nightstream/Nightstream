import NightstreamFPrime.Spec.Compression.Akita.Fold
import NightstreamFPrime.Spec.Compression.Lib.Cube

/-! The honest prover of one fold of Akita with direct coefficient packing, for one claim
(ePrint 2026/1983, revision of 2026-10-15, Figure 4, steps 1 and 2), the deterministic part of
Akita Lemma 10.7, and the layout of the scalar opening.

- `Encodable b δ` is `Enc` of Akita (147): each coefficient is in `[−M, T]`, the interval of `δ`
  balanced digits in base `b`. `gadget_decompose_of_encodable` gives `G_{b,n} G^{-1}_{b,n} z = z`
  for such a vector, which is Akita (74) for an admissible response.
- `OpeningShape.encode` gives the digits `ê` of the planes of the partials (Akita Section 5.4),
  and `OpeningShape.partialPlane_encode` states that `OpeningShape.partialPlane` decodes them.
- `honest` is the witness of the honest prover, and `honest_satisfies` states that it satisfies
  the branch relation `Witness.Satisfies` when the response is encodable. Akita Theorem 10.9 has
  the probability that no nonce gives an encodable response in the term `ξ_h` of (155). This file
  has no bound for it.
- `scalar_contract` gives the scalar (59) of the contracted source blocks as the sum of the cells
  of the decoded table with the products of the four weights of the point.
- `Point.ofSegments` has the equality weights of four segments, and `scalar_contract_eq_mle`
  states that this scalar is the multilinear evaluation of the decoded table at the point
  `(r_pack, r_tail, r_pos, r_blk)`, with the first variable on the lowest bit of the cell. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib (eqAt mle mle_eq_cubeSum cubeSum_eq_sum_range)

section Encoding

variable {d : ℕ} [NeZero d] {ι : Type}

/-- The conditions on a response base `b_z`: `b_z` is even and `2 ≤ b_z < q`. -/
structure Radix (b : ℕ) : Prop where
  even : b % 2 = 0
  two : 2 ≤ b
  small : b < fieldModulus

/-- `Enc` of Akita (147) and Definition 10.5 on the coefficient-norm route: each coefficient of
`v` is in `[−M, T]` for `M = (b/2)·Σ_{u<δ} b^u` and `T = (b/2 − 1)·Σ_{u<δ} b^u`. -/
def Encodable (b δ : ℕ) (v : ι → Rq d) : Prop :=
  Digits (-((b / 2 : ℕ) : ℤ) * ∑ u ∈ Finset.range δ, (b : ℤ) ^ u)
    ((((b / 2 : ℕ) : ℤ) - 1) * ∑ u ∈ Finset.range δ, (b : ℤ) ^ u) v

/-- The lift of Akita Section 3.3 of a residue with its centered representative at most `T` is
that representative. -/
theorem lift_eq_valMinAbs {b δ : ℕ} {x : Zq}
    (high : x.valMinAbs ≤ (((b / 2 : ℕ) : ℤ) - 1) * ∑ u ∈ Finset.range δ, (b : ℤ) ^ u) :
    lift b δ x = x.valMinAbs := by
  have below : (x.val : ℤ) < fieldModulus := by exact_mod_cast ZMod.val_lt x
  unfold lift
  rw [ZMod.valMinAbs_def_pos] at high ⊢
  generalize (((b / 2 : ℕ) : ℤ) - 1) * ∑ u ∈ Finset.range δ, (b : ℤ) ^ u = top at high ⊢
  split at high <;> split <;> omega

theorem fieldDigit_balanced {b : ℕ} (radix : Radix b) (δ : ℕ) (x : Zq) (u : ℕ) :
    -((b / 2 : ℕ) : ℤ) ≤ (fieldDigit b δ x u).valMinAbs ∧
      (fieldDigit b δ x u).valMinAbs ≤ ((b / 2 : ℕ) : ℤ) - 1 := by
  have range := balancedDigit_range radix.even radix.two (lift b δ x) u
  have small := radix.small
  have value : (fieldDigit b δ x u).valMinAbs = balancedDigit b (lift b δ x) u :=
    (ZMod.valMinAbs_spec _ _).mpr ⟨rfl, by constructor <;> omega⟩
  rw [value]
  exact range

theorem fieldDigit_sum_of_range {b δ : ℕ} (radix : Radix b) {x : Zq}
    (low : -((b / 2 : ℕ) : ℤ) * ∑ u ∈ Finset.range δ, (b : ℤ) ^ u ≤ x.valMinAbs)
    (high : x.valMinAbs ≤ (((b / 2 : ℕ) : ℤ) - 1) * ∑ u ∈ Finset.range δ, (b : ℤ) ^ u) :
    ∑ u ∈ Finset.range δ, (b : Zq) ^ u * fieldDigit b δ x u = x := by
  have value := lift_eq_valMinAbs high
  have recomposed := balancedDigit_sum radix.even radix.two δ (lift b δ x)
    (by rw [value]; exact low) (by rw [value]; exact high)
  have cast := congrArg (Int.cast : ℤ → Zq) recomposed
  simp only [fieldDigit]
  rw [Int.cast_sum] at cast
  simp only [Int.cast_mul, Int.cast_pow, Int.cast_natCast] at cast
  rw [cast, value, ZMod.coe_valMinAbs]

/-- Akita (74) for an admissible response: `G_{b,n} G^{-1}_{b,n} z = z` at the depth `δ` when
each coefficient of `z` is in the interval of `δ` digits. -/
theorem gadget_decompose_of_encodable {b δ : ℕ} (radix : Radix b) {v : ι → Rq d}
    (fits : Encodable b δ v) : gadget b (decompose b v : ι × Fin δ → Rq d) = v := by
  funext j
  refine ext_coeff fun i small => ?_
  rw [gadget, coeff_sum]
  simp only [coeff_natCast_pow_mul, decompose, coeff_ofCoeffs _ small]
  rw [Fin.sum_univ_eq_sum_range fun u => (b : Zq) ^ u * fieldDigit b δ (coeff (v j) i) u]
  exact fieldDigit_sum_of_range radix (fits j i small).1 (fits j i small).2

theorem decompose_balanced_of_radix {b : ℕ} (radix : Radix b) (δ : ℕ) (v : ι → Rq d) :
    Balanced b (decompose b v : ι × Fin δ → Rq d) := by
  intro entry i small
  rw [decompose, coeff_ofCoeffs _ small]
  exact fieldDigit_balanced radix δ _ _

end Encoding

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor}

namespace OpeningShape

/-- The native blocks of `R_{q,d_D}` of the `k·d_car` coordinates of one partial, in the order
plane `t`, then coefficient `j` (Akita Section 5.4): coefficient `k` of block `y` is coordinate
`y·d_D + k`. -/
noncomputable def native (O : OpeningShape S) (planes : Fin O.planes → Rq O.carrierDegree) :
    Fin O.subcolumns → Rq O.openDegree :=
  fun y => ofCoeffs fun k => sequence planes (y * O.openDegree + k)

/-- The digits `ê` of the planes `e_{i,t}` of the partials: the balanced digits in base `b_1` of
the native blocks of each partial (Akita Section 5.4). -/
noncomputable def encode (O : OpeningShape S)
    (planes : Fin S.blocks → Fin O.planes → Rq O.carrierDegree) : O.Partials :=
  fun column => decompose S.openingBase (O.native (planes column.1)) column.2

/-- `partialPlane` decodes the digits of `encode`. -/
theorem partialPlane_encode (O : OpeningShape S) (base : Base S.openingBase S.openingDigits)
    (planes : Fin S.blocks → Fin O.planes → Rq O.carrierDegree) (i : Fin S.blocks)
    (t : Fin O.planes) : O.partialPlane (O.encode planes) i t = planes i t := by
  refine ext_coeff fun j small => ?_
  have inside : (t : ℕ) * O.carrierDegree + j < O.planes * O.carrierDegree :=
    calc (t : ℕ) * O.carrierDegree + j < ((t : ℕ) + 1) * O.carrierDegree := by
          rw [Nat.add_mul, one_mul]; omega
      _ ≤ O.planes * O.carrierDegree := Nat.mul_le_mul_right _ t.isLt
  have column : ((t : ℕ) * O.carrierDegree + j) / O.openDegree < O.subcolumns :=
    Nat.div_lt_div_of_lt_of_dvd O.openDivides inside
  have digits : (fun entry : Fin O.subcolumns × Fin S.openingDigits => O.encode planes (i, entry)) =
      decompose S.openingBase (O.native (planes i)) := rfl
  have unpacked := sequence_apply (O.native (planes i))
    ⟨((t : ℕ) * O.carrierDegree + j) / O.openDegree, column⟩
    (Nat.mod_lt ((t : ℕ) * O.carrierDegree + j) (NeZero.pos O.openDegree))
  simp only [Nat.div_add_mod'] at unpacked
  rw [partialPlane, coeff_ofCoeffs _ small, digits, gadget_decompose base, unpacked]
  simp only [native]
  rw [coeff_ofCoeffs _ (Nat.mod_lt _ (NeZero.pos O.openDegree)), Nat.div_add_mod']
  exact sequence_apply (planes i) t small

end OpeningShape

variable {E : Type} [CommRing E] [Algebra Zq E]

/-- The digits `ê` of the honest prover: the digits of the planes `L_t(G_{b,M}·s_i)` of the
partials `e_i` (Akita (57), (58) and Section 5.4). -/
noncomputable def honestPartials (β : Module.Basis (Fin O.planes) Zq E) (point : Point O E)
    (source : Fin S.blocks → S.Source) : O.Partials :=
  O.encode fun i t => contract β point t (gadget S.sourceBase (source i))

/-- The digits of `H_1`, `H_2` and the payload `v_pub` of the honest prover for the digits `ê`
(Akita (70) and Figure 4, step 1). -/
noncomputable def honestOpening (key : OpeningKey O) (partials : O.Partials) :
    PathDigits O.chain × O.Payload :=
  pathCommit O.chain key.chain (stack key.matrix fun _ : Fin 1 => partials)

/-- The witness of the honest prover of one fold for the commitment state `state` and the
challenges `c_i` (Akita Figure 4, steps 1 and 2): `t̂` and the digits of `F_1`, `F_2` of the
state, the digits `ê` and the digits of `H_1`, `H_2`, and `ẑ = G^{-1}_{b_z,m_A}(z)` for the
response `z` of Akita (73) and (74). -/
noncomputable def honest (β : Module.Basis (Fin O.planes) Zq E) (key : OpeningKey O)
    (point : Point O E) (fold : Descriptor) (challenge : Fin S.blocks → Rq O.carrierDegree)
    (state : Decommitment S) : Witness O fold where
  inner := state.inner
  chain := state.chain
  partials := honestPartials β point state.source
  openChain := (honestOpening key (honestPartials β point state.source)).1
  response := decompose fold.responseBase (response O challenge state.source)

/-- The deterministic part of Akita Lemma 10.7 for one claim with coefficient packing: when the
response is admissible, "all remaining honest equations of the stage hold". The state is an
algebraic decommitment of `u_pub` (`opens`), the statement has the payload `v_pub` and the value
`v` of the honest prover (`payload`, `value`), the bases satisfy `b_1 ≤ b⋆` (`shape`) and
`b_z ≤ b⋆` (`certified`), and the response `z` satisfies `Enc` of Akita Definition 10.5
(`encodable`). Then the honest witness satisfies the branch relation for the challenges. The
hypothesis `encodable` is not a consequence of the others: Akita completeness is not perfect
(Theorem 10.9, (155)). -/
theorem honest_satisfies {β : Module.Basis (Fin O.planes) Zq E} {st : Statement O E}
    {state : Decommitment S} {table : Table} (challenge : Fin S.blocks → Rq O.carrierDegree)
    (shape : S.Honest) (radix : Radix fold.responseBase)
    (certified : fold.responseBase ≤ S.certifiedBase)
    (fits : PathFits O.openDegree (1 * O.openRank) O.chain)
    (opens : state.Opens st.key st.payload table)
    (encodable : Encodable fold.responseBase fold.responseDigits
      (response O challenge state.source))
    (payload : st.openPayload =
      (honestOpening st.openKey (honestPartials β st.point state.source)).2)
    (value : st.value =
      scalar β st.point fun i t => contract β st.point t (gadget S.sourceBase (state.source i))) :
    (honest β st.openKey st.point fold challenge state).Satisfies β st challenge := by
  have planes : O.partialPlane (honestPartials β st.point state.source) =
      fun i t => contract β st.point t (gadget S.sourceBase (state.source i)) :=
    funext fun i => funext fun t => O.partialPlane_encode shape.opening _ i t
  have recomposed := gadget_decompose_of_encodable radix encodable
  refine {
    carried := {
      rows := opens.rows
      openRows := ?_
      value := ?_
      innerDigits := opens.innerDigits
      chainDigits := opens.chainDigits
      partialDigits := fun column =>
        ((decompose_balanced shape.opening _).mono shape.certified) column.2
      openDigits := (pathCommit_rows fits st.openKey.chain _).2 }
    evaluation := fun t => ?_
    innerRows := ?_
    responseDigits := (decompose_balanced_of_radix radix _ _).mono certified }
  · show Rows O.chain st.openKey.chain _ _ st.openPayload
    rw [payload]
    exact (pathCommit_rows fits st.openKey.chain _).1
  · show scalar β st.point (O.partialPlane (honestPartials β st.point state.source)) = st.value
    rw [planes, value]
  · show ∑ i, challenge i * O.partialPlane (honestPartials β st.point state.source) i t =
      contract β st.point t (gadget S.sourceBase (gadget fold.responseBase
        (decompose fold.responseBase (response O challenge state.source))))
    rw [recomposed, response, gadget_sum, map_sum, planes]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [gadget_smul, contract_embed_smul]
  · show st.key.inner.mulVec (gadget fold.responseBase
      (decompose fold.responseBase (response O challenge state.source))) =
      ∑ i, embed O.stride O.stride_tower (challenge i) • gadget S.openingBase (state.inner i)
    rw [recomposed, response, Matrix.mulVec_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Matrix.mulVec_smul, opens.innerRows]

section Layout

theorem coeff_contract (β : Module.Basis (Fin O.planes) Zq E) (point : Point O E)
    (t : Fin O.planes) (F : Fin S.blockLength → Rq S.innerDegree) {j : ℕ}
    (small : j < O.carrierDegree) :
    coeff (contract β point t F) j = ∑ x, ∑ a : Fin O.stride,
      β.repr (point.position x * point.pack a) t * coeff (F x) (a + O.stride * j) := by
  rw [contract_apply, coeff_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [coeff_sum]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [coeff_smul, coeff_plane _ _ _ small]

/-- Cell `(i·M + x)·d_A + k` of the decoded table is coefficient `k` of entry `x` of
`G_{b,M}·s_i` (Akita Section 4.5, coefficient packing). -/
theorem Shape.decode_cell (S : Shape) (source : Fin S.blocks → S.Source) (i : Fin S.blocks)
    (x : Fin S.blockLength) {k : ℕ} (small : k < S.innerDegree) :
    S.decode source ((i * S.blockLength + x) * S.innerDegree + k) =
      coeff (gadget S.sourceBase (source i) x) k := by
  have inner : (x : ℕ) * S.innerDegree + k < S.blockLength * S.innerDegree :=
    calc (x : ℕ) * S.innerDegree + k < ((x : ℕ) + 1) * S.innerDegree := by
          rw [Nat.add_mul, one_mul]; omega
      _ ≤ S.blockLength * S.innerDegree := Nat.mul_le_mul_right _ x.isLt
  have address : ((i : ℕ) * S.blockLength + x) * S.innerDegree + k =
      (x : ℕ) * S.innerDegree + k + (i : ℕ) * (S.blockLength * S.innerDegree) := by ring
  have quotient : ((x : ℕ) * S.innerDegree + k + (i : ℕ) * (S.blockLength * S.innerDegree)) /
      (S.blockLength * S.innerDegree) = i := by
    rw [Nat.add_mul_div_right _ _ (Nat.zero_lt_of_lt inner), Nat.div_eq_of_lt inner, zero_add]
  have rest : ((x : ℕ) * S.innerDegree + k + (i : ℕ) * (S.blockLength * S.innerDegree)) %
      (S.blockLength * S.innerDegree) = (x : ℕ) * S.innerDegree + k := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt inner]
  rw [address]
  simp only [Shape.decode, quotient, rest, i.isLt, ↓reduceDIte, Fin.eta]
  exact sequence_apply _ x small

/-- The layout of the scalar opening (Akita Section 5.3, after (59)): the scalar (59) of the
contracted source blocks `L(G_{b,M}·s_i)` is the sum over the block `i`, the position `x`, the
coefficient `a` of a group and the group `j` of the cell `(i·M + x)·d_A + a + k·h_car·j` of the
decoded table (Akita (56)) with the weight
`ẽq(r_blk, i)·ẽq(r_pos, x)·ẽq(r_pack, a)·ẽq(r_tail, j)`. -/
theorem scalar_contract (β : Module.Basis (Fin O.planes) Zq E) (point : Point O E)
    (source : Fin S.blocks → S.Source) :
    scalar β point (fun i t => contract β point t (gadget S.sourceBase (source i))) =
      ∑ i : Fin S.blocks, ∑ x : Fin S.blockLength, ∑ a : Fin O.stride,
        ∑ j : Fin O.carrierDegree,
          S.decode source ((i * S.blockLength + x) * S.innerDegree + (a + O.stride * j)) •
            (point.block i * point.position x * point.pack a * point.tail j) := by
  have plane : ∀ (i : Fin S.blocks) (j : Fin O.carrierDegree),
      ∑ t, coeff (contract β point t (gadget S.sourceBase (source i))) j • β t =
        ∑ x : Fin S.blockLength, ∑ a : Fin O.stride,
          S.decode source ((i * S.blockLength + x) * S.innerDegree + (a + O.stride * j)) •
            (point.position x * point.pack a) := by
    intro i j
    simp only [coeff_contract β point _ _ j.isLt, Finset.sum_smul]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun x _ => ?_
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun a _ => ?_
    rw [S.decode_cell source i x (plane_address_lt O.stride_tower a.isLt j.isLt)]
    simp only [mul_comm (β.repr _ _), mul_smul]
    rw [← Finset.smul_sum, β.sum_repr]
  rw [scalar]
  refine Finset.sum_congr rfl fun i _ => ?_
  simp only [plane, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun j _ => ?_
  rw [mul_smul_comm, mul_smul_comm]
  congr 1
  ring

omit [Algebra Zq E] in
theorem eqAt_append (p q : List E) {a : ℕ} (small : a < 2 ^ p.length) (b : ℕ) :
    eqAt (p ++ q) (a + 2 ^ p.length * b) = eqAt p a * eqAt q b := by
  induction p generalizing a with
  | nil =>
      have zero : a = 0 := by simpa using small
      simp [eqAt, zero]
  | cons r rest step =>
      rw [List.length_cons, pow_succ] at small
      have shape : a + 2 ^ (rest.length + 1) * b = a + 2 * (2 ^ rest.length * b) := by
        rw [pow_succ', Nat.mul_assoc]
      have parity : (a + 2 * (2 ^ rest.length * b)) % 2 = a % 2 := by omega
      have half : (a + 2 * (2 ^ rest.length * b)) / 2 = a / 2 + 2 ^ rest.length * b := by omega
      rw [List.cons_append, List.length_cons, shape]
      simp only [eqAt, parity, half]
      rw [step (by omega : a / 2 < 2 ^ rest.length), mul_assoc]

theorem sum_fin_mul {M : Type} [AddCommMonoid M] (m n : ℕ) (f : ℕ → M) :
    ∑ c : Fin (m * n), f c = ∑ i : Fin m, ∑ j : Fin n, f (j + n * i) :=
  ((finProdFinEquiv.sum_comp fun c : Fin (m * n) => f c).symm.trans (Fintype.sum_prod_type _))

/-- A sum over the cells of a table as a sum over the block `i`, the position `x`, the
coefficient `a` of a group and the group `j`. -/
theorem sum_range_cells {M : Type} [AddCommMonoid M] (B L c s : ℕ) (g : ℕ → M) :
    ∑ cell ∈ Finset.range (B * (L * (c * s))), g cell =
      ∑ i : Fin B, ∑ x : Fin L, ∑ a : Fin s, ∑ j : Fin c,
        g (a + s * j + c * s * x + L * (c * s) * i) := by
  rw [Finset.sum_range, sum_fin_mul]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [sum_fin_mul L (c * s) fun n => g (n + L * (c * s) * i)]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [sum_fin_mul c s fun n => g (n + c * s * x + L * (c * s) * i), Finset.sum_comm]

/-- The weights of an opening point with the segments `r_blk`, `r_pos`, `r_pack` and `r_tail`:
`ẽq(r_blk, i)`, `ẽq(r_pos, x)`, `ẽq(r_pack, a)` and `ẽq(r_tail, j)` (Akita Section 5.3). -/
def Point.ofSegments (O : OpeningShape S) (block position pack tail : List E) : Point O E :=
  ⟨fun i => eqAt block i, fun x => eqAt position x, fun a => eqAt pack a, fun j => eqAt tail j⟩

/-- The scalar (59) of the contracted source blocks is the multilinear evaluation of the decoded
table (Akita Section 5.3: "Contracting the block and carrier axes of these partials recovers the
original extension-field claim"). The point is `(r_pack, r_tail, r_pos, r_blk)` with the first
variable on the lowest bit of the cell `(i·M + x)·d_A + a + k·h_car·j`, and `B`, `M`, `k·h_car`
and `d_car` are the powers of two of the lengths of the segments. -/
theorem scalar_contract_eq_mle (β : Module.Basis (Fin O.planes) Zq E)
    (block position pack tail : List E) (blocks : S.blocks = 2 ^ block.length)
    (length : S.blockLength = 2 ^ position.length) (stride : O.stride = 2 ^ pack.length)
    (degree : O.carrierDegree = 2 ^ tail.length) (source : Fin S.blocks → S.Source) :
    scalar β (Point.ofSegments O block position pack tail)
        (fun i t => contract β (Point.ofSegments O block position pack tail) t
          (gadget S.sourceBase (source i))) =
      mle (pack ++ (tail ++ (position ++ block)))
        (fun cell => algebraMap Zq E (S.decode source cell)) := by
  have total : 2 ^ (pack ++ (tail ++ (position ++ block))).length =
      S.blocks * (S.blockLength * (O.carrierDegree * O.stride)) := by
    rw [blocks, length, stride, degree]
    simp only [List.length_append, pow_add]
    ring
  rw [scalar_contract, mle_eq_cubeSum, cubeSum_eq_sum_range, total, sum_range_cells]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun x _ =>
    Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun j _ => ?_
  have cell : (a : ℕ) + O.stride * j + O.carrierDegree * O.stride * x +
      S.blockLength * (O.carrierDegree * O.stride) * i =
        (i * S.blockLength + x) * S.innerDegree + (a + O.stride * j) := by
    rw [← O.stride_tower]
    ring
  have nested : ((i : ℕ) * S.blockLength + x) * S.innerDegree + (a + O.stride * j) =
      a + 2 ^ pack.length * (j + 2 ^ tail.length * (x + 2 ^ position.length * i)) := by
    rw [← O.stride_tower, ← stride, ← degree, ← length]
    ring
  have weight : eqAt (pack ++ (tail ++ (position ++ block)))
      (((i : ℕ) * S.blockLength + x) * S.innerDegree + (a + O.stride * j)) =
        eqAt pack a * (eqAt tail j * (eqAt position x * eqAt block i)) := by
    rw [nested, eqAt_append _ _ (a.isLt.trans_eq stride),
      eqAt_append _ _ (j.isLt.trans_eq degree), eqAt_append _ _ (x.isLt.trans_eq length)]
  rw [cell, weight, Algebra.smul_def]
  simp only [Point.ofSegments]
  ring

end Layout

end NightstreamFPrime.Spec.Compression.Akita
