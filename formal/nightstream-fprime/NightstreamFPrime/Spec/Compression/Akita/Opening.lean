import NightstreamFPrime.Spec.FieldExtensionRing
import NightstreamFPrime.Spec.Compression.Akita.Commitment
import NightstreamFPrime.Spec.Compression.Akita.Embed

/-! The opening path of a fold with direct coefficient packing, for one claim (Akita,
ePrint 2026/1983, revision of 2026-10-15, Sections 5.3 and 5.4, and Appendix L.2).

- `OpeningShape S` has the numbers of the opening of a commitment of shape `S`: `d_car`, `k` and
  `h_car` with `d_A = k·h_car·d_car` (Akita (55)), the degree `d_D` and the rank `n_D` of the
  opening matrix `D` with `d_D ∣ k·d_car` (Akita Section 4.4), and the compression maps `H_1`,
  `H_2`.
- `nativeView` is the layout (346) of Akita Appendix L.2 for the digits of a value in a ring of a
  smaller degree. `Shape.outerView_eq` states that `Shape.outerView` is `nativeView` for the digits
  of an inner image and `(D, d) = (d_A, d_B)`.
- `OpeningShape.Partials` is the digit vector `ê` in the view of `D` (Akita (346) with
  `(D, d) = (k·d_car, d_D)`). The digits have the base `b_1` and the depth `δ_open` of the inner
  images (Akita (69)). `OpeningShape.partialPlane` is the plane `e_{i,t}` of Akita (58): its
  coefficient `j` is coordinate `t·d_car + j` of partial `i`, which is the order of
  `partial_base_field_coordinate_index` in the Akita code (revision `d50f700f`,
  `crates/akita-params/src/layout/subring_packing_geometry.rs`).
- `OpeningKey` has `D` and the maps `H_1`, `H_2`. `OpeningRows` is the row (106) of Akita with the
  rows (47), or (48) without compression maps. `D` has no slices, which is one slice of `stack`.
- `Point` has the equality weights of the four segments `r_blk`, `r_pos`, `r_pack` and `r_tail` of
  the opening point. `contract` is the plane `t` of the contraction `L` of Akita (57),
  `contract_embed_smul` is Akita (61), and `scalar` is the left side of Akita (59).
- The extension field `E` is a commutative ring with a basis of `k` elements over `Z_q`.
  `extensionBasis` is the basis `(1, U)` of `K = F_q[U]/(U² − 2)`, for `k = 2`. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing

/-- `K` as an algebra over `Z_q`. -/
noncomputable scoped instance extensionAlgebra : Algebra Zq K :=
  (carrierEquiv.ringEquiv.symm.toRingHom.comp
    (algebraMap Zq (QuadraticAlgebra Zq 2 0))).toAlgebra

/-- The basis `β = (1, U)` of `K` over `Z_q` (Akita (58) for `k = 2`). -/
noncomputable def extensionBasis : Module.Basis (Fin 2) Zq K :=
  (QuadraticAlgebra.basis (2 : Zq) 0).map
    (AlgEquiv.ofRingEquiv (f := carrierEquiv.ringEquiv.symm) fun _ => rfl).toLinearEquiv

/-- The coordinates of an element of `K` in `extensionBasis` are its two coordinates. -/
theorem extensionBasis_repr (value : K) : ⇑(extensionBasis.repr value) = ![value.c0, value.c1] :=
  rfl

section View

variable {d d' δ : ℕ} [NeZero d] [NeZero d'] {ι : Type}

/-- The digits of values of `R_{q,d}` in native blocks of `R_{q,d'}` (Akita Appendix L.2, (346)):
coefficient `k` of the element at `(j, y, u)` is coefficient `y·d' + k` of digit `u` of value
`j`. -/
noncomputable def nativeView (d' : ℕ) [NeZero d'] (digits : ι × Fin δ → Rq d) :
    ι × Fin (d / d') × Fin δ → Rq d' :=
  fun column => ofCoeffs fun k => coeff (digits (column.1, column.2.2)) (column.2.1 * d' + k)

omit [NeZero d] [NeZero d'] in
theorem native_address_lt (divides : d' ∣ d) (y : Fin (d / d')) {k : ℕ} (small : k < d') :
    y * d' + k < d :=
  calc y * d' + k < (y + 1) * d' := by rw [Nat.add_mul, one_mul]; omega
    _ ≤ d / d' * d' := Nat.mul_le_mul_right _ y.isLt
    _ = d := Nat.div_mul_cancel divides

theorem nativeView_injective (divides : d' ∣ d) :
    Function.Injective (nativeView d' : (ι × Fin δ → Rq d) → _) := by
  intro digits digits' same
  funext entry
  refine ext_coeff fun j small => ?_
  have positive := NeZero.pos d'
  have column := congrArg (fun x => coeff x (j % d')) (congrFun same
    (entry.1, ⟨j / d', Nat.div_lt_div_of_lt_of_dvd divides small⟩, entry.2))
  simp only [nativeView, coeff_ofCoeffs _ (Nat.mod_lt j positive), Nat.div_add_mod'] at column
  exact column

/-- The views of two digit vectors of the alphabet `A_b` differ by at most `b − 1` in each
coefficient (Akita (49)). -/
theorem nativeView_sub_le [Fintype ι] (divides : d' ∣ d) {b : ℕ} {digits digits' : ι × Fin δ → Rq d}
    (left : Balanced b digits) (right : Balanced b digits') :
    vecNormInf (nativeView d' digits - nativeView d' digits') ≤ b - 1 := by
  refine (vecNormInf_le_iff _ _).mpr fun column => (normInf_le_iff _ _).mpr fun k small => ?_
  rw [Pi.sub_apply, coeff_sub, nativeView, nativeView, coeff_ofCoeffs _ small,
    coeff_ofCoeffs _ small]
  refine (magnitude_sub_le_of_interval (left _ _ (native_address_lt divides column.2.1 small))
    (right _ _ (native_address_lt divides column.2.1 small))).trans ?_
  omega

/-- The view of Akita Appendix L.1 is the view (346) with `(D, d) = (d_A, d_B)`. -/
theorem Shape.outerView_eq (S : Shape) : S.outerView = nativeView S.outerDegree :=
  rfl

end View

/-- The shape of the opening of a commitment of shape `S` at a packing fold: the degree `d_car` of
the challenge subring, the degree `k` of the extension field and the factor `h_car` with
`d_A = k·h_car·d_car` (Akita (55)), the degree `d_D` and the rank `n_D` of the opening matrix `D`
with `d_D ∣ k·d_car` (Akita Section 4.4), and the compression maps `H_1` and `H_2`. -/
structure OpeningShape (S : Shape) where
  carrierDegree : ℕ
  [carrierPositive : NeZero carrierDegree]
  planes : ℕ
  packing : ℕ
  tower : planes * packing * carrierDegree = S.innerDegree
  openDegree : ℕ
  [openPositive : NeZero openDegree]
  openDivides : openDegree ∣ planes * carrierDegree
  openRank : ℕ
  chain : Option ChainShape

attribute [instance] OpeningShape.carrierPositive OpeningShape.openPositive

namespace OpeningShape

variable {S : Shape} (O : OpeningShape S)

/-- `k·h_car`, the stride of the embedding `ι` (Akita (38)). -/
def stride : ℕ := O.planes * O.packing

theorem stride_tower : O.stride * O.carrierDegree = S.innerDegree := O.tower

/-- `d_op / d_D` for the width `d_op = k·d_car` of one partial (Akita (68)). -/
def subcolumns : ℕ := O.planes * O.carrierDegree / O.openDegree

/-- The positions of the input of `D` for one claim, in the order block, `R_{q,d_D}` subcolumn,
digit (Akita (346)). Their number is `m_D` of Akita (69). -/
abbrev OpenColumn : Type := Fin S.blocks × Fin O.subcolumns × Fin S.openingDigits

/-- The digit vector `ê` of the partial opening evaluations in the view of `D`. -/
abbrev Partials : Type := O.OpenColumn → Rq O.openDegree

/-- The payload `v_pub` of the opening path. -/
abbrev Payload : Type := Akita.Payload O.openDegree (1 * O.openRank) O.chain

/-- The plane `e_{i,t} ∈ S` of partial `i` (Akita (58)): its coefficient `j` is coordinate
`t·d_car + j` of the `k·d_car` coordinates that the digits of block `i` give in base `b_1`. -/
noncomputable def partialPlane (digits : O.Partials) (i : Fin S.blocks) (t : Fin O.planes) :
    Rq O.carrierDegree :=
  ofCoeffs fun j => sequence
    (gadget S.openingBase fun entry : Fin O.subcolumns × Fin S.openingDigits => digits (i, entry))
    (t * O.carrierDegree + j)

end OpeningShape

variable {S : Shape} {O : OpeningShape S}

/-- The matrices of the opening path: `D ∈ R_{q,d_D}^{n_D × m_D}` and the compression maps `H_1`,
`H_2` (Akita (70)). -/
structure OpeningKey (O : OpeningShape S) where
  matrix : Matrix (Fin O.openRank) O.OpenColumn (Rq O.openDegree)
  chain : PathKey O.chain

/-- The row (106) of Akita, `D ê = v`, with the rows (47) from `v` to the payload `v_pub`, or
`D ê = v_pub` (48). `D` is applied to the complete vector `ê`, which is one slice. -/
def OpeningRows (key : OpeningKey O) (digits : O.Partials) (chain : PathDigits O.chain)
    (payload : O.Payload) : Prop :=
  Rows O.chain key.chain (stack key.matrix fun _ : Fin 1 => digits) chain payload

/-- The equality weights of the segments of the opening point (Akita Section 5.3): `ẽq(r_blk, i)`,
`ẽq(r_pos, x)`, `ẽq(r_pack, a)` and `ẽq(r_tail, j)`. -/
structure Point (O : OpeningShape S) (E : Type) where
  block : Fin S.blocks → E
  position : Fin S.blockLength → E
  pack : Fin O.stride → E
  tail : Fin O.carrierDegree → E

variable {E : Type} [CommRing E] [Module Zq E]

/-- The plane `t` of the contraction `L` of Akita (57): for the coordinates
`[ẽq(r_pos, x)·ẽq(r_pack, a)]_t` in the basis `β`, the map
`F ↦ Σ_x Σ_a [ẽq(r_pos, x)·ẽq(r_pack, a)]_t · plane a (F_x)`, so `L(F) = Σ_t β_t·L_t(F)`. -/
noncomputable def contract (β : Module.Basis (Fin O.planes) Zq E) (point : Point O E)
    (t : Fin O.planes) : (Fin S.blockLength → Rq S.innerDegree) →ₗ[Zq] Rq O.carrierDegree :=
  ∑ x, ∑ a : Fin O.stride, β.repr (point.position x * point.pack a) t •
    (plane O.carrierDegree O.stride a ∘ₗ LinearMap.proj x)

theorem contract_apply (β : Module.Basis (Fin O.planes) Zq E) (point : Point O E)
    (t : Fin O.planes) (F : Fin S.blockLength → Rq S.innerDegree) :
    contract β point t F = ∑ x, ∑ a : Fin O.stride,
      β.repr (point.position x * point.pack a) t • plane O.carrierDegree O.stride a (F x) := by
  simp [contract]

/-- Akita (61) on each plane: `L_t(ι(c)·F) = c·L_t(F)` in `S`. -/
theorem contract_embed_smul (β : Module.Basis (Fin O.planes) Zq E) (point : Point O E)
    (t : Fin O.planes) (c : Rq O.carrierDegree) (F : Fin S.blockLength → Rq S.innerDegree) :
    contract β point t (embed O.stride O.stride_tower c • F) = c * contract β point t F := by
  rw [contract_apply, contract_apply, Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [Pi.smul_apply, smul_eq_mul, plane_embed_mul O.stride_tower a.isLt, mul_smul_comm]

/-- The left side of the scalar opening (59) of Akita for the planes `e_{i,t}`:
`Σ_i ẽq(r_blk, i) · Σ_j ẽq(r_tail, j) · Σ_t β_t·e_{i,t,j}`. -/
noncomputable def scalar (β : Module.Basis (Fin O.planes) Zq E) (point : Point O E)
    (partials : Fin S.blocks → Fin O.planes → Rq O.carrierDegree) : E :=
  ∑ i, point.block i * ∑ j : Fin O.carrierDegree, point.tail j *
    ∑ t, coeff (partials i t) j • β t

end NightstreamFPrime.Spec.Compression.Akita
