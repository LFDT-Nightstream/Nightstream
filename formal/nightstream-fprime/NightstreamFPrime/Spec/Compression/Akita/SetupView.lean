import NightstreamFPrime.Spec.Compression.Akita.Setup
import NightstreamFPrime.Spec.Compression.Akita.Batch
import NightstreamFPrime.Spec.Compression.Akita.RowBatch

/-! The matrices of a level of Akita as views of the shared setup vector (ePrint 2026/1983,
revision of 2026-11-23, Section 4.3 with (35), and Section 9 with (133)).

- `Setup.shared S d n width code` is the prefix view (35) of the flat setup vector `S` with the
  degree `d`, `n` rows and `width` columns, with the column `c` of the matrix at the column
  `code c` of the view. `Setup.shared_sum` states that each entry is linear in `S`.
- `Shape.innerSetupCode`, `Shape.outerSetupCode` and `Batch.openSetupCode` are the orders of the
  columns of `A`, `B` and `D` in the witness layout. Their widths are the widths of
  `Shape.innerInstance`, `Shape.outerInstance` and `Batch.openInstance`.
- `Batch.Statement.over st S` is the statement `st` with the matrices `A` and `B` of each group
  and the matrix `D` of the shared path replaced by these views of `S`. The compression maps of
  `st` stay. `Batch.Statement.over_eq` states that a statement with these views is its statement
  over `S`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

namespace Setup

variable {N : ℕ}

/-- The prefix view (35) of the flat setup vector `S` with the degree `d`, `n` rows and `width`
columns, with the column `c` of the matrix at the column `code c` of the view. -/
noncomputable def shared (S : Fin N → Zq) (d n width : ℕ) [NeZero d] {cols : Type}
    (code : cols → Fin width) : Matrix (Fin n) cols (Rq d) :=
  (view ⟨d, n, width, false, 0⟩ S).submatrix id code

/-- Each entry of a view is linear in the flat setup vector: it is `Σ_p S_p·M(e_p)`, with the
view `M(e_p)` of the unit vector `e_p`. -/
theorem shared_sum (S : Fin N → Zq) (d n width : ℕ) [NeZero d] {cols : Type}
    (code : cols → Fin width) (i : Fin n) (c : cols) :
    shared S d n width code i c = ∑ p, S p • shared (Pi.single p 1) d n width code i c := by
  refine ext_coeff fun l small => ?_
  rw [coeff_sum]
  simp only [coeff_smul, shared, Matrix.submatrix_apply, id, view, Matrix.of_apply,
    coeff_ofCoeffs _ small]
  by_cases inside : ((i : ℕ) * width + code c) * d + l < N
  · simp only [dite_eq_left inside, Pi.single_apply, mul_ite, mul_one, mul_zero]
    rw [Finset.sum_ite_eq, ite_eq_left (Finset.mem_univ _)]
  · simp [inside]

end Setup

namespace Shape

variable (S : Shape)

/-- The order of the columns of `A` in the witness layout: the column `(x, u)` of the position
`x` of the block and the digit `u` is the column `x·δ_com + u` of the view. -/
def innerSetupCode :
    Fin S.blockLength × Fin S.sourceDigits ≃ Fin (S.blockLength * S.sourceDigits) :=
  finProdFinEquiv

/-- The order of the columns of `B` in the witness layout: the position in the slice, the row of
`A`, the subcolumn of `R_{q,d_B}` and the digit, in this order (Akita Section 4.5). -/
def outerSetupCode : Fin S.sliceLength × S.OuterColumn ≃
    Fin (S.sliceLength * (S.innerRank * (S.ratio * S.openingDigits))) :=
  (Equiv.prodCongr (Equiv.refl _)
    ((Equiv.prodCongr (Equiv.refl _) finProdFinEquiv).trans finProdFinEquiv)).trans finProdFinEquiv

end Shape

namespace Batch

variable {n : ℕ} (B : Batch (Fin n))

/-- The order of the columns of `D` in the witness layout: the columns of the groups in the
order `groupOrder` of the pinned code (the groups of the plan in
`crates/akita-types/src/setup_contribution/plan/prepare.rs` of the Akita code; Akita Section 4.3
has "in group order"), and in a group the block, the subcolumn and the digit, in this order
(Akita (69), (70)). -/
def openSetupCode : B.OpenColumn ≃
    Fin (∑ g, (B.shape g).blocks * ((B.opening g).subcolumns * (B.shape g).openingDigits)) :=
  (Equiv.sigmaCongrRight fun _ =>
    (Equiv.prodCongr (Equiv.refl _) finProdFinEquiv).trans finProdFinEquiv).trans
    (place fun g => (B.shape g).blocks * ((B.opening g).subcolumns * (B.shape g).openingDigits)).symm

variable {B} {E : Type} {N : ℕ}

/-- The statement `st` with the matrices of the shared setup vector `S` (Akita Section 4.3 and
(35)): the matrices `A` and `B` of each group and the matrix `D` of the shared path are the views
of `S` in the orders `innerSetupCode`, `outerSetupCode` and `openSetupCode`. The compression maps
of `st` stay. -/
noncomputable def Statement.over (st : B.Statement E) (S : Fin N → Zq) : B.Statement E where
  key g :=
    ⟨Setup.shared S (B.shape g).innerDegree (B.shape g).innerRank _ (B.shape g).innerSetupCode,
      Setup.shared S (B.shape g).outerDegree (B.shape g).outerRank _ (B.shape g).outerSetupCode,
      (st.key g).chain⟩
  openKey := ⟨Setup.shared S B.openDegree B.openRank _ B.openSetupCode, st.openKey.chain⟩
  payload := st.payload
  openPayload := st.openPayload
  point := st.point
  weight := st.weight
  value := st.value

/-- A statement whose matrices `A`, `B` and `D` are the views of `S` is its statement over
`S`. -/
theorem Statement.over_eq (st : B.Statement E) (S : Fin N → Zq)
    (inner : ∀ g, (st.key g).inner =
      Setup.shared S (B.shape g).innerDegree (B.shape g).innerRank _ (B.shape g).innerSetupCode)
    (outer : ∀ g, (st.key g).outer =
      Setup.shared S (B.shape g).outerDegree (B.shape g).outerRank _ (B.shape g).outerSetupCode)
    (opened : st.openKey.matrix = Setup.shared S B.openDegree B.openRank _ B.openSetupCode) :
    st.over S = st := by
  obtain ⟨key, openKey, payload, openPayload, point, weight, value⟩ := st
  simp only [Statement.over] at inner outer opened ⊢
  congr 1
  · funext g
    rw [← inner g, ← outer g]
  · rw [← opened]

end Batch

end NightstreamFPrime.Spec.Compression.Akita
