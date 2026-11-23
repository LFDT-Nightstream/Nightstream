import NightstreamFPrime.Spec.Compression.Akita.SetupView
import NightstreamFPrime.Spec.Compression.Akita.Extract.Offload

/-! The split (132) of the public weight of an offloaded fold of Akita (ePrint 2026/1983, revision
of 2026-11-23, Section 9 with (132) and (133)).

The matrices `A`, `B` and `D` of a level are views of the shared setup vector `S`
(`Batch.Statement.over`), and the compression maps `F` and `H` are local. Each public multiplier
of a ring row is linear in `S` or does not depend on `S`, so the public weight is affine in `S`:
`m_{τ_1}(r_2) = m_local(r_2) + Σ_p S_p·ω_setup(p)`, with `m_local` the weight at `S = 0` and
`ω_setup(p)` the coefficient of `S_p`.

- `Affine f` states that `f(S) = f(0) + Σ_p S_p·(f(e_p) − f(0))` for the unit vectors `e_p`.
- `RingRow.scan` and `Form.scan` are the linear forms `Σ_c Σ_j A_c(α)·α^j·φ(addr_c(j))` of
  the check (112). `AffineRow` and `AffineForm` state that a family of rows or sums depends on `S`
  only through these forms, in the affine way, and that the degree of a row does not depend on
  `S`.
- `Batch.flatRows_affine` states this for each ring row of `Batch.flatRows` for `st.over S`, and
  `Batch.weight_affine` states that the public weight of `Batch.liftedReduction` for `st.over S`
  is affine in `S`.

`Batch.codeScan` in `Extract/SetupFactor.lean` is the split `SetupScan` for the setup prefix. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib Polynomial

section Affine

variable {N : ℕ} {E : Type} [CommRing E] [Algebra Zq E]

/-- The value `f(S)` is affine in the flat setup vector `S`: it is `f(0)` plus
`Σ_p S_p·(f(e_p) − f(0))` for the unit vectors `e_p`. -/
def Affine (f : (Fin N → Zq) → E) : Prop :=
  ∀ S, f S = f 0 + ∑ p, algebraMap Zq E (S p) * (f (Pi.single p 1) - f 0)

namespace Affine

variable {f g : (Fin N → Zq) → E}

theorem congr (same : ∀ S, f S = g S) (affine : Affine g) : Affine f := by
  rw [funext same]
  exact affine

theorem of_eq (same : ∀ S, f S = f 0) : Affine f := fun S => by
  rw [same S, Finset.sum_eq_zero fun p _ => by rw [same (Pi.single p 1), sub_self, mul_zero],
    add_zero]

theorem of_sum (expand : ∀ S, f S = ∑ p, algebraMap Zq E (S p) * f (Pi.single p 1)) :
    Affine f := fun S => by
  have zero : f 0 = 0 := by
    rw [expand 0]
    simp
  rw [zero, zero_add]
  simp only [sub_zero]
  exact expand S

theorem add (left : Affine f) (right : Affine g) : Affine fun S => f S + g S := fun S => by
  show f S + g S = f 0 + g 0 + ∑ p, algebraMap Zq E (S p) *
    (f (Pi.single p 1) + g (Pi.single p 1) - (f 0 + g 0))
  rw [left S, right S]
  simp only [show ∀ p : Fin N, f (Pi.single p 1) + g (Pi.single p 1) - (f 0 + g 0) =
    (f (Pi.single p 1) - f 0) + (g (Pi.single p 1) - g 0) from fun p => by ring, mul_add,
    Finset.sum_add_distrib]
  ring

theorem sub (left : Affine f) (right : Affine g) : Affine fun S => f S - g S := fun S => by
  show f S - g S = f 0 - g 0 + ∑ p, algebraMap Zq E (S p) *
    (f (Pi.single p 1) - g (Pi.single p 1) - (f 0 - g 0))
  rw [left S, right S]
  simp only [show ∀ p : Fin N, f (Pi.single p 1) - g (Pi.single p 1) - (f 0 - g 0) =
    (f (Pi.single p 1) - f 0) - (g (Pi.single p 1) - g 0) from fun p => by ring, mul_sub,
    Finset.sum_sub_distrib]
  ring

theorem mul_left (affine : Affine f) (a : E) : Affine fun S => a * f S := fun S => by
  show a * f S = a * f 0 + ∑ p, algebraMap Zq E (S p) * (a * f (Pi.single p 1) - a * f 0)
  rw [affine S, mul_add, Finset.mul_sum]
  exact congrArg _ (Finset.sum_congr rfl fun p _ => by ring)

theorem mul_right (affine : Affine f) (a : E) : Affine fun S => f S * a :=
  congr (fun S => mul_comm (f S) a) (affine.mul_left a)

theorem sum {ι : Type} (s : Finset ι) {f : ι → (Fin N → Zq) → E}
    (each : ∀ i, Affine (f i)) :
    Affine fun S => ∑ i ∈ s, f i S := fun S => by
  show ∑ i ∈ s, f i S = ∑ i ∈ s, f i 0 + ∑ p, algebraMap Zq E (S p) *
    (∑ i ∈ s, f i (Pi.single p 1) - ∑ i ∈ s, f i 0)
  rw [Finset.sum_congr rfl fun i _ => each i S, Finset.sum_add_distrib, Finset.sum_comm]
  refine congrArg _ (Finset.sum_congr rfl fun p _ => ?_)
  rw [← Finset.sum_sub_distrib, Finset.mul_sum]

theorem ite (c : Prop) [Decidable c] (affine : Affine f) :
    Affine fun S => if c then f S else 0 := by
  by_cases holds : c
  · exact congr (fun S => ite_eq_left holds) affine
  · exact of_eq fun S => by rw [ite_eq_right holds, ite_eq_right holds]

/-- The value at `α` of a ring element that is linear in `S` is affine in `S`. -/
theorem of_ring {d : ℕ} [NeZero d] (α : E) {M : (Fin N → Zq) → Rq d}
    (expand : ∀ S, M S = ∑ p, S p • M (Pi.single p 1)) :
    Affine fun S => aeval α (rep (M S)) :=
  of_sum fun S => by
    show reducedEval α (M S) = ∑ p, algebraMap Zq E (S p) * reducedEval α (M (Pi.single p 1))
    rw [expand S, map_sum]
    exact Finset.sum_congr rfl fun p _ => by rw [map_smul, Algebra.smul_def]

end Affine

theorem aeval_rep_neg {d : ℕ} [NeZero d] (α : E) (x : Rq d) :
    aeval α (rep (-x)) = -aeval α (rep x) := by
  rw [← reducedEval_apply, ← reducedEval_apply, map_neg]

/-- The linear form `Σ_c Σ_{j<d_r} A_{r,c}(α)·α^j·φ(addr_{r,c}(j))` of the row: with
`φ = 1_{some u}` it is the weight `readWeight` of the entry `u` in the check (112). -/
noncomputable def RingRow.scan (row : RingRow) (α : E) (φ : Option ℕ → E) : E :=
  ∑ c, ∑ j ∈ Finset.range row.degree,
    aeval α (rep (row.multiplier c)) * α ^ j * φ (row.address c j)

theorem RingRow.readWeight_eq_scan (row : RingRow) (α : E) (u : ℕ) :
    row.readWeight α u = row.scan α fun o => if o = some u then 1 else 0 := by
  unfold readWeight scan
  simp only [mul_ite, mul_one, mul_zero]

/-- The linear form `Σ_c Σ_{j<d} A_c(α)·α^j·φ(addr_c(j))` of a sum (110). -/
noncomputable def Form.scan {d : ℕ} [NeZero d] (f : Form d) (α : E) (φ : Option ℕ → E) :
    E :=
  ∑ c, ∑ j ∈ Finset.range d, aeval α (rep (f.multiplier c)) * α ^ j * φ (f.address c j)

theorem Form.scan_row {d : ℕ} [NeZero d] (f : Form d) (target : Rq d) (α : E)
    (φ : Option ℕ → E) : (f.row target).scan α φ = f.scan α φ :=
  (Fintype.equivFin f.Occurrence).symm.sum_comp fun c =>
    ∑ j ∈ Finset.range d, aeval α (rep (f.multiplier c)) * α ^ j * φ (f.address c j)

theorem Form.scan_sub {d : ℕ} [NeZero d] (f g : Form d) (α : E) (φ : Option ℕ → E) :
    (f.sub g).scan α φ = f.scan α φ - g.scan α φ := by
  unfold Form.scan Form.sub
  rw [Fintype.sum_sum_type]
  simp only [Sum.elim_inl, Sum.elim_inr, aeval_rep_neg, neg_mul, Finset.sum_neg_distrib,
    sub_eq_add_neg]

theorem Form.scan_ofMatrix {d n : ℕ} [NeZero d] {cols : Type} [Fintype cols]
    (M : Matrix (Fin n) cols (Rq d)) (address : cols → ℕ → Option ℕ) (i : Fin n) (α : E)
    (φ : Option ℕ → E) :
    (Form.ofMatrix M address i).scan α φ =
      ∑ c, ∑ j ∈ Finset.range d, aeval α (rep (M i c)) * α ^ j * φ (address c j) :=
  rfl

theorem Form.scan_response {Sh : Shape} (fold : Descriptor) (key : Key Sh) (a : Fin Sh.innerRank)
    (α : E) (φ : Option ℕ → E) :
    (responseForm fold key a).scan α φ =
      ∑ col, ∑ j ∈ Finset.range Sh.innerDegree,
        aeval α (rep (key.inner a col.1 *
          (fold.responseBase : Rq Sh.innerDegree) ^ (col.2 : ℕ))) * α ^ j *
          φ (Sh.responseCell fold col j) :=
  rfl

variable (E) in
/-- A family of sums (110) that depends on `S` in the affine way: each linear form `Form.scan`
is affine in `S`. -/
def AffineForm {d : ℕ} [NeZero d] (F : (Fin N → Zq) → Form d) : Prop :=
  ∀ (α : E) (φ : Option ℕ → E), Affine fun S => (F S).scan α φ

variable (E) in
/-- A family of ring rows that depends on `S` in the affine way: the degree does not depend on
`S`, and each linear form `RingRow.scan` is affine in `S`. -/
structure AffineRow (F : (Fin N → Zq) → RingRow) : Prop where
  degree : ∀ S, (F S).degree = (F 0).degree
  scan : ∀ (α : E) (φ : Option ℕ → E), Affine fun S => (F S).scan α φ

namespace AffineForm

variable {d : ℕ} [NeZero d]

theorem sub {F : (Fin N → Zq) → Form d} (affine : AffineForm E F) (g : Form d) :
    AffineForm E fun S => (F S).sub g := fun α φ =>
  Affine.congr (fun _ => Form.scan_sub _ _ _ _)
    ((affine α φ).sub (Affine.of_eq (f := fun _ => g.scan α φ) fun _ => rfl))

theorem const_sub {F : (Fin N → Zq) → Form d} (g : Form d) (affine : AffineForm E F) :
    AffineForm E fun S => g.sub (F S) := fun α φ =>
  Affine.congr (fun _ => Form.scan_sub _ _ _ _)
    ((Affine.of_eq (f := fun _ => g.scan α φ) fun _ => rfl).sub (affine α φ))

/-- The row `r` of `M(S)·x` for a matrix `M(S)` that is linear in `S`. -/
theorem ofMatrix {n : ℕ} {cols : Type} [Fintype cols]
    {M : (Fin N → Zq) → Matrix (Fin n) cols (Rq d)}
    (expand : ∀ S i c, M S i c = ∑ p, S p • M (Pi.single p 1) i c)
    (address : cols → ℕ → Option ℕ) (i : Fin n) :
    AffineForm E fun S => Form.ofMatrix (M S) address i := fun α φ =>
  Affine.congr (fun S => Form.scan_ofMatrix (M S) address i α φ)
    (Affine.sum _ fun c => Affine.sum _ fun _ =>
      ((Affine.of_ring α fun S => expand S i c).mul_right _).mul_right _)

/-- The row `a` of `A(S)·G_{b_z,m_A}·ẑ` for a matrix `A(S)` that is linear in `S`. -/
theorem response {Sh : Shape} (fold : Descriptor) {key : (Fin N → Zq) → Key Sh}
    (expand : ∀ S a c, (key S).inner a c = ∑ p, S p • (key (Pi.single p 1)).inner a c)
    (a : Fin Sh.innerRank) : AffineForm E fun S => responseForm fold (key S) a := fun α φ =>
  Affine.congr (fun S => Form.scan_response fold (key S) a α φ)
    (Affine.sum _ fun col => Affine.sum _ fun _ =>
      ((Affine.of_ring α fun S => by
        rw [expand S a col.1, Finset.sum_mul]
        simp only [smul_mul_assoc]).mul_right _).mul_right _)

end AffineForm

namespace AffineRow

variable {F : (Fin N → Zq) → RingRow}

theorem of_eq (same : ∀ S, F S = F 0) : AffineRow E F :=
  ⟨fun S => by rw [same S], fun _ _ => Affine.of_eq fun S => by rw [same S]⟩

theorem relocate (affine : AffineRow E F) (move : ℕ → ℕ) :
    AffineRow E fun S => (F S).relocate move :=
  ⟨affine.degree, fun α φ => affine.scan α fun o => φ (o.map move)⟩

theorem row {d : ℕ} [NeZero d] {F : (Fin N → Zq) → Form d} (affine : AffineForm E F)
    (target : (Fin N → Zq) → Rq d) : AffineRow E fun S => (F S).row (target S) :=
  ⟨fun _ => rfl, fun α φ => Affine.congr (fun _ => Form.scan_row _ _ _ _) (affine α φ)⟩

theorem append {m k : ℕ} {a : (Fin N → Zq) → Fin m → RingRow}
    {b : (Fin N → Zq) → Fin k → RingRow} (left : ∀ i, AffineRow E fun S => a S i)
    (right : ∀ j, AffineRow E fun S => b S j) (r : Fin (m + k)) :
    AffineRow E fun S => Fin.append (a S) (b S) r := by
  refine Fin.addCases (fun i => ?_) (fun j => ?_) r
  · simpa only [Fin.append_left] using left i
  · simpa only [Fin.append_right] using right j

theorem gather {n : ℕ} {count : Fin n → ℕ}
    {rows : (Fin N → Zq) → (g : Fin n) → Fin (count g) → RingRow}
    (each : ∀ g i, AffineRow E fun S => rows S g i) (r : Fin (∑ g, count g)) :
    AffineRow E fun S => Akita.gather (rows S) r :=
  each _ _

theorem imageRow {d n : ℕ} [NeZero d] : (chain : Option ChainShape) →
    {image : (Fin N → Zq) → Fin n → Form d} → (∀ j, AffineForm E fun S => image S j) →
    (first : ℕ → Option ℕ) → (payload : Payload d n chain) →
    ∀ j, AffineRow E fun S => Akita.imageRow chain (image S) first payload j
  | none, _, images, _, payload, j => row (images j) fun _ => payload j
  | some _, _, images, first, _, j => row ((images j).sub (Form.recompose d n first j)) fun _ => 0

/-- The weight `liftWeight` of an entry is affine in `S` when the reads of the quotient do not
depend on `S`. -/
theorem liftWeight (affine : AffineRow E F) {q : (Fin N → Zq) → QuotientRead} (α : E) (u : ℕ)
    (same : ∀ S, (q S).weight (F S).degree α u = (q 0).weight (F 0).degree α u) :
    Affine fun S => (F S).liftWeight (q S) α u :=
  (Affine.congr (fun S => RingRow.readWeight_eq_scan _ _ _) (affine.scan α _)).sub
    (Affine.of_eq fun S => by rw [same S, affine.degree S])

end AffineRow

end Affine

variable {N : ℕ} {E : Type} [Field E] [Algebra Zq E]

/-- The weight of the entry `u` in the row `i` of the field rows of the quotient-lift reduction
(Akita (112), (113)), for the join `(count, slot, basis)`, the weights `lift r` of `liftWeight` of
the ring rows and the weight `scalar` of the scalar row. -/
noncomputable def liftEntry {m : ℕ} (count : ℕ) (slot : Fin m → Fin count)
    (basis : Fin m → E) (lift : Fin m → E) (scalar : E) (i : ℕ) : E :=
  if small : i < count then ∑ r, if slot r = ⟨i, small⟩ then basis r * lift r else 0
  else if i = count then scalar else 0

theorem LiftRelation.liftRows_weightAt (rel : LiftRelation E) (α : E) (i u : ℕ) :
    (rel.liftRows α).weightAt i u = liftEntry rel.join.count rel.join.slot rel.join.basis
      (fun r => (rel.row r).liftWeight (rel.quotient r) α u) (rel.weight u) i := by
  unfold FieldRows.weightAt liftEntry
  by_cases small : i < rel.join.count
  · rw [dite_eq_left (show i < (rel.liftRows α).count from Nat.lt_succ_of_lt small),
      dite_eq_left small]
    show (rel.liftRows α).weight (Fin.castSucc ⟨i, small⟩) u = _
    simp only [LiftRelation.liftRows, Fin.lastCases_castSucc]
  · rw [dite_eq_right small]
    by_cases last : i = rel.join.count
    · subst last
      rw [dite_eq_left (show rel.join.count < (rel.liftRows α).count from Nat.lt_succ_self _),
        ite_eq_left rfl]
      show (rel.liftRows α).weight (Fin.last _) u = _
      simp only [LiftRelation.liftRows, Fin.lastCases_last]
    · rw [ite_eq_right last, dite_eq_right (show ¬ i < (rel.liftRows α).count from by
        show ¬ i < rel.join.count + 1
        omega)]

theorem Affine.liftEntry {m count : ℕ} (slot : Fin m → Fin count) (basis : Fin m → E)
    {lift : (Fin N → Zq) → Fin m → E} {scalar : (Fin N → Zq) → E}
    (lifts : ∀ r, Affine fun S => lift S r) (scalars : Affine scalar) (i : ℕ) :
    Affine fun S => Akita.liftEntry count slot basis (lift S) (scalar S) i := by
  unfold Akita.liftEntry
  by_cases small : i < count
  · exact Affine.congr (fun S => dite_eq_left small)
      (Affine.sum _ fun r => Affine.ite _ ((lifts r).mul_left _))
  · by_cases last : i = count
    · exact Affine.congr (fun S => by rw [dite_eq_right small, ite_eq_left last]) scalars
    · exact Affine.of_eq fun S => by rw [dite_eq_right small, dite_eq_right small,
        ite_eq_right last, ite_eq_right last]

/-- The public weight `m_{τ_1}(r_2)` of a family of field rows is affine in `S` when each weight
is affine in `S`. -/
theorem Affine.batchWeight {rows : (Fin N → Zq) → FieldRows E}
    (entry : ∀ i x, Affine fun S => (rows S).weightAt i x) (τ r2 : List E) :
    Affine fun S => mle r2 ((rows S).batchWeight τ) := by
  simp only [FieldRows.batchWeight, mle_eq_cubeSum, cubeSum_eq_sum_range]
  exact Affine.sum _ fun x => (Affine.sum _ fun i => (entry i x).mul_left _).mul_left _

namespace Batch

variable {n : ℕ} {B : Batch (Fin n)}

/-- Each ring row of `Batch.flatRows` for the statement `st.over S` is affine in `S`: the rows
(104) of `A`, (105) of `B` and (106) of `D` have the multipliers of the views of `S`, and the
other rows do not depend on `S`. -/
theorem flatRows_affine (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
    (challenge : B.Challenge) :
    ∀ r, AffineRow E fun S : Fin N → Zq => B.flatRows β (st.over S) challenge r := by
  have inner : ∀ g (S : Fin N → Zq) a c, ((st.over S).key g).inner a c =
      ∑ p, S p • ((st.over (Pi.single p 1)).key g).inner a c :=
    fun _ S a c => Setup.shared_sum S _ _ _ _ a c
  have outer : ∀ g (S : Fin N → Zq) i c, ((st.over S).key g).outer i c =
      ∑ p, S p • ((st.over (Pi.single p 1)).key g).outer i c :=
    fun _ S i c => Setup.shared_sum S _ _ _ _ i c
  have opened : ∀ (S : Fin N → Zq) i c, (st.over S).openKey.matrix i c =
      ∑ p, S p • (st.over (Pi.single p 1)).openKey.matrix i c :=
    fun S i c => Setup.shared_sum S _ _ _ _ i c
  unfold flatRows groupRows innerRow outerForm openForm
  refine AffineRow.append (AffineRow.gather fun g => ?_) ?_
  · refine AffineRow.append (fun t => AffineRow.of_eq fun _ => rfl) (AffineRow.append ?_ ?_)
    · intro a
      exact AffineRow.relocate (AffineRow.row
        (AffineForm.const_sub _ (AffineForm.response (B.fold g) (inner g) a)) fun _ => 0) _
    · exact fun j => (AffineRow.imageRow _ (fun j => AffineForm.ofMatrix (outer g) _ _) _ _
        j).relocate _
  · refine AffineRow.append (AffineRow.imageRow _ (fun j => AffineForm.ofMatrix opened _ _) _ _)
      (AffineRow.append (AffineRow.gather fun _ _ => AffineRow.of_eq fun _ => rfl)
        (AffineRow.append (fun _ => AffineRow.of_eq fun _ => rfl)
          (AffineRow.append (AffineRow.gather fun _ _ => AffineRow.of_eq fun _ => rfl)
            fun _ => AffineRow.of_eq fun _ => rfl)))

/-- The weights of the quotient digits of a ring row of `Batch.flatRows` for the statement
`st.over S` do not depend on `S`. -/
theorem quotientWeight_over (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
    (challenge : B.Challenge) (μ : ℕ) (fits : B.cells ≤ 2 ^ μ) (base digits : ℕ) (α : E)
    (S : Fin N → Zq) (r : Fin B.planeRows) (x : ℕ) :
    QuotientRead.weight ⟨digits, fun a => (base : Zq) ^ (a : ℕ),
        (B.flatRelation β (st.over S) challenge μ fits).quotientCell (B.gaps digits) digits r⟩
        (B.flatRows β (st.over S) challenge r).degree α x =
      QuotientRead.weight ⟨digits, fun a => (base : Zq) ^ (a : ℕ),
        (B.flatRelation β (st.over (0 : Fin N → Zq)) challenge μ fits).quotientCell (B.gaps digits) digits r⟩
        (B.flatRows β (st.over (0 : Fin N → Zq)) challenge r).degree α x := by
  have degrees : ∀ r : Fin B.planeRows, (B.flatRows β (st.over S) challenge r).degree =
      (B.flatRows β (st.over 0) challenge r).degree :=
    fun r => (flatRows_affine (N := N) β st challenge r).degree S
  have cells : ∀ (j : ℕ) (a : Fin digits),
      (B.flatRelation β (st.over S) challenge μ fits).quotientCell (B.gaps digits) digits r j a =
        (B.flatRelation β (st.over 0) challenge μ fits).quotientCell (B.gaps digits) digits r j
          a :=
    fun j a => by
      show (B.gaps digits).slot (offset (fun r : Fin B.planeRows =>
          digits * (B.flatRows β (st.over S) challenge r).degree) r +
          (a * (B.flatRows β (st.over S) challenge r).degree + j)) =
        (B.gaps digits).slot (offset (fun r : Fin B.planeRows =>
          digits * (B.flatRows β (st.over 0) challenge r).degree) r +
          (a * (B.flatRows β (st.over 0) challenge r).degree + j))
      simp only [degrees]
      rfl
  rw [degrees r, funext fun j => funext fun a => cells j a]

variable [Fintype E]

/-- The public weight `m_{τ_1}(r_2)` of `Batch.liftedReduction` for the statement `st.over S` is
affine in `S` (Akita (132), (133)). -/
theorem weight_affine (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
    (challenge : B.Challenge) (μ : ℕ) (fits : B.cells ≤ 2 ^ μ) (base digits : ℕ)
    (room : B.quotientRoom digits ≤ 2 ^ μ) (batch : ℕ) (rowsFit : B.rowCount + 1 ≤ 2 ^ batch)
    (α : E) (τ r2 : List E) :
    Affine fun S : Fin N → Zq => mle r2 (((liftedReduction β (st.over S) challenge μ fits base
      digits room batch rowsFit).rows α).batchWeight τ) := by
  have rows := flatRows_affine (N := N) β st challenge
  refine Affine.batchWeight (fun i x => ?_) τ r2
  refine Affine.congr (fun S => LiftRelation.liftRows_weightAt _ α i x) ?_
  refine Affine.liftEntry (m := B.planeRows)
    ((Join.gather fun g => (Join.all (β g)).append (Join.plain E (B.groupTail g))).append
      (Join.plain E B.restCount)).slot
    ((Join.gather fun g => (Join.all (β g)).append (Join.plain E (B.groupTail g))).append
      (Join.plain E B.restCount)).basis (fun r => ?_) ?_ i
  · exact AffineRow.liftWeight ((rows r).relocate (B.gaps digits).place) α x fun S =>
      quotientWeight_over β st challenge μ fits base digits α S r x
  · exact Affine.of_eq fun S => by
      simp only [FlatRelation.withQuotients, Batch.flatRelation, Batch.scalarWeight,
        Batch.Statement.over]

end Batch

end NightstreamFPrime.Spec.Compression.Akita
