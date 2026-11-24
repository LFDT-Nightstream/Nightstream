import NightstreamFPrime.Spec.Compression.Akita.Replay.Rows
import NightstreamFPrime.Spec.Compression.Akita.Extract.SetupFactor
import NightstreamFPrime.Spec.Compression.Akita.Instance

/-! The weight of Stage 3 of the specification (`Batch.codeScan` of `Extract/SetupFactor.lean`;
ePrint 2026/1983, Section 9 with (133) and (134)) at the point `ρ` of Stage 3, as a sum over the
entries of the views of `A`, `B` and `D` (Akita (35)).

- `setupAt ρ α d e` is `Σ_{l<d} ẽq(ρ, e·d + l)·α^l`, the weight at `ρ` of the entry `e` of a view
  of degree `d`: `sum_shared`.
- `runAt r α κ d s` is `Σ_{m<d/2^κ} ẽq(r, s/2^κ + m)·(α^{2^κ})^m`, the weight at `r` of the first
  cells of the blocks of `2^κ` cells of a run of `d` cells from the cell `s`: `mle_run`.
- `deltas` has the part that depends on the setup of the weight of a cell in each ring row.
  `codeWeight_sub` states that the weight of `codeScan` at `e_p` minus the weight at 0 is the
  batch of these parts over the field rows, and `rows_mle` gives this batch as a sum over the rows
  of `A` and `B` of each group and the rows of `D`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib Polynomial

section Forms

variable {E : Type} [CommRing E]

/-- `Σ_{l<d} ẽq(ρ, e·d + l)·α^l`, the weight at `ρ` of the entry `e` of a view of degree `d`
(Akita (35)). -/
def setupAt (ρ : List E) (α : E) (d e : ℕ) : E :=
  ∑ l ∈ Finset.range d, eqAt ρ (e * d + l) * α ^ l

/-- `Σ_{m<d/2^κ} ẽq(r, s/2^κ + m)·(α^{2^κ})^m`, the weight at `r` of the first cells of the
blocks of `2^κ` cells of a run of `d` cells from the cell `s` (`materialize_role_tensor_weights`
of the Akita code). -/
def runAt (r : List E) (α : E) (κ d s : ℕ) : E :=
  ∑ m ∈ Finset.range (d / 2 ^ κ), eqAt r (s / 2 ^ κ + m) * (α ^ 2 ^ κ) ^ m

theorem mle_spread (τ : List E) (s m : ℕ) (f : ℕ → E) (fits : s + m ≤ 2 ^ τ.length) :
    mle τ (spread s m f) = ∑ r ∈ Finset.range m, eqAt τ (s + r) * f r := by
  unfold spread
  rw [mle_finset_sum]
  exact Finset.sum_congr rfl fun r member =>
    mle_delta_at τ (by have := Finset.mem_range.mp member; omega) (f r)

private theorem run_block (α : E) {κ : ℕ} (a b y : ℕ) :
    ∑ t ∈ Finset.range (2 ^ κ), (if 2 ^ κ * a + (2 ^ κ * b + t) = 2 ^ κ * y then
      α ^ (2 ^ κ * b + t) else 0) = if a + b = y then (α ^ 2 ^ κ) ^ b else 0 := by
  have positive : 0 < 2 ^ κ := by positivity
  rw [Finset.sum_eq_single 0]
  · by_cases same : a + b = y
    · rw [ite_eq_left (by rw [← same]; ring), ite_eq_left same, Nat.add_zero, pow_mul]
    · rw [ite_eq_right fun equal => same (Nat.eq_of_mul_eq_mul_left positive (by
        rw [Nat.mul_add]; omega)), ite_eq_right same]
  · intro t member other
    refine ite_eq_right fun equal => other ?_
    have low := congrArg (· % 2 ^ κ) equal
    simp only [show 2 ^ κ * a + (2 ^ κ * b + t) = t + 2 ^ κ * (a + b) by ring,
      Nat.add_mul_mod_self_left, Nat.mul_mod_right,
      Nat.mod_eq_of_lt (Finset.mem_range.mp member)] at low
    exact low
  · exact fun outside => absurd (Finset.mem_range.mpr positive) outside

/-- The weight at `r` of the first cells of the blocks of `2^κ` cells of a run of `d` cells from
the cell `s`, for a run that starts at a multiple of `2^κ` and ends in the cube. -/
theorem mle_run (r : List E) (α : E) {κ d s : ℕ} (start : 2 ^ κ ∣ s) (degree : 2 ^ κ ∣ d)
    (fit : s + d ≤ 2 ^ (r.length + κ)) :
    mle r (fun y => ∑ j ∈ Finset.range d, if s + j = 2 ^ κ * y then α ^ j else 0) =
      runAt r α κ d s := by
  obtain ⟨a, rfl⟩ := start
  obtain ⟨b, rfl⟩ := degree
  have positive : 0 < 2 ^ κ := by positivity
  unfold runAt
  rw [Nat.mul_div_cancel_left _ positive, Nat.mul_div_cancel_left _ positive]
  have small : a + b ≤ 2 ^ r.length := by
    rw [pow_add, ← Nat.mul_add, mul_comm (2 ^ r.length)] at fit
    exact Nat.le_of_mul_le_mul_left fit positive
  clear fit
  induction b with
  | zero => simpa using mle_const r (0 : E)
  | succ b ih =>
      rw [Finset.sum_range_succ, ← ih (by omega),
        ← mle_delta_at r (show a + b < 2 ^ r.length by omega), ← mle_add]
      refine mle_congr r fun y _ => ?_
      rw [Nat.mul_succ, Finset.sum_range_add, run_block]

variable [Algebra Zq E]

/-- The weight at `ρ` of the entry `(i, c)` of a view of the flat setup vector is `setupAt` of its
index `i·w + code c`, for a view of at most `2^|ρ|` cells. -/
theorem sum_shared (ρ : List E) (α : E) {d m w : ℕ} [NeZero d] {cols : Type}
    (code : cols → Fin w) (i : Fin m) (c : cols) (fit : m * w * d ≤ 2 ^ ρ.length) :
    ∑ p : Fin (2 ^ ρ.length), eqAt ρ p *
        aeval α (rep (Setup.shared (Pi.single p 1) d m w code i c)) =
      setupAt ρ α d (i * w + code c) := by
  have inside : ∀ l < d, (i * w + code c) * d + l < 2 ^ ρ.length := fun l small =>
    (Setup.Instance.index_lt (I := ⟨d, m, w, false, 0⟩) i (code c) small).trans_le fit
  have entry : ∀ (p : Fin (2 ^ ρ.length)) {l : ℕ} (small : l < d),
      coeff (Setup.shared (Pi.single p 1) d m w code i c) l =
        if (⟨_, inside l small⟩ : Fin (2 ^ ρ.length)) = p then 1 else 0 := fun p l small => by
    simp only [Setup.shared, Matrix.submatrix_apply, id, Setup.view, Matrix.of_apply,
      coeff_ofCoeffs _ small, dite_eq_left (inside l small), Pi.single_apply]
  simp only [aeval_rep, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun l member => ?_
  have small := Finset.mem_range.mp member
  simp only [entry _ small, ite_smul, one_smul, zero_smul, mul_ite, mul_zero]
  rw [Finset.sum_ite_eq, ite_eq_left (Finset.mem_univ _)]

/-- The part that depends on the setup of the weight of the cell `x` in each ring row: the linear
form `RingRow.scan` of the row of `rows` minus that of the row of `rows'`, at the move `ψ`. -/
noncomputable def deltas (α : E) (ψ : Option ℕ → Option ℕ) (x : ℕ) {m : ℕ}
    (rows rows' : Fin m → RingRow) : Fin m → E :=
  fun r => (rows r).scan α (mark ψ x) - (rows' r).scan α (mark ψ x)

theorem deltas_append (α : E) (ψ : Option ℕ → Option ℕ) (x : ℕ) {m k : ℕ}
    (a a' : Fin m → RingRow) (b b' : Fin k → RingRow) :
    deltas α ψ x (Fin.append a b) (Fin.append a' b') =
      Fin.append (deltas α ψ x a a') (deltas α ψ x b b') := by
  funext r
  refine Fin.addCases (fun i => ?_) (fun j => ?_) r
  · simp only [deltas, Fin.append_left]
  · simp only [deltas, Fin.append_right]

theorem imageRow_deltas {d m : ℕ} [NeZero d] (chain : Option ChainShape)
    (image image' : Fin m → Akita.Form d) (first : ℕ → Option ℕ)
    (payload : Akita.Payload d m chain) (α : E)
    (φ : Option ℕ → E) (j : Fin m) :
    (imageRow chain image first payload j).scan α φ -
        (imageRow chain image' first payload j).scan α φ =
      (image j).scan α φ - (image' j).scan α φ := by
  cases chain with
  | none => simp only [imageRow, Form.scan_row]
  | some c =>
      simp only [imageRow, Form.scan_row, Form.scan_sub]
      ring

end Forms

section Lift

variable {E : Type} [Field E] [Algebra Zq E]

omit [Algebra Zq E] in
theorem liftEntry_sub {m count : ℕ} (slot : Fin m → Fin count) (basis : Fin m → E)
    (lift lift' : Fin m → E) {scalar scalar' : E} (same : scalar = scalar') (i : ℕ) :
    liftEntry count slot basis lift scalar i - liftEntry count slot basis lift' scalar' i =
      tableOf (joinTargets ⟨count, slot, basis⟩ fun r => lift r - lift' r) i := by
  subst same
  unfold liftEntry tableOf joinTargets
  by_cases small : i < count
  · rw [dite_eq_left small, dite_eq_left small, dite_eq_left small, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun r _ => ?_
    split
    · ring
    · ring
  · rw [dite_eq_right small, dite_eq_right small, dite_eq_right small, sub_self]

theorem liftWeight_sub {F F' : RingRow} {q q' : QuotientRead} (α : E) (x : ℕ)
    (degree : F.degree = F'.degree) (same : q.weight F.degree α x = q'.weight F'.degree α x)
    (move : ℕ → ℕ) :
    (F.relocate move).liftWeight q α x - (F'.relocate move).liftWeight q' α x =
      F.scan α (mark (fun o => o.map move) x) - F'.scan α (mark (fun o => o.map move) x) := by
  unfold RingRow.liftWeight
  rw [show (F.relocate move).degree = F.degree from rfl,
    show (F'.relocate move).degree = F'.degree from rfl, same, degree,
    RingRow.readWeight_eq_mark, RingRow.readWeight_eq_mark, sub_sub_sub_cancel_right]
  rfl

end Lift

section Batch

variable {n : ℕ} (B : Batch (Fin n)) {E : Type} [Field E] [Algebra Zq E]

/-- The rows (104) of `A` of the group `g`, with the addresses of the fold. -/
noncomputable def innerRows (st : B.Statement E) (challenge : B.Challenge) (g : Fin n) :
    Fin (B.shape g).innerRank → RingRow := fun a =>
  (innerRow (B.opening g) (B.fold g) (st.key g) (challenge g) a).relocate (B.relocate g)

/-- The rows (105) of `B` of each slice of the group `g`, with the addresses of the fold. -/
noncomputable def outerRows (st : B.Statement E) (g : Fin n) :
    Fin ((B.shape g).slices * (B.shape g).outerRank) → RingRow := fun j =>
  (imageRow (B.shape g).chain (outerForm (B.opening g) (B.fold g) (st.key g))
    ((B.opening g).firstCell (B.fold g)) (st.payload g) j).relocate (B.relocate g)

/-- The rows (106) of `D`. -/
noncomputable def openRows (st : B.Statement E) : Fin (1 * B.openRank) → RingRow :=
  imageRow B.chain (B.openForm st.openKey) B.openFirstCell st.openPayload

variable {B}

/-- The field row of the group `y.1` with the index `y.2` of the parts `deltas` of the rows of the
fold for the statements `st.over S` and `st.over 0`. -/
noncomputable def groupDeltas (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (st : B.Statement E) (challenge : B.Challenge) (α : E) (ψ : Option ℕ → Option ℕ) (x : ℕ)
    {N : ℕ} (S : Fin N → Zq) (y : Σ g : Fin n, Fin (B.groupCount g)) : E :=
  joinTargets ((Join.all (β y.1)).append (Join.plain E (B.groupTail y.1)))
    (deltas α ψ x (B.groupRows β (st.over S) challenge y.1)
      (B.groupRows β (st.over (0 : Fin N → Zq)) challenge y.1)) y.2

theorem group_deltas (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
    (challenge : B.Challenge) (α : E) (ψ : Option ℕ → Option ℕ) (x : ℕ) {N : ℕ}
    (S : Fin N → Zq) (g : Fin n) (s i : ℕ) :
    spread s (B.groupCount g) (tableOf fun j => groupDeltas β st challenge α ψ x S ⟨g, j⟩) i =
      spread (s + 1) (B.shape g).innerRank (tableOf (deltas α ψ x
          (innerRows B (st.over S) challenge g) (innerRows B (st.over (0 : Fin N → Zq)) challenge g))) i +
        spread (s + 1 + (B.shape g).innerRank) ((B.shape g).slices * (B.shape g).outerRank)
          (tableOf (deltas α ψ x (outerRows B (st.over S) g) (outerRows B (st.over (0 : Fin N → Zq)) g))) i := by
  have split : (fun j => groupDeltas β st challenge α ψ x S ⟨g, j⟩) =
      Fin.append (fun _ : Fin 1 => (0 : E)) (Fin.append (deltas α ψ x
          (innerRows B (st.over S) challenge g) (innerRows B (st.over (0 : Fin N → Zq)) challenge g))
        (deltas α ψ x (outerRows B (st.over S) g) (outerRows B (st.over (0 : Fin N → Zq)) g))) := by
    show joinTargets ((Join.all (β g)).append (Join.plain E
        ((B.shape g).innerRank + (B.shape g).slices * (B.shape g).outerRank)))
      (deltas α ψ x (Fin.append (fun t =>
          (evaluationRow (B.opening g) (B.fold g) (β g) ((st.over S).point g) (challenge g)
            t).relocate (B.relocate g))
        (Fin.append (innerRows B (st.over S) challenge g) (outerRows B (st.over S) g)))
        (Fin.append (fun t =>
          (evaluationRow (B.opening g) (B.fold g) (β g) ((st.over (0 : Fin N → Zq)).point g) (challenge g)
            t).relocate (B.relocate g))
        (Fin.append (innerRows B (st.over (0 : Fin N → Zq)) challenge g) (outerRows B (st.over (0 : Fin N → Zq)) g)))) = _
    rw [deltas_append, deltas_append, append_targets, all_targets, plain_targets]
    congr 1
    funext _
    exact Finset.sum_eq_zero fun t _ => by
      rw [show deltas α ψ x (fun t =>
          (evaluationRow (B.opening g) (B.fold g) (β g) ((st.over S).point g) (challenge g)
            t).relocate (B.relocate g)) (fun t =>
          (evaluationRow (B.opening g) (B.fold g) (β g) ((st.over (0 : Fin N → Zq)).point g) (challenge g)
            t).relocate (B.relocate g)) t = 0 from sub_self _, mul_zero]
  rw [split]
  show spread s (1 + ((B.shape g).innerRank + (B.shape g).slices * (B.shape g).outerRank))
    (@tableOf E _ (1 + ((B.shape g).innerRank + (B.shape g).slices * (B.shape g).outerRank)) _)
      i = _
  rw [spread_append, spread_append,
    spread_eq_zero (fun r small => by rw [tableOf, dite_eq_left small]) i, zero_add]

/-- The field rows of the parts `deltas` of the rows of the fold: the rows of `A` and of `B` of
each group in the order `groupOrder`, and the rows of `D`. -/
theorem rows_deltas (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
    (challenge : B.Challenge) (α : E) (ψ : Option ℕ → Option ℕ) (x : ℕ) {N : ℕ}
    (S : Fin N → Zq) (i : ℕ) :
    tableOf (joinTargets ((Join.gather fun g => (Join.all (β g)).append
        (Join.plain E (B.groupTail g))).append (Join.plain E B.restCount))
      (deltas α ψ x (B.flatRows β (st.over S) challenge) (B.flatRows β (st.over (0 : Fin N → Zq)) challenge))) i =
      ∑ k, (spread (rowsBefore (fun j => B.groupCount (groupOrder n j)) k + 1)
            (B.shape (groupOrder n k)).innerRank (tableOf (deltas α ψ x
              (innerRows B (st.over S) challenge (groupOrder n k))
              (innerRows B (st.over (0 : Fin N → Zq)) challenge (groupOrder n k)))) i +
          spread (rowsBefore (fun j => B.groupCount (groupOrder n j)) k + 1 +
              (B.shape (groupOrder n k)).innerRank)
            ((B.shape (groupOrder n k)).slices * (B.shape (groupOrder n k)).outerRank)
            (tableOf (deltas α ψ x (outerRows B (st.over S) (groupOrder n k))
              (outerRows B (st.over (0 : Fin N → Zq)) (groupOrder n k)))) i) +
        spread (∑ g, B.groupCount g) (1 * B.openRank)
          (tableOf (deltas α ψ x (openRows B (st.over S)) (openRows B (st.over (0 : Fin N → Zq))))) i := by
  have targets : joinTargets ((Join.gather fun g => (Join.all (β g)).append
        (Join.plain E (B.groupTail g))).append (Join.plain E B.restCount))
      (deltas α ψ x (B.flatRows β (st.over S) challenge) (B.flatRows β (st.over (0 : Fin N → Zq)) challenge)) =
      Fin.append (fun k => groupDeltas β st challenge α ψ x S (place B.groupCount k))
        (Fin.append (deltas α ψ x (openRows B (st.over S)) (openRows B (st.over (0 : Fin N → Zq))))
          (fun _ => 0)) := by
    show joinTargets ((Join.gather fun g => (Join.all (β g)).append
        (Join.plain E (B.groupTail g))).append (Join.plain E (1 * B.openRank +
          (∑ g, firstRank (B.shape g).chain + (firstRank B.chain +
            (∑ g, secondRank (B.shape g).chain + secondRank B.chain))))))
      (deltas α ψ x
        (Fin.append (Akita.gather (count := fun g => B.planes g + B.groupTail g)
            (B.groupRows β (st.over S) challenge))
          (Fin.append (openRows B (st.over S))
            (Fin.append (Akita.gather (batchFirst B (st.over S)))
              (Fin.append (firstRow B.chain (st.over S).openKey.chain B.openFirstCell
                  B.openSecondCell)
                (Fin.append (Akita.gather (batchSecond B (st.over S)))
                  (secondRow B.chain (st.over S).openKey.chain B.openSecondCell
                    (st.over S).openPayload))))))
        (Fin.append (Akita.gather (count := fun g => B.planes g + B.groupTail g)
            (B.groupRows β (st.over (0 : Fin N → Zq)) challenge))
          (Fin.append (openRows B (st.over (0 : Fin N → Zq)))
            (Fin.append (Akita.gather (batchFirst B (st.over (0 : Fin N → Zq))))
              (Fin.append (firstRow B.chain (st.over (0 : Fin N → Zq)).openKey.chain
                  B.openFirstCell B.openSecondCell)
                (Fin.append (Akita.gather (batchSecond B (st.over (0 : Fin N → Zq))))
                  (secondRow B.chain (st.over (0 : Fin N → Zq)).openKey.chain B.openSecondCell
                    (st.over (0 : Fin N → Zq)).openPayload))))))) = _
    rw [deltas_append, append_targets, plain_targets, deltas_append]
    congr 1
    · funext k
      exact gather_targets (fun g => (Join.all (β g)).append (Join.plain E (B.groupTail g)))
        (fun y => deltas α ψ x (B.groupRows β (st.over S) challenge y.1)
          (B.groupRows β (st.over (0 : Fin N → Zq)) challenge y.1) y.2) k
    · congr 1
      funext r
      exact sub_self _
  rw [targets]
  show @tableOf E _ (∑ g, B.groupCount g + (1 * B.openRank + (∑ g, firstRank (B.shape g).chain +
      (firstRank B.chain + (∑ g, secondRank (B.shape g).chain + secondRank B.chain)))))
    (Fin.append (fun k => groupDeltas β st challenge α ψ x S (place B.groupCount k)) _) i = _
  have rest : ∀ s m, spread s m (tableOf fun _ : Fin m => (0 : E)) i = 0 := fun s m =>
    spread_eq_zero (fun r small => by rw [tableOf, dite_eq_left small]) i
  rw [tableOf_eq_spread, spread_append, spread_gather B.groupCount 0, spread_append, rest,
    add_zero]
  simp only [zero_add, group_deltas]

omit [Algebra Zq E] in
theorem mle_spread_fin (τ : List E) (s : ℕ) {m : ℕ} (v : Fin m → E)
    (fits : s + m ≤ 2 ^ τ.length) :
    mle τ (spread s m (tableOf v)) = ∑ r : Fin m, eqAt τ (s + r) * v r := by
  rw [mle_spread τ s m _ fits, ← Fin.sum_univ_eq_sum_range]
  exact Finset.sum_congr rfl fun r _ => by rw [tableOf, dite_eq_left r.2]

theorem codeWeight_sub [Fintype E] (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (st : B.Statement E) (challenge : B.Challenge) (μ : ℕ) (fits : B.cells ≤ 2 ^ μ)
    (base digits : ℕ) (room : B.quotientRoom digits ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) (α : E) (τ : List E) {N : ℕ} (S : Fin N → Zq)
    (x : ℕ) :
    ((Batch.liftedReduction β (st.over S) challenge μ fits base digits room batch rowsFit).rows
        α).batchWeight τ x -
      ((Batch.liftedReduction β (st.over (0 : Fin N → Zq)) challenge μ fits base digits room batch
        rowsFit).rows α).batchWeight τ x =
      mle τ (tableOf (joinTargets ((Join.gather fun g => (Join.all (β g)).append
          (Join.plain E (B.groupTail g))).append (Join.plain E B.restCount))
        (deltas α (fun o => o.map (B.gaps digits).place) x (B.flatRows β (st.over S) challenge)
          (B.flatRows β (st.over (0 : Fin N → Zq)) challenge)))) := by
  unfold FieldRows.batchWeight
  rw [← mle_sub]
  refine congrArg (mle τ) (funext fun i => ?_)
  refine (congrArg₂ (· - ·) (LiftRelation.liftRows_weightAt _ α i x)
    (LiftRelation.liftRows_weightAt _ α i x)).trans ?_
  refine (liftEntry_sub (m := B.planeRows)
    ((Join.gather fun g => (Join.all (β g)).append (Join.plain E (B.groupTail g))).append
      (Join.plain E B.restCount)).slot
    ((Join.gather fun g => (Join.all (β g)).append (Join.plain E (B.groupTail g))).append
      (Join.plain E B.restCount)).basis _ _ ?_ i).trans ?_
  · simp only [FlatRelation.withQuotients, Batch.flatRelation, Batch.scalarWeight,
      Batch.Statement.over]
  · exact congrArg (fun t => tableOf (joinTargets _ t) i) (funext fun r =>
      liftWeight_sub α x ((Batch.flatRows_affine (N := N) β st challenge r).degree S)
        (Batch.quotientWeight_over β st challenge μ fits base digits α S r x) _)

/-- The batch over the field rows of the parts `deltas`: a sum over the rows of `A` and `B` of
each group in the order `groupOrder` and over the rows of `D`. -/
theorem rows_mle (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
    (challenge : B.Challenge) (α : E) (ψ : Option ℕ → Option ℕ) (x : ℕ) {N : ℕ}
    (S : Fin N → Zq) (τ : List E) (rowsFit : B.rowCount + 1 ≤ 2 ^ τ.length) :
    mle τ (tableOf (joinTargets ((Join.gather fun g => (Join.all (β g)).append
        (Join.plain E (B.groupTail g))).append (Join.plain E B.restCount))
      (deltas α ψ x (B.flatRows β (st.over S) challenge)
        (B.flatRows β (st.over (0 : Fin N → Zq)) challenge)))) =
      ∑ k, (∑ a : Fin (B.shape (groupOrder n k)).innerRank,
            eqAt τ (rowsBefore (fun j => B.groupCount (groupOrder n j)) k + 1 + a) *
              deltas α ψ x (innerRows B (st.over S) challenge (groupOrder n k))
                (innerRows B (st.over (0 : Fin N → Zq)) challenge (groupOrder n k)) a +
          ∑ t : Fin ((B.shape (groupOrder n k)).slices * (B.shape (groupOrder n k)).outerRank),
            eqAt τ (rowsBefore (fun j => B.groupCount (groupOrder n j)) k + 1 +
                (B.shape (groupOrder n k)).innerRank + t) *
              deltas α ψ x (outerRows B (st.over S) (groupOrder n k))
                (outerRows B (st.over (0 : Fin N → Zq)) (groupOrder n k)) t) +
        ∑ j : Fin (1 * B.openRank), eqAt τ (∑ g, B.groupCount g + j) *
          deltas α ψ x (openRows B (st.over S)) (openRows B (st.over (0 : Fin N → Zq))) j := by
  have total : ∑ k, B.groupCount (groupOrder n k) = ∑ g, B.groupCount g :=
    Equiv.sum_comp (groupOrder n) B.groupCount
  have rows : B.rowCount = ∑ g, B.groupCount g + (1 * B.openRank +
      (∑ g, firstRank (B.shape g).chain + (firstRank B.chain +
        (∑ g, secondRank (B.shape g).chain + secondRank B.chain)))) := rfl
  rw [mle_congr τ (fun i _ => rows_deltas β st challenge α ψ x S i), mle_add, mle_finset_sum]
  refine congrArg₂ (· + ·) (Finset.sum_congr rfl fun k _ => ?_) (mle_spread_fin τ _ _ (by omega))
  have bound := rowsBefore_add_le (fun j => B.groupCount (groupOrder n j)) k
  have count : B.groupCount (groupOrder n k) = 1 + ((B.shape (groupOrder n k)).innerRank +
      (B.shape (groupOrder n k)).slices * (B.shape (groupOrder n k)).outerRank) := rfl
  rw [mle_add, mle_spread_fin τ _ _ (by omega), mle_spread_fin τ _ _ (by omega)]

end Batch

section Weight

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E] [Algebra Zq E]

theorem shared_zero {N d m w : ℕ} [NeZero d] {cols : Type} (code : cols → Fin w) (i : Fin m)
    (c : cols) : Setup.shared (0 : Fin N → Zq) d m w code i c = 0 := by
  rw [Setup.shared_sum]
  simp

theorem aeval_rep_zero {d : ℕ} [NeZero d] (α : E) : aeval α (rep (0 : Rq d)) = 0 := by
  rw [← reducedEval_apply, map_zero]

theorem aeval_rep_mul_cast {d : ℕ} [NeZero d] (α : E) (X : Rq d) (b f : ℕ) :
    aeval α (rep (X * (b : Rq d) ^ f)) = (b : E) ^ f * aeval α (rep X) := by
  rw [← reducedEval_apply, ← reducedEval_apply, ← Nat.cast_pow,
    ← map_natCast (algebraMap Zq (Rq d)), mul_comm, ← Algebra.smul_def, map_smul,
    Algebra.smul_def, map_natCast, Nat.cast_pow]

theorem scan_relocate (row : RingRow) (move : ℕ → ℕ) (α : E) (φ : Option ℕ → E) :
    (row.relocate move).scan α φ = row.scan α (fun o => φ (o.map move)) :=
  rfl

theorem run_le {m d : ℕ} (j : Fin m) : (j : ℕ) * d + d ≤ m * d := by
  have := Nat.mul_le_mul_right d (Nat.succ_le_of_lt j.isLt)
  rwa [Nat.succ_mul] at this

/-- `Σ_p ẽq(ρ, p)·Σ_y ẽq(r, y)·h(p, 2^κ·y)`: the weight at `ρ` of a table of the setup indices
`p`, each the extension at `r` of the first cells of the blocks of `2^κ` cells of a table of the
cells. -/
noncomputable def setupPair (ρ r : List E) (κ : ℕ) (h : Fin (2 ^ ρ.length) → ℕ → E) : E :=
  ∑ p : Fin (2 ^ ρ.length), eqAt ρ p * mle r (fun y => h p (2 ^ κ * y))

omit [Algebra Zq E] in
theorem setupPair_congr (ρ r : List E) (κ : ℕ) {h h' : Fin (2 ^ ρ.length) → ℕ → E}
    (same : ∀ p x, h p x = h' p x) : setupPair ρ r κ h = setupPair ρ r κ h' := by
  unfold setupPair
  simp only [same]

omit [Algebra Zq E] in
theorem setupPair_add (ρ r : List E) (κ : ℕ) (h h' : Fin (2 ^ ρ.length) → ℕ → E) :
    setupPair ρ r κ (fun p x => h p x + h' p x) = setupPair ρ r κ h + setupPair ρ r κ h' := by
  unfold setupPair
  simp only [mle_add, mul_add, Finset.sum_add_distrib]

omit [Algebra Zq E] in
theorem setupPair_sum {ι : Type} (s : Finset ι) (ρ r : List E) (κ : ℕ)
    (h : ι → Fin (2 ^ ρ.length) → ℕ → E) :
    setupPair ρ r κ (fun p x => ∑ i ∈ s, h i p x) = ∑ i ∈ s, setupPair ρ r κ (h i) := by
  unfold setupPair
  simp only [mle_finset_sum, Finset.mul_sum]
  exact Finset.sum_comm

omit [Algebra Zq E] in
theorem setupPair_mul_left (ρ r : List E) (κ : ℕ) (c : E) (h : Fin (2 ^ ρ.length) → ℕ → E) :
    setupPair ρ r κ (fun p x => c * h p x) = c * setupPair ρ r κ h := by
  unfold setupPair
  simp only [mle_mul_left, Finset.mul_sum]
  exact Finset.sum_congr rfl fun _ _ => by ring

omit [Algebra Zq E] in
theorem pair_sum {ι : Type} [Fintype ι] (ρ r : List E) (κ : ℕ) (c : ι → E)
    (M : ι → Fin (2 ^ ρ.length) → E) (R : ι → ℕ → E) :
    ∑ p : Fin (2 ^ ρ.length), eqAt ρ p * mle r (fun y => ∑ i, c i * M i p * R i (2 ^ κ * y)) =
      ∑ i, c i * (∑ p : Fin (2 ^ ρ.length), eqAt ρ p * M i p) *
        mle r (fun y => R i (2 ^ κ * y)) := by
  simp only [mle_finset_sum, mle_mul_left, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Finset.sum_mul]
  exact Finset.sum_congr rfl fun p _ => by ring

theorem inner_deltas (st : B.Statement E) (challenge : B.Challenge) (digits : ℕ) (α : E)
    (x : ℕ) {N : ℕ} (p : Fin N) (g : Fin n) (a : Fin (B.shape g).innerRank) :
    deltas α (fun o => o.map (B.gaps digits).place) x
        (innerRows B (st.over (Pi.single p 1)) challenge g)
        (innerRows B (st.over (0 : Fin N → Zq)) challenge g) a =
      ∑ col : (Fin (B.shape g).blockLength × Fin (B.shape g).sourceDigits) ×
          Fin (B.fold g).responseDigits,
        -((B.fold g).responseBase : E) ^ (col.2 : ℕ) *
          aeval α (rep (((st.over (Pi.single p 1)).key g).inner a col.1)) *
          ∑ j ∈ Finset.range (B.shape g).innerDegree,
            if B.mainStart g + (B.shape g).responseCode (B.fold g) col * (B.shape g).innerDegree +
              j = x then α ^ j else 0 := by
  have form : ∀ S : Fin N → Zq, (innerRows B (st.over S) challenge g a).scan α
      (mark (fun o => o.map (B.gaps digits).place) x) =
        (imageForm (B.opening g) (B.fold g) (challenge g) a).scan α
          (fun o => mark (fun o => o.map (B.gaps digits).place) x (o.map (B.relocate g))) -
        (responseForm (B.fold g) ((st.over S).key g) a).scan α
          (fun o => mark (fun o => o.map (B.gaps digits).place) x (o.map (B.relocate g))) :=
    fun S => by
      show ((innerRow (B.opening g) (B.fold g) ((st.over S).key g) (challenge g) a).relocate
        (B.relocate g)).scan α _ = _
      rw [scan_relocate]
      show (((imageForm (B.opening g) (B.fold g) (challenge g) a).sub
        (responseForm (B.fold g) ((st.over S).key g) a)).row 0).scan α _ = _
      rw [Form.scan_row, Form.scan_sub]
  have zero : ∀ col : (Fin (B.shape g).blockLength × Fin (B.shape g).sourceDigits) ×
      Fin (B.fold g).responseDigits, ((st.over (0 : Fin N → Zq)).key g).inner a col.1 = 0 :=
    fun col => shared_zero _ _ _
  unfold deltas
  rw [form, form, sub_sub_sub_cancel_left, Form.scan_response, Form.scan_response]
  simp only [zero, zero_mul, aeval_rep_zero, Finset.sum_const_zero]
  rw [zero_sub, ← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl fun col _ => ?_
  rw [aeval_rep_mul_cast, Finset.mul_sum, ← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl fun j member => ?_
  have small := Finset.mem_range.mp member
  have cell : (B.shape g).responseCell (B.fold g) col j =
      some ((B.shape g).responseCode (B.fold g) col * (B.shape g).innerDegree + j) :=
    ite_eq_left small
  rw [cell]
  simp only [Option.map_some, mark, Option.some.injEq,
    Batch.place_main digits g ((cell_lt _ small).trans_le
      (Nat.le_add_right_of_le (Nat.le_add_right _ _))), Nat.add_assoc]
  split_ifs <;> ring

/-- The weight at `ρ` and at `r` of the row `a` of `A` of the group `g`. -/
theorem inner_pair (st : B.Statement E) (challenge : B.Challenge) (digits : ℕ) (α : E)
    (ρ r : List E) {κ : ℕ} (blocked : B.Blocked κ) (cube : B.cells ≤ 2 ^ (r.length + κ))
    (g : Fin n) (a : Fin (B.shape g).innerRank)
    (fit : ((B.shape g).innerInstance false 0).size ≤ 2 ^ ρ.length) :
    setupPair ρ r κ (fun p x => deltas α (fun o => o.map (B.gaps digits).place) x
        (innerRows B (st.over (Pi.single p 1)) challenge g)
        (innerRows B (st.over (0 : Fin (2 ^ ρ.length) → Zq)) challenge g) a) =
      ∑ col : (Fin (B.shape g).blockLength × Fin (B.shape g).sourceDigits) ×
          Fin (B.fold g).responseDigits,
        -((B.fold g).responseBase : E) ^ (col.2 : ℕ) *
          setupAt ρ α (B.shape g).innerDegree
            (a * ((B.shape g).blockLength * (B.shape g).sourceDigits) +
              (B.shape g).innerSetupCode col.1) *
          runAt r α κ (B.shape g).innerDegree
            (B.mainStart g + (B.shape g).responseCode (B.fold g) col * (B.shape g).innerDegree) := by
  unfold setupPair
  calc _ = ∑ p : Fin (2 ^ ρ.length), eqAt ρ p * mle r (fun y =>
          ∑ col : (Fin (B.shape g).blockLength × Fin (B.shape g).sourceDigits) ×
              Fin (B.fold g).responseDigits,
            -((B.fold g).responseBase : E) ^ (col.2 : ℕ) *
              aeval α (rep (Setup.shared (Pi.single p 1) (B.shape g).innerDegree
                (B.shape g).innerRank _ (B.shape g).innerSetupCode a col.1)) *
              ∑ j ∈ Finset.range (B.shape g).innerDegree,
                if B.mainStart g + (B.shape g).responseCode (B.fold g) col *
                  (B.shape g).innerDegree + j = 2 ^ κ * y then α ^ j else 0) :=
        Finset.sum_congr rfl fun p _ => congrArg _ (congrArg _ (funext fun y =>
          inner_deltas st challenge digits α _ p g a))
    _ = _ := pair_sum ρ r κ _ (fun col p => aeval α (rep (Setup.shared (Pi.single p 1)
          (B.shape g).innerDegree (B.shape g).innerRank _ (B.shape g).innerSetupCode a col.1)))
          (fun col x => ∑ j ∈ Finset.range (B.shape g).innerDegree,
            if B.mainStart g + (B.shape g).responseCode (B.fold g) col *
              (B.shape g).innerDegree + j = x then α ^ j else 0)
    _ = _ := Finset.sum_congr rfl fun col _ => by
        have start : 2 ^ κ ∣ B.mainStart g + (B.shape g).responseCode (B.fold g) col *
            (B.shape g).innerDegree :=
          dvd_add (blocked.1 g).1 (Dvd.dvd.mul_left (blocked.1 g).2.2.2.1 _)
        have inside : B.mainStart g + (B.shape g).responseCode (B.fold g) col *
            (B.shape g).innerDegree + (B.shape g).innerDegree ≤ 2 ^ (r.length + κ) := by
          have run := run_le (d := (B.shape g).innerDegree) ((B.shape g).responseCode (B.fold g) col)
          have before := codeOffset_add_le B.mainCells g
          have total := B.chainStart_le_cells
          have main : (B.shape g).responseCells (B.fold g) ≤ B.mainCells g :=
            Nat.le_add_right_of_le (Nat.le_add_right _ _)
          unfold Shape.responseCells at main
          unfold Batch.mainStart
          unfold Batch.chainStart at total
          omega
        rw [sum_shared (d := (B.shape g).innerDegree) (m := (B.shape g).innerRank)
            (w := (B.shape g).blockLength * (B.shape g).sourceDigits) ρ α
            (B.shape g).innerSetupCode a col.1 fit,
          mle_run r α start (blocked.1 g).2.2.2.1 inside]

omit [Algebra Zq E] in
theorem pair_sum' {ι : Type} [Fintype ι] (ρ r : List E) (κ : ℕ)
    (M : ι → Fin (2 ^ ρ.length) → E) (R : ι → ℕ → E) :
    ∑ p : Fin (2 ^ ρ.length), eqAt ρ p * mle r (fun y => ∑ i, M i p * R i (2 ^ κ * y)) =
      ∑ i, (∑ p : Fin (2 ^ ρ.length), eqAt ρ p * M i p) * mle r (fun y => R i (2 ^ κ * y)) := by
  simp only [mle_finset_sum, mle_mul_left, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Finset.sum_mul]
  exact Finset.sum_congr rfl fun p _ => by ring

/-- The position in the code of `t̂` of the element of the input of slice `s` at `(pos, col)`:
the first block of the slice times the width of a block, plus the column of `B`. -/
theorem innerCode_slice (S : Shape) {i : Fin S.blocks} (e : Fin S.sliceLength × S.OuterColumn)
    {start : ℕ} (at_ : (i : ℕ) = start + e.1) :
    (S.innerCode (i, e.2) : ℕ) =
      start * (S.innerRank * (S.ratio * S.openingDigits)) + S.outerSetupCode e := by
  obtain ⟨pos, a, y, u⟩ := e
  simp only [Shape.innerCode, Shape.outerSetupCode, Equiv.trans_apply, Equiv.prodCongr_apply,
    Equiv.prodAssoc_symm_apply, Equiv.refl_apply, Prod.map, finProdFinEquiv_apply_val, at_]
  ring

theorem outer_deltas (st : B.Statement E) (digits : ℕ) (α : E) (x : ℕ) {N : ℕ} (p : Fin N)
    (g : Fin n) (t : Fin ((B.shape g).slices * (B.shape g).outerRank)) :
    deltas α (fun o => o.map (B.gaps digits).place) x (outerRows B (st.over (Pi.single p 1)) g)
        (outerRows B (st.over (0 : Fin N → Zq)) g) t =
      ∑ e : Fin (B.shape g).sliceLength × (B.shape g).OuterColumn,
        aeval α (rep (((st.over (Pi.single p 1)).key g).outer (finProdFinEquiv.symm t).2 e)) *
          if (B.shape g).sliceStart (finProdFinEquiv.symm t).1 + e.1 <
              (B.shape g).sliceStart ((finProdFinEquiv.symm t).1 + 1) then
            ∑ k ∈ Finset.range (B.shape g).outerDegree,
              if B.mainStart g + ((B.opening g).innerStart (B.fold g) +
                  ((B.shape g).sliceStart (finProdFinEquiv.symm t).1 *
                      ((B.shape g).innerRank * ((B.shape g).ratio * (B.shape g).openingDigits)) +
                    (B.shape g).outerSetupCode e) * (B.shape g).outerDegree) + k = x
              then α ^ k else 0
          else 0 := by
  have scanForm : ∀ S : Fin N → Zq, (outerForm (B.opening g) (B.fold g) ((st.over S).key g)
      t).scan α (fun o => mark (fun o => o.map (B.gaps digits).place) x (o.map (B.relocate g))) =
      ∑ e : Fin (B.shape g).sliceLength × (B.shape g).OuterColumn,
        ∑ k ∈ Finset.range (B.shape g).outerDegree,
          aeval α (rep (((st.over S).key g).outer (finProdFinEquiv.symm t).2 e)) * α ^ k *
            mark (fun o => o.map (B.gaps digits).place) x
              (((B.opening g).sliceCell (B.fold g) (finProdFinEquiv.symm t).1 e k).map
                (B.relocate g)) :=
    fun _ => rfl
  have zero : ∀ e : Fin (B.shape g).sliceLength × (B.shape g).OuterColumn,
      ((st.over (0 : Fin N → Zq)).key g).outer (finProdFinEquiv.symm t).2 e = 0 :=
    fun e => shared_zero _ _ _
  have form : ∀ S : Fin N → Zq, (outerRows B (st.over S) g t).scan α
      (mark (fun o => o.map (B.gaps digits).place) x) =
      (imageRow (B.shape g).chain (outerForm (B.opening g) (B.fold g) ((st.over S).key g))
        ((B.opening g).firstCell (B.fold g)) (st.payload g) t).scan α
        (fun o => mark (fun o => o.map (B.gaps digits).place) x (o.map (B.relocate g))) :=
    fun S => scan_relocate _ _ _ _
  unfold deltas
  rw [form, form, imageRow_deltas, scanForm, scanForm]
  simp only [zero, zero_mul, aeval_rep_zero, Finset.sum_const_zero, sub_zero]
  refine Finset.sum_congr rfl fun e _ => ?_
  by_cases inside : (B.shape g).sliceStart (finProdFinEquiv.symm t).1 + e.1 <
      (B.shape g).sliceStart ((finProdFinEquiv.symm t).1 + 1)
  · rw [ite_eq_left inside, Finset.mul_sum]
    refine Finset.sum_congr rfl fun k member => ?_
    have small := Finset.mem_range.mp member
    have cell : (B.opening g).sliceCell (B.fold g) (finProdFinEquiv.symm t).1 e k =
        some ((B.opening g).innerStart (B.fold g) + ((B.shape g).innerCode
          (⟨_, inside.trans_le ((B.shape g).sliceStart_le
            (Nat.succ_le_of_lt (finProdFinEquiv.symm t).1.isLt))⟩, e.2) *
          (B.shape g).outerDegree + k)) :=
      (dite_eq_left inside).trans (ite_eq_left small)
    have code := innerCode_slice (B.shape g) e (i := ⟨_, inside.trans_le ((B.shape g).sliceStart_le
      (Nat.succ_le_of_lt (finProdFinEquiv.symm t).1.isLt))⟩)
      (start := (B.shape g).sliceStart (finProdFinEquiv.symm t).1) rfl
    rw [cell]
    simp only [Option.map_some, mark, Option.some.injEq,
      Batch.place_main digits g (Nat.add_lt_add_left (cell_lt _ small) _)]
    rw [code]
    simp only [Nat.add_assoc]
    split_ifs <;> ring
  · rw [ite_eq_right inside, mul_zero]
    refine Finset.sum_eq_zero fun k _ => ?_
    have cell : (B.opening g).sliceCell (B.fold g) (finProdFinEquiv.symm t).1 e k = none :=
      dite_eq_right inside
    rw [cell]
    simp [mark]

/-- The weight at `ρ` and at `r` of the row `t` of `B` of the slices of the group `g`. -/
theorem outer_pair (st : B.Statement E) (digits : ℕ) (α : E) (ρ r : List E) {κ : ℕ}
    (blocked : B.Blocked κ) (cube : B.cells ≤ 2 ^ (r.length + κ)) (g : Fin n)
    (t : Fin ((B.shape g).slices * (B.shape g).outerRank))
    (fit : (B.shape g).outerInstance.size ≤ 2 ^ ρ.length) :
    setupPair ρ r κ (fun p x => deltas α (fun o => o.map (B.gaps digits).place) x
        (outerRows B (st.over (Pi.single p 1)) g)
        (outerRows B (st.over (0 : Fin (2 ^ ρ.length) → Zq)) g) t) =
      ∑ e : Fin (B.shape g).sliceLength × (B.shape g).OuterColumn,
        setupAt ρ α (B.shape g).outerDegree ((finProdFinEquiv.symm t).2 *
            ((B.shape g).sliceLength * ((B.shape g).innerRank *
              ((B.shape g).ratio * (B.shape g).openingDigits))) + (B.shape g).outerSetupCode e) *
          if (B.shape g).sliceStart (finProdFinEquiv.symm t).1 + e.1 <
              (B.shape g).sliceStart ((finProdFinEquiv.symm t).1 + 1) then
            runAt r α κ (B.shape g).outerDegree (B.mainStart g + ((B.opening g).innerStart (B.fold g) +
              ((B.shape g).sliceStart (finProdFinEquiv.symm t).1 *
                  ((B.shape g).innerRank * ((B.shape g).ratio * (B.shape g).openingDigits)) +
                (B.shape g).outerSetupCode e) * (B.shape g).outerDegree))
          else 0 := by
  unfold setupPair
  calc _ = ∑ p : Fin (2 ^ ρ.length), eqAt ρ p * mle r (fun y =>
          ∑ e : Fin (B.shape g).sliceLength × (B.shape g).OuterColumn,
            aeval α (rep (Setup.shared (Pi.single p 1) (B.shape g).outerDegree
              (B.shape g).outerRank _ (B.shape g).outerSetupCode (finProdFinEquiv.symm t).2 e)) *
            if (B.shape g).sliceStart (finProdFinEquiv.symm t).1 + e.1 <
                (B.shape g).sliceStart ((finProdFinEquiv.symm t).1 + 1) then
              ∑ k ∈ Finset.range (B.shape g).outerDegree,
                if B.mainStart g + ((B.opening g).innerStart (B.fold g) +
                    ((B.shape g).sliceStart (finProdFinEquiv.symm t).1 *
                        ((B.shape g).innerRank * ((B.shape g).ratio * (B.shape g).openingDigits)) +
                      (B.shape g).outerSetupCode e) * (B.shape g).outerDegree) + k = 2 ^ κ * y
                then α ^ k else 0
            else 0) :=
        Finset.sum_congr rfl fun p _ => congrArg _ (congrArg _ (funext fun y =>
          outer_deltas st digits α _ p g t))
    _ = _ := pair_sum' ρ r κ (fun e p => aeval α (rep (Setup.shared (Pi.single p 1)
          (B.shape g).outerDegree (B.shape g).outerRank _ (B.shape g).outerSetupCode
            (finProdFinEquiv.symm t).2 e)))
          (fun e x => if (B.shape g).sliceStart (finProdFinEquiv.symm t).1 + e.1 <
                (B.shape g).sliceStart ((finProdFinEquiv.symm t).1 + 1) then
              ∑ k ∈ Finset.range (B.shape g).outerDegree,
                if B.mainStart g + ((B.opening g).innerStart (B.fold g) +
                    ((B.shape g).sliceStart (finProdFinEquiv.symm t).1 *
                        ((B.shape g).innerRank * ((B.shape g).ratio * (B.shape g).openingDigits)) +
                      (B.shape g).outerSetupCode e) * (B.shape g).outerDegree) + k = x
                then α ^ k else 0
            else 0)
    _ = _ := Finset.sum_congr rfl fun e _ => by
        rw [sum_shared (d := (B.shape g).outerDegree) (m := (B.shape g).outerRank)
          (w := (B.shape g).sliceLength * ((B.shape g).innerRank *
            ((B.shape g).ratio * (B.shape g).openingDigits))) ρ α
          (B.shape g).outerSetupCode (finProdFinEquiv.symm t).2 e fit]
        by_cases inside : (B.shape g).sliceStart (finProdFinEquiv.symm t).1 + e.1 <
            (B.shape g).sliceStart ((finProdFinEquiv.symm t).1 + 1)
        · simp only [ite_eq_left inside]
          have start : 2 ^ κ ∣ B.mainStart g + ((B.opening g).innerStart (B.fold g) +
              ((B.shape g).sliceStart (finProdFinEquiv.symm t).1 *
                  ((B.shape g).innerRank * ((B.shape g).ratio * (B.shape g).openingDigits)) +
                (B.shape g).outerSetupCode e) * (B.shape g).outerDegree) :=
            dvd_add (blocked.1 g).1 (dvd_add (blocked.1 g).2.2.1
              (Dvd.dvd.mul_left (blocked.1 g).2.2.2.2 _))
          have code := innerCode_slice (B.shape g) e (i := ⟨_, inside.trans_le
            ((B.shape g).sliceStart_le (Nat.succ_le_of_lt (finProdFinEquiv.symm t).1.isLt))⟩)
            (start := (B.shape g).sliceStart (finProdFinEquiv.symm t).1) rfl
          have run := run_le (d := (B.shape g).outerDegree) ((B.shape g).innerCode
            (⟨_, inside.trans_le ((B.shape g).sliceStart_le
              (Nat.succ_le_of_lt (finProdFinEquiv.symm t).1.isLt))⟩, e.2))
          have bound : B.mainStart g + ((B.opening g).innerStart (B.fold g) +
              ((B.shape g).sliceStart (finProdFinEquiv.symm t).1 *
                  ((B.shape g).innerRank * ((B.shape g).ratio * (B.shape g).openingDigits)) +
                (B.shape g).outerSetupCode e) * (B.shape g).outerDegree) +
                (B.shape g).outerDegree ≤ 2 ^ (r.length + κ) := by
            have before := codeOffset_add_le B.mainCells g
            have total := B.chainStart_le_cells
            have main : (B.opening g).innerStart (B.fold g) + (B.shape g).blocks *
                (B.shape g).innerRank * (B.shape g).ratio * (B.shape g).openingDigits *
                  (B.shape g).outerDegree = B.mainCells g := rfl
            unfold Batch.mainStart
            unfold Batch.chainStart at total
            rw [code] at run
            omega
          rw [mle_run r α start (blocked.1 g).2.2.2.2 bound]
        · simp only [ite_eq_right inside, mul_zero]
          rw [mle_const r 0, mul_zero]

theorem open_deltas (st : B.Statement E) (digits : ℕ) (α : E) (x : ℕ) {N : ℕ} (p : Fin N)
    (j : Fin (1 * B.openRank)) :
    deltas α (fun o => o.map (B.gaps digits).place) x (openRows B (st.over (Pi.single p 1)))
        (openRows B (st.over (0 : Fin N → Zq))) j =
      ∑ column : B.OpenColumn,
        aeval α (rep ((st.over (Pi.single p 1)).openKey.matrix (finProdFinEquiv.symm j).2
          column)) *
          ∑ k ∈ Finset.range B.openDegree,
            if B.mainStart column.1 + ((B.shape column.1).responseCells (B.fold column.1) +
                (B.opening column.1).partialCode column.2 * B.openDegree) + k = x
            then α ^ k else 0 := by
  have scanForm : ∀ S : Fin N → Zq, (B.openForm (st.over S).openKey j).scan α
      (mark (fun o => o.map (B.gaps digits).place) x) =
      ∑ column : B.OpenColumn, ∑ k ∈ Finset.range B.openDegree,
        aeval α (rep ((st.over S).openKey.matrix (finProdFinEquiv.symm j).2 column)) * α ^ k *
          mark (fun o => o.map (B.gaps digits).place) x
            (((B.opening column.1).partialCell (B.fold column.1) column.2 k).map
              (B.relocate column.1)) :=
    fun _ => rfl
  have zero : ∀ column : B.OpenColumn,
      (st.over (0 : Fin N → Zq)).openKey.matrix (finProdFinEquiv.symm j).2 column = 0 :=
    fun column => shared_zero _ _ _
  have form : ∀ S : Fin N → Zq, (openRows B (st.over S) j).scan α
      (mark (fun o => o.map (B.gaps digits).place) x) =
      (imageRow B.chain (B.openForm (st.over S).openKey) B.openFirstCell st.openPayload j).scan α
        (mark (fun o => o.map (B.gaps digits).place) x) :=
    fun _ => rfl
  unfold deltas
  rw [form, form, imageRow_deltas, scanForm, scanForm]
  simp only [zero, zero_mul, aeval_rep_zero, Finset.sum_const_zero, sub_zero]
  refine Finset.sum_congr rfl fun column _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun k member => ?_
  have small := Finset.mem_range.mp member
  have cell : (B.opening column.1).partialCell (B.fold column.1) column.2 k =
      some ((B.shape column.1).responseCells (B.fold column.1) +
        ((B.opening column.1).partialCode column.2 * B.openDegree + k)) :=
    ite_eq_left small
  rw [cell]
  simp only [Option.map_some, mark, Option.some.injEq,
    Batch.place_main digits column.1 ((Nat.add_lt_add_left (cell_lt _ small) _).trans_le
      (Nat.le_add_right _ _)), Nat.add_assoc]
  split_ifs <;> ring

/-- The weight at `ρ` and at `r` of the row `j` of `D`. -/
theorem open_pair (st : B.Statement E) (digits : ℕ) (α : E) (ρ r : List E) {κ : ℕ}
    (blocked : B.Blocked κ) (cube : B.cells ≤ 2 ^ (r.length + κ)) (j : Fin (1 * B.openRank))
    (fit : B.openInstance.size ≤ 2 ^ ρ.length) :
    setupPair ρ r κ (fun p x => deltas α (fun o => o.map (B.gaps digits).place) x
        (openRows B (st.over (Pi.single p 1)))
        (openRows B (st.over (0 : Fin (2 ^ ρ.length) → Zq))) j) =
      ∑ column : B.OpenColumn,
        setupAt ρ α B.openDegree ((finProdFinEquiv.symm j).2 *
            (∑ g, (B.shape g).blocks * ((B.opening g).subcolumns * (B.shape g).openingDigits)) +
          B.openSetupCode column) *
          runAt r α κ B.openDegree (B.mainStart column.1 +
            ((B.shape column.1).responseCells (B.fold column.1) +
              (B.opening column.1).partialCode column.2 * B.openDegree)) := by
  unfold setupPair
  calc _ = ∑ p : Fin (2 ^ ρ.length), eqAt ρ p * mle r (fun y =>
          ∑ column : B.OpenColumn,
            aeval α (rep (Setup.shared (Pi.single p 1) B.openDegree B.openRank _ B.openSetupCode
              (finProdFinEquiv.symm j).2 column)) *
            ∑ k ∈ Finset.range B.openDegree,
              if B.mainStart column.1 + ((B.shape column.1).responseCells (B.fold column.1) +
                  (B.opening column.1).partialCode column.2 * B.openDegree) + k = 2 ^ κ * y
              then α ^ k else 0) :=
        Finset.sum_congr rfl fun p _ => congrArg _ (congrArg _ (funext fun y =>
          open_deltas st digits α _ p j))
    _ = _ := pair_sum' ρ r κ (fun column p => aeval α (rep (Setup.shared (Pi.single p 1)
          B.openDegree B.openRank _ B.openSetupCode (finProdFinEquiv.symm j).2 column)))
          (fun column x => ∑ k ∈ Finset.range B.openDegree,
            if B.mainStart column.1 + ((B.shape column.1).responseCells (B.fold column.1) +
                (B.opening column.1).partialCode column.2 * B.openDegree) + k = x
            then α ^ k else 0)
    _ = _ := Finset.sum_congr rfl fun column _ => by
        have start : 2 ^ κ ∣ B.mainStart column.1 +
            ((B.shape column.1).responseCells (B.fold column.1) +
              (B.opening column.1).partialCode column.2 * B.openDegree) :=
          dvd_add (blocked.1 column.1).1 (dvd_add (blocked.1 column.1).2.1
            (Dvd.dvd.mul_left blocked.2 _))
        have bound : B.mainStart column.1 +
            ((B.shape column.1).responseCells (B.fold column.1) +
              (B.opening column.1).partialCode column.2 * B.openDegree) + B.openDegree ≤
                2 ^ (r.length + κ) := by
          have run := run_le (d := B.openDegree) ((B.opening column.1).partialCode column.2)
          have before := codeOffset_add_le B.mainCells column.1
          have total := B.chainStart_le_cells
          have main : (B.shape column.1).responseCells (B.fold column.1) +
              (B.shape column.1).blocks * (B.opening column.1).subcolumns *
                (B.shape column.1).openingDigits * B.openDegree + (B.shape column.1).innerCells =
                B.mainCells column.1 := rfl
          unfold Batch.mainStart
          unfold Batch.chainStart at total
          omega
        rw [sum_shared (d := B.openDegree) (m := B.openRank)
            (w := ∑ g, (B.shape g).blocks * ((B.opening g).subcolumns * (B.shape g).openingDigits))
            ρ α B.openSetupCode (finProdFinEquiv.symm j).2 column fit,
          mle_run r α start blocked.2 bound]

variable (B) in
/-- The weight at `ρ` and at `r` of the rows (104) of `A` of the group `g`, after the row
`row`. -/
noncomputable def innerPart (τ ρ r : List E) (α : E) (κ : ℕ) (g : Fin n) (row : ℕ) : E :=
  ∑ a : Fin (B.shape g).innerRank, eqAt τ (row + 1 + a) *
    ∑ col : (Fin (B.shape g).blockLength × Fin (B.shape g).sourceDigits) ×
        Fin (B.fold g).responseDigits,
      -((B.fold g).responseBase : E) ^ (col.2 : ℕ) *
        setupAt ρ α (B.shape g).innerDegree
          (a * ((B.shape g).blockLength * (B.shape g).sourceDigits) +
            (B.shape g).innerSetupCode col.1) *
        runAt r α κ (B.shape g).innerDegree
          (B.mainStart g + (B.shape g).responseCode (B.fold g) col * (B.shape g).innerDegree)

variable (B) in
/-- The weight at `ρ` and at `r` of the rows (105) of `B` of the slices of the group `g`, after
the rows of `A`. -/
noncomputable def outerPart (τ ρ r : List E) (α : E) (κ : ℕ) (g : Fin n) (row : ℕ) : E :=
  ∑ t : Fin ((B.shape g).slices * (B.shape g).outerRank),
    eqAt τ (row + 1 + (B.shape g).innerRank + t) *
      ∑ e : Fin (B.shape g).sliceLength × (B.shape g).OuterColumn,
        setupAt ρ α (B.shape g).outerDegree ((finProdFinEquiv.symm t).2 *
            ((B.shape g).sliceLength * ((B.shape g).innerRank *
              ((B.shape g).ratio * (B.shape g).openingDigits))) + (B.shape g).outerSetupCode e) *
          if (B.shape g).sliceStart (finProdFinEquiv.symm t).1 + e.1 <
              (B.shape g).sliceStart ((finProdFinEquiv.symm t).1 + 1) then
            runAt r α κ (B.shape g).outerDegree (B.mainStart g + ((B.opening g).innerStart (B.fold g) +
              ((B.shape g).sliceStart (finProdFinEquiv.symm t).1 *
                  ((B.shape g).innerRank * ((B.shape g).ratio * (B.shape g).openingDigits)) +
                (B.shape g).outerSetupCode e) * (B.shape g).outerDegree))
          else 0

variable (B) in
/-- The weight at `ρ` and at `r` of the rows (106) of `D`, from the row `row`. -/
noncomputable def openPart (τ ρ r : List E) (α : E) (κ : ℕ) (row : ℕ) : E :=
  ∑ j : Fin (1 * B.openRank), eqAt τ (row + j) *
    ∑ column : B.OpenColumn,
      setupAt ρ α B.openDegree ((finProdFinEquiv.symm j).2 *
          (∑ g, (B.shape g).blocks * ((B.opening g).subcolumns * (B.shape g).openingDigits)) +
        B.openSetupCode column) *
        runAt r α κ B.openDegree (B.mainStart column.1 +
          ((B.shape column.1).responseCells (B.fold column.1) +
            (B.opening column.1).partialCode column.2 * B.openDegree))

/-- The weight of Stage 3 of `Batch.codeScan` at the point `ρ`: the sum over the entries of the
views of `A` and `B` of each group, in the order `groupOrder`, and of `D`, of the weight of the
batching row times the weight at `ρ` of the entry (`setupAt`) times the weight of its run of
cells at `r_2[κ..]` (`runAt`). -/
theorem codeScan_mle [Fintype E] (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (st : B.Statement E) (challenge : B.Challenge) (μ : ℕ) (fits : B.cells ≤ 2 ^ μ)
    (base digits : ℕ) (room : B.quotientRoom digits ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) (ν : ℕ) (flat : Table) (κ : ℕ)
    (blocked : B.Blocked κ) (low : κ ≤ μ) (α : E) (τ r2 ρ : List E) (τLength : τ.length = batch)
    (r2Length : r2.length = μ) (ρLength : ρ.length = ν)
    (innerFit : ∀ g, ((B.shape g).innerInstance false 0).size ≤ 2 ^ ν)
    (outerFit : ∀ g, (B.shape g).outerInstance.size ≤ 2 ^ ν)
    (openFit : B.openInstance.size ≤ 2 ^ ν) :
    mle ρ ((Batch.codeScan β st challenge μ fits base digits room batch rowsFit ν flat κ blocked
        low).weight α τ r2) =
      ∑ k, (innerPart B τ ρ (r2.drop κ) α κ (groupOrder n k)
            (rowsBefore (fun j => B.groupCount (groupOrder n j)) k) +
          outerPart B τ ρ (r2.drop κ) α κ (groupOrder n k)
            (rowsBefore (fun j => B.groupCount (groupOrder n j)) k)) +
        openPart B τ ρ (r2.drop κ) α κ (∑ g, B.groupCount g) := by
  subst τLength r2Length ρLength
  have cube : B.cells ≤ 2 ^ ((r2.drop κ).length + κ) := by
    rw [List.length_drop, Nat.sub_add_cancel low]
    exact fits
  have expand : mle ρ ((Batch.codeScan β st challenge r2.length fits base digits room τ.length
      rowsFit ρ.length flat κ blocked low).weight α τ r2) =
      setupPair ρ (r2.drop κ) κ (fun p x =>
        ((Batch.liftedReduction β (st.over (Pi.single p 1)) challenge r2.length fits base digits
          room τ.length rowsFit).rows α).batchWeight τ x -
        ((Batch.liftedReduction β (st.over (0 : Fin (2 ^ ρ.length) → Zq)) challenge r2.length
          fits base digits room τ.length rowsFit).rows α).batchWeight τ x) := by
    rw [mle_eq_cubeSum, cubeSum_eq_sum_range, ← Fin.sum_univ_eq_sum_range]
    refine Finset.sum_congr rfl fun p _ => ?_
    simp only [Batch.codeScan, dite_eq_left p.isLt, Fin.eta]
  rw [expand, setupPair_congr ρ _ κ (fun p x =>
    (codeWeight_sub β st challenge _ fits base digits room _ rowsFit α τ (Pi.single p 1) x).trans
      (rows_mle β st challenge α _ x (Pi.single p 1) τ rowsFit))]
  simp only [setupPair_add, setupPair_sum, setupPair_mul_left]
  unfold innerPart outerPart openPart
  refine congrArg₂ (· + ·) (Finset.sum_congr rfl fun k _ => congrArg₂ (· + ·) ?_ ?_) ?_
  · exact Finset.sum_congr rfl fun a _ => congrArg _
      (inner_pair st challenge digits α ρ _ blocked cube _ a (innerFit _))
  · exact Finset.sum_congr rfl fun t _ => congrArg _
      (outer_pair st digits α ρ _ blocked cube _ t (outerFit _))
  · exact Finset.sum_congr rfl fun j _ => congrArg _
      (open_pair st digits α ρ _ blocked cube j openFit)

end Weight

end NightstreamFPrime.Spec.Compression.Akita.Replay
