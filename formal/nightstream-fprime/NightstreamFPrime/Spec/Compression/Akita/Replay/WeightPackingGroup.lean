import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingRows

/-! The parts of one group of a packing fold (`groupParts` of `Replay/WeightPacking.lean`) as the
linear forms of the rows of `flatRows` (ePrint 2026/1983, (102), (104), (105) and (47)).

- `partialLifts`, `responseLifts`, `imageLifts`, `outerLifts`, `chainLifts` and
  `contractionValue` are the parts of `groupParts` with the moved addresses as a parameter
  (`groupParts_eq`).
- `Placement.move_one` states that the cells of the code of a fold with one group are the
  positions `Gaps.place` of `OpeningShape.gaps`.
- The value lemmas give each part at `r_2` as the linear forms `Form.scan` of the rows of
  `FlatFold.lean` with the moved addresses, for each map `move` that is a shift on the segments of
  the flat witness and keeps the cells in the cube. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

section Parts

variable (F : FoldInputs) (move : ℕ → ℕ) (P : PackGroup) (row : ℕ)

/-- The partials of the carrier row (102) of a group. -/
def partialLifts : List Lift :=
  (List.finRange P.shape.blocks).flatMap fun i =>
    (List.range P.opening.planes).map fun t =>
      { degree := P.opening.carrierDegree
        bits := min (Nat.log 2 P.opening.carrierDegree) (Nat.log 2 P.opening.openDegree)
        reads := (List.finRange P.shape.openingDigits).map fun u =>
          (natFq (P.shape.openingBase ^ u.val), moved move fun j =>
            P.opening.partialPosition P.fold i u (t * P.opening.carrierDegree + j))
        muls := [(eqIndex F.tau1 row * extBasis.getD t Ext.zero,
          P.challenges.getD i.val (Poly.zero P.opening.carrierDegree))] }

/-- The rows (104) of `A` on `ẑ` of a group. -/
def responseLifts : List Lift :=
  (List.finRange P.shape.blockLength).flatMap fun x =>
    (List.finRange P.shape.sourceDigits).map fun u => Lift.ofTerm
      { degree := P.shape.innerDegree
        reads := (List.finRange P.fold.responseDigits).map fun f =>
          (natFq (P.fold.responseBase ^ f.val), moved move (P.shape.responseCell P.fold ((x, u), f)))
        muls := (List.finRange P.shape.innerRank).map fun a =>
          (-eqIndex F.tau1 (row + 1 + a.val), viewEntry F.setup P.shape.innerDegree
            (P.shape.blockLength * P.shape.sourceDigits) a.val (P.shape.innerSetupCode (x, u)).val) }

/-- The images `Σ_i ι(c_i)·G·t̂_{i,a}` of the rows (104) of a group. -/
def imageLifts : List Lift :=
  (List.finRange P.shape.blocks).flatMap fun i =>
    (List.finRange P.shape.innerRank).map fun a =>
      { degree := P.shape.innerDegree, bits := Nat.log 2 P.shape.outerDegree
        reads := (List.finRange P.shape.openingDigits).map fun u =>
          (natFq (P.shape.openingBase ^ u.val), moved move (P.opening.innerCell P.fold i a u))
        muls := [(eqIndex F.tau1 (row + 1 + a.val),
          P.ambient.getD i.val (Poly.zero P.shape.innerDegree))] }

/-- The rows (105) of `B` of a group. -/
def outerLifts : List Lift :=
  (List.finRange P.shape.slices).flatMap fun s =>
    (List.finRange P.shape.sliceLength).flatMap fun p =>
      (List.finRange P.shape.innerRank).flatMap fun a =>
        (List.finRange P.shape.ratio).flatMap fun y =>
          (List.finRange P.shape.openingDigits).map fun u => Lift.ofTerm
            { degree := P.shape.outerDegree
              reads := [(Fq.one, moved move (P.opening.sliceCell P.fold s (p, (a, y, u))))]
              muls := (List.finRange P.shape.outerRank).map fun r =>
                (eqIndex F.tau1 (row + 1 + P.shape.innerRank + (s.val * P.shape.outerRank + r.val)),
                  viewEntry F.setup P.shape.outerDegree
                    (P.shape.sliceLength * (P.shape.innerRank * (P.shape.ratio * P.shape.openingDigits)))
                    r.val (P.shape.outerSetupCode (p, (a, y, u))).val) }

/-- The compression rows (47) of the path `F` of a group. -/
def chainLifts (first second : ℕ) : List (List Lift) :=
  (chainTerms F.setup (eqIndex F.tau1) P.shape.chain P.shape.outerDegree
    (P.shape.slices * P.shape.outerRank) (moved move (P.opening.firstCell P.fold))
    (moved move (P.opening.secondCell P.fold)) (row + 1 + P.shape.innerRank) first second).map
      (·.map Lift.ofTerm)

/-- The contraction of the carrier row (102) of a group. -/
def contractionValue (T : Tables) : Ext :=
  -(eqIndex F.tau1 row) * sumList (List.finRange P.shape.blockLength) fun x =>
    eqIndex P.segments.2.2.1 x.val * sumList (List.finRange P.shape.sourceDigits) fun u =>
      sumList (List.finRange P.fold.responseDigits) fun f =>
        Ext.scale (natFq (P.shape.sourceBase ^ u.val * P.fold.responseBase ^ f.val))
          (T.read (T.lowsOf (Nat.log 2 P.shape.innerDegree) P.shape.innerDegree fun j =>
            eqIndex P.segments.1 (j % P.opening.stride) * T.pow (j / P.opening.stride))
            (Nat.log 2 P.shape.innerDegree) 1 (moved move (P.shape.responseCell P.fold ((x, u), f))))

/-- The columns of a group in the rows (106) of `D` (`openParts`). -/
def openLifts (dRow column width : ℕ) : List Lift :=
  (List.finRange P.shape.blocks).flatMap fun i =>
    (List.finRange P.opening.subcolumns).flatMap fun y =>
      (List.finRange P.shape.openingDigits).map fun u => Lift.ofTerm
        { degree := P.opening.openDegree
          reads := [(Fq.one, moved move (P.opening.partialCell P.fold (i, y, u)))]
          muls := (List.finRange P.opening.openRank).map fun r =>
            (eqIndex F.tau1 (dRow + r.val), viewEntry F.setup P.opening.openDegree width r.val
              (column + (P.opening.openSetupCode (i, y, u)).val)) }

theorem openParts_eq (gs : List PackGroup) (L : Placement) (k dRow column width : ℕ) :
    openParts F gs L k P dRow column width = openLifts F (L.move gs k F.cells) P dRow column width :=
  rfl

theorem groupParts_eq (gs : List PackGroup) (L : Placement) (k first second : ℕ) (T : Tables) :
    groupParts F gs L k P row first second T =
      ([partialLifts F (L.move gs k F.cells) P row, responseLifts F (L.move gs k F.cells) P row,
          imageLifts F (L.move gs k F.cells) P row, outerLifts F (L.move gs k F.cells) P row] ++
        chainLifts F (L.move gs k F.cells) P row first second,
        contractionValue F (L.move gs k F.cells) P row T) :=
  rfl

end Parts

/-- The cells of the code of a fold with one group and the path `D` of its opening shape are the
positions `Gaps.place` of the quotient segments of Akita (76). -/
theorem Placement.move_one (P : PackGroup) (digits : ℕ) :
    (Placement.of [P] ⟨P.opening.openDegree, P.opening.openRank, P.opening.chain⟩ digits).move [P] 0
      {} = (P.opening.gaps P.fold digits).place := by
  funext x
  have starts := (P.opening.gaps P.fold digits).ordered
  simp only [Placement.move, Placement.of, List.getElem?_cons_zero, beforeIn, before,
    List.take_zero, List.map_nil, List.sum_nil, List.map_cons, List.sum_cons, add_zero, zero_add]
  have main : P.mainCells + digits * (P.mainDegrees + P.opening.openRank * P.opening.openDegree) =
      P.opening.chainStart P.fold + digits * P.opening.mainDegrees := by
    unfold PackGroup.mainCells PackGroup.mainDegrees OpeningShape.mainDegrees
    ring
  have first : digits * (firstRank P.shape.chain * firstDegree P.shape.chain +
      firstRank P.opening.chain * firstDegree P.opening.chain) =
        digits * P.opening.firstDegrees := rfl
  rw [main, first]
  unfold Gaps.place OpeningShape.gaps
  simp only
  unfold OpeningShape.cells OpeningShape.openSecondStart OpeningShape.secondStart
    OpeningShape.openFirstStart at *
  generalize digits * P.opening.mainDegrees = A
  generalize digits * P.opening.firstDegrees = B
  split_ifs <;> omega

theorem runs_one {m d bound : ℕ} {f : ℕ → Option ℕ} (pow : 2 ^ m = d) (runs : Runs m 1 bound f) :
    Runs m (d / 2 ^ m) bound f := by
  rw [pow, Nat.div_self (pow ▸ Nat.two_pow_pos m)]
  exact runs

section Reads

variable {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E]

omit [Degree2 E] in
theorem reads_value {n d : ℕ} (s : Fin n → ℕ) (addr : Fin n → ℕ → Option ℕ) (a : E)
    (φ : ℕ → E) :
    ((((List.finRange n).map fun u => (natFq (s u), addr u)) : List Read).map fun r =>
      algebraMap Zq E r.1.toZq * ∑ j ∈ Finset.range d, a ^ j * (r.2 j).elim 0 φ).sum =
        ∑ u, (s u : E) * ∑ j ∈ Finset.range d, a ^ j * (addr u j).elim 0 φ := by
  rw [List.map_map, sum_finRange]
  refine Finset.sum_congr rfl fun u _ => ?_
  simp only [Function.comp, natFq, Fq.toZq_ofNat, map_natCast]

omit [Degree2 E] in
theorem reads_single {d : ℕ} (addr : ℕ → Option ℕ) (a : E) (φ : ℕ → E) :
    ((([(Fq.one, addr)]) : List Read).map fun r =>
      algebraMap Zq E r.1.toZq * ∑ j ∈ Finset.range d, a ^ j * (r.2 j).elim 0 φ).sum =
        ∑ j ∈ Finset.range d, a ^ j * (addr j).elim 0 φ := by
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Fq.toZq_one, map_one,
    one_mul, add_zero]

theorem Tables.pow_into (α : Ext) (r2 : List Ext) (bits top k : ℕ) :
    (((Tables.of α r2 bits top).pow k).into : E) = α.into ^ k := by
  show (((powersOf α top).getD k (Ext.pow α k)).into : E) = _
  by_cases small : k < top
  · rw [show (powersOf α top).getD k (Ext.pow α k) = (powersOf α top).getD k Ext.zero by
        simp [Array.getD, powersOf, small], powersOf_getD α small]
  · rw [show (powersOf α top).getD k (Ext.pow α k) = Ext.pow α k by
        simp [Array.getD, powersOf, small], Ext.into_pow]

/-- The value of the lift of a term of a degree `d = 2^m` whose reads are runs of `d` cells. -/
theorem Lift.ofTerm_value_into (α : Ext) (r2 : List Ext) (bits top : ℕ) (t : Term)
    [NeZero t.degree] (short : bits ≤ r2.length) (pow : 2 ^ Nat.log 2 t.degree = t.degree)
    (le : Nat.log 2 t.degree ≤ bits) (degree : t.degree ≤ top)
    (runs : ∀ r ∈ t.reads, Runs (Nat.log 2 t.degree) 1 (2 ^ r2.length) r.2) :
    (((Lift.ofTerm t).value (Tables.of α r2 bits top)).into : E) =
      (t.reads.map fun r => algebraMap Zq E r.1.toZq * ∑ j ∈ Finset.range t.degree,
        (α.into : E) ^ j * (r.2 j).elim 0 (eqAt (r2.map Ext.into))).sum *
      (t.muls.map fun m => (m.1.into : E) *
        Polynomial.aeval (α.into : E) (rep (Poly.toRq t.degree m.2))).sum :=
  have : NeZero (Lift.ofTerm t).degree := ‹NeZero t.degree›
  Lift.value_into α r2 bits top (Lift.ofTerm t) short le (dvd_of_eq pow) degree
    fun r member => runs_one pow (runs r member)

end Reads

section Values

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

/-- A read of the main segments `[ẑ | ê | t̂]` that is a run of cells moves to a run of cells by a
map that is a shift on the main segments. -/
theorem Runs.main_of {S : Shape} {O : OpeningShape S} {fold : Descriptor} {m n bound : ℕ}
    {f : ℕ → Option ℕ} {move : ℕ → ℕ}
    (shift : ∀ x c, x + c < O.chainStart fold → move (x + c) = move x + c)
    (bounded : ∀ x < O.cells fold, move x < bound)
    (run : ∀ y < n, ∀ c < 2 ^ m, f (y * 2 ^ m + c) = (f (y * 2 ^ m)).map (· + c))
    (within : ∀ j x, f j = some x → x < O.chainStart fold) :
    Runs m n bound (moved move f) :=
  Runs.moved_of (lo := 0) (hi := O.chainStart fold) (fun x c _ high => shift x c high) run
    (fun j x read => ⟨Nat.zero_le _, within j x read⟩)
    (fun j x read => bounded x ((within j x read).trans_le (O.chainStart_le_cells fold)))

variable (F : FoldInputs) (move : ℕ → ℕ) (P : PackGroup) (row : ℕ) (α : Ext) (r2 : List Ext)
  (bits top : ℕ)

/-- The partials of a group are the left sides of the planes of the carrier row (102), each with
the coefficient `ẽq(τ_1, row)·β_t`. -/
theorem partialLifts_value (short : bits ≤ r2.length)
    (carrierPow : 2 ^ Nat.log 2 P.opening.carrierDegree = P.opening.carrierDegree)
    (openPow : 2 ^ Nat.log 2 P.opening.openDegree = P.opening.openDegree)
    (le : Nat.log 2 P.opening.carrierDegree ≤ bits) (degree : P.opening.carrierDegree ≤ top)
    (shift : ∀ x c, x + c < P.opening.chainStart P.fold → move (x + c) = move x + c)
    (bounded : ∀ x < P.opening.cells P.fold, move x < 2 ^ r2.length)
    (challenge : Fin P.shape.blocks → Rq P.opening.carrierDegree)
    (challenged : ∀ i, challenge i =
      Poly.toRq _ (P.challenges.getD i (Poly.zero P.opening.carrierDegree))) :
    ((partialLifts F move P row).map fun l => ((l.value (Tables.of α r2 bits top)).into : E)).sum =
      ∑ t : Fin P.opening.planes, ((eqIndex F.tau1 row).into : E) *
        (((extBasis.getD t Ext.zero).into : E) *
          (partialForm P.opening P.fold challenge t).scan (α.into : E)
            (fun o => (o.map move).elim 0 (eqAt (r2.map Ext.into)))) := by
  have divides : 2 ^ min (Nat.log 2 P.opening.carrierDegree) (Nat.log 2 P.opening.openDegree) ∣
      P.opening.carrierDegree := (Nat.pow_dvd_pow 2 (min_le_left _ _)).trans (dvd_of_eq carrierPow)
  have opened : 2 ^ min (Nat.log 2 P.opening.carrierDegree) (Nat.log 2 P.opening.openDegree) ∣
      P.opening.openDegree := (Nat.pow_dvd_pow 2 (min_le_right _ _)).trans (dvd_of_eq openPow)
  unfold partialLifts
  rw [sum_flatMap]
  simp only [List.map_map, Function.comp_def, sum_finRange, sum_range_list]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [partialForm_scan, Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Lift.value_into α r2 bits top _ short ((min_le_left _ _).trans le) divides degree
    (fun r member => by
      obtain ⟨u, _, rfl⟩ := List.mem_map.mp member
      exact Runs.main_of shift bounded
        (OpeningShape.partialPosition_run P.opening P.fold i u t divides opened)
        (fun j x read => P.opening.partialPosition_lt_innerStart P.fold i u _ read |>.trans_le
          (Nat.le_add_right _ _)))]
  rw [reads_value]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, Ext.into_mul,
    ← challenged, moved]
  rw [Finset.mul_sum, Finset.mul_sum, Finset.sum_mul]
  refine Finset.sum_congr rfl fun u _ => ?_
  ring

omit [Degree2 E] in
/-- `Σ_t c·β_t·Σ_x Σ_a [v_{x,a}]_t·R_{x,a} = c·Σ_x Σ_a v_{x,a}·R_{x,a}` for the coordinates
`[v]_t` in the basis `β`. -/
theorem basis_collapse {p : ℕ} {ι κ : Type} [Fintype ι] [Fintype κ]
    (β : Module.Basis (Fin p) Zq E) (collapse : ∀ v : E, ∑ t, β t * algebraMap Zq E (β.repr v t) = v)
    (c : E) (v : ι → κ → E) (R : ι → κ → E) :
    ∑ t, c * (β t * ∑ x, ∑ a, algebraMap Zq E (β.repr (v x a) t) * R x a) =
      c * ∑ x, ∑ a, v x a * R x a := by
  rw [← Finset.mul_sum]
  congr 1
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun a _ => ?_
  conv_rhs => rw [← collapse (v x a)]
  rw [Finset.sum_mul]
  exact Finset.sum_congr rfl fun t _ => by ring

theorem ofFn_getD {n k : ℕ} (w : ℕ → Ext) (small : k < n) :
    (Array.ofFn (n := n) fun i => w i.val).getD k Ext.zero = w k := by
  simp [Array.getD, small]

theorem responseCell_lt_chainStart {S : Shape} (O : OpeningShape S) (fold : Descriptor)
    (col : (Fin S.blockLength × Fin S.sourceDigits) × Fin fold.responseDigits) {k x : ℕ}
    (read : S.responseCell fold col k = some x) : x < O.chainStart fold := by
  unfold Shape.responseCell at read
  split at read
  · rename_i small
    cases read
    have inside := cell_lt (S.responseCode fold col) small
    unfold OpeningShape.chainStart OpeningShape.innerStart Shape.responseCells
    omega
  · cases read

/-- The contraction of a group is the right side of the planes of the carrier row (102), each with
the coefficient `ẽq(τ_1, row)·β_t`, for the weights `ẽq(r_pos, x)` and `ẽq(r_pack, a)` of the
opening point: `Σ_t β_t·[ẽq(r_pos, x)·ẽq(r_pack, a)]_t = ẽq(r_pos, x)·ẽq(r_pack, a)`. -/
theorem contractionValue_into (short : bits ≤ r2.length)
    (innerPow : 2 ^ Nat.log 2 P.shape.innerDegree = P.shape.innerDegree)
    (le : Nat.log 2 P.shape.innerDegree ≤ bits)
    (shift : ∀ x c, x + c < P.opening.chainStart P.fold → move (x + c) = move x + c)
    (bounded : ∀ x < P.opening.cells P.fold, move x < 2 ^ r2.length)
    (β : Module.Basis (Fin P.opening.planes) Zq E) (point : Point P.opening E)
    (positioned : ∀ x, point.position x = eqAt (P.segments.2.2.1.map Ext.into) x)
    (packed : ∀ a, point.pack a = eqAt (P.segments.1.map Ext.into) a) :
    ((contractionValue F move P row (Tables.of α r2 bits top)).into : E) =
      -∑ t : Fin P.opening.planes, ((eqIndex F.tau1 row).into : E) * (β t *
        (contractForm P.opening P.fold β point t).scan (α.into : E)
          (fun o => (o.map move).elim 0 (eqAt (r2.map Ext.into)))) := by
  have collapse : ∀ v : E, ∑ t, β t * algebraMap Zq E (β.repr v t) = v := fun v => by
    conv_rhs => rw [← β.sum_repr v]
    exact Finset.sum_congr rfl fun t _ => by rw [Algebra.smul_def, mul_comm]
  have tower : P.shape.innerDegree = P.opening.carrierDegree * P.opening.stride := by
    rw [← P.opening.stride_tower, mul_comm]
  have stride : 0 < P.opening.stride :=
    Nat.pos_of_ne_zero fun zero => by
      have := P.opening.stride_tower
      rw [zero, zero_mul] at this
      exact NeZero.ne P.shape.innerDegree this.symm
  have read : ∀ col, (((Tables.of α r2 bits top).read
      ((Tables.of α r2 bits top).lowsOf (Nat.log 2 P.shape.innerDegree) P.shape.innerDegree
        fun j => eqIndex P.segments.1 (j % P.opening.stride) *
          (Tables.of α r2 bits top).pow (j / P.opening.stride))
      (Nat.log 2 P.shape.innerDegree) 1 (moved move (P.shape.responseCell P.fold col))).into : E) =
      ∑ a : Fin P.opening.stride, point.pack a * ∑ j ∈ Finset.range P.opening.carrierDegree,
        (α.into : E) ^ j * ((P.shape.responseCell P.fold col (a + P.opening.stride * j)).map
          move).elim 0 (eqAt (r2.map Ext.into)) := by
    intro col
    rw [show (Tables.of α r2 bits top).lowsOf (Nat.log 2 P.shape.innerDegree) P.shape.innerDegree
        (fun j => eqIndex P.segments.1 (j % P.opening.stride) *
          (Tables.of α r2 bits top).pow (j / P.opening.stride)) =
        lowSums r2 (Nat.log 2 P.shape.innerDegree) (Array.ofFn (n := P.shape.innerDegree) fun k =>
          eqIndex P.segments.1 (k.val % P.opening.stride) *
            (Tables.of α r2 bits top).pow (k.val / P.opening.stride)) from rfl,
      Tables.read_into α r2 bits top _ le short le_rfl
        (by rw [one_mul, innerPow, Array.size_ofFn])
        (Runs.main_of shift bounded
          (run_of_start (s := P.shape.responseCode P.fold col * P.shape.innerDegree)
            (fun k small => Shape.responseCell_eq P.fold col small) (by rw [one_mul, innerPow]))
          (fun j x read => responseCell_lt_chainStart P.opening P.fold col read)),
      one_mul, innerPow, tower, ← sum_range_mul (M := E), Finset.sum_comm,
      ← Fin.sum_univ_eq_sum_range]
    refine Finset.sum_congr rfl fun a _ => ?_
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j member => ?_
    have small := Finset.mem_range.mp member
    have mod : (j * P.opening.stride + a) % P.opening.stride = a := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt a.isLt]
    have div : (j * P.opening.stride + a) / P.opening.stride = j := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ stride, Nat.div_eq_of_lt a.isLt, zero_add]
    have inside : j * P.opening.stride + a < P.opening.carrierDegree * P.opening.stride := by
      have := Nat.mul_le_mul_right P.opening.stride (Nat.succ_le_of_lt small)
      rw [Nat.succ_mul] at this
      omega
    rw [ofFn_getD (fun k => eqIndex P.segments.1 (k % P.opening.stride) *
        (Tables.of α r2 bits top).pow (k / P.opening.stride)) (by rw [← tower] at inside ⊢; exact inside),
      Ext.into_mul, eqIndex_into, Tables.pow_into, mod, div, ← packed,
      show j * P.opening.stride + a = a + P.opening.stride * j by ring]
    simp only [moved]
    ring
  unfold contractionValue
  simp only [Ext.into_mul, Ext.into_neg, sumList_into, sum_finRange, eqIndex_into, Ext.into_scale,
    natFq, Fq.toZq_ofNat, map_natCast, read, ← positioned]
  simp only [contractForm_scan]
  have inner : ∀ (t : Fin P.opening.planes) (x : Fin P.shape.blockLength) (a : Fin P.opening.stride),
      ∑ u : Fin P.shape.sourceDigits, ∑ f : Fin P.fold.responseDigits,
        algebraMap Zq E (β.repr (point.position x * point.pack a) t) *
          ((P.shape.sourceBase ^ (u : ℕ) * P.fold.responseBase ^ (f : ℕ) : ℕ) : E) *
            ∑ j ∈ Finset.range P.opening.carrierDegree, (α.into : E) ^ j *
              ((P.shape.responseCell P.fold ((x, u), f) (a + P.opening.stride * j)).map move).elim 0
                (eqAt (r2.map Ext.into)) =
        algebraMap Zq E (β.repr (point.position x * point.pack a) t) *
          ∑ u : Fin P.shape.sourceDigits, ∑ f : Fin P.fold.responseDigits,
            ((P.shape.sourceBase ^ (u : ℕ) * P.fold.responseBase ^ (f : ℕ) : ℕ) : E) *
              ∑ j ∈ Finset.range P.opening.carrierDegree, (α.into : E) ^ j *
                ((P.shape.responseCell P.fold ((x, u), f) (a + P.opening.stride * j)).map move).elim 0
                  (eqAt (r2.map Ext.into)) := fun t x a => by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun u _ => by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun f _ => by ring
  simp only [inner, basis_collapse β collapse]
  rw [neg_mul]
  congr 2
  refine Finset.sum_congr rfl fun x _ => ?_
  simp only [Finset.mul_sum]
  conv_rhs => rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun u _ => ?_
  conv_rhs => rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun f _ => Finset.sum_congr rfl fun a _ =>
    Finset.sum_congr rfl fun j _ => ?_
  ring

theorem outerCell_lt_chainStart {S : Shape} (O : OpeningShape S) (fold : Descriptor)
    (i : Fin S.blocks) (col : S.OuterColumn) {k x : ℕ} (read : O.outerCell fold i col k = some x) :
    x < O.chainStart fold := by
  unfold OpeningShape.outerCell at read
  split at read
  · rename_i small
    cases read
    have inside := cell_lt (S.innerCode (i, col)) small
    unfold OpeningShape.chainStart Shape.innerCells
    omega
  · cases read

theorem innerCell_lt_chainStart {S : Shape} (O : OpeningShape S) (fold : Descriptor)
    (i : Fin S.blocks) (a : Fin S.innerRank) (u : Fin S.openingDigits) {k x : ℕ}
    (read : O.innerCell fold i a u k = some x) : x < O.chainStart fold := by
  unfold OpeningShape.innerCell at read
  split at read
  · exact outerCell_lt_chainStart O fold _ _ read
  · cases read

theorem sliceCell_lt_chainStart {S : Shape} (O : OpeningShape S) (fold : Descriptor)
    (s : Fin S.slices) (entry : Fin S.sliceLength × S.OuterColumn) {k x : ℕ}
    (read : O.sliceCell fold s entry k = some x) : x < O.chainStart fold := by
  unfold OpeningShape.sliceCell at read
  split at read
  · exact outerCell_lt_chainStart O fold _ _ read
  · cases read

theorem partialCell_lt_chainStart {S : Shape} (O : OpeningShape S) (fold : Descriptor)
    (col : O.OpenColumn) {k x : ℕ} (read : O.partialCell fold col k = some x) :
    x < O.chainStart fold := by
  unfold OpeningShape.partialCell at read
  split at read
  · rename_i small
    cases read
    have inside := cell_lt (O.partialCode col) small
    unfold OpeningShape.chainStart OpeningShape.innerStart OpeningShape.partialCells
    omega
  · cases read

omit [Degree2 E] in
theorem sum_comm_in2 {M : Type} [AddCommMonoid M] {α β γ : Type} [Fintype α] [Fintype β]
    [Fintype γ] (g : α → β → γ → M) :
    ∑ a, ∑ x, ∑ u, g a x u = ∑ x, ∑ u, ∑ a, g a x u := by
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun x _ => Finset.sum_comm

omit [Degree2 E] in
theorem sum_comm_in3 {M : Type} [AddCommMonoid M] {α β γ δ : Type} [Fintype α] [Fintype β]
    [Fintype γ] [Fintype δ] (g : α → β → γ → δ → M) :
    ∑ a, ∑ x, ∑ u, ∑ f, g a x u f = ∑ x, ∑ u, ∑ f, ∑ a, g a x u f := by
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun x _ => sum_comm_in2 _

/-- The lifts of `ẑ` of a group are the rows (104) of `A` on `ẑ`, each with the coefficient
`ẽq(τ_1, row + 1 + a)` and the sign of the code. -/
theorem responseLifts_value (short : bits ≤ r2.length)
    (innerPow : 2 ^ Nat.log 2 P.shape.innerDegree = P.shape.innerDegree)
    (le : Nat.log 2 P.shape.innerDegree ≤ bits) (degree : P.shape.innerDegree ≤ top)
    (shift : ∀ x c, x + c < P.opening.chainStart P.fold → move (x + c) = move x + c)
    (bounded : ∀ x < P.opening.cells P.fold, move x < 2 ^ r2.length) (key : Key P.shape)
    (keyed : key.inner = Akita.Setup.shared (flatOf F.setup) P.shape.innerDegree P.shape.innerRank
      _ P.shape.innerSetupCode) :
    ((responseLifts F move P row).map fun l => ((l.value (Tables.of α r2 bits top)).into : E)).sum =
      -∑ a : Fin P.shape.innerRank, ((eqIndex F.tau1 (row + 1 + a)).into : E) *
        (responseForm P.fold key a).scan (α.into : E)
          (fun o => (o.map move).elim 0 (eqAt (r2.map Ext.into))) := by
  unfold responseLifts
  rw [sum_flatMap]
  simp only [List.map_map, Function.comp_def, sum_finRange]
  simp only [Form.scan_response, Fintype.sum_prod_type, keyed, shared_flatOf, ← Nat.cast_pow,
    aeval_rep_mul_natCast]
  simp only [Finset.mul_sum, ← Finset.sum_neg_distrib]
  rw [sum_comm_in3]
  refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun u _ => ?_
  rw [Lift.ofTerm_value_into α r2 bits top _ short innerPow le degree
    (fun r member => by
      obtain ⟨f, _, rfl⟩ := List.mem_map.mp member
      exact Runs.main_of shift bounded
        (run_of_start (s := P.shape.responseCode P.fold ((x, u), f) * P.shape.innerDegree)
          (fun k small => Shape.responseCell_eq P.fold _ small) (by rw [one_mul, innerPow]))
        (fun j x read => responseCell_lt_chainStart P.opening P.fold _ read))]
  show (((List.finRange P.fold.responseDigits).map fun f =>
      (natFq (P.fold.responseBase ^ f.val), moved move (P.shape.responseCell P.fold ((x, u), f)))).map
        fun r => algebraMap Zq E r.1.toZq * ∑ j ∈ Finset.range P.shape.innerDegree,
          (α.into : E) ^ j * (r.2 j).elim 0 (eqAt (r2.map Ext.into))).sum *
      (((List.finRange P.shape.innerRank).map fun a =>
        (-eqIndex F.tau1 (row + 1 + a.val), viewEntry F.setup P.shape.innerDegree
          (P.shape.blockLength * P.shape.sourceDigits) a.val (P.shape.innerSetupCode (x, u)).val)).map
            fun m => (m.1.into : E) *
              Polynomial.aeval (α.into : E) (rep (Poly.toRq P.shape.innerDegree m.2))).sum = _
  rw [reads_value, List.map_map, sum_finRange, Finset.sum_mul]
  refine Finset.sum_congr rfl fun f _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun a _ => ?_
  simp only [Function.comp_apply, Ext.into_neg, moved]
  rw [Finset.mul_sum, Finset.sum_mul]
  refine Finset.sum_congr rfl fun j _ => ?_
  ring

/-- The images of a group are the left sides of the rows (104), each with the coefficient
`ẽq(τ_1, row + 1 + a)`, for the challenges `ι(c_i)` in `R_{q,d_A}`. -/
theorem imageLifts_value (short : bits ≤ r2.length)
    (outerPow : 2 ^ Nat.log 2 P.shape.outerDegree = P.shape.outerDegree)
    (le : Nat.log 2 P.shape.outerDegree ≤ bits) (degree : P.shape.innerDegree ≤ top)
    (shift : ∀ x c, x + c < P.opening.chainStart P.fold → move (x + c) = move x + c)
    (bounded : ∀ x < P.opening.cells P.fold, move x < 2 ^ r2.length)
    (challenge : Fin P.shape.blocks → Rq P.opening.carrierDegree)
    (ambiented : ∀ i : Fin P.shape.blocks, Poly.toRq P.shape.innerDegree
      (P.ambient.getD i.val (Poly.zero P.shape.innerDegree)) =
        embed P.opening.stride P.opening.stride_tower (challenge i)) :
    ((imageLifts F move P row).map fun l => ((l.value (Tables.of α r2 bits top)).into : E)).sum =
      ∑ a : Fin P.shape.innerRank, ((eqIndex F.tau1 (row + 1 + a)).into : E) *
        (imageForm P.opening P.fold challenge a).scan (α.into : E)
          (fun o => (o.map move).elim 0 (eqAt (r2.map Ext.into))) := by
  have ratio : P.shape.innerDegree / 2 ^ Nat.log 2 P.shape.outerDegree = P.shape.ratio := by
    rw [outerPow]
    rfl
  have divides : 2 ^ Nat.log 2 P.shape.outerDegree ∣ P.shape.innerDegree :=
    (dvd_of_eq outerPow).trans P.shape.divides
  unfold imageLifts
  rw [sum_flatMap]
  simp only [List.map_map, Function.comp_def, sum_finRange]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [imageForm_scan, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Lift.value_into α r2 bits top _ short le divides degree
    (fun r member => by
      obtain ⟨u, _, rfl⟩ := List.mem_map.mp member
      refine Runs.main_of shift bounded (fun y small c inside => ?_)
        (fun j x read => innerCell_lt_chainStart P.opening P.fold i a u read)
      rw [ratio] at small
      rw [outerPow] at inside ⊢
      rw [OpeningShape.innerCell_eq P.opening P.fold i a u small inside,
        show y * P.shape.outerDegree = y * P.shape.outerDegree + 0 from rfl,
        OpeningShape.innerCell_eq P.opening P.fold i a u small (NeZero.pos _)]
      exact congrArg some (by rw [Nat.add_zero]))]
  rw [reads_value]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, ambiented, moved]
  rw [Finset.mul_sum, Finset.sum_mul]
  refine Finset.sum_congr rfl fun u _ => ?_
  ring

omit [Degree2 E] in
theorem sum_comm_in4 {M : Type} [AddCommMonoid M] {α β γ δ ε : Type} [Fintype α] [Fintype β]
    [Fintype γ] [Fintype δ] [Fintype ε] (g : α → β → γ → δ → ε → M) :
    ∑ a, ∑ x, ∑ u, ∑ f, ∑ e, g a x u f e = ∑ x, ∑ u, ∑ f, ∑ e, ∑ a, g a x u f e := by
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun x _ => sum_comm_in3 _

/-- The lifts of `B` of a group are the rows (105) of `B` on the slices of `t̂`, each with the
coefficient `ẽq(τ_1, row + 1 + n_A + j)`. -/
theorem outerLifts_value (short : bits ≤ r2.length)
    (outerPow : 2 ^ Nat.log 2 P.shape.outerDegree = P.shape.outerDegree)
    (le : Nat.log 2 P.shape.outerDegree ≤ bits) (degree : P.shape.outerDegree ≤ top)
    (shift : ∀ x c, x + c < P.opening.chainStart P.fold → move (x + c) = move x + c)
    (bounded : ∀ x < P.opening.cells P.fold, move x < 2 ^ r2.length) (key : Key P.shape)
    (keyed : key.outer = Akita.Setup.shared (flatOf F.setup) P.shape.outerDegree P.shape.outerRank
      _ P.shape.outerSetupCode) :
    ((outerLifts F move P row).map fun l => ((l.value (Tables.of α r2 bits top)).into : E)).sum =
      ∑ j : Fin (P.shape.slices * P.shape.outerRank),
        ((eqIndex F.tau1 (row + 1 + P.shape.innerRank + j)).into : E) *
          (outerForm P.opening P.fold key j).scan (α.into : E)
            (fun o => (o.map move).elim 0 (eqAt (r2.map Ext.into))) := by
  rw [← finProdFinEquiv.sum_comp, Fintype.sum_prod_type]
  simp only [outerForm, Form.scan_ofMatrix, Equiv.symm_apply_apply, finProdFinEquiv_apply_val,
    Fintype.sum_prod_type, keyed, shared_flatOf, Finset.mul_sum]
  unfold outerLifts
  simp only [sum_flatMap, List.map_map, Function.comp_def, sum_finRange]
  refine Finset.sum_congr rfl fun s _ => ?_
  rw [sum_comm_in4]
  refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun a _ =>
    Finset.sum_congr rfl fun y _ => Finset.sum_congr rfl fun u _ => ?_
  rw [Lift.ofTerm_value_into α r2 bits top _ short outerPow le degree
    (fun r member => by
      rw [List.mem_singleton] at member
      subst member
      refine Runs.main_of shift bounded ?_
        (fun j x read => sliceCell_lt_chainStart P.opening P.fold s _ read)
      by_cases inside : P.shape.sliceStart s + p < P.shape.sliceStart ((s : ℕ) + 1)
      · refine run_of_start (s := P.opening.innerStart P.fold + P.shape.innerCode (⟨P.shape.sliceStart s + p,
          inside.trans_le (P.shape.sliceStart_le (Nat.succ_le_of_lt s.isLt))⟩, (a, y, u)) *
            P.shape.outerDegree) (d := P.shape.outerDegree) (fun k small => ?_) (by rw [one_mul, outerPow])
        unfold OpeningShape.sliceCell
        rw [dite_eq_left inside]
        exact OpeningShape.outerCell_eq P.opening P.fold _ _ small
      · exact run_of_none fun k => by
          unfold OpeningShape.sliceCell
          exact dite_eq_right inside)]
  rw [reads_single, List.map_map, sum_finRange, Finset.mul_sum]
  refine Finset.sum_congr rfl fun r _ => ?_
  simp only [Function.comp_apply, moved]
  rw [show (s : ℕ) * P.shape.outerRank + r = r + P.shape.outerRank * s by ring, Finset.sum_mul]
  refine Finset.sum_congr rfl fun k _ => ?_
  ring

/-- The cell `start + p` of the entry `p < count` of a segment of digits, and none after it. -/
def segCell (start count p : ℕ) : Option ℕ := if p < count then some (start + p) else none

theorem segCell_run {start count b d : ℕ} (hb : d ∣ b) (hN : d ∣ count) {c : ℕ} (small : c < d) :
    segCell start count (b + c) = (segCell start count b).map (· + c) := by
  unfold segCell
  by_cases inside : b < count
  · rw [ite_eq_left (add_lt_of_dvd hb hN inside small), ite_eq_left inside]
    exact congrArg some (Nat.add_assoc _ _ _).symm
  · rw [ite_eq_right (by omega), ite_eq_right inside]
    rfl

/-- A read of `d = 2^m` entries from a multiple of `d` of a segment of digits whose length is a
multiple of `d` is a run of cells. -/
theorem Runs.seg {m d start count b bound : ℕ} {move : ℕ → ℕ} {f : ℕ → Option ℕ}
    (pow : 2 ^ m = d) (form : ∀ k, f k = segCell start count (b + k)) (hb : d ∣ b)
    (hN : d ∣ count)
    (shift : ∀ x c, start ≤ x → x + c < start + count → move (x + c) = move x + c)
    (bounded : ∀ x, start ≤ x → x < start + count → move x < bound) :
    Runs m 1 bound (moved move f) := by
  refine Runs.moved_of (lo := start) (hi := start + count) shift (fun y small c inside => ?_)
    (fun j x read => ?_) (fun j x read => ?_)
  · obtain rfl : y = 0 := by omega
    rw [zero_mul, zero_add, form, form, Nat.add_zero]
    exact segCell_run hb hN (pow ▸ inside)
  · rw [form] at read
    unfold segCell at read
    split at read
    · cases read
      omega
    · cases read
  · rw [form] at read
    unfold segCell at read
    split at read
    · cases read
      exact bounded _ (by omega) (by omega)
    · cases read

variable {move}

/-- The recompositions `G·ξ̂` of the `n` image rows of a path from the row `base`, with the sign
of the code. -/
theorem recomposeLifts_value {d : ℕ} [NeZero d] (n start count base : ℕ) (coef : ℕ → Ext)
    (short : bits ≤ r2.length) (pow : 2 ^ Nat.log 2 d = d) (le : Nat.log 2 d ≤ bits)
    (degree : d ≤ top) (divides : d ∣ count)
    (shift : ∀ x c, start ≤ x → x + c < start + count → move (x + c) = move x + c)
    (bounded : ∀ x, start ≤ x → x < start + count → move x < 2 ^ r2.length) :
    ((List.range n).map fun j => (((Lift.ofTerm
      ({ degree := d, reads := recomposeReads d n (moved move (segCell start count)) j,
         muls := [(-coef (base + j), unitRing d)] } : Term)).value
          (Tables.of α r2 bits top)).into : E)).sum =
      -∑ j : Fin n, ((coef (base + j)).into : E) *
        (Form.recompose d n (segCell start count) j).scan (α.into : E)
          (fun o => (o.map move).elim 0 (eqAt (r2.map Ext.into))) := by
  rw [sum_range_list, ← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [Lift.ofTerm_value_into α r2 bits top _ short pow le degree
    (fun r member => by
      obtain ⟨u, _, rfl⟩ := List.mem_map.mp member
      exact Runs.seg pow (b := j * d + n * d * u) (fun k => by
        show segCell start count (j * d + k + n * d * u) = _
        rw [show j * d + k + n * d * u = j * d + n * d * u + k by ring])
        (Nat.dvd_add (Dvd.intro_left _ rfl) (Dvd.intro_left (n * u) (by ring))) divides shift
          bounded)]
  simp only [recomposeReads, List.map_map, sum_range_list, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, add_zero, toRq_unitRing, aeval_rep_one, mul_one, Ext.into_neg,
    recompose_scan, Function.comp_apply, natFq, Fq.toZq_ofNat, map_natCast, moved]
  ring

/-- The columns of a compression map `K` that is a view of the setup are the rows `K·x` with the
coefficients `ẽq(τ_1, base + r)`. -/
theorem columnLifts_value (setup : Array Fq) {d : ℕ} [NeZero d] (rank width start count base : ℕ)
    (coef : ℕ → Ext) (K : Matrix (Fin rank) (Fin width) (Rq d))
    (keyed : K = Akita.Setup.view ⟨d, rank, width, false, 1⟩ (flatOf setup))
    (short : bits ≤ r2.length) (pow : 2 ^ Nat.log 2 d = d) (le : Nat.log 2 d ≤ bits)
    (degree : d ≤ top) (divides : d ∣ count)
    (shift : ∀ x c, start ≤ x → x + c < start + count → move (x + c) = move x + c)
    (bounded : ∀ x, start ≤ x → x < start + count → move x < 2 ^ r2.length) :
    ((List.range width).map fun col => (((Lift.ofTerm
      ({ degree := d, reads := [(Fq.one, fun k => moved move (segCell start count) (col * d + k))],
         muls := (List.range rank).map fun r => (coef (base + r), viewEntry setup d width r col) } :
          Term)).value (Tables.of α r2 bits top)).into : E)).sum =
      ∑ r : Fin rank, ((coef (base + r)).into : E) *
        (Form.ofMatrix K (fun col k => segCell start count (col * d + k)) r).scan (α.into : E)
          (fun o => (o.map move).elim 0 (eqAt (r2.map Ext.into))) := by
  rw [sum_range_list]
  simp only [Form.scan_ofMatrix, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun col _ => ?_
  rw [Lift.ofTerm_value_into α r2 bits top _ short pow le degree
    (fun r member => by
      rw [List.mem_singleton] at member
      subst member
      exact Runs.seg pow (b := col * d) (fun k => rfl) (Dvd.intro_left _ rfl) divides shift
        bounded)]
  rw [reads_single, List.map_map, sum_range_list, Finset.mul_sum]
  refine Finset.sum_congr rfl fun r _ => ?_
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun k _ => ?_
  simp only [Function.comp_apply, keyed, view_flatOf, moved]
  ring

omit [Degree2 E] in
theorem imageRow_scan {d n : ℕ} [NeZero d] (chain : Option ChainShape) (image : Fin n → Akita.Form d)
    (first : ℕ → Option ℕ) (payload : Payload d n chain) (j : Fin n) (a : E) (φ : Option ℕ → E) :
    (imageRow chain image first payload j).scan a φ = (image j).scan a φ -
      (if chain.isSome then (Form.recompose d n first j).scan a φ else 0) := by
  cases chain with
  | none =>
      rw [imageRow, Form.scan_row, Option.isSome_none, ite_eq_right (by decide), sub_zero]
  | some c =>
      rw [imageRow, Form.scan_row, Form.scan_sub, Option.isSome_some, ite_eq_left rfl]

/-- The parts of a compression path (Akita (47)) from the digit segments at `s_1` and `s_2`: the
recompositions in the `n` image rows from the row `image`, and the rows of `K_1` from `first`
and of `K_2` from `second`. -/
theorem chainLifts_into (setup : Array Fq) (coef : ℕ → Ext) (chain : Option ChainShape)
    (key : PathKey chain) (shared : PathShared (flatOf setup) chain key) {d : ℕ} [NeZero d]
    (n s1 s2 : ℕ) (payload : Payload d n chain) (image first second : ℕ)
    (short : bits ≤ r2.length) (pow : 2 ^ Nat.log 2 d = d) (le : Nat.log 2 d ≤ bits)
    (degree : d ≤ top) (firstPow : 2 ^ Nat.log 2 (firstDegree chain) = firstDegree chain)
    (secondPow : 2 ^ Nat.log 2 (secondDegree chain) = secondDegree chain)
    (firstLe : Nat.log 2 (firstDegree chain) ≤ bits)
    (secondLe : Nat.log 2 (secondDegree chain) ≤ bits) (firstTop : firstDegree chain ≤ top)
    (secondTop : secondDegree chain ≤ top) (imageDivides : d ∣ firstCells chain)
    (firstDivides : firstDegree chain ∣ secondCells chain)
    (shiftFirst : ∀ x c, s1 ≤ x → x + c < s1 + firstCells chain → move (x + c) = move x + c)
    (shiftSecond : ∀ x c, s2 ≤ x → x + c < s2 + secondCells chain → move (x + c) = move x + c)
    (boundFirst : ∀ x, s1 ≤ x → x < s1 + firstCells chain → move x < 2 ^ r2.length)
    (boundSecond : ∀ x, s2 ≤ x → x < s2 + secondCells chain → move x < 2 ^ r2.length) :
    (((chainTerms setup coef chain d n (moved move (segCell s1 (firstCells chain)))
        (moved move (segCell s2 (secondCells chain))) image first second).map
          (·.map Lift.ofTerm)).map fun p => ((liftsValue (Tables.of α r2 bits top) p).into : E)).sum =
      -(∑ j : Fin n, ((coef (image + j)).into : E) * (if chain.isSome then
          (Form.recompose d n (segCell s1 (firstCells chain)) j).scan (α.into : E)
            (fun o => (o.map move).elim 0 (eqAt (r2.map Ext.into))) else 0)) +
      (∑ r : Fin (firstRank chain), ((coef (first + r)).into : E) *
          (firstRow chain key (segCell s1 (firstCells chain)) (segCell s2 (secondCells chain))
            r).scan (α.into : E) (fun o => (o.map move).elim 0 (eqAt (r2.map Ext.into))) +
        ∑ r : Fin (secondRank chain), ((coef (second + r)).into : E) *
          (secondRow chain key (segCell s2 (secondCells chain)) payload r).scan (α.into : E)
            (fun o => (o.map move).elim 0 (eqAt (r2.map Ext.into)))) := by
  cases chain with
  | none =>
      simp only [chainTerms, List.map_nil, List.sum_nil, Option.isSome_none, Bool.false_eq_true,
        ite_false, mul_zero, Finset.sum_const_zero, neg_zero, zero_add]
      rw [Finset.sum_eq_zero fun r _ => r.elim0, Finset.sum_eq_zero fun r _ => r.elim0, add_zero]
  | some c =>
      obtain ⟨firstShared, secondShared⟩ := shared
      simp only [chainTerms, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero,
        liftsValue, sumList_into, List.map_map, Function.comp_def]
      rw [recomposeLifts_value α r2 bits top n s1 _ image coef short pow le degree imageDivides
          shiftFirst boundFirst,
        columnLifts_value (d := c.firstDegree) α r2 bits top setup c.firstRank c.firstWidth s1
          (firstCells (some c)) first
          coef _ firstShared short firstPow firstLe firstTop (Dvd.intro_left _ rfl) shiftFirst
          boundFirst,
        recomposeLifts_value (d := c.firstDegree) α r2 bits top c.firstRank s2 _ first coef short
          firstPow firstLe
          firstTop firstDivides shiftSecond boundSecond,
        columnLifts_value (d := c.secondDegree) α r2 bits top setup c.secondRank c.secondWidth s2
          (secondCells (some c))
          second coef _ secondShared short secondPow secondLe secondTop (Dvd.intro_left _ rfl)
          shiftSecond boundSecond]
      simp only [Option.isSome_some, ite_true, firstRow, secondRow, firstRank, secondRank,
        Form.scan_row, Form.scan_sub, mul_sub, Finset.sum_sub_distrib]
      abel

/-- The columns of a group in `D` are the rows (106) of `D` on `ê`, each with the coefficient
`ẽq(τ_1, dRow + j)`, for the columns of the one group from 0. -/
theorem openLifts_value (dRow : ℕ) (short : bits ≤ r2.length)
    (openPow : 2 ^ Nat.log 2 P.opening.openDegree = P.opening.openDegree)
    (le : Nat.log 2 P.opening.openDegree ≤ bits) (degree : P.opening.openDegree ≤ top)
    (shift : ∀ x c, x + c < P.opening.chainStart P.fold → move (x + c) = move x + c)
    (bounded : ∀ x < P.opening.cells P.fold, move x < 2 ^ r2.length) (openKey : OpeningKey P.opening)
    (keyed : openKey.matrix = Akita.Setup.shared (flatOf F.setup) P.opening.openDegree
      P.opening.openRank _ P.opening.openSetupCode) :
    ((openLifts F move P dRow 0 (P.shape.blocks * (P.opening.subcolumns * P.shape.openingDigits))).map
      fun l => ((l.value (Tables.of α r2 bits top)).into : E)).sum =
      ∑ j : Fin (1 * P.opening.openRank), ((eqIndex F.tau1 (dRow + j)).into : E) *
        (openForm P.opening P.fold openKey j).scan (α.into : E)
          (fun o => (o.map move).elim 0 (eqAt (r2.map Ext.into))) := by
  rw [← finProdFinEquiv.sum_comp, Fintype.sum_prod_type, Fin.sum_univ_one]
  simp only [openForm, Form.scan_ofMatrix, Equiv.symm_apply_apply, finProdFinEquiv_apply_val,
    Fin.val_eq_zero, mul_zero, add_zero, Fintype.sum_prod_type, keyed, shared_flatOf,
    Finset.mul_sum]
  unfold openLifts
  simp only [sum_flatMap, List.map_map, Function.comp_def, sum_finRange]
  rw [sum_comm_in3]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun y _ =>
    Finset.sum_congr rfl fun u _ => ?_
  rw [Lift.ofTerm_value_into α r2 bits top _ short openPow le degree
    (fun r member => by
      rw [List.mem_singleton] at member
      subst member
      exact Runs.main_of shift bounded
        (run_of_start (s := P.shape.responseCells P.fold + P.opening.partialCode (i, y, u) *
          P.opening.openDegree) (d := P.opening.openDegree)
          (fun k small => OpeningShape.partialCell_eq P.opening P.fold _ small)
          (by rw [one_mul, openPow]))
        (fun j x read => partialCell_lt_chainStart P.opening P.fold _ read))]
  rw [reads_single, List.map_map, sum_finRange, Finset.mul_sum]
  refine Finset.sum_congr rfl fun r _ => ?_
  simp only [Function.comp_apply, moved, zero_add]
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun k _ => ?_
  ring

/-- The scalar row (100) of a group with the claim coefficient 1 at `r_2`: `scalarPart` is the
value at `r_2` of `scalarWeight` for the weights `ẽq(r_blk, i)` and `ẽq(r_tail, j)` of the opening
point and the basis `(1, u)`, when `move` keeps the cells of `ê`. -/
theorem scalarPart_into (short : bits ≤ r2.length)
    (carrierPow : 2 ^ Nat.log 2 P.opening.carrierDegree = P.opening.carrierDegree)
    (openPow : 2 ^ Nat.log 2 P.opening.openDegree = P.opening.openDegree)
    (le : Nat.log 2 P.opening.carrierDegree ≤ bits)
    (fixed : ∀ x < P.opening.chainStart P.fold, move x = x)
    (fits : P.opening.cells P.fold ≤ 2 ^ r2.length)
    (β : Module.Basis (Fin P.opening.planes) Zq E)
    (based : ∀ t, β t = (extBasis.getD t Ext.zero).into) (point : Point P.opening E)
    (blocked : ∀ i, point.block i = eqAt (P.segments.2.2.2.map Ext.into) i)
    (tailed : ∀ j, point.tail j = eqAt (P.segments.2.1.map Ext.into) j)
    (claimed : (P.claim.into : E) = 1) :
    ((scalarPart move P (Tables.of α r2 bits top)).into : E) =
      mle (r2.map Ext.into) (scalarWeight P.opening P.fold β point) := by
  have divides : 2 ^ min (Nat.log 2 P.opening.carrierDegree) (Nat.log 2 P.opening.openDegree) ∣
      P.opening.carrierDegree := (Nat.pow_dvd_pow 2 (min_le_left _ _)).trans (dvd_of_eq carrierPow)
  have opened : 2 ^ min (Nat.log 2 P.opening.carrierDegree) (Nat.log 2 P.opening.openDegree) ∣
      P.opening.openDegree := (Nat.pow_dvd_pow 2 (min_le_right _ _)).trans (dvd_of_eq openPow)
  have total : P.opening.carrierDegree /
      2 ^ min (Nat.log 2 P.opening.carrierDegree) (Nat.log 2 P.opening.openDegree) *
        2 ^ min (Nat.log 2 P.opening.carrierDegree) (Nat.log 2 P.opening.openDegree) =
      P.opening.carrierDegree := Nat.div_mul_cancel divides
  have length : (r2.map (Ext.into (E := E))).length = r2.length := List.length_map _
  rw [mle_eq_cubeSum, cubeSum_eq_sum_range, length]
  unfold scalarWeight
  rw [sum_mul_indicator _ _ _ _ fun entry x read =>
    (P.opening.partialPosition_lt P.fold _ _ _ read).trans_le fits]
  simp only [Fintype.sum_prod_type]
  unfold scalarPart
  simp only [Ext.into_mul, claimed, one_mul, sumList_into, sum_finRange, eqIndex_into,
    sum_range_list, Ext.into_scale, natFq, Fq.toZq_ofNat, map_natCast, ← blocked]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Finset.mul_sum, Finset.sum_comm]
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [show (Tables.of α r2 bits top).lowsOf
      (min (Nat.log 2 P.opening.carrierDegree) (Nat.log 2 P.opening.openDegree))
      P.opening.carrierDegree (fun j => eqIndex P.segments.2.1 j) =
    lowSums r2 (min (Nat.log 2 P.opening.carrierDegree) (Nat.log 2 P.opening.openDegree))
      (Array.ofFn (n := P.opening.carrierDegree) fun k => eqIndex P.segments.2.1 k.val) from rfl]
  have below : ∀ (u : Fin P.shape.openingDigits) j x,
      P.opening.partialPosition P.fold i u (t * P.opening.carrierDegree + j) = some x →
        x < P.opening.chainStart P.fold := fun u j x read =>
    (P.opening.partialPosition_lt_innerStart P.fold i u _ read).trans_le (Nat.le_add_right _ _)
  have keep : ∀ (u : Fin P.shape.openingDigits) j,
      ((P.opening.partialPosition P.fold i u (t * P.opening.carrierDegree + j)).map move).elim 0
          (eqAt (r2.map (Ext.into (E := E)))) =
        (P.opening.partialPosition P.fold i u (t * P.opening.carrierDegree + j)).elim 0
          (eqAt (r2.map Ext.into)) := fun u j => by
    cases read : P.opening.partialPosition P.fold i u (t * P.opening.carrierDegree + j) with
    | none => rfl
    | some x =>
        show eqAt _ (move x) = eqAt _ x
        rw [fixed x (below u j x read)]
  conv_rhs => rw [Finset.sum_comm]
  rw [Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun u _ => ?_
  rw [Tables.read_into α r2 bits top _ ((min_le_left _ _).trans le) short le_rfl
    (by rw [total, Array.size_ofFn])
    (Runs.moved_of (lo := 0) (hi := P.opening.chainStart P.fold)
      (fun x c _ high => by rw [fixed _ high, fixed _ (by omega)])
      (OpeningShape.partialPosition_run P.opening P.fold i u t divides opened)
      (fun j x read => ⟨Nat.zero_le _, below u j x read⟩)
      (fun j x read => by
        rw [fixed x (below u j x read)]
        exact (P.opening.partialPosition_lt P.fold i u _ read).trans_le fits)), total,
    Finset.mul_sum, Finset.mul_sum, Finset.mul_sum, Finset.sum_range]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [ofFn_getD (fun k => eqIndex P.segments.2.1 k) j.isLt, eqIndex_into,
    ← tailed, ← based]
  simp only [moved]
  rw [keep, Algebra.smul_def, map_pow, map_natCast, Nat.cast_pow]
  ring

end Values

end NightstreamFPrime.Spec.Compression.Akita.Replay
