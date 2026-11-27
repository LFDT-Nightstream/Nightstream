import NightstreamFPrime.Spec.Compression.Akita.Replay.Fused
import NightstreamFPrime.Spec.Compression.Akita.Extract.LiftSource

/-! The relation weight at `r_2` of one fold with coefficient packing on the quotient-lift route,
without the table of the witness cells (Akita revision `d50f700`:
`RelationMatrixEvaluator::eval_flat_at_point` and `eval_flat_at_point_with_deferred_setup` of
`akita-verifier/src/relation/evaluation.rs`, `PackingStage2::weight_at_point` of
`stages/stage2.rs` with `prepare_compact_factors` of `coefficient_packing_relation/compact.rs`,
`build_compression_relation_weights` of `akita-types/src/proof/compression_relation_weights.rs`,
and the witness layout of `akita-params/src/witness.rs` and `witness/tail.rs`; ePrint 2026/1983,
Figure 5, (100), (102), (104) to (106), (47), (76), (111) and (112)).

- `Tables` has `ẽq(r_2, ·)` and its two factors for chunks of `2^m` cells. `Tables.read` is
  `Σ_{k<n·2^m} w(k)·ẽq(r_2, addr k)` for an address map that is a run of cells in each chunk: a
  chunk that starts at a multiple of `2^m` costs one product (`Σ_c w(y·2^m + c)·ẽq(r_2[..m], c)`
  times `ẽq(r_2[m..], b/2^m)`). The code checks that each segment starts at a multiple of its
  coefficient block (`canonical_relation_lane_index`).
- A `Lift` is a set of occurrences `(c·s, A, addr)` of ring rows of degree `d` that share one read
  vector, and `Lift.value` is the value at `r_2` of their weights (112):
  `Σ_{(c,A)} c·A(α)·Σ_{(s,addr)} s·Σ_j α^j·ẽq(r_2, addr j)`.
- `PackGroup` is one group of a fold, and `Placement` the cells of the code: the main segments
  `[ẑ | ê | t̂]` of the groups in the order of the list (`root_group_order`), the quotient digits
  of the rows before the compression rows, then for each compression map the digits of `F` of each
  group, of `H`, and the quotient digits of these rows.
- `foldParts` are the rows of the fold in the order of the code: for each group its carrier row
  (102) joined by the basis `(1, u)` (the partials, the contraction and the quotient digits), the
  rows (104) in the sign of the code `Σ_i ι(c_i)·G·t̂_{i,a} − A_a·ẑ` and the rows (105); then
  the rows (106) of `D` with the columns of the groups in the order of the list, the compression
  rows (47), the quotient digits (111) with the weights `−(α^{d_r} + 1)·b^a·α^j`, and the scalar
  row (100) with the claim coefficient of each group.
- `foldFused` is the end check of the fused sum-check of the level. With the setup offloaded to
  Stage 3 (the root of `key30.aks`), the rows of the setup matrices are replaced by
  `powerScale(α, r_2[..m])·σ` (`eval_flat_at_point_with_deferred_setup`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

/-- The sums `Σ_{i<n} a_i·c0(p_i)` and `Σ_{i<n} a_i·c1(p_i)` of the words, not reduced. -/
def dotFrom (p : Array Ext) (a : Array Fq) : ℕ → ℕ × ℕ
  | 0 => (0, 0)
  | n + 1 =>
      if (Poly.get a n).word.toNat = 0 then dotFrom p a n
      else ((dotFrom p a n).1 + (Poly.get a n).word.toNat * (p.getD n Ext.zero).c0.word.toNat,
        (dotFrom p a n).2 + (Poly.get a n).word.toNat * (p.getD n Ext.zero).c1.word.toNat)

/-- `Σ_{i<d} a_i·p_i` for `a` over `Z_q` and `p` over `E`: the sums of the products of the words,
reduced once. -/
def dotFq (p : Array Ext) (a : Array Fq) (d : ℕ) : Ext :=
  let sums := dotFrom p a d
  ⟨Fq.ofNat sums.1, Fq.ofNat sums.2⟩

/-- `α^0, …, α^{n−1}`. -/
def powersOf (α : Ext) (n : ℕ) : Array Ext := Array.ofFn (n := n) fun k => Ext.pow α k.val

/-- The basis `(1, u)` of `E` over `Z_q` (`canonical_extension_basis`). -/
def extBasis : List Ext := [Ext.one, ⟨Fq.zero, Fq.one⟩]

/-- The low sums `L^{(m)}_y = Σ_{c<2^m} w_{y·2^m + c}·ẽq(r_2[..m], c)` of the table `w = L^{(0)}`,
by `L^{(m+1)}_y = (1 − r_m)·L^{(m)}_{2y} + r_m·L^{(m)}_{2y+1}`. -/
def lowLevel (r2 : List Ext) (w : Array Ext) : ℕ → Array Ext
  | 0 => w
  | m + 1 =>
      let prev := lowLevel r2 w m
      let rm := r2.getD m Ext.zero
      Array.ofFn (n := prev.size / 2) fun y =>
        (Ext.one - rm) * prev.getD (2 * y.val) Ext.zero + rm * prev.getD (2 * y.val + 1) Ext.zero

/-- The low sums of `w` for `m ≤ bits`. -/
def lowSums (r2 : List Ext) (bits : ℕ) (w : Array Ext) : Array (Array Ext) :=
  Array.ofFn (n := bits + 1) fun m => lowLevel r2 w m.val

/-- The tables of one point `r_2` for the runs of at most `2^bits` cells: `ẽq(r_2, ·)`, for each
`m ≤ bits` the table `ẽq(r_2[m..], ·)`, the powers of `α` and their low sums. -/
structure Tables where
  alpha : Ext
  r2 : List Ext
  bits : ℕ
  eq : CellEq
  high : Array CellEq
  powers : Array Ext
  powerLows : Array (Array Ext)

/-- The tables of `r_2` and `α` for the runs of at most `2^bits` cells and the powers below
`top`. -/
def Tables.of (alpha : Ext) (r2 : List Ext) (bits top : ℕ) : Tables :=
  let powers := powersOf alpha top
  ⟨alpha, r2, bits, CellEq.half r2, (Array.range (bits + 1)).map fun m => CellEq.half (r2.drop m),
    powers, lowSums r2 bits powers⟩

namespace Tables

variable (T : Tables)

/-- `α^k`. -/
def pow (k : ℕ) : Ext := T.powers.getD k (Ext.pow T.alpha k)

/-- `Σ_{c<2^m} L^{(0)}_{y·2^m + c}·ẽq(r_2, b + c)` for the run of `2^m` cells from `b`: one product
`L^{(m)}_y·ẽq(r_2[m..], b/2^m)` when `2^m ∣ b`, else the two halves. -/
def run (Ls : Array (Array Ext)) : ℕ → ℕ → ℕ → Ext
  | 0, y, b => (Ls.getD 0 #[]).getD y Ext.zero * T.eq.get b
  | m + 1, y, b =>
      if b % 2 ^ (m + 1) = 0 then
        (Ls.getD (m + 1) #[]).getD y Ext.zero * (T.high.getD (m + 1) T.eq).get (b / 2 ^ (m + 1))
      else run Ls m (2 * y) b + run Ls m (2 * y + 1) (b + 2 ^ m)

/-- `Σ_{y<n} Σ_{c<2^m} L^{(0)}_{y·2^m + c}·ẽq(r_2, addr(y·2^m) + c)` for an address map that is a run
of cells in each chunk of `2^m`. -/
def read (Ls : Array (Array Ext)) (m n : ℕ) (addr : ℕ → Option ℕ) : Ext :=
  let size := 2 ^ m
  sumList (List.range n) fun y => match addr (y * size) with
    | none => Ext.zero
    | some b => T.run Ls m y b

/-- `Σ_{j<d} α^j·ẽq(r_2, addr j)` in runs of `2^m` cells. -/
def liftRead (m d : ℕ) (addr : ℕ → Option ℕ) : Ext := T.read T.powerLows m (d / 2 ^ m) addr

/-- The low sums of the weights `w_k` for `k < n`. -/
def lowsOf (m n : ℕ) (w : ℕ → Ext) : Array (Array Ext) :=
  lowSums T.r2 m (Array.ofFn (n := n) fun k => w k.val)

end Tables

/-- Occurrences of ring rows of degree `degree` that share the read vector `reads`, read in
chunks of `2^bits` cells. -/
structure Lift where
  degree : ℕ
  bits : ℕ
  muls : List (Ext × Array Fq)
  reads : List Read

/-- `Σ_{(c,A)} c·A(α)·Σ_{(s,addr)} s·Σ_j α^j·ẽq(r_2, addr j)`. -/
def Lift.value (T : Tables) (t : Lift) : Ext :=
  (sumList t.reads fun r => Ext.scale r.1 (T.liftRead t.bits t.degree r.2)) *
    sumList t.muls fun m => m.1 * dotFq T.powers m.2 t.degree

/-- A term whose reads are runs of cells of its degree. -/
def Lift.ofTerm (t : Term) : Lift := ⟨t.degree, Nat.log 2 t.degree, t.muls, t.reads⟩

/-- The sum of the values of the lift terms. -/
def liftsValue (T : Tables) (terms : List Lift) : Ext := sumList terms (·.value T)

/-- One group of a packing fold: its shapes, its fold challenges in the subring `R_{q,d_car}` and
in `R_{q,d_A}`, its opening point `(r_pack, r_tail, r_pos, r_blk)` and the claim coefficient of
its claim. -/
structure PackGroup where
  shape : Shape
  opening : OpeningShape shape
  fold : Descriptor
  challenges : List (Array Fq)
  ambient : List (Array Fq)
  point : List Ext
  claim : Ext

namespace PackGroup

variable (P : PackGroup)

/-- The segments `r_pack`, `r_tail`, `r_pos` and `r_blk` of the opening point. -/
def segments : List Ext × List Ext × List Ext × List Ext :=
  let pack := Nat.log 2 P.opening.stride
  let tail := Nat.log 2 P.opening.carrierDegree
  let position := Nat.log 2 P.shape.blockLength
  (P.point.take pack, (P.point.drop pack).take tail, (P.point.drop (pack + tail)).take position,
    P.point.drop (pack + tail + position))

/-- The cells `[ẑ | ê | t̂]` of the group. -/
def mainCells : ℕ := P.opening.chainStart P.fold

/-- The rows of the group before the rows of `D`: the carrier row, `n_A` and `slices·n_B`. -/
def rowCount : ℕ := 1 + P.shape.innerRank + P.shape.slices * P.shape.outerRank

/-- The width of the columns of the group in `D`. -/
def openWidth : ℕ := P.shape.blocks * (P.opening.subcolumns * P.shape.openingDigits)

/-- The sum of the degrees of the rows of the group before the rows of `D`. -/
def mainDegrees : ℕ :=
  P.opening.planes * P.opening.carrierDegree + (P.shape.innerRank * P.shape.innerDegree +
    P.shape.slices * P.shape.outerRank * P.shape.outerDegree)

end PackGroup

/-- `Σ` of `f` over the elements before index `k` of a list. -/
def before {α : Type} (l : List α) (f : α → ℕ) (k : ℕ) : ℕ := ((l.take k).map f).sum

/-- `Σ` of `f` over the elements of a list whose rank in `rank` is below the rank of index `k`;
without ranks, `before`. -/
def beforeIn {α : Type} (rank : Option (List ℕ)) (l : List α) (f : α → ℕ) (k : ℕ) : ℕ :=
  match rank with
  | none => before l f k
  | some r => (l.zipIdx.filterMap fun (x, j) =>
      if r.getD j 0 < r.getD k 0 then some (f x) else none).sum

/-- The orders of the cells of the groups of a fold: by default the order of the list of groups
(the code). `main`, `columns` and `spans` give a rank to each group for its main segment, its
columns of `D` and its spans of `F`. -/
structure CellOrder where
  main : Option (List ℕ) := none
  columns : Option (List ℕ) := none
  spans : Option (List ℕ) := none

/-- The cells of the witness of the code for the groups `gs` in this order and `digits` quotient
digits (`WitnessLayout::new`, `tail::materialize`): the main segments, the quotient digits of the
rows before the compression rows, then for each compression map the digit spans of `F` of each
group and of `H`, and the quotient digits of the rows of this map. -/
structure Placement where
  main : List ℕ
  mainTotal : ℕ
  quotientMain : ℕ
  firstSpan : ℕ
  openFirst : ℕ
  quotientFirst : ℕ
  secondSpan : ℕ
  openSecond : ℕ
  quotientSecond : ℕ

def Placement.of (gs : List PackGroup) (D : OpenPath) (digits : ℕ) : Placement :=
  let mainTotal := (gs.map (·.mainCells)).sum
  let quotientMain := mainTotal
  let mainQ := digits * ((gs.map (·.mainDegrees)).sum + D.rank * D.degree)
  let firstSpan := quotientMain + mainQ
  let openFirst := firstSpan + (gs.map fun P => firstCells P.shape.chain).sum
  let quotientFirst := openFirst + firstCells D.chain
  let firstQ := digits * ((gs.map fun P => firstRank P.shape.chain * firstDegree P.shape.chain).sum +
    firstRank D.chain * firstDegree D.chain)
  let secondSpan := quotientFirst + firstQ
  let openSecond := secondSpan + (gs.map fun P => secondCells P.shape.chain).sum
  let quotientSecond := openSecond + secondCells D.chain
  ⟨gs.map (·.mainCells), mainTotal, quotientMain, firstSpan, openFirst, quotientFirst, secondSpan,
    openSecond, quotientSecond⟩

/-- The cell of the code of the position `x` of the flat witness `Witness.flat` of group `k` of
`gs` (`Batch.relocate` with the groups in the order of the list and the quotient segments of
Akita (76)). -/
def Placement.move (L : Placement) (gs : List PackGroup) (k : ℕ) (C : CellOrder := {}) :
    ℕ → ℕ :=
  match gs[k]? with
  | none => id
  | some P =>
      let O := P.opening
      let c0 := O.chainStart P.fold
      let c1 := O.openFirstStart P.fold
      let c2 := O.secondStart P.fold
      let c3 := O.openSecondStart P.fold
      let main := beforeIn C.main gs (·.mainCells) k
      let first := L.firstSpan + beforeIn C.spans gs (fun Q => firstCells Q.shape.chain) k
      let second := L.secondSpan + beforeIn C.spans gs (fun Q => secondCells Q.shape.chain) k
      fun x =>
        if x < c0 then main + x
        else if x < c1 then first + (x - c0)
        else if x < c2 then L.openFirst + (x - c1)
        else if x < c3 then second + (x - c2)
        else L.openSecond + (x - c3)

/-- The read `(s, addr)` with the address `addr` moved by `move`. -/
def moved (move : ℕ → ℕ) (f : ℕ → Option ℕ) : ℕ → Option ℕ := fun j => (f j).map move

/-- The order of the cells of the quotient digits of the carrier row: Akita (348), the code
(`WitnessQuotientRowLayout::coefficient_index`) and `FlatRelation.quotientCell` have digit, plane,
coefficient. `planeFirst` is the order plane, digit, coefficient, which a test uses as a changed
input. -/
inductive PackingOrder
  | code
  | planeFirst
deriving DecidableEq, Repr

/-- The cell of digit `a` of coefficient `j` of plane `t` of a carrier quotient from `start`, with
`planes` planes of degree `d` and `digits` digits. -/
def PackingOrder.cell (order : PackingOrder) (start planes digits d a t j : ℕ) : ℕ :=
  match order with
  | .code => start + a * (planes * d) + t * d + j
  | .planeFirst => start + t * (digits * d) + a * d + j

/-- The inputs of the weight of a fold that do not depend on a group. -/
structure FoldInputs where
  setup : Array Fq
  tau1 : List Ext
  digits : ℕ
  low : ℕ
  base : ℕ
  path : OpenPath
  order : PackingOrder := .code
  cells : CellOrder := {}

/-- The reads of the `digits` digits of base `b` of the coefficients of a quotient of degree `d`
whose digit `a` starts at the cell `start a`. -/
def quotientReads (b digits d : ℕ) (start : ℕ → ℕ) : List Read :=
  (List.range digits).map fun a => (natFq (b ^ a), fun j => if j < d then some (start a + j) else none)

/-- The quotient term of a ring row of degree `d` with the coefficient `c` and the digits from
`start`, `digits·d` cells in the order digit, coefficient. -/
def quotientLift (b digits d : ℕ) (α c : Ext) (start : ℕ) : Lift :=
  ⟨d, Nat.log 2 d, [(-(c * (Ext.pow α d + Ext.one)), unitRing d)],
    quotientReads b digits d fun a => start + a * d⟩

/-- The degrees of the rows `(degree, count)` from the cell `start` with `digits·d` cells each:
the first cell of each row. -/
def rowStarts (digits : ℕ) : ℕ → List (ℕ × ℕ) → List (ℕ × ℕ)
  | _, [] => []
  | s, (d, n) :: rest =>
      ((List.range n).map fun r => (d, s + r * (digits * d))) ++ rowStarts digits (s + n * (digits * d)) rest

/-- The parts of group `k` of `gs` (its row coefficients from `row`, its columns of `D` from
`column`): the partials and the contraction of its carrier row (102), the rows (104) on `ẑ`, the
images of (104), the rows (105) and the compression rows (47) of its path `F`. -/
def groupParts (F : FoldInputs) (gs : List PackGroup) (L : Placement) (k : ℕ) (P : PackGroup)
    (row first second : ℕ) (T : Tables) : List (List Lift) × Ext :=
  let S := P.shape
  let O := P.opening
  let fold := P.fold
  let move := L.move gs k F.cells
  let at_ := moved move
  let coef (r : ℕ) : Ext := eqIndex F.tau1 r
  let d := S.innerDegree
  let dCar := O.carrierDegree
  let β (t : ℕ) : Ext := extBasis.getD t Ext.zero
  let (pack, _, position, _) := P.segments
  let lift (t : Term) : Lift := Lift.ofTerm t
  let mPartial := min (Nat.log 2 dCar) (Nat.log 2 O.openDegree)
  let partials : List Lift := (List.finRange S.blocks).flatMap fun i =>
    (List.range O.planes).map fun t =>
      { degree := dCar, bits := mPartial
        reads := (List.finRange S.openingDigits).map fun u =>
          (natFq (S.openingBase ^ u.val), at_ fun j => O.partialPosition fold i u (t * dCar + j))
        muls := [(coef row * β t, P.challenges.getD i.val (Poly.zero dCar))] }
  let widthA := S.blockLength * S.sourceDigits
  let response : List Lift := (List.finRange S.blockLength).flatMap fun x =>
    (List.finRange S.sourceDigits).map fun u => lift
      { degree := d
        reads := (List.finRange fold.responseDigits).map fun f =>
          (natFq (fold.responseBase ^ f.val), at_ (S.responseCell fold ((x, u), f)))
        muls := (List.finRange S.innerRank).map fun a =>
          (-coef (row + 1 + a.val), viewEntry F.setup d widthA a.val (S.innerSetupCode (x, u)).val) }
  let images : List Lift := (List.finRange S.blocks).flatMap fun i =>
    (List.finRange S.innerRank).map fun a =>
      { degree := d, bits := Nat.log 2 S.outerDegree
        reads := (List.finRange S.openingDigits).map fun u =>
          (natFq (S.openingBase ^ u.val), at_ (O.innerCell fold i a u))
        muls := [(coef (row + 1 + a.val), P.ambient.getD i.val (Poly.zero d))] }
  let widthB := S.sliceLength * (S.innerRank * (S.ratio * S.openingDigits))
  let outer : List Lift := (List.finRange S.slices).flatMap fun s =>
    (List.finRange S.sliceLength).flatMap fun p => (List.finRange S.innerRank).flatMap fun a =>
      (List.finRange S.ratio).flatMap fun y => (List.finRange S.openingDigits).map fun u => lift
        { degree := S.outerDegree
          reads := [(Fq.one, at_ (O.sliceCell fold s (p, (a, y, u))))]
          muls := (List.finRange S.outerRank).map fun r =>
            (coef (row + 1 + S.innerRank + (s.val * S.outerRank + r.val)),
              viewEntry F.setup S.outerDegree widthB r.val (S.outerSetupCode (p, (a, y, u))).val) }
  let chains := (chainTerms F.setup coef S.chain S.outerDegree (S.slices * S.outerRank)
    (at_ (O.firstCell fold)) (at_ (O.secondCell fold)) (row + 1 + S.innerRank) first second).map
      (·.map lift)
  -- The contraction of (102): the weight `pack_{k mod kh}·α^{k / kh}` of coefficient `k` of each
  -- element of `ẑ`, with the multiplier `−ẽq(τ_1, row)·ẽq(r_pos, x)·b^u·b_z^f`.
  let kh := O.stride
  let m := Nat.log 2 d
  let Lz := T.lowsOf m d fun j => eqIndex pack (j % kh) * T.pow (j / kh)
  let contraction : Ext := -(coef row) * sumList (List.finRange S.blockLength) fun x =>
    eqIndex position x.val * sumList (List.finRange S.sourceDigits) fun u =>
      sumList (List.finRange fold.responseDigits) fun f =>
        Ext.scale (natFq (S.sourceBase ^ u.val * fold.responseBase ^ f.val))
          (T.read Lz m 1 (at_ (S.responseCell fold ((x, u), f))))
  ([partials, response, images, outer] ++ chains, contraction)

/-- The columns of group `k` of `D` (Akita (106)): the rows of `D` from `dRow`, the columns from
`column` in the width `width`. -/
def openParts (F : FoldInputs) (gs : List PackGroup) (L : Placement) (k : ℕ) (P : PackGroup)
    (dRow column width : ℕ) : List Lift :=
  let O := P.opening
  let at_ := moved (L.move gs k F.cells)
  (List.finRange P.shape.blocks).flatMap fun i =>
    (List.finRange O.subcolumns).flatMap fun y => (List.finRange P.shape.openingDigits).map fun u =>
      Lift.ofTerm
        { degree := O.openDegree
          reads := [(Fq.one, at_ (O.partialCell P.fold (i, y, u)))]
          muls := (List.finRange O.openRank).map fun r =>
            (eqIndex F.tau1 (dRow + r.val),
              viewEntry F.setup O.openDegree width r.val (column + (O.openSetupCode (i, y, u)).val)) }

/-- The scalar row (100) of group `P` at the row `N`: `ϑ·Σ_{i,t,u,j} ẽq(r_blk, i)·β_t·b_1^u·
ẽq(r_tail, j)·ẽq(r_2, x)` at the cell `x` of coordinate `t·d_car + j` of digit `u` of partial
`i`. -/
def scalarPart (move : ℕ → ℕ) (P : PackGroup) (T : Tables) : Ext :=
  let O := P.opening
  let dCar := O.carrierDegree
  let (_, tail, _, block) := P.segments
  let m := min (Nat.log 2 dCar) (Nat.log 2 O.openDegree)
  let Ls := T.lowsOf m dCar fun j => eqIndex tail j
  P.claim * sumList (List.finRange P.shape.blocks) fun i =>
    eqIndex block i.val * sumList (List.range O.planes) fun t =>
      extBasis.getD t Ext.zero * sumList (List.finRange P.shape.openingDigits) fun u =>
        Ext.scale (natFq (P.shape.openingBase ^ u.val))
          (T.read Ls m (dCar / 2 ^ m) (moved move fun j => O.partialPosition P.fold i u (t * dCar + j)))

/-- The carrier quotient of a group: the digits of the planes in the code order (digit, plane,
coefficient) from `start`, with the row coefficient `c`. -/
def carrierQuotient (F : FoldInputs) (P : PackGroup) (α c : Ext) (start : ℕ) : List Lift :=
  let O := P.opening
  let dCar := O.carrierDegree
  (List.range O.planes).map fun t =>
    ⟨dCar, Nat.log 2 dCar,
      [(-(c * extBasis.getD t Ext.zero * (Ext.pow α dCar + Ext.one)), unitRing dCar)],
      quotientReads F.base F.digits dCar fun a => F.order.cell start O.planes F.digits dCar a t 0⟩

/-- `powerScale(α, r_2[..m]) = Σ_{c<2^m} α^c·ẽq(r_2[..m], c)` (`common_alpha_evaluation`). -/
def Tables.common (T : Tables) (m : ℕ) : Ext := (T.powerLows.getD m #[]).getD 0 Ext.zero

/-- The value at `r_2` of the relation weight of the fold, in named parts. With the claim `σ` of
Stage 3 (`setupClaim`), the parts of the setup matrices `A`, `B` and `D` are the one part
`powerScale(α, r_2[..low])·σ` (`eval_flat_at_point_with_deferred_setup`). -/
def foldParts (F : FoldInputs) (gs : List PackGroup) (T : Tables) (setupClaim : Option Ext) :
    List (String × Ext) :=
  let D := F.path
  let L := Placement.of gs D F.digits
  let coef (r : ℕ) : Ext := eqIndex F.tau1 r
  let starts := (List.range gs.length).map fun k => before gs (·.rowCount) k
  let dRow := (gs.map (·.rowCount)).sum
  let hFirst := dRow + D.rank + (gs.map fun P => firstRank P.shape.chain).sum
  let hSecond := hFirst + firstRank D.chain + (gs.map fun P => secondRank P.shape.chain).sum
  let openTotal := (gs.map (·.openWidth)).sum
  let groups := gs.zipIdx.map fun (P, k) =>
    groupParts F gs L k P (starts.getD k 0)
      (dRow + D.rank + before gs (fun Q => firstRank Q.shape.chain) k)
      (hFirst + firstRank D.chain + before gs (fun Q => secondRank Q.shape.chain) k) T
  let opened := gs.zipIdx.flatMap fun (P, k) =>
    openParts F gs L k P dRow (beforeIn F.cells.columns gs (·.openWidth) k) openTotal
  let openCell (start : ℕ) (cells : ℕ) (position : ℕ) : Option ℕ :=
    if position < cells then some (start + position) else none
  let openChains := (chainTerms F.setup coef D.chain D.degree (1 * D.rank)
    (openCell L.openFirst (firstCells D.chain)) (openCell L.openSecond (secondCells D.chain))
    dRow hFirst hSecond).map (·.map Lift.ofTerm)
  let carrierStarts := (List.range gs.length).map fun k =>
    L.quotientMain + F.digits * before gs (·.mainDegrees) k
  let groupQuotients := gs.zipIdx.flatMap fun (P, k) =>
    let start := carrierStarts.getD k 0
    let row := starts.getD k 0
    carrierQuotient F P T.alpha (coef row) start ++
      (rowStarts F.digits (start + F.digits * (P.opening.planes * P.opening.carrierDegree))
        [(P.shape.innerDegree, P.shape.innerRank),
          (P.shape.outerDegree, P.shape.slices * P.shape.outerRank)]).zipIdx.map
        fun ((d, s), r) => quotientLift F.base F.digits d T.alpha (coef (row + 1 + r)) s
  let dQuotients := (rowStarts F.digits (L.quotientMain + F.digits * (gs.map (·.mainDegrees)).sum)
    [(D.degree, D.rank)]).zipIdx.map fun ((d, s), r) =>
      quotientLift F.base F.digits d T.alpha (coef (dRow + r)) s
  let firstRows : List (ℕ × ℕ) := (gs.map fun P => (firstDegree P.shape.chain, firstRank P.shape.chain)) ++
    [(firstDegree D.chain, firstRank D.chain)]
  let secondRows : List (ℕ × ℕ) := (gs.map fun P => (secondDegree P.shape.chain, secondRank P.shape.chain)) ++
    [(secondDegree D.chain, secondRank D.chain)]
  let firstQuotients := (rowStarts F.digits L.quotientFirst firstRows).zipIdx.map fun ((d, s), r) =>
    quotientLift F.base F.digits d T.alpha (coef (dRow + D.rank + r)) s
  let secondQuotients := (rowStarts F.digits L.quotientSecond secondRows).zipIdx.map
    fun ((d, s), r) => quotientLift F.base F.digits d T.alpha (coef (hFirst + firstRank D.chain + r)) s
  let scalar := eqIndex F.tau1 (hSecond + secondRank D.chain) *
    sumList gs.zipIdx fun (P, k) => scalarPart (L.move gs k F.cells) P T
  let named (name : String) (terms : List Lift) : String × Ext := (name, liftsValue T terms)
  let setupPart (j : ℕ) : Bool := setupClaim.isSome && (j == 1 || j == 3)
  (groups.zipIdx.flatMap fun ((parts, contraction), k) =>
      ((parts.zipIdx.filterMap fun (p, j) =>
        if setupPart j then none else some (named s!"g{k}.part{j}" p)) ++
        [(s!"g{k}.contraction", contraction)])) ++
    (match setupClaim with
      | some σ => [("setup", T.common F.low * σ)]
      | none => [named "open" opened]) ++
    (openChains.zipIdx.map fun (p, j) => named s!"open.chain{j}" p) ++
    [named "quotient.groups" groupQuotients, named "quotient.open" dQuotients,
      named "quotient.first" firstQuotients, named "quotient.second" secondQuotients,
      ("scalar", scalar)]

/-- The count of setup elements that the matrices of a packing level read, or with the setup
offloaded to Stage 3 (`offloaded`) the count that its compression maps read. -/
def PackShape.setupCount (P : PackShape) (offloaded : Bool) : ℕ :=
  let D := P.path
  if offloaded then (P.slots.map fun g => chainCount g.shape.chain).foldl max (chainCount D.chain)
  else
    let width := (P.slots.map fun g =>
      g.shape.blocks * (g.opening.subcolumns * g.shape.openingDigits)).sum
    (P.slots.map fun g => max (viewCount g.shape.innerRank
        (g.shape.blockLength * g.shape.sourceDigits) g.shape.innerDegree)
      (max (viewCount g.shape.outerRank
          (g.shape.sliceLength * (g.shape.innerRank * (g.shape.ratio * g.shape.openingDigits)))
          g.shape.outerDegree) (chainCount g.shape.chain))).foldl max
      (max (viewCount D.rank width D.degree) (chainCount D.chain))

/-- `Σ_{x ∈ I_bin} ẽq(r_virt, x)·ẽq(r_2, x)` for the compression digits of the witness: the spans of
`F` and `H` of the two maps (`negative_binary_support_intervals`). -/
def foldSupport (L : Placement) (virt eq : ℕ → Ext) : Ext :=
  let span (a b : ℕ) : Ext := sumList (List.range (b - a)) fun k => virt (a + k) * eq (a + k)
  span L.firstSpan L.quotientFirst + span L.secondSpan L.quotientSecond

/-- The values of the replay at the end of the fused sum-check of a packing level. -/
structure PackingValues where
  compressed : Bool
  tau1 : List Ext
  r2 : List Ext
  virtual : List Ext
  alpha : Ext
  output : Ext
  value : Ext
  batch : Ext
  binary : Ext

/-- The end check of the fused sum-check at a packing level with the groups `gs` in the order of
the code, with the weight `weight` at `r_2`. -/
def packingEnd (F : FoldInputs) (gs : List PackGroup) (V : PackingValues) (weight : Ext) :
    FusedEnd :=
  let L := Placement.of gs F.path F.digits
  let eq := (CellEq.half V.r2).get
  let virt := (CellEq.half V.virtual).get
  { output := V.output
    value := V.value
    batch := V.batch
    binary := if V.compressed then V.binary else Ext.zero
    norm := Ext.zero
    weight
    normWeight := Ext.zero
    virt := eqPoints V.virtual V.r2
    support := if V.compressed then foldSupport L virt eq else Ext.zero }

/-- The tables of a level: the chunks of `2^low` cells and the powers of `α` to the largest
degree. -/
def foldTables (F : FoldInputs) (gs : List PackGroup) (V : PackingValues) : Tables :=
  let top := (gs.map fun P => P.shape.innerDegree).foldl max F.path.degree
  Tables.of V.alpha V.r2 (Nat.log 2 top) top

/-- The end check of the fused sum-check at a packing level (`expected_output_claim` with the
relation oracle and the packing oracle). -/
def foldFused (F : FoldInputs) (gs : List PackGroup) (V : PackingValues)
    (setupClaim : Option Ext) : FusedEnd :=
  packingEnd F gs V (sumList (foldParts F gs (foldTables F gs V) setupClaim) (·.2))

end NightstreamFPrime.Spec.Compression.Akita.Replay
