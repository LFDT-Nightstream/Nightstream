import NightstreamFPrime.Spec.Compression.Akita.Replay.Weight
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceNorm
import NightstreamFPrime.Spec.Compression.Akita.SetupInstances

/-! The weight at `r_2` of the rows of one fold with the evaluation-trace opening method, for one
claim in one group (Akita revision `d50f700`: `PreparedDirectRelation` of
`akita-verifier/src/relation/evaluation.rs`, `PreparedEvaluationTrace` of
`relation/evaluation_trace.rs`, `eval_boolean_pair_tensor_families` of the physical L2 oracle of
`stages/stage2.rs`; ePrint 2026/1983, Figure 5, (103) to (106), (101), (91) and (125)).

The rows are those of `traceRelation` (`Extract/TraceSource.lean`) in its order: the row (103),
the `n_A` rows (104), the rows (105) of `B` and (106) of `D`, then the scalar row (101). Each
address is the address map of the specification (`Shape.responseCell`,
`OpeningShape.partialPosition`, `OpeningShape.innerCell`, `OpeningShape.sliceCell`,
`OpeningShape.partialCell`), and each matrix is the prefix view of the flat setup in the column
order of the specification (`Shape.innerSetupCode`, `Shape.outerSetupCode`,
`OpeningShape.openSetupCode`; `Setup.shared`): entry `(r, c)` is the ring element
`r·width + code c` of the setup.

- `traceTerms` has one term for each column of `ẑ` (the rows of `A` and the contraction of
  (103)), each block (the partials of (103)), each block and row of `A` (the image of (104)),
  each column of `B` and each column of `D`.
- `traceScalar` is the scalar row (101): `θ·Σ_{i,u,j} χ_blk(i)·b_1^u·T_{ρ_pack}(X^j)·ẽq(r_2, x)`
  at the cell `x` of coordinate `j` of digit `u` of partial `i`.
- `normValue` is `ẑ(r_virt)` at `r_2`: `Σ_x ẽq(r_2, x)·K_{r_virt}(x)` with the kernel
  `Shape.normKernel` (Akita (91)), and `supportValue` is `Σ_{x ∈ I_bin} ẽq(r_virt, x)·ẽq(r_2, x)`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

variable {S : Shape} (O : OpeningShape S) (fold : Descriptor)

/-- The public inputs of the weight of a trace fold: the flat setup, the fold challenges `c_i` in
`R_{q,d_A}`, the batching point `τ_1`, the opening point `ρ` of the level and the factor `θ`. -/
structure TraceInputs where
  setup : Array Fq
  challenges : List (Array Fq)
  tau1 : List Ext
  point : List Ext
  theta : Ext

/-- The element `1` of `R_{q,d}`. -/
def unitRing (d : ℕ) : Array Fq := Poly.ofFn d fun k => if k = 0 then Fq.one else Fq.zero

/-- `X^j` in `R_{q,d}`. -/
def monomial (d j : ℕ) : Array Fq := Poly.ofFn d fun k => if k = j then Fq.one else Fq.zero

/-- Entry `(r, c)` of the prefix view of the flat setup with the degree `d` and the width
`width`. -/
def viewEntry (setup : Array Fq) (d width r c : ℕ) : Array Fq := Poly.ring d setup (r * width + c)

/-- `Fq` of a natural number. -/
def natFq (n : ℕ) : Fq := Fq.ofNat n

/-- The digits `Σ_u 2^u·ξ_{v + n·d·u}` of entry `entry` of an image of `n` elements of degree
`d` (`Form.recompose`). -/
def recomposeReads (d n : ℕ) (cell : ℕ → Option ℕ) (entry : ℕ) : List Read :=
  (List.range binaryDigits).map fun u => (natFq (2 ^ u), fun k => cell (entry * d + k + n * d * u))

/-- The terms of the rows of a compression path (Akita (47)): the recomposition of `ξ̂_1` in the
`n` image rows from the row `image`, the rows `K_1·ξ̂_1 − G·ξ̂_2` from the row `first`, and the
rows `K_2·ξ̂_2` from the row `second`. `K_1` and `K_2` are the prefix views of the setup. -/
def chainTerms (setup : Array Fq) (coef : ℕ → Ext) : Option ChainShape → (d n : ℕ) →
    (firstCell secondCell : ℕ → Option ℕ) → (image first second : ℕ) → List (List Term)
  | none, _, _, _, _, _, _, _ => []
  | some c, d, n, firstCell, secondCell, image, first, second =>
      [(List.range n).map fun j =>
          { degree := d, reads := recomposeReads d n firstCell j
            muls := [(-coef (image + j), unitRing d)] },
        (List.range c.firstWidth).map fun col =>
          { degree := c.firstDegree
            reads := [(Fq.one, fun k => firstCell (col * c.firstDegree + k))]
            muls := (List.range c.firstRank).map fun r =>
              (coef (first + r), viewEntry setup c.firstDegree c.firstWidth r col) },
        (List.range c.firstRank).map fun r =>
          { degree := c.firstDegree, reads := recomposeReads c.firstDegree c.firstRank secondCell r
            muls := [(-coef (first + r), unitRing c.firstDegree)] },
        (List.range c.secondWidth).map fun col =>
          { degree := c.secondDegree
            reads := [(Fq.one, fun k => secondCell (col * c.secondDegree + k))]
            muls := (List.range c.secondRank).map fun r =>
              (coef (second + r), viewEntry setup c.secondDegree c.secondWidth r col) }]

variable {O fold}

/-- The coefficient `ẽq(τ_1, row)` of a row. -/
def TraceInputs.coef (I : TraceInputs) (row : ℕ) : Ext := eqIndex I.tau1 row

/-- Fold challenge `i` in `R_{q,d}`. -/
def TraceInputs.challenge (I : TraceInputs) (d i : ℕ) : Array Fq := I.challenges.getD i (Poly.zero d)

/-- The reads `Σ_f b_z^f·ẑ_{(x,u),f}` of the column `(x, u)` of `ẑ`. -/
def responseReads (x : Fin S.blockLength) (u : Fin S.sourceDigits) : List Read :=
  (List.finRange fold.responseDigits).map fun f =>
    (natFq (fold.responseBase ^ f.val), S.responseCell fold ((x, u), f))

/-- The term of the column `(x, u)` of `ẑ`: the rows (104) of `A` and the contraction of (103),
with the sign of the code: `Σ_i c_i·G·t̂_{i,a} − A_a·ẑ` (Akita (104), left side minus right
side). -/
def responseTerm (I : TraceInputs) (x : Fin S.blockLength) (u : Fin S.sourceDigits) : Term :=
  { degree := S.innerDegree
    reads := responseReads (fold := fold) x u
    muls := ((List.finRange S.innerRank).map fun a =>
        (-I.coef (1 + a.val), viewEntry I.setup S.innerDegree (S.blockLength * S.sourceDigits)
          a.val (S.innerSetupCode (x, u)).val)) ++
      [(-I.coef 0, Poly.scale S.innerDegree (natFq (S.sourceBase ^ u.val))
        (Poly.subMul S.innerDegree (S.innerDegree / 4)
          (eqIndex (positionPoint S I.point) x.val) (unitRing S.innerDegree)))] }

/-- The terms of `ẑ`. -/
def responseTerms (I : TraceInputs) : List Term :=
  (List.finRange S.blockLength).flatMap fun x =>
    (List.finRange S.sourceDigits).map (responseTerm (fold := fold) I x)

/-- The term of partial `i` of (103). -/
def partialTerm (I : TraceInputs) (i : Fin S.blocks) : Term :=
  { degree := S.innerDegree
    reads := (List.finRange S.openingDigits).map fun u =>
      (natFq (S.openingBase ^ u.val), O.partialPosition fold i u)
    muls := [(I.coef 0, I.challenge S.innerDegree i.val)] }

/-- The terms of the partials of (103). -/
def partialTerms (I : TraceInputs) : List Term :=
  (List.finRange S.blocks).map (partialTerm (O := O) (fold := fold) I)

/-- The term of the image `c_i·G·t̂_{i,a}` of (104). -/
def imageTerm (I : TraceInputs) (i : Fin S.blocks) (a : Fin S.innerRank) : Term :=
  { degree := S.innerDegree
    reads := (List.finRange S.openingDigits).map fun u =>
      (natFq (S.openingBase ^ u.val), O.innerCell fold i a u)
    muls := [(I.coef (1 + a.val), I.challenge S.innerDegree i.val)] }

/-- The terms of the images `Σ_i c_i·G·t̂_{i,a}` of (104). -/
def imageTerms (I : TraceInputs) : List Term :=
  (List.finRange S.blocks).flatMap fun i =>
    (List.finRange S.innerRank).map (imageTerm (O := O) (fold := fold) I i)

/-- The term of the column `(p, (a, y, u))` of slice `s` of `B`. -/
def outerTerm (I : TraceInputs) (s : Fin S.slices) (p : Fin S.sliceLength) (a : Fin S.innerRank)
    (y : Fin S.ratio) (u : Fin S.openingDigits) : Term :=
  { degree := S.outerDegree
    reads := [(Fq.one, O.sliceCell fold s (p, (a, y, u)))]
    muls := (List.finRange S.outerRank).map fun r =>
      (I.coef (1 + S.innerRank + (s.val * S.outerRank + r.val)),
        viewEntry I.setup S.outerDegree
          (S.sliceLength * (S.innerRank * (S.ratio * S.openingDigits))) r.val
          (S.outerSetupCode (p, (a, y, u))).val) }

/-- The terms of the rows (105) of `B`: one for each column of each slice. -/
def outerTerms (I : TraceInputs) : List Term := (List.finRange S.slices).flatMap fun s =>
  (List.finRange S.sliceLength).flatMap fun p => (List.finRange S.innerRank).flatMap fun a =>
    (List.finRange S.ratio).flatMap fun y =>
      (List.finRange S.openingDigits).map (outerTerm (O := O) (fold := fold) I s p a y)

/-- The term of the column `(i, y, u)` of `D`. -/
def openTerm (I : TraceInputs) (i : Fin S.blocks) (y : Fin O.subcolumns)
    (u : Fin S.openingDigits) : Term :=
  { degree := O.openDegree
    reads := [(Fq.one, O.partialCell fold (i, y, u))]
    muls := (List.finRange O.openRank).map fun r =>
      (I.coef (1 + S.innerRank + S.slices * S.outerRank + r.val),
        viewEntry I.setup O.openDegree (S.blocks * (O.subcolumns * S.openingDigits)) r.val
          (O.openSetupCode (i, y, u)).val) }

/-- The terms of the rows (106) of `D`: one for each column. -/
def openTerms (I : TraceInputs) : List Term := (List.finRange S.blocks).flatMap fun i =>
  (List.finRange O.subcolumns).flatMap fun y =>
    (List.finRange S.openingDigits).map (openTerm (O := O) (fold := fold) I i y)

/-- The parts of the rows of a trace fold, in this order: the rows (104) of `A` and the
contraction of (103) on `ẑ`, the partials of (103), the images of (104), the rows (105) of `B`,
the rows (106) of `D`, and the compression rows (47) of the two paths. -/
def traceParts (I : TraceInputs) : List (List Term) :=
  [responseTerms (S := S) (fold := fold) I, partialTerms (O := O) (fold := fold) I,
    imageTerms (O := O) (fold := fold) I, outerTerms (O := O) (fold := fold) I,
    openTerms (O := O) (fold := fold) I] ++
    chainTerms I.setup I.coef S.chain S.outerDegree (S.slices * S.outerRank) (O.firstCell fold)
      (O.secondCell fold) (1 + S.innerRank) (1 + S.innerRank + S.slices * S.outerRank + O.openRank)
      (1 + S.innerRank + S.slices * S.outerRank + O.openRank + firstRank S.chain +
        firstRank O.chain) ++
    chainTerms I.setup I.coef O.chain O.openDegree (1 * O.openRank) (O.openFirstCell fold)
      (O.openSecondCell fold) (1 + S.innerRank + S.slices * S.outerRank)
      (1 + S.innerRank + S.slices * S.outerRank + O.openRank + firstRank S.chain)
      (1 + S.innerRank + S.slices * S.outerRank + O.openRank + firstRank S.chain +
        firstRank O.chain + secondRank S.chain)

/-- The terms of the rows (103) to (106) of a trace fold. -/
def traceTerms (I : TraceInputs) : List Term := (traceParts (O := O) (fold := fold) I).flatten

/-- The count of setup elements that a view of `rank` rows, `width` columns and degree `d` reads. -/
def viewCount (rank width d : ℕ) : ℕ := rank * width * d

/-- The count of setup elements that the compression maps of a path read. -/
def chainCount : Option ChainShape → ℕ
  | none => 0
  | some c => max (viewCount c.firstRank c.firstWidth c.firstDegree)
      (viewCount c.secondRank c.secondWidth c.secondDegree)

variable (O) in
/-- The count of setup elements that the matrices of a trace fold read: the longest prefix of
the views of `A`, `B`, `D` and the compression maps. -/
def traceSetupCount : ℕ :=
  max (viewCount S.innerRank (S.blockLength * S.sourceDigits) S.innerDegree)
    (max (viewCount S.outerRank (S.sliceLength * (S.innerRank * (S.ratio * S.openingDigits)))
      S.outerDegree)
      (max (viewCount O.openRank (S.blocks * (O.subcolumns * S.openingDigits)) O.openDegree)
        (max (chainCount S.chain) (chainCount O.chain))))

/-- The scalar row (101) at `r_2`, with its row coefficient `ẽq(τ_1, N)`. -/
def traceScalar (I : TraceInputs) (eq : ℕ → Ext) : Ext :=
  let d := S.innerDegree
  let pack := packPoint S I.point
  let block := blockPoint S I.point
  let T : Array Ext := Array.ofFn (n := d) fun j => functional (d / 4) pack (monomial d j.val)
  let cells : Ext := sumList (List.finRange S.blocks) fun i =>
    eqIndex block i.val * sumList (List.finRange S.openingDigits) fun u =>
      Ext.scale (natFq (S.openingBase ^ u.val)) (sumList (List.range d) fun j =>
        (O.partialPosition fold i u j).elim Ext.zero fun x => T.getD j Ext.zero * eq x)
  eqIndex I.tau1 O.traceRowCount * (I.theta * cells)

/-- `Σ_x ẽq(r_2, x)·K_{r_virt}(x)` for the kernel `Shape.normKernel` (Akita (91)):
`Σ_{c, k} ẽq(r_virt, c·d + k)·Σ_a b_z^a·ẽq(r_2, (c·δ_f + a)·d + k)`. -/
def normValue (S : Shape) (fold : Descriptor) (virt eq : ℕ → Ext) : Ext :=
  let d := S.innerDegree
  sumList (List.range (S.blockLength * S.sourceDigits)) fun c =>
    sumList (List.range d) fun k =>
      virt (c * d + k) * sumList (List.range fold.responseDigits) fun a =>
        Ext.scale (natFq (fold.responseBase ^ a)) (eq ((c * fold.responseDigits + a) * d + k))

/-- `Σ_{x ∈ I_bin} ẽq(r_virt, x)·ẽq(r_2, x)` for `I_bin = [chainStart, cells)`. -/
def supportValue (O : OpeningShape S) (fold : Descriptor) (virt eq : ℕ → Ext) : Ext :=
  sumList (List.range (O.cells fold - O.chainStart fold)) fun k =>
    virt (O.chainStart fold + k) * eq (O.chainStart fold + k)

end NightstreamFPrime.Spec.Compression.Akita.Replay
