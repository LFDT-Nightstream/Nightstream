import NightstreamFPrime.Spec.Compression.Akita.Shapes
import NightstreamFPrime.Spec.Compression.Akita.Extract.Tower

/-! The numbers of the schedule file `crates/ns-compress/artifacts/key20.aks` for the tower of
`Extract/Tower.lean`: the root with three opening groups, the packing level 1, the trace levels 2
to 5 and the terminal (Akita, ePrint 2026/1983, revision of 2026-10-15, (28), (53), (55), (69),
(75), (76), (84) and Section 9.3). `tests/AkitaSchedules.lean` compares each number that the file
holds with the file.

- `root` is the fold of the key polynomial, `W` and `V`. `level₁` to `level₅` are the levels
  after it, and `terminal` is the shape of the terminal. `norm₃`, `norm₄`, `norm₅` and
  `normTerminal` have `Γ`, `S_max` and `ν_norm` of the Euclidean route.
- The file has the live blocks of a level in `live_blocks`, and `Shape.blocks` is this number.
- These numbers are not in the file. They are fixed in the Akita code (revision `d50f700f`). The
  compression maps have the ring degrees 64 and 32, the rank 1 and one position for each of the
  64 binary digits of each coefficient of their image
  (`crates/akita-params/src/compression.rs`). A quotient has `quotientDigits = ⌈64/3⌉` digits
  (`crates/akita-params/src/witness/tail.rs`). `Γ` is 13 in `R_{q,128}` and 18 in `R_{q,64}`
  (`crates/akita-challenges/src/config.rs`). The extension degree `k` is 2.
- These numbers are choices of this file. `Level.batch` and `rootBatch` are the least `s_rel`
  with a place for each ring row. `level₅.vars` is the least `μ` with a place for each cell of
  the flat witness. The opening shape of a trace level has `k·h_car = 1`, because its fold
  challenges are
  in `R_{q,d_A}`. The fields of `terminal` for `B` are those of `level₅`, and the terminal does
  not read them.
- The conditions on these numbers that the levels of `mixedTower_suffix` read are theorems:
  `root_sized`, `root_room`, `root_honest`, `root_points`, `level₁_sized`, `traces_sized`,
  `norms_sized`, `quarters`, `points`, `blocks_live`, `paths` and `covers`.
- `root_degrees` gives the sum of the degrees of the ring rows of the root. The end
  `Batch.quotientRoom = cells + δ_quot·Σ_r d_r` of the witness of the model is the end of the
  witness of the Akita code, and the file has it with the last alignment of the code in
  `output_witness_len` (`tests/AkitaSchedules.lean`).
- `live_cells`: the live blocks of each level after the root and of the terminal have each cell
  that the flat witness of the level before it can use (Akita Section 4.5, (40) to (42)). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

/-- The input of each compression map has a position for each of 64 binary digits of its
image. -/
theorem pathFits_of {d n : ℕ} {c : ChainShape} (first : n * d * 64 ≤ c.firstWidth * c.firstDegree)
    (second : c.firstRank * c.firstDegree * 64 ≤ c.secondWidth * c.secondDegree) :
    PathFits d n (some c) :=
  ⟨(Nat.mul_le_mul_left _ binaryDigits_le).trans first,
    (Nat.mul_le_mul_left _ binaryDigits_le).trans second⟩

/-- `U_dir` of a trace level is `N_A·(Σ_{a<δ_f} b_z^a·b⋆/2)²` for `N_A = M·δ_com·d_A`. -/
theorem Shape.directLimit_eq (S : Shape) (fold : Descriptor) :
    S.directLimit fold = S.blockLength * S.sourceDigits * S.innerDegree *
      (∑ a : Fin fold.responseDigits, fold.responseBase ^ (a : ℕ) * (S.certifiedBase / 2)) ^ 2 := by
  rw [Shape.directLimit, directBound, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin]

namespace Key20

/-- The compression maps of an image of `width` coefficients (Akita Section 4.5). -/
def maps (width : ℕ) : ChainShape where
  firstDegree := 64
  firstRank := 1
  firstWidth := width
  secondDegree := 32
  secondRank := 1
  secondWidth := 128

/-- The commitment shape of `W`: a table of `2^16` cells in 16 blocks of 8 elements of
`R_{q,512}`. -/
def shapeW : Shape where
  innerDegree := 512
  outerDegree := 128
  divides := ⟨4, rfl⟩
  blocks := 16
  blockLength := 8
  sourceBase := 2 ^ 10
  sourceDigits := 7
  innerRank := 2
  openingBase := 8
  openingDigits := 22
  certifiedBase := 8
  outerRank := 1
  slices := 8
  chain := some (maps 1024)

/-- The commitment shape of `V`: a table of `2^16` cells in 8 blocks of 8 elements of
`R_{q,1024}`. -/
def shapeV : Shape where
  innerDegree := 1024
  outerDegree := 128
  divides := ⟨8, rfl⟩
  blocks := 8
  blockLength := 8
  sourceBase := 2 ^ 10
  sourceDigits := 7
  innerRank := 1
  openingBase := 8
  openingDigits := 22
  certifiedBase := 8
  outerRank := 1
  slices := 1
  chain := some (maps 128)

/-- The root fold: the groups of the key polynomial, `W` and `V` in this order, with the shared
matrix `D` of rank 1 over `R_{q,128}` and its compression maps. -/
def root : Batch (Fin 3) where
  shape := ![Shape.key20, shapeW, shapeV]
  carrierDegree := ![128, 64, 64]
  carrierPositive g := ⟨by revert g; decide⟩
  planes _ := 2
  packing := ![4, 4, 8]
  tower := by decide
  fold := ![Descriptor.key20, ⟨51, 8, 6⟩, ⟨51, 8, 6⟩]
  openDegree := 128
  openDivides := by decide
  openRank := 1
  chain := some (maps 128)

/-- `μ` of the root: the count of variables of the table of level 1. -/
def rootVars : ℕ := 22

/-- `s_rel` of the root. -/
def rootBatch : ℕ := 6

/-- `δ_quot` of the root and of level 1, in the base 8. -/
def quotientDigits : ℕ := 22

/-- The numbers of a level after the root: the commitment shape of its source, the shape of its
opening, its opening descriptor, the count `μ` of variables of its flat witness and `s_rel`. -/
structure Level where
  shape : Shape
  opening : OpeningShape shape
  fold : Descriptor
  vars : ℕ
  batch : ℕ

/-- `ν` of the tensor reduction after the checks of a level: the final point has `μ`
coordinates. -/
def Level.tail (L : Level) : ℕ := L.vars - 1

/-- The numbers of the Euclidean route of a level: `Γ`, `S_max` and `ν_norm`, for a response of
`2^ν_norm` coordinates. -/
structure Squared where
  operator : ℕ
  limit : ℕ
  vars : ℕ

/-- Level 1, with coefficient packing in the subring of degree 64. -/
def level₁ : Level where
  shape := {
    innerDegree := 1024, outerDegree := 128, divides := ⟨8, rfl⟩, blocks := 28
    blockLength := 128, sourceBase := 8, sourceDigits := 1, innerRank := 1, openingBase := 8
    openingDigits := 22, certifiedBase := 8, outerRank := 1, slices := 8
    chain := some (maps 1024) }
  opening := {
    carrierDegree := 64, planes := 2, packing := 8, tower := rfl, openDegree := 128
    openDivides := ⟨1, rfl⟩, openRank := 1, chain := some (maps 128) }
  fold := ⟨51, 8, 4⟩
  vars := 21
  batch := 5

/-- Level 2, with evaluation trace and the coefficient norm. -/
def level₂ : Level where
  shape := {
    innerDegree := 128, outerDegree := 128, divides := ⟨1, rfl⟩, blocks := 21
    blockLength := 512, sourceBase := 8, sourceDigits := 1, innerRank := 4, openingBase := 32
    openingDigits := 13, certifiedBase := 32, outerRank := 1, slices := 1
    chain := some (maps 128) }
  opening := {
    carrierDegree := 128, planes := 1, packing := 1, tower := rfl, openDegree := 128
    openDivides := ⟨1, rfl⟩, openRank := 1, chain := some (maps 128) }
  fold := ⟨31, 32, 2⟩
  vars := 19
  batch := 4

/-- Level 3, with evaluation trace and the Euclidean norm. -/
def level₃ : Level where
  shape := {
    innerDegree := 128, outerDegree := 128, divides := ⟨1, rfl⟩, blocks := 11
    blockLength := 256, sourceBase := 32, sourceDigits := 1, innerRank := 4, openingBase := 32
    openingDigits := 13, certifiedBase := 32, outerRank := 1, slices := 1
    chain := some (maps 128) }
  opening := {
    carrierDegree := 128, planes := 1, packing := 1, tower := rfl, openDegree := 128
    openDivides := ⟨1, rfl⟩, openRank := 1, chain := some (maps 128) }
  fold := ⟨31, 32, 3⟩
  vars := 18
  batch := 4

/-- Level 4, with evaluation trace, the Euclidean norm and no compression maps. -/
def level₄ : Level where
  shape := {
    innerDegree := 64, outerDegree := 64, divides := ⟨1, rfl⟩, blocks := 7
    blockLength := 512, sourceBase := 32, sourceDigits := 1, innerRank := 8, openingBase := 64
    openingDigits := 11, certifiedBase := 64, outerRank := 3, slices := 1, chain := none }
  opening := {
    carrierDegree := 64, planes := 1, packing := 1, tower := rfl, openDegree := 64
    openDivides := ⟨1, rfl⟩, openRank := 2, chain := none }
  fold := ⟨53, 64, 2⟩
  vars := 17
  batch := 4

/-- Level 5, with evaluation trace, the Euclidean norm and no compression maps. -/
def level₅ : Level where
  shape := {
    innerDegree := 64, outerDegree := 64, divides := ⟨1, rfl⟩, blocks := 7
    blockLength := 256, sourceBase := 64, sourceDigits := 1, innerRank := 8, openingBase := 64
    openingDigits := 11, certifiedBase := 64, outerRank := 3, slices := 1, chain := none }
  opening := {
    carrierDegree := 64, planes := 1, packing := 1, tower := rfl, openDegree := 64
    openDivides := ⟨1, rfl⟩, openRank := 2, chain := none }
  fold := ⟨53, 64, 2⟩
  vars := 17
  batch := 4

/-- The shape of the terminal: 5 blocks of 256 elements of `R_{q,64}` under `A_term` of rank 8. -/
def terminal : Shape := { level₅.shape with blocks := 5 }

def norm₃ : Squared := ⟨13, 990011035, 15⟩

def norm₄ : Squared := ⟨18, 1325539644, 15⟩

def norm₅ : Squared := ⟨18, 3029804899, 14⟩

def normTerminal : Squared := ⟨18, 2417093711, 14⟩

/-- The trace levels. -/
def traces : List Level := [level₂, level₃, level₄, level₅]

/-- The conditions of `liftBatch` and `liftBatch_valid` on the numbers of the root: `fits`,
`rowsFit`, `response`, `uniform` for the base 8, and `small`. -/
theorem root_sized : root.cells ≤ 2 ^ rootVars ∧ root.rowCount + 1 ≤ 2 ^ rootBatch ∧
    (∀ g, 2 ≤ (root.fold g).responseBase) ∧ (∀ g, (root.shape g).certifiedBase = 8) ∧
    8 < fieldModulus := by decide

/-- The ring rows of the root have the sum 5760 of their degrees, and the flat witness has
3532800 cells: the quotient digits of Akita (76) end at the cell `3532800 + 22·5760 = 3659520`. -/
theorem root_degrees : root.degreeSum = 5760 ∧ root.cells = 3532800 := by decide

/-- The condition `room` of `liftBatch` for the root with the 22 quotient digits of Akita: the
witness ends at the cell 3659520, before the `2^22` cells of the cube. -/
theorem root_room : root.quotientRoom quotientDigits ≤ 2 ^ rootVars := by decide

/-- Each commitment shape of the root has the conditions `Shape.Honest`. -/
theorem root_honest : ∀ g, (root.shape g).Honest := by
  have honestW : shapeW.Honest :=
    ⟨⟨by decide, by decide, by decide, by decide⟩, ⟨by decide, by decide, by decide, by decide⟩,
      le_rfl, by decide, pathFits_of (by decide) (by decide)⟩
  have honestV : shapeV.Honest :=
    ⟨⟨by decide, by decide, by decide, by decide⟩, ⟨by decide, by decide, by decide, by decide⟩,
      le_rfl, by decide, pathFits_of (by decide) (by decide)⟩
  intro g
  fin_cases g
  exacts [Shape.key20_honest, honestW, honestV]

/-- The opening point of each group of the root has segments for `RootLinked`: `B`, `M`,
`k·h_car` and `d_car` are powers of two, and the lengths have the sums 20, 16 and 16. -/
theorem root_points : ∀ g, (root.shape g).blocks = 2 ^ ![5, 4, 3] g ∧
    (root.shape g).blockLength = 2 ^ ![5, 3, 3] g ∧ (root.opening g).stride = 2 ^ ![3, 3, 4] g ∧
    root.carrierDegree g = 2 ^ ![7, 6, 6] g := by decide

/-- The conditions of `liftFold` and `LevelSized` on the numbers of level 1: `fits`, `room`,
`rowsFit`, `base` and `small`. -/
theorem level₁_sized : level₁.opening.cells level₁.fold ≤ 2 ^ level₁.vars ∧
    level₁.opening.quotientRoom level₁.fold quotientDigits ≤ 2 ^ level₁.vars ∧
    level₁.opening.rowCount + 1 ≤ 2 ^ level₁.batch ∧ 2 ≤ level₁.fold.responseBase ∧
    level₁.shape.certifiedBase < fieldModulus := by decide

/-- The conditions of `committedTrace`, `committedTraceSq` and `LevelSized` on the numbers of a
trace level: `fits`, `rowsFit`, `base` and `small`. -/
theorem traces_sized : ∀ L ∈ traces, L.opening.cells L.fold ≤ 2 ^ L.vars ∧
    L.opening.traceRowCount + 1 ≤ 2 ^ L.batch ∧ 2 ≤ L.fold.responseBase ∧
    L.shape.certifiedBase < fieldModulus := by decide

/-- The conditions `room` and `noWrap` of `NormSized` for the levels 3 to 5. -/
theorem norms_sized : ∀ entry ∈ [(level₃, norm₃), (level₄, norm₄), (level₅, norm₅)],
    entry.1.shape.blockLength * entry.1.shape.sourceDigits * entry.1.shape.innerDegree ≤
        2 ^ entry.1.vars ∧
      max (entry.1.shape.directLimit entry.1.fold) entry.2.limit < fieldModulus := by
  simp only [Shape.directLimit_eq]
  decide

/-- The response of each level on the Euclidean route has `2^ν_norm` coordinates, which is
`physical_response_len` of the file. -/
theorem norms_response : ∀ entry ∈ [(level₃, norm₃), (level₄, norm₄), (level₅, norm₅)],
    entry.1.shape.blockLength * entry.1.shape.sourceDigits * entry.1.shape.innerDegree =
      2 ^ entry.2.vars := by decide

/-- `4·m = d_A` for the receivers with evaluation trace: `m` is 32 at the levels 2 and 3, and 16
at the levels 4 and 5 and at the terminal. -/
theorem quarters : 4 * 32 = level₂.shape.innerDegree ∧ 4 * 32 = level₃.shape.innerDegree ∧
    4 * 16 = level₄.shape.innerDegree ∧ 4 * 16 = level₅.shape.innerDegree ∧
    4 * 16 = terminal.innerDegree := by decide

/-- The segments of the points of the links other than the block segment: `k·h_car`, `d_car` and
`M` of level 1 (`Tier.Linked`), and `2·m` and `M` of each receiver with evaluation trace
(`Tier.TraceLinked`, `Terminal.Linked`). -/
theorem points : level₁.opening.stride = 2 ^ 4 ∧ level₁.opening.carrierDegree = 2 ^ 6 ∧
    level₁.shape.blockLength = 2 ^ 7 ∧ 2 * 32 = 2 ^ 6 ∧ level₂.shape.blockLength = 2 ^ 9 ∧
    level₃.shape.blockLength = 2 ^ 8 ∧ 2 * 16 = 2 ^ 5 ∧ level₄.shape.blockLength = 2 ^ 9 ∧
    level₅.shape.blockLength = 2 ^ 8 ∧ terminal.blockLength = 2 ^ 8 := by decide

/-- The block segment of each link (`Tier.Linked`, `Tier.TraceLinked`, `Terminal.Linked`): the
live blocks `B` = 28, 21, 11, 7, 7 of the levels 1 to 5 and 5 of the terminal are at most
`2^{r_blk}` for `r_blk` = 5, 5, 4, 3, 3, 3 (Akita Section 4.5, (40) to (42)). With the segments
of `points`, the point of each link has the coordinates of the final claim of its sender: `μ` of
the root, and `ν = μ − 1` of the levels 1 to 5. -/
theorem blocks_live : (level₁.shape.blocks ≤ 2 ^ 5 ∧ 4 + (6 + (7 + 5)) = rootVars) ∧
    (level₂.shape.blocks ≤ 2 ^ 5 ∧ 6 + (9 + 5) = level₁.tail) ∧
    (level₃.shape.blocks ≤ 2 ^ 4 ∧ 6 + (8 + 4) = level₂.tail) ∧
    (level₄.shape.blocks ≤ 2 ^ 3 ∧ 5 + (9 + 3) = level₃.tail) ∧
    (level₅.shape.blocks ≤ 2 ^ 3 ∧ 5 + (8 + 3) = level₄.tail) ∧
    (terminal.blocks ≤ 2 ^ 3 ∧ 5 + (8 + 3) = level₅.tail) := by decide

/-- The live blocks of each receiver have each cell that the witness of its sender can use: the
end of the quotient digits of the root and of level 1, and the end of the flat witness of the
levels 2 to 5. An honest table is 0 from these cells, so it is the table that the live blocks
decode to (Akita Section 4.5, (40) to (42)). -/
theorem live_cells :
    root.quotientRoom quotientDigits ≤
        level₁.shape.blocks * (level₁.shape.blockLength * level₁.shape.innerDegree) ∧
      level₁.opening.quotientRoom level₁.fold quotientDigits ≤
        level₂.shape.blocks * (level₂.shape.blockLength * level₂.shape.innerDegree) ∧
      level₂.opening.cells level₂.fold ≤
        level₃.shape.blocks * (level₃.shape.blockLength * level₃.shape.innerDegree) ∧
      level₃.opening.cells level₃.fold ≤
        level₄.shape.blocks * (level₄.shape.blockLength * level₄.shape.innerDegree) ∧
      level₄.opening.cells level₄.fold ≤
        level₅.shape.blocks * (level₅.shape.blockLength * level₅.shape.innerDegree) ∧
      level₅.opening.cells level₅.fold ≤
        terminal.blocks * (terminal.blockLength * terminal.innerDegree) := by decide

/-- The compression maps of `B` and of `D` at the levels 1 to 3 and the compression maps of `D`
at the root have a position for each binary digit of their image (`PathFits`). -/
theorem paths : PathFits root.openDegree (1 * root.openRank) root.chain ∧
    ∀ L ∈ [level₁, level₂, level₃],
      PathFits L.shape.outerDegree (L.shape.slices * L.shape.outerRank) L.shape.chain ∧
        PathFits L.opening.openDegree (1 * L.opening.openRank) L.opening.chain := by
  simp only [List.forall_mem_cons, List.not_mem_nil, false_imp_iff, implies_true, and_true]
  exact ⟨pathFits_of (by decide) (by decide),
    ⟨pathFits_of (by decide) (by decide), pathFits_of (by decide) (by decide)⟩,
    ⟨pathFits_of (by decide) (by decide), pathFits_of (by decide) (by decide)⟩,
    pathFits_of (by decide) (by decide), pathFits_of (by decide) (by decide)⟩

/-- The base 8 of the quotient digits has the conditions `Radix`, and each element of `Z_q` has
`quotientDigits` balanced digits: the hypotheses `radix` and `covers` of `liftedFold_winsTo`. -/
theorem covers : Radix 8 ∧ ∀ x : Zq,
    -((8 / 2 : ℕ) : ℤ) * ∑ u ∈ Finset.range quotientDigits, ((8 : ℕ) : ℤ) ^ u ≤ x.valMinAbs ∧
      x.valMinAbs ≤ (((8 / 2 : ℕ) : ℤ) - 1) *
        ∑ u ∈ Finset.range quotientDigits, ((8 : ℕ) : ℤ) ^ u := by
  refine ⟨⟨by decide, by decide, by decide⟩, fun x => ?_⟩
  have near := ZMod.natAbs_valMinAbs_le x
  have half : fieldModulus / 2 = 9223372036854775778 := by decide
  have total : ∑ u ∈ Finset.range quotientDigits, ((8 : ℕ) : ℤ) ^ u = 10540996613548315209 := by
    decide
  rw [total]
  omega

end Key20

end NightstreamFPrime.Spec.Compression.Akita
