import NightstreamFPrime.Spec.Compression.Akita.Key20

/-! The numbers of the schedule file `crates/ns-compress/artifacts/key30.aks`, the file of the key
of `2^19` blocks (Akita, ePrint 2026/1983, revision of 2026-11-23, (28), (53), (55), (69), (75),
(76), (84), Section 9 with (136), and Section 9.3). `tests/AkitaSchedules.lean` compares each
number that the file holds with the file.

- The file has two rows: `w27` for `W` of 27 variables and `w26` for `W` of 26 variables. `Row`
  holds the parameters that vary by row.
- `root` is the fold of the key polynomial, `W` and `V`. `level₁` is the fold of the setup prefix
  and of the witness of the root (Akita (136)): the root is offloaded, and the file has the
  natural length `2^23` of the setup prefix in `setup_natural_len`. `level₂` and `(level₃ r)` have
  coefficient packing with the quotient-lift check, `level₄` to `level₆` have evaluation trace,
  and `terminal` is the shape of the terminal. The levels 5 and 6 and the terminal have the
  numbers of the levels 4 and 5 and of the terminal of `key20.aks`, and the level 5 has its own
  `S_max`.
- The certified base `b⋆` is 8 for each group of the root and 16 for each group of level 1: the
  file has the bounds 7 and 15 of the Module-SIS instances of `B` and `D`, which are `b⋆ − 1`
  (Akita (49)).
- A quotient has `⌈64/3⌉ = 22` digits at the root and `64/4 = 16` digits at the levels 1 to 3
  (`quotientDigits₀`, `quotientDigits`). The numbers that the file does not hold are those of
  `Key20.lean`.
- `covers`: `q ≤ 8^22` and `q ≤ 16^16`, which are the hypothesis `covers` of `liftedFold_winsTo`
  at the root and at the levels 1 to 3. The Akita code uses the residues
  `[−8·Σ_u 16^u, 7·Σ_u 16^u]` at the levels 1 to 3, and `lift` gives these residues.
- The conditions that the levels of the suffix read are theorems: `root_sized`, `level₁_sized`,
  `level₂_sized`, `level₃_sized`, `level₄_sized`, `tails_sized`, `norms_sized`, `quarters`,
  `root_points`,
  `level₁_points`, `points`, `honest` and `live_cells`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

namespace Key30

open Key20 (maps Squared)

/-- The numbers of a row of `key30.aks` that the other row does not have: the count of variables
of `W`, its block length at the root, the live blocks at levels 1 to 3, and the squared norm cap at level 5. -/
structure Row where
  vars : ℕ
  length : ℕ
  blocks₁ : ℕ
  blocks₂ : ℕ
  blocks₃ : ℕ
  cap₅ : ℕ

/-- The second row, for `W` of 27 variables. -/
def w27 : Row := ⟨27, 256, 148, 43, 23, 1370973736⟩

/-- The first row, for `W` of 26 variables. -/
def w26 : Row := ⟨26, 128, 139, 42, 22, 1360588800⟩

def rows : List Row := [w26, w27]

variable (r : Row)

/-- The commitment shape of `W`: 256 blocks of `M` elements of `R_{q,2048}`. -/
def shapeW : Shape where
  innerDegree := 2048
  outerDegree := 128
  divides := ⟨16, rfl⟩
  blocks := 256
  blockLength := r.length
  sourceBase := 2 ^ 16
  sourceDigits := 4
  innerRank := 1
  openingBase := 8
  openingDigits := 22
  certifiedBase := 8
  outerRank := 1
  slices := 8
  chain := some (maps 1024)

/-- The commitment shape of `V`: a table of `2^26` cells in 256 blocks of 512 elements of
`R_{q,512}`. -/
def shapeV : Shape where
  innerDegree := 512
  outerDegree := 256
  divides := ⟨2, rfl⟩
  blocks := 256
  blockLength := 512
  sourceBase := 2 ^ 13
  sourceDigits := 5
  innerRank := 3
  openingBase := 8
  openingDigits := 22
  certifiedBase := 8
  outerRank := 1
  slices := 2
  chain := some (maps 512)

/-- The root fold: the groups of the key polynomial, `W` and `V` in this order, with the shared
matrix `D` of rank 1 over `R_{q,256}` and its compression maps. -/
def root : Batch (Fin 3) where
  shape := ![Shape.key30, shapeW r, shapeV]
  carrierDegree _ := 128
  carrierPositive _ := ⟨by decide⟩
  planes _ := 2
  packing := ![8, 8, 2]
  tower g := by fin_cases g <;> rfl
  fold := ![Descriptor.key30, ⟨31, 8, 9⟩, ⟨31, 8, 8⟩]
  openDegree := 256
  openDivides _ := ⟨1, rfl⟩
  openRank := 1
  chain := some (maps 256)

/-- The commitment shape of the setup prefix `S^♭` (Akita Section 9, (136)): a table of `2^23`
cells in 128 blocks of 128 elements of `R_{q,512}`. -/
def shapeS : Shape where
  innerDegree := 512
  outerDegree := 256
  divides := ⟨2, rfl⟩
  blocks := 128
  blockLength := 128
  sourceBase := 2 ^ 8
  sourceDigits := 8
  innerRank := 2
  openingBase := 16
  openingDigits := 16
  certifiedBase := 16
  outerRank := 1
  slices := 1
  chain := some (maps 256)

/-- The commitment shape of the witness of level 1: live blocks of 2048 elements of
`R_{q,512}`. -/
def shapeW₁ : Shape where
  innerDegree := 512
  outerDegree := 256
  divides := ⟨2, rfl⟩
  blocks := r.blocks₁
  blockLength := 2048
  sourceBase := 8
  sourceDigits := 1
  innerRank := 2
  openingBase := 16
  openingDigits := 16
  certifiedBase := 16
  outerRank := 1
  slices := 2
  chain := some (maps 512)

/-- The fold of level 1: the group of the setup prefix and the group of the witness in this
order (Akita (136)), with the shared matrix `D` of rank 1 over `R_{q,256}`. -/
def level₁ : Batch (Fin 2) where
  shape := ![shapeS, shapeW₁ r]
  carrierDegree _ := 128
  carrierPositive _ := ⟨by decide⟩
  planes _ := 2
  packing _ := 2
  tower g := by fin_cases g <;> rfl
  fold := ![⟨31, 16, 4⟩, ⟨31, 16, 3⟩]
  openDegree := 256
  openDivides _ := ⟨1, rfl⟩
  openRank := 1
  chain := some (maps 256)

/-- Level 2, with coefficient packing in the subring of degree 64. -/
def level₂ : Key20.Level where
  shape := {
    innerDegree := 128, outerDegree := 128, divides := ⟨1, rfl⟩, blocks := r.blocks₂
    blockLength := 2048, sourceBase := 16, sourceDigits := 1, innerRank := 6, openingBase := 16
    openingDigits := 16, certifiedBase := 16, outerRank := 1, slices := 1
    chain := some (maps 128) }
  opening := {
    carrierDegree := 64, planes := 2, packing := 1, tower := rfl, openDegree := 128
    openDivides := ⟨1, rfl⟩, openRank := 1, chain := some (maps 128) }
  fold := ⟨51, 16, 3⟩
  vars := 21
  batch := 4

/-- Level 3, with coefficient packing in the subring of degree 64. -/
def level₃ : Key20.Level where
  shape := {
    innerDegree := 128, outerDegree := 128, divides := ⟨1, rfl⟩, blocks := r.blocks₃
    blockLength := 512, sourceBase := 16, sourceDigits := 1, innerRank := 5, openingBase := 16
    openingDigits := 16, certifiedBase := 16, outerRank := 1, slices := 1
    chain := some (maps 128) }
  opening := {
    carrierDegree := 64, planes := 2, packing := 1, tower := rfl, openDegree := 128
    openDivides := ⟨1, rfl⟩, openRank := 1, chain := some (maps 128) }
  fold := ⟨51, 16, 3⟩
  vars := 19
  batch := 4

/-- Level 4, with evaluation trace and the coefficient norm. -/
def level₄ : Key20.Level where
  shape := {
    innerDegree := 128, outerDegree := 128, divides := ⟨1, rfl⟩, blocks := 8
    blockLength := 512, sourceBase := 16, sourceDigits := 1, innerRank := 4, openingBase := 32
    openingDigits := 13, certifiedBase := 32, outerRank := 1, slices := 1
    chain := some (maps 128) }
  opening := {
    carrierDegree := 128, planes := 1, packing := 1, tower := rfl, openDegree := 128
    openDivides := ⟨1, rfl⟩, openRank := 1, chain := some (maps 128) }
  fold := ⟨31, 32, 2⟩
  vars := 18
  batch := 4

/-- Level 5: the shape, the opening and the descriptor of the level 4 of `key20.aks`. -/
def level₅ : Key20.Level := Key20.level₄

/-- Level 6: the numbers of the level 5 of `key20.aks`. -/
def level₆ : Key20.Level := Key20.level₅

/-- The shape of the terminal: that of `key20.aks`. -/
def terminal : Shape := Key20.terminal

def norm₅ : Squared := ⟨18, r.cap₅, 15⟩

def norm₆ : Squared := Key20.norm₅

def normTerminal : Squared := Key20.normTerminal

/-- `μ` of the root: the count of variables of the table of the witness group of level 1. -/
def rootVars : ℕ := 28

/-- `s_rel` of the root. -/
def rootBatch : ℕ := 5

/-- `δ_quot` of the root, in the base 8. -/
def quotientDigits₀ : ℕ := 22

/-- `μ` of level 1. -/
def vars₁ : ℕ := 24

/-- `s_rel` of level 1. -/
def batch₁ : ℕ := 5

/-- `δ_quot` of the levels 1 to 3, in the base 16. -/
def quotientDigits : ℕ := 16

/-- `ν_S`: the count of variables of the setup prefix (Akita (134)). -/
def setupVars : ℕ := 23

/-- The conditions of the root on its numbers: `fits`, `room`, `rowsFit`, `response`, `uniform`
for the base 8, and `small`. -/
theorem root_sized : ∀ r ∈ rows, (root r).cells ≤ 2 ^ rootVars ∧
    (root r).quotientRoom quotientDigits₀ ≤ 2 ^ rootVars ∧
    (root r).rowCount + 1 ≤ 2 ^ rootBatch ∧ (∀ g, 2 ≤ ((root r).fold g).responseBase) ∧
    (∀ g, ((root r).shape g).certifiedBase = 8) ∧ 8 < fieldModulus := by decide

/-- The ring rows of the root have the sum 9600 of their degrees, and the flat witness has
154075136 cells for `W` of 27 variables and 144637952 cells for `W` of 26 variables. -/
theorem root_degrees : (root w27).degreeSum = 9600 ∧ (root w27).cells = 154075136 ∧
    (root w26).degreeSum = 9600 ∧ (root w26).cells = 144637952 := by decide

/-- The conditions of `liftBatch` and `liftBatch_valid` on the numbers of level 1: `fits`, `room`,
`rowsFit`, `response`, `uniform` for the base 16, and `small`. -/
theorem level₁_sized : ∀ r ∈ rows, (level₁ r).cells ≤ 2 ^ vars₁ ∧
    (level₁ r).quotientRoom quotientDigits ≤ 2 ^ vars₁ ∧
    (level₁ r).rowCount + 1 ≤ 2 ^ batch₁ ∧ (∀ g, 2 ≤ ((level₁ r).fold g).responseBase) ∧
    (∀ g, ((level₁ r).shape g).certifiedBase = 16) ∧ 16 < fieldModulus := by decide

/-- The conditions of `liftFold` and `LevelSized` on the numbers of level 2. -/
theorem level₂_sized : ∀ r ∈ rows,
    (level₂ r).opening.cells (level₂ r).fold ≤ 2 ^ (level₂ r).vars ∧
    (level₂ r).opening.quotientRoom (level₂ r).fold quotientDigits ≤ 2 ^ (level₂ r).vars ∧
    (level₂ r).opening.rowCount + 1 ≤ 2 ^ (level₂ r).batch ∧
    2 ≤ (level₂ r).fold.responseBase ∧ (level₂ r).shape.certifiedBase < fieldModulus := by decide

/-- The conditions of `liftFold` and `LevelSized` on the numbers of level 3: the quotient
digits end at cell 510976 for `w26` and at cell 523264 for `w27` in the cube of `2^19` cells. -/
theorem level₃_sized : ∀ r ∈ rows, (level₃ r).opening.cells (level₃ r).fold ≤ 2 ^ (level₃ r).vars ∧
    (level₃ r).opening.quotientRoom (level₃ r).fold quotientDigits ≤ 2 ^ (level₃ r).vars ∧
    (level₃ r).opening.rowCount + 1 ≤ 2 ^ (level₃ r).batch ∧ 2 ≤ (level₃ r).fold.responseBase ∧
    (level₃ r).shape.certifiedBase < fieldModulus := by decide

/-- The conditions of `committedTrace` and `LevelSized` on the numbers of level 4. -/
theorem level₄_sized : level₄.opening.cells level₄.fold ≤ 2 ^ level₄.vars ∧
    level₄.opening.traceRowCount + 1 ≤ 2 ^ level₄.batch ∧ 2 ≤ level₄.fold.responseBase ∧
    level₄.shape.certifiedBase < fieldModulus := by decide

/-- The conditions of `committedTraceSq` and `LevelSized` on the numbers of the levels 5
and 6. -/
theorem tails_sized : ∀ L ∈ [level₅, level₆], L.opening.cells L.fold ≤ 2 ^ L.vars ∧
    L.opening.traceRowCount + 1 ≤ 2 ^ L.batch ∧ 2 ≤ L.fold.responseBase ∧
    L.shape.certifiedBase < fieldModulus := by decide

/-- The conditions `room` and `noWrap` of `NormSized` for the levels 5 and 6. -/
theorem norms_sized : ∀ r ∈ rows, ∀ entry ∈ [(level₅, (norm₅ r)), (level₆, norm₆)],
    entry.1.shape.blockLength * entry.1.shape.sourceDigits * entry.1.shape.innerDegree ≤
        2 ^ entry.1.vars ∧
      max (entry.1.shape.directLimit entry.1.fold) entry.2.limit < fieldModulus := by
  simp only [Shape.directLimit_eq]
  decide

/-- The response of each level on the Euclidean route has `2^ν_norm` coordinates, which is
`physical_response_len` of the file. -/
theorem norms_response : ∀ r ∈ rows, ∀ entry ∈ [(level₅, (norm₅ r)), (level₆, norm₆)],
    entry.1.shape.blockLength * entry.1.shape.sourceDigits * entry.1.shape.innerDegree =
      2 ^ entry.2.vars := by decide

/-- `4·m = d_A` for the receivers with evaluation trace: `m` is 32 at the level 4, and 16 at the
levels 5 and 6 and at the terminal. -/
theorem quarters : 4 * 32 = level₄.shape.innerDegree ∧ 4 * 16 = level₅.shape.innerDegree ∧
    4 * 16 = level₆.shape.innerDegree ∧ 4 * 16 = terminal.innerDegree := by decide

/-- The opening point of each group of the root has segments for `RootLinked`: `B`, `M`,
`k·h_car` and `d_car` are powers of two. The lengths have the sums 30, 27 or 26, and 26. -/
theorem root_points : ∀ r ∈ rows, ∀ g, ((root r).shape g).blocks = 2 ^ ![9, 8, 8] g ∧
    ((root r).shape g).blockLength = 2 ^ ![10, r.vars - 19, 9] g ∧
    ((root r).opening g).stride = 2 ^ ![4, 4, 2] g ∧ (root r).carrierDegree g = 2 ^ 7 ∧
    ![4, 4, 2] g + (7 + (![10, r.vars - 19, 9] g + ![9, 8, 8] g)) = ![30, r.vars, 26] g := by
  decide

/-- The segments of the opening points of the two groups of level 1 (Akita (136)): the point
`ρ_S` of the setup prefix has `2 + 7 + 7 + 7 = ν_S` coordinates, and the point `r_2` of the witness
has `2 + 7 + 11 + 8 = μ` coordinates for the 148 or 139 live blocks. -/
theorem level₁_points : ∀ r ∈ rows, ∀ g, ((level₁ r).shape g).blocks ≤ 2 ^ ![7, 8] g ∧
    ((level₁ r).shape g).blockLength = 2 ^ ![7, 11] g ∧ ((level₁ r).opening g).stride = 2 ^ 2 ∧
    (level₁ r).carrierDegree g = 2 ^ 7 ∧
    2 + (7 + (![7, 11] g + ![7, 8] g)) = ![setupVars, rootVars] g := by
  decide

/-- The segments of the points of the links after level 1: `k·h_car`, `d_car`, `M` and the live
blocks of the levels 2 and 3 (`Tier.Linked`), and `2·m`, `M` and the live blocks of each receiver
with evaluation trace (`Tier.TraceLinked`, `Terminal.Linked`). The point of each link has the
coordinates of the final claim of its sender: `μ` of the levels 1 and 2, and `ν = μ − 1` of the
levels 3 to 6. -/
theorem points : ∀ r ∈ rows, ((level₂ r).opening.stride = 2 ^ 1 ∧
      (level₂ r).opening.carrierDegree = 2 ^ 6 ∧ (level₂ r).shape.blockLength = 2 ^ 11 ∧
      (level₂ r).shape.blocks ≤ 2 ^ 6 ∧ 1 + (6 + (11 + 6)) = vars₁ ∧
      1 + (6 + (9 + 5)) = (level₂ r).vars) ∧
    ((level₃ r).opening.stride = 2 ^ 1 ∧ (level₃ r).opening.carrierDegree = 2 ^ 6 ∧
      (level₃ r).shape.blockLength = 2 ^ 9 ∧ (level₃ r).shape.blocks ≤ 2 ^ 5) ∧
    (2 * 32 = 2 ^ 6 ∧ level₄.shape.blockLength = 2 ^ 9 ∧ level₄.shape.blocks ≤ 2 ^ 3 ∧
      6 + (9 + 3) = (level₃ r).tail) ∧
    (2 * 16 = 2 ^ 5 ∧ level₅.shape.blockLength = 2 ^ 9 ∧ level₅.shape.blocks ≤ 2 ^ 3 ∧
      5 + (9 + 3) = level₄.tail) ∧
    (level₆.shape.blockLength = 2 ^ 8 ∧ level₆.shape.blocks ≤ 2 ^ 3 ∧
      5 + (8 + 3) = level₅.tail) ∧
    (terminal.blockLength = 2 ^ 8 ∧ terminal.blocks ≤ 2 ^ 3 ∧ 5 + (8 + 3) = level₆.tail) := by
  decide

/-- Each commitment shape of the root and the shape of the setup prefix have the conditions
`Shape.Honest`. A later shape has one source digit, and its source is a table of digits. -/
theorem honest : (∀ r ∈ rows, ∀ g, ((root r).shape g).Honest) ∧ shapeS.Honest := by
  have honestW : ∀ r ∈ rows, (shapeW r).Honest := by
    intro r member
    simp only [rows, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl <;>
      exact ⟨⟨by decide, by decide, by decide, by decide⟩,
        ⟨by decide, by decide, by decide, by decide⟩, le_rfl, by decide,
        pathFits_of (by decide) (by decide)⟩
  have honestV : shapeV.Honest :=
    ⟨⟨by decide, by decide, by decide, by decide⟩, ⟨by decide, by decide, by decide, by decide⟩,
      le_rfl, by decide, pathFits_of (by decide) (by decide)⟩
  refine ⟨fun r member g => ?_, ⟨by decide, by decide, by decide, by decide⟩,
    ⟨by decide, by decide, by decide, by decide⟩, le_rfl, by decide,
    pathFits_of (by decide) (by decide)⟩
  fin_cases g
  exacts [Shape.key30_honest, honestW r member, honestV]

/-- The live blocks of each receiver have each cell that the witness of its sender can use: the
end of the quotient digits of the root and of the levels 1 to 3, and the end of the flat witness
of the levels 4 to 6 (Akita Section 4.5, (40) to (42)). -/
theorem live_cells : ∀ r ∈ rows,
    ((root r).quotientRoom quotientDigits₀ ≤
          (shapeW₁ r).blocks * ((shapeW₁ r).blockLength * (shapeW₁ r).innerDegree) ∧
        (level₁ r).quotientRoom quotientDigits ≤
          (level₂ r).shape.blocks * ((level₂ r).shape.blockLength * (level₂ r).shape.innerDegree) ∧
        (level₂ r).opening.quotientRoom (level₂ r).fold quotientDigits ≤
          (level₃ r).shape.blocks * ((level₃ r).shape.blockLength * (level₃ r).shape.innerDegree)) ∧
    (level₃ r).opening.quotientRoom (level₃ r).fold quotientDigits ≤
        level₄.shape.blocks * (level₄.shape.blockLength * level₄.shape.innerDegree) ∧
      level₄.opening.cells level₄.fold ≤
        level₅.shape.blocks * (level₅.shape.blockLength * level₅.shape.innerDegree) ∧
      level₅.opening.cells level₅.fold ≤
        level₆.shape.blocks * (level₆.shape.blockLength * level₆.shape.innerDegree) ∧
      level₆.opening.cells level₆.fold ≤
        terminal.blocks * (terminal.blockLength * terminal.innerDegree) := by decide

/-- The bases 8 and 16 of the quotient digits have the conditions `Radix`, with
`q ≤ 8^quotientDigits₀` at the root and `q ≤ 16^quotientDigits` at the levels 1 to 3: the
hypotheses `radix` and `covers` of `liftedFold_winsTo` and of `Batch.liftedFold_winsTo`. At the
levels 1 to 3 the lift of an element above `7·Σ_{u<16} 16^u` is negative (`lift`), as in the Akita
code. -/
theorem covers : (Radix 8 ∧ fieldModulus ≤ 8 ^ quotientDigits₀) ∧
    Radix 16 ∧ fieldModulus ≤ 16 ^ quotientDigits :=
  ⟨⟨⟨by decide, by decide, by decide⟩, by decide⟩, ⟨by decide, by decide, by decide⟩, by decide⟩

end Key30

end NightstreamFPrime.Spec.Compression.Akita
