import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictProductionCompiler

/-!
Bounded coefficient correspondence for the production-shaped strict-`PiDEC`
canonical-X receipt.

Assurance tier: artifact-checked for the fixed `54 x 5`, fourteen-child
profile exported by the direct live `enforce_dec_v_strict` invocation.

Owns: reconstruction of the compiler layout from the exact
canonical-to-actual coordinate map; bounded comparison of all 4,590 physical
A/B/C rows with `PiDecStrictProductionCompiler.uniformXRows`; and exact
physical owner/index correspondence.

Does not own: row satisfaction, whole-`PiDEC` acceptance, fixed-point private
columns, protocol composition, commitment binding, or row removal authority.

Emits constraints: no.

| Stage path | Checked equation | Physical owner | Multiplicity |
|---|---|---|---:|
| `nifs.pi_dec.public_x.recompose` | `parent = sum 2^i child_i` | live strict emitter receipt | 270 |
| `nifs.pi_dec.public_x.sign` | shared centered sign rows | live strict emitter receipt | 540 |
| `nifs.pi_dec.public_x.digits` | child selector rows | live strict emitter receipt | 3,780 |
-/

namespace Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.ArtifactRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX

private abbrev artifactCoordinates :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.coordinates

/-- Ordered physical rows returned by the live canonical-X emitter receipt. -/
abbrev artifactRows :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.rows

private def artifactRowChunks : List (List PhysicalRow) :=
  [Generated.Rows.Chunk0.values, Generated.Rows.Chunk1.values,
   Generated.Rows.Chunk2.values, Generated.Rows.Chunk3.values,
   Generated.Rows.Chunk4.values, Generated.Rows.Chunk5.values,
   Generated.Rows.Chunk6.values, Generated.Rows.Chunk7.values,
   Generated.Rows.Chunk8.values, Generated.Rows.Chunk9.values,
   Generated.Rows.Chunk10.values, Generated.Rows.Chunk11.values,
   Generated.Rows.Chunk12.values, Generated.Rows.Chunk13.values,
   Generated.Rows.Chunk14.values, Generated.Rows.Chunk15.values,
   Generated.Rows.Chunk16.values, Generated.Rows.Chunk17.values,
   Generated.Rows.Chunk18.values, Generated.Rows.Chunk19.values,
   Generated.Rows.Chunk20.values]

private theorem artifactRows_eq_chunks :
    artifactRows = artifactRowChunks.flatten := by
  rfl

private def coordinateAt (index : Nat) : CoordinateColumns :=
  artifactCoordinates.getD index
    { parent := 0, children := [], sign := 0, product := 0 }

private def emptyCommitment : CommitmentLayout where
  dCol := 0
  kappaCol := 0
  dataCols := []

private def claim (xColumns : List Nat) : ClaimLayout where
  commitment := emptyCommitment
  adv := none
  xActiveCols := xColumns
  xInactiveCol := 0
  xRows := 54
  xWidth := 5
  xRowsCol := 0
  xWidthCol := 0
  mIn := 270
  mInCol := 0
  yRingCols := []
  ctCols := []
  rCols := []
  sColCols := []
  foldDigestCols := []

private def parentClaim : ClaimLayout :=
  claim (artifactCoordinates.map (fun coordinate => coordinate.parent))

private def childClaim (child : Nat) : ClaimLayout :=
  claim (artifactCoordinates.map (fun coordinate =>
    coordinate.children.getD child 0))

private def baseLayout : PiDecStrictCompiler.Layout where
  radix := 2
  ringDimension := 54
  extensionLimbs := 2
  firstAllocatedColumn := 0
  parent := parentClaim
  children := (List.range 14).map childClaim

/-- Concrete layout reconstructed only from the live receipt's exact column
map. Fields unused by the isolated canonical-X compiler are empty. -/
def layout : PiDecStrictProductionCompiler.Layout where
  base := baseLayout
  xSignTraces := artifactCoordinates.map fun coordinate =>
    (coordinate.sign, coordinate.product)
  childCount := by
    simp [baseLayout, Nightstream.SuperNeo.Concrete.productionGlobalParams]

/-- The fixed receipt reconstructs a well-shaped isolated canonical-X layout.
All non-X lists are empty by construction; the only data-dependent length is
the checked 270-coordinate map. -/
theorem layout_shape_valid :
    PiDecStrictProductionCompiler.ShapeValid layout := by
  refine {
    base := {
      ringPositive := by decide
      powersCanonical := ?_
      commitmentLengths := ?_
      xShapes := ?_
      activeXLengths := ?_
      yShapes := ?_
      rShapes := ?_
      sColShapes := ?_
      ctShapes := ?_
      foldDigestShapes := ?_
    }
    radixTwo := rfl
    ringDimension := rfl
    extensionLimbs := rfl
    traceCount := ?_
    semanticYFits := ?_
  }
  · intro coefficient coefficientMember
    apply PiDecStrictCanonicalX.powers_canonical coefficient
    simpa [layout, baseLayout, PiDecStrictCanonicalX.powers,
      Nightstream.SuperNeo.Concrete.productionGlobalParams] using
      coefficientMember
  · intro child childMember
    change child ∈ (List.range 14).map childClaim at childMember
    rcases List.mem_map.mp childMember with ⟨index, indexMember, rfl⟩
    rfl
  · intro child childMember
    change child ∈ (List.range 14).map childClaim at childMember
    rcases List.mem_map.mp childMember with ⟨index, indexMember, rfl⟩
    exact ⟨rfl, rfl, rfl⟩
  · intro current currentMember
    change current ∈ parentClaim :: (List.range 14).map childClaim at currentMember
    have activeColumnsFive : PiDecStrictCompiler.activeColumns layout.base = 5 := by
      rfl
    rcases List.mem_cons.mp currentMember with currentParent | currentChild
    · subst current
      rw [activeColumnsFive]
      simp only [parentClaim, claim, List.length_map]
      exact Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.coordinates_length
    · rcases List.mem_map.mp currentChild with ⟨index, indexMember, rfl⟩
      rw [activeColumnsFive]
      simp only [childClaim, claim, List.length_map]
      exact Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.coordinates_length
  · intro child childMember
    change child ∈ (List.range 14).map childClaim at childMember
    rcases List.mem_map.mp childMember with ⟨index, indexMember, rfl⟩
    change 0 = 0 ∧ ∀ row, row < 0 → 0 = ([].getD row []).length
    simp
  · intro child childMember
    change child ∈ (List.range 14).map childClaim at childMember
    rcases List.mem_map.mp childMember with ⟨index, indexMember, rfl⟩
    rfl
  · intro child childMember
    change child ∈ (List.range 14).map childClaim at childMember
    rcases List.mem_map.mp childMember with ⟨index, indexMember, rfl⟩
    rfl
  · intro current currentMember
    change current ∈ parentClaim :: (List.range 14).map childClaim at currentMember
    rcases List.mem_cons.mp currentMember with currentParent | currentChild
    · subst current
      simp [parentClaim, claim]
    · rcases List.mem_map.mp currentChild with ⟨index, indexMember, rfl⟩
      simp [childClaim, claim]
  · intro child childMember
    change child ∈ (List.range 14).map childClaim at childMember
    rcases List.mem_map.mp childMember with ⟨index, indexMember, rfl⟩
    rfl
  · simp [layout, PiDecStrictProductionCompiler.logicalXCount, baseLayout,
      parentClaim, claim, PiDecStrictCompiler.activeColumns,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.coordinates_length]
  · intro row rowLt
    simp [layout, baseLayout, parentClaim, claim] at rowLt

/-- Canonical-column order used by the Rust indexed compiler: constant one,
all parents, fourteen child-major blocks, then interleaved sign/product
traces. -/
def canonicalToActual : List Nat :=
  [0] ++
  artifactCoordinates.map (fun coordinate => coordinate.parent) ++
  (List.range 14).flatMap (fun child =>
    artifactCoordinates.map (fun coordinate =>
      coordinate.children.getD child 0)) ++
  artifactCoordinates.flatMap (fun coordinate =>
    [coordinate.sign, coordinate.product])

private theorem length_flatMap_constant_list
    {Input Output : Type} (values : List Input) (width : Nat)
    (body : Input → List Output)
    (bodyLength : ∀ value ∈ values, (body value).length = width) :
    (values.flatMap body).length = values.length * width := by
  induction values with
  | nil => simp
  | cons head tail inductionHypothesis =>
      rw [List.flatMap_cons, List.length_append]
      rw [bodyLength head (by simp)]
      rw [inductionHypothesis (fun value member =>
        bodyLength value (by simp [member]))]
      simp only [List.length_cons, Nat.succ_mul]
      omega

set_option maxRecDepth 100000 in
theorem canonicalToActual_length : canonicalToActual.length = 4591 := by
  have childBlocks :
      ((List.range 14).flatMap (fun child =>
        artifactCoordinates.map (fun coordinate =>
          coordinate.children.getD child 0))).length = 14 * 270 := by
    apply length_flatMap_constant_list
    intro child childMember
    simp only [List.length_map]
    exact Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.coordinates_length
  have traces :
      (artifactCoordinates.flatMap (fun coordinate =>
        [coordinate.sign, coordinate.product])).length = 270 * 2 := by
    apply length_flatMap_constant_list
    intro coordinate coordinateMember
    rfl
  simp only [canonicalToActual, List.length_append, List.length_cons,
    List.length_nil, List.length_map,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.coordinates_length,
    childBlocks, traces]

/-! ## Sparse term-order normalization

The live builder snapshot stores terms in increasing physical-column order.
The independent Lean compiler preserves emitter construction order. These
rows have no duplicate or zero terms, so insertion sorting is the exact
representation bridge; no equation is changed or inferred from labels. -/

private def insertTerm (term : Nat × Nat) : List (Nat × Nat) → List (Nat × Nat)
  | [] => [term]
  | head :: tail =>
      if term.1 < head.1 then term :: head :: tail
      else head :: insertTerm term tail

private def normalizeTerms (terms : List (Nat × Nat)) : List (Nat × Nat) :=
  terms.foldr insertTerm []

/-- Sort sparse terms into the live builder snapshot's column order. -/
def normalizeRow (row : Row) : Row where
  a := normalizeTerms row.a
  b := normalizeTerms row.b
  c := normalizeTerms row.c

private theorem insertTerm_perm (term : Nat × Nat) (terms : List (Nat × Nat)) :
    (insertTerm term terms).Perm (term :: terms) := by
  induction terms with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [insertTerm]
      split
      · rfl
      · exact (inductionHypothesis.cons head).trans
          (List.Perm.swap head term tail).symm

private theorem normalizeTerms_perm (terms : List (Nat × Nat)) :
    (normalizeTerms terms).Perm terms := by
  induction terms with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      change (insertTerm head (normalizeTerms tail)).Perm (head :: tail)
      exact (insertTerm_perm head (normalizeTerms tail)).trans
        (inductionHypothesis.cons head)

private theorem normalizeRow_holds_iff
    (assignment : Nat → Nat) (row : Row) :
    RowHolds assignment (normalizeRow row) ↔ RowHolds assignment row := by
  unfold RowHolds normalizeRow
  rw [Program.lcEval_eq_of_perm assignment (normalizeTerms_perm row.a),
    Program.lcEval_eq_of_perm assignment (normalizeTerms_perm row.b),
    Program.lcEval_eq_of_perm assignment (normalizeTerms_perm row.c)]

/-- Term sorting is a representation-only operation: satisfaction of the
normalized independent compiler schedule implies satisfaction of the same
rows in emitter construction order. This theorem is generic in the row list
and assignment and performs no artifact computation. -/
theorem satisfies_of_normalizedRows
    {rows : List Row} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows.map normalizeRow) assignment) :
    Satisfies rows assignment := by
  intro row rowMember
  apply (normalizeRow_holds_iff assignment row).mp
  exact satisfies (normalizeRow row) (List.mem_map.mpr ⟨row, rowMember, rfl⟩)

/-- Term sorting also preserves honest satisfaction in the forward
direction. Like `satisfies_of_normalizedRows`, this is a generic permutation
theorem and does not inspect the generated receipt. -/
theorem normalizedRows_satisfy_of
    {rows : List Row} {assignment : Nat → Nat}
    (satisfies : Satisfies rows assignment) :
    Satisfies (rows.map normalizeRow) assignment := by
  intro normalized normalizedMember
  rcases List.mem_map.mp normalizedMember with ⟨row, rowMember, rfl⟩
  apply (normalizeRow_holds_iff assignment row).mpr
  exact satisfies row rowMember

private def outerRange (start count : Nat) : List Nat :=
  List.range' start count

private def recompositionInstructionsFor (outerRows : List Nat) :
    List Instruction :=
  let powers := radixPowers layout.base.radix layout.base.children.length
  outerRows.flatMap fun row =>
    (List.range (activeColumns layout.base)).map fun column =>
      recompositionCheck
        (xColumn layout.base layout.base.parent row column)
        (layout.base.children.map fun child =>
          xColumn layout.base child row column)
        powers

private def canonicalityInstructionsFor (outerRows : List Nat) :
    List Instruction :=
  outerRows.flatMap fun row =>
    (List.range (activeColumns layout.base)).flatMap fun column =>
      PiDecStrictCanonicalX.canonicalityInstructions
        (PiDecStrictProductionCompiler.coordinateLayout layout row column)

private def coefficientFragment (instructions : List Instruction) : List Row :=
  (CheckedProgram.rows instructions).map normalizeRow

private def recompositionFragment (outerRows : List Nat) : List Row :=
  coefficientFragment (recompositionInstructionsFor outerRows)

private def canonicalityFragment (outerRows : List Nat) : List Row :=
  coefficientFragment (canonicalityInstructionsFor outerRows)

private theorem coefficientFragment_append
    (left right : List Instruction) :
    coefficientFragment (left ++ right) =
      coefficientFragment left ++ coefficientFragment right := by
  simp [coefficientFragment, CheckedProgram.rows]

private theorem recompositionFragment_append (left right : List Nat) :
    recompositionFragment (left ++ right) =
      recompositionFragment left ++ recompositionFragment right := by
  simp [recompositionFragment, recompositionInstructionsFor,
    coefficientFragment, CheckedProgram.rows]

private theorem canonicalityFragment_append (left right : List Nat) :
    canonicalityFragment (left ++ right) =
      canonicalityFragment left ++ canonicalityFragment right := by
  simp [canonicalityFragment, canonicalityInstructionsFor,
    coefficientFragment, CheckedProgram.rows]

private theorem fragments_flatten_recomposition (chunks : List (List Nat)) :
    (chunks.map recompositionFragment).flatten =
      recompositionFragment chunks.flatten := by
  induction chunks with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.flatten_cons]
      rw [inductionHypothesis, recompositionFragment_append]

private theorem fragments_flatten_canonicality (chunks : List (List Nat)) :
    (chunks.map canonicalityFragment).flatten =
      canonicalityFragment chunks.flatten := by
  induction chunks with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.flatten_cons]
      rw [inductionHypothesis, canonicalityFragment_append]

private def recompositionOuterChunks : List (List Nat) :=
  [outerRange 0 20, outerRange 20 20, outerRange 40 14]

private def canonicalityOuterChunks : List (List Nat) :=
  (List.range 18).map fun chunk => outerRange (3 * chunk) 3

/- The executable inputs here contain only 54 natural-number row indices. -/
private theorem recompositionOuterChunks_cover :
    recompositionOuterChunks.flatten = List.range 54 := by native_decide

private theorem canonicalityOuterChunks_cover :
    canonicalityOuterChunks.flatten = List.range 54 := by native_decide

/-! ## Bounded coefficient certificates

Each `native_decide` below compares one proof-free shard: a dense 100/100/70
recomposition shard or a sparse 240-row canonicality shard. No decision
procedure receives the 4,590-row aggregate. -/

private theorem chunk0_coefficients :
    Generated.Rows.Chunk0.values.map (fun record => record.row) =
      recompositionFragment (outerRange 0 20) := by native_decide

private theorem chunk1_coefficients :
    Generated.Rows.Chunk1.values.map (fun record => record.row) =
      recompositionFragment (outerRange 20 20) := by native_decide

private theorem chunk2_coefficients :
    Generated.Rows.Chunk2.values.map (fun record => record.row) =
      recompositionFragment (outerRange 40 14) := by native_decide

private theorem chunk3_coefficients :
    Generated.Rows.Chunk3.values.map (fun record => record.row) =
      canonicalityFragment (outerRange 0 3) := by native_decide

private theorem chunk4_coefficients :
    Generated.Rows.Chunk4.values.map (fun record => record.row) =
      canonicalityFragment (outerRange 3 3) := by native_decide

private theorem chunk5_coefficients :
    Generated.Rows.Chunk5.values.map (fun record => record.row) =
      canonicalityFragment (outerRange 6 3) := by native_decide

private theorem chunk6_coefficients :
    Generated.Rows.Chunk6.values.map (fun record => record.row) =
      canonicalityFragment (outerRange 9 3) := by native_decide

private theorem chunk7_coefficients :
    Generated.Rows.Chunk7.values.map (fun record => record.row) =
      canonicalityFragment (outerRange 12 3) := by native_decide

private theorem chunk8_coefficients :
    Generated.Rows.Chunk8.values.map (fun record => record.row) =
      canonicalityFragment (outerRange 15 3) := by native_decide

private theorem chunk9_coefficients :
    Generated.Rows.Chunk9.values.map (fun record => record.row) =
      canonicalityFragment (outerRange 18 3) := by native_decide

private theorem chunk10_coefficients :
    Generated.Rows.Chunk10.values.map (fun record => record.row) =
      canonicalityFragment (outerRange 21 3) := by native_decide

private theorem chunk11_coefficients :
    Generated.Rows.Chunk11.values.map (fun record => record.row) =
      canonicalityFragment (outerRange 24 3) := by native_decide

private theorem chunk12_coefficients :
    Generated.Rows.Chunk12.values.map (fun record => record.row) =
      canonicalityFragment (outerRange 27 3) := by native_decide

private theorem chunk13_coefficients :
    Generated.Rows.Chunk13.values.map (fun record => record.row) =
      canonicalityFragment (outerRange 30 3) := by native_decide

private theorem chunk14_coefficients :
    Generated.Rows.Chunk14.values.map (fun record => record.row) =
      canonicalityFragment (outerRange 33 3) := by native_decide

private theorem chunk15_coefficients :
    Generated.Rows.Chunk15.values.map (fun record => record.row) =
      canonicalityFragment (outerRange 36 3) := by native_decide

private theorem chunk16_coefficients :
    Generated.Rows.Chunk16.values.map (fun record => record.row) =
      canonicalityFragment (outerRange 39 3) := by native_decide

private theorem chunk17_coefficients :
    Generated.Rows.Chunk17.values.map (fun record => record.row) =
      canonicalityFragment (outerRange 42 3) := by native_decide

private theorem chunk18_coefficients :
    Generated.Rows.Chunk18.values.map (fun record => record.row) =
      canonicalityFragment (outerRange 45 3) := by native_decide

private theorem chunk19_coefficients :
    Generated.Rows.Chunk19.values.map (fun record => record.row) =
      canonicalityFragment (outerRange 48 3) := by native_decide

private theorem chunk20_coefficients :
    Generated.Rows.Chunk20.values.map (fun record => record.row) =
      canonicalityFragment (outerRange 51 3) := by native_decide

private theorem coefficientChunks_exact :
    artifactRowChunks.map (fun chunk =>
      chunk.map (fun record => record.row)) =
      recompositionOuterChunks.map recompositionFragment ++
      canonicalityOuterChunks.map canonicalityFragment := by
  simp only [artifactRowChunks, List.map_cons, List.map_nil,
    chunk0_coefficients, chunk1_coefficients, chunk2_coefficients,
    chunk3_coefficients, chunk4_coefficients, chunk5_coefficients,
    chunk6_coefficients, chunk7_coefficients, chunk8_coefficients,
    chunk9_coefficients, chunk10_coefficients, chunk11_coefficients,
    chunk12_coefficients, chunk13_coefficients, chunk14_coefficients,
    chunk15_coefficients, chunk16_coefficients, chunk17_coefficients,
    chunk18_coefficients, chunk19_coefficients, chunk20_coefficients]
  rfl

private theorem recompositionFragments_cover :
    (recompositionOuterChunks.map recompositionFragment).flatten =
      (CheckedProgram.rows
        (xRecompositionInstructions layout.base
          (radixPowers layout.base.radix
            layout.base.children.length))).map normalizeRow := by
  rw [fragments_flatten_recomposition, recompositionOuterChunks_cover]
  rfl

private theorem canonicalityFragments_cover :
    (canonicalityOuterChunks.map canonicalityFragment).flatten =
      (CheckedProgram.rows
        (PiDecStrictProductionCompiler.canonicalXInstructions layout)).map
          normalizeRow := by
  rw [fragments_flatten_canonicality, canonicalityOuterChunks_cover]
  rfl

set_option maxRecDepth 100000 in
/-- Every live physical canonical-X row has the same normalized sparse
coefficients as the independent Lean `uniformXRows` compiler. -/
theorem coefficients_exact :
    artifactRows.map (fun record => record.row) =
      (PiDecStrictProductionCompiler.uniformXRows layout).map normalizeRow := by
  calc
    artifactRows.map (fun record => record.row) =
        (artifactRowChunks.map (fun chunk =>
          chunk.map (fun record => record.row))).flatten := by
      rw [artifactRows_eq_chunks, List.map_flatten]
    _ = (recompositionOuterChunks.map recompositionFragment ++
          canonicalityOuterChunks.map canonicalityFragment).flatten := by
      rw [coefficientChunks_exact]
    _ = (recompositionOuterChunks.map recompositionFragment).flatten ++
          (canonicalityOuterChunks.map canonicalityFragment).flatten := by
      rw [List.flatten_append]
    _ = (PiDecStrictProductionCompiler.uniformXRows layout).map
          normalizeRow := by
      rw [recompositionFragments_cover, canonicalityFragments_cover]
      simp [PiDecStrictProductionCompiler.uniformXRows,
        PiDecStrictProductionCompiler.uniformXInstructions,
        CheckedProgram.rows]

/-! ## Exact physical ownership -/

abbrev Ownership := Nat × Nat × RowOwner

def ownershipOf (record : PhysicalRow) : Ownership :=
  (record.relativeIndex, record.physicalIndex, record.owner)

private def recompositionOwnershipRange (start count : Nat) :
    List Ownership :=
  (outerRange start count).map fun activeIndex =>
    (activeIndex,
      Generated.Metadata.recompositionRowStart + activeIndex,
      .recomposition activeIndex)

private def canonicalityOwnershipRange (start count : Nat) :
    List Ownership :=
  (outerRange start count).map fun offset =>
    let activeIndex := offset / 16
    let localRow := offset % 16
    let owner :=
      if localRow = 0 then RowOwner.signProduct activeIndex
      else if localRow = 1 then RowOwner.signZero activeIndex
      else RowOwner.childDigit activeIndex (localRow - 2)
    (270 + offset,
      Generated.Metadata.canonicalityRowStart + offset,
      owner)

/-- Expected ordered ownership receipt for the complete canonical-X program. -/
def expectedOwnership : List Ownership :=
  recompositionOwnershipRange 0 270 ++
  canonicalityOwnershipRange 0 4320

private def recompositionOwnershipChunks : List (List Ownership) :=
  [recompositionOwnershipRange 0 100,
   recompositionOwnershipRange 100 100,
   recompositionOwnershipRange 200 70]

private def canonicalityOwnershipChunks : List (List Ownership) :=
  (List.range 18).map fun chunk =>
    canonicalityOwnershipRange (240 * chunk) 240

/- The ownership checks project the same bounded record shards as the
coefficient checks: dense 100/100/70 or sparse 240 records. -/
private theorem chunk0_ownership :
    Generated.Rows.Chunk0.values.map ownershipOf =
      recompositionOwnershipRange 0 100 := by native_decide
private theorem chunk1_ownership :
    Generated.Rows.Chunk1.values.map ownershipOf =
      recompositionOwnershipRange 100 100 := by native_decide
private theorem chunk2_ownership :
    Generated.Rows.Chunk2.values.map ownershipOf =
      recompositionOwnershipRange 200 70 := by native_decide
private theorem chunk3_ownership :
    Generated.Rows.Chunk3.values.map ownershipOf =
      canonicalityOwnershipRange 0 240 := by native_decide
private theorem chunk4_ownership :
    Generated.Rows.Chunk4.values.map ownershipOf =
      canonicalityOwnershipRange 240 240 := by native_decide
private theorem chunk5_ownership :
    Generated.Rows.Chunk5.values.map ownershipOf =
      canonicalityOwnershipRange 480 240 := by native_decide
private theorem chunk6_ownership :
    Generated.Rows.Chunk6.values.map ownershipOf =
      canonicalityOwnershipRange 720 240 := by native_decide
private theorem chunk7_ownership :
    Generated.Rows.Chunk7.values.map ownershipOf =
      canonicalityOwnershipRange 960 240 := by native_decide
private theorem chunk8_ownership :
    Generated.Rows.Chunk8.values.map ownershipOf =
      canonicalityOwnershipRange 1200 240 := by native_decide
private theorem chunk9_ownership :
    Generated.Rows.Chunk9.values.map ownershipOf =
      canonicalityOwnershipRange 1440 240 := by native_decide
private theorem chunk10_ownership :
    Generated.Rows.Chunk10.values.map ownershipOf =
      canonicalityOwnershipRange 1680 240 := by native_decide
private theorem chunk11_ownership :
    Generated.Rows.Chunk11.values.map ownershipOf =
      canonicalityOwnershipRange 1920 240 := by native_decide
private theorem chunk12_ownership :
    Generated.Rows.Chunk12.values.map ownershipOf =
      canonicalityOwnershipRange 2160 240 := by native_decide
private theorem chunk13_ownership :
    Generated.Rows.Chunk13.values.map ownershipOf =
      canonicalityOwnershipRange 2400 240 := by native_decide
private theorem chunk14_ownership :
    Generated.Rows.Chunk14.values.map ownershipOf =
      canonicalityOwnershipRange 2640 240 := by native_decide
private theorem chunk15_ownership :
    Generated.Rows.Chunk15.values.map ownershipOf =
      canonicalityOwnershipRange 2880 240 := by native_decide
private theorem chunk16_ownership :
    Generated.Rows.Chunk16.values.map ownershipOf =
      canonicalityOwnershipRange 3120 240 := by native_decide
private theorem chunk17_ownership :
    Generated.Rows.Chunk17.values.map ownershipOf =
      canonicalityOwnershipRange 3360 240 := by native_decide
private theorem chunk18_ownership :
    Generated.Rows.Chunk18.values.map ownershipOf =
      canonicalityOwnershipRange 3600 240 := by native_decide
private theorem chunk19_ownership :
    Generated.Rows.Chunk19.values.map ownershipOf =
      canonicalityOwnershipRange 3840 240 := by native_decide
private theorem chunk20_ownership :
    Generated.Rows.Chunk20.values.map ownershipOf =
      canonicalityOwnershipRange 4080 240 := by native_decide

private theorem ownershipChunks_exact :
    artifactRowChunks.map (fun chunk => chunk.map ownershipOf) =
      recompositionOwnershipChunks ++ canonicalityOwnershipChunks := by
  simp only [artifactRowChunks, List.map_cons, List.map_nil,
    chunk0_ownership, chunk1_ownership, chunk2_ownership,
    chunk3_ownership, chunk4_ownership, chunk5_ownership,
    chunk6_ownership, chunk7_ownership, chunk8_ownership,
    chunk9_ownership, chunk10_ownership, chunk11_ownership,
    chunk12_ownership, chunk13_ownership, chunk14_ownership,
    chunk15_ownership, chunk16_ownership, chunk17_ownership,
    chunk18_ownership, chunk19_ownership, chunk20_ownership]
  rfl

private theorem recompositionOwnershipRange_append
    (start left right : Nat) :
    recompositionOwnershipRange start left ++
      recompositionOwnershipRange (start + left) right =
        recompositionOwnershipRange start (left + right) := by
  unfold recompositionOwnershipRange outerRange
  rw [← List.map_append, List.range'_append_1]

private theorem canonicalityOwnershipRange_append
    (start left right : Nat) :
    canonicalityOwnershipRange start left ++
      canonicalityOwnershipRange (start + left) right =
        canonicalityOwnershipRange start (left + right) := by
  unfold canonicalityOwnershipRange outerRange
  rw [← List.map_append, List.range'_append_1]

private theorem recompositionOwnershipChunks_cover :
    recompositionOwnershipChunks.flatten =
      recompositionOwnershipRange 0 270 := by
  simp only [recompositionOwnershipChunks, List.flatten_cons,
    List.flatten_nil, List.append_nil]
  rw [← List.append_assoc]
  rw [recompositionOwnershipRange_append 0 100 100]
  rw [recompositionOwnershipRange_append 0 200 70]

private theorem canonicalityOwnership_block_chunks (blockCount : Nat) :
    ((List.range blockCount).map fun chunk =>
      canonicalityOwnershipRange (240 * chunk) 240).flatten =
        canonicalityOwnershipRange 0 (blockCount * 240) := by
  induction blockCount with
  | zero => simp [canonicalityOwnershipRange, outerRange]
  | succ blockCount inductionHypothesis =>
      rw [List.range_succ, List.map_append, List.flatten_append]
      simp only [List.map_singleton, List.flatten_cons,
        List.flatten_nil, List.append_nil]
      rw [inductionHypothesis]
      have append := canonicalityOwnershipRange_append
        0 (blockCount * 240) 240
      rw [Nat.mul_comm 240 blockCount]
      rw [Nat.add_mul, Nat.one_mul]
      simpa only [Nat.zero_add] using append

private theorem canonicalityOwnershipChunks_cover :
    canonicalityOwnershipChunks.flatten =
      canonicalityOwnershipRange 0 4320 := by
  unfold canonicalityOwnershipChunks
  exact canonicalityOwnership_block_chunks 18

set_option maxRecDepth 100000 in
/-- The generated receipt assigns each ordered row its literal physical row
and indexed equation owner. This theorem states dataflow identity only; it
does not infer semantics from the labels. -/
theorem ownership_exact :
    artifactRows.map ownershipOf = expectedOwnership := by
  calc
    artifactRows.map ownershipOf =
        (artifactRowChunks.map (fun chunk =>
          chunk.map ownershipOf)).flatten := by
      rw [artifactRows_eq_chunks, List.map_flatten]
    _ = (recompositionOwnershipChunks ++
          canonicalityOwnershipChunks).flatten := by
      rw [ownershipChunks_exact]
    _ = recompositionOwnershipChunks.flatten ++
          canonicalityOwnershipChunks.flatten := by
      rw [List.flatten_append]
    _ = expectedOwnership := by
      rw [recompositionOwnershipChunks_cover,
        canonicalityOwnershipChunks_cover]
      rfl

private def ownershipPhysicalIndex (entry : Ownership) : Nat := entry.2.1

private theorem recompositionOwnershipRange_physicalIndices
    (start count : Nat) :
    (recompositionOwnershipRange start count).map ownershipPhysicalIndex =
      List.range'
        (Generated.Metadata.recompositionRowStart + start) count := by
  simp [recompositionOwnershipRange, ownershipPhysicalIndex, outerRange,
    List.map_map, Function.comp_def, List.map_add_range']

private theorem canonicalityOwnershipRange_physicalIndices
    (start count : Nat) :
    (canonicalityOwnershipRange start count).map ownershipPhysicalIndex =
      List.range'
        (Generated.Metadata.canonicalityRowStart + start) count := by
  simp [canonicalityOwnershipRange, ownershipPhysicalIndex, outerRange,
    List.map_map, Function.comp_def, List.map_add_range']

/-- The exported physical indices are exactly the two live-emitter intervals,
in emission order. In particular, this is an index theorem, not an inference
from stage labels. -/
theorem physicalIndices_exact :
    artifactRows.map (fun record => record.physicalIndex) =
      List.range' Generated.Metadata.recompositionRowStart 270 ++
      List.range' Generated.Metadata.canonicalityRowStart 4320 := by
  calc
    artifactRows.map (fun record => record.physicalIndex) =
        (artifactRows.map ownershipOf).map ownershipPhysicalIndex := by
      simp [List.map_map, ownershipOf, ownershipPhysicalIndex,
        Function.comp_def]
    _ = expectedOwnership.map ownershipPhysicalIndex := by
      rw [ownership_exact]
    _ = List.range' Generated.Metadata.recompositionRowStart 270 ++
          List.range' Generated.Metadata.canonicalityRowStart 4320 := by
      simp [expectedOwnership, List.map_append,
        recompositionOwnershipRange_physicalIndices,
        canonicalityOwnershipRange_physicalIndices]

/-- No physical row is multiply owned by the canonical-X receipt. -/
theorem physicalIndices_unique :
    (artifactRows.map (fun record => record.physicalIndex)).Nodup := by
  rw [physicalIndices_exact, List.nodup_append]
  refine ⟨List.nodup_range', List.nodup_range', ?_⟩
  intro recompositionIndex recompositionMember
    canonicalityIndex canonicalityMember
  simp only [List.mem_range'_1] at recompositionMember canonicalityMember
  have rangeFacts :=
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.physical_ranges_exact
  omega

private theorem expectedOwnership_length : expectedOwnership.length = 4590 := by
  simp [expectedOwnership, recompositionOwnershipRange,
    canonicalityOwnershipRange, outerRange]

/-- Exact conservation contract for the live canonical-X emitter receipt.

Together, `ownersExact`, `physicalIndicesExact`, and
`physicalIndicesUnique` say that all 4,590 emitted rows have exactly one
indexed equation owner and that no physical row is owned twice. The interval
field records the independently generated gap between the recomposition and
canonicality families. -/
structure PhysicalOwnerPartition : Prop where
  ownersExact : artifactRows.map ownershipOf = expectedOwnership
  artifactCountExact : artifactRows.length = 4590
  ownerCountExact : expectedOwnership.length = 4590
  physicalIndicesExact :
    artifactRows.map (fun record => record.physicalIndex) =
      List.range' Generated.Metadata.recompositionRowStart 270 ++
      List.range' Generated.Metadata.canonicalityRowStart 4320
  physicalRangesDisjoint :
    Generated.Metadata.recompositionRowEnd ≤
      Generated.Metadata.canonicalityRowStart
  physicalIndicesUnique :
    (artifactRows.map (fun record => record.physicalIndex)).Nodup

/-- The complete live receipt is an exact, disjoint physical-row partition. -/
theorem physical_owner_partition : PhysicalOwnerPartition where
  ownersExact := ownership_exact
  artifactCountExact :=
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.rows_length
  ownerCountExact := expectedOwnership_length
  physicalIndicesExact := physicalIndices_exact
  physicalRangesDisjoint :=
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.physical_ranges_exact.2.2.1
  physicalIndicesUnique := physicalIndices_unique

end Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.ArtifactRows
