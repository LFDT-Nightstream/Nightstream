import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.SourceRows.Padding
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics

/-!
Bounded production certificate for the delayed combined-NC `y_zcol`
padding rows.

Owns: the first 300 generated source rows, their exact partition into fifteen
20-row output ranges, and their coefficient-level identification with the ten
padding lanes 54 through 63 and both quadratic-extension coordinates.

Does not own: child authority, transcript order, commitment binding, source
assignment construction, selective rewrite acceptance, terminal semantics,
costs, or permission to remove rows.

Emits constraints: none.

Assurance tier: artifact-checked for the three generated row certificates and
the generated boundary layout; model-level for the generic satisfaction-to-
zero theorem.  The latter concludes field-residue zero for both coordinates,
not equality of an arbitrary non-canonical `Nat` assignment word.

Certificate accounting is deliberately bounded:

* source certificates inspect 120, 120, and 60 proof-free `RawSourceRow`
  records;
* five layout certificates inspect three outputs apiece (at most 252 nested
  column/owner records), with ten padding pairs and twenty coordinate owners
  per output;
* chunk cardinalities inspect 128, 128, and 44 rows independently;
* no-overlap is derived in the kernel from three independently checked
  source-row bounds of cardinality 120, 120, and 60;
* the 300-row coverage equality is a kernel proof from `take_append_drop`, not
  a closed computation or a `native_decide` invocation.

No certificate decodes rows or constructs proof-carrying generated values.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.padding_artifact` | Check the exact generated zero-padding rows and their unique source ownership. | checked artifact |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.PaddingArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Decoder

/-! ## Compact proof-free projections -/

def rawTerms (terms : List RawTerm) : List (Nat × Nat) :=
  terms.map fun term => (term.column, term.coefficient)

def rawRow (row : RawSourceRow) : Row where
  a := rawTerms row.a
  b := rawTerms row.b
  c := rawTerms row.c

def rawRows (rows : List RawSourceRow) : List Row :=
  rows.map rawRow

/-- The literal unit-times-constant-one row expected for one padding
coordinate.  Its source row and padded column are supplied independently by
the generated boundary schedule. -/
def expectedRow (sourceRow column : Nat) : RawSourceRow :=
  { schemaVersion := supportedSchemaVersion
    rows := Metadata.sourceRelationRows
    columns := Metadata.sourceRelationColumns
    sourceRow
    a := [{ column, coefficient := 1 }]
    b := [{ column := 0, coefficient := 1 }]
    c := [] }

/-- The ten generated quadratic-extension column pairs for one output. -/
def paddingPairsForOutput (output : Nat) : List RawKColumns :=
  match Metadata.boundary.outputYZcolColumns[output]? with
  | none => []
  | some columns =>
      (columns.drop activeLaneCount).take paddingLaneCount

/-- The twenty scalar columns, in physical row order, owned by one output. -/
def paddingCoordinateColumnsForOutput (output : Nat) : List Nat :=
  (paddingPairsForOutput output).flatMap fun columns =>
    [columns.c0, columns.c1]

/-- Boundary-derived expected source rows for one output.  Failed lookups
produce an empty list, so the fail-closed layout certificate below must prove
all lookups and exact cardinalities before this schedule is useful. -/
def expectedOutputRows (output : Nat) : List RawSourceRow :=
  match Metadata.boundary.outputYZcolPaddingRows[output]?,
      Metadata.boundary.outputYZcolColumns[output]? with
  | some rowRange, some columns =>
      (List.range paddingLaneCount).flatMap fun paddingOffset =>
        match columns[activeLaneCount + paddingOffset]? with
        | none => []
        | some pair =>
            [expectedRow (rowRange.start + 2 * paddingOffset) pair.c0,
             expectedRow (rowRange.start + 2 * paddingOffset + 1) pair.c1]
  | _, _ => []

def outputShard0 : List Nat := List.range' 0 6
def outputShard1 : List Nat := List.range' 6 6
def outputShard2 : List Nat := List.range' 12 3

def layoutShard0 : List Nat := List.range' 0 3
def layoutShard1 : List Nat := List.range' 3 3
def layoutShard2 : List Nat := List.range' 6 3
def layoutShard3 : List Nat := List.range' 9 3
def layoutShard4 : List Nat := List.range' 12 3

def expectedShard0 : List RawSourceRow :=
  outputShard0.flatMap expectedOutputRows

def expectedShard1 : List RawSourceRow :=
  outputShard1.flatMap expectedOutputRows

def expectedShard2 : List RawSourceRow :=
  outputShard2.flatMap expectedOutputRows

/-! The source-row split is output-aligned while retaining the literal order
of the generated 128-row chunks. -/

def sourceShard0 : List RawSourceRow :=
  SourceRows.Chunk0.values.take 120

def sourceShard1 : List RawSourceRow :=
  SourceRows.Chunk0.values.drop 120 ++ SourceRows.Chunk1.values.take 112

def sourceShard2 : List RawSourceRow :=
  SourceRows.Chunk1.values.drop 112 ++ SourceRows.Chunk2.values.take 44

/-! ## Exact boundary cardinality and ownership -/

/-- Fail-closed shape for one output.  Besides the ten-pair/twenty-coordinate
cardinality, the final equality binds the exact physical source-row range. -/
def OutputLayoutValid (output : Nat) : Prop :=
  match Metadata.boundary.outputYZcolPaddingRows[output]?,
      Metadata.boundary.outputYZcolColumns[output]? with
  | some rowRange, some columns =>
      rowRange.stop = rowRange.start + outputPaddingRowsPerOutput ∧
      columns.length = paddedLaneCount ∧
      (columns.drop activeLaneCount).length = paddingLaneCount ∧
      (paddingPairsForOutput output).length = paddingLaneCount ∧
      (paddingCoordinateColumnsForOutput output).length =
        outputPaddingRowsPerOutput ∧
      (expectedOutputRows output).length = outputPaddingRowsPerOutput ∧
      (expectedOutputRows output).map RawSourceRow.sourceRow =
        List.range' rowRange.start outputPaddingRowsPerOutput
  | _, _ => False

instance (output : Nat) : Decidable (OutputLayoutValid output) := by
  unfold OutputLayoutValid
  cases rangeLookup :
      Metadata.boundary.outputYZcolPaddingRows[output]? <;>
    cases columnsLookup :
      Metadata.boundary.outputYZcolColumns[output]? <;>
      infer_instance

theorem boundary_output_count :
    Metadata.boundary.outputYZcolColumns.length = outputCount := by
  native_decide

theorem boundary_padding_range_count :
    Metadata.boundary.outputYZcolPaddingRows.length = outputCount := by
  native_decide

/-! These five computations cover three output records apiece.  Each output
contains 64 column pairs and projects exactly ten padding pairs, twenty
scalar coordinates, and twenty source-row numbers.  Thus each closed layout
computation stays below the explicit 256-record ceiling. -/

set_option maxRecDepth 100000 in
theorem layoutShard0_valid :
    ∀ output ∈ layoutShard0, OutputLayoutValid output := by
  native_decide

set_option maxRecDepth 100000 in
theorem layoutShard1_valid :
    ∀ output ∈ layoutShard1, OutputLayoutValid output := by
  native_decide

set_option maxRecDepth 100000 in
theorem layoutShard2_valid :
    ∀ output ∈ layoutShard2, OutputLayoutValid output := by
  native_decide

set_option maxRecDepth 100000 in
theorem layoutShard3_valid :
    ∀ output ∈ layoutShard3, OutputLayoutValid output := by
  native_decide

set_option maxRecDepth 100000 in
theorem layoutShard4_valid :
    ∀ output ∈ layoutShard4, OutputLayoutValid output := by
  native_decide

theorem layoutShard_coverage :
    layoutShard0 ++ layoutShard1 ++ layoutShard2 ++ layoutShard3 ++
        layoutShard4 =
      List.range outputCount := by
  native_decide

/-- All fifteen output layouts follow from the five bounded certificates. -/
theorem outputLayoutValid (output : Fin outputCount) :
    OutputLayoutValid output.val := by
  have member :
      output.val ∈ layoutShard0 ++ layoutShard1 ++ layoutShard2 ++
        layoutShard3 ++ layoutShard4 := by
    rw [layoutShard_coverage]
    exact List.mem_range.mpr output.isLt
  by_cases member0 : output.val ∈ layoutShard0
  · exact layoutShard0_valid output.val member0
  by_cases member1 : output.val ∈ layoutShard1
  · exact layoutShard1_valid output.val member1
  by_cases member2 : output.val ∈ layoutShard2
  · exact layoutShard2_valid output.val member2
  by_cases member3 : output.val ∈ layoutShard3
  · exact layoutShard3_valid output.val member3
  have member4 : output.val ∈ layoutShard4 := by
    simp only [List.mem_append] at member
    rcases member with (((member0' | member1') | member2') | member3') | member4
    · exact False.elim (member0 member0')
    · exact False.elim (member1 member1')
    · exact False.elim (member2 member2')
    · exact False.elim (member3 member3')
    · exact member4
  exact layoutShard4_valid output.val member4

/-- Every one of the fifteen outputs has exactly ten padding K pairs and
twenty scalar coordinate owners. -/
theorem outputYZcolPadding_cardinality (output : Fin outputCount) :
    (paddingPairsForOutput output.val).length = paddingLaneCount ∧
      (paddingCoordinateColumnsForOutput output.val).length =
        outputPaddingRowsPerOutput := by
  have valid := outputLayoutValid output
  unfold OutputLayoutValid at valid
  cases rangeLookup :
      Metadata.boundary.outputYZcolPaddingRows[output.val]? with
  | none =>
      simp [rangeLookup] at valid
  | some rowRange =>
      cases columnsLookup :
          Metadata.boundary.outputYZcolColumns[output.val]? with
      | none =>
          simp [rangeLookup, columnsLookup] at valid
      | some columns =>
          simp only [rangeLookup, columnsLookup] at valid
          exact ⟨valid.2.2.2.1, valid.2.2.2.2.1⟩

/-- Every output owns its advertised complete 20-row half-open range, in
strict physical row order. -/
theorem outputYZcolPadding_rowOwnership (output : Fin outputCount) :
    ∃ rowRange,
      Metadata.boundary.outputYZcolPaddingRows[output.val]? = some rowRange ∧
      rowRange.stop = rowRange.start + outputPaddingRowsPerOutput ∧
      (expectedOutputRows output.val).map RawSourceRow.sourceRow =
        List.range' rowRange.start outputPaddingRowsPerOutput := by
  have valid := outputLayoutValid output
  unfold OutputLayoutValid at valid
  cases rangeLookup :
      Metadata.boundary.outputYZcolPaddingRows[output.val]? with
  | none =>
      simp [rangeLookup] at valid
  | some rowRange =>
      cases columnsLookup :
          Metadata.boundary.outputYZcolColumns[output.val]? with
      | none =>
          simp [rangeLookup, columnsLookup] at valid
      | some columns =>
          simp only [rangeLookup, columnsLookup] at valid
          exact ⟨rowRange, rfl, valid.1,
            valid.2.2.2.2.2.2⟩

/-- The three output index shards cover exactly outputs 0 through 14. -/
theorem outputShard_coverage :
    outputShard0 ++ outputShard1 ++ outputShard2 = List.range outputCount := by
  native_decide

/-- Output ownership does not overlap.  This computation examines fifteen
`Nat` indices, not generated rows. -/
theorem outputShard_noOverlap :
    (outputShard0 ++ outputShard1 ++ outputShard2).Nodup := by
  native_decide

/-- The fifteen advertised physical ranges are ordered and pairwise
disjoint.  This computation examines fifteen `RawRowRange` records. -/
theorem outputPaddingRanges_orderedDisjoint :
    OrderedDisjointRanges Metadata.boundary.outputYZcolPaddingRows := by
  native_decide

/-! ## Exact generated source rows -/

/-- The first source chunk contributes 120 complete output rows and eight
rows of the next output. -/
theorem chunk0_count : SourceRows.Chunk0.values.length = 128 := by
  native_decide

/-- The second source chunk completes that output, owns five more outputs,
and contributes sixteen rows of output 12. -/
theorem chunk1_count : SourceRows.Chunk1.values.length = 128 := by
  native_decide

/-- Exactly 44 rows of the third source chunk finish outputs 12 through 14. -/
theorem chunk2_padding_count :
    (SourceRows.Chunk2.values.take 44).length = 44 := by
  native_decide

theorem sourceShard0_count : sourceShard0.length = 120 := by
  native_decide

theorem sourceShard1_count : sourceShard1.length = 120 := by
  native_decide

theorem sourceShard2_count : sourceShard2.length = 60 := by
  native_decide

/-! Each equality below compares at most 120 actual proof-free source-row
records with the independently reconstructed boundary schedule.  In
particular, family tags and generated row-range prose play no role. -/

set_option maxRecDepth 100000 in
theorem sourceShard0_exact : sourceShard0 = expectedShard0 := by
  native_decide

set_option maxRecDepth 100000 in
theorem sourceShard1_exact : sourceShard1 = expectedShard1 := by
  native_decide

set_option maxRecDepth 100000 in
theorem sourceShard2_exact : sourceShard2 = expectedShard2 := by
  native_decide

def shard1FirstRow : Nat :=
  match Metadata.boundary.outputYZcolPaddingRows[6]? with
  | none => 0
  | some rowRange => rowRange.start

def shard2FirstRow : Nat :=
  match Metadata.boundary.outputYZcolPaddingRows[12]? with
  | none => 0
  | some rowRange => rowRange.start

/-! No computation below sees more than one source shard.  These three
proof-free certificates inspect 120, 120, and 60 source rows, respectively;
pairwise no-overlap is then a generic arithmetic consequence. -/

theorem sourceShard0_upperBound :
    ∀ row ∈ sourceShard0, row.sourceRow < shard1FirstRow := by
  native_decide

theorem sourceShard1_bounds :
    ∀ row ∈ sourceShard1,
      shard1FirstRow ≤ row.sourceRow ∧ row.sourceRow < shard2FirstRow := by
  native_decide

theorem sourceShard2_lowerBound :
    ∀ row ∈ sourceShard2, shard2FirstRow ≤ row.sourceRow := by
  native_decide

theorem shardFirstRows_ordered : shard1FirstRow < shard2FirstRow := by
  native_decide

theorem sourceShard01_noOverlap :
    ∀ left ∈ sourceShard0, ∀ right ∈ sourceShard1,
      left.sourceRow ≠ right.sourceRow := by
  intro left leftMember right rightMember
  have leftBound := sourceShard0_upperBound left leftMember
  have rightBound := (sourceShard1_bounds right rightMember).1
  omega

theorem sourceShard02_noOverlap :
    ∀ left ∈ sourceShard0, ∀ right ∈ sourceShard2,
      left.sourceRow ≠ right.sourceRow := by
  intro left leftMember right rightMember
  have leftBound := sourceShard0_upperBound left leftMember
  have middleBound := shardFirstRows_ordered
  have rightBound := sourceShard2_lowerBound right rightMember
  omega

theorem sourceShard12_noOverlap :
    ∀ left ∈ sourceShard1, ∀ right ∈ sourceShard2,
      left.sourceRow ≠ right.sourceRow := by
  intro left leftMember right rightMember
  have leftBound := (sourceShard1_bounds left leftMember).2
  have rightBound := sourceShard2_lowerBound right rightMember
  omega

/-- Symbolic coverage of the exact generated prefix.  This theorem invokes
no evaluator: the two complete chunks and the 44-row final prefix are split
only by the generic `take_append_drop` identity. -/
theorem generatedPrefix_coverage :
    SourceRows.Chunk0.values ++ SourceRows.Chunk1.values ++
        SourceRows.Chunk2.values.take 44 =
      sourceShard0 ++ sourceShard1 ++ sourceShard2 := by
  rw [← List.take_append_drop 120 SourceRows.Chunk0.values,
    ← List.take_append_drop 112 SourceRows.Chunk1.values]
  simp only [sourceShard0, sourceShard1, sourceShard2, List.append_assoc]

/-- The independently checked shard cardinalities sum to exactly 300. -/
theorem sourceShard_total_count :
    sourceShard0.length + sourceShard1.length + sourceShard2.length = 300 := by
  rw [sourceShard0_count, sourceShard1_count, sourceShard2_count]

/-! ## Generic semantic transport -/

/-- Satisfaction is kept as three bounded obligations so no definition needs
to normalize a 300-row proof-carrying aggregate. -/
def SourceRowsSatisfy (assignment : Nat → Nat) : Prop :=
  Satisfies (rawRows sourceShard0) assignment ∧
  Satisfies (rawRows sourceShard1) assignment ∧
  Satisfies (rawRows sourceShard2) assignment

/-- Semantic zero statement for one generated output.  Lookup failure is
fail-closed.  Offsets 0 through 9 denote physical lanes 54 through 63. -/
def OutputPaddingZero (assignment : Nat → Nat) (output : Nat) : Prop :=
  match Metadata.boundary.outputYZcolColumns[output]? with
  | none => False
  | some columns =>
      ∀ paddingOffset, paddingOffset < paddingLaneCount →
        ∀ pair,
          columns[activeLaneCount + paddingOffset]? = some pair →
          Semantics.rawKColumnsValue pair assignment =
            ProjectionProgram.K.zero

/-- One literal padding row is exactly a unit coordinate multiplied by the
constant-one column.  No generated collection is evaluated here. -/
theorem expectedRow_holds_iff_coordinate_zero
    (assignment : Nat → Nat) (sourceRow column : Nat)
    (constantOne : assignment 0 = 1) :
    RowHolds assignment (rawRow (expectedRow sourceRow column)) ↔
      assignment column % goldilocksP = 0 := by
  simp [RowHolds, rawRow, rawTerms, expectedRow, lcEval, constantOne,
    goldilocksP]

private theorem outputRows_satisfy_of_shard
    {indices : List Nat} {rows : List RawSourceRow}
    {assignment : Nat → Nat} {output : Nat}
    (exactRows : rows = indices.flatMap expectedOutputRows)
    (sourceSatisfies : Satisfies (rawRows rows) assignment)
    (outputMember : output ∈ indices) :
    Satisfies (rawRows (expectedOutputRows output)) assignment := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨raw, rawMember, rfl⟩
  apply sourceSatisfies (rawRow raw)
  rw [exactRows]
  apply List.mem_map.mpr
  exact ⟨raw,
    List.mem_flatMap.mpr ⟨output, outputMember, rawMember⟩,
    rfl⟩

private theorem expectedCoordinateRow_mem
    {output paddingOffset : Nat} {rowRange : RawRowRange}
    {columns : List RawKColumns} {pair : RawKColumns}
    (rangeLookup :
      Metadata.boundary.outputYZcolPaddingRows[output]? = some rowRange)
    (columnsLookup :
      Metadata.boundary.outputYZcolColumns[output]? = some columns)
    (paddingOffsetLt : paddingOffset < paddingLaneCount)
    (pairLookup :
      columns[activeLaneCount + paddingOffset]? = some pair) :
    expectedRow (rowRange.start + 2 * paddingOffset) pair.c0 ∈
        expectedOutputRows output ∧
      expectedRow (rowRange.start + 2 * paddingOffset + 1) pair.c1 ∈
        expectedOutputRows output := by
  constructor
  · simp only [expectedOutputRows, rangeLookup, columnsLookup]
    apply List.mem_flatMap.mpr
    refine ⟨paddingOffset, List.mem_range.mpr paddingOffsetLt, ?_⟩
    rw [pairLookup]
    simp
  · simp only [expectedOutputRows, rangeLookup, columnsLookup]
    apply List.mem_flatMap.mpr
    refine ⟨paddingOffset, List.mem_range.mpr paddingOffsetLt, ?_⟩
    rw [pairLookup]
    simp

/-- Generic kernel soundness for one boundary-derived output.  The layout
premise supplies the fail-closed range lookup; source satisfaction supplies
the actual equations. -/
theorem outputPaddingZero_of_expectedRows
    {assignment : Nat → Nat} {output : Nat}
    (layout : OutputLayoutValid output)
    (constantOne : assignment 0 = 1)
    (sourceSatisfies :
      Satisfies (rawRows (expectedOutputRows output)) assignment) :
    OutputPaddingZero assignment output := by
  unfold OutputLayoutValid at layout
  unfold OutputPaddingZero
  cases rangeLookup :
      Metadata.boundary.outputYZcolPaddingRows[output]? with
  | none =>
      simp [rangeLookup] at layout
  | some rowRange =>
      cases columnsLookup :
          Metadata.boundary.outputYZcolColumns[output]? with
      | none =>
          simp [rangeLookup, columnsLookup] at layout
      | some columns =>
          simp only [columnsLookup]
          intro paddingOffset paddingOffsetLt pair pairLookup
          have members := expectedCoordinateRow_mem rangeLookup columnsLookup
            paddingOffsetLt pairLookup
          have c0Holds :
              RowHolds assignment
                (rawRow
                  (expectedRow (rowRange.start + 2 * paddingOffset)
                    pair.c0)) :=
            sourceSatisfies _ (List.mem_map.mpr
              ⟨_, members.1, rfl⟩)
          have c1Holds :
              RowHolds assignment
                (rawRow
                  (expectedRow (rowRange.start + 2 * paddingOffset + 1)
                    pair.c1)) :=
            sourceSatisfies _ (List.mem_map.mpr
              ⟨_, members.2, rfl⟩)
          have c0Zero : assignment pair.c0 % goldilocksP = 0 :=
            (expectedRow_holds_iff_coordinate_zero assignment _ _
              constantOne).mp c0Holds
          have c1Zero : assignment pair.c1 % goldilocksP = 0 :=
            (expectedRow_holds_iff_coordinate_zero assignment _ _
              constantOne).mp c1Holds
          have c0Zero' : ProjectionProgram.baseAt assignment pair.c0 = 0 := by
            apply Fin.ext
            exact c0Zero
          have c1Zero' : ProjectionProgram.baseAt assignment pair.c1 = 0 := by
            apply Fin.ext
            exact c1Zero
          change
            (⟨ProjectionProgram.baseAt assignment pair.c0,
              ProjectionProgram.baseAt assignment pair.c1⟩ :
                ProjectionProgram.K) =
              ⟨0, 0⟩
          rw [c0Zero', c1Zero']

/-- Main generic soundness theorem.  Satisfaction of the exact three source
shards plus the explicit constant-one invariant forces every generated
`y_zcol` padding K-coordinate (outputs 0..14, lanes 54..63) to zero. -/
theorem outputYZcolPaddingZero
    {assignment : Nat → Nat}
    (sourceSatisfies : SourceRowsSatisfy assignment)
    (constantOne : assignment 0 = 1)
    (output : Fin outputCount) :
    OutputPaddingZero assignment output.val := by
  have outputMember :
      output.val ∈ outputShard0 ++ outputShard1 ++ outputShard2 := by
    rw [outputShard_coverage]
    exact List.mem_range.mpr output.isLt
  by_cases member0 : output.val ∈ outputShard0
  · apply outputPaddingZero_of_expectedRows
      (outputLayoutValid output) constantOne
    exact outputRows_satisfy_of_shard sourceShard0_exact
      sourceSatisfies.1 member0
  by_cases member1 : output.val ∈ outputShard1
  · apply outputPaddingZero_of_expectedRows
      (outputLayoutValid output) constantOne
    exact outputRows_satisfy_of_shard sourceShard1_exact
      sourceSatisfies.2.1 member1
  have member2 : output.val ∈ outputShard2 := by
    simp only [List.mem_append] at outputMember
    rcases outputMember with (member0' | member1') | member2
    · exact False.elim (member0 member0')
    · exact False.elim (member1 member1')
    · exact member2
  apply outputPaddingZero_of_expectedRows
      (outputLayoutValid output) constantOne
  exact outputRows_satisfy_of_shard sourceShard2_exact
    sourceSatisfies.2.2 member2

/-- Lookup-form corollary exposing the exact boundary coordinates.  A
`Fin paddingLaneCount` offset is added to `activeLaneCount`, so this statement
ranges over precisely lanes 54 through 63. -/
theorem outputYZcolPaddingCoordinateZero
    {assignment : Nat → Nat}
    (sourceSatisfies : SourceRowsSatisfy assignment)
    (constantOne : assignment 0 = 1)
    (output : Fin outputCount)
    (paddingOffset : Fin paddingLaneCount)
    {pair : RawKColumns}
    (lookup :
      (Metadata.boundary.outputYZcolColumns[output.val]?).bind
          (fun columns =>
            columns[activeLaneCount + paddingOffset.val]?) =
        some pair) :
    Semantics.rawKColumnsValue pair assignment = ProjectionProgram.K.zero := by
  have zero := outputYZcolPaddingZero sourceSatisfies constantOne output
  unfold OutputPaddingZero at zero
  cases columnsLookup :
      Metadata.boundary.outputYZcolColumns[output.val]? with
  | none =>
      simp [columnsLookup] at lookup
  | some columns =>
      have pairLookup :
          columns[activeLaneCount + paddingOffset.val]? = some pair := by
        simpa [columnsLookup] using lookup
      simp only [columnsLookup] at zero
      exact zero paddingOffset.val paddingOffset.isLt pair pairLookup

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.PaddingArtifact
