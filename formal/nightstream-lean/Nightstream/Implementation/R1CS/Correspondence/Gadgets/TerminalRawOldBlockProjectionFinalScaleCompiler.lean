import Nightstream.Implementation.R1CS.Correspondence.Gadgets.TerminalRawOldBlockProjectionCompiler

/-!
Exact compiler contract for the final-round-factorized terminal raw-witness
projection.

The direct compiler multiplies every live block weight through every Boolean
point coordinate.  When every live block lies in the low half of the final
round, that round contributes the same factor `1 - oldBlock[last]` to every
summand.  This compiler therefore owns four row families:

* the compact chi tensor for the strict point prefix;
* the unchanged raw-witness/chi products for every live coordinate;
* one five-row extension-field multiplication per active lane, multiplying
  the complete lane sum by the common final factor; and
* two terminal rows equating the parent limbs to that scaled output.

This module is generic and proof-sized.  It does not choose the production
profile, identify physical columns, or authorize generated rows.  A generated
artifact must provide the physical row bijection and instantiate `finalPoint`
with the omitted verifier-owned old-block coordinate.
-/

namespace Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionFinalScaleCompiler

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionProgram

abbrev PrefixLayout :=
  Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.Layout

abbrev PrefixTensorRowIndex :=
  Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.TensorRowIndex

abbrev PrefixTensorRowsSatisfied :=
  Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.TensorRowsSatisfied

abbrev PrefixCoordinateRowsSatisfied :=
  Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.CoordinateRowsSatisfied

/-- Explicit producer metadata for the enabled factorization.  Keeping these
values in the layout prevents a generated artifact from selecting the mode by
reverse-engineering row counts. -/
structure FactorMetadata where
  factorFinalRound : Bool
  tensorVariables : Nat
  factoredVariable : Nat
  fullOldBlock : Nat -> KColumns
  finalPoint : KColumns

/-- The direct compiler layout restricted to the tensor prefix, plus the one
common-factor multiplication trace owned by every active lane.  The parent
and raw witness columns remain those of `prefix`. -/
structure Layout where
  base : PrefixLayout
  factor : FactorMetadata
  scale : Fin base.activeLanes -> KMulTrace

def laneProductTerms (layout : Layout)
    (lane : Fin layout.base.activeLanes) : KTerms where
  c0 :=
    Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.laneTerms
      layout.base KColumns.c0 lane
  c1 :=
    Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.laneTerms
      layout.base KColumns.c1 lane

def finalFactorTerms (layout : Layout) : KTerms :=
  Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.oneMinusPointTerms
    layout.factor.finalPoint

def terminalTerms (layout : Layout)
    (lane : Fin layout.base.activeLanes) (limb : KColumns -> Nat) :
    List (Nat × Nat) :=
  [(limb (layout.scale lane).output, 1)]

def terminalRowsFor (layout : Layout)
    (lane : Fin layout.base.activeLanes) : List Row :=
  [builderLinearRow (layout.base.parent lane).c0
      (terminalTerms layout lane KColumns.c0),
   builderLinearRow (layout.base.parent lane).c1
      (terminalTerms layout lane KColumns.c1)]

/-- Compact ownership index for the four exact compiler families. -/
inductive RowIndex (layout : Layout) where
  | tensor (index : PrefixTensorRowIndex layout.base)
  | coordinate (coordinate : Fin layout.base.logicalWidth) (limb : Fin 2)
  | scale (lane : Fin layout.base.activeLanes) (definition : Fin 5)
  | terminal (lane : Fin layout.base.activeLanes) (limb : Fin 2)

def expectedRow {layout : Layout} : RowIndex layout -> Row
  | .tensor index =>
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.expectedRow
        (.tensor index)
  | .coordinate coordinate limb =>
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.expectedRow
        (.coordinate coordinate limb)
  | .scale lane definition =>
      (((layout.scale lane).definitions).get
        ⟨definition.val, by simp [KMulTrace.definitions]⟩).builderRow
  | .terminal lane limb =>
      (terminalRowsFor layout lane).get
        ⟨limb.val, by simp [terminalRowsFor]⟩

/-- Indexed satisfaction of exactly the optimized row families. -/
def RowsSatisfied (layout : Layout) (assignment : Nat -> Nat) : Prop :=
  forall index : RowIndex layout, RowHolds assignment (expectedRow index)

theorem RowsSatisfied.tensor
    {layout : Layout} {assignment : Nat -> Nat}
    (satisfies : RowsSatisfied layout assignment) :
    PrefixTensorRowsSatisfied layout.base assignment :=
  fun index => satisfies (.tensor index)

theorem RowsSatisfied.coordinate
    {layout : Layout} {assignment : Nat -> Nat}
    (satisfies : RowsSatisfied layout assignment) :
    PrefixCoordinateRowsSatisfied layout.base assignment :=
  fun coordinate limb => satisfies (.coordinate coordinate limb)

/-- Pure syntactic validity.  `blocksFitPrefix` is the exact condition that
makes the omitted next tensor round all-low; it contains no assignment or
acceptance proposition. -/
structure ShapeValid (layout : Layout) : Prop where
  baseShape :
    Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.ShapeValid
      layout.base
  factorEnabled : layout.factor.factorFinalRound = true
  tensorVariables : layout.factor.tensorVariables =
    layout.base.blockVariables
  factoredVariable : layout.factor.factoredVariable =
    layout.factor.tensorVariables
  prefixPointColumns : forall round,
    layout.base.oldBlock round = layout.factor.fullOldBlock round.val
  finalPointColumn : layout.factor.finalPoint =
    layout.factor.fullOldBlock layout.factor.factoredVariable
  blocksFitPrefix :
    Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.blockCount
      layout.base <=
    2 ^ layout.base.blockVariables
  scaleOperands : forall lane,
    (layout.scale lane).left = laneProductTerms layout lane /\
      (layout.scale lane).right = finalFactorTerms layout
  scaleDefinitionCanonical : forall lane definition,
    definition ∈ (layout.scale lane).definitions -> definition.Canonical
  scaleTraceShape : forall lane, (layout.scale lane).SumLayoutValid

theorem terminalTerms_canonical
    (layout : Layout) (lane : Fin layout.base.activeLanes)
    (limb : KColumns -> Nat) :
    CanonicalTerms (terminalTerms layout lane limb) := by
  simp [CanonicalTerms, terminalTerms, goldilocksP]

/-- Each satisfying tensor family computes exactly the compact semantic
prefix.  This endpoint deliberately depends only on tensor rows. -/
theorem coordinateChiTerms_value_eq_tensorValue
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : RowsSatisfied layout assignment)
    (coordinate : Fin layout.base.logicalWidth) :
    (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateChiTerms
      layout.base coordinate).value assignment =
      (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.tensorValues
        layout.base assignment).getD
        (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateBlock
          layout.base coordinate) K.one := by
  exact Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateChiTerms_value_eq_tensorValue_of_tensorRows
    valid.baseShape canonical one satisfies.tensor coordinate

/-- The unchanged two-row coordinate family binds a raw-witness LC to its
compact-prefix chi weight. -/
theorem coordinate_product_sound
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : RowsSatisfied layout assignment)
    (coordinate : Fin layout.base.logicalWidth) :
    (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.productColumns
      layout.base coordinate).value assignment =
      K.mul
        (K.ofBase (residue (lcEval assignment
          (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.rawTerms
            layout.base coordinate))))
        ((Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateChiTerms
          layout.base coordinate).value
          assignment) := by
  exact Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinate_product_sound_of_coordinateRows
    valid.baseShape
    canonical one satisfies.coordinate coordinate

theorem laneProductTerms_value
    (layout : Layout) (assignment : Nat -> Nat)
    (lane : Fin layout.base.activeLanes) :
    (laneProductTerms layout lane).value assignment =
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.projectedLane
        layout.base assignment lane := by
  rfl

/-- The five scale rows multiply the complete lane sum by the one-minus final
point coordinate. -/
theorem scale_sound
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : RowsSatisfied layout assignment)
    (lane : Fin layout.base.activeLanes) :
    (layout.scale lane).output.value assignment =
      K.mul
        (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.projectedLane
          layout.base assignment lane)
        (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.K.sub
          K.one (layout.factor.finalPoint.value assignment)) := by
  let trace := layout.scale lane
  have definitionsHold : DefinitionsHold assignment trace.definitions := by
    intro definition member
    rcases List.mem_iff_getElem.mp member with
      ⟨definitionIndex, definitionLt, definitionEq⟩
    have definitionLtFive : definitionIndex < 5 := by
      simpa [KMulTrace.definitions] using definitionLt
    let definitionAt := trace.definitions.get
      ⟨definitionIndex, definitionLt⟩
    have rowHolds := satisfies (.scale lane
      ⟨definitionIndex, definitionLtFive⟩)
    have canonicalDefinition : definitionAt.Canonical :=
      valid.scaleDefinitionCanonical lane definitionAt (List.get_mem _ _)
    have holdsAt : Definition.Holds assignment definitionAt := by
      apply builderDefinition_sound canonical one definitionAt
        canonicalDefinition
      simpa [expectedRow, trace, definitionAt] using rowHolds
    rw [← definitionEq]
    exact holdsAt
  have multiplication := KMulTrace.sound trace assignment
    (valid.scaleTraceShape lane) definitionsHold
  have operands := valid.scaleOperands lane
  rw [multiplication, operands.1, operands.2,
    laneProductTerms_value, finalFactorTerms,
    Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.oneMinusPointTerms_value
      assignment one]

/-- The two terminal rows bind the parent to the scale output, with no direct
parent-to-prefix-sum equation left in the optimized vocabulary. -/
theorem terminal_sound
    {layout : Layout} (_valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : RowsSatisfied layout assignment)
    (lane : Fin layout.base.activeLanes) :
    (layout.base.parent lane).value assignment =
      (layout.scale lane).output.value assignment := by
  have row0 : RowHolds assignment
      (builderLinearRow (layout.base.parent lane).c0
        (terminalTerms layout lane KColumns.c0)) := by
    simpa [expectedRow, terminalRowsFor] using
      satisfies (.terminal lane ⟨0, by decide⟩)
  have row1 : RowHolds assignment
      (builderLinearRow (layout.base.parent lane).c1
        (terminalTerms layout lane KColumns.c1)) := by
    simpa [expectedRow, terminalRowsFor] using
      satisfies (.terminal lane ⟨1, by decide⟩)
  have c0 := builderLinearRow_sound canonical one _ _
    (terminalTerms_canonical layout lane KColumns.c0) row0
  have c1 := builderLinearRow_sound canonical one _ _
    (terminalTerms_canonical layout lane KColumns.c1) row1
  simp only [KColumns.value, K.mk.injEq]
  constructor
  · apply Fin.ext
    simpa [terminalTerms, baseAt, residue, lcEval] using
      congrArg residue c0
  · apply Fin.ext
    simpa [terminalTerms, baseAt, residue, lcEval] using
      congrArg residue c1

/-- The prefix product sum has exactly the independent decoded raw-column
meaning; no derived product column remains in the conclusion. -/
theorem projectedLane_eq_decodedRawProjection
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : RowsSatisfied layout assignment)
    (lane : Fin layout.base.activeLanes) :
    Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.projectedLane
        layout.base assignment lane =
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.decodedRawProjection
        layout.base assignment lane := by
  rw [Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.projectedLane_eq_productFold]
  unfold Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.decodedRawProjection
  let coordinates :=
    Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.laneCoordinates
      layout.base lane
  change coordinates.foldr
      (fun coordinate suffix =>
        K.add
          ((Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.productColumns
            layout.base coordinate).value assignment)
          suffix)
      K.zero =
    coordinates.foldr
      (fun coordinate suffix =>
        K.add
          (K.mul
            (K.ofBase (residue (lcEval assignment
              (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.rawTerms
                layout.base coordinate))))
            ((Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateChiTerms
              layout.base coordinate).value
              assignment))
          suffix)
      K.zero
  induction coordinates with
  | nil => rfl
  | cons coordinate tail inductionHypothesis =>
      simp only [List.foldr_cons]
      rw [coordinate_product_sound valid canonical one satisfies coordinate,
        inductionHypothesis]

/-- Soundness of all four optimized families.  The output is the decoded raw
projection at the tensor prefix, scaled exactly once by the omitted all-low
coordinate. -/
theorem rows_imply_scaledDecodedRawProjection
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : RowsSatisfied layout assignment) :
    forall lane, (layout.base.parent lane).value assignment =
      K.mul
        (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.decodedRawProjection
          layout.base assignment lane)
        (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.K.sub
          K.one (layout.factor.finalPoint.value assignment)) := by
  intro lane
  rw [terminal_sound valid canonical one satisfies lane,
    scale_sound valid canonical one satisfies lane,
    projectedLane_eq_decodedRawProjection valid canonical one satisfies lane]

/-- Local honest completeness.  The premises are exact SSA equations for the
four families, never row satisfaction or semantic projection acceptance. -/
theorem rows_complete
    {layout : Layout} (valid : ShapeValid layout)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (tensorHolds : forall index : PrefixTensorRowIndex layout.base,
      Definition.Holds assignment
        ((Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.tensorTrace
          index).definitions.get
          ⟨index.definition.val, by simp [KMulTrace.definitions]⟩))
    (coordinateHolds : forall coordinate,
      Definition.Holds assignment
          ((Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateDefinitions
            layout.base coordinate).get
            ⟨0, by simp [Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateDefinitions]⟩) /\
        Definition.Holds assignment
          ((Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateDefinitions
            layout.base coordinate).get
            ⟨1, by simp [Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateDefinitions]⟩))
    (scaleHolds : forall lane definition,
      definition ∈ (layout.scale lane).definitions ->
        Definition.Holds assignment definition)
    (terminalHolds : forall lane,
      assignment (layout.base.parent lane).c0 =
          lcEval assignment (terminalTerms layout lane KColumns.c0) /\
        assignment (layout.base.parent lane).c1 =
          lcEval assignment (terminalTerms layout lane KColumns.c1)) :
    RowsSatisfied layout assignment := by
  intro index
  cases index with
  | tensor tensorIndex =>
      let level := layout.base.tensorLevels.get tensorIndex.level
      let trace :=
        Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.tensorTrace
          tensorIndex
      let definition := trace.definitions.get
        ⟨tensorIndex.definition.val, by simp [KMulTrace.definitions]⟩
      apply builderDefinition_complete canonical one definition
      · apply valid.baseShape.tensorDefinitionCanonical level
          (List.get_mem _ tensorIndex.level) tensorIndex.multiplication
          definition
        exact List.get_mem _ _
      · exact tensorHolds tensorIndex
  | coordinate coordinate limb =>
      have cases : limb = 0 ∨ limb = 1 := by omega
      rcases cases with rfl | rfl
      · apply builderDefinition_complete canonical one _ (by trivial)
        exact (coordinateHolds coordinate).1
      · apply builderDefinition_complete canonical one _ (by trivial)
        exact (coordinateHolds coordinate).2
  | scale lane definitionIndex =>
      let definition := (layout.scale lane).definitions.get
        ⟨definitionIndex.val, by simp [KMulTrace.definitions]⟩
      apply builderDefinition_complete canonical one definition
      · exact valid.scaleDefinitionCanonical lane definition
          (List.get_mem _ _)
      · exact scaleHolds lane definition (List.get_mem _ _)
  | terminal lane limb =>
      have cases : limb = 0 ∨ limb = 1 := by omega
      rcases cases with rfl | rfl
      · exact builderLinearRow_complete one _ _
          (terminalTerms_canonical layout lane KColumns.c0)
          (terminalHolds lane).1
      · exact builderLinearRow_complete one _ _
          (terminalTerms_canonical layout lane KColumns.c1)
          (terminalHolds lane).2

def tensorMultiplicationCount (layout : Layout) : Nat :=
  Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.tensorMultiplicationCount
    layout.base

/-- Exact conceptual count: five rows per tensor multiplication, two per raw
coordinate, five per lane scale, and two per terminal parent. -/
def rowCount (layout : Layout) : Nat :=
  5 * tensorMultiplicationCount layout +
    2 * layout.base.logicalWidth +
    5 * layout.base.activeLanes +
    2 * layout.base.activeLanes

/-- Artifact-facing physical ownership contract.  Generated profile data must
provide a bijection and exact row identity; counts and labels alone cannot
instantiate this structure. -/
structure ArtifactContract (layout : Layout)
    (artifactRow : Fin (rowCount layout) -> Row) where
  profileRadix : layout.base.radix = 2
  profileChildren : layout.base.childCount = 14
  profileActiveLanes : layout.base.activeLanes = 54
  profilePaddingLanes : 64 - layout.base.activeLanes = 10
  profileLogicalWidth : layout.base.logicalWidth = 11437038
  profileFactorEnabled : layout.factor.factorFinalRound = true
  profileTensorVariables : layout.factor.tensorVariables = 18
  profileFactoredVariable : layout.factor.factoredVariable = 18
  profileBlockCount :
    Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.blockCount
      layout.base = 211797
  profileTensorMultiplications : tensorMultiplicationCount layout = 262143
  profileRows : rowCount layout = 24185169
  shape : ShapeValid layout
  physicalIndex : RowIndex layout -> Fin (rowCount layout)
  physicalIndex_injective : Function.Injective physicalIndex
  physicalIndex_surjective : Function.Surjective physicalIndex
  rowAt : forall index : RowIndex layout,
    artifactRow (physicalIndex index) = expectedRow index

def ArtifactRowsSatisfied
    {layout : Layout} {artifactRow : Fin (rowCount layout) -> Row}
    (_contract : ArtifactContract layout artifactRow)
    (assignment : Nat -> Nat) : Prop :=
  forall index, RowHolds assignment (artifactRow index)

theorem ArtifactContract.rowsSatisfied
    {layout : Layout} {artifactRow : Fin (rowCount layout) -> Row}
    (contract : ArtifactContract layout artifactRow)
    {assignment : Nat -> Nat}
    (satisfies : ArtifactRowsSatisfied contract assignment) :
    RowsSatisfied layout assignment := by
  intro index
  rw [← contract.rowAt index]
  exact satisfies (contract.physicalIndex index)

theorem ArtifactContract.artifactRowsSatisfied_of_rowsSatisfied
    {layout : Layout} {artifactRow : Fin (rowCount layout) -> Row}
    (contract : ArtifactContract layout artifactRow)
    {assignment : Nat -> Nat}
    (satisfies : RowsSatisfied layout assignment) :
    ArtifactRowsSatisfied contract assignment := by
  intro physical
  rcases contract.physicalIndex_surjective physical with ⟨owner, rfl⟩
  rw [contract.rowAt owner]
  exact satisfies owner

theorem artifact_rows_sound
    {layout : Layout} {artifactRow : Fin (rowCount layout) -> Row}
    (contract : ArtifactContract layout artifactRow)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : ArtifactRowsSatisfied contract assignment) :
    forall lane, (layout.base.parent lane).value assignment =
      K.mul
        (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.decodedRawProjection
          layout.base assignment lane)
        (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.K.sub
          K.one (layout.factor.finalPoint.value assignment)) := by
  exact rows_imply_scaledDecodedRawProjection contract.shape canonical one
    (contract.rowsSatisfied satisfies)

end Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionFinalScaleCompiler
