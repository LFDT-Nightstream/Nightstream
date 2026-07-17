import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain

/-!
Semantic and arithmetization dimensions for the two-domain `Pi_CCS` model.

Protocol: SuperNeo `Pi_CCS`, specialized to the Phi81 carrier.
Phase: domain ownership before FE or NC polynomial construction.
Constraint family: none; this file emits no rows.

Owns: the semantic row cube, original CCS width, complete Phi81 carrier
width, source arities, and separately stated coverage predicates for the
existing flat-column/lane and canonical block/lane NC arithmetizations.

Does not own: matrices, assignments, SumCheck polynomials, transcript
challenges, production dimension derivation, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: `SemanticShape` describes what must be checked. An NC
implementation may choose its own Boolean domains, but it does not cover the
semantic carrier merely by declaring widths; it must prove `FlatNcCovers`.
The row, flat-column, and lane domains are intentionally different types.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| `Pi_CCS` | semantic shape | row cube | FE rows use `rowVariables` |
| coefficient embedding | carrier shape | original / complete width | `carrierWidth = ceil(logicalWidth / 54) * 54` |
| `Pi_CCS` | source shape | fresh / running | exactly `freshCount + runningCount` sources |
| split NC | existing arithmetization | flat column / lane cubes | coverage is an explicit proposition, not a trusted equality |
| split NC | packed arithmetization | Phi81 block / lane cubes | block coverage and lane coverage are explicit propositions |
| split NC | packed indexing | Boolean block / lane vertices | one canonical little-endian numeric codec |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Dimensions of the independent Phi81 `Pi_CCS` semantic statement.

`logicalWidth` is the original field-vector/CCS width. Semantic assignments
are carried at `carrierWidth`, because a running CE source may use positions
that were zero padding in a fresh source. -/
structure SemanticShape where
  rowVariables : Nat
  logicalWidth : Nat
  freshCount : Nat
  runningCount : Nat
  matrixCount : Nat
deriving Repr, DecidableEq

namespace SemanticShape

/-- Complete coefficient-carrier width required by the Phi81 embedding. -/
def carrierWidth (shape : SemanticShape) : Nat :=
  PaperJoint.Phi81CarrierLayout.carrierWidth shape.logicalWidth

/-- Number of input sources checked jointly by FE and NC. -/
def sourceCount (shape : SemanticShape) : Nat :=
  shape.freshCount + shape.runningCount

/-- The already-audited paper residual shape over the FE row cube. -/
def paperShape (shape : SemanticShape) : PaperJoint.Shape :=
  { cubeVariables := shape.rowVariables
    freshCount := shape.freshCount
    runningCount := shape.runningCount
    matrixCount := shape.matrixCount
    coefficientCount := ringDegree }

/-- The specialized paper shape has the same source count. -/
theorem paperShape_sourceCount (shape : SemanticShape) :
    shape.paperShape.sourceCount = shape.sourceCount := by
  rfl

/-- The specialized paper shape has exactly the production Phi81 degree. -/
theorem paperShape_coefficientCount (shape : SemanticShape) :
    shape.paperShape.coefficientCount = ringDegree := by
  rfl

/-- Carrier completion cannot remove an original CCS coordinate. -/
theorem logicalWidth_le_carrierWidth (shape : SemanticShape) :
    shape.logicalWidth <= shape.carrierWidth := by
  exact PaperJoint.Phi81CarrierLayout.logicalWidth_le_carrierWidth
    shape.logicalWidth

end SemanticShape

/-- Boolean widths used by the existing flat-column/lane NC decomposition.

This is deliberately not part of `SemanticShape`: another sound
arithmetization may cover the same carrier differently. -/
structure FlatNcDomain where
  columnVariables : Nat
  laneVariables : Nat
deriving Repr, DecidableEq

namespace FlatNcDomain

/-- Cardinality of the padded flat-column cube. -/
def columnCount (domain : FlatNcDomain) : Nat :=
  2 ^ domain.columnVariables

/-- Cardinality of the padded Phi81-lane cube. -/
def laneCount (domain : FlatNcDomain) : Nat :=
  2 ^ domain.laneVariables

/-- Exact coverage obligation for a flat-column/lane NC table.

The first conjunct prevents a completed carried coordinate from escaping the
column cube. The second prevents a real Phi81 lane from escaping the lane
cube. Padding beyond either live domain is an arithmetization detail. -/
def Covers (domain : FlatNcDomain) (shape : SemanticShape) : Prop :=
  shape.carrierWidth <= domain.columnCount /\
    ringDegree <= domain.laneCount

/-- Coverage embeds every semantic carrier coordinate into the flat column
cube without changing its numeric index. -/
def carrierColumn
    {domain : FlatNcDomain}
    {shape : SemanticShape}
    (covers : domain.Covers shape)
    (column : Fin shape.carrierWidth) : Fin domain.columnCount :=
  ⟨column.val, Nat.lt_of_lt_of_le column.isLt covers.1⟩

/-- Coverage embeds every real Phi81 lane into the padded lane cube without
changing its numeric index. -/
def phi81Lane
    {domain : FlatNcDomain}
    {shape : SemanticShape}
    (covers : domain.Covers shape)
    (lane : Fin ringDegree) : Fin domain.laneCount :=
  ⟨lane.val, Nat.lt_of_lt_of_le lane.isLt covers.2⟩

@[simp] theorem carrierColumn_val
    {domain : FlatNcDomain}
    {shape : SemanticShape}
    (covers : domain.Covers shape)
    (column : Fin shape.carrierWidth) :
    (carrierColumn covers column).val = column.val := by
  rfl

@[simp] theorem phi81Lane_val
    {domain : FlatNcDomain}
    {shape : SemanticShape}
    (covers : domain.Covers shape)
    (lane : Fin ringDegree) :
    (phi81Lane covers lane).val = lane.val := by
  rfl

end FlatNcDomain

/-- Boolean widths for the canonical Phi81 block×lane NC decomposition.

Unlike `FlatNcDomain`, the first axis indexes complete 54-coefficient blocks
rather than duplicating each flat carrier coordinate across a second lane
axis. -/
structure BlockNcDomain where
  blockVariables : Nat
  laneVariables : Nat
deriving Repr, DecidableEq

namespace BlockNcDomain

/-- Cardinality of the padded Phi81-block cube. -/
def blockCount (domain : BlockNcDomain) : Nat :=
  2 ^ domain.blockVariables

/-- Cardinality of the padded Phi81-lane cube. -/
def laneCount (domain : BlockNcDomain) : Nat :=
  2 ^ domain.laneVariables

/-- Coverage obligation for a block/lane NC table.

The live table has exactly one cell for every complete-carrier coefficient.
Both Boolean suffixes are arithmetization padding and must be computed as
zero, never supplied by a prover. -/
def Covers (domain : BlockNcDomain) (shape : SemanticShape) : Prop :=
  Phi81ColumnLayout.blockCount shape.carrierWidth <= domain.blockCount /\
    ringDegree <= domain.laneCount

/-- Embed a live Phi81 block into the padded Boolean block cube. -/
def carrierBlock
    {domain : BlockNcDomain}
    {shape : SemanticShape}
    (covers : domain.Covers shape)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    Fin domain.blockCount :=
  ⟨block.val, Nat.lt_of_lt_of_le block.isLt covers.1⟩

/-- Embed a live Phi81 lane into the padded Boolean lane cube. -/
def phi81Lane
    {domain : BlockNcDomain}
    {shape : SemanticShape}
    (covers : domain.Covers shape)
    (lane : Fin ringDegree) : Fin domain.laneCount :=
  ⟨lane.val, Nat.lt_of_lt_of_le lane.isLt covers.2⟩

@[simp] theorem carrierBlock_val
    {domain : BlockNcDomain}
    {shape : SemanticShape}
    (covers : domain.Covers shape)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    (carrierBlock covers block).val = block.val := by
  rfl

@[simp] theorem phi81Lane_val
    {domain : BlockNcDomain}
    {shape : SemanticShape}
    (covers : domain.Covers shape)
    (lane : Fin ringDegree) :
    (phi81Lane covers lane).val = lane.val := by
  rfl

/-- Canonical padded-block index of a Boolean vertex. This is shared by the
NC source polynomial and packed-output projection so their bit order cannot
drift. -/
def blockIndex
    {domain : BlockNcDomain}
    (vertex : BooleanVertex domain.blockVariables) :
    Fin domain.blockCount :=
  ⟨NumericBooleanDomain.index vertex, by
    simpa [blockCount] using
      NumericBooleanDomain.index_lt_twoPow vertex⟩

/-- Canonical padded-lane index of a Boolean vertex. -/
def laneIndex
    {domain : BlockNcDomain}
    (vertex : BooleanVertex domain.laneVariables) :
    Fin domain.laneCount :=
  ⟨NumericBooleanDomain.index vertex, by
    simpa [laneCount] using
      NumericBooleanDomain.index_lt_twoPow vertex⟩

/-- Canonical little-endian Boolean vertex of a padded-block index. -/
def blockVertex
    {domain : BlockNcDomain}
    (block : Fin domain.blockCount) :
    BooleanVertex domain.blockVariables :=
  NumericBooleanDomain.vertex domain.blockVariables block

/-- Canonical little-endian Boolean vertex of a padded-lane index. -/
def laneVertex
    {domain : BlockNcDomain}
    (lane : Fin domain.laneCount) :
    BooleanVertex domain.laneVariables :=
  NumericBooleanDomain.vertex domain.laneVariables lane

@[simp] theorem blockIndex_blockVertex
    {domain : BlockNcDomain}
    (block : Fin domain.blockCount) :
    blockIndex (blockVertex block) = block := by
  apply Fin.ext
  exact NumericBooleanDomain.index_vertex domain.blockVariables block

@[simp] theorem laneIndex_laneVertex
    {domain : BlockNcDomain}
    (lane : Fin domain.laneCount) :
    laneIndex (laneVertex lane) = lane := by
  apply Fin.ext
  exact NumericBooleanDomain.index_vertex domain.laneVariables lane

@[simp] theorem blockVertex_blockIndex
    {domain : BlockNcDomain}
    (vertex : BooleanVertex domain.blockVariables) :
    blockVertex (blockIndex vertex) = vertex := by
  exact NumericBooleanDomain.vertex_index vertex

@[simp] theorem laneVertex_laneIndex
    {domain : BlockNcDomain}
    (vertex : BooleanVertex domain.laneVariables) :
    laneVertex (laneIndex vertex) = vertex := by
  exact NumericBooleanDomain.vertex_index vertex

end BlockNcDomain

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc
