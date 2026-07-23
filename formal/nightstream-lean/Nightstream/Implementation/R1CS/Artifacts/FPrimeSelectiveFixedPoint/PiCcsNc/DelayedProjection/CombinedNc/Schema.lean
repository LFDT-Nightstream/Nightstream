import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCcsRowSchema

/-!
Wire schema for the bounded production combined-NC row projection.

Owns: proof-free sparse source rows, compact thirteen-port emitted rows, the
exact production round column maps, and the delayed block/lane boundary map.

Does not own: field decoding, row satisfaction, compiler rewrites, generated
values, transcript authority, commitment binding, costs, or row removal.

Emits constraints: none.

Every field in this file is untrusted artifact data. In particular, family
tags and row intervals are provenance only; correspondence must decode the
coefficients and compare the actual rows with their independent programs.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `pi_ccs.nc.delayed.combined.source_row` | one exact sparse source A/B/C row | checked after decoding |
| `pi_ccs.nc.delayed.combined.emitted_row` | one exact thirteen-port selective row | checked after decoding |
| `pi_ccs.nc.delayed.combined.round_map` | one local-to-source ten-slot round map | checked after row comparison |
| `pi_ccs.nc.delayed.combined.boundary` | exact delayed-NC input/output column schedule | direct dataflow after decoding |
| `pi_ccs.nc.delayed.combined.padding` | ten padded lanes for each of fifteen outputs | checked after row comparison |
| `pi_ccs.nc.delayed.combined.active_pin` | public writes, selector values, and their exact physical rows | checked after decoding |
| `pi_ccs.nc.delayed.combined.raw_old_block` | fourteen child-major 54-active-plus-10-zero projections at the pending point | checked after execution decoding |
| `pi_ccs.nc.delayed.combined.execution_binding` | transcript and terminal values joined to exact builder and normalized columns | direct dataflow after decoding |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc

abbrev RawFamily :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire.RawFamily

abbrev RawTerm :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire.RawTerm

/-- Half-open row interval. Empty and out-of-bounds ranges remain
representable so the correspondence decoder can reject them. -/
structure RawRowRange where
  start : Nat
  stop : Nat
deriving DecidableEq, Repr, Inhabited

/-- Two source columns carrying one quadratic-extension value. -/
structure RawKColumns where
  c0 : Nat
  c1 : Nat
deriving DecidableEq, Repr, Inhabited

/-- Canonical proof-free coefficient pair for one quadratic-extension value.
The correspondence layer, rather than this wire schema, owns field decoding
and rejects non-canonical coefficients. -/
structure RawK where
  c0 : Nat
  c1 : Nat
deriving DecidableEq, Repr, Inhabited

/-- One child-major lane of the verifier-recomputed projection of a raw
`WitnessMat` at the pending old block.  Active lanes are numbered `0..53`;
lanes `54..63` are verifier-created zeros and carry `padding = true`.

The authority tag lives in the execution header as provenance only.  Neither
that tag nor this value record establishes Rust dataflow by itself. -/
structure RawOldBlockLane where
  schemaVersion : Nat
  child : Nat
  lane : Nat
  padding : Bool
  value : RawK
deriving DecidableEq, Repr, Inhabited

/-- Exact association between one named execution slot, its source-builder
coefficient columns, the normalized production-assignment columns, and the
value observed at both surfaces.  Slot tags and indices are untrusted wire
data; the correspondence decoder checks their complete schedule. -/
structure RawGeneratedKBinding where
  schemaVersion : Nat
  slotKind : Nat
  slotIndex0 : Nat
  slotIndex1 : Nat
  builderC0 : Nat
  builderC1 : Nat
  normalizedC0 : Nat
  normalizedC1 : Nat
  value : RawK
deriving DecidableEq, Repr, Inhabited

/-- One active recursive public write observed at the source builder, the
normalized assignment, and the packed committed witness.  The exporter
checks all three values before emitting this proof-free record. -/
structure RawPublicWrite where
  schemaVersion : Nat
  logicalColumn : Nat
  packedRow : Nat
  packedColumn : Nat
  sourceKind : Nat
  builderColumn : Option Nat
  normalizedSourceColumn : Option Nat
  normalizedColumn : Nat
  width : Nat
  centered : Bool
  aliasSource : Option Nat
  value : Nat
deriving DecidableEq, Repr, Inhabited

/-- One replayed combined-NC SumCheck round.  The fixed production profile
requires exactly five coefficients and twenty-five rounds, checked outside
this permissive artifact schema. -/
structure RawCombinedNcRound where
  schemaVersion : Nat
  index : Nat
  coefficients : List RawK
  challenge : RawK
  claimIn : RawK
  claimOut : RawK
deriving DecidableEq, Repr, Inhabited

/-- Compact header exported from the active post-`Pi_DEC` execution audit.
All values are observations, not authorities.  In particular,
`fieldDecodingTag = 0` records the canonical base-field embedding used by
the exporter. `rawAuthorityTag` is checked only as fail-closed provenance; the concrete
correspondence theorem must derive raw-witness authority from the exported
old-block projection and its production dataflow checks. -/
structure RawExecutionHeader where
  schemaVersion : Nat
  branch : Nat
  proofVariant : Nat
  outputSources : Nat
  outputMatrices : Nat
  outputActiveLanes : Nat
  freshCount : Nat
  runningCount : Nat
  logicalColumns : Nat
  packedRows : Nat
  packedColumns : Nat
  sourceRows : Nat
  sourceColumns : Nat
  finalRows : Nat
  finalColumns : Nat
  publicWriteCount : Nat
  selectorColumns : List Nat
  selectorValues : List Nat
  oldBlock : List RawK
  parentYZcol : List RawK
  radix : RawK
  producerBeta : RawK
  batchWeight : RawK
  betaBlock : List RawK
  betaLane : List RawK
  blockPoint : List RawK
  lanePoint : List RawK
  terminalInitial : RawK
  terminalFinal : RawK
  terminalRhs : RawK
  fieldDecodingTag : Nat
  rawAuthorityTag : Nat
deriving DecidableEq, Repr, Inhabited

/-- Compact exact coefficient progression over a half-open column range. -/
structure RawGeometricRun where
  columnStart : Nat
  length : Nat
  initial : Nat
  ratio : Nat
deriving DecidableEq, Repr, Inhabited

/-- One ordered port of a final selective row. Explicit terms and geometric
runs are additive streams; duplicate contributions are never silently
deduplicated by Lean. -/
structure RawPort where
  explicit : List RawTerm
  geometric : List RawGeometricRun
deriving DecidableEq, Repr, Inhabited

/-- Literal final selective row, including its unique emitted-run metadata.
`family` and `arm` are diagnostics and do not establish row semantics. -/
structure RawEmittedRow where
  schemaVersion : Nat
  rows : Nat
  columns : Nat
  emittedRow : Nat
  runIndex : Nat
  family : RawFamily
  arm : Option Nat
  ports : List RawPort
deriving DecidableEq, Repr

/-- Literal source-arm R1CS row before selective lowering. -/
structure RawSourceRow where
  schemaVersion : Nat
  rows : Nat
  columns : Nat
  sourceRow : Nat
  a : List RawTerm
  b : List RawTerm
  c : List RawTerm
deriving DecidableEq, Repr

/-- One source linear combination with its constant-one contribution kept
separate, matching the selective compiler's provenance surface. -/
structure RawLinearCombination where
  constant : Nat
  terms : List RawTerm
deriving DecidableEq, Repr, Inhabited

/-- Final low-norm slot retaining one source coordinate. -/
structure RawSourceSlot where
  column : Nat
  start : Nat
  width : Nat
deriving DecidableEq, Repr, Inhabited

/-- One compiler-validated linear substitution in source-column order. -/
structure RawSourceDefinition where
  target : Nat
  value : RawLinearCombination
deriving DecidableEq, Repr, Inhabited

/-- One scaled product in a compiler-introduced grouped product sum. -/
structure RawProductFactor where
  left : RawLinearCombination
  right : RawLinearCombination
  coefficient : Nat
deriving DecidableEq, Repr, Inhabited

/-- One retained grouped-product accumulator and its final slot. -/
structure RawDerivedProductSum where
  compilerIndex : Nat
  start : Nat
  width : Nat
  factors : List RawProductFactor
  previous : Option Nat
deriving DecidableEq, Repr, Inhabited

inductive RawRewriteKind where
  | polynomialEvaluation
  | productSum
deriving DecidableEq, Repr, Inhabited

inductive RawRewriteOutput where
  | source (value : RawLinearCombination)
  | derivedProductSum (compilerIndex : Nat)
deriving DecidableEq, Repr, Inhabited

/-- Exact recurrence reconstructed by one selectively emitted rewrite row. -/
structure RawRewriteStep where
  emittedRow : Nat
  rewriteId : Nat
  kind : RawRewriteKind
  sourceRows : List RawRowRange
  output : RawRewriteOutput
  base : RawLinearCombination
  previous : Option Nat
  factors : List RawProductFactor
deriving DecidableEq, Repr, Inhabited

/-- Exact source owner and A/B/C forms of one physically retained emitted
row. -/
structure RawRetainedStep where
  emittedRow : Nat
  sourceRow : Nat
  a : RawLinearCombination
  b : RawLinearCombination
  c : RawLinearCombination
deriving DecidableEq, Repr, Inhabited

/-- Prepared-layout disposition of one source column. -/
inductive RawSourceResolution where
  | constantOne
  | direct (start width : Nat) (centered : Bool)
  | decompositionAlias (source digit start : Nat) (centered : Bool)
  | equalityAlias (source start width : Nat) (centered : Bool)
  | linearDefinition
  | traceEliminated
deriving DecidableEq, Repr, Inhabited

structure RawSourceDecoder where
  column : Nat
  resolution : RawSourceResolution
deriving DecidableEq, Repr, Inhabited

/-- Verifier-owned source of one active public-coordinate write. -/
inductive RawActivePublicSource where
  | constantOne
  | sourceField (column : Nat)
  | fixedZero
deriving DecidableEq, Repr, Inhabited

/-- One exact public-write address and its block/lane decomposition under the
production `D = 54` packing. -/
structure RawPackedPublicCoordinate where
  schemaVersion : Nat
  column : Nat
  block : Nat
  lane : Nat
  source : RawActivePublicSource
deriving DecidableEq, Repr, Inhabited

/-- Exact active-branch pins exported from the same fixed-point projection as
the delayed combined-NC rows. The sparse selector rows carry their emitted-row
and run owner, so labels alone never establish these equations. -/
structure RawActivePins where
  schemaVersion : Nat
  sourceRows : Nat
  sourceColumns : Nat
  finalRows : Nat
  finalColumns : Nat
  constantOneColumn : Nat
  constantOneValue : Nat
  selectorColumns : List Nat
  recursiveSelectorValues : List Nat
  packedLaneCount : Nat
  packedBlockCount : Nat
  publicCoordinateCount : Nat
  selectorDomainRows : List RawEmittedRow
  oneHotRow : RawEmittedRow
deriving DecidableEq, Repr

/-- Exact source-arm schedule for one production quartic SumCheck round.
`columnMap` maps all 43 columns of that isolated verifier gadget into this
source relation. The named fields are retained independently so decoding can
reject a self-inconsistent map. -/
structure RawRoundMap where
  schemaVersion : Nat
  sourceRows : Nat
  sourceColumns : Nat
  roundIndex : Nat
  rowRange : RawRowRange
  firstAllocatedColumn : Nat
  allocatedColumns : List Nat
  coefficientColumns : List RawKColumns
  challengeColumns : RawKColumns
  claimInColumns : RawKColumns
  claimOutColumns : RawKColumns
  columnMap : List Nat
deriving DecidableEq, Repr

/-- Exact source-arm boundary of the steady recursive delayed combined-NC
check. The recursive exporter must reject an absent pending value before
constructing this record, so the two pending column families are not
optional here. -/
structure RawBoundaryMap where
  schemaVersion : Nat
  sourceRows : Nat
  sourceColumns : Nat
  constantOneColumn : Nat
  claimedInitialRows : RawRowRange
  terminalIdentityRows : RawRowRange
  terminalFinalEqualityRows : RawRowRange
  outputYZcolPaddingRows : List RawRowRange
  gammaColumns : RawKColumns
  betaLaneColumns : List RawKColumns
  betaBlockColumns : List RawKColumns
  producerBetaColumns : RawKColumns
  batchWeightColumns : RawKColumns
  pendingOldBlockColumns : List RawKColumns
  pendingParentYZcolColumns : List RawKColumns
  outputYZcolColumns : List (List RawKColumns)
  blockPointColumns : List RawKColumns
  lanePointColumns : List RawKColumns
  claimedInitialColumns : RawKColumns
  finalSumColumns : RawKColumns
  terminalRhsColumns : RawKColumns
deriving DecidableEq, Repr

/-! Fixed production profile checked by the Rust audit and independently
rechecked by the Lean decoder. These values describe this artifact only; they
are not global protocol constants or semantic authority. -/

def supportedSchemaVersion : Nat := 1
def selectivePortCount : Nat := 13
def activeLaneCount : Nat := 54
def laneBitCount : Nat := 6
def paddedLaneCount : Nat := 64
def paddingLaneCount : Nat := paddedLaneCount - activeLaneCount
def blockBitCount : Nat := 19
def outputCount : Nat := 15
def sumcheckRoundCount : Nat := blockBitCount + laneBitCount
def roundCoefficientCount : Nat := 5
def isolatedRoundColumnCount : Nat := 43
def isolatedRoundAllocatedCount : Nat := 28
def isolatedRoundRowCount : Nat := 30
def outputPaddingRowsPerOutput : Nat := 2 * paddingLaneCount

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
