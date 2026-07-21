import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Decoder
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.BoundaryArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionRound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Semantics

/-!
Executable semantics for decoded production combined-NC row data.

Owns: lossless geometric-run expansion, exact source A/B/C conversion to the
repository `R1CS.Row`, thirteen-port selective-polynomial evaluation, exact
affine copies of the isolated ten-coefficient production-round program, and
typed reads of the delayed boundary columns.

Does not own: generated-row truth, source-to-emitted rewrite refinement,
assignment decoding, selector truth, transcript scheduling, raw-child or
parent authority, commitment binding, costs, or row removal.

Emits constraints: none.

This module interprets row data only. A later generated-artifact leaf must
prove that these decoded rows are the rows selected from production before
using the satisfaction predicates below.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Decoder

/-- Stable typed view of the exact generated production boundary. Its fields
remain columns only; the semantic readers below assign no authority by
themselves. -/
def productionBoundary : DecodedBoundaryMap :=
  BoundaryArtifact.decodedBoundary

theorem productionBoundary_raw :
    productionBoundary.raw = Generated.Metadata.boundary :=
  BoundaryArtifact.decodedBoundary_raw

def fieldResidue (value : Nat) : F :=
  ⟨value % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

def termAsNatTerm {columns : Nat} (term : DecodedTerm columns) :
    Nat × Nat :=
  (term.column.val, term.coefficient.val)

def termsAsNatTerms {columns : Nat} (terms : List (DecodedTerm columns)) :
    List (Nat × Nat) :=
  terms.map termAsNatTerm

/-- Offset `i` contributes `initial * ratio^i` at `columnStart + i`. -/
def expandedRunNatTerms {columns : Nat}
    (run : DecodedGeometricRun columns) : List (Nat × Nat) :=
  (List.finRange run.length).map fun offset =>
    ((run.column offset).val, (run.initial * run.ratio ^ offset.val).val)

/-- Exact additive contribution stream of one decoded selective port. -/
def expandedNatTerms {columns : Nat} (port : DecodedPort columns) :
    List (Nat × Nat) :=
  termsAsNatTerms port.explicit ++
    port.geometric.flatMap expandedRunNatTerms

def portValue {columns : Nat} (port : DecodedPort columns)
    (assignment : Nat → Nat) : F :=
  fieldResidue (lcEval assignment (expandedNatTerms port))

def emittedPoint (row : DecodedEmittedRow) (assignment : Nat → Nat) :
    Fin selectivePortCount → F :=
  fun port => portValue (row.port port) assignment

/-- Residual of the exact independent thirteen-port selective polynomial. -/
def emittedResidual (row : DecodedEmittedRow) (assignment : Nat → Nat) : F :=
  evaluate (emittedPoint row assignment)

def EmittedRowHolds (row : DecodedEmittedRow)
    (assignment : Nat → Nat) : Prop :=
  emittedResidual row assignment = 0

instance (row : DecodedEmittedRow) (assignment : Nat → Nat) :
    Decidable (EmittedRowHolds row assignment) := by
  unfold EmittedRowHolds
  infer_instance

def EmittedRowsSatisfy (rows : List DecodedEmittedRow)
    (assignment : Nat → Nat) : Prop :=
  ∀ row ∈ rows, EmittedRowHolds row assignment

def sourceRowToRow (row : DecodedSourceRow) : Row where
  a := termsAsNatTerms row.a
  b := termsAsNatTerms row.b
  c := termsAsNatTerms row.c

/-- Source-row satisfaction is exactly the repository's executable R1CS
semantics; no parallel acceptance predicate is introduced. -/
def SourceRowHolds (row : DecodedSourceRow)
    (assignment : Nat → Nat) : Prop :=
  RowHolds assignment (sourceRowToRow row)

theorem sourceRowHolds_iff (row : DecodedSourceRow)
    (assignment : Nat → Nat) :
    SourceRowHolds row assignment ↔
      RowHolds assignment (sourceRowToRow row) :=
  Iff.rfl

def sourceRows (rows : List DecodedSourceRow) : List Row :=
  rows.map sourceRowToRow

def SourceRowsSatisfy (rows : List DecodedSourceRow)
    (assignment : Nat → Nat) : Prop :=
  Satisfies (sourceRows rows) assignment

theorem sourceRowsSatisfy_iff (rows : List DecodedSourceRow)
    (assignment : Nat → Nat) :
    SourceRowsSatisfy rows assignment ↔
      ∀ row ∈ rows, SourceRowHolds row assignment := by
  constructor
  · intro satisfies row member
    exact satisfies (sourceRowToRow row)
      (List.mem_map.mpr ⟨row, member, rfl⟩)
  · intro holds mapped member
    rcases List.mem_map.mp member with ⟨row, rowMember, rfl⟩
    exact holds row rowMember

/-- Source-linear-combination carrier used by compiler provenance. Constants
are represented on the standard constant-one column zero. -/
def linearCombinationTerms {columns : Nat}
    (value : DecodedLinearCombination columns) : List (Nat × Nat) :=
  (0, value.constant.val) :: termsAsNatTerms value.terms

def linearCombinationValue {columns : Nat}
    (value : DecodedLinearCombination columns)
    (assignment : Nat → Nat) : F :=
  fieldResidue (lcEval assignment (linearCombinationTerms value))

def productFactorValue {columns : Nat} (factor : DecodedProductFactor columns)
    (assignment : Nat → Nat) : F :=
  factor.coefficient *
    linearCombinationValue factor.left assignment *
    linearCombinationValue factor.right assignment

def rawKColumnsToColumns (columns : RawKColumns) :
    ProjectionProgram.KColumns :=
  ⟨columns.c0, columns.c1⟩

def rawKColumnsValue (columns : RawKColumns) (assignment : Nat → Nat) :
    ProjectionProgram.K :=
  (rawKColumnsToColumns columns).value assignment

/-- Exact mapped 30-row program for one decoded production round. -/
def roundMapRows (round : DecodedRoundMap) : List Row :=
  ProductionRound.rows.map
    (Relabel.row round.raw.columnMap)

def RoundMapHolds (round : DecodedRoundMap)
    (assignment : Nat → Nat) : Prop :=
  Satisfies (roundMapRows round) assignment

theorem roundMapMapsOne (round : DecodedRoundMap) :
    Relabel.column round.raw.columnMap 0 = 0 := by
  simpa [Relabel.column] using round.valid.2.2.2.1

/-- Generic kernel bridge for a decoded exact round map. Generated data must
still prove satisfaction of `round.rows`; the map itself grants no acceptance. -/
theorem roundMapAccepted_of_holds (round : DecodedRoundMap)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : RoundMapHolds round assignment) :
    ProductionRound.Accepted
      (Relabel.assignment round.raw.columnMap assignment) := by
  exact ProductionRound.mapped_sound round.raw.columnMap
    (roundMapMapsOne round)
    canonical one holds

def roundColumnMaps (rounds : List DecodedRoundMap) : List (List Nat) :=
  rounds.map fun round => round.raw.columnMap

def roundRows (rounds : List DecodedRoundMap) : List Row :=
  (rounds.map roundMapRows).flatten

def RoundRowsSatisfy (rounds : List DecodedRoundMap)
    (assignment : Nat → Nat) : Prop :=
  Satisfies (roundRows rounds) assignment

theorem roundRowsSatisfy_iff (rounds : List DecodedRoundMap)
    (assignment : Nat → Nat) :
    RoundRowsSatisfy rounds assignment ↔
      ∀ round ∈ rounds, RoundMapHolds round assignment := by
  rw [RoundRowsSatisfy, roundRows,
    satisfies_flatten_iff (rounds.map roundMapRows) assignment]
  constructor
  · intro pieces round member
    exact pieces (roundMapRows round)
      (List.mem_map.mpr ⟨round, member, rfl⟩)
  · intro holds piece member
    rcases List.mem_map.mp member with ⟨round, roundMember, rfl⟩
    exact holds round roundMember

def boundaryGamma (boundary : DecodedBoundaryMap)
    (assignment : Nat → Nat) : ProjectionProgram.K :=
  rawKColumnsValue boundary.raw.gammaColumns assignment

def boundaryProducerBeta (boundary : DecodedBoundaryMap)
    (assignment : Nat → Nat) : ProjectionProgram.K :=
  rawKColumnsValue boundary.raw.producerBetaColumns assignment

def boundaryBatchWeight (boundary : DecodedBoundaryMap)
    (assignment : Nat → Nat) : ProjectionProgram.K :=
  rawKColumnsValue boundary.raw.batchWeightColumns assignment

def boundaryClaimedInitial (boundary : DecodedBoundaryMap)
    (assignment : Nat → Nat) : ProjectionProgram.K :=
  rawKColumnsValue boundary.raw.claimedInitialColumns assignment

def boundaryFinalSum (boundary : DecodedBoundaryMap)
    (assignment : Nat → Nat) : ProjectionProgram.K :=
  rawKColumnsValue boundary.raw.finalSumColumns assignment

def boundaryTerminalRhs (boundary : DecodedBoundaryMap)
    (assignment : Nat → Nat) : ProjectionProgram.K :=
  rawKColumnsValue boundary.raw.terminalRhsColumns assignment

def boundaryBetaLane (boundary : DecodedBoundaryMap)
    (assignment : Nat → Nat) : List ProjectionProgram.K :=
  boundary.raw.betaLaneColumns.map fun columns =>
    rawKColumnsValue columns assignment

def boundaryBetaBlock (boundary : DecodedBoundaryMap)
    (assignment : Nat → Nat) : List ProjectionProgram.K :=
  boundary.raw.betaBlockColumns.map fun columns =>
    rawKColumnsValue columns assignment

def boundaryPendingOldBlock (boundary : DecodedBoundaryMap)
    (assignment : Nat → Nat) : List ProjectionProgram.K :=
  boundary.raw.pendingOldBlockColumns.map fun columns =>
    rawKColumnsValue columns assignment

def boundaryPendingParentYZcol (boundary : DecodedBoundaryMap)
    (assignment : Nat → Nat) : List ProjectionProgram.K :=
  boundary.raw.pendingParentYZcolColumns.map fun columns =>
    rawKColumnsValue columns assignment

def boundaryOutputYZcol (boundary : DecodedBoundaryMap)
    (assignment : Nat → Nat) : List (List ProjectionProgram.K) :=
  boundary.raw.outputYZcolColumns.map fun output =>
    output.map fun columns => rawKColumnsValue columns assignment

def boundaryBlockPoint (boundary : DecodedBoundaryMap)
    (assignment : Nat → Nat) : List ProjectionProgram.K :=
  boundary.raw.blockPointColumns.map fun columns =>
    rawKColumnsValue columns assignment

def boundaryLanePoint (boundary : DecodedBoundaryMap)
    (assignment : Nat → Nat) : List ProjectionProgram.K :=
  boundary.raw.lanePointColumns.map fun columns =>
    rawKColumnsValue columns assignment

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Semantics
