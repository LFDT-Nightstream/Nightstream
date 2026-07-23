import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.RoundMaps
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.SourceRows.Round
import Nightstream.Implementation.R1CS.Artifacts.Projection.IndexedRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Decoder
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionRound

/-!
Exact production-row certificates for the 25 delayed combined-NC SumCheck
rounds.

Owns: successful decoding of every generated round map, exact source-row
indices for every 30-row interval, canonical sparse coefficients, and literal
A/B/C equality with the independently defined five-coefficient production
round after the generated column renaming.

Does not own: source-to-selective rewrite refinement, row satisfaction,
transcript order, quartic high-slot zeroes, terminal-NC semantics, parent or
raw-child authority, commitment binding, costs, or row removal.

Emits constraints: none. The checked source multiplicity is exactly
`25 * 30`; this file makes no claim about rows outside those intervals.

Assurance tier: artifact-checked for the generated fixed production profile.
Generated family labels and interval lengths are not trusted: each
certificate compares the actual sparse coefficients and columns.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.round_artifact` | Check exact generated ownership and coefficients for one materialized round family. | checked artifact |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RoundArtifact

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Decoder
open Nightstream.Implementation.R1CS.ProjectionIndexedRows

/-- Proof-free coefficient projection used by the bounded certificates. -/
def rawTerms (terms : List RawTerm) : List (Nat × Nat) :=
  terms.map fun term => (term.column, term.coefficient)

/-- Proof-free A/B/C projection. Row metadata is checked separately. -/
def rawRow (row : RawSourceRow) : Row where
  a := rawTerms row.a
  b := rawTerms row.b
  c := rawTerms row.c

def rawRows (rows : List RawSourceRow) : List Row :=
  rows.map rawRow

private def rowsPermutationEquivalentListDecidable :
    (source reconstructed : List Row) ->
      Decidable (RowsPermutationEquivalentList source reconstructed)
  | [], [] => isTrue True.intro
  | [], _ :: _ => isFalse id
  | _ :: _, [] => isFalse id
  | source :: sources, reconstructed :: reconstructions =>
      match inferInstanceAs
          (Decidable (RowsPermutationEquivalent source reconstructed)),
        rowsPermutationEquivalentListDecidable sources reconstructions with
      | isTrue head, isTrue tail => isTrue ⟨head, tail⟩
      | isFalse head, isTrue _ => isFalse fun equivalent => head equivalent.1
      | isTrue _, isFalse tail => isFalse fun equivalent => tail equivalent.2
      | isFalse head, isFalse _ => isFalse fun equivalent => head equivalent.1

instance (source reconstructed : List Row) :
    Decidable (RowsPermutationEquivalentList source reconstructed) :=
  rowsPermutationEquivalentListDecidable source reconstructed

def RawTermValid (columns : Nat) (term : RawTerm) : Prop :=
  term.column < columns ∧
  term.coefficient < goldilocksModulus ∧
  term.coefficient ≠ 0

instance (columns : Nat) (term : RawTerm) :
    Decidable (RawTermValid columns term) := by
  unfold RawTermValid
  infer_instance

/-- Fail-closed structural conditions needed to interpret one exported sparse
row without modular aliases, zero terms, or escaping coordinates. -/
def RawSourceRowValid (row : RawSourceRow) : Prop :=
  row.schemaVersion = supportedSchemaVersion ∧
  0 < row.rows ∧
  0 < row.columns ∧
  row.sourceRow < row.rows ∧
  ∀ term ∈ row.a ++ row.b ++ row.c, RawTermValid row.columns term

instance (row : RawSourceRow) : Decidable (RawSourceRowValid row) := by
  unfold RawSourceRowValid RawTermValid
  infer_instance

/-- All data checked for one generated round. The final conjunct is the
load-bearing coefficient equality; preceding fields reject valid-looking
rows attached to the wrong physical interval or relation shape. -/
def RoundArtifactValid (index : Nat) (rows : List RawSourceRow)
    (round : RawRoundMap) : Prop :=
  roundMapValid round ∧
  round.roundIndex = index ∧
  rows.length = isolatedRoundRowCount ∧
  rows.map RawSourceRow.sourceRow =
    (List.range isolatedRoundRowCount).map
      (fun offset => round.rowRange.start + offset) ∧
  (∀ row ∈ rows,
    RawSourceRowValid row ∧
    row.rows = round.sourceRows ∧
    row.columns = round.sourceColumns) ∧
  RowsPermutationEquivalentList (rawRows rows)
    (ProductionRound.rows.map (Relabel.row round.columnMap))

instance (index : Nat) (rows : List RawSourceRow) (round : RawRoundMap) :
    Decidable (RoundArtifactValid index rows round) := by
  unfold RoundArtifactValid RawSourceRowValid RawTermValid
  infer_instance

/-- Lookup is part of the certificate, so a missing or reordered map fails
closed rather than selecting a caller-provided replacement. -/
def Certificate (index : Nat) (rows : List RawSourceRow) : Prop :=
  match RoundMaps.values[index]? with
  | none => False
  | some round => RoundArtifactValid index rows round

instance (index : Nat) (rows : List RawSourceRow) :
    Decidable (Certificate index rows) := by
  unfold Certificate
  cases RoundMaps.values[index]? <;> infer_instance

theorem certificate_lookup {index : Nat} {rows : List RawSourceRow}
    (certificate : Certificate index rows) :
    ∃ round,
      RoundMaps.values[index]? = some round ∧
      RoundArtifactValid index rows round := by
  cases lookup : RoundMaps.values[index]? with
  | none =>
      simp [Certificate, lookup] at certificate
  | some round =>
      exact ⟨round, rfl,
        by simpa [Certificate, lookup] using certificate⟩

theorem decodeRoundMap_of_valid {round : RawRoundMap}
    (valid : roundMapValid round) :
    ∃ decoded, decodeRoundMap round = some decoded := by
  refine ⟨⟨round, valid⟩, ?_⟩
  simp [decodeRoundMap, valid]

/-- A bounded round certificate yields both the generated lookup identity and
successful fail-closed map decoding. -/
theorem certificate_map_decodes {index : Nat} {rows : List RawSourceRow}
    (certificate : Certificate index rows) :
    ∃ round decoded,
      RoundMaps.values[index]? = some round ∧
      decodeRoundMap round = some decoded := by
  rcases certificate_lookup certificate with ⟨round, lookup, valid⟩
  rcases decodeRoundMap_of_valid valid.1 with ⟨decoded, decodes⟩
  exact ⟨round, decoded, lookup, decodes⟩

/-- Kernel-level projection of the exact-row field, reusable without
re-evaluating a generated certificate. -/
theorem certificate_exact_rows {index : Nat} {rows : List RawSourceRow}
    (certificate : Certificate index rows) :
    ∃ round,
      RoundMaps.values[index]? = some round ∧
      RowsPermutationEquivalentList (rawRows rows)
        (ProductionRound.rows.map (Relabel.row round.columnMap)) := by
  rcases certificate_lookup certificate with ⟨round, lookup, valid⟩
  exact ⟨round, lookup, valid.2.2.2.2.2⟩

/-! The map census evaluates exactly 25 proof-free `RawRoundMap` records.
Every map contains 43 mapped columns, 28 allocated columns, and five
coefficient pairs. It does not construct a list of decoded maps. -/

def GeneratedRoundMapsValid : Prop :=
  RoundMaps.values.length = sumcheckRoundCount ∧
  RoundMaps.values.map RawRoundMap.roundIndex =
    List.range sumcheckRoundCount ∧
  ∀ round ∈ RoundMaps.values, roundMapValid round

instance : Decidable GeneratedRoundMapsValid := by
  unfold GeneratedRoundMapsValid
  infer_instance

set_option maxRecDepth 100000 in
theorem generatedRoundMapsValid : GeneratedRoundMapsValid := by
  native_decide

/-- Every one of the 25 generated maps has a typed fail-closed decoding. -/
theorem generatedRoundMapsDecode (round : RawRoundMap)
    (member : round ∈ RoundMaps.values) :
    ∃ decoded, decodeRoundMap round = some decoded := by
  exact decodeRoundMap_of_valid (generatedRoundMapsValid.2.2 round member)

/-! Each following definition contains exactly 30 proof-free
`RawSourceRow` records. Cross-shard intervals take only the necessary tail and
prefix; no theorem filters or decodes the aggregate 8,021-row source list. -/

private def within (rows : List RawSourceRow) (offset : Nat) :
    List RawSourceRow :=
  (rows.drop offset).take isolatedRoundRowCount

private def across (left right : List RawSourceRow) (offset : Nat) :
    List RawSourceRow :=
  ((left.drop offset) ++ right).take isolatedRoundRowCount

def round0Rows : List RawSourceRow := within SourceRows.Chunk5.values 36
def round1Rows : List RawSourceRow := within SourceRows.Chunk5.values 66
def round2Rows : List RawSourceRow := within SourceRows.Chunk5.values 96
def round3Rows : List RawSourceRow :=
  across SourceRows.Chunk5.values SourceRows.Chunk6.values 126
def round4Rows : List RawSourceRow := within SourceRows.Chunk6.values 28
def round5Rows : List RawSourceRow := within SourceRows.Chunk6.values 58
def round6Rows : List RawSourceRow := within SourceRows.Chunk6.values 88
def round7Rows : List RawSourceRow :=
  across SourceRows.Chunk6.values SourceRows.Chunk7.values 118
def round8Rows : List RawSourceRow := within SourceRows.Chunk7.values 20
def round9Rows : List RawSourceRow := within SourceRows.Chunk7.values 50
def round10Rows : List RawSourceRow := within SourceRows.Chunk7.values 80
def round11Rows : List RawSourceRow :=
  across SourceRows.Chunk7.values SourceRows.Chunk8.values 110
def round12Rows : List RawSourceRow := within SourceRows.Chunk8.values 12
def round13Rows : List RawSourceRow := within SourceRows.Chunk8.values 42
def round14Rows : List RawSourceRow := within SourceRows.Chunk8.values 72
def round15Rows : List RawSourceRow :=
  across SourceRows.Chunk8.values SourceRows.Chunk9.values 102
def round16Rows : List RawSourceRow := within SourceRows.Chunk9.values 4
def round17Rows : List RawSourceRow := within SourceRows.Chunk9.values 34
def round18Rows : List RawSourceRow := within SourceRows.Chunk9.values 64
def round19Rows : List RawSourceRow := within SourceRows.Chunk9.values 94
def round20Rows : List RawSourceRow :=
  across SourceRows.Chunk9.values SourceRows.Chunk10.values 124
def round21Rows : List RawSourceRow := within SourceRows.Chunk10.values 26
def round22Rows : List RawSourceRow := within SourceRows.Chunk10.values 56
def round23Rows : List RawSourceRow := within SourceRows.Chunk10.values 86
def round24Rows : List RawSourceRow :=
  across SourceRows.Chunk10.values SourceRows.Chunk11.values 116

/-! Each certificate evaluates exactly its 30 proof-free source-row records
and one proof-free round map. Sparse A/B/C terms are retained verbatim. -/

set_option maxRecDepth 100000 in
theorem round0 : Certificate 0 round0Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round1 : Certificate 1 round1Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round2 : Certificate 2 round2Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round3 : Certificate 3 round3Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round4 : Certificate 4 round4Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round5 : Certificate 5 round5Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round6 : Certificate 6 round6Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round7 : Certificate 7 round7Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round8 : Certificate 8 round8Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round9 : Certificate 9 round9Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round10 : Certificate 10 round10Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round11 : Certificate 11 round11Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round12 : Certificate 12 round12Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round13 : Certificate 13 round13Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round14 : Certificate 14 round14Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round15 : Certificate 15 round15Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round16 : Certificate 16 round16Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round17 : Certificate 17 round17Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round18 : Certificate 18 round18Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round19 : Certificate 19 round19Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round20 : Certificate 20 round20Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round21 : Certificate 21 round21Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round22 : Certificate 22 round22Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round23 : Certificate 23 round23Rows := by native_decide
set_option maxRecDepth 100000 in
theorem round24 : Certificate 24 round24Rows := by native_decide

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RoundArtifact
