import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PublicCarrier
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryPiDecArtifact

/-!
Exact prospective R1CS source layout for the ordered-commitment message.

Assurance tier: artifact-checked layout plus model-level value decoding.

Owns: the ten verifier-owned domain columns, parent-point coordinate columns,
fourteen child-commitment column blocks, their exact fixed-profile lengths,
and evaluation of a relabeled source list into the corresponding raw field
message.

Does not own: an emitted Rust hash call, constant-definition row membership,
the typed point/commitment decoder into `CommitmentFamilyPayload`, Poseidon2
rows, collision resistance, costs, or row removal.

Emits constraints: no.

Authority boundary: this module specifies what a future direct hash owner must
read. It does not claim that the current conservative per-child/aggregate Rust
serializer emits these columns. The caller must separately prove the ten
constant columns and the relevant local-to-global PiDEC column map.

| Stage path | Mathematical obligation | Authority class | Physical order | Lean owner |
|---|---|---|---|---|
| `fprime.accumulator.ordered_commitments.source.domain` | exact packed v1 tag | verifier-owned constant | ten fields | `domainConstantValues` |
| `fprime.accumulator.ordered_commitments.source.point` | common parent point | direct dataflow | coordinate-major `(c0,c1)` | `pointColumns` |
| `fprime.accumulator.ordered_commitments.source.children` | all child commitments in index order | direct dataflow | child-major flat `c_data` | `childCommitmentColumns` |
| `fprime.accumulator.ordered_commitments.source.complete` | one exact 13,620-column source list for this one-coordinate artifact | computed | domain, point, children | `expectedSourceColumns_length` |
| `fprime.accumulator.ordered_commitments.source.values` | relabeled columns evaluate to the independent raw message | derived | unchanged | `expectedSourceColumns_values` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentSourceLayout

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage
open Nightstream.SuperNeo.Concrete

/-- Values of the ten constant columns, in allocation/source order. -/
def domainConstantValues : List Nat := domainNats

def relabelColumns (columnMap columns : List Nat) : List Nat :=
  columns.map (Relabel.column columnMap)

def pairColumns (columns : List (Nat × Nat)) : List Nat :=
  columns.flatMap fun pair => [pair.1, pair.2]

/-- The common PiDEC parent point is the point carried by every strict child. -/
def pointColumns (columnMap : List Nat) : List Nat :=
  relabelColumns columnMap (pairColumns layout.parent.rCols)

/-- Exact child index order, retaining each fixed-profile flat commitment block
without copying shape headers. -/
def childCommitmentColumns (columnMap : List Nat) : List Nat :=
  layout.children.flatMap fun child =>
    relabelColumns columnMap child.commitment.dataCols

def expectedSourceColumns
    (columnMap constantColumns : List Nat) : List Nat :=
  constantColumns ++ pointColumns columnMap ++
    childCommitmentColumns columnMap

/-- Raw local-layout message values before canonical Goldilocks conversion. -/
def payloadNats (localAssignment : Nat → Nat) : List Nat :=
  domainNats ++
    (pairColumns layout.parent.rCols).map localAssignment ++
    layout.children.flatMap fun child =>
      child.commitment.dataCols.map localAssignment

/-- Canonical field-valued form of `payloadNats`. -/
def payloadFields (localAssignment : Nat → Nat) : List F :=
  (payloadNats localAssignment).map residue

@[simp] theorem domainConstantValues_length :
    domainConstantValues.length = 10 := by
  rw [domainConstantValues, domainNats_eq]
  rfl

@[simp] theorem pointColumns_length (columnMap : List Nat) :
    (pointColumns columnMap).length = 2 := by
  simp [pointColumns, relabelColumns, pairColumns, layout]

@[simp] theorem childCommitmentColumns_length (columnMap : List Nat) :
    (childCommitmentColumns columnMap).length = 13608 := by
  simp [childCommitmentColumns, relabelColumns, layout]

theorem expectedSourceColumns_length
    {columnMap constantColumns : List Nat}
    (constantLength : constantColumns.length = 10) :
    (expectedSourceColumns columnMap constantColumns).length = 13620 := by
  simp [expectedSourceColumns, constantLength]

/-- Relabeling changes only physical column identities. Once the constant
columns are pinned, evaluation recovers the exact independent raw message. -/
theorem expectedSourceColumns_values
    {columnMap constantColumns : List Nat}
    {assignment : Nat → Nat}
    (constants : constantColumns.map assignment = domainConstantValues) :
    (expectedSourceColumns columnMap constantColumns).map assignment =
      payloadNats (Relabel.assignment columnMap assignment) := by
  rw [expectedSourceColumns, List.map_append, List.map_append, constants]
  simp [domainConstantValues, payloadNats, pointColumns, List.map_flatMap,
    childCommitmentColumns, relabelColumns, Function.comp_def]
  rfl

theorem domainFields_eq_residues :
    domainFields = domainNats.map residue := by
  apply List.map_congr_left
  intro value _member
  apply Fin.ext
  rfl

/-- Field conversion of the checked physical source list is exactly the raw
Goldilocks message. The final typed-payload equality remains a separate decoder
refinement rather than an assumption hidden here. -/
theorem expectedSourceColumns_fields
    {columnMap constantColumns : List Nat}
    {assignment : Nat → Nat}
    (constants : constantColumns.map assignment = domainConstantValues) :
    ((expectedSourceColumns columnMap constantColumns).map assignment).map
        residue =
      payloadFields (Relabel.assignment columnMap assignment) := by
  rw [expectedSourceColumns_values constants]
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentSourceLayout
