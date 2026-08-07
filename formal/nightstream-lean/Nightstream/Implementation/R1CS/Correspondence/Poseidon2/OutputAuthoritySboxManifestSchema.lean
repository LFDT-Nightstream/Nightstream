import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.Sbox7OutputLayout

/-!
Contract: handwritten schema for the compact Rust-generated output-authority
Poseidon2 S-box call manifest.

Owns: call geometry, stage/hash/boundary ranges, isolated output offsets,
family and use censuses, candidate-column derivation, and explicit obligations
for global source-row matching and whole-matrix no-escape.

Does not own: generated production values, the source matrix, proof that a
Rust acceptance flag is true of that matrix, centered substitution, or
authorization to remove rows or slots.

Emits constraints: no.

Authority boundary: generated data is non-authoritative structure. The Rust
validator replays source rows and scans the complete matrix; Lean models those
obligations but cannot discharge them from summary flags alone.

Assurance tier: model-level.

| Schema branch | Mathematical obligation | Lean representation | Permits row removal? |
|---|---|---|---|
| `CallGeometry` | One renamed 600-row Poseidon2 call | `Poseidon2Call.Call` plus exact ABI metadata | no |
| `Boundaries` | Stage, prehash, hash, and digest ownership intervals | half-open ranges and boundary columns | no |
| `Census` | `5 * 86 = 430`, exact use-role counts | compact natural-number census | no |
| `WholeMatrixNoEscape` | Candidate outputs have exactly one C definition and eight A consumers globally | predicate over a complete matrix-use extractor | no |
| `RustEvidence` | Rust validators accepted exact rows and no-escape | non-authoritative evidence flags | no |
-/

namespace Nightstream.Implementation.R1CS.OutputAuthoritySboxManifest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sbox7OutputLayout

structure NatRange where
  start : Nat
  finish : Nat
deriving DecidableEq, Repr, Inhabited

def NatRange.width (range : NatRange) : Nat :=
  range.finish - range.start

def NatRange.columns (range : NatRange) : List Nat :=
  (List.range range.width).map (range.start + ·)

structure CallGeometry where
  traceIndex : Nat
  rowStart : Nat
  rowEnd : Nat
  inputColumns : List Nat
  firstAllocatedColumn : Nat
  allocatedColumnCount : Nat
  outputColumns : List Nat
deriving DecidableEq, Repr, Inhabited

def CallGeometry.toCall (call : CallGeometry) : Poseidon2Call.Call where
  rowStart := call.rowStart
  rowEnd := call.rowEnd
  inputColumns := call.inputColumns
  firstAllocatedColumn := call.firstAllocatedColumn

def CallGeometry.candidateColumns
    (call : CallGeometry) (isolatedOffsets : List Nat) : List Nat :=
  isolatedOffsets.map (call.firstAllocatedColumn + ·)

def CallGeometry.Valid (call : CallGeometry) : Prop :=
  call.rowEnd = call.rowStart + 600 ∧
    call.inputColumns.length = 8 ∧
    call.allocatedColumnCount = 600 ∧
    call.outputColumns =
      (List.range 8).map (call.firstAllocatedColumn + 592 + ·)

instance (call : CallGeometry) : Decidable call.Valid := by
  unfold CallGeometry.Valid
  infer_instance

def CallGeometry.Before (left right : CallGeometry) : Prop :=
  left.traceIndex < right.traceIndex ∧
    left.rowEnd ≤ right.rowStart ∧
    left.firstAllocatedColumn + left.allocatedColumnCount ≤
      right.firstAllocatedColumn

instance (left right : CallGeometry) : Decidable (left.Before right) := by
  unfold CallGeometry.Before
  infer_instance

structure FamilyRanges where
  initialExternal : NatRange
  partialRounds : NatRange
  terminalExternal : NatRange
deriving DecidableEq, Repr, Inhabited

def FamilyRanges.Valid (families : FamilyRanges) : Prop :=
  families.initialExternal = ⟨0, 32⟩ ∧
    families.partialRounds = ⟨32, 54⟩ ∧
    families.terminalExternal = ⟨54, 86⟩

instance (families : FamilyRanges) : Decidable families.Valid := by
  unfold FamilyRanges.Valid
  infer_instance

structure Boundaries where
  stageRows : NatRange
  stageColumns : NatRange
  prehashRows : NatRange
  prehashColumns : NatRange
  hashRows : NatRange
  hashZeroColumn : Nat
  hashOutputColumns : List Nat
  claimedDigestColumns : List Nat
  semanticStateOutputColumns : List Nat
  permutationTraceRange : NatRange
deriving DecidableEq, Repr, Inhabited

def Boundaries.protectedColumns (boundaries : Boundaries) : List Nat :=
  [boundaries.hashZeroColumn] ++
    boundaries.hashOutputColumns ++
    boundaries.claimedDigestColumns ++
    boundaries.semanticStateOutputColumns

def Boundaries.Valid (boundaries : Boundaries) : Prop :=
  boundaries.stageRows.width = 3034 ∧
    boundaries.stageColumns.width = 3034 ∧
    boundaries.prehashRows.start = boundaries.stageRows.start ∧
    boundaries.prehashRows.width = 12 ∧
    boundaries.prehashRows.finish = boundaries.hashRows.start ∧
    boundaries.hashRows.finish + 4 = boundaries.stageRows.finish ∧
    boundaries.prehashColumns.start = boundaries.stageColumns.start ∧
    boundaries.prehashColumns.width = 12 ∧
    boundaries.prehashColumns.finish = boundaries.hashZeroColumn ∧
    boundaries.claimedDigestColumns =
      (List.range 4).map (boundaries.claimedDigestColumns.headD 0 + ·) ∧
    boundaries.claimedDigestColumns.all (· < boundaries.stageColumns.start) = true ∧
    boundaries.semanticStateOutputColumns =
      (List.range 4).map (boundaries.stageColumns.finish - 4 + ·) ∧
    boundaries.hashOutputColumns.length = 4 ∧
    boundaries.permutationTraceRange.width = 5

instance (boundaries : Boundaries) : Decidable boundaries.Valid := by
  unfold Boundaries.Valid
  infer_instance

structure Census where
  scannedSourceRows : Nat
  scannedSourceColumns : Nat
  prehashBindingRows : Nat
  prehashFreshColumns : Nat
  hashInputFields : Nat
  fullAbsorbRounds : Nat
  partialAbsorbFields : Nat
  padRounds : Nat
  permutations : Nat
  sboxesPerPermutation : Nat
  initialExternalSboxes : Nat
  partialSboxes : Nat
  terminalExternalSboxes : Nat
  candidateSboxOutputs : Nat
  definitionCUses : Nat
  linearAUses : Nat
  totalMatrixUses : Nat
deriving DecidableEq, Repr, Inhabited

def Census.Valid (census : Census) : Prop :=
  census.prehashBindingRows = 12 ∧
    census.prehashFreshColumns = 12 ∧
    census.hashInputFields = 16 ∧
    census.fullAbsorbRounds = 4 ∧
    census.partialAbsorbFields = 0 ∧
    census.padRounds = 1 ∧
    census.permutations = 5 ∧
    census.sboxesPerPermutation = 86 ∧
    census.initialExternalSboxes = census.permutations * 32 ∧
    census.partialSboxes = census.permutations * 22 ∧
    census.terminalExternalSboxes = census.permutations * 32 ∧
    census.candidateSboxOutputs =
      census.permutations * census.sboxesPerPermutation ∧
    census.definitionCUses = census.candidateSboxOutputs ∧
    census.linearAUses = census.candidateSboxOutputs * 8 ∧
    census.totalMatrixUses =
      census.definitionCUses + census.linearAUses

instance (census : Census) : Decidable census.Valid := by
  unfold Census.Valid
  infer_instance

/-- These booleans report that the Rust validator accepted. They are evidence
metadata, not proofs of the corresponding predicates below. -/
structure RustEvidence where
  exactCallRowsAccepted : Bool
  wholeMatrixNoEscapeAccepted : Bool
deriving DecidableEq, Repr, Inhabited

structure Manifest where
  schemaVersion : Nat
  boundaries : Boundaries
  calls : List CallGeometry
  isolatedOutputOffsets : List Nat
  families : FamilyRanges
  census : Census
  rustEvidence : RustEvidence
deriving DecidableEq, Repr, Inhabited

def candidateColumnsFor : List CallGeometry → List Nat → List Nat
  | [], _ => []
  | call :: rest, isolatedOffsets =>
      call.candidateColumns isolatedOffsets ++
        candidateColumnsFor rest isolatedOffsets

def Manifest.candidateColumns (manifest : Manifest) : List Nat :=
  candidateColumnsFor manifest.calls manifest.isolatedOutputOffsets

def Manifest.CandidatesDisjointFromBoundaries (manifest : Manifest) : Prop :=
  ∀ column ∈ manifest.candidateColumns,
    column ∉ manifest.boundaries.protectedColumns

instance (manifest : Manifest) :
    Decidable manifest.CandidatesDisjointFromBoundaries := by
  unfold Manifest.CandidatesDisjointFromBoundaries
  infer_instance

/-- Ordered adjacency is enough for the generated 5-call schedule; it avoids
quadratic all-pairs generated facts. -/
def callsAdjacent : List CallGeometry → Bool
  | [] => true
  | [_] => true
  | left :: right :: rest =>
      decide (left.Before right) && callsAdjacent (right :: rest)

/-- Linear-time order predicate used by the generated certificate. -/
def columnsStrictlyIncreasing : List Nat → Bool
  | [] => true
  | [_] => true
  | left :: right :: rest =>
      decide (left < right) && columnsStrictlyIncreasing (right :: rest)

/-- Kernel-checked shape expected from a promoted generated manifest. The two
Rust evidence fields remain declarations about an external validator. -/
structure Manifest.Certificate (manifest : Manifest) : Prop where
  schemaVersion : manifest.schemaVersion = 1
  boundariesValid : manifest.boundaries.Valid
  everyCallValid : ∀ call ∈ manifest.calls, call.Valid
  callsAdjacent : callsAdjacent manifest.calls = true
  traceOrder : manifest.calls.map CallGeometry.traceIndex =
    (List.range manifest.boundaries.permutationTraceRange.width).map
      (manifest.boundaries.permutationTraceRange.start + ·)
  offsetsExact : manifest.isolatedOutputOffsets =
    Poseidon2Sbox7OutputLayout.outputColumns.map (· - 9)
  offsetsInAllocatedRange :
    ∀ offset ∈ manifest.isolatedOutputOffsets, offset < 600
  familiesValid : manifest.families.Valid
  censusValid : manifest.census.Valid
  callCount : manifest.calls.length = manifest.census.permutations
  offsetCount : manifest.isolatedOutputOffsets.length =
    manifest.census.sboxesPerPermutation
  candidateColumnsIncreasing :
    columnsStrictlyIncreasing manifest.candidateColumns = true
  boundaryDisjoint : manifest.CandidatesDisjointFromBoundaries
  exactRowsEvidence : manifest.rustEvidence.exactCallRowsAccepted = true
  noEscapeEvidence : manifest.rustEvidence.wholeMatrixNoEscapeAccepted = true

/-- Exact global source-row identity remains an obligation over actual program
rows. The generated Rust flag is not a proof of this proposition. -/
def Manifest.SourceCallRowsMatch
    (manifest : Manifest) (programRows : List Row) : Prop :=
  ∀ call ∈ manifest.calls, call.toCall.Matches programRows

inductive MatrixUseRole where
  | linearA
  | multiplicativeB
  | definitionC
deriving DecidableEq, Repr

def expectedCandidateUses : List MatrixUseRole :=
  [.definitionC] ++ List.replicate 8 .linearA

/-- `completeUses column` must enumerate every normalized A/B/C occurrence of
that source column across the complete matrix. Without such an extractor this
predicate cannot be concluded from generated summary data. -/
def Manifest.WholeMatrixNoEscape
    (manifest : Manifest)
    (completeUses : Nat → List MatrixUseRole) : Prop :=
  ∀ column ∈ manifest.candidateColumns,
    completeUses column = expectedCandidateUses

end Nightstream.Implementation.R1CS.OutputAuthoritySboxManifest
