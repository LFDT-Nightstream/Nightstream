import Nightstream.Implementation.NebulaV2.AuthoritativeStateOutputRows
import Nightstream.Implementation.NebulaV2.CompactChainHeaderRows
import Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
import Nightstream.Implementation.NebulaV2.FPrimeIterationInputRows
import Nightstream.Implementation.NebulaV2.InitialMemoryCarryRows
import Nightstream.Implementation.NebulaV2.MemoryCarryPublicRows
import Nightstream.Implementation.NebulaV2.MemoryOpenSegmentBlockRows
import Nightstream.Implementation.R1CS.Canonical.ColumnWindows

/-!
Contract: parametric generated-row manifest schema for the Nebula V2 base branch.

Assurance tier: implementation schema.

Owns the exact base row-family order, exact row cover by construction, the
canonical compact-chain headers, the verifier-authoritative initial memory
carry, the complete segment-open block, the active outgoing carry, the
outgoing state hash, and non-overlapping target-column ownership windows.

Does not own final absolute rows or columns, semantic refinement of the named
base challenge-authority family, application/fresh-claim construction,
accumulator arithmetic, control refinement, computation of the initial-memory
root from the public statement, recursive-size closure, or Rust conformance.
A generated V2 artifact must instantiate this schema.

Emits constraints: through the contained local row programs.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.BaseManifestSchema

open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.ColumnWindows
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

inductive Owner where
  | prelude
  | compactHeaders
  | initialCarryValidation
  | challengeAuthority
  | segmentOpening
  | outgoingCarryValidation
  | stateOutput
  | claimConstruction
  | accumulator
  | counterAndControl
deriving DecidableEq, Fintype, Repr

def ownerOrder : List Owner :=
  [.prelude, .compactHeaders, .initialCarryValidation, .challengeAuthority,
    .segmentOpening, .outgoingCarryValidation, .stateOutput,
    .claimConstruction, .accumulator, .counterAndControl]

theorem ownerOrder_nodup : ownerOrder.Nodup := by decide

theorem Owner.mem_order (owner : Owner) : owner ∈ ownerOrder := by
  cases owner <;> simp [ownerOrder]

/-- These compiler row families remain explicit release obligations. Their
nonempty shape does not grant them any semantic theorem. -/
structure OpaqueRows where
  prelude : List Row
  /-- Mandatory generated rows that must recompute the canonical base input
  digest and base successor-prefix digest, then link all 28 authority lanes
  to the segment-open transcript. -/
  challengeAuthority : List Row
  claimConstruction : List Row
  accumulator : List Row
  counterAndControl : List Row
  challengeAuthorityNonempty : challengeAuthority ≠ []
  claimConstructionNonempty : claimConstruction ≠ []
  accumulatorNonempty : accumulator ≠ []
  counterAndControlNonempty : counterAndControl ≠ []

structure Layouts where
  compactHeaders : CompactChainHeaderRows.Layout
  initialMemoryCarry : MemoryCarryPublicRows.Layout
  initialAuthority : InitialMemoryCarryRows.Layout
  opening : MemoryOpenSegmentBlockRows.Layout
  outgoingMemoryCarry : MemoryCarryPublicRows.Layout
  stateOutput : AuthoritativeStateOutputRows.Layout
  baseIteration : FPrimeIterationInputRows.Layout

structure Layouts.Valid
    (profile : Nightstream.Protocol.NebulaV2.Profile.Identity)
    (layouts : Layouts)
    (seedManifest : SeedSchedule.Manifest) : Prop where
  profileCanonical : MemoryTranscriptHashFrame.ProfileCanonical profile
  compactHeadersValid : layouts.compactHeaders.Valid seedManifest
  initialAuthorityUsesInitialCarry :
    layouts.initialAuthority.carry = layouts.initialMemoryCarry
  openingValid :
    MemoryOpenSegmentBlockRows.ProfileIndexed.Valid profile layouts.opening
  openingUsesInitialCarry :
    layouts.opening.before = layouts.initialMemoryCarry
  openingUsesOutgoingCarry :
    layouts.opening.after = layouts.outgoingMemoryCarry
  operationsHeaderUsesInitialCarry :
    layouts.compactHeaders.operations.digestColumn =
      layouts.initialMemoryCarry.carry.headerColumn .operations
  memoryHeaderUsesInitialCarryInitial :
    layouts.compactHeaders.memory.digestColumn =
      layouts.initialMemoryCarry.carry.headerColumn .initialSnapshot
  memoryHeaderUsesInitialCarryFinal :
    layouts.compactHeaders.memory.digestColumn =
      layouts.initialMemoryCarry.carry.headerColumn .finalSnapshot
  operationsHeaderUsesOutgoingCarry :
    layouts.compactHeaders.operations.digestColumn =
      layouts.outgoingMemoryCarry.carry.headerColumn .operations
  memoryHeaderUsesOutgoingInitial :
    layouts.compactHeaders.memory.digestColumn =
      layouts.outgoingMemoryCarry.carry.headerColumn .initialSnapshot
  memoryHeaderUsesOutgoingFinal :
    layouts.compactHeaders.memory.digestColumn =
      layouts.outgoingMemoryCarry.carry.headerColumn .finalSnapshot
  carryBitsSharedWithStateOutput :
    layouts.outgoingMemoryCarry.publicBits =
      layouts.stateOutput.hash.carry.frame.packing.publicBits
  stateOutputValid : layouts.stateOutput.Valid

structure Artifact (widths : CompilerWidths) where
  profile : Nightstream.Protocol.NebulaV2.Profile.Identity
  rowVariableCount : Nat
  /-- The V2 reference's known rows rule out every cube below 25 variables.
  Generation may select a larger exponent after all omitted row families are
  present; this schema must not freeze a planning value as a key identity. -/
  rowVariableCountMinimum : 25 ≤ rowVariableCount
  verifierKeyDigest : Nightstream.Protocol.NebulaV2.Digest.Value
  relationManifestDigest : Nightstream.Protocol.NebulaV2.Digest.Value
  seedManifest : SeedSchedule.Manifest
  seedManifestProfile : seedManifest.profile = profile
  other : OpaqueRows
  layouts : Layouts
  layoutsValid : layouts.Valid profile seedManifest
  columnWidths : List Nat
  columnWidthCount : columnWidths.length = ownerOrder.length
  columnWidthsPositive : ∀ width ∈ columnWidths, 0 < width
  targetColumnsCovered :
    ∀ row ∈
        (ownerOrder.flatMap fun owner =>
          match owner with
          | .prelude => other.prelude
          | .compactHeaders =>
              CompactChainHeaderRows.rows seedManifest layouts.compactHeaders
          | .initialCarryValidation =>
              MemoryCarryPublicRows.rows layouts.initialMemoryCarry ++
                InitialMemoryCarryRows.rows layouts.initialAuthority
          | .challengeAuthority => other.challengeAuthority
          | .segmentOpening =>
              MemoryOpenSegmentBlockRows.ProfileIndexed.rows profile
                layouts.opening
          | .outgoingCarryValidation =>
              MemoryCarryPublicRows.rows layouts.outgoingMemoryCarry
          | .stateOutput =>
              AuthoritativeStateOutputRows.rows layouts.stateOutput
          | .claimConstruction => other.claimConstruction
          | .accumulator => other.accumulator
          | .counterAndControl =>
              FPrimeIterationInputRows.rows layouts.baseIteration ++
                other.counterAndControl),
      ∀ column, Mentions row.c column →
        column = 0 ∨
          ∃ window ∈ windowsOf 0 columnWidths, window.owns column

structure Artifact.MatchesSelected
    {widths : CompilerWidths} (artifact : Artifact widths)
    (selected : FullClaimNifsReceipt.SelectedVerifier widths) : Prop where
  profile : artifact.profile = selected.profile
  verifierKeyDigest :
    artifact.verifierKeyDigest = selected.verifierKeyDigest
  relationManifestDigest :
    artifact.relationManifestDigest = selected.relationManifestDigest

def Artifact.partRows {widths : CompilerWidths}
    (artifact : Artifact widths) : Owner → List Row
  | .prelude => artifact.other.prelude
  | .compactHeaders =>
      CompactChainHeaderRows.rows artifact.seedManifest
        artifact.layouts.compactHeaders
  | .initialCarryValidation =>
      MemoryCarryPublicRows.rows artifact.layouts.initialMemoryCarry ++
        InitialMemoryCarryRows.rows artifact.layouts.initialAuthority
  | .challengeAuthority => artifact.other.challengeAuthority
  | .segmentOpening =>
      MemoryOpenSegmentBlockRows.ProfileIndexed.rows artifact.profile
        artifact.layouts.opening
  | .outgoingCarryValidation =>
      MemoryCarryPublicRows.rows artifact.layouts.outgoingMemoryCarry
  | .stateOutput =>
      AuthoritativeStateOutputRows.rows artifact.layouts.stateOutput
  | .claimConstruction => artifact.other.claimConstruction
  | .accumulator => artifact.other.accumulator
  | .counterAndControl =>
      FPrimeIterationInputRows.rows artifact.layouts.baseIteration ++
        artifact.other.counterAndControl

def Artifact.programRows {widths : CompilerWidths}
    (artifact : Artifact widths) : List Row :=
  ownerOrder.flatMap artifact.partRows

def Artifact.FitsGeneratedDomain {widths : CompilerWidths}
    (artifact : Artifact widths) : Prop :=
  artifact.programRows.length ≤ 2 ^ artifact.rowVariableCount

def Artifact.columnWindows {widths : CompilerWidths}
    (artifact : Artifact widths) : List Window :=
  windowsOf 0 artifact.columnWidths

theorem Artifact.owner_rows_in_program {widths : CompilerWidths}
    (artifact : Artifact widths) (owner : Owner) :
    ∀ row ∈ artifact.partRows owner, row ∈ artifact.programRows := by
  intro row member
  exact List.mem_flatMap.mpr ⟨owner, owner.mem_order, member⟩

theorem Artifact.owner_rows_included {widths : CompilerWidths}
    (artifact : Artifact widths) (owner : Owner) :
    rowsIncluded (artifact.partRows owner) artifact.programRows = true := by
  rw [rowsIncluded, List.all_eq_true]
  intro row member
  exact decide_eq_true (artifact.owner_rows_in_program owner row member)

theorem Artifact.owner_satisfied {widths : CompilerWidths}
    {artifact : Artifact widths} {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) (owner : Owner) :
    Satisfies (artifact.partRows owner) assignment := by
  intro row member
  exact satisfies row
    (rowsIncluded_sound (artifact.owner_rows_included owner) row member)

theorem Artifact.compactHeaders_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (CompactChainHeaderRows.rows artifact.seedManifest
        artifact.layouts.compactHeaders) assignment :=
  artifact.owner_satisfied satisfies .compactHeaders

theorem Artifact.initialCarry_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (MemoryCarryPublicRows.rows artifact.layouts.initialMemoryCarry)
      assignment := by
  have initial := artifact.owner_satisfied satisfies .initialCarryValidation
  intro row member
  exact initial row (List.mem_append_left _ member)

theorem Artifact.initialAuthority_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (InitialMemoryCarryRows.rows artifact.layouts.initialAuthority)
      assignment := by
  have initial := artifact.owner_satisfied satisfies .initialCarryValidation
  intro row member
  exact initial row (List.mem_append_right _ member)

theorem Artifact.opening_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (MemoryOpenSegmentBlockRows.ProfileIndexed.rows artifact.profile
        artifact.layouts.opening) assignment :=
  artifact.owner_satisfied satisfies .segmentOpening

/-- Satisfaction of the artifact includes the mandatory named base
challenge-authority row family.  Its semantic refinement remains a separate
compiler theorem; nonempty rows alone do not prove the authority equality. -/
theorem Artifact.challengeAuthority_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies artifact.other.challengeAuthority assignment :=
  artifact.owner_satisfied satisfies .challengeAuthority

theorem Artifact.outgoingCarry_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (MemoryCarryPublicRows.rows artifact.layouts.outgoingMemoryCarry)
      assignment :=
  artifact.owner_satisfied satisfies .outgoingCarryValidation

theorem Artifact.stateOutput_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (AuthoritativeStateOutputRows.rows artifact.layouts.stateOutput)
      assignment :=
  artifact.owner_satisfied satisfies .stateOutput

/-- The base input iteration is an explicit row, not an opaque control
promise. -/
theorem Artifact.baseIteration_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies (FPrimeIterationInputRows.rows
      artifact.layouts.baseIteration) assignment := by
  have control := artifact.owner_satisfied satisfies .counterAndControl
  intro row member
  exact control row (List.mem_append_left _ member)

theorem Artifact.compactHeaders_rows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    (artifact.partRows .compactHeaders).length = 4848 :=
  CompactChainHeaderRows.rows_length_exact
    artifact.layoutsValid.compactHeadersValid

theorem Artifact.initialCarry_rows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    (artifact.partRows .initialCarryValidation).length = 7101 := by
  simp [Artifact.partRows, MemoryCarryPublicRows.rows_length_exact,
    InitialMemoryCarryRows.rows_length_exact]

theorem Artifact.opening_rows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    (artifact.partRows .segmentOpening).length = 11544 :=
  MemoryOpenSegmentBlockRows.ProfileIndexed.rows_length_exact
    artifact.layoutsValid.openingValid

theorem Artifact.outgoingCarry_rows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    (artifact.partRows .outgoingCarryValidation).length = 7094 :=
  MemoryCarryPublicRows.rows_length_exact _

theorem Artifact.stateOutput_rows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    (artifact.partRows .stateOutput).length = 24497 :=
  AuthoritativeStateOutputRows.rows_length_exact
    artifact.layoutsValid.stateOutputValid

theorem Artifact.programRows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    artifact.programRows.length =
      (ownerOrder.map fun owner => (artifact.partRows owner).length).sum := by
  simp [Artifact.programRows, List.length_flatMap]

theorem Artifact.knownRows_lower_bound {widths : CompilerWidths}
    (artifact : Artifact widths) :
    55084 ≤ artifact.programRows.length := by
  rw [artifact.programRows_length]
  simp only [ownerOrder, List.map_cons, List.map_nil, List.sum_cons,
    List.sum_nil]
  rw [artifact.compactHeaders_rows_length,
    artifact.initialCarry_rows_length, artifact.opening_rows_length,
    artifact.outgoingCarry_rows_length, artifact.stateOutput_rows_length]
  omega

structure RowRange where
  owner : Owner
  start : Nat
  stop : Nat
deriving DecidableEq, Repr

def rangesFrom {widths : CompilerWidths} (artifact : Artifact widths) :
    Nat → List Owner → List RowRange
  | _, [] => []
  | cursor, owner :: rest =>
      { owner := owner
        start := cursor
        stop := cursor + (artifact.partRows owner).length } ::
      rangesFrom artifact
        (cursor + (artifact.partRows owner).length) rest

def Covers : Nat → Nat → List RowRange → Prop
  | cursor, finish, [] => cursor = finish
  | cursor, finish, range :: rest =>
      range.start = cursor ∧ range.start ≤ range.stop ∧
        Covers range.stop finish rest

private theorem rangesFrom_cover {widths : CompilerWidths}
    (artifact : Artifact widths) (cursor : Nat) :
    ∀ owners,
      Covers cursor
        (cursor + (owners.map fun owner =>
          (artifact.partRows owner).length).sum)
        (rangesFrom artifact cursor owners) := by
  intro owners
  induction owners generalizing cursor with
  | nil => simp [rangesFrom, Covers]
  | cons owner rest inductionHypothesis =>
      simp only [rangesFrom, Covers, List.map_cons, List.sum_cons]
      refine ⟨trivial, by omega, ?_⟩
      have tail := inductionHypothesis
        (cursor + (artifact.partRows owner).length)
      convert tail using 1 <;> omega

def Artifact.rowRanges {widths : CompilerWidths}
    (artifact : Artifact widths) : List RowRange :=
  rangesFrom artifact 0 ownerOrder

theorem Artifact.rowRanges_exact_cover {widths : CompilerWidths}
    (artifact : Artifact widths) :
    Covers 0 artifact.programRows.length artifact.rowRanges := by
  rw [artifact.programRows_length]
  simpa [Artifact.rowRanges] using rangesFrom_cover artifact 0 ownerOrder

end Nightstream.Implementation.NebulaV2.BaseManifestSchema
