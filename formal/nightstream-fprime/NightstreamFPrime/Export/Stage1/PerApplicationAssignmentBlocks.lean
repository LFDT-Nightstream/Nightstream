import NightstreamFPrime.Export.AffineRuns
import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentPlan

/-!
Owns compact source-index metadata for the final retained assignment program.
Every block is derived from the existing Lean opcode interpreter. The run
codec changes only representation; `sourceRuns_expand` proves exact source
order and `sourceRuns_count` proves total coverage.

This module does not install the metadata in a sealed package or authorize a
Rust interpreter.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationAssignmentBlocks

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open PerApplicationCanonicalAssignment
open PerApplicationAssignmentPlan

abbrev ProgramApplication := Lifecycle.Stage1.Application.Program

/-- The three value domains consumed by the retained assignment. -/
inductive SourceDomain where
  | retained
  | piCcsPayload
  | physicalBase
deriving Repr, DecidableEq

def SourceDomain.format : Format SourceDomain where
  encode
    | .retained => .atom 0
    | .piCcsPayload => .atom 1
    | .physicalBase => .atom 2
  decode
    | .atom 0 => .ok .retained
    | .atom 1 => .ok .piCcsPayload
    | .atom 2 => .ok .physicalBase
    | _ => .error "invalid assignment source domain"
  decode_encode := by
    intro domain
    cases domain <;> rfl

def slotKindFormat : Format LowNormSlot.Kind where
  encode
    | .bit => .atom 0
    | .centered => .atom 1
    | .field => .atom 2
  decode
    | .atom 0 => .ok .bit
    | .atom 1 => .ok .centered
    | .atom 2 => .ok .field
    | _ => .error "invalid retained slot kind"
  decode_encode := by
    intro kind
    cases kind <;> rfl

/-- A value-free packet exposes the same block geometry and source-index
functions as every real packet. -/
def zeroRaw (application : ProgramApplication) : RawValues application where
  base := fun _ => 0
  groupValue := fun _ _ => 0
  products := fun _ => 0

def entry (application : ProgramApplication) (kind : BlockKind) :
    CanonicalBlockAssignment.BlockValue :=
  kind.expand (zeroRaw application)

def sourceDomainOf : BlockKind → SourceDomain
  | .piCcsPayload => .piCcsPayload
  | .applicationInput | .applicationWitness | .applicationOutput |
      .applicationLocal => .physicalBase
  | _ => .retained

/-- Exact normalized source index selected by one retained slot. Payload
indices omit their retained-prefix displacement; every other domain already
starts at zero. -/
def sourceIndex (application : ProgramApplication)
    (kind : BlockKind)
    (slot : Fin (entry application kind).block.slotCount) : Nat :=
  let selected := ((entry application kind).block.source slot).val
  match kind with
  | .piCcsPayload =>
      selected - PiCCSActionPayloadBlock.prefixSourceWidth application
  | _ => selected

def sourceIndices (application : ProgramApplication)
    (kind : BlockKind) : List Nat :=
  List.ofFn (sourceIndex application kind)

def sourceRunsFor (application : ProgramApplication)
    (kind : BlockKind) : List AffineRuns.Run :=
  AffineRuns.compress (sourceIndices application kind)

theorem sourceIndices_length (application : ProgramApplication)
    (kind : BlockKind) :
    (sourceIndices application kind).length =
      (entry application kind).block.slotCount := by
  simp [sourceIndices]

/-- Run expansion recovers every Lean-selected source index in exact order. -/
theorem sourceRuns_expand (application : ProgramApplication)
    (kind : BlockKind) :
  AffineRuns.expand (sourceRunsFor application kind) =
      sourceIndices application kind := by
  exact AffineRuns.expand_compress _

/-- The compact source map covers every retained slot exactly once. -/
theorem sourceRuns_count (application : ProgramApplication)
    (kind : BlockKind) :
    ((sourceRunsFor application kind).map AffineRuns.Run.count).sum =
      (entry application kind).block.slotCount := by
  rw [← AffineRuns.expand_length, sourceRuns_expand,
    sourceIndices_length]

structure BlockPlan where
  opcode : BlockKind
  slotKind : LowNormSlot.Kind
  slotCount : Nat
  sourceDomain : SourceDomain
  sourceRuns : List AffineRuns.Run
deriving Repr, DecidableEq

def BlockPlan.format : Format BlockPlan where
  encode := fun block => .array [
    BlockKind.format.encode block.opcode,
    slotKindFormat.encode block.slotKind,
    .atom block.slotCount,
    SourceDomain.format.encode block.sourceDomain,
    AffineRuns.format.encode block.sourceRuns]
  decode
    | .array [opcode, slotKind, .atom slotCount, sourceDomain, sourceRuns] => do
        pure {
          opcode := ← BlockKind.format.decode opcode
          slotKind := ← slotKindFormat.decode slotKind
          slotCount := slotCount
          sourceDomain := ← SourceDomain.format.decode sourceDomain
          sourceRuns := ← AffineRuns.format.decode sourceRuns }
    | _ => .error "invalid assignment block plan"
  decode_encode := by
    intro block
    cases block
    simp only [BlockKind.format.decode_encode, slotKindFormat.decode_encode,
      SourceDomain.format.decode_encode, AffineRuns.decode_encode]
    rfl

def BlockPlan.ofKind (application : ProgramApplication)
    (kind : BlockKind) : BlockPlan :=
  { opcode := kind
    slotKind := (entry application kind).block.kind
    slotCount := (entry application kind).block.slotCount
    sourceDomain := sourceDomainOf kind
    sourceRuns := sourceRunsFor application kind }

theorem BlockPlan.ofKind_sourceRuns_count
    (application : ProgramApplication) (kind : BlockKind) :
    (((BlockPlan.ofKind application kind).sourceRuns.map
      AffineRuns.Run.count).sum) =
      (BlockPlan.ofKind application kind).slotCount := by
  simpa [BlockPlan.ofKind] using sourceRuns_count application kind

def canonical (application : ProgramApplication) : List BlockPlan :=
  canonicalKinds.map (BlockPlan.ofKind application)

@[simp] theorem canonical_length (application : ProgramApplication) :
    (canonical application).length = 45 := by
  simp [canonical]

theorem canonical_opcodes (application : ProgramApplication) :
    (canonical application).map BlockPlan.opcode = canonicalKinds := by
  rfl

def format : Format (List BlockPlan) := Codec.list BlockPlan.format

theorem decode_encode (blocks : List BlockPlan) :
    format.decode (format.encode blocks) = .ok blocks :=
  format.decode_encode blocks

end NightstreamFPrime.Export.Stage1.PerApplicationAssignmentBlocks
