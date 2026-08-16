import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.Ports

/-!
Wire schema for the exact Rust schedule-linked private-overlay fixture.

Owns untrusted dimensions, row boundaries, selector columns, schedule maps,
linked field digit ranges, radices, and physical selective-port indices. A
successful decode fixes them to the independently reviewed fixture geometry.

Does not own component semantics, production dimensions, or the complete
F-prime relation.

Emits constraints: no. Semantic consumers recompute every emitted link row.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryScheduledLinkedOverlayFixture.Artifact

open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports

structure RawArtifact where
  schemaVersion : Nat
  rows : Nat
  columns : Nat
  publicColumns : Nat
  scheduledRowEnd : Nat
  overlayRowEnd : Nat
  overlayKindEqualityRowEnd : Nat
  overlayActivationRowEnd : Nat
  fieldLinkRowEnd : Nat
  ringPaddingRowEnd : Nat
  ringPaddingColumnStart : Nat
  portCount : Nat
  generalSelectorPort : Nat
  aPort : Nat
  bPort : Nat
  cPort : Nat
  scheduleSelectorColumns : List Nat
  overlaySelectorColumns : List Nat
  lifecycleGroups : List Nat
  phaseKinds : List Nat
  overlayKinds : List Nat
  phaseFieldStarts : List Nat
  overlayFieldStarts : List Nat
  fieldWidths : List Nat
  fieldRadices : List Nat
deriving DecidableEq, Repr

/-- The only geometry accepted by this fixture bridge. -/
def Valid (raw : RawArtifact) : Prop :=
  raw.schemaVersion = 1 /\
    raw.rows = 384 /\
    raw.columns = 540 /\
    raw.publicColumns = 54 /\
    raw.scheduledRowEnd = 348 /\
    raw.overlayRowEnd = 376 /\
    raw.overlayKindEqualityRowEnd = 378 /\
    raw.overlayActivationRowEnd = 381 /\
    raw.fieldLinkRowEnd = 383 /\
    raw.ringPaddingRowEnd = 384 /\
    raw.ringPaddingColumnStart = 539 /\
    raw.portCount = 13 /\
    raw.generalSelectorPort = Role.generalSelector.index.val /\
    raw.aPort = Role.a.index.val /\
    raw.bPort = Role.b.index.val /\
    raw.cPort = Role.c.index.val /\
    raw.scheduleSelectorColumns = [378, 379, 380] /\
    raw.overlaySelectorColumns = [432, 433] /\
    raw.lifecycleGroups = [0, 1, 1] /\
    raw.phaseKinds = [0, 1, 0] /\
    raw.overlayKinds = [0, 1, 0] /\
    raw.phaseFieldStarts = [270, 270] /\
    raw.overlayFieldStarts = [434, 434] /\
    raw.fieldWidths = [41, 41] /\
    raw.fieldRadices = [3, 3]

instance (raw : RawArtifact) : Decidable (Valid raw) := by
  unfold Valid
  infer_instance

structure Decoded where
  raw : RawArtifact
  valid : Valid raw

def decode (raw : RawArtifact) : Option Decoded :=
  if valid : Valid raw then some ⟨raw, valid⟩ else none

end Nightstream.Implementation.R1CS.FPrimeFullHistoryScheduledLinkedOverlayFixture.Artifact
