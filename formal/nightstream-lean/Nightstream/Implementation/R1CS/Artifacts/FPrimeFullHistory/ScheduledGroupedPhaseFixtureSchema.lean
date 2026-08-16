import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.Ports

/-!
Wire schema for the exact Rust schedule-over-grouped-phase fixture.

Owns untrusted dimensions, row boundaries, selector columns, schedule maps,
cursor-bit ranges, and physical selective-port indices. A successful decode
fixes these values to the independently reviewed fixture geometry.

Does not own component semantics, production phase counts, or the complete
F-prime relation.

Emits constraints: no. Semantic consumers recompute every emitted link row.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryScheduledGroupedPhaseFixture.Artifact

open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports

structure RawArtifact where
  schemaVersion : Nat
  rows : Nat
  columns : Nat
  publicColumns : Nat
  commonRowEnd : Nat
  phaseRowEnd : Nat
  scheduleTotalRowEnd : Nat
  lifecycleEqualityRowEnd : Nat
  phaseKindEqualityRowEnd : Nat
  lifecycleActivationRowEnd : Nat
  phaseKindActivationRowEnd : Nat
  cursorBindingRowEnd : Nat
  portCount : Nat
  generalSelectorPort : Nat
  aPort : Nat
  bPort : Nat
  cPort : Nat
  commonSelectorColumns : List Nat
  phaseKindSelectorColumns : List Nat
  scheduleSelectorColumns : List Nat
  lifecycleGroups : List Nat
  phaseKinds : List Nat
  beforeCursorStart : Nat
  beforeCursorEnd : Nat
  afterCursorStart : Nat
  afterCursorEnd : Nat
deriving DecidableEq, Repr

/-- The only geometry accepted by this fixture bridge. -/
def Valid (raw : RawArtifact) : Prop :=
  raw.schemaVersion = 1 /\
    raw.rows = 406 /\
    raw.columns = 324 /\
    raw.publicColumns = 54 /\
    raw.commonRowEnd = 169 /\
    raw.phaseRowEnd = 338 /\
    raw.scheduleTotalRowEnd = 339 /\
    raw.lifecycleEqualityRowEnd = 341 /\
    raw.phaseKindEqualityRowEnd = 343 /\
    raw.lifecycleActivationRowEnd = 346 /\
    raw.phaseKindActivationRowEnd = 349 /\
    raw.cursorBindingRowEnd = 355 /\
    raw.portCount = 13 /\
    raw.generalSelectorPort = Role.generalSelector.index.val /\
    raw.aPort = Role.a.index.val /\
    raw.bPort = Role.b.index.val /\
    raw.cPort = Role.c.index.val /\
    raw.commonSelectorColumns = [54, 55] /\
    raw.phaseKindSelectorColumns = [162, 163] /\
    raw.scheduleSelectorColumns = [270, 271, 272] /\
    raw.lifecycleGroups = [0, 1, 1] /\
    raw.phaseKinds = [0, 1, 0] /\
    raw.beforeCursorStart = 1 /\
    raw.beforeCursorEnd = 3 /\
    raw.afterCursorStart = 3 /\
    raw.afterCursorEnd = 5

instance (raw : RawArtifact) : Decidable (Valid raw) := by
  unfold Valid
  infer_instance

structure Decoded where
  raw : RawArtifact
  valid : Valid raw

def decode (raw : RawArtifact) : Option Decoded :=
  if valid : Valid raw then some ⟨raw, valid⟩ else none

end Nightstream.Implementation.R1CS.FPrimeFullHistoryScheduledGroupedPhaseFixture.Artifact
