import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.Ports

/-!
Wire schema for the exact Rust grouped-phase composition fixture.

Owns untrusted dimensions, row boundaries, selector columns, the phase-group
map, and physical selective-port indices. A successful decode fixes these
values to the independently reviewed fixture geometry.

Does not own either component relation, row truth, production phase counts,
or Nebula F-prime semantics.

Emits constraints: no. The semantic consumer recomputes every link row.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryGroupedPhaseFixture.Artifact

open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports

structure RawArtifact where
  schemaVersion : Nat
  rows : Nat
  columns : Nat
  publicColumns : Nat
  commonRowEnd : Nat
  phaseRowEnd : Nat
  groupEqualityRowEnd : Nat
  phaseActivationRowEnd : Nat
  portCount : Nat
  generalSelectorPort : Nat
  aPort : Nat
  bPort : Nat
  cPort : Nat
  commonSelectorColumns : List Nat
  phaseSelectorColumns : List Nat
  phaseGroups : List Nat
deriving DecidableEq, Repr

/-- The only geometry accepted by this fixture bridge. -/
def Valid (raw : RawArtifact) : Prop :=
  raw.schemaVersion = 1 /\
    raw.rows = 340 /\
    raw.columns = 270 /\
    raw.publicColumns = 54 /\
    raw.commonRowEnd = 166 /\
    raw.phaseRowEnd = 335 /\
    raw.groupEqualityRowEnd = 337 /\
    raw.phaseActivationRowEnd = 340 /\
    raw.portCount = 13 /\
    raw.generalSelectorPort = Role.generalSelector.index.val /\
    raw.aPort = Role.a.index.val /\
    raw.bPort = Role.b.index.val /\
    raw.cPort = Role.c.index.val /\
    raw.commonSelectorColumns = [54, 55] /\
    raw.phaseSelectorColumns = [162, 163, 164] /\
    raw.phaseGroups = [0, 1, 1]

instance (raw : RawArtifact) : Decidable (Valid raw) := by
  unfold Valid
  infer_instance

structure Decoded where
  raw : RawArtifact
  valid : Valid raw

def decode (raw : RawArtifact) : Option Decoded :=
  if valid : Valid raw then some ⟨raw, valid⟩ else none

end Nightstream.Implementation.R1CS.FPrimeFullHistoryGroupedPhaseFixture.Artifact
