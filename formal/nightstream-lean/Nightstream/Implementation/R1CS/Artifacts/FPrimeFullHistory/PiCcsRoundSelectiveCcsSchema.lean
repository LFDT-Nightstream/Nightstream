import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCCSRoundRows
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.Ports

/-!
Wire schema for the exact selective-CCS recipe of one phased PiCCS round.

Owns untrusted dimensions, canonical column starts, physical port indices,
and the two nontrivial Goldilocks coefficients. A successful decode fixes all
values to the independent Lean row geometry.

Does not own row truth, Poseidon2 replay, recursive orchestration, terminal
integration, or the complete F-prime relation.

Emits constraints: no. The semantic consumer recomputes every matrix row.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsRoundSelectiveCcs.Artifact

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRoundRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports

structure RawArtifact where
  schemaVersion : Nat
  degree : Nat
  coefficientCount : Nat
  currentStart : Nat
  coefficientStart : Nat
  challengeStart : Nat
  nextStart : Nat
  auxiliaryStart : Nat
  rows : Nat
  columns : Nat
  rowVariables : Nat
  portCount : Nat
  generalSelectorPort : Nat
  aPort : Nat
  bPort : Nat
  cPort : Nat
  nonresidue : Nat
  minusOne : Nat
deriving DecidableEq, Repr

/-- A decoded recipe has the sole production geometry accepted by this
bridge. Generated counts and coefficients are not semantic authority. -/
def Valid (raw : RawArtifact) : Prop :=
  raw.schemaVersion = 1 /\
    raw.degree = 9 /\
    raw.coefficientCount = 10 /\
    raw.currentStart = 1 /\
    raw.coefficientStart = 3 /\
    raw.challengeStart = 23 /\
    raw.nextStart = 25 /\
    raw.auxiliaryStart = 27 /\
    raw.rows = 31 /\
    raw.columns = 54 /\
    raw.rowVariables = 5 /\
    raw.portCount = 13 /\
    raw.generalSelectorPort = Role.generalSelector.index.val /\
    raw.aPort = Role.a.index.val /\
    raw.bPort = Role.b.index.val /\
    raw.cPort = Role.c.index.val /\
    raw.nonresidue = 7 /\
    raw.minusOne = goldilocksP - 1

instance (raw : RawArtifact) : Decidable (Valid raw) := by
  unfold Valid
  infer_instance

structure Decoded where
  raw : RawArtifact
  valid : Valid raw

def decode (raw : RawArtifact) : Option Decoded :=
  if valid : Valid raw then some ⟨raw, valid⟩ else none

/-- Canonical caller-visible and auxiliary placement selected by the decoded
recipe. -/
def Decoded.layout (artifact : Decoded) : Layout where
  currentStart := artifact.raw.currentStart
  coefficientStart := artifact.raw.coefficientStart
  challengeStart := artifact.raw.challengeStart
  nextStart := artifact.raw.nextStart
  auxiliaryStart := artifact.raw.auxiliaryStart

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsRoundSelectiveCcs.Artifact
