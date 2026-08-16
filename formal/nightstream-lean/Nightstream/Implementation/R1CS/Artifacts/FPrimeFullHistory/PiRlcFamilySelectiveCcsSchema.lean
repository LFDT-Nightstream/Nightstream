import Nightstream.Implementation.Nebula.NIFS.PiRLC.RingCombinationRows
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.Ports

/-!
Wire schema for the compact exact selective-CCS recipe of one PiRLC family.

Owns untrusted dimensions, column starts, physical port indices, and the two
nontrivial Goldilocks coefficients. A successful decode fixes every value to
the independent Lean relation geometry.

Does not own row truth, input authority, Poseidon2 binding, Rust execution, or
the complete F-prime relation.

Emits constraints: no. The semantic consumer recomputes every matrix row.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcFamilySelectiveCcs.Artifact

open Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports

structure RawArtifact where
  schemaVersion : Nat
  sourceCount : Nat
  laneCount : Nat
  challengeStart : Nat
  inputStart : Nat
  outputStart : Nat
  productStart : Nat
  productRows : Nat
  rows : Nat
  columns : Nat
  rowVariables : Nat
  portCount : Nat
  generalSelectorPort : Nat
  aPort : Nat
  bPort : Nat
  cPort : Nat
  minusOne : Nat
  minusTwo : Nat
deriving DecidableEq, Repr

/-- A decoded recipe has the sole production geometry accepted by this
bridge. No generated count or coefficient is semantic authority. -/
def Valid (raw : RawArtifact) : Prop :=
  raw.schemaVersion = 1 /\
    raw.sourceCount = sourceCount /\
    raw.laneCount = laneCount /\
    raw.challengeStart = 1 /\
    raw.inputStart = 811 /\
    raw.outputStart = 1621 /\
    raw.productStart = 1675 /\
    raw.productRows = productCount /\
    raw.rows = 43794 /\
    raw.columns = 45415 /\
    raw.rowVariables = 16 /\
    raw.portCount = 13 /\
    raw.generalSelectorPort = Role.generalSelector.index.val /\
    raw.aPort = Role.a.index.val /\
    raw.bPort = Role.b.index.val /\
    raw.cPort = Role.c.index.val /\
    raw.minusOne = goldilocksP - 1 /\
    raw.minusTwo = goldilocksP - 2

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
  base := artifact.raw.productStart
  challengeSymbol := fun source lane =>
    artifact.raw.challengeStart + source.val * artifact.raw.laneCount + lane.val
  input := fun source lane =>
    artifact.raw.inputStart + source.val * artifact.raw.laneCount + lane.val
  output := fun lane => artifact.raw.outputStart + lane.val

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcFamilySelectiveCcs.Artifact
