import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PublicCarrier
import Nightstream.Implementation.R1CS.Core.Relabel

/-!
Explicit local-to-production column transport for the fixed F' NIFS carrier.

Protocol: production `Pi_RLC` to strict-`Pi_DEC` handoff.
Phase: column-name refinement.
Constraint family: verifier-visible CE claim columns; this file emits no rows.

Owns: structural relabeling of commitment and claim layouts; exact agreement
between decoding relabeled global columns and decoding local columns under
`Relabel.assignment` for commitments, public inputs, and points.

Does not own: proof that a generated column map is complete or injective,
strict-PiDEC row soundness, active evaluation widths, private openings,
projection identities, costs, or row removal.

Emits constraints: no. It transports column identifiers only.

Authority boundary: no digest or prover-supplied equality is used. Every
global column is computed from the explicit checked column map. A default map
entry remains column zero, exactly as in `Relabel.column`; profile modules must
separately prove that every authoritative local column lies inside the map.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.layout.commitment.relabel` | map every commitment coordinate exactly once | computed | `relabelCommitment` |
| `nifs.pi_dec.layout.claim.relabel` | map every column-bearing claim field and preserve dimensions | computed | `relabelClaim` |
| `nifs.pi_dec.decode.commitment.relabel` | global mapped decoding equals local-assignment decoding | derived | `decodedPackedCommitment_relabel` |
| `nifs.pi_dec.decode.public_input.relabel` | all 270 mapped `X` values equal local-assignment decoding | derived | `decodedPackedInput_relabel` |
| `nifs.pi_dec.decode.point.relabel` | mapped extension-coordinate pairs decode identically | derived | `decodedPoint_relabel` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.RelabeledCarrier

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper

/-- Relabel every column in one commitment carrier. -/
def relabelCommitment (columnMap : List Nat)
    (commitment : CommitmentLayout) : CommitmentLayout where
  dCol := Relabel.column columnMap commitment.dCol
  kappaCol := Relabel.column columnMap commitment.kappaCol
  dataCols := commitment.dataCols.map (Relabel.column columnMap)

/-- Relabel all three optional auxiliary commitments. -/
def relabelAdv (columnMap : List Nat) (adv : AdvLayout) : AdvLayout where
  ops := relabelCommitment columnMap adv.ops
  is := relabelCommitment columnMap adv.is
  fs := relabelCommitment columnMap adv.fs

/-- Relabel both limbs of each extension-field column pair. -/
def relabelPairs (columnMap : List Nat)
    (columns : List (Nat × Nat)) : List (Nat × Nat) :=
  columns.map fun pair =>
    (Relabel.column columnMap pair.1, Relabel.column columnMap pair.2)

/-- Relabel every column-bearing field of a strict-PiDEC claim while retaining
its verifier-owned scalar dimensions. -/
def relabelClaim (columnMap : List Nat) (claim : ClaimLayout) : ClaimLayout where
  commitment := relabelCommitment columnMap claim.commitment
  adv := claim.adv.map (relabelAdv columnMap)
  xActiveCols := claim.xActiveCols.map (Relabel.column columnMap)
  xInactiveCol := Relabel.column columnMap claim.xInactiveCol
  xRows := claim.xRows
  xWidth := claim.xWidth
  xRowsCol := Relabel.column columnMap claim.xRowsCol
  xWidthCol := Relabel.column columnMap claim.xWidthCol
  mIn := claim.mIn
  mInCol := Relabel.column columnMap claim.mInCol
  yRingCols := claim.yRingCols.map fun row =>
    row.map (Relabel.column columnMap)
  ctCols := relabelPairs columnMap claim.ctCols
  rCols := relabelPairs columnMap claim.rCols
  sColCols := relabelPairs columnMap claim.sColCols
  foldDigestCols := claim.foldDigestCols.map (Relabel.column columnMap)

/-- Mapping columns before reading an assignment is exactly reading through
`Relabel.assignment`. -/
theorem values_relabel (columnMap : List Nat) (assignment : Nat -> Nat)
    (columns : List Nat) :
    values assignment (columns.map (Relabel.column columnMap)) =
      values (Relabel.assignment columnMap assignment) columns := by
  simp [values, Relabel.assignment, Function.comp_def]

theorem decodedPackedCommitment_relabel
    (columnMap : List Nat) (assignment : Nat -> Nat)
    (claim : ClaimLayout) :
    decodedPackedCommitment assignment (relabelClaim columnMap claim) =
      decodedPackedCommitment
        (Relabel.assignment columnMap assignment) claim := by
  apply PackedCommitment.eq_of_data_eq
  exact values_relabel columnMap assignment claim.commitment.dataCols

theorem decodedPackedInput_relabel
    (columnMap : List Nat) (assignment : Nat -> Nat)
    (claim : ClaimLayout) :
    decodedPackedInput assignment (relabelClaim columnMap claim) =
      decodedPackedInput
        (Relabel.assignment columnMap assignment) claim := by
  apply PackedPublicInput.eq_of_data_eq
  exact values_relabel columnMap assignment claim.xActiveCols

theorem extensionValues_relabel
    (columnMap : List Nat) (assignment : Nat -> Nat)
    (columns : List (Nat × Nat)) :
    extensionValues assignment (relabelPairs columnMap columns) =
      extensionValues (Relabel.assignment columnMap assignment) columns := by
  simp [extensionValues, extensionValue, relabelPairs,
    Relabel.assignment, Function.comp_def]

theorem decodedPoint_relabel
    (columnMap : List Nat) (assignment : Nat -> Nat)
    (claim : ClaimLayout) :
    decodedPoint assignment (relabelClaim columnMap claim) =
      decodedPoint (Relabel.assignment columnMap assignment) claim := by
  exact extensionValues_relabel columnMap assignment claim.rCols

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.RelabeledCarrier
