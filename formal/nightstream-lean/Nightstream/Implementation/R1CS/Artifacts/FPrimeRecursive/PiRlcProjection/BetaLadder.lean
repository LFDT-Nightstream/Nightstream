import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.Generated.BetaLadderData
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.YZcolProjection

/-!
Stable artifact facade for the active shared PiRLC beta ladder.

Owns: one checked 55-power ladder, its exact 272 normalized source rows, and
exact column linkage to both returned-parent `y_zcol` evaluator leaves.

Does not own: beta transcript derivation, rho evaluations, semantic parent
authority, whole-matrix embedding, projection soundness, or row removal.

Emits constraints: no.

Assurance tier: artifact-checked local ownership after the Rust drift test.
This is not whole-verifier Rust conformance.

| Child path | Mathematical obligation | Exact owner | Consumer |
|---|---|---:|---|
| `projection_shared.beta_ladder` | build `1, beta, ..., beta^54` | 272 source rows | all projection identities |
| `identities.y_zcol.evaluations.output.{limb0,limb1}` | consume powers `0..53` | zero additional rows | exact 54-power prefix |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.BetaLadder

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjectionBetaLadderData

def expectedStagePath : String :=
  "nifs.pi_rlc.verify.projection_shared.beta_ladder"

def powerCount : Nat := 55

def owner : PiRlcProjectionBetaLadderOwner where
  stagePath := stagePath
  rowStart := rowStart
  rowEnd := rowEnd
  allocatedStart := allocatedStart
  allocatedEnd := allocatedEnd
  betaColumns := betaColumns
  powerColumns := powerColumns

def ownedRowDefinitions : List (Nat × Program.Definition) :=
  owner.rowDefinitions

def ownedSourceRows : List Row := sourceRows.map Prod.snd

def SourceRowsMatch : Prop :=
  ActiveIndexedRows.indexedRowsMatch sourceRows ownedRowDefinitions = true

instance : Decidable SourceRowsMatch := by
  unfold SourceRowsMatch
  infer_instance

def DataValid : Prop :=
  stagePath = expectedStagePath ∧
  owner.Valid powerCount ∧
  sourceRows.length = 272 ∧
  (sourceRows.map Prod.fst).Nodup ∧
  (ownedRowDefinitions.map Prod.fst).Nodup ∧
  SourceRowsMatch ∧
  powerColumns.take
      FPrimeRecursiveYZcolProjectionData.activeLaneCount =
    FPrimeRecursiveYZcolProjectionData.sharedPowerColumns

instance : Decidable DataValid := by
  unfold DataValid
  infer_instance

theorem data_check : DataValid := by
  set_option maxRecDepth 100000 in
    decide

theorem owner_valid : owner.Valid powerCount := by
  set_option maxRecDepth 100000 in
    decide

theorem power_count : powerColumns.length = 55 := by
  decide

theorem owned_row_count : ownedRowDefinitions.length = 272 := by
  set_option maxRecDepth 100000 in
    decide

theorem source_row_count : sourceRows.length = 272 := by
  set_option maxRecDepth 100000 in
    decide

theorem source_rows_distinct : (sourceRows.map Prod.fst).Nodup := by
  set_option maxRecDepth 100000 in
    decide

theorem source_rows_match : SourceRowsMatch := by
  set_option maxRecDepth 100000 in
    decide

theorem y_zcol_power_prefix :
    powerColumns.take
        FPrimeRecursiveYZcolProjectionData.activeLaneCount =
      FPrimeRecursiveYZcolProjectionData.sharedPowerColumns := by
  decide

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.BetaLadder
