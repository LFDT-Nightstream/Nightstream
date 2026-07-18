import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.Generated.FPrimeRecursiveYZcolProjectionData

/-!
Stable artifact facade for the two fixed-profile parent `y_zcol` output
evaluation leaves.

Owns: the checked two-limb owner tree, its exact 216 normalized source rows,
their index-preserving correspondence to reconstructed equations, and
fixed-profile facts about active/padded widths and shared power columns.

Does not own: parent-opening authority, beta transcript derivation, beta-ladder
rows, the ten padded-lane zero/canonicalization checks, semantic projection
soundness, bad-root bounds, whole-identity rows, global cost reconciliation,
or permission to remove rows.

Emits constraints: no.

Assurance tier: artifact-checked local ownership after the Rust drift test
replays the fixed profile and byte-compares the generated module. This is not
yet whole-verifier Rust conformance.

Authority boundary: the generator binds coefficient columns directly to the
returned `FPrimeStepOutput.nifs_parent.y_zcol` wires. That is typed wire
ownership, not a proof that the returned parent is the canonical semantic
parent or that the supplied powers came from the verifier transcript.

| Child path | Mathematical obligation | Exact source-R1CS owner | Lean owner |
|---|---|---:|---|
| `nifs.pi_rlc.verify.identities.y_zcol.evaluations.output.limb0` | evaluate the 54 active `c0` coefficients at beta | 108 source rows | first `owners` entry |
| `nifs.pi_rlc.verify.identities.y_zcol.evaluations.output.limb1` | evaluate the 54 active `c1` coefficients at beta | 108 source rows | second `owners` entry |
| `nifs.pi_rlc.verify.identities.y_zcol.evaluations.output` | compose both leaves and record their shared power inputs | 216 source rows | `ownedRowDefinitions` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjection

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjectionData

def expectedStagePath : String :=
  "nifs.pi_rlc.verify.identities.y_zcol.evaluations.output"

def expectedLimb0StagePath : String :=
  "nifs.pi_rlc.verify.identities.y_zcol.evaluations.output.limb0"

def expectedLimb1StagePath : String :=
  "nifs.pi_rlc.verify.identities.y_zcol.evaluations.output.limb1"

def limb0Owner : YZcolOutputEvaluationOwner := owners.getD 0 default

def limb1Owner : YZcolOutputEvaluationOwner := owners.getD 1 default

/-- Every owned physical row paired with its reconstructed leaf equation. -/
def ownedRowDefinitions : List (Nat × Program.Definition) :=
  owners.flatMap YZcolOutputEvaluationOwner.rowDefinitions

/-- The exact normalized Rust rows, with their absolute source indices
removed only for R1CS satisfaction. -/
def ownedSourceRows : List Row := sourceRows.map Prod.snd

/-- Index-preserving, term-order-insensitive comparison between exact source
rows and reconstructed builder equations. -/
def SourceRowsMatch : Prop :=
  ActiveIndexedRows.indexedRowsMatch
    sourceRows ownedRowDefinitions = true

instance : Decidable SourceRowsMatch := by
  unfold SourceRowsMatch
  infer_instance

/-- Complete fixed-profile artifact predicate. It checks organization and
local evaluator layout only; no semantic authority premise is smuggled in. -/
def DataValid : Prop :=
  stagePath = expectedStagePath ∧
  activeLaneCount = 54 ∧
  paddedLaneCount = 64 ∧
  sharedPowerColumns.length = activeLaneCount ∧
  owners.length = 2 ∧
  limb0Owner.stagePath = expectedLimb0StagePath ∧
  limb1Owner.stagePath = expectedLimb1StagePath ∧
  limb0Owner.limb = 0 ∧
  limb1Owner.limb = 1 ∧
  limb0Owner.identityIndex + 1 = limb1Owner.identityIndex ∧
  limb0Owner.evaluationRowEnd ≤ limb1Owner.evaluationRowStart ∧
  limb0Owner.evaluationAllocatedEnd ≤
    limb1Owner.evaluationAllocatedStart ∧
  limb0Owner.powerColumns = sharedPowerColumns ∧
  limb1Owner.powerColumns = sharedPowerColumns ∧
  limb0Owner.Valid activeLaneCount ∧
  limb1Owner.Valid activeLaneCount ∧
  sourceRows.length = 216 ∧
  (sourceRows.map Prod.fst).Nodup ∧
  (ownedRowDefinitions.map Prod.fst).Nodup ∧
  SourceRowsMatch

instance : Decidable DataValid := by
  unfold DataValid
  infer_instance

/-- Kernel-checked local-layout certificate for the generated artifact. -/
theorem data_check : DataValid := by
  set_option maxRecDepth 100000 in
    decide

theorem owner_count : owners.length = 2 := by
  decide

theorem owner_limb_order : owners.map (·.limb) = [0, 1] := by
  decide

theorem active_lane_count : activeLaneCount = 54 := by
  decide

theorem padded_lane_count : paddedLaneCount = 64 := by
  decide

theorem limb0_valid : limb0Owner.Valid activeLaneCount := by
  set_option maxRecDepth 100000 in
    decide

theorem limb1_valid : limb1Owner.Valid activeLaneCount := by
  set_option maxRecDepth 100000 in
    decide

theorem owned_row_count : ownedRowDefinitions.length = 216 := by
  set_option maxRecDepth 100000 in
    decide

theorem source_row_count : sourceRows.length = 216 := by
  set_option maxRecDepth 100000 in
    decide

theorem source_rows_distinct : (sourceRows.map Prod.fst).Nodup := by
  set_option maxRecDepth 100000 in
    decide

theorem source_rows_match : SourceRowsMatch := by
  set_option maxRecDepth 100000 in
    decide

end Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjection
