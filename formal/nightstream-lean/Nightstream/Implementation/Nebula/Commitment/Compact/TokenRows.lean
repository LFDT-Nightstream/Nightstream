import Nightstream.Implementation.Nebula.Commitment.Compact.AjtaiStageRows
import Nightstream.Protocol.Nebula.Digest

/-!
Contract: exact two-stage `CompactCommitV2` token relation.

Assurance tier: implementation-to-protocol bridge.

Owns the selected primary and short seeded setups for both token roles, the
exact primary-output-to-short-input link, all 134,082 rows, and the theorem
that satisfying rows compute `CompactCommit.Key.token` from the supplied
canonical commitment fields.

Does not own commitment-field parsing from the full claim, three lane-token
instances, Poseidon2 lane chains, prechallenge knowledge, transcript rows,
Rust refinement, Module-SIS hardness, or absolute generated columns.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.CompactTokenRows

open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.CompactCommit

def primarySetup (manifest : SeedSchedule.Manifest)
    (role : CompactCommit.Role) :
    SeededAjtai.Setup primaryRank primaryMessageRingColumns :=
  match role with
  | .operations => manifest.setup .tokenPrimaryOperations
  | .memory => manifest.setup .tokenPrimaryMemory

def shortSetup (manifest : SeedSchedule.Manifest)
    (role : CompactCommit.Role) :
    SeededAjtai.Setup shortRank shortMessageRingColumns :=
  match role with
  | .operations => manifest.setup .tokenShortOperations
  | .memory => manifest.setup .tokenShortMemory

/-- Concrete protocol key selected by one exact seven-role V2 seed manifest.
The `Seed` type is only a role selector. The authority-bearing bytes are the
four distinct `SeededAjtai.Setup` values selected above. -/
def key (manifest : SeedSchedule.Manifest) :
    CompactCommit.Key Digest.Value CompactCommit.Role where
  profile := manifest.profile
  plan := manifest.plan
  primarySeed := fun role => role
  primarySeedIndependent := CompactCommit.roles_distinct
  shortSeed := fun role => role
  shortSeedIndependent := CompactCommit.roles_distinct
  primaryFromSeed := fun role =>
    CompactAjtaiStageRows.semanticOutput (primarySetup manifest role)
  shortFromSeed := fun role =>
    CompactAjtaiStageRows.semanticOutput (shortSetup manifest role)

@[simp] theorem key_profile (manifest : SeedSchedule.Manifest) :
    (key manifest).profile = manifest.profile := rfl

@[simp] theorem key_plan (manifest : SeedSchedule.Manifest) :
    (key manifest).plan = manifest.plan := rfl

@[simp] theorem key_primary
    (manifest : SeedSchedule.Manifest) (role : CompactCommit.Role)
    (message : RingMessage primaryMessageRingColumns) :
    (key manifest).primary role message =
      CompactAjtaiStageRows.semanticOutput
        (primarySetup manifest role) message := by
  rfl

@[simp] theorem key_short
    (manifest : SeedSchedule.Manifest) (role : CompactCommit.Role)
    (message : RingMessage shortMessageRingColumns) :
    (key manifest).short role message =
      CompactAjtaiStageRows.semanticOutput
        (shortSetup manifest role) message := by
  rfl

/-- The four token maps use four different verifier-key seeds. This includes
cross-stage separation, which the abstract `CompactCommit.Key` selector type
does not itself express. -/
theorem four_setup_seeds_pairwise_distinct
    (manifest : SeedSchedule.Manifest) :
    (manifest.setup .tokenPrimaryOperations).seed.bytes ≠
        (manifest.setup .tokenPrimaryMemory).seed.bytes ∧
      (manifest.setup .tokenPrimaryOperations).seed.bytes ≠
        (manifest.setup .tokenShortOperations).seed.bytes ∧
      (manifest.setup .tokenPrimaryOperations).seed.bytes ≠
        (manifest.setup .tokenShortMemory).seed.bytes ∧
      (manifest.setup .tokenPrimaryMemory).seed.bytes ≠
        (manifest.setup .tokenShortOperations).seed.bytes ∧
      (manifest.setup .tokenPrimaryMemory).seed.bytes ≠
        (manifest.setup .tokenShortMemory).seed.bytes ∧
      (manifest.setup .tokenShortOperations).seed.bytes ≠
        (manifest.setup .tokenShortMemory).seed.bytes := by
  exact ⟨manifest.different_roles_have_different_seeds (by decide),
    manifest.different_roles_have_different_seeds (by decide),
    manifest.different_roles_have_different_seeds (by decide),
    manifest.different_roles_have_different_seeds (by decide),
    manifest.different_roles_have_different_seeds (by decide),
    manifest.different_roles_have_different_seeds (by decide)⟩

/-- Relative columns for one complete two-stage token. The primary output
columns are definitionally the source field columns of the short stage. -/
structure Layout where
  commitmentFieldColumn : Fin commitmentFieldCount → Nat
  primaryDigitStart : Fin commitmentFieldCount → Nat
  primaryOutputColumn : Fin primaryOutputFieldCount → Nat
  shortDigitStart : Fin primaryOutputFieldCount → Nat
  tokenOutputColumn : Fin tokenFieldCount → Nat

def Layout.primary (layout : Layout) :
    CompactAjtaiStageRows.Layout commitmentFieldCount primaryRank where
  fieldColumn := layout.commitmentFieldColumn
  digitStart := layout.primaryDigitStart
  outputColumn := layout.primaryOutputColumn

def Layout.short (layout : Layout) :
    CompactAjtaiStageRows.Layout primaryOutputFieldCount shortRank where
  fieldColumn := layout.primaryOutputColumn
  digitStart := layout.shortDigitStart
  outputColumn := layout.tokenOutputColumn

def primaryRows (manifest : SeedSchedule.Manifest)
    (role : CompactCommit.Role) (layout : Layout) : List Row :=
  CompactAjtaiStageRows.rows
    (primarySetup manifest role) primaryPacking layout.primary

def shortRows (manifest : SeedSchedule.Manifest)
    (role : CompactCommit.Role) (layout : Layout) : List Row :=
  CompactAjtaiStageRows.rows
    (shortSetup manifest role) shortPacking layout.short

def rows (manifest : SeedSchedule.Manifest)
    (role : CompactCommit.Role) (layout : Layout) : List Row :=
  primaryRows manifest role layout ++ shortRows manifest role layout

theorem primaryRows_length (manifest : SeedSchedule.Manifest)
    (role : CompactCommit.Role) (layout : Layout) :
    (primaryRows manifest role layout).length = 120636 := by
  simpa [primaryRows, commitmentFieldCount, primaryRank,
    CompactCommit.ringDegree] using
    CompactAjtaiStageRows.rows_length
      (primarySetup manifest role) primaryPacking layout.primary

theorem shortRows_length (manifest : SeedSchedule.Manifest)
    (role : CompactCommit.Role) (layout : Layout) :
    (shortRows manifest role layout).length = 13446 := by
  simpa [shortRows, primaryOutputFieldCount, shortRank,
    CompactCommit.ringDegree] using
    CompactAjtaiStageRows.rows_length
      (shortSetup manifest role) shortPacking layout.short

theorem rows_length_exact (manifest : SeedSchedule.Manifest)
    (role : CompactCommit.Role) (layout : Layout) :
    (rows manifest role layout).length = 134082 := by
  simp [rows, primaryRows_length, shortRows_length]

private theorem primary_holds
    {manifest : SeedSchedule.Manifest} {role : CompactCommit.Role}
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows manifest role layout) assignment) :
    Satisfies (primaryRows manifest role layout) assignment := by
  intro row member
  exact holds row (List.mem_append_left _ member)

private theorem short_holds
    {manifest : SeedSchedule.Manifest} {role : CompactCommit.Role}
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows manifest role layout) assignment) :
    Satisfies (shortRows manifest role layout) assignment := by
  intro row member
  exact holds row (List.mem_append_right _ member)

def CommitmentPlaced (layout : Layout) (assignment : Nat → Nat)
    (commitment : CommitmentEncoding) : Prop :=
  CompactAjtaiStageRows.FieldsPlaced layout.primary assignment commitment

/-- Main two-stage row soundness theorem. The short stage consumes the exact
primary output columns; it cannot replace them with an independently supplied
intermediate vector. -/
theorem token_exact
    {manifest : SeedSchedule.Manifest} {role : CompactCommit.Role}
    {layout : Layout} {assignment : Nat → Nat}
    {commitment : CommitmentEncoding}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : CommitmentPlaced layout assignment commitment)
    (holds : Satisfies (rows manifest role layout) assignment) :
    ∀ output,
      assignment (layout.tokenOutputColumn output) =
        ((key manifest).token role commitment output).val := by
  let primaryOutput : PrimaryOutput :=
    CompactAjtaiStageRows.semanticOutput (primarySetup manifest role)
      (packFields primaryPacking commitment)
  have primaryExact : ∀ output,
      assignment (layout.primaryOutputColumn output) =
        (primaryOutput output).val := by
    exact CompactAjtaiStageRows.output_exact canonical one placed
      (primary_holds holds)
  have shortPlaced :
      CompactAjtaiStageRows.FieldsPlaced
        layout.short assignment primaryOutput := primaryExact
  have shortExact := CompactAjtaiStageRows.output_exact
    canonical one shortPlaced
    (short_holds holds)
  intro output
  simpa [CompactCommit.Key.token, CompactCommit.Key.primary,
    CompactCommit.Key.short, key, primaryOutput] using shortExact output

end Nightstream.Implementation.Nebula.CompactTokenRows
