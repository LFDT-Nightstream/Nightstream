import Mathlib.Tactic.DeriveFintype
import Nightstream.Implementation.R1CS.Core.SeededAjtai
import Nightstream.Protocol.NebulaV2.CompactCommit
import Nightstream.Protocol.NebulaV2.Digest
import Nightstream.Protocol.NebulaV2.ProductionProfileCandidates

/-!
Contract: exact verifier-key-owned Ajtai seed schedule for Nebula V2.

Assurance tier: implementation model.

Owns the seven independent setup roles, their exact row ranks, their
verifier-key-owned ring-column counts, successful pure ChaCha8 sampler
executions, and the requirement that all role seeds are distinct. Initial and
final snapshots use the one `bundleSnapshot` role by construction.

Does not own ChaCha8 pseudorandomness, Module-SIS hardness, Rust manifest
serialization, concrete final V2 lane widths, or generated-row refinement.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.SeedSchedule

open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CompactCommit

/-- No two authority roles reuse one Ajtai matrix. Initial and final snapshot
commitments deliberately share `bundleSnapshot`, so they are not two roles. -/
inductive Role where
  | bundleFull
  | bundleOperations
  | bundleSnapshot
  | tokenPrimaryOperations
  | tokenPrimaryMemory
  | tokenShortOperations
  | tokenShortMemory
deriving DecidableEq, Fintype, Repr

theorem role_count : Fintype.card Role = 7 := by
  decide

/-- Ring-column widths fixed by the final V2 verifier-key manifest. Compact
token widths are protocol constants and are not fields here. -/
structure Geometry where
  fullAssignmentRingColumns : Nat
  operationsRingColumns : Nat
  snapshotRingColumns : Nat
  fullPositive : 0 < fullAssignmentRingColumns
  operationsPositive : 0 < operationsRingColumns
  snapshotPositive : 0 < snapshotRingColumns
deriving Repr

def Role.rows : Role → Nat
  | .bundleFull | .bundleOperations | .bundleSnapshot => commitmentRank
  | .tokenPrimaryOperations | .tokenPrimaryMemory => primaryRank
  | .tokenShortOperations | .tokenShortMemory => shortRank

def Role.columns (geometry : Geometry) : Role → Nat
  | .bundleFull => geometry.fullAssignmentRingColumns
  | .bundleOperations => geometry.operationsRingColumns
  | .bundleSnapshot => geometry.snapshotRingColumns
  | .tokenPrimaryOperations | .tokenPrimaryMemory =>
      primaryMessageRingColumns
  | .tokenShortOperations | .tokenShortMemory =>
      shortMessageRingColumns

theorem exact_fixed_role_geometry (geometry : Geometry) :
    Role.bundleFull.rows = 18 ∧
      Role.bundleOperations.rows = 18 ∧
      Role.bundleSnapshot.rows = 18 ∧
      Role.tokenPrimaryOperations.rows = 2 ∧
      Role.tokenPrimaryMemory.rows = 2 ∧
      Role.tokenShortOperations.rows = 1 ∧
      Role.tokenShortMemory.rows = 1 ∧
      Role.tokenPrimaryOperations.columns geometry = 738 ∧
      Role.tokenPrimaryMemory.columns geometry = 738 ∧
      Role.tokenShortOperations.columns geometry = 82 ∧
      Role.tokenShortMemory.columns geometry = 82 := by
  norm_num [Role.rows, Role.columns, commitmentRank, primaryRank, shortRank,
    primaryMessageRingColumns, shortMessageRingColumns,
    commitmentFieldCount, primaryOutputFieldCount, ringDegree,
    ShiftedTernary41V1.digitCount]

/-- Exact selected setup for every role. `SeededAjtai.Setup` already carries
proof that the bounded pure ChaCha8 rejection sampler succeeds and selects one
finite verifier key. -/
structure Manifest where
  profile : Profile.Identity
  /-- The current package has exact numeric frames only for the V2 reference
  and the four separately versioned field-native candidates. -/
  profileSupported : ProductionProfileCandidates.SupportedIdentity profile
  /-- Verifier-owned digest of the complete canonical plan. The deployed
  verifier must recompute it from the plan manifest. -/
  plan : Digest.Value
  geometry : Geometry
  setup : (role : Role) →
    SeededAjtai.Setup role.rows (role.columns geometry)
  seedsDistinct : Function.Injective
    (fun role => (setup role).seed.bytes)

namespace Manifest

def setupIdentity (manifest : Manifest) (role : Role) :
    SeededAjtai.Identity :=
  (manifest.setup role).identity

theorem different_roles_have_different_seeds
    (manifest : Manifest) {left right : Role}
    (different : left ≠ right) :
    (manifest.setup left).seed.bytes ≠
      (manifest.setup right).seed.bytes := by
  intro equal
  exact different (manifest.seedsDistinct equal)

theorem compact_primary_seeds_are_distinct (manifest : Manifest) :
    (manifest.setup .tokenPrimaryOperations).seed.bytes ≠
      (manifest.setup .tokenPrimaryMemory).seed.bytes :=
  manifest.different_roles_have_different_seeds (by decide)

theorem compact_short_seeds_are_distinct (manifest : Manifest) :
    (manifest.setup .tokenShortOperations).seed.bytes ≠
      (manifest.setup .tokenShortMemory).seed.bytes :=
  manifest.different_roles_have_different_seeds (by decide)

end Manifest

end Nightstream.Implementation.NebulaV2.SeedSchedule
