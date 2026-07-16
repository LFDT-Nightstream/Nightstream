import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.LinearCompiler
import Nightstream.SuperNeo.Concrete.Relation

/-!
Contract: compact fail-closed schema for the generated fixed-F-prime source
census and its independently ordered encoded/CE coordinate ownership census.

Owns: source-ordered segments, independently coordinate-ordered owner runs,
exact universe partition proofs, exhaustive source roles, compact
ordinary-field placement, the bridge to the generic linear compiler, and the
theorem placing every eligible 41-coordinate word in one commitment-bound
fresh CE assignment with verifier-owned `b = 2` norm.

Does not own: concrete fixed-F-prime segment data, handwritten counts,
manifest generation, Rust witness materialization, public/private-bit
encoding, canonical-u64 encoding, SIS opening semantics, or derived-column
lowering. It also does not own a combined fixed materializer or proof that
selector-composed branches share one assignment arena.

Emits constraints: no. A production generator must instantiate
`GeneratedArtifact`; this file deliberately stores runs rather than one record
per encoded or CE coordinate. Production now uses 41-coordinate ordinary
slots, but this generic schema does not claim that their exact source-to-slot
placement has already been exported as a checked `GeneratedArtifact`.

Authority boundary: ownership and role data are census evidence only. CE
authority comes from `CE.Holds`, whose opening recomputes the commitment and
checks the verifier-selected norm on the same assignment. The norm applies to
the exact 41 CE coordinates, never to the decoded source residue.

ABI boundary: encoded coordinate zero and CE coordinate zero are each required
to have an explicitly excluded owner. This reflects the concrete low-norm ABI
in which assignment coordinate zero is ONE and the whole assignment is the CE
committed witness. Source order is not coordinate order. Coordinate-only
owners cover selectors, alignment padding, and synthetic fields. One artifact
describes one concrete production branch; base and recursive artifacts remain
separate until a combined materializer and selector-composition proof exist.

| Source-column role | Ordinary 41-coordinate encoding? | Owner |
|---|---:|---|
| constant one | no | assignment ABI |
| ordinary private field | yes | this compiler path |
| private Boolean | no | private-bit binding |
| public bit | no | public-input/bit binding |
| canonical u64 | no | canonical-u64 path |
| SIS opening | no | SIS opening path |
| linearly derived | no | source linear compiler |
| structural balanced alias | no | shifted-ternary layout |
| gadget-derived | no | emitting gadget |
| product-derived | no | emitting product gate |
| gadget temporary | no | emitting gadget |

| Surface | Mathematical obligation | Main result | Tier |
|---|---|---|---|
| compact runs | first start zero, positive lengths, abutting neighbors, exact final end | `ExactPartition` | artifact interface |
| exact universes | every in-range coordinate has exactly one owner; distinct owners are disjoint | `existsUniqueOwner`, `distinctOwnersDisjoint` | kernel theorem |
| source census | every source column occurs once and has one exhaustive role | `Valid.existsUniqueSlotForSource` | artifact theorem |
| coordinate census | source-backed and coordinate-only runs are ordered independently of source columns | `CoordinateOwnerRun` | artifact interface |
| ordinary placement | one 41-coordinate encoded and CE block per eligible source segment | `Valid.ordinaryOwnerFor` | artifact theorem |
| compiler bridge | eligible field order fixes source, encoded, and CE coordinates | `CompilerBinding` | refinement interface |
| CE authority | all eligible coordinates come from one committed, norm-checked assignment | `eligibleSlots_share_committed_freshCe_assignment` | conditional model theorem |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFieldLayout

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.CenteredTernaryField
open Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged
open Nightstream.Implementation.R1CS.CenteredTernaryLinearCompiler
open Nightstream.SuperNeo

/-- Exhaustive fail-closed source-column classification. Adding a production
kind requires extending this type and the explicit exclusion theorem. -/
inductive SlotRole where
  | constantOne
  | ordinaryPrivateField
  | privateBoolean
  | publicBit
  | canonicalU64
  | sisOpening
  | linearlyDerived
  | structuralBalancedAlias
  | gadgetDerived
  | productDerived
  | gadgetTemporary
deriving DecidableEq, Repr, Inhabited

namespace SlotRole

def Eligible (role : SlotRole) : Prop :=
  role = .ordinaryPrivateField

def ExplicitlyExcluded (role : SlotRole) : Prop :=
  role = .constantOne ∨
    role = .privateBoolean ∨
    role = .publicBit ∨
    role = .canonicalU64 ∨
    role = .sisOpening ∨
    role = .linearlyDerived ∨
    role = .structuralBalancedAlias ∨
    role = .gadgetDerived ∨
    role = .productDerived ∨
    role = .gadgetTemporary

instance (role : SlotRole) : Decidable role.Eligible := by
  unfold Eligible
  infer_instance

instance (role : SlotRole) : Decidable role.ExplicitlyExcluded := by
  unfold ExplicitlyExcluded
  infer_instance

theorem eligible_or_explicitlyExcluded (role : SlotRole) :
    role.Eligible ∨ role.ExplicitlyExcluded := by
  cases role <;> simp [Eligible, ExplicitlyExcluded]

theorem eligible_iff_not_explicitlyExcluded (role : SlotRole) :
    role.Eligible ↔ ¬ role.ExplicitlyExcluded := by
  cases role <;> simp [Eligible, ExplicitlyExcluded]

end SlotRole

/-- A compact half-open coordinate interval `[start, start + length)`. -/
structure CoordinateRun where
  start : Nat
  length : Nat
deriving DecidableEq, Repr, Inhabited

namespace CoordinateRun

def endExclusive (run : CoordinateRun) : Nat :=
  run.start + run.length

def Contains (run : CoordinateRun) (coordinate : Nat) : Prop :=
  run.start ≤ coordinate ∧ coordinate < run.endExclusive

instance (run : CoordinateRun) (coordinate : Nat) :
    Decidable (run.Contains coordinate) := by
  unfold Contains
  infer_instance

end CoordinateRun

/-- Recursive exact-partition certificate. At each step the current run must
start at the prior end and have positive length. The empty tail closes only
when the accumulated end is the declared universe size. -/
def ExactPartitionFrom {Owner : Type}
    (runOf : Owner → CoordinateRun)
    (cursor count : Nat) : List Owner → Prop
  | [] => cursor = count
  | owner :: tail =>
      (runOf owner).start = cursor ∧
      0 < (runOf owner).length ∧
      ExactPartitionFrom runOf (runOf owner).endExclusive count tail

def ExactPartition {Owner : Type}
    (runOf : Owner → CoordinateRun)
    (count : Nat) (owners : List Owner) : Prop :=
  ExactPartitionFrom runOf 0 count owners

namespace ExactPartition

private theorem ownerStartsAtOrAfter
    {Owner : Type} {runOf : Owner → CoordinateRun}
    {cursor count : Nat} {owners : List Owner}
    (partition : ExactPartitionFrom runOf cursor count owners)
    {owner : Owner} (member : owner ∈ owners) :
    cursor ≤ (runOf owner).start := by
  induction owners generalizing cursor with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      simp only [ExactPartitionFrom] at partition
      rcases partition with ⟨headStart, headPositive, tailPartition⟩
      rcases List.mem_cons.mp member with rfl | tailMember
      · omega
      · have tailStart := inductionHypothesis tailPartition tailMember
        have cursorLeEnd : cursor ≤ (runOf head).endExclusive := by
          simp only [CoordinateRun.endExclusive]
          omega
        exact Nat.le_trans cursorLeEnd tailStart

private theorem ownerLengthPositiveFrom
    {Owner : Type} {runOf : Owner → CoordinateRun}
    {cursor count : Nat} {owners : List Owner}
    (partition : ExactPartitionFrom runOf cursor count owners)
    {owner : Owner} (member : owner ∈ owners) :
    0 < (runOf owner).length := by
  induction owners generalizing cursor with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      simp only [ExactPartitionFrom] at partition
      rcases partition with ⟨headStart, headPositive, tailPartition⟩
      rcases List.mem_cons.mp member with rfl | tailMember
      · exact headPositive
      · exact inductionHypothesis tailPartition tailMember

private theorem hasFinalEndFrom
    {Owner : Type} {runOf : Owner → CoordinateRun}
    {cursor count : Nat} {owners : List Owner}
    (partition : ExactPartitionFrom runOf cursor count owners)
    (nonempty : owners ≠ []) :
    ∃ owner : Owner,
      owner ∈ owners ∧ (runOf owner).endExclusive = count := by
  induction owners generalizing cursor with
  | nil => exact (nonempty rfl).elim
  | cons head tail inductionHypothesis =>
      simp only [ExactPartitionFrom] at partition
      rcases partition with ⟨headStart, headPositive, tailPartition⟩
      by_cases tailEmpty : tail = []
      · subst tail
        simp only [ExactPartitionFrom] at tailPartition
        exact ⟨head, List.mem_cons.mpr (Or.inl rfl), tailPartition⟩
      · rcases inductionHypothesis tailPartition tailEmpty with
          ⟨owner, member, finalEnd⟩
        exact ⟨owner, List.mem_cons.mpr (Or.inr member), finalEnd⟩

private theorem coversFrom
    {Owner : Type} {runOf : Owner → CoordinateRun}
    {cursor count : Nat} {owners : List Owner}
    (partition : ExactPartitionFrom runOf cursor count owners)
    {coordinate : Nat}
    (cursorLe : cursor ≤ coordinate)
    (coordinateLt : coordinate < count) :
    ∃ owner : Owner,
      owner ∈ owners ∧ (runOf owner).Contains coordinate := by
  induction owners generalizing cursor with
  | nil =>
      simp only [ExactPartitionFrom] at partition
      omega
  | cons head tail inductionHypothesis =>
      simp only [ExactPartitionFrom] at partition
      rcases partition with ⟨headStart, headPositive, tailPartition⟩
      by_cases coordinateLtEnd : coordinate < (runOf head).endExclusive
      · refine ⟨head, List.mem_cons.mpr (Or.inl rfl), ?_⟩
        exact ⟨by omega, coordinateLtEnd⟩
      · have endLe : (runOf head).endExclusive ≤ coordinate :=
          Nat.le_of_not_gt coordinateLtEnd
        rcases inductionHypothesis tailPartition endLe with
          ⟨owner, ownerMember, contains⟩
        exact ⟨owner, List.mem_cons.mpr (Or.inr ownerMember), contains⟩

private theorem ownerUniqueFrom
    {Owner : Type} {runOf : Owner → CoordinateRun}
    {cursor count : Nat} {owners : List Owner}
    (partition : ExactPartitionFrom runOf cursor count owners)
    {coordinate : Nat} {left right : Owner}
    (leftMember : left ∈ owners)
    (leftContains : (runOf left).Contains coordinate)
    (rightMember : right ∈ owners)
    (rightContains : (runOf right).Contains coordinate) :
    left = right := by
  induction owners generalizing cursor left right with
  | nil => simp at leftMember
  | cons head tail inductionHypothesis =>
      simp only [ExactPartitionFrom] at partition
      rcases partition with ⟨headStart, headPositive, tailPartition⟩
      rcases List.mem_cons.mp leftMember with rfl | leftTail
      · rcases List.mem_cons.mp rightMember with rfl | rightTail
        · rfl
        · have rightStart := ownerStartsAtOrAfter tailPartition rightTail
          rcases leftContains with ⟨leftLower, leftUpper⟩
          rcases rightContains with ⟨rightLower, rightUpper⟩
          exfalso
          omega
      · rcases List.mem_cons.mp rightMember with rfl | rightTail
        · have leftStart := ownerStartsAtOrAfter tailPartition leftTail
          rcases leftContains with ⟨leftLower, leftUpper⟩
          rcases rightContains with ⟨rightLower, rightUpper⟩
          exfalso
          omega
        · exact inductionHypothesis tailPartition leftTail leftContains
            rightTail rightContains

theorem firstStartsAtZero
    {Owner : Type} {runOf : Owner → CoordinateRun}
    {count : Nat} {head : Owner} {tail : List Owner}
    (partition : ExactPartition runOf count (head :: tail)) :
    (runOf head).start = 0 := by
  exact partition.1

theorem ownerLengthPositive
    {Owner : Type} {runOf : Owner → CoordinateRun}
    {count : Nat} {owners : List Owner}
    (partition : ExactPartition runOf count owners)
    {owner : Owner} (member : owner ∈ owners) :
    0 < (runOf owner).length :=
  ownerLengthPositiveFrom partition member

theorem firstAbutsSecond
    {Owner : Type} {runOf : Owner → CoordinateRun}
    {count : Nat} {first second : Owner} {tail : List Owner}
    (partition : ExactPartition runOf count (first :: second :: tail)) :
    (runOf first).endExclusive = (runOf second).start := by
  exact partition.2.2.1.symm

theorem finalEndsAtCount
    {Owner : Type} {runOf : Owner → CoordinateRun}
    {count : Nat} {owners : List Owner}
    (partition : ExactPartition runOf count owners)
    (nonempty : owners ≠ []) :
    ∃ owner : Owner,
      owner ∈ owners ∧ (runOf owner).endExclusive = count :=
  hasFinalEndFrom partition nonempty

theorem existsUniqueOwner
    {Owner : Type} {runOf : Owner → CoordinateRun}
    {count : Nat} {owners : List Owner}
    (partition : ExactPartition runOf count owners)
    {coordinate : Nat} (coordinateLt : coordinate < count) :
    ∃ owner : Owner,
      owner ∈ owners ∧
      (runOf owner).Contains coordinate ∧
      ∀ other : Owner,
        other ∈ owners →
        (runOf other).Contains coordinate →
        other = owner := by
  rcases coversFrom partition (Nat.zero_le coordinate) coordinateLt with
    ⟨owner, ownerMember, ownerContains⟩
  refine ⟨owner, ownerMember, ownerContains, ?_⟩
  intro other otherMember otherContains
  exact ownerUniqueFrom partition otherMember otherContains
    ownerMember ownerContains

theorem distinctOwnersDisjoint
    {Owner : Type} {runOf : Owner → CoordinateRun}
    {count : Nat} {owners : List Owner}
    (partition : ExactPartition runOf count owners)
    {left right : Owner}
    (leftMember : left ∈ owners)
    (rightMember : right ∈ owners)
    (different : left ≠ right) :
    ∀ coordinate,
      ¬ ((runOf left).Contains coordinate ∧
        (runOf right).Contains coordinate) := by
  intro coordinate both
  apply different
  exact ownerUniqueFrom partition leftMember both.1 rightMember both.2

end ExactPartition

/-- One source-ordered ownership segment. This list is the source census only;
its order is never reused as encoded or CE coordinate order. -/
structure SourceSegment where
  ownerPath : String
  role : SlotRole
  source : CoordinateRun
deriving DecidableEq, Repr, Inhabited

namespace SourceSegment

def eligibleFieldCount (segment : SourceSegment) : Nat :=
  if segment.role = .ordinaryPrivateField then segment.source.length else 0

def eligibleCountOf : List SourceSegment → Nat
  | [] => 0
  | segment :: tail => segment.eligibleFieldCount + eligibleCountOf tail

/-- Source-order lookup for the ordinary-field compiler domain. The default
case is unreachable whenever the index is below `eligibleCountOf`. -/
def eligibleSourceColumnAt : List SourceSegment → Nat → Nat
  | [], _ => 0
  | segment :: tail, field =>
      if segment.role = .ordinaryPrivateField then
        if field < segment.source.length then
          segment.source.start + field
        else
          eligibleSourceColumnAt tail (field - segment.source.length)
      else
        eligibleSourceColumnAt tail field

end SourceSegment

/-- Coordinate ownership is either tied to one exact source segment or exists
only in the lowered assignment, as for selectors, padding, and synthetic
fields. -/
inductive CoordinateOwner where
  | source (segment : SourceSegment)
  | coordinateOnly (ownerPath : String) (role : SlotRole)
deriving DecidableEq, Repr, Inhabited

namespace CoordinateOwner

def role : CoordinateOwner → SlotRole
  | .source segment => segment.role
  | .coordinateOnly _ role => role

def ownerPath : CoordinateOwner → String
  | .source segment => segment.ownerPath
  | .coordinateOnly ownerPath _ => ownerPath

end CoordinateOwner

/-- One coordinate-ordered owner run. The encoded and CE runs share an owner
but retain separate starts so their equality is never assumed without
generated evidence. -/
structure CoordinateOwnerRun where
  owner : CoordinateOwner
  encoded : CoordinateRun
  ce : CoordinateRun
deriving DecidableEq, Repr, Inhabited

namespace CoordinateOwnerRun

def role (run : CoordinateOwnerRun) : SlotRole :=
  run.owner.role

def ownerPath (run : CoordinateOwnerRun) : String :=
  run.owner.ownerPath

end CoordinateOwnerRun

/-- One ordinary field's source column and the starts of its two exact
41-coordinate blocks. Placements are compiler-binding evidence, not a flat
artifact table. -/
structure FieldPlacement where
  sourceColumn : Nat
  encodedStart : Nat
  ceStart : Nat
deriving DecidableEq, Repr, Inhabited

/-- Pure compact data emitted from the exact fixed-F-prime builder trace.
Source segments and coordinate owners are independent ordered axes. -/
structure Manifest where
  schemaVersion : Nat
  profile : String
  sourceColumnCount : Nat
  encodedColumnCount : Nat
  ceAssignmentLength : Nat
  sourceSegments : List SourceSegment
  coordinateOwners : List CoordinateOwnerRun
deriving DecidableEq, Repr, Inhabited

namespace Manifest

def eligibleSegments (manifest : Manifest) : List SourceSegment :=
  manifest.sourceSegments.filter fun segment =>
    decide (segment.role = .ordinaryPrivateField)

def eligibleCount (manifest : Manifest) : Nat :=
  SourceSegment.eligibleCountOf manifest.sourceSegments

def eligibleSourceColumnAt (manifest : Manifest) (field : Nat) : Nat :=
  SourceSegment.eligibleSourceColumnAt manifest.sourceSegments field

/-- Kernel obligations one generated concrete-branch manifest must discharge. -/
structure Valid (manifest : Manifest) : Prop where
  sourceNonempty : 0 < manifest.sourceColumnCount
  sourcePartition : ExactPartition SourceSegment.source
    manifest.sourceColumnCount manifest.sourceSegments
  encodedPartition : ExactPartition CoordinateOwnerRun.encoded
    manifest.encodedColumnCount manifest.coordinateOwners
  cePartition : ExactPartition CoordinateOwnerRun.ce
    manifest.ceAssignmentLength manifest.coordinateOwners
  sourceLinks : ∀ owner ∈ manifest.coordinateOwners,
    ∀ segment, owner.owner = .source segment →
      segment ∈ manifest.sourceSegments
  eligibleOwnersHaveSource : ∀ owner ∈ manifest.coordinateOwners,
    owner.role.Eligible →
      ∃ segment, owner.owner = .source segment
  constantOneIffSourceZero : ∀ segment ∈ manifest.sourceSegments,
    ∀ sourceColumn, segment.source.Contains sourceColumn →
      (segment.role = .constantOne ↔ sourceColumn = 0)
  ordinaryShape : ∀ segment ∈ manifest.sourceSegments,
      segment.role.Eligible →
        ∃ owner : CoordinateOwnerRun,
          owner ∈ manifest.coordinateOwners ∧
          owner.owner = .source segment ∧
          owner.encoded.length = segment.source.length * digitCount ∧
          owner.ce.length = segment.source.length * digitCount ∧
          ∀ other : CoordinateOwnerRun,
            other ∈ manifest.coordinateOwners →
            other.owner = .source segment →
            other = owner
  /-- Concrete encoded-assignment ABI: coordinate zero exists and its unique
  owner is explicitly excluded from ordinary-field optimization. -/
  encodedZeroReserved : 0 < manifest.encodedColumnCount ∧
    ∀ owner ∈ manifest.coordinateOwners,
      owner.encoded.Contains 0 → owner.role.ExplicitlyExcluded
  /-- Concrete committed-witness ABI: CE coordinate zero is also ONE and may
  never enter ordinary-field placement. -/
  ceZeroReserved : 0 < manifest.ceAssignmentLength ∧
    ∀ owner ∈ manifest.coordinateOwners,
      owner.ce.Contains 0 → owner.role.ExplicitlyExcluded

theorem Valid.existsUniqueSlotForSource {manifest : Manifest}
    (valid : manifest.Valid) {sourceColumn : Nat}
    (sourceColumnLt : sourceColumn < manifest.sourceColumnCount) :
    ∃ segment : SourceSegment,
      segment ∈ manifest.sourceSegments ∧
      segment.source.Contains sourceColumn ∧
      ∀ other : SourceSegment,
        other ∈ manifest.sourceSegments →
        other.source.Contains sourceColumn →
        other = segment :=
  ExactPartition.existsUniqueOwner valid.sourcePartition sourceColumnLt

theorem Valid.encodedCoordinateHasUniqueOwner {manifest : Manifest}
    (valid : manifest.Valid) {coordinate : Nat}
    (coordinateLt : coordinate < manifest.encodedColumnCount) :
    ∃ owner : CoordinateOwnerRun,
      owner ∈ manifest.coordinateOwners ∧
      owner.encoded.Contains coordinate ∧
      ∀ other : CoordinateOwnerRun,
        other ∈ manifest.coordinateOwners →
        other.encoded.Contains coordinate →
        other = owner :=
  ExactPartition.existsUniqueOwner valid.encodedPartition coordinateLt

theorem Valid.ceCoordinateHasUniqueOwner {manifest : Manifest}
    (valid : manifest.Valid) {coordinate : Nat}
    (coordinateLt : coordinate < manifest.ceAssignmentLength) :
    ∃ owner : CoordinateOwnerRun,
      owner ∈ manifest.coordinateOwners ∧
      owner.ce.Contains coordinate ∧
      ∀ other : CoordinateOwnerRun,
        other ∈ manifest.coordinateOwners →
        other.ce.Contains coordinate →
        other = owner :=
  ExactPartition.existsUniqueOwner valid.cePartition coordinateLt

theorem Valid.encodedDistinctOwnersDisjoint {manifest : Manifest}
    (valid : manifest.Valid)
    {left right : CoordinateOwnerRun}
    (leftMember : left ∈ manifest.coordinateOwners)
    (rightMember : right ∈ manifest.coordinateOwners)
    (different : left ≠ right) :
    ∀ coordinate,
      ¬ (left.encoded.Contains coordinate ∧
        right.encoded.Contains coordinate) :=
  ExactPartition.distinctOwnersDisjoint
    valid.encodedPartition leftMember rightMember different

theorem Valid.ceDistinctOwnersDisjoint {manifest : Manifest}
    (valid : manifest.Valid)
    {left right : CoordinateOwnerRun}
    (leftMember : left ∈ manifest.coordinateOwners)
    (rightMember : right ∈ manifest.coordinateOwners)
    (different : left ≠ right) :
    ∀ coordinate,
      ¬ (left.ce.Contains coordinate ∧
        right.ce.Contains coordinate) :=
  ExactPartition.distinctOwnersDisjoint
    valid.cePartition leftMember rightMember different

theorem Valid.ordinaryOwnerFor {manifest : Manifest}
    (valid : manifest.Valid)
    {segment : SourceSegment}
    (segmentMember : segment ∈ manifest.sourceSegments)
    (eligible : segment.role.Eligible) :
    ∃ owner : CoordinateOwnerRun,
      owner ∈ manifest.coordinateOwners ∧
      owner.owner = .source segment ∧
      owner.encoded.length = segment.source.length * digitCount ∧
      owner.ce.length = segment.source.length * digitCount ∧
      ∀ other : CoordinateOwnerRun,
        other ∈ manifest.coordinateOwners →
        other.owner = .source segment →
      other = owner :=
  valid.ordinaryShape segment segmentMember eligible

theorem Valid.coordinateOnlyOwnerIsExcluded {manifest : Manifest}
    (valid : manifest.Valid) {owner : CoordinateOwnerRun}
    (ownerMember : owner ∈ manifest.coordinateOwners)
    {ownerPath : String} {role : SlotRole}
    (coordinateOnly : owner.owner = .coordinateOnly ownerPath role) :
    role.ExplicitlyExcluded := by
  rcases SlotRole.eligible_or_explicitlyExcluded role with eligible | excluded
  · have ownerEligible : owner.role.Eligible := by
      simpa [CoordinateOwnerRun.role, CoordinateOwner.role, coordinateOnly]
        using eligible
    rcases valid.eligibleOwnersHaveSource owner ownerMember ownerEligible with
      ⟨segment, sourceOwner⟩
    simp [coordinateOnly] at sourceOwner
  · exact excluded

theorem Valid.sourceZeroHasConstantOneOwner {manifest : Manifest}
    (valid : manifest.Valid) :
    ∃ segment : SourceSegment,
      segment ∈ manifest.sourceSegments ∧
      segment.source.Contains 0 ∧
      segment.role = .constantOne := by
  rcases valid.existsUniqueSlotForSource valid.sourceNonempty with
    ⟨segment, member, contains, unique⟩
  exact ⟨segment, member, contains,
    (valid.constantOneIffSourceZero segment member 0 contains).2 rfl⟩

theorem Valid.encodedZeroHasExcludedOwner {manifest : Manifest}
    (valid : manifest.Valid) :
    ∃ owner : CoordinateOwnerRun,
      owner ∈ manifest.coordinateOwners ∧
      owner.encoded.Contains 0 ∧
      owner.role.ExplicitlyExcluded := by
  rcases valid.encodedCoordinateHasUniqueOwner valid.encodedZeroReserved.1 with
    ⟨owner, member, contains, unique⟩
  exact ⟨owner, member, contains,
    valid.encodedZeroReserved.2 owner member contains⟩

theorem Valid.ceZeroHasExcludedOwner {manifest : Manifest}
    (valid : manifest.Valid) :
    ∃ owner : CoordinateOwnerRun,
      owner ∈ manifest.coordinateOwners ∧
      owner.ce.Contains 0 ∧
      owner.role.ExplicitlyExcluded := by
  rcases valid.ceCoordinateHasUniqueOwner valid.ceZeroReserved.1 with
    ⟨owner, member, contains, unique⟩
  exact ⟨owner, member, contains,
    valid.ceZeroReserved.2 owner member contains⟩

/-- Assignment-level bridge. The compiler supplies compact run-derived
placements; every placement reads the corresponding coordinate from one
concrete CE witness list. -/
def BindsEncodedAssignment (manifest : Manifest)
    (placement : Fin manifest.eligibleCount → FieldPlacement)
    (encoded : Nat → Nat)
    (assignment : Nightstream.SuperNeo.Concrete.Assignment) : Prop :=
  assignment.length = manifest.ceAssignmentLength ∧
    ∀ field : Fin manifest.eligibleCount,
      ∀ index, index < digitCount →
        encoded ((placement field).encodedStart + index) =
          (assignment.getD ((placement field).ceStart + index) 0).val

end Manifest

/-- One concrete production branch must export this pair from generated compact
data. Base and recursive branches use separate artifacts until a combined
materializer and selector-composition proof exist. The schema itself
deliberately contains no fixed-F-prime census. -/
structure GeneratedArtifact where
  manifest : Manifest
  valid : manifest.Valid

/-- Exact ordering bridge from compact generated ordinary segments to the
generic linear-substitution compiler and CE assignment. `placementOwned`
connects each ordinary source field to its source-backed owner run without
requiring source and coordinate order to coincide. -/
structure CompilerBinding (artifact : GeneratedArtifact) where
  layout :
    Nightstream.Implementation.R1CS.CenteredTernaryLinearCompiler.Layout
      artifact.manifest.eligibleCount
  placement : Fin artifact.manifest.eligibleCount → FieldPlacement
  ceCoordinate : Fin artifact.manifest.eligibleCount → Nat → Nat
  placementSource_eq : ∀ field,
    (placement field).sourceColumn =
      artifact.manifest.eligibleSourceColumnAt field.val
  placementOwned : ∀ field,
    ∃ segment : SourceSegment,
      ∃ owner : CoordinateOwnerRun,
        ∃ offset : Nat,
          segment ∈ artifact.manifest.sourceSegments ∧
          segment.role.Eligible ∧
          owner ∈ artifact.manifest.coordinateOwners ∧
          owner.owner = .source segment ∧
          offset < segment.source.length ∧
          (placement field).sourceColumn = segment.source.start + offset ∧
          (placement field).encodedStart =
            owner.encoded.start + offset * digitCount ∧
          (placement field).ceStart =
            owner.ce.start + offset * digitCount
  placementWithin : ∀ field,
    (placement field).encodedStart + digitCount ≤
        artifact.manifest.encodedColumnCount ∧
      (placement field).ceStart + digitCount ≤
        artifact.manifest.ceAssignmentLength
  sourceColumn_eq : ∀ field,
    layout.sourceColumn field =
      (placement field).sourceColumn
  encodedColumn_eq : ∀ field digit, digit < digitCount →
    layout.encodedColumn field digit =
      (placement field).encodedStart + digit
  ceCoordinate_eq : ∀ field digit, digit < digitCount →
    ceCoordinate field digit =
      (placement field).ceStart + digit

private theorem getD_mem_of_lt {values : List Nightstream.SuperNeo.Concrete.F}
    {index : Nat} (indexLt : index < values.length) :
    values.getD index 0 ∈ values := by
  have member := List.getElem_mem (l := values) indexLt
  rwa [List.getElem_eq_getD 0] at member

/-- Every coordinate of every eligible ordinary-field word is read from one
and the same assignment whose Ajtai commitment and strict norm are checked by
the accepted fresh CE opening. The source residue is obtained later through
`decodedPrivateColumn`; this theorem makes no claim that it is norm-bounded. -/
theorem eligibleSlots_share_committed_freshCe_assignment
    (artifact : GeneratedArtifact)
    (compiler : CompilerBinding artifact)
    (context : Nightstream.SuperNeo.Concrete.Context)
    (params : GlobalParams)
    (statement : Nightstream.SuperNeo.Concrete.CEStatement)
    (encoded : Nat → Nat)
    (assignment : Nightstream.SuperNeo.Concrete.Assignment)
    (baseBound : params.b = 2)
    (freshStage : statement.stage = .fresh)
    (binding : artifact.manifest.BindsEncodedAssignment
      compiler.placement encoded assignment)
    (accepted : CE.Holds
      (Nightstream.SuperNeo.Concrete.relationSemantics context)
      params statement assignment) :
    Nightstream.SuperNeo.Concrete.ajtaiCommit context.ajtaiKey assignment =
        statement.commitment ∧
      Nightstream.SuperNeo.Concrete.normBounded 2 assignment ∧
      PrivateCoordinatesNormBoundTwo compiler.layout encoded := by
  have expanded :=
    (Nightstream.SuperNeo.Concrete.ceMembership_iff
      context params statement assignment).mp accepted
  have normTwo : Nightstream.SuperNeo.Concrete.normBounded 2 assignment := by
    have normAtStage := expanded.2.2.1
    simpa [NormStage.bound, freshStage, baseBound] using normAtStage
  refine ⟨expanded.1, normTwo, ?_⟩
  intro field digit
  let placement := compiler.placement field
  have within := compiler.placementWithin field
  change placement.encodedStart + digitCount ≤
      artifact.manifest.encodedColumnCount ∧
    placement.ceStart + digitCount ≤
      artifact.manifest.ceAssignmentLength at within
  have ceCoordinateLtManifest :
      placement.ceStart + digit.val < artifact.manifest.ceAssignmentLength := by
    omega
  have ceCoordinateLt :
      compiler.ceCoordinate field digit.val < assignment.length := by
    rw [compiler.ceCoordinate_eq field digit.val digit.isLt, binding.1]
    exact ceCoordinateLtManifest
  have assignmentMember :
      assignment.getD (compiler.ceCoordinate field digit.val) 0 ∈ assignment :=
    getD_mem_of_lt ceCoordinateLt
  have assignmentCentered :
      CenteredResidue
        (assignment.getD (compiler.ceCoordinate field digit.val) 0).val :=
    concrete_normBounded_two_implies_centered normTwo assignmentMember
  have encodedEq :=
    binding.2 field digit.val digit.isLt
  rw [compiler.encodedColumn_eq field digit.val digit.isLt, encodedEq,
    ← compiler.ceCoordinate_eq field digit.val digit.isLt]
  exact normBoundTwo_iff_centeredResidue.mpr assignmentCentered

/-- Regression against the invalid inference that a decoded source field must
itself lie in the centered alphabet. The canonical source residue `2` is not a
centered digit, while its exact 41-coordinate word is norm-bounded and decodes
back to `2`. -/
theorem normBounded_word_can_decode_nonCentered_source :
    (∀ digit : Fin digitCount, NormBoundTwo (finiteEncode 2 digit)) ∧
      decodeFiniteWord (finiteEncode 2) = 2 ∧
      ¬ CenteredResidue 2 := by
  refine ⟨?_, decodeFiniteWord_finiteEncode (by native_decide), by native_decide⟩
  intro digit
  exact normBoundTwo_iff_centeredResidue.mpr (finiteEncode_alphabet 2 digit)

end Nightstream.Implementation.R1CS.FPrimeFieldLayout
