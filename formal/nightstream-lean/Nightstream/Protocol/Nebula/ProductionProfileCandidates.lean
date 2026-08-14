import Nightstream.Protocol.Nebula.MemoryWireGeometry
import Nightstream.Protocol.Nebula.Profile

/-!
Contract: successor field-native profile candidates for Nebula on SuperNeo.

The bit-serial factor-one V2 profile remains an exact reference relation. It
is not reused as the production profile. This file owns four distinct
field-native candidate identities, their segment partitions, their delayed
lifecycle counts, and the exact mixed-coordinate size of the proposed public
carrier.

No candidate is selected here. Selection requires compiled relation and
terminal artifacts plus measured rows, columns, nonzero entries, memory, and
time. These arithmetic projections are not performance measurements.

Does not own generated rows, a verifier key, a plan digest, Rust, terminal
verification, security estimates, or benchmark results.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.ProductionProfileCandidates

/-- Candidate identity. Each factor has a distinct public profile version. -/
inductive Id where
  | e1
  | e4
  | e8
  | e16
deriving DecidableEq, Repr

def all : List Id := [.e1, .e4, .e8, .e16]

theorem mem_all (candidate : Id) : candidate ∈ all := by
  cases candidate <;> simp [all]

/-- Sequential checked steps proved inside one fresh claim. -/
def checkedStepsPerFreshClaim : Id -> Nat
  | .e1 => 1
  | .e4 => 4
  | .e8 => 8
  | .e16 => 16

/-- The versions are different because each factor selects a different
relation, lifecycle, transcript census, security census, and verifier key. -/
def version : Id -> Nat
  | .e1 => 3
  | .e4 => 4
  | .e8 => 5
  | .e16 => 6

def identity (candidate : Id) : Profile.Identity where
  name := .paddedRowIdentityMemoryFieldNative
  version := version candidate
  checkedStepsPerFreshClaim := checkedStepsPerFreshClaim candidate
  commitmentEncoding := .shiftedTernary41V1

/-- Exact finite set of protocol identities that the current V2 Lean package
can encode in verifier-key-owned transcript frames. This predicate is a
configuration check. It is not a cryptographic assumption. -/
def SupportedIdentity (profile : Profile.Identity) : Prop :=
  profile = Profile.v2 \/ exists candidate, profile = identity candidate

theorem v2_supported : SupportedIdentity Profile.v2 :=
  Or.inl rfl

theorem identity_supported (candidate : Id) :
    SupportedIdentity (identity candidate) :=
  Or.inr <| Exists.intro candidate rfl

/-- The four candidates and the bit-serial V2 reference cannot share a
profile identity or verifier key. -/
theorem identities_pairwise_distinct :
    (Profile.v2 :: all.map identity).Pairwise (fun left right => left ≠ right) := by
  decide

theorem identity_ne_v2 (candidate : Id) : identity candidate ≠ Profile.v2 := by
  cases candidate <;> decide

def stepsPerSegment : Nat := 1088
def maximumSegments : Nat := 64

theorem checkedSteps_positive (candidate : Id) :
    0 < checkedStepsPerFreshClaim candidate := by
  cases candidate <;> decide

theorem checkedSteps_divide_segment (candidate : Id) :
    checkedStepsPerFreshClaim candidate ∣ stepsPerSegment := by
  cases candidate <;> decide

def claimsPerSegment (candidate : Id) : Nat :=
  stepsPerSegment / checkedStepsPerFreshClaim candidate

def maximumClaims (candidate : Id) : Nat :=
  maximumSegments * claimsPerSegment candidate

def maximumAugmentedInvocations (candidate : Id) : Nat :=
  maximumClaims candidate + 1

/-- Every candidate partitions one segment exactly. A batch cannot leave an
unchecked suffix at the segment boundary. -/
theorem exact_segment_partition (candidate : Id) :
    claimsPerSegment candidate * checkedStepsPerFreshClaim candidate =
      stepsPerSegment := by
  exact Nat.div_mul_cancel (checkedSteps_divide_segment candidate)

/-- A local claim whose index is in range cannot cross a segment boundary. -/
theorem local_batch_end_le_segment
    (candidate : Id) {claim : Nat}
    (inRange : claim < claimsPerSegment candidate) :
    claim * checkedStepsPerFreshClaim candidate +
        checkedStepsPerFreshClaim candidate <= stepsPerSegment := by
  cases candidate <;>
    simp [claimsPerSegment, checkedStepsPerFreshClaim, stepsPerSegment] at inRange ⊢ <;>
    omega

/-- Exact lifecycle counts used only to compare the four candidates. -/
theorem candidate_count_table :
    claimsPerSegment .e1 = 1088 /\
      maximumClaims .e1 = 69632 /\
      maximumAugmentedInvocations .e1 = 69633 /\
    claimsPerSegment .e4 = 272 /\
      maximumClaims .e4 = 17408 /\
      maximumAugmentedInvocations .e4 = 17409 /\
    claimsPerSegment .e8 = 136 /\
      maximumClaims .e8 = 8704 /\
      maximumAugmentedInvocations .e8 = 8705 /\
    claimsPerSegment .e16 = 68 /\
      maximumClaims .e16 = 4352 /\
      maximumAugmentedInvocations .e16 = 4353 := by
  decide

/-! ## Field-native public-carrier geometry -/

/-- Only the CCS public bits remain fixed narrow claim coordinates. Candidate
identity is type- and verifier-key-bound. The external application statement
is parsed at the verifier boundary and is not an unauthenticated fresh-claim
sidecar. -/
def fixedNarrowCoordinates : Nat := 540

/-- Bounded counters stay bit-decomposed in the successor relation. -/
def memoryCounterCoordinatesPerCheckedStep : Nat :=
  MemoryWireGeometry.stepCounterBits

/-- Challenges, products, and roots stay as native Goldilocks coordinates.
They must not be expanded into the reference relation's 64-bit words. -/
def memoryFieldCoordinatesPerCheckedStep : Nat :=
  MemoryWireGeometry.challengeBaseFieldLimbs +
    2 * MemoryWireGeometry.productStateBaseFieldLimbs +
    3 * MemoryWireGeometry.rootsBaseFieldLimbs

/-- Complete mixed-coordinate memory suffix for one checked step. -/
def memorySuffixCoordinatesPerCheckedStep : Nat :=
  memoryCounterCoordinatesPerCheckedStep +
    memoryFieldCoordinatesPerCheckedStep

theorem memorySuffixCoordinate_split_exact :
    memoryCounterCoordinatesPerCheckedStep = 116 /\
      memoryFieldCoordinatesPerCheckedStep = 76 /\
      memorySuffixCoordinatesPerCheckedStep = 192 := by
  decide

def narrowCoordinates (candidate : Id) : Nat :=
  fixedNarrowCoordinates +
    checkedStepsPerFreshClaim candidate *
      memoryCounterCoordinatesPerCheckedStep

/-- Complete fourteen-running paper carrier at one augmented-relation
exponent. Each additional cube variable adds one extension-field coordinate,
which is two Goldilocks coordinates. -/
def runningFieldCoordinatesFor (rowVariables : Nat) : Nat :=
  83160 + 2 * rowVariables

/-- Fixed-25 reference value. Production artifacts use
`runningFieldCoordinatesFor` with their selected exponent. -/
def runningFieldCoordinates : Nat := runningFieldCoordinatesFor 25

/-- The mandatory four-component fresh bundle remains as field values. -/
def bundleFieldCoordinates : Nat := 3888

def memoryFieldCoordinates (candidate : Id) : Nat :=
  checkedStepsPerFreshClaim candidate *
    memoryFieldCoordinatesPerCheckedStep

/-- Mixed field-native envelope size. This is a coordinate count, not a
generated row count. Physical column reuse can remove equality rows. -/
def fieldNativeEnvelopeCoordinates (candidate : Id) : Nat :=
  narrowCoordinates candidate + runningFieldCoordinates +
    bundleFieldCoordinates + memoryFieldCoordinates candidate

/-- Exact field-native envelope at the artifact-selected augmented-relation
exponent. -/
def fieldNativeEnvelopeCoordinatesFor
    (candidate : Id) (rowVariables : Nat) : Nat :=
  narrowCoordinates candidate + runningFieldCoordinatesFor rowVariables +
    bundleFieldCoordinates + memoryFieldCoordinates candidate

def bitSerialReferenceEnvelopeBits : Nat := 5587724
def bitSerialReferenceRunningBridgeRows : Nat := 11066930

theorem fixedNarrowCoordinates_exact : fixedNarrowCoordinates = 540 := by decide

theorem fieldNativeEnvelopeCoordinate_table :
    fieldNativeEnvelopeCoordinates .e1 = 87830 /\
      fieldNativeEnvelopeCoordinates .e4 = 88406 /\
      fieldNativeEnvelopeCoordinates .e8 = 89174 /\
      fieldNativeEnvelopeCoordinates .e16 = 90710 := by
  decide

theorem fieldNativeEnvelopeCoordinate_table_at_26 :
    fieldNativeEnvelopeCoordinatesFor .e1 26 = 87832 /\
      fieldNativeEnvelopeCoordinatesFor .e4 26 = 88408 /\
      fieldNativeEnvelopeCoordinatesFor .e8 26 = 89176 /\
      fieldNativeEnvelopeCoordinatesFor .e16 26 = 90712 := by
  decide

theorem fieldNative_envelope_is_smaller_than_reference_bit_image :
    forall candidate,
      fieldNativeEnvelopeCoordinates candidate <
        bitSerialReferenceEnvelopeBits := by
  intro candidate
  cases candidate <;> decide

/-- A generated relation can use zero copy rows only if its manifest proves
that the NIFS input and carried state use the same physical columns. -/
structure AliasContract where
  runningCarrierColumn : Fin runningFieldCoordinates -> Nat
  nifsRunningColumn : Fin runningFieldCoordinates -> Nat
  runningColumnsEqual : forall coordinate,
    nifsRunningColumn coordinate = runningCarrierColumn coordinate
  bundleCarrierColumn : Fin bundleFieldCoordinates -> Nat
  nifsBundleColumn : Fin bundleFieldCoordinates -> Nat
  bundleColumnsEqual : forall coordinate,
    nifsBundleColumn coordinate = bundleCarrierColumn coordinate

/-- Zero-copy alias contract at the generated augmented-relation exponent.
The running width is derived from the same exponent as the NIFS shape. -/
structure AliasContractFor (rowVariables : Nat) where
  runningCarrierColumn : Fin (runningFieldCoordinatesFor rowVariables) -> Nat
  nifsRunningColumn : Fin (runningFieldCoordinatesFor rowVariables) -> Nat
  runningColumnsEqual : forall coordinate,
    nifsRunningColumn coordinate = runningCarrierColumn coordinate
  bundleCarrierColumn : Fin bundleFieldCoordinates -> Nat
  nifsBundleColumn : Fin bundleFieldCoordinates -> Nat
  bundleColumnsEqual : forall coordinate,
    nifsBundleColumn coordinate = bundleCarrierColumn coordinate

/-- Exact data that must come from each compiled candidate before profile and
key selection. Times use integer microseconds and sizes use bytes. -/
structure CompiledMeasurement where
  exactLogicalRows : Nat
  exactAssignmentWidth : Nat
  exactNonzeroEntries : Nat
  recursiveRowVariables : Nat
  terminalLogicalRows : Nat
  terminalRowVariables : Nat
  proofBytes : Nat
  proverPeakBytes : Nat
  gpuPeakBytes : Nat
  proverTimeMicros : Nat
  terminalTimeMicros : Nat

/-- Key freeze requires a compiled measurement for every candidate. The
selection policy and artifact digests remain verifier-key-owned inputs. -/
structure BenchmarkMatrix where
  measurement : Id -> CompiledMeasurement

end Nightstream.Protocol.Nebula.ProductionProfileCandidates
