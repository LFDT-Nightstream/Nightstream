import Mathlib.Logic.Equiv.Fin.Basic
import Nightstream.Protocol.Nebula.Encoding
import Nightstream.Protocol.Nebula.Profile

/-!
Contract: exact structural definition and collision reduction for the
two-stage `CompactCommitV2` lane token.

Assurance tier: model-level and Module-SIS reduction boundary.

Owns the fixed V2 ranks and field counts, the exact shifted-ternary encoder,
independent operations and memory primary roles, a role-separated short map,
and the deterministic reduction of a token collision to a primary or short
Ajtai binding failure.

Does not own concrete Ajtai matrix arithmetic, ChaCha8 setup refinement,
Module-SIS hardness, Rust or R1CS conformance, token serialization, or
Poseidon2 leaf hashing.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.CompactCommit

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ShiftedTernary41V1

def ringDegree : Nat := 54
def commitmentRank : Nat := 18
def primaryRank : Nat := 2
def shortRank : Nat := 1

def commitmentFieldCount : Nat := commitmentRank * ringDegree
def primaryOutputFieldCount : Nat := primaryRank * ringDegree
def tokenFieldCount : Nat := shortRank * ringDegree
def primaryMessageRingColumns : Nat :=
  commitmentFieldCount * digitCount / ringDegree
def shortMessageRingColumns : Nat :=
  primaryOutputFieldCount * digitCount / ringDegree

theorem exact_dimensions :
    commitmentFieldCount = 972 ∧
      primaryOutputFieldCount = 108 ∧
      tokenFieldCount = 54 ∧
      primaryMessageRingColumns = 738 ∧
      shortMessageRingColumns = 82 := by
  decide

abbrev FieldVector (count : Nat) := Fin count → CanonicalGoldilocks

/-- Each field becomes the exact fixed-width little-endian list of canonical
field coordinates for the centered digits `(-1, 0, 1)`. -/
abbrev CenteredMessage (fieldCount : Nat) := Fin fieldCount → List Nat

def centeredWord (value : CanonicalGoldilocks) : List Nat :=
  (trits value).map fieldDigit

private theorem fieldDigit_injective_below_three
    {left right : Nat} (leftBound : left < 3) (rightBound : right < 3)
    (equal : fieldDigit left = fieldDigit right) :
    left = right := by
  interval_cases left <;> interval_cases right <;>
    simp_all [fieldDigit, modulus]

private theorem map_fieldDigit_injective_bounded :
    ∀ {left right : List Nat},
      (∀ value ∈ left, value < 3) →
      (∀ value ∈ right, value < 3) →
      left.map fieldDigit = right.map fieldDigit →
      left = right
  | [], [], _, _, _ => rfl
  | [], _ :: _, _, _, equal => by simp at equal
  | _ :: _, [], _, _, equal => by simp at equal
  | leftHead :: leftTail, rightHead :: rightTail,
      leftBound, rightBound, equal => by
      simp only [List.map_cons, List.cons.injEq] at equal
      have headEqual : leftHead = rightHead :=
        fieldDigit_injective_below_three
          (leftBound leftHead (by simp))
          (rightBound rightHead (by simp)) equal.1
      subst rightHead
      have tailEqual : leftTail = rightTail :=
        map_fieldDigit_injective_bounded
          (fun value member => leftBound value (by simp [member]))
          (fun value member => rightBound value (by simp [member]))
          equal.2
      rw [tailEqual]

def encodeFields {fieldCount : Nat} :
    FieldVector fieldCount → CenteredMessage fieldCount :=
  fun fields index => centeredWord (fields index)

theorem encodeFields_word_length
    {fieldCount : Nat} (fields : FieldVector fieldCount)
    (index : Fin fieldCount) :
    (encodeFields fields index).length = digitCount :=
  by simp [encodeFields, centeredWord, trits_length]

theorem encodeFields_word_canonical
    {fieldCount : Nat} (fields : FieldVector fieldCount)
    (index : Fin fieldCount) :
    ∀ coordinate ∈ encodeFields fields index,
      coordinate = modulus - 1 ∨ coordinate = 0 ∨ coordinate = 1 := by
  intro coordinate member
  rcases List.mem_map.mp member with ⟨trit, tritMember, rfl⟩
  have bound := trits_bounded (fields index) trit tritMember
  interval_cases trit <;> simp [fieldDigit]

theorem encodeFields_injective {fieldCount : Nat} :
    Function.Injective
      (encodeFields : FieldVector fieldCount → CenteredMessage fieldCount) := by
  intro left right equal
  funext index
  apply trits_injective
  exact map_fieldDigit_injective_bounded
    (trits_bounded (left index)) (trits_bounded (right index))
    (congrFun equal index)

/-! ## Exact ring-column packing -/

def tritAt (value : CanonicalGoldilocks) (index : Fin digitCount) : Nat :=
  (trits value).get
    ⟨index.val, by rw [trits_length]; exact index.isLt⟩

theorem tritAt_lt_three
    (value : CanonicalGoldilocks) (index : Fin digitCount) :
    tritAt value index < 3 := by
  apply trits_bounded value
  exact List.get_mem _ _

/-- Signed coefficient used by the Module-SIS witness. This is an integer in
`{-1,0,1}`, not the base-field representative `q-1` of minus one. -/
def signedDigit
    (value : CanonicalGoldilocks) (index : Fin digitCount) : Int :=
  (tritAt value index : Int) - 1

def signedDigits (value : CanonicalGoldilocks) : Fin digitCount → Int :=
  signedDigit value

theorem signedDigits_unit_bound
    (value : CanonicalGoldilocks) (index : Fin digitCount) :
    (signedDigits value index).natAbs ≤ 1 := by
  have bound := tritAt_lt_three value index
  have alternatives :
      tritAt value index = 0 ∨ tritAt value index = 1 ∨
        tritAt value index = 2 := by
    omega
  rcases alternatives with equal | equal | equal <;>
    simp [signedDigits, signedDigit, equal]

theorem signedDigits_injective : Function.Injective signedDigits := by
  intro left right equal
  apply trits_injective
  apply List.ext_get
  · rw [trits_length, trits_length]
  · intro index leftBound rightBound
    let digit : Fin digitCount :=
      ⟨index, by simpa [trits_length] using leftBound⟩
    have atDigit := congrFun equal digit
    have tritEqual : tritAt left digit = tritAt right digit := by
      unfold signedDigits signedDigit at atDigit
      omega
    simpa [tritAt, digit] using tritEqual

/-- Exact equality between field digits and degree-54 ring coefficients. -/
structure Packing (fieldCount ringColumns : Nat) where
  coefficientCountExact :
    ringColumns * ringDegree = fieldCount * digitCount

namespace Packing

def indexEquiv
    {fieldCount ringColumns : Nat}
    (packing : Packing fieldCount ringColumns) :
    (Fin ringColumns × Fin ringDegree) ≃
      (Fin fieldCount × Fin digitCount) :=
  (finProdFinEquiv (m := ringColumns) (n := ringDegree)).trans
    ((finCongr packing.coefficientCountExact).trans
      (finProdFinEquiv (m := fieldCount) (n := digitCount)).symm)

end Packing

abbrev RingMessage (ringColumns : Nat) :=
  Fin ringColumns → Fin ringDegree → Int

/-- Canonical row-major packing of field words into complete ring columns. -/
def packFields
    {fieldCount ringColumns : Nat}
    (packing : Packing fieldCount ringColumns)
    (fields : FieldVector fieldCount) : RingMessage ringColumns :=
  fun column coefficient =>
    let source := packing.indexEquiv (column, coefficient)
    signedDigit (fields source.1) source.2

theorem packFields_injective
    {fieldCount ringColumns : Nat}
    (packing : Packing fieldCount ringColumns) :
    Function.Injective (packFields packing) := by
  intro left right equal
  funext field
  apply signedDigits_injective
  funext digit
  let target := packing.indexEquiv.symm (field, digit)
  have atTarget := congrFun (congrFun equal target.1) target.2
  simpa [packFields, target, signedDigits] using atTarget

theorem packFields_unit_bound
    {fieldCount ringColumns : Nat}
    (packing : Packing fieldCount ringColumns)
    (fields : FieldVector fieldCount)
    (column : Fin ringColumns) (coefficient : Fin ringDegree) :
    (packFields packing fields column coefficient).natAbs ≤ 1 := by
  exact signedDigits_unit_bound _ _

def primaryPacking :
    Packing commitmentFieldCount primaryMessageRingColumns where
  coefficientCountExact := by decide

def shortPacking :
    Packing primaryOutputFieldCount shortMessageRingColumns where
  coefficientCountExact := by decide

/-- The operations and memory lanes use different primary seeded maps. The
initial and final snapshots both use `memory`. -/
inductive Role where
  | operations
  | memory
deriving DecidableEq, Repr

theorem roles_distinct : Role.operations ≠ Role.memory := by
  decide

abbrev CommitmentEncoding := FieldVector commitmentFieldCount
abbrev PrimaryOutput := FieldVector primaryOutputFieldCount
abbrev Token := FieldVector tokenFieldCount

/-- Verifier-key-owned token maps. Both stages carry explicit, independent
role seeds. This makes matrix separation data flow, not only a label. -/
structure Key (Plan Seed : Type) where
  profile : Profile.Identity
  plan : Plan
  primarySeed : Role → Seed
  primarySeedIndependent :
    primarySeed .operations ≠ primarySeed .memory
  shortSeed : Role → Seed
  shortSeedIndependent :
    shortSeed .operations ≠ shortSeed .memory
  primaryFromSeed :
    Seed → RingMessage primaryMessageRingColumns → PrimaryOutput
  shortFromSeed :
    Seed → RingMessage shortMessageRingColumns → Token

namespace Key

def primary
    {Plan Seed : Type} (key : Key Plan Seed) (role : Role) :
    RingMessage primaryMessageRingColumns → PrimaryOutput :=
  key.primaryFromSeed (key.primarySeed role)

def short
    {Plan Seed : Type} (key : Key Plan Seed) (role : Role) :
    RingMessage shortMessageRingColumns → Token :=
  key.shortFromSeed (key.shortSeed role)

/-- The exact two-stage V2 token. Both stages consume
`ShiftedTernary41V1` words. -/
def token
    {Plan Seed : Type} (key : Key Plan Seed)
    (role : Role) (commitment : CommitmentEncoding) : Token :=
  key.short role
    (packFields shortPacking
      (key.primary role (packFields primaryPacking commitment)))

end Key

/-- Two different canonical commitment encodings collide in the selected
rank-two primary map. This is the first named Module-SIS/Ajtai event. -/
def PrimaryBindingFailure
    {Plan Seed : Type} (key : Key Plan Seed) (role : Role) : Prop :=
  ∃ left right : CommitmentEncoding,
    left ≠ right ∧
      key.primary role (packFields primaryPacking left) =
        key.primary role (packFields primaryPacking right)

/-- Two different rank-two outputs collide after exact re-encoding in the
role-separated rank-one short map. This is the second named
Module-SIS/Ajtai event. -/
def ShortBindingFailure
    {Plan Seed : Type} (key : Key Plan Seed) (role : Role) : Prop :=
  ∃ left right : PrimaryOutput,
    left ≠ right ∧
      key.short role (packFields shortPacking left) =
        key.short role (packFields shortPacking right)

/-- A compact token collision is not accepted as a hash collision. It reduces
deterministically to exactly one of the two Ajtai binding events. -/
theorem token_collision_implies_primary_or_short_failure
    {Plan Seed : Type} (key : Key Plan Seed) (role : Role)
    {left right : CommitmentEncoding}
    (different : left ≠ right)
    (collision : key.token role left = key.token role right) :
    PrimaryBindingFailure key role ∨ ShortBindingFailure key role := by
  by_cases primaryEqual :
      key.primary role (packFields primaryPacking left) =
        key.primary role (packFields primaryPacking right)
  · exact Or.inl ⟨left, right, different, primaryEqual⟩
  · exact Or.inr
      ⟨key.primary role (packFields primaryPacking left),
        key.primary role (packFields primaryPacking right),
        primaryEqual, collision⟩

theorem token_injective_of_no_binding_failure
    {Plan Seed : Type} (key : Key Plan Seed) (role : Role)
    (primarySecure : ¬ PrimaryBindingFailure key role)
    (shortSecure : ¬ ShortBindingFailure key role) :
    Function.Injective (key.token role) := by
  intro left right equalToken
  by_contra different
  rcases token_collision_implies_primary_or_short_failure
      key role different equalToken with primaryFailure | shortFailure
  · exact primarySecure primaryFailure
  · exact shortSecure shortFailure

/-- A primary map that is constant on two different canonical commitments is
an explicit binding failure. This prevents a zero map from satisfying the
security interface merely because it is linear. -/
theorem constant_primary_pair_is_failure
    {Plan Seed : Type} (key : Key Plan Seed) (role : Role)
    {left right : CommitmentEncoding}
    (different : left ≠ right)
    (equalPrimary :
      key.primary role (packFields primaryPacking left) =
        key.primary role (packFields primaryPacking right)) :
    PrimaryBindingFailure key role :=
  ⟨left, right, different, equalPrimary⟩

end Nightstream.Protocol.Nebula.CompactCommit
