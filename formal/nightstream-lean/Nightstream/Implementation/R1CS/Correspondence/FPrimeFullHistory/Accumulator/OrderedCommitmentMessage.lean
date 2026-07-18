import Nightstream.Implementation.R1CS.Core.SevenBytePacking
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.Accumulator.CarrierCodec

/-!
Exact field message for the fixed-profile ordered-child accumulator carrier.

Assurance tier: model-level representation refinement.

Owns: the fresh protocol tag, Rust-compatible seven-byte packing, the exact
tag-plus-carrier field order, pre-hash injectivity, and reduction of equal
digests to exact carrier equality or a collision in the supplied field hash.

Does not own: Poseidon2 permutation semantics or collision resistance, an
emitted Rust call, R1CS columns, constant-definition rows, Ajtai opening
binding, cost totals, or row removal.

Emits constraints: no.

Authority boundary: the fixed profile owns commitment width and child count;
the specialized shape owns point length. They are not redundantly serialized.
The unique tag names this exact payload, and the sponge's padded list schedule
binds its total length. A digest has authority only when later correspondence
proves that Poseidon2 was recomputed over `serialize payload`.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.accumulator.ordered_commitments.message.domain` | bind the exact v1 payload type | verifier-owned constant | `domainNats_eq` |
| `fprime.accumulator.ordered_commitments.message.carrier` | retain point then all fourteen commitments | computed | `CarrierCodec.encodeCommitmentFamily` |
| `fprime.accumulator.ordered_commitments.message.injective` | distinct carriers have distinct field messages | derived | `serialize_injective` |
| `fprime.accumulator.ordered_commitments.message.hash` | compress exactly the serialized field list | security boundary | `payloadDigest`, `digest_eq_or_fieldHashCollision` |
| `fprime.accumulator.ordered_commitments.message.scheme` | instantiate the generic claim-hash branch | computed | `claimDigest_eq_payloadDigest` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage

open Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec
open Nightstream.Protocol.FPrime.AccumulatorBinding
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildCommitmentAuthority

/-- Exact ASCII bytes of
`neo.fold.clean/f_prime/accumulator/ordered_child_commitments/v1`.

This tag is intentionally distinct from the legacy full-CE child digest and
the legacy nested child-digest accumulator formats. -/
def domainBytes : List Nat :=
  [110, 101, 111, 46, 102, 111, 108, 100, 46, 99, 108, 101, 97,
   110, 47, 102, 95, 112, 114, 105, 109, 101, 47, 97, 99, 99,
   117, 109, 117, 108, 97, 116, 111, 114, 47, 111, 114, 100,
   101, 114, 101, 100, 95, 99, 104, 105, 108, 100, 95, 99,
   111, 109, 109, 105, 116, 109, 101, 110, 116, 115, 47, 118, 49]

def domainNats : List Nat :=
  Nightstream.Implementation.R1CS.SevenBytePacking.packBytesAsNats domainBytes

/-- Independent closed check of the ASCII bytes and seven-byte packing. -/
theorem domainNats_eq :
    domainNats =
      [63, 30521782141150574, 31069335676202596,
       30796712693949999, 30809842190987109,
       13355146924878965, 28259039672824431,
       26850539575206751, 30808768617738083,
       13922220031372901] := by
  decide

/-- Every packed domain word is already canonical in Goldilocks. -/
theorem domainNats_canonical :
    ∀ value ∈ domainNats, value < goldilocksModulus := by
  rw [domainNats_eq]
  decide

def fieldOfNat (value : Nat) : F :=
  ⟨value % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

def domainFields : List F :=
  domainNats.map fieldOfNat

@[simp] theorem domainFields_length : domainFields.length = 10 := by
  rw [domainFields, domainNats_eq]
  rfl

/-- Exact Poseidon2 field preimage before the sponge schedule is applied. -/
def serialize
    {shape : Shape}
    {verifierRows count : Nat}
    (payload :
      CommitmentFamilyPayload shape (Commitment.Value verifierRows) count) :
    List F :=
  domainFields ++ encodeCommitmentFamily payload

@[simp] theorem serialize_length
    {shape : Shape}
    {verifierRows count : Nat}
    (payload :
      CommitmentFamilyPayload shape (Commitment.Value verifierRows) count) :
    (serialize payload).length =
      10 + commitmentFamilyFieldCount shape verifierRows count := by
  simp [serialize]

/-- The domain prefix adds separation but no ambiguity: removing its fixed ten
fields recovers the already-injective carrier codec exactly. -/
theorem serialize_injective
    {shape : Shape}
    {verifierRows count : Nat} :
    Function.Injective
      (serialize :
        CommitmentFamilyPayload shape (Commitment.Value verifierRows) count ->
          List F) := by
  intro left right same
  apply encodeCommitmentFamily_injective
  have tails := congrArg (List.drop domainFields.length) same
  simpa [serialize] using tails

universe uDigest

/-- Direct compression of the exact field message. Production instantiates
`hashFields` with the fixed Goldilocks Poseidon2 sponge. -/
def payloadDigest
    {shape : Shape}
    {verifierRows count : Nat}
    {Digest : Type uDigest}
    (hashFields : List F -> Digest)
    (payload :
      CommitmentFamilyPayload shape (Commitment.Value verifierRows) count) :
    Digest :=
  hashFields (serialize payload)

/-- Collision of the supplied field hash on two distinct field lists. This is
the only remaining compression failure after `serialize_injective`. -/
def FieldHashCollision
    {Digest : Type uDigest}
    (hashFields : List F -> Digest) : Prop :=
  ∃ left right : List F,
    left ≠ right ∧ hashFields left = hashFields right

theorem digest_eq_or_fieldHashCollision
    {shape : Shape}
    {verifierRows count : Nat}
    {Digest : Type uDigest}
    (hashFields : List F -> Digest)
    (left right :
      CommitmentFamilyPayload shape (Commitment.Value verifierRows) count)
    (sameDigest :
      payloadDigest hashFields left = payloadDigest hashFields right) :
    left = right ∨ FieldHashCollision hashFields := by
  classical
  by_cases samePayload : left = right
  · exact Or.inl samePayload
  · apply Or.inr
    exact ⟨serialize left, serialize right,
      fun same => samePayload (serialize_injective same), sameDigest⟩

/-- Generic accumulator schemes contain a second, nested-accumulator branch.
That branch is not used by the direct ordered-commitment handle, so its hash
remains an explicit separate input instead of being silently assigned a legacy
serialization. -/
def claimScheme
    {shape : Shape}
    {verifierRows count : Nat}
    {Digest : Type uDigest}
    (hashFields : List F -> Digest)
    (hashAccumulator : AccumulatorPreimage Digest -> Digest) :
    Scheme
      (CommitmentFamilyPayload shape (Commitment.Value verifierRows) count)
      (List F) Digest where
  encodeClaim := encodeCommitmentFamily
  hash
    | .claim fields => hashFields (domainFields ++ fields)
    | .accumulator preimage => hashAccumulator preimage

theorem claimDigest_eq_payloadDigest
    {shape : Shape}
    {verifierRows count : Nat}
    {Digest : Type uDigest}
    (hashFields : List F -> Digest)
    (hashAccumulator : AccumulatorPreimage Digest -> Digest)
    (payload :
      CommitmentFamilyPayload shape (Commitment.Value verifierRows) count) :
    claimDigest (claimScheme hashFields hashAccumulator) payload =
      payloadDigest hashFields payload := by
  rfl

/-! ## Fixed Phi81 profile -/

def fixedPreimageFieldCount (rowVariables : Nat) : Nat :=
  10 + (2 * rowVariables + 13608)

theorem fixed_serialize_length
    {shape : Shape}
    (payload : FixedCommitmentFamilyPayload shape) :
    (serialize payload).length = fixedPreimageFieldCount shape.rowVariables := by
  rw [serialize_length, fixed_commitment_family_field_count]
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.OrderedCommitmentMessage
