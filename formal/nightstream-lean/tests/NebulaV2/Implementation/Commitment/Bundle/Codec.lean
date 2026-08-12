import Nightstream.Implementation.NebulaV2.Commitment.Bundle.Codec

set_option autoImplicit false

namespace tests.NebulaV2CommitmentBundleCodec

open Nightstream.Implementation.NebulaV2.CommitmentBundleCodec
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CommitmentBundle

def zero : ShiftedTernary41V1.CanonicalGoldilocks :=
  ⟨0, by norm_num [ShiftedTernary41V1.modulus]⟩

def one : ShiftedTernary41V1.CanonicalGoldilocks :=
  ⟨1, by norm_num [ShiftedTernary41V1.modulus]⟩

def firstCoordinate : Coordinate := ⟨0, by decide⟩

def zeroValue : Value := fun _component _coordinate => zero

def changedOperations : Value := fun component coordinate =>
  if component = .operations ∧ coordinate = firstCoordinate then one
  else zero

theorem bundle_encoding_has_exact_width :
    (encode zeroValue).length = 248832 :=
  encode_exact_length zeroValue

theorem bundle_component_order_is_normative :
    componentOrder =
      [.full, .operations, .initialSnapshot, .finalSnapshot] :=
  rfl

theorem changed_component_changes_bundle : changedOperations ≠ zeroValue := by
  intro equal
  have atChanged := congrFun (congrFun equal .operations) firstCoordinate
  have valuesEqual := congrArg Subtype.val atChanged
  change 1 = 0 at valuesEqual
  omega

/-- Changing only the operations component cannot preserve the mandatory
bundle bit string. -/
theorem changed_component_changes_encoding :
    encode changedOperations ≠ encode zeroValue := by
  intro equalEncoding
  exact changed_component_changes_bundle (encode_injective equalEncoding)

end tests.NebulaV2CommitmentBundleCodec
