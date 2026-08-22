import Mathlib.Data.List.Defs

/-!
Owns the small canonical value codec used by the circuit-package serializer.
The wire value has only natural-number atoms and arrays. Package modules own
all schema tags and validation; this module owns only lossless composition.
-/

namespace NightstreamFPrime.Export.Codec

/-- Canonical structured value before deterministic text rendering. -/
inductive Value where
  | atom (value : Nat)
  | array (values : List Value)
deriving Repr

/-- A codec carries its round-trip theorem with its implementation. -/
structure Format (Alpha : Type) where
  encode : Alpha → Value
  decode : Value → Except String Alpha
  decode_encode : ∀ value, decode (encode value) = .ok value

variable {Alpha Beta : Type}

def nat : Format Nat where
  encode := Value.atom
  decode
    | .atom value => .ok value
    | .array _ => .error "expected a natural-number atom"
  decode_encode := by
    intro value
    rfl

def pair (left : Format Alpha) (right : Format Beta) : Format (Alpha × Beta) where
  encode := fun value => .array [left.encode value.1, right.encode value.2]
  decode
    | .array [leftValue, rightValue] => do
        pure (← left.decode leftValue, ← right.decode rightValue)
    | _ => .error "expected a two-element array"
  decode_encode := by
    rintro ⟨leftValue, rightValue⟩
    simp only
    rw [left.decode_encode, right.decode_encode]
    rfl

def option (item : Format Alpha) : Format (Option Alpha) where
  encode
    | none => .array [.atom 0]
    | some value => .array [.atom 1, item.encode value]
  decode
    | .array [.atom 0] => .ok none
    | .array [.atom 1, value] => return some (← item.decode value)
    | _ => .error "expected a canonical option array"
  decode_encode := by
    intro value
    cases value with
    | none => rfl
    | some value =>
        simp only
        rw [item.decode_encode]
        rfl

def decodeList (item : Format Alpha) : List Value → Except String (List Alpha)
  | [] => .ok []
  | value :: rest => return (← item.decode value) :: (← decodeList item rest)

theorem decodeList_map_encode (item : Format Alpha) (values : List Alpha) :
    decodeList item (values.map item.encode) = .ok values := by
  induction values with
  | nil => rfl
  | cons value rest ih =>
      simp only [List.map_cons, decodeList]
      rw [item.decode_encode, ih]
      rfl

def list (item : Format Alpha) : Format (List Alpha) where
  encode := fun values => .array (values.map item.encode)
  decode
    | .array values => decodeList item values
    | .atom _ => .error "expected an array"
  decode_encode := decodeList_map_encode item

/-- Deterministic JSON-compatible rendering. The Rust decoder accepts only
this numeric-array subset, so no escaping or object-key ordering is involved. -/
def Value.render : Value → String
  | .atom value => toString value
  | .array values =>
      "[" ++ String.intercalate "," (values.map Value.render) ++ "]"

end NightstreamFPrime.Export.Codec
