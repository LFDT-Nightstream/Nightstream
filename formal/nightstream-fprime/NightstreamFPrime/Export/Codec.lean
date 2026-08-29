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

/-- Linear accumulator renderer for executable emission. It writes the same
numeric-array syntax as `Value.render` without retaining one rendered string
for every sibling value. -/
def Value.renderInto : Value → ByteArray → ByteArray
  | .atom value, output => output ++ (toString value).toUTF8
  | .array values, output =>
      let renderers := values.map Value.renderInto
      let output := output.push 91
      let output := match renderers with
        | [] => output
        | first :: rest =>
            rest.foldl (fun current (renderChild : ByteArray → ByteArray) =>
              renderChild (current.push 44)) (first output)
      output.push 93

def Value.renderBytes (value : Value) : ByteArray :=
  value.renderInto ByteArray.empty

private def leftBracketBytes : ByteArray := ByteArray.empty.push 91
private def rightBracketBytes : ByteArray := ByteArray.empty.push 93
private def commaBytes : ByteArray := ByteArray.empty.push 44
private def newlineBytes : ByteArray := ByteArray.empty.push 10

/-- Write common punctuation without allocating a new singleton byte array. -/
def writeByte (handle : IO.FS.Handle) (byte : UInt8) : IO Unit :=
  handle.write <| match byte with
    | 91 => leftBracketBytes
    | 93 => rightBracketBytes
    | 44 => commaBytes
    | 10 => newlineBytes
    | _ => ByteArray.empty.push byte

/-- Write the canonical numeric-array encoding directly to a file handle.
The traversal uses the same atom, bracket, comma, and child order as
`renderInto`, but it does not retain the complete artifact in memory. -/
partial def Value.writeCanonical (handle : IO.FS.Handle) : Value → IO Nat
  | .atom value => do
      let text := toString value
      handle.putStr text
      pure text.length
  | .array values => do
      writeByte handle 91
      let rec writeValues : List Value → IO Nat
        | [] => pure 0
        | [value] => value.writeCanonical handle
        | value :: rest => do
            let first ← value.writeCanonical handle
            writeByte handle 44
            let tail ← writeValues rest
            pure (first + 1 + tail)
      let children ← writeValues values
      writeByte handle 93
      pure (children + 2)

/-- Write a canonical array with one item action at a time. -/
partial def writeListWith (handle : IO.FS.Handle)
    (writeItem : Alpha → IO Unit) : List Alpha → IO Unit
  | values => do
      writeByte handle 91
      let rec writeValues : List Alpha → IO Unit
        | [] => pure ()
        | value :: rest => do
            writeItem value
            match rest with
            | [] => pure ()
            | _ =>
              writeByte handle 44
              writeValues rest
      writeValues values
      writeByte handle 93

/-- Write a canonical codec list without first constructing its complete
`Value.array`. Only the current encoded item is retained. -/
def writeListCanonical (handle : IO.FS.Handle)
    (item : Format Alpha) (values : List Alpha) : IO Unit :=
  writeListWith handle
    (fun value => do
      let _ ← (item.encode value).writeCanonical handle
      pure ())
    values

/-- Append items to an array that a caller opened. `first` records whether
the array is still empty, so callers can stream several logical list segments
without constructing their concatenation. -/
partial def writeArrayItemsWith (handle : IO.FS.Handle)
    (writeItem : Alpha → IO Unit) : Bool → List Alpha → IO Bool
  | first, [] => pure first
  | first, value :: rest => do
      if !first then
        writeByte handle 44
      writeItem value
      writeArrayItemsWith handle writeItem false rest

end NightstreamFPrime.Export.Codec
