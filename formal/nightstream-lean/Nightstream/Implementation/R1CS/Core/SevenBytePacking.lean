/-!
Canonical seven-byte packing shared by field-valued protocol messages.

Owns: byte length followed by zero-padded seven-byte little-endian limbs.
Does not own: a field conversion, protocol tag, hash, or emitted constraint.
Emits constraints: no.

| Operation | Mathematical obligation |
|---|---|
| `packSevenAt` | Interpret at most seven bytes as one little-endian natural number |
| `packBytesAsNats` | Prefix the byte length, then pack every seven-byte chunk in order |
-/

namespace Nightstream.Implementation.R1CS.SevenBytePacking

/-- One zero-padded seven-byte little-endian limb. Seven bytes always fit
canonically in the Goldilocks field. -/
def packSevenAt (bytes : List Nat) (start : Nat) : Nat :=
  bytes.getD start 0 +
    256 * bytes.getD (start + 1) 0 +
    256 ^ 2 * bytes.getD (start + 2) 0 +
    256 ^ 3 * bytes.getD (start + 3) 0 +
    256 ^ 4 * bytes.getD (start + 4) 0 +
    256 ^ 5 * bytes.getD (start + 5) 0 +
    256 ^ 6 * bytes.getD (start + 6) 0

/-- Rust's `pack_bytes_as_fields` before conversion into Goldilocks: the byte
length followed by seven-byte little-endian limbs. -/
def packBytesAsNats (bytes : List Nat) : List Nat :=
  bytes.length ::
    (List.range ((bytes.length + 6) / 7)).map
      (fun chunk => packSevenAt bytes (7 * chunk))

@[simp] theorem packBytesAsNats_length (bytes : List Nat) :
    (packBytesAsNats bytes).length = 1 + (bytes.length + 6) / 7 := by
  simp [packBytesAsNats, Nat.add_comm]

end Nightstream.Implementation.R1CS.SevenBytePacking
