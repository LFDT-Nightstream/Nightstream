import NightstreamFPrime.Spec.Compression.Akita.Replay.Keccak
import NightstreamFPrime.Spec.Compression.Akita.Replay.Field

/-! The flat setup vector of the pinned Akita code from its seed (Akita revision `d50f700`,
`akita-types/src/proof/setup.rs`, `derive_public_matrix_prefix`, `setup/range.rs`, `fill_page`;
`jolt-algebra` `sample_uniform_below`), and the field digest of the terminal
(`akita-transcript/src/proof_stream/digest.rs`, `field_digest`).

These are DEFINITIONS of the concrete instance: the model of Akita has a uniform setup vector and
a random oracle for the digest. The evidence is the digest of the last block of `t` of the
terminal of each vector, which reads `A` (`tests/AkitaTerminalVectors.lean`).

- The setup is in pages of 4096 elements. Page `j` is one SHAKE256 stream after the header
  `pageHeader seed j`: six fields of a length-prefixed label and value (`absorb_len_prefixed`),
  each length a `u64` little endian. An element is the next 8 bytes as a little-endian word,
  and a word `q` or more is rejected. `page` takes the first `count` kept words of the first
  `count + 64` words; it is `none` when more than 64 of them are rejected (the code reads on).
- `expandTasks` is `expand` with the pages in parallel tasks (`expandTasks_eq`).
- `fieldDigest digest` is the hash `digest` (SHA3-256 in the code, `Public.digest`) of the 32-byte
  tag `akita-pcs/field-digest/v8` and each value as 8 bytes little endian. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

namespace Setup

/-- SHAKE256 (FIPS 202 Section 6.2) with `count` bytes of output. -/
def shake256 (input : ByteArray) (count : ℕ) : ByteArray := Keccak.hash 136 0x1F input count

/-- `value` as `count` bytes, little endian. -/
def littleEndian (count value : ℕ) : ByteArray :=
  ByteArray.mk ((Array.range count).map fun i => (value / 2 ^ (8 * i) % 256).toUInt8)

/-- `absorb_len_prefixed`. -/
def field (label data : ByteArray) : ByteArray :=
  littleEndian 8 label.size ++ label ++ littleEndian 8 data.size ++ data

/-- The elements of one page (`PublicMatrixDerivation::Shake256PagedV1`). -/
def pageSize : ℕ := 4096

/-- The header of the stream of page `page` (`SetupSeedPageXof::new`); the modulus is 32 bytes
big endian (`field_modulus_be_bytes`). -/
def pageHeader (seed : ByteArray) (page : ℕ) : ByteArray :=
  field "domain".toUTF8 "akita/commitment/public-field-stream".toUTF8 ++
    field "derivation".toUTF8 "shake256-paged-v1".toUTF8 ++
    field "page_field_elements".toUTF8 (littleEndian 8 pageSize) ++
    field "seed".toUTF8 seed ++
    field "field".toUTF8 (ByteArray.mk ((littleEndian 32 fieldModulus).data.reverse)) ++
    field "page".toUTF8 (littleEndian 8 page)

/-- The word of the 8 bytes from `at`, little endian. -/
def word (bytes : ByteArray) (at_ : ℕ) : UInt64 :=
  (List.range 8).foldr (fun i acc => (acc <<< 8) ||| (bytes.get! (at_ + i)).toUInt64) 0

/-- The first `count` elements of page `page`. -/
def page (seed : ByteArray) (page count : ℕ) : Option (Array Fq) := Id.run do
  let bytes := shake256 (pageHeader seed page) (8 * (count + 64))
  let mut kept : Array Fq := Array.emptyWithCapacity count
  for i in [0:count + 64] do
    if kept.size = count then break
    if let some a := Fq.ofWord? (word bytes (8 * i)) then kept := kept.push a
  return if kept.size = count then some kept else none

/-- The first `count` elements of the flat setup vector (`derive_public_matrix_prefix`). -/
def expand (seed : ByteArray) (count : ℕ) : Option (Array Fq) :=
  (List.range ((count + pageSize - 1) / pageSize)).foldlM (fun out j => do
    let elements ← page seed j (min pageSize (count - j * pageSize))
    return out ++ elements) (Array.emptyWithCapacity count)

/-- `expand` with one task for each page. -/
def expandTasks (seed : ByteArray) (count : ℕ) : Option (Array Fq) :=
  ((List.range ((count + pageSize - 1) / pageSize)).map fun j =>
      Task.spawn fun _ => page seed j (min pageSize (count - j * pageSize))).foldlM
    (fun out t => do return out ++ (← t.get)) (Array.emptyWithCapacity count)

theorem expandTasks_eq (seed : ByteArray) (count : ℕ) :
    expandTasks seed count = expand seed count := by
  unfold expandTasks expand
  rw [List.foldlM_map]
  rfl

/-- `sample_akita_setup_seed`: the word `0xDEADBEEFCAFEBABE` little endian, then 24 zero bytes. -/
def sampleSeed : ByteArray := littleEndian 8 0xDEADBEEFCAFEBABE ++ littleEndian 24 0

end Setup

/-- `field_digest` with the hash `digest`; the code has SHA3-256. -/
def fieldDigest (digest : ByteArray → ByteArray) (values : Array Fq) : ByteArray :=
  digest (values.foldl (fun out a => out ++ Setup.littleEndian 8 a.word.toNat)
    ("akita-pcs/field-digest/v8".toUTF8 ++ ByteArray.mk (Array.replicate 7 0)))

end NightstreamFPrime.Spec.Compression.Akita.Replay
