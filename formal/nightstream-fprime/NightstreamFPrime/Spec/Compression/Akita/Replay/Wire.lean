import NightstreamFPrime.Spec.Compression.Akita.Replay.Script

/-! The wire grammar of an Akita opening proof of the pinned code (Akita revision `d50f700`):
the prover messages of a script in the bytes of the proof.

- A field element is 8 bytes, little endian, below `q` (`FieldAtom::from_narg`). An element of
  `E` is `c0` then `c1` (`ExtensionAtom`). `E_resp` is a `u128` of 16 bytes. A digest is 32
  bytes. The payload of `z` is a `u32` length of 4 bytes, at most `zMax`, and that many bytes
  (`receive_bounded_bytes`).
- A proof-of-work nonce of a site with `g > 0` bits is LEB128 of at most `⌈(g + 7)/7⌉` bytes
  (`SpongePow::read_nonce`); the response nonce of a fold is LEB128 of at most 5 bytes with at
  most 32 bits (`NonceAtom::from_narg`). `decode` reads the bytes of a nonce; whether its
  encoding is canonical and its value is in range is a check of the work (`Data.nonce`).
- `decode` reads the messages of the script in order. It fails (`Malformed`) when the proof ends
  inside a message, at an element that is not below `q`, at a nonce longer than its maximum and
  at a length of `z` above `zMax`. Otherwise it returns the messages and the count of bytes
  after the last message.
- `verifyBytes` is the outcome of the byte layer: `malformed`, `trailing` when bytes remain, or
  the verdict of the later stages on the messages. `verifyBytes_malformed` and
  `verifyBytes_trailing` are the two statements of `docs/akita-work/match-design.md` section 5.1
  for this layer. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

inductive Data
  | fields (values : Array Fq)
  | exts (values : Array Ext)
  | integer (value : ℕ)
  | bytes (value : ByteArray)
  | nonce (value : ℕ) (canonical : Bool)

structure Message where
  offset : ℕ
  size : ℕ
  data : Data

structure Malformed where
  op : ℕ
  offset : ℕ
deriving DecidableEq, Repr

namespace Wire

def word (proof : ByteArray) (at_ : ℕ) : UInt64 :=
  (List.range 8).foldr (fun i acc => (acc <<< 8) ||| (proof.get! (at_ + i)).toUInt64) 0

def number (proof : ByteArray) (at_ n : ℕ) : ℕ :=
  (List.range n).foldr (fun i acc => acc * 256 + (proof.get! (at_ + i)).toNat) 0

def fields (proof : ByteArray) (at_ count : ℕ) : Option (Array Fq) :=
  if proof.size < at_ + 8 * count then none
  else (List.range count).foldlM (fun values i =>
    (Fq.ofWord? (word proof (at_ + 8 * i))).map values.push) (Array.emptyWithCapacity count)

def exts (proof : ByteArray) (at_ count : ℕ) : Option (Array Ext) := do
  let values ← fields proof at_ (2 * count)
  return (Array.range count).map fun i => ⟨values[2 * i]!, values[2 * i + 1]!⟩

/-- A LEB128 nonce of at most `most` bytes at `at`: its value, its length and whether it is
canonical (no zero last byte after the first). -/
def leb (proof : ByteArray) (at_ most : ℕ) : Option (ℕ × ℕ × Bool) :=
  go 0 0
where
  go (i value : ℕ) : Option (ℕ × ℕ × Bool) :=
    if i < most then
      if at_ + i < proof.size then
        let byte := (proof.get! (at_ + i)).toNat
        let value := value + byte % 128 * 2 ^ (7 * i)
        if byte < 128 then some (value, i + 1, i == 0 || byte != 0) else go (i + 1) value
      else none
    else none
  termination_by most - i

def read (proof : ByteArray) (at_ : ℕ) : Op → Option (Option Message)
  | .fields _ count => (fields proof at_ count).map fun v => some ⟨at_, 8 * count, .fields v⟩
  | .exts _ count => (exts proof at_ count).map fun v => some ⟨at_, 16 * count, .exts v⟩
  | .integer _ =>
      if proof.size < at_ + 16 then none
      else some (some ⟨at_, 16, .integer (number proof at_ 16)⟩)
  | .digest _ =>
      if proof.size < at_ + 32 then none
      else some (some ⟨at_, 32, .bytes (proof.extract at_ (at_ + 32))⟩)
  | .bounded _ most =>
      if proof.size < at_ + 4 then none else
        let length := number proof at_ 4
        if most < length ∨ proof.size < at_ + 4 + length then none
        else some (some ⟨at_, 4 + length, .bytes (proof.extract (at_ + 4) (at_ + 4 + length))⟩)
  | .work _ _ bits =>
      if bits = 0 then some none
      else (leb proof at_ ((bits + 13) / 7)).map fun (value, size, canonical) =>
        some ⟨at_, size, .nonce value canonical⟩
  | .foldNonce _ _ =>
      (leb proof at_ 5).bind fun (value, size, canonical) =>
        if value < 2 ^ 32 then some (some ⟨at_, size, .nonce value canonical⟩) else none
  | _ => some none

end Wire

def decode (ops : Array Op) (proof : ByteArray) : Except Malformed (Array Message × ℕ) := do
  let (messages, cursor) ← ops.zipIdx.foldlM (fun (messages, cursor) (op, i) =>
    match Wire.read proof cursor op with
    | none => .error ⟨i, cursor⟩
    | some none => .ok (messages, cursor)
    | some (some m) => .ok (messages.push m, cursor + m.size)) (#[], 0)
  return (messages, proof.size - cursor)

/-- The verdict on decoded messages; `noWork` names the operation of the script and the site of
the work that fails. -/
inductive Verdict
  | accept
  | noWork (op : ℕ) (site : String)
  | reject (check : String)
deriving DecidableEq, Repr

inductive Outcome
  | malformed (site : Malformed)
  | trailing
  | verdict (verdict : Verdict)
deriving DecidableEq, Repr

def verifyBytes (after : Array Message → Verdict) (ops : Array Op) (proof : ByteArray) :
    Outcome :=
  match decode ops proof with
  | .error site => .malformed site
  | .ok (messages, rest) => if rest = 0 then .verdict (after messages) else .trailing

theorem verifyBytes_malformed (after : Array Message → Verdict) (ops : Array Op)
    (proof : ByteArray) (site : Malformed) :
    verifyBytes after ops proof = .malformed site ↔ decode ops proof = .error site := by
  unfold verifyBytes
  rcases decode ops proof with error | ⟨messages, rest⟩
  · simp
  · by_cases empty : rest = 0 <;> simp [empty]

theorem verifyBytes_trailing (after : Array Message → Verdict) (ops : Array Op)
    (proof : ByteArray) :
    verifyBytes after ops proof = .trailing ↔
      ∃ messages rest, decode ops proof = .ok (messages, rest) ∧ rest ≠ 0 := by
  unfold verifyBytes
  rcases decode ops proof with error | ⟨messages, rest⟩
  · simp
  · by_cases empty : rest = 0 <;> simp [empty]

theorem verifyBytes_verdict (after : Array Message → Verdict) (ops : Array Op)
    (proof : ByteArray) (messages : Array Message)
    (decoded : decode ops proof = .ok (messages, 0)) :
    verifyBytes after ops proof = .verdict (after messages) := by
  simp [verifyBytes, decoded]

theorem fields_canonical {proof : ByteArray} {at_ count : ℕ} {values : Array Fq}
    (decoded : Wire.fields proof at_ count = some values) :
    ∀ a ∈ values, a.word.toNat < fieldModulus := by
  have step : ∀ (indices : List ℕ) (start : Array Fq),
      (∀ a ∈ start, a.word.toNat < fieldModulus) →
      indices.foldlM (fun values i =>
        (Fq.ofWord? (Wire.word proof (at_ + 8 * i))).map values.push) start = some values →
      ∀ a ∈ values, a.word.toNat < fieldModulus := by
    intro indices
    induction indices with
    | nil =>
        intro start below folded
        simp only [List.foldlM_nil] at folded
        cases folded
        exact below
    | cons i rest ih =>
        intro start below folded
        simp only [List.foldlM_cons] at folded
        cases read : Fq.ofWord? (Wire.word proof (at_ + 8 * i)) with
        | none => simp [read] at folded
        | some a =>
            simp only [read, Option.map_some, Option.bind_eq_bind, Option.bind_some] at folded
            refine ih (start.push a) ?_ folded
            intro b member
            rcases Array.mem_push.mp member with old | new
            · exact below b old
            · obtain ⟨same, lt⟩ := Fq.ofWord?_canonical read
              rw [new, same]
              exact lt
  unfold Wire.fields at decoded
  split at decoded
  · cases decoded
  · exact step _ _ (by simp) decoded

end NightstreamFPrime.Spec.Compression.Akita.Replay
