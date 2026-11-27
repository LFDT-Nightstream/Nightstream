import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinExists

/-! The byte strings that the code absorbs for the stage `k` of the model, for `key20.aks` (Akita
revision `d50f700`; `docs/akita-work/w20/design.md`, decisions D2 and D3).

- `Op.input proof m` are the bytes that the step of a reading operation absorbs for the message
  `m`. `inputAt` is `Op.input` of the message at an index, and `inputsFrom … l n` are the absorbed
  bytes of the reading operations of `l`, with `n` the index of the first message.
- `stageIndex20 pub k` is the index of the work of the stage `k` (`workIndices20`), and the index
  of the terminal for `k ≥ 473`. `stageStart20 pub k` is the index after the work of the stage
  `k - 1`.
- `inputs20 … k` are the absorbed bytes of the reading operations from `stageStart20 pub k` up to
  and with the work of the stage `k`. `traceAt20 … k` is `inputs20 … k` as a trace of at most 5
  strings of at most 16384 bytes (`traceOf`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

/-- The bytes that the step of a reading operation absorbs for the message `m`: the nonce as one
word, the length and the bytes of the payload of `z`, and otherwise the bytes of the message. -/
def Op.input (proof : ByteArray) (m : Message) : Op → ByteArray
  | .bounded _ _ =>
      le64 (m.size - 4) ++ wordPadded (proof.extract (m.offset + 4) (m.offset + m.size))
  | .work _ _ _ | .foldNonce _ _ => match m.data with
      | .nonce nonce _ => le64 nonce
      | _ => .empty
  | _ => proof.extract m.offset (m.offset + m.size)

/-- The count of the reading operations of `l`. -/
def readCount (l : List (Op × ℕ)) : ℕ := (l.filter (·.1.reads)).length

/-- The bytes that the step of the operation absorbs for the message at the index `n`. -/
def inputAt (proof : ByteArray) (messages : Array Message) (op : Op) (n : ℕ) : ByteArray :=
  match messages[n]? with
  | some m => op.input proof m
  | none => .empty

/-- The absorbed bytes of the reading operations of `l`, in order; `n` is the index of the first
message. -/
def inputsFrom (proof : ByteArray) (messages : Array Message) : List (Op × ℕ) → ℕ → List ByteArray
  | [], _ => []
  | x :: l, n => (if x.1.reads then [inputAt proof messages x.1 n] else []) ++
      inputsFrom proof messages l (n + if x.1.reads then 1 else 0)

/-- The index of the operation before which the code has the wire message `k` of the model: the
work of the stage `k`, and the terminal for `k ≥ 473`. -/
def stageIndex20 (pub : Public) (k : ℕ) : ℕ :=
  if k < 473 then workIndices20.getD k 0 else pub.ops.size - 1

theorem stageIndex20_lt (pub : Public) {k : ℕ} (inside : k < 473) :
    stageIndex20 pub k = workIndices20.getD k 0 := by
  simp only [stageIndex20, inside, ite_true]

/-- The index of the first operation of the stage `k`: the operation after the work of the stage
`k - 1`. -/
def stageStart20 (pub : Public) (k : ℕ) : ℕ := if k = 0 then 0 else stageIndex20 pub (k - 1) + 1

/-- The byte strings that the code absorbs for the stage `k`: the absorbed bytes of each reading
operation after the work of the stage `k - 1` up to and with the work of the stage `k`. -/
def inputs20 (pub : Public) (proof : ByteArray) (messages : Array Message) (k : ℕ) :
    List ByteArray :=
  inputsFrom proof messages
    ((pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1)).drop (stageStart20 pub k))
    (readCount (pub.ops.zipIdx.toList.take (stageStart20 pub k)))

/-- The bytes of a byte array as a list. -/
def bytesOf (bytes : ByteArray) : List (Fin 256) :=
  bytes.data.toList.map fun byte => ⟨byte.toNat, byte.toNat_lt⟩

/-- The first `count` strings, each cut to `bytes` bytes, as a trace. -/
def traceOf (count bytes : ℕ) (strings : List ByteArray) : Extract.Trace count bytes :=
  ⟨(strings.take count).map fun string =>
      ⟨(bytesOf string).take bytes, by rw [List.length_take]; exact Nat.min_le_left _ _⟩,
    by rw [List.length_map, List.length_take]; exact Nat.min_le_left _ _⟩

/-- The trace of the stage `k`: the byte strings `inputs20 … k`. -/
def traceAt20 (pub : Public) (proof : ByteArray) (messages : Array Message) (k : ℕ) :
    Extract.Trace 5 16384 :=
  traceOf 5 16384 (inputs20 pub proof messages k)

end NightstreamFPrime.Spec.Compression.Akita.Replay
