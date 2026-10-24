import CompressionLean.Circuit.Hash

/-!
Owns the v1.2 fold transcript of `crates/neo-spartan/src/circuit/fold.rs` as
builder programs: the additive sponge (`absorb`, `squeezeDigest`,
`statePrefix`) and the layer-1 handoff (`handoff`). Each one calls `Build` in
the order of its Rust counterpart.
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Word)

/-- The layer-1 domain label (`hash::DOMAIN`, `hash.rs:24`). -/
def compressDomain : String := "Nightstream/SuperNeo/compress/v1"

/-- `neo_transcript::domain_chunk_v1_2`: the label's bytes, eight per word,
little-endian, zero-padded to one rate chunk. The label has at most 96
bytes. -/
def domainChunk (label : String) : Array Word :=
  let bytes := label.toUTF8.data
  Array.ofFn fun word : Fin rate =>
    Word.ofNat ((List.range 8).foldl (fun acc i => acc + (bytes.getD (8 * word + i) 0).toNat * 256 ^ i) 0)

/-- The fold transcript's 16 lanes; it never holds a partial chunk
(`fold.rs:15-17`). -/
structure FoldTranscript where
  state : Array Var

namespace FoldTranscript

/-- The zero state (`fold.rs:27-30`). -/
def new : Build FoldTranscript := do
  let zeroWord ← Build.const 0
  pure ⟨Array.replicate 16 zeroWord⟩

/-- A transcript at a known state (`fold.rs:33-35`). -/
def fromState (state : Fin 16 → Var) : FoldTranscript := ⟨Array.ofFn state⟩

/-- Add each word into the rate lanes; permute after every complete or
partial chunk (`fold.rs:39-46`). -/
def absorb (t : FoldTranscript) (words : Array Var) : Build FoldTranscript := do
  let mut state := t.state
  for chunk in chunks words do
    for (word, lane) in chunk.zipIdx do
      state := state.set! lane (← Build.add state[lane]! word)
    state := Array.ofFn (← Build.permute fun lane => state[lane]!)
  pure ⟨state⟩

/-- `fold.rs:49-52`. -/
def absorbConstants (t : FoldTranscript) (words : Array Word) : Build FoldTranscript := do
  let words ← words.mapM Build.const
  t.absorb words

/-- The first four lanes; the state does not change (`prefix`, `fold.rs:62-64`). -/
def statePrefix (t : FoldTranscript) : DigestVar := fun lane => t.state[lane]!

/-- The first four lanes, then one permutation (`fold.rs:55-59`). -/
def squeezeDigest (t : FoldTranscript) : Build (DigestVar × FoldTranscript) := do
  let digest := t.statePrefix
  let lanes ← Build.permute fun lane => t.state[lane]!
  pure (digest, ⟨Array.ofFn lanes⟩)

/-- Name layer 1 in the transcript and return the layer-1 seed
(`fold.rs:68-72`). -/
def handoff (t : FoldTranscript) : Build DigestVar := do
  let t ← t.absorbConstants (domainChunk compressDomain)
  pure (← t.squeezeDigest).1

end FoldTranscript

end CompressionLean.Circuit
