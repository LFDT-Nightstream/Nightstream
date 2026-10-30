import CompressionLean.Ports.Circuit.Hash

namespace CompressionLean.Circuit

open CompressionLean.Replay (Word)

def compressDomain : String := "Nightstream/SuperNeo/compress/v1"

/-- Rust `neo_transcript::domain_chunk_v1_2`; the label has at most 96 bytes. -/
def domainChunk (label : String) : Array Word :=
  let bytes := label.toUTF8.data
  Array.ofFn fun word : Fin rate =>
    Word.ofNat ((List.range 8).foldl (fun acc i => acc + (bytes.getD (8 * word + i) 0).toNat * 256 ^ i) 0)

structure FoldTranscript where
  state : Array Var

namespace FoldTranscript

def new : Build FoldTranscript := do
  let zeroWord ← Build.const 0
  pure ⟨Array.replicate 16 zeroWord⟩

def fromState (state : Fin 16 → Var) : FoldTranscript := ⟨Array.ofFn state⟩

def absorb (t : FoldTranscript) (words : Array Var) : Build FoldTranscript := do
  let mut state := t.state
  for chunk in chunks words do
    for (word, lane) in chunk.zipIdx do
      state := state.set! lane (← Build.add state[lane]! word)
    state := Array.ofFn (← Build.permute fun lane => state[lane]!)
  pure ⟨state⟩

def absorbConstants (t : FoldTranscript) (words : Array Word) : Build FoldTranscript := do
  let words ← words.mapM Build.const
  t.absorb words

def statePrefix (t : FoldTranscript) : DigestVar := fun lane => t.state[lane]!

def squeezeDigest (t : FoldTranscript) : Build (DigestVar × FoldTranscript) := do
  let digest := t.statePrefix
  let lanes ← Build.permute fun lane => t.state[lane]!
  pure (digest, ⟨Array.ofFn lanes⟩)

def handoff (t : FoldTranscript) : Build DigestVar := do
  let t ← t.absorbConstants (domainChunk compressDomain)
  pure (← t.squeezeDigest).1

end FoldTranscript

end CompressionLean.Circuit
