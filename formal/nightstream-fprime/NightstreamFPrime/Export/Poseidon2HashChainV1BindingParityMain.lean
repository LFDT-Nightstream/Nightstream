import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1BindingParity

open NightstreamFPrime.Lifecycle.VerifierContext
open NightstreamFPrime.Spec

private def usage : String :=
  "usage: emitPoseidon2HashChainV1BindingParity <id0> <id1> <id2> <id3> <relation0> <relation1> <relation2> <relation3> <application0> <application1> <application2> <application3> <nifs0> <nifs1> <nifs2> <nifs3> <commitment0> <commitment1> <commitment2> <commitment3> <output-path>"

private def parseDigestWord (label word : String) : Except String F := do
  if word.isEmpty || word.length > 20 ||
      !(word.toList.all Char.isDigit) then
    throw s!"invalid digest word {label}: expected decimal UInt64"
  let some value := word.toNat?
    | throw s!"invalid digest word {label}: expected decimal UInt64"
  if _ : value < UInt64.size then
    if canonical : value < goldilocksModulus then
      pure ⟨value, canonical⟩
    else
      throw s!"invalid digest word {label}: not a canonical Goldilocks element"
  else
    throw s!"invalid digest word {label}: expected decimal UInt64"

private def parseDigest (label w0 w1 w2 w3 : String) :
    Except String Digest4 := do
  let c0 ← parseDigestWord s!"{label}0" w0
  let c1 ← parseDigestWord s!"{label}1" w1
  let c2 ← parseDigestWord s!"{label}2" w2
  let c3 ← parseDigestWord s!"{label}3" w3
  pure { c0, c1, c2, c3 }

private def parseInputs
    (i0 i1 i2 i3 r0 r1 r2 r3 a0 a1 a2 a3 n0 n1 n2 n3 c0 c1 c2 c3 :
      String) :
    Except String (Digest4 × Digest4 × Digest4 × Digest4 × Digest4) := do
  let structural ← parseDigest "id" i0 i1 i2 i3
  let relation ← parseDigest "relation" r0 r1 r2 r3
  let application ← parseDigest "application" a0 a1 a2 a3
  let nifsKey ← parseDigest "nifs" n0 n1 n2 n3
  let commitmentKey ← parseDigest "commitment" c0 c1 c2 c3
  pure (structural, relation, application, nifsKey, commitmentKey)

private def run
    (i0 i1 i2 i3 r0 r1 r2 r3 a0 a1 a2 a3 n0 n1 n2 n3 c0 c1 c2 c3 path :
      String) : IO UInt32 := do
  match parseInputs i0 i1 i2 i3 r0 r1 r2 r3 a0 a1 a2 a3 n0 n1 n2 n3
      c0 c1 c2 c3 with
  | .error error =>
      IO.eprintln error
      pure 2
  | .ok (structural, relation, application, nifsKey, commitmentKey) =>
      NightstreamFPrime.Export.ParityEmitter.runIO
        "emitted_poseidon2_hash_chain_v1_binding_parity"
        (NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1BindingParity.parityValueIO
          structural relation application nifsKey commitmentKey) [path]

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | [i0, i1, i2, i3, r0, r1, r2, r3, a0, a1, a2, a3,
      n0, n1, n2, n3, c0, c1, c2, c3, path] =>
      run i0 i1 i2 i3 r0 r1 r2 r3 a0 a1 a2 a3 n0 n1 n2 n3
        c0 c1 c2 c3 path
  | ["--", i0, i1, i2, i3, r0, r1, r2, r3, a0, a1, a2, a3,
      n0, n1, n2, n3, c0, c1, c2, c3, path] =>
      run i0 i1 i2 i3 r0 r1 r2 r3 a0 a1 a2 a3 n0 n1 n2 n3
        c0 c1 c2 c3 path
  | _ => do
      IO.eprintln usage
      pure 2
