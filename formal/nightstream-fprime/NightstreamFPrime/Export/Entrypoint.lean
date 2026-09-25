import NightstreamFPrime.Export.Main
import NightstreamFPrime.Export.Stage1.Wide.Emitter

/-! Executable wrapper for the canonical Stage 1 package emitter. -/

def main (arguments : List String) : IO UInt32 := do
  let arguments := if arguments.head? = some "--" then arguments.drop 1 else arguments
  match arguments with
  | ["--poseidon2-hash-chain-v1", path] =>
    NightstreamFPrime.Export.Stage1.Wide.Emitter.emitSelected ⟨path⟩ false
    return 0
  | ["--poseidon2-hash-chain-v1-expanded", path] =>
    NightstreamFPrime.Export.Stage1.Wide.Emitter.emitSelected ⟨path⟩ true
    return 0
  | _ => NightstreamFPrime.Export.Main.run arguments
