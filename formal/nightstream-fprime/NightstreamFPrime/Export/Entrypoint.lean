import NightstreamFPrime.Export.Stage1.Wide.Emitter

/-! Executable entry point for the canonical wide-sampler package. -/

def main (arguments : List String) : IO UInt32 := do
  let arguments := if arguments.head? = some "--" then arguments.drop 1 else arguments
  match arguments with
  | [path] =>
    NightstreamFPrime.Export.Stage1.Wide.Emitter.emitSelected ⟨path⟩ false
    return 0
  | ["--expanded", path] =>
    NightstreamFPrime.Export.Stage1.Wide.Emitter.emitSelected ⟨path⟩ true
    return 0
  | _ =>
    IO.eprintln "usage: lake exe emit -- [--expanded] <output-path>"
    return 2
