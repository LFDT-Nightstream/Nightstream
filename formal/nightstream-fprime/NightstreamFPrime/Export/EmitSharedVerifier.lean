import NightstreamFPrime.Export.SharedVerifier

def main (arguments : List String) : IO UInt32 := do
  match arguments with
  | [path] | ["--", path] =>
      NightstreamFPrime.Export.SharedVerifier.write path
      pure 0
  | _ =>
      IO.eprintln "usage: emitSharedVerifier <output-path>"
      pure 2
