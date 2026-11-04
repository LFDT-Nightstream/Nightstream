import NightstreamFPrime.Export.SharedFormulas

def main (arguments : List String) : IO UInt32 := do
  match arguments with
  | [path] | ["--", path] =>
      NightstreamFPrime.Export.SharedFormulas.write path
      pure 0
  | _ =>
      IO.eprintln "usage: emitSharedFormulas <output-path>"
      pure 2
