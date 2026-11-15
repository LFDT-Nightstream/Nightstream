import NightstreamFPrime.Export.CompressionParity

def main (arguments : List String) : IO UInt32 := do
  let path ← match arguments with
    | [path] | ["--", path] => pure path
    | _ => throw (IO.userError "usage: emitCompressionParity OUTPUT.json")
  IO.FS.writeFile path (NightstreamFPrime.Export.CompressionParity.fixture.compress ++ "\n")
  pure 0
