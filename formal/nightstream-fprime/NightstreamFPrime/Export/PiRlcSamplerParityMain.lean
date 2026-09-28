import NightstreamFPrime.Export.Stage1.PiRlcSamplerParity

def main (arguments : List String) : IO UInt32 := do
  let path ← match arguments with
    | [path] | ["--", path] => pure path
    | _ => throw (IO.userError "usage: emitPiRlcSamplerParity OUTPUT.json")
  IO.FS.writeFile path (NightstreamFPrime.Export.Stage1.PiRlcSamplerParity.fixture.compress ++ "\n")
  pure 0
