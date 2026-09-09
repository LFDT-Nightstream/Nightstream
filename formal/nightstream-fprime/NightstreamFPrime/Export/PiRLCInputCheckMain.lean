import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.PiRLCInputCheck

private def run (p0 p1 p2 p3 inputPath outputPath : String) : IO UInt32 := do
  let text ← IO.FS.readFile inputPath
  let parsed := do
    let packageIdentity ← NightstreamFPrime.Export.ParityEmitter.parseVerifierKey p0 p1 p2 p3
    let input ← NightstreamFPrime.Export.Stage1.PiCCSInputCheck.parse text
    pure (packageIdentity, input)
  match parsed with
  | .error error =>
      IO.eprintln error
      pure 2
  | .ok (packageIdentity, input) =>
      NightstreamFPrime.Export.ParityEmitter.runIO "checked_pi_rlc_input"
        (NightstreamFPrime.Export.Stage1.PiRLCInputCheck.checkValueIO input packageIdentity)
        [outputPath]

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | [p0, p1, p2, p3, inputPath, outputPath] => run p0 p1 p2 p3 inputPath outputPath
  | ["--", p0, p1, p2, p3, inputPath, outputPath] => run p0 p1 p2 p3 inputPath outputPath
  | _ => do
      IO.eprintln "usage: checkPiRLCInput <package[4]> <PiCCS-input> <output-path>"
      pure 2
