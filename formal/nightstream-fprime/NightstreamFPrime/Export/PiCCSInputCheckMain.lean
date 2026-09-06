import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.PiCCSInputCheck

private def run (inputPath outputPath : String) : IO UInt32 := do
  let text ← IO.FS.readFile inputPath
  match NightstreamFPrime.Export.Stage1.PiCCSInputCheck.parse text with
  | .error error =>
      IO.eprintln error
      pure 2
  | .ok input =>
      NightstreamFPrime.Export.ParityEmitter.emit "checked_pi_ccs_input"
        (NightstreamFPrime.Export.Stage1.PiCCSInputCheck.checkValue input)
        outputPath
      pure 0

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | [inputPath, outputPath] => run inputPath outputPath
  | ["--", inputPath, outputPath] => run inputPath outputPath
  | _ => do
      IO.eprintln "usage: checkPiCCSInput <input-json-path> <output-path>"
      pure 2
