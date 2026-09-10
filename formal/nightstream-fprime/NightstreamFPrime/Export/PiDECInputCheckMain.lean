import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.PiDECInputCheck

private def run (p0 p1 p2 p3 inputPath messagesPath outputPath : String) : IO UInt32 := do
  let inputText ← IO.FS.readFile inputPath
  let messagesText ← IO.FS.readFile messagesPath
  let parsed := do
    let identity ← NightstreamFPrime.Export.ParityEmitter.parseVerifierKey p0 p1 p2 p3
    let input ← NightstreamFPrime.Export.Stage1.PiCCSInputCheck.parse inputText
    let messages ← NightstreamFPrime.Export.Stage1.PiCCSInputCheck.parseRunning messagesText
    pure (identity, input, messages)
  match parsed with
  | .error error =>
      IO.eprintln error
      pure 2
  | .ok (identity, input, messages) =>
      NightstreamFPrime.Export.ParityEmitter.runIO "checked_pi_dec_input"
        (NightstreamFPrime.Export.Stage1.PiDECInputCheck.checkValueIO input messages identity)
        [outputPath]

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | [p0, p1, p2, p3, inputPath, messagesPath, outputPath] =>
      run p0 p1 p2 p3 inputPath messagesPath outputPath
  | ["--", p0, p1, p2, p3, inputPath, messagesPath, outputPath] =>
      run p0 p1 p2 p3 inputPath messagesPath outputPath
  | _ => do
      IO.eprintln "usage: checkPiDECInput <package[4]> <PiCCS-input> <children> <output>"
      pure 2
