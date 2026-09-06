import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.RecursiveStepFixture

private def run (w0 w1 w2 w3 inputPath childrenPath outputPath : String) :
    IO UInt32 := do
  match NightstreamFPrime.Export.ParityEmitter.parseVerifierKey w0 w1 w2 w3 with
  | .error error => IO.eprintln error; pure 2
  | .ok context =>
      let inputText ← IO.FS.readFile inputPath
      let childrenText ← IO.FS.readFile childrenPath
      match NightstreamFPrime.Export.Stage1.PiCCSInputCheck.parse inputText,
          NightstreamFPrime.Export.Stage1.PiCCSInputCheck.parseRunning childrenText with
      | .ok input, .ok children =>
          NightstreamFPrime.Export.ParityEmitter.runIO "emitted_recursive_step_fixture"
            (NightstreamFPrime.Export.Stage1.RecursiveStepFixture.valueIO context input children)
            [outputPath]
      | .error error, _ | _, .error error => IO.eprintln error; pure 2

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | [w0, w1, w2, w3, inputPath, childrenPath, outputPath] =>
      run w0 w1 w2 w3 inputPath childrenPath outputPath
  | ["--", w0, w1, w2, w3, inputPath, childrenPath, outputPath] =>
      run w0 w1 w2 w3 inputPath childrenPath outputPath
  | _ => do
      IO.eprintln "usage: emitRecursiveStepFixture <context[4]> <PiCCS-input> <child-running> <output>"
      pure 2
