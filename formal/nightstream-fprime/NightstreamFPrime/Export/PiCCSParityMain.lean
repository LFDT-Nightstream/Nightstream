import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.PiCCSParity

private def usage : String :=
  "usage: emitPiCCSParity <vk0> <vk1> <vk2> <vk3> <output-path>"

private def run (w0 w1 w2 w3 path : String) : IO UInt32 := do
  match NightstreamFPrime.Export.ParityEmitter.parseVerifierKey w0 w1 w2 w3 with
  | .error error =>
      IO.eprintln error
      pure 2
  | .ok verifierKey =>
      NightstreamFPrime.Export.ParityEmitter.runIO "emitted_pi_ccs_parity"
        (NightstreamFPrime.Export.Stage1.PiCCSParity.parityValueIO
          verifierKey.toList) [path]

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | [w0, w1, w2, w3, path] => run w0 w1 w2 w3 path
  | ["--", w0, w1, w2, w3, path] => run w0 w1 w2 w3 path
  | _ => do
      IO.eprintln usage
      pure 2
