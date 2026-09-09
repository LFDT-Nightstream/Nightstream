import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.PiDECParity

private def usage : String :=
  "usage: emitPiDECParity <context0> <context1> <context2> <context3> " ++
    "<package0> <package1> <package2> <package3> <output-path>"

private def run (c0 c1 c2 c3 p0 p1 p2 p3 path : String) : IO UInt32 := do
  let parsed := do
    let context ←
      NightstreamFPrime.Export.ParityEmitter.parseVerifierKey c0 c1 c2 c3
    let packageIdentity ←
      (NightstreamFPrime.Export.ParityEmitter.parseVerifierKey p0 p1 p2 p3).mapError
        (fun error => s!"package identity: {error}")
    pure (context, packageIdentity)
  match parsed with
  | .error error =>
      IO.eprintln error
      pure 2
  | .ok (context, packageIdentity) =>
      NightstreamFPrime.Export.ParityEmitter.runIO "emitted_pi_dec_parity"
        (NightstreamFPrime.Export.Stage1.PiDECParity.parityValueIO
          context packageIdentity) [path]

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | [c0, c1, c2, c3, p0, p1, p2, p3, path] =>
      run c0 c1 c2 c3 p0 p1 p2 p3 path
  | ["--", c0, c1, c2, c3, p0, p1, p2, p3, path] =>
      run c0 c1 c2 c3 p0 p1 p2 p3 path
  | _ => do
      IO.eprintln usage
      pure 2
