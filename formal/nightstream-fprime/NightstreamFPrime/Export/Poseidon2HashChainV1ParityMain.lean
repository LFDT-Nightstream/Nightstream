import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Parity
import NightstreamFPrime.Export.Stage1.Wide.BaseStepFixture
import NightstreamFPrime.Export.Stage1.Wide.FixedPoint

open NightstreamFPrime.Export.Stage1

private def terminalLayout : Except String NightstreamFPrime.Export.Package.TerminalLayout := do
  let some compiled := NightstreamFPrime.Layout.PiRlcWideSampler.RangePlan.compile?
    | throw "wide range compilation failed"
  return {
    rowStart := 0
    rowCount := (Wide.FixedPoint.structuralPlan Poseidon2HashChainV1Package.application
      compiled Poseidon2HashChainV1Package.fits).rowCount
    runningClaims := NightstreamFPrime.Lifecycle.productionShape.runningCount
    freshClaims := NightstreamFPrime.Lifecycle.productionShape.freshCount }

private def usage : String :=
  "usage: emitPoseidon2HashChainV1Parity " ++
    "<context0> <context1> <context2> <context3> <output-path>"

private def run (c0 c1 c2 c3 path : String) : IO UInt32 := do
  match NightstreamFPrime.Export.ParityEmitter.parseVerifierKey c0 c1 c2 c3 with
  | .error error =>
      IO.eprintln error
      pure 2
  | .ok context =>
      let terminal ← match terminalLayout with
        | .ok terminal => pure terminal
        | .error error => throw (IO.userError error)
      NightstreamFPrime.Export.ParityEmitter.runIO
        "emitted_poseidon2_hash_chain_v1_parity"
        (NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Parity.parityValueIO
          Wide.BaseStepFixture.batch terminal context) [path]

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | [c0, c1, c2, c3, path] => run c0 c1 c2 c3 path
  | ["--", c0, c1, c2, c3, path] => run c0 c1 c2 c3 path
  | _ => do
      IO.eprintln usage
      pure 2
