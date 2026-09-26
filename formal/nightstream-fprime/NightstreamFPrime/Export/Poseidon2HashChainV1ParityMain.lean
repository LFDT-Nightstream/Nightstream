import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Parity
import NightstreamFPrime.Export.Stage1.Wide.BaseStepFixture
import NightstreamFPrime.Export.Stage1.Wide.FixedPoint

open NightstreamFPrime.Export.Stage1

private def terminalLayout : NightstreamFPrime.Export.Package.TerminalLayout := {
    rowStart := 0
    rowCount := 3248694 + ApplicationDirectPlan.rowCount Poseidon2HashChainV1Package.application
    runningClaims := NightstreamFPrime.Lifecycle.productionShape.runningCount
    freshClaims := NightstreamFPrime.Lifecycle.productionShape.freshCount }

private theorem terminalLayout_rows (compiled : NightstreamFPrime.Layout.PiRlcWideSampler.RangePlan.Compiled) :
    terminalLayout.rowCount = (Wide.FixedPoint.structuralPlan Poseidon2HashChainV1Package.application
      compiled Poseidon2HashChainV1Package.fits).rowCount := by
  exact (Wide.Stage1Plan.plan_rows _ _ _ _).symm

private def usage : String :=
  "usage: emitPoseidon2HashChainV1Parity " ++
    "<context0> <context1> <context2> <context3> <output-path>"

private def run (c0 c1 c2 c3 path : String) : IO UInt32 := do
  match NightstreamFPrime.Export.ParityEmitter.parseVerifierKey c0 c1 c2 c3 with
  | .error error =>
      IO.eprintln error
      pure 2
  | .ok context =>
      NightstreamFPrime.Export.ParityEmitter.runIO
        "emitted_poseidon2_hash_chain_v1_parity"
        (NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Parity.parityValueIO
          Wide.BaseStepFixture.batch terminalLayout context) [path]

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | [c0, c1, c2, c3, path] => run c0 c1 c2 c3 path
  | ["--", c0, c1, c2, c3, path] => run c0 c1 c2 c3 path
  | _ => do
      IO.eprintln usage
      pure 2
