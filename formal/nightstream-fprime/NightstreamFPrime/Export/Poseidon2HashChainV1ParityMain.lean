import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Parity

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
          context) [path]

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | [c0, c1, c2, c3, path] => run c0 c1 c2 c3 path
  | ["--", c0, c1, c2, c3, path] => run c0 c1 c2 c3 path
  | _ => do
      IO.eprintln usage
      pure 2
