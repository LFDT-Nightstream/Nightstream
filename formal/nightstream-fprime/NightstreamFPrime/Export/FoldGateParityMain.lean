import NightstreamFPrime.Export.FoldGateParity
import NightstreamFPrime.Export.ParityEmitter

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.run "emitted_fold_gate_parity"
    NightstreamFPrime.Export.FoldGateParity.parityValue arguments
