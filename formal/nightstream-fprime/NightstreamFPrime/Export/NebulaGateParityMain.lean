import NightstreamFPrime.Export.NebulaGateParity
import NightstreamFPrime.Export.ParityEmitter

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.run "emitted_nebula_gate_parity"
    NightstreamFPrime.Export.NebulaGateParity.parityValue arguments
