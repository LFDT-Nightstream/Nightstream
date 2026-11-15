import NightstreamFPrime.Export.MatrixPassParity
import NightstreamFPrime.Export.ParityEmitter

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.run "emitted_matrix_pass_parity"
    NightstreamFPrime.Export.MatrixPassParity.parityValue arguments
