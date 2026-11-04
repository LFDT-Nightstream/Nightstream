import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.PiCCSTestErrorParity

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.run "emitted_piccs_test_error_parity"
    NightstreamFPrime.Export.Stage1.PiCCSTestErrorParity.parityValue arguments
