import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.PiDECParity

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.runIO "emitted_pi_dec_parity"
    NightstreamFPrime.Export.Stage1.PiDECParity.parityValueIO arguments
