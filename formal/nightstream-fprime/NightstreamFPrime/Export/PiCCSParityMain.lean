import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.PiCCSParity

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.run "emitted_pi_ccs_parity"
    (NightstreamFPrime.Export.Stage1.PiCCSParity.parityValue ()) arguments
