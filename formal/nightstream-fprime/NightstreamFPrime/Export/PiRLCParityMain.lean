import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.PiRLCParity

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.run "emitted_pi_rlc_parity"
    NightstreamFPrime.Export.Stage1.PiRLCParity.parityValue arguments
