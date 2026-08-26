import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.PilotParity

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.run "emitted_pilot_parity"
    NightstreamFPrime.Export.Stage1.PilotParity.parityValue arguments
