import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.AjtaiSetupV1Parity

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.run "emitted_ajtai_setup_v1_parity"
    NightstreamFPrime.Export.Stage1.AjtaiSetupV1Parity.parityValue arguments
