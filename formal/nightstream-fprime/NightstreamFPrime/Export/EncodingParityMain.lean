import NightstreamFPrime.Export.EncodingParity
import NightstreamFPrime.Export.ParityEmitter

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.run "emitted_encoding_parity"
    NightstreamFPrime.Export.EncodingParity.parityValue arguments
