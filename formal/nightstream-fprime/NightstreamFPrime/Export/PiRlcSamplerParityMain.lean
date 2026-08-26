import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.PiRlcSamplerParity

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.run "emitted_pi_rlc_sampler_parity"
    NightstreamFPrime.Export.Stage1.PiRlcSamplerParity.parityValue arguments
