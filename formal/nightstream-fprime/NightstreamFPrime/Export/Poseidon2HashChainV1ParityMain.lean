import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Parity

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.runIO
    "emitted_poseidon2_hash_chain_v1_parity"
    NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Parity.parityValueIO
    arguments
