import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.AjtaiSparseCommitmentV1Parity

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.run "emitted_ajtai_sparse_commitment_v1_parity"
    (NightstreamFPrime.Export.Stage1.AjtaiSparseCommitmentV1Parity.parityValue ()) arguments
