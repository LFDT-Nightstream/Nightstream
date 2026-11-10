import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.AjtaiSetupV1Parity
import NightstreamFPrime.Spec
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.GoldilocksCausal
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.run "emitted_ajtai_setup_v1_parity"
    NightstreamFPrime.Export.Stage1.AjtaiSetupV1Parity.parityValue arguments
