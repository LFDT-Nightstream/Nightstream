import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Export.Stage1.PiCCSTestErrorParity
import NightstreamFPrime.Spec
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FieldCausal
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.run "emitted_piccs_test_error_parity"
    NightstreamFPrime.Export.Stage1.PiCCSTestErrorParity.parityValue arguments
