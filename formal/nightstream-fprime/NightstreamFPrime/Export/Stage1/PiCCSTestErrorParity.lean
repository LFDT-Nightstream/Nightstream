import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Lifecycle.Nifs.SelectedTestNumerator
import NightstreamFPrime.Spec
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.GoldilocksCausal
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

/-! Rust recomputes the numerator from the same counts and must match it. -/

namespace NightstreamFPrime.Export.Stage1.PiCCSTestErrorParity

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Lifecycle

def schema : Nat := 1

def parityValue : Value :=
  .array [
    .atom schema,
    .atom productionShape.cubeVariables,
    .atom Nifs.SelectedTestNumerator.width,
    .atom productionShape.freshCount,
    .atom productionShape.runningCount,
    .atom productionShape.coefficientCount,
    .atom productionShape.matrixCount,
    .atom Nifs.SelectedTestNumerator.numerator]

end NightstreamFPrime.Export.Stage1.PiCCSTestErrorParity
