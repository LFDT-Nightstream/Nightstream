import NightstreamFPrime.Export.SharedFormulas
import NightstreamFPrime.Export.MatrixProgram
import NightstreamFPrime.Layout.MatrixProgram.RingProduct
import NightstreamFPrime.Layout.MatrixProgram.Poseidon
import NightstreamFPrime.Layout.ProductionRelation.OrdinarySourcePlan
import NightstreamFPrime.Spec
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FieldCausal
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

def main (arguments : List String) : IO UInt32 := do
  match arguments with
  | [path] | ["--", path] =>
      NightstreamFPrime.Export.SharedFormulas.write path
      pure 0
  | _ =>
      IO.eprintln "usage: emitSharedFormulas <output-path>"
      pure 2
