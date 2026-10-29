import NightstreamFPrime.Lifecycle.Pilot
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Completeness
import NightstreamFPrime.Lifecycle.PiRLC.v1_2.Completeness
import NightstreamFPrime.Lifecycle.PiDEC.v1_2.Completeness
import NightstreamFPrime.Lifecycle.Stage1.Application
import NightstreamFPrime.Lifecycle.Stage1.RunningTransition
import NightstreamFPrime.Lifecycle.Stage1.NextPreimage

namespace NightstreamFPrime.Lifecycle.Stage1

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra

structure Interface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Application.Program) where
  pilot : Pilot.Interface
  piCcs : PiCCS.v1_2.Formal.Interface logicalWidth
    (ProductionKey.degreeBound relation) publicFits
  piRlc : PiRLC.v1_2.Formal.Interface logicalWidth publicFits
  piDec : PiDEC.v1_2.Formal.Interface logicalWidth publicFits
  running : RunningTransition.Interface logicalWidth publicFits
  application : Application.Interface program.witnessWordCount
  nextPreimage : NextPreimage.Interface

end NightstreamFPrime.Lifecycle.Stage1
