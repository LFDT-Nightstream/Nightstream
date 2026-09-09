import NightstreamFPrime.Lifecycle.Pilot
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.Completeness
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.Completeness
import NightstreamFPrime.Lifecycle.Stage1.Application
import NightstreamFPrime.Lifecycle.Stage1.RunningTransition
import NightstreamFPrime.Lifecycle.Stage1.NextPreimage

/-!
Owns the symbolic interface of one Stage 1 augmented circuit.

The verifier selects `relation`, `ajtai`, and one closed application program
before this interface is constructed. This record carries only the symbolic
wire views used by the existing opaque phase circuits. Physical columns and
the outer HyperNova terminal verifier belong to `Layout/` and `Terminal`.
-/

namespace NightstreamFPrime.Lifecycle.Stage1

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra

/-- Symbolic child interfaces for one fixed Stage 1 application. -/
structure Interface
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Application.Program) where
  pilot : Pilot.Interface
  piCcs : PiCCS.v1_1.Formal.Interface logicalWidth
    (ProductionKey.degreeBound relation) publicFits
  piRlc : PiRLC.v1_1.Formal.Interface logicalWidth publicFits
  piDec : PiDEC.v1_1.Formal.Interface logicalWidth publicFits
  running : RunningTransition.Interface logicalWidth publicFits
  application : Application.Interface program.witnessWordCount
  nextPreimage : NextPreimage.Interface

end NightstreamFPrime.Lifecycle.Stage1
