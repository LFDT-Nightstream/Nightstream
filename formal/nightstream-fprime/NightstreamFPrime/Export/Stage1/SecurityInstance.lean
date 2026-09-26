import NightstreamFPrime.Export.Stage1.PiCCSInputCheck

/-! The width, relation and Ajtai key that one extraction statement is about.
The selected package supplies one value. The proofs read only these fields. -/

namespace NightstreamFPrime.Export.Stage1

open NightstreamFPrime.Lifecycle NightstreamFPrime.Spec Spec.Folding.PiCCS.PaperJoint

structure SecurityInstance where
  logicalWidth : Nat
  publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth
  relation : ProductionKey.LogicalRelation logicalWidth publicFits
  ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)

namespace SecurityInstance

variable (inst : SecurityInstance)

/-- The checker input's running claims at this instance's width. -/
abbrev running (input : PiCCSInputCheck.Input) :=
  PiCCSInputCheck.runningAt (width := inst.logicalWidth) (fits := inst.publicFits) input

/-- The checker input's fresh claim at this instance's width. -/
abbrev fresh (input : PiCCSInputCheck.Input) :=
  PiCCSInputCheck.freshAt (width := inst.logicalWidth) (fits := inst.publicFits) input

end SecurityInstance

end NightstreamFPrime.Export.Stage1
