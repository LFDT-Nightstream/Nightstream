import NightstreamFPrime.Export.Stage1.Wide.InputSupport
import NightstreamFPrime.Export.Stage1.Wide.PiRLCOutput

/-! Execute the compact PiRLC witness on the actual Stage 1 input forms.
Common coordinates are preserved. The other Stage 1 phases still require
their own compatible source completion before production selection. -/

namespace NightstreamFPrime.Export.Stage1.Wide.Stage1Witness

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def assignment (program : RetainedLayout.Program)
    (base : Assignment F (RetainedLayout.logicalWidth program)) :=
  PiRLCWitness.assignment (Stage1Plan.piRlcInterface program) base

theorem complete (program : RetainedLayout.Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (base : Assignment F (RetainedLayout.logicalWidth program))
    (one : base (Stage1Plan.piRlcInterface program).oneColumn = 1) :
    (Stage1Plan.piRlc program compiled).RowsZero (assignment program base) :=
  PiRLCWitness.complete compiled (Stage1Plan.piRlcInterface program) base
    (InputSupport.inputsBefore program) one

theorem common_unchanged (program : RetainedLayout.Program)
    (base : Assignment F (RetainedLayout.logicalWidth program))
    (column : Fin (PerApplicationFixedPoint.logicalWidth program)) (common : FormSupport.Common program column) :
    assignment program base (RetainedLayout.column program column) =
      base (RetainedLayout.column program column) :=
  PiRLCWitness.assignment_before (Stage1Plan.piRlcInterface program) base _
    (FormSupport.common_before program column common)

theorem common_form_unchanged (program : RetainedLayout.Program)
    (base : Assignment F (RetainedLayout.logicalWidth program))
    (form : SparseForm (PerApplicationFixedPoint.logicalWidth program))
    (supported : FormSupport.Supported (FormSupport.Common program) form) :
    (form.mapColumns (RetainedLayout.column program)).eval (assignment program base) =
      (form.mapColumns (RetainedLayout.column program)).eval base := by
  apply PiRLCWitness.disjoint_form
  intro entry member
  exact Or.inl (FormSupport.renamed_before program form supported entry member)

theorem output (program : RetainedLayout.Program)
    (base : Assignment F (RetainedLayout.logicalWidth program))
    (family : PiRLCOutput.Family) (block : Fin family.blockCount) (cell : Fin family.cellCount) :
    Phi81ProductPlan.evalState (assignment program base)
      (PiRLCGeometry.output (Stage1Plan.piRlcInterface program) (PiRLCOutput.terminal family block cell)) =
      PiRLCOutput.ordered (PiRLCWitness.initial (Stage1Plan.piRlcInterface program) base)
        (PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program) base) family block cell :=
  PiRLCOutput.witness_output _ _ _ _ _

theorem accepted_output (program : RetainedLayout.Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    {relationWidth : Nat}
    {publicFits : ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationWidth}
    (relation : Lifecycle.ProductionKey.LogicalRelation relationWidth publicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 program)
    (values : Assignment F (RetainedLayout.logicalWidth program))
    (one : values (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (rows : (Stage1Plan.plan program compiled relation fits).RowsZero values)
    (family : PiRLCOutput.Family) (block : Fin family.blockCount) (cell : Fin family.cellCount) :
    Phi81ProductPlan.evalState values
      (PiRLCGeometry.output (Stage1Plan.piRlcInterface program) (PiRLCOutput.terminal family block cell)) =
      PiRLCOutput.ordered (PiRLCWitness.initial (Stage1Plan.piRlcInterface program) values)
        (PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program) values) family block cell := by
  exact PiRLCOutput.soundness compiled _ values one
    ((Stage1Plan.rows_iff program compiled relation fits values).mp rows).2.1 family block cell

end NightstreamFPrime.Export.Stage1.Wide.Stage1Witness
