import NightstreamFPrime.Export.Stage1.Wide.PiRLCSoundness

/-! The final retained values are the four PiRLC families' ordered K+k sums.
Both arbitrary accepted assignments and the direct witness have this meaning. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PiRLCOutput

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation PiRlcWideSampler
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiRLCGeometry

abbrev Family := PiRLCProductRingSchedule.Family

def terminal (family : Family) (block : Fin family.blockCount) (cell : Fin family.cellCount) : RingIndex :=
  ({ family, source := ⟨16, by decide⟩, block, cell } : Descriptor).invocation

def ordered (initial : PiRLCValues.Initial) (values : PiRLCValues.Values)
    (family : Family) (block : Fin family.blockCount) (cell : Fin family.cellCount) : RingF :=
  Lifecycle.PiRLC.v1_1.CombinationFamily.rightCombination (count := 17) fun source =>
    ringFMul (PiRLCValues.challenge initial source.val)
      (values ({ family, source, block, cell } : Descriptor).invocation)

theorem direct_eq_ordered (initial : PiRLCValues.Initial) (values : PiRLCValues.Values)
    (family : Family) (block : Fin family.blockCount) (cell : Fin family.cellCount) :
    PiRLCValues.output initial values (terminal family block cell) = ordered initial values family block cell := by
  unfold PiRLCValues.output terminal
  rw [PiRLCProductRingSchedule.descriptor_invocation]
  funext lane
  simp only [PiRLCValues.partialSum, PiRLCValues.term, withSource,
    ordered, Lifecycle.PiRLC.v1_1.CombinationFamily.rightCombination,
    ringFAdd, ringFZero, Fin.val_zero, Fin.val_succ]
  norm_num
  abel

theorem soundness {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns)
    (assignment : Assignment F columns) (one : assignment interface.oneColumn = 1)
    (rows : (plan compiled interface).RowsZero assignment)
    (family : Family) (block : Fin family.blockCount) (cell : Fin family.cellCount) :
    Phi81ProductPlan.evalState assignment (output interface (terminal family block cell)) =
      ordered (PiRLCWitness.initial interface assignment) (PiRLCWitness.inputValues interface assignment)
        family block cell := by
  rw [PiRLCSoundness.output_eq_direct compiled interface assignment one rows, direct_eq_ordered]

theorem witness_output {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (family : Family) (block : Fin family.blockCount) (cell : Fin family.cellCount) :
    Phi81ProductPlan.evalState (PiRLCWitness.assignment interface base)
        (output interface (terminal family block cell)) =
      ordered (PiRLCWitness.initial interface base) (PiRLCWitness.inputValues interface base)
        family block cell := by
  rw [PiRLCWitness.output_eq, direct_eq_ordered]

end NightstreamFPrime.Export.Stage1.Wide.PiRLCOutput
