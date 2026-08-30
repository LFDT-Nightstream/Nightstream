import NightstreamFPrime.Circuit.Sequence
import NightstreamFPrime.Lifecycle.Stage1.Interface

/-!
Owns the first complete opaque-child order for the Stage 1 augmented circuit.

This file proves exact offsets, aggregate footprints, coverage, and soundness
for arbitrary satisfying assignments. It does not yet export the final
`Stage1.circuit`: that requires the base/recursive completeness proof and the
cross-phase relation theorem in `Soundness.lean`.

HyperNova's terminal checks are outside `F'`. They are therefore not an
operation in this circuit. `Stage1.Terminal` owns that outer verifier boundary.
-/

namespace NightstreamFPrime.Lifecycle.Stage1

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def priorChild
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Application.Program) (interface : Interface relation program) :
    FormalCircuit :=
  Pilot.priorCircuit interface.pilot

def outputHashChild
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Application.Program) (interface : Interface relation program) :
    FormalCircuit :=
  Pilot.outputCircuit interface.pilot

noncomputable def piCcsChild
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation)) : FormalCircuit :=
  PiCCS.v1_1.Formal.circuit relation ajtai interface.piCcs template

noncomputable def piRlcChild
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program) :
    FormalCircuit :=
  PiRLC.v1_1.Formal.circuit relation ajtai interface.piRlc

noncomputable def piDecChild
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program) :
    FormalCircuit :=
  PiDEC.v1_1.Formal.circuit relation ajtai interface.piDec

def runningChild
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Application.Program) (interface : Interface relation program) :
    FormalCircuit :=
  RunningTransition.circuit interface.running

def applicationChild
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Application.Program) (interface : Interface relation program) :
    FormalCircuit :=
  program.circuit interface.application

def priorOffset (offset : Nat) : Nat := offset

def outputHashOffset
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Application.Program) (interface : Interface relation program)
    (offset : Nat) : Nat :=
  priorOffset offset + (priorChild relation program interface).privateCount
    (priorOffset offset)

noncomputable def piCcsOffset
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Application.Program) (interface : Interface relation program)
    (offset : Nat) : Nat :=
  outputHashOffset relation program interface offset +
    (outputHashChild relation program interface).privateCount
      (outputHashOffset relation program interface offset)

noncomputable def piRlcOffset
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) : Nat :=
  piCcsOffset relation program interface offset +
    (piCcsChild relation ajtai program interface template).privateCount
      (piCcsOffset relation program interface offset)

noncomputable def piDecOffset
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) : Nat :=
  piRlcOffset relation ajtai program interface template offset +
    (piRlcChild relation ajtai program interface).privateCount
      (piRlcOffset relation ajtai program interface template offset)

noncomputable def runningOffset
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) : Nat :=
  piDecOffset relation ajtai program interface template offset +
    (piDecChild relation ajtai program interface).privateCount
      (piDecOffset relation ajtai program interface template offset)

noncomputable def applicationOffset
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) : Nat :=
  runningOffset relation ajtai program interface template offset +
    (runningChild relation program interface).privateCount
      (runningOffset relation ajtai program interface template offset)

noncomputable def finalOffset
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) : Nat :=
  applicationOffset relation ajtai program interface template offset +
    (applicationChild relation program interface).privateCount
      (applicationOffset relation ajtai program interface template offset)

def childOp (name : String) (child : FormalCircuit) (offset : Nat) : Op :=
  Sequence.childOp name child offset

@[simp] theorem childOp_privateCount (name : String) (child : FormalCircuit)
    (offset : Nat) :
    (childOp name child offset).localLength = child.privateCount offset := by
  rfl

@[simp] theorem childOp_rowCount (name : String) (child : FormalCircuit)
    (offset : Nat) :
    (childOp name child offset).rowCount = child.rowCount offset := by
  rfl

noncomputable def opsAt
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) : List Op :=
  [childOp "stage1.prior_state_hash" (priorChild relation program interface)
      (priorOffset offset),
    childOp "stage1.output_hash" (outputHashChild relation program interface)
      (outputHashOffset relation program interface offset),
    childOp "stage1.piccs.v1_1"
      (piCcsChild relation ajtai program interface template)
      (piCcsOffset relation program interface offset),
    childOp "stage1.pirlc.v1_1"
      (piRlcChild relation ajtai program interface)
      (piRlcOffset relation ajtai program interface template offset),
    childOp "stage1.pidec.v1_1"
      (piDecChild relation ajtai program interface)
      (piDecOffset relation ajtai program interface template offset),
    childOp "stage1.running_transition"
      (runningChild relation program interface)
      (runningOffset relation ajtai program interface template offset),
    childOp "stage1.application" (applicationChild relation program interface)
      (applicationOffset relation ajtai program interface template offset)]

noncomputable def main
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation)) : Circuit Unit :=
  fun offset =>
    ((), finalOffset relation ajtai program interface template offset,
      opsAt relation ajtai program interface template offset)

@[simp] theorem main_ops
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation)) (offset : Nat) :
    Circuit.ops (main relation ajtai program interface template) offset =
      opsAt relation ajtai program interface template offset := by
  rfl

noncomputable def logicalPrivateCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) : Nat :=
  (priorChild relation program interface).privateCount (priorOffset offset) +
  (outputHashChild relation program interface).privateCount
    (outputHashOffset relation program interface offset) +
  (piCcsChild relation ajtai program interface template).privateCount
    (piCcsOffset relation program interface offset) +
  (piRlcChild relation ajtai program interface).privateCount
    (piRlcOffset relation ajtai program interface template offset) +
  (piDecChild relation ajtai program interface).privateCount
    (piDecOffset relation ajtai program interface template offset) +
  (runningChild relation program interface).privateCount
    (runningOffset relation ajtai program interface template offset) +
  (applicationChild relation program interface).privateCount
    (applicationOffset relation ajtai program interface template offset)

noncomputable def logicalRowCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) : Nat :=
  (priorChild relation program interface).rowCount (priorOffset offset) +
  (outputHashChild relation program interface).rowCount
    (outputHashOffset relation program interface offset) +
  (piCcsChild relation ajtai program interface template).rowCount
    (piCcsOffset relation program interface offset) +
  (piRlcChild relation ajtai program interface).rowCount
    (piRlcOffset relation ajtai program interface template offset) +
  (piDecChild relation ajtai program interface).rowCount
    (piDecOffset relation ajtai program interface template offset) +
  (runningChild relation program interface).rowCount
    (runningOffset relation ajtai program interface template offset) +
  (applicationChild relation program interface).rowCount
    (applicationOffset relation ajtai program interface template offset)

theorem localLength_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation)) (offset : Nat) :
    localLength (Circuit.ops
      (main relation ajtai program interface template) offset) =
      logicalPrivateCount relation ajtai program interface template offset := by
  rw [main_ops]
  simp only [opsAt, localLength, List.map_cons, List.map_nil, List.sum_cons,
    List.sum_nil, childOp_privateCount, Nat.add_zero, logicalPrivateCount]
  omega

theorem flatConstraints_length_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation)) (offset : Nat) :
    (flatConstraints (Circuit.ops
      (main relation ajtai program interface template) offset)).length =
      logicalRowCount relation ajtai program interface template offset := by
  rw [flatConstraints_length_eq_rowCount]
  rw [main_ops]
  simp only [opsAt, rowCount, List.map_cons, List.map_nil, List.sum_cons,
    List.sum_nil, childOp_rowCount, Nat.add_zero, logicalRowCount]
  omega

structure Assumptions
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env) : Prop where
  prior : (priorChild relation program interface).assumptions
    (priorOffset offset) env
  outputHash : (outputHashChild relation program interface).assumptions
    (outputHashOffset relation program interface offset) env
  piCcs : (piCcsChild relation ajtai program interface template).assumptions
    (piCcsOffset relation program interface offset) env
  piRlc : (piRlcChild relation ajtai program interface).assumptions
    (piRlcOffset relation ajtai program interface template offset) env
  piDec : (piDecChild relation ajtai program interface).assumptions
    (piDecOffset relation ajtai program interface template offset) env
  running : (runningChild relation program interface).assumptions
    (runningOffset relation ajtai program interface template offset) env
  application : (applicationChild relation program interface).assumptions
    (applicationOffset relation ajtai program interface template offset) env

structure SpecHolds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env) : Prop where
  prior : (priorChild relation program interface).spec (priorOffset offset) env
  outputHash : (outputHashChild relation program interface).spec
    (outputHashOffset relation program interface offset) env
  piCcs : (piCcsChild relation ajtai program interface template).spec
    (piCcsOffset relation program interface offset) env
  piRlc : (piRlcChild relation ajtai program interface).spec
    (piRlcOffset relation ajtai program interface template offset) env
  piDec : (piDecChild relation ajtai program interface).spec
    (piDecOffset relation ajtai program interface template offset) env
  running : (runningChild relation program interface).spec
    (runningOffset relation ajtai program interface template offset) env
  application : (applicationChild relation program interface).spec
    (applicationOffset relation ajtai program interface template offset) env

private theorem childSpec_of_rows (name : String) (child : FormalCircuit)
    (childOffset : Nat) (env : Env) (operations : List Op)
    (rows : holds env operations)
    (member : childOp name child childOffset ∈ operations)
    (assumptions : child.assumptions childOffset env) :
    child.spec childOffset env := by
  exact (rows _ member) assumptions

/-- Arbitrary satisfying assignments imply every opaque Stage 1 child spec.
No child operation list is unfolded. -/
theorem soundness
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env)
    (assumptions : Assumptions relation ajtai program interface template
      offset env)
    (rows : holds env (Circuit.ops
      (main relation ajtai program interface template) offset)) :
    SpecHolds relation ajtai program interface template offset env := by
  rw [main_ops] at rows
  refine {
    prior := childSpec_of_rows "stage1.prior_state_hash" _ _ env _ rows
      (by simp [opsAt]) assumptions.prior
    outputHash := childSpec_of_rows "stage1.output_hash" _ _ env _ rows
      (by simp [opsAt]) assumptions.outputHash
    piCcs := childSpec_of_rows "stage1.piccs.v1_1" _ _ env _ rows
      (by simp [opsAt]) assumptions.piCcs
    piRlc := childSpec_of_rows "stage1.pirlc.v1_1" _ _ env _ rows
      (by simp [opsAt]) assumptions.piRlc
    piDec := childSpec_of_rows "stage1.pidec.v1_1" _ _ env _ rows
      (by simp [opsAt]) assumptions.piDec
    running := childSpec_of_rows "stage1.running_transition" _ _ env _ rows
      (by simp [opsAt]) assumptions.running
    application := childSpec_of_rows "stage1.application" _ _ env _ rows
      (by simp [opsAt]) assumptions.application }

/-- The parent has exactly seven opaque children, once each, in protocol
order. This is the mechanical coverage statement used by later layout proofs. -/
theorem opsAt_coverage
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Application.Program) (interface : Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation)) (offset : Nat) :
    (opsAt relation ajtai program interface template offset).length = 7 := by
  rfl

end NightstreamFPrime.Lifecycle.Stage1
