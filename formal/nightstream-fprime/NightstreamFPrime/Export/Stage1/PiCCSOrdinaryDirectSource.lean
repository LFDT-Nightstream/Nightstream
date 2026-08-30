import NightstreamFPrime.Export.Stage1.PackageCompleteness
import NightstreamFPrime.Layout.ProductionRelation.OrdinarySourcePlan
import NightstreamFPrime.Layout.Stage1.SpartanBounds

/-!
Owns indexed access to the exact canonical PiCCS ordinary rows for the direct
14-matrix compiler.

The row list remains the established Lean lowering. This module proves its
size, source-column bound, indexed satisfaction, and equality to the package
row order. It does not select retained low-norm coordinates.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSOrdinaryDirectSource

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Exact PiCCS source rows after the canonical Spartan column permutation. -/
def sourceRows (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List R1CS.Row :=
  (PiCCSArithmetic.arithmeticRows logicalWidth publicFits).map
    Rows.CompiledRow.toR1CS

theorem sourceRows_length
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (sourceRows logicalWidth publicFits).length = 811669 := by
  rw [sourceRows, List.length_map,
    PiCCSArithmetic.arithmeticRows_length logicalWidth publicFits relation]

/-- Every canonical source row is confined to the exact Spartan domain. -/
theorem sourceRows_varsBelow
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ∀ row ∈ sourceRows logicalWidth publicFits,
      row.VarsBelow Spartan.spartanColumnCount := by
  rw [sourceRows, PiCCSCompleteness.arithmeticRows_toR1CS_eq relation]
  apply Spartan.remapRows_varsBelow
  have loweredScope := R1CS.lowerConstraints_rows_varsBelow
    (PiCCSCompleteness.emittedConstraints logicalWidth publicFits)
    PiCCSArithmetic.initialClaimFreshStart
    (PackageCompleteness.piCcsEmittedConstraints_varsBelow relation
      (fun _ => 0))
  rw [PiCCSCompleteness.emittedConstraints_totalFreshCount relation] at loweredScope
  have endEq : PiCCSArithmetic.initialClaimFreshStart + 731605 =
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset := by
    unfold PiCCSArithmetic.initialClaimFreshStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.initialClaimFreshStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptFreshStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeFreshStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementAbsorptionFreshStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementBindingFreshStart
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase
    rw [NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
    norm_num [NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset]
  rw [endEq] at loweredScope
  intro row member
  exact R1CS.Row.VarsBelow.mono row (loweredScope row member) (by
    norm_num [NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset,
      Spartan.SourceColumnCount])

theorem sourceRows_rowCount_le
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (sourceRows logicalWidth publicFits).length ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [sourceRows_length relation]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

def sourceListIndex
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 811669) : Fin (sourceRows logicalWidth publicFits).length :=
  Fin.cast (sourceRows_length relation).symm index

def programRow
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 811669) : R1CS.Row :=
  (sourceRows logicalWidth publicFits).get (sourceListIndex relation index)

/-- Indexed PiCCS source rows remain below the exact Spartan source width.
This named theorem prevents executable consumers from unfolding the complete
supported-program certificate to recover one row bound. -/
theorem programRow_bounded
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 811669) :
    (programRow relation index).VarsBelow Spartan.spartanColumnCount := by
  exact sourceRows_varsBelow relation _
    (List.get_mem _ (sourceListIndex relation index))

private theorem ofFn_cast_get {Alpha : Type} (rows : List Alpha) {count : Nat}
    (lengthEq : rows.length = count) :
    List.ofFn (fun index : Fin count =>
      rows.get (Fin.cast lengthEq.symm index)) = rows := by
  subst count
  simpa using List.ofFn_get rows

theorem programRows_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List.ofFn (programRow relation) = sourceRows logicalWidth publicFits := by
  unfold programRow sourceListIndex
  exact ofFn_cast_get _ (sourceRows_length relation)

theorem programRow_eq_of_same_shape
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : ProductionKey.LogicalRelation logicalWidth publicFits) :
    programRow left = programRow right := by
  apply List.ofFn_injective
  rw [programRows_eq left, programRows_eq right]

structure SupportedProgram (rows : List R1CS.Row) where
  rowCount : Nat
  rowCount_le : rowCount ≤ 2 ^ NightstreamFPrime.Lifecycle.cubeVariables
  row : Fin rowCount → R1CS.Row
  exactRows : List.ofFn row = rows
  bounded : ∀ index, (row index).VarsBelow Spartan.spartanColumnCount

def SupportedProgram.toProgram {rows : List R1CS.Row}
    (source : SupportedProgram rows) :
    OrdinarySourcePlan.Program Spartan.spartanColumnCount where
  rowCount := source.rowCount
  rowCount_le := source.rowCount_le
  row := source.row
  bounded := source.bounded

def supportedProgram
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    SupportedProgram (sourceRows logicalWidth publicFits) where
  rowCount := 811669
  rowCount_le := by norm_num [NightstreamFPrime.Lifecycle.cubeVariables]
  row := programRow relation
  exactRows := programRows_eq relation
  bounded := programRow_bounded relation

/-- Proof-oriented indexed access without an artifact-sized theorem term. -/
def program
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    OrdinarySourcePlan.Program Spartan.spartanColumnCount :=
  (supportedProgram relation).toProgram

@[simp] theorem program_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (program relation).rowCount = 811669 := by
  rfl

/-- Program construction depends on the relation shape, not its matrix
entries. The relation value is used only to supply proof certificates for the
fixed row layout. -/
theorem program_eq_of_same_shape
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : ProductionKey.LogicalRelation logicalWidth publicFits) :
    program left = program right := by
  rfl

private theorem holds_iff_rowsHold_ofFn {count : Nat}
    (rowAt : Fin count → R1CS.Row) (env : Env) :
    (∀ index, (rowAt index).Holds env) ↔
      R1CS.RowsHold env (List.ofFn rowAt) := by
  unfold R1CS.RowsHold
  exact List.forall_mem_ofFn_iff.symm

private theorem predicate_iff_of_eq {Alpha : Type} (predicate : Alpha → Prop)
    {left right : Alpha} (equal : left = right) :
    predicate left ↔ predicate right := by
  cases equal
  rfl

/-- Indexed canonical PiCCS rows hold exactly when the complete Lean-lowered
row list holds in package order. -/
theorem programRows_hold_iff_rowsHold
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    (∀ index : Fin 811669, (programRow relation index).Holds env) ↔
      R1CS.RowsHold env (sourceRows logicalWidth publicFits) := by
  exact (holds_iff_rowsHold_ofFn (programRow relation) env).trans
    (predicate_iff_of_eq (R1CS.RowsHold env) (programRows_eq relation))

private theorem supportedHolds_iff_rowsHold {rows : List R1CS.Row}
    (source : SupportedProgram rows) (env : Env) :
    source.toProgram.Holds env ↔ R1CS.RowsHold env rows := by
  exact (holds_iff_rowsHold_ofFn source.row env).trans
    (predicate_iff_of_eq (R1CS.RowsHold env) source.exactRows)

/-- The indexed program is exactly list-based satisfaction in package order. -/
theorem program_holds_iff_rowsHold
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    (program relation).Holds env ↔
      R1CS.RowsHold env (sourceRows logicalWidth publicFits) := by
  exact supportedHolds_iff_rowsHold (supportedProgram relation) env

/-- The source program uses the exact canonical compiled PiCCS row list. -/
theorem sourceRows_eq_canonical
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    sourceRows logicalWidth publicFits =
      (PiCCSArithmetic.arithmeticRows logicalWidth publicFits).map
        Rows.CompiledRow.toR1CS := by
  rfl

end NightstreamFPrime.Export.Stage1.PiCCSOrdinaryDirectSource
