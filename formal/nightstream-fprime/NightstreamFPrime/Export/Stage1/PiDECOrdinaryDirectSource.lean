import NightstreamFPrime.Export.Stage1.PiDECDirectSupport
import NightstreamFPrime.Layout.ProductionRelation.OrdinarySourcePlan
import NightstreamFPrime.Layout.R1CS.Support
import NightstreamFPrime.Layout.Stage1.SpartanBounds

/-!
Owns indexed access to the four nonempty canonical PiDEC row packets for the
direct 14-matrix compiler. Each packet keeps its own proved expression support;
the exact R1CS append law fixes their combined order.
-/

namespace NightstreamFPrime.Export.Stage1.PiDECOrdinaryDirectSource

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.Stage1.PiDECSourceSupport
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiDEC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def publicConstraints (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  PiDECArithmetic.publicInputConstraints logicalWidth publicFits

def commitmentConstraints (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  PiDECArithmetic.commitmentConstraints logicalWidth publicFits

def evalKConstraints (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  PiDECArithmetic.evalKConstraints logicalWidth publicFits

def evalAConstraints (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List Expr :=
  PiDECArithmetic.evalAConstraints logicalWidth publicFits

def publicFreshStart : Nat := PiDECStarts.publicInputFreshStart

def commitmentFreshStart (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Nat :=
  publicFreshStart + R1CS.totalFreshCount
    (publicConstraints logicalWidth publicFits)

def evalKFreshStart (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Nat :=
  commitmentFreshStart logicalWidth publicFits + R1CS.totalFreshCount
    (commitmentConstraints logicalWidth publicFits)

def evalAFreshStart (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : Nat :=
  evalKFreshStart logicalWidth publicFits + R1CS.totalFreshCount
    (evalKConstraints logicalWidth publicFits)

theorem publicFreshStart_eq :
    publicFreshStart = PiDECStarts.phaseFreshStart := by
  rfl

theorem commitmentFreshStart_eq :
    ∀ (logicalWidth : Nat)
      (publicFits : ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth logicalWidth),
    commitmentFreshStart logicalWidth publicFits =
      PiDECStarts.phaseFreshStart + R1CS.totalFreshCount
        (publicConstraints logicalWidth publicFits) := by
  intro logicalWidth publicFits
  rw [commitmentFreshStart, publicFreshStart_eq]

theorem evalKFreshStart_eq :
    ∀ (logicalWidth : Nat)
      (publicFits : ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth logicalWidth),
    evalKFreshStart logicalWidth publicFits =
      PiDECStarts.phaseFreshStart + R1CS.totalFreshCount
          (publicConstraints logicalWidth publicFits) +
        R1CS.totalFreshCount
          (commitmentConstraints logicalWidth publicFits) := by
  intro logicalWidth publicFits
  rw [evalKFreshStart, commitmentFreshStart_eq]

theorem evalAFreshStart_eq :
    ∀ (logicalWidth : Nat)
      (publicFits : ringDegree * publicRingColumns ≤
        Phi81CarrierLayout.carrierWidth logicalWidth),
    evalAFreshStart logicalWidth publicFits =
      PiDECStarts.phaseFreshStart + R1CS.totalFreshCount
          (publicConstraints logicalWidth publicFits) +
        R1CS.totalFreshCount
          (commitmentConstraints logicalWidth publicFits) +
          R1CS.totalFreshCount
            (evalKConstraints logicalWidth publicFits) := by
  intro logicalWidth publicFits
  rw [evalAFreshStart, evalKFreshStart_eq]

def publicRows (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List R1CS.Row :=
  Spartan.remapRows
    (R1CS.lowerConstraints (publicConstraints logicalWidth publicFits)
      publicFreshStart).rows

def commitmentRows (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List R1CS.Row :=
  Spartan.remapRows
    (R1CS.lowerConstraints (commitmentConstraints logicalWidth publicFits)
      (commitmentFreshStart logicalWidth publicFits)).rows

def evalKRows (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List R1CS.Row :=
  Spartan.remapRows
    (R1CS.lowerConstraints (evalKConstraints logicalWidth publicFits)
      (evalKFreshStart logicalWidth publicFits)).rows

def evalARows (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List R1CS.Row :=
  Spartan.remapRows
    (R1CS.lowerConstraints (evalAConstraints logicalWidth publicFits)
      (evalAFreshStart logicalWidth publicFits)).rows

/-- Exact nonempty PiDEC parent order. -/
def sourceRows (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List R1CS.Row :=
  publicRows logicalWidth publicFits ++
    commitmentRows logicalWidth publicFits ++
    evalKRows logicalWidth publicFits ++ evalARows logicalWidth publicFits

theorem lowerFour_rows (first second third fourth : List Expr) (start : Nat) :
    (R1CS.lowerConstraints (first ++ second ++ third ++ fourth) start).rows =
      (R1CS.lowerConstraints first start).rows ++
        (R1CS.lowerConstraints second
          (start + R1CS.totalFreshCount first)).rows ++
        (R1CS.lowerConstraints third
          (start + R1CS.totalFreshCount first +
            R1CS.totalFreshCount second)).rows ++
        (R1CS.lowerConstraints fourth
          (start + R1CS.totalFreshCount first +
            R1CS.totalFreshCount second +
            R1CS.totalFreshCount third)).rows := by
  rw [R1CS.lowerConstraints_append_rows,
    R1CS.lowerConstraints_append_rows,
    R1CS.lowerConstraints_append_rows]
  simp only [R1CS.totalFreshCount_append, Nat.add_assoc]

section

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

private def phaseInterface (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    Formal.Interface logicalWidth publicFits :=
  PiDECArithmetic.phaseInterface logicalWidth publicFits

private theorem freshDeltas
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    [0, R1CS.totalFreshCount (publicConstraints logicalWidth publicFits),
      R1CS.totalFreshCount (commitmentConstraints logicalWidth publicFits),
      R1CS.totalFreshCount (evalKConstraints logicalWidth publicFits),
      R1CS.totalFreshCount (evalAConstraints logicalWidth publicFits), 0] =
        [0, 17820, 0, 0, 0, 0] := by
  simpa only [NightstreamFPrime.Layout.PiDEC.v1_1.physicalFreshDeltas,
    NightstreamFPrime.Layout.PiDEC.v1_1.childConstraintLists,
    publicConstraints, commitmentConstraints, evalKConstraints,
    evalAConstraints, PiDECArithmetic.publicInputConstraints,
    PiDECArithmetic.commitmentConstraints, PiDECArithmetic.evalKConstraints,
    PiDECArithmetic.evalAConstraints, PiDECArithmetic.phaseInterface,
    List.map_cons, List.map_nil] using
    (NightstreamFPrime.Layout.PiDEC.v1_1.physicalFreshDeltas_eq relation
      (phaseInterface logicalWidth publicFits) PiDECInputs.phaseOffset
      (PiDECInputs.inputShapes relation))

theorem publicFreshCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount (publicConstraints logicalWidth publicFits) = 17820 := by
  simpa using congrArg (fun values : List Nat => values.getD 1 0)
    (freshDeltas relation)

theorem commitmentFreshCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount (commitmentConstraints logicalWidth publicFits) = 0 := by
  simpa using congrArg (fun values : List Nat => values.getD 2 0)
    (freshDeltas relation)

theorem evalKFreshCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount (evalKConstraints logicalWidth publicFits) = 0 := by
  simpa using congrArg (fun values : List Nat => values.getD 3 0)
    (freshDeltas relation)

theorem evalAFreshCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalFreshCount (evalAConstraints logicalWidth publicFits) = 0 := by
  simpa using congrArg (fun values : List Nat => values.getD 4 0)
    (freshDeltas relation)

private theorem rowDeltas
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    [0, R1CS.totalRowCount (publicConstraints logicalWidth publicFits),
      R1CS.totalRowCount (commitmentConstraints logicalWidth publicFits),
      R1CS.totalRowCount (evalKConstraints logicalWidth publicFits),
      R1CS.totalRowCount (evalAConstraints logicalWidth publicFits), 0] =
        [0, 22680, 972, 108, 1512, 0] := by
  simpa only [NightstreamFPrime.Layout.PiDEC.v1_1.physicalRowDeltas,
    NightstreamFPrime.Layout.PiDEC.v1_1.childConstraintLists,
    publicConstraints, commitmentConstraints, evalKConstraints,
    evalAConstraints, PiDECArithmetic.publicInputConstraints,
    PiDECArithmetic.commitmentConstraints, PiDECArithmetic.evalKConstraints,
    PiDECArithmetic.evalAConstraints, PiDECArithmetic.phaseInterface,
    List.map_cons, List.map_nil] using
    (NightstreamFPrime.Layout.PiDEC.v1_1.physicalRowDeltas_eq relation
      (phaseInterface logicalWidth publicFits) PiDECInputs.phaseOffset
      (PiDECInputs.inputShapes relation))

theorem publicRowCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalRowCount (publicConstraints logicalWidth publicFits) = 22680 := by
  simpa using congrArg (fun values : List Nat => values.getD 1 0)
    (rowDeltas relation)

theorem commitmentRowCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalRowCount (commitmentConstraints logicalWidth publicFits) = 972 := by
  simpa using congrArg (fun values : List Nat => values.getD 2 0)
    (rowDeltas relation)

theorem evalKRowCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalRowCount (evalKConstraints logicalWidth publicFits) = 108 := by
  simpa using congrArg (fun values : List Nat => values.getD 3 0)
    (rowDeltas relation)

theorem evalARowCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    R1CS.totalRowCount (evalAConstraints logicalWidth publicFits) = 1512 := by
  simpa using congrArg (fun values : List Nat => values.getD 4 0)
    (rowDeltas relation)

theorem sourceRows_length
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (sourceRows logicalWidth publicFits).length = 25272 := by
  simp only [sourceRows, List.length_append, publicRows, commitmentRows,
    evalKRows, evalARows, Spartan.remapRows, List.length_map,
    R1CS.lowerConstraints_rows_length]
  rw [publicRowCount relation, commitmentRowCount relation,
    evalKRowCount relation, evalARowCount relation]

theorem constraints_eq_sources :
    PiDECArithmetic.constraints logicalWidth publicFits =
      publicConstraints logicalWidth publicFits ++
        commitmentConstraints logicalWidth publicFits ++
        evalKConstraints logicalWidth publicFits ++
        evalAConstraints logicalWidth publicFits := by
  rfl

/-- The four packets concatenate to the exact canonical compiled PiDEC rows. -/
theorem sourceRows_eq_canonical :
    sourceRows logicalWidth publicFits =
      (PiDECArithmetic.canonicalPlan logicalWidth publicFits).rows.map
        Rows.CompiledRow.toR1CS := by
  rw [PiDECArithmetic.Plan.rows_toR1CS]
  change sourceRows logicalWidth publicFits = Spartan.remapRows
    (R1CS.lowerConstraints
      (PiDECArithmetic.constraints logicalWidth publicFits)
      PiDECStarts.phaseFreshStart).rows
  rw [constraints_eq_sources, lowerFour_rows]
  simp only [sourceRows, publicRows, commitmentRows, evalKRows, evalARows,
    Spartan.remapRows, List.map_append]
  rw [publicFreshStart_eq, commitmentFreshStart_eq logicalWidth publicFits,
    evalKFreshStart_eq logicalWidth publicFits,
    evalAFreshStart_eq logicalWidth publicFits]

private theorem remappedRows_varsSatisfy (constraints : List Expr) (start : Nat)
    (scope : PiDECDirectSupport.ConstraintsSupported constraints)
    (freshSupported : ∀ column,
      InRange start (R1CS.totalFreshCount constraints) column → Source column) :
    ∀ row ∈ Spartan.remapRows
        (R1CS.lowerConstraints constraints start).rows,
      row.VarsSatisfy Target := by
  have lowered := R1CS.lowerConstraints_rows_varsSatisfy constraints start Source
    scope.get
  apply Spartan.remapRows_varsSatisfy Source Target _
  · intro row member
    apply (lowered row member).mono row
    intro column support
    rcases support with source | fresh
    · exact source
    · exact freshSupported column fresh
  · intro column support
    exact source_target column support

theorem publicRows_varsSatisfy
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ∀ row ∈ publicRows logicalWidth publicFits, row.VarsSatisfy Target := by
  apply remappedRows_varsSatisfy (publicConstraints logicalWidth publicFits)
    publicFreshStart
    (PiDECDirectSupport.productionPublicConstraints_supported
      (logicalWidth := logicalWidth) (publicFits := publicFits))
  intro column support
  apply fresh_source column
  rw [publicFreshCount relation] at support
  simpa [publicFreshStart, PiDECStarts.publicInputFreshStart,
    PiDECStarts.inputFreshStart, freshCount] using support

theorem commitmentRows_varsSatisfy
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ∀ row ∈ commitmentRows logicalWidth publicFits,
      row.VarsSatisfy Target := by
  apply remappedRows_varsSatisfy
    (commitmentConstraints logicalWidth publicFits)
    (commitmentFreshStart logicalWidth publicFits)
    (PiDECDirectSupport.productionCommitmentConstraints_supported
      (logicalWidth := logicalWidth) (publicFits := publicFits))
  intro column support
  rw [commitmentFreshCount relation] at support
  unfold InRange at support
  omega

theorem evalKRows_varsSatisfy
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ∀ row ∈ evalKRows logicalWidth publicFits, row.VarsSatisfy Target := by
  apply remappedRows_varsSatisfy (evalKConstraints logicalWidth publicFits)
    (evalKFreshStart logicalWidth publicFits)
    (PiDECDirectSupport.productionEvalKConstraints_supported
      (logicalWidth := logicalWidth) (publicFits := publicFits))
  intro column support
  rw [evalKFreshCount relation] at support
  unfold InRange at support
  omega

theorem evalARows_varsSatisfy
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ∀ row ∈ evalARows logicalWidth publicFits, row.VarsSatisfy Target := by
  apply remappedRows_varsSatisfy (evalAConstraints logicalWidth publicFits)
    (evalAFreshStart logicalWidth publicFits)
    (PiDECDirectSupport.productionEvalAConstraints_supported
      (logicalWidth := logicalWidth) (publicFits := publicFits))
  intro column support
  rw [evalAFreshCount relation] at support
  unfold InRange at support
  omega

theorem target_lt_spartanColumnCount {column : Nat} (support : Target column) :
    column < Spartan.spartanColumnCount := by
  rcases support with ⟨source, sourceSupport, rfl⟩
  exact Spartan.sourceToSpartan_lt source
    (PiDECSourceSupport.source_lt_sourceColumnCount sourceSupport)

private theorem varsBelow_of_target (rows : List R1CS.Row)
    (scope : ∀ row ∈ rows, row.VarsSatisfy Target) :
    ∀ row ∈ rows, row.VarsBelow Spartan.spartanColumnCount := by
  intro row member
  exact (scope row member).mono row
    (fun _ support => target_lt_spartanColumnCount support)

theorem publicRows_varsBelow
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ∀ row ∈ publicRows logicalWidth publicFits,
      row.VarsBelow Spartan.spartanColumnCount :=
  varsBelow_of_target (publicRows logicalWidth publicFits)
    (publicRows_varsSatisfy relation)

theorem commitmentRows_varsBelow
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ∀ row ∈ commitmentRows logicalWidth publicFits,
      row.VarsBelow Spartan.spartanColumnCount :=
  varsBelow_of_target (commitmentRows logicalWidth publicFits)
    (commitmentRows_varsSatisfy relation)

theorem evalKRows_varsBelow
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ∀ row ∈ evalKRows logicalWidth publicFits,
      row.VarsBelow Spartan.spartanColumnCount :=
  varsBelow_of_target (evalKRows logicalWidth publicFits)
    (evalKRows_varsSatisfy relation)

theorem evalARows_varsBelow
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ∀ row ∈ evalARows logicalWidth publicFits,
      row.VarsBelow Spartan.spartanColumnCount :=
  varsBelow_of_target (evalARows logicalWidth publicFits)
    (evalARows_varsSatisfy relation)

theorem publicRows_length
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (publicRows logicalWidth publicFits).length = 22680 := by
  simp only [publicRows, Spartan.remapRows, List.length_map,
    R1CS.lowerConstraints_rows_length]
  exact publicRowCount relation

theorem commitmentRows_length
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (commitmentRows logicalWidth publicFits).length = 972 := by
  simp only [commitmentRows, Spartan.remapRows, List.length_map,
    R1CS.lowerConstraints_rows_length]
  exact commitmentRowCount relation

theorem evalKRows_length
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (evalKRows logicalWidth publicFits).length = 108 := by
  simp only [evalKRows, Spartan.remapRows, List.length_map,
    R1CS.lowerConstraints_rows_length]
  exact evalKRowCount relation

theorem evalARows_length
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (evalARows logicalWidth publicFits).length = 1512 := by
  simp only [evalARows, Spartan.remapRows, List.length_map,
    R1CS.lowerConstraints_rows_length]
  exact evalARowCount relation

def publicListIndex
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 22680) : Fin (publicRows logicalWidth publicFits).length :=
  Fin.cast (publicRows_length relation).symm index

def commitmentListIndex
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 972) : Fin (commitmentRows logicalWidth publicFits).length :=
  Fin.cast (commitmentRows_length relation).symm index

def evalKListIndex
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 108) : Fin (evalKRows logicalWidth publicFits).length :=
  Fin.cast (evalKRows_length relation).symm index

def evalAListIndex
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 1512) : Fin (evalARows logicalWidth publicFits).length :=
  Fin.cast (evalARows_length relation).symm index

def publicProgramRow
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 22680) : R1CS.Row :=
  (publicRows logicalWidth publicFits).get (publicListIndex relation index)

def commitmentProgramRow
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 972) : R1CS.Row :=
  (commitmentRows logicalWidth publicFits).get
    (commitmentListIndex relation index)

def evalKProgramRow
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 108) : R1CS.Row :=
  (evalKRows logicalWidth publicFits).get (evalKListIndex relation index)

def evalAProgramRow
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 1512) : R1CS.Row :=
  (evalARows logicalWidth publicFits).get (evalAListIndex relation index)

private theorem ofFn_cast_get {Alpha : Type} (rows : List Alpha) {count : Nat}
    (lengthEq : rows.length = count) :
    List.ofFn (fun index : Fin count =>
      rows.get (Fin.cast lengthEq.symm index)) = rows := by
  subst count
  simpa using List.ofFn_get rows

theorem publicProgramRows_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List.ofFn (publicProgramRow relation) =
      publicRows logicalWidth publicFits := by
  unfold publicProgramRow publicListIndex
  exact ofFn_cast_get (publicRows logicalWidth publicFits)
    (publicRows_length relation)

theorem commitmentProgramRows_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List.ofFn (commitmentProgramRow relation) =
      commitmentRows logicalWidth publicFits := by
  unfold commitmentProgramRow commitmentListIndex
  exact ofFn_cast_get (commitmentRows logicalWidth publicFits)
    (commitmentRows_length relation)

theorem evalKProgramRows_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List.ofFn (evalKProgramRow relation) = evalKRows logicalWidth publicFits := by
  unfold evalKProgramRow evalKListIndex
  exact ofFn_cast_get (evalKRows logicalWidth publicFits)
    (evalKRows_length relation)

theorem evalAProgramRows_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List.ofFn (evalAProgramRow relation) = evalARows logicalWidth publicFits := by
  unfold evalAProgramRow evalAListIndex
  exact ofFn_cast_get (evalARows logicalWidth publicFits)
    (evalARows_length relation)

theorem publicProgramRow_varsSatisfy
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 22680) :
    (publicProgramRow relation index).VarsSatisfy Target :=
  publicRows_varsSatisfy relation _
    (List.get_mem _ (publicListIndex relation index))

theorem commitmentProgramRow_varsSatisfy
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 972) :
    (commitmentProgramRow relation index).VarsSatisfy Target :=
  commitmentRows_varsSatisfy relation _
    (List.get_mem _ (commitmentListIndex relation index))

theorem evalKProgramRow_varsSatisfy
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 108) :
    (evalKProgramRow relation index).VarsSatisfy Target :=
  evalKRows_varsSatisfy relation _
    (List.get_mem _ (evalKListIndex relation index))

theorem evalAProgramRow_varsSatisfy
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 1512) :
    (evalAProgramRow relation index).VarsSatisfy Target :=
  evalARows_varsSatisfy relation _
    (List.get_mem _ (evalAListIndex relation index))

theorem publicProgramRow_bounded
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 22680) :
    SourceCompiler.RowBounded Spartan.spartanColumnCount
      (publicProgramRow relation index) :=
  (publicProgramRow_varsSatisfy relation index).mono _
    (fun _ support => target_lt_spartanColumnCount support)

theorem commitmentProgramRow_bounded
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 972) :
    SourceCompiler.RowBounded Spartan.spartanColumnCount
      (commitmentProgramRow relation index) :=
  (commitmentProgramRow_varsSatisfy relation index).mono _
    (fun _ support => target_lt_spartanColumnCount support)

theorem evalKProgramRow_bounded
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 108) :
    SourceCompiler.RowBounded Spartan.spartanColumnCount
      (evalKProgramRow relation index) :=
  (evalKProgramRow_varsSatisfy relation index).mono _
    (fun _ support => target_lt_spartanColumnCount support)

theorem evalAProgramRow_bounded
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin 1512) :
    SourceCompiler.RowBounded Spartan.spartanColumnCount
      (evalAProgramRow relation index) :=
  (evalAProgramRow_varsSatisfy relation index).mono _
    (fun _ support => target_lt_spartanColumnCount support)

end

end NightstreamFPrime.Export.Stage1.PiDECOrdinaryDirectSource
