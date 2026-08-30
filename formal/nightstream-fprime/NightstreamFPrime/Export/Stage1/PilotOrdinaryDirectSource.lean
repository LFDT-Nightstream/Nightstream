import NightstreamFPrime.Export.Pilot
import NightstreamFPrime.Gadgets.Range.CanonicalPublicU64
import NightstreamFPrime.Layout.ProductionRelation.OrdinarySourcePlan
import NightstreamFPrime.Layout.R1CS.Support

/-!
Owns indexed access and exact source support for the 1,330 non-Poseidon pilot
rows. The source rows remain the canonical pilot witness-instruction rows
followed by the canonical assertion rows.
-/

namespace NightstreamFPrime.Export.Stage1.PilotOrdinaryDirectSource

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export.Package

def instructionRows : List R1CS.Row :=
  (PilotData.witnessInstructions ()).map fun instruction =>
    Rows.CompiledRow.toR1CS (.witness instruction)

def assertionRows : List R1CS.Row :=
  (PilotData.assertionRows ()).map fun assertion =>
    Rows.CompiledRow.toR1CS (.assertion assertion)

/-- Exact canonical package order for all non-Poseidon pilot rows. -/
def sourceRows : List R1CS.Row := instructionRows ++ assertionRows

theorem sourceRows_length : sourceRows.length = 1330 := by
  unfold sourceRows instructionRows assertionRows
  simp only [List.length_append, List.length_map]
  rw [PilotData.witnessInstructions, PilotData.assertionRows,
    List.length_append, Rows.witnessInstructionsTR_eq,
    Rows.assertionRowsTR_eq, Pilot.digestRows_length]
  have partition := Rows.witnessInstructions_length_add_assertionRows_length
    (PilotData.priorExtraRows ())
  rw [Pilot.priorExtraRows_length] at partition
  omega

private theorem priorDigestStart_eq :
    PilotProduction.priorDigestStart = 6891852 := by
  unfold PilotProduction.priorDigestStart
  rw [PilotProduction.witnessOffset_eq, PilotProduction.absorbCount_eq]
  norm_num [PilotProduction.permutationRecipeCount,
    PilotValues.permutationRecipeCount,
    PilotValues.permutationOutputLocalStart]

private theorem priorHashEnd_eq :
    PriorStateHash.hashEnd PilotProduction.priorInterface
      PilotProduction.witnessOffset = 6891860 := by
  unfold PriorStateHash.hashEnd
  rw [PilotProduction.priorHashLogicalLength_eq,
    PilotProduction.witnessOffset_eq]

private theorem priorPublicInputStart_eq :
    PilotProduction.priorPublicInputStart = 45937 := by rfl

private theorem outputStateStart_eq :
    PilotProduction.lifecycleOutputOffset +
      PilotValues.absorbCount * 592 + 584 = 13691828 := by
  rw [PilotProduction.lifecycleOutputOffset_eq]
  norm_num [PilotValues.absorbCount, PilotValues.stateHashWords,
    PilotValues.stateHashBaseWords, Spec.Poseidon2.rate]

private theorem outputDigestStart_eq :
    PilotProduction.outputDigestStart = 92144 := by rfl

private theorem sourceColumnCount_eq :
    PilotValues.sourceColumnCount = 13692624 := by rfl

private theorem outputChainAbsorbCount_eq :
    PilotData.outputChain.absorbCount = 11485 := by rfl

private theorem outputChainWitnessStart_eq :
    PilotData.outputChain.witnessStart = 6891850 := by rfl

private theorem outputChainDigestStart_eq :
    PilotData.outputChain.digestStart = 13692621 := by rfl

private theorem outputStateSource_eq (lane : Fin 4) :
    PilotProduction.lifecycleOutputOffset +
        PilotData.outputChain.absorbCount * 592 + 584 + lane.val =
      13691828 + lane.val := by
  rw [PilotProduction.lifecycleOutputOffset_eq,
    outputChainAbsorbCount_eq]

private theorem outputStateTarget_eq (lane : Fin 4) :
    PilotData.outputChain.witnessStart +
        PilotData.outputChain.absorbCount * 592 + 584 + lane.val =
      13691554 + lane.val := by
  rw [outputChainWitnessStart_eq, outputChainAbsorbCount_eq]

theorem priorDigest_targetColumn (lane : Fin 4) :
    PilotSpartan.sourceToSpartan
        (PilotProduction.priorDigestStart + lane.val) =
      6891578 + lane.val := by
  have sourceEq : PilotProduction.priorDigestStart + lane.val =
      PilotProduction.witnessOffset +
        (PilotProduction.absorbCount * 592 + 584 + lane.val) := by
    unfold PilotProduction.priorDigestStart
    norm_num [PilotProduction.permutationRecipeCount,
      PilotValues.permutationRecipeCount,
      PilotValues.permutationOutputLocalStart]
    omega
  rw [sourceEq, PilotSpartan.sourceToSpartan_pilotWitness]
  rw [PilotProduction.absorbCount_eq]
  norm_num [PilotSpartan.witnessPrivateStart_value]
  omega

theorem outputState_targetColumn (lane : Fin 4) :
    PilotSpartan.sourceToSpartan
        (PilotProduction.lifecycleOutputOffset +
          PilotValues.absorbCount * 592 + 584 + lane.val) =
      13691554 + lane.val := by
  have sourceEq : PilotProduction.lifecycleOutputOffset +
      PilotValues.absorbCount * 592 + 584 + lane.val =
      PilotProduction.witnessOffset +
        (PilotValues.hashWitnessCount + PilotValues.priorCanonicalPrivateCount +
          PilotValues.absorbCount * 592 + 584 + lane.val) := by
    rw [PilotProduction.lifecycleOutputOffset_eq,
      PilotProduction.witnessOffset_eq]
    norm_num [PilotValues.absorbCount, PilotValues.hashWitnessCount,
      PilotValues.permutationRecipeCount, PilotValues.stateHashWords,
      PilotValues.stateHashBaseWords, PilotValues.priorCanonicalPrivateCount,
      Spec.Poseidon2.rate]
    omega
  rw [sourceEq, PilotSpartan.sourceToSpartan_pilotWitness]
  have hashCount : PilotValues.hashWitnessCount = 6799712 := by rfl
  have privateCount : PilotValues.priorCanonicalPrivateCount = 264 := by rfl
  have absorbCount : PilotValues.absorbCount = 11485 := by rfl
  rw [PilotSpartan.witnessPrivateStart_value, hashCount, privateCount,
    absorbCount]
  change 91874 + (13599680 + lane.val) = 13691554 + lane.val
  rw [← Nat.add_assoc]

def InRange (start count column : Nat) : Prop :=
  start ≤ column ∧ column < start + count

/-- Exact logical sources read before R1CS lowering. -/
inductive LogicalSource (column : Nat) : Prop where
  | priorDigest : InRange PilotProduction.priorDigestStart 4 column →
      LogicalSource column
  | priorPublic : InRange PilotProduction.priorPublicInputStart 270 column →
      LogicalSource column
  | canonicalLocal : InRange
      (PriorStateHash.hashEnd PilotProduction.priorInterface
        PilotProduction.witnessOffset) 264 column → LogicalSource column
  | outputState : InRange
      (PilotProduction.lifecycleOutputOffset +
        PilotValues.absorbCount * 592 + 584) 4 column → LogicalSource column
  | outputDigest : InRange PilotProduction.outputDigestStart 4 column →
      LogicalSource column

/-- Logical sources plus the exact 788 R1CS-fresh interval. -/
def PhysicalSource : Nat → Prop :=
  R1CS.SourceOrFresh LogicalSource PilotValues.logicalColumnCount
    PilotValues.sourceColumnCount

/-- Source support after the canonical pilot Spartan permutation. -/
def Target (column : Nat) : Prop :=
  ∃ source, PhysicalSource source ∧
    PilotSpartan.sourceToSpartan source = column

private theorem priorDigest_source (word : Fin 4) :
    (PilotProduction.fastPriorDigest word).VarsSatisfy LogicalSource := by
  apply LogicalSource.priorDigest
  unfold InRange
  have bound := word.isLt
  omega

private theorem priorPublic_source (word : Fin 4) (bit : Nat)
    (_bitBound : bit < CanonicalPublicU64.bitCount) :
    (PilotProduction.fastPriorWordInterface word).bit
        (PriorStateHash.wordOffset PilotProduction.priorInterface
          PilotProduction.witnessOffset word) bit |>.VarsSatisfy
      LogicalSource := by
  apply LogicalSource.priorPublic
  unfold InRange
  constructor
  · omega
  · have bounded := (PriorStateHash.digestBitIndexNat word bit).isLt
    change (PriorStateHash.digestBitIndexNat word bit).val < 270 at bounded
    omega

private theorem canonicalLocal_source (word : Fin 4) (index : Nat)
    (indexBound : index < CanonicalPublicU64.privateCount) :
    LogicalSource
      (PriorStateHash.wordOffset PilotProduction.priorInterface
        PilotProduction.witnessOffset word + index) := by
  apply LogicalSource.canonicalLocal
  unfold InRange PriorStateHash.wordOffset
  have wordBound := word.isLt
  norm_num [NightstreamFPrime.Gadgets.Range.CanonicalPublicU64.privateCount,
    CanonicalU64.auxiliaryCount] at indexBound ⊢
  omega

private theorem wordConstraints_varsSatisfy (word : Fin 4) :
    ∀ expression ∈ PilotProduction.priorWordConstraints word,
      expression.VarsSatisfy LogicalSource := by
  rw [PilotProduction.priorWordConstraints_eq]
  unfold Layout.Range.CanonicalPublicU64.logicalConstraints
  rw [← PilotProduction.fastPriorWordInterface_eq word]
  exact NightstreamFPrime.Gadgets.Range.CanonicalPublicU64.flatConstraints_varsSatisfy
    (PilotProduction.fastPriorWordInterface word)
    (PriorStateHash.wordOffset PilotProduction.priorInterface
      PilotProduction.witnessOffset word)
    LogicalSource (priorDigest_source word)
    (priorPublic_source word) (canonicalLocal_source word)

private theorem wordOps_eq :
    PriorStateHash.wordOps PilotProduction.priorInterface
        PilotProduction.witnessOffset =
      [PriorStateHash.wordOp PilotProduction.priorInterface
          PilotProduction.witnessOffset 0,
       PriorStateHash.wordOp PilotProduction.priorInterface
          PilotProduction.witnessOffset 1,
       PriorStateHash.wordOp PilotProduction.priorInterface
          PilotProduction.witnessOffset 2,
       PriorStateHash.wordOp PilotProduction.priorInterface
          PilotProduction.witnessOffset 3] := by
  simp [PriorStateHash.wordOps, List.ofFn_succ]

private theorem priorWordConstraintsAll_varsSatisfy :
    ∀ expression ∈ PilotProduction.priorWordConstraintsAll (),
      expression.VarsSatisfy LogicalSource := by
  intro expression member
  unfold PilotProduction.priorWordConstraintsAll at member
  rw [wordOps_eq] at member
  simp only [flatConstraints, List.mem_flatMap, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with ⟨operation, operationMember, constraintMember⟩
  rcases operationMember with rfl | rfl | rfl | rfl
  · exact wordConstraints_varsSatisfy 0 expression constraintMember
  · exact wordConstraints_varsSatisfy 1 expression constraintMember
  · exact wordConstraints_varsSatisfy 2 expression constraintMember
  · exact wordConstraints_varsSatisfy 3 expression constraintMember

private theorem priorBindingConstraints_varsSatisfy :
    ∀ expression ∈ PilotProduction.priorBindingConstraints,
      expression.VarsSatisfy LogicalSource := by
  intro expression member
  unfold PilotProduction.priorBindingConstraints at member
  simp only [flatConstraints, List.mem_flatMap] at member
  rcases member with ⟨operation, operationMember, constraintMember⟩
  simp only [PriorStateHash.bindingAssertions, List.mem_cons] at operationMember
  rcases operationMember with rfl | operationMember
  · simp only [Op.flatConstraints, List.mem_singleton] at constraintMember
    subst expression
    apply Expr.VarsSatisfy.sub
    · rw [PilotProduction.priorInterface_publicInput_apply]
      apply LogicalSource.priorPublic
      unfold InRange
      norm_num [PriorStateHash.markerIndex]
    · trivial
  · rw [List.mem_ofFn'] at operationMember
    rcases operationMember with ⟨lane, rfl⟩
    simp only [Op.flatConstraints, List.mem_singleton] at constraintMember
    subst expression
    rw [PilotProduction.priorInterface_publicInput_apply]
    apply LogicalSource.priorPublic
    unfold InRange
    have laneBound := lane.isLt
    norm_num [PriorStateHash.tailIndex] at laneBound ⊢
    omega

theorem priorExtraConstraints_varsSatisfy :
    ∀ expression ∈ PilotData.priorExtraConstraints (),
      expression.VarsSatisfy LogicalSource := by
  rw [PilotData.priorExtraConstraints_eq]
  intro expression member
  rcases List.mem_append.mp member with word | binding
  · exact priorWordConstraintsAll_varsSatisfy expression word
  · exact priorBindingConstraints_varsSatisfy expression binding

private theorem priorLoweredRows_varsSatisfy :
    ∀ row ∈ (R1CS.lowerConstraints (PilotData.priorExtraConstraints ())
        PilotValues.logicalColumnCount).rows,
      row.VarsSatisfy PhysicalSource := by
  have freshCount :
      R1CS.totalFreshCount (PilotData.priorExtraConstraints ()) = 788 := by
    rw [PilotData.priorExtraConstraints_eq,
      R1CS.totalFreshCount_append,
      PilotProduction.priorWordConstraints_freshCount,
      PilotProduction.priorBindingConstraints_freshCount]
  have rows := R1CS.lowerConstraints_rows_varsSatisfy
    (PilotData.priorExtraConstraints ()) PilotValues.logicalColumnCount
    LogicalSource priorExtraConstraints_varsSatisfy
  rw [freshCount] at rows
  simpa only [PhysicalSource] using rows

private theorem remapCombination_varsSatisfy
    (combination : R1CS.LinearCombination)
    (scope : combination.VarsSatisfy PhysicalSource) :
    (PilotSpartan.remapCombination combination).VarsSatisfy Target := by
  intro term member
  unfold PilotSpartan.remapCombination at member
  rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
  exact ⟨source.1, scope source sourceMember, rfl⟩

private theorem remapRow_varsSatisfy (row : R1CS.Row)
    (scope : row.VarsSatisfy PhysicalSource) :
    (PilotSpartan.remapRow row).VarsSatisfy Target :=
  ⟨remapCombination_varsSatisfy row.a scope.1,
    remapCombination_varsSatisfy row.b scope.2.1,
    remapCombination_varsSatisfy row.c scope.2.2⟩

private theorem priorRows_eq :
    (PilotData.priorExtraRows ()).map Rows.CompiledRow.toR1CS =
      PilotSpartan.remapRows
        (R1CS.lowerConstraints (PilotData.priorExtraConstraints ())
          PilotValues.logicalColumnCount).rows := by
  rw [PilotData.priorExtraRows, PilotData.remapCompiledRows_toR1CS,
    Rows.compileRowsTR_toR1CS, Rows.lowerConstraintsTR_eq]

private theorem priorRows_varsSatisfy :
    ∀ row ∈ (PilotData.priorExtraRows ()).map Rows.CompiledRow.toR1CS,
      row.VarsSatisfy Target := by
  rw [priorRows_eq]
  intro row member
  unfold PilotSpartan.remapRows at member
  rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
  exact remapRow_varsSatisfy source
    (priorLoweredRows_varsSatisfy source sourceMember)

private theorem outputDigestRows_varsSatisfy :
    ∀ row ∈ (PilotData.digestRows PilotData.outputChain).map
        (fun assertion => Rows.CompiledRow.toR1CS (.assertion assertion)),
      row.VarsSatisfy Target := by
  intro row member
  rw [List.mem_map] at member
  rcases member with ⟨source, sourceMember, rfl⟩
  rw [PilotData.digestRows, List.mem_ofFn'] at sourceMember
  rcases sourceMember with ⟨lane, rfl⟩
  have outputStateMapped :
      PilotSpartan.sourceToSpartan
          (PilotProduction.lifecycleOutputOffset +
            PilotData.outputChain.absorbCount * 592 + 584 + lane.val) =
        PilotData.outputChain.witnessStart +
          PilotData.outputChain.absorbCount * 592 + 584 + lane.val := by
    rw [outputStateSource_eq lane, outputStateTarget_eq lane]
    unfold PilotSpartan.sourceToSpartan
    rw [if_neg (by rw [PilotSpartan.priorPublicStart_value]; omega)]
    rw [if_neg (by rw [PilotSpartan.outputPreimageStart_value]; omega)]
    rw [if_neg (by rw [PilotSpartan.outputDigestStart_value]; omega)]
    rw [if_neg (by rw [PilotSpartan.witnessStart_value]; omega)]
    rw [PilotSpartan.witnessStart_value,
      PilotSpartan.witnessPrivateStart_value]
    omega
  have outputDigestMapped :
      PilotSpartan.sourceToSpartan
          (PilotProduction.outputDigestStart + lane.val) =
        PilotData.outputChain.digestStart + lane.val := by
    rw [outputDigestStart_eq, outputChainDigestStart_eq]
    have laneBound := lane.isLt
    unfold PilotSpartan.sourceToSpartan
    rw [if_neg (by rw [PilotSpartan.priorPublicStart_value]; omega)]
    rw [if_neg (by rw [PilotSpartan.outputPreimageStart_value]; omega)]
    rw [if_neg (by rw [PilotSpartan.outputDigestStart_value]; omega)]
    rw [if_pos (by rw [PilotSpartan.witnessStart_value]; omega)]
    rw [PilotSpartan.outputDigestStart_value,
      PilotSpartan.secondPublicStart_value]
    omega
  constructor
  · intro term member
    simp only [Rows.CompiledRow.toR1CS, SparseRow.toR1CS,
      PilotData.digestRow, PilotData.zeroCombination,
      PilotData.oneCombination, SparseCombination.toR1CS,
      List.mem_map, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with ⟨sourceTerm, sourceMember, rfl⟩
    rcases sourceMember with rfl | rfl
    · refine ⟨PilotProduction.lifecycleOutputOffset +
          PilotData.outputChain.absorbCount * 592 + 584 + lane.val,
        ?_, ?_⟩
      · apply Or.inl
        apply LogicalSource.outputState
        rw [outputStateStart_eq, outputStateSource_eq lane]
        unfold InRange
        have laneBound := lane.isLt
        omega
      · exact outputStateMapped
    · refine ⟨PilotProduction.outputDigestStart + lane.val, ?_, ?_⟩
      · apply Or.inl
        apply LogicalSource.outputDigest
        unfold InRange
        have laneBound := lane.isLt
        omega
      · exact outputDigestMapped
  · constructor
    · intro term member
      simp [Rows.CompiledRow.toR1CS, SparseRow.toR1CS,
        PilotData.digestRow, PilotData.oneCombination,
        SparseCombination.toR1CS] at member
    · intro term member
      simp [Rows.CompiledRow.toR1CS, SparseRow.toR1CS,
        PilotData.digestRow, PilotData.zeroCombination,
        SparseCombination.toR1CS] at member

theorem sourceRows_varsSatisfy :
    ∀ row ∈ sourceRows, row.VarsSatisfy Target := by
  intro row member
  unfold sourceRows instructionRows assertionRows at member
  rcases List.mem_append.mp member with instruction | assertion
  · rw [List.mem_map] at instruction
    rcases instruction with ⟨source, sourceMember, rfl⟩
    exact priorRows_varsSatisfy
      (Rows.CompiledRow.toR1CS (.witness source)) (by
        rw [List.mem_map]
        exact ⟨.witness source, by
          exact (Rows.witnessInstructions_member
            (PilotData.priorExtraRows ()) source).mp (by
              simpa [PilotData.witnessInstructions,
                Rows.witnessInstructionsTR_eq] using sourceMember),
          rfl⟩)
  · rw [List.mem_map] at assertion
    rcases assertion with ⟨source, sourceMember, rfl⟩
    rw [PilotData.assertionRows, List.mem_append] at sourceMember
    rcases sourceMember with prior | digest
    · exact priorRows_varsSatisfy
        (Rows.CompiledRow.toR1CS (.assertion source)) (by
          rw [List.mem_map]
          exact ⟨.assertion source,
            (Rows.assertionRows_member
              (PilotData.priorExtraRows ()) source).mp (by
                simpa [Rows.assertionRowsTR_eq] using prior), rfl⟩)
    · exact outputDigestRows_varsSatisfy
        (Rows.CompiledRow.toR1CS (.assertion source)) (by
          rw [List.mem_map]
          exact ⟨source, digest, rfl⟩)

theorem physicalSource_lt (source : Nat)
    (support : PhysicalSource source) : source < PilotValues.sourceColumnCount := by
  rcases support with logical | fresh
  · rcases logical with digest | publicSource | localSource | outputState |
      outputDigest
    · rw [priorDigestStart_eq] at digest
      rw [sourceColumnCount_eq]
      unfold InRange at digest
      omega
    · rw [priorPublicInputStart_eq] at publicSource
      rw [sourceColumnCount_eq]
      unfold InRange at publicSource
      omega
    · rw [priorHashEnd_eq] at localSource
      rw [sourceColumnCount_eq]
      unfold InRange at localSource
      omega
    · rw [outputStateStart_eq] at outputState
      rw [sourceColumnCount_eq]
      unfold InRange at outputState
      omega
    · rw [outputDigestStart_eq] at outputDigest
      rw [sourceColumnCount_eq]
      unfold InRange at outputDigest
      omega
  · rw [sourceColumnCount_eq] at fresh ⊢
    exact fresh.2

theorem target_lt (column : Nat) (support : Target column) :
    column < PilotSpartan.spartanColumnCount := by
  rcases support with ⟨source, sourceSupport, rfl⟩
  apply PilotSpartan.sourceToSpartan_lt source
  simpa [PilotSpartan.SourceColumnCount] using
    physicalSource_lt source sourceSupport

def programRow (index : Fin 1330) : R1CS.Row :=
  sourceRows.get (Fin.cast sourceRows_length.symm index)

private theorem ofFn_cast_get {Alpha : Type} (rows : List Alpha) {count : Nat}
    (lengthEq : rows.length = count) :
    List.ofFn (fun index : Fin count =>
      rows.get (Fin.cast lengthEq.symm index)) = rows := by
  subst count
  simpa using List.ofFn_get rows

theorem programRows_eq : List.ofFn programRow = sourceRows := by
  unfold programRow
  exact ofFn_cast_get sourceRows sourceRows_length

theorem sourceRows_varsBelow :
    ∀ row ∈ sourceRows, row.VarsBelow PilotSpartan.spartanColumnCount := by
  intro row member
  exact (sourceRows_varsSatisfy row member).mono row target_lt

structure SupportedProgram (rows : List R1CS.Row) where
  rowCount : Nat
  rowCount_le : rowCount ≤ 2 ^ Lifecycle.cubeVariables
  row : Fin rowCount → R1CS.Row
  exactRows : List.ofFn row = rows
  bounded : ∀ index, (row index).VarsBelow PilotSpartan.spartanColumnCount

def SupportedProgram.toProgram {rows : List R1CS.Row}
    (source : SupportedProgram rows) :
    OrdinarySourcePlan.Program PilotSpartan.spartanColumnCount where
  rowCount := source.rowCount
  rowCount_le := source.rowCount_le
  row := source.row
  bounded := source.bounded

/-- Stable row-boundedness interface for executable package decoders. -/
theorem programRow_bounded (index : Fin 1330) :
    SourceCompiler.RowBounded PilotSpartan.spartanColumnCount
      (programRow index) := by
  exact sourceRows_varsBelow _
    (List.get_mem _ (Fin.cast sourceRows_length.symm index))

def supportedProgram : SupportedProgram sourceRows where
  rowCount := 1330
  rowCount_le := by norm_num [Lifecycle.cubeVariables]
  row := programRow
  exactRows := programRows_eq
  bounded := programRow_bounded

def program : OrdinarySourcePlan.Program PilotSpartan.spartanColumnCount :=
  supportedProgram.toProgram

@[simp] theorem program_rowCount : program.rowCount = 1330 := by rfl

private theorem holds_iff_rowsHold_ofFn {count : Nat}
    (rowAt : Fin count → R1CS.Row) (env : Env) :
    (∀ index, (rowAt index).Holds env) ↔ R1CS.RowsHold env (List.ofFn rowAt) := by
  unfold R1CS.RowsHold
  exact List.forall_mem_ofFn_iff.symm

private theorem predicate_iff_of_eq {Alpha : Type} (predicate : Alpha → Prop)
    {left right : Alpha} (equal : left = right) :
    predicate left ↔ predicate right := by
  cases equal
  rfl

/-- Indexed canonical pilot rows hold exactly when the complete Lean-lowered
row list holds in package order. -/
theorem programRows_hold_iff_rowsHold (env : Env) :
    (∀ index : Fin 1330, (programRow index).Holds env) ↔
      R1CS.RowsHold env sourceRows := by
  exact (holds_iff_rowsHold_ofFn programRow env).trans
    (predicate_iff_of_eq (R1CS.RowsHold env) programRows_eq)

private theorem supportedHolds_iff_rowsHold {rows : List R1CS.Row}
    (source : SupportedProgram rows) (env : Env) :
    source.toProgram.Holds env ↔ R1CS.RowsHold env rows := by
  exact (holds_iff_rowsHold_ofFn source.row env).trans
    (predicate_iff_of_eq (R1CS.RowsHold env) source.exactRows)

theorem program_holds_iff_rowsHold (env : Env) :
    program.Holds env ↔ R1CS.RowsHold env sourceRows := by
  exact supportedHolds_iff_rowsHold supportedProgram env

/-- Exact combined-row satisfaction recovers every original pilot ordinary
instruction and assertion predicate. -/
theorem sourceRows_hold_implies_packagePredicates (env : Env)
    (holds : R1CS.RowsHold env sourceRows) :
    (∀ instruction ∈ PilotData.witnessInstructions (),
        instruction.Holds env) ∧
      ∀ assertion ∈ PilotData.assertionRows (), assertion.Holds env := by
  have separated : R1CS.RowsHold env instructionRows ∧
      R1CS.RowsHold env assertionRows := by
    exact (R1CS.rowsHold_append env instructionRows assertionRows).mp holds
  constructor
  · intro instruction member
    have rowMember :
        Rows.CompiledRow.toR1CS (.witness instruction) ∈ instructionRows := by
      unfold instructionRows
      exact List.mem_map.mpr ⟨instruction, member, rfl⟩
    have rowHolds := separated.1 _ rowMember
    change instruction.toR1CS.Holds env at rowHolds
    exact (witnessInstruction_toR1CS_holds instruction env).mp rowHolds
  · intro assertion member
    have rowMember :
        Rows.CompiledRow.toR1CS (.assertion assertion) ∈ assertionRows := by
      unfold assertionRows
      exact List.mem_map.mpr ⟨assertion, member, rfl⟩
    have rowHolds := separated.2 _ rowMember
    change assertion.toR1CS.Holds env at rowHolds
    exact (sparseRow_holds assertion env).mp rowHolds

/-- Exact ordinary-row satisfaction implies the original logical post-hash
constraints without reconstructing the complete pilot package. -/
theorem sourceRows_hold_implies_priorConstraints (env : Env)
    (holds : R1CS.RowsHold env sourceRows) :
    ConstraintsHold (PilotSpartan.pullback env)
      (PilotData.priorExtraConstraints ()) := by
  have predicates := sourceRows_hold_implies_packagePredicates env holds
  have physical : R1CS.RowsHold env
      ((PilotData.priorExtraRows ()).map Rows.CompiledRow.toR1CS) := by
    apply (Rows.compiledRows_hold_iff
      (PilotData.priorExtraRows ()) env).mpr
    constructor
    · intro instruction member
      apply predicates.1 instruction
      simpa [PilotData.witnessInstructions,
        Rows.witnessInstructionsTR_eq] using member
    · intro assertion member
      apply predicates.2 assertion
      rw [PilotData.assertionRows, List.mem_append]
      exact Or.inl (by
        simpa [Rows.assertionRowsTR_eq] using member)
  rw [PilotData.priorExtraRows,
    PilotData.remapCompiledRows_toR1CS,
    Rows.compileRowsTR_toR1CS] at physical
  have sourceRowsHold := (PilotSpartan.remapRows_hold env
    (Rows.lowerConstraintsTR (PilotData.priorExtraConstraints ())
      PilotValues.logicalColumnCount).rows).mp physical
  rw [Rows.lowerConstraintsTR_eq] at sourceRowsHold
  exact R1CS.lowerConstraints_sound (PilotSpartan.pullback env)
    (PilotData.priorExtraConstraints ()) PilotValues.logicalColumnCount
    sourceRowsHold

end NightstreamFPrime.Export.Stage1.PilotOrdinaryDirectSource
