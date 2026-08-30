import NightstreamFPrime.Export.Package
import NightstreamFPrime.Export.RowSemantics
import NightstreamFPrime.Export.Stage1.Rows
import NightstreamFPrime.Lifecycle.Stage1.Application
import NightstreamFPrime.Lifecycle.VerifierContext
import NightstreamFPrime.Layout.Stage1.ApplicationSemantics

/-!
Owns the standalone canonical package for one Lean-authored Stage 1
application.

The plan contains every application-owned executable row category and the
exact column map for its four-word input state, fixed-width witness, and
four-word output state. Its identifier is recomputed from the complete
prefix-free plan encoding. The final Stage 1 package will embed one such plan;
this module does not change the current candidate package.
-/

namespace NightstreamFPrime.Export.Stage1.ApplicationPackage

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Canonical executable application slice. All indices are final physical
package indices, so Rust performs no application-specific layout choice. -/
structure Plan where
  schemaVersion : Nat
  witnessWordCount : Nat
  inputColumns : List Nat
  witnessColumns : List Nat
  outputColumns : List Nat
  privateStart : Nat
  privateCount : Nat
  rowStart : Nat
  rowCount : Nat
  hashChains : List HashChain
  permutationInvocations : List PermutationInvocation
  compactRowTemplates : List CompactRowTemplate
  compactRowInvocations : List CompactRowInvocation
  witnessBatches : List WitnessBatch
  witnessInstructions : List WitnessInstruction
  assertionRows : List SparseRow
deriving Repr

def Plan.format : Format Plan where
  encode := fun value => .array [
    .atom value.schemaVersion,
    .atom value.witnessWordCount,
    (list nat).encode value.inputColumns,
    (list nat).encode value.witnessColumns,
    (list nat).encode value.outputColumns,
    .atom value.privateStart,
    .atom value.privateCount,
    .atom value.rowStart,
    .atom value.rowCount,
    (list HashChain.format).encode value.hashChains,
    (list PermutationInvocation.format).encode value.permutationInvocations,
    (list CompactRowTemplate.format).encode value.compactRowTemplates,
    (list CompactRowInvocation.format).encode value.compactRowInvocations,
    (list WitnessBatch.format).encode value.witnessBatches,
    (list WitnessInstruction.format).encode value.witnessInstructions,
    (list SparseRow.format).encode value.assertionRows]
  decode
    | .array [.atom schemaVersion, .atom witnessWordCount,
        inputColumns, witnessColumns, outputColumns,
        .atom privateStart, .atom privateCount, .atom rowStart,
        .atom rowCount, hashChains, permutationInvocations,
        compactRowTemplates, compactRowInvocations, witnessBatches,
        witnessInstructions, assertionRows] => do
      pure {
        schemaVersion
        witnessWordCount
        inputColumns := ← (list nat).decode inputColumns
        witnessColumns := ← (list nat).decode witnessColumns
        outputColumns := ← (list nat).decode outputColumns
        privateStart
        privateCount
        rowStart
        rowCount
        hashChains := ← (list HashChain.format).decode hashChains
        permutationInvocations :=
          ← (list PermutationInvocation.format).decode permutationInvocations
        compactRowTemplates :=
          ← (list CompactRowTemplate.format).decode compactRowTemplates
        compactRowInvocations :=
          ← (list CompactRowInvocation.format).decode compactRowInvocations
        witnessBatches := ← (list WitnessBatch.format).decode witnessBatches
        witnessInstructions :=
          ← (list WitnessInstruction.format).decode witnessInstructions
        assertionRows := ← (list SparseRow.format).decode assertionRows }
    | _ => .error "invalid Stage 1 application plan"
  decode_encode := by
    intro value
    cases value
    simp [Format.decode_encode] <;> rfl

theorem Plan.decode_encode (value : Plan) :
    Plan.format.decode (Plan.format.encode value) = .ok value :=
  Plan.format.decode_encode value

theorem Plan.encode_injective : Function.Injective Plan.format.encode := by
  intro left right encoded
  have decoded := congrArg Plan.format.decode encoded
  rw [Plan.decode_encode, Plan.decode_encode] at decoded
  exact Except.ok.inj decoded

/-- Fixed production Poseidon2 template size. Application permutations use
the same canonical template as the complete Stage 1 package. -/
def permutationRowCount : Nat := 592

/-- Exact expanded application row count under the canonical package rules. -/
def Plan.expandedRowCount (plan : Plan) : Nat :=
  (plan.hashChains.map fun chain => chain.witnessLength).sum +
    plan.permutationInvocations.length * permutationRowCount +
    compactRowCountFor plan.compactRowTemplates plan.compactRowInvocations +
    plan.witnessInstructions.length + plan.assertionRows.length

/-- Structural validity required before a plan can enter a final package.
Row- and column-ownership theorems remain obligations of the concrete
application layout. -/
structure Plan.WellFormed (plan : Plan) : Prop where
  schema : plan.schemaVersion = 1
  inputWidth : plan.inputColumns.length =
    Lifecycle.Stage1.Application.stateWordCount
  witnessWidth : plan.witnessColumns.length = plan.witnessWordCount
  outputWidth : plan.outputColumns.length =
    Lifecycle.Stage1.Application.stateWordCount
  expandedRows : plan.expandedRowCount = plan.rowCount

def identityDomain : List F :=
  ([78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47,
    70, 80, 114, 105, 109, 101, 47, 97, 112, 112, 108, 105, 99, 97,
    116, 105, 111, 110, 47, 118, 49] : List Nat).map Poseidon2.ofNat

/-- Full authoritative token sequence. A verifier can recompute it from the
embedded plan; a prover-carried digest never replaces these words. -/
def authorityWords (plan : Plan) : List F :=
  Package.valuePreimage (Plan.format.encode plan)

/-- Domain-separated identity of one exact application plan. -/
def identifier (plan : Plan) : VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList
    (Poseidon2.hash (identityDomain ++ authorityWords plan))

def identifierWords (plan : Plan) : List F :=
  (identifier plan).toList

@[simp] theorem identifierWords_length (plan : Plan) :
    (identifierWords plan).length = 4 := by
  exact VerifierContext.Digest4.toList_length (identifier plan)

/-- The application identity is always recomputed from the complete canonical
plan token sequence. -/
theorem identifier_recomputed (plan : Plan) :
    identifierWords plan =
      (VerifierContext.Digest4.ofList
        (Poseidon2.hash (identityDomain ++ authorityWords plan))).toList := by
  rfl

/-! ## Direct compiler from one selected Lean application -/

/-- Exact final-package columns used by the selected application. -/
structure Columns (witnessWordCount : Nat) where
  input : Lifecycle.Stage1.Application.StateIndex → Nat
  witness : Fin witnessWordCount → Nat
  output : Lifecycle.Stage1.Application.StateIndex → Nat

def Columns.interface {witnessWordCount : Nat}
    (columns : Columns witnessWordCount) :
    Lifecycle.Stage1.Application.Interface witnessWordCount where
  input := fun _ index => .var (columns.input index)
  witness := fun _ index => .var (columns.witness index)
  output := fun _ index => .var (columns.output index)

def operations (program : Lifecycle.Stage1.Application.Program)
    (columns : Columns program.witnessWordCount) (privateStart : Nat) :
    List Op :=
  Circuit.ops (program.circuit columns.interface).main privateStart

def constraints (program : Lifecycle.Stage1.Application.Program)
    (columns : Columns program.witnessWordCount) (privateStart : Nat) :
    List Expr :=
  flatConstraints (operations program columns privateStart)

def r1csFreshStart (program : Lifecycle.Stage1.Application.Program)
    (columns : Columns program.witnessWordCount) (privateStart : Nat) : Nat :=
  privateStart + localLength (operations program columns privateStart)

def compiledRows (program : Lifecycle.Stage1.Application.Program)
    (columns : Columns program.witnessWordCount) (privateStart rowStart : Nat) :
    List Rows.CompiledRow :=
  let source := constraints program columns privateStart
  let freshStart := r1csFreshStart program columns privateStart
  Rows.compileRowsTR freshStart rowStart
    (Rows.lowerConstraintsTR source freshStart).rows

/-- Canonical ordinary-row plan. The executable uses the stack-safe lowering
and row classifiers already proved equal to the structural definitions. -/
def ofProgram (program : Lifecycle.Stage1.Application.Program)
    (columns : Columns program.witnessWordCount) (privateStart rowStart : Nat) :
    Plan :=
  let ops := operations program columns privateStart
  let source := flatConstraints ops
  let rows := compiledRows program columns privateStart rowStart
  {
    schemaVersion := 1
    witnessWordCount := program.witnessWordCount
    inputColumns := List.ofFn columns.input
    witnessColumns := List.ofFn columns.witness
    outputColumns := List.ofFn columns.output
    privateStart
    privateCount := localLength ops + R1CS.totalFreshCount source
    rowStart
    rowCount := rows.length
    hashChains := []
    permutationInvocations := []
    compactRowTemplates := []
    compactRowInvocations := []
    witnessBatches := witnesses ops
    witnessInstructions := Rows.witnessInstructionsTR rows
    assertionRows := Rows.assertionRowsTR rows }

theorem ofProgram_wellFormed
    (program : Lifecycle.Stage1.Application.Program)
    (columns : Columns program.witnessWordCount) (privateStart rowStart : Nat) :
    (ofProgram program columns privateStart rowStart).WellFormed := by
  refine {
    schema := rfl
    inputWidth := by simp [ofProgram]
    witnessWidth := by simp [ofProgram]
    outputWidth := by simp [ofProgram]
    expandedRows := ?_ }
  simp only [Plan.expandedRowCount, ofProgram, List.map_nil, List.sum_nil,
    List.length_nil, zero_mul, zero_add, compactRowCountFor]
  exact Rows.witnessInstructionsTR_length_add_assertionRowsTR_length
    (compiledRows program columns privateStart rowStart)

theorem ofProgram_witnessBatches
    (program : Lifecycle.Stage1.Application.Program)
    (columns : Columns program.witnessWordCount) (privateStart rowStart : Nat) :
    (ofProgram program columns privateStart rowStart).witnessBatches =
      witnesses (operations program columns privateStart) := by
  rfl

theorem ofProgram_compiledRows_toR1CS
    (program : Lifecycle.Stage1.Application.Program)
    (columns : Columns program.witnessWordCount) (privateStart rowStart : Nat) :
    (compiledRows program columns privateStart rowStart).map
        Rows.CompiledRow.toR1CS =
      (R1CS.lowerConstraints (constraints program columns privateStart)
        (r1csFreshStart program columns privateStart)).rows := by
  unfold compiledRows
  rw [Rows.compileRowsTR_toR1CS, Rows.lowerConstraintsTR_eq]

/-- Independent satisfaction of every stored ordinary application row implies
the exact transition of the selected Lean program for an arbitrary
assignment. -/
theorem rows_imply_programHolds
    (program : Lifecycle.Stage1.Application.Program)
    (columns : Columns program.witnessWordCount) (privateStart rowStart : Nat)
    (env : Env)
    (assumptions :
      (program.circuit columns.interface).assumptions privateStart env)
    (instructions : ∀ instruction ∈
      (ofProgram program columns privateStart rowStart).witnessInstructions,
      instruction.Holds env)
    (assertions : ∀ assertion ∈
      (ofProgram program columns privateStart rowStart).assertionRows,
      assertion.Holds env) :
    Lifecycle.Stage1.Application.Holds program.step columns.interface
      privateStart env := by
  let rows := compiledRows program columns privateStart rowStart
  have classified :
      (∀ instruction ∈ Rows.witnessInstructions rows,
          instruction.Holds env) ∧
        ∀ assertion ∈ Rows.assertionRows rows, assertion.Holds env := by
    constructor
    · simpa [ofProgram, rows, Rows.witnessInstructionsTR_eq] using instructions
    · simpa [ofProgram, rows, Rows.assertionRowsTR_eq] using assertions
  have encodedRows : R1CS.RowsHold env
      (rows.map Rows.CompiledRow.toR1CS) :=
    (Rows.compiledRows_hold_iff rows env).mpr classified
  have loweredRows : R1CS.RowsHold env
      (R1CS.lowerConstraints (constraints program columns privateStart)
        (r1csFreshStart program columns privateStart)).rows := by
    rw [← ofProgram_compiledRows_toR1CS]
    exact encodedRows
  have flattened : holdsFlat env
      (operations program columns privateStart) := by
    exact R1CS.lowerConstraints_sound env
      (constraints program columns privateStart)
      (r1csFreshStart program columns privateStart) loweredRows
  exact program.soundness columns.interface privateStart env assumptions
    (holdsFlat_implies_holds env
      (operations program columns privateStart) flattened)

/-! ## Verifier-context application binding -/

/-- Replace only the application component of a verifier-owned authority with
the full canonical words of the selected Lean program plan. -/
def bindApplication (base : VerifierContext.Authority)
    (program : Lifecycle.Stage1.Application.Program)
    (columns : Columns program.witnessWordCount) (privateStart rowStart : Nat) :
    VerifierContext.Authority :=
  { base with
    applicationWords :=
      authorityWords (ofProgram program columns privateStart rowStart) }

@[simp] theorem bindApplication_applicationWords
    (base : VerifierContext.Authority)
    (program : Lifecycle.Stage1.Application.Program)
    (columns : Columns program.witnessWordCount) (privateStart rowStart : Nat) :
    (bindApplication base program columns privateStart rowStart).applicationWords =
      authorityWords (ofProgram program columns privateStart rowStart) := by
  rfl

theorem bindApplication_descriptor_application
    (base : VerifierContext.Authority)
    (program : Lifecycle.Stage1.Application.Program)
    (columns : Columns program.witnessWordCount) (privateStart rowStart : Nat) :
    (VerifierContext.descriptor
      (bindApplication base program columns privateStart rowStart)).application =
      VerifierContext.componentDigest 2
        (authorityWords
          (ofProgram program columns privateStart rowStart)) := by
  rfl

/-! ## Production zero-copy specialization -/

def productionColumns (program : Lifecycle.Stage1.Application.Program) :
    Columns program.witnessWordCount where
  input := Layout.Stage1.ApplicationInputs.inputColumn
  witness := Layout.Stage1.ApplicationInputs.witnessColumn
  output := Layout.Stage1.ApplicationInputs.outputColumn

@[simp] theorem productionColumns_interface
    (program : Lifecycle.Stage1.Application.Program) :
    (productionColumns program).interface =
      Layout.Stage1.ApplicationInputs.interface program := by
  rfl

def productionPlan (program : Lifecycle.Stage1.Application.Program)
    (rowStart : Nat) : Plan :=
  ofProgram program (productionColumns program)
    (Layout.Stage1.ApplicationInputs.localStart program) rowStart

theorem productionPlan_wellFormed
    (program : Lifecycle.Stage1.Application.Program) (rowStart : Nat) :
    (productionPlan program rowStart).WellFormed := by
  exact ofProgram_wellFormed program (productionColumns program)
    (Layout.Stage1.ApplicationInputs.localStart program) rowStart

/-- The independently checked canonical application rows imply the exact
typed transition between the prior and next state-hash preimages. -/
theorem productionRows_imply_typedTransition
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (program : Lifecycle.Stage1.Application.Program) (rowStart : Nat)
    (env : Env)
    (prior next : HashPreimage
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (witness : AppWitness)
    (priorFixed : Layout.PilotProduction.FixedPreimage prior)
    (nextFixed : Layout.PilotProduction.FixedPreimage next)
    (priorRepresents : PriorStateHash.RepresentsPreimage
      Layout.PilotProduction.priorInterface
      Layout.PilotProduction.witnessOffset
      (Layout.Stage1.Spartan.pullback env) prior)
    (nextRepresents : OutputHash.RepresentsPreimage
      Layout.PilotProduction.outputInterface
      Layout.PilotProduction.lifecycleOutputOffset
      (Layout.Stage1.Spartan.pullback env) next)
    (witnessRepresents :
      Lifecycle.Stage1.Application.witnessValue
          (Layout.Stage1.ApplicationInputs.interface program)
          (Layout.Stage1.ApplicationInputs.localStart program) env = witness)
    (instructions : ∀ instruction ∈
      (productionPlan program rowStart).witnessInstructions,
      instruction.Holds env)
    (assertions : ∀ assertion ∈
      (productionPlan program rowStart).assertionRows,
      assertion.Holds env) :
    next.current = program.step prior.current witness := by
  have assumptions := program.assumptions
    (Layout.Stage1.ApplicationInputs.interface program)
    (Layout.Stage1.ApplicationInputs.localStart program) env
    (Layout.Stage1.ApplicationInputs.externalBelow program)
  have semantic := rows_imply_programHolds program
    (productionColumns program)
    (Layout.Stage1.ApplicationInputs.localStart program) rowStart env
    (by simpa using assumptions) (by simpa [productionPlan] using instructions)
    (by simpa [productionPlan] using assertions)
  have inputEquals :=
    Layout.Stage1.ApplicationInputs.inputState_eq_current
      program env prior priorFixed priorRepresents
  have outputEquals :=
    Layout.Stage1.ApplicationInputs.outputState_eq_current
      program env next nextFixed nextRepresents
  unfold Lifecycle.Stage1.Application.Holds at semantic
  calc
    next.current =
        Lifecycle.Stage1.Application.outputState
          (Layout.Stage1.ApplicationInputs.interface program)
          (Layout.Stage1.ApplicationInputs.localStart program) env :=
      outputEquals.symm
    _ = program.step
        (Lifecycle.Stage1.Application.inputState
          (Layout.Stage1.ApplicationInputs.interface program)
          (Layout.Stage1.ApplicationInputs.localStart program) env)
        (Lifecycle.Stage1.Application.witnessValue
          (Layout.Stage1.ApplicationInputs.interface program)
          (Layout.Stage1.ApplicationInputs.localStart program) env) := by
      simpa using semantic
    _ = program.step prior.current witness :=
      congrArg₂ program.step inputEquals witnessRepresents

end NightstreamFPrime.Export.Stage1.ApplicationPackage
