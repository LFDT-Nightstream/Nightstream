import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepConstructionPlans

/-!
Contract: split the exact physical Step cost into the fixed F-prime/NIFS
shell and the application-selected `step` call.

Both summands come from the conserved receipt program.  The application term
is additionally proved equal to the selected signature's exact `callCost`;
it is never supplied as numeric metadata.

Emits constraints: no.  This module partitions an existing receipt list.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationStepCostSplit

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

private theorem cost_extensionality
    {left right : Cost}
    (recurringRows :
      left.recurringRows = right.recurringRows)
    (committedColumns :
      left.committedColumns = right.committedColumns)
    (publicColumns :
      left.publicColumns = right.publicColumns)
    (auxiliaryColumns :
      left.auxiliaryColumns = right.auxiliaryColumns) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem cost_add_comm (left right : Cost) :
    left + right = right + left := by
  apply cost_extensionality <;>
    simp only [Cost.add_recurringRows, Cost.add_committedColumns,
      Cost.add_publicColumns, Cost.add_auxiliaryColumns] <;>
    omega

theorem committedLayout_cost_exact (width : Nat) :
    (committedLayout width).cost = ⟨0, width, 0, 0⟩ := by
  induction width with
  | zero =>
      rfl
  | succ width inductionHypothesis =>
      change
        Cost.oneColumn Ownership.committedColumn +
            (committedLayout width).cost =
          ⟨0, width + 1, 0, 0⟩
      rw [inductionHypothesis]
      apply cost_extensionality <;>
        simp [Cost.oneColumn] <;>
        omega

theorem auxiliaryLayout_cost_exact (width : Nat) :
    (auxiliaryLayout width).cost = ⟨0, 0, 0, width⟩ := by
  induction width with
  | zero =>
      rfl
  | succ width inductionHypothesis =>
      change
        Cost.oneColumn Ownership.auxiliaryColumn +
            (auxiliaryLayout width).cost =
          ⟨0, 0, 0, width + 1⟩
      rw [inductionHypothesis]
      apply cost_extensionality <;>
        simp [Cost.oneColumn] <;>
        omega

/-- The complete Step advice allocation is the sum of its seven committed
codec widths. No public or auxiliary column is part of the input schema. -/
theorem stepInputSchema_cost_exact (parameters : Parameters) :
    (stepInputSchema parameters).cost =
      ⟨0,
        parameters.widths.iteration +
          parameters.widths.state +
          parameters.widths.state +
          parameters.widths.running +
          parameters.widths.fresh +
          parameters.widths.witness +
          parameters.widths.nifsProof,
        0, 0⟩ := by
  simp only [stepInputSchema, Schema.cost, List.map_cons, List.map_nil,
    Cost.sum, Port.cost, Ports.committedNat, Ports.committedState,
    Ports.committedRunning, Ports.committedFresh, Ports.committedWitness,
    Ports.committedNifsProof, dataPort, committedLayout_cost_exact]
  apply cost_extensionality <;>
    simp only [Cost.add_recurringRows, Cost.add_committedColumns,
      Cost.add_publicColumns, Cost.add_auxiliaryColumns, Cost.zero] <;>
    omega

/-- The selected NIFS call adds only its committed running output to the
intrinsic proof-carrying footprint. -/
theorem nifsVerify_callCost_exact (parameters : Parameters) :
    (signature parameters).callCost Call.nifsVerify =
      parameters.footprints.nifsVerify.cost +
        (committedLayout parameters.widths.running).cost :=
  rfl

private theorem columnCost_append
    (left right : List OwnedColumn) :
    columnCost (left ++ right) = columnCost left + columnCost right := by
  unfold columnCost
  rw [List.map_append, Cost.sum_append]

private theorem columnBundle_cost
    {layout : Layout}
    (bundle : ColumnBundle layout) :
    columnCost bundle.columns = layout.cost := by
  unfold columnCost Layout.cost
  rw [← bundle.ownerships_exact]
  simp only [List.map_map, Function.comp_def]

private theorem schemaBundles_cost
    {types : TypeSystem}
    {schema : Schema types}
    (bundles : SchemaBundles schema) :
    columnCost bundles.columns = schema.cost := by
  induction bundles with
  | nil =>
      rfl
  | @cons port tail head rest inductionHypothesis =>
      simp only [SchemaBundles.columns, SchemaBundles.portColumns,
        List.flatten_cons, Schema.cost, List.map_cons, Cost.sum]
      rw [columnCost_append, columnBundle_cost]
      have restCost :
          columnCost rest.portColumns.flatten = tail.cost := by
        simpa only [SchemaBundles.columns] using inductionHypothesis
      rw [restCost]
      rfl

private theorem layoutBundles_cost
    {layouts : List Layout}
    (bundles : LayoutBundles layouts) :
    columnCost bundles.columns =
      Cost.sum (layouts.map Layout.cost) := by
  induction bundles with
  | nil =>
      rfl
  | @cons layout tail head rest inductionHypothesis =>
      simp only [LayoutBundles.columns, LayoutBundles.bundleColumns,
        List.flatten_cons, List.map_cons, Cost.sum]
      rw [columnCost_append, columnBundle_cost]
      have restCost :
          columnCost rest.bundleColumns.flatten =
            Cost.sum (tail.map Layout.cost) := by
        simpa only [LayoutBundles.columns] using inductionHypothesis
      rw [restCost]

private theorem rowCost_eq_length (rows : List OwnedRow) :
    rowCost rows = ⟨rows.length, 0, 0, 0⟩ := by
  induction rows with
  | nil =>
      rfl
  | cons _ tail inductionHypothesis =>
      change Cost.oneRow + rowCost tail =
        ⟨tail.length + 1, 0, 0, 0⟩
      rw [inductionHypothesis]
      apply cost_extensionality <;>
        simp only [Cost.add_recurringRows, Cost.add_committedColumns,
          Cost.add_publicColumns, Cost.add_auxiliaryColumns, Cost.oneRow] <;>
        omega

private theorem receiptsCost_eq_physicalCost
    (receipts : List InstructionReceipt) :
    Cost.sum (receipts.map InstructionReceipt.cost) =
      physicalCost
        (receipts.flatMap fun receipt => receipt.allocations)
        (receipts.flatMap fun receipt => receipt.rows) := by
  induction receipts with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, Cost.sum, List.flatMap_cons]
      rw [physicalCost_append, ← inductionHypothesis]
      rfl

private theorem inputReceipts_cost
    {types : TypeSystem}
    (schema : Schema types) :
    Cost.sum
        ((InputReceipts.receipts schema).map InstructionReceipt.cost) =
      schema.cost := by
  rw [receiptsCost_eq_physicalCost, InputReceipts.allocations_exact,
    InputReceipts.rows_empty]
  unfold physicalCost
  rw [rowCost_eq_length]
  have columns :
      columnCost (schemaOwnedColumns (inputColumns schema)) =
        schema.cost := by
    rw [← Columns.toSchemaBundles_columns]
    exact schemaBundles_cost (inputColumns schema).toSchemaBundles
  rw [columns]
  cases schema.cost
  rfl

/-- A physical call receipt has exactly the signature-owned call cost:
recurring rows, output coordinates, and temporary coordinates. -/
theorem callReceipt_cost_exact
    {signature : Signature}
    {family : Family signature.types}
    {call : signature.Call}
    (recipe : CallRecipe signature family call)
    {context : Schema signature.types}
    {references :
      Refs signature.types context (signature.callInputs call)}
    (frame : CallFrame family call references) :
    InstructionReceipt.cost (InstructionReceipt.ofCall recipe frame) =
      signature.callCost call := by
  unfold InstructionReceipt.cost InstructionReceipt.ofCall
    physicalCost CallFrame.allocations Signature.callCost
    CallFootprint.cost
  rw [columnCost_append, schemaBundles_cost, layoutBundles_cost,
    rowCost_eq_length, recipe.rowCount frame]
  rw [cost_add_comm
    ((signature.callOutputs call).cost +
      Cost.sum
        ((signature.callFootprint call).temporaries.map Layout.cost))
    ⟨(signature.callFootprint call).recurringRows, 0, 0, 0⟩]
  rw [cost_add_comm
    (signature.callOutputs call).cost
    (Cost.sum
      ((signature.callFootprint call).temporaries.map Layout.cost))]
  exact (Cost.add_assoc _ _ _).symm

private theorem invokePlan_cost_exact
    {parameters : Parameters}
    {profile : Profile parameters}
    {context : Schema (typeSystem parameters)}
    {call : (SelectedSignature parameters).Call}
    {operands :
      Refs (typeSystem parameters) context
        ((SelectedSignature parameters).callInputs call)}
    {path : OwnerPath}
    {inputColumns : Columns context}
    {one active : ColumnId}
    (plan :
      InvokePlan parameters profile call operands path inputColumns one active) :
    plan.receipt.cost =
      (signature parameters).callCost call :=
  callReceipt_cost_exact plan.recipe plan.frame

private theorem literalPlan_cost_exact
    {parameters : Parameters}
    {profile : Profile parameters}
    {context : Schema (typeSystem parameters)}
    {port : Port (typeSystem parameters)}
    {value : (typeSystem parameters).Value port.kind}
    {path : OwnerPath}
    {inputColumns : Columns context}
    {one active : ColumnId}
    (plan :
      LiteralPlan parameters profile port value path inputColumns one active) :
    plan.receipt.cost =
      port.layout.cost +
        ⟨port.layout.owners.length, 0, 0, 0⟩ := by
  unfold LiteralPlan.receipt InstructionReceipt.cost physicalCost
    InstructionReceipt.ofLiteral
  rw [columnBundle_cost, rowCost_eq_length,
    LiteralPinRecipe.row_count, plan.recipe.widthAgrees]

private theorem assertPlan_cost_exact
    {parameters : Parameters}
    {profile : Profile parameters}
    {context : Schema (typeSystem parameters)}
    {condition : Ref (typeSystem parameters) context .bit}
    {path : OwnerPath}
    {inputColumns : Columns context}
    {one active : ColumnId}
    (plan :
      AssertPlan parameters profile condition path inputColumns one active) :
    plan.receipt.cost = Cost.oneRow := by
  unfold AssertPlan.receipt InstructionReceipt.cost physicalCost
    InstructionReceipt.ofAssertion
  rw [rowCost_eq_length, BoolAssertRecipe.row_count]
  rfl

private theorem activationReceipt_cost_exact
    (path : OwnerPath)
    (one active selector : ColumnId)
    (selected : Bool) :
    (if selected then
        CanonicalBranchPlan.trueActivationReceipt path one active selector
      else
        CanonicalBranchPlan.falseActivationReceipt path one active selector
    ).cost =
      ⟨1, 0, 0, 1⟩ := by
  cases selected <;>
    rfl

private theorem trueActivationReceipt_cost_exact
    (path : OwnerPath)
    (one active selector : ColumnId) :
    (CanonicalBranchPlan.trueActivationReceipt
      path one active selector).cost =
        ⟨1, 0, 0, 1⟩ := by
  simpa using
    activationReceipt_cost_exact path one active selector true

private theorem falseActivationReceipt_cost_exact
    (path : OwnerPath)
    (one active selector : ColumnId) :
    (CanonicalBranchPlan.falseActivationReceipt
      path one active selector).cost =
        ⟨1, 0, 0, 1⟩ := by
  simpa using
    activationReceipt_cost_exact path one active selector false

private theorem joinReceipt_cost_exact
    {types : TypeSystem}
    (path : OwnerPath)
    (selector : ColumnId)
    (port : Port types)
    (onTrue onFalse : Bundle port) :
    (CanonicalBranchPlan.onePortJoinReceipt
      path selector port onTrue onFalse).cost =
        port.layout.cost +
          ⟨port.layout.owners.length, 0, 0, 0⟩ := by
  unfold CanonicalBranchPlan.onePortJoinReceipt InstructionReceipt.cost
    physicalCost InstructionReceipt.ofMux
  rw [columnBundle_cost, rowCost_eq_length, MuxRecipe.row_count]

/-- Exact compact cost list for the fifteen non-input Step receipts. -/
def stepBodyCosts (parameters : Parameters) : List Cost :=
  [ (signature parameters).callCost Call.step,
    (signature parameters).callCost Call.iterationZero,
    ⟨1, 0, 0, 1⟩,
    ⟨1, 0, 0, 1⟩,
    (signature parameters).callCost Call.stateEqual,
    Cost.oneRow,
    (Ports.committedRunning parameters).layout.cost +
      ⟨parameters.widths.running, 0, 0, 0⟩,
    (signature parameters).callCost Call.hashPrior,
    (signature parameters).callCost Call.freshPublic,
    (signature parameters).callCost Call.encodeInstance,
    (signature parameters).callCost Call.encodedEqual,
    Cost.oneRow,
    (signature parameters).callCost Call.nifsVerify,
    (Ports.committedRunning parameters).layout.cost +
      ⟨parameters.widths.running, 0, 0, 0⟩,
    (signature parameters).callCost Call.hashNext ]

/-- Exact compact cost of the complete canonical Step program. -/
def compactStepCost (parameters : Parameters) : Cost :=
  Cost.oneColumn .publicColumn +
    (stepInputSchema parameters).cost +
      Cost.sum (stepBodyCosts parameters)

private theorem bodyReceipts_cost_exact
    {parameters : Parameters}
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    Cost.sum
        ((CanonicalStepPlan.bodyReceipts parameters profile recipes
          defaultAdmissible).map InstructionReceipt.cost) =
      Cost.sum (stepBodyCosts parameters) := by
  simp only [CanonicalStepPlan.bodyReceipts, stepBodyCosts,
    List.map_cons, List.map_nil, Cost.sum,
    CanonicalStepPlan.applyPlan, CanonicalStepPlan.selectorPlan,
    CanonicalStepPlan.baseEqualityPlan,
    CanonicalStepPlan.baseAssertionPlan,
    CanonicalStepPlan.baseLiteralPlan,
    CanonicalStepPlan.recursiveHashPlan,
    CanonicalStepPlan.recursiveFreshPublicPlan,
    CanonicalStepPlan.recursiveEncodePlan,
    CanonicalStepPlan.recursiveEncodedEqualityPlan,
    CanonicalStepPlan.recursiveAssertionPlan,
    CanonicalStepPlan.recursiveNifsPlan,
    CanonicalStepPlan.continuationHashPlan,
    PrimitivePlan.receipt,
    invokePlan_cost_exact, literalPlan_cost_exact,
    assertPlan_cost_exact, trueActivationReceipt_cost_exact,
    falseActivationReceipt_cost_exact, joinReceipt_cost_exact,
    Ports.committedRunning, dataPort, committedLayout, ownedLayout,
    List.length_replicate]

namespace CompleteApplicationCertification

/-- Receipts before the application call: verifier one and exact Step inputs. -/
def stepPrefixReceipts
    {parameters : Parameters}
    (_certificate : CompleteApplicationCertification parameters) :
    List InstructionReceipt :=
  InstructionReceipt.prelude ::
    InputReceipts.receipts (stepInputSchema parameters)

/-- The canonical invocation owned by the application-selected Step call. -/
def applicationStepInvokePlan
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :=
  CanonicalStepPlan.applyInvokePlan parameters certificate.baseProfile
    certificate.allRecipes

/-- The one physical receipt owned by the application-selected Step call. -/
def applicationStepReceipt
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    InstructionReceipt :=
  (CanonicalStepPlan.applyPlan.{0} parameters certificate.baseProfile
    certificate.allRecipes).receipt

/-- The application receipt contains exactly the selected recipe rows at the
canonical physical call frame. -/
@[simp] theorem applicationStepReceipt_rows
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    (applicationStepReceipt certificate).rows =
      certificate.phase5.step.rows
        (applicationStepInvokePlan certificate).frame := by
  unfold applicationStepReceipt CanonicalStepPlan.applyPlan
    PrimitivePlan.receipt InvokePlan.receipt
  change
    (applicationStepInvokePlan certificate).recipe.rows
        (applicationStepInvokePlan certificate).frame =
      certificate.phase5.step.rows
        (applicationStepInvokePlan certificate).frame
  rw [show (applicationStepInvokePlan certificate).recipe =
      certificate.phase5.step by
    exact CompleteApplicationCertification.allRecipes_step certificate]

/-- Receipts after the application call, in exact typed-program order. -/
def stepSuffixReceipts
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    List InstructionReceipt :=
  (CanonicalStepPlan.bodyReceipts parameters certificate.baseProfile
    certificate.allRecipes certificate.defaultRunningAdmissible).tail

/-- The fixed protocol receipt stream excludes exactly the application call
receipt and retains every other allocation and row. -/
def fixedProtocolReceipts
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    List InstructionReceipt :=
  stepPrefixReceipts certificate ++ stepSuffixReceipts certificate

/-- Exact fixed F-prime/NIFS shell cost, folded from its physical receipts. -/
def fixedProtocolCost
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) : Cost :=
  Cost.sum
    ((fixedProtocolReceipts certificate).map InstructionReceipt.cost)

/-- Exact proof-carrying application cost, folded from its one physical call
receipt. -/
def applicationStepCost
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) : Cost :=
  (applicationStepReceipt certificate).cost

/-- The canonical Step receipt stream contains the application call exactly
between the input prefix and the fixed protocol suffix. -/
theorem stepReceipts_exact_split
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.canonicalStep.program.physical.receipts =
      stepPrefixReceipts certificate ++
        applicationStepReceipt certificate ::
          stepSuffixReceipts certificate := by
  simp [CompleteApplicationCertification.canonicalStep,
    CanonicalEncodingRealization.step,
    CanonicalStepPlan.aligned,
    CanonicalStepPlan.physical,
    CanonicalStepPlan.receipts,
    stepPrefixReceipts, applicationStepReceipt, stepSuffixReceipts,
    CanonicalStepPlan.bodyReceipts]

/-- The application summand is the selected signature's exact call cost,
never a caller-provided natural number. -/
theorem applicationStepCost_eq_callCost
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    applicationStepCost certificate =
      (signature parameters).callCost Call.step := by
  unfold applicationStepCost applicationStepReceipt
  exact callReceipt_cost_exact _ _

/-- Exact application-parametric decomposition of the complete physical Step
program. -/
theorem stepCost_eq_fixedProtocol_add_application
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.stepCost =
      fixedProtocolCost certificate + applicationStepCost certificate := by
  rw [certificate.stepCost_eq_receiptFold,
    stepReceipts_exact_split certificate]
  simp only [List.map_append, List.map_cons, Cost.sum_append, Cost.sum,
    fixedProtocolCost, fixedProtocolReceipts, applicationStepCost]
  rw [cost_add_comm
    (InstructionReceipt.cost (applicationStepReceipt certificate))
    (Cost.sum
      ((stepSuffixReceipts certificate).map InstructionReceipt.cost))]
  exact
    (Cost.add_assoc
      (Cost.sum
        ((stepPrefixReceipts certificate).map InstructionReceipt.cost))
      (Cost.sum
        ((stepSuffixReceipts certificate).map InstructionReceipt.cost))
      (InstructionReceipt.cost (applicationStepReceipt certificate))).symm

/-- The complete physical Step cost reduces to the compact list of selected
call costs and fixed primitive costs.  This avoids reduction through the
large emitted receipt payload. -/
theorem stepCost_eq_compact
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.stepCost = compactStepCost parameters := by
  rw [certificate.stepCost_eq_receiptFold]
  simp only [CompleteApplicationCertification.canonicalStep,
    CanonicalEncodingRealization.step, CanonicalStepPlan.aligned,
    CanonicalStepPlan.physical, CanonicalStepPlan.receipts,
    List.map_cons, List.map_append, Cost.sum, Cost.sum_append,
    InstructionReceipt.prelude, InstructionReceipt.cost,
    physicalCost, columnCost, rowCost]
  rw [inputReceipts_cost, bodyReceipts_cost_exact]
  rfl

end CompleteApplicationCertification

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationStepCostSplit
