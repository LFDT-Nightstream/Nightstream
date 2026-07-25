import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalBranchPlan
import Nightstream.Implementation.Lowering.Goldilocks.InputReceipts

/-!
Contract: small compositional scoping lemmas for canonical fixed-one receipt
sequences.

Owns:
- exact context-ID coverage by an accumulated allocation list;
- preservation of coverage when a receipt is appended;
- coverage of canonical call, literal, and one-port join outputs;
- composition of two already scoped receipt sequences.

Does not own: Step/Terminal traversal, receipt selection, owner order, costs,
row semantics, or any whole-program certificate.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks

universe u

namespace ReceiptScoping

/-- A singleton typed context has exactly the identities of its head bundle. -/
theorem singletonColumnsIds
    {types : TypeSystem.{u}}
    {port : Port types}
    (columns : Columns [port]) :
    columns.toSchemaBundles.ids =
      (HVec.head columns).toColumnBundle.ids := by
  cases columns with
  | cons head tail =>
      cases tail
      simp [Columns.toSchemaBundles, SchemaBundles.ids,
        SchemaBundles.columns, SchemaBundles.portColumns,
        HVec.head, ColumnBundle.ids, Bundle.toColumnBundle_columns]

/-- Every coordinate of one exact typed context has already been allocated. -/
def Covers
    {types : TypeSystem.{u}}
    {schema : Schema types}
    (columns : Columns schema)
    (available : List ColumnId) : Prop :=
  ∀ id, id ∈ columns.toSchemaBundles.ids -> id ∈ available

namespace Covers

/-- Head and tail coverage compose for one exact typed-context constructor. -/
theorem cons
    {types : TypeSystem.{u}}
    {port : Port types}
    {tail : Schema types}
    {head : Bundle port}
    {rest : Columns tail}
    {available : List ColumnId}
    (headCovers :
      ∀ id, id ∈ head.toColumnBundle.ids -> id ∈ available)
    (tailCovers : Covers rest available) :
    Covers (HVec.cons head rest) available := by
  intro id member
  rw [Columns.toSchemaBundles.eq_2,
    SchemaBundles.ids_cons] at member
  rcases List.mem_append.mp member with headMember | tailMember
  · exact headCovers id headMember
  · exact tailCovers id tailMember

theorem weaken
    {types : TypeSystem.{u}}
    {schema : Schema types}
    {columns : Columns schema}
    {available : List ColumnId}
    (covers : Covers columns available)
    (later : List ColumnId) :
    Covers columns (available ++ later) := by
  intro id member
  exact List.mem_append_left later (covers id member)

theorem append
    {types : TypeSystem.{u}}
    {left right : Schema types}
    {leftColumns : Columns left}
    {rightColumns : Columns right}
    {available : List ColumnId}
    (leftCovers : Covers leftColumns available)
    (rightCovers : Covers rightColumns available) :
    Covers (leftColumns.append rightColumns) available := by
  induction leftColumns with
  | nil =>
      exact rightCovers
  | cons head rest inductionHypothesis =>
      rw [HVec.append.eq_2]
      apply cons
      · intro id headMember
        apply leftCovers id
        rw [Columns.toSchemaBundles.eq_2,
          SchemaBundles.ids_cons, List.mem_append]
        exact Or.inl headMember
      · apply inductionHypothesis
        · intro id tailMember
          apply leftCovers id
          rw [Columns.toSchemaBundles.eq_2,
            SchemaBundles.ids_cons, List.mem_append]
          exact Or.inr tailMember

theorem primitiveInputs
    {parameters :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.Parameters}
    {schema :
      Schema
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.typeSystem
          parameters)}
    {columns : Columns schema}
    {one active : ColumnId}
    {available : List ColumnId}
    (covers : Covers columns available)
    (oneAvailable : one ∈ available)
    (activeAvailable : active ∈ available) :
    PrimitivePlan.InputsAvailable columns one active available := by
  intro id member
  rcases List.mem_append.mp member with controls | context
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at controls
    rcases controls with rfl | rfl
    · exact oneAvailable
    · exact activeAvailable
  · exact covers id context

end Covers

namespace InvokePlan

/-- A canonical call receipt allocates every coordinate of its exact output
schema, so those coordinates are covered immediately after the receipt. -/
theorem outputCoversAfter
    {parameters :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.Parameters}
    {profile : Profile parameters}
    {context :
      Schema
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.typeSystem
          parameters)}
    {call : (SelectedSignature parameters).Call}
    {operands :
      Refs
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.typeSystem
          parameters)
        context ((SelectedSignature parameters).callInputs call)}
    {path : OwnerPath}
    {inputColumns : Columns context}
    {one active : ColumnId}
    (plan :
      InvokePlan parameters profile call operands path
        inputColumns one active)
    (available : List ColumnId) :
    Covers
      (instructionColumns path
        ((SelectedSignature parameters).callOutputs call))
      (available ++ plan.receipt.columnIds) := by
  intro id member
  apply List.mem_append_right available
  rw [← plan.outputsExact] at member
  have receiptIds :
      plan.receipt.columnIds =
        plan.frame.outputs.ids ++ plan.frame.temporaries.ids := by
    simp [InvokePlan.receipt, InstructionReceipt.columnIds,
      CallFrame.allocations, SchemaBundles.ids, LayoutBundles.ids,
      List.map_append]
  rw [receiptIds]
  exact List.mem_append_left plan.frame.temporaries.ids member

end InvokePlan

namespace LiteralPlan

/-- A canonical literal receipt allocates its one exact typed output bundle. -/
theorem outputCoversAfter
    {parameters :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.Parameters}
    {profile : Profile parameters}
    {context :
      Schema
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.typeSystem
          parameters)}
    {port :
      Port
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.typeSystem
          parameters)}
    {value :
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.typeSystem
        parameters).Value port.kind}
    {path : OwnerPath}
    {inputColumns : Columns context}
    {one active : ColumnId}
    (plan :
      LiteralPlan parameters profile port value path
        inputColumns one active)
    (available : List ColumnId) :
    Covers (instructionColumns path [port])
      (available ++ plan.receipt.columnIds) := by
  intro id member
  apply List.mem_append_right available
  rw [singletonColumnsIds] at member
  rw [← plan.outputExact] at member
  simpa [LiteralPlan.receipt, InstructionReceipt.columnIds,
    ColumnBundle.ids] using member

end LiteralPlan

/-- Exact fresh-output coverage selected by the primitive-plan constructor.
Assertions allocate no fresh output. -/
def PrimitivePlan.FreshOutputsCoveredAfter
    {parameters :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.Parameters}
    {profile : Profile parameters}
    {input output :
      Schema
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.typeSystem
          parameters)}
    {primitive :
      Primitive (SelectedSignature parameters) input output}
    {path : OwnerPath}
    {inputColumns : Columns input}
    {one active : ColumnId}
    (plan :
      PrimitivePlan parameters profile primitive path
        inputColumns one active)
    (available : List ColumnId) : Prop :=
  match plan with
  | .invoke (call := call) plan =>
      Covers
        (instructionColumns path
          ((SelectedSignature parameters).callOutputs call))
        (available ++ plan.receipt.columnIds)
  | .literal (port := port) plan =>
      Covers (instructionColumns path [port])
        (available ++ plan.receipt.columnIds)
  | .assertTrue _ =>
      True

/-- Every supported primitive plan establishes its constructor-selected exact
fresh-output coverage. -/
theorem PrimitivePlan.freshOutputsCoveredAfter
    {parameters :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.Parameters}
    {profile : Profile parameters}
    {input output :
      Schema
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.typeSystem
          parameters)}
    {primitive :
      Primitive (SelectedSignature parameters) input output}
    {path : OwnerPath}
    {inputColumns : Columns input}
    {one active : ColumnId}
    (plan :
      PrimitivePlan parameters profile primitive path
        inputColumns one active)
    (available : List ColumnId) :
    PrimitivePlan.FreshOutputsCoveredAfter plan available := by
  cases plan with
  | invoke plan =>
      exact InvokePlan.outputCoversAfter plan available
  | literal plan =>
      exact LiteralPlan.outputCoversAfter plan available
  | assertTrue =>
      trivial

/-- Exact post-primitive context coverage selected by the primitive-plan
constructor. -/
def PrimitivePlan.ResultCoveredAfter
    {parameters :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.Parameters}
    {profile : Profile parameters}
    {input output :
      Schema
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.typeSystem
          parameters)}
    {primitive :
      Primitive (SelectedSignature parameters) input output}
    {path : OwnerPath}
    {inputColumns : Columns input}
    {one active : ColumnId}
    (plan :
      PrimitivePlan parameters profile primitive path
        inputColumns one active)
    (available : List ColumnId) : Prop :=
  match plan with
  | .invoke (call := call) plan =>
      Covers
        ((instructionColumns path
          ((SelectedSignature parameters).callOutputs call)).append
            inputColumns)
        (available ++ plan.receipt.columnIds)
  | .literal (port := port) plan =>
      Covers
        ((instructionColumns path [port]).append inputColumns)
        (available ++ plan.receipt.columnIds)
  | .assertTrue plan =>
      Covers inputColumns (available ++ plan.receipt.columnIds)

/-- A scoped primitive preserves its exact input context and adds exactly its
fresh outputs. -/
theorem PrimitivePlan.resultCoveredAfter
    {parameters :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.Parameters}
    {profile : Profile parameters}
    {input output :
      Schema
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.typeSystem
          parameters)}
    {primitive :
      Primitive (SelectedSignature parameters) input output}
    {path : OwnerPath}
    {inputColumns : Columns input}
    {one active : ColumnId}
    (plan :
      PrimitivePlan parameters profile primitive path
        inputColumns one active)
    (available : List ColumnId)
    (inputCovers : Covers inputColumns available) :
    PrimitivePlan.ResultCoveredAfter plan available := by
  cases plan with
  | invoke plan =>
      exact Covers.append
        (InvokePlan.outputCoversAfter plan available)
        (inputCovers.weaken plan.receipt.columnIds)
  | literal plan =>
      exact Covers.append
        (LiteralPlan.outputCoversAfter plan available)
        (inputCovers.weaken plan.receipt.columnIds)
  | assertTrue plan =>
      exact inputCovers.weaken plan.receipt.columnIds

/-- The selected-true activation coordinate is available immediately after
its canonical activation receipt. -/
theorem trueActivationAvailableAfter
    (path : OwnerPath)
    (one active selector : ColumnId)
    (available : List ColumnId) :
    activationColumn path true ∈
      available ++
        (CanonicalBranchPlan.trueActivationReceipt
          path one active selector).columnIds := by
  apply List.mem_append_right available
  simp [CanonicalBranchPlan.trueActivationReceipt,
    CanonicalBranchPlan.activationRecipe,
    InstructionReceipt.columnIds,
    InstructionReceipt.ofTrueActivation]

/-- The selected-false activation coordinate is available immediately after
its canonical activation receipt. -/
theorem falseActivationAvailableAfter
    (path : OwnerPath)
    (one active selector : ColumnId)
    (available : List ColumnId) :
    activationColumn path false ∈
      available ++
        (CanonicalBranchPlan.falseActivationReceipt
          path one active selector).columnIds := by
  apply List.mem_append_right available
  simp [CanonicalBranchPlan.falseActivationReceipt,
    CanonicalBranchPlan.activationRecipe,
    InstructionReceipt.columnIds,
    InstructionReceipt.ofFalseActivation]

/-- A canonical one-port join receipt allocates every joined coordinate. -/
theorem joinOutputCoversAfter
    {types : TypeSystem.{u}}
    (path : OwnerPath)
    (selector : ColumnId)
    (port : Port types)
    (onTrue onFalse : Bundle port)
    (available : List ColumnId) :
    Covers (branchJoinColumns path [port])
      (available ++
        (CanonicalBranchPlan.onePortJoinReceipt
          path selector port onTrue onFalse).columnIds) := by
  intro id member
  apply List.mem_append_right available
  rw [singletonColumnsIds] at member
  simpa [CanonicalBranchPlan.onePortJoinReceipt,
    CanonicalBranchPlan.onePortJoinRecipe,
    InstructionReceipt.columnIds, InstructionReceipt.ofMux,
    ColumnBundle.ids] using member

/-- Sequentially scoped receipt lists compose at the exact flattened
allocation boundary. -/
theorem wellScoped_append
    (available : List ColumnId)
    (left right : List InstructionReceipt)
    (leftScoped : ReceiptsWellScoped available left)
    (rightScoped :
      ReceiptsWellScoped
        (available ++ left.flatMap InstructionReceipt.columnIds)
        right) :
    ReceiptsWellScoped available (left ++ right) := by
  induction left generalizing available with
  | nil =>
      simpa using rightScoped
  | cons head tail inductionHypothesis =>
      rcases leftScoped with ⟨headScoped, tailScoped⟩
      constructor
      · exact headScoped
      · apply inductionHypothesis
          (available := available ++ head.columnIds)
          tailScoped
        simpa [List.flatMap_cons, List.append_assoc] using rightScoped

end ReceiptScoping

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
