import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceProgram
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceSchedule

/-!
Direct source semantics for the two rewrite families in the focused `y_zcol`
slice.

Owns: the mathematical target of a polynomial-evaluation rewrite and of an
extension-product rewrite, plus their derivation from the independently
reconstructed source program.

Does not own: generated rewrite labels, compact coefficients, selector truth,
source-to-compact agreement, final checks, protocol authority, security
events, or permission to remove rows.

Emits constraints: no.

The targets below deliberately mention only source trace inputs and outputs;
they do not mention compiler fragments or selected rows. This keeps the
semantic direction source trace → direct equation → compact refinement.

| Source family | Mathematical obligation | Authority class |
|---|---|---|
| evaluation | output is the coefficient/power inner product | derived from source definitions |
| product | output is one quadratic-extension product | derived from source definitions |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.DirectSemantics

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram

/-- Direct equation implemented by one complete evaluation rewrite. -/
def EvaluationDirect (trace : EvalTrace) (assignment : Nat → Nat) : Prop :=
  trace.output.value assignment =
    K.add
      (K.ofBase (baseAt assignment (trace.coefficients.headD 0)))
      ((trace.ExpectedProducts assignment).foldr K.add K.zero)

/-- Direct equation implemented by one complete product-sum rewrite. -/
def ProductDirect (trace : KMulTrace) (assignment : Nat → Nat) : Prop :=
  trace.output.value assignment =
    K.mul (trace.left.value assignment) (trace.right.value assignment)

theorem evaluationDirect_of_definitions
    (trace : EvalTrace) (assignment : Nat → Nat)
    (layout : trace.LayoutValid)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    EvaluationDirect trace assignment := by
  have products := trace.products_sound assignment layout definitionsHold
  have output := trace.output_value assignment layout definitionsHold
  obtain ⟨head, tail, coefficientsEq⟩ :=
    List.exists_cons_of_ne_nil layout.1
  have headEq : trace.coefficients.head layout.1 =
      trace.coefficients.headD 0 := by
    simpa [coefficientsEq]
  unfold EvaluationDirect
  rw [output, products, headEq]

theorem productDirect_of_definitions
    (trace : KMulTrace) (assignment : Nat → Nat)
    (layout : trace.SumLayoutValid)
    (definitionsHold : DefinitionsHold assignment trace.definitions) :
    ProductDirect trace assignment := by
  exact trace.sound assignment layout definitionsHold

set_option maxRecDepth 100000 in
private theorem evaluationDefinitionsCovered :
    ∀ owner ∈ SourceSchedule.evaluationOwners,
      ∀ definition ∈ owner.trace.definitions,
        definition ∈ SourceProgram.sourceDefinitions := by
  native_decide

set_option maxRecDepth 100000 in
private theorem productDefinitionsCovered :
    ∀ owner ∈ SourceSchedule.productOwners,
      ∀ definition ∈ owner.trace.definitions,
        definition ∈ SourceProgram.sourceDefinitions := by
  native_decide

/-- The recomputed source assignment satisfies every scheduled evaluation's
direct equation by source definitions alone. -/
theorem sourceEvaluationsDirect (assignment : Nat → Nat) :
    ∀ owner ∈ SourceSchedule.evaluationOwners,
      EvaluationDirect owner.trace
        (SourceProgram.sourceAssignment assignment) := by
  intro owner ownerMember
  apply evaluationDirect_of_definitions owner.trace
    (SourceProgram.sourceAssignment assignment)
    (SourceSchedule.evaluation_layouts owner ownerMember)
  intro definition definitionMember
  exact SourceProgram.sourceAssignmentDefinitionsHold assignment definition
    (evaluationDefinitionsCovered owner ownerMember definition definitionMember)

/-- The recomputed source assignment satisfies every scheduled extension
product's direct equation by source definitions alone. -/
theorem sourceProductsDirect (assignment : Nat → Nat) :
    ∀ owner ∈ SourceSchedule.productOwners,
      ProductDirect owner.trace
        (SourceProgram.sourceAssignment assignment) := by
  intro owner ownerMember
  apply productDirect_of_definitions owner.trace
    (SourceProgram.sourceAssignment assignment)
    (SourceSchedule.product_layouts owner ownerMember)
  intro definition definitionMember
  exact SourceProgram.sourceAssignmentDefinitionsHold assignment definition
    (productDefinitionsCovered owner ownerMember definition definitionMember)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.DirectSemantics
