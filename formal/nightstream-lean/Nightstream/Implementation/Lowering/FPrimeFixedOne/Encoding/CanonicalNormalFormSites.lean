import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalEncoding
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepPlan
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalTerminalProgram

/-!
Contract: exact finite normal-form specifications induced by canonical
assertion and branch-join recipes.

Owns:
- a deterministic fresh residual identity for the alternative assertion
  recipe;
- exact direct-assertion row equality;
- one branch-join specification per canonical mux coordinate;
- exact selected-mux rows and owner/ordinal census.

Does not own: whole-program certificate construction, minimum theorems,
semantic R1CS refinement, Rust emission, or generated artifacts.

The assignment stored in each local semantic specification is the constant
one assignment.  Candidate correctness is already quantified over every
well-formed specification; this fixed inhabitant only packages the
assignment-irrelevant physical rows and costs into a whole-program
certificate.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.PrimitiveNormalForm

namespace CanonicalNormalFormSites

/-- A total assignment used only to inhabit the assignment fields of local
normal-form specifications.  Physical rows and costs do not depend on it. -/
def assignment : ColumnId -> Field :=
  fun _ => 1

/-- Candidate-local assertion residual, placed strictly after every bundle
index mentioned by the direct assertion recipe. -/
def assertionResidual (recipe : BoolAssertRecipe) : ColumnId :=
  { owner := recipe.owner
    bundleIndex :=
      recipe.one.bundleIndex +
        recipe.active.bundleIndex +
        recipe.condition.bundleIndex + 1
    coordinateIndex := 0 }

theorem assertionResidual_ne_one (recipe : BoolAssertRecipe) :
    assertionResidual recipe ≠ recipe.one := by
  intro equal
  have bundleEqual := congrArg ColumnId.bundleIndex equal
  simp only [assertionResidual] at bundleEqual
  omega

theorem assertionResidual_ne_active (recipe : BoolAssertRecipe) :
    assertionResidual recipe ≠ recipe.active := by
  intro equal
  have bundleEqual := congrArg ColumnId.bundleIndex equal
  simp only [assertionResidual] at bundleEqual
  omega

theorem assertionResidual_ne_condition (recipe : BoolAssertRecipe) :
    assertionResidual recipe ≠ recipe.condition := by
  intro equal
  have bundleEqual := congrArg ColumnId.bundleIndex equal
  simp only [assertionResidual] at bundleEqual
  omega

/-- The exact finite-class specification induced by one canonical direct
assertion recipe. -/
def assertionSpecification
    (recipe : BoolAssertRecipe) :
    GatedAssertion.Specification where
  owner := recipe.owner
  firstOrdinal := recipe.ordinal
  assignment := assignment
  one := recipe.one
  active := recipe.active
  condition := recipe.condition
  residual := assertionResidual recipe
  constantOne := rfl
  residual_ne_one := assertionResidual_ne_one recipe
  residual_ne_active := assertionResidual_ne_active recipe
  residual_ne_condition := assertionResidual_ne_condition recipe

/-- The selected direct candidate emits definitionally the canonical
assertion recipe's one exact owned row. -/
theorem assertionRowsExact (recipe : BoolAssertRecipe) :
    GatedAssertion.Candidate.rows
        .direct (assertionSpecification recipe) =
      recipe.rows :=
  rfl

private def joinSpecificationsFrom
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (selector : ColumnId) :
    List OwnedColumn ->
      List OwnedColumn ->
      List OwnedColumn ->
      List BranchJoin.Specification
  | joined :: joinedTail,
      onTrue :: trueTail,
      onFalse :: falseTail =>
      { owner := owner
        firstOrdinal := firstOrdinal
        assignment := assignment
        one := oneColumn
        selector := selector
        joined := joined.id
        onTrue := onTrue.id
        onFalse := onFalse.id
        constantOne := rfl
        selectorBoolean := Or.inr rfl } ::
        joinSpecificationsFrom owner (firstOrdinal + 1) selector
          joinedTail trueTail falseTail
  | _, _, _ => []

/-- One local normal-form site for every coordinate emitted by an exact mux
recipe, in the recipe's physical order. -/
def joinSpecifications
    {layout : Layout}
    (recipe : MuxRecipe layout) :
    List BranchJoin.Specification :=
  joinSpecificationsFrom recipe.owner recipe.firstOrdinal recipe.selector
    recipe.joined.columns recipe.onTrue.columns recipe.onFalse.columns

/-- Flattening the selected one-row candidate at each induced site recovers
the canonical mux recipe's complete ordered row list. -/
theorem joinRowsExact
    {layout : Layout}
    (recipe : MuxRecipe layout) :
    recipe.rows =
      (joinSpecifications recipe).flatMap
        (fun specification =>
          BranchJoin.Candidate.rows .selectedMux specification) := by
  unfold joinSpecifications MuxRecipe.rows
  generalize recipe.owner = owner
  generalize recipe.firstOrdinal = firstOrdinal
  generalize recipe.selector = selector
  generalize recipe.joined.columns = joined
  generalize recipe.onTrue.columns = onTrue
  generalize recipe.onFalse.columns = onFalse
  induction joined generalizing firstOrdinal onTrue onFalse with
  | nil =>
      simp [joinSpecificationsFrom, muxRowsFrom]
  | cons joined joinedTail inductionHypothesis =>
      cases onTrue with
      | nil =>
          simp [joinSpecificationsFrom, muxRowsFrom]
      | cons onTrue trueTail =>
          cases onFalse with
          | nil =>
              simp [joinSpecificationsFrom, muxRowsFrom]
          | cons onFalse falseTail =>
              simp only [joinSpecificationsFrom, muxRowsFrom,
                BranchJoin.Candidate.rows, List.flatMap_cons,
                List.singleton_append, List.cons.injEq]
              constructor
              · rfl
              · simpa only [BranchJoin.Candidate.rows] using
                  inductionHypothesis
                    (firstOrdinal + 1) trueTail falseTail

/-- The induced local sites use the mux receipt owner and consecutive
occurrence-local ordinals, with no skipped or duplicated coordinate. -/
theorem joinOwnerOrdinalsExact
    {layout : Layout}
    (recipe : MuxRecipe layout) :
    (joinSpecifications recipe).map
        (fun specification =>
          (specification.owner, specification.firstOrdinal)) =
      (List.range' recipe.firstOrdinal
        recipe.joined.columns.length).map
          (fun ordinal => (recipe.owner, ordinal)) := by
  have trueLength :
      recipe.joined.columns.length =
        recipe.onTrue.columns.length := by
    rw [recipe.joined.length_eq, recipe.onTrue.length_eq]
  have falseLength :
      recipe.joined.columns.length =
        recipe.onFalse.columns.length := by
    rw [recipe.joined.length_eq, recipe.onFalse.length_eq]
  unfold joinSpecifications
  generalize recipe.joined.columns = joined at trueLength falseLength ⊢
  generalize recipe.onTrue.columns = onTrue at trueLength falseLength ⊢
  generalize recipe.onFalse.columns = onFalse at trueLength falseLength ⊢
  generalize recipe.owner = owner at ⊢
  generalize recipe.firstOrdinal = firstOrdinal at ⊢
  generalize recipe.selector = selector at ⊢
  induction joined generalizing firstOrdinal onTrue onFalse with
  | nil =>
      simp [joinSpecificationsFrom]
  | cons joined joinedTail inductionHypothesis =>
      cases onTrue with
      | nil =>
          simp at trueLength
      | cons onTrue trueTail =>
          cases onFalse with
          | nil =>
              simp at falseLength
          | cons onFalse falseTail =>
              simp only [List.length_cons, Nat.succ.injEq] at trueLength falseLength
              simp only [joinSpecificationsFrom, List.map_cons,
                List.length_cons, List.range'_succ, List.cons.injEq]
              constructor
              · trivial
              · exact inductionHypothesis
                  trueTail trueLength falseTail falseLength
                  (firstOrdinal + 1)

/-- A one-port schema owns exactly the columns in its unique bundle.  This
structural bridge keeps exact-receipt proofs independent of allocator
reduction details. -/
theorem singletonSchemaOwnedColumns
    {types : TypeSystem}
    {port : Port types}
    (columns : Columns [port]) :
    schemaOwnedColumns columns =
      bundleOwnedColumns port (HVec.head columns) := by
  cases columns with
  | cons head tail =>
      cases tail
      simp [schemaOwnedColumns, HVec.head]

end CanonicalNormalFormSites

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
