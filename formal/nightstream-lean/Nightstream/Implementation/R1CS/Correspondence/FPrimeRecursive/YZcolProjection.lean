import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.YZcolProjection
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.IndexedRows
import Nightstream.Implementation.R1CS.Core.Projection.Interpretation
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.LimbDecomposition

/-!
Conditional semantic refinement for the two fixed-profile parent `y_zcol`
output-evaluation source-R1CS leaves.

Owns: exact normalized-source-row interpretation, the explicit algebra
transport from the R1CS projection interpreter to the independent concrete
SuperNeo carrier, and composition of the two generated leaves into the
supplied packed-parent projection.

Does not own: satisfaction of these definitions by the whole production
matrix, the beta-ladder rows, transcript timing, parent commitment/opening
authority, selection of the canonical NIFS parent, the ten padded-lane
zero/canonicalization checks, gadget-native/low-norm lowering, encoded costs,
bad-root probability, global costs, or permission to remove rows.

Emits constraints: no.

Assurance tier: conditional model-level semantics over artifact-checked
fixed-profile source rows. The generator drift test establishes artifact
freshness, not whole-verifier Rust conformance. Whole-R1CS conformance still
requires exact indexed embedding of these rows, transcript authority, and
semantic decoding of the supplied parent.

Authority boundary: neither the generated columns nor their two scalar
outputs are authority. The main theorem requires semantic parent-column
matching explicitly and proves only deterministic evaluator correctness.

| Stage path | Mathematical obligation | Premise/owner | Result |
|---|---|---|---|
| `nifs.pi_rlc.verify.identities.y_zcol.evaluations.output.limb0` | evaluate all 54 parent `c0` coefficients at beta | exact normalized source rows plus shared powers | first semantic limb evaluation |
| `nifs.pi_rlc.verify.identities.y_zcol.evaluations.output.limb1` | evaluate all 54 parent `c1` coefficients at beta | exact normalized source rows plus shared powers | second semantic limb evaluation |
| `nifs.pi_rlc.verify.identities.y_zcol.evaluations.output` | recombine `E0 + u * E1` | both leaf results plus `ParentColumnsMatch` | canonical `projectedValue parent beta` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjection.Refinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjectionData
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.LimbDecomposition

/-- The two Goldilocks modulus definitions are intentionally transported
explicitly instead of relying on their current definitional equality. -/
def toSemanticField (value : ProjectionProgram.F) : F :=
  ⟨value.val, by
    simpa [goldilocksP, goldilocksModulus] using value.isLt⟩

@[simp] theorem toSemanticField_add
    (left right : ProjectionProgram.F) :
    toSemanticField (left + right) =
      toSemanticField left + toSemanticField right := by
  apply Fin.ext
  rfl

@[simp] theorem toSemanticField_mul
    (left right : ProjectionProgram.F) :
    toSemanticField (left * right) =
      toSemanticField left * toSemanticField right := by
  apply Fin.ext
  rfl

@[simp] theorem toSemanticField_seven :
    toSemanticField (7 : ProjectionProgram.F) = (7 : F) := by
  apply Fin.ext
  rfl

/-- Structure-preserving transport between the implementation interpreter's
quadratic extension and the independent SuperNeo carrier. -/
def toSemanticK (value : ProjectionProgram.K) : K :=
  ⟨toSemanticField value.c0, toSemanticField value.c1⟩

@[simp] theorem toSemanticK_zero :
    toSemanticK ProjectionProgram.K.zero = K.zero := by
  rfl

@[simp] theorem toSemanticK_add (left right : ProjectionProgram.K) :
    toSemanticK (ProjectionProgram.K.add left right) =
      K.add (toSemanticK left) (toSemanticK right) := by
  rfl

@[simp] theorem toSemanticK_mul (left right : ProjectionProgram.K) :
    toSemanticK (ProjectionProgram.K.mul left right) =
      K.mul (toSemanticK left) (toSemanticK right) := by
  rcases left with ⟨left0, left1⟩
  rcases right with ⟨right0, right1⟩
  simp only [toSemanticK, ProjectionProgram.K.mul, K.mul, K.mk.injEq,
    toSemanticField_add, toSemanticField_mul, toSemanticField_seven]
  constructor
  · apply Fin.ext
    simp [Fin.mul_assoc]
  · trivial

/-- Homomorphic transport of constant-first Horner evaluation. This is the
single bridge between the duplicate implementation/model carrier types. -/
theorem eval_transport (coefficients : List ProjectionProgram.K)
    (point : ProjectionProgram.K) :
    toSemanticK
        (ProjectionCheck.eval ProjectionProgram.K.ops coefficients point) =
      ProjectionCheck.eval projectionOps
        (coefficients.map toSemanticK) (toSemanticK point) := by
  induction coefficients with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      change toSemanticK
          (ProjectionProgram.K.add head
            (ProjectionProgram.K.mul point
              (ProjectionCheck.eval ProjectionProgram.K.ops tail point))) =
        K.add (toSemanticK head)
          (K.mul (toSemanticK point)
            (ProjectionCheck.eval projectionOps
              (tail.map toSemanticK) (toSemanticK point)))
      rw [toSemanticK_add, toSemanticK_mul, inductionHypothesis]

/-- Semantic base-field coefficient list read from physical columns. -/
def liftedColumns (assignment : Nat -> Nat) (columns : List Nat) :
    List K :=
  columns.map fun column =>
    K.embed
      (toSemanticField (ProjectionProgram.baseAt assignment column))

theorem basePolynomial_transport (assignment : Nat -> Nat)
    (columns : List Nat) :
    (ProjectionProgram.basePolynomial assignment columns).map toSemanticK =
      liftedColumns assignment columns := by
  simp [ProjectionProgram.basePolynomial, liftedColumns, toSemanticK,
    toSemanticField, K.embed]

/-- One evaluator leaf is correct when its exact definitions hold and its
power inputs are the advertised powers of `point`. -/
theorem leaf_definitions_sound
    (owner : YZcolOutputEvaluationOwner)
    (assignment : Nat -> Nat)
    (point : ProjectionProgram.K)
    (layout : owner.evalTrace.LayoutValid)
    (powersValid : owner.evalTrace.PowersValid assignment point)
    (definitionsHold :
      ProjectionProgram.DefinitionsHold assignment
        owner.evalTrace.definitions) :
    toSemanticK (owner.evalTrace.output.value assignment) =
      ProjectionCheck.eval projectionOps
        (liftedColumns assignment owner.parentCoefficientColumns)
        (toSemanticK point) := by
  have physical := owner.evalTrace.sound assignment point layout powersValid
    definitionsHold
  calc
    toSemanticK (owner.evalTrace.output.value assignment) =
        toSemanticK
          (ProjectionProgram.Polynomial.eval
            (ProjectionProgram.basePolynomial assignment
              owner.parentCoefficientColumns) point) :=
      congrArg toSemanticK physical
    _ = ProjectionCheck.eval projectionOps
        ((ProjectionProgram.basePolynomial assignment
          owner.parentCoefficientColumns).map toSemanticK)
        (toSemanticK point) :=
      eval_transport
        (ProjectionProgram.basePolynomial assignment
          owner.parentCoefficientColumns) point
    _ = ProjectionCheck.eval projectionOps
        (liftedColumns assignment owner.parentCoefficientColumns)
        (toSemanticK point) := by
      rw [basePolynomial_transport]

/-- Definition schedule reconstructed from the two artifact owners. -/
def ownedDefinitions : List Program.Definition :=
  ownedRowDefinitions.map Prod.snd

/-- Exact builder-row schedule corresponding to `ownedDefinitions`. -/
def ownedBuilderRows : List Row :=
  ownedRowDefinitions.map fun entry => entry.2.builderRow

/-- The two leaf schedules in semantic execution order. -/
def evaluatorDefinitions : List Program.Definition :=
  limb0Owner.evalTrace.definitions ++ limb1Owner.evalTrace.definitions

theorem ownedDefinitions_eq_evaluatorDefinitions :
    ownedDefinitions = evaluatorDefinitions := by
  set_option maxRecDepth 100000 in
    decide

theorem ownedDefinitions_canonical :
    ∀ definition ∈ ownedDefinitions, definition.Canonical := by
  set_option maxRecDepth 100000 in
    decide

/-- Exact source-row satisfaction yields all 216 reconstructed SSA
definitions. Assignment canonicality and the constant-one wire stay explicit
because they are global R1CS invariants, not artifact facts. -/
theorem ownedSourceRows_definitionsHold
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (sourceSatisfies : Satisfies ownedSourceRows assignment) :
    ProjectionProgram.DefinitionsHold assignment evaluatorDefinitions := by
  have builderRows : Satisfies ownedBuilderRows assignment := by
    exact ActiveIndexedRows.builderRows_satisfied_of_indexedRowsMatch
      FPrimeRecursiveYZcolProjectionData.sourceRows ownedRowDefinitions
      source_rows_match sourceSatisfies
  have normalizedBuilderRows :
      Satisfies (ownedDefinitions.map Program.Definition.builderRow)
        assignment := by
    simpa [ownedBuilderRows, ownedDefinitions, List.map_map]
      using builderRows
  have definitionsHold := Program.builderDefinitions_sound
    assignmentCanonical constantOne ownedDefinitions_canonical
    normalizedBuilderRows
  rw [ownedDefinitions_eq_evaluatorDefinitions] at definitionsHold
  exact definitionsHold

/-- One verifier-owned power ladder feeds both leaves. This is arithmetic
power validity only; transcript derivation of `point` remains a higher-level
obligation. -/
def SharedPowersValid (assignment : Nat → Nat)
    (point : ProjectionProgram.K) : Prop :=
  sharedPowerColumns.map (fun power => power.value assignment) =
    ProjectionProgram.K.powersFrom point ProjectionProgram.K.one
      sharedPowerColumns.length

theorem limb0_powers_of_shared
    {assignment : Nat → Nat} {point : ProjectionProgram.K}
    (shared : SharedPowersValid assignment point) :
    limb0Owner.evalTrace.PowersValid assignment point := by
  apply limb0Owner.evalTrace.powersValid_of_ladderPrefix assignment point
    sharedPowerColumns
  · decide
  · decide
  · exact shared

theorem limb1_powers_of_shared
    {assignment : Nat → Nat} {point : ProjectionProgram.K}
    (shared : SharedPowersValid assignment point) :
    limb1Owner.evalTrace.PowersValid assignment point := by
  apply limb1Owner.evalTrace.powersValid_of_ladderPrefix assignment point
    sharedPowerColumns
  · decide
  · decide
  · exact shared

/-- Decode the generated physical parent columns as the two canonical limb
coefficient lists of one independent semantic parent. This is a required
semantic binding premise, not a consequence of artifact labels. -/
def ParentColumnsMatch (assignment : Nat -> Nat)
    (parent : RingK) : Prop :=
  liftedColumns assignment limb0Owner.parentCoefficientColumns =
      projectionC0Coefficients (coefficients parent) ∧
    liftedColumns assignment limb1Owner.parentCoefficientColumns =
      projectionC1Coefficients (coefficients parent)

/-- The exact two evaluator leaves compute the canonical packed-parent
projection, conditional on their still-explicit row, power, and semantic
column-binding premises. -/
theorem evaluationDefinitions_refine_parentProjection
    {assignment : Nat -> Nat}
    {point : ProjectionProgram.K}
    {parent : RingK}
    (definitionsHold :
      ProjectionProgram.DefinitionsHold assignment evaluatorDefinitions)
    (sharedPowers : SharedPowersValid assignment point)
    (parentColumns : ParentColumnsMatch assignment parent) :
    K.add
        (toSemanticK (limb0Owner.evalTrace.output.value assignment))
        (K.mul extensionGenerator
          (toSemanticK (limb1Owner.evalTrace.output.value assignment))) =
      projectedValue parent (toSemanticK point) := by
  have limb0Definitions :
      ProjectionProgram.DefinitionsHold assignment
        limb0Owner.evalTrace.definitions := by
    intro definition member
    exact definitionsHold definition (by
      apply List.mem_append_left
      exact member)
  have limb1Definitions :
      ProjectionProgram.DefinitionsHold assignment
        limb1Owner.evalTrace.definitions := by
    intro definition member
    exact definitionsHold definition (by
      apply List.mem_append_right
      exact member)
  have limb0 := leaf_definitions_sound limb0Owner assignment point
    limb0_valid.layout (limb0_powers_of_shared sharedPowers)
    limb0Definitions
  have limb1 := leaf_definitions_sound limb1Owner assignment point
    limb1_valid.layout (limb1_powers_of_shared sharedPowers)
    limb1Definitions
  rw [parentColumns.1] at limb0
  rw [parentColumns.2] at limb1
  rw [limb0, limb1]
  exact (projectedValue_eq_limbEvaluations parent
    (toSemanticK point)).symm

/-- The exact 216 normalized production rows refine the semantic packed-parent
projection, conditional on the shared ladder and supplied-parent column
binding. -/
theorem ownedSourceRows_refine_parentProjection
    {assignment : Nat → Nat}
    {point : ProjectionProgram.K}
    {parent : RingK}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (sourceSatisfies : Satisfies ownedSourceRows assignment)
    (sharedPowers : SharedPowersValid assignment point)
    (parentColumns : ParentColumnsMatch assignment parent) :
    PairRightScalarMatches parent
      (K.add
        (toSemanticK (limb0Owner.evalTrace.output.value assignment))
        (K.mul extensionGenerator
          (toSemanticK
            (limb1Owner.evalTrace.output.value assignment))))
      (toSemanticK point) := by
  unfold PairRightScalarMatches
  exact evaluationDefinitions_refine_parentProjection
    (ownedSourceRows_definitionsHold assignmentCanonical constantOne
      sourceSatisfies)
    sharedPowers parentColumns

/-- Exact absolute-index embedding of the generated leaves in a larger R1CS
row list. This remains an explicit premise until the complete production row
artifact is available in Lean. -/
def SourceRowsEmbedded (fullRows : List Row) : Prop :=
  ActiveIndexedRows.SourceRowsEmbedded
    FPrimeRecursiveYZcolProjectionData.sourceRows fullRows

/-- Whole-row satisfaction specializes to the exact generated source leaves
when their absolute indices are supplied by a separate full-program bridge. -/
theorem ownedSourceRows_satisfied_of_embedded
    {fullRows : List Row} {assignment : Nat → Nat}
    (embedded : SourceRowsEmbedded fullRows)
    (fullSatisfies : Satisfies fullRows assignment) :
    Satisfies ownedSourceRows assignment :=
  ActiveIndexedRows.sourceRows_satisfied_of_embedded embedded fullSatisfies

/-- Conditional full-program entry point. The conclusion is still only the
packed-parent evaluator obligation; transcript and parent-opening authority
remain separate higher-level theorems. -/
theorem fullRows_refine_parentProjection
    {fullRows : List Row}
    {assignment : Nat → Nat}
    {point : ProjectionProgram.K}
    {parent : RingK}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (embedded : SourceRowsEmbedded fullRows)
    (fullSatisfies : Satisfies fullRows assignment)
    (sharedPowers : SharedPowersValid assignment point)
    (parentColumns : ParentColumnsMatch assignment parent) :
    PairRightScalarMatches parent
      (K.add
        (toSemanticK (limb0Owner.evalTrace.output.value assignment))
        (K.mul extensionGenerator
          (toSemanticK
            (limb1Owner.evalTrace.output.value assignment))))
      (toSemanticK point) :=
  ownedSourceRows_refine_parentProjection assignmentCanonical constantOne
    (ownedSourceRows_satisfied_of_embedded embedded fullSatisfies)
    sharedPowers parentColumns

end Nightstream.Implementation.R1CS.FPrimeRecursiveYZcolProjection.Refinement
