import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Completeness
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.DirectSemantics
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.DerivedProgram
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.RewriteProvenance
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.SourceArtifactFacts
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ArtifactRows
import Nightstream.Implementation.R1CS.Correspondence.Projection.ProjectionComplete

/-!
Independent honest source and abstract rewrite execution for the bounded
selective fixed-point `y_zcol` projection slice.

Owns: the exact honest source boundary, centered-ternary materialization of
its retained fields, reconstruction of the checked compiler closure,
and deterministic construction of the intermediate abstract product-sum
fields.

Does not own: producer authority for the honest source boundary, projection
bad-root soundness, transcript security, production-wide assignment
conformance, or permission to remove rows.

Emits constraints: no.

| Leaf | Mathematical obligation | Authority class |
|---|---|---|
| `honest.source_boundary` | an executable source run plus the two direct sampled wire equations | independent semantic input |
| `honest.retained_words` | every retained source value is encoded in its checked balanced slot | computed |
| `honest.compiler_closure` | decoded retained words determine every checked compiler definition | derived |
| `honest.abstract_rewrite` | intermediate product sums are constructed in checked rewrite order | computed |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Completeness

private abbrev certificate := Materialized.Checked.sourceArtifact.certificate

/-- Independent honest boundary for the focused source program. The source is
the output of deterministic execution from an unconstrained seed, and the only
semantic premises are constant one and the two direct sampled wire equations.
No source-row satisfaction, decoded equality, or selected-row acceptance occurs
here. -/
structure HonestSourceBoundary (source : Nat → Nat) : Type where
  seed : Nat → Nat
  seedOne : seed 0 = 1
  sourceEq : SourceProgram.sourceAssignment seed = source
  lowSampledWireEquation :
    (Checked.lowTrace.pairProductValues source).foldr
        ProjectionProgram.K.add ProjectionProgram.K.zero =
      ProjectionProgram.K.add
        (Checked.lowTrace.quotientPhiProduct.output.value source)
        (Checked.lowTrace.outputEvaluation.output.value source)
  highSampledWireEquation :
    (Checked.highTrace.pairProductValues source).foldr
        ProjectionProgram.K.add ProjectionProgram.K.zero =
      ProjectionProgram.K.add
        (Checked.highTrace.quotientPhiProduct.output.value source)
        (Checked.highTrace.outputEvaluation.output.value source)

namespace HonestSourceBoundary

theorem canonical {source : Nat → Nat}
    (honest : HonestSourceBoundary source) :
    ∀ column, source column < goldilocksP := by
  rw [← honest.sourceEq]
  exact SourceProgram.sourceAssignmentCanonical honest.seed

theorem constantOne {source : Nat → Nat}
    (honest : HonestSourceBoundary source) : source 0 = 1 := by
  rw [← honest.sourceEq]
  exact SourceProgram.sourceAssignmentConstantOne honest.seedOne

theorem compilerDefinitions {source : Nat → Nat}
    (honest : HonestSourceBoundary source) :
    ∀ definition ∈ SourceDecode.compilerDefinitions,
      definition.Holds source := by
  rw [← honest.sourceEq]
  exact SourceProgram.sourceAssignmentCompilerDefinitionsHold honest.seedOne

theorem sourceDefinitions {source : Nat → Nat}
    (honest : HonestSourceBoundary source) :
    ProjectionProgram.DefinitionsHold source
      SourceProgram.sourceDefinitions := by
  rw [← honest.sourceEq]
  exact SourceProgram.sourceAssignmentDefinitionsHold honest.seed

theorem finalChecks {source : Nat → Nat}
    (honest : HonestSourceBoundary source) :
    Satisfies certificate.checks source := by
  have lowChecks := Checked.lowTrace.checks_complete source
    honest.constantOne honest.lowSampledWireEquation
  have highChecks := Checked.highTrace.checks_complete source
    honest.constantOne honest.highSampledWireEquation
  have traceChecks :
      Satisfies
        (Checked.traces.flatMap ProjectionProgram.ProjectionTrace.checks)
        source := by
    change Satisfies
      (Checked.lowTrace.checks ++ Checked.highTrace.checks) source
    intro row member
    simp only [List.mem_append] at member
    rcases member with low | high
    · exact lowChecks row low
    · exact highChecks row high
  have coverage := ArtifactRows.certificate_covers Checked.structureValid
  intro row member
  exact traceChecks row ((coverage.checksIff row).mp member)

end HonestSourceBoundary

def materializedValues (source derived : Nat → Nat) : SlotOwner → Nat
  | .source column => source column
  | .derived compilerIndex => derived compilerIndex % goldilocksP

def materializedAssignment (source derived : Nat → Nat) : Nat → Nat :=
  materializeAssignment (materializedValues source derived)

theorem materializedAssignment_constantOne (source derived : Nat → Nat) :
    materializedAssignment source derived
      Materialized.Checked.constantOneColumn = 1 :=
  Completeness.materializeAssignment_constantOne _

theorem materializedAssignment_selectorOne (source derived : Nat → Nat) :
    materializedAssignment source derived
      Materialized.Checked.steadySelectorColumn = 1 :=
  Completeness.materializeAssignment_selectorOne _

theorem materializedAssignment_canonical (source derived : Nat → Nat) :
    AssignmentCanonical (materializedAssignment source derived) :=
  Completeness.materializeAssignment_canonical _

private theorem retainedSeed_agrees
    {source derived : Nat → Nat}
    (honest : HonestSourceBoundary source) :
    AgreeOn
      (SourceDecode.retainedSeed (materializedAssignment source derived))
      source SourceDecode.compilerKnownColumns := by
  intro column member
  simp only [SourceDecode.compilerKnownColumns, List.mem_cons,
    List.mem_map] at member
  rcases member with rfl | retained
  · have constantColumn : Materialized.Checked.constantOneColumn = 0 := by
      exact constantOneColumnZero
    have encodedOne := materializedAssignment_constantOne source derived
    rw [constantColumn] at encodedOne
    rw [honest.constantOne]
    simp [SourceDecode.retainedSeed, encodedOne, goldilocksP]
  · rcases retained with ⟨slot, slotMember, rfl⟩
    have positive := slot.columnPositive
    have nonzero : slot.column ≠ 0 := Nat.ne_of_gt positive
    rcases retainedSlotFast_exists slot slotMember with
      ⟨found, lookup, foundMember, foundColumn⟩
    simp only [SourceDecode.retainedSeed, nonzero, if_false, lookup]
    simpa [materializedAssignment, materializedValues, foundColumn] using
      (Completeness.sourceSlot_decodes (materializedValues source derived)
        found foundMember (honest.canonical found.column))

/-- The centered words reconstruct exactly the honest compiler assignment on
every declared input and derived compiler column. This is proved from source
definitions and slot decoding, not assumed as decoded equality. -/
theorem compilerAssignment_agrees
    {source derived : Nat → Nat}
    (honest : HonestSourceBoundary source) :
    AgreeOn
      (SourceDecode.compilerAssignment (materializedAssignment source derived))
      source
      (knownAfter SourceDecode.compilerKnownColumns
        SourceDecode.compilerDefinitions) := by
  exact run_agrees_of_holds SourceDecode.compilerProgramWellFormed
    (retainedSeed_agrees honest) honest.compilerDefinitions

/-! ## Deterministic intermediate product-sum program -/

def abstractSourceValue (source : Nat → Nat)
    (linear : SourceDecode.DecodedSourceLinearCombination) : F :=
  Materialized.Semantics.fieldResidue
    (lcEval source linear.programTerms)

def abstractFactorValue (source : Nat → Nat)
    (factor : DecodedProductFactor) : F :=
  Materialized.Semantics.fieldResidue factor.coefficient *
    abstractSourceValue source factor.left *
    abstractSourceValue source factor.right

def abstractFactorValueAt (source : Nat → Nat)
    (factors : List DecodedProductFactor) (index : Nat) : F :=
  match factors[index]? with
  | none => 0
  | some factor => abstractFactorValue source factor

def abstractFactorSum (source : Nat → Nat)
    (factors : List DecodedProductFactor) : F :=
  abstractFactorValueAt source factors 0 +
    abstractFactorValueAt source factors 1 +
    abstractFactorValueAt source factors 2 +
    abstractFactorValueAt source factors 3 +
    abstractFactorValueAt source factors 4

def derivedPreviousValue (state : Nat → F) :
    Option DecodedDerivedSlot → F
  | none => 0
  | some slot => state slot.compilerIndex

def derivedRhs (source : Nat → Nat) (state : Nat → F)
    (step : DecodedRewriteStep) : F :=
  abstractSourceValue source step.base +
    derivedPreviousValue state step.previous +
    abstractFactorSum source step.factors

def setDerived (state : Nat → F) (column : Nat) (value : F) : Nat → F :=
  fun candidate => if candidate = column then value else state candidate

def executeDerived (source : Nat → Nat) (state : Nat → F)
    (step : DecodedRewriteStep) : Nat → F :=
  match step.output with
  | .source _ => state
  | .derivedProductSum slot =>
      setDerived state slot.compilerIndex (derivedRhs source state step)

def runDerived (source : Nat → Nat) :
    (Nat → F) → List DecodedRewriteStep → Nat → F
  | state, [] => state
  | state, step :: rest =>
      runDerived source (executeDerived source state step) rest

def initialDerivedState : Nat → F := fun _ => 0

def derivedAssignment (source : Nat → Nat) : Nat → F :=
  runDerived source initialDerivedState decodedRewriteSteps

def DerivedAgreeOn (left right : Nat → F) (known : List Nat) : Prop :=
  ∀ column ∈ known, left column = right column

private theorem setDerived_same (state : Nat → F) (column : Nat)
    (value : F) : setDerived state column value column = value := by
  simp [setDerived]

private theorem setDerived_other (state : Nat → F) {column other : Nat}
    (value : F) (different : other ≠ column) :
    setDerived state column value other = state other := by
  simp [setDerived, different]

private theorem executeDerived_preserves
    {source : Nat → Nat} {state : Nat → F}
    {known : List Nat} {step : DecodedRewriteStep}
    (fresh : match step.output with
      | .source _ => True
      | .derivedProductSum slot => slot.compilerIndex ∉ known) :
    DerivedAgreeOn (executeDerived source state step) state known := by
  intro column member
  cases outputEq : step.output with
  | source output => simp [executeDerived, outputEq]
  | derivedProductSum slot =>
      unfold executeDerived
      rw [outputEq]
      apply setDerived_other
      intro equal
      have freshSlot : slot.compilerIndex ∉ known := by
        simpa [outputEq] using fresh
      apply freshSlot
      rw [← equal]
      exact member

private theorem runDerived_preserves
    {source : Nat → Nat} {state : Nat → F}
    {known : List Nat} {steps : List DecodedRewriteStep}
    (valid : DerivedWellFormed known steps) :
    DerivedAgreeOn (runDerived source state steps) state known := by
  induction steps generalizing known state with
  | nil => intro _ _; rfl
  | cons step rest inductionHypothesis =>
    cases valid with
    | source previous isSource tail =>
      simpa [runDerived, executeDerived, isSource] using
        inductionHypothesis (state := state) tail
    | derived previous isDerived fresh tail =>
      intro column member
      have tailPreserves := inductionHypothesis
        (state := executeDerived source state step) tail column
        (by simp [member])
      exact tailPreserves.trans
        (executeDerived_preserves (by simpa [isDerived] using fresh)
          column member)

def AbstractStepHolds (source : Nat → Nat) (derived : Nat → F)
    (step : DecodedRewriteStep) : Prop :=
  match step.output with
  | .source output =>
      abstractSourceValue source output = derivedRhs source derived step
  | .derivedProductSum slot =>
      derived slot.compilerIndex = derivedRhs source derived step

/-- Terminal equations are evaluated during deterministic forward execution.
This predicate contains no selected rows or encoded assignment. The next
section derives it from the independently stated source family equations. -/
def TerminalsHoldFrom (source : Nat → Nat) :
    (Nat → F) → List DecodedRewriteStep → Prop
  | _, [] => True
  | state, step :: rest =>
      (match step.output with
        | .source output =>
            abstractSourceValue source output = derivedRhs source state step
        | .derivedProductSum _ => True) ∧
      TerminalsHoldFrom source (executeDerived source state step) rest

structure TerminalsEvidence (source : Nat → Nat) (initial : Nat → F)
    (steps : List DecodedRewriteStep) : Prop where
  holds : TerminalsHoldFrom source initial steps

abbrev RewriteTerminalsHold (source : Nat → Nat) : Prop :=
  TerminalsEvidence source initialDerivedState decodedRewriteSteps

private theorem previousValue_eq_of_agree
    {left right : Nat → F} {known : List Nat}
    (agreement : DerivedAgreeOn left right known)
    (previous : Option DecodedDerivedSlot)
    (knownPrevious : match previous with
      | none => True
      | some slot => slot.compilerIndex ∈ known) :
    derivedPreviousValue left previous = derivedPreviousValue right previous := by
  cases previous with
  | none => rfl
  | some slot => exact agreement slot.compilerIndex knownPrevious

private theorem derivedRhs_eq_of_agree
    {source : Nat → Nat} {left right : Nat → F} {known : List Nat}
    (agreement : DerivedAgreeOn left right known)
    (step : DecodedRewriteStep) (previous : PreviousKnown known step) :
    derivedRhs source left step = derivedRhs source right step := by
  unfold derivedRhs
  rw [previousValue_eq_of_agree agreement step.previous previous]

structure AbstractStepsHold (source : Nat → Nat) (derived : Nat → F)
    (steps : List DecodedRewriteStep) : Prop where
  holds : ∀ step ∈ steps, AbstractStepHolds source derived step

private theorem runDerived_steps_hold
    {source : Nat → Nat} {initial : Nat → F}
    {known : List Nat} {steps : List DecodedRewriteStep}
    (valid : DerivedWellFormed known steps)
    (terminals : TerminalsEvidence source initial steps) :
    AbstractStepsHold source (runDerived source initial steps) steps := by
  rcases terminals with ⟨terminals⟩
  constructor
  induction steps generalizing known initial with
  | nil => intro step member; simp at member
  | cons head rest inductionHypothesis =>
    cases valid with
    | source previous isSource tail =>
      intro step member
      simp only [TerminalsHoldFrom, isSource] at terminals
      simp only [List.mem_cons] at member
      rcases member with rfl | inTail
      · simp only [AbstractStepHolds, isSource]
        rw [terminals.1]
        apply derivedRhs_eq_of_agree
        · intro column columnMember
          symm
          simpa [runDerived, executeDerived, isSource] using
            runDerived_preserves (source := source) (state := initial) tail
              column columnMember
        · exact previous
      · simpa [runDerived, executeDerived, isSource] using
          inductionHypothesis tail terminals.2 step inTail
    | @derived _ _ _ slot previous isDerived fresh tail =>
      intro step member
      simp only [TerminalsHoldFrom, isDerived] at terminals
      simp only [List.mem_cons] at member
      rcases member with rfl | inTail
      · simp only [AbstractStepHolds, isDerived]
        have outputPreserved := runDerived_preserves
          (source := source)
          (state := executeDerived source initial step) tail
          slot.compilerIndex (by simp)
        rw [runDerived, outputPreserved]
        simp only [executeDerived, isDerived, setDerived_same]
        apply derivedRhs_eq_of_agree
        · intro column columnMember
          have finalPreserves := runDerived_preserves
            (source := source)
            (state := executeDerived source initial step)
            (known := slot.compilerIndex :: known) tail
            column (List.mem_cons_of_mem _ columnMember)
          symm
          simpa only [executeDerived, isDerived] using
            finalPreserves.trans
              (executeDerived_preserves (by simpa [isDerived] using fresh)
                column columnMember)
        · exact previous
      · simpa [runDerived] using
          inductionHypothesis tail terminals.2 step inTail

theorem constructedAbstractStepsHold {source : Nat → Nat}
    (terminals : RewriteTerminalsHold source) :
    AbstractStepsHold source (derivedAssignment source)
      decodedRewriteSteps := by
  change AbstractStepsHold source
    (runDerived source initialDerivedState decodedRewriteSteps)
    decodedRewriteSteps
  apply runDerived_steps_hold
    (source := source) (initial := initialDerivedState)
    (known := []) (steps := decodedRewriteSteps)
  · exact derivedProgramWellFormed
  · exact terminals


end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
