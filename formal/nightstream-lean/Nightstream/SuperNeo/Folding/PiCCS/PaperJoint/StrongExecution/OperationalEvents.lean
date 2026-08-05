import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution

/-!
Literal finite-experiment events for the causal paper `Pi_CCS` execution.

Owns: Boolean events for two-run witness disagreement, exact success-gated
source extraction, the fixed-first-witness bad event on a fresh second run,
and the pointwise Appendix-D.4 extraction cover.

Does not own: a probability distribution, rejection sampling, runtime,
Schwartz--Zippel or SumCheck probability bounds, Fiat--Shamir, Rust, R1CS,
artifacts, or costs.

Emits constraints: no.

The witness used in the bad event is read from the first execution.  The
second execution contributes only its independently generated causal prefix.
Thus neither root event can adapt its witness to the fresh second coins.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution

universe uExtension uCommitment uPublicInput

private noncomputable def propositionCheck (proposition : Prop) : Bool :=
  @ite Bool proposition (Classical.propDecidable proposition) true false

@[simp] private theorem propositionCheck_eq_true
    (proposition : Prop) :
    propositionCheck proposition = true <-> proposition := by
  simp [propositionCheck]

@[simp] private theorem propositionCheck_eq_false
    (proposition : Prop) :
    propositionCheck proposition = false <-> ¬ proposition := by
  simp [propositionCheck]

/-- The literal raw two-run disagreement event from Definition 10.  Both
executions must accept in the corrected ambient target relation, both target
witnesses must exist, and those witnesses must differ. -/
noncomputable def witnessDisagreement
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (executions : Execution Extension shape columns ×
      Execution Extension shape columns) : Bool :=
  ambientCheck context executions.1 && ambientCheck context executions.2 &&
    match executions.1.target, executions.2.target with
    | some left, some right => propositionCheck (left ≠ right)
    | _, _ => false

/-- Extraction is source membership of the witness fixed by the first
successful execution.  The fresh second execution is used only to expose a
bad root when this event is false. -/
noncomputable def sourceExtracted
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (executions : Execution Extension shape columns ×
      Execution Extension shape columns) : Bool :=
  match executions.1.target with
  | none => false
  | some witness => propositionCheck
      (SourceHolds context.extensionOps context.lift context.openingMaps
        context.params context.statement witness)

/-- Exact output event of the paper's success-gated extractor. The first
component is the successful retry witness fixed before the fresh initial-run
coins are analyzed. The fresh run must also succeed with the same witness,
and that retained witness must satisfy the source relation. -/
noncomputable def successGatedSourceExtracted
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (executions : Execution Extension shape columns ×
      Execution Extension shape columns) : Bool :=
  ambientCheck context executions.1 && ambientCheck context executions.2 &&
    match executions.1.target, executions.2.target with
    | some retained, some fresh => propositionCheck
        (retained = fresh /\
          SourceHolds context.extensionOps context.lift context.openingMaps
            context.params context.statement retained)
    | _, _ => false

/-- The exact fixed-first bad event on the fresh second execution. -/
noncomputable def fixedFirstBad
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (executions : Execution Extension shape columns ×
      Execution Extension shape columns) : Bool :=
  match executions.1.target with
  | none => false
  | some witness => propositionCheck
      (MixingFailure context executions.2.causalRun witness \/
        SumCheckFailure context executions.2.causalRun witness)

/-- The literal public-output projection mismatch event.  It is retained as
an actual event rather than defined to be zero; verifier construction proves
that it is always false. -/
noncomputable def outputPhiMismatch
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (executions : Execution Extension shape columns ×
      Execution Extension shape columns) : Bool :=
  propositionCheck
    (outputPhi (context.statement.publicOutput executions.1.causalRun.probe) ≠
      outputPhi (context.statement.publicOutput executions.2.causalRun.probe))

theorem witnessDisagreement_eq_true_iff
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (executions : Execution Extension shape columns ×
      Execution Extension shape columns) :
    witnessDisagreement context executions = true <->
      (AmbientSuccess context executions.1 /\
        AmbientSuccess context executions.2) /\
      exists left right,
        executions.1.target = some left /\
        executions.2.target = some right /\
        left ≠ right := by
  cases leftTarget : executions.1.target <;>
    cases rightTarget : executions.2.target <;>
    simp [witnessDisagreement, ambientCheck_eq_true_iff, AmbientSuccess,
      leftTarget, rightTarget]

theorem witnessDisagreement_implies_first_success
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (first second : Execution Extension shape columns)
    (disagreement : witnessDisagreement context (first, second) = true) :
    ambientCheck context first = true := by
  exact (ambientCheck_eq_true_iff context first).2
    ((witnessDisagreement_eq_true_iff context (first, second)).1
      disagreement).1.1

theorem witnessDisagreement_implies_second_success
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (first second : Execution Extension shape columns)
    (disagreement : witnessDisagreement context (first, second) = true) :
    ambientCheck context second = true := by
  exact (ambientCheck_eq_true_iff context second).2
    ((witnessDisagreement_eq_true_iff context (first, second)).1
      disagreement).1.2

theorem outputPhiMismatch_eq_false
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (executions : Execution Extension shape columns ×
      Execution Extension shape columns) :
    outputPhiMismatch context executions = false := by
  simp [outputPhiMismatch,
    repeatedPublicOutputs_same_phi context.statement
      executions.1.causalRun.probe executions.2.causalRun.probe]

/-- Two successful executions with no literal disagreement have the same
target witness.  This is derived from the event definitions; witness equality
is not an extraction premise. -/
theorem targets_eq_of_success_of_no_disagreement
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (first second : Execution Extension shape columns)
    (firstSuccess : ambientCheck context first = true)
    (secondSuccess : ambientCheck context second = true)
    (noDisagreement : witnessDisagreement context (first, second) = false) :
    exists left right,
      first.target = some left /\ second.target = some right /\ left = right := by
  have firstSemantic := (ambientCheck_eq_true_iff context first).1 firstSuccess
  have secondSemantic :=
    (ambientCheck_eq_true_iff context second).1 secondSuccess
  cases leftTarget : first.target with
  | none => simp [AmbientSuccess, leftTarget] at firstSemantic
  | some left =>
      cases rightTarget : second.target with
      | none => simp [AmbientSuccess, rightTarget] at secondSemantic
      | some right =>
          have notDifferent : ¬ (left ≠ right) := by
            have checkedFalse : propositionCheck (left ≠ right) = false := by
              simpa [witnessDisagreement, firstSuccess, secondSuccess,
                leftTarget, rightTarget] using noDisagreement
            exact (propositionCheck_eq_false (left ≠ right)).1 checkedFalse
          exact ⟨left, right, rfl, rfl,
            Classical.not_not.mp notDifferent⟩

/-- Appendix D.4's pointwise cover on the actual events.  The first witness
is fixed before the fresh second prefix is inspected.  If both executions
succeed and their witnesses do not disagree, the deterministic paper theorem
on the second prefix yields source truth or one of the two named failures. -/
theorem extraction_or_fixedFirstBad
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (first second : Execution Extension shape columns)
    (firstSuccess : ambientCheck context first = true)
    (secondSuccess : ambientCheck context second = true)
    (noDisagreement : witnessDisagreement context (first, second) = false) :
    sourceExtracted context (first, second) = true \/
      fixedFirstBad context (first, second) = true := by
  rcases targets_eq_of_success_of_no_disagreement context first second
      firstSuccess secondSuccess noDisagreement with
    ⟨firstWitness, secondWitness, firstTarget, secondTarget, witnessesEqual⟩
  subst secondWitness
  have secondSemantic :=
    (ambientCheck_eq_true_iff context second).1 secondSuccess
  have secondFacts :
      second.causalRun.probe.FixedWidthAccepted context.extensionOps
          context.lift context.statement context.sumcheckWidth /\
        AmbientOutputHolds context.extensionOps context.lift
          context.openingMaps context.params context.statement
          second.causalRun.probe firstWitness := by
    simpa [AmbientSuccess, secondTarget] using secondSemantic
  rcases acceptedPrefix_extracts_fixedWitness_or_badEvent context
      second.causalRun firstWitness secondFacts.2 secondFacts.1 with
    source | mixing | sumCheck
  · exact Or.inl (by
      simp [sourceExtracted, firstTarget, source])
  · exact Or.inr (by
      simp [fixedFirstBad, firstTarget, mixing])
  · exact Or.inr (by
      simp [fixedFirstBad, firstTarget, sumCheck])

/-- The same pointwise cover with the exact success-gated output event. -/
theorem successGatedExtraction_or_fixedFirstBad
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (first second : Execution Extension shape columns)
    (firstSuccess : ambientCheck context first = true)
    (secondSuccess : ambientCheck context second = true)
    (noDisagreement : witnessDisagreement context (first, second) = false) :
    successGatedSourceExtracted context (first, second) = true \/
      fixedFirstBad context (first, second) = true := by
  rcases extraction_or_fixedFirstBad context first second firstSuccess
      secondSuccess noDisagreement with extracted | bad
  · rcases targets_eq_of_success_of_no_disagreement context first second
      firstSuccess secondSuccess noDisagreement with
      ⟨retained, fresh, retainedTarget, freshTarget, equal⟩
    subst fresh
    have source :
        SourceHolds context.extensionOps context.lift context.openingMaps
          context.params context.statement retained := by
      simpa [sourceExtracted, retainedTarget] using extracted
    exact Or.inl (by
      simp [successGatedSourceExtracted, firstSuccess, secondSuccess,
        retainedTarget, freshTarget, source])
  · exact Or.inr bad

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents
