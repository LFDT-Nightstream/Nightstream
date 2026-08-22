import NightstreamFPrime.Spec.Folding.PiDEC

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiDEC/PaperVerifier.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Paper-exact public verifier for SuperNeo `Pi_DEC`.

Protocol: SuperNeo Section 7.5.
Phase: one combined `CE(B)` parent to exactly `k` fresh `CE(b)` children.
Constraint family: semantic verifier equations only; this file emits no rows.

Assurance tier: model-level.

Owns: verifier-computed child public inputs, copied structure and point,
fixed evaluation-message arity, prover-supplied child commitment/evaluation
messages, the fixed-arity tuple boundary and two paper equations, projection
to the recomposition verifier, reduction of knowledge, and honest completeness.

Does not own: deterministic child commitments or evaluations, child CE
membership as a verifier check, commitment binding, extraction, a concrete
radix implementation, transcript binding, Rust/R1CS refinement, costs, or row
removal.

Emits constraints: no.

Authority boundary: the prover sends only child commitments and evaluation
arrays. The verifier computes every child public input from the public parent,
copies the parent structure and point, marks each output fresh, and checks only
commitment and evaluation recomposition. `PiDEC.Accepted` remains the weaker
recomposition relation used by the existing knowledge theorem.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.paper.public_split` | child `i` public input is `split_b(parent.x)_i` | computed | `PublicInputSplit.split`, `children` |
| `nifs.pi_dec.paper.public_split.recompose` | the public split recomposes to the parent input | derived | `PublicInputSplit.recompose_split` |
| `nifs.pi_dec.paper.public_split.projection` | splitting commutes with authoritative public projection | derived | `PublicInputSplit.split_project` |
| `nifs.pi_dec.paper.copy` | structure and point are copied and stage is fresh | computed | `children` |
| `nifs.pi_dec.paper.commitment` | parent commitment recomposes from prover messages | checked | `Accepted.commitmentEquation` |
| `nifs.pi_dec.paper.evaluation_arity` | parent and every child message contain exactly the structure-owned `t` evaluations | checked/typed | `EvaluationArity`, `Accepted.parentEvaluationSize`, `Accepted.messageEvaluationSize` |
| `nifs.pi_dec.paper.evaluations` | all `t` parent evaluations recompose from prover messages | checked | `Accepted.evaluationEquation` |
| `nifs.pi_dec.paper.relaxed` | paper output satisfies public recomposition acceptance | derived | `Accepted.toRecompositionAccepted` |
| `nifs.pi_dec.paper.complete` | the honest private split satisfies the paper verifier | derived | `complete` |
-/

namespace NightstreamFPrime.Spec.Folding.PiDEC.PaperVerifier

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment

/-- Verifier-owned public radix split and the laws relating it to the existing
assignment/recomposition algebra. -/
structure PublicInputSplit
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params) where
  split : PublicInput → Fin params.k → PublicInput
  recompose_split : ∀ input,
    algebra.recomposePublicInput (split input) = input
  split_project : ∀ assignment child,
    split (semantics.projectPublicInput assignment) child =
      semantics.projectPublicInput (algebra.splitAssignment assignment child)

/-- Verifier-owned evaluation-vector arity. The paper uses a statically sized
`t`-tuple; this sidecar makes that typing obligation explicit because the
generic `CE.Instance` carrier stores evaluations in an `Array`. -/
structure EvaluationArity
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment) where
  count : Structure → Nat
  evaluations_size : ∀ system assignment point,
    (semantics.evaluations system assignment point).size = count system

/-- The exact prover message in Section 7.5. Public inputs, structure, point,
and norm stage are absent because the verifier computes them. -/
structure ChildMessage
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment) where
  commitment : Commitment
  evaluations : Array Evaluation

/-- Public parent plus the `k` prover messages consumed by the paper verifier. -/
structure Attempt
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (params : GlobalParams) where
  parent : CE.Instance Structure PublicInput Point Evaluation Commitment
  messages : Fin params.k → ChildMessage Evaluation Commitment

/-- The full output family computed by the verifier from its parent and the
prover messages. -/
def children
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    (publicSplit : PublicInputSplit algebra)
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment params) :
    Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment :=
  fun child => {
    constraintSystem := attempt.parent.constraintSystem
    commitment := (attempt.messages child).commitment
    publicInput := publicSplit.split attempt.parent.publicInput child
    point := attempt.parent.point
    evaluations := (attempt.messages child).evaluations
    stage := .fresh
  }

@[simp] theorem children_publicInput
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    (publicSplit : PublicInputSplit algebra)
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment params)
    (child : Fin params.k) :
    (children publicSplit attempt child).publicInput =
      publicSplit.split attempt.parent.publicInput child := by
  rfl

/-- The Section-7.5 fixed-tuple shape, its two arithmetic equations, and the
required combined input stage. All child fields omitted here are computed by
`children`. The size fields model static paper typing; a concrete fixed-layout
refinement may discharge them without arithmetic constraints. -/
structure Accepted
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (evaluationArity : EvaluationArity semantics)
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment
      params) : Prop where
  parentCombined : attempt.parent.stage = .combined
  parentEvaluationSize :
    attempt.parent.evaluations.size =
      evaluationArity.count attempt.parent.constraintSystem
  messageEvaluationSize : ∀ child,
    (attempt.messages child).evaluations.size =
      evaluationArity.count attempt.parent.constraintSystem
  commitmentEquation :
    attempt.parent.commitment =
      algebra.recomposeCommitment fun child =>
        (attempt.messages child).commitment
  evaluationEquation :
    attempt.parent.evaluations =
      algebra.recomposeEvaluations fun child =>
        (attempt.messages child).evaluations

/-- Forget the operational message boundary and expose the full parent/child
family expected by the recomposition model. -/
def toRecompositionAttempt
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    (publicSplit : PublicInputSplit algebra)
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment params) :
    PiDEC.Attempt Structure PublicInput Point Evaluation Commitment params := {
  parent := attempt.parent
  children := children publicSplit attempt
}

/-- Extract exactly the message fields from a candidate full output family. -/
def messagesOf
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    (output : Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment) :
    Fin params.k → ChildMessage Evaluation Commitment :=
  fun child => {
    commitment := (output child).commitment
    evaluations := (output child).evaluations
  }

/-- Operational attempt induced by a parent and a candidate full output. -/
def attemptForOutput
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (output : Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment) :
    Attempt Structure PublicInput Point Evaluation Commitment params := {
  parent := parent
  messages := messagesOf output
}

/-- Exact relation between one accepted operational verifier run and its full
computed output. `checks` contains fixed tuple arities and the two paper
equations; the separate equality records that the candidate family is the
verifier's constructed result. -/
structure OutputAccepted
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (publicSplit : PublicInputSplit algebra)
    (evaluationArity : EvaluationArity semantics)
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (output : Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment) : Prop where
  outputComputed :
    children publicSplit (attemptForOutput parent output) = output
  checks : Accepted algebra evaluationArity (attemptForOutput parent output)

namespace Accepted

/-- The operational paper verifier refines the established recomposition
relation. The reverse implication is intentionally false in general. -/
theorem toRecompositionAccepted
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    {evaluationArity : EvaluationArity semantics}
    {attempt : Attempt Structure PublicInput Point Evaluation Commitment params}
    (accepted : Accepted algebra evaluationArity attempt)
    (publicSplit : PublicInputSplit algebra) :
    PiDEC.Accepted algebra (toRecompositionAttempt publicSplit attempt) := {
  parentCombined := accepted.parentCombined
  childFresh := fun _ => rfl
  sameStructure := fun _ => rfl
  samePoint := fun _ => rfl
  commitmentEquation := accepted.commitmentEquation
  publicInputEquation :=
    (publicSplit.recompose_split attempt.parent.publicInput).symm
  evaluationEquation := accepted.evaluationEquation
}

end Accepted

namespace OutputAccepted

/-- Every exact output carries the verifier-computed child public input. -/
theorem childPublicInput_eq
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    {publicSplit : PublicInputSplit algebra}
    {evaluationArity : EvaluationArity semantics}
    {parent : CE.Instance Structure PublicInput Point Evaluation Commitment}
    {output : Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    (accepted : OutputAccepted algebra publicSplit evaluationArity parent output)
    (child : Fin params.k) :
    (output child).publicInput =
      publicSplit.split parent.publicInput child := by
  have childEq := congrFun accepted.outputComputed child
  calc
    (output child).publicInput =
        (children publicSplit (attemptForOutput parent output) child).publicInput :=
      (congrArg CE.Instance.publicInput childEq).symm
    _ = publicSplit.split parent.publicInput child := rfl

/-- Exact outputs over equal parent public inputs have identical child public
inputs, independently of their prover-supplied commitments/evaluations. -/
theorem publicInputs_eq_of_parentPublicInput_eq
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    {publicSplit : PublicInputSplit algebra}
    {evaluationArity : EvaluationArity semantics}
    {leftParent rightParent :
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    {leftOutput rightOutput : Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    (leftAccepted :
      OutputAccepted algebra publicSplit evaluationArity leftParent leftOutput)
    (rightAccepted :
      OutputAccepted algebra publicSplit evaluationArity rightParent rightOutput)
    (parentPublicInputEq :
      leftParent.publicInput = rightParent.publicInput) :
    (fun child => (leftOutput child).publicInput) =
      (fun child => (rightOutput child).publicInput) := by
  funext child
  rw [leftAccepted.childPublicInput_eq child,
    rightAccepted.childPublicInput_eq child, parentPublicInputEq]

/-- Exact outputs contain exactly the structure-owned number of evaluation
claims; trailing or default-filled entries cannot pass this relation. -/
theorem childEvaluations_size
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    {publicSplit : PublicInputSplit algebra}
    {evaluationArity : EvaluationArity semantics}
    {parent : CE.Instance Structure PublicInput Point Evaluation Commitment}
    {output : Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    (accepted :
      OutputAccepted algebra publicSplit evaluationArity parent output)
    (child : Fin params.k) :
    (output child).evaluations.size =
      evaluationArity.count parent.constraintSystem := by
  exact accepted.checks.messageEvaluationSize child

/-- The public parent also contains exactly the structure-owned number of
evaluation claims. -/
theorem parentEvaluations_size
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    {publicSplit : PublicInputSplit algebra}
    {evaluationArity : EvaluationArity semantics}
    {parent : CE.Instance Structure PublicInput Point Evaluation Commitment}
    {output : Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    (accepted :
      OutputAccepted algebra publicSplit evaluationArity parent output) :
    parent.evaluations.size =
      evaluationArity.count parent.constraintSystem :=
  accepted.checks.parentEvaluationSize

/-- Exact full-output paper acceptance implies the established recomposition
predicate over that same parent and output family. -/
theorem toRecompositionAccepted
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    {publicSplit : PublicInputSplit algebra}
    {evaluationArity : EvaluationArity semantics}
    {parent : CE.Instance Structure PublicInput Point Evaluation Commitment}
    {output : Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    (accepted :
      OutputAccepted algebra publicSplit evaluationArity parent output) :
    PiDEC.Accepted algebra { parent := parent, children := output } := by
  have relaxed := accepted.checks.toRecompositionAccepted publicSplit
  change PiDEC.Accepted algebra {
    parent := parent
    children := children publicSplit (attemptForOutput parent output)
  } at relaxed
  rw [accepted.outputComputed] at relaxed
  exact relaxed

end OutputAccepted

/-- Equal parent public inputs produce equal child public inputs. This is the
children-from-parent direction missing from the recomposition-only relation;
it deliberately says nothing about prover-supplied commitments/evaluations. -/
theorem childPublicInputs_eq_of_parentPublicInput_eq
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    (publicSplit : PublicInputSplit algebra)
    (left right : Attempt Structure PublicInput Point Evaluation Commitment
      params)
    (parentPublicInputEq :
      left.parent.publicInput = right.parent.publicInput) :
    (fun child => (children publicSplit left child).publicInput) =
      (fun child => (children publicSplit right child).publicInput) := by
  funext child
  exact congrArg (fun input => publicSplit.split input child)
    parentPublicInputEq

/-- Honest messages computed from the private parent split. -/
def honestMessages
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment) :
    Fin params.k → ChildMessage Evaluation Commitment :=
  fun child => {
    commitment := semantics.commit (algebra.splitAssignment assignment child)
    evaluations := semantics.evaluations parent.constraintSystem
      (algebra.splitAssignment assignment child) parent.point
  }

/-- Honest paper-verifier attempt for one valid combined parent opening. -/
def honestAttempt
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment) :
    Attempt Structure PublicInput Point Evaluation Commitment params := {
  parent := parent
  messages := honestMessages algebra parent assignment
}

/-- With an authoritative parent public opening, operational honest children
are exactly the established private-split children. -/
theorem honestChildren_eq_childrenOf
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (publicSplit : PublicInputSplit algebra)
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment)
    (parentPublicInput :
      semantics.projectPublicInput assignment = parent.publicInput) :
    children publicSplit (honestAttempt algebra parent assignment) =
      PiDEC.childrenOf algebra parent assignment := by
  funext child
  have publicInputEq :
      publicSplit.split parent.publicInput child =
        semantics.projectPublicInput
          (algebra.splitAssignment assignment child) := by
    calc
      publicSplit.split parent.publicInput child =
          publicSplit.split
            (semantics.projectPublicInput assignment) child := by
        rw [parentPublicInput]
      _ = semantics.projectPublicInput
          (algebra.splitAssignment assignment child) :=
        publicSplit.split_project assignment child
  simp [children, honestAttempt, honestMessages, PiDEC.childrenOf,
    publicInputEq]

/-- Honest completeness of the exact operational paper verifier. Child CE
membership is a theorem about the output, not an extra runtime check. -/
theorem complete
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (publicSplit : PublicInputSplit algebra)
    (evaluationArity : EvaluationArity semantics)
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment)
    (parentCombined : parent.stage = .combined)
    (parentValid : CE.Holds semantics params parent assignment) :
    let attempt := honestAttempt algebra parent assignment
    Accepted algebra evaluationArity attempt ∧
      ∀ child, CE.Holds semantics params (children publicSplit attempt child)
        (algebra.splitAssignment assignment child) := by
  dsimp only
  have recomposition := PiDEC.complete semantics params algebra parent
    assignment parentCombined parentValid
  have childrenEq := honestChildren_eq_childrenOf algebra publicSplit parent
    assignment parentValid.1.2.1
  constructor
  · exact {
      parentCombined := recomposition.1.parentCombined
      parentEvaluationSize := by
        simp only [honestAttempt]
        rw [← parentValid.2.2]
        exact evaluationArity.evaluations_size _ _ _
      messageEvaluationSize := by
        intro child
        simpa only [honestAttempt, honestMessages] using
          evaluationArity.evaluations_size parent.constraintSystem
            (algebra.splitAssignment assignment child) parent.point
      commitmentEquation := recomposition.1.commitmentEquation
      evaluationEquation := recomposition.1.evaluationEquation
    }
  · intro child
    rw [childrenEq]
    exact recomposition.2 child

/-- Full-output form of honest completeness, with the established canonical
private children used only as the honest output constructor. -/
theorem output_complete
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (publicSplit : PublicInputSplit algebra)
    (evaluationArity : EvaluationArity semantics)
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment)
    (parentCombined : parent.stage = .combined)
    (parentValid : CE.Holds semantics params parent assignment) :
    let output := PiDEC.childrenOf algebra parent assignment
    OutputAccepted algebra publicSplit evaluationArity parent output ∧
      ∀ child, CE.Holds semantics params (output child)
        (algebra.splitAssignment assignment child) := by
  dsimp only
  have operational := complete semantics params algebra publicSplit
    evaluationArity parent assignment parentCombined parentValid
  refine ⟨{
    outputComputed := ?_
    checks := ?_
  }, ?_⟩
  · exact honestChildren_eq_childrenOf algebra publicSplit parent assignment
      parentValid.1.2.1
  · simpa [attemptForOutput, messagesOf, honestAttempt, honestMessages,
      PiDEC.childrenOf] using operational.1
  · intro child
    have childrenEq := honestChildren_eq_childrenOf algebra publicSplit parent
      assignment parentValid.1.2.1
    rw [← childrenEq]
    exact operational.2 child

/-- Reduction of knowledge for the exact operational verifier, inherited only
after deriving the weaker recomposition equations. -/
theorem reduce_knowledge
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (publicSplit : PublicInputSplit algebra)
    (evaluationArity : EvaluationArity semantics)
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment params)
    (childAssignments : Fin params.k → Assignment)
    (kPositive : 0 < params.k)
    (accepted : Accepted algebra evaluationArity attempt)
    (childrenValid : ∀ child,
      CE.Holds semantics params (children publicSplit attempt child)
        (childAssignments child)) :
    CE.Holds semantics params attempt.parent
      (algebra.recomposeAssignment childAssignments) :=
  PiDEC.reduce_knowledge semantics params algebra
    (toRecompositionAttempt publicSplit attempt) childAssignments kPositive
    (accepted.toRecompositionAccepted publicSplit) childrenValid

/-- The parent-opening binding dichotomy also transfers through the exact
operational verifier. -/
theorem parent_eq_recompose_or_bindingCollision
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (publicSplit : PublicInputSplit algebra)
    (evaluationArity : EvaluationArity semantics)
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment params)
    (parentAssignment : Assignment)
    (childAssignments : Fin params.k → Assignment)
    (accepted : Accepted algebra evaluationArity attempt)
    (parentValid : CE.Holds semantics params attempt.parent parentAssignment)
    (childrenValid : ∀ child,
      CE.Holds semantics params (children publicSplit attempt child)
        (childAssignments child)) :
    parentAssignment = algebra.recomposeAssignment childAssignments ∨
      Nonempty (PiDEC.ParentOpeningBindingCollision semantics params
        attempt.parent.commitment) :=
  PiDEC.accepted_parent_eq_recompose_or_bindingCollision semantics params
    algebra (toRecompositionAttempt publicSplit attempt) parentAssignment
    childAssignments (accepted.toRecompositionAccepted publicSplit)
    parentValid childrenValid

end NightstreamFPrime.Spec.Folding.PiDEC.PaperVerifier
