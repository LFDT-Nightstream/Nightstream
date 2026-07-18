import Nightstream.SuperNeo.CheckPlan
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold

/-!
Exact obligation tree for one certificate-independent concrete Phi81 NIFS
realization.

Protocol: SuperNeo NIFS.
Phases: incoming authority, `Pi_CCS`, `Pi_RLC`, and `Pi_DEC`.
Constraint families: mathematical obligations only; this file emits no rows.

Assurance tier: model-level.

Owns: one raw candidate language, stable protocol/phase/family ownership for
the retained semantic leaves, and exact equivalence between accepting every
leaf and satisfying the witness-indexed `SemanticFold.Realization` at the
candidate's exact point and challenge vector.

Does not own: executable transcript replay, SumCheck soundness, sampler
shortfall probability, child-opening extraction, per-leaf necessity,
Rust/R1CS refinement, physical rows, costs, or row removal.

Emits constraints: no.

Authority boundary: `Candidate` carries raw point and challenge data without
proofs. In particular, strong-set membership is a removable leaf rather than
an invariant hidden in the carrier type. `target` also binds the exact raw
witness used by the candidate; it cannot be satisfied by silently choosing a
different point or challenge vector. Physical transcript and extraction
checks remain separate refinement/security boundaries.

The incoming-authority leaf validates strict parent/child recomposition for
the child vector already present in the candidate. It does not prove that a
parent-only digest binds that vector across F-prime steps; that is a separate
outer authority obligation and has an exact production-profile
counterexample.

| Phase | Family | Leaf stage path | Mathematical obligation | Authority class |
|---|---|---|---|---|
| `Pi_CCS` | paper relation | `nifs.semantic.pi_ccs.relation.fresh_ccs` | every fresh assignment satisfies CCS | checked |
| `Pi_CCS` | paper relation | `nifs.semantic.pi_ccs.relation.all_source_norm` | every source coordinate has strict norm | checked |
| `Pi_CCS` | paper relation | `nifs.semantic.pi_ccs.relation.carried_evaluations` | every running claim equals its prior evaluation | checked |
| `Pi_CCS` | input authority | `nifs.semantic.pi_ccs.input.polynomial` | polynomial input is the source projection | checked |
| `Pi_CCS` | input authority | `nifs.semantic.pi_ccs.input.product` | public source product opens the same sources | checked |
| incoming | accumulator authority | `nifs.semantic.running.authority` | bootstrap absence or strict active-parent recomposition | checked |
| `Pi_RLC` | challenge | `nifs.semantic.pi_rlc.challenge.strong_set` | every raw scalar belongs to the strong set | checked |
| `Pi_RLC` | parent | `nifs.semantic.pi_rlc.parent.exact` | public parent equals the canonical combination | computed |
| `Pi_DEC` | children | `nifs.semantic.pi_dec.children.exact` | public children equal the canonical radix split | computed |

The child-owner files expose the next level without duplicating it here:
`Semantics.Paper` splits the three relation leaves, `InputAuthority` splits
fresh/running field bindings, and `RunningAuthority` splits bootstrap from
active parent recomposition. The transcript, output-extraction, sampler, and
child-opening bridges are intentionally not counted as semantic leaves; they
must refine this plan or expose a named security event before any physical
row can be removed.
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

/-- Stable phase ownership. This is semantic order, not row order. -/
inductive Phase where
  | piCcs
  | incoming
  | piRlc
  | piDec
  deriving DecidableEq

/-- Stable family ownership within a phase. -/
inductive Family where
  | paperRelation
  | inputAuthority
  | accumulatorAuthority
  | challenge
  | parent
  | children
  deriving DecidableEq

/-- The exact semantic leaves retained by `SemanticFold.Holds`. -/
inductive Leaf where
  | freshCcs
  | allSourceNorm
  | carriedEvaluations
  | polynomialInput
  | sourceProduct
  | incomingAuthority
  | challengeStrongSet
  | parentExact
  | childrenExact
  deriving DecidableEq

/-- Review classification used by the semantic and eventual physical ledgers. -/
inductive AuthorityClass where
  | checked
  | computed
  | directDataflow
  | derived
  | securityBoundary
  deriving DecidableEq

/-- Parent phase of each retained semantic leaf. -/
def phase : Leaf -> Phase
  | .freshCcs | .allSourceNorm | .carriedEvaluations
  | .polynomialInput | .sourceProduct => .piCcs
  | .incomingAuthority => .incoming
  | .challengeStrongSet | .parentExact => .piRlc
  | .childrenExact => .piDec

/-- Parent family of each retained semantic leaf. -/
def family : Leaf -> Family
  | .freshCcs | .allSourceNorm | .carriedEvaluations => .paperRelation
  | .polynomialInput | .sourceProduct => .inputAuthority
  | .incomingAuthority => .accumulatorAuthority
  | .challengeStrongSet => .challenge
  | .parentExact => .parent
  | .childrenExact => .children

/-- Semantic authority classification of each leaf. -/
def authority : Leaf -> AuthorityClass
  | .parentExact | .childrenExact => .computed
  | _ => .checked

/-- Stable review path. Exact physical paths must refine one of these owners
rather than inventing a second semantic name. -/
def path : Leaf -> String
  | .freshCcs => "nifs.semantic.pi_ccs.relation.fresh_ccs"
  | .allSourceNorm => "nifs.semantic.pi_ccs.relation.all_source_norm"
  | .carriedEvaluations =>
      "nifs.semantic.pi_ccs.relation.carried_evaluations"
  | .polynomialInput => "nifs.semantic.pi_ccs.input.polynomial"
  | .sourceProduct => "nifs.semantic.pi_ccs.input.product"
  | .incomingAuthority => "nifs.semantic.running.authority"
  | .challengeStrongSet => "nifs.semantic.pi_rlc.challenge.strong_set"
  | .parentExact => "nifs.semantic.pi_rlc.parent.exact"
  | .childrenExact => "nifs.semantic.pi_dec.children.exact"

/-- Stable semantic review order. -/
def checks : List Leaf :=
  [.freshCcs, .allSourceNorm, .carriedEvaluations, .polynomialInput,
    .sourceProduct, .incomingAuthority, .challengeStrongSet, .parentExact,
    .childrenExact]

theorem mem_checks (leaf : Leaf) : leaf ∈ checks := by
  cases leaf <;> simp [checks]

/-- Raw verifier-language candidate. No semantic proposition is stored in
this carrier. -/
structure Candidate
    (shape : SemanticShape)
    (State : Type uState)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat)
    (arity : BatchArity productionGlobalParams) where
  context :
    Context shape State publicRingColumns publicFits verifierRows arity
  data : Data shape
  point : CubePoint K shape.rowVariables
  challenges : Fin arity.total -> RingF
  parent :
    Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows)
  children : Fin productionGlobalParams.k ->
    Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows)

namespace Candidate

/-- Raw point/challenge carrier interpreted as the semantic fold witness. -/
def witness
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (candidate :
      Candidate shape State publicRingColumns publicFits verifierRows arity) :
    SemanticFold.Witness candidate.context := {
  point := candidate.point
  challenges := candidate.challenges
}

end Candidate

section

variable {shape : SemanticShape}
variable {State : Type uState}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}
variable {arity : BatchArity productionGlobalParams}

/-- One leaf owns exactly one proposition of the independent realization. -/
def semantics :
    Leaf ->
      Candidate shape State publicRingColumns publicFits verifierRows arity ->
        Prop
  | .freshCcs, candidate =>
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Paper.FreshCcsHolds
        candidate.data
  | .allSourceNorm, candidate =>
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Paper.AllSourceNormsHold
        candidate.data
  | .carriedEvaluations, candidate =>
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Paper.CarriedEvaluationsHold
        candidate.data
  | .polynomialInput, candidate =>
      SemanticFold.PublicInputBound candidate.context candidate.data
  | .sourceProduct, candidate =>
      SemanticFold.InputBound candidate.context candidate.data
  | .incomingAuthority, candidate =>
      RunningAuthority.Accepted candidate.context
  | .challengeStrongSet, candidate =>
      SemanticFold.ChallengesValid candidate.context candidate.witness
  | .parentExact, candidate =>
      candidate.parent =
        SemanticFold.parentOf candidate.context candidate.data
          candidate.witness
  | .childrenExact, candidate =>
      candidate.children =
        SemanticFold.childrenOf candidate.context candidate.data
          candidate.witness

/-- The candidate realizes the independent fold at its exact raw point and
challenge vector. Indexing `Realization` by the candidate witness prevents
existential witness substitution. -/
def target
    (candidate :
      Candidate shape State publicRingColumns publicFits verifierRows arity) :
    Prop :=
  SemanticFold.Realization candidate.context candidate.data candidate.parent
    candidate.children candidate.witness

/-- The nine-leaf tree is exactly the candidate-indexed independent fold
relation. This theorem makes no per-leaf necessity or physical row claim. -/
theorem accepts_iff_target
    (candidate :
      Candidate shape State publicRingColumns publicFits verifierRows arity) :
    CheckPlan.Accepts semantics checks candidate ↔ target candidate := by
  constructor
  · intro accepted
    exact {
      paper := ⟨
        accepted .freshCcs (mem_checks .freshCcs),
        accepted .allSourceNorm (mem_checks .allSourceNorm),
        accepted .carriedEvaluations (mem_checks .carriedEvaluations)⟩
      input := {
        publicInput :=
          accepted .polynomialInput (mem_checks .polynomialInput)
        sources := accepted .sourceProduct (mem_checks .sourceProduct)
      }
      running :=
        accepted .incomingAuthority (mem_checks .incomingAuthority)
      challengesValid :=
        accepted .challengeStrongSet (mem_checks .challengeStrongSet)
      parent_eq := accepted .parentExact (mem_checks .parentExact)
      children_eq := accepted .childrenExact (mem_checks .childrenExact)
    }
  · intro holds leaf _member
    cases leaf with
    | freshCcs => exact holds.paper.1
    | allSourceNorm => exact holds.paper.2.1
    | carriedEvaluations => exact holds.paper.2.2
    | polynomialInput => exact holds.input.publicInput
    | sourceProduct => exact holds.input.sources
    | incomingAuthority => exact holds.running
    | challengeStrongSet =>
        exact holds.challengesValid
    | parentExact =>
        exact holds.parent_eq
    | childrenExact =>
        exact holds.children_eq

/-- Exactness in the generic inclusion-minimality calculus. -/
theorem exact :
    CheckPlan.Exact
      (semantics (shape := shape) (State := State)
        (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)
        (arity := arity))
      target checks := by
  intro candidate
  exact accepts_iff_target candidate

end

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold.ObligationPlan
