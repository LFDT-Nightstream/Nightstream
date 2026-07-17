import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Types

/-!
Canonical outgoing `Pi_DEC` obligations for the concrete Phi81 NIFS.

Protocol: SuperNeo NIFS.
Phase: computed `Pi_RLC` parent to canonical `Pi_DEC` children.
Constraint family: commitment, public-input, and evaluation recomposition;
this file emits no rows.

Owns: the exact proof that child structure, point, and fresh stage are
constructed from the computed parent, leaving only the three public
recomposition equations as retained `Pi_DEC` obligations.

Does not own: the arithmetic implementation of recomposition, child opening
validity, binding security, Rust/R1CS refinement, physical costs, necessity,
or row removal.

Emits constraints: no.

Authority boundary: the theorem applies only to `derive context certificate`
and `Execution.piDecChildren`. It does not accept complete prover-supplied
child statements. A child payload supplies only commitment, public input, and
evaluations; relation structure, evaluation point, and fresh stage are
verifier-computed.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.concrete.pi_dec.child_structure` | child structure equals computed parent structure | computed | `accepted_of_recomposition` |
| `nifs.concrete.pi_dec.child_point` | child point equals computed parent point | computed | `accepted_of_recomposition` |
| `nifs.concrete.pi_dec.child_stage` | every canonical child is fresh | computed | `accepted_of_recomposition` |
| `nifs.concrete.pi_dec.parent_stage` | canonical `Pi_RLC` parent is combined | computed | `accepted_of_recomposition` |
| `nifs.concrete.pi_dec.commitment` | child commitments recompose to the parent commitment | retained check | `RecompositionEquations.commitment` |
| `nifs.concrete.pi_dec.public_input` | child public inputs recompose to the parent public input | retained check | `RecompositionEquations.publicInput` |
| `nifs.concrete.pi_dec.evaluations` | child evaluations recompose to the parent evaluations | retained check | `RecompositionEquations.evaluations` |
| `nifs.concrete.pi_dec.exact` | generic acceptance iff the three retained equations | exact model theorem | `accepted_iff_recomposition` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiDec

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

universe uState

variable
  {shape : SemanticShape}
  {domain : FlatNcDomain}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits :
    ringDegree * publicRingColumns <= shape.carrierWidth}
  {arity : BatchArity productionGlobalParams}

/-- The sole non-derived public equations in the canonical outgoing
`Pi_DEC` attempt. -/
structure RecompositionEquations
    (context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity)
    (certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) : Prop where
  commitment :
    (derive context certificate).piRlcOutput.commitment =
      (decAlgebra context.key).recomposeCommitment
        (fun child => (certificate.piDecPayloads child).commitment)
  publicInput :
    (derive context certificate).piRlcOutput.publicInput =
      (decAlgebra context.key).recomposePublicInput
        (fun child => (certificate.piDecPayloads child).publicInput)
  evaluations :
    (derive context certificate).piRlcOutput.evaluations =
      (decAlgebra context.key).recomposeEvaluations
        (fun child => (certificate.piDecPayloads child).evaluations)

/-- The three retained recomposition equations imply every field of generic
`PiDEC.Accepted` for the canonical attempt. -/
theorem accepted_of_recomposition
    {context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity}
    {certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (equations : RecompositionEquations context certificate) :
    PiDEC.Accepted (decAlgebra context.key)
      ((derive context certificate).piDecAttempt certificate) := by
  refine {
    parentCombined := ?_
    childFresh := ?_
    sameStructure := ?_
    samePoint := ?_
    commitmentEquation := ?_
    publicInputEquation := ?_
    evaluationEquation := ?_
  }
  · rfl
  · intro child
    rfl
  · intro child
    rfl
  · intro child
    rfl
  · simpa [Execution.piDecAttempt, Execution.piDecChildren,
      PiDecChildPayload.materialize] using equations.commitment
  · simpa [Execution.piDecAttempt, Execution.piDecChildren,
      PiDecChildPayload.materialize] using equations.publicInput
  · simpa [Execution.piDecAttempt, Execution.piDecChildren,
      PiDecChildPayload.materialize] using equations.evaluations

/-- Generic `PiDEC.Accepted` for the canonical attempt implies exactly the
three retained recomposition equations. -/
theorem recomposition_of_accepted
    {context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity}
    {certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (accepted :
      PiDEC.Accepted (decAlgebra context.key)
        ((derive context certificate).piDecAttempt certificate)) :
    RecompositionEquations context certificate := by
  refine {
    commitment := ?_
    publicInput := ?_
    evaluations := ?_
  }
  · simpa [Execution.piDecAttempt, Execution.piDecChildren,
      PiDecChildPayload.materialize] using accepted.commitmentEquation
  · simpa [Execution.piDecAttempt, Execution.piDecChildren,
      PiDecChildPayload.materialize] using accepted.publicInputEquation
  · simpa [Execution.piDecAttempt, Execution.piDecChildren,
      PiDecChildPayload.materialize] using accepted.evaluationEquation

/-- Exact retained/eliminated ledger for canonical outgoing `Pi_DEC`.
Inherited child fields and the parent stage are construction facts; generic
acceptance is equivalent to only the three recomposition equations. -/
theorem accepted_iff_recomposition
    {context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity}
    {certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput} :
    PiDEC.Accepted (decAlgebra context.key)
        ((derive context certificate).piDecAttempt certificate) ↔
      RecompositionEquations context certificate := by
  constructor
  · exact recomposition_of_accepted
  · exact accepted_of_recomposition

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiDec
