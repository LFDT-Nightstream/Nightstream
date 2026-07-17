import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CarrierEquality
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiDec

/-!
Executable retained-equation checker for canonical outgoing `Pi_DEC`.

Protocol: SuperNeo NIFS.
Phase: verifier-computed `Pi_RLC` parent to canonical `Pi_DEC` child payloads.
Constraint family: commitment, public-input, and evaluation recomposition;
this file emits no rows.

Owns: one Boolean check for each of the three non-derived outgoing `Pi_DEC`
equations and exact equivalence with `DerivedPiDec.RecompositionEquations`.

Does not own: inherited child structure/point/stage, child opening validity,
incoming running authority, Rust/R1CS lowering, physical rows, costs,
necessity, or row removal.

Emits constraints: no.

Authority boundary: the parent is always `(derive context certificate).piRlcOutput`
and the right-hand sides are recomputed from the exact typed child payloads
with verifier-fixed radix weights. No parent digest or second derived carrier
is accepted as authority.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.concrete.pi_dec.commitment` | computed child commitments equal the computed parent commitment | checked | `check`, `check_eq_true_iff_recomposition` |
| `nifs.concrete.pi_dec.public_input` | computed child public inputs equal the computed parent public input | checked | `check`, `check_eq_true_iff_recomposition` |
| `nifs.concrete.pi_dec.evaluations` | computed child evaluation arrays equal the computed parent evaluations | checked | `check`, `check_eq_true_iff_recomposition` |
| `nifs.concrete.pi_dec.exact_checker` | Boolean result iff all and only the retained equations hold | exact model theorem | `check_eq_true_iff_recomposition` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiDec.Checker

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open CarrierEquality

universe uState

/-- Execute exactly the three retained recomposition comparisons. -/
def check
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity)
    (certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) : Bool :=
  commitmentEqual
      (derive context certificate).piRlcOutput.commitment
      ((decAlgebra context.key).recomposeCommitment fun child =>
        (certificate.piDecPayloads child).commitment) &&
    (publicInputEqual
        (derive context certificate).piRlcOutput.publicInput
        ((decAlgebra context.key).recomposePublicInput fun child =>
          (certificate.piDecPayloads child).publicInput) &&
      evaluationsEqual
        (derive context certificate).piRlcOutput.evaluations
        ((decAlgebra context.key).recomposeEvaluations fun child =>
          (certificate.piDecPayloads child).evaluations))

/-- The Boolean checker accepts exactly the independently stated retained
`Pi_DEC` equations. -/
theorem check_eq_true_iff_recomposition
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity)
    (certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) :
    check context certificate = true <->
      RecompositionEquations context certificate := by
  simp only [check, Bool.and_eq_true, commitmentEqual_eq_true_iff,
    publicInputEqual_eq_true_iff, evaluationsEqual_eq_true_iff]
  constructor
  · rintro ⟨commitment, publicInput, evaluations⟩
    exact {
      commitment := commitment
      publicInput := publicInput
      evaluations := evaluations
    }
  · intro equations
    exact ⟨equations.commitment, equations.publicInput,
      equations.evaluations⟩

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiDec.Checker
