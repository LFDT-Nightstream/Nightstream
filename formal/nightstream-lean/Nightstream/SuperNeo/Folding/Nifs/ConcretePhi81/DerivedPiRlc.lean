import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Types

/-!
Canonical outgoing `Pi_RLC` obligations for the concrete Phi81 NIFS.

Protocol: SuperNeo NIFS.
Phase: derived `Pi_CCS` product to computed `Pi_RLC` parent.
Constraint family: source-structure consistency only; this file emits no rows.

Owns: the exact proof that the canonical dataflow computes every public
`Pi_RLC` output equation by construction, leaving only equality of each
source structure with the verifier-selected structure as a retained
obligation.

Does not own: source-structure decoding, `Pi_CCS` acceptance, challenge
sampling, `Pi_DEC`, Rust/R1CS refinement, physical costs, or row removal.

Emits constraints: no.

Authority boundary: the theorem applies only to `derive context certificate`.
It does not authorize a prover-supplied parent or an arbitrary `PiRLC.Attempt`.
The commitment, public input, evaluations, point, and stage of the outgoing
parent are computed by `PiRLC.combinedOutput`; they are not accepted as a
second copy to compare against.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.concrete.pi_rlc.source_structure` | every canonical source uses `context.system` | retained check | `SourceStructuresBound` |
| `nifs.concrete.pi_rlc.input_stage` | every materialized `Pi_CCS` output is fresh | computed | `equations_of_sourceStructures` |
| `nifs.concrete.pi_rlc.input_point` | every materialized source uses the FE-derived point | computed | `equations_of_sourceStructures` |
| `nifs.concrete.pi_rlc.parent_stage` | the computed parent is combined | computed | `equations_of_sourceStructures` |
| `nifs.concrete.pi_rlc.parent_commitment` | parent commitment is the canonical challenge combination | computed | `equations_of_sourceStructures` |
| `nifs.concrete.pi_rlc.parent_public_input` | parent public input is the canonical challenge combination | computed | `equations_of_sourceStructures` |
| `nifs.concrete.pi_rlc.parent_evaluations` | parent evaluations are the canonical challenge combination | computed | `equations_of_sourceStructures` |
| `nifs.concrete.pi_rlc.exact` | generic public equations iff the sole retained structure family | exact model theorem | `equations_iff_sourceStructures` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiRlc

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits :
    ringDegree * publicRingColumns <= shape.carrierWidth}
  {arity : BatchArity productionGlobalParams}

/-- The sole non-derived public equation in the canonical outgoing `Pi_RLC`
attempt. All sources must use the verifier-selected relation structure. -/
def SourceStructuresBound
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity) : Prop :=
  forall source,
    (context.input.source source).constraintSystem = context.system

/-- Every generic `Pi_RLC.Equations` field for the canonical derived attempt
follows from source-structure consistency. The expensive parent fields are
computed once; no second authoritative copy is checked. -/
theorem equations_of_sourceStructures
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (structures : SourceStructuresBound context) :
    PiRLC.Equations (rlcAlgebra context.key)
      ((derive context certificate).piRlcAttempt certificate) := by
  refine {
    inputFresh := ?_
    sameStructure := ?_
    samePoint := ?_
    outputCombined := ?_
    commitmentEquation := ?_
    publicInputEquation := ?_
    evaluationEquation := ?_
  }
  · intro source
    rfl
  · intro source
    exact structures source
  · intro source
    rfl
  · rfl
  · rfl
  · rfl
  · rfl

/-- Generic public `Pi_RLC` equations for the canonical derived attempt imply
the sole retained source-structure family. -/
theorem sourceStructures_of_equations
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (equations :
      PiRLC.Equations (rlcAlgebra context.key)
        ((derive context certificate).piRlcAttempt certificate)) :
    SourceStructuresBound context := by
  intro source
  exact equations.sameStructure source

/-- Exact retained/eliminated ledger for the canonical outgoing `Pi_RLC`
attempt. This is an equivalence to independently defined generic phase
equations, not a restatement of a caller-supplied verifier predicate. -/
theorem equations_iff_sourceStructures
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput} :
    PiRLC.Equations (rlcAlgebra context.key)
        ((derive context certificate).piRlcAttempt certificate) <->
      SourceStructuresBound context := by
  constructor
  · exact sourceStructures_of_equations
  · exact equations_of_sourceStructures

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiRlc
