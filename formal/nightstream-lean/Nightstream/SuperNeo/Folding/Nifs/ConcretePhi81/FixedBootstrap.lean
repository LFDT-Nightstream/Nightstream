import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Result

/-!
Fixed zero-running bootstrap carrier for the concrete Phi81 NIFS.

Protocol: production SuperNeo NIFS bootstrap.
Phase: one fresh CCS claim and no incoming CE accumulator.
Constraint family: typed carrier and semantic result only; this file emits no
rows.

Owns: the exact production arity `1 + 0 = 1`; the fixed-profile context and
certificate facade; and absence of incoming parent authority.

Does not own: the generic result equations or semantic transition, which are
owned by `Result`; the outer F-prime lifecycle; equivalence to HyperNova's
default initialization; executable checking; Rust/R1CS refinement; costs;
necessity; or row removal.

Emits constraints: no.

Authority boundary: the bootstrap input contains no running claims and must
carry no parent. Generic result authority delegates definitionally to
`Result`; this facade adds only the profile-specific absent-parent contract.
It does not claim that omitting HyperNova's default vector is semantics
neutral.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.fixed_bootstrap.arity` | exactly one fresh source and zero running sources | typed/computed | `arity` |
| `nifs.fixed_bootstrap.input.parent_absence` | zero-running input carries no parent authority | checked | `runningAuthority_iff_parentAbsent`, `ResultTransition.parentAbsent` |
| `nifs.fixed_bootstrap.result` | expose the generic complete result at bootstrap arity | delegated | `Result.FoldResult`, `Result.resultOf` |
| `nifs.fixed_bootstrap.semantic` | expose the generic semantic transition at bootstrap arity | delegated | `Result.ResultTransition` |
| `nifs.fixed_bootstrap.derived` | expose generic result facts at bootstrap arity | delegated | `Result.ResultTransition.*` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedBootstrap

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

/-- Exact first recursive-fold arity used by production: one fresh source and
no synthetic running source. -/
def arity : BatchArity productionGlobalParams :=
  BatchArity.bootstrap productionGlobalParams 1 (by decide) (by decide)

@[simp] theorem arity_freshCount : arity.freshCount = 1 := rfl

@[simp] theorem arity_mode : arity.mode = .bootstrap := rfl

@[simp] theorem arity_runningCount :
    arity.mode.count productionGlobalParams = 0 := rfl

@[simp] theorem arity_total : arity.total = 1 := rfl

/-- Concrete verifier context specialized to the fixed bootstrap arity. -/
abbrev Context
    (shape : SemanticShape)
    (State : Type uState)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) :=
  ConcretePhi81.Context shape State publicRingColumns publicFits
    verifierRows arity

/-- Raw verifier-visible certificate specialized to the fixed bootstrap
arity. -/
abbrev Certificate
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows) :=
  ConcretePhi81.Certificate (arity := arity)
    publicRingColumns publicFits verifierRows context.piCcsInput

/-- Complete bootstrap fold result. The parent is retained only as authority
for the next active fold. -/
abbrev FoldResult
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) :=
  Result.FoldResult shape publicRingColumns publicFits verifierRows

/-- Construct valid incoming-authority evidence for the zero-running mode. -/
def runningAuthority_of_parentAbsent
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows)
    (parentAbsent : context.runningParent = none) :
    RunningAuthority.Accepted context :=
  .bootstrap arity_mode parentAbsent

/-- For the fixed bootstrap profile, exact parent absence is the entire
incoming-authority contract. This is the target future native/R1CS checker
must refine, including the concrete `Option.none` tag. -/
theorem runningAuthority_iff_parentAbsent
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows) :
    RunningAuthority.Accepted context ↔ context.runningParent = none :=
  RunningAuthority.Accepted.iff_parentAbsent_of_bootstrap
    (context := context) arity_mode

/-- Compute both public result surfaces from one shared phase execution. -/
abbrev resultOf
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows)
    (certificate : Certificate context) :
    FoldResult shape publicRingColumns publicFits verifierRows :=
  Result.resultOf context certificate

@[simp] theorem resultOf_parent
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows)
    (certificate : Certificate context) :
    (resultOf context certificate).parent =
      (ConcretePhi81.derive context certificate).piRlcOutput := rfl

@[simp] theorem resultOf_children
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows)
    (certificate : Certificate context) :
    (resultOf context certificate).children =
      ConcretePhi81.outputChildren context certificate := rfl

/-- Independent semantic transition for the complete bootstrap result. -/
abbrev ResultTransition
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Context shape State publicRingColumns publicFits verifierRows)
    (result : FoldResult shape publicRingColumns publicFits verifierRows) :
    Prop :=
  Result.ResultTransition context result

/-- The complete result projects to the child-only concrete transition. -/
theorem ResultTransition.children_transition
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      Context shape State publicRingColumns publicFits verifierRows}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (transition : ResultTransition context result) :
    ConcretePhi81.Transition context result.children := by
  exact Result.ResultTransition.children_transition
    (arity := arity) transition

/-- Semantic bootstrap acceptance proves that the incoming parent carrier was
absent; this is not inferred from an empty digest. -/
theorem ResultTransition.parentAbsent
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      Context shape State publicRingColumns publicFits verifierRows}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (transition : ResultTransition context result) :
    context.runningParent = none := by
  rcases transition with ⟨_data, _witness, realized⟩
  exact realized.running.parentAbsent_of_bootstrap arity_mode

/-- Strict outgoing `Pi_DEC` binding preserves the sole fresh-source
structure in every returned accumulator child. -/
theorem ResultTransition.childStructure_eq_fresh
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      Context shape State publicRingColumns publicFits verifierRows}
    {result : FoldResult shape publicRingColumns publicFits verifierRows}
    (transition : ResultTransition context result)
    (child : Fin productionGlobalParams.k) :
    (result.children child).constraintSystem =
      (context.input.fresh ⟨0, arity.freshPositive⟩).constraintSystem := by
  exact Result.ResultTransition.childStructure_eq_fresh
    (arity := arity) transition child

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedBootstrap
