import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.RunningAuthority
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiRlc
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiDec
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.OutputRefinement

/-!
Physical-to-semantic refinement for the concrete Phi81 NIFS verifier over the
exact Split-NC `Pi_CCS → Pi_RLC → Pi_DEC` dataflow.

Protocol: SuperNeo NIFS.
Phase: complete three-phase semantic verifier.
Constraint family: logical acceptance only; this file emits no rows.

Assurance tier: model-level.

Owns: executable verifier acceptance separated from semantic source authority;
checked incoming accumulator authority; sampler-derived
`Pi_RLC` challenge authority; the independently proved concrete `Pi_RLC`
equations; the three retained outgoing `Pi_DEC` recomposition equations over
canonical children; the minimal certificate-indexed refinement evidence; its bridge to
`SemanticFold.Holds`; and deterministic soundness with output mismatch and
FE/NC bad events explicit.

Does not own: Poseidon2 instantiation of the abstract sampler machine,
extraction, Ajtai/MSIS binding security, F-prime selection/lifecycle, Rust,
R1CS, rows, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: `Accepted` reads public context and raw certificate only.
It contains no private source data, opening witness, paper truth, or output
truth. Bootstrap requires an absent incoming parent; active mode validates the
complete transcript-bound parent against the exact running children.
`SemanticInput` re-exports the binding owned by `SemanticFold`: one rich
independent source family is tied to the
public polynomial input and source product. `OutputBound` remains a separate
semantic outcome: physical acceptance alone does not assume it. A failed
output binding or SumCheck mixing claim is returned explicitly rather than
silently promoted to verifier authority. `CertificateRefinement` deliberately
does not carry `TailAccepted`: the independent fold follows from source/output
binding, sampler replay, and exact child openings. This is a model-level
dependency result, not permission to delete physical tail rows before the
child-opening security reduction and exact R1CS refinement are closed.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.concrete.input.projection` | public polynomial input is exactly the source projection | semantic bridge | `PublicInputBound` |
| `nifs.concrete.input.sources` | every public source field opens the aligned semantic source | semantic bridge | `InputBound` |
| `nifs.concrete.running_authority` | bootstrap has no parent; active parent strictly recomposes to the `k` running children | checked | `RunningAuthority.Accepted` |
| `nifs.concrete.pi_ccs` | exact physical FE→NC transcript accepts | checked | `PiCcsAccepted` |
| `nifs.concrete.pi_ccs.output` | complete `yRing`/`yZcol` binds to the independent sources; `yZcol` terminates here rather than becoming a CE field | explicit semantic outcome | `OutputBound`, `CertificateRefinement.packedYZcolBound` |
| `nifs.concrete.pi_rlc.sampler` | replay binds every challenge and derives production-set membership | checked/derived | `Accepted.sampler`, `TailAccepted.piRlcAccepted` |
| `nifs.concrete.pi_rlc.source_structure` | every materialized source uses the verifier-selected structure | checked | `TailAccepted.sourceStructures` |
| `nifs.concrete.pi_rlc.derived_equations` | stage, point, commitment, public input, and evaluations of the parent are canonical | computed | `TailAccepted.piRlcEquations` |
| `nifs.concrete.pi_rlc.parent_opening` | the computed parent opens at the canonical challenge-folded assignment | derived | `CertificateRefinement.piRlcParentOpening` |
| `nifs.concrete.pi_dec.recomposition` | canonical child payloads recompose to the one derived parent | checked | `TailAccepted.piDecRecomposition` |
| `nifs.concrete.pi_dec.inherited` | child structure, point, and fresh stage are inherited from the parent | computed | `TailAccepted.piDec` |
| `nifs.concrete.pi_dec.openings` | every public child has the canonical split private opening | explicit security boundary | `ChildOpenings` |
| `nifs.concrete.refinement` | paper source truth, public/source binding, output authority, sampler replay, and child openings | certificate-indexed bridge evidence | `CertificateRefinement` |
| `nifs.concrete.semantic` | physical result equals the certificate-independent honest fold | derived refinement | `CertificateRefinement.toSemanticFold` |
| `nifs.concrete.soundness` | physical acceptance plus semantic input and child-opening authority implies transition, output mismatch, or named FE/NC bad event | derived conditional on extraction/binding | `accepted_implies_transition_or_outputUnbound_or_badEvent` |
| `nifs.concrete.completeness` | honest paper sources and valid challenges construct all three phases and valid children | derived | `complete_of_paperObligations` |
| `nifs.concrete.completeness.outcome` | honest paper sources produce a transition or one exact bounded-sampler shortfall | exhaustive model outcome | `complete_or_samplerShortfall` |
| `nifs.concrete.public_relation` | existential independent source data and computed parent, with no raw certificate | independent specification | `Transition` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits :
    ringDegree * publicRingColumns <= shape.carrierWidth}
  {arity : BatchArity productionGlobalParams}

/-- Public-input projection bridge, owned by the certificate-independent
semantic module and re-exported here for physical refinement signatures. -/
abbrev PublicInputBound
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape) :=
  SemanticFold.PublicInputBound context data

/-- Public source-product bridge, owned by the certificate-independent
semantic module and re-exported here for physical refinement signatures. -/
abbrev InputBound
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape) :=
  SemanticFold.InputBound context data

/-- Complete two-surface semantic input binding. -/
abbrev SemanticInput
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape) :=
  SemanticFold.Input context data

/-- Exact physical Split-NC phase acceptance over the public polynomial input.
No source witness or semantic output binding is read here. -/
def PiCcsAccepted
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) : Prop :=
  Protocol.BlockLane.Accepted StatementInput.polynomial context.piCcsSchedule
    context.priorState context.profile context.piCcsStatement
    certificate.piCcs

/-- Complete semantic output binding at the two points derived by the same
physical transcript. This remains an explicit soundness outcome, not a field
of `Accepted`. -/
def OutputBound
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity)
    (data : Data shape)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) : Prop :=
  Protocol.BlockLane.OutputBound context.covers data
    (derive context certificate).piCcs certificate.piCcs.output

/-- Private-opening authority for every public Π_DEC child. Public
recomposition alone cannot establish this property; a concrete soundness
theorem must obtain it from extraction and commitment binding. -/
def ChildOpenings
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) : Prop :=
  ∀ child,
    CE.Holds (semantics context.key) productionGlobalParams
      (outputChildren context certificate child)
      ((decAlgebra context.key).splitAssignment
        (PiRLC.combinedWitness (rlcAlgebra context.key)
          certificate.piRlcChallenges
          (InputAuthority.productAssignments data context.alignment))
        child)

/-- Tail acceptance over the unique derived `Pi_CCS` product and `Pi_RLC`
parent. -/
structure TailAccepted
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) : Prop where
  sourceStructures : DerivedPiRlc.SourceStructuresBound context
  piDecRecomposition :
    DerivedPiDec.RecompositionEquations context certificate

namespace TailAccepted

/-- The canonical dataflow derives all public `Pi_RLC` equations from the one
retained source-structure family. This is an eliminated check family, not an
additional field of `TailAccepted`. -/
def piRlcEquations
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (tail : TailAccepted context certificate) :
    PiRLC.Equations (rlcAlgebra context.key)
      ((derive context certificate).piRlcAttempt certificate) :=
  DerivedPiRlc.equations_of_sourceStructures tail.sourceStructures

/-- Exact sampler replay supplies the challenge-validity theorem needed to
assemble complete model-level Π_RLC acceptance. There is no independent
challenge-membership check in the concrete transition. -/
def piRlcAccepted
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (tail : TailAccepted context certificate)
    (sampler : Sampler.CertificateAccepted context certificate) :
    PiRLC.Accepted (rlcAlgebra context.key)
      ((derive context certificate).piRlcAttempt certificate) :=
  tail.piRlcEquations.withChallengesValid
    (Sampler.certificateAccepted_challengesValid sampler)

/-- The retained recomposition equations assemble complete model-level
`PiDEC.Accepted`; inherited child fields are construction facts rather than
independent checks. -/
def piDec
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (tail : TailAccepted context certificate) :
    PiDEC.Accepted (decAlgebra context.key)
      ((derive context certificate).piDecAttempt certificate) :=
  DerivedPiDec.accepted_of_recomposition tail.piDecRecomposition

end TailAccepted

/-- Complete physical verifier acceptance. -/
structure Accepted
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) : Prop where
  running : RunningAuthority.Accepted context
  piCcs : PiCcsAccepted context certificate
  sampler : Sampler.CertificateAccepted context certificate
  tail : TailAccepted context certificate

/-- Certificate-indexed refinement evidence for one physical execution. This
is deliberately not the independent NIFS relation: it records the output and
child-opening facts needed to prove that the execution refines
`SemanticFold.Holds`. -/
structure CertificateRefinement
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity)
    (data : Data shape)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) : Prop where
  paper : Semantics.Paper.Holds data
  input : SemanticInput context data
  running : RunningAuthority.Accepted context
  output : OutputBound context data certificate
  sampler : Sampler.CertificateAccepted context certificate
  children : ChildOpenings context data certificate

namespace CertificateRefinement

/-- Exact raw semantic witness decoded from one physical execution. Validity
is proved separately by sampler refinement. -/
def semanticWitness
    {context :
      Context shape State publicRingColumns publicFits verifierRows arity}
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) :
    SemanticFold.Witness context where
  point := (derive context certificate).piCcs.fePoint.row
  challenges := certificate.piRlcChallenges

/-- Packed `yZcol` remains bound to the independent sources at the derived
BlockLane point, but is deliberately absent from the CE parent and children. -/
theorem packedYZcolBound
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    {data : Data shape}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (holds : CertificateRefinement context data certificate) :
    Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock context.covers data
      (derive context certificate).piCcs.ncPoint.block
      certificate.piCcs.output := by
  exact holds.output.2

/-- The complete certificate-indexed evidence refines the independent fold
relation. The proof first replaces the message-shaped `Pi_CCS` product by
`PiCCS.honestOutputs`, then proves every extracted child is the corresponding
`PiDEC.childrenOf` value. No certificate field appears in the conclusion. -/
theorem toSemanticRealization
    {context :
      Context shape State publicRingColumns publicFits verifierRows arity}
    {data : Data shape}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (holds : CertificateRefinement context data certificate) :
    SemanticFold.Realization context data
      (derive context certificate).piRlcOutput
      (outputChildren context certificate)
      (semanticWitness certificate) := by
  let witness := semanticWitness certificate
  have systemEq :
      context.system = SemanticFold.systemOf context data := by
    simpa [Context.system, SemanticFold.systemOf] using
      (holds.input.sources.fresh ⟨0, arity.freshPositive⟩).constraintSystem
  have outputsEq :
      (derive context certificate).piCcsOutputs =
        SemanticFold.outputs context data witness := by
    change
      OutputProduct.materialize publicRingColumns publicFits context.alignment
          context.input (derive context certificate).piCcs.fePoint.row
          certificate.piCcs.output =
        PiCCS.honestOutputs (semantics context.key) context.input
          (InputAuthority.productAssignments data context.alignment)
          (derive context certificate).piCcs.fePoint.row
    simpa [semantics] using
      (Protocol.OutputRefinement.materializedOutputs_eq_honestOutputs_of_yRingEq
        publicRingColumns publicFits (commit context.key) data context.alignment
        context.input (derive context certificate).piCcs.fePoint.row
        certificate.piCcs.output production_norm_stages.1 holds.paper
        holds.input.sources holds.output.1)
  have parentEq :
      (derive context certificate).piRlcOutput =
        SemanticFold.parentOf context data witness := by
    change
      PiRLC.combinedOutput (rlcAlgebra context.key) context.system
          (derive context certificate).piCcs.fePoint.row
          (derive context certificate).piCcsOutputs
          certificate.piRlcChallenges =
        PiRLC.combinedOutput (rlcAlgebra context.key)
          (SemanticFold.systemOf context data)
          (derive context certificate).piCcs.fePoint.row
          (SemanticFold.outputs context data witness)
          certificate.piRlcChallenges
    rw [systemEq, outputsEq]
  have childrenEq :
      outputChildren context certificate =
        SemanticFold.childrenOf context data witness := by
    funext child
    calc
      outputChildren context certificate child =
          PiDEC.childrenOf (decAlgebra context.key)
            (derive context certificate).piRlcOutput
            (PiRLC.combinedWitness (rlcAlgebra context.key)
              certificate.piRlcChallenges
              (InputAuthority.productAssignments data context.alignment))
            child := by
        apply SemanticFold.child_eq_childrenOf_of_holds context
        · rfl
        · rfl
        · rfl
        · exact holds.children child
      _ = SemanticFold.childrenOf context data witness child := by
        simpa [SemanticFold.childrenOf, SemanticFold.combinedAssignment,
          SemanticFold.assignments, witness] using
          congrArg
            (fun parent =>
              PiDEC.childrenOf (decAlgebra context.key) parent
                (PiRLC.combinedWitness (rlcAlgebra context.key)
                  certificate.piRlcChallenges
                  (InputAuthority.productAssignments data context.alignment))
                child)
            parentEq
  exact {
    paper := holds.paper
    input := holds.input
    running := holds.running
    challengesValid :=
      Sampler.certificateAccepted_challengesValid holds.sampler
    parent_eq := parentEq
    children_eq := childrenEq
  }

/-- The exact physical realization instantiates the existential independent
fold relation. -/
theorem toSemanticFold
    {context :
      Context shape State publicRingColumns publicFits verifierRows arity}
    {data : Data shape}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (holds : CertificateRefinement context data certificate) :
    SemanticFold.Holds context data
      (derive context certificate).piRlcOutput
      (outputChildren context certificate) :=
  ⟨semanticWitness certificate, holds.toSemanticRealization⟩

/-- The computed `Pi_RLC` parent has the canonical combined opening. This is
now a corollary of the certificate-independent fold refinement; it does not
consume either outgoing tail-check family. -/
theorem piRlcParentOpening
    {context :
      Context shape State publicRingColumns publicFits verifierRows arity}
    {data : Data shape}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (holds : CertificateRefinement context data certificate) :
    CE.Holds (semantics context.key) productionGlobalParams
      (derive context certificate).piRlcOutput
      (PiRLC.combinedWitness (rlcAlgebra context.key)
        certificate.piRlcChallenges
        (InputAuthority.productAssignments data context.alignment)) := by
  simpa [SemanticFold.combinedAssignment, SemanticFold.assignments,
    semanticWitness] using holds.toSemanticRealization.parentOpening

/-- Actual certificate children carry exactly the verifier-computed public
split of the derived parent whenever the existing semantic openings bind both
sides. No deterministic child commitment or evaluation equality is used. -/
theorem childPublicInput_eq_splitParent
    {context :
      Context shape State publicRingColumns publicFits verifierRows arity}
    {data : Data shape}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (holds : CertificateRefinement context data certificate)
    (child : Fin productionGlobalParams.k) :
    (outputChildren context certificate child).publicInput =
      PiDECAlgebra.PublicInput.splitPublicInput
        (derive context certificate).piRlcOutput.publicInput child := by
  let parentAssignment :=
    PiRLC.combinedWitness (rlcAlgebra context.key)
      certificate.piRlcChallenges
      (InputAuthority.productAssignments data context.alignment)
  have parentOpening := holds.piRlcParentOpening
  have childOpening := holds.children child
  calc
    (outputChildren context certificate child).publicInput =
        Phi81Relation.projectPublicInput
          ((decAlgebra context.key).splitAssignment parentAssignment child) :=
      childOpening.1.2.1.symm
    _ = PiDECAlgebra.PublicInput.splitPublicInput
          (Phi81Relation.projectPublicInput parentAssignment) child :=
      (PiDECAlgebra.PublicInput.splitPublicInput_project
        parentAssignment child).symm
    _ = PiDECAlgebra.PublicInput.splitPublicInput
          (derive context certificate).piRlcOutput.publicInput child :=
      congrArg
        (fun publicInput =>
          PiDECAlgebra.PublicInput.splitPublicInput publicInput child)
        parentOpening.1.2.1

/-- The derived parent evaluation array has exactly the relation-owned matrix
arity. -/
theorem parentEvaluations_size
    {context :
      Context shape State publicRingColumns publicFits verifierRows arity}
    {data : Data shape}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (holds : CertificateRefinement context data certificate) :
    (derive context certificate).piRlcOutput.evaluations.size =
      shape.matrixCount := by
  let parentAssignment :=
    PiRLC.combinedWitness (rlcAlgebra context.key)
      certificate.piRlcChallenges
      (InputAuthority.productAssignments data context.alignment)
  have parentOpening := holds.piRlcParentOpening
  calc
    (derive context certificate).piRlcOutput.evaluations.size =
        (Phi81Relation.evaluations
          (derive context certificate).piRlcOutput.constraintSystem
          parentAssignment
          (derive context certificate).piRlcOutput.point).size :=
      congrArg Array.size parentOpening.2.2.symm
    _ = shape.matrixCount := Phi81Relation.evaluations_size _ _ _

/-- Every actual certificate child evaluation array has exactly the
relation-owned matrix arity. -/
theorem childEvaluations_size
    {context :
      Context shape State publicRingColumns publicFits verifierRows arity}
    {data : Data shape}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (holds : CertificateRefinement context data certificate)
    (child : Fin productionGlobalParams.k) :
    (outputChildren context certificate child).evaluations.size =
      shape.matrixCount := by
  let parentAssignment :=
    PiRLC.combinedWitness (rlcAlgebra context.key)
      certificate.piRlcChallenges
      (InputAuthority.productAssignments data context.alignment)
  have childOpening := holds.children child
  calc
    (outputChildren context certificate child).evaluations.size =
        (Phi81Relation.evaluations
          (outputChildren context certificate child).constraintSystem
          ((decAlgebra context.key).splitAssignment parentAssignment child)
          (outputChildren context certificate child).point).size :=
      congrArg Array.size childOpening.2.2.symm
    _ = shape.matrixCount := Phi81Relation.evaluations_size _ _ _

end CertificateRefinement

/-- Named FE/NC algebraic failure, interpreted only through an explicit
public-input projection bridge. -/
def PiCcsBadEvent
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity)
    (data : Data shape)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) : Prop :=
  ∃ bound : PublicInputBound context data,
    let pre :=
      Protocol.TranscriptAuthority.BlockLane.derivePreSumcheck
        context.piCcsSchedule context.priorState context.piCcsStatement
    let sourceCertificate :=
      Protocol.BlockLane.certificateAtSources data certificate.piCcs bound
    Protocol.BlockLane.BadEvent context.profile context.covers data
      pre.challenges (derive context certificate).piCcs sourceCertificate
      context.challengeSetSize

/-- External concrete NIFS relation. The public context and child output remain
visible; independent source data and the computed parent are existential. No
raw verifier certificate occurs in this specification. -/
def Transition
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity)
    (output : Fin productionGlobalParams.k ->
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) : Prop :=
  ∃ data : Data shape,
    ∃ parent :
        Phi81Relation.CEStatement
          (RelationShape shape publicRingColumns publicFits)
          (CommitmentValue verifierRows),
      SemanticFold.Holds context data parent output

/-- Physical Split-NC acceptance implies the independent paper obligations,
an explicit output-binding failure, or one named FE/NC bad event once the
public polynomial input is known to be the source projection. -/
theorem accepted_implies_paper_or_outputUnbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    {data : Data shape}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (publicInputBound : PublicInputBound context data)
    (accepted : Accepted context certificate) :
    Semantics.Paper.Holds data ∨
      ¬ OutputBound context data certificate ∨
      PiCcsBadEvent context data certificate := by
  rcases
      Protocol.BlockLane.accepted_implies_paperObligations_or_unbound_or_badEvent
        noZeroDivisors context.covers StatementInput.polynomial
        context.piCcsSchedule context.priorState context.profile
        context.piCcsStatement data publicInputBound certificate.piCcs
        context.challengeSetSize accepted.piCcs with
    paper | unbound | bad
  · exact Or.inl paper
  · exact Or.inr (Or.inl (by
      simpa [OutputBound, derive] using unbound))
  · apply Or.inr
    apply Or.inr
    refine ⟨publicInputBound, ?_⟩
    simpa [derive] using bad

/-- Exact Split-NC soundness lifts through both shared tail phases without
smuggling semantic output truth into physical acceptance. -/
theorem accepted_implies_refinement_or_outputUnbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    {data : Data shape}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (input : SemanticInput context data)
    (children : ChildOpenings context data certificate)
    (accepted : Accepted context certificate) :
    CertificateRefinement context data certificate ∨
      ¬ OutputBound context data certificate ∨
      PiCcsBadEvent context data certificate := by
  rcases accepted_implies_paper_or_outputUnbound_or_badEvent
      noZeroDivisors input.publicInput accepted with
    paper | outputUnbound | bad
  · by_cases output : OutputBound context data certificate
    · exact Or.inl {
        paper := paper
        input := input
        running := accepted.running
        output := output
        sampler := accepted.sampler
        children := children
      }
    · exact Or.inr (Or.inl output)
  · exact Or.inr (Or.inl outputUnbound)
  · exact Or.inr (Or.inr bad)

/-- Public projection of physical acceptance is the independent concrete
transition, an explicit output-binding failure, or one named FE/NC bad event. -/
theorem accepted_implies_transition_or_outputUnbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    {data : Data shape}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (input : SemanticInput context data)
    (children : ChildOpenings context data certificate)
    (accepted : Accepted context certificate) :
    Transition context (outputChildren context certificate) ∨
      ¬ OutputBound context data certificate ∨
      PiCcsBadEvent context data certificate := by
  rcases accepted_implies_refinement_or_outputUnbound_or_badEvent
      noZeroDivisors input children accepted with
    holds | outputUnbound | bad
  · exact Or.inl ⟨data, (derive context certificate).piRlcOutput,
      holds.toSemanticFold⟩
  · exact Or.inr (Or.inl outputUnbound)
  · exact Or.inr (Or.inr bad)

/-- The exact honest-completeness failure that remains possible after the
canonical `Pi_CCS` prefix has accepted and its semantic output has been bound.
The failure names one challenge coordinate whose fixed 64-candidate prefix
contains fewer than the required 54 accepted coefficients. -/
def HonestSamplerShortfall
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity)
    (data : Data shape) : Prop :=
  ∃ piCcsCertificate :
      Protocol.BlockLane.Certificate context.piCcsInput
        PiCcsDomains.production,
    Protocol.BlockLane.Accepted StatementInput.polynomial
        context.piCcsSchedule context.priorState context.profile
        context.piCcsStatement piCcsCertificate ∧
      Protocol.BlockLane.OutputBound context.covers data
        (Protocol.BlockLane.derive StatementInput.polynomial
          context.piCcsSchedule context.priorState context.profile
          context.piCcsStatement piCcsCertificate)
        piCcsCertificate.output ∧
      Exists fun coordinate : Fin arity.total =>
        Nifs.NonInteractive.PiRlcSampler.ShortfallAt
          (Sampler.Specification context.piRlcMachine)
          Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.candidateBound
          (Protocol.BlockLane.derive StatementInput.polynomial
            context.piCcsSchedule context.priorState context.profile
            context.piCcsStatement piCcsCertificate).finalState
          coordinate.val

/-- Finish the concrete NIFS honest construction from the one canonical
honest `Pi_CCS` prefix and one successful bounded sampler batch. Keeping this
step separate prevents the public completeness theorem from requiring one
challenge vector to work for every accepted `Pi_CCS` certificate. -/
private theorem complete_of_honestPiCcsAndSampler
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity)
    (data : Data shape)
    (paper : Semantics.Paper.Holds data)
    (input : SemanticInput context data)
    (running : RunningAuthority.Accepted context)
    (piCcsCertificate :
      Protocol.BlockLane.Certificate context.piCcsInput
        PiCcsDomains.production)
    (piCcsTranscript :
      Protocol.BlockLane.Accepted StatementInput.polynomial
        context.piCcsSchedule context.priorState context.profile
        context.piCcsStatement piCcsCertificate)
    (outputAuthority :
      Protocol.BlockLane.OutputBound context.covers data
        (Protocol.BlockLane.derive StatementInput.polynomial
          context.piCcsSchedule context.priorState context.profile
          context.piCcsStatement piCcsCertificate)
        piCcsCertificate.output)
    (outputsHold :
      ProductHolds publicRingColumns publicFits (commit context.key)
        (OutputProduct.materialize publicRingColumns publicFits
          context.alignment context.input
          (Protocol.BlockLane.derive StatementInput.polynomial
            context.piCcsSchedule context.priorState context.profile
            context.piCcsStatement piCcsCertificate).fePoint.row
          piCcsCertificate.output)
        (InputAuthority.productAssignments data context.alignment))
    (challenges : Fin arity.total -> RingF)
    (samplerBound :
      Sampler.Bound context.piRlcMachine
        (Protocol.BlockLane.derive StatementInput.polynomial
          context.piCcsSchedule context.priorState context.profile
          context.piCcsStatement piCcsCertificate).finalState
        challenges) :
    ∃ certificate :
        Certificate (arity := arity)
          publicRingColumns publicFits verifierRows context.piCcsInput,
      Accepted context certificate ∧
        CertificateRefinement context data certificate ∧
        ∀ child,
          CE.Holds (semantics context.key) productionGlobalParams
            (outputChildren context certificate child)
            ((decAlgebra context.key).splitAssignment
              (PiRLC.combinedWitness (rlcAlgebra context.key) challenges
                (InputAuthority.productAssignments
                  data context.alignment))
              child) := by
  rcases input with ⟨publicInputBound, inputBound⟩
  let piCcsExecution :=
    Protocol.BlockLane.derive StatementInput.polynomial context.piCcsSchedule
      context.priorState context.profile context.piCcsStatement
      piCcsCertificate
  let piCcsOutputs :=
    OutputProduct.materialize publicRingColumns publicFits context.alignment
      context.input piCcsExecution.fePoint.row
      piCcsCertificate.output
  let assignments :=
    InputAuthority.productAssignments data context.alignment
  have challengesValid :
      ∀ source, (rlcAlgebra context.key).challengeValid
        (challenges source) := by
    exact samplerBound.challengeValid
  have outputsValid :
      ∀ source,
        CE.Holds (semantics context.key) productionGlobalParams
          (piCcsOutputs source) (assignments source) := by
    simpa [piCcsOutputs, piCcsExecution, assignments] using outputsHold
  let system := context.system
  have systemBound :
      system =
        Phi81Relation.Structure.ofSourceData
          publicRingColumns publicFits data := by
    simpa [system, Context.system] using
      (inputBound.fresh ⟨0, arity.freshPositive⟩).constraintSystem
  let point := piCcsExecution.fePoint.row
  have rlcComplete :=
    PiRLC.complete (semantics context.key) productionGlobalParams
      (rlcAlgebra context.key) arity system point piCcsOutputs challenges
      assignments (fun _ => rfl)
      (fun source => by
        calc
          (piCcsOutputs source).constraintSystem =
              (context.input.source source).constraintSystem := by
            rfl
          _ = Phi81Relation.Structure.ofSourceData
                publicRingColumns publicFits data :=
            InputAuthority.BoundToSources.sourceStructure
              publicRingColumns publicFits (commit context.key) data
              context.alignment context.input inputBound source
          _ = system := systemBound.symm)
      (fun _ => rfl)
      challengesValid outputsValid
      (Phi81Relation.evaluationPointValid_holds system point)
  let rlcOutput :=
    PiRLC.combinedOutput (rlcAlgebra context.key) system point piCcsOutputs
      challenges
  let combinedAssignment :=
    PiRLC.combinedWitness (rlcAlgebra context.key) challenges assignments
  have decComplete :=
    PiDEC.complete (semantics context.key) productionGlobalParams
      (decAlgebra context.key) rlcOutput combinedAssignment rfl rlcComplete.2
  let children :=
    PiDEC.childrenOf (decAlgebra context.key) rlcOutput combinedAssignment
  let certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput := {
    piCcs := piCcsCertificate
    piRlcChallenges := challenges
    piDecPayloads := fun child =>
      PiDecChildPayload.ofStatement (children child)
  }
  have tail : TailAccepted context certificate := by
    refine ⟨?_, ?_⟩
    · intro source
      calc
        (context.input.source source).constraintSystem =
            Phi81Relation.Structure.ofSourceData
              publicRingColumns publicFits data :=
          InputAuthority.BoundToSources.sourceStructure
            publicRingColumns publicFits (commit context.key) data
            context.alignment context.input inputBound source
        _ = system := systemBound.symm
        _ = context.system := rfl
    · apply DerivedPiDec.recomposition_of_accepted
      simpa [certificate, derive, piCcsExecution, piCcsOutputs, system, point,
        rlcOutput, children, Execution.piDecAttempt,
        Execution.piDecChildren, PiDecChildPayload.ofStatement,
        PiDecChildPayload.materialize] using decComplete.1
  have sampler : Sampler.CertificateAccepted context certificate := by
    refine ⟨?_⟩
    simpa [Sampler.CertificateBound, certificate, derive, piCcsExecution,
      Execution.piRlcInitialState] using samplerBound
  have physical : Accepted context certificate := {
    running := running
    piCcs := piCcsTranscript
    sampler := sampler
    tail := tail
  }
  have childOpenings : ChildOpenings context data certificate := by
    intro child
    simpa [ChildOpenings, certificate, children, combinedAssignment,
      assignments] using decComplete.2 child
  have semantic : CertificateRefinement context data certificate := {
    paper := paper
    input := ⟨publicInputBound, inputBound⟩
    running := running
    output := by
      simpa [certificate, derive, piCcsExecution] using outputAuthority
    sampler := sampler
    children := childOpenings
  }
  refine ⟨certificate, physical, semantic, ?_⟩
  exact childOpenings

/-- Honest completeness of the exact concrete composition, conditional only
on bounded sampler availability for the honest Split-NC prefix. Membership in
the production challenge set is derived from that replay witness rather than
accepted as a separate premise.

The remaining implementation refinement must instantiate
`context.piRlcMachine` with the exact Poseidon2 schedule and prove that native
and R1CS execution provide this same bounded batch. -/
theorem complete_of_paperObligations
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity)
    (data : Data shape)
    (paper : Semantics.Paper.Holds data)
    (input : SemanticInput context data)
    (running : RunningAuthority.Accepted context)
    (challenges : Fin arity.total -> RingF)
    (samplerAvailable :
      ∀ piCcsCertificate :
          Protocol.BlockLane.Certificate context.piCcsInput
            PiCcsDomains.production,
        Protocol.BlockLane.Accepted StatementInput.polynomial
            context.piCcsSchedule context.priorState context.profile
            context.piCcsStatement piCcsCertificate →
          Sampler.Bound context.piRlcMachine
            (Protocol.BlockLane.derive StatementInput.polynomial
              context.piCcsSchedule context.priorState context.profile
              context.piCcsStatement piCcsCertificate).finalState
            challenges) :
    ∃ certificate :
        Certificate (arity := arity)
          publicRingColumns publicFits verifierRows context.piCcsInput,
      Accepted context certificate ∧
        CertificateRefinement context data certificate ∧
        ∀ child,
          CE.Holds (semantics context.key) productionGlobalParams
            (outputChildren context certificate child)
            ((decAlgebra context.key).splitAssignment
              (PiRLC.combinedWitness (rlcAlgebra context.key) challenges
                (InputAuthority.productAssignments
                  data context.alignment))
              child) := by
  rcases
      Protocol.BlockLane.OutputRefinement.complete_of_paperObligations
        context.covers StatementInput.polynomial context.piCcsSchedule
        context.priorState context.profile context.piCcsStatement data
        input.publicInput publicRingColumns publicFits (commit context.key)
        context.alignment context.input production_norm_stages.1 paper
        input.sources with
    ⟨piCcsCertificate, piCcsTranscript, outputAuthority, outputsHold⟩
  have samplerBound :
      Sampler.Bound context.piRlcMachine
        (Protocol.BlockLane.derive StatementInput.polynomial
          context.piCcsSchedule context.priorState context.profile
          context.piCcsStatement piCcsCertificate).finalState
        challenges := by
    exact samplerAvailable piCcsCertificate piCcsTranscript
  exact complete_of_honestPiCcsAndSampler context data paper input running
    piCcsCertificate piCcsTranscript outputAuthority outputsHold challenges
    samplerBound

/-- Honest concrete NIFS completeness without a hidden total-sampler
assumption. The independently constructed `Pi_CCS` prefix either extends
through one complete transcript-bound challenge batch and both tail phases,
or the result names the exact bounded-sampler coordinate that shortfalls. -/
theorem complete_or_samplerShortfall
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity)
    (data : Data shape)
    (paper : Semantics.Paper.Holds data)
    (input : SemanticInput context data)
    (running : RunningAuthority.Accepted context) :
    (∃ challenges : Fin arity.total -> RingF,
      ∃ certificate :
          Certificate (arity := arity)
            publicRingColumns publicFits verifierRows context.piCcsInput,
        Accepted context certificate ∧
          CertificateRefinement context data certificate ∧
          ∀ child,
            CE.Holds (semantics context.key) productionGlobalParams
              (outputChildren context certificate child)
              ((decAlgebra context.key).splitAssignment
                (PiRLC.combinedWitness (rlcAlgebra context.key) challenges
                  (InputAuthority.productAssignments
                    data context.alignment))
                child)) \/
      HonestSamplerShortfall context data := by
  rcases
      Protocol.BlockLane.OutputRefinement.complete_of_paperObligations
        context.covers StatementInput.polynomial context.piCcsSchedule
        context.priorState context.profile context.piCcsStatement data
        input.publicInput publicRingColumns publicFits (commit context.key)
        context.alignment context.input production_norm_stages.1 paper
        input.sources with
    ⟨piCcsCertificate, piCcsTranscript, outputAuthority, outputsHold⟩
  let piRlcInitialState :=
    (Protocol.BlockLane.derive StatementInput.polynomial
      context.piCcsSchedule context.priorState context.profile
      context.piCcsStatement piCcsCertificate).finalState
  rcases Sampler.exists_bound_or_exists_shortfall context.piRlcMachine
      arity.total piRlcInitialState with bound | shortfall
  · rcases bound with ⟨challenges, ⟨samplerBound⟩⟩
    apply Or.inl
    refine ⟨challenges, ?_⟩
    apply complete_of_honestPiCcsAndSampler context data paper input running
      piCcsCertificate piCcsTranscript outputAuthority outputsHold challenges
    simpa [piRlcInitialState] using samplerBound
  · apply Or.inr
    refine ⟨piCcsCertificate, piCcsTranscript, outputAuthority, ?_⟩
    simpa [piRlcInitialState] using shortfall

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
