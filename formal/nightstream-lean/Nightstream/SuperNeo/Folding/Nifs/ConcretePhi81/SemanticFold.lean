import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.RunningAuthority
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Paper
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.InputAuthority

/-!
Certificate-independent semantics for one concrete Phi81 NIFS fold.

Protocol: SuperNeo NIFS.
Phase: mathematical `Pi_CCS -> Pi_RLC -> Pi_DEC` transition.
Constraint family: semantic obligations only; this file emits no rows.

Assurance tier: model-level.

Owns: binding one independent source family to both public input surfaces; one
raw `(row point, Pi_RLC challenges)` carrier and its separate strong-set
obligation; canonical honest `Pi_CCS` outputs; the canonical combined parent;
canonical radix children; and the resulting parent-plus-children relation.

Does not own: a raw certificate, SumCheck messages, Fiat--Shamir replay,
sampler traces, bad-event probabilities, executable checking, Rust, R1CS,
rows, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: `Holds` is stated from the paper relation and authoritative
source assignments. It never asks an implementation what the output means.
The evaluation point and valid `Pi_RLC` challenges are existential semantic
choices; every remaining output field is computed. Physical transcript and
extraction arguments must refine into this relation in a separate module.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.semantic.input.polynomial` | public polynomial input is the exact source projection | checked bridge | `PublicInputBound` |
| `nifs.semantic.input.product` | every public source field binds to the same independent source family | checked bridge | `InputBound` |
| `nifs.semantic.pi_ccs.outputs` | re-evaluate every authoritative source at one typed row point | computed | `outputs` |
| `nifs.semantic.pi_rlc.challenge` | every scalar belongs to the verifier-owned strong set | checked | `ChallengesValid`, `Holds.challengesValid` |
| `nifs.semantic.pi_rlc.parent` | combine the honest outputs and their assignments with the same challenges | computed | `parentOf`, `combinedAssignment` |
| `nifs.semantic.pi_dec.children` | split the combined assignment and compute every fresh child | computed | `childrenOf` |
| `nifs.semantic.running` | bootstrap parent absence or strict active-parent recomposition | checked | `RunningAuthority.Accepted` |
| `nifs.semantic.fold` | paper truth, input authority, incoming authority, and exact computed result | independent specification | `Holds` |
| `nifs.semantic.completeness` | every paper-valid bound input and semantic witness yields its computed result | derived | `complete` |
| `nifs.semantic.opening.parent` | the computed parent has the combined private opening | derived | `Holds.parentOpening` |
| `nifs.semantic.opening.children` | every computed child has its radix-split private opening | derived | `Holds.childOpening` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
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

/-- The public polynomial verifier input is the exact projection of the
independent source family. -/
def PublicInputBound
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape) : Prop :=
  context.piCcsInput = PublicInput.ofSources data

/-- The complete public source product is the exact public image of the same
independent source family. -/
def InputBound
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape) : Prop :=
  InputAuthority.BoundToSources publicRingColumns publicFits
    (commit context.key) data context.alignment context.input

/-- Both public input surfaces bind to one source family. They remain separate
fields because neither surface is definitionally derivable from the other. -/
structure Input
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape) : Prop where
  publicInput : PublicInputBound context data
  sources : InputBound context data

/-- The only non-source values needed to state one fold realization.

This carrier deliberately contains no validity proof. Keeping the raw
challenge vector separate from `ChallengesValid` is what permits a later
removal witness to mutate or omit that obligation. Transcript replay is also
absent; a physical refinement must prove that its derived values instantiate
this carrier. -/
structure Witness
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity) where
  point : CubePoint K shape.rowVariables
  challenges : Fin arity.total -> RingF

/-- Strong-set membership is an explicit semantic obligation, not evidence
baked into the candidate carrier. -/
def ChallengesValid
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (witness : Witness context) : Prop :=
  forall source,
    (rlcAlgebra context.key).challengeValid (witness.challenges source)

/-- Authoritative assignments in the exact public-product order. -/
def assignments
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape) :
    Fin arity.total -> SourceAssignment shape :=
  InputAuthority.productAssignments data context.alignment

/-- The sole relation structure derived from the independent source data. -/
def systemOf
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape) :
    Phi81Relation.Structure
      (RelationShape shape publicRingColumns publicFits) :=
  Phi81Relation.Structure.ofSourceData publicRingColumns publicFits data

/-- Canonical `Pi_CCS` handoff: every source is re-evaluated at the same typed
row point. No prover output message appears in this definition. -/
def outputs
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape)
    (witness : Witness context) :
    Product shape publicRingColumns publicFits (CommitmentValue verifierRows)
      productionGlobalParams arity :=
  PiCCS.honestOutputs (semantics context.key) context.input
    (assignments context data) witness.point

/-- Canonical private opening of the combined parent. -/
def combinedAssignment
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape)
    (witness : Witness context) :
    Phi81Relation.Assignment
      (RelationShape shape publicRingColumns publicFits) :=
  PiRLC.combinedWitness (rlcAlgebra context.key) witness.challenges
    (assignments context data)

/-- Canonical public `Pi_RLC` parent computed from honest `Pi_CCS` outputs. -/
def parentOf
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape)
    (witness : Witness context) :
    Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows) :=
  PiRLC.combinedOutput (rlcAlgebra context.key) (systemOf context data)
    witness.point (outputs context data witness) witness.challenges

/-- Canonical public `Pi_DEC` children computed from the parent opening. -/
def childrenOf
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape)
    (witness : Witness context) :
    Fin productionGlobalParams.k ->
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows) :=
  PiDEC.childrenOf (decAlgebra context.key) (parentOf context data witness)
    (combinedAssignment context data witness)

/-- Public `Pi_DEC` view used to state strict recomposition for any proposed
parent-plus-children result. -/
def piDecAttempt
    (parent :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows))
    (children : Fin productionGlobalParams.k ->
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) :
    PiDEC.Attempt
      (Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.PublicInput
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.Point
        (RelationShape shape publicRingColumns publicFits))
      Phi81Relation.Evaluation (CommitmentValue verifierRows)
      productionGlobalParams := {
  parent := parent
  children := children
}

/-- Certificate-independent mathematical fold relation. The parent and child
family are public; the rich source data and the two irreducible fold choices
remain semantic witnesses. -/
structure Holds
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape)
    (parent :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows))
    (children : Fin productionGlobalParams.k ->
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) : Prop where
  paper : SplitNc.Semantics.Paper.Holds data
  input : Input context data
  running : RunningAuthority.Accepted context
  witness : Witness context
  challengesValid : ChallengesValid context witness
  parent_eq : parent = parentOf context data witness
  children_eq : children = childrenOf context data witness

/-- Perfect completeness of the certificate-independent relation: once the
paper statement, both public bindings, incoming authority, and one valid
challenge witness are available, the computed result satisfies `Holds` by
construction. -/
theorem complete
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape)
    (paper : SplitNc.Semantics.Paper.Holds data)
    (input : Input context data)
    (running : RunningAuthority.Accepted context)
    (witness : Witness context)
    (challengesValid : ChallengesValid context witness) :
    Holds context data (parentOf context data witness)
      (childrenOf context data witness) := {
  paper := paper
  input := input
  running := running
  witness := witness
  challengesValid := challengesValid
  parent_eq := rfl
  children_eq := rfl
}

/-- A CE statement with the canonical child structure, point, stage, and
opening is definitionally the corresponding `Pi_DEC.childrenOf` statement.
This is the extraction-to-canonical-child bridge used by the physical layer. -/
theorem child_eq_childrenOf_of_holds
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (parent :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows))
    (assignment :
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (child : Fin productionGlobalParams.k)
    (candidate :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows))
    (sameStructure :
      candidate.constraintSystem = parent.constraintSystem)
    (samePoint : candidate.point = parent.point)
    (fresh : candidate.stage = .fresh)
    (candidateHolds :
      CE.Holds (semantics context.key) productionGlobalParams candidate
        ((decAlgebra context.key).splitAssignment assignment child)) :
    candidate =
      PiDEC.childrenOf (decAlgebra context.key) parent assignment child := by
  have commitmentEq :
      candidate.commitment =
        (semantics context.key).commit
          ((decAlgebra context.key).splitAssignment assignment child) :=
    candidateHolds.1.1.symm
  have publicInputEq :
      candidate.publicInput =
        (semantics context.key).projectPublicInput
          ((decAlgebra context.key).splitAssignment assignment child) :=
    candidateHolds.1.2.1.symm
  have evaluationsEq :
      candidate.evaluations =
        (semantics context.key).evaluations parent.constraintSystem
          ((decAlgebra context.key).splitAssignment assignment child)
          parent.point := by
    calc
      candidate.evaluations =
          (semantics context.key).evaluations candidate.constraintSystem
            ((decAlgebra context.key).splitAssignment assignment child)
            candidate.point := candidateHolds.2.2.symm
      _ = (semantics context.key).evaluations parent.constraintSystem
            ((decAlgebra context.key).splitAssignment assignment child)
            parent.point := by rw [sameStructure, samePoint]
  rcases candidate with
    ⟨candidateSystem, candidateCommitment, candidatePublicInput,
      candidatePoint, candidateEvaluations, candidateStage⟩
  rcases parent with
    ⟨parentSystem, parentCommitment, parentPublicInput, parentPoint,
      parentEvaluations, parentStage⟩
  simp only [PiDEC.childrenOf]
  simp_all [semantics, productSemantics, commit, decAlgebra,
    PiDECAlgebra.Algebra.concrete]

namespace Holds

/-- Every canonical `Pi_CCS` output is a genuine fresh CE opening of the
corresponding authoritative assignment. -/
theorem outputsHold
    {context :
      Context shape State publicRingColumns publicFits verifierRows arity}
    {data : Data shape}
    {parent :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)}
    {children : Fin productionGlobalParams.k ->
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)}
    (holds : Holds context data parent children) :
    forall source,
      CE.Holds (semantics context.key) productionGlobalParams
        (outputs context data holds.witness source)
        (assignments context data source) := by
  apply PiCCS.product_complete (semantics context.key) productionGlobalParams
    arity context.input (assignments context data) holds.witness.point
  · exact InputAuthority.BoundToSources.sourceFresh publicRingColumns
      publicFits (commit context.key) data context.alignment context.input
      holds.input.sources
  · exact InputAuthority.allSourcesHold publicRingColumns publicFits
      (commit context.key) data context.alignment context.input
      production_norm_stages.1 holds.paper holds.input.sources
  · intro source
    exact Phi81Relation.evaluationPointValid_holds
      (context.input.source source).constraintSystem holds.witness.point

/-- The public parent has exactly the combined private opening. -/
theorem parentOpening
    {context :
      Context shape State publicRingColumns publicFits verifierRows arity}
    {data : Data shape}
    {parent :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)}
    {children : Fin productionGlobalParams.k ->
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)}
    (holds : Holds context data parent children) :
    CE.Holds (semantics context.key) productionGlobalParams parent
      (combinedAssignment context data holds.witness) := by
  rw [holds.parent_eq]
  apply PiRLC.combinedOutput_holds (semantics context.key)
    productionGlobalParams (rlcAlgebra context.key) arity
    (systemOf context data) holds.witness.point
    (outputs context data holds.witness) holds.witness.challenges
    (assignments context data)
  · intro source
    rfl
  · intro source
    simpa [outputs, systemOf, PiCCS.honestOutputs, PiCCS.honestOutput] using
      (InputAuthority.BoundToSources.sourceStructure publicRingColumns
        publicFits (commit context.key) data context.alignment context.input
        holds.input.sources source)
  · intro source
    rfl
  · exact holds.challengesValid
  · exact holds.outputsHold
  · exact Phi81Relation.evaluationPointValid_holds
      (systemOf context data) holds.witness.point

/-- Every public child has exactly its canonical radix-split opening. -/
theorem childOpening
    {context :
      Context shape State publicRingColumns publicFits verifierRows arity}
    {data : Data shape}
    {parent :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)}
    {children : Fin productionGlobalParams.k ->
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)}
    (holds : Holds context data parent children)
    (child : Fin productionGlobalParams.k) :
    CE.Holds (semantics context.key) productionGlobalParams (children child)
      ((decAlgebra context.key).splitAssignment
        (combinedAssignment context data holds.witness) child) := by
  have parentHolds := holds.parentOpening
  rw [holds.parent_eq] at parentHolds
  rw [holds.children_eq]
  exact PiDEC.childrenOf_holds (semantics context.key) productionGlobalParams
    (decAlgebra context.key) (parentOf context data holds.witness)
    (combinedAssignment context data holds.witness) rfl parentHolds child

/-- The semantic result satisfies the strict public `Pi_DEC` equations. -/
theorem piDecAccepted
    {context :
      Context shape State publicRingColumns publicFits verifierRows arity}
    {data : Data shape}
    {parent :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)}
    {children : Fin productionGlobalParams.k ->
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)}
    (holds : Holds context data parent children) :
    PiDEC.Accepted (decAlgebra context.key) (piDecAttempt parent children) := by
  have parentHolds := holds.parentOpening
  rw [holds.parent_eq] at parentHolds
  rw [holds.parent_eq, holds.children_eq]
  simpa [piDecAttempt] using
    (PiDEC.complete (semantics context.key) productionGlobalParams
      (decAlgebra context.key) (parentOf context data holds.witness)
      (combinedAssignment context data holds.witness) rfl parentHolds).1

end Holds

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticFold
