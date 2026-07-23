import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
import Nightstream.SuperNeo.Folding.Nifs.PaperProfile
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Paper
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.InputAuthority

/-!
Paper-owned fixed-active SuperNeo NIFS relation for the concrete Phi81
algebras.

Protocol: SuperNeo Sections 7.3--7.5.
Phase: one `CCS(2)` source plus fourteen `CE(2)` sources, followed by
`Pi_CCS -> Pi_RLC -> Pi_DEC` and fourteen `CE(2)` outputs.
Constraint family: independent semantic obligations only; this file emits no
rows.

Assurance tier: model-level.

Owns: the concrete Phi81 instantiation of the independent production profile;
the verifier commitment key and source-partition alignment; authoritative
source binding; the three Section 7.3 relation obligations; one common row
point; fifteen strong-set challenges; the coordinatewise Section-7.5 public
split and fixed matrix-evaluation arity; exact operational `Pi_DEC` public
acceptance; and the canonical
honest-completeness dataflow.

Does not own: Split-NC polynomial inputs, either SumCheck message flow,
Fiat--Shamir replay, incoming or outgoing parent caches, transcript state,
HyperNova lifecycle state, Rust, R1CS, costs, necessity, or row removal.

Authority boundary: the public source is exactly one fresh CCS statement and
the complete fourteen-element running CE product. The paper verifier computes
the target public inputs, structure, point, and fresh stage from the combined
parent; the prover supplies only fixed-arity child commitments and evaluations.
The combined `Pi_RLC` parent is an internal computed intermediate, never an
additional public authority. Source assignments, row point, and challenges
are existential semantic data; public source fields must bind to those
assignments explicitly. Target-child CE openings are premises of the
knowledge reduction, not checks added to public `Pi_DEC` acceptance.

This relation states the coherent relation-level composition. It does not
claim that the current Split-NC two-SumCheck verifier is message-for-message
equivalent to the paper's disputed displayed single-`Q` formula.

Emits constraints: no.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.paper.fixed_active.profile` | instantiate the generic `1 + 14 -> 14` paper profile with concrete Phi81 algebras | refinement | `toGenericProfile`, `Realization.toGeneric` |
| `nifs.paper.fixed_active.source` | every public field binds to one authoritative source family | checked | `InputBound` |
| `nifs.paper.fixed_active.pi_ccs` | fresh CCS, all strict norms, carried evaluations, and one common output point | independent/computed | `Paper.Holds`, `outputs` |
| `nifs.paper.fixed_active.pi_rlc` | fifteen valid challenges determine one combined parent | checked/computed | `ChallengesValid`, `parentOf` |
| `nifs.paper.fixed_active.pi_dec` | compute child public/copy fields, enforce exact matrix-evaluation arity, and check commitment/evaluation recomposition | checked/computed | `decPublicInputSplit`, `decEvaluationArity`, `Realization.piDecAccepted`, `childrenOf`, `complete` |
| `nifs.paper.fixed_active.transition` | public-verifier source-to-children acceptance relation | independent specification | `Transition` |
| `nifs.paper.fixed_active.completeness` | valid premises construct the canonical target | derived | `complete` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

/-- Exact selected SuperNeo profile: one fresh CCS source and all fourteen
running CE sources. -/
abbrev arity : BatchArity productionGlobalParams :=
  Nightstream.SuperNeo.Folding.Nifs.PaperProfile.arity

@[simp] theorem arity_freshCount : arity.freshCount = 1 := rfl

@[simp] theorem arity_mode : arity.mode = .active := rfl

@[simp] theorem arity_runningCount :
    arity.mode.count productionGlobalParams = 14 := rfl

@[simp] theorem arity_total : arity.total = 15 := rfl

@[simp] theorem outputCount : productionGlobalParams.k = 14 := rfl

@[simp] theorem baseNormBound : productionGlobalParams.b = 2 := rfl

namespace PaperProfile

/-- Static verifier data that belongs to the paper relation. Transcript and
lifecycle fields are deliberately absent. -/
structure Profile
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  key : PiRLCAlgebra.Commitment.Key
    (RelationShape shape publicRingColumns publicFits) verifierRows
  alignment : SourceAlignment shape productionGlobalParams arity

/-- Exact public input: one fresh CCS statement followed by fourteen running
CE statements. -/
abbrev Source
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) :=
  SourceProduct shape publicRingColumns publicFits
    (PiRLCAlgebra.Commitment.Value verifierRows) productionGlobalParams arity

/-- Exact public output: fourteen fresh CE children. -/
abbrev Target
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) :=
  Fin productionGlobalParams.k ->
    Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (PiRLCAlgebra.Commitment.Value verifierRows)

namespace Profile

/-- The aligned semantic source family has exactly one fresh assignment. -/
@[simp] theorem freshCount_eq
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows) :
    shape.freshCount = 1 := by
  simpa using profile.alignment.freshCount_eq.symm

/-- The aligned semantic source family has exactly fourteen running
assignments. -/
@[simp] theorem runningCount_eq
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows) :
    shape.runningCount = 14 := by
  simpa using profile.alignment.runningCount_eq.symm

/-- The aligned semantic source family has exactly fifteen assignments. -/
@[simp] theorem sourceCount_eq
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows) :
    shape.sourceCount = 15 := by
  simpa using profile.alignment.total_eq_sourceCount.symm

end Profile

/-- Raw semantic coins for one relation realization. Validity remains a
separate obligation so challenge membership can later receive an independent
necessity witness. -/
structure Witness (shape : SemanticShape) where
  point : CubePoint K shape.rowVariables
  challenges : Fin arity.total -> RingF

/-- Sole typed commitment map selected by the paper profile's verifier key. -/
def commit
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows)
    (assignment : SourceAssignment shape) :
    PiRLCAlgebra.Commitment.Value verifierRows :=
  PiRLCAlgebra.Commitment.commit profile.key assignment

/-- Concrete relation semantics selected by the paper profile's verifier
key. -/
def semantics
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows) :
    RelationSemantics
      (Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits))
      (SourceAssignment shape)
      (Phi81Relation.PublicInput
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.Point
        (RelationShape shape publicRingColumns publicFits))
      Phi81Relation.Evaluation
      (PiRLCAlgebra.Commitment.Value verifierRows) :=
  productSemantics publicRingColumns publicFits (commit profile)

/-- Independently proved concrete `Pi_RLC` algebra selected by the profile. -/
def rlcAlgebra
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows) :=
  PiRLCAlgebra.Algebra.concrete profile.key

/-- Independently proved concrete `Pi_DEC` algebra selected by the profile. -/
def decAlgebra
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows) :=
  PiDECAlgebra.Algebra.concrete profile.key

/-- Verifier-owned coordinatewise public split selected by the same concrete
profile. -/
def decPublicInputSplit
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows) :=
  PiDECAlgebra.PaperVerifier.publicInputSplit profile.key

/-- Verifier-owned matrix-evaluation arity selected by the relation shape. -/
def decEvaluationArity
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows) :=
  PiDECAlgebra.PaperVerifier.evaluationArity profile.key

/-- Install the concrete relation and independently proved Phi81 tail
algebras into the abstract paper profile. -/
def toGenericProfile
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows) :
    Nightstream.SuperNeo.Folding.Nifs.PaperProfile.Profile
      (Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits))
      (SourceAssignment shape)
      (Phi81Relation.PublicInput
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.Point
        (RelationShape shape publicRingColumns publicFits))
      Phi81Relation.Evaluation
      (PiRLCAlgebra.Commitment.Value verifierRows) RingF := {
  semantics := semantics profile
  rlcAlgebra := rlcAlgebra profile
  decAlgebra := decAlgebra profile
  decPublicInputSplit := decPublicInputSplit profile
  decEvaluationArity := decEvaluationArity profile
}

/-- Public source statements bind to one authoritative source family in the
exact fresh/running partition. -/
def InputBound
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
  (profile : Profile shape publicRingColumns publicFits verifierRows)
  (source : Source shape publicRingColumns publicFits verifierRows)
  (data : Data shape) : Prop :=
  InputAuthority.BoundToSources publicRingColumns publicFits
    (commit profile) data profile.alignment source

/-- Authoritative assignments in exact public-product order. -/
def assignments
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows)
    (data : Data shape) : Fin arity.total -> SourceAssignment shape :=
  InputAuthority.productAssignments data profile.alignment

/-- Install the concrete authoritative assignment family and raw coins into
the abstract paper witness. -/
def toGenericWitness
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows)
    (data : Data shape)
    (witness : Witness shape) :
    Nightstream.SuperNeo.Folding.Nifs.PaperProfile.Witness
      (SourceAssignment shape)
      (Phi81Relation.Point
        (RelationShape shape publicRingColumns publicFits)) RingF := {
  assignments := assignments profile data
  point := witness.point
  challenges := witness.challenges
}

/-- Sole relation structure derived from the independent source data. -/
def systemOf
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (_profile : Profile shape publicRingColumns publicFits verifierRows)
    (data : Data shape) :
    Phi81Relation.Structure
      (RelationShape shape publicRingColumns publicFits) :=
  Phi81Relation.Structure.ofSourceData publicRingColumns publicFits data

/-- Canonical `Pi_CCS` outputs: re-evaluate every authoritative source at one
shared row point. -/
def outputs
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows)
    (source : Source shape publicRingColumns publicFits verifierRows)
    (data : Data shape)
    (witness : Witness shape) :
    Product shape publicRingColumns publicFits
      (PiRLCAlgebra.Commitment.Value verifierRows) productionGlobalParams
      arity :=
  PiCCS.honestOutputs (semantics profile) source (assignments profile data)
    witness.point

/-- Strong-set membership for all fifteen `Pi_RLC` challenges. -/
def ChallengesValid
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows)
    (witness : Witness shape) : Prop :=
  forall source,
    (rlcAlgebra profile).challengeValid (witness.challenges source)

/-- Canonical private opening of the internal combined parent. -/
def combinedAssignment
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows)
    (data : Data shape)
    (witness : Witness shape) : SourceAssignment shape :=
  PiRLC.combinedWitness (rlcAlgebra profile) witness.challenges
    (assignments profile data)

/-- Canonical `Pi_RLC` parent. It is an internal value, not part of the
public paper target. -/
def parentOf
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows)
    (source : Source shape publicRingColumns publicFits verifierRows)
    (data : Data shape)
    (witness : Witness shape) :
    Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (PiRLCAlgebra.Commitment.Value verifierRows) :=
  PiRLC.combinedOutput (rlcAlgebra profile) (systemOf profile data)
    witness.point (outputs profile source data witness) witness.challenges

/-- Canonical public `Pi_DEC` children computed from the internal parent and
its authoritative opening. -/
def childrenOf
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows)
    (source : Source shape publicRingColumns publicFits verifierRows)
    (data : Data shape)
    (witness : Witness shape) :
    Target shape publicRingColumns publicFits verifierRows :=
  PiDEC.childrenOf (decAlgebra profile) (parentOf profile source data witness)
    (combinedAssignment profile data witness)

/-- One exact source-to-target realization of the fixed paper profile. -/
structure Realization
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows)
    (source : Source shape publicRingColumns publicFits verifierRows)
    (data : Data shape)
    (target : Target shape publicRingColumns publicFits verifierRows)
    (witness : Witness shape) : Prop where
  paper : Semantics.Paper.Holds data
  input : InputBound profile source data
  challengesValid : ChallengesValid profile witness
  piDecAccepted : PiDEC.PaperVerifier.OutputAccepted (decAlgebra profile)
    (decPublicInputSplit profile) (decEvaluationArity profile)
    (parentOf profile source data witness) target

/-- Public source binding identifies the generic profile's fresh-source
structure with the sole structure derived from authoritative Split-NC data. -/
theorem genericSystem_eq_of_inputBound
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows)
    (source : Source shape publicRingColumns publicFits verifierRows)
    (data : Data shape)
    (input : InputBound profile source data) :
    Nightstream.SuperNeo.Folding.Nifs.PaperProfile.systemOf source =
      systemOf profile data := by
  let first : Fin arity.total :=
    Fin.castAdd (arity.mode.count productionGlobalParams)
      Nightstream.SuperNeo.Folding.Nifs.PaperProfile.firstFresh
  have bound := InputAuthority.BoundToSources.sourceStructure
    publicRingColumns publicFits (commit profile) data profile.alignment source
    input first
  simpa [first, Nightstream.SuperNeo.Folding.Nifs.PaperProfile.systemOf,
    Nightstream.SuperNeo.Folding.Nifs.PaperProfile.firstFresh,
    PiCCS.InputProduct.source, PiCCS.Source.constraintSystem, systemOf] using
    bound

/-- Under exact public/source binding, the abstract and concrete internal
parents are the same statement. -/
theorem genericParent_eq_of_inputBound
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows)
    (source : Source shape publicRingColumns publicFits verifierRows)
    (data : Data shape)
    (witness : Witness shape)
    (input : InputBound profile source data) :
    Nightstream.SuperNeo.Folding.Nifs.PaperProfile.parentOf
        (toGenericProfile profile) source
        (toGenericWitness profile data witness) =
      parentOf profile source data witness := by
  unfold Nightstream.SuperNeo.Folding.Nifs.PaperProfile.parentOf
    Nightstream.SuperNeo.Folding.Nifs.PaperProfile.outputs
    parentOf outputs
  rw [genericSystem_eq_of_inputBound profile source data input]
  rfl

/-- Under the exact public/source binding, the abstract paper computation and
the concrete Phi81 computation return the same fourteen children. -/
theorem genericChildren_eq_of_inputBound
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows)
    (source : Source shape publicRingColumns publicFits verifierRows)
    (data : Data shape)
    (witness : Witness shape)
    (input : InputBound profile source data) :
    Nightstream.SuperNeo.Folding.Nifs.PaperProfile.childrenOf
        (toGenericProfile profile) source
        (toGenericWitness profile data witness) =
      childrenOf profile source data witness := by
  unfold Nightstream.SuperNeo.Folding.Nifs.PaperProfile.childrenOf
    Nightstream.SuperNeo.Folding.Nifs.PaperProfile.parentOf
    Nightstream.SuperNeo.Folding.Nifs.PaperProfile.outputs
    Nightstream.SuperNeo.Folding.Nifs.PaperProfile.combinedAssignment
    childrenOf parentOf outputs combinedAssignment
  rw [genericSystem_eq_of_inputBound profile source data input]
  rfl

/-- Independent fixed-active public-verifier acceptance relation. Only the
source and fourteen children are public; source data and semantic coins remain
existential. Child CE membership remains a separate soundness premise. -/
def Transition
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows)
    (source : Source shape publicRingColumns publicFits verifierRows)
    (target : Target shape publicRingColumns publicFits verifierRows) : Prop :=
  exists data : Data shape,
    exists witness : Witness shape,
      Realization profile source data target witness

/-- Indexed honest completeness of the fixed paper relation: once source
truth, public binding, and valid challenges are supplied, the canonical
children form one exact realization of the operational `Pi_DEC` verifier. -/
theorem completeRealization
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows)
    (source : Source shape publicRingColumns publicFits verifierRows)
    (data : Data shape)
    (paper : Semantics.Paper.Holds data)
    (input : InputBound profile source data)
    (witness : Witness shape)
    (challengesValid : ChallengesValid profile witness) :
    Realization profile source data
      (childrenOf profile source data witness) witness := by
  have outputsValid : forall index,
      CE.Holds (semantics profile) productionGlobalParams
        (outputs profile source data witness index)
        (assignments profile data index) := by
    apply PiCCS.product_complete (semantics profile) productionGlobalParams
      arity source (assignments profile data) witness.point
    · exact InputAuthority.BoundToSources.sourceFresh publicRingColumns
        publicFits (commit profile) data profile.alignment source input
    · exact InputAuthority.allSourcesHold publicRingColumns publicFits
        (commit profile) data profile.alignment source
        production_norm_stages.1 paper input
    · intro index
      exact Phi81Relation.evaluationPointValid_holds
        (source.source index).constraintSystem witness.point
  have parentValid :
      CE.Holds (semantics profile) productionGlobalParams
        (parentOf profile source data witness)
        (combinedAssignment profile data witness) := by
    apply PiRLC.combinedOutput_holds (semantics profile)
      productionGlobalParams (rlcAlgebra profile) arity
      (systemOf profile data) witness.point
      (outputs profile source data witness) witness.challenges
      (assignments profile data)
    · intro index
      rfl
    · intro index
      simpa [outputs, systemOf, PiCCS.honestOutputs,
        PiCCS.honestOutput] using
        (InputAuthority.BoundToSources.sourceStructure publicRingColumns
          publicFits (commit profile) data profile.alignment source input index)
    · intro index
      rfl
    · exact challengesValid
    · exact outputsValid
    · exact Phi81Relation.evaluationPointValid_holds
        (systemOf profile data) witness.point
  have piDec := PiDEC.PaperVerifier.output_complete (semantics profile)
    productionGlobalParams (decAlgebra profile) (decPublicInputSplit profile)
    (decEvaluationArity profile)
    (parentOf profile source data witness)
    (combinedAssignment profile data witness) rfl parentValid
  refine {
    paper := paper
    input := input
    challengesValid := challengesValid
    piDecAccepted := ?_
  }
  simpa only [childrenOf] using piDec.1

/-- Conditional honest completeness of the fixed public transition. This is
the existential wrapper around `completeRealization`. -/
theorem complete
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (profile : Profile shape publicRingColumns publicFits verifierRows)
    (source : Source shape publicRingColumns publicFits verifierRows)
    (data : Data shape)
    (paper : Semantics.Paper.Holds data)
    (input : InputBound profile source data)
    (witness : Witness shape)
    (challengesValid : ChallengesValid profile witness) :
    Transition profile source (childrenOf profile source data witness) := by
  exact ⟨data, witness,
    completeRealization profile source data paper input witness challengesValid⟩

namespace Realization

/-- The concrete source relation and binding instantiate the abstract
paper-level NIFS graph. This is a refinement theorem, not a type alias. -/
theorem toGeneric
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {profile : Profile shape publicRingColumns publicFits verifierRows}
    {source : Source shape publicRingColumns publicFits verifierRows}
    {data : Data shape}
    {target : Target shape publicRingColumns publicFits verifierRows}
    {witness : Witness shape}
    (holds : Realization profile source data target witness) :
    Nightstream.SuperNeo.Folding.Nifs.PaperProfile.Realization
      (toGenericProfile profile) source target
      (toGenericWitness profile data witness) := by
  refine {
    sourceFresh := ?_
    sourceHolds := ?_
    commonStructure := ?_
    runningCommonPoint := ?_
    newPointValid := ?_
    challengesValid := holds.challengesValid
    piDecAccepted := ?_
  }
  · exact InputAuthority.BoundToSources.sourceFresh publicRingColumns
      publicFits (commit profile) data profile.alignment source holds.input
  · exact InputAuthority.allSourcesHold publicRingColumns publicFits
      (commit profile) data profile.alignment source
      production_norm_stages.1 holds.paper holds.input
  · intro index
    exact (InputAuthority.BoundToSources.sourceStructure publicRingColumns
      publicFits (commit profile) data profile.alignment source holds.input
      index).trans
        (genericSystem_eq_of_inputBound profile source data holds.input).symm
  · intro left right
    exact (holds.input.running left).point.trans
      (holds.input.running right).point.symm
  · exact Phi81Relation.evaluationPointValid_holds
      (Nightstream.SuperNeo.Folding.Nifs.PaperProfile.systemOf source)
      witness.point
  · simpa only [genericParent_eq_of_inputBound profile source data witness
      holds.input] using holds.piDecAccepted

/-- Every canonical `Pi_CCS` output is a genuine fresh CE opening of the
corresponding authoritative assignment. -/
theorem outputsHold
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {profile : Profile shape publicRingColumns publicFits verifierRows}
    {source : Source shape publicRingColumns publicFits verifierRows}
    {data : Data shape}
    {target : Target shape publicRingColumns publicFits verifierRows}
    {witness : Witness shape}
    (holds : Realization profile source data target witness) :
    forall index,
      CE.Holds (semantics profile) productionGlobalParams
        (outputs profile source data witness index)
        (assignments profile data index) := by
  apply PiCCS.product_complete (semantics profile) productionGlobalParams
    arity source (assignments profile data) witness.point
  · exact InputAuthority.BoundToSources.sourceFresh publicRingColumns
      publicFits (commit profile) data profile.alignment source holds.input
  · exact InputAuthority.allSourcesHold publicRingColumns publicFits
      (commit profile) data profile.alignment source
      production_norm_stages.1 holds.paper holds.input
  · intro index
    exact Phi81Relation.evaluationPointValid_holds
      (source.source index).constraintSystem witness.point

/-- The internal combined parent has exactly the canonical challenge-folded
private opening. -/
theorem parentOpening
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {profile : Profile shape publicRingColumns publicFits verifierRows}
    {source : Source shape publicRingColumns publicFits verifierRows}
    {data : Data shape}
    {target : Target shape publicRingColumns publicFits verifierRows}
    {witness : Witness shape}
    (holds : Realization profile source data target witness) :
    CE.Holds (semantics profile) productionGlobalParams
      (parentOf profile source data witness)
      (combinedAssignment profile data witness) := by
  apply PiRLC.combinedOutput_holds (semantics profile)
    productionGlobalParams (rlcAlgebra profile) arity
    (systemOf profile data) witness.point
    (outputs profile source data witness) witness.challenges
    (assignments profile data)
  · intro index
    rfl
  · intro index
    simpa [outputs, systemOf, PiCCS.honestOutputs, PiCCS.honestOutput] using
      (InputAuthority.BoundToSources.sourceStructure publicRingColumns
        publicFits (commit profile) data profile.alignment source
        holds.input index)
  · intro index
    rfl
  · exact holds.challengesValid
  · exact holds.outputsHold
  · exact Phi81Relation.evaluationPointValid_holds
      (systemOf profile data) witness.point

/-- The candidate target is exactly an output of the operational `Pi_DEC`
verifier against the source-derived parent. Only its commitments/evaluations
are prover-supplied fields. -/
theorem outputAccepted
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {profile : Profile shape publicRingColumns publicFits verifierRows}
    {source : Source shape publicRingColumns publicFits verifierRows}
    {data : Data shape}
    {target : Target shape publicRingColumns publicFits verifierRows}
    {witness : Witness shape}
    (holds : Realization profile source data target witness) :
    PiDEC.PaperVerifier.OutputAccepted (decAlgebra profile)
      (decPublicInputSplit profile) (decEvaluationArity profile)
      (parentOf profile source data witness) target :=
  holds.piDecAccepted

/-- Compatibility projection into the older recomposition relation. -/
theorem outputRecompositionAccepted
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {profile : Profile shape publicRingColumns publicFits verifierRows}
    {source : Source shape publicRingColumns publicFits verifierRows}
    {data : Data shape}
    {target : Target shape publicRingColumns publicFits verifierRows}
    {witness : Witness shape}
    (holds : Realization profile source data target witness) :
    PiDEC.Accepted (decAlgebra profile) {
      parent := parentOf profile source data witness
      children := target
    } :=
  holds.piDecAccepted.toRecompositionAccepted

/-- Valid target openings either recompose to the source-derived combined
opening or expose the standard parent-opening binding collision. This theorem
does not force the deterministic honest split. -/
theorem parentOpening_eq_recompose_or_bindingCollision
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {profile : Profile shape publicRingColumns publicFits verifierRows}
    {source : Source shape publicRingColumns publicFits verifierRows}
    {data : Data shape}
    {target : Target shape publicRingColumns publicFits verifierRows}
    {witness : Witness shape}
    (holds : Realization profile source data target witness)
    (childAssignments : Fin productionGlobalParams.k -> SourceAssignment shape)
    (childrenValid : forall child,
      CE.Holds (semantics profile) productionGlobalParams (target child)
        (childAssignments child)) :
    combinedAssignment profile data witness =
        (decAlgebra profile).recomposeAssignment childAssignments \/
      Nonempty (PiDEC.ParentOpeningBindingCollision (semantics profile)
        productionGlobalParams
        (parentOf profile source data witness).commitment) := by
  exact PiDEC.accepted_parent_eq_recompose_or_bindingCollision
    (semantics profile) productionGlobalParams (decAlgebra profile) {
      parent := parentOf profile source data witness
      children := target
    } (combinedAssignment profile data witness) childAssignments
      holds.outputRecompositionAccepted holds.parentOpening childrenValid

end Realization

namespace Transition

/-- Every paper transition contains source data and coins for which the target
passes the exact operational `Pi_DEC` verifier. -/
theorem outputAccepted
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {profile : Profile shape publicRingColumns publicFits verifierRows}
    {source : Source shape publicRingColumns publicFits verifierRows}
    {target : Target shape publicRingColumns publicFits verifierRows}
    (transition : Transition profile source target) :
    exists data : Data shape,
      exists witness : Witness shape,
        PiDEC.PaperVerifier.OutputAccepted (decAlgebra profile)
          (decPublicInputSplit profile) (decEvaluationArity profile)
          (parentOf profile source data witness) target := by
  rcases transition with ⟨data, witness, realized⟩
  exact ⟨data, witness, realized.piDecAccepted⟩

end Transition

end PaperProfile

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive
