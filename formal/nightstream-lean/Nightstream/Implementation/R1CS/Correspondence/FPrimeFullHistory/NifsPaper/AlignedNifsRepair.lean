import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCeRelation
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiCcs
import Nightstream.SuperNeo.Concrete.Parameters
import Nightstream.SuperNeo.Folding.Nifs

/-!
Fail-closed semantic repair contract for the fixed F′ SuperNeo carrier.

Owns: the paper dimension predicate after inserting thirteen authoritative
public zeros; an honest aligned fresh-CCS constructor; preservation of the
legacy CCS relation and norm predicate at that constructor; and rejection of
the exact linked 270-coefficient counterexample by aligned CCS and CE
membership.

Does not own: a Rust column insertion, sparse-matrix migration, an aligned
Ajtai setup artifact, Π_CCS/NIFS verifier refinement, Fiat--Shamir, generated
R1CS rows, constraint counts, or permission to remove constraints.

Emits constraints: no.

Authority boundary: the new commitment is recomputed from the complete
aligned assignment under a verifier-owned key. The thirteen coefficients are
part of the paper-visible public input; no legacy digest or 257-field
projection is accepted as authority for them.

| Protocol | Phase | Constraint family | Mathematical obligation | Result |
|---|---|---|---|---|
| SuperNeo relation | dimensions | field/ring carrier | both public and total widths are exact multiples of 54 | `PaperDimensions` |
| Π_CCS input | fresh opening | public padding | construct `[x, 0^13, w]` and recompute its commitment | `alignedFreshStatement` |
| Π_CCS input | fresh semantics | CCS and norm | honest legacy membership transports to the aligned relation | `alignedFreshStatement_holds` |
| NIFS | honest transition | phase composition | aligned fresh inputs and already-aligned running inputs realize a candidate paper transition | `alignedNifsTransition_complete` |
| fixed NIFS/F′ | exact bad carrier | CCS opening | the linked accepted carrier cannot satisfy fresh aligned CCS | `fixedCarrierTail_not_alignedCCS` |
| fixed NIFS/F′ | exact bad carrier | CE opening | the linked accepted carrier cannot satisfy fresh aligned CE | `fixedCarrierTail_not_alignedCE` |

The final two theorems close only the concrete carrier counterexample. They do
not establish that production now implements this repair.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedPublicInput
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCcsRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCeRelation

/-- Paper Definition 1 eligibility for the repaired public and total field
dimensions. This stays an explicit premise: adding thirteen public columns
does not make every arbitrary legacy total width ring-aligned. -/
def PaperDimensions (system : Structure) : Prop :=
  alignedPublicWidth = ringDegree * 5 ∧
    alignedPublicWidth ≤ (alignStructure system).columns ∧
    ∃ totalRingColumns,
      (alignStructure system).columns = ringDegree * totalRingColumns

/-- The exact 257-column fixture becomes a 270-column, five-ring-column paper
relation after the public insertion. -/
theorem fixedCarrier_paperDimensions (system : Structure)
    (columns : system.columns = logicalPublicWidth) :
    PaperDimensions system := by
  constructor
  · exact aligned_dimensions.2.2.2
  constructor
  · simp [alignStructure, columns, logicalPublicWidth, alignedPublicWidth,
      paddingWidth]
  · refine ⟨5, ?_⟩
    simp [alignStructure, columns, logicalPublicWidth, paddingWidth, ringDegree]

/-- Canonical fresh statement for the repaired relation. Its commitment and
270-field public input are derived from the complete aligned assignment. -/
def alignedFreshStatement (ajtaiKey : AjtaiKey) (system : Structure)
    (assignment : Assignment) : CCSStatement :=
  canonicalCCSStatement (alignedContext ajtaiKey) (alignStructure system)
    .fresh (insertPublicPadding assignment)

/-- Honest fresh completeness of the repair. This proof uses the independent
relation and norm transport theorems; no old verifier acceptance is a premise. -/
theorem alignedFreshStatement_holds (params : GlobalParams)
    (ajtaiKey : AjtaiKey) (system : Structure) (assignment : Assignment)
    (positiveFreshBound : 0 < params.b)
    (hasPublic : logicalPublicWidth ≤ system.columns)
    (assignmentLength : assignment.length = system.columns)
    (wellFormed : system.WellFormed)
    (bounded : normBounded params.b assignment)
    (satisfied : ccsSatisfied system assignment) :
    CCS.Holds (relationSemantics (alignedContext ajtaiKey)) params
      (alignedFreshStatement ajtaiKey system assignment)
      (insertPublicPadding assignment) := by
  apply canonicalCCS_holds
  · apply (normBounded_insertPublicPadding_iff params.b
      positiveFreshBound assignment).2
    exact bounded
  · exact (ccsSatisfied_align_iff system assignment hasPublic
      assignmentLength wellFormed).2 satisfied

/-! ## Candidate NIFS closure after alignment -/

/-- One repaired input product. Fresh CCS statements are reconstructed from
legacy assignments; running CE statements must already belong to the repaired
relation and may therefore carry nonzero values in all 270 public
coefficients after ring-linear folding. -/
def alignedInput
    {params : GlobalParams} {arity : BatchArity params}
    (ajtaiKey : AjtaiKey) (system : Structure)
    (freshAssignments : Fin arity.freshCount → Assignment)
    (running : Fin (arity.mode.count params) → CEStatement) :
    PiCCS.InputProduct Structure PublicInput Point Evaluation Commitment
      params arity where
  fresh index := alignedFreshStatement ajtaiKey system (freshAssignments index)
  running := running

/-- Assignment order shared by Π_CCS, Π_RLC, and Π_DEC: repaired fresh
assignments first, followed by already-aligned running assignments. -/
def alignedSourceAssignments
    {params : GlobalParams} {arity : BatchArity params}
    (freshAssignments : Fin arity.freshCount → Assignment)
    (runningAssignments : Fin (arity.mode.count params) → Assignment) :
    Fin arity.total → Assignment :=
  Fin.addCases
    (fun index => insertPublicPadding (freshAssignments index))
    runningAssignments

/-- Perfect completeness of the repaired carrier inside the independent
three-phase candidate NIFS semantics. The theorem derives fresh-source
membership from the legacy relation; running membership remains an explicit
inductive invariant.

This does not close the candidate model's documented joint-Q, finite
certificate, extractor, or Fiat--Shamir boundaries, and it says nothing about
the current Rust verifier. -/
theorem alignedNifsTransition_complete
    {Scalar Challenge Value : Type}
    (params : GlobalParams) (arity : BatchArity params)
    (ajtaiKey : AjtaiKey) (system : Structure)
    (freshAssignments : Fin arity.freshCount → Assignment)
    (running : Fin (arity.mode.count params) → CEStatement)
    (runningAssignments : Fin (arity.mode.count params) → Assignment)
    (point : Point)
    (sumcheckOps : SumCheck.Ops Challenge Value)
    (rlcAlgebra : PiRLC.Algebra Structure Assignment PublicInput Point
      Evaluation Commitment Scalar
      (relationSemantics (alignedContext ajtaiKey)) params)
    (decAlgebra : PiDEC.Algebra Structure Assignment PublicInput Point
      Evaluation Commitment
      (relationSemantics (alignedContext ajtaiKey)) params)
    (fe nc : SumCheck.Instance Challenge Value)
    (challenges : Fin arity.total → Scalar)
    (dimensions : PaperDimensions system)
    (positiveFreshBound : 0 < params.b)
    (hasPublic : logicalPublicWidth ≤ system.columns)
    (wellFormed : system.WellFormed)
    (freshLength : ∀ index,
      (freshAssignments index).length = system.columns)
    (freshBounded : ∀ index,
      normBounded params.b (freshAssignments index))
    (freshSatisfied : ∀ index,
      ccsSatisfied system (freshAssignments index))
    (runningFresh : ∀ index, (running index).stage = .fresh)
    (runningStructure : ∀ index,
      (running index).constraintSystem = alignStructure system)
    (runningValid : ∀ index,
      CE.Holds (relationSemantics (alignedContext ajtaiKey)) params
        (running index) (runningAssignments index))
    (pointValid : evaluationPointValid (alignStructure system) point)
    (feTruth : SumCheck.TruthPath sumcheckOps fe)
    (ncTruth : SumCheck.TruthPath sumcheckOps nc)
    (feHonest : SumCheck.Honest fe)
    (ncHonest : SumCheck.Honest nc)
    (challengesValid : ∀ index,
      rlcAlgebra.challengeValid (challenges index)) :
    ∃ output,
      PaperDimensions system ∧
      Nifs.PaperNifsTransition sumcheckOps rlcAlgebra decAlgebra
        (alignedInput ajtaiKey system freshAssignments running) output := by
  let input := alignedInput ajtaiKey system freshAssignments running
  let assignments := alignedSourceAssignments freshAssignments runningAssignments
  have sourceFresh : ∀ index, (input.source index).stage = .fresh := by
    intro index
    refine Fin.addCases ?_ ?_ index
    · intro freshIndex
      simp [input, alignedInput, PiCCS.InputProduct.source,
        PiCCS.Source.stage, alignedFreshStatement, canonicalCCSStatement]
    · intro runningIndex
      simpa [input, alignedInput, PiCCS.InputProduct.source,
        PiCCS.Source.stage] using runningFresh runningIndex
  have sourceValid : ∀ index,
      (input.source index).Holds
        (relationSemantics (alignedContext ajtaiKey)) params
        (assignments index) := by
    intro index
    refine Fin.addCases ?_ ?_ index
    · intro freshIndex
      simpa [input, assignments, alignedInput, alignedSourceAssignments,
        PiCCS.InputProduct.source, PiCCS.Source.Holds] using
        alignedFreshStatement_holds params ajtaiKey system
          (freshAssignments freshIndex) positiveFreshBound hasPublic
          (freshLength freshIndex) wellFormed (freshBounded freshIndex)
          (freshSatisfied freshIndex)
    · intro runningIndex
      simpa [input, assignments, alignedInput, alignedSourceAssignments,
        PiCCS.InputProduct.source, PiCCS.Source.Holds] using
        runningValid runningIndex
  have sameStructure : ∀ index,
      (input.source index).constraintSystem = alignStructure system := by
    intro index
    refine Fin.addCases ?_ ?_ index
    · intro freshIndex
      simp [input, alignedInput, PiCCS.InputProduct.source,
        PiCCS.Source.constraintSystem, alignedFreshStatement,
        canonicalCCSStatement]
    · intro runningIndex
      simpa [input, alignedInput, PiCCS.InputProduct.source,
        PiCCS.Source.constraintSystem] using runningStructure runningIndex
  have transition := Nifs.paperNifsTransition_complete
    (relationSemantics (alignedContext ajtaiKey)) params sumcheckOps rlcAlgebra
    decAlgebra arity (alignStructure system) point input assignments fe nc
    challenges sourceFresh sourceValid sameStructure pointValid feTruth ncTruth
    feHonest ncHonest challengesValid
  exact ⟨_, dimensions, transition⟩

/-- The exact linked implementation carrier, interpreted as all 270 paper
coefficients rather than the truncated 257-field view. -/
def fixedCarrierTail : Assignment :=
  Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.baseFields
    Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.linkedTailFullDecode

theorem fixedCarrierTail_not_normBounded :
    ¬ normBounded productionGlobalParams.b fixedCarrierTail := by
  simpa [fixedCarrierTail, productionGlobalParams] using
    PiCcs.fixedCarrierArtifact_linkedTail_not_normBounded

/-- Any fresh aligned CCS statement rejects the exact carrier accepted by the
current recursive F′ execution. Commitment or row self-consistency cannot
override the failed full-carrier norm obligation. -/
theorem fixedCarrierTail_not_alignedCCS (ajtaiKey : AjtaiKey)
    (statement : CCSStatement) (fresh : statement.stage = .fresh) :
    ¬ CCS.Holds (relationSemantics (alignedContext ajtaiKey))
      productionGlobalParams statement fixedCarrierTail := by
  intro accepted
  apply fixedCarrierTail_not_normBounded
  simpa [fresh, productionGlobalParams] using accepted.1.2.2

/-- The same exact carrier also cannot satisfy any fresh aligned CE statement. -/
theorem fixedCarrierTail_not_alignedCE (ajtaiKey : AjtaiKey)
    (statement : CEStatement) (fresh : statement.stage = .fresh) :
    ¬ CE.Holds (relationSemantics (alignedContext ajtaiKey))
      productionGlobalParams statement fixedCarrierTail := by
  intro accepted
  apply fixedCarrierTail_not_normBounded
  simpa [fresh, productionGlobalParams] using accepted.1.2.2

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair
