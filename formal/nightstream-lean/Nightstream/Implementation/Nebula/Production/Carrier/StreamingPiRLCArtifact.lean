import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCAuthority
import Nightstream.Implementation.Nebula.Production.FPrime.Fresh.RelationCompilerFor
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.PiRlcFamilySelectiveCcs

/-!
Contract: exact generated selective-CCS bridge for one production PiRLC phase.

Assurance tier: artifact-checked compiler correspondence.

Owns the fail-closed generated recipe decode, the canonical 45,415-column
phase layout, exact compilation of all 43,794 handwritten rows into the
thirteen-port selective relation, and the same-assignment implication from
generated CCS satisfaction to `FamilyPhaseRelation`.

Does not own the Rust witness encoder, Poseidon2 replay rows, PiCCS input
authority, recursive orchestration, terminal integration, or security
assumptions.

Emits constraints: 43,794 selective product rows in a 16-variable Boolean row
domain. Padding rows are zero by the direct compiler theorem.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationRows
open Nightstream.Implementation.Nebula.ProductionFreshRelationCompilerFor
open Nightstream.Implementation.Nebula.ProductionFreshRelationCompilerFor.NumericBridge
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiRlcFamilySelectiveCcs.Artifact
open Nightstream.Implementation.R1CS.SelectiveCcs
open Nightstream.Implementation.R1CS.SelectiveCcs.LeanCompiler
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

namespace Generated

abbrev rawArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.PiRlcFamilySelectiveCcs.rawArtifact

theorem rawArtifact_valid : Valid rawArtifact := by
  decide

def artifact : Decoded := ⟨rawArtifact, rawArtifact_valid⟩

theorem decode_rawArtifact : decode rawArtifact = some artifact := by
  simp [decode, artifact, rawArtifact_valid]

def layout : Layout := artifact.layout

@[simp] theorem layout_base : layout.base = 1675 := by
  rfl

@[simp] theorem layout_challengeSymbol
    (source : ProductPiRlcRingCombinationRows.Source) (lane : Lane) :
    layout.challengeSymbol source lane = 1 + source.val * 54 + lane.val := by
  rfl

@[simp] theorem layout_input
    (source : ProductPiRlcRingCombinationRows.Source) (lane : Lane) :
    layout.input source lane = 811 + source.val * 54 + lane.val := by
  rfl

@[simp] theorem layout_output (lane : Lane) :
    layout.output lane = 1621 + lane.val := by
  rfl

def sourceRows : List R1CS.Row := rows layout

@[simp] theorem sourceRows_length : sourceRows.length = 43794 := by
  exact rows_length layout

def columns : Nat := rawArtifact.columns

@[simp] theorem columns_eq : columns = 45415 := by
  rfl

theorem columns_positive : 0 < columns := by
  decide

private theorem productColumn_below
    (source : ProductPiRlcRingCombinationRows.Source)
    (left right : Lane) :
    productColumn layout source left right < columns := by
  have offsetLt := productOffset_lt source left right
  simp only [productColumn, layout_base, columns_eq]
  change 1675 + productOffset source left right < 45415
  have productExact : productCount = 43740 := productCount_eq
  omega

private theorem centeredChallenge_below
    (source : ProductPiRlcRingCombinationRows.Source) (lane : Lane) :
    TermsBelow columns (centeredChallenge layout source lane) := by
  intro term member
  simp only [centeredChallenge, List.mem_cons, List.not_mem_nil, or_false]
    at member
  rcases member with equal | equal
  · subst term
    simp only [layout_challengeSymbol, columns_eq]
    have sourceLt := source.isLt
    have laneLt := lane.isLt
    change source.val < 15 at sourceLt
    change lane.val < 54 at laneLt
    omega
  · subst term
    simp [columns_eq]

private theorem input_below
    (source : ProductPiRlcRingCombinationRows.Source) (lane : Lane) :
    layout.input source lane < columns := by
  simp only [layout_input, columns_eq]
  have sourceLt := source.isLt
  have laneLt := lane.isLt
  change source.val < 15 at sourceLt
  change lane.val < 54 at laneLt
  omega

private theorem output_below (lane : Lane) :
    layout.output lane < columns := by
  simp only [layout_output, columns_eq]
  have laneLt := lane.isLt
  change lane.val < 54 at laneLt
  omega

private theorem rawTerms_below
    (source : ProductPiRlcRingCombinationRows.Source)
    (degree coefficient : Nat) :
    TermsBelow columns (rawTerms layout source degree coefficient) := by
  intro term member
  unfold rawTerms at member
  rcases List.mem_filterMap.mp member with ⟨left, _, selected⟩
  split at selected
  · simp only [Option.some.injEq] at selected
    subst term
    exact productColumn_below source left _
  · simp at selected

private theorem termsBelow_append
    {left right : List (Nat × Nat)}
    (leftBelow : TermsBelow columns left)
    (rightBelow : TermsBelow columns right) :
    TermsBelow columns (left ++ right) := by
  intro term member
  rcases List.mem_append.mp member with leftMember | rightMember
  · exact leftBelow term leftMember
  · exact rightBelow term rightMember

private theorem sourceOutputTerms_below
    (source : ProductPiRlcRingCombinationRows.Source) (output : Lane) :
    TermsBelow columns (sourceOutputTerms layout source output) := by
  unfold sourceOutputTerms
  apply termsBelow_append
  · exact termsBelow_append (rawTerms_below source output.val 1)
      (rawTerms_below source (foldedDegree output) (goldilocksP - 1))
  split
  · exact rawTerms_below source (output.val + 81) 1
  · intro term member
    simp at member

private theorem outputTerms_below (output : Lane) :
    TermsBelow columns (outputTerms layout output) := by
  intro term member
  unfold outputTerms at member
  rcases List.mem_flatMap.mp member with ⟨source, _, sourceMember⟩
  exact sourceOutputTerms_below source output term sourceMember

private theorem productRow_below
    (source : ProductPiRlcRingCombinationRows.Source)
    (left right : Lane) :
    RowBelow columns (productRow layout source left right) := by
  refine ⟨centeredChallenge_below source left, ?_, ?_⟩
  · intro term member
    simp only [productRow, List.mem_cons, List.not_mem_nil, or_false] at member
    subst term
    exact input_below source right
  · intro term member
    simp only [productRow, List.mem_cons, List.not_mem_nil, or_false] at member
    subst term
    exact productColumn_below source left right

private theorem outputRow_below (output : Lane) :
    RowBelow columns (outputRow layout output) := by
  refine ⟨?_, outputTerms_below output, ?_⟩
  · intro term member
    simp only [outputRow, List.mem_cons, List.not_mem_nil, or_false] at member
    subst term
    simp [columns_eq]
  · intro term member
    simp only [outputRow, List.mem_cons, List.not_mem_nil, or_false] at member
    subst term
    exact output_below output

theorem sourceRows_below : RowsBelow columns sourceRows := by
  intro row member
  unfold sourceRows rows at member
  rcases List.mem_append.mp member with productMember | outputMember
  · unfold productRows at productMember
    rcases List.mem_flatMap.mp productMember with
      ⟨source, _, sourceMember⟩
    unfold sourceProductRows at sourceMember
    rcases List.mem_flatMap.mp sourceMember with ⟨left, _, leftMember⟩
    rcases List.mem_map.mp leftMember with ⟨right, _, rfl⟩
    exact productRow_below source left right
  · unfold outputRows at outputMember
    rcases List.mem_map.mp outputMember with ⟨output, _, rfl⟩
    exact outputRow_below output

def directProgram : List (DirectRows.SourceRow columns) :=
  NumericBridge.directProgram columns_positive sourceRows

@[simp] theorem directProgram_length : directProgram.length = 43794 := by
  simp [directProgram]

def one : Fin columns := ⟨0, columns_positive⟩

def relation := DirectRows.relation one directProgram

def profile : RelationProfile.Profile directProgram.length columns where
  rowVariables := rawArtifact.rowVariables
  rowDomain := by
    rw [directProgram_length]
    change RelationProfile.ExactRowDomain 43794 16
    constructor
    · norm_num
    · intro smaller smallerLt
      have exponentLe : smaller ≤ 15 := by omega
      have powerLe : 2 ^ smaller ≤ 2 ^ 15 :=
        Nat.pow_le_pow_right (by decide) exponentLe
      exact powerLe.trans_lt (by norm_num)
  publicRingColumns := 0
  publicFits := by simp

def system := DirectRows.paperSystem relation profile

def numericAssignment (assignment : Fin columns → F) : Nat → Nat :=
  fun column =>
    if within : column < columns then assignment ⟨column, within⟩ |>.val else 0

theorem numericAssignment_canonical (assignment : Fin columns → F) :
    ∀ column, numericAssignment assignment column < goldilocksP := by
  intro column
  unfold numericAssignment
  split
  · simpa [goldilocksP, goldilocksModulus] using
      (assignment ⟨column, ‹column < columns›⟩).isLt
  · simp [goldilocksP]

theorem numericAssignment_one
    (assignment : Fin columns → F) (constantOne : assignment one = 1) :
    numericAssignment assignment 0 = 1 := by
  unfold numericAssignment
  rw [dif_pos columns_positive]
  have sameValue := congrArg Fin.val constantOne
  simpa [one] using sameValue

/-- Satisfaction of every generated selective-CCS row implies the same
concrete phase relation as the 43,794 handwritten rows. The finite field
assignment is read once; its exact numeric view is used by both sides. -/
theorem generated_selective_ccs_implies_concrete_phase
    (assignment : Fin columns → F)
    (constantOne : assignment one = 1)
    (satisfied : ConstraintSatisfied baseOps system assignment)
    (range : ∀ source lane,
      numericAssignment assignment (layout.challengeSymbol source lane) < 5)
    (inputSetup : InputBindingSetup)
    (before after : FamilyState) (family : Family)
    (challengesExact :
      decodedChallenges layout (numericAssignment assignment) range =
        before.challenges)
    (cursorExact : before.familyCursor =
      ProductPiRlcAlgebraRows.familyOrdinal family)
    (transition : FamilyTransition inputSetup before after family
      (decodedInputs layout (numericAssignment assignment)
        (numericAssignment_canonical assignment))
      (ProductPiRlcRingCombinationSound.outputRing layout
        (numericAssignment assignment)
        (numericAssignment_canonical assignment))) :
    FamilyPhaseRelation inputSetup before after
      family
      (decodedInputs layout (numericAssignment assignment)
        (numericAssignment_canonical assignment))
      (ProductPiRlcRingCombinationSound.outputRing layout
        (numericAssignment assignment)
        (numericAssignment_canonical assignment)) := by
  have directSatisfied :
      ∀ index : Fin directProgram.length,
        (directProgram.get index).Holds assignment :=
    (DirectRows.constraintSatisfied_iff one directProgram profile assignment
      constantOne).mp satisfied
  have localSatisfied : Satisfies sourceRows (numericAssignment assignment) :=
    (NumericBridge.directProgram_satisfied_iff columns_positive sourceRows
      sourceRows_below assignment).mp directSatisfied
  exact local_rows_imply_concrete_phase
    (numericAssignment_canonical assignment)
    (numericAssignment_one assignment constantOne) range localSatisfied
    inputSetup before after family challengesExact cursorExact transition

end Generated

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact
