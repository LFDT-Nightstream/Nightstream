import Nightstream.Implementation.Nebula.Production.FPrime.Fresh.RelationCompilerFor
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.PiCcsRoundSelectiveCcs
import Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckSupport

/-!
Contract: exact generated selective-CCS bridge for one production PiCCS round.

Assurance tier: artifact-checked Rust emitter correspondence.

Owns fail-closed generated recipe decoding, the canonical 54-column phase
layout, exact compilation of all 31 arithmetic rows into the thirteen-port
selective relation, and the same-assignment implication from generated CCS
satisfaction to `RoundPhaseRelation`.

The Rust artifact owner invokes the real compact phase emitter and compares
every A, B, and C row with an independent recipe. Lean independently
recomputes the same row program and proves its semantics.

Does not own Poseidon2 permutation rows, recursive orchestration, the PiCCS
start or finish phases, terminal integration, or security assumptions.

Emits constraints: 31 selective product rows in a five-variable Boolean row
domain. Padding rows are zero by the direct compiler theorem.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRoundArtifact

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Nebula.ProductionFreshRelationCompilerFor
open Nightstream.Implementation.Nebula.ProductionFreshRelationCompilerFor.NumericBridge
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsAuthority
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRoundRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckHonest
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckSupport
open Nightstream.Implementation.R1CS.Canonical.KHornerHonest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiCcsRoundSelectiveCcs.Artifact
open Nightstream.Implementation.R1CS.SelectiveCcs
open Nightstream.Implementation.R1CS.SelectiveCcs.LeanCompiler
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

namespace Generated

abbrev rawArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.PiCcsRoundSelectiveCcs.rawArtifact

theorem rawArtifact_valid : Valid rawArtifact := by
  decide

def artifact : Decoded := ⟨rawArtifact, rawArtifact_valid⟩

theorem decode_rawArtifact : decode rawArtifact = some artifact := by
  simp [decode, artifact, rawArtifact_valid]

def layout : Layout := artifact.layout

@[simp] theorem layout_currentStart : layout.currentStart = 1 := by
  rfl

@[simp] theorem layout_coefficientStart : layout.coefficientStart = 3 := by
  rfl

@[simp] theorem layout_challengeStart : layout.challengeStart = 23 := by
  rfl

@[simp] theorem layout_nextStart : layout.nextStart = 25 := by
  rfl

@[simp] theorem layout_auxiliaryStart : layout.auxiliaryStart = 27 := by
  rfl

def sourceRows : List R1CS.Row := rows layout

@[simp] theorem sourceRows_length : sourceRows.length = 31 := by
  exact rows_length layout

def columns : Nat := rawArtifact.columns

@[simp] theorem columns_eq : columns = 54 := by
  rfl

theorem columns_positive : 0 < columns := by
  decide

private theorem carriedAt_below
    {start base : Nat} (highBelow : start + 1 < base) :
    CarriedBelow (carriedAt start) base := by
  constructor <;> intro column mentioned
  · simp only [carriedAt, Mentions, List.map_cons, List.map_nil,
      List.mem_singleton] at mentioned
    subst column
    omega
  · simp only [carriedAt, Mentions, List.map_cons, List.map_nil,
      List.mem_singleton] at mentioned
    subst column
    exact highBelow

private theorem current_below : CarriedBelow (current layout) 27 := by
  apply carriedAt_below
  simp

private theorem challenge_below : CarriedBelow (challenge layout) 27 := by
  apply carriedAt_below
  simp

private theorem next_below : CarriedBelow (next layout) 27 := by
  apply carriedAt_below
  simp

private theorem round_below : RoundBelow (round layout) 27 := by
  intro value member
  unfold round coefficients at member
  rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
  apply carriedAt_below
  simp only [layout_coefficientStart]
  have indexLt := index.isLt
  omega

private theorem mentioned_of_member
    {terms : List (Nat × Nat)} {term : Nat × Nat}
    (member : term ∈ terms) : Mentions terms term.1 := by
  exact List.mem_map.mpr ⟨term, member, rfl⟩

theorem sourceRows_below : RowsBelow columns sourceRows := by
  intro row member
  refine ⟨?_, ?_, ?_⟩ <;> intro term termMember
  · have below := chainRows_columns_below_end
      (current layout) [round layout] [challenge layout] (next layout) 27
      (by decide) current_below
      (by simpa using round_below)
      (by simpa using challenge_below)
      next_below rfl row member term.1
      (Or.inl (mentioned_of_member termMember))
    simpa [columns, sourceRows, rows] using below
  · have below := chainRows_columns_below_end
      (current layout) [round layout] [challenge layout] (next layout) 27
      (by decide) current_below
      (by simpa using round_below)
      (by simpa using challenge_below)
      next_below rfl row member term.1
      (Or.inr (Or.inl (mentioned_of_member termMember)))
    simpa [columns, sourceRows, rows] using below
  · have below := chainRows_columns_below_end
      (current layout) [round layout] [challenge layout] (next layout) 27
      (by decide) current_below
      (by simpa using round_below)
      (by simpa using challenge_below)
      next_below rfl row member term.1
      (Or.inr (Or.inr (mentioned_of_member termMember)))
    simpa [columns, sourceRows, rows] using below

def directProgram : List (DirectRows.SourceRow columns) :=
  NumericBridge.directProgram columns_positive sourceRows

@[simp] theorem directProgram_length : directProgram.length = 31 := by
  simp [directProgram]

def one : Fin columns := ⟨0, columns_positive⟩

def relation := DirectRows.relation one directProgram

def profile : RelationProfile.Profile directProgram.length columns where
  rowVariables := rawArtifact.rowVariables
  rowDomain := by
    rw [directProgram_length]
    change RelationProfile.ExactRowDomain 31 5
    constructor
    · norm_num
    · intro smaller smallerLt
      have exponentLe : smaller ≤ 4 := by omega
      have powerLe : 2 ^ smaller ≤ 2 ^ 4 :=
        Nat.pow_le_pow_right (by decide) exponentLe
      exact powerLe.trans_lt (by norm_num)
  publicRingColumns := 0
  publicFits := by simp

def system := DirectRows.paperSystem relation profile

def numericAssignment (assignment : Fin columns → F) : Nat → Nat :=
  fun column =>
    if within : column < columns then assignment ⟨column, within⟩ |>.val else 0

theorem numericAssignment_one
    (assignment : Fin columns → F) (constantOne : assignment one = 1) :
    numericAssignment assignment 0 = 1 := by
  unfold numericAssignment
  rw [dif_pos columns_positive]
  have sameValue := congrArg Fin.val constantOne
  simpa [one] using sameValue

/-- Satisfaction of every generated selective-CCS row implies the fused
production round relation over the exact same finite assignment. -/
theorem generated_selective_ccs_implies_roundPhaseRelation
    (assignment : Fin columns → F)
    (constantOne : assignment one = 1)
    (satisfied : ConstraintSatisfied baseOps system assignment)
    {rowVariables : Nat}
    (roundIndex : Fin (ProductNifsCodec.shapeFor rowVariables).cubeVariables)
    (polynomial : ProductionRound)
    (before after : Continuation ConcreteK BindingState)
    (placed : ControlPlacement layout (numericAssignment assignment)
      roundIndex polynomial before after) :
    RoundPhaseRelation (ProductPoseidon2.transcriptFor rowVariables)
      ConcreteCarrier.extensionOps.toOps roundIndex polynomial before after := by
  have directSatisfied :
      ∀ index : Fin directProgram.length,
        (directProgram.get index).Holds assignment :=
    (DirectRows.constraintSatisfied_iff one directProgram profile assignment
      constantOne).mp satisfied
  have localSatisfied : Satisfies sourceRows (numericAssignment assignment) :=
    (NumericBridge.directProgram_satisfied_iff columns_positive sourceRows
      sourceRows_below assignment).mp directSatisfied
  exact rows_imply_roundPhaseRelation layout (numericAssignment assignment)
    (numericAssignment_one assignment constantOne) localSatisfied
    roundIndex polynomial before after placed

end Generated

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRoundArtifact
