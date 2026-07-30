import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputRows

/-!
Contract: positional receipts and column conservation for selected NIFS
output materialization.

The receipt list owns row positions, not row values. Conservation classifies
every mentioned column as the constant wire or as one of the authoritative
proof/output coordinates compared by the same child-materialization program.
The program allocates no columns.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputAudit

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev TranscriptState := Poseidon2Duplex.State

section SelectedFrame

variable {shape : SemanticShape}
variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}
variable {keys : Fin 1 →
  SelectedKey shape TranscriptState publicRingColumns publicFits verifierRows}
variable {defaultRunning :
  SelectedRunning shape publicRingColumns publicFits verifierRows}
variable {machine :
  Machine
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    Digest AppState Witness
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    Encoded 1}
variable {terminalRelations :
  TerminalRelations
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    RunningWitness
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    FreshWitness 1}
variable {terminalChecks :
  Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
    terminalRelations}
variable {widths : Widths} {footprints : Footprints}

local notation "Selected" =>
  ConcreteNifsParameters.selected keys defaultRunning machine
    terminalRelations terminalChecks widths footprints

private abbrev FamilyFor
    (application : Poseidon23ApplicationProfile Selected) :=
  application.family Selected

private abbrev FrameFor
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)} :=
  Nightstream.Implementation.Lowering.Goldilocks.CallFrame
    (signature := signature Selected)
    (FamilyFor application) Call.nifsVerify
    (Refs.cons runningRef
      (Refs.cons freshRef (Refs.cons proofRef .nil)))

/-! ## Positional receipts -/

inductive RowOwner where
  | childMaterialization (index : Nat)
deriving DecidableEq, Repr

private def blank : Row := ⟨[], [], []⟩

def ownedRow
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    RowOwner → Row
  | .childMaterialization index =>
      (ConcreteNifsOutputRows.rows application profile frame).getD index blank

def owners
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    List RowOwner :=
  (List.range
    (ConcreteNifsOutputRows.rows application profile frame).length).map
      RowOwner.childMaterialization

private theorem map_getD_range {α : Type} (list : List α) (fallback : α) :
    (List.range list.length).map (fun index => list.getD index fallback) =
      list := by
  induction list with
  | nil => rfl
  | cons head tail hypothesis =>
      rw [List.length_cons, List.range_succ_eq_map, List.map_cons,
        List.map_map]
      exact congrArg (head :: ·) hypothesis

private theorem nodup_map_of_injective {α β : Type} (f : α → β)
    (injective : ∀ first second, f first = f second → first = second) :
    ∀ {list : List α}, list.Nodup → (list.map f).Nodup
  | [], _ => by simp
  | head :: tail, nodup => by
      rw [List.nodup_cons] at nodup
      rw [List.map_cons, List.nodup_cons]
      refine ⟨?_, nodup_map_of_injective f injective nodup.2⟩
      intro member
      rcases List.mem_map.1 member with ⟨other, otherMember, equal⟩
      exact nodup.1 (injective _ _ equal ▸ otherMember)

theorem rows_eq_map_owners
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ConcreteNifsOutputRows.rows application profile frame =
      (owners application profile frame).map
        (ownedRow application profile frame) := by
  unfold owners
  simp only [List.map_map, Function.comp_def, ownedRow]
  exact (map_getD_range _ _).symm

theorem owners_nodup
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (owners application profile frame).Nodup := by
  unfold owners
  exact nodup_map_of_injective _
    (fun _ _ equal => by cases equal; rfl) List.nodup_range

/-- Exactly one structured receipt for every output row position. -/
theorem ownership_is_positional
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (ConcreteNifsOutputRows.rows application profile frame).length =
        (owners application profile frame).length
      ∧ (owners application profile frame).Nodup
      ∧ ConcreteNifsOutputRows.rows application profile frame =
        (owners application profile frame).map
          (ownedRow application profile frame) := by
  refine ⟨?_, owners_nodup application profile frame,
    rows_eq_map_owners application profile frame⟩
  rw [rows_eq_map_owners, List.length_map]

/-! ## Column conservation -/

/-- An authoritative proof/output coordinate compared by the child block. -/
def AuthoritativeMention
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (column : Nat) : Prop :=
  (∃ child row lane,
      Mentions
          (ConcreteNifsOutputRows.outputChildCommitment
            application profile frame child row lane) column
        ∨ Mentions
          (ConcreteNifsOutputRows.proofChildCommitment
            application profile frame child row lane) column)
  ∨ (∃ child publicCoordinate,
      Mentions
          (ConcreteNifsOutputRows.outputChildPublic
            application profile frame child publicCoordinate) column
        ∨ Mentions
          (ConcreteNifsOutputRows.proofChildPublic
            application profile frame child publicCoordinate) column)
  ∨ (∃ child coordinate,
      let output :=
        ConcreteNifsOutputRows.outputChildPoint
          application profile frame child coordinate
      let parent :=
        ConcreteNifsOutputRows.outputParentPoint
          application profile frame coordinate
      Mentions output.low column ∨ Mentions output.high column
        ∨ Mentions parent.low column ∨ Mentions parent.high column)
  ∨ (∃ child matrix lane,
      let output :=
        ConcreteNifsOutputRows.outputChildEvaluation
          application profile frame child matrix lane
      let proof :=
        ConcreteNifsOutputRows.proofChildEvaluation
          application profile frame child matrix lane
      Mentions output.low column ∨ Mentions output.high column
        ∨ Mentions proof.low column ∨ Mentions proof.high column)

private theorem equalityRow_conservation
    (left right : LinComb) (column : Nat)
    (mentioned :
      Mentions (KEquality.equalityRow left right).a column
        ∨ Mentions (KEquality.equalityRow left right).b column
        ∨ Mentions (KEquality.equalityRow left right).c column) :
    column = 0 ∨ Mentions left column ∨ Mentions right column := by
  rcases mentioned with mentioned | mentioned | mentioned
  · exact Or.inr (Or.inl mentioned)
  · exact Or.inl (by
      simpa only [KEquality.equalityRow, Mentions, List.map_cons,
        List.map_nil, List.mem_singleton] using mentioned)
  · exact Or.inr (Or.inr mentioned)

/-- Every row reads only the constant wire and the exact authoritative
coordinates named by the selected proof and output codecs. -/
theorem rows_conservation
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (row : Row)
    (member :
      row ∈ ConcreteNifsOutputRows.rows application profile frame)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column
        ∨ Mentions row.c column) :
    column = 0 ∨ AuthoritativeMention application profile frame column := by
  rcases List.mem_flatMap.1 member with
    ⟨child, _, childMember⟩
  simp only [ConcreteNifsOutputRows.childRows, List.mem_append] at childMember
  rcases childMember with
    ((commitmentMember | publicMember) | pointMember) | evaluationMember
  · rcases List.mem_flatMap.1 commitmentMember with
      ⟨sourceRow, _, laneMember⟩
    rcases List.mem_map.1 laneMember with ⟨lane, _, rfl⟩
    rcases equalityRow_conservation _ _ column mentioned with
      constant | output | proof
    · exact Or.inl constant
    · exact Or.inr (Or.inl ⟨child, sourceRow, lane, Or.inl output⟩)
    · exact Or.inr (Or.inl ⟨child, sourceRow, lane, Or.inr proof⟩)
  · rcases List.mem_map.1 publicMember with
      ⟨publicCoordinate, _, rfl⟩
    rcases equalityRow_conservation _ _ column mentioned with
      constant | output | proof
    · exact Or.inl constant
    · exact Or.inr (Or.inr (Or.inl
        ⟨child, publicCoordinate, Or.inl output⟩))
    · exact Or.inr (Or.inr (Or.inl
        ⟨child, publicCoordinate, Or.inr proof⟩))
  · rcases List.mem_flatMap.1 pointMember with
      ⟨coordinate, _, rowMember⟩
    rcases KEquality.rows_conservation _ _ row rowMember column mentioned with
      constant | outputLow | outputHigh | parentLow | parentHigh
    · exact Or.inl constant
    · exact Or.inr (Or.inr (Or.inr (Or.inl
        ⟨child, coordinate, Or.inl outputLow⟩)))
    · exact Or.inr (Or.inr (Or.inr (Or.inl
        ⟨child, coordinate, Or.inr (Or.inl outputHigh)⟩)))
    · exact Or.inr (Or.inr (Or.inr (Or.inl
        ⟨child, coordinate, Or.inr (Or.inr (Or.inl parentLow))⟩)))
    · exact Or.inr (Or.inr (Or.inr (Or.inl
        ⟨child, coordinate, Or.inr (Or.inr (Or.inr parentHigh))⟩)))
  · rcases List.mem_flatMap.1 evaluationMember with
      ⟨matrix, _, laneRowsMember⟩
    rcases List.mem_flatMap.1 laneRowsMember with
      ⟨lane, _, rowMember⟩
    rcases KEquality.rows_conservation _ _ row rowMember column mentioned with
      constant | outputLow | outputHigh | proofLow | proofHigh
    · exact Or.inl constant
    · exact Or.inr (Or.inr (Or.inr (Or.inr
        ⟨child, matrix, lane, Or.inl outputLow⟩)))
    · exact Or.inr (Or.inr (Or.inr (Or.inr
        ⟨child, matrix, lane, Or.inr (Or.inl outputHigh)⟩)))
    · exact Or.inr (Or.inr (Or.inr (Or.inr
        ⟨child, matrix, lane,
          Or.inr (Or.inr (Or.inl proofLow))⟩)))
    · exact Or.inr (Or.inr (Or.inr (Or.inr
        ⟨child, matrix, lane,
          Or.inr (Or.inr (Or.inr proofHigh))⟩)))

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputAudit
