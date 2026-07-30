import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointRows

/-!
Contract: exact positional receipts and column conservation for the selected
PiRLC parent-point binding.

A receipt records both the FE row coordinate and which of the two physical
Goldilocks equality rows it emits.  Receipts own positions rather than row
values: distinct coordinates are allowed to emit structurally equal rows.

The slice allocates nothing.  Every mentioned column is therefore either the
constant-one wire, an operational transcript challenge coordinate, or the
corresponding coordinate of the decoded running output's parent point.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointAudit

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
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

/-! ## Positional receipts -/

/-- One receipt per emitted row: the FE coordinate and extension half. -/
structure RowOwner (variables : Nat) where
  coordinate : Fin variables
  half : KEquality.RowOwner
deriving DecidableEq

def ownedRow
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      Nightstream.Implementation.Lowering.Goldilocks.CallFrame
        (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (owner : RowOwner shape.rowVariables) : Row :=
  KEquality.ownedRow
    (ConcreteNifsPiRlcPointRows.transcriptCoordinate
      application profile frame owner.coordinate)
    (ConcreteNifsPiRlcPointRows.outputCoordinate
      application profile frame owner.coordinate)
    owner.half

def owners (variables : Nat) : List (RowOwner variables) :=
  (List.ofFn fun coordinate : Fin variables => coordinate).flatMap
    (fun coordinate =>
      KEquality.allOwners.map (RowOwner.mk coordinate))

private theorem flatMap_eq_of_pointwise
    {α β : Type} (list : List α) (left right : α → List β)
    (same : ∀ value, left value = right value) :
    list.flatMap left = list.flatMap right := by
  induction list with
  | nil => rfl
  | cons head tail hypothesis =>
      rw [List.flatMap_cons, List.flatMap_cons, same head, hypothesis]

theorem rows_eq_map_owners
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      Nightstream.Implementation.Lowering.Goldilocks.CallFrame
        (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    ConcreteNifsPiRlcPointRows.rows application profile frame =
      (owners shape.rowVariables).map
        (ownedRow application profile frame) := by
  unfold ConcreteNifsPiRlcPointRows.rows owners
  rw [List.map_flatMap]
  simp only [List.map_map, Function.comp_def, ownedRow,
    ConcreteNifsPiRlcPointRows.coordinateRows]
  apply flatMap_eq_of_pointwise
  intro coordinate
  exact KEquality.rows_eq_map_owners _ _

private theorem nodup_ofFn_of_injective {α : Type} :
    ∀ {size : Nat} (function : Fin size → α),
      Function.Injective function → (List.ofFn function).Nodup
  | 0, _, _ => by simp
  | _ + 1, function, injective => by
      rw [List.ofFn_succ, List.nodup_cons]
      constructor
      · intro member
        rcases List.mem_ofFn.mp member with ⟨index, equal⟩
        exact Fin.succ_ne_zero index (injective equal)
      · exact nodup_ofFn_of_injective
          (fun index => function index.succ)
          (fun left right equal => Fin.succ_inj.mp (injective equal))

private theorem ownerBlocks_nodup
    {variables : Nat} :
    ∀ coordinates : List (Fin variables),
      coordinates.Nodup →
        (coordinates.flatMap fun coordinate =>
          KEquality.allOwners.map (RowOwner.mk coordinate)).Nodup
  | [], _ => by simp
  | coordinate :: rest, nodup => by
      rw [List.nodup_cons] at nodup
      rw [List.flatMap_cons, List.nodup_append]
      refine ⟨
        LinCombNormal.nodup_map KEquality.allOwners
          (RowOwner.mk coordinate)
          (fun left right equal => by
            cases equal
            rfl)
          KEquality.allOwners_nodup,
        ownerBlocks_nodup rest nodup.2,
        ?_⟩
      intro left leftMember right rightMember equal
      rcases List.mem_map.1 leftMember with
        ⟨leftHalf, _, rfl⟩
      rcases List.mem_flatMap.1 rightMember with
        ⟨rightCoordinate, rightCoordinateMember, rightMember⟩
      rcases List.mem_map.1 rightMember with
        ⟨rightHalf, _, rfl⟩
      have coordinatesEqual : coordinate = rightCoordinate := by
        cases equal
        rfl
      exact nodup.1 (coordinatesEqual ▸ rightCoordinateMember)

theorem owners_nodup (variables : Nat) :
    (owners variables).Nodup := by
  apply ownerBlocks_nodup
  exact nodup_ofFn_of_injective id (fun _ _ equal => equal)

/-- Exactly one semantically tagged receipt per emitted row position. -/
theorem ownership_is_positional
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      Nightstream.Implementation.Lowering.Goldilocks.CallFrame
        (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    (ConcreteNifsPiRlcPointRows.rows application profile frame).length =
        (owners shape.rowVariables).length
      ∧ (owners shape.rowVariables).Nodup
      ∧ ConcreteNifsPiRlcPointRows.rows application profile frame =
        (owners shape.rowVariables).map
          (ownedRow application profile frame) := by
  refine ⟨?_, owners_nodup shape.rowVariables,
    rows_eq_map_owners application profile frame⟩
  rw [rows_eq_map_owners, List.length_map]

/-! ## Column conservation -/

/-- Every column mentioned by the point slice is the constant wire or belongs
to the exact transcript/output coordinate named by a receipt. -/
theorem rows_conservation
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      Nightstream.Implementation.Lowering.Goldilocks.CallFrame
        (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (row : Row)
    (member :
      row ∈ ConcreteNifsPiRlcPointRows.rows application profile frame)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    column = 0
      ∨ (∃ coordinate : Fin shape.rowVariables,
          Mentions
              (ConcreteNifsPiRlcPointRows.transcriptCoordinate
                application profile frame coordinate).low column
            ∨ Mentions
              (ConcreteNifsPiRlcPointRows.transcriptCoordinate
                application profile frame coordinate).high column)
      ∨ (∃ coordinate : Fin shape.rowVariables,
          Mentions
              (ConcreteNifsPiRlcPointRows.outputCoordinate
                application profile frame coordinate).low column
            ∨ Mentions
              (ConcreteNifsPiRlcPointRows.outputCoordinate
                application profile frame coordinate).high column) := by
  rcases List.mem_flatMap.1 member with
    ⟨coordinate, _, coordinateMember⟩
  rcases KEquality.rows_conservation
      (ConcreteNifsPiRlcPointRows.transcriptCoordinate
        application profile frame coordinate)
      (ConcreteNifsPiRlcPointRows.outputCoordinate
        application profile frame coordinate)
      row coordinateMember column mentioned with
    wire | transcriptLow | transcriptHigh | outputLow | outputHigh
  · exact Or.inl wire
  · exact Or.inr (Or.inl ⟨coordinate, Or.inl transcriptLow⟩)
  · exact Or.inr (Or.inl ⟨coordinate, Or.inr transcriptHigh⟩)
  · exact Or.inr (Or.inr ⟨coordinate, Or.inl outputLow⟩)
  · exact Or.inr (Or.inr ⟨coordinate, Or.inr outputHigh⟩)

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointAudit
