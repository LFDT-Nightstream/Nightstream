import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRunningAuthorityRows

/-!
Contract: exact positional receipts and column conservation for the selected
running-authority row slice.

Receipts own row positions, not row values: two independent obligations may
legitimately emit equal `Row` values.  Conservation says every mentioned column
is the constant wire or belongs to an authoritative child/parent coordinate
selected directly from the decoded running carrier.  The slice allocates no
columns.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRunningAuthorityAudit

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRunningAuthorityRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
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

private abbrev FamilyFor (application : Poseidon23ApplicationProfile Selected) :=
  application.family Selected

def pointRows
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
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) : List Row :=
  KConsistency.consistencyRows
    (ConcreteNifsRunningAuthorityRows.pointPairs application profile frame)

def radixRows
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
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) : List Row :=
  Phi81RadixRows.rows
    (ConcreteNifsRunningAuthorityRows.fCoordinates application profile frame)
    (ConcreteNifsRunningAuthorityRows.evaluationCoordinates
      application profile frame)

theorem rows_eq_parts
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
    ConcreteNifsRunningAuthorityRows.rows application profile frame =
      pointRows application profile frame ++
        radixRows application profile frame := rfl

/-! ## Positional receipts -/

inductive RowOwner where
  | point (index : Nat)
  | radix (index : Nat)
deriving DecidableEq, Repr

private def blank : Row := ⟨[], [], []⟩

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
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) : RowOwner → Row
  | .point index => (pointRows application profile frame).getD index blank
  | .radix index => (radixRows application profile frame).getD index blank

def owners
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
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) : List RowOwner :=
  (List.range (pointRows application profile frame).length).map RowOwner.point
    ++ (List.range (radixRows application profile frame).length).map
      RowOwner.radix

private theorem map_getD_range {α : Type} (list : List α) (fallback : α) :
    (List.range list.length).map (fun index => list.getD index fallback) =
      list := by
  induction list with
  | nil => rfl
  | cons head tail hypothesis =>
      rw [List.length_cons, List.range_succ_eq_map, List.map_cons,
        List.map_map]
      exact congrArg (head :: ·) hypothesis

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
    ConcreteNifsRunningAuthorityRows.rows application profile frame =
      (owners application profile frame).map
        (ownedRow application profile frame) := by
  rw [rows_eq_parts, owners, List.map_append]
  simp only [List.map_map, Function.comp_def, ownedRow]
  rw [map_getD_range, map_getD_range]

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

theorem owners_nodup
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
    (owners application profile frame).Nodup := by
  unfold owners
  refine List.nodup_append.2 ⟨?_, ?_, ?_⟩
  · exact nodup_map_of_injective _ (fun _ _ equal => by cases equal; rfl)
      List.nodup_range
  · exact nodup_map_of_injective _ (fun _ _ equal => by cases equal; rfl)
      List.nodup_range
  · intro left leftMember right rightMember equal
    rcases List.mem_map.1 leftMember with ⟨_, _, rfl⟩
    rcases List.mem_map.1 rightMember with ⟨_, _, rfl⟩
    cases equal

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
    (ConcreteNifsRunningAuthorityRows.rows application profile frame).length =
        (owners application profile frame).length
      ∧ (owners application profile frame).Nodup
      ∧ ConcreteNifsRunningAuthorityRows.rows application profile frame =
        (owners application profile frame).map
          (ownedRow application profile frame) := by
  refine ⟨?_, owners_nodup application profile frame,
    rows_eq_map_owners application profile frame⟩
  rw [rows_eq_map_owners, List.length_map]

/-! ## Column conservation -/

/-- Every column mentioned by the slice is an authoritative running-carrier
operand or the constant-one wire. -/
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
      row ∈ ConcreteNifsRunningAuthorityRows.rows application profile frame)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    column = 0
      ∨ (∃ pair ∈
          ConcreteNifsRunningAuthorityRows.pointPairs application profile frame,
          Mentions pair.1.low column ∨ Mentions pair.1.high column
            ∨ Mentions pair.2.low column ∨ Mentions pair.2.high column)
      ∨ (∃ coordinate ∈
          ConcreteNifsRunningAuthorityRows.fCoordinates
            application profile frame,
          (∃ child, Mentions (coordinate.children child) column)
            ∨ Mentions coordinate.parent column)
      ∨ (∃ coordinate ∈
          ConcreteNifsRunningAuthorityRows.evaluationCoordinates
            application profile frame,
          (∃ child,
            Mentions (coordinate.children child).low column
              ∨ Mentions (coordinate.children child).high column)
            ∨ Mentions coordinate.parent.low column
            ∨ Mentions coordinate.parent.high column) := by
  rw [rows_eq_parts] at member
  rcases List.mem_append.1 member with inPoint | inRadix
  · obtain ⟨pair, pairMember, support⟩ :=
      KConsistency.consistencyRows_conservation
        (ConcreteNifsRunningAuthorityRows.pointPairs application profile frame)
        row inPoint column mentioned
    rcases support with isOne | inChildLow | inChildHigh
        | inParentLow | inParentHigh
    · exact Or.inl isOne
    · exact Or.inr (Or.inl
        ⟨pair, pairMember, Or.inl inChildLow⟩)
    · exact Or.inr (Or.inl
        ⟨pair, pairMember, Or.inr (Or.inl inChildHigh)⟩)
    · exact Or.inr (Or.inl
        ⟨pair, pairMember, Or.inr (Or.inr (Or.inl inParentLow))⟩)
    · exact Or.inr (Or.inl
        ⟨pair, pairMember, Or.inr (Or.inr (Or.inr inParentHigh))⟩)
  · have support :=
      Phi81RadixRows.rows_conservation
        (ConcreteNifsRunningAuthorityRows.fCoordinates application profile frame)
        (ConcreteNifsRunningAuthorityRows.evaluationCoordinates
          application profile frame)
        row inRadix column mentioned
    rcases support with isOne | inF | inK
    · exact Or.inl isOne
    · exact Or.inr (Or.inr (Or.inl inF))
    · exact Or.inr (Or.inr (Or.inr inK))

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRunningAuthorityAudit
