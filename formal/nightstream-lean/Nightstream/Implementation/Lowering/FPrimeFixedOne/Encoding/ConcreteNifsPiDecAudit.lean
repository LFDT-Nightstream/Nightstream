import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiDecRows

/-!
Contract: positional receipts and column conservation for the selected
outgoing `Pi_DEC` slice.

The receipt list owns positions rather than row values: equal physical rows
may encode distinct retained equations.  Conservation states that every
mentioned column is either the constant wire or belongs to one of the
authoritative proof-child/output-parent coordinates selected by the same
slice.  The slice allocates no columns.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiDecAudit

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiDecRows
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

private abbrev FamilyFor
    (application : Poseidon23ApplicationProfile Selected) :=
  application.family Selected

/-! ## Positional receipts -/

/-- One receipt for each outgoing radix row position.  The coordinate family
is recovered from the row program's fixed commitment/public/evaluation order;
the index distinguishes obligations even when their row values coincide. -/
inductive RowOwner where
  | outgoingRadix (index : Nat)
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
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    RowOwner → Row
  | .outgoingRadix index =>
      (ConcreteNifsPiDecRows.rows application profile frame).getD index blank

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
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    List RowOwner :=
  (List.range
    (ConcreteNifsPiDecRows.rows application profile frame).length).map
      RowOwner.outgoingRadix

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
    ConcreteNifsPiDecRows.rows application profile frame =
      (owners application profile frame).map
        (ownedRow application profile frame) := by
  unfold owners
  simp only [List.map_map, Function.comp_def, ownedRow]
  exact (map_getD_range _ _).symm

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
  exact nodup_map_of_injective _
    (fun _ _ equal => by cases equal; rfl) List.nodup_range

/-- Exactly one outgoing-`Pi_DEC` receipt for every emitted row position. -/
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
    (ConcreteNifsPiDecRows.rows application profile frame).length =
        (owners application profile frame).length
      ∧ (owners application profile frame).Nodup
      ∧ ConcreteNifsPiDecRows.rows application profile frame =
        (owners application profile frame).map
          (ownedRow application profile frame) := by
  refine ⟨?_, owners_nodup application profile frame,
    rows_eq_map_owners application profile frame⟩
  rw [rows_eq_map_owners, List.length_map]

/-! ## Column conservation -/

/-- Every mentioned column belongs to one authoritative proof-child or
output-parent coordinate, apart from the constant-one wire. -/
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
      row ∈ ConcreteNifsPiDecRows.rows application profile frame)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    column = 0
      ∨ (∃ coordinate ∈
          ConcreteNifsPiDecRows.fCoordinates application profile frame,
          (∃ child, Mentions (coordinate.children child) column)
            ∨ Mentions coordinate.parent column)
      ∨ (∃ coordinate ∈
          ConcreteNifsPiDecRows.evaluationCoordinates
            application profile frame,
          (∃ child,
            Mentions (coordinate.children child).low column
              ∨ Mentions (coordinate.children child).high column)
            ∨ Mentions coordinate.parent.low column
            ∨ Mentions coordinate.parent.high column) :=
  Phi81RadixRows.rows_conservation
    (ConcreteNifsPiDecRows.fCoordinates application profile frame)
    (ConcreteNifsPiDecRows.evaluationCoordinates application profile frame)
    row member column mentioned

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiDecAudit
