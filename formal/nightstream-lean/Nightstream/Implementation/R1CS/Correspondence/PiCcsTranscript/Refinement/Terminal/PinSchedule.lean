import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Schedule
import Nightstream.Implementation.R1CS.Core.ConstantPins

/-!
Exact verifier-owned constant tree for the terminal `Pi_CCS` binding and
pre-SumCheck challenge schedule.

Assurance tier: implementation/R1CS correspondence. Every ordinary constant
piece is grouped by protocol phase, checked against the exact generated owner,
and decoded from accepted equations under canonical-field assumptions.

Owns: fixed header-bundle fields for this terminal profile; raw-message length
and domain words; running-count value; and every raw-squeeze word for the
seven engine and five `beta_m` Poseidon2 calls.

Does not own: authority of the fixed header bundle, instance-digest wires,
checked-parent wires, initial transcript state, Poseidon2 call semantics,
inter-call connectivity, challenge partitioning, Rust conformance, costs, or
row removal.

Emits constraints: no.

Authority boundary: a constant is not trusted because it appears in generated
rows. Its value follows from an independently accepted R1CS equation, a
canonical assignment, and the verifier's constant-one column. The header
bundle still requires a separate derivation from public parameters.

| Protocol | Phase | Constraint family | Exact verifier-owned values |
|---|---|---|---|
| `Pi_CCS` | header | fixed bundle and raw boundary | four bundle fields, length `5`, tag `11` |
| `Pi_CCS` | instance | raw boundary | length `5`, tag `12` |
| `Pi_CCS` | running authority | three raw boundaries | `[1,4]`, `[2,5,14]`, `[5,13]` |
| `Pi_CCS` | main challenges | domain and squeeze words | first `[1,2,1]`, then six `1`s |
| `Pi_CCS` | `beta_m` | domain and squeeze words | first `[1,3,1]`, then four `1`s |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.PinSchedule

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate

set_option maxRecDepth 1000000

namespace Artifact

def headerPins : List (Nat × Nat) :=
  [(1623283, 17168707872888128320),
   (1623284, 11050799198242575901),
   (1623285, 16730522141919911230),
   (1623286, 5655123306428251295),
   (1623287, 11), (1623288, 5)]

def instancePins : List (Nat × Nat) :=
  [(1624489, 12), (1624490, 5)]

def runningDomainPins : List (Nat × Nat) :=
  [(1625091, 4), (1625092, 1)]

def runningParentPins : List (Nat × Nat) :=
  [(1625693, 5), (1625694, 14), (1625695, 2),
   (1625696, 13), (1625697, 5)]

def mainFirstPins : List (Nat × Nat) :=
  [(1626898, 1), (1626899, 2), (1626900, 1)]

def mainLaterCount : Nat := 6

def mainLaterColumn (index : Fin mainLaterCount) : Nat :=
  1627501 + 601 * index.val

def mainLaterPins (index : Fin mainLaterCount) : List (Nat × Nat) :=
  [(mainLaterColumn index, 1)]

def betaMFirstPins : List (Nat × Nat) :=
  [(1631107, 1), (1631108, 3), (1631109, 1)]

def betaMLaterCount : Nat := 4

def betaMLaterColumn (index : Fin betaMLaterCount) : Nat :=
  1631710 + 601 * index.val

def betaMLaterPins (index : Fin betaMLaterCount) : List (Nat × Nat) :=
  [(betaMLaterColumn index, 1)]

def headerPiece : Piece := Rows.pieceAt ⟨14, by decide⟩
def instancePiece : Piece := Rows.pieceAt ⟨17, by decide⟩
def runningDomainPiece : Piece := Rows.pieceAt ⟨19, by decide⟩
def runningParentPiece : Piece := Rows.pieceAt ⟨21, by decide⟩
def mainFirstPiece : Piece := Rows.pieceAt ⟨24, by decide⟩

def mainLaterPieceIndex (index : Fin mainLaterCount) : Fin Rows.pieceCount :=
  ⟨26 + 2 * index.val, by
    have indexLt := index.isLt
    simp only [mainLaterCount, Rows.pieceCount] at indexLt ⊢
    omega⟩

def mainLaterPiece (index : Fin mainLaterCount) : Piece :=
  Rows.pieceAt (mainLaterPieceIndex index)

def betaMFirstPiece : Piece := Rows.pieceAt ⟨38, by decide⟩

def betaMLaterPieceIndex (index : Fin betaMLaterCount) : Fin Rows.pieceCount :=
  ⟨40 + 2 * index.val, by
    have indexLt := index.isLt
    simp only [betaMLaterCount, Rows.pieceCount] at indexLt ⊢
    omega⟩

def betaMLaterPiece (index : Fin betaMLaterCount) : Piece :=
  Rows.pieceAt (betaMLaterPieceIndex index)

def expectedHeaderPiece : Piece :=
  { rowStart := 1572953, rowEnd := 1572959
    payload := .ordinary (ConstantPins.rows headerPins) }

def expectedInstancePiece : Piece :=
  { rowStart := 1574159, rowEnd := 1574161
    payload := .ordinary (ConstantPins.rows instancePins) }

def expectedRunningDomainPiece : Piece :=
  { rowStart := 1574761, rowEnd := 1574763
    payload := .ordinary (ConstantPins.rows runningDomainPins) }

def expectedRunningParentPiece : Piece :=
  { rowStart := 1575363, rowEnd := 1575368
    payload := .ordinary (ConstantPins.rows runningParentPins) }

def expectedMainFirstPiece : Piece :=
  { rowStart := 1576568, rowEnd := 1576571
    payload := .ordinary (ConstantPins.rows mainFirstPins) }

def expectedMainLaterPiece (index : Fin mainLaterCount) : Piece :=
  { rowStart := 1577171 + 601 * index.val
    rowEnd := 1577172 + 601 * index.val
    payload := .ordinary (ConstantPins.rows (mainLaterPins index)) }

def expectedBetaMFirstPiece : Piece :=
  { rowStart := 1580777, rowEnd := 1580780
    payload := .ordinary (ConstantPins.rows betaMFirstPins) }

def expectedBetaMLaterPiece (index : Fin betaMLaterCount) : Piece :=
  { rowStart := 1581380 + 601 * index.val
    rowEnd := 1581381 + 601 * index.val
    payload := .ordinary (ConstantPins.rows (betaMLaterPins index)) }

/-- Closed protocol/phase pin tree over every constant-only owner piece in
the terminal binding and challenge prefix. -/
theorem pinTree_eq :
    headerPiece = expectedHeaderPiece /\
    instancePiece = expectedInstancePiece /\
    runningDomainPiece = expectedRunningDomainPiece /\
    runningParentPiece = expectedRunningParentPiece /\
    mainFirstPiece = expectedMainFirstPiece /\
    (forall index : Fin mainLaterCount,
      mainLaterPiece index = expectedMainLaterPiece index) /\
    betaMFirstPiece = expectedBetaMFirstPiece /\
    (forall index : Fin betaMLaterCount,
      betaMLaterPiece index = expectedBetaMLaterPiece index) := by
  decide

theorem headerPiece_eq : headerPiece = expectedHeaderPiece := pinTree_eq.1
theorem instancePiece_eq : instancePiece = expectedInstancePiece := pinTree_eq.2.1
theorem runningDomainPiece_eq :
    runningDomainPiece = expectedRunningDomainPiece := pinTree_eq.2.2.1
theorem runningParentPiece_eq :
    runningParentPiece = expectedRunningParentPiece := pinTree_eq.2.2.2.1
theorem mainFirstPiece_eq :
    mainFirstPiece = expectedMainFirstPiece := pinTree_eq.2.2.2.2.1
theorem mainLaterPiece_eq (index : Fin mainLaterCount) :
    mainLaterPiece index = expectedMainLaterPiece index :=
  pinTree_eq.2.2.2.2.2.1 index
theorem betaMFirstPiece_eq :
    betaMFirstPiece = expectedBetaMFirstPiece :=
  pinTree_eq.2.2.2.2.2.2.1
theorem betaMLaterPiece_eq (index : Fin betaMLaterCount) :
    betaMLaterPiece index = expectedBetaMLaterPiece index :=
  pinTree_eq.2.2.2.2.2.2.2 index

theorem headerPiece_mem : headerPiece ∈
    FPrimeFullHistoryTerminalPiCcsTranscript.owner.pieces := Rows.pieceAt_mem _
theorem instancePiece_mem : instancePiece ∈
    FPrimeFullHistoryTerminalPiCcsTranscript.owner.pieces := Rows.pieceAt_mem _
theorem runningDomainPiece_mem : runningDomainPiece ∈
    FPrimeFullHistoryTerminalPiCcsTranscript.owner.pieces := Rows.pieceAt_mem _
theorem runningParentPiece_mem : runningParentPiece ∈
    FPrimeFullHistoryTerminalPiCcsTranscript.owner.pieces := Rows.pieceAt_mem _
theorem mainFirstPiece_mem : mainFirstPiece ∈
    FPrimeFullHistoryTerminalPiCcsTranscript.owner.pieces := Rows.pieceAt_mem _
theorem mainLaterPiece_mem (index : Fin mainLaterCount) :
    mainLaterPiece index ∈
      FPrimeFullHistoryTerminalPiCcsTranscript.owner.pieces := Rows.pieceAt_mem _
theorem betaMFirstPiece_mem : betaMFirstPiece ∈
    FPrimeFullHistoryTerminalPiCcsTranscript.owner.pieces := Rows.pieceAt_mem _
theorem betaMLaterPiece_mem (index : Fin betaMLaterCount) :
    betaMLaterPiece index ∈
      FPrimeFullHistoryTerminalPiCcsTranscript.owner.pieces := Rows.pieceAt_mem _

end Artifact

private theorem rowsIncluded_self (rows : List Row) :
    rowsIncluded rows rows = true := by
  simp [rowsIncluded]

private theorem acceptedPins
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : FPrimeFullHistoryTerminalPiCcsTranscript.Accepted assignment)
    (pins : List (Nat × Nat))
    (valuesCanonical : ConstantPins.ValuesCanonical pins)
    (piece : Piece)
    (piecePayload : piece.payload = .ordinary (ConstantPins.rows pins))
    (pieceMember : piece ∈
      FPrimeFullHistoryTerminalPiCcsTranscript.owner.pieces) :
    forall pin, pin ∈ pins -> assignment pin.1 = pin.2 := by
  have pieceAccepted := accepted piece pieceMember
  rw [Piece.Accepted, piecePayload, Payload.Accepted] at pieceAccepted
  exact ConstantPins.sound valuesCanonical
    (rowsIncluded_self (ConstantPins.rows pins)) canonical one pieceAccepted

theorem headerPinsCanonical : ConstantPins.ValuesCanonical Artifact.headerPins := by decide
theorem instancePinsCanonical : ConstantPins.ValuesCanonical Artifact.instancePins := by decide
theorem runningDomainPinsCanonical :
    ConstantPins.ValuesCanonical Artifact.runningDomainPins := by decide
theorem runningParentPinsCanonical :
    ConstantPins.ValuesCanonical Artifact.runningParentPins := by decide
theorem mainFirstPinsCanonical :
    ConstantPins.ValuesCanonical Artifact.mainFirstPins := by decide
theorem mainLaterPinsCanonical : forall index : Fin Artifact.mainLaterCount,
    ConstantPins.ValuesCanonical (Artifact.mainLaterPins index) := by decide
theorem betaMFirstPinsCanonical :
    ConstantPins.ValuesCanonical Artifact.betaMFirstPins := by decide
theorem betaMLaterPinsCanonical : forall index : Fin Artifact.betaMLaterCount,
    ConstantPins.ValuesCanonical (Artifact.betaMLaterPins index) := by decide

/-- Exact accepted constant facts, grouped by protocol phase rather than by
their accidental row adjacency. -/
structure Facts (assignment : Nat -> Nat) : Prop where
  header : forall pin, pin ∈ Artifact.headerPins -> assignment pin.1 = pin.2
  instanceBoundary : forall pin, pin ∈ Artifact.instancePins ->
    assignment pin.1 = pin.2
  runningDomain : forall pin, pin ∈ Artifact.runningDomainPins -> assignment pin.1 = pin.2
  runningParent : forall pin, pin ∈ Artifact.runningParentPins -> assignment pin.1 = pin.2
  mainFirst : forall pin, pin ∈ Artifact.mainFirstPins -> assignment pin.1 = pin.2
  mainLater : forall index pin, pin ∈ Artifact.mainLaterPins index -> assignment pin.1 = pin.2
  betaMFirst : forall pin, pin ∈ Artifact.betaMFirstPins -> assignment pin.1 = pin.2
  betaMLater : forall index pin, pin ∈ Artifact.betaMLaterPins index -> assignment pin.1 = pin.2

theorem facts
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : FPrimeFullHistoryTerminalPiCcsTranscript.Accepted assignment) :
    Facts assignment := by
  refine {
    header := ?_
    instanceBoundary := ?_
    runningDomain := ?_
    runningParent := ?_
    mainFirst := ?_
    mainLater := ?_
    betaMFirst := ?_
    betaMLater := ?_
  }
  · exact acceptedPins canonical one accepted Artifact.headerPins
      headerPinsCanonical Artifact.headerPiece
      (by rw [Artifact.headerPiece_eq]; rfl) Artifact.headerPiece_mem
  · exact acceptedPins canonical one accepted Artifact.instancePins
      instancePinsCanonical Artifact.instancePiece
      (by rw [Artifact.instancePiece_eq]; rfl) Artifact.instancePiece_mem
  · exact acceptedPins canonical one accepted Artifact.runningDomainPins
      runningDomainPinsCanonical Artifact.runningDomainPiece
      (by rw [Artifact.runningDomainPiece_eq]; rfl)
      Artifact.runningDomainPiece_mem
  · exact acceptedPins canonical one accepted Artifact.runningParentPins
      runningParentPinsCanonical Artifact.runningParentPiece
      (by rw [Artifact.runningParentPiece_eq]; rfl)
      Artifact.runningParentPiece_mem
  · exact acceptedPins canonical one accepted Artifact.mainFirstPins
      mainFirstPinsCanonical Artifact.mainFirstPiece
      (by rw [Artifact.mainFirstPiece_eq]; rfl) Artifact.mainFirstPiece_mem
  · intro index
    exact acceptedPins canonical one accepted (Artifact.mainLaterPins index)
      (mainLaterPinsCanonical index) (Artifact.mainLaterPiece index)
      (by rw [Artifact.mainLaterPiece_eq]; rfl)
      (Artifact.mainLaterPiece_mem index)
  · exact acceptedPins canonical one accepted Artifact.betaMFirstPins
      betaMFirstPinsCanonical Artifact.betaMFirstPiece
      (by rw [Artifact.betaMFirstPiece_eq]; rfl) Artifact.betaMFirstPiece_mem
  · intro index
    exact acceptedPins canonical one accepted (Artifact.betaMLaterPins index)
      (betaMLaterPinsCanonical index) (Artifact.betaMLaterPiece index)
      (by rw [Artifact.betaMLaterPiece_eq]; rfl)
      (Artifact.betaMLaterPiece_mem index)

variable {assignment : Nat -> Nat}

theorem Facts.headerLength (self : Facts assignment) : assignment 1623288 = 5 :=
  self.header (1623288, 5) (by simp [Artifact.headerPins])
theorem Facts.headerTag (self : Facts assignment) : assignment 1623287 = 11 :=
  self.header (1623287, 11) (by simp [Artifact.headerPins])
theorem Facts.instanceLength (self : Facts assignment) : assignment 1624490 = 5 :=
  self.instanceBoundary (1624490, 5) (by simp [Artifact.instancePins])
theorem Facts.instanceTag (self : Facts assignment) : assignment 1624489 = 12 :=
  self.instanceBoundary (1624489, 12) (by simp [Artifact.instancePins])
theorem Facts.runningDomainLength (self : Facts assignment) : assignment 1625092 = 1 :=
  self.runningDomain (1625092, 1) (by simp [Artifact.runningDomainPins])
theorem Facts.runningDomainTag (self : Facts assignment) : assignment 1625091 = 4 :=
  self.runningDomain (1625091, 4) (by simp [Artifact.runningDomainPins])
theorem Facts.runningCountLength (self : Facts assignment) : assignment 1625695 = 2 :=
  self.runningParent (1625695, 2) (by simp [Artifact.runningParentPins])
theorem Facts.runningCountTag (self : Facts assignment) : assignment 1625693 = 5 :=
  self.runningParent (1625693, 5) (by simp [Artifact.runningParentPins])
theorem Facts.runningCount (self : Facts assignment) : assignment 1625694 = 14 :=
  self.runningParent (1625694, 14) (by simp [Artifact.runningParentPins])
theorem Facts.parentLength (self : Facts assignment) : assignment 1625697 = 5 :=
  self.runningParent (1625697, 5) (by simp [Artifact.runningParentPins])
theorem Facts.parentTag (self : Facts assignment) : assignment 1625696 = 13 :=
  self.runningParent (1625696, 13) (by simp [Artifact.runningParentPins])
theorem Facts.mainDomainLength (self : Facts assignment) : assignment 1626898 = 1 :=
  self.mainFirst (1626898, 1) (by simp [Artifact.mainFirstPins])
theorem Facts.mainDomainTag (self : Facts assignment) : assignment 1626899 = 2 :=
  self.mainFirst (1626899, 2) (by simp [Artifact.mainFirstPins])
theorem Facts.mainFirstSqueeze (self : Facts assignment) : assignment 1626900 = 1 :=
  self.mainFirst (1626900, 1) (by simp [Artifact.mainFirstPins])
theorem Facts.mainLaterSqueeze (self : Facts assignment)
    (index : Fin Artifact.mainLaterCount) :
    assignment (Artifact.mainLaterColumn index) = 1 :=
  self.mainLater index (Artifact.mainLaterColumn index, 1)
    (by simp [Artifact.mainLaterPins])
theorem Facts.betaMDomainLength (self : Facts assignment) : assignment 1631107 = 1 :=
  self.betaMFirst (1631107, 1) (by simp [Artifact.betaMFirstPins])
theorem Facts.betaMDomainTag (self : Facts assignment) : assignment 1631108 = 3 :=
  self.betaMFirst (1631108, 3) (by simp [Artifact.betaMFirstPins])
theorem Facts.betaMFirstSqueeze (self : Facts assignment) : assignment 1631109 = 1 :=
  self.betaMFirst (1631109, 1) (by simp [Artifact.betaMFirstPins])
theorem Facts.betaMLaterSqueeze (self : Facts assignment)
    (index : Fin Artifact.betaMLaterCount) :
    assignment (Artifact.betaMLaterColumn index) = 1 :=
  self.betaMLater index (Artifact.betaMLaterColumn index, 1)
    (by simp [Artifact.betaMLaterPins])

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.PinSchedule
