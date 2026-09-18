import Nightstream.Implementation.R1CS.Correspondence.Sumcheck.SumcheckRoundSound
import Nightstream.SuperNeo.SumCheck

/-!
Compiler correspondence for a complete FE or NC claimed SumCheck chain.

Each generated map is an exact affine copy of `enforce_sumcheck_round`.
`Linked` states the structural wire reuse performed by Rust: one round's
Horner output is literally the next round's input claim.  Exact mapped-row
satisfaction therefore yields `SumCheck.Accepted` without an
accepted-implies-valid assumption.
-/

namespace Nightstream.Implementation.R1CS.SumcheckChainSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.SumcheckRoundArtifact
open Nightstream.Implementation.R1CS.SumcheckRoundSound

abbrev ColumnMap := List Nat
abbrev K := ProjectionProgram.K

def ops : Nightstream.SuperNeo.SumCheck.Ops K K where
  zero := ProjectionProgram.K.zero
  one := ProjectionProgram.K.one
  add := ProjectionProgram.K.add

def mappedAssignment (columnMap : ColumnMap) (assignment : Nat → Nat) :
    Nat → Nat :=
  Relabel.assignment columnMap assignment

def claimed (columnMap : ColumnMap) (assignment : Nat → Nat) (point : K) : K :=
  polynomial (mappedAssignment columnMap assignment) point

def round (columnMap : ColumnMap) (assignment : Nat → Nat) :
    Nightstream.SuperNeo.SumCheck.Round K K where
  claimed := claimed columnMap assignment
  expected := claimed columnMap assignment
  challenge := challengeValue (mappedAssignment columnMap assignment)
  degree := degree

def initial : List ColumnMap → (Nat → Nat) → K
  | [], _ => ProjectionProgram.K.zero
  | columnMap :: _, assignment =>
      claimInValue (mappedAssignment columnMap assignment)

def terminal : List ColumnMap → (Nat → Nat) → K
  | [], _ => ProjectionProgram.K.zero
  | [columnMap], assignment =>
      claimOutValue (mappedAssignment columnMap assignment)
  | _ :: tail, assignment => terminal tail assignment

def transcript (maps : List ColumnMap) (assignment : Nat → Nat) :
    Nightstream.SuperNeo.SumCheck.Instance K K where
  claimedInitial := initial maps assignment
  trueInitial := initial maps assignment
  terminal := terminal maps assignment
  rounds := maps.map fun columnMap => round columnMap assignment
  maxDegree := degree
  challengeSetSize := goldilocksP * goldilocksP

def Rows (columnMap : ColumnMap) : List Row :=
  SumcheckRoundArtifact.rows.map (Relabel.row columnMap)

def Holds (maps : List ColumnMap) (assignment : Nat → Nat) : Prop :=
  ∀ columnMap ∈ maps, Satisfies (Rows columnMap) assignment

def MapsOne (maps : List ColumnMap) : Prop :=
  ∀ columnMap ∈ maps, Relabel.column columnMap 0 = 0

instance (maps : List ColumnMap) : Decidable (MapsOne maps) := by
  unfold MapsOne
  infer_instance

def Link (left right : ColumnMap) : Prop :=
  Relabel.column left claimOutColumns.1 =
      Relabel.column right claimInColumns.1 ∧
    Relabel.column left claimOutColumns.2 =
      Relabel.column right claimInColumns.2

instance (left right : ColumnMap) : Decidable (Link left right) := by
  unfold Link
  infer_instance

def Linked : List ColumnMap → Prop
  | [] => True
  | [_] => True
  | left :: right :: tail => Link left right ∧ Linked (right :: tail)

private def linkedDecidable : (maps : List ColumnMap) → Decidable (Linked maps)
  | [] => isTrue trivial
  | [_] => isTrue trivial
  | left :: right :: tail => by
      letI := linkedDecidable (right :: tail)
      unfold Linked
      infer_instance

instance (maps : List ColumnMap) : Decidable (Linked maps) :=
  linkedDecidable maps

private theorem link_value
    {left right : ColumnMap}
    {assignment : Nat → Nat}
    (link : Link left right) :
    claimOutValue (mappedAssignment left assignment) =
      claimInValue (mappedAssignment right assignment) := by
  rcases link with ⟨low, high⟩
  have low' : Relabel.column left 41 = Relabel.column right 1 := by
    simpa [claimOutColumns, claimInColumns] using low
  have high' : Relabel.column left 42 = Relabel.column right 7 := by
    simpa [claimOutColumns, claimInColumns] using high
  simp only [claimOutValue, claimInValue, columns, claimOutColumns,
    claimInColumns, ProjectionProgram.KColumns.value,
    ProjectionProgram.baseAt, mappedAssignment, Relabel.assignment]
  rw [low', high']

private theorem map_sound
    {columnMap : ColumnMap}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (mapsOne : Relabel.column columnMap 0 = 0)
    (holds : Satisfies (Rows columnMap) assignment) :
    SumcheckRoundSound.Accepted (mappedAssignment columnMap assignment) := by
  exact mapped_sound columnMap mapsOne canonical one holds

private theorem chain_sound
    (maps : List ColumnMap)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (mapsOne : MapsOne maps)
    (linked : Linked maps)
    (holds : Holds maps assignment) :
    Nightstream.SuperNeo.SumCheck.Chain ops
      (fun current => current.claimed)
      (initial maps assignment)
      (maps.map fun columnMap => round columnMap assignment)
      (terminal maps assignment) := by
  induction maps with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      have headOne : Relabel.column head 0 = 0 := mapsOne head (by simp)
      have headHolds : Satisfies (Rows head) assignment :=
        holds head (by simp)
      have headSound := map_sound canonical one headOne headHolds
      change
        initial (head :: tail) assignment =
            ops.add ((round head assignment).claimed ops.zero)
              ((round head assignment).claimed ops.one) ∧
          Nightstream.SuperNeo.SumCheck.Chain ops
            (fun current => current.claimed)
            ((round head assignment).claimed (round head assignment).challenge)
            (tail.map fun columnMap => round columnMap assignment)
            (terminal (head :: tail) assignment)
      constructor
      · simpa [initial, round, claimed, ops] using headSound.initial
      · cases tail with
        | nil =>
          simpa [round, claimed, terminal] using headSound.terminal.symm
        | cons next rest =>
          rcases linked with ⟨headLink, tailLinked⟩
          have tailOne : MapsOne (next :: rest) := by
            intro columnMap member
            exact mapsOne columnMap (by simp [member])
          have tailHolds : Holds (next :: rest) assignment := by
            intro columnMap member
            exact holds columnMap (by simp [member])
          have tailChain := inductionHypothesis tailOne tailLinked tailHolds
          have forwarded :
              claimed head assignment
                  (challengeValue (mappedAssignment head assignment)) =
                initial (next :: rest) assignment := by
            calc
              claimed head assignment
                  (challengeValue (mappedAssignment head assignment)) =
                  claimOutValue (mappedAssignment head assignment) :=
                headSound.terminal.symm
              _ = claimInValue (mappedAssignment next assignment) :=
                link_value headLink
              _ = initial (next :: rest) assignment := rfl
          have forwarded' :
              (round head assignment).claimed (round head assignment).challenge =
                initial (next :: rest) assignment := by
            simpa [round, claimed] using forwarded
          rw [forwarded']
          exact tailChain

/-- Exact generated round rows and structural claim forwarding imply the
generic model's executable claimed-chain acceptance predicate. -/
theorem accepted
    (maps : List ColumnMap)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (mapsOne : MapsOne maps)
    (linked : Linked maps)
    (holds : Holds maps assignment) :
    Nightstream.SuperNeo.SumCheck.Accepted ops (transcript maps assignment) := by
  constructor
  · intro current member
    rcases List.mem_map.mp member with ⟨columnMap, _, rfl⟩
    simp [round, transcript]
  · exact chain_sound maps canonical one mapsOne linked holds

end Nightstream.Implementation.R1CS.SumcheckChainSound
