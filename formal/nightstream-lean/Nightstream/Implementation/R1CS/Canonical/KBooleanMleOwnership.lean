import Nightstream.Implementation.R1CS.Canonical.KBooleanMleHonest

/-!
Contract: positional row ownership and whole-program column conservation for
the canonical Boolean-MLE program.

Receipts identify an interpolation node and one of its three Karatsuba rows.
They do not identify a row merely by its list position: the receipt is decoded
through the node's actual left/right operands and frame.  Equal row values at
different positions therefore remain distinct structural emissions.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KBooleanMleOwnership

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KMulChain
open Nightstream.Implementation.R1CS.Canonical.KBooleanMleSupport
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- One postorder interpolation node before it is flattened into three rows. -/
structure Block where
  left : Carried
  right : Carried
  frame : Frame

def blockRows (block : Block) : List Row :=
  KMul.rows block.left block.right block.frame

/-- The interpolation blocks in exactly the emitter's postorder. -/
def blocks (frames : Nat → Frame) :
    {variables : Nat} →
      BooleanTable Carried variables → List Carried → Nat → List Block
  | 0, .leaf _, _, _ => []
  | tailVariables + 1, .branch low high, coordinates, step =>
      let tail := KBooleanMle.tailCoordinates coordinates
      let lowValue := KBooleanMle.carried frames low tail step
      let highValue :=
        KBooleanMle.carried frames high tail
          (step + KBooleanMle.frameCount tailVariables)
      blocks frames low tail step ++
        blocks frames high tail
          (step + KBooleanMle.frameCount tailVariables) ++
        [⟨KBooleanMle.headCoordinate coordinates,
          KLinear.subCarried highValue lowValue,
          KBooleanMle.rootFrame frames step tailVariables⟩]

theorem blocks_length
    (frames : Nat → Frame) :
    ∀ {variables : Nat}
      (table : BooleanTable Carried variables)
      (coordinates : List Carried) (step : Nat),
      (blocks frames table coordinates step).length =
        KBooleanMle.frameCount variables
  | 0, .leaf _, _, _ => rfl
  | tailVariables + 1, .branch low high, coordinates, step => by
      simp only [blocks, List.length_append, List.length_cons, List.length_nil,
        Nat.add_zero,
        blocks_length frames low (KBooleanMle.tailCoordinates coordinates) step,
        blocks_length frames high (KBooleanMle.tailCoordinates coordinates)
          (step + KBooleanMle.frameCount tailVariables),
        KBooleanMle.frameCount]
      omega

/-- The row emitter is exactly the flattened postorder block list. -/
theorem rows_eq_flatMap_blocks
    (frames : Nat → Frame) :
    ∀ {variables : Nat}
      (table : BooleanTable Carried variables)
      (coordinates : List Carried) (step : Nat),
      KBooleanMle.rows frames table coordinates step =
        (blocks frames table coordinates step).flatMap blockRows
  | 0, .leaf _, _, _ => rfl
  | tailVariables + 1, .branch low high, coordinates, step => by
      simp only [KBooleanMle.rows, blocks, List.flatMap_append,
        List.flatMap_cons, List.flatMap_nil, List.append_nil, blockRows,
        rows_eq_flatMap_blocks frames low
          (KBooleanMle.tailCoordinates coordinates) step,
        rows_eq_flatMap_blocks frames high
          (KBooleanMle.tailCoordinates coordinates)
          (step + KBooleanMle.frameCount tailVariables)]

/-- Which interpolation node, and which Karatsuba row within it. -/
structure RowOwner where
  node : Nat
  slot : KMulOwnership.RowOwner
deriving DecidableEq

def owners (count : Nat) : List RowOwner :=
  (List.range count).flatMap
    (fun node => KMulOwnership.allOwners.map (RowOwner.mk node))

theorem owners_length (count : Nat) :
    (owners count).length = 3 * count := by
  unfold owners
  have blocksLength :
      ∀ nodes : List Nat,
        (nodes.flatMap
          (fun node => KMulOwnership.allOwners.map (RowOwner.mk node))).length =
          3 * nodes.length := by
    intro nodes
    induction nodes with
    | nil => rfl
    | cons node rest inductionHypothesis =>
        simp only [List.flatMap_cons, List.length_append, List.length_map,
          KMulOwnership.allOwners_length, List.length_cons, inductionHypothesis]
        omega
  simpa using blocksLength (List.range count)

theorem owners_nodup (count : Nat) : (owners count).Nodup := by
  unfold owners
  induction count with
  | zero => simp
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.flatMap_append]
      refine List.nodup_append.2 ⟨inductionHypothesis, ?_, ?_⟩
      · simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
        exact LinCombNormal.nodup_map KMulOwnership.allOwners
          (RowOwner.mk count)
          (fun left right equal => by
            simp only [RowOwner.mk.injEq] at equal
            exact equal.2)
          KMulOwnership.allOwners_nodup
      · intro left leftMember right rightMember equal
        have leftLt : left.node < count := by
          rcases List.mem_flatMap.1 leftMember with
            ⟨node, nodeMember, ownerMember⟩
          rcases List.mem_map.1 ownerMember with ⟨slot, _, rfl⟩
          exact List.mem_range.1 nodeMember
        have rightAt : right.node = count := by
          simp only [List.flatMap_cons, List.flatMap_nil,
            List.append_nil] at rightMember
          rcases List.mem_map.1 rightMember with ⟨slot, _, rfl⟩
          rfl
        rw [equal, rightAt] at leftLt
        omega

private def blankBlock : Block :=
  ⟨KLinear.zeroCarried, KLinear.zeroCarried, ⟨0, 0, 0⟩⟩

def ownedRow (nodeBlocks : List Block) (owner : RowOwner) : Row :=
  let block := nodeBlocks.getD owner.node blankBlock
  KMulOwnership.ownedRow block.left block.right block.frame owner.slot

private theorem flatMap_getD_range
    {α β : Type} (fallback : α) (f : α → List β) :
    ∀ list : List α,
      (List.range list.length).flatMap
          (fun index => f (list.getD index fallback)) =
        list.flatMap f
  | [] => rfl
  | head :: tail => by
      rw [List.length_cons, List.range_succ_eq_map, List.flatMap_cons,
        List.flatMap_map]
      exact congrArg (f head ++ ·) (flatMap_getD_range fallback f tail)

/-- The emitted row list is exactly the unique receipt list's image. -/
theorem rows_eq_map_owners
    (frames : Nat → Frame)
    {variables : Nat}
    (table : BooleanTable Carried variables)
    (coordinates : List Carried) (step : Nat) :
    KBooleanMle.rows frames table coordinates step =
      (owners (KBooleanMle.frameCount variables)).map
        (ownedRow (blocks frames table coordinates step)) := by
  rw [rows_eq_flatMap_blocks]
  have length :=
    blocks_length frames table coordinates step
  rw [owners, ← length, List.map_flatMap]
  simp only [List.map_map, Function.comp_def, ownedRow]
  rw [flatMap_getD_range blankBlock
    (fun block =>
      KMulOwnership.allOwners.map
        (KMulOwnership.ownedRow block.left block.right block.frame))]
  induction blocks frames table coordinates step with
  | nil => rfl
  | cons block rest inductionHypothesis =>
      simp only [List.flatMap_cons, blockRows]
      rw [KMulOwnership.rows_eq_map_owners, inductionHypothesis]

/-- Exactly one structural receipt per emitted position. -/
theorem ownership_is_positional
    (frames : Nat → Frame)
    {variables : Nat}
    (table : BooleanTable Carried variables)
    (coordinates : List Carried) (step : Nat) :
    (KBooleanMle.rows frames table coordinates step).length =
        (owners (KBooleanMle.frameCount variables)).length ∧
      (owners (KBooleanMle.frameCount variables)).Nodup ∧
      KBooleanMle.rows frames table coordinates step =
        (owners (KBooleanMle.frameCount variables)).map
          (ownedRow (blocks frames table coordinates step)) := by
  refine ⟨?_, owners_nodup _, rows_eq_map_owners frames table coordinates step⟩
  rw [KBooleanMle.rows_length, owners_length]

/-- Under the canonical placement, every mentioned column belongs either to
the source region or to this MLE's exact declared auxiliary allocation. -/
theorem rows_conservation
    (base : Nat)
    {variables : Nat}
    (table : BooleanTable Carried variables)
    (coordinates : List Carried)
    (tableBelow : TableBelowBase table base)
    (coordinatesBelow : CoordinatesBelowBase coordinates base)
    (row : Row)
    (member :
      row ∈ KBooleanMle.rows (KFrames.frameAt base) table coordinates 0)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨ Mentions row.c column) :
    column < base ∨ column ∈ KBooleanMle.columns base variables := by
  have upper :=
    KBooleanMleSupport.rows_below base table coordinates 0
      tableBelow coordinatesBelow row member column mentioned
  by_cases source : column < base
  · exact Or.inl source
  · exact Or.inr ((KFrames.frameColumns_mem_iff _ _ _).2
      ⟨by omega, by simpa [KBooleanMle.columns] using upper⟩)

end Nightstream.Implementation.R1CS.Canonical.KBooleanMleOwnership
