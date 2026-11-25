import NightstreamFPrime.Export.Generic.CircuitRelation
import Std.Data.HashMap.Lemmas

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.CircuitRelation

open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.Frame

private abbrev Buckets := Std.HashMap Nat (List Nat)

private def putStart (index : Buckets) (bucket start : Nat) : Buckets :=
  index.insert bucket (start :: index.getD bucket [])

private def addBlock (index : Buckets) (block : Block) : Buckets :=
  let bucket := block.start / 1080
  putStart (putStart index bucket block.start) (bucket + 1) block.start

private def marked (index : Buckets) (column : Nat) : Bool :=
  (index.getD (column / 1080) []).any fun start => start ≤ column && column < start + 1080

private theorem marked_put (index : Buckets) (bucket start column : Nat) :
    marked (putStart index bucket start) column =
      ((bucket == column / 1080 && decide (start ≤ column) && decide (column < start + 1080)) ||
        marked index column) := by
  unfold marked putStart
  rw [Std.HashMap.getD_insert]
  by_cases same : bucket = column / 1080 <;> simp [same]

private theorem marked_add (index : Buckets) (block : Block) (column : Nat) :
    marked (addBlock index block) column = (block.dead column || marked index column) := by
  rw [addBlock, marked_put, marked_put]
  by_cases inside : block.start ≤ column ∧ column < block.start + 1080
  · have interval : block.start / 1080 = column / 1080 ∨
        block.start / 1080 + 1 = column / 1080 := by omega
    rcases interval with low | high <;> simp [Block.dead, inside.1, inside.2, *]
  · simp only [Block.dead]
    rcases not_and_or.mp inside with h | h <;> simp [h]

private theorem marked_fold (blocks : List Block) (index : Buckets) (column : Nat) :
    marked (blocks.foldl addBlock index) column = (deadIn blocks column || marked index column) := by
  induction blocks generalizing index with
  | nil => simp [deadIn]
  | cons block rest ih =>
    simp only [List.foldl_cons, ih, marked_add, deadIn, List.any_cons]
    ac_rfl

def packableFast (publicColumns : Nat) (pieces : List Piece) : Bool :=
  let blocks := blocksOf pieces
  let index := blocks.foldl addBlock {}
  let dead := marked index
  ordered blocks && (blocks.all fun block => block.lanes.all fun lane =>
    (linear? lane).isSome && avoids (fun column => block.covers column || dead column) lane) &&
    (blocks.all fun block => decide (publicColumns ≤ block.start)) && pieces.all (avoidsDead dead)

@[csimp] theorem packable_eq_packableFast : @packable = @packableFast := by
  funext publicColumns pieces
  have lookup : marked ((blocksOf pieces).foldl addBlock {}) = deadIn (blocksOf pieces) := by
    funext column
    rw [marked_fold]
    simp [marked]
  simp only [packableFast, lookup, packable, checkFrame, startsFrom]

end NightstreamFPrime.Export.Generic.CircuitRelation
