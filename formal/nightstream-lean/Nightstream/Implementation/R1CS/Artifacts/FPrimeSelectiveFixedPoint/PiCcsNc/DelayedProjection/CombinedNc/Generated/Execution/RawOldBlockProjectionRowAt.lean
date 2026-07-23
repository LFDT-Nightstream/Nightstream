/-
Generated file: production combined-NC execution artifact; do not hand-edit.

Owns: the Rust-owned active raw-old-block owner, inverse offsets, canonical columns, and exact indexed A/B/C row formula.

Does not own: row satisfaction, commitment binding, semantic acceptance,
security reductions, costs, or permission to remove rows.

Emits constraints: no.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.execution` | The generated execution payload named by `Owns` above | computed artifact |
-/

import Nightstream.Implementation.R1CS.Core.Projection.Trace
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan

def constantOneColumn : Nat := 0
def oldBlockFirstColumn : Nat := 1
def parentFirstColumn : Nat := 39
def witnessFamilyFirstColumn : Nat := 147
def tensorFirstColumn : Nat := 160118679
def productFirstColumn : Nat := 161429394
def finalScaleFirstColumn : Nat := 184303470
def canonicalColumnCount : Nat := 184303740
def tensorVariables : Nat := 18
def factoredVariable : Nat := 18
def tensorRows : Nat := 1310715
def productRows : Nat := 22874076
def finalScaleRows : Nat := 270
def terminalRows : Nat := 108
def totalRows : Nat := 24185169
def productRowFirst : Nat := 1310715
def finalScaleRowFirst : Nat := 24184791
def terminalRowFirst : Nat := 24185061
def tensorRoundMulCounts : List Nat := [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
def tensorRoundHighCounts : List Nat := [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 80725]
def tensorRoundMulStarts : List Nat := [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131071]

def kColumnsAt (first : Nat) : KColumns := { c0 := first, c1 := first + 1 }
def oldBlockColumnsNat (round : Nat) : KColumns :=
  kColumnsAt (oldBlockFirstColumn + 2 * round)
def oldBlockColumns (round : Fin blockVariables) : KColumns :=
  oldBlockColumnsNat round.val
def parentColumnsNat (lane : Nat) : KColumns :=
  kColumnsAt (parentFirstColumn + 2 * lane)
def parentColumns (lane : Fin activeLanes) : KColumns :=
  parentColumnsNat lane.val
def witnessEntriesPerChild : Nat := activeLanes * blockCount
def childWitnessFirstNat (child : Nat) : Nat :=
  witnessFamilyFirstColumn + child * witnessEntriesPerChild
def childWitnessFirst (child : Fin childCount) : Nat :=
  childWitnessFirstNat child.val
def witnessOffset (lane block : Nat) : Nat := lane * blockCount + block
def childWitnessColumn (child lane block : Nat) : Nat :=
  childWitnessFirstNat child + witnessOffset lane block
def tensorRoundMulCount (round : Nat) : Nat := Nat.min blockCount (2 ^ round)
def tensorRoundHighCount (round : Nat) : Nat :=
  Nat.min (blockCount - 2 ^ round) (2 ^ round)
def tensorRoundMulStart (round : Nat) : Nat :=
  (List.range round).foldl (fun count prior => count + tensorRoundMulCount prior) 0
def tensorMulOrdinal (round parent : Nat) : Nat :=
  tensorRoundMulStart round + parent
def tensorMulFirstColumn (round parent : Nat) : Nat :=
  tensorFirstColumn + 5 * tensorMulOrdinal round parent
def tensorOutputColumns (round parent : Nat) : KColumns :=
  kColumnsAt (tensorMulFirstColumn round parent + 3)
def productColumn (lane block limb : Nat) : Nat :=
  productFirstColumn + 2 * witnessOffset lane block + limb
def finalScaleOutput (lane : Nat) : KColumns :=
  kColumnsAt (finalScaleFirstColumn + 5 * lane + 3)
def tensorPhysicalRow (round parent definition : Nat) : Nat :=
  5 * tensorMulOrdinal round parent + definition
def productPhysicalRow (lane block limb : Nat) : Nat :=
  productRowFirst + 2 * witnessOffset lane block + limb
def finalScalePhysicalRow (lane definition : Nat) : Nat :=
  finalScaleRowFirst + 5 * lane + definition
def terminalPhysicalRow (lane limb : Nat) : Nat :=
  terminalRowFirst + 2 * lane + limb
def emptyRow : Row := { a := [], b := [], c := [] }
def emptyKTerms : KTerms := { c0 := [], c1 := [] }
def tensorRoot : KTerms := { c0 := [(constantOneColumn, 1)], c1 := [] }
def pointTerms (round : Nat) : KTerms := KTerms.ofColumns (oldBlockColumnsNat round)
def oneMinusPointTerms (round : Nat) : KTerms :=
  let point := oldBlockColumnsNat round
  { c0 := [(constantOneColumn, 1), (point.c0, goldilocksP - 1)]
    c1 := [(point.c1, goldilocksP - 1)] }
def subtractOutput (terms : KTerms) (output : KColumns) : KTerms :=
  { c0 := terms.c0 ++ [(output.c0, goldilocksP - 1)]
    c1 := terms.c1 ++ [(output.c1, goldilocksP - 1)] }
def tensorTermsAt : Nat -> Nat -> KTerms
  | 0, index => if index = 0 then tensorRoot else emptyKTerms
  | round + 1, index =>
      let count := tensorRoundMulCount round
      let high := tensorRoundHighCount round
      if index < count then
        if index < high then
          subtractOutput (tensorTermsAt round index)
            (tensorOutputColumns round index)
        else KTerms.ofColumns (tensorOutputColumns round index)
      else
        let parent := index - count
        if parent < high then
          KTerms.ofColumns (tensorOutputColumns round parent)
        else emptyKTerms
def tensorTrace (round parent : Nat) : KMulTrace :=
  let left := tensorTermsAt round parent
  let right := if parent < tensorRoundHighCount round then
    pointTerms round else oneMinusPointTerms round
  let first := tensorMulFirstColumn round parent
  { left := left
    right := right
    sumLeft := left.c0 ++ left.c1
    sumRight := right.c0 ++ right.c1
    productC0 := first
    productC1 := first + 1
    productSum := first + 2
    output := kColumnsAt (first + 3) }
def rawTerms (lane block : Nat) : List (Nat × Nat) :=
  (List.range childCount).map fun child =>
    (childWitnessColumn child lane block, radixBase ^ child % goldilocksP)
def chiTerms (block : Nat) : KTerms :=
  tensorTermsAt tensorVariables block


def laneTerms (lane limb : Nat) : List (Nat × Nat) :=
  (List.range blockCount).map fun block => (productColumn lane block limb, 1)
def finalScaleTrace (lane : Nat) : KMulTrace :=
  let left : KTerms := { c0 := laneTerms lane 0, c1 := laneTerms lane 1 }
  let right := oneMinusPointTerms factoredVariable
  let first := finalScaleFirstColumn + 5 * lane
  { left := left
    right := right
    sumLeft := left.c0 ++ left.c1
    sumRight := right.c0 ++ right.c1
    productC0 := first
    productC1 := first + 1
    productSum := first + 2
    output := kColumnsAt (first + 3) }
inductive RowOwner where
  | tensor (round parent definition : Nat)
  | product (lane block limb : Nat)
  | finalScale (lane definition : Nat)
  | terminal (lane limb : Nat)
deriving DecidableEq, Repr

def tensorOwner (ordinal : Nat) : Nat × Nat :=
  if ordinal < 1 then (0, ordinal - 0) else (if ordinal < 3 then (1, ordinal - 1) else (if ordinal < 7 then (2, ordinal - 3) else (if ordinal < 15 then (3, ordinal - 7) else (if ordinal < 31 then (4, ordinal - 15) else (if ordinal < 63 then (5, ordinal - 31) else (if ordinal < 127 then (6, ordinal - 63) else (if ordinal < 255 then (7, ordinal - 127) else (if ordinal < 511 then (8, ordinal - 255) else (if ordinal < 1023 then (9, ordinal - 511) else (if ordinal < 2047 then (10, ordinal - 1023) else (if ordinal < 4095 then (11, ordinal - 2047) else (if ordinal < 8191 then (12, ordinal - 4095) else (if ordinal < 16383 then (13, ordinal - 8191) else (if ordinal < 32767 then (14, ordinal - 16383) else (if ordinal < 65535 then (15, ordinal - 32767) else (if ordinal < 131071 then (16, ordinal - 65535) else ((17, ordinal - 131071))))))))))))))))))

def ownerAtNat (row : Nat) : RowOwner :=
  if row < tensorRows then
    let ordinal := row / 5
    let owner := tensorOwner ordinal
    .tensor owner.1 owner.2 (row % 5)
  else if row < finalScaleRowFirst then
    let offset := row - productRowFirst
    let product := offset / 2
    .product (product / blockCount) (product % blockCount) (offset % 2)
  else if row < terminalRowFirst then
    let offset := row - finalScaleRowFirst
    .finalScale (offset / 5) (offset % 5)
  else
    let offset := row - terminalRowFirst
    .terminal (offset / 2) (offset % 2)
def ownerAt (row : Fin totalRows) : RowOwner := ownerAtNat row.val
def tensorRow (round parent definition : Nat) : Row :=
  match (tensorTrace round parent).definitions[definition]? with
  | some current => current.builderRow
  | none => emptyRow
def productRow (lane block limb : Nat) : Row :=
  let chi := chiTerms block
  let selected := if limb = 0 then chi.c0 else chi.c1
  (Definition.mk (productColumn lane block limb)
    (.product (rawTerms lane block) selected)).builderRow
def finalScaleRow (lane definition : Nat) : Row :=
  match (finalScaleTrace lane).definitions[definition]? with
  | some current => current.builderRow
  | none => emptyRow
def terminalRow (lane limb : Nat) : Row :=
  builderLinearRow
    (if limb = 0 then (parentColumnsNat lane).c0 else (parentColumnsNat lane).c1)
    [(if limb = 0 then (finalScaleTrace lane).output.c0
      else (finalScaleTrace lane).output.c1, 1)]
def artifactRowForOwner : RowOwner -> Row
  | .tensor round parent definition => tensorRow round parent definition
  | .product lane block limb => productRow lane block limb
  | .finalScale lane definition => finalScaleRow lane definition
  | .terminal lane limb => terminalRow lane limb
def artifactRow (row : Fin totalRows) : Row := artifactRowForOwner (ownerAt row)

/-- Serialization of the crate-private `RawOldBlockProjectionColumnMap`
stored in `TerminalPendingProjectionAudit`. Production constructs it
inside `enforce_raw_old_block_projection`; it is not a prover or
theorem-caller authority input. -/
structure EmitterLayout where
  rowFirst : Nat
  rowStop : Nat
  oldBlock : List KColumns
  parent : List KColumns
  finalWitnessFirst : List Nat
  tensorFirst : Nat
  productFirst : Nat
  finalScaleFirst : Nat
deriving DecidableEq, Repr
structure ColumnInterval where
  start : Nat
  stop : Nat
deriving DecidableEq, Repr
def allDistinct : List Nat -> Bool
  | [] => true
  | head :: tail => !(tail.contains head) && allDistinct tail
def emitterScalarColumns (emitter : EmitterLayout) : List Nat :=
  [constantOneColumn] ++
  emitter.oldBlock.flatMap (fun columns => [columns.c0, columns.c1]) ++
  emitter.parent.flatMap (fun columns => [columns.c0, columns.c1])
def childIntervals (emitter : EmitterLayout) : List ColumnInterval :=
  emitter.finalWitnessFirst.map fun first =>
    { start := first, stop := first + witnessEntriesPerChild }
def emitterIntervals (emitter : EmitterLayout) : List ColumnInterval :=
  childIntervals emitter ++
  [ { start := emitter.tensorFirst, stop := emitter.productFirst },
    { start := emitter.productFirst, stop := emitter.finalScaleFirst },
    { start := emitter.finalScaleFirst, stop := emitter.finalScaleFirst + finalScaleRows } ]
def intervalsDisjoint (left right : ColumnInterval) : Bool :=
  decide (left.stop <= right.start) || decide (right.stop <= left.start)
def intervalsPairwiseDisjoint : List ColumnInterval -> Bool
  | [] => true
  | head :: tail =>
      tail.all (intervalsDisjoint head) && intervalsPairwiseDisjoint tail
def intervalContains (interval : ColumnInterval) (column : Nat) : Bool :=
  decide (interval.start <= column) && decide (column < interval.stop)
def scalarsOutsideIntervals (emitter : EmitterLayout) : Bool :=
  (emitterScalarColumns emitter).all fun column =>
    !((emitterIntervals emitter).any fun interval => intervalContains interval column)
def selectLimb (columns : KColumns) (limb : Nat) : Nat :=
  if limb = 0 then columns.c0 else columns.c1
def emitterColumnMap (emitter : EmitterLayout) (column : Nat) : Nat :=
  if column = constantOneColumn then constantOneColumn
  else if column < parentFirstColumn then
    let offset := column - oldBlockFirstColumn
    selectLimb (emitter.oldBlock.getD (offset / 2) default) (offset % 2)
  else if column < witnessFamilyFirstColumn then
    let offset := column - parentFirstColumn
    selectLimb (emitter.parent.getD (offset / 2) default) (offset % 2)
  else if column < tensorFirstColumn then
    let offset := column - witnessFamilyFirstColumn
    emitter.finalWitnessFirst.getD (offset / witnessEntriesPerChild) 0 +
      offset % witnessEntriesPerChild
  else if column < productFirstColumn then
    emitter.tensorFirst + column - tensorFirstColumn
  else if column < finalScaleFirstColumn then
    emitter.productFirst + column - productFirstColumn
  else emitter.finalScaleFirst + column - finalScaleFirstColumn
def findKColumnAux : List KColumns -> Nat -> Nat -> Option (Nat × Nat)
  | [], _, _ => none
  | head :: tail, index, column =>
      if column = head.c0 then some (index, 0)
      else if column = head.c1 then some (index, 1)
      else findKColumnAux tail (index + 1) column
def findWitnessIntervalAux : List Nat -> Nat -> Nat -> Option (Nat × Nat)
  | [], _, _ => none
  | first :: tail, child, column =>
      if first ≤ column ∧ column < first + witnessEntriesPerChild then
        some (child, column - first)
      else findWitnessIntervalAux tail (child + 1) column
def emitterColumnInverse (emitter : EmitterLayout) (column : Nat) : Option Nat :=
  if column = constantOneColumn then some constantOneColumn
  else if emitter.tensorFirst ≤ column ∧ column < emitter.productFirst then
    some (tensorFirstColumn + (column - emitter.tensorFirst))
  else if emitter.productFirst ≤ column ∧ column < emitter.productFirst + productRows then
    some (productFirstColumn + (column - emitter.productFirst))
  else if emitter.finalScaleFirst ≤ column ∧ column < emitter.finalScaleFirst + finalScaleRows then
    some (finalScaleFirstColumn + (column - emitter.finalScaleFirst))
  else
    match findKColumnAux emitter.oldBlock 0 column with
    | some (round, limb) => some (oldBlockFirstColumn + 2 * round + limb)
    | none =>
        match findKColumnAux emitter.parent 0 column with
        | some (lane, limb) => some (parentFirstColumn + 2 * lane + limb)
        | none =>
            match findWitnessIntervalAux emitter.finalWitnessFirst 0 column with
            | some (child, offset) =>
                some (witnessFamilyFirstColumn + child * witnessEntriesPerChild + offset)
            | none => none
def physicalRow (emitter : EmitterLayout) (row : Fin totalRows) : Nat :=
  emitter.rowFirst + row.val
def expectedRowStop (emitter : EmitterLayout) : Nat := emitter.rowFirst + totalRows
def emitterShapePinned (emitter : EmitterLayout) : Bool :=
  emitter.rowStop == expectedRowStop emitter &&
  emitter.oldBlock.length == blockVariables &&
  emitter.parent.length == activeLanes &&
  emitter.finalWitnessFirst.length == childCount &&
  emitter.productFirst == emitter.tensorFirst + tensorRows &&
  emitter.finalScaleFirst == emitter.productFirst + productRows
def emitterColumnMapValid (emitter : EmitterLayout) : Bool :=
  emitterShapePinned emitter &&
  allDistinct (emitterScalarColumns emitter) &&
  intervalsPairwiseDisjoint (emitterIntervals emitter) &&
  scalarsOutsideIntervals emitter
def actualRow (emitter : EmitterLayout) (row : Fin totalRows) : Row :=
  renameRow (emitterColumnMap emitter) (artifactRow row)
def projectionChildWitnessFirst (emitter : EmitterLayout) (child : Nat) : Nat :=
  emitter.finalWitnessFirst.getD child 0
def ajtaiChildWitnessFirst (emitter : EmitterLayout) (child : Nat) : Nat :=
  emitter.finalWitnessFirst.getD child 0

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt
