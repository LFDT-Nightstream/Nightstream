import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidatesSound
import Nightstream.Implementation.Lowering.Typed.Cost
import Nightstream.SuperNeo.Concrete.Phi81StrongSet

/-!
Contract: Lean-owned physical selection of the first 54 accepted candidates
from one 64-candidate `Pi_RLC` scalar.

The selector reads only columns already constrained by
`PiRlcCanonicalCandidates`: the verifier-derived accept bit, residue/symbol,
and accepted-prefix expression.  It allocates a four-bit acceptance slack and,
for each output position, an eleven-way one-hot selector, three products per
candidate, and one output column.  The output binding applies the canonical
centered Goldilocks embedding, so the selected 54 columns are immediately the
Phi81 ring coefficients consumed by `PiRLC`.

Candidate zero's prefix is the literal empty linear combination.  Consequently
this canonical construction needs no allocated zero-prefix column and emits
2,598 rows rather than the historical 2,599-row tail.

This file owns construction, exact row/column cost, contiguous allocation, and
restriction lemmas.  Soundness, honest completeness, and full conservation are
separate proof responsibilities.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelector

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

def outputCount : Nat := 54
def selectionWindow : Nat := 11
def slackBitCount : Nat := 4
def positionAuxiliaryCount : Nat := 45
def scalarAuxiliaryCount : Nat :=
  5 + outputCount * positionAuxiliaryCount

theorem parameter_values :
    outputCount = 54 ∧ selectionWindow = 11 ∧
      scalarAuxiliaryCount = 2435 := by
  decide

/-- The canonical centered embedding is the field shift by `-2`. -/
theorem embedCoefficient_val_eq_shift
    (coefficient : ProductionAlphabet.Coefficient) :
    (Phi81StrongSet.embedCoefficient coefficient).val =
      (coefficient.val + (goldilocksP - 2)) % goldilocksP := by
  unfold Phi81StrongSet.embedCoefficient
    Phi81StrongSet.centeredRepresentative
  have bounded := coefficient.isLt
  change coefficient.val < 5 at bounded
  have cases :
      coefficient.val = 0 ∨ coefficient.val = 1 ∨
        coefficient.val = 2 ∨ coefficient.val = 3 ∨
          coefficient.val = 4 := by
    omega
  rcases cases with value | value | value | value | value <;>
    simp [value, goldilocksP, goldilocksModulus]

def scalarBase (selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) : Nat :=
  selectorBase + coordinate.val * scalarAuxiliaryCount

def slackColumn (selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) : Nat :=
  scalarBase selectorBase coordinate

def slackBitColumn (selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (offset : Nat) : Nat :=
  scalarBase selectorBase coordinate + 1 + offset

def positionBase (selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount) : Nat :=
  scalarBase selectorBase coordinate + 5 +
    position.val * positionAuxiliaryCount

def selectorColumn (selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount)
    (offset : Fin selectionWindow) : Nat :=
  positionBase selectorBase coordinate position + offset.val

def symbolProductColumn (selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount)
    (offset : Fin selectionWindow) : Nat :=
  positionBase selectorBase coordinate position + 11 + 3 * offset.val

def acceptProductColumn (selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount)
    (offset : Fin selectionWindow) : Nat :=
  symbolProductColumn selectorBase coordinate position offset + 1

def prefixProductColumn (selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount)
    (offset : Fin selectionWindow) : Nat :=
  symbolProductColumn selectorBase coordinate position offset + 2

def outputColumn (selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount) : Nat :=
  positionBase selectorBase coordinate position + 44

def candidateAt (position : Fin outputCount)
    (offset : Fin selectionWindow) : Fin candidatesPerScalar :=
  ⟨position.val + offset.val, by
    have positionLt := position.isLt
    have offsetLt := offset.isLt
    simp only [outputCount] at positionLt
    simp only [selectionWindow] at offsetLt
    simp only [candidatesPerScalar]
    omega⟩

@[simp] theorem candidateAt_val
    (position : Fin outputCount) (offset : Fin selectionWindow) :
    (candidateAt position offset).val = position.val + offset.val := rfl

def candidateSourceLayout
    (duplexBase u64Base candidateBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (candidate : Fin candidatesPerScalar) :
    PiRlcCanonicalCandidate.Layout :=
  candidateLayout duplexBase u64Base candidateBase initial coordinate candidate

def acceptSource
    (duplexBase u64Base candidateBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (candidate : Fin candidatesPerScalar) : Nat :=
  PiRlcCanonicalCandidate.acceptColumn
    (candidateSourceLayout duplexBase u64Base candidateBase initial coordinate
      candidate)

def symbolSource
    (duplexBase u64Base candidateBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (candidate : Fin candidatesPerScalar) : Nat :=
  PiRlcCanonicalCandidate.residueColumn
    (candidateSourceLayout duplexBase u64Base candidateBase initial coordinate
      candidate)

def prefixSource
    (duplexBase u64Base candidateBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (candidate : Fin candidatesPerScalar) : LinComb :=
  (candidateSourceLayout duplexBase u64Base candidateBase initial coordinate
    candidate).prior

def finalCountSource
    (duplexBase u64Base candidateBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count) : Nat :=
  PiRlcCanonicalCandidate.cumulativeColumn
    (candidateSourceLayout duplexBase u64Base candidateBase initial coordinate
      ⟨63, by decide⟩)

def slackTerms (selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) : LinComb :=
  (List.range slackBitCount).map fun offset =>
    (slackBitColumn selectorBase coordinate offset, 2 ^ offset)

def acceptanceBoundRows
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count) : List Row :=
  (List.range slackBitCount).map
      (fun offset => bitRow (slackBitColumn selectorBase coordinate offset)) ++
    [ ⟨[(slackColumn selectorBase coordinate, 1)], [(0, 1)],
        slackTerms selectorBase coordinate⟩,
      ⟨[(finalCountSource duplexBase u64Base candidateBase initial coordinate,
          1)],
        [(0, 1)],
        [(slackColumn selectorBase coordinate, 1), (0, outputCount)]⟩ ]

def selectorTerms (selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount) : LinComb :=
  (List.finRange selectionWindow).map fun offset =>
    (selectorColumn selectorBase coordinate position offset, 1)

def oneHotRows (selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount) : List Row :=
  (List.finRange selectionWindow).map
      (fun offset =>
        bitRow (selectorColumn selectorBase coordinate position offset)) ++
    [⟨selectorTerms selectorBase coordinate position, [(0, 1)], [(0, 1)]⟩]

def productRowsAt
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count) (position : Fin outputCount)
    (offset : Fin selectionWindow) : List Row :=
  let candidate := candidateAt position offset
  [ ⟨[(selectorColumn selectorBase coordinate position offset, 1)],
      [(symbolSource duplexBase u64Base candidateBase initial coordinate
        candidate, 1)],
      [(symbolProductColumn selectorBase coordinate position offset, 1)]⟩,
    ⟨[(selectorColumn selectorBase coordinate position offset, 1)],
      [(acceptSource duplexBase u64Base candidateBase initial coordinate
        candidate, 1)],
      [(acceptProductColumn selectorBase coordinate position offset, 1)]⟩,
    ⟨[(selectorColumn selectorBase coordinate position offset, 1)],
      prefixSource duplexBase u64Base candidateBase initial coordinate
        candidate,
      [(prefixProductColumn selectorBase coordinate position offset, 1)]⟩ ]

def productRows
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount) : List Row :=
  (List.finRange selectionWindow).flatMap fun offset =>
    productRowsAt duplexBase u64Base candidateBase selectorBase initial
      coordinate position offset

def symbolProductTerms (selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount) : LinComb :=
  (List.finRange selectionWindow).map fun offset =>
    (symbolProductColumn selectorBase coordinate position offset, 1)

/-- Selected raw symbol shifted into its canonical centered field image. -/
def centeredSymbolTerms (selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount) : LinComb :=
  symbolProductTerms selectorBase coordinate position ++
    [(0, goldilocksP - 2)]

def acceptProductTerms (selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount) : LinComb :=
  (List.finRange selectionWindow).map fun offset =>
    (acceptProductColumn selectorBase coordinate position offset, 1)

def prefixProductTerms (selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount) : LinComb :=
  (List.finRange selectionWindow).map fun offset =>
    (prefixProductColumn selectorBase coordinate position offset, 1)

def positionTerms (position : Fin outputCount) : LinComb :=
  if position.val = 0 then [] else [(0, position.val)]

def bindingRows (selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount) : List Row :=
  [ ⟨acceptProductTerms selectorBase coordinate position, [(0, 1)], [(0, 1)]⟩,
    ⟨prefixProductTerms selectorBase coordinate position, [(0, 1)],
      positionTerms position⟩,
    ⟨[(outputColumn selectorBase coordinate position, 1)], [(0, 1)],
      centeredSymbolTerms selectorBase coordinate position⟩ ]

def positionRows
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount) : List Row :=
  oneHotRows selectorBase coordinate position ++
    productRows duplexBase u64Base candidateBase selectorBase initial
      coordinate position ++
    bindingRows selectorBase coordinate position

def scalarRows
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count) : List Row :=
  acceptanceBoundRows duplexBase u64Base candidateBase selectorBase initial
      coordinate ++
    (List.finRange outputCount).flatMap fun position =>
      positionRows duplexBase u64Base candidateBase selectorBase initial
        coordinate position

def rows
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder) : List Row :=
  (List.finRange count).flatMap fun coordinate =>
    scalarRows duplexBase u64Base candidateBase selectorBase initial coordinate

theorem acceptanceBoundRows_length
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count) :
    (acceptanceBoundRows duplexBase u64Base candidateBase selectorBase initial
      coordinate).length = 6 := by
  simp [acceptanceBoundRows, slackBitCount]

theorem positionRows_length
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount) :
    (positionRows duplexBase u64Base candidateBase selectorBase initial
      coordinate position).length = 48 := by
  simp [positionRows, oneHotRows, productRows, productRowsAt, bindingRows,
    selectionWindow]
  decide

private theorem sum_const {α : Type} (items : List α) (value : Nat) :
    (items.map (fun _ => value)).sum = items.length * value := by
  rw [List.map_const', List.sum_replicate_nat]

theorem scalarRows_length
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count) :
    (scalarRows duplexBase u64Base candidateBase selectorBase initial
      coordinate).length = 2598 := by
  simp [scalarRows, acceptanceBoundRows_length, positionRows_length]
  rw [sum_const]
  decide

theorem rows_length
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder) :
    (rows duplexBase u64Base candidateBase selectorBase count initial).length =
      count * 2598 := by
  simp [rows, scalarRows_length]
  rw [sum_const]
  simp

theorem fixedActive_rows_length
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder) :
    (rows duplexBase u64Base candidateBase selectorBase 15 initial).length =
      38970 := by
  rw [rows_length]

def allocation (selectorBase count : Nat) : List Nat :=
  (List.range (count * scalarAuxiliaryCount)).map
    (fun offset => selectorBase + offset)

theorem allocation_length (selectorBase count : Nat) :
    (allocation selectorBase count).length =
      count * scalarAuxiliaryCount := by
  simp [allocation]

theorem fixedActive_allocation_length (selectorBase : Nat) :
    (allocation selectorBase 15).length = 36525 := by
  rw [allocation_length]
  decide

theorem allocation_nodup (selectorBase count : Nat) :
    (allocation selectorBase count).Nodup := by
  unfold allocation
  exact nodup_map _ _ (fun _ _ equal => by omega) List.nodup_range

theorem allocation_nonzero
    (selectorBase count column : Nat) (positive : 0 < selectorBase)
    (member : column ∈ allocation selectorBase count) :
    column ≠ 0 := by
  unfold allocation at member
  rcases List.mem_map.mp member with ⟨offset, _, rfl⟩
  omega

theorem allocation_mem_iff
    (selectorBase count column : Nat) :
    column ∈ allocation selectorBase count ↔
      selectorBase ≤ column ∧
        column < selectorBase + count * scalarAuxiliaryCount := by
  unfold allocation
  constructor
  · intro member
    rcases List.mem_map.mp member with ⟨offset, inRange, rfl⟩
    have bounded := List.mem_range.mp inRange
    omega
  · intro ⟨lower, upper⟩
    exact List.mem_map.mpr
      ⟨column - selectorBase, List.mem_range.mpr (by omega), by omega⟩

def cost (count : Nat) : Lowering.Typed.Cost where
  recurringRows := count * 2598
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := count * scalarAuxiliaryCount

theorem cost_rows
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder) :
    (rows duplexBase u64Base candidateBase selectorBase count initial).length =
      (cost count).recurringRows :=
  rows_length _ _ _ _ _ _

theorem cost_columns (selectorBase count : Nat) :
    (allocation selectorBase count).length =
      (cost count).auxiliaryColumns :=
  allocation_length _ _

theorem satisfies_scalar
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder) (assignment : Nat → Nat)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) :
    Satisfies
      (scalarRows duplexBase u64Base candidateBase selectorBase initial
        coordinate)
      assignment := by
  intro row member
  apply satisfied row
  unfold rows
  exact List.mem_flatMap.mpr
    ⟨coordinate, List.mem_finRange coordinate, member⟩

theorem satisfies_acceptanceBoundRows
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder) (assignment : Nat → Nat)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) :
    Satisfies
      (acceptanceBoundRows duplexBase u64Base candidateBase selectorBase
        initial coordinate)
      assignment := by
  intro row member
  apply satisfies_scalar duplexBase u64Base candidateBase selectorBase count
    initial assignment satisfied coordinate row
  unfold scalarRows
  exact List.mem_append_left _ member

theorem satisfies_position
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder) (assignment : Nat → Nat)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) (position : Fin outputCount) :
    Satisfies
      (positionRows duplexBase u64Base candidateBase selectorBase initial
        coordinate position)
      assignment := by
  intro row member
  apply satisfies_scalar duplexBase u64Base candidateBase selectorBase count
    initial assignment satisfied coordinate row
  unfold scalarRows
  apply List.mem_append_right
  exact List.mem_flatMap.mpr
    ⟨position, List.mem_finRange position, member⟩

theorem satisfies_oneHotRows
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder) (assignment : Nat → Nat)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) (position : Fin outputCount) :
    Satisfies (oneHotRows selectorBase coordinate position) assignment := by
  intro row member
  apply satisfies_position duplexBase u64Base candidateBase selectorBase count
    initial assignment satisfied coordinate position row
  unfold positionRows
  apply List.mem_append_left
  exact List.mem_append_left _ member

theorem satisfies_productRowsAt
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder) (assignment : Nat → Nat)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) (position : Fin outputCount)
    (offset : Fin selectionWindow) :
    Satisfies
      (productRowsAt duplexBase u64Base candidateBase selectorBase initial
        coordinate position offset)
      assignment := by
  intro row member
  apply satisfies_position duplexBase u64Base candidateBase selectorBase count
    initial assignment satisfied coordinate position row
  unfold positionRows
  apply List.mem_append_left
  apply List.mem_append_right
  unfold productRows
  exact List.mem_flatMap.mpr
    ⟨offset, List.mem_finRange offset, member⟩

theorem satisfies_bindingRows
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder) (assignment : Nat → Nat)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) (position : Fin outputCount) :
    Satisfies (bindingRows selectorBase coordinate position) assignment := by
  intro row member
  apply satisfies_position duplexBase u64Base candidateBase selectorBase count
    initial assignment satisfied coordinate position row
  unfold positionRows
  exact List.mem_append_right _ member

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelector
