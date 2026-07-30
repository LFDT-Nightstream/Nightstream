import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelector

/-!
Contract: exact categorical column conservation for the canonical
first-accepted selector batch.

Every selector operand is the shared constant wire, a column in the exact
candidate-family allocation, or a column in the exact selector-family
allocation.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorConservation

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelector

def Allowed
    (candidateBase selectorBase count column : Nat) : Prop :=
  column = 0 ∨
    column ∈ PiRlcCanonicalCandidates.allocation candidateBase count ∨
    column ∈ allocation selectorBase count

private def CombAllowed
    (candidateBase selectorBase count : Nat) (comb : LinComb) : Prop :=
  ∀ column, Mentions comb column →
    Allowed candidateBase selectorBase count column

private def RowAllowed
    (candidateBase selectorBase count : Nat) (row : Row) : Prop :=
  CombAllowed candidateBase selectorBase count row.a ∧
    CombAllowed candidateBase selectorBase count row.b ∧
      CombAllowed candidateBase selectorBase count row.c

private theorem allowed_constant
    (candidateBase selectorBase count : Nat) :
    Allowed candidateBase selectorBase count 0 :=
  Or.inl rfl

private theorem allowed_candidate
    (candidateBase selectorBase count column : Nat)
    (member :
      column ∈ PiRlcCanonicalCandidates.allocation candidateBase count) :
    Allowed candidateBase selectorBase count column :=
  Or.inr (Or.inl member)

private theorem allowed_selector
    (candidateBase selectorBase count column : Nat)
    (member : column ∈ allocation selectorBase count) :
    Allowed candidateBase selectorBase count column :=
  Or.inr (Or.inr member)

private theorem combAllowed_nil
    (candidateBase selectorBase count : Nat) :
    CombAllowed candidateBase selectorBase count [] := by
  intro column mentioned
  simp [Mentions] at mentioned

private theorem combAllowed_single
    (candidateBase selectorBase count column coefficient : Nat)
    (allowed : Allowed candidateBase selectorBase count column) :
    CombAllowed candidateBase selectorBase count [(column, coefficient)] := by
  intro target mentioned
  have equal := (mentions_single column target coefficient).mp mentioned
  subst target
  exact allowed

private theorem combAllowed_append
    (candidateBase selectorBase count : Nat)
    (left right : LinComb)
    (leftAllowed : CombAllowed candidateBase selectorBase count left)
    (rightAllowed : CombAllowed candidateBase selectorBase count right) :
    CombAllowed candidateBase selectorBase count (left ++ right) := by
  intro column mentioned
  rw [mentions_append] at mentioned
  exact mentioned.elim (leftAllowed column) (rightAllowed column)

private theorem combAllowed_pair
    (candidateBase selectorBase count : Nat)
    (leftColumn leftCoefficient rightColumn rightCoefficient : Nat)
    (leftAllowed : Allowed candidateBase selectorBase count leftColumn)
    (rightAllowed : Allowed candidateBase selectorBase count rightColumn) :
    CombAllowed candidateBase selectorBase count
      [(leftColumn, leftCoefficient), (rightColumn, rightCoefficient)] :=
  combAllowed_append candidateBase selectorBase count
    [(leftColumn, leftCoefficient)] [(rightColumn, rightCoefficient)]
    (combAllowed_single candidateBase selectorBase count
      leftColumn leftCoefficient leftAllowed)
    (combAllowed_single candidateBase selectorBase count
      rightColumn rightCoefficient rightAllowed)

private theorem selectorAllocated_of_local
    (candidateBase selectorBase count : Nat)
    (coordinate : Fin count) (column : Nat)
    (lower : scalarBase selectorBase coordinate ≤ column)
    (upper :
      column <
        scalarBase selectorBase coordinate + scalarAuxiliaryCount) :
    Allowed candidateBase selectorBase count column := by
  apply allowed_selector
  rw [allocation_mem_iff]
  have coordinateLt := coordinate.isLt
  have coordinateBound : coordinate.val + 1 ≤ count := by omega
  have scaled :=
    Nat.mul_le_mul_right scalarAuxiliaryCount coordinateBound
  simp only [Nat.add_mul, Nat.one_mul] at scaled
  simp only [scalarBase] at lower upper
  exact ⟨by omega, by omega⟩

private theorem slack_allowed
    (candidateBase selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) :
    Allowed candidateBase selectorBase count
      (slackColumn selectorBase coordinate) := by
  apply selectorAllocated_of_local candidateBase selectorBase count coordinate
  · unfold slackColumn
    omega
  · simp only [slackColumn, scalarAuxiliaryCount, outputCount,
      positionAuxiliaryCount]
    omega

private theorem slackBit_allowed
    (candidateBase selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (offset : Nat)
    (bounded : offset < slackBitCount) :
    Allowed candidateBase selectorBase count
      (slackBitColumn selectorBase coordinate offset) := by
  apply selectorAllocated_of_local candidateBase selectorBase count coordinate
  · unfold slackBitColumn
    omega
  · simp only [slackBitColumn, slackBitCount, scalarAuxiliaryCount,
      outputCount, positionAuxiliaryCount] at *
    omega

private theorem selectorColumn_allowed
    (candidateBase selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount)
    (offset : Fin selectionWindow) :
    Allowed candidateBase selectorBase count
      (selectorColumn selectorBase coordinate position offset) := by
  apply selectorAllocated_of_local candidateBase selectorBase count coordinate
  · unfold selectorColumn positionBase
    omega
  · have positionLt := position.isLt
    have offsetLt := offset.isLt
    simp only [selectorColumn, positionBase, outputCount, selectionWindow,
      scalarAuxiliaryCount, positionAuxiliaryCount] at *
    omega

private theorem symbolProduct_allowed
    (candidateBase selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount)
    (offset : Fin selectionWindow) :
    Allowed candidateBase selectorBase count
      (symbolProductColumn selectorBase coordinate position offset) := by
  apply selectorAllocated_of_local candidateBase selectorBase count coordinate
  · unfold symbolProductColumn positionBase
    omega
  · have positionLt := position.isLt
    have offsetLt := offset.isLt
    simp only [symbolProductColumn, positionBase, outputCount, selectionWindow,
      scalarAuxiliaryCount, positionAuxiliaryCount] at *
    omega

private theorem acceptProduct_allowed
    (candidateBase selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount)
    (offset : Fin selectionWindow) :
    Allowed candidateBase selectorBase count
      (acceptProductColumn selectorBase coordinate position offset) := by
  apply selectorAllocated_of_local candidateBase selectorBase count coordinate
  · unfold acceptProductColumn symbolProductColumn positionBase
    omega
  · have positionLt := position.isLt
    have offsetLt := offset.isLt
    simp only [acceptProductColumn, symbolProductColumn, positionBase,
      outputCount, selectionWindow, scalarAuxiliaryCount,
      positionAuxiliaryCount] at *
    omega

private theorem prefixProduct_allowed
    (candidateBase selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount)
    (offset : Fin selectionWindow) :
    Allowed candidateBase selectorBase count
      (prefixProductColumn selectorBase coordinate position offset) := by
  apply selectorAllocated_of_local candidateBase selectorBase count coordinate
  · unfold prefixProductColumn symbolProductColumn positionBase
    omega
  · have positionLt := position.isLt
    have offsetLt := offset.isLt
    simp only [prefixProductColumn, symbolProductColumn, positionBase,
      outputCount, selectionWindow, scalarAuxiliaryCount,
      positionAuxiliaryCount] at *
    omega

private theorem output_allowed
    (candidateBase selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount) :
    Allowed candidateBase selectorBase count
      (outputColumn selectorBase coordinate position) := by
  apply selectorAllocated_of_local candidateBase selectorBase count coordinate
  · unfold outputColumn positionBase
    omega
  · have positionLt := position.isLt
    simp only [outputColumn, positionBase, outputCount,
      scalarAuxiliaryCount, positionAuxiliaryCount] at *
    omega

private theorem candidateLocalAllocation
    (duplexBase u64Base candidateBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar)
    (column : Nat)
    (member :
      column ∈ PiRlcCanonicalCandidate.allocation
        (candidateSourceLayout duplexBase u64Base candidateBase initial
          coordinate candidate)) :
    column ∈ PiRlcCanonicalCandidates.allocation candidateBase count := by
  exact PiRlcCanonicalCandidates.occurrence_allocation_mem
    duplexBase u64Base candidateBase initial coordinate candidate column
    member

private theorem acceptSource_allowed
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar) :
    Allowed candidateBase selectorBase count
      (acceptSource duplexBase u64Base candidateBase initial coordinate
        candidate) := by
  apply allowed_candidate
  apply candidateLocalAllocation duplexBase u64Base candidateBase initial
    coordinate candidate
  rw [PiRlcCanonicalCandidate.allocation_mem_iff]
  simp only [acceptSource, candidateSourceLayout,
    PiRlcCanonicalCandidate.acceptColumn,
    PiRlcCanonicalCandidate.auxiliaryCount]
  omega

private theorem symbolSource_allowed
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar) :
    Allowed candidateBase selectorBase count
      (symbolSource duplexBase u64Base candidateBase initial coordinate
        candidate) := by
  apply allowed_candidate
  apply candidateLocalAllocation duplexBase u64Base candidateBase initial
    coordinate candidate
  rw [PiRlcCanonicalCandidate.allocation_mem_iff]
  simp only [symbolSource, candidateSourceLayout,
    PiRlcCanonicalCandidate.residueColumn,
    PiRlcCanonicalCandidate.auxiliaryCount]
  omega

private theorem finalCountSource_allowed
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count) :
    Allowed candidateBase selectorBase count
      (finalCountSource duplexBase u64Base candidateBase initial
        coordinate) := by
  apply allowed_candidate
  apply candidateLocalAllocation duplexBase u64Base candidateBase initial
    coordinate ⟨63, by decide⟩
  rw [PiRlcCanonicalCandidate.allocation_mem_iff]
  simp only [finalCountSource, candidateSourceLayout,
    PiRlcCanonicalCandidate.cumulativeColumn,
    PiRlcCanonicalCandidate.auxiliaryCount]
  omega

private theorem prefixSource_allowed
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar) :
    CombAllowed candidateBase selectorBase count
      (prefixSource duplexBase u64Base candidateBase initial coordinate
        candidate) := by
  intro column mentioned
  simp only [prefixSource, candidateSourceLayout,
    PiRlcCanonicalCandidates.candidateLayout] at mentioned
  unfold PiRlcCanonicalCandidates.prior at mentioned
  split at mentioned
  · simp [Mentions] at mentioned
  · simp only [Mentions, List.map_cons, List.map_nil, List.mem_cons,
      List.not_mem_nil, or_false] at mentioned
    rcases mentioned with ⟨rfl, rfl⟩
    apply allowed_candidate
    rw [PiRlcCanonicalCandidates.allocation_mem_iff]
    have coordinateLt := coordinate.isLt
    have candidateLt := candidate.isLt
    simp only [PiRlcCanonicalCandidates.occurrenceBase,
      PiRlcCanonicalCandidates.occurrenceIndex,
      PiRlcCanonicalCandidates.candidatesPerScalar,
      PiRlcCanonicalCandidate.auxiliaryCount] at candidateLt ⊢
    omega

private theorem slackTerms_allowed
    (candidateBase selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) :
    CombAllowed candidateBase selectorBase count
      (slackTerms selectorBase coordinate) := by
  intro column mentioned
  unfold slackTerms Mentions at mentioned
  rw [List.map_map] at mentioned
  change column ∈
    (List.range slackBitCount).map
      (slackBitColumn selectorBase coordinate) at mentioned
  rcases List.mem_map.mp mentioned with ⟨offset, inRange, rfl⟩
  exact slackBit_allowed candidateBase selectorBase coordinate offset
    (List.mem_range.mp inRange)

private theorem selectorTerms_allowed
    (candidateBase selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount) :
    CombAllowed candidateBase selectorBase count
      (selectorTerms selectorBase coordinate position) := by
  intro column mentioned
  unfold selectorTerms Mentions at mentioned
  rw [List.map_map] at mentioned
  change column ∈
    (List.finRange selectionWindow).map
      (selectorColumn selectorBase coordinate position) at mentioned
  rcases List.mem_map.mp mentioned with ⟨offset, _, rfl⟩
  exact selectorColumn_allowed candidateBase selectorBase coordinate
    position offset

private theorem symbolProductTerms_allowed
    (candidateBase selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount) :
    CombAllowed candidateBase selectorBase count
      (symbolProductTerms selectorBase coordinate position) := by
  intro column mentioned
  unfold symbolProductTerms Mentions at mentioned
  rw [List.map_map] at mentioned
  change column ∈
    (List.finRange selectionWindow).map
      (symbolProductColumn selectorBase coordinate position) at mentioned
  rcases List.mem_map.mp mentioned with ⟨offset, _, rfl⟩
  exact symbolProduct_allowed candidateBase selectorBase coordinate
    position offset

private theorem centeredSymbolTerms_allowed
    (candidateBase selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount) :
    CombAllowed candidateBase selectorBase count
      (centeredSymbolTerms selectorBase coordinate position) := by
  intro column mentioned
  unfold centeredSymbolTerms Mentions at mentioned
  rw [List.map_append] at mentioned
  simp only [List.map_cons, List.map_nil, List.mem_append, List.mem_cons,
    List.not_mem_nil, or_false] at mentioned
  rcases mentioned with inProducts | constant
  · exact symbolProductTerms_allowed candidateBase selectorBase coordinate
      position column inProducts
  · subst column
    exact allowed_constant candidateBase selectorBase count

private theorem acceptProductTerms_allowed
    (candidateBase selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount) :
    CombAllowed candidateBase selectorBase count
      (acceptProductTerms selectorBase coordinate position) := by
  intro column mentioned
  unfold acceptProductTerms Mentions at mentioned
  rw [List.map_map] at mentioned
  change column ∈
    (List.finRange selectionWindow).map
      (acceptProductColumn selectorBase coordinate position) at mentioned
  rcases List.mem_map.mp mentioned with ⟨offset, _, rfl⟩
  exact acceptProduct_allowed candidateBase selectorBase coordinate
    position offset

private theorem prefixProductTerms_allowed
    (candidateBase selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount) :
    CombAllowed candidateBase selectorBase count
      (prefixProductTerms selectorBase coordinate position) := by
  intro column mentioned
  unfold prefixProductTerms Mentions at mentioned
  rw [List.map_map] at mentioned
  change column ∈
    (List.finRange selectionWindow).map
      (prefixProductColumn selectorBase coordinate position) at mentioned
  rcases List.mem_map.mp mentioned with ⟨offset, _, rfl⟩
  exact prefixProduct_allowed candidateBase selectorBase coordinate
    position offset

private theorem positionTerms_allowed
    (candidateBase selectorBase : Nat) {count : Nat}
    (position : Fin outputCount) :
    CombAllowed candidateBase selectorBase count (positionTerms position) := by
  unfold positionTerms
  split
  · exact combAllowed_nil candidateBase selectorBase count
  · exact combAllowed_single candidateBase selectorBase count 0 position.val
      (allowed_constant candidateBase selectorBase count)

private theorem bitRow_allowed
    (candidateBase selectorBase count column : Nat)
    (allowed : Allowed candidateBase selectorBase count column) :
    RowAllowed candidateBase selectorBase count (bitRow column) :=
  ⟨combAllowed_single candidateBase selectorBase count column 1 allowed,
    combAllowed_pair candidateBase selectorBase count
      column 1 0 (goldilocksP - 1) allowed
      (allowed_constant candidateBase selectorBase count),
    combAllowed_nil candidateBase selectorBase count⟩

private theorem acceptanceBoundRows_allowed
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (row : Row)
    (member :
      row ∈ acceptanceBoundRows duplexBase u64Base candidateBase selectorBase
        initial coordinate) :
    RowAllowed candidateBase selectorBase count row := by
  simp only [acceptanceBoundRows, List.mem_append] at member
  rcases member with inBits | inTail
  · rcases List.mem_map.mp inBits with ⟨offset, inRange, rfl⟩
    exact bitRow_allowed candidateBase selectorBase count _
      (slackBit_allowed candidateBase selectorBase coordinate offset
        (List.mem_range.mp inRange))
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at inTail
    rcases inTail with rfl | rfl
    · exact
        ⟨combAllowed_single candidateBase selectorBase count _ 1
            (slack_allowed candidateBase selectorBase coordinate),
          combAllowed_single candidateBase selectorBase count 0 1
            (allowed_constant candidateBase selectorBase count),
          slackTerms_allowed candidateBase selectorBase coordinate⟩
    · exact
        ⟨combAllowed_single candidateBase selectorBase count _ 1
            (finalCountSource_allowed duplexBase u64Base candidateBase
              selectorBase initial coordinate),
          combAllowed_single candidateBase selectorBase count 0 1
            (allowed_constant candidateBase selectorBase count),
          combAllowed_pair candidateBase selectorBase count _ 1 0 outputCount
            (slack_allowed candidateBase selectorBase coordinate)
            (allowed_constant candidateBase selectorBase count)⟩

private theorem oneHotRows_allowed
    (candidateBase selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount)
    (row : Row) (member : row ∈ oneHotRows selectorBase coordinate position) :
    RowAllowed candidateBase selectorBase count row := by
  simp only [oneHotRows, List.mem_append] at member
  rcases member with inBits | inSum
  · rcases List.mem_map.mp inBits with ⟨offset, _, rfl⟩
    exact bitRow_allowed candidateBase selectorBase count _
      (selectorColumn_allowed candidateBase selectorBase coordinate
        position offset)
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at inSum
    subst row
    exact
      ⟨selectorTerms_allowed candidateBase selectorBase coordinate position,
        combAllowed_single candidateBase selectorBase count 0 1
          (allowed_constant candidateBase selectorBase count),
        combAllowed_single candidateBase selectorBase count 0 1
          (allowed_constant candidateBase selectorBase count)⟩

private theorem productRowsAt_allowed
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount) (offset : Fin selectionWindow)
    (row : Row)
    (member :
      row ∈ productRowsAt duplexBase u64Base candidateBase selectorBase
        initial coordinate position offset) :
    RowAllowed candidateBase selectorBase count row := by
  simp only [productRowsAt, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact
      ⟨combAllowed_single candidateBase selectorBase count _ 1
          (selectorColumn_allowed candidateBase selectorBase coordinate
            position offset),
        combAllowed_single candidateBase selectorBase count _ 1
          (symbolSource_allowed duplexBase u64Base candidateBase selectorBase
            initial coordinate (candidateAt position offset)),
        combAllowed_single candidateBase selectorBase count _ 1
          (symbolProduct_allowed candidateBase selectorBase coordinate
            position offset)⟩
  · exact
      ⟨combAllowed_single candidateBase selectorBase count _ 1
          (selectorColumn_allowed candidateBase selectorBase coordinate
            position offset),
        combAllowed_single candidateBase selectorBase count _ 1
          (acceptSource_allowed duplexBase u64Base candidateBase selectorBase
            initial coordinate (candidateAt position offset)),
        combAllowed_single candidateBase selectorBase count _ 1
          (acceptProduct_allowed candidateBase selectorBase coordinate
            position offset)⟩
  · exact
      ⟨combAllowed_single candidateBase selectorBase count _ 1
          (selectorColumn_allowed candidateBase selectorBase coordinate
            position offset),
        prefixSource_allowed duplexBase u64Base candidateBase selectorBase
          initial coordinate (candidateAt position offset),
        combAllowed_single candidateBase selectorBase count _ 1
          (prefixProduct_allowed candidateBase selectorBase coordinate
            position offset)⟩

private theorem productRows_allowed
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount)
    (row : Row)
    (member :
      row ∈ productRows duplexBase u64Base candidateBase selectorBase
        initial coordinate position) :
    RowAllowed candidateBase selectorBase count row := by
  rcases List.mem_flatMap.mp member with ⟨offset, _, localMember⟩
  exact productRowsAt_allowed duplexBase u64Base candidateBase selectorBase
    initial coordinate position offset row localMember

private theorem bindingRows_allowed
    (candidateBase selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (position : Fin outputCount)
    (row : Row) (member : row ∈ bindingRows selectorBase coordinate position) :
    RowAllowed candidateBase selectorBase count row := by
  simp only [bindingRows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact
      ⟨acceptProductTerms_allowed candidateBase selectorBase coordinate
          position,
        combAllowed_single candidateBase selectorBase count 0 1
          (allowed_constant candidateBase selectorBase count),
        combAllowed_single candidateBase selectorBase count 0 1
          (allowed_constant candidateBase selectorBase count)⟩
  · exact
      ⟨prefixProductTerms_allowed candidateBase selectorBase coordinate
          position,
        combAllowed_single candidateBase selectorBase count 0 1
          (allowed_constant candidateBase selectorBase count),
        positionTerms_allowed candidateBase selectorBase position⟩
  · exact
      ⟨combAllowed_single candidateBase selectorBase count _ 1
          (output_allowed candidateBase selectorBase coordinate position),
        combAllowed_single candidateBase selectorBase count 0 1
          (allowed_constant candidateBase selectorBase count),
        centeredSymbolTerms_allowed candidateBase selectorBase coordinate
          position⟩

private theorem positionRows_allowed
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount)
    (row : Row)
    (member :
      row ∈ positionRows duplexBase u64Base candidateBase selectorBase
        initial coordinate position) :
    RowAllowed candidateBase selectorBase count row := by
  simp only [positionRows, List.mem_append] at member
  rcases member with (inOneHot | inProducts) | inBindings
  · exact oneHotRows_allowed candidateBase selectorBase coordinate position
      row inOneHot
  · exact productRows_allowed duplexBase u64Base candidateBase selectorBase
      initial coordinate position row inProducts
  · exact bindingRows_allowed candidateBase selectorBase coordinate position
      row inBindings

private theorem scalarRows_allowed
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (row : Row)
    (member :
      row ∈ scalarRows duplexBase u64Base candidateBase selectorBase
        initial coordinate) :
    RowAllowed candidateBase selectorBase count row := by
  simp only [scalarRows, List.mem_append] at member
  rcases member with inBound | inPositions
  · exact acceptanceBoundRows_allowed duplexBase u64Base candidateBase
      selectorBase initial coordinate row inBound
  · rcases List.mem_flatMap.mp inPositions with
      ⟨position, _, localMember⟩
    exact positionRows_allowed duplexBase u64Base candidateBase selectorBase
      initial coordinate position row localMember

/-- Every operand of every selector row belongs to one explicit column
class. -/
theorem rows_conservation
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    (row : Row)
    (rowMember :
      row ∈ rows duplexBase u64Base candidateBase selectorBase count initial)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    Allowed candidateBase selectorBase count column := by
  rcases List.mem_flatMap.mp rowMember with
    ⟨coordinate, _, localMember⟩
  have allowed :=
    scalarRows_allowed duplexBase u64Base candidateBase selectorBase initial
      coordinate row localMember
  exact mentioned.elim (allowed.1 column)
    (fun side => side.elim (allowed.2.1 column) (allowed.2.2 column))

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorConservation
