import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidatesBatchHonest

/-!
Contract: honest witness construction for the family-major batch of canonical
`Pi_RLC` first-accepted selectors.

Candidate values are caller-owned only in the physical sense: their exact
verifier values are carried by `SourcesMatch`, and a separate placement
contract keeps every such read before the selector allocation.  This module
owns the selector row frame, canonical witness values, and coordinate-major
threading for `PiRlcCanonicalSelector.rows`.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorBatchHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelector
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorHonest
open Nightstream.SuperNeo.Sampling
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-- Boundary after one complete selector coordinate. -/
def scalarEnd
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count) : Nat :=
  scalarBase selectorBase coordinate + scalarAuxiliaryCount

private def CombBelow (boundary : Nat) (comb : LinComb) : Prop :=
  ∀ column, Mentions comb column → column < boundary

private def RowBelow (boundary : Nat) (row : Row) : Prop :=
  CombBelow boundary row.a ∧
    CombBelow boundary row.b ∧
      CombBelow boundary row.c

private theorem combBelow_nil (boundary : Nat) :
    CombBelow boundary [] := by
  intro column mentioned
  simp [Mentions] at mentioned

private theorem combBelow_single
    (boundary column coefficient : Nat) (bounded : column < boundary) :
    CombBelow boundary [(column, coefficient)] := by
  intro target mentioned
  have equal := (mentions_single column target coefficient).mp mentioned
  subst target
  exact bounded

private theorem combBelow_append
    (boundary : Nat) (left right : LinComb)
    (leftBelow : CombBelow boundary left)
    (rightBelow : CombBelow boundary right) :
    CombBelow boundary (left ++ right) := by
  intro column mentioned
  rw [mentions_append] at mentioned
  exact mentioned.elim (leftBelow column) (rightBelow column)

private theorem combBelow_pair
    (boundary leftColumn leftCoefficient rightColumn rightCoefficient : Nat)
    (leftBound : leftColumn < boundary)
    (rightBound : rightColumn < boundary) :
    CombBelow boundary
      [(leftColumn, leftCoefficient), (rightColumn, rightCoefficient)] :=
  combBelow_append boundary
    [(leftColumn, leftCoefficient)] [(rightColumn, rightCoefficient)]
    (combBelow_single boundary leftColumn leftCoefficient leftBound)
    (combBelow_single boundary rightColumn rightCoefficient rightBound)

private theorem constant_below
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (positive : 0 < scalarBase selectorBase coordinate) :
    0 < scalarEnd selectorBase coordinate := by
  unfold scalarEnd
  omega

private theorem scalarBase_lt_end
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count) :
    scalarBase selectorBase coordinate < scalarEnd selectorBase coordinate := by
  simp only [scalarEnd, scalarAuxiliaryCount, outputCount,
    positionAuxiliaryCount]
  omega

private theorem slackColumn_below
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count) :
    slackColumn selectorBase coordinate < scalarEnd selectorBase coordinate := by
  simp only [slackColumn, scalarEnd, scalarAuxiliaryCount, outputCount,
    positionAuxiliaryCount]
  omega

private theorem slackBitColumn_below
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (offset : Nat) (bounded : offset < slackBitCount) :
    slackBitColumn selectorBase coordinate offset <
      scalarEnd selectorBase coordinate := by
  simp only [slackBitColumn, scalarEnd, scalarAuxiliaryCount, outputCount,
    positionAuxiliaryCount, slackBitCount] at bounded ⊢
  omega

private theorem selectorColumn_below
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount) (offset : Fin selectionWindow) :
    selectorColumn selectorBase coordinate position offset <
      scalarEnd selectorBase coordinate := by
  have positionLt := position.isLt
  have offsetLt := offset.isLt
  simp only [selectorColumn, positionBase, scalarEnd, scalarAuxiliaryCount,
    outputCount, positionAuxiliaryCount, selectionWindow] at positionLt offsetLt ⊢
  omega

private theorem symbolProductColumn_below
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount) (offset : Fin selectionWindow) :
    symbolProductColumn selectorBase coordinate position offset <
      scalarEnd selectorBase coordinate := by
  have positionLt := position.isLt
  have offsetLt := offset.isLt
  simp only [symbolProductColumn, positionBase, scalarEnd,
    scalarAuxiliaryCount, outputCount, positionAuxiliaryCount,
    selectionWindow] at positionLt offsetLt ⊢
  omega

private theorem acceptProductColumn_below
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount) (offset : Fin selectionWindow) :
    acceptProductColumn selectorBase coordinate position offset <
      scalarEnd selectorBase coordinate := by
  have bound :=
    symbolProductColumn_below selectorBase coordinate position offset
  have positionLt := position.isLt
  have offsetLt := offset.isLt
  simp only [acceptProductColumn, symbolProductColumn, positionBase,
    scalarEnd, scalarAuxiliaryCount, outputCount, positionAuxiliaryCount,
    selectionWindow] at bound positionLt offsetLt ⊢
  omega

private theorem prefixProductColumn_below
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount) (offset : Fin selectionWindow) :
    prefixProductColumn selectorBase coordinate position offset <
      scalarEnd selectorBase coordinate := by
  have positionLt := position.isLt
  have offsetLt := offset.isLt
  simp only [prefixProductColumn, symbolProductColumn, positionBase,
    scalarEnd, scalarAuxiliaryCount, outputCount, positionAuxiliaryCount,
    selectionWindow] at positionLt offsetLt ⊢
  omega

private theorem outputColumn_below
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount) :
    outputColumn selectorBase coordinate position <
      scalarEnd selectorBase coordinate := by
  have positionLt := position.isLt
  simp only [outputColumn, positionBase, scalarEnd, scalarAuxiliaryCount,
    outputCount, positionAuxiliaryCount] at positionLt ⊢
  omega

private theorem slackTerms_below
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count) :
    CombBelow (scalarEnd selectorBase coordinate)
      (slackTerms selectorBase coordinate) := by
  intro column mentioned
  unfold slackTerms Mentions at mentioned
  rw [List.map_map] at mentioned
  change
    column ∈
      (List.range slackBitCount).map
        (slackBitColumn selectorBase coordinate) at mentioned
  rcases List.mem_map.mp mentioned with ⟨offset, inRange, rfl⟩
  exact slackBitColumn_below selectorBase coordinate offset
    (List.mem_range.mp inRange)

private theorem selectorTerms_below
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount) :
    CombBelow (scalarEnd selectorBase coordinate)
      (selectorTerms selectorBase coordinate position) := by
  intro column mentioned
  unfold selectorTerms Mentions at mentioned
  rw [List.map_map] at mentioned
  change
    column ∈
      (List.finRange selectionWindow).map
        (selectorColumn selectorBase coordinate position) at mentioned
  rcases List.mem_map.mp mentioned with ⟨offset, _, rfl⟩
  exact selectorColumn_below selectorBase coordinate position offset

private theorem symbolProductTerms_below
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount) :
    CombBelow (scalarEnd selectorBase coordinate)
      (symbolProductTerms selectorBase coordinate position) := by
  intro column mentioned
  unfold symbolProductTerms Mentions at mentioned
  rw [List.map_map] at mentioned
  change
    column ∈
      (List.finRange selectionWindow).map
        (symbolProductColumn selectorBase coordinate position) at mentioned
  rcases List.mem_map.mp mentioned with ⟨offset, _, rfl⟩
  exact symbolProductColumn_below selectorBase coordinate position offset

private theorem centeredSymbolTerms_below
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount)
    (positive : 0 < scalarBase selectorBase coordinate) :
    CombBelow (scalarEnd selectorBase coordinate)
      (centeredSymbolTerms selectorBase coordinate position) := by
  intro column mentioned
  unfold centeredSymbolTerms Mentions at mentioned
  rw [List.map_append] at mentioned
  simp only [List.map_cons, List.map_nil, List.mem_append, List.mem_cons,
    List.not_mem_nil, or_false] at mentioned
  rcases mentioned with inProducts | constant
  · exact symbolProductTerms_below selectorBase coordinate position column
      inProducts
  · subst column
    exact constant_below selectorBase coordinate positive

private theorem acceptProductTerms_below
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount) :
    CombBelow (scalarEnd selectorBase coordinate)
      (acceptProductTerms selectorBase coordinate position) := by
  intro column mentioned
  unfold acceptProductTerms Mentions at mentioned
  rw [List.map_map] at mentioned
  change
    column ∈
      (List.finRange selectionWindow).map
        (acceptProductColumn selectorBase coordinate position) at mentioned
  rcases List.mem_map.mp mentioned with ⟨offset, _, rfl⟩
  exact acceptProductColumn_below selectorBase coordinate position offset

private theorem prefixProductTerms_below
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount) :
    CombBelow (scalarEnd selectorBase coordinate)
      (prefixProductTerms selectorBase coordinate position) := by
  intro column mentioned
  unfold prefixProductTerms Mentions at mentioned
  rw [List.map_map] at mentioned
  change
    column ∈
      (List.finRange selectionWindow).map
        (prefixProductColumn selectorBase coordinate position) at mentioned
  rcases List.mem_map.mp mentioned with ⟨offset, _, rfl⟩
  exact prefixProductColumn_below selectorBase coordinate position offset

private theorem positionTerms_below
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount)
    (positive : 0 < scalarBase selectorBase coordinate) :
    CombBelow (scalarEnd selectorBase coordinate) (positionTerms position) := by
  unfold positionTerms
  split
  · exact combBelow_nil _
  · exact combBelow_single _ 0 position.val
      (constant_below selectorBase coordinate positive)

private theorem bitRow_below
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (positive : 0 < scalarBase selectorBase coordinate)
    (column : Nat) (bounded : column < scalarEnd selectorBase coordinate) :
    RowBelow (scalarEnd selectorBase coordinate) (bitRow column) := by
  unfold RowBelow bitRow
  exact
    ⟨combBelow_single _ column 1 bounded,
      combBelow_pair _ column 1 0 (goldilocksP - 1)
        bounded (constant_below selectorBase coordinate positive),
      combBelow_nil _⟩

private theorem acceptanceBoundRows_below
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (below :
      SourcesBelow duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate)
    (positive : 0 < scalarBase selectorBase coordinate)
    (row : Row)
    (member :
      row ∈ acceptanceBoundRows duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate) :
    RowBelow (scalarEnd selectorBase coordinate) row := by
  simp only [acceptanceBoundRows, List.mem_append] at member
  rcases member with bitMember | tailMember
  · rcases List.mem_map.mp bitMember with ⟨offset, inRange, rfl⟩
    exact bitRow_below selectorBase coordinate positive _ <|
      slackBitColumn_below selectorBase coordinate offset
        (List.mem_range.mp inRange)
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at tailMember
    rcases tailMember with rfl | rfl
    · exact
        ⟨combBelow_single _ _ 1
            (slackColumn_below selectorBase coordinate),
          combBelow_single _ 0 1
            (constant_below selectorBase coordinate positive),
          slackTerms_below selectorBase coordinate⟩
    · exact
        ⟨combBelow_single _ _ 1
            (Nat.lt_trans below.finalCount
              (scalarBase_lt_end selectorBase coordinate)),
          combBelow_single _ 0 1
            (constant_below selectorBase coordinate positive),
          combBelow_pair _ _ 1 0 outputCount
            (slackColumn_below selectorBase coordinate)
            (constant_below selectorBase coordinate positive)⟩

private theorem oneHotRows_below
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (positive : 0 < scalarBase selectorBase coordinate)
    (position : Fin outputCount)
    (row : Row) (member : row ∈ oneHotRows selectorBase coordinate position) :
    RowBelow (scalarEnd selectorBase coordinate) row := by
  simp only [oneHotRows, List.mem_append] at member
  rcases member with bitMember | sumMember
  · rcases List.mem_map.mp bitMember with ⟨offset, _, rfl⟩
    exact bitRow_below selectorBase coordinate positive _
      (selectorColumn_below selectorBase coordinate position offset)
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at sumMember
    subst row
    exact
      ⟨selectorTerms_below selectorBase coordinate position,
        combBelow_single _ 0 1
          (constant_below selectorBase coordinate positive),
        combBelow_single _ 0 1
          (constant_below selectorBase coordinate positive)⟩

private theorem productRowsAt_below
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (below :
      SourcesBelow duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate)
    (position : Fin outputCount) (offset : Fin selectionWindow)
    (row : Row)
    (member :
      row ∈ productRowsAt duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate position offset) :
    RowBelow (scalarEnd selectorBase coordinate) row := by
  simp only [productRowsAt, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact
      ⟨combBelow_single _ _ 1
          (selectorColumn_below selectorBase coordinate position offset),
        combBelow_single _ _ 1
          (Nat.lt_trans (below.symbol (candidateAt position offset))
            (scalarBase_lt_end selectorBase coordinate)),
        combBelow_single _ _ 1
          (symbolProductColumn_below selectorBase coordinate position offset)⟩
  · exact
      ⟨combBelow_single _ _ 1
          (selectorColumn_below selectorBase coordinate position offset),
        combBelow_single _ _ 1
          (Nat.lt_trans (below.accept (candidateAt position offset))
            (scalarBase_lt_end selectorBase coordinate)),
        combBelow_single _ _ 1
          (acceptProductColumn_below selectorBase coordinate position offset)⟩
  · exact
      ⟨combBelow_single _ _ 1
          (selectorColumn_below selectorBase coordinate position offset),
        (fun column mentioned =>
          Nat.lt_trans
            (below.prefixRead (candidateAt position offset) column mentioned)
            (scalarBase_lt_end selectorBase coordinate)),
        combBelow_single _ _ 1
          (prefixProductColumn_below selectorBase coordinate position offset)⟩

private theorem productRows_below
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (below :
      SourcesBelow duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate)
    (position : Fin outputCount)
    (row : Row)
    (member :
      row ∈ productRows duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate position) :
    RowBelow (scalarEnd selectorBase coordinate) row := by
  rcases List.mem_flatMap.mp member with ⟨offset, _, localMember⟩
  exact productRowsAt_below duplexBase u64Base candidateBase selectorBase
    initialBuilder coordinate below position offset row localMember

private theorem bindingRows_below
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (positive : 0 < scalarBase selectorBase coordinate)
    (position : Fin outputCount)
    (row : Row) (member : row ∈ bindingRows selectorBase coordinate position) :
    RowBelow (scalarEnd selectorBase coordinate) row := by
  simp only [bindingRows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact
      ⟨acceptProductTerms_below selectorBase coordinate position,
        combBelow_single _ 0 1
          (constant_below selectorBase coordinate positive),
        combBelow_single _ 0 1
          (constant_below selectorBase coordinate positive)⟩
  · exact
      ⟨prefixProductTerms_below selectorBase coordinate position,
        combBelow_single _ 0 1
          (constant_below selectorBase coordinate positive),
        positionTerms_below selectorBase coordinate position positive⟩
  · exact
      ⟨combBelow_single _ _ 1
          (outputColumn_below selectorBase coordinate position),
        combBelow_single _ 0 1
          (constant_below selectorBase coordinate positive),
        centeredSymbolTerms_below selectorBase coordinate position positive⟩

private theorem positionRows_below
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (below :
      SourcesBelow duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate)
    (positive : 0 < scalarBase selectorBase coordinate)
    (position : Fin outputCount)
    (row : Row)
    (member :
      row ∈ positionRows duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate position) :
    RowBelow (scalarEnd selectorBase coordinate) row := by
  simp only [positionRows, List.mem_append] at member
  rcases member with (oneHot | products) | bindings
  · exact oneHotRows_below selectorBase coordinate positive position row oneHot
  · exact productRows_below duplexBase u64Base candidateBase selectorBase
      initialBuilder coordinate below position row products
  · exact bindingRows_below selectorBase coordinate positive position row
      bindings

/-- Every operand of one selector scalar lies before the next scalar block. -/
theorem scalarRows_mentions_lt
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (below :
      SourcesBelow duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate)
    (positive : 0 < scalarBase selectorBase coordinate)
    (row : Row)
    (member :
      row ∈ scalarRows duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨ Mentions row.c column) :
    column < scalarEnd selectorBase coordinate := by
  have rowBelow : RowBelow (scalarEnd selectorBase coordinate) row := by
    simp only [scalarRows, List.mem_append] at member
    rcases member with acceptance | positions
    · exact acceptanceBoundRows_below duplexBase u64Base candidateBase
        selectorBase initialBuilder coordinate below positive row acceptance
    · rcases List.mem_flatMap.mp positions with ⟨position, _, localMember⟩
      exact positionRows_below duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate below positive position row localMember
  rcases rowBelow with ⟨aBelow, bBelow, cBelow⟩
  exact mentioned.elim (aBelow column)
    (fun right => right.elim (bBelow column) (cBelow column))

/-! ## Canonical selector witness values -/

theorem scalarLocalValue_lt
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (localOffset : Nat) :
    scalarLocalValue duplexBase u64Base candidateBase initialBuilder coordinate
      assignment candidates lengthExact enough localOffset < goldilocksP := by
  unfold scalarLocalValue
  split
  · have bound := slackValue_le_ten candidates lengthExact
    simp only [goldilocksP]
    omega
  · split
    · have bound := slackBit_le_one candidates (localOffset - 1)
      simp only [goldilocksP]
      omega
    · dsimp only
      split
      · unfold selectorValue
        split <;> simp only [goldilocksP] <;> omega
      · split
        · exact Nat.mod_lt _ (by decide)
        · split <;>
            exact Nat.mod_lt _ (by decide)

theorem scalarWitness_canonical
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates) :
    ∀ column,
      scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
        coordinate assignment candidates lengthExact enough column <
        goldilocksP := by
  intro column
  unfold scalarWitness
  split
  · exact canonical _
  · split
    · exact scalarLocalValue_lt duplexBase u64Base candidateBase
        initialBuilder coordinate assignment canonical candidates lengthExact
        enough _
    · exact canonical _

/-! ## Candidate/selector placement -/

/-- First column after the complete family-major candidate allocation. -/
def candidateEnd (candidateBase count : Nat) : Nat :=
  candidateBase +
    count * PiRlcCanonicalCandidates.candidatesPerScalar *
      PiRlcCanonicalCandidate.auxiliaryCount

/-- Every candidate value read by any selector coordinate lies before the
shared selector allocation.  Unlike `SourcesBelow`, this boundary is
coordinate-independent and can therefore preserve one coordinate's sources
while another coordinate's selector witness is installed. -/
structure SourcesBeforeSelector
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count) : Prop where
  accept :
    ∀ candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar,
      acceptSource duplexBase u64Base candidateBase initialBuilder coordinate
          candidate <
        selectorBase
  symbol :
    ∀ candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar,
      symbolSource duplexBase u64Base candidateBase initialBuilder coordinate
          candidate <
        selectorBase
  prefixRead :
    ∀ (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar)
      column,
      Mentions
          (prefixSource duplexBase u64Base candidateBase initialBuilder
            coordinate candidate)
          column →
        column < selectorBase
  finalCount :
    finalCountSource duplexBase u64Base candidateBase initialBuilder
        coordinate <
      selectorBase

/-- One exact allocation-separation inequality constructs the global source
boundary for every selector coordinate. -/
theorem sourcesBeforeSelector_of_candidateEnd
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (separated : candidateEnd candidateBase count ≤ selectorBase) :
    ∀ coordinate : Fin count,
      SourcesBeforeSelector duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate := by
  intro coordinate
  constructor
  · intro candidate
    have coordinateLt := coordinate.isLt
    have candidateLt := candidate.isLt
    simp only [acceptSource, candidateSourceLayout,
      PiRlcCanonicalCandidate.acceptColumn,
      PiRlcCanonicalCandidates.candidateLayout,
      PiRlcCanonicalCandidates.occurrenceBase,
      PiRlcCanonicalCandidates.occurrenceIndex, candidateEnd,
      PiRlcCanonicalCandidates.candidatesPerScalar,
      PiRlcCanonicalCandidate.auxiliaryCount] at separated candidateLt ⊢
    omega
  · intro candidate
    have coordinateLt := coordinate.isLt
    have candidateLt := candidate.isLt
    simp only [symbolSource, candidateSourceLayout,
      PiRlcCanonicalCandidate.residueColumn,
      PiRlcCanonicalCandidates.candidateLayout,
      PiRlcCanonicalCandidates.occurrenceBase,
      PiRlcCanonicalCandidates.occurrenceIndex, candidateEnd,
      PiRlcCanonicalCandidates.candidatesPerScalar,
      PiRlcCanonicalCandidate.auxiliaryCount] at separated candidateLt ⊢
    omega
  · intro candidate column mentioned
    simp only [prefixSource, candidateSourceLayout,
      PiRlcCanonicalCandidates.candidateLayout] at mentioned
    unfold PiRlcCanonicalCandidates.prior at mentioned
    split at mentioned
    · simp only [Mentions, List.map_nil, List.not_mem_nil] at mentioned
    · simp only [Mentions, List.map_cons, List.map_nil, List.mem_cons,
        List.not_mem_nil, or_false, Prod.mk.injEq] at mentioned
      rcases mentioned with ⟨rfl, rfl⟩
      have coordinateLt := coordinate.isLt
      have candidateLt := candidate.isLt
      simp only [PiRlcCanonicalCandidates.occurrenceBase,
        PiRlcCanonicalCandidates.occurrenceIndex, candidateEnd,
        PiRlcCanonicalCandidates.candidatesPerScalar,
        PiRlcCanonicalCandidate.auxiliaryCount]
        at separated candidateLt ⊢
      omega
  · have coordinateLt := coordinate.isLt
    simp only [finalCountSource, candidateSourceLayout,
      PiRlcCanonicalCandidate.cumulativeColumn,
      PiRlcCanonicalCandidates.candidateLayout,
      PiRlcCanonicalCandidates.occurrenceBase,
      PiRlcCanonicalCandidates.occurrenceIndex, candidateEnd,
      PiRlcCanonicalCandidates.candidatesPerScalar,
      PiRlcCanonicalCandidate.auxiliaryCount] at separated ⊢
    omega

/-- A global source boundary specializes to the local freshness contract of
any selector scalar. -/
theorem SourcesBeforeSelector.toSourcesBelow
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (before :
      SourcesBeforeSelector duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate) :
    SourcesBelow duplexBase u64Base candidateBase selectorBase initialBuilder
      coordinate where
  accept candidate := by
    exact Nat.lt_of_lt_of_le (before.accept candidate) (by
      unfold scalarBase
      omega)
  symbol candidate := by
    exact Nat.lt_of_lt_of_le (before.symbol candidate) (by
      unfold scalarBase
      omega)
  prefixRead candidate column mentioned := by
    exact Nat.lt_of_lt_of_le
      (before.prefixRead candidate column mentioned) (by
        unfold scalarBase
        omega)
  finalCount := by
    exact Nat.lt_of_lt_of_le before.finalCount (by
      unfold scalarBase
      omega)

/-! ## Coordinate-major selector witness threading -/

/-- Boundary after the first `processed` selector coordinates. -/
def batchPrefixBoundary (selectorBase processed : Nat) : Nat :=
  selectorBase + processed * scalarAuxiliaryCount

/-- Apply complete selector witnesses to the first `processed` coordinates. -/
def batchPrefixWitness
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (candidates : ∀ coordinate : Fin count, List ProductionAlphabet.Chunk)
    (sourceMatches :
      ∀ coordinate : Fin count,
        SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
          initial (candidates coordinate))
    (enough :
      ∀ coordinate : Fin count,
        FirstAccepted.Enough ProductionAlphabet.verifier outputCount
          (candidates coordinate)) :
    Nat → Nat → Nat
  | 0 => initial
  | processed + 1 =>
      if bounded : processed < count then
        scalarWitness duplexBase u64Base candidateBase selectorBase
          initialBuilder ⟨processed, bounded⟩
          (batchPrefixWitness duplexBase u64Base candidateBase selectorBase
            count initialBuilder initial candidates sourceMatches enough
            processed)
          (candidates ⟨processed, bounded⟩)
          (sourceMatches ⟨processed, bounded⟩).lengthExact
          (enough ⟨processed, bounded⟩)
      else
        batchPrefixWitness duplexBase u64Base candidateBase selectorBase count
          initialBuilder initial candidates sourceMatches enough processed

theorem batchPrefixWitness_succ
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (candidates : ∀ coordinate : Fin count, List ProductionAlphabet.Chunk)
    (sourceMatches :
      ∀ coordinate : Fin count,
        SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
          initial (candidates coordinate))
    (enough :
      ∀ coordinate : Fin count,
        FirstAccepted.Enough ProductionAlphabet.verifier outputCount
          (candidates coordinate))
    {processed : Nat} (bounded : processed < count) :
    batchPrefixWitness duplexBase u64Base candidateBase selectorBase count
        initialBuilder initial candidates sourceMatches enough
        (processed + 1) =
      scalarWitness duplexBase u64Base candidateBase selectorBase
        initialBuilder ⟨processed, bounded⟩
        (batchPrefixWitness duplexBase u64Base candidateBase selectorBase count
          initialBuilder initial candidates sourceMatches enough processed)
        (candidates ⟨processed, bounded⟩)
        (sourceMatches ⟨processed, bounded⟩).lengthExact
        (enough ⟨processed, bounded⟩) := by
  simp [batchPrefixWitness, bounded]

/-- Every installed selector block preserves the complete candidate
allocation. -/
theorem batchPrefixWitness_before_selectorBase
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (candidates : ∀ coordinate : Fin count, List ProductionAlphabet.Chunk)
    (sourceMatches :
      ∀ coordinate : Fin count,
        SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
          initial (candidates coordinate))
    (enough :
      ∀ coordinate : Fin count,
        FirstAccepted.Enough ProductionAlphabet.verifier outputCount
          (candidates coordinate))
    (processed : Nat)
    {column : Nat} (before : column < selectorBase) :
    batchPrefixWitness duplexBase u64Base candidateBase selectorBase count
        initialBuilder initial candidates sourceMatches enough processed
        column =
      initial column := by
  induction processed with
  | zero => rfl
  | succ processed hypothesis =>
      simp only [batchPrefixWitness]
      split
      next bounded =>
        rw [scalarWitness_before duplexBase u64Base candidateBase selectorBase
          initialBuilder ⟨processed, bounded⟩
          (batchPrefixWitness duplexBase u64Base candidateBase selectorBase
            count initialBuilder initial candidates sourceMatches enough
            processed)
          (candidates ⟨processed, bounded⟩)
          (sourceMatches ⟨processed, bounded⟩).lengthExact
          (enough ⟨processed, bounded⟩) (by
            unfold scalarBase
            omega)]
        exact hypothesis
      next =>
        exact hypothesis

/-- The exact candidate-value contract survives every preceding selector
block. -/
theorem batchPrefixSourcesMatch
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (candidates : ∀ coordinate : Fin count, List ProductionAlphabet.Chunk)
    (sourceMatches :
      ∀ coordinate : Fin count,
        SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
          initial (candidates coordinate))
    (enough :
      ∀ coordinate : Fin count,
        FirstAccepted.Enough ProductionAlphabet.verifier outputCount
          (candidates coordinate))
    (sourcesBefore :
      ∀ coordinate : Fin count,
        SourcesBeforeSelector duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate)
    (processed : Nat) (coordinate : Fin count) :
    SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
      (batchPrefixWitness duplexBase u64Base candidateBase selectorBase count
        initialBuilder initial candidates sourceMatches enough processed)
      (candidates coordinate) := by
  let sources := sourceMatches coordinate
  refine
    { lengthExact := sources.lengthExact
      accept := ?_
      symbol := ?_
      prefixExact := ?_
      finalCount := ?_ }
  · intro candidate
    rw [batchPrefixWitness_before_selectorBase duplexBase u64Base
      candidateBase selectorBase count initialBuilder initial candidates
      sourceMatches enough processed
      ((sourcesBefore coordinate).accept candidate)]
    exact sources.accept candidate
  · intro candidate
    rw [batchPrefixWitness_before_selectorBase duplexBase u64Base
      candidateBase selectorBase count initialBuilder initial candidates
      sourceMatches enough processed
      ((sourcesBefore coordinate).symbol candidate)]
    exact sources.symbol candidate
  · intro candidate
    rw [KMulHonest.lcEval_congr
      (batchPrefixWitness duplexBase u64Base candidateBase selectorBase count
        initialBuilder initial candidates sourceMatches enough processed)
      initial
      (prefixSource duplexBase u64Base candidateBase initialBuilder coordinate
        candidate)]
    · exact sources.prefixExact candidate
    · intro column mentioned
      exact batchPrefixWitness_before_selectorBase duplexBase u64Base
        candidateBase selectorBase count initialBuilder initial candidates
        sourceMatches enough processed
        ((sourcesBefore coordinate).prefixRead candidate column mentioned)
  · rw [batchPrefixWitness_before_selectorBase duplexBase u64Base
      candidateBase selectorBase count initialBuilder initial candidates
      sourceMatches enough processed (sourcesBefore coordinate).finalCount]
    exact sources.finalCount

theorem batchPrefixWitness_canonical
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (initialCanonical : ∀ column, initial column < goldilocksP)
    (candidates : ∀ coordinate : Fin count, List ProductionAlphabet.Chunk)
    (sourceMatches :
      ∀ coordinate : Fin count,
        SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
          initial (candidates coordinate))
    (enough :
      ∀ coordinate : Fin count,
        FirstAccepted.Enough ProductionAlphabet.verifier outputCount
          (candidates coordinate)) :
    ∀ processed column,
      batchPrefixWitness duplexBase u64Base candidateBase selectorBase count
        initialBuilder initial candidates sourceMatches enough processed
        column <
      goldilocksP := by
  intro processed
  induction processed with
  | zero => exact initialCanonical
  | succ processed hypothesis =>
      simp only [batchPrefixWitness]
      split
      next bounded =>
        exact scalarWitness_canonical duplexBase u64Base candidateBase
          selectorBase initialBuilder ⟨processed, bounded⟩
          (batchPrefixWitness duplexBase u64Base candidateBase selectorBase
            count initialBuilder initial candidates sourceMatches enough
            processed)
          hypothesis (candidates ⟨processed, bounded⟩)
          (sourceMatches ⟨processed, bounded⟩).lengthExact
          (enough ⟨processed, bounded⟩)
      next =>
        exact hypothesis

theorem batchPrefixWitness_stable
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (candidates : ∀ coordinate : Fin count, List ProductionAlphabet.Chunk)
    (sourceMatches :
      ∀ coordinate : Fin count,
        SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
          initial (candidates coordinate))
    (enough :
      ∀ coordinate : Fin count,
        FirstAccepted.Enough ProductionAlphabet.verifier outputCount
          (candidates coordinate))
    {start finish : Nat} (ordered : start ≤ finish)
    (finishBounded : finish ≤ count)
    {column : Nat}
    (before : column < batchPrefixBoundary selectorBase start) :
    batchPrefixWitness duplexBase u64Base candidateBase selectorBase count
        initialBuilder initial candidates sourceMatches enough finish column =
      batchPrefixWitness duplexBase u64Base candidateBase selectorBase count
        initialBuilder initial candidates sourceMatches enough start column := by
  induction finish generalizing start with
  | zero =>
      have startZero : start = 0 := by omega
      subst start
      rfl
  | succ finish hypothesis =>
      by_cases atEnd : start = finish + 1
      · subst start
        rfl
      · have startLe : start ≤ finish := by omega
        have finishLt : finish < count := by omega
        rw [batchPrefixWitness_succ duplexBase u64Base candidateBase
          selectorBase count initialBuilder initial candidates sourceMatches
          enough finishLt]
        rw [scalarWitness_before duplexBase u64Base candidateBase selectorBase
          initialBuilder ⟨finish, finishLt⟩
          (batchPrefixWitness duplexBase u64Base candidateBase selectorBase
            count initialBuilder initial candidates sourceMatches enough finish)
          (candidates ⟨finish, finishLt⟩)
          (sourceMatches ⟨finish, finishLt⟩).lengthExact
          (enough ⟨finish, finishLt⟩) (by
            simp only [scalarBase, batchPrefixBoundary] at before ⊢
            exact Nat.lt_of_lt_of_le before
              (Nat.add_le_add_left
                (Nat.mul_le_mul_right scalarAuxiliaryCount startLe)
                selectorBase))]
        exact hypothesis startLe (by omega) before

/-- Final family-major selector witness. -/
def batchWitness
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (candidates : ∀ coordinate : Fin count, List ProductionAlphabet.Chunk)
    (sourceMatches :
      ∀ coordinate : Fin count,
        SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
          initial (candidates coordinate))
    (enough :
      ∀ coordinate : Fin count,
        FirstAccepted.Enough ProductionAlphabet.verifier outputCount
          (candidates coordinate)) :
    Nat → Nat :=
  batchPrefixWitness duplexBase u64Base candidateBase selectorBase count
    initialBuilder initial candidates sourceMatches enough count

private theorem rowHolds_congr
    (left right : Nat → Nat) (row : Row)
    (agree :
      ∀ column,
        Mentions row.a column ∨ Mentions row.b column ∨
          Mentions row.c column →
        left column = right column) :
    RowHolds left row ↔ RowHolds right row := by
  unfold RowHolds
  rw [KMulHonest.lcEval_congr left right row.a
      (fun column member => agree column (Or.inl member)),
    KMulHonest.lcEval_congr left right row.b
      (fun column member => agree column (Or.inr (Or.inl member))),
    KMulHonest.lcEval_congr left right row.c
      (fun column member => agree column (Or.inr (Or.inr member)))]

theorem batchStage_complete
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (initialCanonical : ∀ column, initial column < goldilocksP)
    (constantWire : initial 0 = 1)
    (positive : 0 < selectorBase)
    (candidates : ∀ coordinate : Fin count, List ProductionAlphabet.Chunk)
    (sourceMatches :
      ∀ coordinate : Fin count,
        SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
          initial (candidates coordinate))
    (enough :
      ∀ coordinate : Fin count,
        FirstAccepted.Enough ProductionAlphabet.verifier outputCount
          (candidates coordinate))
    (sourcesBefore :
      ∀ coordinate : Fin count,
        SourcesBeforeSelector duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate)
    {processed : Nat} (processedLt : processed < count) :
    Satisfies
      (scalarRows duplexBase u64Base candidateBase selectorBase initialBuilder
        ⟨processed, processedLt⟩)
      (batchPrefixWitness duplexBase u64Base candidateBase selectorBase count
        initialBuilder initial candidates sourceMatches enough
        (processed + 1)) := by
  let coordinate : Fin count := ⟨processed, processedLt⟩
  let prior :=
    batchPrefixWitness duplexBase u64Base candidateBase selectorBase count
      initialBuilder initial candidates sourceMatches enough processed
  have priorCanonical : ∀ column, prior column < goldilocksP :=
    batchPrefixWitness_canonical duplexBase u64Base candidateBase selectorBase
      count initialBuilder initial initialCanonical candidates sourceMatches
      enough processed
  have priorConstant : prior 0 = 1 := by
    change
      batchPrefixWitness duplexBase u64Base candidateBase selectorBase count
        initialBuilder initial candidates sourceMatches enough processed 0 = 1
    rw [batchPrefixWitness_before_selectorBase duplexBase u64Base
      candidateBase selectorBase count initialBuilder initial candidates
      sourceMatches enough processed positive]
    exact constantWire
  have priorSources :
      SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
        prior (candidates coordinate) :=
    batchPrefixSourcesMatch duplexBase u64Base candidateBase selectorBase count
      initialBuilder initial candidates sourceMatches enough sourcesBefore
      processed coordinate
  rw [batchPrefixWitness_succ duplexBase u64Base candidateBase selectorBase
    count initialBuilder initial candidates sourceMatches enough processedLt]
  apply PiRlcCanonicalSelectorComplete.scalarRows_complete duplexBase u64Base
    candidateBase selectorBase initialBuilder coordinate prior priorCanonical
    priorConstant (candidates coordinate) priorSources (enough coordinate)
    (sourcesBefore coordinate).toSourcesBelow
  unfold scalarBase
  exact Nat.lt_of_lt_of_le positive (Nat.le_add_right selectorBase _)

/-- Honest completeness of the exact family-major selector batch. -/
theorem rows_complete
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    (initial : Nat → Nat)
    (initialCanonical : ∀ column, initial column < goldilocksP)
    (constantWire : initial 0 = 1)
    (positive : 0 < selectorBase)
    (candidates : ∀ coordinate : Fin count, List ProductionAlphabet.Chunk)
    (sourceMatches :
      ∀ coordinate : Fin count,
        SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
          initial (candidates coordinate))
    (enough :
      ∀ coordinate : Fin count,
        FirstAccepted.Enough ProductionAlphabet.verifier outputCount
          (candidates coordinate))
    (sourcesBefore :
      ∀ coordinate : Fin count,
        SourcesBeforeSelector duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate) :
    Satisfies
      (rows duplexBase u64Base candidateBase selectorBase count initialBuilder)
      (batchWitness duplexBase u64Base candidateBase selectorBase count
        initialBuilder initial candidates sourceMatches enough) := by
  intro row member
  rcases List.mem_flatMap.mp member with
    ⟨coordinate, _, rowMember⟩
  have stage :=
    batchStage_complete duplexBase u64Base candidateBase selectorBase count
      initialBuilder initial initialCanonical constantWire positive candidates
      sourceMatches enough sourcesBefore coordinate.isLt
  have stageHolds : RowHolds
      (batchPrefixWitness duplexBase u64Base candidateBase selectorBase count
        initialBuilder initial candidates sourceMatches enough
        (coordinate.val + 1))
      row :=
    stage row (by simpa using rowMember)
  apply
    (rowHolds_congr
      (batchPrefixWitness duplexBase u64Base candidateBase selectorBase count
        initialBuilder initial candidates sourceMatches enough
        (coordinate.val + 1))
      (batchWitness duplexBase u64Base candidateBase selectorBase count
        initialBuilder initial candidates sourceMatches enough)
      row ?_).mp
  · exact stageHolds
  · intro column mentioned
    symm
    apply batchPrefixWitness_stable duplexBase u64Base candidateBase
      selectorBase count initialBuilder initial candidates sourceMatches enough
      (start := coordinate.val + 1) (finish := count)
    · omega
    · exact Nat.le_refl _
    · have localBound :=
        scalarRows_mentions_lt duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate
          (sourcesBefore coordinate).toSourcesBelow
          (by
            unfold scalarBase
            exact Nat.lt_of_lt_of_le positive
              (Nat.le_add_right selectorBase _))
          row rowMember column mentioned
      simp only [scalarEnd, scalarBase, batchPrefixBoundary] at localBound ⊢
      simpa [Nat.add_mul, Nat.add_assoc] using localBound

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorBatchHonest
