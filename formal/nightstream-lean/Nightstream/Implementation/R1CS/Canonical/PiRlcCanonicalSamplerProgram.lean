import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachineHonest
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerConservation
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalU64Placement

/-!
Contract: the complete Lean-owned fixed-active `Pi_RLC` sampler row program.

Owns the exact concatenation of the 135-call transcript and the 15-coordinate
sampler suffix, their physical placement, receipt fold, exact cost, and one
honest satisfying assignment.

Does not own the surrounding NIFS verifier, Fiat–Shamir security, or Rust
conformance.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPlacement
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerHonest
open Nightstream.SuperNeo.Sampling
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

def coordinateCount : Nat := 15
def transcriptCalls : Nat := coordinateCount * 9

def u64Base (duplexBase : Nat) : Nat :=
  duplexBase + transcriptCalls * SymbolicDuplex.stride

def candidateBase (duplexBase : Nat) : Nat :=
  PiRlcCanonicalCandidatesBatchHonest.u64End
    (u64Base duplexBase) coordinateCount

def selectorBase (duplexBase : Nat) : Nat :=
  PiRlcCanonicalSelectorBatchHonest.candidateEnd
    (candidateBase duplexBase) coordinateCount

def transcriptRows
    (duplexBase : Nat) (constants : Constants) (lanes : State) : List Row :=
  SymbolicDuplex.rows duplexBase constants
    (PiRlcCanonicalSymbolicMachineHonest.fixedBuilder duplexBase lanes)

def suffixRows
    (duplexBase : Nat) (lanes : State) : List Row :=
  PiRlcCanonicalSamplerHonest.suffixRows
    duplexBase (u64Base duplexBase) (candidateBase duplexBase)
    (selectorBase duplexBase) coordinateCount
    (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)

def rows
    (duplexBase : Nat) (constants : Constants) (lanes : State) : List Row :=
  transcriptRows duplexBase constants lanes ++ suffixRows duplexBase lanes

theorem transcriptRows_satisfied
    (duplexBase : Nat) (constants : Constants) (lanes : State)
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows duplexBase constants lanes) assignment) :
    Satisfies (transcriptRows duplexBase constants lanes) assignment :=
  fun row member => satisfied row (List.mem_append_left _ member)

theorem suffixRows_satisfied
    (duplexBase : Nat) (constants : Constants) (lanes : State)
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows duplexBase constants lanes) assignment) :
    Satisfies (suffixRows duplexBase lanes) assignment :=
  fun row member => satisfied row (List.mem_append_right _ member)

theorem u64Rows_satisfied
    (duplexBase : Nat) (constants : Constants) (lanes : State)
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows duplexBase constants lanes) assignment) :
    Satisfies
      (PiRlcCanonicalU64.rows duplexBase (u64Base duplexBase)
        coordinateCount
        (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes))
      assignment := by
  intro row member
  apply suffixRows_satisfied duplexBase constants lanes assignment satisfied
  unfold suffixRows PiRlcCanonicalSamplerHonest.suffixRows
  exact List.mem_append_left _ (List.mem_append_left _ member)

theorem candidateRows_satisfied
    (duplexBase : Nat) (constants : Constants) (lanes : State)
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows duplexBase constants lanes) assignment) :
    Satisfies
      (PiRlcCanonicalCandidates.rows duplexBase (u64Base duplexBase)
        (candidateBase duplexBase) coordinateCount
        (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes))
      assignment := by
  intro row member
  apply suffixRows_satisfied duplexBase constants lanes assignment satisfied
  unfold suffixRows PiRlcCanonicalSamplerHonest.suffixRows
  exact List.mem_append_left _ (List.mem_append_right _ member)

theorem selectorRows_satisfied
    (duplexBase : Nat) (constants : Constants) (lanes : State)
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows duplexBase constants lanes) assignment) :
    Satisfies
      (PiRlcCanonicalSelector.rows duplexBase (u64Base duplexBase)
        (candidateBase duplexBase) (selectorBase duplexBase) coordinateCount
        (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes))
      assignment := by
  intro row member
  apply suffixRows_satisfied duplexBase constants lanes assignment satisfied
  unfold suffixRows PiRlcCanonicalSamplerHonest.suffixRows
  exact List.mem_append_right _ member

def transcriptAllocation (duplexBase : Nat) : List Nat :=
  PiRlcCanonicalSymbolicMachineHonest.fixedAllocation duplexBase

def suffixAllocation (duplexBase : Nat) : List Nat :=
  PiRlcCanonicalSamplerHonest.suffixAllocation
    (u64Base duplexBase) (candidateBase duplexBase)
    (selectorBase duplexBase) coordinateCount

def allocation (duplexBase : Nat) : List Nat :=
  transcriptAllocation duplexBase ++ suffixAllocation duplexBase

def cost : Nightstream.Implementation.Lowering.Typed.Cost :=
  ⟨transcriptCalls * SymbolicDuplex.stride +
      (PiRlcCanonicalSamplerHonest.suffixCost coordinateCount).recurringRows,
    0,
    0,
    transcriptCalls * SymbolicDuplex.stride +
      (PiRlcCanonicalSamplerHonest.suffixCost coordinateCount).auxiliaryColumns⟩

theorem cost_recurringRows : cost.recurringRows = 143610 := by
  rfl

theorem cost_auxiliaryColumns : cost.auxiliaryColumns = 136845 := by
  rfl

theorem rows_length
    (duplexBase : Nat) (constants : Constants) (lanes : State) :
    (rows duplexBase constants lanes).length = cost.recurringRows := by
  unfold rows transcriptRows suffixRows coordinateCount
  rw [List.length_append,
    PiRlcCanonicalSymbolicMachineHonest.fixedRows_length,
    PiRlcCanonicalSamplerHonest.fixedActive_suffixRows_length,
    cost_recurringRows]

theorem allocation_length (duplexBase : Nat) :
    (allocation duplexBase).length = cost.auxiliaryColumns := by
  unfold allocation transcriptAllocation suffixAllocation coordinateCount
  rw [List.length_append,
    PiRlcCanonicalSymbolicMachineHonest.fixedAllocation_length,
    PiRlcCanonicalSamplerHonest.fixedActive_suffixAllocation_length,
    cost_auxiliaryColumns]

theorem transcriptAllocation_eq (duplexBase : Nat) :
    transcriptAllocation duplexBase =
      SymbolicDuplexPhysical.temporaryColumns duplexBase transcriptCalls := by
  rfl

theorem u64_separated (duplexBase : Nat) :
    PiRlcCanonicalCandidatesBatchHonest.u64End
        (u64Base duplexBase) coordinateCount ≤
      candidateBase duplexBase := by
  exact Nat.le_refl _

theorem candidate_separated (duplexBase : Nat) :
    PiRlcCanonicalSelectorBatchHonest.candidateEnd
        (candidateBase duplexBase) coordinateCount ≤
      selectorBase duplexBase := by
  exact Nat.le_refl _

private theorem suffixAllocation_ge
    (duplexBase column : Nat)
    (member : column ∈ suffixAllocation duplexBase) :
    u64Base duplexBase ≤ column := by
  unfold suffixAllocation PiRlcCanonicalSamplerHonest.suffixAllocation at member
  simp only [List.mem_append] at member
  rcases member with (inU64 | inCandidates) | inSelectors
  · exact
      (PiRlcCanonicalU64.allocation_mem_iff
        (u64Base duplexBase) coordinateCount column).mp inU64 |>.1
  · have lower :=
      (PiRlcCanonicalCandidates.allocation_mem_iff
        (candidateBase duplexBase) coordinateCount column).mp inCandidates |>.1
    exact Nat.le_trans
      (Nat.le_trans
        (Nat.le_add_right (u64Base duplexBase) _)
        (u64_separated duplexBase))
      lower
  · have lower :=
      (PiRlcCanonicalSelector.allocation_mem_iff
        (selectorBase duplexBase) coordinateCount column).mp inSelectors |>.1
    exact Nat.le_trans
      (Nat.le_trans
        (Nat.le_trans
          (Nat.le_add_right (u64Base duplexBase) _)
          (u64_separated duplexBase))
        (Nat.le_trans
          (Nat.le_add_right (candidateBase duplexBase) _)
          (candidate_separated duplexBase)))
      lower

theorem allocation_nodup (duplexBase : Nat) :
    (allocation duplexBase).Nodup := by
  unfold allocation
  rw [List.nodup_append]
  refine
    ⟨PiRlcCanonicalSymbolicMachineHonest.fixedAllocation_nodup duplexBase,
      PiRlcCanonicalSamplerHonest.suffixAllocation_nodup
        (u64Base duplexBase) (candidateBase duplexBase)
        (selectorBase duplexBase) coordinateCount
        (u64_separated duplexBase) (candidate_separated duplexBase),
      ?_⟩
  intro transcriptColumn transcriptMember suffixColumn suffixMember equal
  subst suffixColumn
  rw [transcriptAllocation_eq] at transcriptMember
  have below :=
    SymbolicDuplexPhysical.temporaryColumns_lt_end
      duplexBase transcriptCalls transcriptColumn
      transcriptMember
  have above := suffixAllocation_ge duplexBase transcriptColumn suffixMember
  have notAbove : ¬ u64Base duplexBase ≤ transcriptColumn := by
    exact Nat.not_le_of_gt below
  exact notAbove above

theorem allocation_ge
    (duplexBase column : Nat)
    (member : column ∈ allocation duplexBase) :
    duplexBase ≤ column := by
  rcases List.mem_append.mp member with inTranscript | inSuffix
  · unfold transcriptAllocation
      PiRlcCanonicalSymbolicMachineHonest.fixedAllocation
      SymbolicDuplexPhysical.temporaryColumns at inTranscript
    rcases List.mem_ofFn.mp inTranscript with ⟨position, rfl⟩
    unfold SymbolicDuplexPhysical.temporaryColumn
    omega
  · exact Nat.le_trans
      (Nat.le_add_right duplexBase _)
      (suffixAllocation_ge duplexBase column inSuffix)

/-- Every allocated sampler column lies below the exact contiguous allocation
end.  This is a value bound, not an inference from the allocation's length. -/
theorem allocation_lt_end
    (duplexBase column : Nat)
    (member : column ∈ allocation duplexBase) :
    column < duplexBase + cost.auxiliaryColumns := by
  rw [cost_auxiliaryColumns]
  rcases List.mem_append.mp member with inTranscript | inSuffix
  · rw [transcriptAllocation_eq] at inTranscript
    have below :=
      SymbolicDuplexPhysical.temporaryColumns_lt_end
        duplexBase transcriptCalls column
        inTranscript
    simp only [transcriptCalls, coordinateCount,
      SymbolicDuplex.stride] at below
    omega
  · unfold suffixAllocation PiRlcCanonicalSamplerHonest.suffixAllocation at inSuffix
    simp only [List.mem_append] at inSuffix
    rcases inSuffix with (inU64 | inCandidates) | inSelectors
    · have upper :=
        (PiRlcCanonicalU64.allocation_mem_iff
          (u64Base duplexBase) coordinateCount column).mp inU64 |>.2
      simp only [u64Base, coordinateCount, transcriptCalls,
        SymbolicDuplex.stride,
        PiRlcCanonicalU64.lanesPerScalar,
        CanonicalU64Recipe.auxiliaryCount] at upper ⊢
      omega
    · have upper :=
        (PiRlcCanonicalCandidates.allocation_mem_iff
          (candidateBase duplexBase) coordinateCount column).mp
            inCandidates |>.2
      simp only [candidateBase,
        PiRlcCanonicalCandidatesBatchHonest.u64End, u64Base,
        coordinateCount, transcriptCalls,
        SymbolicDuplex.stride,
        PiRlcCanonicalU64.lanesPerScalar,
        CanonicalU64Recipe.auxiliaryCount,
        PiRlcCanonicalCandidates.candidatesPerScalar,
        PiRlcCanonicalCandidate.auxiliaryCount] at upper ⊢
      omega
    · have upper :=
        (PiRlcCanonicalSelector.allocation_mem_iff
          (selectorBase duplexBase) coordinateCount column).mp inSelectors |>.2
      simp only [selectorBase,
        PiRlcCanonicalSelectorBatchHonest.candidateEnd, candidateBase,
        PiRlcCanonicalCandidatesBatchHonest.u64End, u64Base,
        coordinateCount, transcriptCalls,
        SymbolicDuplex.stride,
        PiRlcCanonicalU64.lanesPerScalar,
        CanonicalU64Recipe.auxiliaryCount,
        PiRlcCanonicalCandidates.candidatesPerScalar,
        PiRlcCanonicalCandidate.auxiliaryCount,
        PiRlcCanonicalSelector.scalarAuxiliaryCount,
        PiRlcCanonicalSelector.outputCount,
        PiRlcCanonicalSelector.positionAuxiliaryCount] at upper ⊢
      omega

/-! ## Combined structured row receipts -/

def TranscriptOwner
    (duplexBase : Nat) (lanes : State) : Type :=
  SymbolicDuplexPhysical.RowOwner
    (PiRlcCanonicalSymbolicMachineHonest.fixedBuilder
      duplexBase lanes).entries

abbrev RowOwner
    (duplexBase : Nat) (lanes : State) :=
  Sum (TranscriptOwner duplexBase lanes)
    PiRlcCanonicalSamplerOwnership.RowOwner

def owners (duplexBase : Nat) (lanes : State) :
    List (RowOwner duplexBase lanes) :=
  (SymbolicDuplexPhysical.owners
      (PiRlcCanonicalSymbolicMachineHonest.fixedBuilder
        duplexBase lanes).entries).map Sum.inl ++
    (PiRlcCanonicalSamplerOwnership.owners
      duplexBase (u64Base duplexBase) (candidateBase duplexBase)
      (selectorBase duplexBase) coordinateCount
      (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)).map
      Sum.inr

def ownedRow
    (duplexBase : Nat) (constants : Constants) (lanes : State) :
    RowOwner duplexBase lanes → Row :=
  Sum.elim
    (SymbolicDuplexPhysical.ownedRow duplexBase constants)
    (PiRlcCanonicalSamplerOwnership.ownedRow
      duplexBase (u64Base duplexBase) (candidateBase duplexBase)
      (selectorBase duplexBase) coordinateCount
      (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes))

private theorem map_sum
    {α β γ : Type}
    (left : α → γ) (right : β → γ)
    (lefts : List α) (rights : List β) :
    (lefts.map Sum.inl ++ rights.map Sum.inr).map
        (Sum.elim left right) =
      lefts.map left ++ rights.map right := by
  rw [List.map_append, List.map_map, List.map_map]
  rfl

theorem owners_nodup (duplexBase : Nat) (lanes : State) :
    (owners duplexBase lanes).Nodup := by
  unfold owners
  apply List.nodup_append.mpr
  refine
    ⟨Poseidon2Ownership.nodup_map_of_injective Sum.inl
        (fun first second equal => by cases equal; rfl)
        (SymbolicDuplexPhysical.owners_nodup _),
      Poseidon2Ownership.nodup_map_of_injective Sum.inr
        (fun first second equal => by cases equal; rfl)
        (PiRlcCanonicalSamplerOwnership.owners_nodup
          duplexBase (u64Base duplexBase) (candidateBase duplexBase)
          (selectorBase duplexBase) coordinateCount
          (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)),
      ?_⟩
  intro left leftMember right rightMember equal
  rcases List.mem_map.1 leftMember with ⟨owner, _, rfl⟩
  rcases List.mem_map.1 rightMember with ⟨other, _, rfl⟩
  cases equal

theorem rows_eq_map_owners
    (duplexBase : Nat) (constants : Constants) (lanes : State) :
    rows duplexBase constants lanes =
      (owners duplexBase lanes).map
        (ownedRow duplexBase constants lanes) := by
  have transcriptEq :
      transcriptRows duplexBase constants lanes =
        (SymbolicDuplexPhysical.owners
          (PiRlcCanonicalSymbolicMachineHonest.fixedBuilder
            duplexBase lanes).entries).map
          (SymbolicDuplexPhysical.ownedRow duplexBase constants) := by
    exact
      (PiRlcCanonicalSymbolicMachineHonest.fixedRows_ownership
        duplexBase constants lanes).2.2
  have suffixEq :
      suffixRows duplexBase lanes =
        (PiRlcCanonicalSamplerOwnership.owners
          duplexBase (u64Base duplexBase) (candidateBase duplexBase)
          (selectorBase duplexBase) coordinateCount
          (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)).map
          (PiRlcCanonicalSamplerOwnership.ownedRow
            duplexBase (u64Base duplexBase) (candidateBase duplexBase)
            (selectorBase duplexBase) coordinateCount
            (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)) := by
    exact
      PiRlcCanonicalSamplerOwnership.rows_eq_map_owners
        duplexBase (u64Base duplexBase) (candidateBase duplexBase)
        (selectorBase duplexBase) coordinateCount
        (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
  unfold rows
  rw [transcriptEq, suffixEq]
  unfold owners
  exact
    (map_sum
      (SymbolicDuplexPhysical.ownedRow duplexBase constants)
      (PiRlcCanonicalSamplerOwnership.ownedRow
        duplexBase (u64Base duplexBase) (candidateBase duplexBase)
        (selectorBase duplexBase) coordinateCount
        (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes))
      _ _).symm

theorem ownership_is_positional
    (duplexBase : Nat) (constants : Constants) (lanes : State) :
    (rows duplexBase constants lanes).length =
        (owners duplexBase lanes).length
      ∧ (owners duplexBase lanes).Nodup
      ∧ rows duplexBase constants lanes =
          (owners duplexBase lanes).map
            (ownedRow duplexBase constants lanes) := by
  refine
    ⟨?_, owners_nodup duplexBase lanes,
      rows_eq_map_owners duplexBase constants lanes⟩
  rw [rows_eq_map_owners, List.length_map]

/-! ## Internally constructed placement -/

theorem inputsBelow
    (duplexBase : Nat) (lanes : State) :
    PiRlcCanonicalU64Honest.InputsBelow
      duplexBase (u64Base duplexBase) coordinateCount
      (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes) := by
  apply PiRlcCanonicalU64Placement.inputsBelow_of_transcriptEnd
  · exact PiRlcCanonicalSymbolicMachineHonest.initialBuilder_absorbed lanes
  · simp only [PiRlcCanonicalU64Placement.transcriptEnd, u64Base,
      transcriptCalls,
      PiRlcCanonicalSymbolicMachineHonest.initialBuilder_entries_length,
      Nat.zero_add]
    exact Nat.le_refl _

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

/-! ## Honest completeness -/

/-- The exact final assignment constructed by the fixed transcript, canonical
u64, candidate, and selector witnesses.  Naming this assignment makes
cross-program preservation available to larger verifier occurrences; it does
not add a caller-supplied semantic field. -/
def honestAssignment
    (field : PiRlcCanonicalCandidateHonest.FieldInverse)
    (duplexBase : Nat) (constants : Constants) (lanes : State)
    (initial : Nat → Nat)
    (enough :
      ∀ coordinate : Fin coordinateCount,
        FirstAccepted.Enough ProductionAlphabet.verifier
          PiRlcCanonicalSelector.outputCount
          (PiRlcCanonicalSamplerHonest.honestCandidates field
            duplexBase (u64Base duplexBase) (candidateBase duplexBase)
            coordinateCount
            (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
            (PiRlcCanonicalSymbolicMachineHonest.fixedWitness
              duplexBase constants lanes initial)
            (inputsBelow duplexBase lanes)
            (PiRlcCanonicalCandidatesBatchHonest.sourcesBelow_of_u64End
              duplexBase (u64Base duplexBase) (candidateBase duplexBase)
              coordinateCount
              (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
              (u64_separated duplexBase))
            coordinate)) : Nat → Nat :=
  let transcriptAssignment :=
    PiRlcCanonicalSymbolicMachineHonest.fixedWitness
      duplexBase constants lanes initial
  let initialBuilder :=
    PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes
  let u64Below := inputsBelow duplexBase lanes
  let candidateBelow :=
    PiRlcCanonicalCandidatesBatchHonest.sourcesBelow_of_u64End
      duplexBase (u64Base duplexBase) (candidateBase duplexBase)
      coordinateCount initialBuilder (u64_separated duplexBase)
  PiRlcCanonicalSamplerHonest.finalWitness field
    duplexBase (u64Base duplexBase) (candidateBase duplexBase)
    (selectorBase duplexBase) coordinateCount initialBuilder
    transcriptAssignment u64Below candidateBelow enough

/-- The complete sampler witness preserves every caller-owned source before
the sampler's allocation base. -/
theorem honestAssignment_before_base
    (field : PiRlcCanonicalCandidateHonest.FieldInverse)
    (duplexBase : Nat) (constants : Constants) (lanes : State)
    (initial : Nat → Nat)
    (enough :
      ∀ coordinate : Fin coordinateCount,
        FirstAccepted.Enough ProductionAlphabet.verifier
          PiRlcCanonicalSelector.outputCount
          (PiRlcCanonicalSamplerHonest.honestCandidates field
            duplexBase (u64Base duplexBase) (candidateBase duplexBase)
            coordinateCount
            (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
            (PiRlcCanonicalSymbolicMachineHonest.fixedWitness
              duplexBase constants lanes initial)
            (inputsBelow duplexBase lanes)
            (PiRlcCanonicalCandidatesBatchHonest.sourcesBelow_of_u64End
              duplexBase (u64Base duplexBase) (candidateBase duplexBase)
              coordinateCount
              (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
              (u64_separated duplexBase))
            coordinate))
    {column : Nat} (before : column < duplexBase) :
    honestAssignment field duplexBase constants lanes initial enough column =
      initial column := by
  unfold honestAssignment
  rw [PiRlcCanonicalSamplerHonest.finalWitness_before_u64Base field
    duplexBase (u64Base duplexBase) (candidateBase duplexBase)
    (selectorBase duplexBase) coordinateCount
    (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
    (PiRlcCanonicalSymbolicMachineHonest.fixedWitness
      duplexBase constants lanes initial)
    (inputsBelow duplexBase lanes)
    (PiRlcCanonicalCandidatesBatchHonest.sourcesBelow_of_u64End
      duplexBase (u64Base duplexBase) (candidateBase duplexBase)
      coordinateCount
      (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
      (u64_separated duplexBase))
    enough (u64_separated duplexBase) (candidate_separated duplexBase)
    (by
      unfold u64Base
      omega)]
  apply SymbolicDuplexHonest.witnesses_preserve_before
    (boundary := duplexBase) (column := column)
  · intro entry _member
    simp only [SymbolicDuplexHonest.outputBase,
      SymbolicDuplexHonest.callBase, SymbolicDuplex.stride]
    omega
  · exact before

/-- Every column written by the named sampler witness is represented by its
canonical Goldilocks residue. -/
theorem honestAssignment_canonical
    (field : PiRlcCanonicalCandidateHonest.FieldInverse)
    (duplexBase : Nat) (constants : Constants) (lanes : State)
    (initial : Nat → Nat)
    (initialCanonical : ∀ column, initial column < goldilocksP)
    (enough :
      ∀ coordinate : Fin coordinateCount,
        FirstAccepted.Enough ProductionAlphabet.verifier
          PiRlcCanonicalSelector.outputCount
          (PiRlcCanonicalSamplerHonest.honestCandidates field
            duplexBase (u64Base duplexBase) (candidateBase duplexBase)
            coordinateCount
            (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
            (PiRlcCanonicalSymbolicMachineHonest.fixedWitness
              duplexBase constants lanes initial)
            (inputsBelow duplexBase lanes)
            (PiRlcCanonicalCandidatesBatchHonest.sourcesBelow_of_u64End
              duplexBase (u64Base duplexBase) (candidateBase duplexBase)
              coordinateCount
              (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
              (u64_separated duplexBase))
            coordinate)) :
    ∀ column,
      honestAssignment field duplexBase constants lanes initial enough column <
        goldilocksP := by
  let transcript :=
    PiRlcCanonicalSymbolicMachineHonest.fixedWitness
      duplexBase constants lanes initial
  let builder :=
    PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes
  let u64 :=
    PiRlcCanonicalSamplerHonest.u64Witness field duplexBase (u64Base duplexBase)
      coordinateCount builder transcript
  let candidate :=
    PiRlcCanonicalSamplerHonest.candidateWitness field duplexBase
      (u64Base duplexBase) (candidateBase duplexBase) coordinateCount
      builder transcript
  let below :=
    PiRlcCanonicalCandidatesBatchHonest.sourcesBelow_of_u64End
      duplexBase (u64Base duplexBase) (candidateBase duplexBase)
      coordinateCount builder (u64_separated duplexBase)
  have transcriptCanonical : ∀ column, transcript column < goldilocksP :=
    PiRlcCanonicalSymbolicMachineHonest.fixedWitness_residues
      duplexBase constants lanes initial initialCanonical
  have u64Canonical : ∀ column, u64 column < goldilocksP :=
    PiRlcCanonicalU64Honest.batchWitness_canonical field duplexBase
      (u64Base duplexBase) coordinateCount builder transcript
      transcriptCanonical
  have candidateCanonical : ∀ column, candidate column < goldilocksP :=
    PiRlcCanonicalCandidatesBatchHonest.batchWitness_canonical field
      duplexBase (u64Base duplexBase) (candidateBase duplexBase)
      coordinateCount builder u64 u64Canonical below
      (fun coordinate candidate =>
        PiRlcCanonicalCandidatesBatchHonest.sourceBitsBoolean_of_u64Witness
          field duplexBase (u64Base duplexBase) (candidateBase duplexBase)
          coordinateCount builder transcript
          (inputsBelow duplexBase lanes) coordinate candidate)
  intro column
  unfold honestAssignment PiRlcCanonicalSamplerHonest.finalWitness
  exact
    PiRlcCanonicalSelectorBatchHonest.batchPrefixWitness_canonical
      duplexBase (u64Base duplexBase) (candidateBase duplexBase)
      (selectorBase duplexBase) coordinateCount builder candidate
      candidateCanonical
      (PiRlcCanonicalSamplerHonest.honestCandidates field duplexBase
        (u64Base duplexBase) (candidateBase duplexBase) coordinateCount
        builder transcript (inputsBelow duplexBase lanes) below)
      (PiRlcCanonicalSamplerHonest.candidateSourcesMatch field duplexBase
        (u64Base duplexBase) (candidateBase duplexBase) coordinateCount
        builder transcript (inputsBelow duplexBase lanes) below)
      enough coordinateCount column

/-- The named sampler witness preserves the caller's constant-one wire. -/
theorem honestAssignment_constantWire
    (field : PiRlcCanonicalCandidateHonest.FieldInverse)
    (duplexBase : Nat) (constants : Constants) (lanes : State)
    (initial : Nat → Nat)
    (enough :
      ∀ coordinate : Fin coordinateCount,
        FirstAccepted.Enough ProductionAlphabet.verifier
          PiRlcCanonicalSelector.outputCount
          (PiRlcCanonicalSamplerHonest.honestCandidates field
            duplexBase (u64Base duplexBase) (candidateBase duplexBase)
            coordinateCount
            (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
            (PiRlcCanonicalSymbolicMachineHonest.fixedWitness
              duplexBase constants lanes initial)
            (inputsBelow duplexBase lanes)
            (PiRlcCanonicalCandidatesBatchHonest.sourcesBelow_of_u64End
              duplexBase (u64Base duplexBase) (candidateBase duplexBase)
              coordinateCount
              (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
              (u64_separated duplexBase))
            coordinate))
    (positive : 0 < duplexBase)
    (constantWire : initial 0 = 1) :
    honestAssignment field duplexBase constants lanes initial enough 0 = 1 :=
  (honestAssignment_before_base field duplexBase constants lanes initial enough
    positive).trans constantWire

/-- The named honest assignment satisfies every row of the selected fixed
sampler program. -/
theorem honestAssignment_satisfies
    (field : PiRlcCanonicalCandidateHonest.FieldInverse)
    (duplexBase : Nat) (constants : Constants) (lanes : State)
    (initial : Nat → Nat)
    (positive : 0 < duplexBase)
    (lanesInPrefix :
      ∀ lane : Fin width, ValueInPrefix duplexBase (lanes lane))
    (initialCanonical : ∀ column, initial column < goldilocksP)
    (constantWire : initial 0 = 1)
    (enough :
      ∀ coordinate : Fin coordinateCount,
        FirstAccepted.Enough ProductionAlphabet.verifier
          PiRlcCanonicalSelector.outputCount
          (PiRlcCanonicalSamplerHonest.honestCandidates field
            duplexBase (u64Base duplexBase) (candidateBase duplexBase)
            coordinateCount
            (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
            (PiRlcCanonicalSymbolicMachineHonest.fixedWitness
              duplexBase constants lanes initial)
            (inputsBelow duplexBase lanes)
            (PiRlcCanonicalCandidatesBatchHonest.sourcesBelow_of_u64End
              duplexBase (u64Base duplexBase) (candidateBase duplexBase)
              coordinateCount
              (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
              (u64_separated duplexBase))
            coordinate)) :
    Satisfies (rows duplexBase constants lanes)
      (honestAssignment field duplexBase constants lanes initial enough) := by
  let transcriptAssignment :=
    PiRlcCanonicalSymbolicMachineHonest.fixedWitness
      duplexBase constants lanes initial
  let initialBuilder :=
    PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes
  let u64Below := inputsBelow duplexBase lanes
  let candidateBelow :=
    PiRlcCanonicalCandidatesBatchHonest.sourcesBelow_of_u64End
      duplexBase (u64Base duplexBase) (candidateBase duplexBase)
      coordinateCount initialBuilder (u64_separated duplexBase)
  let assignment :=
    PiRlcCanonicalSamplerHonest.finalWitness field
      duplexBase (u64Base duplexBase) (candidateBase duplexBase)
      (selectorBase duplexBase) coordinateCount initialBuilder
      transcriptAssignment u64Below candidateBelow enough
  change Satisfies (rows duplexBase constants lanes) assignment
  have lanesBefore :
      ∀ lane : Fin width, ∀ column,
        Mentions (lanes lane) column →
          column <
            SymbolicDuplexHonest.outputBase duplexBase 0 := by
    intro lane column mentioned
    simpa only [SymbolicDuplexHonest.outputBase,
      SymbolicDuplexHonest.callBase] using
      lanesInPrefix lane column mentioned
  have transcriptSatisfied :
      Satisfies (transcriptRows duplexBase constants lanes)
        transcriptAssignment :=
    PiRlcCanonicalSymbolicMachineHonest.fixedRows_honest
      duplexBase constants lanes initial lanesBefore positive initialCanonical
      constantWire
  have transcriptCanonical :
      ∀ column, transcriptAssignment column < goldilocksP :=
    PiRlcCanonicalSymbolicMachineHonest.fixedWitness_residues
      duplexBase constants lanes initial initialCanonical
  have transcriptConstant : transcriptAssignment 0 = 1 := by
    change
      PiRlcCanonicalSymbolicMachineHonest.fixedWitness
        duplexBase constants lanes initial 0 = 1
    rw [PiRlcCanonicalSymbolicMachineHonest.fixedWitness_constantWire
      duplexBase constants lanes initial positive]
    exact constantWire
  have suffixSatisfied :
      Satisfies (suffixRows duplexBase lanes) assignment := by
    exact PiRlcCanonicalSamplerHonest.suffixRows_complete field
      duplexBase (u64Base duplexBase) (candidateBase duplexBase)
      (selectorBase duplexBase) coordinateCount initialBuilder
      transcriptAssignment transcriptCanonical transcriptConstant
      (by
        unfold u64Base
        omega)
      u64Below (u64_separated duplexBase) (candidate_separated duplexBase)
      enough
  intro row member
  rcases List.mem_append.1 member with inTranscript | inSuffix
  · have holds := transcriptSatisfied row inTranscript
    apply (rowHolds_congr transcriptAssignment assignment row ?_).mp holds
    intro column mentioned
    have conserved :=
      PiRlcCanonicalSymbolicMachineHonest.fixedRows_conservation
        duplexBase constants lanes positive lanesInPrefix row inTranscript
        column mentioned
    have beforeU64 : column < u64Base duplexBase := by
      rcases conserved with inPrefix | inAllocation
      · unfold u64Base
        omega
      · have below :=
          SymbolicDuplexPhysical.temporaryColumns_lt_end
            duplexBase transcriptCalls column
            (by
              rw [← transcriptAllocation_eq]
              exact inAllocation)
        exact below
    symm
    exact PiRlcCanonicalSamplerHonest.finalWitness_before_u64Base field
      duplexBase (u64Base duplexBase) (candidateBase duplexBase)
      (selectorBase duplexBase) coordinateCount initialBuilder
      transcriptAssignment u64Below candidateBelow enough
      (u64_separated duplexBase) (candidate_separated duplexBase)
      beforeU64
  · exact suffixSatisfied row inSuffix

/-- Existential honest completeness, retained as the public construction
interface used by callers that do not need the exact assignment. -/
theorem rows_complete
    (field : PiRlcCanonicalCandidateHonest.FieldInverse)
    (duplexBase : Nat) (constants : Constants) (lanes : State)
    (initial : Nat → Nat)
    (positive : 0 < duplexBase)
    (lanesInPrefix :
      ∀ lane : Fin width, ValueInPrefix duplexBase (lanes lane))
    (initialCanonical : ∀ column, initial column < goldilocksP)
    (constantWire : initial 0 = 1)
    (enough :
      ∀ coordinate : Fin coordinateCount,
        FirstAccepted.Enough ProductionAlphabet.verifier
          PiRlcCanonicalSelector.outputCount
          (PiRlcCanonicalSamplerHonest.honestCandidates field
            duplexBase (u64Base duplexBase) (candidateBase duplexBase)
            coordinateCount
            (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
            (PiRlcCanonicalSymbolicMachineHonest.fixedWitness
              duplexBase constants lanes initial)
            (inputsBelow duplexBase lanes)
            (PiRlcCanonicalCandidatesBatchHonest.sourcesBelow_of_u64End
              duplexBase (u64Base duplexBase) (candidateBase duplexBase)
              coordinateCount
              (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
              (u64_separated duplexBase))
            coordinate)) :
    ∃ assignment,
      Satisfies (rows duplexBase constants lanes) assignment :=
  ⟨honestAssignment field duplexBase constants lanes initial enough,
    honestAssignment_satisfies field duplexBase constants lanes initial
      positive lanesInPrefix initialCanonical constantWire enough⟩

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgram
