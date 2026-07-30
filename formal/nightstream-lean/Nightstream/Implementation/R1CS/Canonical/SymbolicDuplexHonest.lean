import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplex
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation
import Nightstream.Implementation.R1CS.Canonical.KMulHonest

/-!
Contract: an explicit honest assignment for a well-placed symbolic Poseidon2
duplex.

Owns: the call-local witness, sequential witness composition, the structural
placement predicate consumed by that composition, and honest completeness of
the exact `SymbolicDuplex.rows` list.

Does not own: a protocol absorption schedule.  Callers prove `WellPlaced` while
building their typed transcript.  The predicate is structural: every call is
position-indexed and every carried source precedes that call's output/auxiliary
space.  It contains no acceptance proposition or row-satisfaction premise.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest
open Nightstream.Implementation.R1CS.Canonical.Poseidon2HonestFrom
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexSemantics

def callBase (base call : Nat) : Nat :=
  base + call * SymbolicDuplex.stride

def outputBase (base call : Nat) : Nat :=
  callBase base call

def callEnd (base call : Nat) : Nat :=
  callBase base call + SymbolicDuplex.stride

/-- Every source consumed by an entry precedes that entry's output and
auxiliary columns. -/
def SourcesBefore (base : Nat) (entry : SymbolicDuplex.Entry) : Prop :=
  ∀ lane : Fin width, ∀ column,
    Mentions (entry.state lane) column →
      column < outputBase base entry.call

/-- The one-call witness.  Input-port columns are unused by a carried-entry
program and remain untouched.  The eight output ports and 344 S-box columns
receive the reference execution; every other column is preserved. -/
def entryWitness
    (base : Nat) (constants : Constants)
    (entry : SymbolicDuplex.Entry) (z : Nat → Nat) : Nat → Nat :=
  fun column =>
    if beforeOutputs : column < outputBase base entry.call then
      z column
    else if output : column < callBase base entry.call + width then
      referencePermutation constants (evalState z entry.state)
        ⟨column - outputBase base entry.call, by
          simp only [outputBase, callBase, width] at *
          omega⟩
    else if auxiliary : column < callEnd base entry.call then
      chainSlot
        (sboxInputValue constants (evalState z entry.state)
          ((column - (callBase base entry.call + width)) / columnsPerSbox))
        ((column - (callBase base entry.call + width)) % columnsPerSbox)
    else
      z column

theorem entryWitness_before
    (base : Nat) (constants : Constants)
    (entry : SymbolicDuplex.Entry) (z : Nat → Nat)
    (column : Nat) (before : column < outputBase base entry.call) :
    entryWitness base constants entry z column = z column := by
  unfold entryWitness
  rw [dif_pos before]

theorem entryWitness_after
    (base : Nat) (constants : Constants)
    (entry : SymbolicDuplex.Entry) (z : Nat → Nat)
    (column : Nat) (after : callEnd base entry.call ≤ column) :
    entryWitness base constants entry z column = z column := by
  unfold entryWitness
  rw [dif_neg (by
      simp only [outputBase, callEnd, callBase, SymbolicDuplex.stride,
        Poseidon2Layout.canonicalColumnTotal, width, sboxCount,
        externalRounds, partialRounds, columnsPerSbox] at *
      omega),
    dif_neg (by
      simp only [callEnd, callBase, SymbolicDuplex.stride,
        Poseidon2Layout.canonicalColumnTotal, width, sboxCount,
        externalRounds, partialRounds, columnsPerSbox] at *
      omega),
    dif_neg (Nat.not_lt_of_ge after)]

theorem entryWitness_output
    (base : Nat) (constants : Constants)
    (entry : SymbolicDuplex.Entry) (z : Nat → Nat)
    (lane : Fin width) :
    entryWitness base constants entry z
        ((SymbolicDuplex.layoutAt base entry.call).outputPort lane) =
      referencePermutation constants (evalState z entry.state) lane := by
  have laneLt := lane.isLt
  simp only [width] at laneLt
  unfold entryWitness
  simp only [SymbolicDuplex.layoutAt, outputBase, callBase,
    SymbolicDuplex.stride, width]
  rw [dif_neg (by omega), dif_pos (by omega)]
  apply congrArg (referencePermutation constants (evalState z entry.state))
  apply Fin.ext
  change
    base + entry.call * SymbolicDuplex.stride + lane.val -
        (base + entry.call * SymbolicDuplex.stride) =
      lane.val
  omega

theorem entryWitness_sbox
    (base : Nat) (constants : Constants)
    (entry : SymbolicDuplex.Entry) (z : Nat → Nat)
    (index : Fin sboxCount) (slot : Fin columnsPerSbox) :
    entryWitness base constants entry z
        (sboxColumn (SymbolicDuplex.layoutAt base entry.call) index slot) =
      chainSlot
        (sboxInputValue constants (evalState z entry.state) index.val)
        slot.val := by
  have indexLt := index.isLt
  have slotLt := slot.isLt
  simp only [sboxCount, externalRounds, width, partialRounds] at indexLt
  simp only [columnsPerSbox] at slotLt
  unfold entryWitness
  simp only [SymbolicDuplex.layoutAt,
    sboxColumn, outputBase, callEnd, callBase, SymbolicDuplex.stride,
    width, sboxCount, externalRounds,
    partialRounds, columnsPerSbox]
  rw [dif_neg (by omega), dif_neg (by omega), dif_pos (by omega)]
  have divEq :
      ((base + entry.call * 352 + 8 + 4 * index.val + slot.val -
          (base + entry.call * 352 + 8)) / 4) = index.val := by
    omega
  have modEq :
      ((base + entry.call * 352 + 8 + 4 * index.val + slot.val -
          (base + entry.call * 352 + 8)) % 4) = slot.val := by
    omega
  rw [divEq, modEq]

theorem entryWitness_residues
    (base : Nat) (constants : Constants)
    (entry : SymbolicDuplex.Entry) (z : Nat → Nat)
    (residues : ∀ column, z column < goldilocksP) :
    ∀ column, entryWitness base constants entry z column < goldilocksP := by
  intro column
  unfold entryWitness
  split
  · exact residues _
  · split
    · exact refTerminal_lt _ _ _ _
    · split
      · exact chainSlot_lt _ _
      · exact residues _

theorem entryWitness_constantWire
    (base : Nat) (constants : Constants)
    (entry : SymbolicDuplex.Entry) (z : Nat → Nat)
    (basePositive : 0 < base) :
    entryWitness base constants entry z 0 = z 0 := by
  apply entryWitness_before
  simp only [outputBase, callBase]
  omega

theorem entryWitness_entryAgrees
    (base : Nat) (constants : Constants)
    (entry : SymbolicDuplex.Entry) (z : Nat → Nat)
    (sources : SourcesBefore base entry) (lane : Fin width) :
    lcEval (entryWitness base constants entry z) (entry.state lane) =
      evalState z entry.state lane := by
  unfold evalState
  refine (KMulHonest.lcEval_congr z
    (entryWitness base constants entry z) (entry.state lane)
    (fun column mentioned => ?_)).symm
  exact (entryWitness_before base constants entry z column
    (sources lane column mentioned)).symm

/-- One call's witness supplies every premise of carried-entry honest
completeness. -/
theorem entryWitness_honest
    (base : Nat) (constants : Constants)
    (entry : SymbolicDuplex.Entry) (z : Nat → Nat)
    (sources : SourcesBefore base entry) :
    SymbolicDuplex.EntryHonest base constants
      (entryWitness base constants entry z) entry
      (evalState z entry.state) where
  entryAgrees :=
    entryWitness_entryAgrees base constants entry z sources
  sboxAgrees :=
    entryWitness_sbox base constants entry z
  outputAgrees :=
    entryWitness_output base constants entry z

/-- Install call witnesses from left to right. -/
def witnesses
    (base : Nat) (constants : Constants) :
    List SymbolicDuplex.Entry → (Nat → Nat) → (Nat → Nat)
  | [], z => z
  | entry :: rest, z =>
      witnesses base constants rest
        (entryWitness base constants entry z)

/-- Later calls preserve any prefix below all of their output spaces. -/
theorem witnesses_preserve_before
    (base : Nat) (constants : Constants) :
    ∀ (entries : List SymbolicDuplex.Entry) (z : Nat → Nat)
      (boundary column : Nat),
      (∀ entry ∈ entries, boundary ≤ outputBase base entry.call) →
      column < boundary →
      witnesses base constants entries z column = z column
  | [], _, _, _, _, _ => rfl
  | entry :: rest, z, boundary, column, allAfter, below => by
      rw [witnesses,
        witnesses_preserve_before base constants rest
          (entryWitness base constants entry z) boundary column
          (fun other member => allAfter other (by simp [member])) below,
        entryWitness_before base constants entry z column
          (Nat.lt_of_lt_of_le below (allAfter entry (by simp)))]

theorem witnesses_residues
    (base : Nat) (constants : Constants) :
    ∀ (entries : List SymbolicDuplex.Entry) (z : Nat → Nat),
      (∀ column, z column < goldilocksP) →
      ∀ column, witnesses base constants entries z column < goldilocksP
  | [], _, residues => residues
  | entry :: rest, z, residues =>
      witnesses_residues base constants rest
        (entryWitness base constants entry z)
        (entryWitness_residues base constants entry z residues)

theorem witnesses_constantWire
    (base : Nat) (constants : Constants) (basePositive : 0 < base) :
    ∀ (entries : List SymbolicDuplex.Entry) (z : Nat → Nat),
      witnesses base constants entries z 0 = z 0
  | [], _ => rfl
  | entry :: rest, z => by
      rw [witnesses,
        witnesses_constantWire base constants basePositive rest
          (entryWitness base constants entry z),
        entryWitness_constantWire base constants entry z basePositive]

private theorem outputPort_lt_callEnd
    (base call : Nat) (lane : Fin width) :
    (SymbolicDuplex.layoutAt base call).outputPort lane < callEnd base call := by
  have laneLt := lane.isLt
  simp only [width] at laneLt
  simp only [SymbolicDuplex.layoutAt,
    callEnd, callBase, SymbolicDuplex.stride,
    width]
  omega

private theorem sboxColumn_lt_callEnd
    (base call : Nat) (index : Fin sboxCount)
    (slot : Fin columnsPerSbox) :
    sboxColumn (SymbolicDuplex.layoutAt base call) index slot <
      callEnd base call := by
  have indexLt := index.isLt
  have slotLt := slot.isLt
  simp only [sboxCount, externalRounds, width, partialRounds] at indexLt
  simp only [columnsPerSbox] at slotLt
  simp only [SymbolicDuplex.layoutAt,
    sboxColumn, callEnd, callBase, SymbolicDuplex.stride,
    width, columnsPerSbox]
  omega

private theorem callEnd_le_laterOutput
    (base first second : Nat) (later : first < second) :
    callEnd base first ≤ outputBase base second := by
  simp only [callEnd, outputBase, callBase, SymbolicDuplex.stride]
  omega

/-- Constructive per-entry honesty for one list under one final assignment. -/
inductive EntriesHonest
    (base : Nat) (constants : Constants) (assignment : Nat → Nat) :
    List SymbolicDuplex.Entry → Prop
  | nil : EntriesHonest base constants assignment []
  | cons {head : SymbolicDuplex.Entry} {tail : List SymbolicDuplex.Entry}
      (headHonest :
        ∃ values,
          SymbolicDuplex.EntryHonest base constants assignment head values)
      (tailHonest : EntriesHonest base constants assignment tail) :
      EntriesHonest base constants assignment (head :: tail)

/-- The final sequential assignment is honest for every entry.  The value
consumed by each entry is constructed at the point that entry is installed;
later calls preserve it because their column spaces are strictly later. -/
theorem witnesses_entries_honest
    (base : Nat) (constants : Constants) :
    ∀ (entries : List SymbolicDuplex.Entry) (z : Nat → Nat),
      entries.Pairwise (fun first second => first.call < second.call) →
      (∀ entry ∈ entries, SourcesBefore base entry) →
      EntriesHonest base constants
        (witnesses base constants entries z) entries := by
  intro entries
  induction entries with
  | nil =>
      intro _ _ _
      exact EntriesHonest.nil
  | cons firstEntry rest inductionHypothesis =>
      intro z ordered sources
      have headBeforeLater :
          ∀ other ∈ rest, callEnd base firstEntry.call ≤
            outputBase base other.call := by
        intro other otherMember
        exact callEnd_le_laterOutput base firstEntry.call other.call
          ((List.pairwise_cons.mp ordered).1 other otherMember)
      have preserveHead :
          ∀ column, column < callEnd base firstEntry.call →
            witnesses base constants rest
                (entryWitness base constants firstEntry z) column =
              entryWitness base constants firstEntry z column :=
        fun column below =>
          witnesses_preserve_before base constants rest
            (entryWitness base constants firstEntry z)
            (callEnd base firstEntry.call) column headBeforeLater below
      apply EntriesHonest.cons
      · refine ⟨evalState z firstEntry.state, ?_⟩
        have self :=
          entryWitness_honest base constants firstEntry z
            (sources firstEntry (by simp))
        refine
          { entryAgrees := fun lane => ?_
            sboxAgrees := fun index slot => ?_
            outputAgrees := fun lane => ?_ }
        · change
            lcEval
                (witnesses base constants rest
                  (entryWitness base constants firstEntry z))
                (firstEntry.state lane) =
              evalState z firstEntry.state lane
          calc
            lcEval
                (witnesses base constants rest
                  (entryWitness base constants firstEntry z))
                (firstEntry.state lane) =
              lcEval (entryWitness base constants firstEntry z)
                (firstEntry.state lane) := by
                  apply KMulHonest.lcEval_congr
                  intro column mentioned
                  exact preserveHead column
                    (Nat.lt_trans
                      ((sources firstEntry (by simp)) lane column mentioned)
                      (by
                        simp only [outputBase, callEnd, callBase,
                          SymbolicDuplex.stride,
                          Poseidon2Layout.canonicalColumnTotal, width,
                          sboxCount, externalRounds, partialRounds,
                          columnsPerSbox]
                        omega))
            _ = evalState z firstEntry.state lane :=
              self.entryAgrees lane
        · change
            witnesses base constants rest
                (entryWitness base constants firstEntry z)
                (sboxColumn
                  (SymbolicDuplex.layoutAt base firstEntry.call) index slot) =
              _
          rw [preserveHead _
            (sboxColumn_lt_callEnd base firstEntry.call index slot)]
          exact self.sboxAgrees index slot
        · change
            witnesses base constants rest
                (entryWitness base constants firstEntry z)
                ((SymbolicDuplex.layoutAt base firstEntry.call).outputPort lane) =
              _
          rw [preserveHead _
            (outputPort_lt_callEnd base firstEntry.call lane)]
          exact self.outputAgrees lane
      · exact
          inductionHypothesis
            (entryWitness base constants firstEntry z)
            (List.pairwise_cons.mp ordered).2
            (fun other otherMember =>
              sources other (by simp [otherMember]))

/-- Existential per-entry honesty is enough to satisfy the exact concatenated
row list; no global values oracle is required. -/
theorem rowsFrom_honest_of_exists
    (base : Nat) (constants : Constants) (assignment : Nat → Nat)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1) :
    ∀ entries : List SymbolicDuplex.Entry,
      EntriesHonest base constants assignment entries →
      Satisfies (SymbolicDuplex.rowsFrom base constants entries) assignment
  | [], _ => by
      intro row member
      cases member
  | head :: rest, honest => by
      cases honest with
      | cons headExists tailHonest =>
          rcases headExists with ⟨values, headHonest⟩
          have headSatisfied :
              Satisfies (SymbolicDuplex.entryRows base constants head)
                assignment := by
            unfold SymbolicDuplex.entryRows
            exact honest_satisfies_normalizedFrom
              (SymbolicDuplex.layoutAt base head.call) head.state constants
              values assignment residues constantWire headHonest.entryAgrees
              headHonest.sboxAgrees headHonest.outputAgrees
          have tailSatisfied :=
            rowsFrom_honest_of_exists base constants assignment residues
              constantWire rest tailHonest
          intro row member
          rcases List.mem_append.mp member with inHead | inTail
          · exact headSatisfied row inHead
          · exact tailSatisfied row inTail

/-- Structural placement of one accumulated builder.  `calls` states that
entry call IDs are exactly their list positions; the other two fields bind
every stored entry and the current carried state below the next call. -/
structure WellPlaced (base : Nat) (builder : SymbolicDuplex.Builder) : Prop where
  calls :
    builder.entries.map SymbolicDuplex.Entry.call =
      List.range builder.entries.length
  entrySources :
    ∀ entry ∈ builder.entries, SourcesBefore base entry
  lanesBefore :
    ∀ lane : Fin width, ∀ column,
      Mentions (builder.lanes lane) column →
        column < outputBase base builder.entries.length

theorem WellPlaced.callsPairwise
    {base : Nat} {builder : SymbolicDuplex.Builder}
    (placed : WellPlaced base builder) :
    builder.entries.Pairwise
      (fun first second => first.call < second.call) := by
  apply List.pairwise_map.mp
  rw [placed.calls]
  exact List.pairwise_lt_range

/-- A well-placed builder has an explicit honest assignment for every emitted
row. -/
theorem rows_honest
    (base : Nat) (constants : Constants)
    (builder : SymbolicDuplex.Builder) (initial : Nat → Nat)
    (placed : WellPlaced base builder)
    (basePositive : 0 < base)
    (initialResidues : ∀ column, initial column < goldilocksP)
    (constantWire : initial 0 = 1) :
    Satisfies (SymbolicDuplex.rows base constants builder)
      (witnesses base constants builder.entries initial) := by
  unfold SymbolicDuplex.rows
  apply rowsFrom_honest_of_exists base constants
    (witnesses base constants builder.entries initial)
    (witnesses_residues base constants builder.entries initial
      initialResidues)
    (by
      rw [witnesses_constantWire base constants basePositive]
      exact constantWire)
  exact witnesses_entries_honest base constants builder.entries initial
    placed.callsPairwise placed.entrySources

end Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexHonest
