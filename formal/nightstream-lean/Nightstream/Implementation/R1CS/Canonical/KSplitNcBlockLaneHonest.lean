import Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneRows
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence

/-!
Contract: constructive honest completeness for the exact FE-row, FE-lane,
and block×lane NC claimed-chain program.

Owns:
- the sequential numeric witness over the three disjoint Horner intervals;
- preservation of the caller-owned prefix and earlier row groups;
- transport of the three exact frozen chains across those extensions; and
- satisfaction of `KSplitNcBlockLaneRows.rows`.

Does not own the Fiat--Shamir transcript, verifier-owned endpoint formulas,
call-frame decoding, or the selected NIFS wrapper.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckHonest
open Nightstream.Implementation.R1CS.Canonical.KHornerSupport
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps.toOps

/-- Assignment after the FE row-coordinate chain. -/
def afterFeRow
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domains : Domains}
    (columns : KSplitNcBlockLaneRows.Columns input domains)
    (base : Nat) (assignment : Nat → Nat) : Nat → Nat :=
  chainWitness assignment
    columns.fe.rowSource.rowRounds
    columns.fe.rowSource.rowChallenges
    base

/-- Assignment after both FE chains. -/
def afterFeLane
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domains : Domains}
    (columns : KSplitNcBlockLaneRows.Columns input domains)
    (base : Nat) (assignment : Nat → Nat) : Nat → Nat :=
  chainWitness (afterFeRow columns base assignment)
    columns.fe.laneSource.rowRounds
    columns.fe.laneSource.rowChallenges
    (KSplitNcFeRows.laneBase columns.fe base)

/-- Final assignment after FE-row, FE-lane, and NC chains. -/
def witness
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domains : Domains}
    (columns : KSplitNcBlockLaneRows.Columns input domains)
    (base : Nat) (assignment : Nat → Nat) : Nat → Nat :=
  chainWitness (afterFeLane columns base assignment)
    columns.nc.rowRounds
    columns.nc.rowChallenges
    (KSplitNcBlockLaneRows.ncBase columns base)

/-- The combined witness writes only in the compact allocation beginning at
`base`. -/
theorem witness_off_block
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domains : Domains}
    (columns : KSplitNcBlockLaneRows.Columns input domains)
    (base : Nat) (assignment : Nat → Nat)
    (column : Nat) (below : column < base) :
    witness columns base assignment column = assignment column := by
  unfold witness afterFeLane afterFeRow
  rw [
    chainWitness_off_block _ columns.nc.rowRounds
      columns.nc.rowChallenges
      (KSplitNcBlockLaneRows.ncBase columns base) column
      (by
        exact Nat.lt_of_lt_of_le below
          (by
            unfold KSplitNcBlockLaneRows.ncBase
            exact Nat.le_add_right _ _)),
    chainWitness_off_block _ columns.fe.laneSource.rowRounds
      columns.fe.laneSource.rowChallenges
      (KSplitNcFeRows.laneBase columns.fe base) column
      (by unfold KSplitNcFeRows.laneBase; omega),
    chainWitness_off_block _ columns.fe.rowSource.rowRounds
      columns.fe.rowSource.rowChallenges base column below]

/-- The combined FE/NC claimed-chain witness preserves canonical
representatives across all three sequential Horner intervals. -/
theorem witness_residues
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domains : Domains}
    (columns : KSplitNcBlockLaneRows.Columns input domains)
    (base : Nat) (assignment : Nat → Nat)
    (residues : ∀ column, assignment column < goldilocksP) :
    ∀ column, witness columns base assignment column < goldilocksP := by
  unfold witness afterFeLane afterFeRow
  apply chainWitness_residues
  apply chainWitness_residues
  apply chainWitness_residues
  exact residues

private theorem satisfies_append
    {left right : List Row} {assignment : Nat → Nat}
    (leftSatisfied : Satisfies left assignment)
    (rightSatisfied : Satisfies right assignment) :
    Satisfies (left ++ right) assignment := by
  intro row member
  exact (List.mem_append.1 member).elim
    (leftSatisfied row) (rightSatisfied row)

private theorem afterFeRow_agrees_below
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domains : Domains}
    (columns : KSplitNcBlockLaneRows.Columns input domains)
    (base : Nat) (assignment : Nat → Nat)
    (column : Nat) (below : column < base) :
    afterFeRow columns base assignment column = assignment column :=
  chainWitness_off_block assignment
    columns.fe.rowSource.rowRounds
    columns.fe.rowSource.rowChallenges base column below

private theorem afterFeLane_agrees_below
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domains : Domains}
    (columns : KSplitNcBlockLaneRows.Columns input domains)
    (base : Nat) (assignment : Nat → Nat)
    (column : Nat) (below : column < base) :
    afterFeLane columns base assignment column = assignment column := by
  unfold afterFeLane
  rw [
    chainWitness_off_block _ columns.fe.laneSource.rowRounds
      columns.fe.laneSource.rowChallenges
      (KSplitNcFeRows.laneBase columns.fe base) column
      (by unfold KSplitNcFeRows.laneBase; omega),
    afterFeRow_agrees_below columns base assignment column below]

/-- Honest completeness of the three exact claimed chains.

The semantic premises are precisely the three frozen `FixedPhase.Chain`
relations, evaluated on the caller's source assignment.  No R1CS row equation
or operational acceptance conclusion is supplied. -/
theorem rows_honest
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domains : Domains}
    (columns : KSplitNcBlockLaneRows.Columns input domains)
    (base : Nat) (assignment : Nat → Nat)
    (basePositive : 0 < base)
    (constantWire : assignment 0 = 1)
    (rowPlaced : columns.fe.rowSource.BelowBase base)
    (lanePlaced : columns.fe.laneSource.BelowBase base)
    (ncPlaced : columns.nc.BelowBase base)
    (rowChain :
      FixedPhase.Chain ops
        (columns.fe.rowSource.paperCurrent assignment)
        (columns.fe.rowSource.paperRounds assignment)
        (columns.fe.rowSource.paperChallenges assignment)
        (columns.fe.rowSource.paperTerminal assignment))
    (laneChain :
      FixedPhase.Chain ops
        (columns.fe.laneSource.paperCurrent assignment)
        (columns.fe.laneSource.paperRounds assignment)
        (columns.fe.laneSource.paperChallenges assignment)
        (columns.fe.laneSource.paperTerminal assignment))
    (ncChain :
      FixedPhase.Chain ops
        (columns.nc.paperCurrent assignment)
        (columns.nc.paperRounds assignment)
        (columns.nc.paperChallenges assignment)
        (columns.nc.paperTerminal assignment)) :
    Satisfies (KSplitNcBlockLaneRows.rows columns base)
      (witness columns base assignment) := by
  let laneBase := KSplitNcFeRows.laneBase columns.fe base
  let ncBase := KSplitNcBlockLaneRows.ncBase columns base
  let afterRow := afterFeRow columns base assignment
  let afterLane := afterFeLane columns base assignment
  let final := witness columns base assignment
  have laneBaseOrdered : base ≤ laneBase := by
    unfold laneBase KSplitNcFeRows.laneBase
    omega
  have ncBaseOrdered : laneBase ≤ ncBase := by
    unfold laneBase ncBase KSplitNcBlockLaneRows.ncBase
      KSplitNcFeRows.laneBase
    rw [KSplitNcFeRows.auxiliary_count]
    omega
  have rowEnd_eq_laneBase :
      base +
          columns.fe.rowSource.rounds.length *
            (3 * SumCheck.Fe.Drow input) =
        laneBase := by
    unfold laneBase KSplitNcFeRows.laneBase
      KSplitNcFeRows.Columns.rowSource
    rfl
  have laneEnd_eq_ncBase :
      laneBase +
          columns.fe.laneSource.rounds.length * (3 * 2) =
        ncBase := by
    unfold laneBase ncBase KSplitNcBlockLaneRows.ncBase
      KSplitNcFeRows.laneBase
    rw [KSplitNcFeRows.auxiliary_count]
    simp only [KSplitNcFeRows.Columns.laneSource]
    omega
  have laneBasePositive : 0 < laneBase :=
    Nat.lt_of_lt_of_le basePositive laneBaseOrdered
  have ncBasePositive : 0 < ncBase :=
    Nat.lt_of_lt_of_le laneBasePositive ncBaseOrdered
  have afterRowConstant : afterRow 0 = 1 := by
    rw [show afterRow 0 = assignment 0 by
      exact afterFeRow_agrees_below columns base assignment 0 basePositive]
    exact constantWire
  have afterLaneConstant : afterLane 0 = 1 := by
    rw [show afterLane 0 = assignment 0 by
      exact afterFeLane_agrees_below columns base assignment 0 basePositive]
    exact constantWire
  have laneData :=
    columns.fe.laneSource.paperData_eq_of_agreeBelow
      lanePlaced afterRow assignment
      (afterFeRow_agrees_below columns base assignment)
  have ncData :=
    columns.nc.paperData_eq_of_agreeBelow
      ncPlaced afterLane assignment
      (afterFeLane_agrees_below columns base assignment)
  have rowSatisfied :
      Satisfies
        (chainRows
          (carried columns.fe.rowSource.current)
          columns.fe.rowSource.rowRounds
          columns.fe.rowSource.rowChallenges
          (carried columns.fe.rowSource.terminal)
          base)
        afterRow := by
    exact columns.fe.rowSource.numericRows_honest
      base basePositive rowPlaced assignment constantWire rowChain
  have laneSatisfied :
      Satisfies
        (chainRows
          (carried columns.fe.laneSource.current)
          columns.fe.laneSource.rowRounds
          columns.fe.laneSource.rowChallenges
          (carried columns.fe.laneSource.terminal)
          laneBase)
        afterLane := by
    apply columns.fe.laneSource.numericRows_honest
      laneBase laneBasePositive (lanePlaced.mono laneBaseOrdered)
      afterRow afterRowConstant
    simpa only [laneData.1, laneData.2.1, laneData.2.2.1,
      laneData.2.2.2] using laneChain
  have ncSatisfied :
      Satisfies
        (chainRows
          (carried columns.nc.current)
          columns.nc.rowRounds
          columns.nc.rowChallenges
          (carried columns.nc.terminal)
          ncBase)
        final := by
    apply columns.nc.numericRows_honest
      ncBase ncBasePositive
      (ncPlaced.mono (Nat.le_trans laneBaseOrdered ncBaseOrdered))
      afterLane afterLaneConstant
    simpa only [ncData.1, ncData.2.1, ncData.2.2.1,
      ncData.2.2.2] using ncChain
  have rowAtAfterLane :
      Satisfies
        (chainRows
          (carried columns.fe.rowSource.current)
          columns.fe.rowSource.rowRounds
          columns.fe.rowSource.rowChallenges
          (carried columns.fe.rowSource.terminal)
          base)
        afterLane := by
    refine satisfies_extend _ afterRow afterLane ?_ rowSatisfied
    intro row member column mentioned
    symm
    apply chainWitness_off_block
    exact columns.fe.rowSource.numericRows_columns_below_end
      base basePositive rowPlaced row member column mentioned
  have rowAtFinal :
      Satisfies
        (chainRows
          (carried columns.fe.rowSource.current)
          columns.fe.rowSource.rowRounds
          columns.fe.rowSource.rowChallenges
          (carried columns.fe.rowSource.terminal)
          base)
        final := by
    refine satisfies_extend _ afterLane final ?_ rowAtAfterLane
    intro row member column mentioned
    symm
    apply chainWitness_off_block
    have belowLane :=
      columns.fe.rowSource.numericRows_columns_below_end
        base basePositive rowPlaced row member column mentioned
    exact Nat.lt_of_lt_of_le
      (by simpa only [rowEnd_eq_laneBase] using belowLane)
      ncBaseOrdered
  have laneAtFinal :
      Satisfies
        (chainRows
          (carried columns.fe.laneSource.current)
          columns.fe.laneSource.rowRounds
          columns.fe.laneSource.rowChallenges
          (carried columns.fe.laneSource.terminal)
          laneBase)
        final := by
    refine satisfies_extend _ afterLane final ?_ laneSatisfied
    intro row member column mentioned
    symm
    apply chainWitness_off_block
    have belowNc :=
      columns.fe.laneSource.numericRows_columns_below_end
        laneBase laneBasePositive (lanePlaced.mono laneBaseOrdered)
        row member column mentioned
    simpa only [laneEnd_eq_ncBase] using belowNc
  unfold KSplitNcBlockLaneRows.rows KSplitNcFeRows.rows
    KSplitNcNcRows.rows
  exact satisfies_append (satisfies_append rowAtFinal laneAtFinal) ncSatisfied

end Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneHonest
