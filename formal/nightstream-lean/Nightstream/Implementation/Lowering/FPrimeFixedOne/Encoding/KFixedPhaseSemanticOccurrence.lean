import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhasePhysicalOccurrence
import Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge

/-!
Contract: bind the physical fixed-phase SumCheck occurrence to authoritative
paper-carrier values decoded directly from its selected source columns.

No semantic equality or acceptance proposition is stored in this carrier.
Current value, round coefficients, challenges, and terminal value are all
decoded from their exact column coordinates.  Soundness reflects the physical
row chain into the unchanged paper `Concrete.K` carrier; honest completeness
transports the exact paper chain back into a constructed physical witness.

Does not own Fiat--Shamir challenge derivation, PiCCS initial/terminal
polynomials, or the enclosing `nifsVerify` call.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckHonest
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhasePhysicalOccurrence
open Nightstream.SuperNeo.SumCheck.Finite

/-- One authoritative extension value, as two existing physical columns. -/
def carried (columns : KColumns) : Carried :=
  ⟨[(columns.c0, 1)], [(columns.c1, 1)]⟩

/-- Decoding the singleton row combinations is exactly the selected
projection-carrier value. -/
theorem decodeCarried_carried
    (assignment : Nat → Nat) (columns : KColumns) :
    decodeCarried assignment (carried columns) =
      columns.value assignment := by
  apply KBridge.toPair_injective
  rw [toPair_decodeCarried]
  rcases columns with ⟨c0, c1⟩
  simp [carried, KHorner.carriedValue, KMul.lcEval_singleton_col,
    KBridge.toPair, KColumns.value, baseAt, ProjectionProgram.residue]

/-- One fixed-width verifier message, with no semantic ghosts. -/
structure RoundColumns (degree : Nat) where
  coefficients : List KColumns
  coefficients_length : coefficients.length = degree + 1

def RoundColumns.rowRound
    {degree : Nat} (round : RoundColumns degree) : Round degree where
  coefficients := round.coefficients.map carried
  coefficients_length := by
    rw [List.length_map, round.coefficients_length]

/-- The paper polynomial decoded from exactly the same columns. -/
def RoundColumns.paperPolynomial
    {degree : Nat} (round : RoundColumns degree)
    (assignment : Nat → Nat) :
    FixedPolynomial Nightstream.SuperNeo.Concrete.K degree where
  coefficients :=
    round.coefficients.map fun columns =>
      ofProjection (columns.value assignment)
  coefficients_length := by
    rw [List.length_map, round.coefficients_length]

private theorem fixedPolynomial_eq_of_coefficients_eq
    {Field : Type} {degree : Nat}
    {left right : FixedPolynomial Field degree}
    (equal : left.coefficients = right.coefficients) :
    left = right := by
  cases left with
  | mk leftCoefficients leftLength =>
      cases right with
      | mk rightCoefficients rightLength =>
          simp only at equal
          subst rightCoefficients
          rfl

theorem RoundColumns.map_paperPolynomial
    {degree : Nat} (round : RoundColumns degree)
    (assignment : Nat → Nat) :
    mapPolynomial (round.paperPolynomial assignment) =
      round.rowRound.polynomial assignment := by
  cases round with
  | mk coefficients coefficientsLength =>
      apply fixedPolynomial_eq_of_coefficients_eq
      simp only [paperPolynomial, mapPolynomial, rowRound, Round.polynomial,
        List.map_map, Function.comp_apply]
      apply List.map_congr_left
      intro columns _
      change
        toProjection (ofProjection (columns.value assignment)) =
          decodeCarried assignment (carried columns)
      rw [toProjection_ofProjection, decodeCarried_carried]

structure SourceColumns (degree : Nat) where
  current : KColumns
  rounds : List (RoundColumns degree)
  challenges : List KColumns
  terminal : KColumns
  sameLength : rounds.length = challenges.length

structure SourceColumns.BelowBase
    {degree : Nat} (source : SourceColumns degree) (base : Nat) : Prop where
  currentLow : source.current.c0 < base
  currentHigh : source.current.c1 < base
  rounds :
    ∀ round ∈ source.rounds, ∀ columns ∈ round.coefficients,
      columns.c0 < base ∧ columns.c1 < base
  challenges :
    ∀ columns ∈ source.challenges,
      columns.c0 < base ∧ columns.c1 < base
  terminalLow : source.terminal.c0 < base
  terminalHigh : source.terminal.c1 < base

/-- Source placement is monotone in the allocation boundary. -/
theorem SourceColumns.BelowBase.mono
    {degree base nextBase : Nat} {source : SourceColumns degree}
    (placed : source.BelowBase base) (ordered : base ≤ nextBase) :
    source.BelowBase nextBase where
  currentLow := Nat.lt_of_lt_of_le placed.currentLow ordered
  currentHigh := Nat.lt_of_lt_of_le placed.currentHigh ordered
  rounds := by
    intro round roundMember columns columnMember
    exact
      ⟨Nat.lt_of_lt_of_le
          (placed.rounds round roundMember columns columnMember).1 ordered,
        Nat.lt_of_lt_of_le
          (placed.rounds round roundMember columns columnMember).2 ordered⟩
  challenges := by
    intro columns member
    exact
      ⟨Nat.lt_of_lt_of_le (placed.challenges columns member).1 ordered,
        Nat.lt_of_lt_of_le (placed.challenges columns member).2 ordered⟩
  terminalLow := Nat.lt_of_lt_of_le placed.terminalLow ordered
  terminalHigh := Nat.lt_of_lt_of_le placed.terminalHigh ordered

def SourceColumns.rowRounds
    {degree : Nat} (source : SourceColumns degree) :
    List (Round degree) :=
  source.rounds.map RoundColumns.rowRound

def SourceColumns.rowChallenges
    {degree : Nat} (source : SourceColumns degree) : List Carried :=
  source.challenges.map carried

def SourceColumns.paperCurrent
    {degree : Nat} (source : SourceColumns degree)
    (assignment : Nat → Nat) : Nightstream.SuperNeo.Concrete.K :=
  ofProjection (source.current.value assignment)

def SourceColumns.paperRounds
    {degree : Nat} (source : SourceColumns degree)
    (assignment : Nat → Nat) :
    List (FixedPolynomial Nightstream.SuperNeo.Concrete.K degree) :=
  source.rounds.map fun round => round.paperPolynomial assignment

def SourceColumns.paperChallenges
    {degree : Nat} (source : SourceColumns degree)
    (assignment : Nat → Nat) : List Nightstream.SuperNeo.Concrete.K :=
  source.challenges.map fun columns =>
    ofProjection (columns.value assignment)

def SourceColumns.paperTerminal
    {degree : Nat} (source : SourceColumns degree)
    (assignment : Nat → Nat) : Nightstream.SuperNeo.Concrete.K :=
  ofProjection (source.terminal.value assignment)

private theorem columns_value_eq_of_agreeBelow
    (columns : KColumns) {base : Nat}
    (low : columns.c0 < base) (high : columns.c1 < base)
    (left right : Nat → Nat)
    (agree : ∀ column, column < base → left column = right column) :
    columns.value left = columns.value right := by
  rcases columns with ⟨c0, c1⟩
  simp only [KColumns.value, baseAt]
  rw [agree c0 low, agree c1 high]

private theorem round_paperPolynomial_eq_of_agreeBelow
    {degree base : Nat} (round : RoundColumns degree)
    (below :
      ∀ columns ∈ round.coefficients,
        columns.c0 < base ∧ columns.c1 < base)
    (left right : Nat → Nat)
    (agree : ∀ column, column < base → left column = right column) :
    round.paperPolynomial left = round.paperPolynomial right := by
  apply fixedPolynomial_eq_of_coefficients_eq
  simp only [RoundColumns.paperPolynomial]
  apply List.map_congr_left
  intro columns member
  apply congrArg ofProjection
  exact columns_value_eq_of_agreeBelow columns
    (below columns member).1 (below columns member).2 left right agree

/-- Every semantic value decoded by one fixed-phase occurrence is unchanged
when two assignments agree below the source boundary.  This is the
composition lemma used when later witness blocks extend an earlier one. -/
theorem SourceColumns.paperData_eq_of_agreeBelow
    {degree base : Nat} (source : SourceColumns degree)
    (placed : source.BelowBase base)
    (left right : Nat → Nat)
    (agree : ∀ column, column < base → left column = right column) :
    source.paperCurrent left = source.paperCurrent right ∧
      source.paperRounds left = source.paperRounds right ∧
      source.paperChallenges left = source.paperChallenges right ∧
      source.paperTerminal left = source.paperTerminal right := by
  constructor
  · apply congrArg ofProjection
    exact columns_value_eq_of_agreeBelow source.current
      placed.currentLow placed.currentHigh left right agree
  constructor
  · unfold SourceColumns.paperRounds
    apply List.map_congr_left
    intro round member
    exact round_paperPolynomial_eq_of_agreeBelow round
      (placed.rounds round member) left right agree
  constructor
  · unfold SourceColumns.paperChallenges
    apply List.map_congr_left
    intro columns member
    apply congrArg ofProjection
    exact columns_value_eq_of_agreeBelow columns
      (placed.challenges columns member).1
      (placed.challenges columns member).2 left right agree
  · apply congrArg ofProjection
    exact columns_value_eq_of_agreeBelow source.terminal
      placed.terminalLow placed.terminalHigh left right agree

private theorem carried_below
    (columns : KColumns) {base : Nat}
    (low : columns.c0 < base) (high : columns.c1 < base) :
    CarriedBelow (carried columns) base := by
  constructor <;> intro column mentioned
  · simp [carried, Mentions] at mentioned
    subst column
    exact low
  · simp [carried, Mentions] at mentioned
    subst column
    exact high

@[simp] theorem SourceColumns.toProjection_paperCurrent
    {degree : Nat} (source : SourceColumns degree)
    (assignment : Nat → Nat) :
    toProjection (source.paperCurrent assignment) =
      source.current.value assignment := by
  simp [SourceColumns.paperCurrent]

@[simp] theorem SourceColumns.toProjection_paperTerminal
    {degree : Nat} (source : SourceColumns degree)
    (assignment : Nat → Nat) :
    toProjection (source.paperTerminal assignment) =
      source.terminal.value assignment := by
  simp [SourceColumns.paperTerminal]

theorem SourceColumns.map_paperRounds
    {degree : Nat} (source : SourceColumns degree)
    (assignment : Nat → Nat) :
    (source.paperRounds assignment).map mapPolynomial =
      source.rowRounds.map fun round => round.polynomial assignment := by
  unfold SourceColumns.paperRounds SourceColumns.rowRounds
  simp only [List.map_map, Function.comp_apply]
  apply List.map_congr_left
  intro round _
  exact round.map_paperPolynomial assignment

theorem SourceColumns.map_paperChallenges
    {degree : Nat} (source : SourceColumns degree)
    (assignment : Nat → Nat) :
    (source.paperChallenges assignment).map toProjection =
      source.rowChallenges.map (decodeCarried assignment) := by
  unfold SourceColumns.paperChallenges SourceColumns.rowChallenges
  simp only [List.map_map, Function.comp_apply]
  apply List.map_congr_left
  intro columns _
  simp only [Function.comp_apply]
  rw [toProjection_ofProjection, decodeCarried_carried]

private theorem round_below
    {degree base : Nat} (source : SourceColumns degree)
    (placed : source.BelowBase base)
    (round : RoundColumns degree) (member : round ∈ source.rounds) :
    RoundBelow round.rowRound base := by
  intro value valueMember
  rcases List.mem_map.mp valueMember with ⟨columns, columnsMember, rfl⟩
  exact carried_below columns
    (placed.rounds round member columns columnsMember).1
    (placed.rounds round member columns columnsMember).2

/-- Construct the physical occurrence from source columns and a fresh
allocation boundary. -/
def SourceColumns.physical
    {degree : Nat} (source : SourceColumns degree)
    (base : Nat) (sourceOwner owner : PhysicalOwner)
    (firstOrdinal : Nat) (basePositive : 0 < base)
    (placed : source.BelowBase base) :
    PhysicalOccurrence (carried source.current)
      source.rowRounds source.rowChallenges (carried source.terminal) base where
  sourceOwner := sourceOwner
  owner := owner
  firstOrdinal := firstOrdinal
  basePositive := basePositive
  currentBelow :=
    carried_below source.current placed.currentLow placed.currentHigh
  roundsBelow := by
    intro round member
    rcases List.mem_map.mp member with ⟨columns, columnsMember, rfl⟩
    exact round_below source placed columns columnsMember
  challengesBelow := by
    intro challenge member
    rcases List.mem_map.mp member with ⟨columns, columnsMember, rfl⟩
    exact carried_below columns
      (placed.challenges columns columnsMember).1
      (placed.challenges columns columnsMember).2
  terminalBelow :=
    carried_below source.terminal placed.terminalLow placed.terminalHigh
  sameLength := by
    simpa [SourceColumns.rowRounds, SourceColumns.rowChallenges] using
      source.sameLength

/-- **Physical soundness reaches the unchanged paper fixed-phase chain.**

The only hypotheses are physical placement, the constant wire, and row
satisfaction.  There is no caller-supplied transcript interpretation. -/
theorem SourceColumns.rows_sound
    {degree : Nat} (source : SourceColumns degree)
    (base : Nat) (sourceOwner owner : PhysicalOwner)
    (firstOrdinal : Nat) (basePositive : 0 < base)
    (placed : source.BelowBase base)
    (assignment : ColumnId → Nightstream.SuperNeo.Concrete.F)
    (constantWire :
      assignment
        { owner := .prelude, bundleIndex := 0, coordinateIndex := 0 } = 1)
    (satisfied :
      Satisfies
        (source.physical base sourceOwner owner firstOrdinal
          basePositive placed).rows assignment) :
    let physical :=
      source.physical base sourceOwner owner firstOrdinal basePositive placed
    let numeric := numericAssignment physical.map assignment
    FixedPhase.Chain
      Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.extensionOps.toOps
      (source.paperCurrent numeric)
      (source.paperRounds numeric)
      (source.paperChallenges numeric)
      (source.paperTerminal numeric) := by
  dsimp only
  let physical :=
    source.physical base sourceOwner owner firstOrdinal basePositive placed
  let numeric := numericAssignment physical.map assignment
  have rowChain := physical.rows_sound assignment constantWire satisfied
  apply chain_of_toProjection
  rw [source.toProjection_paperCurrent,
    source.map_paperRounds, source.map_paperChallenges,
    source.toProjection_paperTerminal]
  simpa only [decodeCarried_carried] using rowChain

/-- Honest paper executions construct the explicit numeric witness used by
larger canonical row programs.  Unlike `SourceColumns.rows_honest`, this
statement keeps the numeric namespace visible so a caller can extend the
witness with later disjoint allocations before translating the final
assignment into typed columns. -/
theorem SourceColumns.numericRows_honest
    {degree : Nat} (source : SourceColumns degree)
    (base : Nat) (basePositive : 0 < base)
    (placed : source.BelowBase base)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (chain :
      FixedPhase.Chain
        Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.extensionOps.toOps
        (source.paperCurrent assignment)
        (source.paperRounds assignment)
        (source.paperChallenges assignment)
        (source.paperTerminal assignment)) :
    Satisfies
      (chainRows
        (carried source.current)
        source.rowRounds
        source.rowChallenges
        (carried source.terminal)
        base)
      (chainWitness assignment source.rowRounds source.rowChallenges base) := by
  apply chainWitness_satisfies assignment constantWire
  · exact basePositive
  · exact carried_below source.current
      placed.currentLow placed.currentHigh
  · intro round member
    rcases List.mem_map.mp member with
      ⟨columns, columnsMember, rfl⟩
    exact round_below source placed columns columnsMember
  · intro challenge member
    rcases List.mem_map.mp member with
      ⟨columns, columnsMember, rfl⟩
    exact carried_below columns
      (placed.challenges columns columnsMember).1
      (placed.challenges columns columnsMember).2
  · exact carried_below source.terminal
      placed.terminalLow placed.terminalHigh
  · have rowChain :=
      chain_toProjection
        (source.paperCurrent assignment)
        (source.paperTerminal assignment)
        (source.paperRounds assignment)
        (source.paperChallenges assignment)
        chain
    rw [source.toProjection_paperCurrent,
      source.map_paperRounds, source.map_paperChallenges,
      source.toProjection_paperTerminal] at rowChain
    simpa only [decodeCarried_carried] using rowChain

/-- Every column mentioned by the numeric occurrence lies below the end of
its exact Horner allocation. -/
theorem SourceColumns.numericRows_columns_below_end
    {degree : Nat} (source : SourceColumns degree)
    (base : Nat) (basePositive : 0 < base)
    (placed : source.BelowBase base)
    (row : Nightstream.Implementation.R1CS.Row)
    (member :
      row ∈
        chainRows
          (carried source.current)
          source.rowRounds
          source.rowChallenges
          (carried source.terminal)
          base)
    (column : Nat)
    (mentioned :
      Mentions row.a column ∨ Mentions row.b column ∨
        Mentions row.c column) :
    column < base + source.rounds.length * (3 * degree) := by
  have bound :=
    KFixedPhaseSumCheckSupport.chainRows_columns_below_end
      (carried source.current)
      source.rowRounds
      source.rowChallenges
      (carried source.terminal)
      base
      basePositive
      (carried_below source.current
        placed.currentLow placed.currentHigh)
      (by
        intro round roundMember
        rcases List.mem_map.mp roundMember with
          ⟨columns, columnsMember, rfl⟩
        exact round_below source placed columns columnsMember)
      (by
        intro challenge challengeMember
        rcases List.mem_map.mp challengeMember with
          ⟨columns, columnsMember, rfl⟩
        exact carried_below columns
          (placed.challenges columns columnsMember).1
          (placed.challenges columns columnsMember).2)
      (carried_below source.terminal
        placed.terminalLow placed.terminalHigh)
      (by
        simpa [SourceColumns.rowRounds, SourceColumns.rowChallenges] using
          source.sameLength)
      row member column mentioned
  simpa [SourceColumns.rowRounds] using bound

/-- Honest paper executions construct satisfying typed rows over the same
authoritative source coordinates. -/
theorem SourceColumns.rows_honest
    {degree : Nat} (source : SourceColumns degree)
    (base : Nat) (sourceOwner owner : PhysicalOwner)
    (firstOrdinal : Nat) (basePositive : 0 < base)
    (placed : source.BelowBase base)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (chain :
      FixedPhase.Chain
        Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.extensionOps.toOps
        (source.paperCurrent assignment)
        (source.paperRounds assignment)
        (source.paperChallenges assignment)
        (source.paperTerminal assignment)) :
    ∃ completed : ColumnId → Nightstream.SuperNeo.Concrete.F,
      Satisfies
        (source.physical base sourceOwner owner firstOrdinal
          basePositive placed).rows completed := by
  let physical :=
    source.physical base sourceOwner owner firstOrdinal basePositive placed
  apply physical.rows_honest assignment constantWire
  have rowChain :=
    chain_toProjection
      (source.paperCurrent assignment)
      (source.paperTerminal assignment)
      (source.paperRounds assignment)
      (source.paperChallenges assignment)
      chain
  rw [source.toProjection_paperCurrent,
    source.map_paperRounds, source.map_paperChallenges,
    source.toProjection_paperTerminal] at rowChain
  simpa only [decodeCarried_carried] using rowChain

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence
