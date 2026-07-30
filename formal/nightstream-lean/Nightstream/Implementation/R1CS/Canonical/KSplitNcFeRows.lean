import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence
import Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Interface

/-!
Lean-owned mixed-width R1CS program for the Split-NC FE claimed chain.

Assurance tier: model-level.

Owns: the row-degree prefix, the exact degree-two lane suffix, their shared
boundary value, exact emitted rows and cost, row-satisfaction soundness, and
the bridge to the unchanged mixed-width FE claimed-chain relation.

Does not own: transcript generation, the FE initial or terminal formulas,
call-frame decoding, output authority, Rust, or generated rows.

The lane suffix is encoded at its physical three-coefficient width. It is not
widened to the row degree before emitting rows; widening exists only in the
semantic bridge to the generic fixed-phase verifier.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcFeRows

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps.toOps
private abbrev KColumns :=
  Nightstream.Implementation.R1CS.ProjectionProgram.KColumns

/-- Exact source columns for the mixed-width FE chain. The boundary is named
once and is consumed as the terminal of the row phase and the initial value of
the lane phase. -/
structure Columns (rowDegree : Nat) where
  initial : KColumns
  rowRounds : List (RoundColumns rowDegree)
  rowChallenges : List KColumns
  rowSameLength : rowRounds.length = rowChallenges.length
  boundary : KColumns
  laneRounds : List (RoundColumns 2)
  laneChallenges : List KColumns
  laneSameLength : laneRounds.length = laneChallenges.length
  terminal : KColumns

def Columns.rowSource
    {rowDegree : Nat} (columns : Columns rowDegree) :
    SourceColumns rowDegree where
  current := columns.initial
  rounds := columns.rowRounds
  challenges := columns.rowChallenges
  terminal := columns.boundary
  sameLength := columns.rowSameLength

def Columns.laneSource
    {rowDegree : Nat} (columns : Columns rowDegree) :
    SourceColumns 2 where
  current := columns.boundary
  rounds := columns.laneRounds
  challenges := columns.laneChallenges
  terminal := columns.terminal
  sameLength := columns.laneSameLength

/-- The lane Horner allocation begins immediately after the row-prefix
allocation. -/
def laneBase
    {rowDegree : Nat} (columns : Columns rowDegree) (base : Nat) : Nat :=
  base + columns.rowRounds.length * (3 * rowDegree)

/-- Exact mixed-width numeric program: one fixed-width row chain followed by
one degree-two lane chain. -/
def rows
    {rowDegree : Nat} (columns : Columns rowDegree) (base : Nat) :
    List Row :=
  chainRows
      (carried columns.initial)
      columns.rowSource.rowRounds
      columns.rowSource.rowChallenges
      (carried columns.boundary)
      base ++
    chainRows
      (carried columns.boundary)
      columns.laneSource.rowRounds
      columns.laneSource.rowChallenges
      (carried columns.terminal)
      (laneBase columns base)

/-- Exact mixed-width cost. Source/message columns are shared reads; only the
two Horner intervals are allocated here. -/
def cost
    {rowDegree : Nat} (columns : Columns rowDegree) : Cost :=
  chainCost rowDegree columns.rowRounds.length +
    chainCost 2 columns.laneRounds.length

theorem rows_length
    {rowDegree : Nat} (columns : Columns rowDegree) (base : Nat) :
    (rows columns base).length =
      columns.rowRounds.length * (3 * rowDegree + 2) + 2 +
        (columns.laneRounds.length * 8 + 2) := by
  unfold rows
  rw [List.length_append,
    chainRows_length
      (carried columns.initial)
      columns.rowSource.rowRounds
      columns.rowSource.rowChallenges
      (carried columns.boundary)
      base
      (by
        simpa [Columns.rowSource, SourceColumns.rowRounds,
          SourceColumns.rowChallenges] using columns.rowSameLength),
    chainRows_length
      (carried columns.boundary)
      columns.laneSource.rowRounds
      columns.laneSource.rowChallenges
      (carried columns.terminal)
      (laneBase columns base)
      (by
        simpa [Columns.laneSource, SourceColumns.rowRounds,
          SourceColumns.rowChallenges] using columns.laneSameLength)]
  simp only [Columns.rowSource, Columns.laneSource, SourceColumns.rowRounds,
    List.length_map]

theorem rows_cost
    {rowDegree : Nat} (columns : Columns rowDegree) (base : Nat) :
    (rows columns base).length = (cost columns).recurringRows := by
  rw [rows_length]
  simp [cost, chainCost]

theorem auxiliary_count
    {rowDegree : Nat} (columns : Columns rowDegree) :
    (cost columns).auxiliaryColumns =
      columns.rowRounds.length * (3 * rowDegree) +
        columns.laneRounds.length * 6 := by
  simp [cost, chainCost]

private theorem satisfies_append_left
    {left right : List Row} {assignment : Nat -> Nat}
    (satisfied : Satisfies (left ++ right) assignment) :
    Satisfies left assignment :=
  fun row member => satisfied row (List.mem_append_left _ member)

private theorem satisfies_append_right
    {left right : List Row} {assignment : Nat -> Nat}
    (satisfied : Satisfies (left ++ right) assignment) :
    Satisfies right assignment :=
  fun row member => satisfied row (List.mem_append_right _ member)

private theorem source_chain_sound_at
    {degree base : Nat}
    (source : SourceColumns degree)
    (assignment : Nat -> Nat)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (chainRows
          (carried source.current)
          source.rowRounds
          source.rowChallenges
          (carried source.terminal)
          base)
        assignment) :
    FixedPhase.Chain ops
      (source.paperCurrent assignment)
      (source.paperRounds assignment)
      (source.paperChallenges assignment)
      (source.paperTerminal assignment) := by
  have rowChain :=
    chainRows_sound assignment constantWire
      (carried source.current)
      source.rowRounds
      source.rowChallenges
      (carried source.terminal)
      base
      satisfied
  apply chain_of_toProjection
  rw [source.toProjection_paperCurrent,
    source.map_paperRounds, source.map_paperChallenges,
    source.toProjection_paperTerminal]
  simpa only [decodeCarried_carried] using rowChain

/-- Satisfaction determines both exact fixed-phase chains. No initial,
boundary, terminal, or challenge value is supplied as a semantic premise. -/
theorem rows_sound
    {rowDegree : Nat} (columns : Columns rowDegree) (base : Nat)
    (assignment : Nat -> Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows columns base) assignment) :
    FixedPhase.Chain ops
        (columns.rowSource.paperCurrent assignment)
        (columns.rowSource.paperRounds assignment)
        (columns.rowSource.paperChallenges assignment)
        (columns.rowSource.paperTerminal assignment) /\
      FixedPhase.Chain ops
        (columns.laneSource.paperCurrent assignment)
        (columns.laneSource.paperRounds assignment)
        (columns.laneSource.paperChallenges assignment)
        (columns.laneSource.paperTerminal assignment) := by
  exact
    ⟨source_chain_sound_at columns.rowSource assignment constantWire
        (satisfies_append_left satisfied),
      source_chain_sound_at columns.laneSource assignment constantWire
        (satisfies_append_right satisfied)⟩

private theorem chain_append
    {Field : Type} {degree : Nat}
    (fieldOps : Ops Field)
    (middle terminal : Field) :
    forall
      (current : Field)
      (left right : List (FixedPolynomial Field degree))
      (leftChallenges rightChallenges : List Field),
      FixedPhase.Chain fieldOps current left leftChallenges middle ->
      FixedPhase.Chain fieldOps middle right rightChallenges terminal ->
      FixedPhase.Chain fieldOps current (left ++ right)
        (leftChallenges ++ rightChallenges) terminal
  | current, [], right, [], rightChallenges, leftChain, rightChain => by
      simp only [FixedPhase.Chain] at leftChain
      simpa [leftChain] using rightChain
  | _, [], _, _ :: _, _, leftChain, _ => by
      simp [FixedPhase.Chain] at leftChain
  | _, _ :: _, _, [], _, leftChain, _ => by
      simp [FixedPhase.Chain] at leftChain
  | current, polynomial :: left, right,
      challenge :: leftChallenges, rightChallenges,
      leftChain, rightChain => by
      simp only [FixedPhase.Chain] at leftChain
      simp only [List.cons_append, FixedPhase.Chain]
      exact
        ⟨leftChain.1,
          chain_append fieldOps middle terminal
            (polynomial.evaluate fieldOps challenge)
            left right leftChallenges rightChallenges
            leftChain.2 rightChain⟩

private theorem lane_chain_widen
    {shape : SemanticShape}
    (input : PublicInput shape)
    (current terminal : K)
    (rounds : List SumCheck.Fe.LaneMessage)
    (challenges : List K) :
    FixedPhase.Chain ops current rounds challenges terminal ->
      FixedPhase.Chain ops current
        (rounds.map (SumCheck.Fe.laneToUniform input))
        challenges terminal := by
  intro chain
  induction rounds generalizing current challenges with
  | nil =>
      cases challenges with
      | nil => simpa [FixedPhase.Chain] using chain
      | cons _ _ => simp [FixedPhase.Chain] at chain
  | cons round rounds inductionHypothesis =>
      cases challenges with
      | nil => simp [FixedPhase.Chain] at chain
      | cons challenge challenges =>
          simp only [FixedPhase.Chain] at chain
          simp only [List.map_cons, FixedPhase.Chain]
          constructor
          · simpa only [
              SumCheck.Fe.lane_evaluate_uniform input round ops.zero,
              SumCheck.Fe.lane_evaluate_uniform input round ops.one] using
                chain.1
          · rw [SumCheck.Fe.lane_evaluate_uniform input round challenge]
            exact inductionHypothesis
              (current := round.evaluate ops challenge)
              (challenges := challenges)
              chain.2

private theorem chain_split
    {Field : Type} {degree : Nat}
    (fieldOps : Ops Field)
    (current terminal : Field)
    (left right : List (FixedPolynomial Field degree))
    (leftChallenges rightChallenges : List Field)
    (sameLength : left.length = leftChallenges.length)
    (chain :
      FixedPhase.Chain fieldOps current (left ++ right)
        (leftChallenges ++ rightChallenges) terminal) :
    ∃ middle,
      FixedPhase.Chain fieldOps current left leftChallenges middle ∧
      FixedPhase.Chain fieldOps middle right rightChallenges terminal := by
  induction left generalizing current leftChallenges with
  | nil =>
      cases leftChallenges with
      | nil =>
          exact ⟨current, rfl, chain⟩
      | cons _ _ =>
          simp only [List.length_nil, List.length_cons] at sameLength
          omega
  | cons polynomial polynomials inductionHypothesis =>
      cases leftChallenges with
      | nil =>
          simp only [List.length_cons, List.length_nil] at sameLength
          omega
      | cons challenge challenges =>
          simp only [List.length_cons, Nat.succ.injEq] at sameLength
          simp only [List.cons_append, FixedPhase.Chain] at chain
          rcases inductionHypothesis
              (current := polynomial.evaluate fieldOps challenge)
              (leftChallenges := challenges)
              sameLength chain.2 with
            ⟨middle, leftChain, rightChain⟩
          exact ⟨middle, ⟨chain.1, leftChain⟩, rightChain⟩

private theorem lane_chain_narrow
    {shape : SemanticShape}
    (input : PublicInput shape)
    (current terminal : K)
    (rounds : List SumCheck.Fe.LaneMessage)
    (challenges : List K) :
    FixedPhase.Chain ops current
        (rounds.map (SumCheck.Fe.laneToUniform input))
        challenges terminal ->
      FixedPhase.Chain ops current rounds challenges terminal := by
  intro chain
  induction rounds generalizing current challenges with
  | nil =>
      cases challenges with
      | nil => simpa [FixedPhase.Chain] using chain
      | cons _ _ => simp [FixedPhase.Chain] at chain
  | cons round rounds inductionHypothesis =>
      cases challenges with
      | nil => simp [FixedPhase.Chain] at chain
      | cons challenge challenges =>
          simp only [List.map_cons, FixedPhase.Chain] at chain
          simp only [FixedPhase.Chain]
          constructor
          · simpa only [
              SumCheck.Fe.lane_evaluate_uniform input round ops.zero,
              SumCheck.Fe.lane_evaluate_uniform input round ops.one] using
                chain.1
          · rw [← SumCheck.Fe.lane_evaluate_uniform input round challenge]
            exact inductionHypothesis
              (current :=
                (SumCheck.Fe.laneToUniform input round).evaluate
                  ops challenge)
              (challenges := challenges)
              chain.2

/-- Exact inverse of the semantic row/lane concatenation used by FE
acceptance. An accepted mixed-width certificate determines one concrete
boundary value together with the two physical fixed-phase chains. -/
theorem accepted_splits
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (initial terminal : K)
    (point : Polynomial.Fe.Point shape domain)
    (certificate : SumCheck.Fe.Certificate input domain)
    (accepted :
      SumCheck.Fe.Accepted initial terminal point certificate) :
    ∃ boundary,
      FixedPhase.Chain ops initial
          (List.ofFn certificate.rowRounds)
          point.row.coordinates boundary ∧
        FixedPhase.Chain ops boundary
          (List.ofFn certificate.laneRounds)
          point.lane.coordinates terminal := by
  unfold SumCheck.Fe.Accepted at accepted
  have rowLength :
      (List.ofFn certificate.rowRounds).length =
        point.row.coordinates.length := by
    simp [point.row.dimension]
  rcases chain_split ops initial terminal
      (List.ofFn certificate.rowRounds)
      ((List.ofFn certificate.laneRounds).map
        (SumCheck.Fe.laneToUniform input))
      point.row.coordinates point.lane.coordinates
      rowLength
      (by
        simpa only [SumCheck.Fe.Certificate.uniformRounds,
          Polynomial.Fe.Point.coordinates] using accepted) with
    ⟨boundary, rowChain, laneChain⟩
  exact
    ⟨boundary, rowChain,
      lane_chain_narrow input boundary terminal
        (List.ofFn certificate.laneRounds)
        point.lane.coordinates laneChain⟩

/-- Deterministic result of replaying one claimed-chain prefix.  The
mismatched-width branches are total only so the function is usable before a
chain proof is available; an accepted chain never reaches them. -/
def chainTerminal
    {Field : Type} {degree : Nat}
    (fieldOps : Ops Field) :
    Field → List (FixedPolynomial Field degree) → List Field → Field
  | current, [], [] => current
  | current, polynomial :: polynomials, challenge :: challenges =>
      chainTerminal fieldOps
        (polynomial.evaluate fieldOps challenge) polynomials challenges
  | current, _, _ => current

private theorem chainTerminal_eq
    {Field : Type} {degree : Nat}
    (fieldOps : Ops Field)
    (current terminal : Field)
    (rounds : List (FixedPolynomial Field degree))
    (challenges : List Field)
    (chain : FixedPhase.Chain fieldOps current rounds challenges terminal) :
    chainTerminal fieldOps current rounds challenges = terminal := by
  induction rounds generalizing current challenges with
  | nil =>
      cases challenges with
      | nil => simpa [chainTerminal, FixedPhase.Chain] using chain
      | cons _ _ => simp [FixedPhase.Chain] at chain
  | cons polynomial polynomials inductionHypothesis =>
      cases challenges with
      | nil => simp [FixedPhase.Chain] at chain
      | cons challenge challenges =>
          simp only [FixedPhase.Chain] at chain
          simp only [chainTerminal]
          exact inductionHypothesis
            (current := polynomial.evaluate fieldOps challenge)
            (challenges := challenges) chain.2

/-- The verifier-determined claim at the physical FE row/lane boundary. -/
def boundaryValue
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (initial : K)
    (point : Polynomial.Fe.Point shape domain)
    (certificate : SumCheck.Fe.Certificate input domain) : K :=
  chainTerminal ops initial
    (List.ofFn certificate.rowRounds) point.row.coordinates

/-- Exact FE acceptance splits at the deterministic boundary value, not at a
caller-chosen carried claim. -/
theorem accepted_splits_at_boundaryValue
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (initial terminal : K)
    (point : Polynomial.Fe.Point shape domain)
    (certificate : SumCheck.Fe.Certificate input domain)
    (accepted : SumCheck.Fe.Accepted initial terminal point certificate) :
    FixedPhase.Chain ops initial
        (List.ofFn certificate.rowRounds)
        point.row.coordinates
        (boundaryValue initial point certificate) ∧
      FixedPhase.Chain ops
        (boundaryValue initial point certificate)
        (List.ofFn certificate.laneRounds)
        point.lane.coordinates terminal := by
  rcases accepted_splits initial terminal point certificate accepted with
    ⟨boundary, rowChain, laneChain⟩
  have boundaryEqual :
      boundaryValue initial point certificate = boundary := by
    exact chainTerminal_eq ops initial boundary
      (List.ofFn certificate.rowRounds) point.row.coordinates rowChain
  rw [boundaryEqual]
  exact ⟨rowChain, laneChain⟩

/-- Static equations connecting the emitted source columns to one exact FE
certificate and verifier-derived point.

This structure carries only serialization equalities. The final selected-call
theorem must derive it from call-frame decoding and transcript rows. -/
structure Agrees
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (columns : Columns (SumCheck.Fe.Drow input))
    (assignment : Nat -> Nat)
    (initial terminal : K)
    (point : Polynomial.Fe.Point shape domain)
    (certificate : SumCheck.Fe.Certificate input domain) : Prop where
  initial :
    columns.rowSource.paperCurrent assignment = initial
  rowRounds :
    columns.rowSource.paperRounds assignment =
      List.ofFn certificate.rowRounds
  rowChallenges :
    columns.rowSource.paperChallenges assignment =
      point.row.coordinates
  boundary :
    columns.rowSource.paperTerminal assignment =
      columns.laneSource.paperCurrent assignment
  laneRounds :
    columns.laneSource.paperRounds assignment =
      List.ofFn certificate.laneRounds
  laneChallenges :
    columns.laneSource.paperChallenges assignment =
      point.lane.coordinates
  terminal :
    columns.laneSource.paperTerminal assignment = terminal

/-- The exact mixed-width FE claimed-chain relation follows from emitted rows
once the static codec/transcript projections are connected to those same
columns. No claimed-chain equation is an `Agrees` field. -/
theorem accepted_of_rows
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {input : PublicInput shape}
    (columns : Columns (SumCheck.Fe.Drow input))
    (base : Nat)
    (assignment : Nat -> Nat)
    (constantWire : assignment 0 = 1)
    (initial terminal : K)
    (point : Polynomial.Fe.Point shape domain)
    (certificate : SumCheck.Fe.Certificate input domain)
    (agrees :
      Agrees columns assignment initial terminal point certificate)
    (satisfied : Satisfies (rows columns base) assignment) :
    SumCheck.Fe.Accepted initial terminal point certificate := by
  rcases rows_sound columns base assignment constantWire satisfied with
    ⟨rowChain, laneChain⟩
  rw [agrees.initial, agrees.rowRounds, agrees.rowChallenges] at rowChain
  rw [← agrees.boundary, agrees.laneRounds, agrees.laneChallenges,
    agrees.terminal] at laneChain
  have widenedLane :
      FixedPhase.Chain ops
        (columns.rowSource.paperTerminal assignment)
        ((List.ofFn certificate.laneRounds).map
          (SumCheck.Fe.laneToUniform input))
        point.lane.coordinates terminal :=
    lane_chain_widen input
      (columns.rowSource.paperTerminal assignment)
      terminal
      (List.ofFn certificate.laneRounds)
      point.lane.coordinates
      laneChain
  have combined :
      FixedPhase.Chain ops initial
        (List.ofFn certificate.rowRounds ++
          (List.ofFn certificate.laneRounds).map
            (SumCheck.Fe.laneToUniform input))
        (point.row.coordinates ++ point.lane.coordinates)
        terminal :=
    chain_append ops
      (columns.rowSource.paperTerminal assignment)
      terminal initial
      (List.ofFn certificate.rowRounds)
      ((List.ofFn certificate.laneRounds).map
        (SumCheck.Fe.laneToUniform input))
      point.row.coordinates point.lane.coordinates
      rowChain widenedLane
  simpa only [SumCheck.Fe.Accepted,
    SumCheck.Fe.Certificate.uniformRounds,
    Polynomial.Fe.Point.coordinates] using combined

end Nightstream.Implementation.R1CS.Canonical.KSplitNcFeRows
