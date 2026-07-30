import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.KFixedPhaseSemanticOccurrence
import Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.BlockLane

/-!
Lean-owned R1CS program for the exact degree-four Split-NC NC claimed chain.

Assurance tier: model-level.

Owns: the five-coefficient round program, exact rows and cost, row-satisfaction
soundness, and the bridge to the unchanged block/lane NC claimed-chain
relation.

Does not own: transcript generation, NC terminal construction, packed-output
authority, call-frame decoding, Rust, or generated rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcNcRows

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

/-- The protocol degree is four, hence every physical message has five
constant-first coefficients. -/
abbrev Columns := SourceColumns 4

def rows (columns : Columns) (base : Nat) : List Row :=
  chainRows
    (carried columns.current)
    columns.rowRounds
    columns.rowChallenges
    (carried columns.terminal)
    base

def cost (columns : Columns) : Cost :=
  chainCost 4 columns.rounds.length

theorem rows_length (columns : Columns) (base : Nat) :
    (rows columns base).length = columns.rounds.length * 14 + 2 := by
  unfold rows
  rw [chainRows_length
    (carried columns.current)
    columns.rowRounds
    columns.rowChallenges
    (carried columns.terminal)
    base
    (by
      simpa [SourceColumns.rowRounds, SourceColumns.rowChallenges] using
        columns.sameLength)]
  simp [SourceColumns.rowRounds]

theorem rows_cost (columns : Columns) (base : Nat) :
    (rows columns base).length = (cost columns).recurringRows := by
  rw [rows_length]
  simp [cost, chainCost]

theorem auxiliary_count (columns : Columns) :
    (cost columns).auxiliaryColumns = columns.rounds.length * 12 := by
  simp [cost, chainCost]

/-- Satisfaction reconstructs the exact paper-carrier claimed chain. -/
theorem rows_sound
    (columns : Columns)
    (base : Nat)
    (assignment : Nat -> Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows columns base) assignment) :
    FixedPhase.Chain ops
      (columns.paperCurrent assignment)
      (columns.paperRounds assignment)
      (columns.paperChallenges assignment)
      (columns.paperTerminal assignment) := by
  have rowChain :=
    chainRows_sound assignment constantWire
      (carried columns.current)
      columns.rowRounds
      columns.rowChallenges
      (carried columns.terminal)
      base
      satisfied
  apply chain_of_toProjection
  rw [columns.toProjection_paperCurrent,
    columns.map_paperRounds, columns.map_paperChallenges,
    columns.toProjection_paperTerminal]
  simpa only [decodeCarried_carried] using rowChain

/-- Static serialization equations for one exact NC certificate and
transcript-derived challenge vector. The final selected-call theorem must
derive these equations from decoded columns and transcript rows. -/
structure Agrees
    {domain : BlockNcDomain}
    (columns : Columns)
    (assignment : Nat -> Nat)
    (initial terminal : K)
    (point : Polynomial.Nc.BlockLane.Point domain)
    (certificate : Transcript.Nc.BlockLane.Certificate domain) : Prop where
  initial : columns.paperCurrent assignment = initial
  rounds :
    columns.paperRounds assignment =
      certificate.toSumCheck.rounds
  challenges :
    columns.paperChallenges assignment = point.coordinates
  terminal : columns.paperTerminal assignment = terminal

/-- Exact NC claimed-chain acceptance follows from the emitted rows and the
static source-column projection equations. -/
theorem accepted_of_rows
    {domain : BlockNcDomain}
    (columns : Columns)
    (base : Nat)
    (assignment : Nat -> Nat)
    (constantWire : assignment 0 = 1)
    (initial terminal : K)
    (point : Polynomial.Nc.BlockLane.Point domain)
    (certificate : Transcript.Nc.BlockLane.Certificate domain)
    (agrees : Agrees columns assignment initial terminal point certificate)
    (satisfied : Satisfies (rows columns base) assignment) :
    SumCheck.Nc.Accepted initial point.coordinates terminal
      certificate.toSumCheck := by
  have chain := rows_sound columns base assignment constantWire satisfied
  simpa only [agrees.initial, agrees.rounds, agrees.challenges,
    agrees.terminal] using chain

end Nightstream.Implementation.R1CS.Canonical.KSplitNcNcRows
