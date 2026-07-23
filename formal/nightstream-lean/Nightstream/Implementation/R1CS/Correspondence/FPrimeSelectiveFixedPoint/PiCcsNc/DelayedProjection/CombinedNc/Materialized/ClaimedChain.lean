import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionRound
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Schema
import Nightstream.SuperNeo.SumCheck.FixedPhase

/-!
Claimed-chain semantics for the materialized production combined-NC rounds.

Owns: exact construction of one fixed-width quartic message from each
five-coefficient production round, structural forwarding between adjacent
round maps, and derivation of the complete verifier-visible claimed chain
from the independently proved equations of every round.

Does not own: generated-map validity, source-to-selective refinement, initial
or terminal formula rows, transcript sampling, semantic SumCheck soundness,
raw-child authority, state continuity, commitment binding, costs, or row
removal.

Emits constraints: none.  This is a generic kernel theorem over the concrete
43-column/30-row round vocabulary; generated artifacts are joined in a
separate bounded certificate leaf.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.claimed_chain` | Thread claimed SumCheck values through the exact materialized round chain. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ClaimedChain

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.SuperNeo.SumCheck.Finite

/-- SumCheck operations on the exact quadratic-extension carrier interpreted
by the R1CS projection program. -/
def ops : Ops K where
  zero := K.zero
  one := K.one
  add := K.add
  mul := K.mul

def mappedAssignment (round : RawRoundMap) (assignment : Nat → Nat) :
    Nat → Nat :=
  Relabel.assignment round.columnMap assignment

/-- Exact five-slot message read from one generated round map. -/
def roundMessage (round : RawRoundMap) (assignment : Nat → Nat) :
    FixedPolynomial K ProductionRound.degree where
  coefficients :=
    ProductionRound.coefficientValues (mappedAssignment round assignment)
  coefficients_length := ProductionRound.coefficient_count

def challenge (round : RawRoundMap) (assignment : Nat → Nat) : K :=
  ProductionRound.challengeValue (mappedAssignment round assignment)

def claimIn (round : RawRoundMap) (assignment : Nat → Nat) : K :=
  ProductionRound.claimInValue (mappedAssignment round assignment)

def claimOut (round : RawRoundMap) (assignment : Nat → Nat) : K :=
  ProductionRound.claimOutValue (mappedAssignment round assignment)

def initial : List RawRoundMap → (Nat → Nat) → K
  | [], _ => K.zero
  | round :: _, assignment => claimIn round assignment

def terminal : List RawRoundMap → (Nat → Nat) → K
  | [], _ => K.zero
  | [round], assignment => claimOut round assignment
  | _ :: rounds, assignment => terminal rounds assignment

def challenges (rounds : List RawRoundMap) (assignment : Nat → Nat) :
    List K :=
  rounds.map fun round => challenge round assignment

def certificate (rounds : List RawRoundMap) (assignment : Nat → Nat) :
    FixedPhase.Certificate K ProductionRound.degree where
  rounds := rounds.map fun round => roundMessage round assignment

/-- Adjacent maps forward both extension limbs by literal source-column
reuse.  No value equality is supplied by the caller. -/
def Link (left right : RawRoundMap) : Prop :=
  Relabel.column left.columnMap ProductionRound.claimOutColumns.1 =
      Relabel.column right.columnMap ProductionRound.claimInColumns.1 ∧
    Relabel.column left.columnMap ProductionRound.claimOutColumns.2 =
      Relabel.column right.columnMap ProductionRound.claimInColumns.2

def Linked : List RawRoundMap → Prop
  | [] => True
  | [_] => True
  | left :: right :: rounds => Link left right ∧ Linked (right :: rounds)

/-- Every map's exact 30-row equations have already been derived. -/
def RoundsAccepted (rounds : List RawRoundMap)
    (assignment : Nat → Nat) : Prop :=
  ∀ round ∈ rounds,
    ProductionRound.Accepted (mappedAssignment round assignment)

private theorem evaluateCoefficients_eq_projectionEval
    (coefficients : List K) (point : K) :
    Message.evaluateCoefficients ops point coefficients =
      Nightstream.SuperNeo.ProjectionCheck.eval K.ops coefficients point := by
  induction coefficients with
  | nil => rfl
  | cons coefficient coefficients inductionHypothesis =>
      change K.add coefficient
          (K.mul point
            (Message.evaluateCoefficients ops point coefficients)) =
        K.add coefficient
          (K.mul point
            (Nightstream.SuperNeo.ProjectionCheck.eval K.ops coefficients
              point))
      rw [inductionHypothesis]

theorem roundMessage_evaluate (round : RawRoundMap)
    (assignment : Nat → Nat) (point : K) :
    (roundMessage round assignment).evaluate ops point =
      ProductionRound.polynomial (mappedAssignment round assignment) point := by
  exact evaluateCoefficients_eq_projectionEval
    (ProductionRound.coefficientValues (mappedAssignment round assignment))
    point

private theorem link_value {left right : RawRoundMap}
    {assignment : Nat → Nat} (link : Link left right) :
    claimOut left assignment = claimIn right assignment := by
  rcases link with ⟨low, high⟩
  unfold claimOut claimIn ProductionRound.claimOutValue
    ProductionRound.claimInValue ProductionRound.columns
    mappedAssignment
  simp only [ProjectionProgram.KColumns.value, ProjectionProgram.baseAt,
    Relabel.assignment]
  rw [low, high]

private theorem chain
    (rounds : List RawRoundMap) {assignment : Nat → Nat}
    (linked : Linked rounds)
    (accepted : RoundsAccepted rounds assignment) :
    FixedPhase.Chain ops (initial rounds assignment)
      (certificate rounds assignment).rounds
      (challenges rounds assignment) (terminal rounds assignment) := by
  induction rounds with
  | nil => rfl
  | cons round rounds inductionHypothesis =>
      have roundAccepted :
          ProductionRound.Accepted (mappedAssignment round assignment) :=
        accepted round (by simp)
      change
        initial (round :: rounds) assignment =
            ops.add
              ((roundMessage round assignment).evaluate ops ops.zero)
              ((roundMessage round assignment).evaluate ops ops.one) ∧
          FixedPhase.Chain ops
            ((roundMessage round assignment).evaluate ops
              (challenge round assignment))
            (rounds.map fun next => roundMessage next assignment)
            (rounds.map fun next => challenge next assignment)
            (terminal (round :: rounds) assignment)
      constructor
      · simpa [initial, claimIn, roundMessage_evaluate, ops] using
          roundAccepted.initial
      · cases rounds with
        | nil =>
            simpa [terminal, claimOut, challenge, roundMessage_evaluate] using
              roundAccepted.terminal.symm
        | cons next rest =>
            rcases linked with ⟨roundLink, tailLinked⟩
            have tailAccepted :
                RoundsAccepted (next :: rest) assignment := by
              intro candidate member
              exact accepted candidate (by simp [member])
            have tailChain := inductionHypothesis tailLinked tailAccepted
            have forwarded :
                (roundMessage round assignment).evaluate ops
                    (challenge round assignment) =
                  initial (next :: rest) assignment := by
              calc
                (roundMessage round assignment).evaluate ops
                    (challenge round assignment) =
                    claimOut round assignment := by
                  simpa [challenge, claimOut, roundMessage_evaluate] using
                    roundAccepted.terminal.symm
                _ = claimIn next assignment := link_value roundLink
                _ = initial (next :: rest) assignment := rfl
            rw [forwarded]
            exact tailChain

/-- Exact materialized round equations and literal adjacent-column reuse imply
the complete claimed SumCheck chain.  Initial and terminal values remain the
actual first/last mapped assignment reads; later boundary-row theorems must
identify them with the production combined-NC formulas. -/
theorem accepted
    (rounds : List RawRoundMap) {assignment : Nat → Nat}
    (linked : Linked rounds)
    (roundsAccepted : RoundsAccepted rounds assignment) :
    FixedPhase.Chain ops (initial rounds assignment)
      (certificate rounds assignment).rounds
      (challenges rounds assignment) (terminal rounds assignment) :=
  chain rounds linked roundsAccepted

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ClaimedChain
