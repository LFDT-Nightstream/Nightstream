import Nightstream.Implementation.R1CS.Canonical.KQuotientIdentityHonest
import Nightstream.Implementation.R1CS.Canonical.KTraceProgram

/-!
Contract: a concrete occurrence-bound projection collision.

The older `NifsRecipeShape` fixture proves that the abstract frozen event is
nonempty.  This module binds the same phenomenon to one actual
`KTraceProgram.Occurrence`: its coefficient columns, emitted rows, assignment,
and identity list are fixed together.

This is not a reachable NIFS transcript or a production attack.  It proves
only that the occurrence-bound event branch cannot be erased from the
projection row program's deterministic soundness theorem.
-/

set_option autoImplicit false
set_option maxRecDepth 8000

namespace Nightstream.Implementation.R1CS.Canonical.KTraceBadRootFixture

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KHornerHonest
open Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity
open Nightstream.Implementation.R1CS.Canonical.KProjectionTrace
open Nightstream.Implementation.R1CS.Canonical.KTraceDecoder
open Nightstream.Implementation.R1CS.Canonical.KTraceProgram
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.SuperNeo.ProjectionCheck

/-- Column zero is the constant-one wire; every other source column is zero. -/
def sourceAssignment (column : Nat) : Nat :=
  if column = 0 then 1 else 0

def coefficientOne : List Nat :=
  0 :: List.replicate 53 1

def coefficientX : List Nat :=
  1 :: 0 :: List.replicate 52 1

def coefficientZero53 : List Nat :=
  List.replicate 53 1

/-- The quotient form `X = 1 + 0 * Φ₈₁`, checked at `beta = 1`. -/
def trace : KProjectionTrace.Trace where
  beta := ⟨0, 1⟩
  pairs := [⟨coefficientOne, coefficientX⟩]
  output := coefficientOne
  quotient := coefficientZero53
  maxDegree := 106

theorem trace_valid : trace.Valid := by
  decide

/-- The fixed trace is accepted at its selected point but is not a coefficient
identity. -/
theorem trace_badRoot : BadRoot K.ops (trace.identity sourceAssignment) := by
  refine ⟨?_, ?_, ?_⟩
  · decide
  · decide
  · decide

def layout : BatchLayout where
  traces := [trace]
  sharedBeta := trace.beta
  betaShared := by
    intro item member
    simp only [List.mem_singleton] at member
    subst item
    rfl
  valid := by
    intro item member
    simp only [List.mem_singleton] at member
    subst item
    exact trace_valid

def occurrence : Occurrence :=
  ⟨2, layout⟩

/-- The canonical auxiliary assignment for this occurrence. -/
def witness : Nat → Nat :=
  KQuotientIdentity.identityWitness sourceAssignment
    (decodePoint trace.beta) 2 (decodedPairs trace)
    (decodeVector trace.output) (decodeVector trace.quotient) decodeModulus

theorem witness_identity_not_exact :
    ¬ (trace.identity witness).Exact := by
  decide

private theorem coefficientOne_below :
    ∀ column ∈ coefficientOne, column < 2 := by
  intro column member
  simp [coefficientOne] at member
  omega

private theorem coefficientX_below :
    ∀ column ∈ coefficientX, column < 2 := by
  intro column member
  simp [coefficientX] at member
  omega

private theorem coefficientZero53_below :
    ∀ column ∈ coefficientZero53, column < 2 := by
  intro column member
  simp [coefficientZero53] at member
  omega

private theorem betaLow :
    BelowBase (decodePoint trace.beta).low 2 := by
  intro column member
  simp [decodePoint, trace, Mentions] at member
  omega

private theorem betaHigh :
    BelowBase (decodePoint trace.beta).high 2 := by
  intro column member
  simp [decodePoint, trace, Mentions] at member
  omega

private theorem pairsBelow :
    ∀ pair ∈ decodedPairs trace,
      (∀ c ∈ pair.1, BelowBase c.low 2 ∧ BelowBase c.high 2) ∧
      (∀ c ∈ pair.2, BelowBase c.low 2 ∧ BelowBase c.high 2) := by
  intro pair member
  have image :
      pair = (decodeVector coefficientOne, decodeVector coefficientX) := by
    simpa [decodedPairs, trace] using member
  rw [image]
  exact ⟨decodeVector_belowBase coefficientOne 2 coefficientOne_below,
    decodeVector_belowBase coefficientX 2 coefficientX_below⟩

private theorem outputBelow :
    ∀ c ∈ decodeVector trace.output,
      BelowBase c.low 2 ∧ BelowBase c.high 2 := by
  exact decodeVector_belowBase coefficientOne 2 coefficientOne_below

private theorem quotientBelow :
    ∀ c ∈ decodeVector trace.quotient,
      BelowBase c.low 2 ∧ BelowBase c.high 2 := by
  exact decodeVector_belowBase coefficientZero53 2 coefficientZero53_below

private theorem projectedCollision :
    KQuotientIdentity.pairSum
        ((decodedPairs trace).map fun pair =>
          mulPair
            (KQuotientIdentity.projected sourceAssignment
              (decodePoint trace.beta) pair.1)
            (KQuotientIdentity.projected sourceAssignment
              (decodePoint trace.beta) pair.2))
      =
      addPair
        (KQuotientIdentity.projected sourceAssignment
          (decodePoint trace.beta) (decodeVector trace.output))
        (mulPair
          (KQuotientIdentity.projected sourceAssignment
            (decodePoint trace.beta) (decodeVector trace.quotient))
          (KQuotientIdentity.projected sourceAssignment
            (decodePoint trace.beta) decodeModulus)) := by
  rfl

/-- The event branch is reachable from the selected occurrence's own emitted
rows, not merely inhabited by an unrelated identity. -/
theorem occurrence_rows_satisfied :
    Satisfies occurrence.rows witness := by
  have satisfied := KQuotientIdentity.identityRows_honest
    sourceAssignment (decodePoint trace.beta) 2 (decodedPairs trace)
    (decodeVector trace.output) (decodeVector trace.quotient) decodeModulus
    (by decide) (by decide)
    (by
      intro pair member
      have image :
          pair = (decodeVector coefficientOne, decodeVector coefficientX) := by
        simpa [decodedPairs, trace] using member
      rw [image]
      exact ⟨by decide, by decide⟩)
    (by decide) (by decide) decodeModulus_length
    betaLow betaHigh pairsBelow outputBelow quotientBelow
    (decodeModulus_belowBase 2 (by decide)) projectedCollision
  simpa [occurrence, Occurrence.rows, layout, KTraceProgram.rows,
    rowsFrom, witness] using satisfied

theorem witness_constantWire : witness 0 = 1 := by
  rw [witness, KQuotientIdentity.identityWitness_off_block
    sourceAssignment (decodePoint trace.beta) 2 (decodedPairs trace)
    (decodeVector trace.output) (decodeVector trace.quotient) decodeModulus
    0 (by decide)]
  rfl

theorem occurrence_not_exact_at_witness :
    ¬ occurrence.Exact witness := by
  intro exact
  apply witness_identity_not_exact
  exact exact (trace.identity witness) (by
    simp [Occurrence.identities, occurrence, layout,
      KProjectionTrace.BatchIdentity])

/-- The exact occurrence-bound event is reached by a satisfying assignment to
that occurrence's own rows.  It therefore cannot be removed from a generic
soundness contract for this projection program. -/
theorem occurrence_badRoot_of_satisfied_rows :
    occurrence.BadRoot witness := by
  rcases Occurrence.exact_or_badRoot occurrence witness witness_constantWire
      occurrence_rows_satisfied with exact | event
  · exact absurd exact occurrence_not_exact_at_witness
  · exact event

/-- The event-free soundness shape that an unconditional `CallRecipe` would
need if its semantic result asserted exact projection identities. -/
def EventFreeOccurrenceSoundness : Prop :=
  ∀ assignment : Nat → Nat,
    assignment 0 = 1 →
    Satisfies occurrence.rows assignment →
    occurrence.Exact assignment

/-- The current exact-only call contract cannot represent this projection
program: its own satisfying witness takes the occurrence-bound event branch. -/
theorem not_eventFreeOccurrenceSoundness :
    ¬ EventFreeOccurrenceSoundness := by
  intro sound
  exact occurrence_not_exact_at_witness
    (sound witness witness_constantWire occurrence_rows_satisfied)

theorem occurrence_badRoot_at_source :
    occurrence.BadRoot sourceAssignment := by
  refine ⟨trace.identity sourceAssignment, ?_, trace_badRoot⟩
  simp [Occurrence.identities, occurrence, layout, KProjectionTrace.BatchIdentity]

theorem occurrence_not_exact_at_source :
    ¬ occurrence.Exact sourceAssignment :=
  fun exact => (Occurrence.exact_excludes_badRoot exact)
    occurrence_badRoot_at_source

end Nightstream.Implementation.R1CS.Canonical.KTraceBadRootFixture
