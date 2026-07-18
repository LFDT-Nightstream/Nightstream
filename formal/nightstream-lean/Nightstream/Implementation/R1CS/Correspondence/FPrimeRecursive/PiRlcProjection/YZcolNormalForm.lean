import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.YZcolIdentities
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.EvaluationBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.Reduction.Trace
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection

/-!
Conditional Phi81 normal form for the two active PiRLC `y_zcol` identities.

Owns: interpretation of coefficient-wise exactness for the two physical limb
traces as one typed `RingK` PiRLC source aggregate, plus composition with the
exact source-row theorem's named bad-root branch.

Does not own: transcript derivation of the 15 challenges, semantic binding of
the physical challenge/input/output columns, parent commitment opening,
whole-matrix row embedding, bad-root probability, padding, encoded lowering,
or row removal.

Emits constraints: no.

Assurance tier: conditional model-level semantics over the artifact-checked
active traces. `SemanticColumnsMatch` is an explicit future refinement
premise, not an acceptance predicate or authority source.

Authority boundary: exact quotient rows prove the polynomial remainder
statement without giving the quotient semantic authority. The parent/source
claim follows only when every rho ring, paired input ring, and paired output
ring is independently bound to its semantic value.

| Protocol → phase → family | Mathematical obligation | Authority class | Remaining boundary |
|---|---|---|---|
| `identities.y_zcol.normal_form.limb0` | exact low-limb trace is the Phi81 product sum | derived | semantic column binding |
| `identities.y_zcol.normal_form.limb1` | exact high-limb trace is the Phi81 product sum | derived | semantic column binding |
| `identities.y_zcol.normal_form.pair` | pair both limbs into the typed `RingK` fold | derived | none beyond both limb equations |
| `identities.y_zcol.authority.columns` | physical rho/input/output columns equal semantic values | checked premise | transcript and opening refinement |
| `identities.y_zcol.authority.parent` | accepted rows imply exact parent/source binding or a bad root | security boundary | bad-root probability |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolNormalForm

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities.Refinement
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.EvaluationBridge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Reduction
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection
open Nightstream.SuperNeo.ProjectionCheck

def sourceCount : Nat := 15

theorem limb0_pair_count : limb0Trace.pairs.length = sourceCount := by
  decide

theorem limb1_pair_count : limb1Trace.pairs.length = sourceCount := by
  decide

def limb0Pair (index : Fin sourceCount) : ProjectionProgram.PairTrace :=
  limb0Trace.pairs.get (Fin.cast limb0_pair_count.symm index)

def limb1Pair (index : Fin sourceCount) : ProjectionProgram.PairTrace :=
  limb1Trace.pairs.get (Fin.cast limb1_pair_count.symm index)

def lowChallengeRings (assignment : Nat → Nat) : Fin sourceCount → Ring :=
  fun index => values assignment (limb0Pair index).rhoColumns

def highChallengeRings (assignment : Nat → Nat) : Fin sourceCount → Ring :=
  fun index => values assignment (limb1Pair index).rhoColumns

def lowInputRings (assignment : Nat → Nat) : Fin sourceCount → Ring :=
  fun index => values assignment (limb0Pair index).inputColumns

def highInputRings (assignment : Nat → Nat) : Fin sourceCount → Ring :=
  fun index => values assignment (limb1Pair index).inputColumns

def lowOutputRing (assignment : Nat → Nat) : Ring :=
  values assignment limb0Trace.outputColumns

def highOutputRing (assignment : Nat → Nat) : Ring :=
  values assignment limb1Trace.outputColumns

def decodedChallenges (assignment : Nat → Nat) :
    Fin sourceCount → RingF :=
  fun index => ringOfList (lowChallengeRings assignment index)

def decodedInputs (assignment : Nat → Nat) :
    Fin sourceCount → RingK :=
  fun index => pairRings
    (lowInputRings assignment index)
    (highInputRings assignment index)

def decodedOutput (assignment : Nat → Nat) : RingK :=
  pairRings (lowOutputRing assignment) (highOutputRing assignment)

theorem limb0_output_width :
    limb0Trace.outputColumns.length = ringDegree := by
  native_decide

theorem limb1_output_width :
    limb1Trace.outputColumns.length = ringDegree := by
  native_decide

theorem limb0_quotient_width :
    limb0Trace.quotientColumns.length = 53 := by
  native_decide

theorem limb1_quotient_width :
    limb1Trace.quotientColumns.length = 53 := by
  native_decide

theorem limb0_max_degree : limb0Trace.maxDegree = 106 := by
  native_decide

theorem limb1_max_degree : limb1Trace.maxDegree = 106 := by
  native_decide

theorem limb0_pair_widths_indexed (index : Fin sourceCount) :
    (limb0Pair index).rhoColumns.length = ringDegree ∧
      (limb0Pair index).inputColumns.length = ringDegree := by
  apply limb0_pair_widths
  simpa [limb0Pair] using
    (List.get_mem limb0Trace.pairs
      (Fin.cast limb0_pair_count.symm index))

theorem limb1_pair_widths_indexed (index : Fin sourceCount) :
    (limb1Pair index).rhoColumns.length = ringDegree ∧
      (limb1Pair index).inputColumns.length = ringDegree := by
  apply limb1_pair_widths
  simpa [limb1Pair] using
    (List.get_mem limb1Trace.pairs
      (Fin.cast limb1_pair_count.symm index))

/-- Exactness of the low physical trace eliminates its quotient witness and
returns the unique Phi81 remainder in the 54 output columns. -/
theorem limb0_exact_output
    {assignment : Nat → Nat}
    (exact : (limb0Trace.identity assignment).Exact) :
    lowOutputRing assignment =
      phi81Combine (lowChallengeRings assignment)
        (lowInputRings assignment) := by
  simpa [lowOutputRing, lowChallengeRings, lowInputRings, limb0Pair]
    using exact_output_eq_phi81Combine
      (count := sourceCount) assignment limb0Trace limb0_pair_count
      (fun index => (limb0_pair_widths_indexed index).1)
      (fun index => (limb0_pair_widths_indexed index).2)
      limb0_output_width limb0_quotient_width limb0_max_degree exact

/-- High-limb counterpart of `limb0_exact_output`. -/
theorem limb1_exact_output
    {assignment : Nat → Nat}
    (exact : (limb1Trace.identity assignment).Exact) :
    highOutputRing assignment =
      phi81Combine (highChallengeRings assignment)
        (highInputRings assignment) := by
  simpa [highOutputRing, highChallengeRings, highInputRings, limb1Pair]
    using exact_output_eq_phi81Combine
      (count := sourceCount) assignment limb1Trace limb1_pair_count
      (fun index => (limb1_pair_widths_indexed index).1)
      (fun index => (limb1_pair_widths_indexed index).2)
      limb1_output_width limb1_quotient_width limb1_max_degree exact

/-- Both limb traces consume the same 15 physical rho coefficient vectors.
This is fixed-artifact linkage only; transcript authority remains separate. -/
theorem challenge_columns_shared : ∀ index : Fin sourceCount,
    (limb1Pair index).rhoColumns = (limb0Pair index).rhoColumns := by
  native_decide

theorem challenge_rings_shared (assignment : Nat → Nat) :
    highChallengeRings assignment = lowChallengeRings assignment := by
  funext index
  simp [highChallengeRings, lowChallengeRings, challenge_columns_shared]

private theorem exact_limb0_of_batch
    {assignment : Nat → Nat}
    (exact : BatchExact
      (ProjectionProgram.BatchIdentity traces assignment)) :
    (limb0Trace.identity assignment).Exact := by
  apply exact
  simp [ProjectionProgram.BatchIdentity, traces]

private theorem exact_limb1_of_batch
    {assignment : Nat → Nat}
    (exact : BatchExact
      (ProjectionProgram.BatchIdentity traces assignment)) :
    (limb1Trace.identity assignment).Exact := by
  apply exact
  simp [ProjectionProgram.BatchIdentity, traces]

/-- Coefficient-wise exactness of both physical limb identities is exactly
the independent typed `RingK` PiRLC evaluation fold. -/
theorem batchExact_decodedOutput_eq_sourceAggregate
    {assignment : Nat → Nat}
    (exact : BatchExact
      (ProjectionProgram.BatchIdentity traces assignment)) :
    decodedOutput assignment =
      sourceAggregate (decodedChallenges assignment)
        (decodedInputs assignment) := by
  have low := limb0_exact_output (exact_limb0_of_batch exact)
  have high := limb1_exact_output (exact_limb1_of_batch exact)
  rw [challenge_rings_shared assignment] at high
  unfold decodedOutput decodedChallenges decodedInputs sourceAggregate
  rw [low, high]
  exact pairRings_phi81Combine
    (lowChallengeRings assignment)
    (lowInputRings assignment)
    (highInputRings assignment)

/-- Explicit semantic bindings still required above the physical identity
rows. These fields must later be derived from transcript replay, source
decoding, and an authoritative parent opening. -/
structure SemanticColumnsMatch
    (assignment : Nat → Nat)
    (challenges : Fin sourceCount → RingF)
    (inputs : Fin sourceCount → RingK)
    (parent : RingK) : Prop where
  challenges : decodedChallenges assignment = challenges
  inputs : decodedInputs assignment = inputs
  output : decodedOutput assignment = parent

/-- The exact physical batch refines the intended semantic parent transition
once, and only once, the three explicit column bindings are supplied. -/
theorem batchExact_parent_eq_sourceAggregate
    {assignment : Nat → Nat}
    {challenges : Fin sourceCount → RingF}
    {inputs : Fin sourceCount → RingK}
    {parent : RingK}
    (columns : SemanticColumnsMatch assignment challenges inputs parent)
    (exact : BatchExact
      (ProjectionProgram.BatchIdentity traces assignment)) :
    parent = sourceAggregate challenges inputs := by
  calc
    parent = decodedOutput assignment := columns.output.symm
    _ = sourceAggregate (decodedChallenges assignment)
        (decodedInputs assignment) :=
      batchExact_decodedOutput_eq_sourceAggregate exact
    _ = sourceAggregate challenges inputs := by
      rw [columns.challenges, columns.inputs]

/-- Complete local R1CS-to-semantic boundary for the two active `y_zcol`
identities. The failure branch remains the exact sampled bad-root event. -/
theorem completeSourceRows_parent_eq_sourceAggregate_or_badRoot
    {assignment : Nat → Nat}
    {challenges : Fin sourceCount → RingF}
    {inputs : Fin sourceCount → RingK}
    {parent : RingK}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (betaSatisfies : Satisfies betaSourceRows assignment)
    (rhoSatisfies : Satisfies rhoSourceRows assignment)
    (outputSatisfies : Satisfies outputSourceRows assignment)
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment)
    (columns : SemanticColumnsMatch assignment challenges inputs parent) :
    parent = sourceAggregate challenges inputs ∨
      BatchBadRoot ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity traces assignment) := by
  rcases completeSourceRows_batchExact_or_badRoot assignmentCanonical
      constantOne betaSatisfies rhoSatisfies outputSatisfies localSatisfies with
    exact | badRoot
  · exact Or.inl (batchExact_parent_eq_sourceAggregate columns exact)
  · exact Or.inr badRoot

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolNormalForm
