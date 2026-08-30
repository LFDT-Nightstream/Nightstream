import NightstreamFPrime.Layout.Stage1.PiCCSInputs
import NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
import NightstreamFPrime.Lifecycle.Stage1.Accumulator

/-!
Owns the zero-copy Stage 1 accumulator view.

The complete NIFS proof reuses the existing PiCCS round and output wires and
the existing PiDEC child commitment and evaluation wires. The accumulator
output is the existing typed PiDEC running vector. This module allocates no
column or row and defines no second verifier relation.
-/

namespace NightstreamFPrime.Layout.Stage1.AccumulatorInputs

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def piCcsInterface
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  PiCCSInputs.interface logicalWidth publicFits

def running
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (env : Env) :
    Running (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.evalRunning
    (piCcsInterface logicalWidth publicFits) PiCCSInputs.phaseOffset env

def fresh
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (env : Env) :
    Fresh (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.evalFresh
    (piCcsInterface logicalWidth publicFits) PiCCSInputs.phaseOffset env

/-- One complete NIFS proof projected from the canonical phase wires. -/
def proof
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) : Proof (ProductionKey.degreeBound relation) where
  piCcsRounds := fun roundIndex =>
    ((piCcsInterface logicalWidth publicFits).round
      PiCCSInputs.phaseOffset roundIndex).semanticPolynomial env
  piCcsOutput :=
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.evalOutput
      (piCcsInterface logicalWidth publicFits) PiCCSInputs.phaseOffset env
  piDecCommitments :=
    (RunningTransitionInputs.piDecRunningOutput relation env).commitments
  piDecEvaluations :=
    (RunningTransitionInputs.piDecRunningOutput relation env).evaluations

/-- The NIFS output uses the exact typed PiDEC child vector. -/
def output
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    Running (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  RunningTransitionInputs.piDecRunningOutput relation env

@[simp] theorem piCcsEvalProof_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.evalProof relation
        (piCcsInterface logicalWidth publicFits) PiCCSInputs.phaseOffset env
        (proof relation env) =
      proof relation env := by
  rfl

@[simp] theorem proof_piDecCommitments
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    (proof relation env).piDecCommitments =
      (output relation env).commitments := by
  rfl

@[simp] theorem proof_piDecEvaluations
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    (proof relation env).piDecEvaluations =
      (output relation env).evaluations := by
  rfl

end NightstreamFPrime.Layout.Stage1.AccumulatorInputs
