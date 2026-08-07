import Nightstream.Checks.Common
import Nightstream.HyperNova.Construction2.State
import Nightstream.Protocol.FPrime.Step
import Nightstream.SuperNeo.Concrete.Parameters
import Nightstream.SuperNeo.ProjectionCheck
import Nightstream.SuperNeo.SumCheck

namespace Nightstream.Checks.Protocol

namespace Parameters

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete

/-- Executable cross-checks for the concrete Appendix-B.2 profile. -/
def probes : List Nightstream.Checks.Probe :=
  [ ⟨"params_goldilocks_q", fun _ => productionGlobalParams.q == 18446744069414584321, true⟩
  , ⟨"params_fresh_bound", fun _ => productionGlobalParams.b == 2, true⟩
  , ⟨"params_decomposition_arity", fun _ => productionGlobalParams.k == 14, true⟩
  , ⟨"params_max_fresh", fun _ => productionGlobalParams.maxFresh == 61, true⟩
  , ⟨"params_big_b", fun _ => productionGlobalParams.bigB == 16384, true⟩
  , ⟨"params_expansion_t", fun _ => productionGlobalParams.expansionT == 216, true⟩
  ]

end Parameters

namespace M2Probe

open Nightstream.SuperNeo.SumCheck

def ops : Ops Nat Nat where
  zero := 0
  one := 1
  add := Nat.add

def expected : Nat → Nat
  | 0 => 2
  | 1 => 3
  | _ => 7

def forged : Nat → Nat
  | 0 => 4
  | 1 => 4
  | _ => 7

def honestTranscript : Instance Nat Nat where
  claimedInitial := 5
  trueInitial := 5
  terminal := 7
  rounds := [{ claimed := expected, expected := expected, challenge := 2, degree := 2 }]
  maxDegree := 2
  challengeSetSize := 97

/-- False claim which passes only because the polynomials collide at challenge two. -/
def forgedTranscript : Instance Nat Nat where
  claimedInitial := 8
  trueInitial := 5
  terminal := 7
  rounds := [{ claimed := forged, expected := expected, challenge := 2, degree := 2 }]
  maxDegree := 2
  challengeSetSize := 97

def malformedTranscript : Instance Nat Nat :=
  { forgedTranscript with claimedInitial := 9 }

end M2Probe

namespace Folding

/-- M2 probes keep acceptance visibly separate from claim truth. -/
def probes : List Nightstream.Checks.Probe :=
  [ ⟨"sumcheck_honest_accepts",
      fun _ => Nightstream.SuperNeo.SumCheck.check M2Probe.ops M2Probe.honestTranscript, true⟩
  , ⟨"sumcheck_forged_collision_acceptance_observed",
      fun _ => Nightstream.SuperNeo.SumCheck.check M2Probe.ops M2Probe.forgedTranscript, true⟩
  , ⟨"sumcheck_forged_claim_truth_is_false",
      fun _ => decide
        (M2Probe.forgedTranscript.claimedInitial = M2Probe.forgedTranscript.trueInitial), false⟩
  , ⟨"sumcheck_malformed_chain_is_rejected",
      fun _ => Nightstream.SuperNeo.SumCheck.check M2Probe.ops M2Probe.malformedTranscript, false⟩
  ]

end Folding

namespace M3Probe

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime

def hashSemantics : XOut.Semantics Unit Unit Unit Nat Unit Unit where
  hash := fun _ => 17
  nebulaDigest := id

def context : XOut.Context Unit Unit Unit Nat where
  params := ()
  structureDigest := ()
  piCcsHeader := ()
  publicInputLength := none
  initialSemanticState := 0

def stepSemantics : Step.Semantics Nat Nat Nat Nat Unit Unit where
  emptyRunning := 0
  initialAccumulatorDigest := 0
  initialNebula := none
  runningDigest := id
  chunkDigest := fun start fresh => 100 + start + fresh.length
  freshLink := fun digest fresh => digest == fresh
  nifsVerify := fun transcript running latest proof =>
    if proof = running + latest.length + transcript.chunkCount then
      some (running + latest.length)
    else
      none
  applicationStep := fun _ _ _ => true
  nebulaVerify := fun prior opening next => decide (opening = none ∧ next = prior)

abbrev ProbeState := State Nat Nat Nat Unit
abbrev ProbeInput := Step.Input Nat Unit Unit
abbrev ProbeProof := Step.Proof Nat Nat Unit

def initial : ProbeState where
  chunkCount := 0
  stepCount := 0
  z0 := 17
  zi := 17
  initialSemanticState := 0
  semanticState := 0
  pc := 1
  accumulatorDigest := 0
  publicTrace := 17
  proof := .initial

def baseInput : ProbeInput where
  nextLatest := [17]
  nebulaOpen := none
  nebulaNext := none

def baseProof : ProbeProof where
  fold := .noFold
  nebulaOpen := none
  semanticStateDigest := 0
  xOut := 17

def afterBase : ProbeState :=
  Step.advancedState stepSemantics initial 0 baseInput baseProof

def recursiveInput : ProbeInput where
  nextLatest := [17]
  nebulaOpen := none
  nebulaNext := none

def recursiveProof : ProbeProof where
  fold := .recursive 2
  nebulaOpen := none
  semanticStateDigest := 1
  xOut := 17

def afterRecursive : ProbeState :=
  Step.advancedState stepSemantics afterBase 1 recursiveInput recursiveProof

end M3Probe

namespace FPrime

/-- M3 probes cover honest branches and single-coordinate forgeries. -/
def probes : List Nightstream.Checks.Probe :=
  [ ⟨"fprime_base_accepts", fun _ =>
      Nightstream.Protocol.FPrime.Step.check M3Probe.hashSemantics
        M3Probe.stepSemantics .stateless M3Probe.context M3Probe.initial
        M3Probe.afterBase M3Probe.baseInput M3Probe.baseProof, true⟩
  , ⟨"fprime_base_rejects_xout_forgery", fun _ =>
      Nightstream.Protocol.FPrime.Step.check M3Probe.hashSemantics
        M3Probe.stepSemantics .stateless M3Probe.context M3Probe.initial
        M3Probe.afterBase M3Probe.baseInput { M3Probe.baseProof with xOut := 18 }, false⟩
  , ⟨"fprime_recursive_accepts", fun _ =>
      Nightstream.Protocol.FPrime.Step.check M3Probe.hashSemantics
        M3Probe.stepSemantics .stateless M3Probe.context M3Probe.afterBase
        M3Probe.afterRecursive M3Probe.recursiveInput M3Probe.recursiveProof, true⟩
  , ⟨"fprime_recursive_rejects_nifs_forgery", fun _ =>
      Nightstream.Protocol.FPrime.Step.check M3Probe.hashSemantics
        M3Probe.stepSemantics .stateless M3Probe.context M3Probe.afterBase
        M3Probe.afterRecursive M3Probe.recursiveInput
        { M3Probe.recursiveProof with fold := .recursive 3 }, false⟩
  , ⟨"fprime_xout_preimage_observes_counter", fun _ =>
      decide (Nightstream.Protocol.FPrime.XOut.preimage M3Probe.hashSemantics
        .stateless M3Probe.context M3Probe.afterBase ≠
        Nightstream.Protocol.FPrime.XOut.preimage M3Probe.hashSemantics
          .stateless M3Probe.context { M3Probe.afterBase with chunkCount := 2 }), true⟩
  ]

end FPrime

namespace ProjectionProbe

open Nightstream.SuperNeo.ProjectionCheck

def ops : Ops Nat where
  zero := 0
  add := fun left right => (left + right) % 97
  mul := fun left right => (left * right) % 97

def fixedBetaForgery : Identity Nat where
  lhs := [90, 1]
  rhs := [0, 0]
  beta := 7
  maxDegree := 1

end ProjectionProbe

namespace Projection

/-- The one-point check accepts a nonzero polynomial exactly on the named bad
root; it must never be presented as deterministic coefficient equality. -/
def probes : List Nightstream.Checks.Probe :=
  [ ⟨"pirlc_projection_fixed_beta_accepts_root_collision", fun _ =>
      decide (Nightstream.SuperNeo.ProjectionCheck.Accepted
        ProjectionProbe.ops ProjectionProbe.fixedBetaForgery), true⟩
  , ⟨"pirlc_projection_root_collision_is_not_exact", fun _ =>
      decide ProjectionProbe.fixedBetaForgery.Exact, false⟩
  ]

end Projection

def run : IO Bool := do
  let parametersOk ← Nightstream.Checks.runProbes Parameters.probes
  let foldingOk ← Nightstream.Checks.runProbes Folding.probes
  let fPrimeOk ← Nightstream.Checks.runProbes FPrime.probes
  let projectionOk ← Nightstream.Checks.runProbes Projection.probes
  pure (parametersOk && foldingOk && fPrimeOk && projectionOk)

end Nightstream.Checks.Protocol
