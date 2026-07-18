import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.ProductionRingAlgebra
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.DiagnosticProfile
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Recursive.RingAssembly

/-!
Recursive-bootstrap `Pi_RLC` sampler wiring into the independent paper-facing
artifact.

Assurance tier: artifact-checked. This module inspects the fixed-carrier
29-leaf recursive projection fixture, proves the one-input bootstrap
arity, binds every public projection pair to the same verifier-derived
54-column challenge, and transports those columns to the connected Poseidon2
sampler machine.

Owns: the recursive public-role-to-trace map; exclusion of the two delayed
`y_zcol` traces; exact one-input arity; challenge-column sharing for all 29
roles; coefficient decoding; unary production membership; and the concrete
recursive `ChallengeWiringArtifact` constructor.

Does not own: the external low-norm invertibility theorem, the post-`Pi_CCS`
initial transcript-state binding, public input/output carrier columns,
projection identities, Rust conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: recursive bootstrap has one fresh `Pi_RLC` input and no
synthetic running inputs. Challenge-column values equal the Poseidon2 machine
output only under canonical-assignment, constant-one, and accepted-row
premises; the post-`Pi_CCS` initial-state binding remains open. Unary machine
membership does not bind those columns or imply the pairwise strong-set law.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.identities.public` | first 29 recursive traces are public and two `y_zcol` traces are excluded | checked | `publicRoleIndex_census` |
| `nifs.pi_rlc.shape` | bootstrap has one fresh and zero running inputs | checked | `recursiveTotal_eq_one` |
| `nifs.pi_rlc.challenge` | all public pairs use the same 54 challenge columns | direct dataflow | `publicShared`, `challengeWiringArtifact` |
| `nifs.pi_rlc.challenge.sampler.selection.bind.symbol` | columns equal machine output conditional on canonical/one/accepted rows | checked | `decodedRing_eq_machineRing` |
| `nifs.pi_rlc.challenge.sampler.selection` | machine output is a typed canonical challenge; column authority remains separate | derived | `machineRing_member` |
| `nifs.pi_rlc.challenge.sampler` | pairwise strong-set security is a separate low-norm boundary | security boundary | `ProductionRingAlgebra.productionRingAlgebra_strong` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveSamplerArtifact

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Recursive

set_option maxRecDepth 1000000
set_option maxHeartbeats 8000000

/-! ## Protocol -> public projection tree -/

/-- Exact zero-based position of a paper-public role in recursive projection
order. -/
def publicRoleIndex : PublicRole DiagnosticProfile.matrixCount -> Nat
  | .commitment lane => lane.val
  | .x column => 18 + column.val
  | .yRing row limb => 23 + 2 * row.val + limb.val

/-- The recursive census contains the diagnostic public leaves followed by
the two delayed-NC `y_zcol` traces. -/
theorem recursiveTrace_count :
    recursiveTraces.length = DiagnosticProfile.traceCount := by
  decide

theorem publicRoleIndex_lt (role : PublicRole DiagnosticProfile.matrixCount) :
    publicRoleIndex role < recursiveTraces.length := by
  rw [recursiveTrace_count, DiagnosticProfile.traceCount_eq_31]
  cases role with
  | commitment lane =>
      exact Nat.lt_trans lane.isLt (by decide)
  | x column =>
      simp only [publicRoleIndex]
      omega
  | yRing row limb =>
      simp only [publicRoleIndex, DiagnosticProfile.matrixCount] at row ⊢
      omega

/-- Generated trace selected by one typed paper-public role. -/
def recursivePublicTrace
    (role : PublicRole DiagnosticProfile.matrixCount) :
    ProjectionProgram.ProjectionTrace :=
  recursiveTraces.get ⟨publicRoleIndex role, publicRoleIndex_lt role⟩

/-- Every recursive public trace contains the sole bootstrap input pair. -/
theorem recursivePublicTrace_pairArity :
    forall role, (recursivePublicTrace role).pairs.length = recursiveArity.total := by
  intro role
  cases role with
  | commitment lane =>
      revert lane
      decide
  | x column =>
      revert column
      decide
  | yRing row limb =>
      revert row limb
      decide

/-- Typed public subtree; its index cannot express delayed `y_zcol`. -/
def tree : TraceTree recursiveArity DiagnosticProfile.matrixCount where
  publicTrace := recursivePublicTrace
  publicPairArity := recursivePublicTrace_pairArity

theorem publicRoleIndex_census :
    (publicOrder DiagnosticProfile.matrixCount).map publicRoleIndex =
      List.range DiagnosticProfile.publicLeafCount := by
  decide

/-! ## Phase -> verifier-owned challenge columns -/

theorem recursiveTotal_eq_one : recursiveArity.total = 1 := by
  rfl

/-- Recursive bootstrap has exactly one input, so every typed input index names
the same sole sampler output. -/
def challengeColumns (_index : Fin recursiveArity.total) : List Nat :=
  RingAssembly.challengeColumns

@[simp] theorem challengeColumns_length (index : Fin recursiveArity.total) :
    (challengeColumns index).length = Concrete.ringDegree := by
  simpa [challengeColumns, Concrete.ringDegree,
    Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.coefficientCount]
    using RingAssembly.challengeColumns_length

/-- All 29 recursive public projection pairs use the exact challenge vector
decoded by the recursive sampler. -/
theorem publicShared : forall role index,
    (tree.publicPairAt role index).rhoColumns = challengeColumns index := by
  intro role index
  cases role with
  | commitment lane =>
      revert lane index
      decide
  | x column =>
      revert column index
      decide
  | yRing row limb =>
      revert row limb index
      decide

/-! ## Family -> coefficient and ring interpretation -/

/-- Paper-list view of the sole verifier-owned machine challenge. -/
def machineRing
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (_index : Fin recursiveArity.total) : Ring :=
  List.ofFn (RingAssembly.machineChallenge assignment canonical)

theorem values_challengeColumns_eq_decoded
    (assignment : Nat -> Nat) (index : Fin recursiveArity.total) :
    values assignment (challengeColumns index) =
      List.ofFn (RingAssembly.decodedChallenge assignment) := by
  unfold values challengeColumns RingAssembly.challengeColumns
  rw [List.map_ofFn]
  apply congrArg List.ofFn
  funext position
  apply Fin.ext
  rfl

/-- Accepted recursive sampler rows determine the complete ring challenge. -/
theorem decodedRing_eq_machineRing
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (index : Fin recursiveArity.total) :
    values assignment (challengeColumns index) =
      machineRing assignment canonical index := by
  rw [values_challengeColumns_eq_decoded]
  exact congrArg List.ofFn
    (RingAssembly.decodedChallenge_eq_machineChallenge
      prime canonical one accepted)

/-! ## Machine membership and equation wiring -/

theorem machineRing_member
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (index : Fin recursiveArity.total) :
    ProductionRingAlgebra.ChallengeMember
      (machineRing assignment canonical index) := by
  refine ⟨RingAssembly.machineScalar assignment canonical, ?_⟩
  unfold machineRing ProductionRingAlgebra.canonicalRing
  exact congrArg List.ofFn
    (RingAssembly.machineChallenge_eq_embedScalar assignment canonical)

/-- Bind only the challenge columns of a broader recursive batch facade. -/
def bindChallenges
    (columns : BatchColumns Concrete.productionGlobalParams recursiveArity
      DiagnosticProfile.matrixCount) :
    BatchColumns Concrete.productionGlobalParams recursiveArity
      DiagnosticProfile.matrixCount :=
  { columns with challenges := challengeColumns }

/-- Static rho-column sharing used by the public PiRLC equations. -/
def challengeWiringArtifact
    (columns : BatchColumns Concrete.productionGlobalParams recursiveArity
      DiagnosticProfile.matrixCount) :
    ChallengeWiringArtifact (bindChallenges columns) tree where
  publicShared := publicShared

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveSamplerArtifact
