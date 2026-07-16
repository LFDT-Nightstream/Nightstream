import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.ProductionRingAlgebra
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Recursive.RingAssembly

/-!
Recursive-bootstrap `Pi_RLC` sampler wiring into the independent paper-facing
artifact.

Assurance tier: implementation/R1CS correspondence. This module constructs the
exact 29-leaf recursive public projection tree, proves the one-input bootstrap
arity, binds every public projection pair to the same verifier-derived
54-column challenge, and transports those columns to the connected Poseidon2
sampler machine.

Owns: the recursive public-role-to-trace map; exclusion of the two delayed
`y_zcol` traces; exact one-input arity; challenge-column sharing for all 29
roles; coefficient decoding; unary production membership; and the concrete
recursive `SamplerArtifact` constructor.

Does not own: the external low-norm invertibility theorem, the post-`Pi_CCS`
initial transcript-state binding, public input/output carrier columns,
projection identities, Rust conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: recursive bootstrap has one fresh `Pi_RLC` input and no
synthetic running inputs. Its challenge columns are accepted only through the
connected Poseidon2 sampler machine. Unary membership does not imply the
separate pairwise strong-set theorem.

| Protocol | Phase | Constraint family | Indexed leaf | Exact obligation |
|---|---|---|---|---|
| `Pi_RLC` | projection | public trace census | 29 public roles | first 29 recursive traces; two `y_zcol` traces excluded |
| `Pi_RLC` | arity | bootstrap input | one fresh, zero running | `recursiveArity.total = 1` |
| `Pi_RLC` | challenge binding | shared rho columns | role x sole input | all public pairs use the canonical 54 columns |
| `Pi_RLC` | ring assembly | decoded field vector | 54 coefficients | production columns equal machine output |
| `Pi_RLC` | membership | canonical challenge | sole ring challenge | derived from typed five-symbol output |
| Definition 17 | pairwise security | invertible challenge difference | distinct challenges | separate `productionRingAlgebra_strong` theorem |
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
def publicRoleIndex : PublicRole -> Nat
  | .commitment lane => lane.val
  | .x column => 18 + column.val
  | .yRing row limb => 23 + 2 * row.val + limb.val

/-- The recursive census contains 29 paper-public traces followed by the two
delayed-NC `y_zcol` traces. -/
theorem recursiveTrace_count : recursiveTraces.length = 31 := by
  decide

theorem publicRoleIndex_lt (role : PublicRole) :
    publicRoleIndex role < recursiveTraces.length := by
  rw [recursiveTrace_count]
  cases role with
  | commitment lane =>
      exact Nat.lt_trans lane.isLt (by decide)
  | x column =>
      simp only [publicRoleIndex]
      omega
  | yRing row limb =>
      simp only [publicRoleIndex]
      omega

/-- Generated trace selected by one typed paper-public role. -/
def recursivePublicTrace (role : PublicRole) : ProjectionProgram.ProjectionTrace :=
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
def tree : TraceTree recursiveArity where
  publicTrace := recursivePublicTrace
  publicPairArity := recursivePublicTrace_pairArity

theorem publicRoleIndex_census :
    publicOrder.map publicRoleIndex = List.range 29 := by
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

/-! ## Paper sampler artifact -/

def ChallengeMembershipPremise
    (ring : RingAlgebra)
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment) : Prop :=
  forall index, ring.challengeValid (machineRing assignment canonical index)

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

theorem productionChallengeMembership
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment) :
    ChallengeMembershipPremise ProductionRingAlgebra.productionRingAlgebra
      assignment canonical := by
  intro index
  exact machineRing_member assignment canonical index

/-- Bind only the challenge columns of a broader recursive batch facade. -/
def bindChallenges
    (columns : BatchColumns Concrete.productionGlobalParams recursiveArity) :
    BatchColumns Concrete.productionGlobalParams recursiveArity :=
  { columns with challenges := challengeColumns }

theorem samplerArtifact_of_membership
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (ring : RingAlgebra)
    (membership : ChallengeMembershipPremise ring assignment canonical)
    (columns : BatchColumns Concrete.productionGlobalParams recursiveArity) :
    SamplerArtifact ring assignment (bindChallenges columns) tree where
  width := challengeColumns_length
  publicShared := publicShared
  challengeMembership index := by
    change ring.challengeValid (values assignment (challengeColumns index))
    rw [decodedRing_eq_machineRing prime canonical one accepted index]
    exact membership index

/-- Concrete recursive-bootstrap sampler artifact. Pairwise strong-set
security remains a separate mathematical theorem. -/
theorem productionSamplerArtifact
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (columns : BatchColumns Concrete.productionGlobalParams recursiveArity) :
    SamplerArtifact ProductionRingAlgebra.productionRingAlgebra assignment
      (bindChallenges columns) tree :=
  samplerArtifact_of_membership prime canonical one accepted
    ProductionRingAlgebra.productionRingAlgebra
    (productionChallengeMembership assignment canonical) columns

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveSamplerArtifact
