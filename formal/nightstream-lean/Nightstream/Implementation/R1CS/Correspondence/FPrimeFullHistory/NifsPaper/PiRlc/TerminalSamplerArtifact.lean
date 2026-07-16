import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.ProductionRingAlgebra
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.RingAssembly

/-!
Terminal `Pi_RLC` sampler wiring into the independent paper-facing artifact.

Assurance tier: implementation/R1CS correspondence. This module constructs
the exact 29-leaf public projection tree, proves that every leaf shares the
same verifier-derived challenge columns, and transports the decoded columns
to the transcript-machine-derived Phi81 coefficient vectors.

Owns: the terminal public-role-to-trace map; the exclusion of the two delayed
`y_zcol` traces from the paper tree; exact challenge-column sharing for all
29 roles and 15 inputs; and the conditional constructor for the paper
`SamplerArtifact`.

Does not own: the external low-norm invertibility theorem, the post-PiCCS
initial transcript state, public input/output carrier columns, projection
identities, Rust conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: production challenge columns are accepted only through
the connected Poseidon2 sampler machine. Unary production membership is
derived from the typed machine output. Pairwise strong-set security remains a
separate theorem and cannot be inferred from this artifact.

| Protocol | Phase | Constraint family | Indexed leaf | Exact obligation |
|---|---|---|---|---|
| `Pi_RLC` | projection | public trace census | 29 public roles | first 29 terminal traces exactly; two `y_zcol` traces excluded |
| `Pi_RLC` | challenge binding | shared rho columns | role x input | every public pair uses the canonical 54 challenge columns |
| `Pi_RLC` | ring assembly | decoded field vector | input x coefficient | production column values equal the verifier-owned machine output |
| `Pi_RLC` | membership | canonical production challenge | one of 15 ring challenges | derived from typed five-symbol machine output |
| `Pi_RLC` | paper bridge | `SamplerArtifact` | complete terminal sampler | exact wiring plus unary membership; no pairwise claim |
| Definition 17 | pairwise security | invertible challenge difference | two distinct challenges | separate `ProductionRingAlgebra.productionRingAlgebra_strong` theorem |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalSamplerArtifact

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal

set_option maxRecDepth 1000000
set_option maxHeartbeats 8000000

/-! ## Public trace tree -/

/-- Exact zero-based position of one paper-public role in the production
terminal projection order. -/
def publicRoleIndex : PublicRole -> Nat
  | .commitment lane => lane.val
  | .x column => 18 + column.val
  | .yRing row limb => 23 + 2 * row.val + limb.val

/-- The complete generated terminal census contains 29 paper-public traces
followed by the two delayed-NC `y_zcol` traces. -/
theorem terminalTrace_count : terminalTraces.length = 31 := by
  decide

theorem publicRoleIndex_lt (role : PublicRole) :
    publicRoleIndex role < terminalTraces.length := by
  rw [terminalTrace_count]
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
def terminalPublicTrace (role : PublicRole) : ProjectionProgram.ProjectionTrace :=
  terminalTraces.get ⟨publicRoleIndex role, publicRoleIndex_lt role⟩

/-- Every selected public trace contains the exact active terminal arity of
15 input pairs. -/
theorem terminalPublicTrace_pairArity :
    forall role, (terminalPublicTrace role).pairs.length = terminalArity.total := by
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

/-- Typed paper-public subtree. Its `PublicRole` index cannot express either
delayed-NC `y_zcol` trace. -/
def tree : TraceTree terminalArity where
  publicTrace := terminalPublicTrace
  publicPairArity := terminalPublicTrace_pairArity

/-- The typed public-role order is exactly the first 29 zero-based positions
of the generated terminal census. This proves the census boundary without
forcing the kernel to compare the large trace records field-by-field. -/
theorem publicRoleIndex_census :
    publicOrder.map publicRoleIndex = List.range 29 := by
  decide

/-! ## Shared verifier-owned challenge columns -/

theorem terminalTotal_eq_scalarCount :
    terminalArity.total = ScalarRows.scalarCount := by
  rfl

/-- Convert a paper terminal input index into the sampler's scalar index. -/
def scalarIndex (index : Fin terminalArity.total) : Fin ScalarRows.scalarCount :=
  Fin.cast terminalTotal_eq_scalarCount index

/-- Canonical production challenge columns for one terminal input. -/
def challengeColumns (index : Fin terminalArity.total) : List Nat :=
  RingAssembly.challengeColumns (scalarIndex index)

@[simp] theorem challengeColumns_length (index : Fin terminalArity.total) :
    (challengeColumns index).length = Concrete.ringDegree := by
  simpa [challengeColumns, Concrete.ringDegree,
    Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.coefficientCount]
    using RingAssembly.challengeColumns_length (scalarIndex index)

/-- Every one of the `29 * 15` public projection pairs uses the exact same
54-column challenge vector selected by its input index. -/
theorem publicShared : forall role index,
    (tree.publicPairAt role index).rhoColumns = challengeColumns index := by
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

/-! ## Field-list and Phi81 carrier equality -/

/-- Paper list view of the verifier-owned machine challenge. -/
def machineRing
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (index : Fin terminalArity.total) : Ring :=
  List.ofFn (RingAssembly.machineChallenge assignment canonical (scalarIndex index))

/-- Decoding the canonical challenge columns through the paper carrier is
exactly the list view of `TerminalRing.decodedChallenge`. -/
theorem values_challengeColumns_eq_decoded
    (assignment : Nat -> Nat) (index : Fin terminalArity.total) :
    values assignment (challengeColumns index) =
      List.ofFn (RingAssembly.decodedChallenge assignment (scalarIndex index)) := by
  unfold values challengeColumns RingAssembly.challengeColumns
  rw [List.map_ofFn]
  apply congrArg List.ofFn
  funext position
  apply Fin.ext
  rfl

/-- Accepted terminal sampler rows force the complete paper-list challenge to
equal the verifier-owned machine challenge. -/
theorem decodedRing_eq_machineRing
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (index : Fin terminalArity.total) :
    values assignment (challengeColumns index) =
      machineRing assignment canonical index := by
  rw [values_challengeColumns_eq_decoded]
  exact congrArg List.ofFn
    (RingAssembly.decodedChallenge_eq_machineChallenge
      prime canonical one accepted (scalarIndex index))

/-! ## Unary membership and paper artifact -/

/-- Generic unary membership premise for an arbitrary paper-facing algebra.
This deliberately makes no pairwise strong-set claim. -/
def ChallengeMembershipPremise
    (ring : RingAlgebra)
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment) : Prop :=
  forall index, ring.challengeValid (machineRing assignment canonical index)

/-- Every verifier-owned machine vector is an exact canonical member of the
independently defined production challenge carrier. No accepted-row premise
is required for this typed range fact. -/
theorem machineRing_member
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (index : Fin terminalArity.total) :
    ProductionRingAlgebra.ChallengeMember
      (machineRing assignment canonical index) := by
  refine ⟨MachineOutput.scalar assignment canonical (scalarIndex index), ?_⟩
  unfold machineRing ProductionRingAlgebra.canonicalRing
  exact congrArg List.ofFn
    (RingAssembly.machineChallenge_eq_embedScalar
      assignment canonical (scalarIndex index))

theorem productionChallengeMembership
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment) :
    ChallengeMembershipPremise ProductionRingAlgebra.productionRingAlgebra
      assignment canonical := by
  intro index
  exact machineRing_member assignment canonical index

/-- Bind only the challenge-column field of a broader terminal batch facade.
All public input, output, and point fields retain their caller-supplied values
and require separate carrier refinements. -/
def bindChallenges
    (columns : BatchColumns Concrete.productionGlobalParams terminalArity) :
    BatchColumns Concrete.productionGlobalParams terminalArity :=
  { columns with challenges := challengeColumns }

/-- Exact wiring plus unary membership constructs the terminal paper sampler
artifact for an arbitrary algebra. Pairwise strong-set security is not a field
of `SamplerArtifact` and must be supplied to the later soundness theorem. -/
theorem samplerArtifact_of_membership
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (ring : RingAlgebra)
    (membership : ChallengeMembershipPremise ring assignment canonical)
    (columns : BatchColumns Concrete.productionGlobalParams terminalArity) :
    SamplerArtifact ring assignment (bindChallenges columns) tree where
  width := challengeColumns_length
  publicShared := publicShared
  challengeMembership index := by
    change ring.challengeValid (values assignment (challengeColumns index))
    rw [decodedRing_eq_machineRing prime canonical one accepted index]
    exact membership index

/-- The concrete production sampler artifact now needs no ad hoc semantic
premise: typed machine output proves unary membership. This still does not
assert the pairwise Definition-17 law. -/
theorem productionSamplerArtifact
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (columns : BatchColumns Concrete.productionGlobalParams terminalArity) :
    SamplerArtifact ProductionRingAlgebra.productionRingAlgebra assignment
      (bindChallenges columns) tree :=
  samplerArtifact_of_membership prime canonical one accepted
    ProductionRingAlgebra.productionRingAlgebra
    (productionChallengeMembership assignment canonical) columns

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalSamplerArtifact
