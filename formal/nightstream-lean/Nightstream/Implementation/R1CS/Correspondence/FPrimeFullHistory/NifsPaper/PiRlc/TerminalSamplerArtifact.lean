import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.ProductionRingAlgebra
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.DiagnosticProfile
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.RingAssembly

/-!
Terminal `Pi_RLC` sampler wiring into the independent paper-facing artifact.

Assurance tier: artifact-checked. This module inspects the fixed-carrier
29-leaf terminal projection fixture and proves that every leaf shares the
same verifier-derived challenge columns, and transports the decoded columns
to the transcript-machine-derived Phi81 coefficient vectors.

Owns: the terminal public-role-to-trace map; the exclusion of the two delayed
`y_zcol` traces from the paper tree; exact challenge-column sharing for all
29 roles and 15 inputs; typed challenge range facts; and the static
`ChallengeWiringArtifact` constructor.

Does not own: the external low-norm invertibility theorem, the post-PiCCS
initial transcript state, public input/output carrier columns, projection
identities, Rust conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: challenge-column values equal the connected Poseidon2
machine output only under canonical-assignment, constant-one, and accepted-row
premises; the post-`Pi_CCS` initial-state binding remains open. Unary machine
membership does not bind those columns or imply pairwise strong-set security.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.identities.public` | first 29 terminal traces are public and two `y_zcol` traces are excluded | checked | `publicRoleIndex_census` |
| `nifs.pi_rlc.challenge` | every public pair uses the same 54 challenge columns | direct dataflow | `publicShared`, `challengeWiringArtifact` |
| `nifs.pi_rlc.challenge.sampler.selection.bind.symbol` | columns equal machine output conditional on canonical/one/accepted rows | checked | `decodedRing_eq_machineRing` |
| `nifs.pi_rlc.challenge.sampler.selection` | machine output is a typed canonical challenge; column authority remains separate | derived | `machineRing_member` |
| `nifs.pi_rlc.challenge.sampler` | pairwise strong-set security is a separate low-norm boundary | security boundary | `ProductionRingAlgebra.productionRingAlgebra_strong` |
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
def publicRoleIndex : PublicRole DiagnosticProfile.matrixCount -> Nat
  | .commitment lane => lane.val
  | .x column => 18 + column.val
  | .yRing row limb => 23 + 2 * row.val + limb.val

/-- The generated terminal census contains the diagnostic public leaves
followed by the two delayed-NC `y_zcol` traces. -/
theorem terminalTrace_count :
    terminalTraces.length = DiagnosticProfile.traceCount := by
  decide

theorem publicRoleIndex_lt (role : PublicRole DiagnosticProfile.matrixCount) :
    publicRoleIndex role < terminalTraces.length := by
  rw [terminalTrace_count, DiagnosticProfile.traceCount_eq_31]
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
def terminalPublicTrace
    (role : PublicRole DiagnosticProfile.matrixCount) :
    ProjectionProgram.ProjectionTrace :=
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
def tree : TraceTree terminalArity DiagnosticProfile.matrixCount where
  publicTrace := terminalPublicTrace
  publicPairArity := terminalPublicTrace_pairArity

/-- The typed public-role order is exactly the first 29 zero-based positions
of the generated terminal census. This proves the census boundary without
forcing the kernel to compare the large trace records field-by-field. -/
theorem publicRoleIndex_census :
    (publicOrder DiagnosticProfile.matrixCount).map publicRoleIndex =
      List.range DiagnosticProfile.publicLeafCount := by
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

/-! ## Machine membership and equation wiring -/

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

/-- Bind only the challenge-column field of a broader terminal batch facade.
All public input, output, and point fields retain their caller-supplied values
and require separate carrier refinements. -/
def bindChallenges
    (columns : BatchColumns Concrete.productionGlobalParams terminalArity
      DiagnosticProfile.matrixCount) :
    BatchColumns Concrete.productionGlobalParams terminalArity
      DiagnosticProfile.matrixCount :=
  { columns with challenges := challengeColumns }

/-- Static rho-column sharing used by the public PiRLC equations. -/
def challengeWiringArtifact
    (columns : BatchColumns Concrete.productionGlobalParams terminalArity
      DiagnosticProfile.matrixCount) :
    ChallengeWiringArtifact (bindChallenges columns) tree where
  publicShared := publicShared

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalSamplerArtifact
