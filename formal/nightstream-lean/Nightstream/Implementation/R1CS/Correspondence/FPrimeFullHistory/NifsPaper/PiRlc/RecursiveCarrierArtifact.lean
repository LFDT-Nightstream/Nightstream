import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.TraceCarrier
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.CarrierCodec
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.DiagnosticProfile
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.RecursiveSamplerArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.RelabeledCarrier

/-!
Recursive-bootstrap fixed-carrier diagnostic for `Pi_RLC`.

Assurance tier: artifact-checked. Exact generated-list facts are guarded and
do not by themselves establish Rust-conformant semantics.

Protocol: SuperNeo `Pi_RLC` inside recursive full-history F'.
Phase: generated projection carrier and strict-`Pi_DEC` parent binding.
Constraint family: 29 diagnostic projection leaves; challenge rows are owned by
`RecursiveSamplerArtifact`.

Owns: the exact first-29-trace census; all input/output 54-column widths; the
recursive local-to-global strict-`Pi_DEC` parent layout; exact output-parent
column identity; and the carrier, parent, and challenge-wiring artifacts.

Does not own: projection-row satisfaction, exact quotient-ring reduction,
input CE source authority, PiCCS output authority, evaluation padding,
delayed NC, strict-PiDEC row soundness, costs, or row removal.

Emits constraints: no. It classifies and connects generated columns.

Authority boundary: parent equality is structural equality against
`recursiveColumnMap`; no assignment values or digests are premises. The input
point facade names the same verifier-selected point but does not yet prove
that every source claim is wired to those point columns. Exact
generated-list equalities use `native_decide`; the trust-dependency guard makes
`Lean.trustCompiler` explicit instead of presenting artifact validation as a
pure kernel-normalization result.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.identities.public` | exactly the first 29 generated traces are paper-public and the final two are delayed-NC | checked | `publicTrace_census`, `trace_partition` |
| `nifs.pi_rlc.verify.identities.public` | every public input/output polynomial has 54 coefficients | checked | `inputWidth`, `outputWidth` |
| `nifs.pi_rlc.shape.parent` | output names the recursively relabeled strict-PiDEC parent | computed | `parentClaim`, `columns` |
| `nifs.pi_rlc.shape.parent.evaluations` | the diagnostic parent has exactly three physical evaluation rows | checked | `parentEvaluationCount` |
| `nifs.pi_rlc.verify.fold_wires` | commitment, X, evaluations, and point are column-identical to that parent | derived | `parentArtifact` |
| `nifs.pi_rlc.verify.identities.public` | equation inputs, outputs, and point are direct batch-carrier fields | direct dataflow | `equationWiringArtifact` |
| `nifs.pi_rlc.verify.identities.public` | exact generated carrier evidence | derived | `carrierArtifact` |
| `nifs.pi_rlc.challenge` | every public identity reads the same installed batch challenge columns | direct dataflow | `challengeWiringArtifact` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveCarrierArtifact

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc

/-- Strict-PiDEC parent in recursive full-history global column space. -/
def parentClaim : PiDecStrictCompiler.ClaimLayout :=
  RelabeledCarrier.relabelClaim recursiveColumnMap layout.parent

/-- The verifier-selected parent point, shared by every reduction input. -/
def point : PointColumns :=
  { r := parentClaim.rCols }

/-- Exact diagnostic recursive batch extracted from generated traces. -/
def columns : BatchColumns Concrete.productionGlobalParams recursiveArity
    DiagnosticProfile.matrixCount :=
  TraceCarrier.batchColumns RecursiveSamplerArtifact.tree parentClaim
    RecursiveSamplerArtifact.challengeColumns point

/-- Generated paper-public projection prefix of the diagnostic fixture. -/
def publicTraces : List ProjectionProgram.ProjectionTrace :=
  recursiveTraces.take DiagnosticProfile.publicLeafCount

/-- Exact physical evaluation count of the current diagnostic parent layout. -/
theorem parentEvaluationCount :
    parentClaim.yRingCols.length = DiagnosticProfile.matrixCount := by
  native_decide

/-- Final two generated traces, owned by delayed-NC rather than paper PiRLC. -/
def delayedTraces : List ProjectionProgram.ProjectionTrace :=
  recursiveTraces.drop DiagnosticProfile.publicLeafCount

set_option maxRecDepth 1000000
set_option maxHeartbeats 8000000

theorem publicTrace_census :
    RecursiveSamplerArtifact.tree.flatten = publicTraces := by
  native_decide

theorem publicTrace_count :
    publicTraces.length = DiagnosticProfile.publicLeafCount := by
  native_decide

theorem delayedTrace_count :
    delayedTraces.length = DiagnosticProfile.delayedNcLeafCount := by
  native_decide

theorem trace_partition : publicTraces ++ delayedTraces = recursiveTraces := by
  exact List.take_append_drop DiagnosticProfile.publicLeafCount recursiveTraces

theorem publicTrace_positions :
    (publicOrder DiagnosticProfile.matrixCount).map
      RecursiveSamplerArtifact.publicRoleIndex =
        List.range DiagnosticProfile.publicLeafCount :=
  RecursiveSamplerArtifact.publicRoleIndex_census

theorem inputWidth : forall role index,
    (RecursiveSamplerArtifact.tree.publicPairAt role index).inputColumns.length =
      Concrete.ringDegree := by
  intro role index
  cases role with
  | commitment lane =>
      revert lane index
      native_decide
  | x column =>
      revert column index
      native_decide
  | yRing row limb =>
      revert row limb index
      native_decide

theorem outputWidth : forall role,
    (RecursiveSamplerArtifact.tree.publicTrace role).outputColumns.length =
      Concrete.ringDegree := by
  intro role
  cases role with
  | commitment lane =>
      revert lane
      native_decide
  | x column =>
      revert column
      native_decide
  | yRing row limb =>
      revert row limb
      native_decide

/-- Equation-only column wiring, independent of generated-list census and
width checking. -/
theorem equationWiringArtifact :
    EquationWiringArtifact columns RecursiveSamplerArtifact.tree :=
  TraceCarrier.equationWiringArtifact RecursiveSamplerArtifact.tree parentClaim
    RecursiveSamplerArtifact.challengeColumns point

/-- Exact generated diagnostic-carrier refinement. -/
theorem carrierArtifact :
    CarrierArtifact (CarrierCodec.canonical DiagnosticProfile.matrixCount) columns
      RecursiveSamplerArtifact.tree publicTraces :=
  TraceCarrier.carrierArtifact
    (CarrierCodec.canonical DiagnosticProfile.matrixCount)
    RecursiveSamplerArtifact.tree parentClaim
    RecursiveSamplerArtifact.challengeColumns point publicTraces
    (CarrierCodec.canonical_artifact DiagnosticProfile.matrixCount)
    publicTrace_census
    inputWidth outputWidth

/-- The generated combined output columns are exactly the globally relabeled
strict-PiDEC parent carrier. -/
theorem parentArtifact : ParentArtifact columns where
  commitment := by native_decide
  x := by native_decide
  evaluationRows := parentEvaluationCount
  yRing := by
    intro row limb
    revert row limb
    native_decide
  r := by rfl

theorem bindChallenges_columns :
    RecursiveSamplerArtifact.bindChallenges columns = columns := by
  unfold RecursiveSamplerArtifact.bindChallenges columns
    TraceCarrier.batchColumns
  rfl

/-- Static challenge wiring for this exact diagnostic batch. -/
theorem challengeWiringArtifact :
    ChallengeWiringArtifact columns RecursiveSamplerArtifact.tree := by
  rw [← bindChallenges_columns]
  exact RecursiveSamplerArtifact.challengeWiringArtifact columns

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveCarrierArtifact
