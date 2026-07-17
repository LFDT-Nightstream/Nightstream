import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc

/-!
Matrix-indexed carrier extraction from a typed `Pi_RLC` projection tree.

Assurance tier: model-level. Profile modules separately prove the
artifact-checked generated census and widths.

Protocol: SuperNeo `Pi_RLC` inside the fixed F' NIFS.
Phase: generated projection trace to a CE carrier.
Constraint family: 18 commitment, five public-input, and
`2 * matrixCount` evaluation leaves.

Owns: the unique structural extraction of input and output carrier columns
from `TraceTree`; the shared-point batch facade; and the generic constructor
that turns exact census/width facts into `CarrierArtifact`.

Does not own: transcript challenges, strict-PiDEC parent identity, evaluation
padding, delayed NC, projection-row soundness, quotient-ring semantics, costs,
or row removal.

Emits constraints: no. It names columns already present in projection traces.

Authority boundary: this module performs no value comparison and accepts no
prover digest. Profile modules must prove the trace census and all 54-column
widths against generated data before using `carrierArtifact`.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.identities.public` | each typed input leaf is the matching trace-pair input | computed | `inputProjection`, `inputProjection_at` |
| `nifs.pi_rlc.verify.identities.public` | each typed output leaf is the matching trace output | computed | `outputProjection`, `outputProjection_at` |
| `nifs.pi_rlc.verify.fold_wires` | every reduction input shares the verifier-selected output point | direct dataflow | `batchColumns` |
| `nifs.pi_rlc.verify.identities.public` | equation inputs, outputs, and point are direct batch-carrier fields | direct dataflow | `equationWiringArtifact` |
| `nifs.pi_rlc.verify.identities.public` | census plus exact widths constructs diagnostic carrier evidence | derived | `carrierArtifact` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TraceCarrier

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc

/-- Public columns for one input index, read directly from its trace pairs. -/
def inputProjection
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (tree : TraceTree arity matrixCount) (index : Fin arity.total) :
    ProjectionColumns matrixCount where
  commitment lane := (tree.publicPairAt (.commitment lane) index).inputColumns
  x column := (tree.publicPairAt (.x column) index).inputColumns
  yRing row limb := (tree.publicPairAt (.yRing row limb) index).inputColumns

@[simp] theorem inputProjection_at
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (tree : TraceTree arity matrixCount) (index : Fin arity.total)
    (role : PublicRole matrixCount) :
    (inputProjection tree index).at role =
      (tree.publicPairAt role index).inputColumns := by
  cases role <;> rfl

/-- Public output columns, read directly from the corresponding trace. -/
def outputProjection
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (tree : TraceTree arity matrixCount) : ProjectionColumns matrixCount where
  commitment lane := (tree.publicTrace (.commitment lane)).outputColumns
  x column := (tree.publicTrace (.x column)).outputColumns
  yRing row limb := (tree.publicTrace (.yRing row limb)).outputColumns

@[simp] theorem outputProjection_at
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (tree : TraceTree arity matrixCount) (role : PublicRole matrixCount) :
    (outputProjection tree).at role = (tree.publicTrace role).outputColumns := by
  cases role <;> rfl

/-- Typed batch facade. Padding and delayed-NC carriers cannot enter it. -/
def batchColumns
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (tree : TraceTree arity matrixCount) (parentClaim : ClaimLayout)
    (challenges : Fin arity.total -> List Nat) (point : PointColumns) :
    BatchColumns params arity matrixCount where
  parentClaim := parentClaim
  challenges := challenges
  inputs := inputProjection tree
  output := outputProjection tree
  inputPoints := fun _ => point
  outputPoint := point

/-- The three public-equation column equalities are definitional consequences
of `batchColumns`; no generated census or width proof is involved. -/
theorem equationWiringArtifact
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (tree : TraceTree arity matrixCount) (parentClaim : ClaimLayout)
    (challenges : Fin arity.total -> List Nat) (point : PointColumns) :
    EquationWiringArtifact
      (batchColumns tree parentClaim challenges point) tree where
  inputColumns role index := by
    change (tree.publicPairAt role index).inputColumns =
      (inputProjection tree index).at role
    rw [inputProjection_at]
  outputColumns role := by
    change (tree.publicTrace role).outputColumns =
      (outputProjection tree).at role
    rw [outputProjection_at]
  pointColumns _ := rfl

/-- The structural extraction leaves only census and exact-width evidence for
a profile to discharge. -/
theorem carrierArtifact
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (codec : CarrierCodec matrixCount) (tree : TraceTree arity matrixCount)
    (parentClaim : ClaimLayout) (challenges : Fin arity.total -> List Nat)
    (point : PointColumns) (productionTraces : List ProjectionTrace)
    (codecLayout : CodecArtifact codec)
    (census : tree.flatten = productionTraces)
    (inputWidth : forall role index,
      (tree.publicPairAt role index).inputColumns.length = Concrete.ringDegree)
    (outputWidth : forall role,
      (tree.publicTrace role).outputColumns.length = Concrete.ringDegree) :
    CarrierArtifact codec
      (batchColumns tree parentClaim challenges point) tree productionTraces where
  census := census
  codecLayout := codecLayout
  inputWidth role index := by
    change ((inputProjection tree index).at role).length = Concrete.ringDegree
    rw [inputProjection_at]
    exact inputWidth role index
  outputWidth role := by
    change ((outputProjection tree).at role).length = Concrete.ringDegree
    rw [outputProjection_at]
    exact outputWidth role
  wiring := equationWiringArtifact tree parentClaim challenges point

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TraceCarrier
