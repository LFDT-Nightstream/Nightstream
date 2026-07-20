import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol

/-!
Independent source-trace schedule for the focused selective `y_zcol` bridge.

Owns: the exact mathematical order of the 49 polynomial evaluations and 86
extension-field products reconstructed from the checked source artifact.

Does not own: selectively emitted rows, generated rewrite metadata, low-norm
columns, assignment satisfaction, transcript authority, security bounds, or
permission to remove rows.

Emits constraints: no.

The order is derived from source-program ownership: shared rho evaluations;
then each limb's input, parent, and quotient evaluations; and independently
the shared power ladder followed by each limb's pair and quotient/Phi products.
Generated rewrite provenance must be compared against this schedule before a
family label can acquire semantic meaning.

| Schedule leaf | Mathematical obligation | Authority class |
|---|---|---|
| evaluations | 49 source-owned evaluation traces in mathematical order | direct dataflow |
| products | 86 source-owned extension products in dependency order | direct dataflow |
| layouts | every scheduled trace has the expected source-row shape | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceSchedule

open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol

private abbrev artifact :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Checked.artifact

private abbrev lowLimb :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Checked.lowLimb

private abbrev highLimb :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Checked.highLimb

private def limbEvaluations (limb : LimbOwner) : List EvaluationOwner :=
  limb.pairs.map PairOwner.inputEvaluation ++
    [limb.parentEvaluation, limb.quotientEvaluation]

/-- Source-semantic evaluation owners in the order used by the selective
evaluation-rewrite planner. -/
def evaluationOwners : List EvaluationOwner :=
  artifact.shared.rhoEvaluations ++
    limbEvaluations lowLimb ++
    limbEvaluations highLimb

private def limbProducts (limb : LimbOwner) : List KProductOwner :=
  limb.pairs.map PairOwner.rhoProduct ++ [limb.quotientPhiProduct]

/-- Source-semantic product owners in the order used by the selective
product-sum rewrite planner. -/
def productOwners : List KProductOwner :=
  artifact.shared.ladderProducts ++
    limbProducts lowLimb ++
    limbProducts highLimb

def evaluationTraces : List EvalTrace :=
  evaluationOwners.map EvaluationOwner.trace

def productTraces : List KMulTrace :=
  productOwners.map KProductOwner.trace

def evaluationSourceBlocks : List RowBlock :=
  evaluationOwners.map EvaluationOwner.rows

def productSourceBlocks : List RowBlock :=
  productOwners.map KProductOwner.rows

theorem evaluation_count : evaluationOwners.length = 49 := by
  native_decide

theorem product_count : productOwners.length = 86 := by
  native_decide

theorem evaluation_trace_count : evaluationTraces.length = 49 := by
  simp [evaluationTraces, evaluation_count]

theorem product_trace_count : productTraces.length = 86 := by
  simp [productTraces, product_count]

/-- Every scheduled evaluation has the exact source-trace layout. -/
theorem evaluation_layouts :
    ∀ owner ∈ evaluationOwners, owner.trace.LayoutValid := by
  set_option maxRecDepth 100000 in
    native_decide

/-- Every scheduled product has the exact five-row Karatsuba layout. -/
theorem product_layouts :
    ∀ owner ∈ productOwners, owner.trace.SumLayoutValid := by
  set_option maxRecDepth 100000 in
    native_decide

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceSchedule
