import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction
import Nightstream.SuperNeo.SumCheck.HypercubeTruth

/-!
Model-level initial sum for the independent Split-NC norm polynomial.

Protocol: SuperNeo `Pi_CCS`, split NC branch.
Phase: exact Boolean-cube sum before the first NC SumCheck message.
Constraint family: equality-gated, gamma-compressed source cubics only; this
file emits no rows.

Owns: one source specialization at `betaM` and `betaA`, its gamma mixture, the
typed column-by-lane Boolean sum, the exact generic SumCheck cube adapter, the
literal zero initial claim, and honest initial-sum completeness.

Does not own: the mixing-root soundness dichotomy, SumCheck messages, degree,
transcript derivation, `yZcol`, terminal binding, Rust, R1CS, row emission,
row removal, or constraint counts.

Emits constraints: no.

Authority boundary: `hypercubeSum` evaluates the source-derived polynomial.
`claimedInitial` is definitionally zero and is never supplied by a prover.
Honest completeness consumes only `Semantics.Nc.Truth`; it does not consume an
output message, a polynomial callback, or a no-zero-divisor assumption.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.initial.source_specialization` | equality-weight one source's complete padded cubic table | independent semantic residual | `sourceResidualAtBeta` |
| `nifs.pi_ccs.nc.initial.gamma_mix` | compress every source specialization under the named exponent convention | computed | `mixedResidualAtBeta` |
| `nifs.pi_ccs.nc.initial.boolean_sum` | sum the exact equality-gated NC polynomial over column then lane | computed | `hypercubeSum` |
| `nifs.pi_ccs.nc.initial.sumcheck.total` | totalize only the unreachable wrong-arity branch | semantic adapter | `sumcheckPolynomial` |
| `nifs.pi_ccs.nc.initial.sumcheck.bridge` | recursive completions equal the typed column-by-lane cube | derived | `sumcheckHypercubeSum_eq_hypercubeSum` |
| `nifs.pi_ccs.nc.initial.residual.bridge` | typed polynomial cube equals the independent source-specialization mix | derived | `hypercubeSum_eq_mixedResidualAtBeta` |
| `nifs.pi_ccs.nc.initial.claim` | verifier initial claim is literally zero | computed | `claimedInitial` |
| assurance | honest full-carrier norm truth zeros the source mix and every convention's cube sum | derived | `mixedResidualAtBeta_eq_zero_of_truth`, `claimedInitial_eq_sumcheckHypercubeSum_of_truth` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum

set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

/-- Equality-weighted Boolean specialization of one independently derived
source cubic table. -/
def sourceResidualAtBeta
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (source : Fin shape.sourceCount) : K :=
  FiniteSumAlgebra.sumMap ops
    (BooleanVertex.all domain.columnVariables) fun column =>
      FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.laneVariables) fun lane =>
          K.mul
            (K.mul
              (column.equalityWeight ops coins.betaM)
              (lane.equalityWeight ops coins.betaA))
            (SourceProjection.rangeValueAt covers data source {
              column := column.toCubePoint ops
              lane := lane.toCubePoint ops })

/-- Gamma compression of the independent source specializations. -/
def mixedResidualAtBeta
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain) : K :=
  FiniteSumAlgebra.sumMap ops
    (canonicalFinIndices shape.sourceCount) fun source =>
      SignedJointIdentity.gammaTerm ops coins.gamma
        (sourceExponent shape convention source)
        (sourceResidualAtBeta covers data coins source)

/-- Exact typed product-cube sum of the independent NC polynomial. -/
def hypercubeSum
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain) : K :=
  FiniteSumAlgebra.sumMap ops
    (BooleanVertex.all domain.columnVariables) fun column =>
      FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.laneVariables) fun lane =>
          qAtPoint convention covers data coins {
            column := column.toCubePoint ops
            lane := lane.toCubePoint ops }

private def selectorAt
    {domain : FlatNcDomain}
    (coins : Coins domain)
    (column : BooleanVertex domain.columnVariables)
    (lane : BooleanVertex domain.laneVariables) : K :=
  K.mul
    (column.equalityWeight ops coins.betaM)
    (lane.equalityWeight ops coins.betaA)

private def sourcePointTerm
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (source : Fin shape.sourceCount)
    (column : BooleanVertex domain.columnVariables)
    (lane : BooleanVertex domain.laneVariables) : K :=
  SignedJointIdentity.gammaTerm ops coins.gamma
    (sourceExponent shape convention source)
    (K.mul (selectorAt coins column lane)
      (SourceProjection.rangeValueAt covers data source {
        column := column.toCubePoint ops
        lane := lane.toCubePoint ops }))

private theorem qAtPoint_toCubePoint_eq_sourcePointSum
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (column : BooleanVertex domain.columnVariables)
    (lane : BooleanVertex domain.laneVariables) :
    qAtPoint convention covers data coins {
        column := column.toCubePoint ops
        lane := lane.toCubePoint ops } =
      FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.sourceCount) fun source =>
          sourcePointTerm convention covers data coins source column lane := by
  unfold qAtPoint mixedRangeAt sourcePointTerm selectorAt
  rw [SumCheckTruthPath.pointEquality_toCubePoint_eq_equalityWeight ops laws]
  rw [SumCheckTruthPath.pointEquality_toCubePoint_eq_equalityWeight ops laws]
  change
    ops.mul
        (ops.mul
          (column.equalityWeight ops coins.betaM)
          (lane.equalityWeight ops coins.betaA))
        (FiniteSumAlgebra.sumMap ops
          (canonicalFinIndices shape.sourceCount) _) =
      FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.sourceCount) _
  rw [← FiniteSumAlgebra.sumMap_mul_left ops laws]
  apply FiniteSumAlgebra.sumMap_congr
  intro source _
  unfold SignedJointIdentity.gammaTerm
  let selector := ops.mul
    (column.equalityWeight ops coins.betaM)
    (lane.equalityWeight ops coins.betaA)
  let weight := TargetPolynomial.power ops.toOps coins.gamma
    (sourceExponent shape convention source)
  let residual := SourceProjection.rangeValueAt covers data source {
    column := column.toCubePoint ops
    lane := lane.toCubePoint ops }
  change ops.mul selector (ops.mul weight residual) =
    ops.mul weight (ops.mul selector residual)
  calc
    ops.mul selector (ops.mul weight residual) =
        ops.mul (ops.mul selector weight) residual :=
      (laws.mul_assoc selector weight residual).symm
    _ = ops.mul (ops.mul weight selector) residual := by
      rw [laws.mul_comm selector weight]
    _ = ops.mul weight (ops.mul selector residual) :=
      laws.mul_assoc weight selector residual

/-- The typed Boolean sum is exactly the independently grouped source
specialization mix. This is a finite Fubini/distributivity theorem, not a
soundness assumption. -/
theorem hypercubeSum_eq_mixedResidualAtBeta
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain) :
    hypercubeSum convention covers data coins =
      mixedResidualAtBeta convention covers data coins := by
  unfold hypercubeSum mixedResidualAtBeta
  calc
    FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.columnVariables) (fun column =>
          FiniteSumAlgebra.sumMap ops
            (BooleanVertex.all domain.laneVariables) (fun lane =>
              qAtPoint convention covers data coins {
                column := column.toCubePoint ops
                lane := lane.toCubePoint ops })) =
      FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.columnVariables) (fun column =>
          FiniteSumAlgebra.sumMap ops
            (BooleanVertex.all domain.laneVariables) (fun lane =>
              FiniteSumAlgebra.sumMap ops
                (canonicalFinIndices shape.sourceCount) (fun source =>
                  sourcePointTerm convention covers data coins source
                    column lane))) := by
        apply FiniteSumAlgebra.sumMap_congr
        intro column _
        apply FiniteSumAlgebra.sumMap_congr
        intro lane _
        exact qAtPoint_toCubePoint_eq_sourcePointSum
          convention covers data coins column lane
    _ = FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.columnVariables) (fun column =>
          FiniteSumAlgebra.sumMap ops
            (canonicalFinIndices shape.sourceCount) (fun source =>
              FiniteSumAlgebra.sumMap ops
                (BooleanVertex.all domain.laneVariables) (fun lane =>
                  sourcePointTerm convention covers data coins source
                    column lane))) := by
        apply FiniteSumAlgebra.sumMap_congr
        intro column _
        exact FiniteSumAlgebra.sumMap_swap ops laws
          (BooleanVertex.all domain.laneVariables)
          (canonicalFinIndices shape.sourceCount)
          (fun lane source =>
            sourcePointTerm convention covers data coins source column lane)
    _ = FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.sourceCount) (fun source =>
          FiniteSumAlgebra.sumMap ops
            (BooleanVertex.all domain.columnVariables) (fun column =>
              FiniteSumAlgebra.sumMap ops
                (BooleanVertex.all domain.laneVariables) (fun lane =>
                  sourcePointTerm convention covers data coins source
                    column lane))) :=
        FiniteSumAlgebra.sumMap_swap ops laws
          (BooleanVertex.all domain.columnVariables)
          (canonicalFinIndices shape.sourceCount)
          (fun column source =>
            FiniteSumAlgebra.sumMap ops
              (BooleanVertex.all domain.laneVariables) fun lane =>
                sourcePointTerm convention covers data coins source
                  column lane)
    _ = FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.sourceCount) (fun source =>
          SignedJointIdentity.gammaTerm ops coins.gamma
            (sourceExponent shape convention source)
            (sourceResidualAtBeta covers data coins source)) := by
        apply FiniteSumAlgebra.sumMap_congr
        intro source _
        unfold sourcePointTerm SignedJointIdentity.gammaTerm
          sourceResidualAtBeta selectorAt
        let weight := TargetPolynomial.power ops.toOps coins.gamma
          (sourceExponent shape convention source)
        calc
          FiniteSumAlgebra.sumMap ops
              (BooleanVertex.all domain.columnVariables) (fun column =>
                FiniteSumAlgebra.sumMap ops
                  (BooleanVertex.all domain.laneVariables) (fun lane =>
                    ops.mul weight
                      (ops.mul
                        (ops.mul
                          (column.equalityWeight ops coins.betaM)
                          (lane.equalityWeight ops coins.betaA))
                        (SourceProjection.rangeValueAt covers data source {
                          column := column.toCubePoint ops
                          lane := lane.toCubePoint ops })))) =
            FiniteSumAlgebra.sumMap ops
              (BooleanVertex.all domain.columnVariables) (fun column =>
                ops.mul weight
                  (FiniteSumAlgebra.sumMap ops
                    (BooleanVertex.all domain.laneVariables) (fun lane =>
                      ops.mul
                        (ops.mul
                          (column.equalityWeight ops coins.betaM)
                          (lane.equalityWeight ops coins.betaA))
                        (SourceProjection.rangeValueAt covers data source {
                          column := column.toCubePoint ops
                          lane := lane.toCubePoint ops })))) := by
              apply FiniteSumAlgebra.sumMap_congr
              intro column _
              exact FiniteSumAlgebra.sumMap_mul_left ops laws weight _ _
          _ = ops.mul weight
              (FiniteSumAlgebra.sumMap ops
                (BooleanVertex.all domain.columnVariables) (fun column =>
                  FiniteSumAlgebra.sumMap ops
                    (BooleanVertex.all domain.laneVariables) (fun lane =>
                      ops.mul
                        (ops.mul
                          (column.equalityWeight ops coins.betaM)
                          (lane.equalityWeight ops coins.betaA))
                        (SourceProjection.rangeValueAt covers data source {
                          column := column.toCubePoint ops
                          lane := lane.toCubePoint ops })))) :=
              FiniteSumAlgebra.sumMap_mul_left ops laws weight _ _
    _ = _ := rfl

/-- Total semantic evaluator used only to instantiate generic SumCheck truth.
The public evaluator remains fail-closed; exact round-count checking makes the
default branch unreachable. -/
def sumcheckPolynomial
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (coordinates : List K) : K :=
  (polynomial convention covers data coins coordinates).getD K.zero

/-- On an exact typed point, totalization agrees with the source-derived NC
polynomial. -/
theorem sumcheckPolynomial_coordinates_eq_qAtPoint
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (point : Point domain) :
    sumcheckPolynomial convention covers data coins point.coordinates =
      qAtPoint convention covers data coins point := by
  unfold sumcheckPolynomial
  rw [polynomial_coordinates_eq_qAtPoint]
  rfl

/-- Generic recursive SumCheck initial sum for the exact NC round count. -/
def sumcheckHypercubeSum
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain) : K :=
  SumCheck.Finite.HypercubeTruth.sumCompletions ops.toOps
    (sumcheckPolynomial convention covers data coins) []
    (domain.columnVariables + domain.laneVariables)

/-- The recursive completion sum and typed product sum enumerate the same
column-then-lane Boolean points. -/
theorem sumcheckHypercubeSum_eq_hypercubeSum
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain) :
    sumcheckHypercubeSum convention covers data coins =
      hypercubeSum convention covers data coins := by
  unfold sumcheckHypercubeSum
  rw [SumCheck.Finite.HypercubeTruth.sumCompletions_add]
  rw [SumCheckTruthPath.sumCompletions_eq_vertexSum ops laws]
  unfold hypercubeSum FiniteSumAlgebra.sumMap
  simp only [List.nil_append]
  congr 1
  apply List.map_congr_left
  intro column _
  rw [SumCheckTruthPath.sumCompletions_eq_vertexSum ops laws]
  congr 1
  apply List.map_congr_left
  intro lane _
  exact sumcheckPolynomial_coordinates_eq_qAtPoint
    convention covers data coins {
      column := column.toCubePoint ops
      lane := lane.toCubePoint ops }

/-- The verifier's NC initial claim is a constant, not a certificate field. -/
def claimedInitial : K := K.zero

private theorem rangeValueAt_toCubePoint_eq_zero_of_truth
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (truth : Semantics.Nc.Truth data)
    (source : Fin shape.sourceCount)
    (column : BooleanVertex domain.columnVariables)
    (lane : BooleanVertex domain.laneVariables) :
    SourceProjection.rangeValueAt covers data source {
        column := column.toCubePoint ops
        lane := lane.toCubePoint ops } = K.zero := by
  have zeroAtIndex :=
    SourceProjection.booleanResidualsZero_of_truth covers data truth source
      (columnIndex column) (laneIndex lane)
  simpa [booleanPoint] using zeroAtIndex

/-- Every named gamma convention has zero pointwise source mixture under
honest full-carrier norm truth. -/
theorem mixedRangeAt_toCubePoint_eq_zero_of_truth
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (truth : Semantics.Nc.Truth data)
    (column : BooleanVertex domain.columnVariables)
    (lane : BooleanVertex domain.laneVariables) :
    mixedRangeAt convention covers data coins {
        column := column.toCubePoint ops
        lane := lane.toCubePoint ops } = K.zero := by
  unfold mixedRangeAt
  calc
    FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.sourceCount) _ =
      FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.sourceCount) (fun _ => ops.zero) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro source _
          unfold SignedJointIdentity.gammaTerm
          rw [rangeValueAt_toCubePoint_eq_zero_of_truth
            covers data truth source column lane]
          exact laws.mul_zero _
    _ = ops.zero := FiniteSumAlgebra.sumMap_zero ops laws _

/-- Honest norm truth makes the typed NC Boolean sum zero for every exponent
convention and every challenge tuple. -/
theorem hypercubeSum_eq_zero_of_truth
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (truth : Semantics.Nc.Truth data) :
    hypercubeSum convention covers data coins = K.zero := by
  unfold hypercubeSum
  calc
    FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.columnVariables) _ =
      FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all domain.columnVariables) (fun _ => ops.zero) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro column _
          calc
            FiniteSumAlgebra.sumMap ops
                (BooleanVertex.all domain.laneVariables) _ =
              FiniteSumAlgebra.sumMap ops
                (BooleanVertex.all domain.laneVariables)
                (fun _ => ops.zero) := by
                  apply FiniteSumAlgebra.sumMap_congr
                  intro lane _
                  unfold qAtPoint
                  rw [mixedRangeAt_toCubePoint_eq_zero_of_truth
                    convention covers data coins truth column lane]
                  exact laws.mul_zero _
            _ = ops.zero := FiniteSumAlgebra.sumMap_zero ops laws _
    _ = ops.zero := FiniteSumAlgebra.sumMap_zero ops laws _

/-- Honest norm truth also zeros the independently grouped source
specialization mix. -/
theorem mixedResidualAtBeta_eq_zero_of_truth
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (truth : Semantics.Nc.Truth data) :
    mixedResidualAtBeta convention covers data coins = K.zero := by
  rw [← hypercubeSum_eq_mixedResidualAtBeta]
  exact hypercubeSum_eq_zero_of_truth
    convention covers data coins truth

/-- Honest full-carrier norm truth makes the literal zero claim equal the
generic recursive NC SumCheck cube sum. No output message or trusted
`trueInitial` value appears in the statement. -/
theorem claimedInitial_eq_sumcheckHypercubeSum_of_truth
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (truth : Semantics.Nc.Truth data) :
    claimedInitial =
      sumcheckHypercubeSum convention covers data coins := by
  rw [sumcheckHypercubeSum_eq_hypercubeSum]
  exact (hypercubeSum_eq_zero_of_truth
    convention covers data coins truth).symm

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum
