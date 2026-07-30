import Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
import Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpoints

/-!
Contract: exact finite column support for the three selected Split-NC endpoint
programs.

The endpoint emitters already derive their row and allocation counts. This
module proves that authoritative inputs placed before the endpoint allocation,
together with explicitly materialized endpoint claims inside the enclosing
boundary, suffice to bound every row dependency by that boundary. It emits no
rows and assigns no semantic meaning to a column.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointSupport

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

private theorem feInitial_ordinal_lt
    {shape : SemanticShape}
    (index : KSplitNcFeInitial.Index shape) :
    KSplitNcFeInitial.ordinal index <
      shape.matrixCount * shape.runningCount := by
  unfold KSplitNcFeInitial.ordinal
  calc
    index.1.val * shape.runningCount + index.2.val <
        index.1.val * shape.runningCount + shape.runningCount :=
      Nat.add_lt_add_left index.2.isLt _
    _ = (index.1.val + 1) * shape.runningCount := by
      rw [Nat.add_mul, Nat.one_mul]
    _ ≤ shape.matrixCount * shape.runningCount :=
      Nat.mul_le_mul_right shape.runningCount
        (Nat.succ_le_iff.mpr index.1.isLt)

private theorem feInitial_mle_end
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcFeInitial.allocationWidth input ≤ boundary)
    (index : KSplitNcFeInitial.Index shape) :
    KSplitNcFeInitial.mleBase input index +
        3 * KBooleanMle.frameCount domain.laneVariables ≤
      boundary := by
  have ordinalBound :
      KSplitNcFeInitial.ordinal index + 1 ≤
        shape.matrixCount * shape.runningCount :=
    Nat.succ_le_iff.mpr (feInitial_ordinal_lt index)
  have scaled :=
    Nat.mul_le_mul_left
      (KSplitNcFeInitial.rowsPerMle domain) ordinalBound
  unfold KSplitNcFeInitial.mleBase at ⊢
  unfold KSplitNcFeInitial.allocationWidth at allocationEnd
  unfold KSplitNcFeInitial.rowsPerMle at scaled allocationEnd ⊢
  simp only [Nat.mul_succ] at scaled
  omega

private theorem feInitial_table_below
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (sourcesBelow :
      ∀ running matrix lane,
        CarriedBelow
          (input.claimedYRing running matrix lane) input.frameBase)
    (index : KSplitNcFeInitial.Index shape) :
    KBooleanMleSupport.TableBelowBase
      (KSplitNcFeInitial.table input index)
      (KSplitNcFeInitial.mleBase input index) := by
  unfold KSplitNcFeInitial.table
  apply paddedTable_below
  intro lane
  exact carried_mono
    (sourcesBelow index.2 index.1 lane)
    (by unfold KSplitNcFeInitial.mleBase; omega)

private theorem feInitial_coordinates_below
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (alphaBelow :
      ∀ coordinate, CarriedBelow (input.alpha coordinate) input.frameBase)
    (index : KSplitNcFeInitial.Index shape) :
    KBooleanMleSupport.CoordinatesBelowBase
      (KSplitNcFeInitial.alphaCoordinates input)
      (KSplitNcFeInitial.mleBase input index) := by
  unfold KSplitNcFeInitial.alphaCoordinates
  apply coordinates_below_ofFn
  intro coordinate
  exact carried_mono (alphaBelow coordinate)
    (by unfold KSplitNcFeInitial.mleBase; omega)

private theorem feInitial_mleRows_below
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (boundary : Nat)
    (alphaBelow :
      ∀ coordinate, CarriedBelow (input.alpha coordinate) input.frameBase)
    (sourcesBelow :
      ∀ running matrix lane,
        CarriedBelow
          (input.claimedYRing running matrix lane) input.frameBase)
    (allocationEnd :
      input.frameBase + KSplitNcFeInitial.allocationWidth input ≤ boundary) :
    RowsBelow (KSplitNcFeInitial.mleRows input) boundary := by
  intro row member column mentioned
  rcases List.mem_flatMap.mp member with
    ⟨index, _, inRows⟩
  exact boolean_rows_below
    (KSplitNcFeInitial.table input index)
    (KSplitNcFeInitial.alphaCoordinates input)
    (feInitial_table_below input sourcesBelow index)
    (feInitial_coordinates_below input alphaBelow index)
    (feInitial_mle_end input boundary allocationEnd index)
    row inRows column mentioned

private theorem feInitial_mleOutput_below
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (boundary : Nat)
    (alphaBelow :
      ∀ coordinate, CarriedBelow (input.alpha coordinate) input.frameBase)
    (sourcesBelow :
      ∀ running matrix lane,
        CarriedBelow
          (input.claimedYRing running matrix lane) input.frameBase)
    (allocationEnd :
      input.frameBase + KSplitNcFeInitial.allocationWidth input ≤ boundary)
    (index : KSplitNcFeInitial.Index shape) :
    CarriedBelow (KSplitNcFeInitial.mleOutput input index) boundary := by
  exact boolean_output_below
    (KSplitNcFeInitial.table input index)
    (KSplitNcFeInitial.alphaCoordinates input)
    (feInitial_table_below input sourcesBelow index)
    (feInitial_coordinates_below input alphaBelow index)
    (feInitial_mle_end input boundary allocationEnd index)

private theorem feInitial_matrixBlock_below
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (boundary : Nat)
    (alphaBelow :
      ∀ coordinate, CarriedBelow (input.alpha coordinate) input.frameBase)
    (sourcesBelow :
      ∀ running matrix lane,
        CarriedBelow
          (input.claimedYRing running matrix lane) input.frameBase)
    (allocationEnd :
      input.frameBase + KSplitNcFeInitial.allocationWidth input ≤ boundary)
    (matrix : Fin shape.matrixCount) :
    ∀ coefficient ∈ KSplitNcFeInitial.matrixBlock input matrix,
      CarriedBelow coefficient boundary := by
  intro coefficient member
  rcases List.mem_append.mp member with inZeros | inOutputs
  · have same : coefficient = KLinear.zeroCarried :=
      List.eq_of_mem_replicate inZeros
    subst coefficient
    exact zero_below boundary
  · rcases List.mem_map.mp inOutputs with
      ⟨running, _, rfl⟩
    exact feInitial_mleOutput_below input boundary alphaBelow
      sourcesBelow allocationEnd (matrix, running)

private theorem feInitial_coefficients_below
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (boundary : Nat)
    (alphaBelow :
      ∀ coordinate, CarriedBelow (input.alpha coordinate) input.frameBase)
    (sourcesBelow :
      ∀ running matrix lane,
        CarriedBelow
          (input.claimedYRing running matrix lane) input.frameBase)
    (allocationEnd :
      input.frameBase + KSplitNcFeInitial.allocationWidth input ≤ boundary) :
    ∀ coefficient ∈ KSplitNcFeInitial.coefficients input,
      CarriedBelow coefficient boundary := by
  intro coefficient member
  rcases List.mem_append.mp member with inZeros | inMatrices
  · have same : coefficient = KLinear.zeroCarried :=
      List.eq_of_mem_replicate inZeros
    subst coefficient
    exact zero_below boundary
  · rcases List.mem_flatMap.mp inMatrices with
      ⟨matrix, _, inBlock⟩
    exact feInitial_matrixBlock_below input boundary alphaBelow
      sourcesBelow allocationEnd matrix coefficient inBlock

private theorem feInitial_horner_end
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcFeInitial.allocationWidth input ≤ boundary) :
    KSplitNcFeInitial.hornerBase input +
        3 * ((KSplitNcFeInitial.coefficients input).length - 1) ≤
      boundary := by
  rw [KSplitNcFeInitial.coefficients_length]
  unfold KSplitNcFeInitial.hornerBase
  unfold KSplitNcFeInitial.allocationWidth at allocationEnd
  omega

theorem feInitial_rows_below
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain)
    (boundary : Nat)
    (positive : 0 < boundary)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (alphaBelow :
      ∀ coordinate, CarriedBelow (input.alpha coordinate) input.frameBase)
    (sourcesBelow :
      ∀ running matrix lane,
        CarriedBelow
          (input.claimedYRing running matrix lane) input.frameBase)
    (initialBelow : CarriedBelow input.initial boundary)
    (allocationEnd :
      input.frameBase + KSplitNcFeInitial.allocationWidth input ≤ boundary) :
    RowsBelow (KSplitNcFeInitial.rows input) boundary := by
  intro row member column mentioned
  rcases List.mem_append.mp member with inPrefix | inEquality
  · rcases List.mem_append.mp inPrefix with inMle | inHorner
    · exact feInitial_mleRows_below input boundary alphaBelow
        sourcesBelow allocationEnd row inMle column mentioned
    · exact horner_rows_below input.gamma
        (KSplitNcFeInitial.coefficients input)
        (KSplitNcFeInitial.hornerBase input) 0 boundary
        (carried_mono gammaBelow (by
          exact Nat.le_trans
            (Nat.le_add_right input.frameBase
              (KSplitNcFeInitial.allocationWidth input))
            allocationEnd))
        (feInitial_coefficients_below input boundary alphaBelow
          sourcesBelow allocationEnd)
        (by simpa using
          feInitial_horner_end input boundary allocationEnd)
        row inHorner column mentioned
  · apply equality_rows_below
      (KSplitNcFeInitial.evaluated input) input.initial boundary positive
    · exact horner_output_below input.gamma
        (KSplitNcFeInitial.coefficients input)
        (KSplitNcFeInitial.hornerBase input) 0 boundary
        (feInitial_coefficients_below input boundary alphaBelow
          sourcesBelow allocationEnd)
        (by simpa using
          feInitial_horner_end input boundary allocationEnd)
    · exact initialBelow
    · exact inEquality
    · exact mentioned

private theorem feTerminal_external_mono
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcFeTerminal.allocationWidth input ≤ boundary)
    {value : Carried}
    (below : CarriedBelow value input.frameBase) :
    CarriedBelow value boundary :=
  carried_mono below
    (Nat.le_trans
      (Nat.le_add_right input.frameBase
        (KSplitNcFeTerminal.allocationWidth input))
      allocationEnd)

private theorem feTerminal_fresh_sparse_end
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcFeTerminal.allocationWidth input ≤ boundary)
    (fresh : Fin shape.freshCount) :
    (KSplitNcFeTerminal.freshPolynomialInput input fresh).frameBase +
        3 * KSparsePolynomial.totalDegreeSum
          (KSplitNcFeTerminal.freshPolynomialInput input fresh).polynomial.terms ≤
      boundary := by
  change input.frameBase +
        fresh.val * KSplitNcFeTerminal.sparseRowsPerFresh input +
        KSplitNcFeTerminal.sparseRowsPerFresh input ≤ boundary
  have freshBound : fresh.val + 1 ≤ shape.freshCount :=
    Nat.succ_le_iff.mpr fresh.isLt
  have scaled :=
    Nat.mul_le_mul_left
      (KSplitNcFeTerminal.sparseRowsPerFresh input) freshBound
  unfold KSplitNcFeTerminal.allocationWidth at allocationEnd
  simp only [Nat.mul_succ] at scaled
  rw [Nat.mul_comm fresh.val
    (KSplitNcFeTerminal.sparseRowsPerFresh input)]
  omega

private theorem feTerminal_freshRows_below
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (boundary : Nat) (positive : 0 < boundary)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase)
    (allocationEnd :
      input.frameBase + KSplitNcFeTerminal.allocationWidth input ≤ boundary) :
    RowsBelow (KSplitNcFeTerminal.freshRows input) boundary := by
  intro row member column mentioned
  rcases List.mem_flatMap.mp member with
    ⟨fresh, _, inRows⟩
  apply sparsePolynomial_rows_below
      (KSplitNcFeTerminal.freshPolynomialInput input fresh)
      positive
  · intro matrix
    exact feTerminal_external_mono input boundary allocationEnd
      (messageBelow (Data.freshIndex fresh) matrix
        Phi81CoefficientKernel.constant)
  · exact feTerminal_fresh_sparse_end input boundary allocationEnd fresh
  · exact inRows
  · exact mentioned

private theorem feTerminal_freshOutput_below
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (boundary : Nat) (positive : 0 < boundary)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase)
    (allocationEnd :
      input.frameBase + KSplitNcFeTerminal.allocationWidth input ≤ boundary)
    (fresh : Fin shape.freshCount) :
    CarriedBelow
      (KSparsePolynomial.output
        (KSplitNcFeTerminal.freshPolynomialInput input fresh))
      boundary := by
  apply sparsePolynomial_output_below
      (KSplitNcFeTerminal.freshPolynomialInput input fresh)
      positive
  · intro matrix
    exact feTerminal_external_mono input boundary allocationEnd
      (messageBelow (Data.freshIndex fresh) matrix
        Phi81CoefficientKernel.constant)
  · exact feTerminal_fresh_sparse_end input boundary allocationEnd fresh

private theorem feTerminal_freshOutputs_below
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (boundary : Nat) (positive : 0 < boundary)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase)
    (allocationEnd :
      input.frameBase + KSplitNcFeTerminal.allocationWidth input ≤ boundary) :
    ∀ output ∈ KSplitNcFeTerminal.freshOutputs input,
      CarriedBelow output boundary := by
  intro output member
  rcases List.mem_map.mp member with ⟨fresh, _, rfl⟩
  exact feTerminal_freshOutput_below input boundary positive
    messageBelow allocationEnd fresh

private theorem feTerminal_fresh_horner_end
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcFeTerminal.allocationWidth input ≤ boundary) :
    KSplitNcFeTerminal.freshHornerBase input +
        3 * ((KSplitNcFeTerminal.freshOutputs input).length - 1) ≤
      boundary := by
  rw [KSplitNcFeTerminal.freshOutputs_length]
  unfold KSplitNcFeTerminal.freshHornerBase
  unfold KSplitNcFeTerminal.allocationWidth at allocationEnd
  omega

private theorem feTerminal_carriedTarget_below
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcFeTerminal.allocationWidth input ≤ boundary) :
    CarriedBelow (KSplitNcFeTerminal.carriedTarget input) boundary := by
  constructor <;> intro column mentioned
  · have same : column = KSplitNcFeTerminal.carriedTargetBase input := by
      simpa [KSplitNcFeTerminal.carriedTarget, Mentions] using mentioned
    subst column
    unfold KSplitNcFeTerminal.allocationWidth at allocationEnd
    unfold KSplitNcFeTerminal.carriedTargetBase
      KSplitNcFeTerminal.carriedBase
      KSplitNcFeTerminal.carriedInternalWidth
      KSplitNcFeTerminal.freshHornerBase
    unfold KSplitNcFeTerminal.carriedInternalWidth at allocationEnd
    omega
  · have same :
        column = KSplitNcFeTerminal.carriedTargetBase input + 1 := by
      simpa [KSplitNcFeTerminal.carriedTarget, Mentions] using mentioned
    subst column
    unfold KSplitNcFeTerminal.allocationWidth at allocationEnd
    unfold KSplitNcFeTerminal.carriedTargetBase
      KSplitNcFeTerminal.carriedBase
      KSplitNcFeTerminal.carriedInternalWidth
      KSplitNcFeTerminal.freshHornerBase
    unfold KSplitNcFeTerminal.carriedInternalWidth at allocationEnd
    omega

private theorem feTerminal_carried_allocation_end
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcFeTerminal.allocationWidth input ≤ boundary) :
    (KSplitNcFeTerminal.carriedInput input).frameBase +
        KSplitNcFeInitial.allocationWidth
          (KSplitNcFeTerminal.carriedInput input) ≤
      boundary := by
  change KSplitNcFeTerminal.carriedBase input +
      KSplitNcFeTerminal.carriedInternalWidth input ≤ boundary
  unfold KSplitNcFeTerminal.carriedBase
    KSplitNcFeTerminal.freshHornerBase
  unfold KSplitNcFeTerminal.allocationWidth at allocationEnd
  omega

private theorem feTerminal_product_end
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcFeTerminal.allocationWidth input ≤ boundary) :
    KSplitNcFeTerminal.productBase input + 12 ≤ boundary := by
  unfold KSplitNcFeTerminal.allocationWidth at allocationEnd
  unfold KSplitNcFeTerminal.productBase
    KSplitNcFeTerminal.equalityBase
    KSplitNcFeTerminal.carriedTargetBase
    KSplitNcFeTerminal.carriedBase
    KSplitNcFeTerminal.carriedInternalWidth
    KSplitNcFeTerminal.freshHornerBase
    KSplitNcFeTerminal.pointEqualityRows
  unfold KSplitNcFeTerminal.carriedInternalWidth
    KSplitNcFeTerminal.pointEqualityRows at allocationEnd
  omega

private theorem feTerminal_point_end
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcFeTerminal.allocationWidth input ≤ boundary)
    {variables : Nat}
    (pointInput : KPointEquality.Input variables)
    (baseMatch :
      pointInput.frameBase + KSplitNcFeTerminal.pointEqualityRows variables ≤
        KSplitNcFeTerminal.productBase input) :
    pointInput.frameBase + 3 * variables + 3 * (variables - 1) ≤
      boundary := by
  unfold KSplitNcFeTerminal.pointEqualityRows at baseMatch
  have productEnd := feTerminal_product_end input boundary allocationEnd
  omega

private theorem feTerminal_le_carriedBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) :
    input.frameBase ≤
      (KSplitNcFeTerminal.carriedInput input).frameBase := by
  change input.frameBase ≤ KSplitNcFeTerminal.carriedBase input
  unfold KSplitNcFeTerminal.carriedBase
    KSplitNcFeTerminal.freshHornerBase
  omega

private theorem feTerminal_freshLane_end
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) :
    (KSplitNcFeTerminal.freshLaneEqualityInput input).frameBase +
        KSplitNcFeTerminal.pointEqualityRows domain.laneVariables ≤
      KSplitNcFeTerminal.productBase input := by
  change KSplitNcFeTerminal.equalityBase input +
      KSplitNcFeTerminal.pointEqualityRows domain.laneVariables ≤
    KSplitNcFeTerminal.productBase input
  unfold KSplitNcFeTerminal.productBase
  omega

private theorem feTerminal_freshRow_end
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) :
    (KSplitNcFeTerminal.freshRowEqualityInput input).frameBase +
        KSplitNcFeTerminal.pointEqualityRows shape.rowVariables ≤
      KSplitNcFeTerminal.productBase input := by
  change KSplitNcFeTerminal.equalityBase input +
        KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
        KSplitNcFeTerminal.pointEqualityRows shape.rowVariables ≤
    KSplitNcFeTerminal.productBase input
  unfold KSplitNcFeTerminal.productBase
  omega

private theorem feTerminal_carriedLane_end
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) :
    (KSplitNcFeTerminal.carriedLaneEqualityInput input).frameBase +
        KSplitNcFeTerminal.pointEqualityRows domain.laneVariables ≤
      KSplitNcFeTerminal.productBase input := by
  change KSplitNcFeTerminal.equalityBase input +
        KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
        KSplitNcFeTerminal.pointEqualityRows shape.rowVariables +
        KSplitNcFeTerminal.pointEqualityRows domain.laneVariables ≤
    KSplitNcFeTerminal.productBase input
  unfold KSplitNcFeTerminal.productBase
  omega

private theorem feTerminal_carriedRow_end
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) :
    (KSplitNcFeTerminal.carriedRowEqualityInput input).frameBase +
        KSplitNcFeTerminal.pointEqualityRows shape.rowVariables ≤
      KSplitNcFeTerminal.productBase input := by
  change KSplitNcFeTerminal.equalityBase input +
        2 * KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
        KSplitNcFeTerminal.pointEqualityRows shape.rowVariables +
        KSplitNcFeTerminal.pointEqualityRows shape.rowVariables ≤
    KSplitNcFeTerminal.productBase input
  unfold KSplitNcFeTerminal.productBase
  omega

theorem feTerminal_rows_below
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (boundary : Nat) (positive : 0 < boundary)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (alphaBelow :
      ∀ coordinate, CarriedBelow (input.alpha coordinate) input.frameBase)
    (betaABelow :
      ∀ coordinate, CarriedBelow (input.betaA coordinate) input.frameBase)
    (betaRBelow :
      ∀ coordinate, CarriedBelow (input.betaR coordinate) input.frameBase)
    (pointLaneBelow :
      ∀ coordinate, CarriedBelow (input.pointLane coordinate) input.frameBase)
    (pointRowBelow :
      ∀ coordinate, CarriedBelow (input.pointRow coordinate) input.frameBase)
    (priorPointBelow :
      ∀ coordinate, CarriedBelow (input.priorPoint coordinate) input.frameBase)
    (messageBelow :
      ∀ source matrix lane,
        CarriedBelow
          (input.messageYRing source matrix lane) input.frameBase)
    (terminalBelow : CarriedBelow input.terminal boundary)
    (allocationEnd :
      input.frameBase + KSplitNcFeTerminal.allocationWidth input ≤ boundary) :
    RowsBelow (KSplitNcFeTerminal.rows input) boundary := by
  have gammaBoundary :=
    feTerminal_external_mono input boundary allocationEnd gammaBelow
  have freshOutputsBoundary :=
    feTerminal_freshOutputs_below input boundary positive
      messageBelow allocationEnd
  have freshHornerRows :
      RowsBelow (KSplitNcFeTerminal.freshHornerRows input) boundary :=
    horner_rows_below input.gamma
      (KSplitNcFeTerminal.freshOutputs input)
      (KSplitNcFeTerminal.freshHornerBase input) 0 boundary
      gammaBoundary freshOutputsBoundary
      (by simpa using
        feTerminal_fresh_horner_end input boundary allocationEnd)
  have freshOutputBoundary :
      CarriedBelow (KSplitNcFeTerminal.freshOutput input) boundary :=
    horner_output_below input.gamma
      (KSplitNcFeTerminal.freshOutputs input)
      (KSplitNcFeTerminal.freshHornerBase input) 0 boundary
      freshOutputsBoundary
      (by simpa using
        feTerminal_fresh_horner_end input boundary allocationEnd)
  have carriedTargetBoundary :=
    feTerminal_carriedTarget_below input boundary allocationEnd
  have carriedRows :
      RowsBelow (KSplitNcFeTerminal.carriedRows input) boundary := by
    unfold KSplitNcFeTerminal.carriedRows
    apply feInitial_rows_below
        (KSplitNcFeTerminal.carriedInput input)
        boundary positive
    · exact carried_mono gammaBelow (feTerminal_le_carriedBase input)
    · intro coordinate
      exact carried_mono (pointLaneBelow coordinate)
        (feTerminal_le_carriedBase input)
    · intro running matrix lane
      exact carried_mono
        (messageBelow (Data.runningIndex running) matrix lane)
        (feTerminal_le_carriedBase input)
    · exact carriedTargetBoundary
    · exact feTerminal_carried_allocation_end
        input boundary allocationEnd
  have freshLaneRows :
      RowsBelow
        (KPointEquality.rows
          (KSplitNcFeTerminal.freshLaneEqualityInput input))
        boundary := by
    apply pointEquality_rows_below _ positive
    · intro coordinate
      exact feTerminal_external_mono input boundary allocationEnd
        (pointLaneBelow coordinate)
    · intro coordinate
      exact feTerminal_external_mono input boundary allocationEnd
        (betaABelow coordinate)
    · exact feTerminal_point_end input boundary allocationEnd
        _ (feTerminal_freshLane_end input)
  have freshRowRows :
      RowsBelow
        (KPointEquality.rows
          (KSplitNcFeTerminal.freshRowEqualityInput input))
        boundary := by
    apply pointEquality_rows_below _ positive
    · intro coordinate
      exact feTerminal_external_mono input boundary allocationEnd
        (pointRowBelow coordinate)
    · intro coordinate
      exact feTerminal_external_mono input boundary allocationEnd
        (betaRBelow coordinate)
    · exact feTerminal_point_end input boundary allocationEnd
        _ (feTerminal_freshRow_end input)
  have carriedLaneRows :
      RowsBelow
        (KPointEquality.rows
          (KSplitNcFeTerminal.carriedLaneEqualityInput input))
        boundary := by
    apply pointEquality_rows_below _ positive
    · intro coordinate
      exact feTerminal_external_mono input boundary allocationEnd
        (pointLaneBelow coordinate)
    · intro coordinate
      exact feTerminal_external_mono input boundary allocationEnd
        (alphaBelow coordinate)
    · exact feTerminal_point_end input boundary allocationEnd
        _ (feTerminal_carriedLane_end input)
  have carriedRowRows :
      RowsBelow
        (KPointEquality.rows
          (KSplitNcFeTerminal.carriedRowEqualityInput input))
        boundary := by
    apply pointEquality_rows_below _ positive
    · intro coordinate
      exact feTerminal_external_mono input boundary allocationEnd
        (pointRowBelow coordinate)
    · intro coordinate
      exact feTerminal_external_mono input boundary allocationEnd
        (priorPointBelow coordinate)
    · exact feTerminal_point_end input boundary allocationEnd
        _ (feTerminal_carriedRow_end input)
  have freshLaneOutput :=
    pointEquality_output_below
      (KSplitNcFeTerminal.freshLaneEqualityInput input)
      positive
      (fun coordinate =>
        feTerminal_external_mono input boundary allocationEnd
          (betaABelow coordinate))
      (feTerminal_point_end input boundary allocationEnd
        (KSplitNcFeTerminal.freshLaneEqualityInput input)
        (feTerminal_freshLane_end input))
  have freshRowOutput :=
    pointEquality_output_below
      (KSplitNcFeTerminal.freshRowEqualityInput input)
      positive
      (fun coordinate =>
        feTerminal_external_mono input boundary allocationEnd
          (betaRBelow coordinate))
      (feTerminal_point_end input boundary allocationEnd
        (KSplitNcFeTerminal.freshRowEqualityInput input)
        (feTerminal_freshRow_end input))
  have carriedLaneOutput :=
    pointEquality_output_below
      (KSplitNcFeTerminal.carriedLaneEqualityInput input)
      positive
      (fun coordinate =>
        feTerminal_external_mono input boundary allocationEnd
          (alphaBelow coordinate))
      (feTerminal_point_end input boundary allocationEnd
        (KSplitNcFeTerminal.carriedLaneEqualityInput input)
        (feTerminal_carriedLane_end input))
  have carriedRowOutput :=
    pointEquality_output_below
      (KSplitNcFeTerminal.carriedRowEqualityInput input)
      positive
      (fun coordinate =>
        feTerminal_external_mono input boundary allocationEnd
          (priorPointBelow coordinate))
      (feTerminal_point_end input boundary allocationEnd
        (KSplitNcFeTerminal.carriedRowEqualityInput input)
        (feTerminal_carriedRow_end input))
  have productEnd :=
    feTerminal_product_end input boundary allocationEnd
  have freshSelectorRows :
      RowsBelow
        (KMul.rows
          (KPointEquality.equalityCarried
            (KSplitNcFeTerminal.freshLaneEqualityInput input))
          (KPointEquality.equalityCarried
            (KSplitNcFeTerminal.freshRowEqualityInput input))
          (KSplitNcFeTerminal.freshSelectorFrame input))
        boundary :=
    mul_rows_below _ _ (KSplitNcFeTerminal.productBase input) 0
      boundary freshLaneOutput freshRowOutput (by omega)
  have freshSelectorBoundary :
      CarriedBelow (KSplitNcFeTerminal.freshSelector input) boundary :=
    frame_output_below (KSplitNcFeTerminal.productBase input)
      0 boundary (by omega)
  have freshContributionRows :
      RowsBelow
        (KMul.rows
          (KSplitNcFeTerminal.freshSelector input)
          (KSplitNcFeTerminal.freshOutput input)
          (KSplitNcFeTerminal.freshContributionFrame input))
        boundary :=
    mul_rows_below _ _ (KSplitNcFeTerminal.productBase input) 1
      boundary freshSelectorBoundary freshOutputBoundary (by omega)
  have carriedSelectorRows :
      RowsBelow
        (KMul.rows
          (KPointEquality.equalityCarried
            (KSplitNcFeTerminal.carriedLaneEqualityInput input))
          (KPointEquality.equalityCarried
            (KSplitNcFeTerminal.carriedRowEqualityInput input))
          (KSplitNcFeTerminal.carriedSelectorFrame input))
        boundary :=
    mul_rows_below _ _ (KSplitNcFeTerminal.productBase input) 2
      boundary carriedLaneOutput carriedRowOutput (by omega)
  have carriedSelectorBoundary :
      CarriedBelow (KSplitNcFeTerminal.carriedSelector input) boundary :=
    frame_output_below (KSplitNcFeTerminal.productBase input)
      2 boundary (by omega)
  have carriedContributionRows :
      RowsBelow
        (KMul.rows
          (KSplitNcFeTerminal.carriedSelector input)
          (KSplitNcFeTerminal.carriedTarget input)
          (KSplitNcFeTerminal.carriedContributionFrame input))
        boundary :=
    mul_rows_below _ _ (KSplitNcFeTerminal.productBase input) 3
      boundary carriedSelectorBoundary carriedTargetBoundary (by omega)
  have freshContributionBoundary :
      CarriedBelow (KSplitNcFeTerminal.freshContribution input) boundary :=
    frame_output_below (KSplitNcFeTerminal.productBase input)
      1 boundary (by omega)
  have carriedContributionBoundary :
      CarriedBelow (KSplitNcFeTerminal.carriedContribution input) boundary :=
    frame_output_below (KSplitNcFeTerminal.productBase input)
      3 boundary (by omega)
  have finalRows :
      RowsBelow
        (KEquality.rows
          (KSplitNcFeTerminal.terminalExpression input) input.terminal)
        boundary :=
    equality_rows_below _ _ boundary positive
      (add_below freshContributionBoundary carriedContributionBoundary)
      terminalBelow
  intro row member column mentioned
  rcases List.mem_flatten.mp member with
    ⟨group, groupMember, rowMember⟩
  simp only [KSplitNcFeTerminal.rowGroups, List.mem_cons,
    List.not_mem_nil, or_false] at groupMember
  rcases groupMember with
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact feTerminal_freshRows_below input boundary positive
      messageBelow allocationEnd row rowMember column mentioned
  · exact freshHornerRows row rowMember column mentioned
  · exact carriedRows row rowMember column mentioned
  · exact freshLaneRows row rowMember column mentioned
  · exact freshRowRows row rowMember column mentioned
  · exact carriedLaneRows row rowMember column mentioned
  · exact carriedRowRows row rowMember column mentioned
  · exact freshSelectorRows row rowMember column mentioned
  · exact freshContributionRows row rowMember column mentioned
  · exact carriedSelectorRows row rowMember column mentioned
  · exact carriedContributionRows row rowMember column mentioned
  · exact finalRows row rowMember column mentioned

private theorem nc_mle_end
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcNcEndpoint.allocationWidth input ≤ boundary)
    (source : Fin shape.sourceCount) :
    KSplitNcNcEndpoint.mleBase input source +
        3 * KBooleanMle.frameCount domain.laneVariables ≤
      boundary := by
  change input.frameBase +
        KSplitNcNcEndpoint.rowsPerMle domain * source.val +
        KSplitNcNcEndpoint.rowsPerMle domain ≤ boundary
  have sourceBound : source.val + 1 ≤ shape.sourceCount :=
    Nat.succ_le_iff.mpr source.isLt
  have scaled :=
    Nat.mul_le_mul_left
      (KSplitNcNcEndpoint.rowsPerMle domain) sourceBound
  unfold KSplitNcNcEndpoint.allocationWidth at allocationEnd
  simp only [Nat.mul_succ] at scaled
  omega

private theorem nc_table_below
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (messageBelow :
      ∀ source lane,
        CarriedBelow (input.messageYZcol source lane) input.frameBase)
    (source : Fin shape.sourceCount) :
    KBooleanMleSupport.TableBelowBase
      (KSplitNcNcEndpoint.sourceTable input source)
      (KSplitNcNcEndpoint.mleBase input source) := by
  unfold KSplitNcNcEndpoint.sourceTable
  apply paddedTable_below
  intro lane
  exact carried_mono (messageBelow source lane)
    (by unfold KSplitNcNcEndpoint.mleBase; omega)

private theorem nc_coordinates_below
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (pointLaneBelow :
      ∀ coordinate,
        CarriedBelow (input.pointLane coordinate) input.frameBase)
    (source : Fin shape.sourceCount) :
    KBooleanMleSupport.CoordinatesBelowBase
      (KSplitNcNcEndpoint.laneCoordinates input)
      (KSplitNcNcEndpoint.mleBase input source) := by
  unfold KSplitNcNcEndpoint.laneCoordinates
  apply coordinates_below_ofFn
  intro coordinate
  exact carried_mono (pointLaneBelow coordinate)
    (by unfold KSplitNcNcEndpoint.mleBase; omega)

private theorem nc_mleRows_below
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (boundary : Nat)
    (pointLaneBelow :
      ∀ coordinate,
        CarriedBelow (input.pointLane coordinate) input.frameBase)
    (messageBelow :
      ∀ source lane,
        CarriedBelow (input.messageYZcol source lane) input.frameBase)
    (allocationEnd :
      input.frameBase + KSplitNcNcEndpoint.allocationWidth input ≤ boundary) :
    RowsBelow (KSplitNcNcEndpoint.mleRows input) boundary := by
  intro row member column mentioned
  rcases List.mem_flatMap.mp member with ⟨source, _, inRows⟩
  exact boolean_rows_below
    (KSplitNcNcEndpoint.sourceTable input source)
    (KSplitNcNcEndpoint.laneCoordinates input)
    (nc_table_below input messageBelow source)
    (nc_coordinates_below input pointLaneBelow source)
    (nc_mle_end input boundary allocationEnd source)
    row inRows column mentioned

private theorem nc_mleOutput_below
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (boundary : Nat)
    (pointLaneBelow :
      ∀ coordinate,
        CarriedBelow (input.pointLane coordinate) input.frameBase)
    (messageBelow :
      ∀ source lane,
        CarriedBelow (input.messageYZcol source lane) input.frameBase)
    (allocationEnd :
      input.frameBase + KSplitNcNcEndpoint.allocationWidth input ≤ boundary)
    (source : Fin shape.sourceCount) :
    CarriedBelow (KSplitNcNcEndpoint.mleOutput input source) boundary :=
  boolean_output_below
    (KSplitNcNcEndpoint.sourceTable input source)
    (KSplitNcNcEndpoint.laneCoordinates input)
    (nc_table_below input messageBelow source)
    (nc_coordinates_below input pointLaneBelow source)
    (nc_mle_end input boundary allocationEnd source)

private theorem nc_norm_end
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcNcEndpoint.allocationWidth input ≤ boundary)
    (source : Fin shape.sourceCount) :
    (KSplitNcNcEndpoint.normInput input source).frameBase + 6 ≤ boundary := by
  change input.frameBase +
        KSplitNcNcEndpoint.rowsPerMle domain * shape.sourceCount +
        6 * source.val + 6 ≤ boundary
  have sourceBound : source.val + 1 ≤ shape.sourceCount :=
    Nat.succ_le_iff.mpr source.isLt
  have scaled := Nat.mul_le_mul_left 6 sourceBound
  unfold KSplitNcNcEndpoint.allocationWidth at allocationEnd
  simp only [Nat.mul_succ] at scaled
  omega

private theorem nc_normRows_below
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (boundary : Nat) (positive : 0 < boundary)
    (pointLaneBelow :
      ∀ coordinate,
        CarriedBelow (input.pointLane coordinate) input.frameBase)
    (messageBelow :
      ∀ source lane,
        CarriedBelow (input.messageYZcol source lane) input.frameBase)
    (allocationEnd :
      input.frameBase + KSplitNcNcEndpoint.allocationWidth input ≤ boundary) :
    RowsBelow (KSplitNcNcEndpoint.normRows input) boundary := by
  intro row member column mentioned
  rcases List.mem_flatMap.mp member with ⟨source, _, inRows⟩
  exact strictNorm_rows_below
    (KSplitNcNcEndpoint.normInput input source)
    boundary positive
    (nc_mleOutput_below input boundary pointLaneBelow messageBelow
      allocationEnd source)
    (nc_norm_end input boundary allocationEnd source)
    row inRows column mentioned

private theorem nc_normOutput_below
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcNcEndpoint.allocationWidth input ≤ boundary)
    (source : Fin shape.sourceCount) :
    CarriedBelow
      (KStrictNorm.output (KSplitNcNcEndpoint.normInput input source))
      boundary :=
  strictNorm_output_below
    (KSplitNcNcEndpoint.normInput input source)
    boundary (nc_norm_end input boundary allocationEnd source)

private theorem nc_normOutputs_below
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcNcEndpoint.allocationWidth input ≤ boundary) :
    ∀ output ∈ KSplitNcNcEndpoint.normOutputs input,
      CarriedBelow output boundary := by
  intro output member
  rcases List.mem_map.mp member with ⟨source, _, rfl⟩
  exact nc_normOutput_below input boundary allocationEnd source

private theorem nc_mixed_end
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcNcEndpoint.allocationWidth input ≤ boundary) :
    KSplitNcNcEndpoint.mixedBase input +
        3 * ((KSplitNcNcEndpoint.normOutputs input).length - 1) ≤
      boundary := by
  rw [KSplitNcNcEndpoint.normOutputs_length]
  unfold KSplitNcNcEndpoint.mixedBase KSplitNcNcEndpoint.normBase
  unfold KSplitNcNcEndpoint.allocationWidth at allocationEnd
  omega

private theorem nc_product_end
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcNcEndpoint.allocationWidth input ≤ boundary) :
    KSplitNcNcEndpoint.productBase input + 6 ≤ boundary := by
  unfold KSplitNcNcEndpoint.allocationWidth at allocationEnd
  unfold KSplitNcNcEndpoint.productBase
    KSplitNcNcEndpoint.equalityBase
    KSplitNcNcEndpoint.mixedBase
    KSplitNcNcEndpoint.normBase
    KSplitNcNcEndpoint.pointEqualityRows
  unfold KSplitNcNcEndpoint.pointEqualityRows at allocationEnd
  omega

private theorem nc_block_end
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) :
    (KSplitNcNcEndpoint.blockEqualityInput input).frameBase +
        KSplitNcNcEndpoint.pointEqualityRows domain.blockVariables ≤
      KSplitNcNcEndpoint.productBase input := by
  change KSplitNcNcEndpoint.equalityBase input +
      KSplitNcNcEndpoint.pointEqualityRows domain.blockVariables ≤
    KSplitNcNcEndpoint.productBase input
  unfold KSplitNcNcEndpoint.productBase
  omega

private theorem nc_lane_end
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) :
    (KSplitNcNcEndpoint.laneEqualityInput input).frameBase +
        KSplitNcNcEndpoint.pointEqualityRows domain.laneVariables ≤
      KSplitNcNcEndpoint.productBase input := by
  change KSplitNcNcEndpoint.equalityBase input +
        KSplitNcNcEndpoint.pointEqualityRows domain.blockVariables +
        KSplitNcNcEndpoint.pointEqualityRows domain.laneVariables ≤
    KSplitNcNcEndpoint.productBase input
  unfold KSplitNcNcEndpoint.productBase
  omega

private theorem nc_point_end
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (boundary : Nat)
    (allocationEnd :
      input.frameBase + KSplitNcNcEndpoint.allocationWidth input ≤ boundary)
    {variables : Nat}
    (pointInput : KPointEquality.Input variables)
    (baseMatch :
      pointInput.frameBase + KSplitNcNcEndpoint.pointEqualityRows variables ≤
        KSplitNcNcEndpoint.productBase input) :
    pointInput.frameBase + 3 * variables + 3 * (variables - 1) ≤
      boundary := by
  unfold KSplitNcNcEndpoint.pointEqualityRows at baseMatch
  have productEnd := nc_product_end input boundary allocationEnd
  omega

theorem nc_rows_below
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (boundary : Nat) (positive : 0 < boundary)
    (gammaBelow : CarriedBelow input.gamma input.frameBase)
    (betaBlockBelow :
      ∀ coordinate,
        CarriedBelow (input.betaBlock coordinate) input.frameBase)
    (betaABelow :
      ∀ coordinate, CarriedBelow (input.betaA coordinate) input.frameBase)
    (pointBlockBelow :
      ∀ coordinate,
        CarriedBelow (input.pointBlock coordinate) input.frameBase)
    (pointLaneBelow :
      ∀ coordinate,
        CarriedBelow (input.pointLane coordinate) input.frameBase)
    (messageBelow :
      ∀ source lane,
        CarriedBelow (input.messageYZcol source lane) input.frameBase)
    (initialBelow : CarriedBelow input.initial boundary)
    (terminalBelow : CarriedBelow input.terminal boundary)
    (allocationEnd :
      input.frameBase + KSplitNcNcEndpoint.allocationWidth input ≤ boundary) :
    RowsBelow (KSplitNcNcEndpoint.rows input) boundary := by
  have gammaBoundary :=
    carried_mono gammaBelow
      (Nat.le_trans
        (Nat.le_add_right input.frameBase
          (KSplitNcNcEndpoint.allocationWidth input))
        allocationEnd)
  have mleRowsBoundary :=
    nc_mleRows_below input boundary pointLaneBelow messageBelow allocationEnd
  have normRowsBoundary :=
    nc_normRows_below input boundary positive pointLaneBelow messageBelow
      allocationEnd
  have normOutputsBoundary :=
    nc_normOutputs_below input boundary allocationEnd
  have mixedRowsBoundary :
      RowsBelow (KSplitNcNcEndpoint.mixedRows input) boundary :=
    horner_rows_below input.gamma (KSplitNcNcEndpoint.normOutputs input)
      (KSplitNcNcEndpoint.mixedBase input) 0 boundary
      gammaBoundary normOutputsBoundary
      (by simpa using nc_mixed_end input boundary allocationEnd)
  have mixedOutputBoundary :
      CarriedBelow (KSplitNcNcEndpoint.mixedOutput input) boundary :=
    horner_output_below input.gamma (KSplitNcNcEndpoint.normOutputs input)
      (KSplitNcNcEndpoint.mixedBase input) 0 boundary
      normOutputsBoundary
      (by simpa using nc_mixed_end input boundary allocationEnd)
  have blockRowsBoundary :
      RowsBelow
        (KPointEquality.rows
          (KSplitNcNcEndpoint.blockEqualityInput input))
        boundary :=
    pointEquality_rows_below _ positive
      (fun coordinate =>
        carried_mono (pointBlockBelow coordinate)
          (Nat.le_trans
            (Nat.le_add_right input.frameBase
              (KSplitNcNcEndpoint.allocationWidth input))
            allocationEnd))
      (fun coordinate =>
        carried_mono (betaBlockBelow coordinate)
          (Nat.le_trans
            (Nat.le_add_right input.frameBase
              (KSplitNcNcEndpoint.allocationWidth input))
            allocationEnd))
      (nc_point_end input boundary allocationEnd
        (KSplitNcNcEndpoint.blockEqualityInput input)
        (nc_block_end input))
  have laneRowsBoundary :
      RowsBelow
        (KPointEquality.rows
          (KSplitNcNcEndpoint.laneEqualityInput input))
        boundary :=
    pointEquality_rows_below _ positive
      (fun coordinate =>
        carried_mono (pointLaneBelow coordinate)
          (Nat.le_trans
            (Nat.le_add_right input.frameBase
              (KSplitNcNcEndpoint.allocationWidth input))
            allocationEnd))
      (fun coordinate =>
        carried_mono (betaABelow coordinate)
          (Nat.le_trans
            (Nat.le_add_right input.frameBase
              (KSplitNcNcEndpoint.allocationWidth input))
            allocationEnd))
      (nc_point_end input boundary allocationEnd
        (KSplitNcNcEndpoint.laneEqualityInput input)
        (nc_lane_end input))
  have blockOutputBoundary :=
    pointEquality_output_below
      (KSplitNcNcEndpoint.blockEqualityInput input) positive
      (fun coordinate =>
        carried_mono (betaBlockBelow coordinate)
          (Nat.le_trans
            (Nat.le_add_right input.frameBase
              (KSplitNcNcEndpoint.allocationWidth input))
            allocationEnd))
      (nc_point_end input boundary allocationEnd
        (KSplitNcNcEndpoint.blockEqualityInput input)
        (nc_block_end input))
  have laneOutputBoundary :=
    pointEquality_output_below
      (KSplitNcNcEndpoint.laneEqualityInput input) positive
      (fun coordinate =>
        carried_mono (betaABelow coordinate)
          (Nat.le_trans
            (Nat.le_add_right input.frameBase
              (KSplitNcNcEndpoint.allocationWidth input))
            allocationEnd))
      (nc_point_end input boundary allocationEnd
        (KSplitNcNcEndpoint.laneEqualityInput input)
        (nc_lane_end input))
  have productEnd := nc_product_end input boundary allocationEnd
  have selectorRowsBoundary :=
    mul_rows_below _ _ (KSplitNcNcEndpoint.productBase input) 0 boundary
      blockOutputBoundary laneOutputBoundary (by omega)
  have selectorBoundary :=
    frame_output_below (KSplitNcNcEndpoint.productBase input) 0 boundary
      (by omega)
  have terminalExpressionRowsBoundary :=
    mul_rows_below _ _ (KSplitNcNcEndpoint.productBase input) 1 boundary
      selectorBoundary mixedOutputBoundary (by omega)
  have terminalExpressionBoundary :=
    frame_output_below (KSplitNcNcEndpoint.productBase input) 1 boundary
      (by omega)
  have initialRowsBoundary :=
    equality_rows_below KLinear.zeroCarried input.initial boundary positive
      (zero_below boundary) initialBelow
  have finalRowsBoundary :=
    equality_rows_below (KSplitNcNcEndpoint.terminalExpression input)
      input.terminal boundary positive terminalExpressionBoundary terminalBelow
  intro row member column mentioned
  rcases List.mem_flatten.mp member with
    ⟨group, groupMember, rowMember⟩
  simp only [KSplitNcNcEndpoint.rowGroups, List.mem_cons,
    List.not_mem_nil, or_false] at groupMember
  rcases groupMember with
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact initialRowsBoundary row rowMember column mentioned
  · exact mleRowsBoundary row rowMember column mentioned
  · exact normRowsBoundary row rowMember column mentioned
  · exact mixedRowsBoundary row rowMember column mentioned
  · exact blockRowsBoundary row rowMember column mentioned
  · exact laneRowsBoundary row rowMember column mentioned
  · exact selectorRowsBoundary row rowMember column mentioned
  · exact terminalExpressionRowsBoundary row rowMember column mentioned
  · exact finalRowsBoundary row rowMember column mentioned

end Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointSupport
