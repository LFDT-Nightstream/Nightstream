import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionTerminalBridge.CarrierAlgebra
/-!
Concrete terminal bridge for the fixed production combined-NC artifact.
Owns: the explicit value contract for the generated terminal input columns
and the kernel algebra that interprets the materialized terminal program as
the production claims-level terminal.  The output table is read in its
physical, little-endian numeric lane order; the bridge proves rather than
assumes the permutation to the independently typed Boolean-table evaluator.
Does not own: construction of the assignment, transcript replay, parent or
raw-child authority, commitment binding, `y_ring`, Poseidon2, Ajtai, costs,
or row removal.  In particular, neither `terminalRhsColumns` nor
`finalSumColumns` occurs in `TerminalColumnBindings`.
Emits constraints: none.
Assurance tier: model-level until the focused Lean target is validated and
the production encoder is proved to supply every input binding below.
-/
/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.terminal.kernels` | Prove the base terminal polynomial and selector-evaluation kernels. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionTerminalBridge
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open Nightstream.Implementation.R1CS.PiCcsNc
open Nightstream.Implementation.R1CS.PiCcsNc.Authority
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc
private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws
private abbrev KColumns := ProjectionProgram.KColumns
universe uState
/-- Interpret an exact materialized K-column pair in the independent
production carrier. -/
def sourceValue (assignment : Nat -> Nat) (columns : KColumns) :
    Nightstream.SuperNeo.Concrete.K :=
  ProductionMessageAcceptance.toConcreteK (columns.value assignment)
/-- The physical output table, retaining all sixty-four generated lane
columns for each of the fifteen sources. -/
def materializedOutputRows (assignment : Nat -> Nat) : List (List K) :=
  TerminalProgram.outputYZcolColumns.map fun output =>
    output.map (sourceValue assignment)
private theorem map_materializedOutputRows
    (assignment : Nat -> Nat) (value : List K -> K) :
    (materializedOutputRows assignment).map value =
      TerminalProgram.outputYZcolColumns.map fun output =>
        value (output.map (sourceValue assignment)) := by
  unfold materializedOutputRows
  rw [List.map_map]
  rfl
/-- The independently typed output message in the same natural numeric lane
order used by the production columns. Padding values are computed by
`Terminal.paddedYZcol`, not supplied by the prover. -/
def productionOutputRows
    {shape : SemanticShape}
    (message : Claims shape) : List (List K) :=
  (canonicalFinIndices shape.sourceCount).map fun source =>
    (canonicalFinIndices PiCcsDomains.production.nc.laneCount).map fun lane =>
      Terminal.paddedYZcol (domain := PiCcsDomains.production.nc)
        message source lane
private theorem map_productionOutputRows
    {shape : SemanticShape} (message : Claims shape) (value : List K -> K) :
    (productionOutputRows message).map value =
      (canonicalFinIndices shape.sourceCount).map fun source =>
        value ((canonicalFinIndices PiCcsDomains.production.nc.laneCount).map
          fun lane => Terminal.paddedYZcol
            (domain := PiCcsDomains.production.nc) message source lane) := by
  unfold productionOutputRows
  rw [List.map_map]
  rfl
/-- Exact input-column contract for the recursive production terminal.  All
fields read generated boundary columns.  No terminal result, acceptance
predicate, digest, or semantic projection is a field. -/
structure TerminalColumnBindings
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (pending : ProductionDelayedBlockLane)
    (assignment : Nat -> Nat) : Prop where
  pendingEq : context.pending = some pending
  gamma : sourceValue assignment TerminalProgram.gammaColumns =
    context.ncCoins.gamma
  betaLane : TerminalProgram.betaLaneColumns.map (sourceValue assignment) =
    context.ncCoins.betaA.coordinates
  betaBlock : TerminalProgram.betaBlockColumns.map (sourceValue assignment) =
    context.ncCoins.betaBlock.coordinates
  producerBeta : sourceValue assignment TerminalProgram.producerBetaColumns =
    context.producerBeta
  batchWeight : sourceValue assignment TerminalProgram.batchWeightColumns =
    context.batchWeight
  pendingOldBlock :
    TerminalProgram.pendingOldBlockColumns.map (sourceValue assignment) =
      pending.oldBlock.coordinates
  outputRows : materializedOutputRows assignment =
    productionOutputRows certificate.piCcs.output
  blockPoint : TerminalProgram.blockPointColumns.map (sourceValue assignment) =
    (ProductionPiCcs.ncPoint context certificate).block.coordinates
  lanePoint : TerminalProgram.lanePointColumns.map (sourceValue assignment) =
    (ProductionPiCcs.ncPoint context certificate).lane.coordinates
/-! ## Carrier and finite-fold transport -/
@[simp] private theorem sourceValue_zero (assignment : Nat -> Nat) :
    ProductionMessageAcceptance.toConcreteK ProjectionProgram.K.zero =
      Nightstream.SuperNeo.Concrete.K.zero :=
  ProductionMessageAcceptance.toConcreteK_zero
@[simp] private theorem sourceValue_one :
    ProductionMessageAcceptance.toConcreteK ProjectionProgram.K.one =
      Nightstream.SuperNeo.Concrete.K.one :=
  ProductionMessageAcceptance.toConcreteK_one
private theorem map_sumK (values : List ProjectionProgram.K) :
    ProductionMessageAcceptance.toConcreteK (TerminalProgram.sumK values) =
      BooleanTable.finiteSum ops
        (values.map ProductionMessageAcceptance.toConcreteK) := by
  induction values with
  | nil => rfl
  | cons value values inductionHypothesis =>
      simp only [TerminalProgram.sumK, List.foldr_cons, List.map_cons,
        BooleanTable.finiteSum]
      rw [ProductionMessageAcceptance.toConcreteK_add]
      exact congrArg
        (K.add (ProductionMessageAcceptance.toConcreteK value))
        inductionHypothesis
theorem map_dotValue
    (left right : List KColumns) (assignment : Nat -> Nat) :
    ProductionMessageAcceptance.toConcreteK
        (TerminalProgram.dotValue left right assignment) =
      BooleanTable.finiteSum ops
        ((left.zip right).map fun pair =>
          K.mul (sourceValue assignment pair.1)
            (sourceValue assignment pair.2)) := by
  unfold TerminalProgram.dotValue
  rw [map_sumK]
  rw [List.map_map]
  apply congrArg (BooleanTable.finiteSum ops)
  apply List.map_congr_left
  intro pair _
  exact ProductionMessageAcceptance.toConcreteK_mul _ _
private theorem sourceConstantOne
    (assignment : Nat -> Nat) (constantOne : assignment 0 = 1) :
    ProductionMessageAcceptance.toConcreteField
        (ProjectionProgram.baseAt assignment 0) = (1 : F) := by
  unfold ProjectionProgram.baseAt
  rw [constantOne, ProjectionProgram.residue_one, mappedFieldOne]
theorem source_oneTerms
    (assignment : Nat -> Nat) (constantOne : assignment 0 = 1) :
    ProductionMessageAcceptance.toConcreteK
        (TerminalProgram.oneTerms.value assignment) = K.one := by
  simp only [TerminalProgram.oneTerms, ProjectionProgram.KTerms.value,
    ProductionMessageAcceptance.toConcreteK, K.one, K.mk.injEq]
  constructor <;> apply Fin.ext <;>
    simp [Nightstream.Implementation.R1CS.lcEval, constantOne,
      ProductionMessageAcceptance.toConcreteField, ProjectionProgram.residue,
      Nightstream.Implementation.R1CS.goldilocksP, goldilocksModulus] <;> rfl
theorem source_baseTwoTerms
    (assignment : Nat -> Nat) (constantOne : assignment 0 = 1) :
    ProductionMessageAcceptance.toConcreteK
        (TerminalProgram.baseTwoTerms.value assignment) = K.embed 2 := by
  simp only [TerminalProgram.baseTwoTerms, ProjectionProgram.KTerms.value,
    ProductionMessageAcceptance.toConcreteK, K.embed, K.mk.injEq]
  constructor <;> apply Fin.ext <;>
    simp [Nightstream.Implementation.R1CS.lcEval, constantOne,
      ProductionMessageAcceptance.toConcreteField, ProjectionProgram.residue,
      Nightstream.Implementation.R1CS.goldilocksP, goldilocksModulus] <;> rfl
private theorem sourceTerms_subtract
    (assignment : Nat -> Nat) (left right : KColumns) :
    ProductionMessageAcceptance.toConcreteK
        ((TerminalProgram.subtractTerms left right).value assignment) =
      K.sub (sourceValue assignment left) (sourceValue assignment right) := by
  simp only [TerminalProgram.subtractTerms, ProjectionProgram.KTerms.value,
    sourceValue, ProjectionProgram.KColumns.value,
    ProductionMessageAcceptance.toConcreteK, K.sub, K.mk.injEq]
  constructor <;> rw [mappedLinear2, mappedNegOne_mul] <;>
    simp only [ProjectionProgram.residue_one, mappedFieldOne, Fin.one_mul,
      Fin.sub_eq_add_neg]
private theorem sourceTerms_oneMinus
    (assignment : Nat -> Nat) (constantOne : assignment 0 = 1)
    (value : KColumns) :
    ProductionMessageAcceptance.toConcreteK
        ((TerminalProgram.oneMinusTerms value).value assignment) =
      K.sub K.one (sourceValue assignment value) := by
  simp only [TerminalProgram.oneMinusTerms, ProjectionProgram.KTerms.value,
    sourceValue, ProjectionProgram.KColumns.value,
    ProductionMessageAcceptance.toConcreteK, K.sub, K.one, K.mk.injEq]
  constructor
  · rw [mappedLinear2, sourceConstantOne assignment constantOne,
      mappedNegOne_mul]
    simp only [ProjectionProgram.residue_one, mappedFieldOne, Fin.one_mul,
      Fin.sub_eq_add_neg]
  · rw [mappedLinear1, mappedNegOne_mul]
    simp only [Fin.zero_add, Fin.sub_eq_add_neg]
private theorem sourceTerms_eqFactor
    (assignment : Nat -> Nat) (constantOne : assignment 0 = 1)
    (product left right : KColumns)
    (productEq : sourceValue assignment product =
      K.mul (sourceValue assignment left) (sourceValue assignment right)) :
    ProductionMessageAcceptance.toConcreteK
        ((TerminalProgram.eqFactorTerms product left right).value assignment) =
      SumCheckTruthPath.equalityFactor ops
        (sourceValue assignment left) (sourceValue assignment right) := by
  have constantValue := sourceConstantOne assignment constantOne
  have decoded : ProductionMessageAcceptance.toConcreteK
        ((TerminalProgram.eqFactorTerms product left right).value assignment) =
      K.sub
        (K.sub
          (K.add
            (K.add (sourceValue assignment product)
              (sourceValue assignment product))
            K.one)
          (sourceValue assignment left))
        (sourceValue assignment right) := by
    simp only [TerminalProgram.eqFactorTerms, ProjectionProgram.KTerms.value,
      sourceValue, ProjectionProgram.KColumns.value,
      ProductionMessageAcceptance.toConcreteK, K.sub, K.add, K.one,
      K.mk.injEq]
    constructor
    · rw [mappedLinear4, mappedTwo_mul, constantValue,
        mappedNegOne_mul, mappedNegOne_mul]
      simp only [ProjectionProgram.residue_one, Fin.one_mul, Fin.add_zero,
        Fin.zero_add, Fin.sub_eq_add_neg]
      ac_rfl
    · rw [mappedLinear3, mappedTwo_mul, mappedNegOne_mul,
        mappedNegOne_mul]
      simp only [Fin.add_zero, Fin.zero_add, Fin.sub_eq_add_neg]
  rw [decoded, productEq]
  exact affineProduct_eq_equalityFactor _ _
theorem sourceTerms_selectorFactor
    (assignment : Nat -> Nat) (constantOne : assignment 0 = 1)
    (selected coordinate : KColumns) :
    ProductionMessageAcceptance.toConcreteK
        ((TerminalProgram.selectorFactorTerms selected coordinate).value
          assignment) =
      K.add (K.sub K.one (sourceValue assignment coordinate))
        (sourceValue assignment selected) := by
  have constantValue := sourceConstantOne assignment constantOne
  simp only [TerminalProgram.selectorFactorTerms,
    ProjectionProgram.KTerms.value, sourceValue,
    ProjectionProgram.KColumns.value, ProductionMessageAcceptance.toConcreteK,
    K.add, K.sub, K.one, K.mk.injEq]
  constructor
  · rw [mappedLinear3, constantValue, mappedNegOne_mul]
    simp only [ProjectionProgram.residue_one, mappedFieldOne, Fin.one_mul,
      Fin.zero_add, Fin.add_zero, Fin.sub_eq_add_neg]
  · rw [mappedLinear2, mappedNegOne_mul]
    simp only [ProjectionProgram.residue_one, mappedFieldOne, Fin.one_mul,
      Fin.zero_add, Fin.add_zero, Fin.sub_eq_add_neg]
/-! ## Numeric chi and typed Boolean-table evaluation -/
/-- Natural numeric lane weights used by the physical output-column order. -/
def numericChi (coordinates : List K) : List K :=
  (List.range (2 ^ coordinates.length)).map fun lane =>
    MixedPolynomial.chi coordinates lane
private theorem numericChi_nil : numericChi [] = [K.one] := by
  rfl
private theorem testBit_add_twoPow_low
    {width mask bit : Nat}
    (maskLt : mask < 2 ^ width) (bitLt : bit < width) :
    Nat.testBit (mask + 2 ^ width) bit = Nat.testBit mask bit := by
  have modulo : (mask + 2 ^ width) % 2 ^ width = mask := by
    calc
      (mask + 2 ^ width) % 2 ^ width = mask % 2 ^ width := by
        simpa using Nat.add_mul_mod_self_left mask (2 ^ width) 1
      _ = mask := Nat.mod_eq_of_lt maskLt
  have projected := Nat.testBit_mod_two_pow
    (mask + 2 ^ width) width bit
  rw [modulo] at projected
  simp [bitLt] at projected
  exact projected.symm
private theorem testBit_add_twoPow_self
    {width mask : Nat} (maskLt : mask < 2 ^ width) :
    Nat.testBit (mask + 2 ^ width) width = true := by
  unfold Nat.testBit
  rw [Nat.shiftRight_eq_div_pow]
  have powerPositive : 0 < 2 ^ width := Nat.two_pow_pos width
  have quotient : (mask + 2 ^ width) / 2 ^ width = 1 := by
    calc
      (mask + 2 ^ width) / 2 ^ width = mask / 2 ^ width + 1 := by
        simpa using Nat.add_mul_div_right mask 1 powerPositive
      _ = 1 := by rw [Nat.div_eq_of_lt maskLt]
  rw [quotient]
  decide
private theorem testBit_eq_false_of_lt_twoPow
    {width mask : Nat} (maskLt : mask < 2 ^ width) :
    Nat.testBit mask width = false := by
  unfold Nat.testBit
  rw [Nat.shiftRight_eq_div_pow, Nat.div_eq_of_lt maskLt]
  decide
/-- One physical chi expansion appends the next little-endian coordinate as
the next numeric bit: the old table is the low half, followed by the high
half. -/
theorem numericChi_append
    (coordinates : List K) (coordinate : K) :
    numericChi (coordinates ++ [coordinate]) =
      (numericChi coordinates).map
          (fun weight => K.mul weight (K.sub K.one coordinate)) ++
        (numericChi coordinates).map
          (fun weight => K.mul weight coordinate) := by
  unfold numericChi
  rw [show 2 ^ (coordinates ++ [coordinate]).length =
      2 ^ coordinates.length + 2 ^ coordinates.length by
    simp [Nat.pow_succ, Nat.mul_two]]
  rw [List.range_add, List.map_append]
  simp only [List.map_map, Function.comp_apply]
  congr 1
  · apply List.map_congr_left
    intro index member
    simp only [Function.comp_apply]
    have indexLt := List.mem_range.mp member
    unfold MixedPolynomial.chi
    rw [List.length_append]
    simp only [List.length_singleton, Nat.add_comm,
      Nightstream.Implementation.R1CS.PiCcsNc.productRange]
    have lowerProduct :
        Nightstream.Implementation.R1CS.PiCcsNc.productRange
            coordinates.length
            (MixedPolynomial.chiFactor (coordinates ++ [coordinate]) index) =
          Nightstream.Implementation.R1CS.PiCcsNc.productRange
            coordinates.length
            (MixedPolynomial.chiFactor coordinates index) := by
      apply Nightstream.Implementation.R1CS.PiCcsNc.productRange_congr
      intro bit bitLt
      unfold MixedPolynomial.chiFactor
      simp only [List.getD_eq_getElem?_getD]
      rw [List.getElem?_append_left bitLt]
    rw [lowerProduct]
    simp [MixedPolynomial.chiFactor,
      testBit_eq_false_of_lt_twoPow indexLt,
      List.getD_eq_getElem?_getD]
  · apply List.map_congr_left
    intro index member
    simp only [Function.comp_apply]
    have indexLt := List.mem_range.mp member
    unfold MixedPolynomial.chi
    rw [List.length_append]
    simp only [List.length_singleton, Nat.add_comm,
      Nightstream.Implementation.R1CS.PiCcsNc.productRange]
    have lowerProduct :
        Nightstream.Implementation.R1CS.PiCcsNc.productRange
            coordinates.length
            (MixedPolynomial.chiFactor (coordinates ++ [coordinate])
              (index + 2 ^ coordinates.length)) =
          Nightstream.Implementation.R1CS.PiCcsNc.productRange
            coordinates.length
            (MixedPolynomial.chiFactor coordinates index) := by
      apply Nightstream.Implementation.R1CS.PiCcsNc.productRange_congr
      intro bit bitLt
      unfold MixedPolynomial.chiFactor
      have bitEq : Nat.testBit (index + 2 ^ coordinates.length) bit =
          Nat.testBit index bit := by
        exact testBit_add_twoPow_low indexLt bitLt
      rw [bitEq]
      simp only [List.getD_eq_getElem?_getD]
      rw [List.getElem?_append_left bitLt]
    rw [lowerProduct]
    have bitTrue : Nat.testBit (index + 2 ^ coordinates.length)
        coordinates.length = true := by
      exact testBit_add_twoPow_self indexLt
    simp [MixedPolynomial.chiFactor, bitTrue,
      List.getD_eq_getElem?_getD]
def expandChiValues (weights : List K) (coordinate : K) : List K :=
  weights.map (fun weight => K.mul weight (K.sub K.one coordinate)) ++
    weights.map (fun weight => K.mul weight coordinate)
private theorem foldl_expandChiValues_eq_numericChi
    (coordinates : List K) :
    coordinates.foldl expandChiValues [K.one] = numericChi coordinates := by
  have reversed : forall values : List K,
      values.reverse.foldl expandChiValues [K.one] =
        numericChi values.reverse := by
    intro values
    induction values with
    | nil => rfl
    | cons coordinate values inductionHypothesis =>
        rw [List.reverse_cons, List.foldl_append, numericChi_append,
          inductionHypothesis]
        rfl
  simpa using reversed coordinates.reverse
def chiColumnsFrom : Nat -> List KColumns -> List KColumns -> List KColumns
  | _, current, [] => current
  | base, current, coordinate :: coordinates =>
      let layer : TerminalProgram.ChiLayer :=
        { base := base, input := current, coordinate := coordinate }
      chiColumnsFrom layer.next layer.output coordinates
private theorem chiLayersFrom_final
    (base : Nat) (current coordinates : List KColumns) :
    (match (TerminalProgram.chiLayersFrom base current coordinates).reverse with
      | [] => current
      | layer :: _ => layer.output) =
      chiColumnsFrom base current coordinates := by
  induction coordinates generalizing base current with
  | nil => rfl
  | cons coordinate coordinates inductionHypothesis =>
      let layer : TerminalProgram.ChiLayer :=
        { base := base, input := current, coordinate := coordinate }
      have tailResult := inductionHypothesis (base := layer.next)
        (current := layer.output)
      simp only [TerminalProgram.chiLayersFrom, chiColumnsFrom,
        List.reverse_cons]
      generalize (TerminalProgram.chiLayersFrom layer.next layer.output
        coordinates).reverse = reversed at tailResult ⊢
      cases reversed <;> simpa using tailResult
private theorem mappedChiExpected
    (assignment : Nat -> Nat) (constantOne : assignment 0 = 1)
    (layer : TerminalProgram.ChiLayer) :
    (layer.expected assignment).map
        ProductionMessageAcceptance.toConcreteK =
      expandChiValues
        (layer.input.map (sourceValue assignment))
        (sourceValue assignment layer.coordinate) := by
  unfold TerminalProgram.ChiLayer.expected expandChiValues
  rw [List.map_append]
  congr 1
  · simp only [List.map_map, Function.comp_apply]
    apply List.map_congr_left
    intro value _
    unfold sourceValue
    change ProductionMessageAcceptance.toConcreteK
        (ProjectionProgram.K.mul (value.value assignment)
          ((TerminalProgram.oneMinusTerms layer.coordinate).value assignment)) =
      K.mul (ProductionMessageAcceptance.toConcreteK (value.value assignment))
        (K.sub K.one (ProductionMessageAcceptance.toConcreteK
          (layer.coordinate.value assignment)))
    rw [ProductionMessageAcceptance.toConcreteK_mul]
    exact congrArg
      (K.mul (ProductionMessageAcceptance.toConcreteK (value.value assignment)))
      (sourceTerms_oneMinus assignment constantOne layer.coordinate)
  · simp only [List.map_map, Function.comp_apply]
    apply List.map_congr_left
    intro value _
    unfold sourceValue
    change ProductionMessageAcceptance.toConcreteK
        (ProjectionProgram.K.mul (value.value assignment)
          (layer.coordinate.value assignment)) =
      K.mul (ProductionMessageAcceptance.toConcreteK (value.value assignment))
        (ProductionMessageAcceptance.toConcreteK
          (layer.coordinate.value assignment))
    rw [ProductionMessageAcceptance.toConcreteK_mul]
private theorem chiColumnsFrom_values
    (assignment : Nat -> Nat) (constantOne : assignment 0 = 1)
    (base : Nat) (current coordinates : List KColumns)
    (computed : forall layer,
      layer ∈ TerminalProgram.chiLayersFrom base current coordinates ->
        layer.output.map (fun output => output.value assignment) =
          layer.expected assignment) :
    (chiColumnsFrom base current coordinates).map (sourceValue assignment) =
      (coordinates.map (sourceValue assignment)).foldl expandChiValues
        (current.map (sourceValue assignment)) := by
  induction coordinates generalizing base current with
  | nil => rfl
  | cons coordinate coordinates inductionHypothesis =>
      let layer : TerminalProgram.ChiLayer :=
        { base := base, input := current, coordinate := coordinate }
      have headComputed :
          layer.output.map (fun output => output.value assignment) =
            layer.expected assignment :=
        computed layer (by simp [TerminalProgram.chiLayersFrom, layer])
      have tailComputed : forall candidate,
          candidate ∈ TerminalProgram.chiLayersFrom layer.next layer.output
              coordinates ->
            candidate.output.map (fun output => output.value assignment) =
              candidate.expected assignment := by
        intro candidate member
        exact computed candidate (by
          simp [TerminalProgram.chiLayersFrom, layer, member])
      have mappedHead :
          layer.output.map (sourceValue assignment) =
            expandChiValues (current.map (sourceValue assignment))
              (sourceValue assignment coordinate) := by
        calc
          layer.output.map (sourceValue assignment) =
              (layer.output.map fun output => output.value assignment).map
                ProductionMessageAcceptance.toConcreteK := by
            simp [sourceValue, List.map_map, Function.comp_def]
          _ = (layer.expected assignment).map
                ProductionMessageAcceptance.toConcreteK := by rw [headComputed]
          _ = _ := mappedChiExpected assignment constantOne layer
      simp only [chiColumnsFrom, List.map_cons, List.foldl]
      rw [inductionHypothesis (base := layer.next) (current := layer.output)
        tailComputed, mappedHead]
private theorem chiColumns_values_eq_numericChi
    (assignment : Nat -> Nat) (constantOne : assignment 0 = 1)
    (computed : TerminalProgram.Computed assignment) :
    TerminalProgram.chiColumns.map (sourceValue assignment) =
      numericChi
        (TerminalProgram.lanePointColumns.map (sourceValue assignment)) := by
  have initial :
      [TerminalProgram.chiInitial.output].map (sourceValue assignment) =
        [K.one] := by
    simp only [List.map_singleton]
    unfold sourceValue
    rw [computed.chiInitial]
    exact congrArg List.singleton (source_oneTerms assignment constantOne)
  have columnsEq : TerminalProgram.chiColumns =
      chiColumnsFrom TerminalProgram.chiInitial.next
        [TerminalProgram.chiInitial.output]
        TerminalProgram.lanePointColumns := by
    unfold TerminalProgram.chiColumns TerminalProgram.chiLayers
    simpa using chiLayersFrom_final TerminalProgram.chiInitial.next
      [TerminalProgram.chiInitial.output] TerminalProgram.lanePointColumns
  rw [columnsEq]
  rw [chiColumnsFrom_values assignment constantOne
    TerminalProgram.chiInitial.next [TerminalProgram.chiInitial.output]
    TerminalProgram.lanePointColumns computed.chiLayers]
  rw [initial, foldl_expandChiValues_eq_numericChi]
private theorem outputTracesFrom_values
    (base : Nat) (outputs : List (List KColumns)) (chi : List KColumns) :
    (TerminalProgram.outputTracesFrom base outputs chi).map
        TerminalProgram.OutputTrace.values = outputs := by
  induction outputs generalizing base with
  | nil => rfl
  | cons output outputs inductionHypothesis =>
      simp [TerminalProgram.outputTracesFrom, inductionHypothesis]
private theorem outputTracesFrom_chi
    (base : Nat) (outputs : List (List KColumns)) (chi : List KColumns) :
    forall trace,
      trace ∈ TerminalProgram.outputTracesFrom base outputs chi ->
        trace.chi = chi := by
  induction outputs generalizing base with
  | nil => simp [TerminalProgram.outputTracesFrom]
  | cons output outputs inductionHypothesis =>
      intro trace member
      simp only [TerminalProgram.outputTracesFrom, List.mem_cons] at member
      rcases member with rfl | member
      · rfl
      · exact inductionHypothesis _ _ member
private theorem map_zip_values
    {Left Right LeftValue RightValue Value : Type}
    (left : List Left) (right : List Right)
    (leftValue : Left -> LeftValue) (rightValue : Right -> RightValue)
    (combine : LeftValue -> RightValue -> Value) :
    (left.zip right).map (fun pair =>
        combine (leftValue pair.1) (rightValue pair.2)) =
      ((left.map leftValue).zip (right.map rightValue)).map (fun pair =>
        combine pair.1 pair.2) := by
  induction left generalizing right with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      cases right with
      | nil => rfl
      | cons rightHead rightTail =>
          simp only [List.zip_cons_cons, List.map_cons, List.cons.injEq]
          constructor
          · trivial
          · exact inductionHypothesis rightTail
private theorem outputTraceEvaluations
    (assignment : Nat -> Nat)
    (computed : TerminalProgram.Computed assignment) :
    TerminalProgram.outputTraces.map (fun trace =>
        sourceValue assignment trace.evaluation.output) =
      TerminalProgram.outputYZcolColumns.map fun row =>
        BooleanTable.finiteSum ops
          ((row.map (sourceValue assignment)).zip
              (TerminalProgram.chiColumns.map (sourceValue assignment)) |>.map
            fun pair => K.mul pair.1 pair.2) := by
  have values := outputTracesFrom_values TerminalProgram.chiNext
    TerminalProgram.outputYZcolColumns TerminalProgram.chiColumns
  have sameChi := outputTracesFrom_chi TerminalProgram.chiNext
    TerminalProgram.outputYZcolColumns TerminalProgram.chiColumns
  have generic : forall
      (traces : List TerminalProgram.OutputTrace)
      (outputs : List (List KColumns)),
      traces.map TerminalProgram.OutputTrace.values = outputs ->
      (forall trace, trace ∈ traces ->
        trace.chi = TerminalProgram.chiColumns) ->
      (forall trace, trace ∈ traces -> trace.Computed assignment) ->
      traces.map (fun trace =>
          sourceValue assignment trace.evaluation.output) =
        outputs.map fun row =>
          BooleanTable.finiteSum ops
            ((row.map (sourceValue assignment)).zip
                (TerminalProgram.chiColumns.map (sourceValue assignment)) |>.map
              fun pair => K.mul pair.1 pair.2) := by
    intro traces
    induction traces with
    | nil =>
        intro outputs values _sameChi _computed
        simpa using values.symm
    | cons trace traces inductionHypothesis =>
        intro outputs values sameChi tracesComputed
        cases outputs with
        | nil => simp at values
        | cons output outputs =>
            simp only [List.map_cons, List.cons.injEq] at values
            have traceComputed := tracesComputed trace (by simp)
            have traceChi := sameChi trace (by simp)
            have tailComputed : forall candidate, candidate ∈ traces ->
                candidate.Computed assignment := by
              intro candidate member
              exact tracesComputed candidate (by simp [member])
            have tailChi : forall candidate, candidate ∈ traces ->
                candidate.chi = TerminalProgram.chiColumns := by
              intro candidate member
              exact sameChi candidate (by simp [member])
            simp only [List.map_cons, List.cons.injEq]
            constructor
            · change ProductionMessageAcceptance.toConcreteK
                  (trace.evaluation.output.value assignment) = _
              rw [traceComputed.evaluation, map_dotValue, traceChi, values.1]
              exact congrArg (BooleanTable.finiteSum ops)
                (map_zip_values output TerminalProgram.chiColumns
                  (sourceValue assignment) (sourceValue assignment) K.mul)
            · exact inductionHypothesis outputs values.2 tailChi tailComputed
  exact generic TerminalProgram.outputTraces
    TerminalProgram.outputYZcolColumns values sameChi computed.outputs
theorem finiteSum_eq_of_perm
    {left right : List K} (permutation : left.Perm right) :
    BooleanTable.finiteSum ops left = BooleanTable.finiteSum ops right := by
  induction permutation with
  | nil => rfl
  | cons value permutation inductionHypothesis =>
      simp only [BooleanTable.finiteSum, inductionHypothesis]
  | swap left right values =>
      simp only [BooleanTable.finiteSum]
      calc
        ops.add right (ops.add left (BooleanTable.finiteSum ops values)) =
            ops.add (ops.add right left) (BooleanTable.finiteSum ops values) :=
          (laws.add_assoc _ _ _).symm
        _ = ops.add (ops.add left right) (BooleanTable.finiteSum ops values) := by
          exact congrArg
            (fun value => ops.add value (BooleanTable.finiteSum ops values))
            (laws.add_comm right left)
        _ = ops.add left (ops.add right (BooleanTable.finiteSum ops values)) :=
          laws.add_assoc _ _ _
  | trans _ _ leftHypothesis rightHypothesis =>
      exact leftHypothesis.trans rightHypothesis
private theorem sumMap_eq_of_perm
    {Index : Type} {left right : List Index}
    (permutation : left.Perm right)
    (value : Index -> Nightstream.SuperNeo.Concrete.K) :
    FiniteSumAlgebra.sumMap ops left value =
      FiniteSumAlgebra.sumMap ops right value := by
  unfold FiniteSumAlgebra.sumMap
  exact finiteSum_eq_of_perm (permutation.map value)
private theorem perm_of_nodup_mem_iff
    {Index : Type} [BEq Index] [LawfulBEq Index]
    {left right : List Index}
    (leftNodup : left.Nodup) (rightNodup : right.Nodup)
    (sameMembers : forall index, index ∈ left <-> index ∈ right) :
    left.Perm right := by
  rw [List.perm_iff_count]
  intro index
  rw [leftNodup.count, rightNodup.count]
  by_cases member : index ∈ left
  · have rightMember := (sameMembers index).mp member
    simp [member, rightMember]
  · have rightMember : index ∉ right := by
      exact fun present => member ((sameMembers index).mpr present)
    simp [member, rightMember]
theorem laneIndices_perm :
    ((BooleanVertex.all PiCcsDomains.production.nc.laneVariables).map
        BlockNcDomain.laneIndex).Perm
      (canonicalFinIndices PiCcsDomains.production.nc.laneCount) := by
  apply perm_of_nodup_mem_iff
  · apply (BooleanVertex.all_nodup
      PiCcsDomains.production.nc.laneVariables).map
      BlockNcDomain.laneIndex
    intro left right different equal
    apply different
    calc
      left = BlockNcDomain.laneVertex
          (BlockNcDomain.laneIndex left) :=
        (BlockNcDomain.laneVertex_laneIndex left).symm
      _ = BlockNcDomain.laneVertex
          (BlockNcDomain.laneIndex right) := by rw [equal]
      _ = right := BlockNcDomain.laneVertex_laneIndex right
  · exact canonicalFinIndices_nodup _
  · intro lane
    constructor
    · intro _
      exact List.mem_ofFn.mpr ⟨lane, rfl⟩
    · intro _
      exact List.mem_map.mpr
        ⟨BlockNcDomain.laneVertex lane, BooleanVertex.mem_all _,
          BlockNcDomain.laneIndex_laneVertex lane⟩
private theorem foldl_range_mul_eq_productRange
    (count : Nat) (term : Nat -> Nightstream.SuperNeo.Concrete.K) :
    (List.range count).foldl
        (fun accumulated index =>
          Nightstream.SuperNeo.Concrete.K.mul accumulated (term index))
        Nightstream.SuperNeo.Concrete.K.one =
      Nightstream.Implementation.R1CS.PiCcsNc.productRange count term := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.foldl_append, inductionHypothesis]
      rfl
private theorem testBitWeight_eq_chi
    {variables : Nat} (index : Fin (2 ^ variables))
    (point : CubePoint Nightstream.SuperNeo.Concrete.K variables) :
    NumericBooleanDomain.testBitWeight ops point index =
      MixedPolynomial.chi point.coordinates index.val := by
  unfold NumericBooleanDomain.testBitWeight MixedPolynomial.chi
    MixedPolynomial.chiFactor
  rw [point.dimension]
  simp only [ConcreteCarrier.derived_sub_eq_concrete_sub]
  let factor : Nat -> Nightstream.SuperNeo.Concrete.K := fun bit =>
    if Nat.testBit index.val bit then
      point.coordinates.getD bit K.zero
    else
      K.sub K.one (point.coordinates.getD bit K.zero)
  calc
    _ = ((canonicalFinIndices variables).map
          (fun bit => factor bit.val)).foldl K.mul K.one := by
      rw [List.foldl_map]
      rfl
    _ = ((List.range variables).map factor).foldl K.mul K.one := by
      congr 1
      simpa only [List.map_map, Function.comp_apply] using
        congrArg (List.map factor) (canonicalFinIndices_values variables)
    _ = (List.range variables).foldl
          (fun accumulated bit => K.mul accumulated (factor bit)) K.one := by
      rw [List.foldl_map]
    _ = Nightstream.Implementation.R1CS.PiCcsNc.productRange variables factor :=
      foldl_range_mul_eq_productRange variables factor
theorem equalityWeight_eq_chi
    {variables : Nat} (vertex : BooleanVertex variables)
    (point : CubePoint Nightstream.SuperNeo.Concrete.K variables) :
    vertex.equalityWeight ops point =
      MixedPolynomial.chi point.coordinates
        (NumericBooleanDomain.index vertex) := by
  let index : Fin (2 ^ variables) :=
    ⟨NumericBooleanDomain.index vertex,
      NumericBooleanDomain.index_lt_twoPow vertex⟩
  calc
    vertex.equalityWeight ops point =
        (NumericBooleanDomain.vertex variables index).equalityWeight ops point := by
      rw [NumericBooleanDomain.vertex_index]
    _ = NumericBooleanDomain.tensorWeight ops index point :=
      (NumericBooleanDomain.tensorWeight_eq_equalityWeight ops index point).symm
    _ = NumericBooleanDomain.testBitWeight ops point index :=
      NumericBooleanDomain.tensorWeight_eq_testBitWeight ops
        (NumericBooleanDomain.WeightProductLaws.ofInterpolationEvaluationLaws
          laws) index point
    _ = MixedPolynomial.chi point.coordinates index.val :=
      testBitWeight_eq_chi index point
theorem numericChi_eq_canonical
    {variables : Nat}
    (point : CubePoint Nightstream.SuperNeo.Concrete.K variables) :
    numericChi point.coordinates =
      (canonicalFinIndices (2 ^ variables)).map fun lane =>
        MixedPolynomial.chi point.coordinates lane.val := by
  unfold numericChi
  rw [point.dimension]
  calc
    (List.range (2 ^ variables)).map
        (MixedPolynomial.chi point.coordinates) =
      ((canonicalFinIndices (2 ^ variables)).map Fin.val).map
        (MixedPolynomial.chi point.coordinates) := by
          rw [canonicalFinIndices_values]
    _ = _ := by
      rw [List.map_map]
      apply List.map_congr_left
      intro lane _
      rfl
theorem zip_maps
    {Index Left Right : Type}
    (indices : List Index) (left : Index -> Left) (right : Index -> Right) :
    (indices.map left).zip (indices.map right) =
      indices.map fun index => (left index, right index) := by
  induction indices with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp [inductionHypothesis]
private theorem terminalValueAt_eq_numericDot
    {shape : SemanticShape}
    (message : Claims shape)
    (source : Fin shape.sourceCount)
    (point : CubePoint Nightstream.SuperNeo.Concrete.K
      PiCcsDomains.production.nc.laneVariables) :
    Terminal.valueAt (domain := PiCcsDomains.production.nc)
        message source point =
      BooleanTable.finiteSum ops
        ((((canonicalFinIndices PiCcsDomains.production.nc.laneCount).map
            fun lane => Terminal.paddedYZcol
              (domain := PiCcsDomains.production.nc) message source lane).zip
          (numericChi point.coordinates)).map fun pair =>
            K.mul pair.1 pair.2) := by
  unfold Terminal.valueAt Terminal.laneTable
  rw [← BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
    ops laws point]
  unfold BooleanReproduction.equalityWeighted
  calc
    FiniteSumAlgebra.sumMap ops
        (BooleanVertex.all PiCcsDomains.production.nc.laneVariables)
        (fun lane => K.mul (lane.equalityWeight ops point)
          (Terminal.paddedYZcol message source
            (BlockNcDomain.laneIndex lane))) =
      FiniteSumAlgebra.sumMap ops
        ((BooleanVertex.all
          PiCcsDomains.production.nc.laneVariables).map
            BlockNcDomain.laneIndex)
        (fun lane => K.mul
          (Terminal.paddedYZcol message source lane)
          (MixedPolynomial.chi point.coordinates lane.val)) := by
            unfold FiniteSumAlgebra.sumMap
            rw [List.map_map]
            apply congrArg (BooleanTable.finiteSum ops)
            apply List.map_congr_left
            intro lane _
            simp only [Function.comp_apply]
            rw [equalityWeight_eq_chi]
            exact laws.mul_comm _ _
    _ = FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices PiCcsDomains.production.nc.laneCount)
        (fun lane => K.mul
          (Terminal.paddedYZcol message source lane)
          (MixedPolynomial.chi point.coordinates lane.val)) :=
      sumMap_eq_of_perm laneIndices_perm _
    _ = BooleanTable.finiteSum ops
        ((((canonicalFinIndices PiCcsDomains.production.nc.laneCount).map
            fun lane => Terminal.paddedYZcol message source lane).zip
          (numericChi point.coordinates)).map fun pair =>
            K.mul pair.1 pair.2) := by
      simp only [BlockNcDomain.laneCount]
      rw [numericChi_eq_canonical point]
      unfold FiniteSumAlgebra.sumMap
      rw [zip_maps]
      rw [List.map_map]
      apply congrArg (BooleanTable.finiteSum ops)
      apply List.map_congr_left
      intro lane _
      rfl
private theorem productionOutputRows_numericDot
    {shape : SemanticShape} (message : Claims shape)
    (point : CubePoint K PiCcsDomains.production.nc.laneVariables) :
    (productionOutputRows message).map (fun row =>
        BooleanTable.finiteSum ops
          ((row.zip (numericChi point.coordinates)).map fun pair =>
            K.mul pair.1 pair.2)) =
      (canonicalFinIndices shape.sourceCount).map fun source =>
        Terminal.valueAt (domain := PiCcsDomains.production.nc)
          message source point := by
  rw [map_productionOutputRows]
  apply List.map_congr_left
  intro source _
  exact (terminalValueAt_eq_numericDot message source point).symm
theorem outputEvaluationValues_eq
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (pending : ProductionDelayedBlockLane)
    (assignment : Nat -> Nat)
    (constantOne : assignment 0 = 1)
    (computed : TerminalProgram.Computed assignment)
    (bindings : TerminalColumnBindings context certificate pending assignment) :
    TerminalProgram.outputTraces.map (fun trace =>
        sourceValue assignment trace.evaluation.output) =
      (canonicalFinIndices shape.sourceCount).map fun source =>
        Terminal.valueAt (domain := PiCcsDomains.production.nc)
          certificate.piCcs.output source
          (ProductionPiCcs.ncPoint context certificate).lane := by
  have chi := chiColumns_values_eq_numericChi assignment constantOne computed
  have evaluations := outputTraceEvaluations assignment computed
  calc
    TerminalProgram.outputTraces.map (fun trace =>
        sourceValue assignment trace.evaluation.output) =
      TerminalProgram.outputYZcolColumns.map fun row =>
        BooleanTable.finiteSum ops
          (((row.map (sourceValue assignment)).zip
            (TerminalProgram.chiColumns.map (sourceValue assignment))).map
              fun pair => K.mul pair.1 pair.2) := evaluations
    _ = TerminalProgram.outputYZcolColumns.map fun row =>
        BooleanTable.finiteSum ops
          (((row.map (sourceValue assignment)).zip (numericChi
            (TerminalProgram.lanePointColumns.map
              (sourceValue assignment)))).map fun pair =>
                K.mul pair.1 pair.2) := by
      apply List.map_congr_left
      intro row _
      rw [chi]
    _ = (materializedOutputRows assignment).map fun row =>
        BooleanTable.finiteSum ops
          ((row.zip (numericChi
            (TerminalProgram.lanePointColumns.map
              (sourceValue assignment)))).map fun pair =>
                K.mul pair.1 pair.2) := by
      symm
      exact map_materializedOutputRows assignment _
    _ = (productionOutputRows certificate.piCcs.output).map fun row =>
        BooleanTable.finiteSum ops
          ((row.zip (numericChi
            (ProductionPiCcs.ncPoint context certificate).lane.coordinates)
            ).map fun pair => K.mul pair.1 pair.2) := by
      rw [bindings.outputRows, bindings.lanePoint]
    _ = (canonicalFinIndices shape.sourceCount).map fun source =>
        Terminal.valueAt (domain := PiCcsDomains.production.nc)
          certificate.piCcs.output source
          (ProductionPiCcs.ncPoint context certificate).lane := by
      exact productionOutputRows_numericDot certificate.piCcs.output
        (ProductionPiCcs.ncPoint context certificate).lane
private theorem outputResidualValues_eq_map_range
    (assignment : Nat -> Nat)
    (computed : TerminalProgram.Computed assignment) :
    TerminalProgram.outputTraces.map (fun trace =>
        sourceValue assignment trace.residual.output) =
      (TerminalProgram.outputTraces.map fun trace =>
        sourceValue assignment trace.evaluation.output).map fun value =>
          K.mul (K.mul (K.add value (K.embed 1)) value)
            (K.sub value (K.embed 1)) := by
  rw [List.map_map]
  apply List.map_congr_left
  intro trace member
  have traceComputed := computed.outputs trace member
  calc
    sourceValue assignment trace.residual.output =
        ProductionMessageAcceptance.toConcreteK
          ((TerminalProgram.subtractTerms trace.cube.output
            trace.evaluation.output).value assignment) := by
      unfold sourceValue
      rw [traceComputed.residual]
    _ = K.sub (sourceValue assignment trace.cube.output)
          (sourceValue assignment trace.evaluation.output) :=
      sourceTerms_subtract assignment trace.cube.output trace.evaluation.output
    _ = K.sub
          (K.mul (sourceValue assignment trace.square.output)
            (sourceValue assignment trace.evaluation.output))
          (sourceValue assignment trace.evaluation.output) := by
      unfold sourceValue
      rw [traceComputed.cube,
        ProductionMessageAcceptance.toConcreteK_mul]
    _ = K.sub
          (K.mul
            (K.mul (sourceValue assignment trace.evaluation.output)
              (sourceValue assignment trace.evaluation.output))
            (sourceValue assignment trace.evaluation.output))
          (sourceValue assignment trace.evaluation.output) := by
      unfold sourceValue
      rw [traceComputed.square,
        ProductionMessageAcceptance.toConcreteK_mul]
    _ = K.mul
          (K.mul
            (K.add (sourceValue assignment trace.evaluation.output)
              (K.embed 1))
            (sourceValue assignment trace.evaluation.output))
          (K.sub (sourceValue assignment trace.evaluation.output)
            (K.embed 1)) :=
      cubic_sub_eq_range _
private theorem outputResidualValues_eq
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (pending : ProductionDelayedBlockLane)
    (assignment : Nat -> Nat)
    (constantOne : assignment 0 = 1)
    (computed : TerminalProgram.Computed assignment)
    (bindings : TerminalColumnBindings context certificate pending assignment) :
    TerminalProgram.outputResiduals.map (sourceValue assignment) =
      (canonicalFinIndices shape.sourceCount).map fun source =>
        Terminal.rangeAt (domain := PiCcsDomains.production.nc)
          certificate.piCcs.output source
          (ProductionPiCcs.ncPoint context certificate).lane := by
  unfold TerminalProgram.outputResiduals
  rw [List.map_map]
  change TerminalProgram.outputTraces.map (fun trace =>
      sourceValue assignment trace.residual.output) = _
  rw [outputResidualValues_eq_map_range assignment computed]
  rw [outputEvaluationValues_eq context certificate pending assignment
    constantOne computed bindings]
  rw [List.map_map]
  rfl
/-! ## Power schedules and source mixing -/
private theorem projectionPowersFrom_range
    (point : ProjectionProgram.K) : forall (offset count : Nat),
    ProjectionProgram.K.powersFrom point
        (ProjectionProgram.K.pow point offset) count =
      (List.range' offset count).map (ProjectionProgram.K.pow point)
  | offset, 0 => rfl
  | offset, count + 1 => by
      rw [List.range'_succ]
      simp only [ProjectionProgram.K.powersFrom, List.map_cons,
        List.cons.injEq]
      constructor
      · trivial
      · have nextCurrent : ProjectionProgram.K.mul
              (ProjectionProgram.K.pow point offset) point =
            ProjectionProgram.K.pow point (offset + 1) := by
            rw [ProjectionProgram.K.pow]
        simpa only [nextCurrent] using
          projectionPowersFrom_range point (offset + 1) count
private theorem projectionPowersFrom_one
    (point : ProjectionProgram.K) (count : Nat) :
    ProjectionProgram.K.powersFrom point ProjectionProgram.K.one count =
      (canonicalFinIndices count).map fun index =>
        ProjectionProgram.K.pow point index.val := by
  calc
    ProjectionProgram.K.powersFrom point ProjectionProgram.K.one count =
        (List.range' 0 count).map (ProjectionProgram.K.pow point) := by
      simpa using projectionPowersFrom_range point 0 count
    _ = (List.range count).map (ProjectionProgram.K.pow point) := by
      simp [List.range'_eq_map_range]
    _ = ((canonicalFinIndices count).map Fin.val).map
          (ProjectionProgram.K.pow point) := by
      rw [canonicalFinIndices_values]
    _ = _ := by
      rw [List.map_map]
      apply List.map_congr_left
      intro index _
      rfl
theorem map_projectionPow
    (point : ProjectionProgram.K) : forall exponent,
    ProductionMessageAcceptance.toConcreteK
        (ProjectionProgram.K.pow point exponent) =
      TargetPolynomial.power ops.toOps
        (ProductionMessageAcceptance.toConcreteK point) exponent
  | 0 => ProductionMessageAcceptance.toConcreteK_one
  | exponent + 1 => by
      rw [ProjectionProgram.K.pow,
        ProductionMessageAcceptance.toConcreteK_mul,
        TargetPolynomial.power, map_projectionPow point exponent]
      exact laws.mul_comm _ _
theorem map_projectionPowersFrom
    (point : ProjectionProgram.K) (count : Nat) :
    (ProjectionProgram.K.powersFrom point ProjectionProgram.K.one count).map
        ProductionMessageAcceptance.toConcreteK =
      (canonicalFinIndices count).map fun index =>
        TargetPolynomial.power ops.toOps
          (ProductionMessageAcceptance.toConcreteK point) index.val := by
  rw [projectionPowersFrom_one]
  rw [List.map_map]
  apply List.map_congr_left
  intro index _
  exact map_projectionPow point index.val
private theorem sourceCount_eq_outputCount
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows) :
    shape.sourceCount = outputCount := by
  have aligned := context.alignment.total_eq_sourceCount
  simpa [outputCount, FixedActive.arity, BatchArity.total] using aligned.symm
theorem runningCount_eq_outputCount_sub_one
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows) :
    shape.runningCount = outputCount - 1 := by
  have aligned := context.alignment.runningCount_eq
  simpa [outputCount, FixedActive.arity] using aligned.symm
theorem ordinarySumValue_eq_mixedRangeAt
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (pending : ProductionDelayedBlockLane)
    (assignment : Nat -> Nat)
    (constantOne : assignment 0 = 1)
    (computed : TerminalProgram.Computed assignment)
    (bindings : TerminalColumnBindings context certificate pending assignment) :
    sourceValue assignment TerminalProgram.ordinarySum.output =
      Terminal.mixedRangeAt certificate.piCcs.output context.ncCoins
        (ProductionPiCcs.ncPoint context certificate).lane := by
  have residuals := outputResidualValues_eq context certificate pending
    assignment constantOne computed bindings
  have countEq := sourceCount_eq_outputCount context
  calc
    sourceValue assignment TerminalProgram.ordinarySum.output =
        ProductionMessageAcceptance.toConcreteK
          (TerminalProgram.dotValue TerminalProgram.gammaPowers.powers
            TerminalProgram.outputResiduals assignment) := by
      unfold sourceValue
      rw [computed.ordinarySumValue]
    _ = BooleanTable.finiteSum ops
        ((TerminalProgram.gammaPowers.powers.zip
          TerminalProgram.outputResiduals).map fun pair =>
            K.mul (sourceValue assignment pair.1)
              (sourceValue assignment pair.2)) :=
      map_dotValue TerminalProgram.gammaPowers.powers
        TerminalProgram.outputResiduals assignment
    _ = BooleanTable.finiteSum ops
        (((TerminalProgram.gammaPowers.powers.map (sourceValue assignment)).zip
          (TerminalProgram.outputResiduals.map (sourceValue assignment))).map
            fun pair => K.mul pair.1 pair.2) := by
      apply congrArg (BooleanTable.finiteSum ops)
      exact map_zip_values TerminalProgram.gammaPowers.powers
        TerminalProgram.outputResiduals (sourceValue assignment)
        (sourceValue assignment) K.mul
    _ = BooleanTable.finiteSum ops
        (((canonicalFinIndices outputCount).map fun source =>
            TargetPolynomial.power ops.toOps context.ncCoins.gamma source.val
          ).zip
          ((canonicalFinIndices shape.sourceCount).map fun source =>
            Terminal.rangeAt (domain := PiCcsDomains.production.nc)
              certificate.piCcs.output source
              (ProductionPiCcs.ncPoint context certificate).lane) |>.map
            fun pair => K.mul pair.1 pair.2) := by
      have mappedPowers :
          TerminalProgram.gammaPowers.powers.map (sourceValue assignment) =
            (canonicalFinIndices outputCount).map fun source =>
              TargetPolynomial.power ops.toOps context.ncCoins.gamma
                source.val := by
        calc
          TerminalProgram.gammaPowers.powers.map (sourceValue assignment) =
              (TerminalProgram.gammaPowers.powers.map fun power =>
                power.value assignment).map
                  ProductionMessageAcceptance.toConcreteK := by
            simp [sourceValue, List.map_map, Function.comp_def]
          _ = (ProjectionProgram.K.powersFrom
                (TerminalProgram.gammaColumns.value assignment)
                ProjectionProgram.K.one outputCount).map
                  ProductionMessageAcceptance.toConcreteK := by
            rw [computed.gammaPowerValues]
          _ = (canonicalFinIndices outputCount).map fun source =>
              TargetPolynomial.power ops.toOps
                (sourceValue assignment TerminalProgram.gammaColumns)
                source.val := by
            simpa [sourceValue] using map_projectionPowersFrom
              (TerminalProgram.gammaColumns.value assignment) outputCount
          _ = (canonicalFinIndices outputCount).map fun source =>
              TargetPolynomial.power ops.toOps context.ncCoins.gamma
                source.val := by rw [bindings.gamma]
      rw [mappedPowers, residuals]
    _ = FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.sourceCount) fun source =>
          SignedJointIdentity.gammaTerm ops context.ncCoins.gamma
            (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing.sourceExponent
              shape .paperNc source)
            (Terminal.rangeAt (domain := PiCcsDomains.production.nc)
              certificate.piCcs.output source
              (ProductionPiCcs.ncPoint context certificate).lane) := by
      rw [← countEq]
      rw [zip_maps]
      unfold FiniteSumAlgebra.sumMap SignedJointIdentity.gammaTerm
      rw [List.map_map]
      rfl
    _ = Terminal.mixedRangeAt certificate.piCcs.output context.ncCoins
        (ProductionPiCcs.ncPoint context certificate).lane := by
      rfl
/-! ## Equality traces -/
/-- The materialized tail convention returns the incoming accumulator when
there are no tail coordinates, and otherwise the last emitted fold. -/
private def eqTailOutputOr
    (accumulator : KColumns) (traces : List TerminalProgram.EqTailTrace) :
    KColumns :=
  match traces.reverse with
  | [] => accumulator
  | trace :: _ => trace.fold.output
private theorem eqTailOutputOr_cons
    (accumulator : KColumns) (trace : TerminalProgram.EqTailTrace)
    (traces : List TerminalProgram.EqTailTrace) :
    eqTailOutputOr accumulator (trace :: traces) =
      eqTailOutputOr trace.fold.output traces := by
  cases traces with
  | nil => rfl
  | cons next rest =>
      unfold eqTailOutputOr
      simp only [List.reverse_cons]
      generalize rest.reverse = reversed
      cases reversed <;> rfl
/-- Left-to-right equality accumulation over the exact paired column lists
used to construct an `EqTrace`. -/
private def foldEqualityColumns
    (assignment : Nat -> Nat)
    (accumulator : Nightstream.SuperNeo.Concrete.K)
    (left right : List KColumns) : Nightstream.SuperNeo.Concrete.K :=
  (left.zip right).foldl (fun current pair =>
    K.mul current
      (SumCheckTruthPath.equalityFactor ops
        (sourceValue assignment pair.1)
        (sourceValue assignment pair.2))) accumulator
private theorem eqTailTracesFrom_value
    (assignment : Nat -> Nat) (constantOne : assignment 0 = 1) :
    forall (base : Nat) (accumulator : KColumns)
      (left right : List KColumns),
      (forall trace,
        trace ∈ TerminalProgram.eqTailTracesFrom base accumulator left right ->
          trace.Computed assignment) ->
      sourceValue assignment
          (eqTailOutputOr accumulator
            (TerminalProgram.eqTailTracesFrom base accumulator left right)) =
        foldEqualityColumns assignment (sourceValue assignment accumulator)
          left right
  | _, accumulator, [], _, _ => rfl
  | _, accumulator, _ :: _, [], _ => rfl
  | base, accumulator, left :: lefts, right :: rights, computed => by
      let trace : TerminalProgram.EqTailTrace :=
        { base := base, accumulator := accumulator, left := left,
          right := right }
      have headComputed : trace.Computed assignment :=
        computed trace (by simp [TerminalProgram.eqTailTracesFrom, trace])
      have tailComputed : forall candidate,
          candidate ∈ TerminalProgram.eqTailTracesFrom trace.next
            trace.fold.output lefts rights ->
          candidate.Computed assignment := by
        intro candidate member
        exact computed candidate (by
          simp [TerminalProgram.eqTailTracesFrom, trace, member])
      have productEq : sourceValue assignment trace.product.output =
          K.mul (sourceValue assignment trace.left)
            (sourceValue assignment trace.right) := by
        unfold sourceValue
        rw [headComputed.product,
          ProductionMessageAcceptance.toConcreteK_mul]
      have factorEq : sourceValue assignment trace.factor.output =
          SumCheckTruthPath.equalityFactor ops
            (sourceValue assignment trace.left)
            (sourceValue assignment trace.right) := by
        unfold sourceValue
        rw [headComputed.factor]
        exact sourceTerms_eqFactor assignment constantOne
          trace.product.output trace.left trace.right productEq
      have foldEq : sourceValue assignment trace.fold.output =
          K.mul (sourceValue assignment trace.accumulator)
            (SumCheckTruthPath.equalityFactor ops
              (sourceValue assignment trace.left)
              (sourceValue assignment trace.right)) := by
        unfold sourceValue
        rw [headComputed.fold,
          ProductionMessageAcceptance.toConcreteK_mul]
        exact congrArg (K.mul (sourceValue assignment trace.accumulator))
          factorEq
      rw [TerminalProgram.eqTailTracesFrom.eq_def, eqTailOutputOr_cons]
      rw [eqTailTracesFrom_value assignment constantOne trace.next
        trace.fold.output lefts rights tailComputed]
      unfold foldEqualityColumns
      simp only [List.zip_cons_cons, List.foldl_cons]
      rw [foldEq]
private theorem foldEqualityColumns_eq_pointEquality
    (assignment : Nat -> Nat) : forall
      (accumulator : Nightstream.SuperNeo.Concrete.K)
      (left right : List KColumns),
      left.length = right.length ->
      foldEqualityColumns assignment accumulator left right =
        K.mul accumulator
          (SumCheckTruthPath.pointEqualityCoordinates ops
            (left.map (sourceValue assignment))
            (right.map (sourceValue assignment)))
  | accumulator, [], [], _ => by
      unfold foldEqualityColumns SumCheckTruthPath.pointEqualityCoordinates
      exact laws.mul_one accumulator |>.symm
  | _, [], _ :: _, lengthEq => by simp at lengthEq
  | _, _ :: _, [], lengthEq => by simp at lengthEq
  | accumulator, left :: lefts, right :: rights, lengthEq => by
      simp only [List.length_cons, Nat.succ.injEq] at lengthEq
      unfold foldEqualityColumns
      simp only [List.zip_cons_cons, List.foldl_cons]
      simp only [List.map_cons,
        SumCheckTruthPath.pointEqualityCoordinates]
      calc
        (lefts.zip rights).foldl
            (fun current pair =>
              K.mul current
                (SumCheckTruthPath.equalityFactor ops
                  (sourceValue assignment pair.1)
                  (sourceValue assignment pair.2)))
            (K.mul accumulator
              (SumCheckTruthPath.equalityFactor ops
                (sourceValue assignment left)
                (sourceValue assignment right))) =
          K.mul
            (K.mul accumulator
              (SumCheckTruthPath.equalityFactor ops
                (sourceValue assignment left)
                (sourceValue assignment right)))
            (SumCheckTruthPath.pointEqualityCoordinates ops
              (lefts.map (sourceValue assignment))
              (rights.map (sourceValue assignment))) :=
          foldEqualityColumns_eq_pointEquality assignment _ _ _ lengthEq
        _ = K.mul accumulator
            (K.mul
              (SumCheckTruthPath.equalityFactor ops
                (sourceValue assignment left)
                (sourceValue assignment right))
              (SumCheckTruthPath.pointEqualityCoordinates ops
                (lefts.map (sourceValue assignment))
                (rights.map (sourceValue assignment)))) :=
          laws.mul_assoc _ _ _
private theorem eqTrace_output_value
    (assignment : Nat -> Nat) (constantOne : assignment 0 = 1)
    (trace : TerminalProgram.EqTrace)
    (computed : trace.Computed assignment)
    (firstPresent : trace.first?.isSome)
    (lengthEq : trace.left.length = trace.right.length) :
    sourceValue assignment trace.output =
      SumCheckTruthPath.pointEqualityCoordinates ops
        (trace.left.map (sourceValue assignment))
        (trace.right.map (sourceValue assignment)) := by
  cases leftEq : trace.left with
  | nil =>
      simp [TerminalProgram.EqTrace.first?, leftEq] at firstPresent
  | cons left lefts =>
      cases rightEq : trace.right with
      | nil =>
          simp [TerminalProgram.EqTrace.first?, leftEq, rightEq] at firstPresent
      | cons right rights =>
          let first : TerminalProgram.EqFirstTrace :=
            { base := trace.base, left := left, right := right }
          have firstEq : trace.first? = some first := by
            simp [TerminalProgram.EqTrace.first?, leftEq, rightEq, first]
          have firstComputed : first.Computed assignment :=
            computed.first first firstEq
          have productEq : sourceValue assignment first.product.output =
              K.mul (sourceValue assignment first.left)
                (sourceValue assignment first.right) := by
            unfold sourceValue
            rw [firstComputed.product,
              ProductionMessageAcceptance.toConcreteK_mul]
          have factorEq : sourceValue assignment first.factor.output =
              SumCheckTruthPath.equalityFactor ops
                (sourceValue assignment first.left)
                (sourceValue assignment first.right) := by
            unfold sourceValue
            rw [firstComputed.factor]
            exact sourceTerms_eqFactor assignment constantOne
              first.product.output first.left first.right productEq
          have tailEq : trace.tail =
              TerminalProgram.eqTailTracesFrom first.next
                first.factor.output lefts rights := by
            simp [TerminalProgram.EqTrace.tail, leftEq, rightEq, firstEq]
          have tailComputed : forall candidate,
              candidate ∈ TerminalProgram.eqTailTracesFrom first.next
                first.factor.output lefts rights ->
              candidate.Computed assignment := by
            intro candidate member
            apply computed.tail candidate
            rw [tailEq]
            exact member
          have tailLength : lefts.length = rights.length := by
            simpa [leftEq, rightEq] using lengthEq
          have outputEq : trace.output =
              eqTailOutputOr first.factor.output trace.tail := by
            unfold TerminalProgram.EqTrace.output eqTailOutputOr
            rw [firstEq]
            cases trace.tail.reverse <;> rfl
          calc
            sourceValue assignment trace.output =
                sourceValue assignment
                  (eqTailOutputOr first.factor.output
                    (TerminalProgram.eqTailTracesFrom first.next
                      first.factor.output lefts rights)) := by
              rw [outputEq, tailEq]
            _ = foldEqualityColumns assignment
                  (sourceValue assignment first.factor.output) lefts rights :=
              eqTailTracesFrom_value assignment constantOne first.next
                first.factor.output lefts rights tailComputed
            _ = K.mul (sourceValue assignment first.factor.output)
                  (SumCheckTruthPath.pointEqualityCoordinates ops
                    (lefts.map (sourceValue assignment))
                    (rights.map (sourceValue assignment))) :=
              foldEqualityColumns_eq_pointEquality assignment _ _ _ tailLength
            _ = K.mul
                  (SumCheckTruthPath.equalityFactor ops
                    (sourceValue assignment left)
                    (sourceValue assignment right))
                  (SumCheckTruthPath.pointEqualityCoordinates ops
                    (lefts.map (sourceValue assignment))
                    (rights.map (sourceValue assignment))) := by
              rw [factorEq]
            _ = SumCheckTruthPath.pointEqualityCoordinates ops
                  ((left :: lefts).map (sourceValue assignment))
                  ((right :: rights).map (sourceValue assignment)) := rfl
theorem blockEqualityValue_eq
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (pending : ProductionDelayedBlockLane)
    (assignment : Nat -> Nat)
    (constantOne : assignment 0 = 1)
    (computed : TerminalProgram.Computed assignment)
    (bindings : TerminalColumnBindings context certificate pending assignment) :
    sourceValue assignment TerminalProgram.blockEquality.output =
      SumCheckTruthPath.pointEquality ops
        (ProductionPiCcs.ncPoint context certificate).block
        context.ncCoins.betaBlock := by
  have lengthEq : TerminalProgram.blockEquality.left.length =
      TerminalProgram.blockEquality.right.length := by
    have leftLength := congrArg List.length bindings.blockPoint
    have rightLength := congrArg List.length bindings.betaBlock
    calc
      TerminalProgram.blockEquality.left.length =
          (ProductionPiCcs.ncPoint context certificate).block.coordinates.length :=
        by simpa [TerminalProgram.blockEquality] using leftLength
      _ = PiCcsDomains.production.nc.blockVariables :=
        (ProductionPiCcs.ncPoint context certificate).block.dimension
      _ = context.ncCoins.betaBlock.coordinates.length :=
        context.ncCoins.betaBlock.dimension.symm
      _ = TerminalProgram.blockEquality.right.length := by
        simpa [TerminalProgram.blockEquality] using rightLength.symm
  rw [SumCheckTruthPath.pointEquality]
  rw [← bindings.blockPoint, ← bindings.betaBlock]
  exact eqTrace_output_value assignment constantOne
    TerminalProgram.blockEquality computed.blockEqualityComputed
    (by simp [TerminalProgram.blockEquality_first]) lengthEq
theorem laneEqualityValue_eq
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (pending : ProductionDelayedBlockLane)
    (assignment : Nat -> Nat)
    (constantOne : assignment 0 = 1)
    (computed : TerminalProgram.Computed assignment)
    (bindings : TerminalColumnBindings context certificate pending assignment) :
    sourceValue assignment TerminalProgram.laneEquality.output =
      SumCheckTruthPath.pointEquality ops
        (ProductionPiCcs.ncPoint context certificate).lane
        context.ncCoins.betaA := by
  have lengthEq : TerminalProgram.laneEquality.left.length =
      TerminalProgram.laneEquality.right.length := by
    have leftLength := congrArg List.length bindings.lanePoint
    have rightLength := congrArg List.length bindings.betaLane
    calc
      TerminalProgram.laneEquality.left.length =
          (ProductionPiCcs.ncPoint context certificate).lane.coordinates.length :=
        by simpa [TerminalProgram.laneEquality] using leftLength
      _ = PiCcsDomains.production.nc.laneVariables :=
        (ProductionPiCcs.ncPoint context certificate).lane.dimension
      _ = context.ncCoins.betaA.coordinates.length :=
        context.ncCoins.betaA.dimension.symm
      _ = TerminalProgram.laneEquality.right.length := by
        simpa [TerminalProgram.laneEquality] using rightLength.symm
  rw [SumCheckTruthPath.pointEquality]
  rw [← bindings.lanePoint, ← bindings.betaLane]
  exact eqTrace_output_value assignment constantOne
    TerminalProgram.laneEquality computed.laneEqualityComputed
    (by simp [TerminalProgram.laneEquality_first]) lengthEq
theorem oldBlockEqualityValue_eq
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (pending : ProductionDelayedBlockLane)
    (assignment : Nat -> Nat)
    (constantOne : assignment 0 = 1)
    (computed : TerminalProgram.Computed assignment)
    (bindings : TerminalColumnBindings context certificate pending assignment) :
    sourceValue assignment TerminalProgram.oldBlockEquality.output =
      SumCheckTruthPath.pointEquality ops
        (ProductionPiCcs.ncPoint context certificate).block pending.oldBlock := by
  have lengthEq : TerminalProgram.oldBlockEquality.left.length =
      TerminalProgram.oldBlockEquality.right.length := by
    have leftLength := congrArg List.length bindings.blockPoint
    have rightLength := congrArg List.length bindings.pendingOldBlock
    calc
      TerminalProgram.oldBlockEquality.left.length =
          (ProductionPiCcs.ncPoint context certificate).block.coordinates.length :=
        by simpa [TerminalProgram.oldBlockEquality] using leftLength
      _ = PiCcsDomains.production.nc.blockVariables :=
        (ProductionPiCcs.ncPoint context certificate).block.dimension
      _ = pending.oldBlock.coordinates.length := pending.oldBlock.dimension.symm
      _ = TerminalProgram.oldBlockEquality.right.length := by
        simpa [TerminalProgram.oldBlockEquality] using rightLength.symm
  rw [SumCheckTruthPath.pointEquality]
  rw [← bindings.blockPoint, ← bindings.pendingOldBlock]
  exact eqTrace_output_value assignment constantOne
    TerminalProgram.oldBlockEquality computed.oldBlockEqualityComputed
    (by simp [TerminalProgram.oldBlockEquality_first]) lengthEq

end ProductionTerminalBridge
