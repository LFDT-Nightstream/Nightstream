import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionTerminalBridge.Kernels

/-!
Delayed terminal kernels for the fixed production combined-NC artifact.

Owns: interpretation of the materialized radix-two running sum and
producer-beta selector.  These are kept separate from the ordinary output
interpolation/equality kernels because they implement the delayed sidecar's
two independent recurrences.

Does not own: assignment construction, transcript replay, raw-child or parent
authority, commitment binding, `y_ring`, result-column assumptions, costs, or
row removal.
-/

/-!
Emits constraints: none; this module proves the delayed-projection terminal kernels.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.terminal.delayed_kernels` | Connect residual weighting and old-point projection terms to the terminal identity. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionTerminalBridge

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open Nightstream.Implementation.R1CS.PiCcsNc
open Nightstream.Implementation.R1CS.PiCcsNc.Authority
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws
private abbrev KColumns := ProjectionProgram.KColumns

universe uState

/-! ## Running-source order and radix powers -/

private theorem canonicalFinIndices_drop
    (fresh running : Nat) :
    (canonicalFinIndices (fresh + running)).drop fresh =
      (canonicalFinIndices running).map (Fin.natAdd fresh) := by
  apply List.ext_get
  · simp [canonicalFinIndices_length]
  · intro index leftBound rightBound
    apply Fin.ext
    simp [canonicalFinIndices]

private theorem freshCount_eq_one
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows) :
    shape.freshCount = 1 := by
  have aligned := context.alignment.freshCount_eq
  simpa [FixedActive.arity] using aligned.symm

private theorem runningEvaluationValues_eq
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
    TerminalProgram.runningValues.map (sourceValue assignment) =
      (canonicalFinIndices shape.runningCount).map fun running =>
        Terminal.valueAt (domain := PiCcsDomains.production.nc)
          certificate.piCcs.output (Data.runningIndex running)
          (ProductionPiCcs.ncPoint context certificate).lane := by
  have evaluations := outputEvaluationValues_eq context certificate pending
    assignment constantOne computed bindings
  have dropped := congrArg (List.drop 1) evaluations
  have leftDropped :
      TerminalProgram.runningValues.map (sourceValue assignment) =
        (TerminalProgram.outputTraces.map fun trace =>
          sourceValue assignment trace.evaluation.output).drop 1 := by
    rw [TerminalProgram.runningValues,
      TerminalProgram.freshOutputCount_eq_one, List.map_drop,
      TerminalProgram.outputEvaluations, List.map_map]
    rfl
  have freshEq := freshCount_eq_one context
  rw [← leftDropped] at dropped
  rw [← freshEq] at dropped
  change TerminalProgram.runningValues.map (sourceValue assignment) =
      (List.map
        (fun source : Fin (shape.freshCount + shape.runningCount) =>
          Terminal.valueAt certificate.piCcs.output source
            (ProductionPiCcs.ncPoint context certificate).lane)
        (canonicalFinIndices
          (shape.freshCount + shape.runningCount))).drop shape.freshCount
    at dropped
  rw [← List.map_drop,
    canonicalFinIndices_drop shape.freshCount shape.runningCount,
    List.map_map] at dropped
  simpa [Data.runningIndex, List.map_map, Function.comp_def] using dropped

private theorem targetPower_embed_two : forall exponent : Nat,
    TargetPolynomial.power ops.toOps (K.embed 2) exponent =
      K.embed
        ⟨2 ^ exponent % goldilocksModulus,
          Nat.mod_lt _ (by decide)⟩
  | 0 => by rfl
  | exponent + 1 => by
      rw [TargetPolynomial.power, targetPower_embed_two exponent]
      change ConcreteCarrier.extensionOps.mul
          (K.embed (2 : F))
          (K.embed
            ⟨2 ^ exponent % goldilocksModulus,
              Nat.mod_lt _ (by decide)⟩) =
        K.embed
          ⟨2 ^ (exponent + 1) % goldilocksModulus,
            Nat.mod_lt _ (by decide)⟩
      rw [laws.mul_comm (K.embed (2 : F))
        (K.embed
          ⟨2 ^ exponent % goldilocksModulus,
            Nat.mod_lt _ (by decide)⟩)]
      rw [← ConcreteCarrier.embed_mul]
      apply congrArg K.embed
      change
          Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.fieldOfNat
              (2 ^ exponent) * (2 : F) =
        Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.fieldOfNat
          (2 ^ (exponent + 1))
      rw [show (2 : F) =
          Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.fieldOfNat 2 by
        rfl]
      rw [← Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.fieldOfNat_mul]
      apply congrArg
        Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.fieldOfNat
      exact (Nat.pow_succ 2 exponent).symm

private theorem mappedRadixWeights
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows) :
    (canonicalFinIndices shape.runningCount).map (fun running =>
      TargetPolynomial.power ops.toOps (K.embed 2) running.val) =
    (canonicalFinIndices shape.runningCount).map (fun running =>
      K.embed (ProductionProjection.productionWeights context running)) := by
  apply List.map_congr_left
  intro running _
  rw [targetPower_embed_two]
  unfold ProductionProjection.productionWeights PiDEC.radixWeight
  have indexValue :
      (context.alignment.productRunningIndex running).val = running.val := rfl
  apply congrArg K.embed
  apply Fin.ext
  change 2 ^ running.val % goldilocksModulus =
    2 ^ (context.alignment.productRunningIndex running).val %
      goldilocksModulus
  rw [indexValue]

private theorem map_zipped_sourceValues
    (assignment : Nat -> Nat) : forall (left right : List KColumns),
    (left.zip right).map (fun pair =>
        K.mul (sourceValue assignment pair.1)
          (sourceValue assignment pair.2)) =
      ((left.map (sourceValue assignment)).zip
        (right.map (sourceValue assignment))).map
          (fun pair => K.mul pair.1 pair.2)
  | [], _ => rfl
  | _ :: _, [] => rfl
  | left :: lefts, right :: rights => by
      simp only [List.zip_cons_cons, List.map_cons, List.cons.injEq]
      exact ⟨True.intro, map_zipped_sourceValues assignment lefts rights⟩

theorem runningSumValue_eq_runningValueFromMessage
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
    sourceValue assignment TerminalProgram.runningSum.output =
      MessageTerminal.runningValueFromMessage certificate.piCcs.output
        (ProductionProjection.productionWeights context)
        (ProductionPiCcs.ncPoint context certificate).lane := by
  have runningValues := runningEvaluationValues_eq context certificate pending
    assignment constantOne computed bindings
  have powers := congrArg
    (List.map ProductionMessageAcceptance.toConcreteK)
    computed.radixPowerValues
  rw [List.map_map, map_projectionPowersFrom] at powers
  have radixConstant : sourceValue assignment
      TerminalProgram.radixConstant.output = K.embed 2 := by
    unfold sourceValue
    rw [computed.radixConstantValue]
    exact source_baseTwoTerms assignment constantOne
  change TerminalProgram.radixPowers.powers.map (sourceValue assignment) = _
    at powers
  unfold sourceValue at radixConstant
  rw [radixConstant] at powers
  have countEq := runningCount_eq_outputCount_sub_one context
  calc
    sourceValue assignment TerminalProgram.runningSum.output =
        ProductionMessageAcceptance.toConcreteK
          (TerminalProgram.dotValue TerminalProgram.radixPowers.powers
            TerminalProgram.runningValues assignment) := by
      unfold sourceValue
      rw [computed.runningSumValue]
    _ = BooleanTable.finiteSum ops
        ((TerminalProgram.radixPowers.powers.zip
          TerminalProgram.runningValues).map fun pair =>
            K.mul (sourceValue assignment pair.1)
              (sourceValue assignment pair.2)) :=
      map_dotValue TerminalProgram.radixPowers.powers
        TerminalProgram.runningValues assignment
    _ = BooleanTable.finiteSum ops
        (((TerminalProgram.radixPowers.powers.map
            (sourceValue assignment)).zip
          (TerminalProgram.runningValues.map (sourceValue assignment))).map
            fun pair => K.mul pair.1 pair.2) := by
      apply congrArg (BooleanTable.finiteSum ops)
      exact map_zipped_sourceValues assignment
        TerminalProgram.radixPowers.powers TerminalProgram.runningValues
    _ = BooleanTable.finiteSum ops
        (((canonicalFinIndices shape.runningCount).map fun running =>
            K.embed (ProductionProjection.productionWeights context running)
          ).zip
          ((canonicalFinIndices shape.runningCount).map fun running =>
            Terminal.valueAt certificate.piCcs.output
              (Data.runningIndex running)
              (ProductionPiCcs.ncPoint context certificate).lane) |>.map
            fun pair => K.mul pair.1 pair.2) := by
      rw [TerminalProgram.runningValues_length, ← countEq] at powers
      rw [powers, mappedRadixWeights context, runningValues]
    _ = MessageTerminal.runningValueFromMessage certificate.piCcs.output
        (ProductionProjection.productionWeights context)
        (ProductionPiCcs.ncPoint context certificate).lane := by
      unfold MessageTerminal.runningValueFromMessage
      rw [zip_maps]
      unfold FiniteSumAlgebra.sumMap
      rw [List.map_map]
      apply congrArg (BooleanTable.finiteSum ops)
      apply List.map_congr_left
      intro running _
      rfl

/-! ## Producer-beta selector trace -/

private def selectorOutputOr
    (accumulator : KColumns) (traces : List TerminalProgram.SelectorStep) :
    KColumns :=
  match traces.reverse with
  | [] => accumulator
  | trace :: _ => trace.fold.output

private theorem selectorOutputOr_cons
    (accumulator : KColumns) (trace : TerminalProgram.SelectorStep)
    (traces : List TerminalProgram.SelectorStep) :
    selectorOutputOr accumulator (trace :: traces) =
      selectorOutputOr trace.fold.output traces := by
  unfold selectorOutputOr
  rw [List.reverse_cons]
  cases reversed : traces.reverse <;>
    simp only [reversed, List.nil_append, List.cons_append]

private def selectorFold
    (accumulator betaPower : K) : List K -> K
  | [] => accumulator
  | coordinate :: coordinates =>
      selectorFold
        (K.mul accumulator
          (K.add (K.sub K.one coordinate)
            (K.mul coordinate betaPower)))
        (K.mul betaPower betaPower) coordinates

private theorem selectorStepsFrom_value
    (assignment : Nat -> Nat) (constantOne : assignment 0 = 1) :
    forall (base : Nat) (accumulator betaPower : KColumns)
      (coordinates : List KColumns),
      (forall trace,
        trace ∈ TerminalProgram.selectorStepsFrom base accumulator betaPower
          coordinates -> trace.Computed assignment) ->
      sourceValue assignment
          (selectorOutputOr accumulator
            (TerminalProgram.selectorStepsFrom base accumulator betaPower
              coordinates)) =
        selectorFold (sourceValue assignment accumulator)
          (sourceValue assignment betaPower)
          (coordinates.map (sourceValue assignment))
  | _, _, _, [], _ => rfl
  | base, accumulator, betaPower, coordinate :: coordinates, computed => by
      let trace : TerminalProgram.SelectorStep :=
        { base := base, accumulator := accumulator, betaPower := betaPower,
          coordinate := coordinate }
      have step : trace.Computed assignment :=
        computed trace (by simp [TerminalProgram.selectorStepsFrom, trace])
      have tail : forall candidate,
          candidate ∈ TerminalProgram.selectorStepsFrom trace.next
            trace.fold.output trace.squareBeta.output coordinates ->
          candidate.Computed assignment := by
        intro candidate member
        exact computed candidate (by
          simp [TerminalProgram.selectorStepsFrom, trace, member])
      have selected : sourceValue assignment trace.selected.output =
          K.mul (sourceValue assignment trace.coordinate)
            (sourceValue assignment trace.betaPower) := by
        unfold sourceValue
        rw [step.selected, ProductionMessageAcceptance.toConcreteK_mul]
      have factor : sourceValue assignment trace.factor.output =
          K.add (K.sub K.one (sourceValue assignment trace.coordinate))
            (K.mul (sourceValue assignment trace.coordinate)
              (sourceValue assignment trace.betaPower)) := by
        unfold sourceValue
        rw [step.factor]
        rw [sourceTerms_selectorFactor assignment constantOne]
        exact congrArg
          (K.add (K.sub K.one (sourceValue assignment trace.coordinate)))
          selected
      have fold : sourceValue assignment trace.fold.output =
          K.mul (sourceValue assignment trace.accumulator)
            (K.add (K.sub K.one (sourceValue assignment trace.coordinate))
              (K.mul (sourceValue assignment trace.coordinate)
                (sourceValue assignment trace.betaPower))) := by
        unfold sourceValue
        rw [step.fold, ProductionMessageAcceptance.toConcreteK_mul]
        exact congrArg (K.mul (sourceValue assignment trace.accumulator)) factor
      have square : sourceValue assignment trace.squareBeta.output =
          K.mul (sourceValue assignment trace.betaPower)
            (sourceValue assignment trace.betaPower) := by
        unfold sourceValue
        rw [step.squareBeta, ProductionMessageAcceptance.toConcreteK_mul]
      rw [TerminalProgram.selectorStepsFrom, selectorOutputOr_cons]
      rw [selectorStepsFrom_value assignment constantOne trace.next
        trace.fold.output trace.squareBeta.output coordinates tail]
      simp only [List.map_cons, selectorFold]
      rw [fold, square]

private def selectorProduct : K -> List K -> K
  | _, [] => K.one
  | betaPower, coordinate :: coordinates =>
      K.mul
        (K.add (K.sub K.one coordinate)
          (K.mul coordinate betaPower))
        (selectorProduct (K.mul betaPower betaPower) coordinates)

private theorem selectorFold_eq_product
    (accumulator betaPower : K) : forall coordinates : List K,
    selectorFold accumulator betaPower coordinates =
      K.mul accumulator (selectorProduct betaPower coordinates)
  | [] => (laws.mul_one accumulator).symm
  | coordinate :: coordinates => by
      rw [selectorFold, selectorFold_eq_product]
      change K.mul
          (K.mul accumulator
            (K.add (K.sub K.one coordinate)
              (K.mul coordinate betaPower)))
          (selectorProduct (K.mul betaPower betaPower) coordinates) =
        K.mul accumulator
          (K.mul
            (K.add (K.sub K.one coordinate)
              (K.mul coordinate betaPower))
            (selectorProduct (K.mul betaPower betaPower) coordinates))
      exact laws.mul_assoc _ _ _

private def shiftLaws : TargetPolynomial.ShiftLaws ops.toOps where
  one_mul := laws.one_mul
  mul_assoc := laws.mul_assoc
  mul_zero := laws.mul_zero
  mul_add := laws.left_distrib

private theorem targetPower_square (base : K) : forall exponent : Nat,
    TargetPolynomial.power ops.toOps base (2 * exponent) =
      TargetPolynomial.power ops.toOps (K.mul base base) exponent
  | 0 => rfl
  | exponent + 1 => by
      rw [show 2 * (exponent + 1) = 2 * exponent + 2 by omega]
      rw [TargetPolynomial.power_add ops.toOps shiftLaws]
      rw [targetPower_square base exponent]
      simp only [TargetPolynomial.power]
      rw [laws.mul_one]
      exact laws.mul_comm _ _

private theorem scaledPowerLow
    (scale base : K) (index : Nat) :
    K.mul scale
        (TargetPolynomial.power ops.toOps base (2 * index)) =
      K.mul scale
        (TargetPolynomial.power ops.toOps (K.mul base base) index) := by
  rw [targetPower_square]

private theorem scaledPowerHigh
    (scale base : K) (index : Nat) :
    K.mul scale
        (TargetPolynomial.power ops.toOps base (1 + 2 * index)) =
      K.mul (K.mul scale base)
        (TargetPolynomial.power ops.toOps (K.mul base base) index) := by
  rw [show 1 + 2 * index = (2 * index) + 1 by omega]
  simp only [TargetPolynomial.power]
  rw [targetPower_square]
  exact (laws.mul_assoc scale base _).symm

private theorem selectorInterpolation
    (scale betaPower coordinate tail : K) :
    K.add (K.mul scale tail)
        (K.mul coordinate
          (K.sub (K.mul (K.mul scale betaPower) tail)
            (K.mul scale tail))) =
      K.mul scale
        (K.mul
          (K.add (K.sub K.one coordinate)
            (K.mul coordinate betaPower)) tail) := by
  rw [← ConcreteCarrier.derived_sub_eq_concrete_sub,
    ← ConcreteCarrier.derived_sub_eq_concrete_sub]
  change ops.add (ops.mul scale tail)
      (ops.mul coordinate
        (ops.sub (ops.mul (ops.mul scale betaPower) tail)
          (ops.mul scale tail))) =
    ops.mul scale
      (ops.mul
        (ops.add (ops.sub ops.one coordinate)
          (ops.mul coordinate betaPower)) tail)
  unfold InterpolationOps.sub
  rw [laws.left_distrib, FiniteSumAlgebra.mul_neg ops laws]
  rw [laws.right_distrib, laws.right_distrib, laws.one_mul, laws.neg_mul]
  rw [laws.left_distrib, laws.left_distrib,
    FiniteSumAlgebra.mul_neg ops laws]
  change ops.add
      (ops.mul scale tail)
      (ops.add
        (ops.mul coordinate (ops.mul (ops.mul scale betaPower) tail))
        (ops.neg (ops.mul coordinate (ops.mul scale tail)))) =
    ops.add
      (ops.add (ops.mul scale tail)
        (ops.neg (ops.mul scale (ops.mul coordinate tail))))
      (ops.mul scale (ops.mul (ops.mul coordinate betaPower) tail))
  letI : Std.Associative ops.mul := ⟨laws.mul_assoc⟩
  letI : Std.Commutative ops.mul := ⟨laws.mul_comm⟩
  have positive :
      ops.mul coordinate (ops.mul (ops.mul scale betaPower) tail) =
        ops.mul scale (ops.mul (ops.mul coordinate betaPower) tail) := by
    ac_rfl
  have negative : ops.mul coordinate (ops.mul scale tail) =
      ops.mul scale (ops.mul coordinate tail) := by
    ac_rfl
  rw [positive, negative]
  letI : Std.Associative ops.add := ⟨laws.add_assoc⟩
  letI : Std.Commutative ops.add := ⟨laws.add_comm⟩
  ac_rfl

private theorem scaledPowerTable_evaluateCoordinates
    (scale betaPower : K) : forall
      (variables : Nat) (coordinates : List K),
      coordinates.length = variables ->
      (BooleanTable.tabulate (variables := variables) fun vertex =>
        K.mul scale
          (TargetPolynomial.power ops.toOps betaPower
            (NumericBooleanDomain.index vertex))).evaluateCoordinates
          ops coordinates =
        K.mul scale (selectorProduct betaPower coordinates)
  | 0, [], _ => by
      change K.mul scale ops.one = K.mul scale K.one
      rfl
  | 0, _ :: _, lengthEq => by simp at lengthEq
  | _ + 1, [], lengthEq => by simp at lengthEq
  | variables + 1, coordinate :: coordinates, lengthEq => by
      have tailLength : coordinates.length = variables :=
        Nat.succ.inj lengthEq
      simp only [BooleanTable.tabulate, BooleanTable.evaluateCoordinates]
      have low := scaledPowerTable_evaluateCoordinates scale
        (K.mul betaPower betaPower) variables coordinates tailLength
      have high := scaledPowerTable_evaluateCoordinates
        (K.mul scale betaPower) (K.mul betaPower betaPower)
        variables coordinates tailLength
      have lowFunctions :
          (fun tail : BooleanVertex variables =>
            K.mul scale
              (TargetPolynomial.power ops.toOps betaPower
                (NumericBooleanDomain.index (.cons false tail)))) =
        (fun tail => K.mul scale
          (TargetPolynomial.power ops.toOps (K.mul betaPower betaPower)
            (NumericBooleanDomain.index tail))) := by
        funext tail
        have exponent : NumericBooleanDomain.index (.cons false tail) =
            2 * NumericBooleanDomain.index tail := by
          simp only [NumericBooleanDomain.index]
          change 0 + 2 * NumericBooleanDomain.index tail = _
          omega
        rw [exponent]
        exact scaledPowerLow scale betaPower
          (NumericBooleanDomain.index tail)
      have highFunctions :
          (fun tail : BooleanVertex variables =>
            K.mul scale
              (TargetPolynomial.power ops.toOps betaPower
                (NumericBooleanDomain.index (.cons true tail)))) =
        (fun tail => K.mul (K.mul scale betaPower)
          (TargetPolynomial.power ops.toOps (K.mul betaPower betaPower)
            (NumericBooleanDomain.index tail))) := by
        funext tail
        have exponent : NumericBooleanDomain.index (.cons true tail) =
            1 + 2 * NumericBooleanDomain.index tail := by
          simp only [NumericBooleanDomain.index]
          change 1 + 2 * NumericBooleanDomain.index tail = _
          rfl
        rw [exponent]
        exact scaledPowerHigh scale betaPower
          (NumericBooleanDomain.index tail)
      rw [lowFunctions, highFunctions, low, high]
      rw [selectorProduct]
      rw [ConcreteCarrier.derived_sub_eq_concrete_sub]
      simpa only [ops, ConcreteCarrier.extensionOps] using
        selectorInterpolation scale betaPower coordinate
          (selectorProduct (K.mul betaPower betaPower) coordinates)

private theorem combinedSelector_eq_product
    {domain : BlockNcDomain}
    (producerBeta : K) (point : CubePoint K domain.laneVariables) :
    Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.betaPowerSelector
        producerBeta point =
      selectorProduct producerBeta point.coordinates := by
  unfold Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.betaPowerSelector
    BooleanTable.evaluate
  change (BooleanTable.tabulate (variables := domain.laneVariables) fun vertex =>
      TargetPolynomial.power ops.toOps producerBeta
        (NumericBooleanDomain.index vertex)).evaluateCoordinates
      ops point.coordinates = selectorProduct producerBeta point.coordinates
  have evaluated := scaledPowerTable_evaluateCoordinates K.one producerBeta
    domain.laneVariables point.coordinates point.dimension
  have oneFunctions :
      (fun vertex : BooleanVertex domain.laneVariables =>
        K.mul K.one
          (TargetPolynomial.power ops.toOps producerBeta
            (NumericBooleanDomain.index vertex))) =
      (fun vertex => TargetPolynomial.power ops.toOps producerBeta
        (NumericBooleanDomain.index vertex)) := by
    funext vertex
    change ops.mul ops.one
      (TargetPolynomial.power ops.toOps producerBeta
        (NumericBooleanDomain.index vertex)) = _
    exact laws.one_mul _
  have oneProduct : K.mul K.one
        (selectorProduct producerBeta point.coordinates) =
      selectorProduct producerBeta point.coordinates := by
    change ops.mul ops.one (selectorProduct producerBeta point.coordinates) = _
    exact laws.one_mul _
  rw [oneFunctions, oneProduct] at evaluated
  exact evaluated

theorem selectorOutputValue_eq_betaPowerSelector
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
    sourceValue assignment TerminalProgram.selectorOutput =
      Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.betaPowerSelector
        context.producerBeta
        (ProductionPiCcs.ncPoint context certificate).lane := by
  have steps : forall trace, trace ∈ TerminalProgram.selectorSteps ->
      trace.Computed assignment := computed.selectorStepsComputed
  have traceValue := selectorStepsFrom_value assignment constantOne
    TerminalProgram.selectorInitial.next TerminalProgram.selectorInitial.output
    TerminalProgram.producerBetaColumns TerminalProgram.lanePointColumns steps
  have initial : sourceValue assignment TerminalProgram.selectorInitial.output =
      K.one := by
    unfold sourceValue
    rw [computed.selectorInitialValue]
    exact source_oneTerms assignment constantOne
  change sourceValue assignment
      (selectorOutputOr TerminalProgram.selectorInitial.output
        TerminalProgram.selectorSteps) = _
  rw [TerminalProgram.selectorSteps]
  rw [traceValue, initial, bindings.producerBeta, bindings.lanePoint]
  rw [selectorFold_eq_product]
  change ops.mul ops.one
      (selectorProduct context.producerBeta
        (ProductionPiCcs.ncPoint context certificate).lane.coordinates) = _
  rw [laws.one_mul]
  exact (combinedSelector_eq_product context.producerBeta
    (ProductionPiCcs.ncPoint context certificate).lane).symm

end ProductionTerminalBridge
