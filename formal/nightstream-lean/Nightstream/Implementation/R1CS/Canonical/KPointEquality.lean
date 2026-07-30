import Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
import Nightstream.Implementation.R1CS.Canonical.KLinear
import Nightstream.Implementation.R1CS.Canonical.KMulChainOwnership
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial

/-!
Contract: the paper PiCCS multilinear point-equality polynomial as canonical
rows over the concrete Goldilocks quadratic extension.

Owns: one affine-factor multiplication per coordinate, the optimized product
chain, exact cost, and soundness to the unchanged
`SumCheckTruthPath.pointEquality`.

The public semantic lemma `equalityFactor_eq_affine` justifies the optimized
factor `(1-r) + l*(r-(1-r))`; the paper definition itself is not changed.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KPointEquality

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge

abbrev ConcreteK := Nightstream.SuperNeo.Concrete.K

structure Input (variables : Nat) where
  left : Fin variables → Carried
  right : Fin variables → Carried
  frameBase : Nat

def indices (variables : Nat) : List (Fin variables) :=
  List.ofFn (fun index => index)

theorem indices_length (variables : Nat) :
    (indices variables).length = variables := by
  unfold indices
  rw [List.length_ofFn]

def intercept {variables : Nat} (input : Input variables)
    (index : Fin variables) : Carried :=
  KLinear.oneMinus (input.right index)

def slope {variables : Nat} (input : Input variables)
    (index : Fin variables) : Carried :=
  KLinear.subCarried (input.right index) (intercept input index)

def factorFrame {variables : Nat} (input : Input variables)
    (index : Fin variables) : Frame :=
  KFrames.frameAt input.frameBase index.val

def factorProduct {variables : Nat} (input : Input variables)
    (index : Fin variables) : Carried :=
  KMulChain.frameOutput (factorFrame input index)

/-- `(1-r) + l*(r-(1-r))`, row-free after the one multiplication. -/
def factor {variables : Nat} (input : Input variables)
    (index : Fin variables) : Carried :=
  KLinear.addCarried (intercept input index) (factorProduct input index)

def factorRows {variables : Nat} (input : Input variables) : List Row :=
  (indices variables).flatMap fun index =>
    KMul.rows (input.left index) (slope input index) (factorFrame input index)

def factors {variables : Nat} (input : Input variables) : List Carried :=
  (indices variables).map (factor input)

theorem factors_length {variables : Nat} (input : Input variables) :
    (factors input).length = variables := by
  unfold factors
  rw [List.length_map, indices_length]

theorem factorRows_length {variables : Nat} (input : Input variables) :
    (factorRows input).length = 3 * variables := by
  unfold factorRows
  rw [List.length_flatMap]
  have each :
      ((indices variables).map
        (fun index =>
          (KMul.rows (input.left index) (slope input index)
            (factorFrame input index)).length)).sum =
        ((indices variables).map (fun _ => 3)).sum := by
    apply congrArg List.sum
    apply List.map_congr_left
    intro index _
    exact KMul.rows_length _ _ _
  rw [each]
  have constantSum : ∀ entries : List (Fin variables),
      (entries.map (fun _ => 3)).sum = 3 * entries.length := by
    intro entries
    induction entries with
    | nil => rfl
    | cons _ rest inductionHypothesis =>
        rw [List.map_cons, List.sum_cons, inductionHypothesis]
        simp only [List.length_cons]
        omega
  rw [constantSum, indices_length]

def productBase {variables : Nat} (input : Input variables) : Nat :=
  input.frameBase + 3 * variables

def productRows {variables : Nat} (input : Input variables) : List Row :=
  match factors input with
  | [] => []
  | first :: rest =>
      KMulChain.rows first (KFrames.frameAt (productBase input)) rest 0

def equalityCarried {variables : Nat} (input : Input variables) : Carried :=
  match factors input with
  | [] => KLinear.oneCarried
  | first :: rest =>
      KMulChain.productCarried first
        (KFrames.frameAt (productBase input)) rest 0

def rows {variables : Nat} (input : Input variables) : List Row :=
  factorRows input ++ productRows input

theorem productRows_length {variables : Nat} (input : Input variables) :
    (productRows input).length = 3 * (variables - 1) := by
  unfold productRows
  split
  next empty =>
    have sized := factors_length input
    rw [empty] at sized
    simp only [List.length_nil] at sized
    have variablesZero : variables = 0 := sized.symm
    subst variables
    rfl
  next first rest equal =>
    rw [KMulChain.rows_length]
    have sized := factors_length input
    rw [equal] at sized
    simp only [List.length_cons] at sized
    omega

theorem rows_length {variables : Nat} (input : Input variables) :
    (rows input).length = 3 * variables + 3 * (variables - 1) := by
  unfold rows
  rw [List.length_append, factorRows_length, productRows_length]

def columns {variables : Nat} (input : Input variables) : List Nat :=
  KFrames.frameColumns input.frameBase variables ++
    KFrames.frameColumns (productBase input) (variables - 1)

theorem columns_length {variables : Nat} (input : Input variables) :
    (columns input).length = 3 * variables + 3 * (variables - 1) := by
  unfold columns
  rw [List.length_append, KFrames.frameColumns_length,
    KFrames.frameColumns_length]

theorem columns_nodup {variables : Nat} (input : Input variables) :
    (columns input).Nodup := by
  unfold columns productBase
  rw [List.nodup_append]
  refine ⟨KFrames.frameColumns_nodup _ _,
    KFrames.frameColumns_nodup _ _, ?_⟩
  intro left leftMember right rightMember equal
  rw [KFrames.frameColumns_mem_iff] at leftMember rightMember
  omega

/-! ## Decoding and factor soundness -/

def decoded (assignment : Nat → Nat) (value : Carried) : ConcreteK :=
  ofProjection
    (KFixedPhaseSumCheck.decodeCarried assignment value)

theorem ofConcrete_decoded (assignment : Nat → Nat) (value : Carried) :
    KConcreteBridge.ofConcrete (decoded assignment value) =
      carriedValue assignment value := by
  unfold decoded
  rw [← toPair_toProjection, toProjection_ofProjection,
    KFixedPhaseSumCheck.toPair_decodeCarried]

def decodedLeft {variables : Nat} (input : Input variables)
    (assignment : Nat → Nat) :
    CubePoint ConcreteK variables where
  coordinates :=
    (indices variables).map fun index => decoded assignment (input.left index)
  dimension := by
    rw [List.length_map, indices_length]

def decodedRight {variables : Nat} (input : Input variables)
    (assignment : Nat → Nat) :
    CubePoint ConcreteK variables where
  coordinates :=
    (indices variables).map fun index => decoded assignment (input.right index)
  dimension := by
    rw [List.length_map, indices_length]

theorem factorRows_satisfied
    {variables : Nat} (input : Input variables)
    (assignment : Nat → Nat)
    (satisfied : Satisfies (factorRows input) assignment)
    (index : Fin variables) :
    Satisfies
      (KMul.rows (input.left index) (slope input index)
        (factorFrame input index)) assignment := by
  intro row member
  exact satisfied row
    (List.mem_flatMap.2
      ⟨index, List.mem_ofFn.mpr ⟨index, rfl⟩, member⟩)

theorem factor_pair_sound
    {variables : Nat} (input : Input variables)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (factorRows input) assignment)
    (index : Fin variables) :
    carriedValue assignment (factor input index) =
      addPair
        (KPairLaws.subPair ⟨1, 0⟩
          (carriedValue assignment (input.right index)))
        (mulPair
          (carriedValue assignment (input.left index))
          (KPairLaws.subPair
            (carriedValue assignment (input.right index))
            (KPairLaws.subPair ⟨1, 0⟩
              (carriedValue assignment (input.right index))))) := by
  have product :=
    KMulChain.frameOutput_sound assignment
      (input.left index) (slope input index) (factorFrame input index)
      (factorRows_satisfied input assignment satisfied index)
  rw [factor, KLinear.carriedValue_add, intercept,
    KLinear.carriedValue_oneMinus _ _ constantWire, factorProduct, product, slope,
    KLinear.carriedValue_sub, intercept,
    KLinear.carriedValue_oneMinus _ _ constantWire]

theorem factor_semantic_sound
    {variables : Nat} (input : Input variables)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (factorRows input) assignment)
    (index : Fin variables) :
    decoded assignment (factor input index) =
      SumCheckTruthPath.equalityFactor ConcreteCarrier.extensionOps
        (decoded assignment (input.left index))
        (decoded assignment (input.right index)) := by
  apply KConcreteBridge.ofConcrete_injective
  rw [ofConcrete_decoded,
    SumCheckTruthPath.equalityFactor_eq_affine
      ConcreteCarrier.extensionLaws]
  rw [ConcreteCarrier.derived_sub_eq_concrete_sub,
    ConcreteCarrier.derived_sub_eq_concrete_sub]
  change
    carriedValue assignment (factor input index) =
      KConcreteBridge.ofConcrete
        (Nightstream.SuperNeo.Concrete.K.add
          (Nightstream.SuperNeo.Concrete.K.sub
            Nightstream.SuperNeo.Concrete.K.one
            (decoded assignment (input.right index)))
          (Nightstream.SuperNeo.Concrete.K.mul
            (decoded assignment (input.left index))
            (Nightstream.SuperNeo.Concrete.K.sub
              (decoded assignment (input.right index))
              (Nightstream.SuperNeo.Concrete.K.sub
                Nightstream.SuperNeo.Concrete.K.one
                (decoded assignment (input.right index))))))
  rw [KConcreteBridge.ofConcrete_add, KConcreteBridge.ofConcrete_sub,
    KConcreteBridge.ofConcrete_mul, KConcreteBridge.ofConcrete_sub,
    KConcreteBridge.ofConcrete_sub,
    KConcreteBridge.ofConcrete_agrees_with_toPair
      Nightstream.SuperNeo.Concrete.K.one
      Nightstream.Implementation.R1CS.ProjectionProgram.K.one rfl rfl,
    ofConcrete_decoded, ofConcrete_decoded]
  exact factor_pair_sound input assignment constantWire satisfied index

/-! ## Product soundness -/

def pairProduct : List Pair → Pair
  | [] => ⟨1, 0⟩
  | value :: rest => mulPair value (pairProduct rest)

theorem pairProduct_canonical :
    ∀ values : List Pair,
      (pairProduct values).low < goldilocksP ∧
        (pairProduct values).high < goldilocksP
  | [] => ⟨by decide, by decide⟩
  | _ :: rest => KPairLaws.mulPair_canonical _ (pairProduct rest)

theorem productValue_eq_pairProduct (initial : Pair)
    (initialLow : initial.low < goldilocksP)
    (initialHigh : initial.high < goldilocksP) :
    ∀ rest : List Pair,
      KMulChain.productValue initial rest =
        pairProduct (initial :: rest)
  | [] => by
      change initial = mulPair initial ⟨1, 0⟩
      rw [KPairLaws.mulPair_comm,
        KPairLaws.mulPair_one_left initial initialLow initialHigh]
  | factor :: rest => by
      change
        KMulChain.productValue (mulPair initial factor) rest =
          mulPair initial (mulPair factor (pairProduct rest))
      rw [productValue_eq_pairProduct (mulPair initial factor)
        (KPairLaws.mulPair_canonical _ _).1
        (KPairLaws.mulPair_canonical _ _).2 rest]
      change
        mulPair (mulPair initial factor) (pairProduct rest) =
          mulPair initial (mulPair factor (pairProduct rest))
      exact KPairLaws.mulPair_assoc initial factor (pairProduct rest)

theorem equality_pair_sound
    {variables : Nat} (input : Input variables)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    carriedValue assignment (equalityCarried input) =
      pairProduct ((factors input).map (carriedValue assignment)) := by
  have factorSatisfied :
      Satisfies (factorRows input) assignment :=
    fun row member =>
      satisfied row (List.mem_append_left _ member)
  have productSatisfied :
      Satisfies (productRows input) assignment :=
    fun row member =>
      satisfied row (List.mem_append_right _ member)
  unfold equalityCarried
  split
  next empty =>
    rw [KLinear.carriedValue_one assignment constantWire]
    simp [empty, pairProduct]
  next first rest equal =>
    have chainSatisfied :
        Satisfies
          (KMulChain.rows first (KFrames.frameAt (productBase input)) rest 0)
          assignment := by
      simpa [productRows, equal] using productSatisfied
    have chain :=
      KMulChain.rows_sound assignment
        (KFrames.frameAt (productBase input)) first rest 0 chainSatisfied
    rw [chain, productValue_eq_pairProduct
      (carriedValue assignment first)
      (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide))
      (rest.map (carriedValue assignment))]
    rw [equal]
    rfl

def semanticProduct : List ConcreteK → ConcreteK
  | [] => Nightstream.SuperNeo.Concrete.K.one
  | value :: rest =>
      Nightstream.SuperNeo.Concrete.K.mul value (semanticProduct rest)

theorem ofConcrete_semanticProduct :
    ∀ values : List ConcreteK,
      KConcreteBridge.ofConcrete (semanticProduct values) =
        pairProduct (values.map KConcreteBridge.ofConcrete)
  | [] => rfl
  | value :: rest => by
      rw [semanticProduct, List.map_cons, pairProduct,
        KConcreteBridge.ofConcrete_mul,
        ofConcrete_semanticProduct rest]

theorem semanticProduct_factors
    {variables : Nat} (input : Input variables)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (factorRows input) assignment) :
    semanticProduct ((indices variables).map fun index =>
      SumCheckTruthPath.equalityFactor ConcreteCarrier.extensionOps
        (decoded assignment (input.left index))
        (decoded assignment (input.right index))) =
      semanticProduct ((factors input).map (decoded assignment)) := by
  congr 1
  unfold factors
  rw [List.map_map]
  apply List.map_congr_left
  intro index _
  exact (factor_semantic_sound input assignment constantWire
    satisfied index).symm

theorem semanticProduct_eq_pointEqualityCoordinates
    {variables : Nat} (input : Input variables)
    (assignment : Nat → Nat) :
    semanticProduct ((indices variables).map fun index =>
      SumCheckTruthPath.equalityFactor ConcreteCarrier.extensionOps
        (decoded assignment (input.left index))
        (decoded assignment (input.right index))) =
      SumCheckTruthPath.pointEqualityCoordinates ConcreteCarrier.extensionOps
        (decodedLeft input assignment).coordinates
        (decodedRight input assignment).coordinates := by
  change
    semanticProduct ((indices variables).map fun index =>
      SumCheckTruthPath.equalityFactor ConcreteCarrier.extensionOps
        (decoded assignment (input.left index))
        (decoded assignment (input.right index))) =
      SumCheckTruthPath.pointEqualityCoordinates ConcreteCarrier.extensionOps
        ((indices variables).map fun index =>
          decoded assignment (input.left index))
        ((indices variables).map fun index =>
          decoded assignment (input.right index))
  generalize indices variables = coordinateIndices
  induction coordinateIndices with
  | nil => rfl
  | cons index rest inductionHypothesis =>
      simp only [List.map_cons, semanticProduct,
        SumCheckTruthPath.pointEqualityCoordinates]
      rw [inductionHypothesis]
      rfl

/-- Satisfying rows compute exactly the unchanged paper point-equality
polynomial on values decoded from the same authoritative coordinate columns. -/
theorem rows_sound
    {variables : Nat} (input : Input variables)
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows input) assignment) :
    decoded assignment (equalityCarried input) =
      SumCheckTruthPath.pointEquality ConcreteCarrier.extensionOps
        (decodedLeft input assignment) (decodedRight input assignment) := by
  apply KConcreteBridge.ofConcrete_injective
  have factorSatisfied :
      Satisfies (factorRows input) assignment :=
    fun row member =>
      satisfied row (List.mem_append_left _ member)
  rw [ofConcrete_decoded]
  calc
    carriedValue assignment (equalityCarried input) =
        pairProduct ((factors input).map (carriedValue assignment)) :=
      equality_pair_sound input assignment constantWire satisfied
    _ = KConcreteBridge.ofConcrete
        (semanticProduct ((factors input).map (decoded assignment))) := by
      rw [ofConcrete_semanticProduct]
      congr 1
      rw [List.map_map]
      apply List.map_congr_left
      intro value _
      exact ofConcrete_decoded assignment value
    _ = KConcreteBridge.ofConcrete
        (semanticProduct ((indices variables).map fun index =>
          SumCheckTruthPath.equalityFactor ConcreteCarrier.extensionOps
            (decoded assignment (input.left index))
            (decoded assignment (input.right index)))) := by
      apply congrArg KConcreteBridge.ofConcrete
      exact (semanticProduct_factors input assignment constantWire
        factorSatisfied).symm
    _ = KConcreteBridge.ofConcrete
        (SumCheckTruthPath.pointEqualityCoordinates
          ConcreteCarrier.extensionOps
          (decodedLeft input assignment).coordinates
          (decodedRight input assignment).coordinates) := by
      apply congrArg KConcreteBridge.ofConcrete
      exact semanticProduct_eq_pointEqualityCoordinates input assignment
    _ = KConcreteBridge.ofConcrete
        (SumCheckTruthPath.pointEquality ConcreteCarrier.extensionOps
          (decodedLeft input assignment) (decodedRight input assignment)) := rfl

end Nightstream.Implementation.R1CS.Canonical.KPointEquality
