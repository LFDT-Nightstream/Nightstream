import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedJointIdentity
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction
import Nightstream.SuperNeo.SumCheck.HypercubeTruth

/-!
Canonical SumCheck truth path for the explicit paper-level joint polynomial.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: joint-polynomial evaluation, Boolean initial sum, and SumCheck terminal.
Constraint family: semantic truth-path owner; this file emits no constraints.

Owns: the full-point equality polynomial, arbitrary-point `F`, `NC`, `Eval`,
and `Q`, a total coordinate-list polynomial with fail-closed arity, its exact
specialization to the existing Boolean `Q`, and the canonical finite SumCheck
expected-round path from that one polynomial.

Does not own: a degree proof for the derived round polynomials, root counting,
challenge sampling, Fiat--Shamir, production FE/NC splitting, concrete table
construction, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: both the semantic initial sum and terminal evaluation are
derived from the same explicit `Q`. No prover message, caller-supplied expected
round function, terminal oracle, Rust evaluator, or old circuit enters this
construction. A coordinate list of the wrong arity evaluates to zero, while a
valid verifier execution must separately enforce the exact round count.

| Protocol | Phase | Mathematical object | Exact result |
|---|---|---|---|
| `Pi_CCS` | point evaluation | `pointEquality` | `prod_i ((1-x_i)(1-r_i)+x_i r_i)` |
| `Pi_CCS` | point evaluation | `qAtPoint` | arbitrary-field-point `Q(x,A,C)` |
| `Pi_CCS` | Boolean restriction | `qAtPoint_toCubePoint_eq_qAt` | arbitrary-point `Q` agrees with the explicit Boolean definition |
| `Pi_CCS` | initial sum | `sumCompletions_jointPolynomial_eq_summedQ` | canonical completion sum equals `sum_x Q` |
| `Pi_CCS` | terminal | `jointPolynomial challenges` | same `Q` at the full challenge vector |
| `Pi_CCS` | truth path | `canonicalGhosts_honest` | expected rounds are derived, finite, and terminal-bound |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckTruthPath

universe uField

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.SumCheck

namespace VertexEncoding

/-- Compatibility name for the neutral canonical Boolean point encoding. -/
abbrev fieldCoordinates
    {Field : Type uField}
    (ops : InterpolationOps Field) :
    {variables : Nat} -> BooleanVertex variables -> List Field
  := BooleanVertex.fieldCoordinates ops

theorem fieldCoordinates_length
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (vertex : BooleanVertex variables) :
    (fieldCoordinates ops vertex).length = variables :=
  BooleanVertex.fieldCoordinates_length ops vertex

/-- Compatibility name for the neutral dimension-checked field point. -/
abbrev toCubePoint
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (vertex : BooleanVertex variables) : CubePoint Field variables :=
  BooleanVertex.toCubePoint ops vertex

end VertexEncoding

/-- One coordinate factor of the multilinear equality polynomial. -/
def equalityFactor
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (left right : Field) : Field :=
  ops.add
    (ops.mul (ops.sub ops.one left) (ops.sub ops.one right))
    (ops.mul left right)

/-- Total coordinate-list equality polynomial. Shape mismatches fail closed. -/
def pointEqualityCoordinates
    {Field : Type uField}
    (ops : InterpolationOps Field) : List Field -> List Field -> Field
  | [], [] => ops.one
  | left :: lefts, right :: rights =>
      ops.mul (equalityFactor ops left right)
        (pointEqualityCoordinates ops lefts rights)
  | _, _ => ops.zero

/-- Equality polynomial between two dimension-checked cube points. -/
def pointEquality
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (left right : CubePoint Field variables) : Field :=
  pointEqualityCoordinates ops left.coordinates right.coordinates

private theorem zero_mul
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (value : Field) :
    ops.mul ops.zero value = ops.zero := by
  rw [laws.mul_comm, laws.mul_zero]

private theorem sub_one_zero
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) :
    ops.sub ops.one ops.zero = ops.one := by
  unfold InterpolationOps.sub
  have negZero : ops.neg ops.zero = ops.zero := by
    have inverse := laws.add_neg ops.zero
    simpa only [laws.zero_add] using inverse
  rw [negZero, laws.add_zero]

private theorem sub_one_one
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) :
    ops.sub ops.one ops.one = ops.zero := by
  exact laws.add_neg ops.one

private theorem equalityFactor_zero
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (right : Field) :
    equalityFactor ops ops.zero right = ops.sub ops.one right := by
  unfold equalityFactor
  rw [sub_one_zero ops laws, laws.one_mul, zero_mul ops laws,
    laws.add_zero]

private theorem equalityFactor_one
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (right : Field) :
    equalityFactor ops ops.one right = right := by
  unfold equalityFactor
  rw [sub_one_one ops laws, zero_mul ops laws, laws.one_mul,
    laws.zero_add]

private theorem add_sub_self_right
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (low high : Field) :
    ops.add low (ops.sub high low) = high := by
  unfold InterpolationOps.sub
  calc
    ops.add low (ops.add high (ops.neg low)) =
        ops.add (ops.add low high) (ops.neg low) :=
      (laws.add_assoc _ _ _).symm
    _ = ops.add (ops.add high low) (ops.neg low) := by
      rw [laws.add_comm low high]
    _ = ops.add high (ops.add low (ops.neg low)) :=
      laws.add_assoc _ _ _
    _ = ops.add high ops.zero := by rw [laws.add_neg]
    _ = high := laws.add_zero high

/-- The independent recursive table MLE specializes to the exact leaf at a
canonical Boolean point. -/
theorem evaluateCoordinates_fieldCoordinates_eq_valueAt
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables : Nat}
    (table : BooleanTable Field variables)
    (vertex : BooleanVertex variables) :
    table.evaluateCoordinates ops (VertexEncoding.fieldCoordinates ops vertex) =
      table.valueAt vertex := by
  induction table with
  | leaf value =>
      cases vertex
      rfl
  | @branch tailVariables low high lowInduction highInduction =>
      cases vertex with
      | cons coordinate tail =>
          cases coordinate
          · simp only [VertexEncoding.fieldCoordinates,
              BooleanVertex.fieldCoordinates,
              BooleanTable.evaluateCoordinates, BooleanTable.valueAt]
            rw [zero_mul ops laws, laws.add_zero,
              lowInduction tail]
          · simp only [VertexEncoding.fieldCoordinates,
              BooleanVertex.fieldCoordinates,
              BooleanTable.evaluateCoordinates, BooleanTable.valueAt]
            rw [laws.one_mul, add_sub_self_right ops laws,
              highInduction tail]

theorem evaluate_toCubePoint_eq_valueAt
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables : Nat}
    (table : BooleanTable Field variables)
    (vertex : BooleanVertex variables) :
    table.evaluate ops (VertexEncoding.toCubePoint ops vertex) =
      table.valueAt vertex := by
  exact evaluateCoordinates_fieldCoordinates_eq_valueAt
    ops laws table vertex

/-- The arbitrary-point equality polynomial specializes to the paper's
Boolean-vertex equality weight. -/
theorem pointEquality_toCubePoint_eq_equalityWeight
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables : Nat}
    (vertex : BooleanVertex variables)
    (right : CubePoint Field variables) :
    pointEquality ops (VertexEncoding.toCubePoint ops vertex) right =
      vertex.equalityWeight ops right := by
  rcases right with ⟨coordinates, dimension⟩
  induction vertex generalizing coordinates with
  | nil =>
      have coordinatesEmpty : coordinates = [] :=
        List.eq_nil_of_length_eq_zero dimension
      subst coordinates
      rfl
  | @cons tailVariables coordinate tail inductionHypothesis =>
      cases coordinates with
      | nil => simp at dimension
      | cons right rights =>
          have tailDimension : rights.length = tailVariables :=
            Nat.succ.inj dimension
          have tailEquality :
              pointEqualityCoordinates ops
                  (VertexEncoding.fieldCoordinates ops tail) rights =
                BooleanVertex.equalityWeightCoordinates ops tail rights := by
            simpa only [pointEquality, VertexEncoding.toCubePoint,
              BooleanVertex.equalityWeight] using
              inductionHypothesis rights tailDimension
          cases coordinate
          · simp only [pointEquality, VertexEncoding.toCubePoint,
              BooleanVertex.toCubePoint_coordinates,
              BooleanVertex.fieldCoordinates,
              pointEqualityCoordinates,
              BooleanVertex.equalityWeight,
              BooleanVertex.equalityWeightCoordinates,
              equalityFactor_zero ops laws]
            rw [tailEquality]
          · simp only [pointEquality, VertexEncoding.toCubePoint,
              BooleanVertex.toCubePoint_coordinates,
              BooleanVertex.fieldCoordinates,
              pointEqualityCoordinates,
              BooleanVertex.equalityWeight,
              BooleanVertex.equalityWeightCoordinates,
              equalityFactor_one ops laws]
            rw [tailEquality]

/-- Arbitrary-point `F(x,C)`. -/
def ccsAtPoint
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (gamma : Field)
    (point : CubePoint Field shape.cubeVariables) : Field :=
  SignedJointIdentity.sumMap ops (canonicalFinIndices shape.freshCount) fun source =>
    SignedJointIdentity.gammaTerm ops gamma source.val
      ((data.ccs source).evaluate ops point)

/-- Arbitrary-point unshifted `NC(x,C)`. -/
def normAtPoint
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (gamma : Field)
    (point : CubePoint Field shape.cubeVariables) : Field :=
  SignedJointIdentity.sumMap ops (canonicalFinIndices shape.sourceCount) fun source =>
    SignedJointIdentity.gammaTerm ops gamma source.val
      ((data.norm source).evaluate ops point)

/-- Arbitrary-point unshifted carried `Eval(x,C)`. -/
def carriedAtPoint
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (gamma : Field)
    (point : CubePoint Field shape.cubeVariables) : Field :=
  ops.mul (pointEquality ops point data.priorPoint) <|
    SignedJointIdentity.sumMap ops (canonicalCarriedCoordinates shape) fun coordinate =>
      SignedJointIdentity.gammaTerm ops gamma coordinate.localGammaExponent
        ((data.carriedImage coordinate).evaluate ops point)

/-- Literal joint polynomial at an arbitrary field point. -/
def qAtPoint
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (point : CubePoint Field shape.cubeVariables) : Field :=
  ops.add
    (ops.mul (pointEquality ops point alpha)
      (ops.add
        (ccsAtPoint ops data gamma point)
        (SignedJointIdentity.gammaTerm ops gamma shape.freshCount
          (normAtPoint ops data gamma point))))
    (SignedJointIdentity.gammaTerm ops gamma shape.carriedEvaluationOffset
      (carriedAtPoint ops data gamma point))

private theorem sumMap_evaluate_toCubePoint_eq_valueAt
    {Field : Type uField}
    {Index : Type}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables : Nat}
    (indices : List Index)
    (tables : Index -> BooleanTable Field variables)
    (weights : Index -> Field)
    (vertex : BooleanVertex variables) :
    SignedJointIdentity.sumMap ops indices (fun index =>
        ops.mul (weights index)
          ((tables index).evaluate ops
            (VertexEncoding.toCubePoint ops vertex))) =
      SignedJointIdentity.sumMap ops indices (fun index =>
        ops.mul (weights index) ((tables index).valueAt vertex)) := by
  unfold SignedJointIdentity.sumMap
  congr 1
  apply List.map_congr_left
  intro index _
  rw [evaluate_toCubePoint_eq_valueAt ops laws]

theorem ccsAtPoint_toCubePoint_eq_ccsAt
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (gamma : Field)
    (vertex : BooleanVertex shape.cubeVariables) :
    ccsAtPoint ops data gamma (VertexEncoding.toCubePoint ops vertex) =
      SignedJointIdentity.ccsAt ops data gamma vertex := by
  unfold ccsAtPoint SignedJointIdentity.ccsAt SignedJointIdentity.gammaTerm
  exact sumMap_evaluate_toCubePoint_eq_valueAt ops laws _ data.ccs
    (fun source => TargetPolynomial.power ops.toOps gamma source.val) vertex

theorem normAtPoint_toCubePoint_eq_normAt
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (gamma : Field)
    (vertex : BooleanVertex shape.cubeVariables) :
    normAtPoint ops data gamma (VertexEncoding.toCubePoint ops vertex) =
      SignedJointIdentity.normAt ops data gamma vertex := by
  unfold normAtPoint SignedJointIdentity.normAt SignedJointIdentity.gammaTerm
  exact sumMap_evaluate_toCubePoint_eq_valueAt ops laws _ data.norm
    (fun source => TargetPolynomial.power ops.toOps gamma source.val) vertex

theorem carriedAtPoint_toCubePoint_eq_carriedAt
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (gamma : Field)
    (vertex : BooleanVertex shape.cubeVariables) :
    carriedAtPoint ops data gamma (VertexEncoding.toCubePoint ops vertex) =
      SignedJointIdentity.carriedAt ops data gamma vertex := by
  unfold carriedAtPoint SignedJointIdentity.carriedAt
  rw [pointEquality_toCubePoint_eq_equalityWeight ops laws]
  congr 1
  unfold SignedJointIdentity.gammaTerm
  exact sumMap_evaluate_toCubePoint_eq_valueAt ops laws _ data.carriedImage
    (fun coordinate =>
      TargetPolynomial.power ops.toOps gamma coordinate.localGammaExponent)
    vertex

/-- The arbitrary-point joint polynomial is not an MLE of a sampled truth
table: it is the literal product expression, and it agrees with the prior
Boolean definition exactly on every cube vertex. -/
theorem qAtPoint_toCubePoint_eq_qAt
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (vertex : BooleanVertex shape.cubeVariables) :
    qAtPoint ops data alpha gamma (VertexEncoding.toCubePoint ops vertex) =
      SignedJointIdentity.qAt ops data alpha gamma vertex := by
  unfold qAtPoint SignedJointIdentity.qAt
  rw [pointEquality_toCubePoint_eq_equalityWeight ops laws,
    ccsAtPoint_toCubePoint_eq_ccsAt ops laws,
    normAtPoint_toCubePoint_eq_normAt ops laws,
    carriedAtPoint_toCubePoint_eq_carriedAt ops laws]

/-- Total coordinate-list form of the joint polynomial. The dependent branch
constructs the sole dimension proof; a wrong-length list fails closed. -/
def jointPolynomial
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (coordinates : List Field) : Field :=
  if dimension : coordinates.length = shape.cubeVariables then
    qAtPoint ops data alpha gamma ⟨coordinates, dimension⟩
  else
    ops.zero

/-- Verifier-computed terminal from the same explicit joint polynomial. -/
def verifierTerminal
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (challenges : List Field) : Field :=
  jointPolynomial ops data alpha gamma challenges

theorem jointPolynomial_fieldCoordinates_eq_qAt
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (vertex : BooleanVertex shape.cubeVariables) :
    jointPolynomial ops data alpha gamma
        (VertexEncoding.fieldCoordinates ops vertex) =
      SignedJointIdentity.qAt ops data alpha gamma vertex := by
  unfold jointPolynomial
  rw [dif_pos (VertexEncoding.fieldCoordinates_length ops vertex)]
  exact qAtPoint_toCubePoint_eq_qAt ops laws data alpha gamma vertex

private theorem finiteSum_append
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) :
    forall left right : List Field,
      BooleanTable.finiteSum ops (left ++ right) =
        ops.add (BooleanTable.finiteSum ops left)
          (BooleanTable.finiteSum ops right)
  | [], right => by
      simp [BooleanTable.finiteSum, laws.zero_add]
  | value :: values, right => by
      simp only [List.cons_append, BooleanTable.finiteSum]
      rw [finiteSum_append ops laws values right]
      exact (laws.add_assoc _ _ _).symm

/-- Generic enumeration theorem connecting the recursive completion sum to
the sole canonical Boolean-vertex order. -/
theorem sumCompletions_eq_vertexSum
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (polynomial : List Field -> Field)
    (fixed : List Field)
    (variables : Nat) :
    SumCheck.Finite.HypercubeTruth.sumCompletions ops.toOps polynomial
        fixed variables =
      BooleanTable.finiteSum ops
        ((BooleanVertex.all variables).map (fun vertex =>
          polynomial
            (fixed ++ VertexEncoding.fieldCoordinates ops vertex))) := by
  induction variables generalizing fixed with
  | zero =>
      simp [SumCheck.Finite.HypercubeTruth.sumCompletions.eq_def,
        BooleanVertex.all, VertexEncoding.fieldCoordinates,
        BooleanTable.finiteSum, laws.add_zero]
  | succ variables inductionHypothesis =>
      change ops.add
          (SumCheck.Finite.HypercubeTruth.sumCompletions ops.toOps polynomial
            (fixed ++ [ops.zero]) variables)
          (SumCheck.Finite.HypercubeTruth.sumCompletions ops.toOps polynomial
            (fixed ++ [ops.one]) variables) = _
      rw [inductionHypothesis (fixed ++ [ops.zero]),
        inductionHypothesis (fixed ++ [ops.one])]
      rw [BooleanVertex.all, List.map_append,
        finiteSum_append ops laws]
      simp only [List.map_map]
      congr 1
      · apply congrArg (BooleanTable.finiteSum ops)
        apply List.map_congr_left
        intro vertex _
        simp [VertexEncoding.fieldCoordinates, List.append_assoc]
      · apply congrArg (BooleanTable.finiteSum ops)
        apply List.map_congr_left
        intro vertex _
        simp [VertexEncoding.fieldCoordinates, List.append_assoc]

/-- The Boolean initial sum of the total joint polynomial is exactly the
independently defined paper `sum_x Q`. -/
theorem sumCompletions_jointPolynomial_eq_summedQ
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) :
    SumCheck.Finite.HypercubeTruth.sumCompletions ops.toOps
        (jointPolynomial ops data alpha gamma) [] shape.cubeVariables =
      SignedJointIdentity.summedQ ops data alpha gamma := by
  rw [sumCompletions_eq_vertexSum ops laws]
  unfold SignedJointIdentity.summedQ SignedJointIdentity.sumMap
  simp only [List.nil_append]
  congr 1
  apply List.map_congr_left
  intro vertex _
  exact jointPolynomial_fieldCoordinates_eq_qAt
    ops laws data alpha gamma vertex

/-- Canonical semantic ghost object. The initial sum is the independently
named paper sum; expected rounds are derived from the same total `Q`. -/
def canonicalGhosts
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (challenges : List Field) : SumCheck.Finite.SemanticGhosts Field where
  trueInitial := SignedJointIdentity.summedQ ops data alpha gamma
  expected := SumCheck.Finite.HypercubeTruth.expectedPolynomials ops.toOps
    (jointPolynomial ops data alpha gamma) challenges

/-- Exact truth path for the joint paper polynomial. The only shape premises
are explicit: one challenge per cube variable and one message per challenge.
The degree of each derived expected round remains a separate theorem. -/
theorem canonicalGhosts_honest
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : SignedJointIdentity.JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (maxDegree challengeSetSize : Nat)
    (initial : Field)
    (challenges : List Field)
    (certificate : SumCheck.Finite.Certificate Field)
    (challengeLength : challenges.length = shape.cubeVariables)
    (sameLength : certificate.rounds.length = challenges.length) :
    (canonicalGhosts ops data alpha gamma challenges).Honest ops.toOps
      maxDegree challengeSetSize initial challenges
      (verifierTerminal ops data alpha gamma challenges) certificate := by
  have generic :=
    SumCheck.Finite.HypercubeTruth.semanticGhosts_honest ops.toOps
      (jointPolynomial ops data alpha gamma) maxDegree challengeSetSize initial
      challenges certificate sameLength
  simpa only [canonicalGhosts, verifierTerminal,
      SumCheck.Finite.HypercubeTruth.semanticGhosts,
      SumCheck.Finite.HypercubeTruth.expectedPolynomials,
      challengeLength,
      sumCompletions_jointPolynomial_eq_summedQ ops laws data alpha gamma]
    using generic

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckTruthPath
