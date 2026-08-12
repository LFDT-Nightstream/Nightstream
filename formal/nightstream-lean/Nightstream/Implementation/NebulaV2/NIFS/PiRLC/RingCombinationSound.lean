import Nightstream.Implementation.NebulaV2.NIFS.PiRLC.RingCombinationRows
import Nightstream.Implementation.NebulaV2.NIFS.PiDEC.LinearCombination
import Nightstream.Implementation.R1CS.Canonical.KLinear
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelector
import Nightstream.Implementation.R1CS.Correspondence.Projection.Phi81.Carrier
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Product

/-!
Contract: typed soundness of one exact V2 PiRLC ring-combination occurrence.

This file decodes canonical row wires independently. It proves that the
selected symbol is the canonical centered Goldilocks coefficient, that every
owned product column is the corresponding field product, and that each output
coefficient is the exact reduction modulo `X^54 + X^27 + 1` of the sum of all
15 challenge-times-source products.

The theorem assumes only canonical wires, the constant-one wire, the exact
row schedule, and the five-symbol range. The transcript and selector layer
derives that range and fixes the symbols in the complete verifier bridge.

This file does not own transcript derivation, placement of all 110 ring
families, or equality with the final paper PiRLC parent claim.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationRows
open Nightstream.Implementation.NebulaV2.ProductPiDecLinearCombination
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81StrongSet
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm

/-- Canonical field view of one wire. -/
def wireField (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (column : Nat) : F :=
  fieldAt assignment canonical column

/-- Field view of a sparse linear combination. -/
def termsField (assignment : Nat -> Nat) (terms : LinComb) : F :=
  ⟨lcEval assignment terms, by
    unfold lcEval
    simpa [goldilocksP, goldilocksModulus] using
      Nat.mod_lt
        (terms.foldl
          (fun accumulated term => accumulated + term.2 * assignment term.1) 0)
        (by decide : 0 < goldilocksP)⟩

@[simp] theorem termsField_nil (assignment : Nat -> Nat) :
    termsField assignment [] = 0 := by
  rfl

theorem termsField_cons
    (assignment : Nat -> Nat) (term : Nat × Nat) (rest : LinComb) :
    termsField assignment (term :: rest) =
      (⟨term.2 % goldilocksP, Nat.mod_lt _ (by decide)⟩ : F) *
          ⟨assignment term.1 % goldilocksP, Nat.mod_lt _ (by decide)⟩ +
        termsField assignment rest := by
  apply Fin.ext
  simp only [termsField, Fin.val_add, Fin.val_mul]
  simp [lcEval_eq_rawSum, rawSum_cons, Nat.add_mod, Nat.mul_mod,
    goldilocksP, goldilocksModulus]

theorem termsField_append
    (assignment : Nat -> Nat) (left right : LinComb) :
    termsField assignment (left ++ right) =
      termsField assignment left + termsField assignment right := by
  apply Fin.ext
  simp only [termsField, Fin.val_add]
  rw [lcEval_eq_rawSum, rawSum_append, Nat.add_mod]
  rw [← lcEval_eq_rawSum, ← lcEval_eq_rawSum]
  simp [goldilocksP, goldilocksModulus]

/-- Canonical head-first field sum over a list. -/
def fieldSum {Alpha : Type} : List Alpha -> (Alpha -> F) -> F
  | [], _ => 0
  | item :: rest, value => value item + fieldSum rest value

theorem termsField_flatMap
    {Alpha : Type} (assignment : Nat -> Nat) (items : List Alpha)
    (terms : Alpha -> LinComb) :
    termsField assignment (items.flatMap terms) =
      fieldSum items (fun item => termsField assignment (terms item)) := by
  induction items with
  | nil => rfl
  | cons item rest inductionHypothesis =>
      simp only [List.flatMap_cons, fieldSum]
      rw [termsField_append, inductionHypothesis]

theorem termsField_filterMap
    {Alpha : Type} (assignment : Nat -> Nat) (items : List Alpha)
    (term : Alpha -> Option (Nat × Nat)) :
    termsField assignment (items.filterMap term) =
      fieldSum items fun item =>
        match term item with
        | none => 0
        | some entry =>
            (⟨entry.2 % goldilocksP, Nat.mod_lt _ (by decide)⟩ : F) *
              ⟨assignment entry.1 % goldilocksP,
                Nat.mod_lt _ (by decide)⟩ := by
  induction items with
  | nil => rfl
  | cons item rest inductionHypothesis =>
      cases selected : term item with
      | none =>
          simp only [List.filterMap_cons, selected, fieldSum, Fin.zero_add]
          exact inductionHypothesis
      | some entry =>
          simp only [List.filterMap_cons, selected, fieldSum]
          rw [termsField_cons, inductionHypothesis]

/-- The physical symbol wire interpreted as a typed alphabet coefficient. -/
def symbolCoefficient
    (layout : Layout) (assignment : Nat -> Nat)
    (source : Source) (lane : Lane)
    (range : assignment (layout.challengeSymbol source lane) < 5) :
    Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.Coefficient :=
  ⟨assignment (layout.challengeSymbol source lane), range⟩

/-- Exact typed challenge ring decoded from the 54 symbol wires. -/
def challengeRing
    (layout : Layout) (assignment : Nat -> Nat)
    (range : forall source lane,
      assignment (layout.challengeSymbol source lane) < 5)
    (source : Source) : RingF :=
  fun lane => embedCoefficient (symbolCoefficient layout assignment source lane
    (range source lane))

/-- Exact typed source ring decoded from canonical input wires. -/
def inputRing
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (source : Source) : RingF :=
  fun lane => wireField assignment canonical (layout.input source lane)

/-- Exact typed output ring decoded from canonical output wires. -/
def outputRing
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) : RingF :=
  fun lane => wireField assignment canonical (layout.output lane)

/-- The row's centered linear form is the independent semantic alphabet
embedding. -/
theorem centeredChallenge_eq_embedCoefficient
    (layout : Layout) (assignment : Nat -> Nat)
    (one : assignment 0 = 1)
    (source : Source) (lane : Lane)
    (range : assignment (layout.challengeSymbol source lane) < 5) :
    termsField assignment (centeredChallenge layout source lane) =
      embedCoefficient
        (symbolCoefficient layout assignment source lane range) := by
  have evaluated :
      lcEval assignment (centeredChallenge layout source lane) =
        (assignment (layout.challengeSymbol source lane) +
          (goldilocksP - 2)) % goldilocksP := by
    simp [centeredChallenge, lcEval, one, Nat.add_comm]
  have embedded :=
    Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelector.embedCoefficient_val_eq_shift
      (symbolCoefficient layout assignment source lane range)
  apply Fin.ext
  change lcEval assignment (centeredChallenge layout source lane) = _
  rw [embedded]
  simpa [symbolCoefficient] using evaluated

/-- Each owned schoolbook column is the corresponding typed field product. -/
theorem productColumn_eq
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : forall source lane,
      assignment (layout.challengeSymbol source lane) < 5)
    (accepted : Accepted layout assignment)
    (source : Source) (left right : Lane) :
    wireField assignment canonical
        (productColumn layout source left right) =
      challengeRing layout assignment range source left *
        inputRing layout assignment canonical source right := by
  have equation := accepted.product source left right
  have centered := centeredChallenge_eq_embedCoefficient layout assignment one
    source left (range source left)
  apply Fin.ext
  simp only [wireField, fieldAt, challengeRing, inputRing, Fin.val_mul]
  rw [← equation]
  have centeredVal := congrArg Fin.val centered
  simp only [termsField] at centeredVal
  rw [centeredVal]
  simp [goldilocksP, goldilocksModulus]

/-- Canonical field image of one public row coefficient. -/
def coefficientField (coefficient : Nat) : F :=
  ⟨coefficient % goldilocksP, by
    simpa [goldilocksP, goldilocksModulus] using
      Nat.mod_lt coefficient (by decide : 0 < goldilocksP)⟩

theorem fieldSum_congr
    {Alpha : Type} (items : List Alpha) (left right : Alpha -> F)
    (equal : forall item, item ∈ items -> left item = right item) :
    fieldSum items left = fieldSum items right := by
  induction items with
  | nil => rfl
  | cons item rest inductionHypothesis =>
      simp only [fieldSum]
      rw [equal item (by simp)]
      apply congrArg (fun suffix => right item + suffix)
      exact inductionHypothesis (fun value member =>
        equal value (by simp [member]))

theorem fieldSum_scale
    {Alpha : Type} (items : List Alpha) (scalar : F) (value : Alpha -> F) :
    fieldSum items (fun item => scalar * value item) =
      scalar * fieldSum items value := by
  induction items with
  | nil => simp [fieldSum]
  | cons item rest inductionHypothesis =>
      simp only [fieldSum, inductionHypothesis]
      exact
        (Nightstream.Implementation.R1CS.ProjectionProgram.fmul_add
          scalar (value item) (fieldSum rest value)).symm

theorem fieldSum_map
    {Alpha Beta : Type} (items : List Alpha) (map : Alpha -> Beta)
    (value : Beta -> F) :
    fieldSum (items.map map) value =
      fieldSum items (fun item => value (map item)) := by
  induction items with
  | nil => rfl
  | cons item rest inductionHypothesis =>
      simp only [List.map_cons, fieldSum, inductionHypothesis]

theorem fieldSum_eq_productFieldListSum
    (items : List Nat) (value : Nat -> F) :
    fieldSum items value = Product.fieldListSum items value := by
  induction items with
  | nil => rfl
  | cons item rest inductionHypothesis =>
      simp only [fieldSum, Product.fieldListSum, inductionHypothesis]

/-- The local finite-index order is exactly the executable natural-index
order used by `rawMulCoeffF`. -/
theorem fieldSum_indices (value : Nat -> F) :
    fieldSum (indices laneCount) (fun index => value index.val) =
      Product.fieldListSum (List.range laneCount) value := by
  calc
    fieldSum (indices laneCount) (fun index => value index.val) =
        fieldSum ((indices laneCount).map Fin.val) value :=
      (fieldSum_map (indices laneCount) Fin.val value).symm
    _ = fieldSum (List.range laneCount) value := by
      rw [indices, List.map_coe_finRange_eq_range]
    _ = Product.fieldListSum (List.range laneCount) value :=
      fieldSum_eq_productFieldListSum _ _

/-- One active sparse term reads exactly the corresponding independent raw
schoolbook term. -/
theorem active_product_eq_rawTerm
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : forall source lane,
      assignment (layout.challengeSymbol source lane) < 5)
    (accepted : Accepted layout assignment)
    (source : Source) (degree : Nat) (left : Lane)
    (active : left.val <= degree /\ degree - left.val < laneCount) :
    wireField assignment canonical
        (productColumn layout source left
          ⟨degree - left.val, active.2⟩) =
      Product.rawTerm
        (challengeRing layout assignment range source)
        (inputRing layout assignment canonical source)
        degree left.val := by
  have semanticActive : Product.supportActive degree left.val := by
    exact ⟨active.1, by
      simpa [laneCount, ringDegree] using active.2⟩
  have rightLt : degree - left.val < ringDegree := semanticActive.2
  have leftLt : left.val < ringDegree := by
    simpa [laneCount, ringDegree] using left.isLt
  rw [productColumn_eq canonical one range accepted source left
    ⟨degree - left.val, active.2⟩]
  unfold Product.rawTerm ringFCoeff
  rw [if_pos semanticActive, dif_pos leftLt, dif_pos rightLt]
  congr 2

/-- The sparse terms for one unreduced coefficient are exactly a public
coefficient times the independent `rawMulCoeffF` result. -/
theorem rawTerms_field
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : forall source lane,
      assignment (layout.challengeSymbol source lane) < 5)
    (accepted : Accepted layout assignment)
    (source : Source) (degree coefficient : Nat) :
    termsField assignment (rawTerms layout source degree coefficient) =
      coefficientField coefficient *
        rawMulCoeffF
          (challengeRing layout assignment range source)
          (inputRing layout assignment canonical source) degree := by
  unfold rawTerms
  rw [termsField_filterMap]
  have pointwise : forall left : Lane, left ∈ indices laneCount ->
      (match
          (if active : left.val <= degree /\ degree - left.val < laneCount then
            some (productColumn layout source left
              ⟨degree - left.val, active.2⟩, coefficient)
          else none) with
        | none => 0
        | some entry =>
            (⟨entry.2 % goldilocksP, Nat.mod_lt _ (by decide)⟩ : F) *
              ⟨assignment entry.1 % goldilocksP,
                Nat.mod_lt _ (by decide)⟩) =
        coefficientField coefficient *
          Product.rawTerm
            (challengeRing layout assignment range source)
            (inputRing layout assignment canonical source)
            degree left.val := by
    intro left _
    by_cases active : left.val <= degree /\ degree - left.val < laneCount
    · simp only [dif_pos active]
      rw [show
          (⟨assignment
                (productColumn layout source left
                  ⟨degree - left.val, active.2⟩) % goldilocksP,
              Nat.mod_lt _ (by decide)⟩ : F) =
            wireField assignment canonical
              (productColumn layout source left
                ⟨degree - left.val, active.2⟩) by
        apply Fin.ext
        change
          assignment (productColumn layout source left
              ⟨degree - left.val, active.2⟩) % goldilocksP =
            assignment (productColumn layout source left
              ⟨degree - left.val, active.2⟩)
        exact Nat.mod_eq_of_lt (canonical _) ]
      rw [active_product_eq_rawTerm canonical one range accepted
        source degree left active]
      rfl
    · have semanticInactive :
          ¬ Product.supportActive degree left.val := by
        intro semanticActive
        apply active
        exact ⟨semanticActive.1, by
          simpa [laneCount, ringDegree] using semanticActive.2⟩
      simp [active, Product.rawTerm, semanticInactive]
  change fieldSum (indices laneCount) _ = _
  rw [fieldSum_congr (indices laneCount) _ _ pointwise]
  rw [fieldSum_scale, fieldSum_indices]
  apply congrArg (fun value => coefficientField coefficient * value)
  simpa [laneCount, ringDegree] using
    (Product.rawMulCoeffF_eq_fieldListSum
      (challengeRing layout assignment range source)
      (inputRing layout assignment canonical source) degree).symm

@[simp] theorem coefficientField_one : coefficientField 1 = (1 : F) := by
  apply Fin.ext
  simp [coefficientField, goldilocksP, goldilocksModulus]

theorem coefficientField_minusOne :
    coefficientField (goldilocksP - 1) = (-1 : F) := by
  apply Fin.ext
  rw [Fin.val_neg]
  simp [coefficientField, goldilocksP, goldilocksModulus]

theorem minusOne_mul (value : F) : (-1 : F) * value = -value := by
  calc
    (-1 : F) * value = -(1 * value) :=
      Lean.Grind.Fin.neg_mul 1 value
    _ = -value := by rw [Fin.one_mul]

/-- One source's output terms implement the exact Phi81 reduction. -/
theorem sourceOutputTerms_field
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : forall source lane,
      assignment (layout.challengeSymbol source lane) < 5)
    (accepted : Accepted layout assignment)
    (source : Source) (output : Lane) :
    termsField assignment (sourceOutputTerms layout source output) =
      ringFMul
        (challengeRing layout assignment range source)
        (inputRing layout assignment canonical source) output := by
  have base := rawTerms_field canonical one range accepted source output.val 1
  have folded := rawTerms_field canonical one range accepted source
    (foldedDegree output) (goldilocksP - 1)
  have twice := rawTerms_field canonical one range accepted source
    (output.val + 81) 1
  unfold sourceOutputTerms
  rw [termsField_append, termsField_append, base, folded,
    coefficientField_one, Fin.one_mul, coefficientField_minusOne,
    minusOne_mul]
  by_cases low : output.val < 27
  · by_cases hasTwice : output.val + 81 <= 106
    · simp [twiceEnabled, hasTwice, twice, ringFMul, foldedDegree, low,
        ringDegree, ringMiddleDegree, Fin.sub_eq_add_neg]
    · simp [twiceEnabled, hasTwice, ringFMul, foldedDegree, low,
        ringDegree, ringMiddleDegree, Fin.sub_eq_add_neg]
  · by_cases hasTwice : output.val + 81 <= 106
    · simp [twiceEnabled, hasTwice, twice, ringFMul, foldedDegree, low,
        ringDegree, ringMiddleDegree, Fin.sub_eq_add_neg]
    · simp [twiceEnabled, hasTwice, ringFMul, foldedDegree, low,
        ringDegree, ringMiddleDegree, Fin.sub_eq_add_neg]

/-- The local source order is the exact head-first source order used by the
independent typed product sum. -/
theorem fieldSum_finRange_productSum
    {count : Nat} (challenges inputs : Fin count -> RingF)
    (output : Fin ringDegree) :
    fieldSum (List.finRange count) (fun source =>
        ringFMul (challenges source) (inputs source) output) =
      Nightstream.Implementation.R1CS.ProjectionPhi81.productSum
        challenges inputs output := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.finRange_succ]
      simp only [fieldSum,
        Nightstream.Implementation.R1CS.ProjectionPhi81.productSum,
        ringFAdd]
      rw [fieldSum_map]
      rw [inductionHypothesis
        (fun source => challenges source.succ)
        (fun source => inputs source.succ)]

/-- The complete output linear form is the exact typed sum of all 15 source
ring products. -/
theorem outputTerms_field
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : forall source lane,
      assignment (layout.challengeSymbol source lane) < 5)
    (accepted : Accepted layout assignment)
    (output : Lane) :
    termsField assignment (outputTerms layout output) =
      Nightstream.Implementation.R1CS.ProjectionPhi81.productSum
        (challengeRing layout assignment range)
        (inputRing layout assignment canonical) output := by
  unfold outputTerms indices
  rw [termsField_flatMap]
  rw [fieldSum_congr (List.finRange sourceCount) _ _
    (fun source _ =>
      sourceOutputTerms_field canonical one range accepted source output)]
  exact fieldSum_finRange_productSum
    (challengeRing layout assignment range)
    (inputRing layout assignment canonical) output

/-- Satisfaction of one complete ring-family occurrence derives the exact
typed PiRLC ring combination. No verifier result is supplied by the caller. -/
theorem rows_imply_ring_combination
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : forall source lane,
      assignment (layout.challengeSymbol source lane) < 5)
    (satisfied : Satisfies (rows layout) assignment) :
    outputRing layout assignment canonical =
      Nightstream.Implementation.R1CS.ProjectionPhi81.productSum
        (challengeRing layout assignment range)
        (inputRing layout assignment canonical) := by
  let accepted := rows_sound canonical one satisfied
  funext output
  calc
    outputRing layout assignment canonical output =
        termsField assignment (outputTerms layout output) := by
      apply Fin.ext
      change assignment (layout.output output) =
        lcEval assignment (outputTerms layout output)
      exact (accepted.output output).symm
    _ = Nightstream.Implementation.R1CS.ProjectionPhi81.productSum
          (challengeRing layout assignment range)
          (inputRing layout assignment canonical) output :=
      outputTerms_field canonical one range accepted output

end Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationSound
