import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Fintype.Card
import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Refinement.ProductSum

/-!
Owns: model-level elimination of one fixed 18-term polynomial-evaluation
chunk and the exact two-limb Karatsuba batching used by a Pi_RLC projection
identity.

Does not own: extraction of production Rust rows, trace non-escape, retained
column decoding, or the generated concrete conformance certificate.

Emits constraints: no.

Authority boundary: coefficients, powers, operands, `W`, and final outputs are
supplied authoritative values. These theorems preserve their exact equations;
they do not establish transcript authority or permit production row removal by
themselves.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `sourceEvaluationChunk18_iff_compact` | `identities.*.evaluations.*` | Eighteen explicit products are equivalent to direct evaluation terms | Shared authoritative coefficients and beta powers | No - this file alone is model-level |
| `sourceQuadraticEvaluation_iff_compact` | `identities.*.evaluations.*` | Both limbs, including coefficient zero only in limb zero, preserve the full evaluation | Shared authoritative coefficients and powers | No - concrete trace bridge required |
| `sourceKaratsuba_iff_direct` | `identities.*.k_products.*` | Exact `p`, `q`, `r`, `W`, and sign rows equal one quadratic-extension product | Commutative ring and supplied `W` | No - this file alone is model-level |
| `sourceFinalLimbBatch_iff_direct` | `identities.*.final_limb_checks` | All input products and q-times-Phi may be substituted in both final limbs | Finite input family and exact Karatsuba rows | No - this file alone is model-level |
| `sourceProjectionIdentity_iff_compact` | complete `identities.*` | All evaluation materializations and terminal Karatsuba rows compose to the direct compact relation | Finite evaluation/input families and shared values | No - concrete trace bridge required |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.ProjectionBatchingRefinement

open scoped BigOperators
open ProductSumRefinement

universe u v

/-- The exact selective-lowering chunk width used by Rust `chunks(18)`. -/
abbrev EvaluationChunkIndex := Fin maxProductTerms

theorem evaluationChunkIndex_card : Fintype.card EvaluationChunkIndex = 18 := by
  simp [EvaluationChunkIndex, maxProductTerms]

/-- Source evaluation shape with one explicit product column per term. -/
def SourceEvaluationChunk18
    {K : Type u} [CommSemiring K]
    (coefficients powers : EvaluationChunkIndex → K) (result : K) : Prop :=
  SourceProductSum (fun _ => 1) coefficients powers result

/-- Compact evaluation shape after substituting all eighteen products. -/
def CompactEvaluationChunk18
    {K : Type u} [CommSemiring K]
    (coefficients powers : EvaluationChunkIndex → K) (result : K) : Prop :=
  DirectProductSum (fun _ => 1) coefficients powers result

/-- Soundness and completeness of eliminating one exact 18-term evaluation
product chunk. -/
theorem sourceEvaluationChunk18_iff_compact
    {K : Type u} [CommSemiring K]
    (coefficients powers : EvaluationChunkIndex → K) (result : K) :
    SourceEvaluationChunk18 coefficients powers result ↔
      CompactEvaluationChunk18 coefficients powers result :=
  sourceProductSum_iff_direct (fun _ => 1) coefficients powers result

/-- Every compact 18-term evaluation has the canonical source-product
reconstruction. -/
theorem sourceEvaluationChunk18_complete
    {K : Type u} [CommSemiring K]
    (coefficients powers : EvaluationChunkIndex → K) (result : K)
    (compact : CompactEvaluationChunk18 coefficients powers result) :
    SourceEvaluationChunk18 coefficients powers result :=
  (sourceEvaluationChunk18_iff_compact coefficients powers result).mpr compact

/-! ### Complete two-limb evaluations -/

/-- Source semantics of one evaluation limb. `coefficientZero` is the
unmultiplied constant term: production passes coefficient zero in limb zero
and zero in limb one. -/
def SourceEvaluationLimb
    {Term : Type u} {K : Type v}
    [CommRing K] [Fintype Term]
    (coefficientZero : K)
    (coefficients powers : Term → K)
    (result : K) : Prop :=
  SourceProductSum (fun _ => 1) coefficients powers (result - coefficientZero)

/-- Direct semantics after eliminating every explicit evaluation product. -/
def CompactEvaluationLimb
    {Term : Type u} {K : Type v}
    [CommRing K] [Fintype Term]
    (coefficientZero : K)
    (coefficients powers : Term → K)
    (result : K) : Prop :=
  DirectProductSum (fun _ => 1) coefficients powers (result - coefficientZero)

/-- Full evaluation-product substitution preserves the constant term and
result exactly. -/
theorem sourceEvaluationLimb_iff_compact
    {Term : Type u} {K : Type v}
    [CommRing K] [Fintype Term]
    (coefficientZero : K)
    (coefficients powers : Term → K)
    (result : K) :
    SourceEvaluationLimb coefficientZero coefficients powers result ↔
      CompactEvaluationLimb coefficientZero coefficients powers result :=
  sourceProductSum_iff_direct (fun _ => 1) coefficients powers
    (result - coefficientZero)

/-- Two base-ring limbs representing one quadratic-extension value. -/
structure QuadraticValue (K : Type u) where
  c0 : K
  c1 : K
deriving DecidableEq

/-- Source semantics of both limbs of one quadratic-extension polynomial
evaluation. The second limb has no coefficient-zero contribution. -/
def SourceQuadraticEvaluation
    {Term : Type u} {K : Type v}
    [CommRing K] [Fintype Term]
    (coefficientZero : K)
    (coefficients powers0 powers1 : Term → K)
    (output : QuadraticValue K) : Prop :=
  SourceEvaluationLimb coefficientZero coefficients powers0 output.c0 ∧
    SourceEvaluationLimb 0 coefficients powers1 output.c1

/-- Compact two-limb evaluation after direct product substitution. -/
def CompactQuadraticEvaluation
    {Term : Type u} {K : Type v}
    [CommRing K] [Fintype Term]
    (coefficientZero : K)
    (coefficients powers0 powers1 : Term → K)
    (output : QuadraticValue K) : Prop :=
  CompactEvaluationLimb coefficientZero coefficients powers0 output.c0 ∧
    CompactEvaluationLimb 0 coefficients powers1 output.c1

/-- Soundness and completeness of the exact coefficient-zero schedule across
both evaluation limbs. -/
theorem sourceQuadraticEvaluation_iff_compact
    {Term : Type u} {K : Type v}
    [CommRing K] [Fintype Term]
    (coefficientZero : K)
    (coefficients powers0 powers1 : Term → K)
    (output : QuadraticValue K) :
    SourceQuadraticEvaluation coefficientZero coefficients powers0 powers1 output ↔
      CompactQuadraticEvaluation coefficientZero coefficients powers0 powers1 output := by
  unfold SourceQuadraticEvaluation CompactQuadraticEvaluation
  constructor
  · rintro ⟨limb0, limb1⟩
    exact ⟨
      (sourceEvaluationLimb_iff_compact coefficientZero coefficients powers0 output.c0).mp limb0,
      (sourceEvaluationLimb_iff_compact 0 coefficients powers1 output.c1).mp limb1⟩
  · rintro ⟨limb0, limb1⟩
    exact ⟨
      (sourceEvaluationLimb_iff_compact coefficientZero coefficients powers0 output.c0).mpr limb0,
      (sourceEvaluationLimb_iff_compact 0 coefficients powers1 output.c1).mpr limb1⟩

theorem quadraticValue_ext
    {K : Type u} {left right : QuadraticValue K}
    (c0 : left.c0 = right.c0) (c1 : left.c1 = right.c1) :
    left = right := by
  cases left
  cases right
  simp_all

/-- Direct multiplication in `K[T]/(T^2-W)`. -/
def quadraticMul
    {K : Type u} [CommRing K]
    (w : K) (left right : QuadraticValue K) : QuadraticValue K where
  c0 := left.c0 * right.c0 + w * (left.c1 * right.c1)
  c1 := left.c0 * right.c1 + left.c1 * right.c0

/-- Exact five-row Karatsuba semantics emitted by Rust:
`p=a0*b0`, `q=a1*b1`, `r=(a0+a1)*(b0+b1)`,
`out0=p+W*q`, and `out1=r-p-q`. -/
def SourceKaratsuba
    {K : Type u} [CommRing K]
    (w : K) (left right output : QuadraticValue K) : Prop :=
  ∃ p q r : K,
    p = left.c0 * right.c0 ∧
    q = left.c1 * right.c1 ∧
    r = (left.c0 + left.c1) * (right.c0 + right.c1) ∧
    output.c0 = p + w * q ∧
    output.c1 = r - p - q

/-- The exact Karatsuba `W` and sign schedule is equivalent to direct
quadratic-extension multiplication. -/
theorem sourceKaratsuba_iff_direct
    {K : Type u} [CommRing K]
    (w : K) (left right output : QuadraticValue K) :
    SourceKaratsuba w left right output ↔
      output = quadraticMul w left right := by
  constructor
  · rintro ⟨p, q, r, hp, hq, hr, h0, h1⟩
    apply quadraticValue_ext
    · simpa [quadraticMul, hp, hq] using h0
    · simp only [quadraticMul]
      rw [h1, hp, hq, hr]
      ring
  · intro direct
    refine ⟨
      left.c0 * right.c0,
      left.c1 * right.c1,
      (left.c0 + left.c1) * (right.c0 + right.c1),
      rfl,
      rfl,
      rfl,
      ?_,
      ?_⟩
    · simpa [quadraticMul] using congrArg QuadraticValue.c0 direct
    · rw [direct]
      simp only [quadraticMul]
      ring

/-- Source projection identity with all Karatsuba intermediates materialized
and both final extension-limb equations explicit. -/
def SourceFinalLimbBatch
    {Input : Type v} [Fintype Input]
    {K : Type u} [CommRing K]
    (w : K)
    (left right : Input → QuadraticValue K)
    (quotient phi output : QuadraticValue K) : Prop :=
  ∃ products : Input → QuadraticValue K,
    ∃ quotientPhi : QuadraticValue K,
      (∀ input, SourceKaratsuba w (left input) (right input) (products input)) ∧
      SourceKaratsuba w quotient phi quotientPhi ∧
      (∑ input, (products input).c0) = quotientPhi.c0 + output.c0 ∧
      (∑ input, (products input).c1) = quotientPhi.c1 + output.c1

/-- The same two final limbs after direct substitution of every exact
Karatsuba product. -/
def DirectFinalLimbBatch
    {Input : Type v} [Fintype Input]
    {K : Type u} [CommRing K]
    (w : K)
    (left right : Input → QuadraticValue K)
    (quotient phi output : QuadraticValue K) : Prop :=
  (∑ input, (quadraticMul w (left input) (right input)).c0) =
      (quadraticMul w quotient phi).c0 + output.c0 ∧
  (∑ input, (quadraticMul w (left input) (right input)).c1) =
      (quadraticMul w quotient phi).c1 + output.c1

/-- Soundness and completeness of eliminating all exact Karatsuba
intermediates while preserving both final extension limbs. -/
theorem sourceFinalLimbBatch_iff_direct
    {Input : Type v} [Fintype Input]
    {K : Type u} [CommRing K]
    (w : K)
    (left right : Input → QuadraticValue K)
    (quotient phi output : QuadraticValue K) :
    SourceFinalLimbBatch w left right quotient phi output ↔
      DirectFinalLimbBatch w left right quotient phi output := by
  constructor
  · rintro ⟨products, quotientPhi, productsSource, quotientPhiSource, limb0, limb1⟩
    have productsDirect : products = fun input => quadraticMul w (left input) (right input) := by
      funext input
      exact (sourceKaratsuba_iff_direct w (left input) (right input) (products input)).mp
        (productsSource input)
    have quotientPhiDirect : quotientPhi = quadraticMul w quotient phi :=
      (sourceKaratsuba_iff_direct w quotient phi quotientPhi).mp quotientPhiSource
    subst products
    subst quotientPhi
    exact ⟨limb0, limb1⟩
  · rintro ⟨limb0, limb1⟩
    refine ⟨
      (fun input => quadraticMul w (left input) (right input)),
      quadraticMul w quotient phi,
      ?_,
      ?_,
      limb0,
      limb1⟩
    · intro input
      exact (sourceKaratsuba_iff_direct w (left input) (right input)
        (quadraticMul w (left input) (right input))).mpr rfl
    · exact (sourceKaratsuba_iff_direct w quotient phi
        (quadraticMul w quotient phi)).mpr rfl

/-! ### Complete projection identity -/

/-- Model of all source rows owned by one projection identity: every
two-limb polynomial evaluation plus all materialized Karatsuba products and
the two final limb checks. -/
def SourceProjectionIdentity
    {Evaluation Input : Type u} (Term : Evaluation → Type u)
    {K : Type v} [CommRing K]
    [Fintype Evaluation] [Fintype Input]
    [∀ evaluation, Fintype (Term evaluation)]
    (coefficientZero : Evaluation → K)
    (coefficients powers0 powers1 : ∀ evaluation, Term evaluation → K)
    (evaluations : Evaluation → QuadraticValue K)
    (w : K)
    (left right : Input → QuadraticValue K)
    (quotient phi output : QuadraticValue K) : Prop :=
  (∀ evaluation,
    SourceQuadraticEvaluation
      (coefficientZero evaluation)
      (coefficients evaluation)
      (powers0 evaluation)
      (powers1 evaluation)
      (evaluations evaluation)) ∧
  SourceFinalLimbBatch w left right quotient phi output

/-- Model of the emitted compact relation: direct evaluation products and
direct two-limb terminal products over the same authoritative values. -/
def CompactProjectionIdentity
    {Evaluation Input : Type u} (Term : Evaluation → Type u)
    {K : Type v} [CommRing K]
    [Fintype Evaluation] [Fintype Input]
    [∀ evaluation, Fintype (Term evaluation)]
    (coefficientZero : Evaluation → K)
    (coefficients powers0 powers1 : ∀ evaluation, Term evaluation → K)
    (evaluations : Evaluation → QuadraticValue K)
    (w : K)
    (left right : Input → QuadraticValue K)
    (quotient phi output : QuadraticValue K) : Prop :=
  (∀ evaluation,
    CompactQuadraticEvaluation
      (coefficientZero evaluation)
      (coefficients evaluation)
      (powers0 evaluation)
      (powers1 evaluation)
      (evaluations evaluation)) ∧
  DirectFinalLimbBatch w left right quotient phi output

/-- The complete source identity and compact identity accept exactly the same
authoritative values. -/
theorem sourceProjectionIdentity_iff_compact
    {Evaluation Input : Type u} (Term : Evaluation → Type u)
    {K : Type v} [CommRing K]
    [Fintype Evaluation] [Fintype Input]
    [∀ evaluation, Fintype (Term evaluation)]
    (coefficientZero : Evaluation → K)
    (coefficients powers0 powers1 : ∀ evaluation, Term evaluation → K)
    (evaluations : Evaluation → QuadraticValue K)
    (w : K)
    (left right : Input → QuadraticValue K)
    (quotient phi output : QuadraticValue K) :
    SourceProjectionIdentity Term coefficientZero coefficients powers0 powers1
        evaluations w left right quotient phi output ↔
      CompactProjectionIdentity Term coefficientZero coefficients powers0 powers1
        evaluations w left right quotient phi output := by
  constructor
  · rintro ⟨sourceEvaluations, sourceFinal⟩
    refine ⟨?_, (sourceFinalLimbBatch_iff_direct w left right quotient phi output).mp sourceFinal⟩
    intro evaluation
    exact (sourceQuadraticEvaluation_iff_compact
      (coefficientZero evaluation)
      (coefficients evaluation)
      (powers0 evaluation)
      (powers1 evaluation)
      (evaluations evaluation)).mp (sourceEvaluations evaluation)
  · rintro ⟨compactEvaluations, compactFinal⟩
    refine ⟨?_, (sourceFinalLimbBatch_iff_direct w left right quotient phi output).mpr compactFinal⟩
    intro evaluation
    exact (sourceQuadraticEvaluation_iff_compact
      (coefficientZero evaluation)
      (coefficients evaluation)
      (powers0 evaluation)
      (powers1 evaluation)
      (evaluations evaluation)).mpr (compactEvaluations evaluation)

/-! ### Exact production chunk shapes -/

/-- Sums of the exact three `chunks(18)` groups emitted for every polynomial
evaluation limb. -/
def threeChunkSums
    {K : Type u} [AddCommMonoid K]
    (values : List K) : List K :=
  [ (values.take maxProductTerms).sum
  , ((values.drop maxProductTerms).take maxProductTerms).sum
  , ((values.drop maxProductTerms).drop maxProductTerms).sum
  ]

/-- Sums of the exact two `chunks(18)` groups emitted for a terminal limb. -/
def twoChunkSums
    {K : Type u} [AddCommMonoid K]
    (values : List K) : List K :=
  [ (values.take maxProductTerms).sum
  , (values.drop maxProductTerms).sum
  ]

def threeChunkLengths {K : Type u} (values : List K) : List Nat :=
  [ (values.take maxProductTerms).length
  , ((values.drop maxProductTerms).take maxProductTerms).length
  , ((values.drop maxProductTerms).drop maxProductTerms).length
  ]

def twoChunkLengths {K : Type u} (values : List K) : List Nat :=
  [ (values.take maxProductTerms).length
  , (values.drop maxProductTerms).length
  ]

theorem takeDrop_sum
    {K : Type u} [AddCommMonoid K]
    (count : Nat) (values : List K) :
    (values.take count).sum + (values.drop count).sum = values.sum := by
  rw [← List.sum_append, List.take_append_drop]

/-- Three carry rows telescope to the original ordered product list. -/
theorem carryChain_threeChunks_iff_sum
    {K : Type u} [CommRing K]
    (result : K) (values : List K) :
    CarryChain 0 (threeChunkSums values) result ↔ result = values.sum := by
  unfold threeChunkSums
  rw [carryChain_zero_iff_direct]
  simp only [List.sum_cons, List.sum_nil, add_zero]
  rw [takeDrop_sum, takeDrop_sum]

/-- Two terminal carry rows telescope to the original ordered factor list. -/
theorem carryChain_twoChunks_iff_sum
    {K : Type u} [CommRing K]
    (result : K) (values : List K) :
    CarryChain 0 (twoChunkSums values) result ↔ result = values.sum := by
  unfold twoChunkSums
  rw [carryChain_zero_iff_direct]
  simp only [List.sum_cons, List.sum_nil, add_zero]
  rw [takeDrop_sum]

/-- Ordered scalar products of one polynomial-evaluation limb. -/
def evaluationProductValues
    {count : Nat} {K : Type u} [CommRing K]
    (coefficients powers : Fin count → K) : List K :=
  List.ofFn fun term => coefficients term * powers term

theorem evaluationProductValues_sum
    {count : Nat} {K : Type u} [CommRing K]
    (coefficients powers : Fin count → K) :
    (evaluationProductValues coefficients powers).sum =
      ∑ term, coefficients term * powers term := by
  induction count with
  | zero => simp [evaluationProductValues]
  | succ count inductionHypothesis =>
      simp only [evaluationProductValues, List.ofFn_succ, List.sum_cons,
        Fin.sum_univ_succ]
      exact congrArg
        (fun value => coefficients 0 * powers 0 + value)
        (inductionHypothesis
          (fun term => coefficients term.succ)
          (fun term => powers term.succ))

theorem evaluation53_chunk_lengths
    {K : Type u} [CommRing K]
    (coefficients powers : Fin 53 → K) :
    threeChunkLengths (evaluationProductValues coefficients powers) =
      [18, 18, 17] := by
  simp [threeChunkLengths, evaluationProductValues, maxProductTerms]

theorem evaluation52_chunk_lengths
    {K : Type u} [CommRing K]
    (coefficients powers : Fin 52 → K) :
    threeChunkLengths (evaluationProductValues coefficients powers) =
      [18, 18, 16] := by
  simp [threeChunkLengths, evaluationProductValues, maxProductTerms]

/-- Exact three-row emitted relation for one evaluation limb. -/
def EmittedThreeChunkEvaluationLimb
    {count : Nat} {K : Type u} [CommRing K]
    (coefficientZero : K)
    (coefficients powers : Fin count → K)
    (result : K) : Prop :=
  CarryChain 0
    (threeChunkSums (evaluationProductValues coefficients powers))
    (result - coefficientZero)

/-- Source product columns and the exact three-row carry chain accept the same
evaluation limb. -/
theorem sourceEvaluationLimb_iff_emittedThreeChunks
    {count : Nat} {K : Type u} [CommRing K]
    (coefficientZero : K)
    (coefficients powers : Fin count → K)
    (result : K) :
    SourceEvaluationLimb coefficientZero coefficients powers result ↔
      EmittedThreeChunkEvaluationLimb coefficientZero coefficients powers result := by
  rw [sourceEvaluationLimb_iff_compact]
  unfold CompactEvaluationLimb DirectProductSum EmittedThreeChunkEvaluationLimb
  rw [carryChain_threeChunks_iff_sum, evaluationProductValues_sum]
  simp only [one_mul]

/-- The 54-coefficient production evaluation has 53 multiplied terms split
exactly as `18 + 18 + 17`. -/
structure Evaluation53 (K : Type u) where
  coefficientZero : K
  coefficients : Fin 53 → K
  powers0 : Fin 53 → K
  powers1 : Fin 53 → K
  output : QuadraticValue K

/-- The 53-coefficient quotient evaluation has 52 multiplied terms split
exactly as `18 + 18 + 16`. -/
structure Evaluation52 (K : Type u) where
  coefficientZero : K
  coefficients : Fin 52 → K
  powers0 : Fin 52 → K
  powers1 : Fin 52 → K
  output : QuadraticValue K

def SourceEvaluation53
    {K : Type u} [CommRing K]
    (evaluation : Evaluation53 K) : Prop :=
  SourceQuadraticEvaluation evaluation.coefficientZero evaluation.coefficients
    evaluation.powers0 evaluation.powers1 evaluation.output

def EmittedEvaluation53
    {K : Type u} [CommRing K]
    (evaluation : Evaluation53 K) : Prop :=
  EmittedThreeChunkEvaluationLimb evaluation.coefficientZero
      evaluation.coefficients evaluation.powers0 evaluation.output.c0 ∧
    EmittedThreeChunkEvaluationLimb 0
      evaluation.coefficients evaluation.powers1 evaluation.output.c1

theorem sourceEvaluation53_iff_emitted
    {K : Type u} [CommRing K]
    (evaluation : Evaluation53 K) :
    SourceEvaluation53 evaluation ↔ EmittedEvaluation53 evaluation := by
  unfold SourceEvaluation53 SourceQuadraticEvaluation EmittedEvaluation53
  constructor
  · rintro ⟨limb0, limb1⟩
    exact ⟨
      (sourceEvaluationLimb_iff_emittedThreeChunks
        evaluation.coefficientZero evaluation.coefficients evaluation.powers0
        evaluation.output.c0).mp limb0,
      (sourceEvaluationLimb_iff_emittedThreeChunks
        0 evaluation.coefficients evaluation.powers1 evaluation.output.c1).mp limb1⟩
  · rintro ⟨limb0, limb1⟩
    exact ⟨
      (sourceEvaluationLimb_iff_emittedThreeChunks
        evaluation.coefficientZero evaluation.coefficients evaluation.powers0
        evaluation.output.c0).mpr limb0,
      (sourceEvaluationLimb_iff_emittedThreeChunks
        0 evaluation.coefficients evaluation.powers1 evaluation.output.c1).mpr limb1⟩

def SourceEvaluation52
    {K : Type u} [CommRing K]
    (evaluation : Evaluation52 K) : Prop :=
  SourceQuadraticEvaluation evaluation.coefficientZero evaluation.coefficients
    evaluation.powers0 evaluation.powers1 evaluation.output

def EmittedEvaluation52
    {K : Type u} [CommRing K]
    (evaluation : Evaluation52 K) : Prop :=
  EmittedThreeChunkEvaluationLimb evaluation.coefficientZero
      evaluation.coefficients evaluation.powers0 evaluation.output.c0 ∧
    EmittedThreeChunkEvaluationLimb 0
      evaluation.coefficients evaluation.powers1 evaluation.output.c1

theorem sourceEvaluation52_iff_emitted
    {K : Type u} [CommRing K]
    (evaluation : Evaluation52 K) :
    SourceEvaluation52 evaluation ↔ EmittedEvaluation52 evaluation := by
  unfold SourceEvaluation52 SourceQuadraticEvaluation EmittedEvaluation52
  constructor
  · rintro ⟨limb0, limb1⟩
    exact ⟨
      (sourceEvaluationLimb_iff_emittedThreeChunks
        evaluation.coefficientZero evaluation.coefficients evaluation.powers0
        evaluation.output.c0).mp limb0,
      (sourceEvaluationLimb_iff_emittedThreeChunks
        0 evaluation.coefficients evaluation.powers1 evaluation.output.c1).mp limb1⟩
  · rintro ⟨limb0, limb1⟩
    exact ⟨
      (sourceEvaluationLimb_iff_emittedThreeChunks
        evaluation.coefficientZero evaluation.coefficients evaluation.powers0
        evaluation.output.c0).mpr limb0,
      (sourceEvaluationLimb_iff_emittedThreeChunks
        0 evaluation.coefficients evaluation.powers1 evaluation.output.c1).mpr limb1⟩

/-! ### Exact terminal operand schedule -/

def productTerm
    {K : Type u} [CommRing K]
    (coefficient left right : K) : BoundedProductTerm K where
  coefficient := coefficient
  left := left
  right := right

/-- Exact ordered 32-factor limb-zero schedule: two factors for each of 15
rho/input pairs, followed by the two negative quotient/Phi factors. -/
def terminalLimb0Terms
    {K : Type u} [CommRing K]
    (w : K)
    (left right : Fin 15 → QuadraticValue K)
    (quotient phi : QuadraticValue K) : List (BoundedProductTerm K) :=
  (List.ofFn fun input =>
    [ productTerm 1 (left input).c0 (right input).c0
    , productTerm w (left input).c1 (right input).c1
    ]).flatten ++
  [ productTerm (-1) quotient.c0 phi.c0
  , productTerm (-w) quotient.c1 phi.c1
  ]

/-- Exact ordered 32-factor limb-one schedule. -/
def terminalLimb1Terms
    {K : Type u} [CommRing K]
    (left right : Fin 15 → QuadraticValue K)
    (quotient phi : QuadraticValue K) : List (BoundedProductTerm K) :=
  (List.ofFn fun input =>
    [ productTerm 1 (left input).c0 (right input).c1
    , productTerm 1 (left input).c1 (right input).c0
    ]).flatten ++
  [ productTerm (-1) quotient.c0 phi.c1
  , productTerm (-1) quotient.c1 phi.c0
  ]

theorem terminalLimb0Terms_length
    {K : Type u} [CommRing K]
    (w : K)
    (left right : Fin 15 → QuadraticValue K)
    (quotient phi : QuadraticValue K) :
    (terminalLimb0Terms w left right quotient phi).length = 32 := by
  simp [terminalLimb0Terms]

theorem terminalLimb1Terms_length
    {K : Type u} [CommRing K]
    (left right : Fin 15 → QuadraticValue K)
    (quotient phi : QuadraticValue K) :
    (terminalLimb1Terms left right quotient phi).length = 32 := by
  simp [terminalLimb1Terms]

theorem terminalLimb0_chunk_lengths
    {K : Type u} [CommRing K]
    (w : K)
    (left right : Fin 15 → QuadraticValue K)
    (quotient phi : QuadraticValue K) :
    twoChunkLengths (terminalLimb0Terms w left right quotient phi) =
      [18, 14] := by
  simp [twoChunkLengths, terminalLimb0Terms, maxProductTerms]

theorem terminalLimb1_chunk_lengths
    {K : Type u} [CommRing K]
    (left right : Fin 15 → QuadraticValue K)
    (quotient phi : QuadraticValue K) :
    twoChunkLengths (terminalLimb1Terms left right quotient phi) =
      [18, 14] := by
  simp [twoChunkLengths, terminalLimb1Terms, maxProductTerms]

theorem terminalLimb0Terms_sum
    {K : Type u} [CommRing K]
    (w : K)
    (left right : Fin 15 → QuadraticValue K)
    (quotient phi : QuadraticValue K) :
    ((terminalLimb0Terms w left right quotient phi).map
      BoundedProductTerm.value).sum =
      (∑ input, (quadraticMul w (left input) (right input)).c0) -
        (quadraticMul w quotient phi).c0 := by
  simp [terminalLimb0Terms, productTerm, BoundedProductTerm.value, quadraticMul,
    Fin.sum_univ_succ]
  ring

theorem terminalLimb1Terms_sum
    {K : Type u} [CommRing K]
    (w : K)
    (left right : Fin 15 → QuadraticValue K)
    (quotient phi : QuadraticValue K) :
    ((terminalLimb1Terms left right quotient phi).map
      BoundedProductTerm.value).sum =
      (∑ input, (quadraticMul w (left input) (right input)).c1) -
        (quadraticMul w quotient phi).c1 := by
  simp [terminalLimb1Terms, productTerm, BoundedProductTerm.value, quadraticMul,
    Fin.sum_univ_succ]
  ring

/-- Exact four-row emitted terminal relation: two carry rows for each of two
32-factor limbs, with chunk sizes `18 + 14`. -/
def EmittedFinalLimbBatch
    {K : Type u} [CommRing K]
    (w : K)
    (left right : Fin 15 → QuadraticValue K)
    (quotient phi output : QuadraticValue K) : Prop :=
  CarryChain 0
      (twoChunkSums ((terminalLimb0Terms w left right quotient phi).map
        BoundedProductTerm.value))
      output.c0 ∧
    CarryChain 0
      (twoChunkSums ((terminalLimb1Terms left right quotient phi).map
        BoundedProductTerm.value))
      output.c1

theorem directFinalLimbBatch_iff_emitted
    {K : Type u} [CommRing K]
    (w : K)
    (left right : Fin 15 → QuadraticValue K)
    (quotient phi output : QuadraticValue K) :
    DirectFinalLimbBatch w left right quotient phi output ↔
      EmittedFinalLimbBatch w left right quotient phi output := by
  unfold EmittedFinalLimbBatch DirectFinalLimbBatch
  rw [carryChain_twoChunks_iff_sum, carryChain_twoChunks_iff_sum]
  rw [terminalLimb0Terms_sum, terminalLimb1Terms_sum]
  constructor
  · rintro ⟨limb0, limb1⟩
    constructor
    · rw [limb0]
      ring
    · rw [limb1]
      ring
  · rintro ⟨limb0, limb1⟩
    constructor
    · rw [limb0]
      ring
    · rw [limb1]
      ring

theorem sourceFinalLimbBatch_iff_emitted
    {K : Type u} [CommRing K]
    (w : K)
    (left right : Fin 15 → QuadraticValue K)
    (quotient phi output : QuadraticValue K) :
    SourceFinalLimbBatch w left right quotient phi output ↔
      EmittedFinalLimbBatch w left right quotient phi output :=
  (sourceFinalLimbBatch_iff_direct w left right quotient phi output).trans
    (directFinalLimbBatch_iff_emitted w left right quotient phi output)

/-! ### Exact complete 1916-row to 106-row identity model -/

/-- Exact source semantics bound to the production operand schedule. -/
def SourceExactProjectionIdentity
    {K : Type u} [CommRing K]
    (w : K)
    (rhoEvaluations : Fin 15 → QuadraticValue K)
    (inputEvaluations : Fin 15 → Evaluation53 K)
    (outputEvaluation : Evaluation53 K)
    (quotientEvaluation : Evaluation52 K)
    (phi : QuadraticValue K) : Prop :=
  (∀ input, SourceEvaluation53 (inputEvaluations input)) ∧
    SourceEvaluation53 outputEvaluation ∧
    SourceEvaluation52 quotientEvaluation ∧
    SourceFinalLimbBatch w
      rhoEvaluations
      (fun input => (inputEvaluations input).output)
      quotientEvaluation.output
      phi
      outputEvaluation.output

/-- Exact emitted semantics: 102 evaluation carry rows plus four terminal
carry rows over the same retained and external operands. -/
def EmittedExactProjectionIdentity
    {K : Type u} [CommRing K]
    (w : K)
    (rhoEvaluations : Fin 15 → QuadraticValue K)
    (inputEvaluations : Fin 15 → Evaluation53 K)
    (outputEvaluation : Evaluation53 K)
    (quotientEvaluation : Evaluation52 K)
    (phi : QuadraticValue K) : Prop :=
  (∀ input, EmittedEvaluation53 (inputEvaluations input)) ∧
    EmittedEvaluation53 outputEvaluation ∧
    EmittedEvaluation52 quotientEvaluation ∧
    EmittedFinalLimbBatch w
      rhoEvaluations
      (fun input => (inputEvaluations input).output)
      quotientEvaluation.output
      phi
      outputEvaluation.output

/-- Exact source-row semantics and the emitted 106-row relation are sound and
complete. The statement binds every terminal operand to the corresponding
retained evaluation output or external rho/Phi value. -/
theorem sourceExactProjectionIdentity_iff_emitted
    {K : Type u} [CommRing K]
    (w : K)
    (rhoEvaluations : Fin 15 → QuadraticValue K)
    (inputEvaluations : Fin 15 → Evaluation53 K)
    (outputEvaluation : Evaluation53 K)
    (quotientEvaluation : Evaluation52 K)
    (phi : QuadraticValue K) :
    SourceExactProjectionIdentity w rhoEvaluations inputEvaluations
        outputEvaluation quotientEvaluation phi ↔
      EmittedExactProjectionIdentity w rhoEvaluations inputEvaluations
        outputEvaluation quotientEvaluation phi := by
  constructor
  · rintro ⟨inputs, output, quotient, final⟩
    exact ⟨
      fun input => (sourceEvaluation53_iff_emitted (inputEvaluations input)).mp (inputs input),
      (sourceEvaluation53_iff_emitted outputEvaluation).mp output,
      (sourceEvaluation52_iff_emitted quotientEvaluation).mp quotient,
      (sourceFinalLimbBatch_iff_emitted w rhoEvaluations
        (fun input => (inputEvaluations input).output)
        quotientEvaluation.output phi outputEvaluation.output).mp final⟩
  · rintro ⟨inputs, output, quotient, final⟩
    exact ⟨
      fun input => (sourceEvaluation53_iff_emitted (inputEvaluations input)).mpr (inputs input),
      (sourceEvaluation53_iff_emitted outputEvaluation).mpr output,
      (sourceEvaluation52_iff_emitted quotientEvaluation).mpr quotient,
      (sourceFinalLimbBatch_iff_emitted w rhoEvaluations
        (fun input => (inputEvaluations input).output)
        quotientEvaluation.output phi outputEvaluation.output).mpr final⟩

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.ProjectionBatchingRefinement
