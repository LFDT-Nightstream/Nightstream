import Nightstream.SuperNeo.SumCheck.FixedPolynomial

/-!
Canonical raw messages for fixed-width SumCheck polynomials.

Owns: removing only redundant high zero coefficients from one fixed-width
polynomial, while retaining `[zero]` as the unique raw representation of the
zero polynomial.

Does not own: prover messages, challenges, SumCheck acceptance, protocol
polynomials, generated artifacts, Rust, R1CS, or costs.

The canonicalizer is defined independently of any implementation artifact.
Its source-decomposition theorem makes the only permitted transformation
explicit: a prefix is retained exactly and a suffix of zero coefficients is
removed.
-/

namespace Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial

universe uField

/-- Remove the maximal redundant high-zero suffix. The empty result records
that every source coefficient was zero; `canonicalMessage` maps precisely that
case to the singleton zero polynomial. -/
private def trimHighZeros
    {Field : Type uField}
    [DecidableEq Field]
    (zero : Field) : List Field -> List Field
  | [] => []
  | coefficient :: coefficients =>
      let higher := trimHighZeros zero coefficients
      if higher = [] ∧ coefficient = zero then [] else coefficient :: higher

private theorem source_eq_trimHighZeros_append
    {Field : Type uField}
    [DecidableEq Field]
    (zero : Field) :
    forall coefficients : List Field,
      exists padding,
        coefficients =
          trimHighZeros zero coefficients ++ List.replicate padding zero
  | [] => by
      exact ⟨0, rfl⟩
  | coefficient :: coefficients => by
      obtain ⟨padding, source⟩ :=
        source_eq_trimHighZeros_append zero coefficients
      by_cases erased :
          trimHighZeros zero coefficients = [] ∧ coefficient = zero
      · refine ⟨padding + 1, ?_⟩
        rcases erased with ⟨higherEmpty, coefficientZero⟩
        subst coefficient
        rw [trimHighZeros, if_pos ⟨higherEmpty, rfl⟩]
        simp only [List.nil_append]
        calc
          zero :: coefficients =
              zero :: (trimHighZeros zero coefficients ++
                List.replicate padding zero) :=
            congrArg (List.cons zero) source
          _ = List.replicate (padding + 1) zero := by
            rw [higherEmpty]
            simp [List.replicate_succ]
      · refine ⟨padding, ?_⟩
        rw [trimHighZeros, if_neg erased]
        simpa only [List.cons_append] using
          congrArg (List.cons coefficient) source

private theorem trimHighZeros_getLast?_ne_zero
    {Field : Type uField}
    [DecidableEq Field]
    (zero : Field) :
    forall coefficients : List Field,
      trimHighZeros zero coefficients ≠ [] ->
        (trimHighZeros zero coefficients).getLast? ≠ some zero
  | [], nonempty => by
      exact (nonempty rfl).elim
  | coefficient :: coefficients, nonempty => by
      by_cases erased :
          trimHighZeros zero coefficients = [] ∧ coefficient = zero
      · simp [trimHighZeros, erased] at nonempty
      · by_cases higherEmpty : trimHighZeros zero coefficients = []
        · have coefficientNonzero : coefficient ≠ zero := by
            intro coefficientZero
            exact erased ⟨higherEmpty, coefficientZero⟩
          simp [trimHighZeros, higherEmpty, coefficientNonzero]
        · have higherLast :=
            trimHighZeros_getLast?_ne_zero zero coefficients higherEmpty
          cases higher : trimHighZeros zero coefficients with
          | nil => exact (higherEmpty higher).elim
          | cons head tail =>
              simpa [trimHighZeros, erased, higher] using higherLast

private theorem evaluateCoefficients_trimHighZeros
    {Field : Type uField}
    [DecidableEq Field]
    (ops : Ops Field)
    (laws : Laws ops)
    (point : Field) :
    forall coefficients : List Field,
      Message.evaluateCoefficients ops point
          (trimHighZeros ops.zero coefficients) =
        Message.evaluateCoefficients ops point coefficients
  | [] => rfl
  | coefficient :: coefficients => by
      have higherEvaluation :=
        evaluateCoefficients_trimHighZeros ops laws point coefficients
      by_cases erased :
          trimHighZeros ops.zero coefficients = [] ∧ coefficient = ops.zero
      · rcases erased with ⟨higherEmpty, coefficientZero⟩
        simp only [trimHighZeros, higherEmpty, coefficientZero, and_self,
          if_pos, Message.evaluateCoefficients]
        rw [← higherEvaluation, higherEmpty]
        simp [Message.evaluateCoefficients, laws.mul_zero, laws.zero_add]
      · simp only [trimHighZeros, if_neg erased,
          Message.evaluateCoefficients]
        rw [higherEvaluation]

/-- Canonical verifier-visible form of one fixed-width polynomial. No low or
interior coefficient is changed; only its redundant high-zero suffix is
discarded. -/
def canonicalMessage
    {Field : Type uField}
    {degree : Nat}
    [DecidableEq Field]
    (ops : Ops Field)
    (polynomial : FixedPolynomial Field degree) : Message Field :=
  let trimmed := trimHighZeros ops.zero polynomial.coefficients
  if trimmed = [] then Message.zero ops else ⟨trimmed⟩

/-- The source fixed-width coefficients are exactly the canonical prefix plus
the removed high-zero suffix. This rules out reordering, interior deletion,
or coefficient rewriting. -/
theorem canonicalMessage_coefficients_eq_prefix_zero_padding
    {Field : Type uField}
    {degree : Nat}
    [DecidableEq Field]
    (ops : Ops Field)
    (polynomial : FixedPolynomial Field degree) :
    exists padding,
      polynomial.coefficients =
        (canonicalMessage ops polynomial).coefficients ++
          List.replicate padding ops.zero := by
  obtain ⟨padding, source⟩ :=
    source_eq_trimHighZeros_append ops.zero polynomial.coefficients
  by_cases empty : trimHighZeros ops.zero polynomial.coefficients = []
  · have sourceNonempty : polynomial.coefficients ≠ [] := by
      intro sourceEmpty
      have width := polynomial.coefficients_length
      rw [sourceEmpty] at width
      simp at width
    cases padding with
    | zero =>
        exact (sourceNonempty (by simpa [empty] using source)).elim
    | succ padding =>
        refine ⟨padding, ?_⟩
        rw [source, empty]
        simp [canonicalMessage, empty, Message.zero, List.replicate_succ]
  · exact ⟨padding, by simpa [canonicalMessage, empty] using source⟩

/-- Canonicalization preserves evaluation at every point. -/
theorem canonicalMessage_evaluate
    {Field : Type uField}
    {degree : Nat}
    [DecidableEq Field]
    (ops : Ops Field)
    (laws : Laws ops)
    (polynomial : FixedPolynomial Field degree)
    (point : Field) :
    (canonicalMessage ops polynomial).evaluate ops point =
      polynomial.evaluate ops point := by
  change
    Message.evaluateCoefficients ops point
        (canonicalMessage ops polynomial).coefficients =
      Message.evaluateCoefficients ops point polynomial.coefficients
  by_cases empty : trimHighZeros ops.zero polynomial.coefficients = []
  · calc
      Message.evaluateCoefficients ops point
          (canonicalMessage ops polynomial).coefficients = ops.zero := by
        simp [canonicalMessage, empty, Message.zero,
          Message.evaluateCoefficients, laws.mul_zero, laws.zero_add]
      _ = Message.evaluateCoefficients ops point
          (trimHighZeros ops.zero polynomial.coefficients) := by
        rw [empty]
        rfl
      _ = Message.evaluateCoefficients ops point polynomial.coefficients :=
        evaluateCoefficients_trimHighZeros ops laws point
          polynomial.coefficients
  · simpa [canonicalMessage, empty] using
      evaluateCoefficients_trimHighZeros ops laws point
        polynomial.coefficients

/-- The output is accepted by the raw verifier's canonical-shape check. -/
theorem canonicalMessage_canonical
    {Field : Type uField}
    {degree : Nat}
    [DecidableEq Field]
    (ops : Ops Field)
    (polynomial : FixedPolynomial Field degree) :
    Message.Canonical ops (canonicalMessage ops polynomial) := by
  by_cases empty : trimHighZeros ops.zero polynomial.coefficients = []
  · simpa [canonicalMessage, empty] using Message.zero_canonical ops
  · refine ⟨?_, ?_⟩
    · simp [canonicalMessage, empty]
    · by_cases singleton :
          (trimHighZeros ops.zero polynomial.coefficients).length = 1
      · exact Or.inl (by simpa [canonicalMessage, empty] using singleton)
      · exact Or.inr (by
          simpa [canonicalMessage, empty] using
            trimHighZeros_getLast?_ne_zero ops.zero
              polynomial.coefficients empty)

/-- Canonicalization can only reduce the verifier-derived degree bound. -/
theorem canonicalMessage_degreeUpperBound_le
    {Field : Type uField}
    {degree : Nat}
    [DecidableEq Field]
    (ops : Ops Field)
    (polynomial : FixedPolynomial Field degree) :
    (canonicalMessage ops polynomial).degreeUpperBound ≤ degree := by
  obtain ⟨padding, source⟩ :=
    canonicalMessage_coefficients_eq_prefix_zero_padding ops polynomial
  have lengthLe :
      (canonicalMessage ops polynomial).coefficients.length ≤
        polynomial.coefficients.length := by
    rw [source, List.length_append]
    omega
  rw [polynomial.coefficients_length] at lengthLe
  unfold Message.degreeUpperBound
  omega

end Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial
