import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.Core

/-!
Compact coefficient certificates for the bounded selective `y_zcol` rows.

Owns: a proof-free normal-form projection for all thirteen compact ports and
the generic kernel bridge from equality of those shapes to the typed rewrite
or retained coefficient obligation.

Does not own: concrete certificate partitions, artifact decoding, row
satisfaction, selector truth, source authority, security events, or permission
to remove rows.

Emits constraints: no.

Assurance tier: checked coefficient data with a kernel-derived semantic lift.

Both certificate obligations are retained for this bounded profile.

| Stable stage path | Exact obligation | Authority class | Artifact owner | Lean owner | Multiplicity |
|---|---|---|---|---|---|
| `pi_rlc.y_zcol.selective.coefficients.shape` | normalized A/B/C port coefficients retain exact column and field residue data | computed | decoded compact rows plus source provenance | `normalizedLinearShape` | thirteen ports per selected row |
| `pi_rlc.y_zcol.selective.coefficients.lift` | proof-free shape equality implies typed sparse-form equivalence | derived | compact certificate | `rewriteCoefficientsMatch_of_shape_check_true`, `retainedCoefficientsMatch_of_shape_check_true` | once per checked pair |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.Coefficients

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

/-- Proof-free sparse term: physical column and canonical field residue. -/
abbrev CompactLinearTerm := Nat × Nat

def compactLinearTerm
    (term : Materialized.LinearForm.Term) : CompactLinearTerm :=
  (term.1, term.2.val)

/-- Canonical sparse form used by native certificates. No decoded proof field
survives this projection. -/
def normalizedLinearShape
    (terms : List Materialized.LinearForm.Term) : List CompactLinearTerm :=
  (Materialized.LinearForm.normalize terms).map compactLinearTerm

/-- One proof-free certificate record. Both arrays have exactly thirteen
entries by construction. -/
structure CoefficientMatchShape where
  actual : Array (List CompactLinearTerm)
  expected : Array (List CompactLinearTerm)
deriving DecidableEq, Repr

def coefficientMatchShapeCheck (shape : CoefficientMatchShape) : Bool :=
  decide (shape.actual = shape.expected)

def coefficientMatchShapesCheck
    (shapes : List CoefficientMatchShape) : Bool :=
  shapes.all coefficientMatchShapeCheck

def rewritePairCoefficientShape (pair : RewritePair) :
    CoefficientMatchShape :=
  { actual := Array.ofFn fun port : Fin 13 =>
      normalizedLinearShape
        (Materialized.LinearForm.portTerms (pair.1.port port))
    expected := Array.ofFn fun port : Fin 13 =>
      normalizedLinearShape (rewritePortLinearForm pair.2 port) }

def retainedPairCoefficientShape (pair : RetainedPair) :
    CoefficientMatchShape :=
  { actual := Array.ofFn fun port : Fin 13 =>
      normalizedLinearShape
        (Materialized.LinearForm.portTerms (pair.1.port port))
    expected := Array.ofFn fun port : Fin 13 =>
      normalizedLinearShape (retainedPortLinearForm pair.2 port) }

private theorem compactLinearTerm_injective :
    Function.Injective compactLinearTerm := by
  intro left right equal
  apply Prod.ext
  · exact congrArg (fun value : CompactLinearTerm => value.1) equal
  · apply Fin.ext
    exact congrArg (fun value : CompactLinearTerm => value.2) equal

private theorem compactLinearTerms_injective :
    Function.Injective (List.map compactLinearTerm) := by
  intro left right equal
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => rfl
      | cons head tail => simp at equal
  | cons leftHead leftTail inductionHypothesis =>
      cases right with
      | nil => simp at equal
      | cons rightHead rightTail =>
          simp only [List.map_cons, List.cons.injEq] at equal
          have headEqual := compactLinearTerm_injective equal.1
          have tailEqual := inductionHypothesis equal.2
          subst rightHead
          subst rightTail
          rfl

private theorem equivalent_of_normalized_shape_eq
    {left right : List Materialized.LinearForm.Term}
    (equal : normalizedLinearShape left = normalizedLinearShape right) :
    Materialized.LinearForm.Equivalent left right := by
  unfold normalizedLinearShape at equal
  unfold Materialized.LinearForm.Equivalent
  exact compactLinearTerms_injective equal

private theorem arrayPortShape_eq
    {actual expected : Fin 13 → List Materialized.LinearForm.Term}
    (equal :
      Array.ofFn (fun port => normalizedLinearShape (actual port)) =
        Array.ofFn (fun port => normalizedLinearShape (expected port)))
    (port : Fin 13) :
    normalizedLinearShape (actual port) =
      normalizedLinearShape (expected port) := by
  have atPort := congrArg
    (fun values : Array (List CompactLinearTerm) => values[port.val]?) equal
  simpa using atPort

/-- Generic kernel bridge for one rewrite pair. The checked value contains
only arrays, lists, and natural numbers. -/
theorem rewriteCoefficientsMatch_of_shape_check_true
    {pair : RewritePair}
    (checked :
      coefficientMatchShapeCheck (rewritePairCoefficientShape pair) = true) :
    RewriteCoefficientsMatch pair.1 pair.2 := by
  have arraysEqual :
      (rewritePairCoefficientShape pair).actual =
        (rewritePairCoefficientShape pair).expected := by
    apply of_decide_eq_true
    simpa only [coefficientMatchShapeCheck] using checked
  intro port
  apply equivalent_of_normalized_shape_eq
  apply arrayPortShape_eq (port := port)
  simpa only [rewritePairCoefficientShape] using arraysEqual

/-- Generic kernel bridge for one physically retained pair. -/
theorem retainedCoefficientsMatch_of_shape_check_true
    {pair : RetainedPair}
    (checked :
      coefficientMatchShapeCheck (retainedPairCoefficientShape pair) = true) :
    RetainedCoefficientsMatch pair.1 pair.2 := by
  have arraysEqual :
      (retainedPairCoefficientShape pair).actual =
        (retainedPairCoefficientShape pair).expected := by
    apply of_decide_eq_true
    simpa only [coefficientMatchShapeCheck] using checked
  intro port
  apply equivalent_of_normalized_shape_eq
  apply arrayPortShape_eq (port := port)
  simpa only [retainedPairCoefficientShape] using arraysEqual

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.Coefficients
