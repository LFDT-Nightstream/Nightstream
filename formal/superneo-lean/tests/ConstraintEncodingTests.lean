import SuperNeo.FPrimeRecursiveVerifier.ConstraintEncoding.ResidualPairFamilies

/-! Theorem-shape and trust-surface regressions for residual pair families. -/

namespace tests.ConstraintEncoding

open SuperNeo
open SuperNeo.FPrimeRecursiveVerifier.ConstraintEncoding
open ResidualPairFamilies

example {left right : F} :
    ResidualPairHolds left right ↔ left = 0 ∧ right = 0 :=
  residualPairHolds_iff

example (left right : OneProductInput) :
    OneProductPairHolds left right ↔
      OneProductHolds left ∧ OneProductHolds right :=
  oneProductPairHolds_iff left right

example (left right : F) :
    CenteredUnitPairHolds left right ↔
      IsCenteredUnit left ∧ IsCenteredUnit right :=
  centeredUnitPairHolds_iff left right

example (coordinates : List Nat) :
    BooleanRows.scheduledCoordinates (BooleanRows.schedule coordinates) =
      coordinates :=
  familySchedule_order_exact coordinates

example (row : BooleanRows.Row Nat) :
    selectorGatedDegree oneProductResidualDegree row ≤ 5 ∧
      selectorGatedDegree centeredUnitResidualDegree row ≤ 7 :=
  ⟨oneProduct_selectorGatedDegree_le_five row,
    centeredUnit_selectorGatedDegree_le_seven row⟩

example := oneProduct_pairRow_is_necessary
example := oneProduct_oddTailRow_is_necessary
example := centeredUnit_pairRow_is_necessary
example := centeredUnit_oddTailRow_is_necessary

/-- info: 'SuperNeo.FPrimeRecursiveVerifier.ConstraintEncoding.ResidualPairFamilies.residualPairHolds_iff' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms residualPairHolds_iff

/-- info: 'SuperNeo.FPrimeRecursiveVerifier.ConstraintEncoding.ResidualPairFamilies.centeredUnitResidual_eq_zero_iff' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms centeredUnitResidual_eq_zero_iff

/-- info: 'SuperNeo.FPrimeRecursiveVerifier.ConstraintEncoding.ResidualPairFamilies.oneProduct_pairRow_is_necessary' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms oneProduct_pairRow_is_necessary

/-- info: 'SuperNeo.FPrimeRecursiveVerifier.ConstraintEncoding.ResidualPairFamilies.centeredUnit_oddTailRow_is_necessary' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms centeredUnit_oddTailRow_is_necessary

end tests.ConstraintEncoding
