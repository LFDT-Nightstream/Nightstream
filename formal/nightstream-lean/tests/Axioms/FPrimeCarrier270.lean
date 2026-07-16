import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the model-level five-ring F' carrier.

| Protocol | Phase | Family | Guarded theorem |
|---|---|---|---|
| F' / CCS | public shape | exact dimensions | `dimensions_exact` |
| F' / CCS | logical lowering | injective ownership / private shift | `alignedIndex_injective`, `assignment_private_shift` |
| F' / CCS | fresh assignment | fixed public padding | `assignment_fixedPublicPadding` |
| F' / CCS | public projection | legacy values plus fixed zeros | `projectPublicInput_exact` |
| F' / CCS | matrix source | aligned column inverse | `legacyIndex?_alignedIndex`, `legacyIndex?_eq_none_iff` |
| F' / CCS | matrix source | old / padding coefficient ownership | `alignedMatrix_at_alignedIndex`, `alignedMatrix_padding_zero` |
| F' / CCS | matrix source | completed carrier ownership | `carrierMatrix_at_alignedCarrierIndex`, `carrierMatrix_completion_zero` |
| F' / CCS | matrix source | numeric / Boolean row bijection | `rowIndex_rowVertex`, `rowVertex_rowIndex` |
| F' / CCS | matrix evaluation | tensor-weight refinement | `productionTensorWeight_eq_equalityWeight` |
| assurance | necessity | norm-valid tail value one | `tailOne_normBounded`, `omittingFixedPadding_enlargesFreshBoundary` |
-/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.dimensions_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.dimensions_exact

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.alignedIndex_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.alignedIndex_injective

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.assignment_private_shift' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.assignment_private_shift

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.assignment_fixedPublicPadding' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.assignment_fixedPublicPadding

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.projectPublicInput_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.projectPublicInput_exact

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.tailOne_normBounded' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.tailOne_normBounded

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.omittingFixedPadding_enlargesFreshBoundary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.omittingFixedPadding_enlargesFreshBoundary

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.rowIndex_lt_twoPow' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.rowIndex_lt_twoPow

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.rowIndex_rowVertex' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.rowIndex_rowVertex

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.rowVertex_rowIndex' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.rowVertex_rowIndex

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.productionTensorWeight_eq_equalityWeight' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.productionTensorWeight_eq_equalityWeight

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap.legacyIndex?_alignedIndex' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap.legacyIndex?_alignedIndex

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap.legacyIndex?_eq_none_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap.legacyIndex?_eq_none_iff

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap.alignedMatrix_at_alignedIndex' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap.alignedMatrix_at_alignedIndex

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap.alignedMatrix_padding_zero' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap.alignedMatrix_padding_zero

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap.carrierMatrix_at_alignedCarrierIndex' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap.carrierMatrix_at_alignedCarrierIndex

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap.carrierMatrix_completion_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap.carrierMatrix_completion_zero
