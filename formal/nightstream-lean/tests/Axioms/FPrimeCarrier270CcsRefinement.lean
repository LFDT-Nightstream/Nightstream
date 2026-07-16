import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the model-level five-ring F' CCS refinement.

| Protocol | Phase | Family | Guarded theorem |
|---|---|---|---|
| F' / CCS | aligned columns | matrix-vector image | `alignedMatrixVectorAt_eq` |
| F' / CCS | carrier completion | matrix-vector image | `carrierMatrixVectorAt_eq` |
| F' / CCS | numeric row | little-endian typed row | `carrierMatrixVectorAt_rowIndex_eq` |
| F' / CCS | lifted structure | identical polynomial | `liftStructure_constraintPolynomial` |
| F' / CCS | lifted structure | image vector | `matrixImagesAt_eq` |
| F' / CCS | lifted structure | residual | `residualAt_eq` |
| F' / CCS | relation membership | zero-set equivalence | `constraintSatisfied_iff` |
-/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement.alignedMatrixVectorAt_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement.alignedMatrixVectorAt_eq

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement.carrierMatrixVectorAt_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement.carrierMatrixVectorAt_eq

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement.carrierMatrixVectorAt_rowIndex_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement.carrierMatrixVectorAt_rowIndex_eq

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement.liftStructure_constraintPolynomial' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement.liftStructure_constraintPolynomial

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement.matrixImagesAt_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement.matrixImagesAt_eq

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement.residualAt_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement.residualAt_eq

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement.constraintSatisfied_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement.constraintSatisfied_iff
