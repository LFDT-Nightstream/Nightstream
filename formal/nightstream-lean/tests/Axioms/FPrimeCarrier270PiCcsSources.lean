import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the model-level F' Split-NC source adapter.

| Protocol | Phase | Family | Guarded theorem |
|---|---|---|---|
| `Pi_CCS` | source shape | complete carrier | `semanticShape_carrierWidth` |
| `Pi_CCS` | matrix source | aligned / completed matrix | `data_matrixSource_matrix` |
| `Pi_CCS` | fresh source | exact completion | `data_freshAssignment_eq` |
| `Pi_CCS` | running source | joint source partition | `data_assignment_runningIndex_eq` |
| FE | fresh CCS | matrix images | `freshMatrixImagesAt_eq` |
| FE | fresh CCS | residual | `freshResidualAt_eq` |
| FE | fresh CCS | batch truth | `freshTruth_iff_legacy` |
-/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources.semanticShape_carrierWidth' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources.semanticShape_carrierWidth

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources.Inputs.data_matrixSource_matrix' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources.Inputs.data_matrixSource_matrix

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources.Inputs.data_freshAssignment_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources.Inputs.data_freshAssignment_eq

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources.Inputs.data_assignment_runningIndex_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources.Inputs.data_assignment_runningIndex_eq

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources.Inputs.freshMatrixImagesAt_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources.Inputs.freshMatrixImagesAt_eq

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources.Inputs.freshResidualAt_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources.Inputs.freshResidualAt_eq

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources.Inputs.freshTruth_iff_legacy' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources.Inputs.freshTruth_iff_legacy
