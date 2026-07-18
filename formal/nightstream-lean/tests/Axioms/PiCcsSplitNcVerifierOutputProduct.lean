import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputProduct
import tests.Axioms.Support

/-!
Fail-closed dependency gate for canonical model-level `Pi_CCS` CE
materialization and its exact `yRing` authority boundary.

| Export | Audited property |
|---|---|
| `outputProduct_shape` | canonical fields satisfy the public `PiCCS.Shape` contract |
| `outputProduct_unique` | the complete CE product has no independently choosable field |
| `materialize_eq_of_yRing_eq` | the delayed-NC `yZcol` payload is absent from CE |
| `yRingBoundToSources_iff_outputEvaluationsBound` | source binding equals every concrete CE evaluation equation |
| point transport | FE and NC authority predicates ignore the other phase's point |
-/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputProduct.outputProduct_shape' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputProduct.outputProduct_shape

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputProduct.outputProduct_unique' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputProduct.outputProduct_unique

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputProduct.materialize_eq_of_yRing_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputProduct.materialize_eq_of_yRing_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputProduct.yRingBoundToSources_iff_outputEvaluationsBound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputProduct.yRingBoundToSources_iff_outputEvaluationsBound

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.yRingBoundToSources_iff_of_rPrime_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.yRingBoundToSources_iff_of_rPrime_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.yZcolBoundToSources_iff_of_sPrime_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.yZcolBoundToSources_iff_of_sPrime_eq
