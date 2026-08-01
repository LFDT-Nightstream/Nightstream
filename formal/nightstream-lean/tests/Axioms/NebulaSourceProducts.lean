import Nightstream.Implementation.Lowering.Nebula.SourceProducts
import tests.Axioms.Support

/-! Fail-closed dependency guards for exact Nebula tuple products. -/

/-- info: 'Nightstream.Implementation.Lowering.Nebula.SourceProducts.operationRun_eq_input_mul_product' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.SourceProducts.operationRun_eq_input_mul_product

/-- info: 'Nightstream.Implementation.Lowering.Nebula.SourceProducts.wasm42x6_public_products_source_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.SourceProducts.wasm42x6_public_products_source_bound
