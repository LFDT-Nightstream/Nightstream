import tests.FPrimeFullHistoryObligationTree
import tests.Axioms.Support

open Nightstream.Assurance.FPrimeFullHistoryObligationTree

/-!
Fail-closed audit for the captured full-history obligation tree. The
`Lean.trustCompiler` dependency is inherited from the existing artifact-checked
full-row certificates; this module does not introduce a new native decision.
-/

/--
info: 'Nightstream.Assurance.FPrimeFullHistoryObligationTree.every_parent_cost_exact' depends on axioms: [propext]
-/
#guard_msgs in
#audit_axioms every_parent_cost_exact

/--
info: 'Nightstream.Assurance.FPrimeFullHistoryObligationTree.every_materialized_row_has_exactly_one_leaf' depends on axioms: [propext,
 Classical.choice,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms every_materialized_row_has_exactly_one_leaf

/--
info: 'Nightstream.Assurance.FPrimeFullHistoryObligationTree.no_row_outside_materialized_leaves' depends on axioms: [propext,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms no_row_outside_materialized_leaves

/--
info: 'Nightstream.Assurance.FPrimeFullHistoryObligationTree.every_lean_evidence_checked' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms every_lean_evidence_checked

/--
info: 'Nightstream.Assurance.FPrimeFullHistoryObligationTree.every_leaf_cross_layer_mapped' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms every_leaf_cross_layer_mapped

/--
info: 'Nightstream.Assurance.FPrimeFullHistoryObligationTree.obligation_tree_retains_terminal_drift' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound]
-/
#guard_msgs in
#audit_axioms obligation_tree_retains_terminal_drift
