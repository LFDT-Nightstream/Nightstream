import Nightstream.Implementation.R1CS.Correspondence.PiCcsMatrix
import tests.Axioms.Support

/-!
Fail-closed kernel-dependency gate for concrete `Pi_CCS` matrix correspondence.

The guards below make the assurance boundary explicit: exhaustive runtime
artifact checks and their semantic transport must remain kernel-checked and
must not acquire `Lean.trustCompiler` or `Classical.choice`.
-/

/-- info: 'Nightstream.Implementation.R1CS.Phi81BarMatrixRefinement.artifact_shape' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Phi81BarMatrixRefinement.artifact_shape

/-- info: 'Nightstream.Implementation.R1CS.Phi81BarMatrixRefinement.runtimeBarEntry_eq_native' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Phi81BarMatrixRefinement.runtimeBarEntry_eq_native

/-- info: 'Nightstream.Implementation.R1CS.Phi81BarMatrixRefinement.runtimeBarBasis_eq_barBasis' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Phi81BarMatrixRefinement.runtimeBarBasis_eq_barBasis

/-- info: 'Nightstream.Implementation.R1CS.Phi81BarMatrixRefinement.runtimeKernel_constant_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Phi81BarMatrixRefinement.runtimeKernel_constant_eq

/-- info: 'Nightstream.Implementation.R1CS.Phi81BarMatrixRefinement.runtimeKernel_weight_eq_phi81Kernel' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Phi81BarMatrixRefinement.runtimeKernel_weight_eq_phi81Kernel

/-- info: 'Nightstream.Implementation.R1CS.Phi81BarMatrixRefinement.runtimeConstantTermLaw' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Phi81BarMatrixRefinement.runtimeConstantTermLaw

/-- info: 'Nightstream.Implementation.R1CS.Phi81MatrixSourceRefinement.runtimeSource_matrix_eq_semantic' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Phi81MatrixSourceRefinement.runtimeSource_matrix_eq_semantic

/-- info: 'Nightstream.Implementation.R1CS.Phi81MatrixSourceRefinement.runtimeSource_paddedMatrixEntry_eq_semantic' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Phi81MatrixSourceRefinement.runtimeSource_paddedMatrixEntry_eq_semantic

/-- info: 'Nightstream.Implementation.R1CS.Phi81MatrixSourceRefinement.runtimeSource_coefficientMatrix_eq_semantic' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Phi81MatrixSourceRefinement.runtimeSource_coefficientMatrix_eq_semantic
