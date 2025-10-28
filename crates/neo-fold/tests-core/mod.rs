#[path = "01_unit_oracle.rs"]
mod unit_oracle;

mod fixture_gen;

#[path = "03_unit_eq_weights.rs"]
mod unit_eq_weights;

#[path = "04_unit_sparse_ops.rs"]
mod unit_sparse_ops;

#[path = "05_integration_pi_ccs_k1_smoke.rs"]
mod integration_pi_ccs_k1_smoke;

#[path = "06_integration_pi_ccs_k2_smoke.rs"]
mod integration_pi_ccs_k2_smoke;

#[path = "07_unit_nc_constraints.rs"]
mod unit_nc_constraints;

#[path = "08_unit_eval.rs"]
mod unit_eval;

#[path = "09_unit_terminal.rs"]
mod unit_terminal;

#[path = "10_unit_f_block_semantics.rs"]
mod unit_f_block_semantics;
