//! WASM operation relations: the R1CS for the opcodes that the base relation
//! does not handle (bitwise, compare, shift, mul, counts, div/rem).
pub mod r1cs;

pub const OPERATIONS_REGION: &str = "operations";
