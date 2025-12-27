//! Tests for output_binding module.

use neo_fold::output_binding::{simple_output_config, OutputBindingConfig};
use neo_math::F;
use neo_memory::output_check::ProgramIO;
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_output_binding_config() {
    let config = OutputBindingConfig::new(16, ProgramIO::new());
    assert_eq!(config.num_bits, 16);
    assert_eq!(config.mem_idx, 0);
}

#[test]
fn test_output_binding_config_builder_mem_idx() {
    let config = OutputBindingConfig::new(16, ProgramIO::new()).with_mem_idx(2);
    assert_eq!(config.mem_idx, 2);
}

#[test]
fn test_simple_output_config() {
    let config = simple_output_config(16, 0x1000, F::from_u64(42));
    assert_eq!(config.num_bits, 16);
    assert_eq!(config.program_io.num_claims(), 1);
}
