use super::{oracle_error, MetalError};
use neo_reductions::PiCcsError;

#[test]
fn device_failure_keeps_its_backend_type() {
    for error in [
        MetalError::Unavailable,
        MetalError::Buffer { bytes: 64 },
        MetalError::Execution("device lost".into()),
    ] {
        assert!(matches!(
            oracle_error(error),
            PiCcsError::BackendFailure { backend: "metal", .. }
        ));
    }
}

#[test]
fn invalid_input_does_not_become_a_device_failure() {
    assert!(matches!(
        oracle_error(MetalError::Shape("claim width")),
        PiCcsError::InvalidInput(_)
    ));
}
