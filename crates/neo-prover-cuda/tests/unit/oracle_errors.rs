use super::{oracle_error, CudaError};
use neo_reductions::PiCcsError;

#[test]
fn device_failure_keeps_its_backend_type() {
    for error in [
        CudaError::Device("no device".into()),
        CudaError::Buffer { bytes: 64 },
        CudaError::Execution("device lost".into()),
    ] {
        assert!(matches!(
            oracle_error(error),
            PiCcsError::BackendFailure { backend: "cuda", .. }
        ));
    }
}

#[test]
fn invalid_input_does_not_become_a_device_failure() {
    assert!(matches!(
        oracle_error(CudaError::Shape("claim width")),
        PiCcsError::InvalidInput(_)
    ));
}
