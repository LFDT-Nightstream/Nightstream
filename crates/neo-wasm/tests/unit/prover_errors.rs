use super::is_backend_error;
use crate::WasmNebulaError;
use neo_fold_clean::{
    frontends::nebula::NebulaFPrimeChainError,
    lifecycle,
    paper::{construction2, nifs, pi_ccs, pi_dec},
};
use neo_reductions::PiCcsError;

fn lifecycle_error(error: nifs::Error) -> lifecycle::Error {
    construction2::Error::from(error).into()
}

#[test]
fn device_errors_trigger_fallback_at_both_lifecycle_entrypoints() {
    for error in [
        nifs::Error::BackendUnavailable {
            backend: "cuda",
            reason: "no device",
        },
        nifs::Error::BackendFailure {
            backend: "metal",
            phase: "commitment",
            reason: "allocation failed".into(),
        },
        pi_ccs::Error::from(neo_fold_clean::engine::optimized::Error::from(
            PiCcsError::BackendFailure {
                backend: "metal",
                reason: "oracle allocation failed".into(),
            },
        ))
        .into(),
        pi_dec::Error::from(neo_fold_clean::engine::optimized::Error::from(
            PiCcsError::BackendFailure {
                backend: "metal",
                reason: "openings failed".into(),
            },
        ))
        .into(),
    ] {
        let error = lifecycle_error(error);
        assert!(super::is_lifecycle_backend_error(&error), "{error}");
        assert!(is_backend_error(&WasmNebulaError::Chain(
            NebulaFPrimeChainError::Lifecycle(error)
        )));
    }
    let error = lifecycle_error(nifs::Error::BackendFailure {
        backend: "metal",
        phase: "commitment",
        reason: "allocation failed".into(),
    });
    assert!(is_backend_error(&WasmNebulaError::Lifecycle(error)));
}

#[test]
fn protocol_and_input_errors_do_not_trigger_fallback() {
    for error in [
        PiCcsError::ProtocolError("Metal buffer allocation failed".into()),
        PiCcsError::InvalidInput("claim width".into()),
        PiCcsError::SumcheckError("terminal identity".into()),
    ] {
        let error = lifecycle_error(pi_ccs::Error::from(neo_fold_clean::engine::optimized::Error::from(error)).into());
        assert!(!is_backend_error(&WasmNebulaError::Lifecycle(error)));
    }
    assert!(!is_backend_error(&WasmNebulaError::EmptyTrace));
}
