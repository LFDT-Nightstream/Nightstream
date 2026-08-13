//! Public backend-selection API checks that do not build a proof fixture.

use neo_wasm::{WasmProver, WasmProverBackend};

#[test]
fn explicit_optimized_cpu_prover_reports_its_backend() {
    let prover = WasmProver::cpu_optimized();
    assert_eq!(prover.backend(), WasmProverBackend::CpuOptimized);
    assert_eq!(prover.backend().as_str(), "cpu-optimized");
    assert_eq!(prover.fallback_reason(), None);
}

#[test]
fn explicit_paper_exact_prover_reports_its_backend() {
    let prover = WasmProver::paper_exact();
    assert_eq!(prover.backend(), WasmProverBackend::PaperExact);
    assert_eq!(prover.backend().as_str(), "paper-exact");
    assert_eq!(prover.fallback_reason(), None);
}

#[test]
fn automatic_prover_reports_a_concrete_backend() {
    let prover = WasmProver::auto();
    assert!(matches!(
        prover.backend(),
        WasmProverBackend::CpuOptimized | WasmProverBackend::Metal | WasmProverBackend::Cuda
    ));
    #[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
    assert_eq!(prover.backend(), WasmProverBackend::Metal);
    #[cfg(all(not(feature = "cuda"), not(all(feature = "metal", target_vendor = "apple"))))]
    assert_eq!(prover.backend(), WasmProverBackend::CpuOptimized);
    if prover.backend() != WasmProverBackend::CpuOptimized {
        assert_eq!(prover.fallback_reason(), None);
    }
}

#[test]
fn explicit_cuda_request_fails_until_the_kernel_is_available() {
    assert!(matches!(
        WasmProver::cuda(),
        Err(neo_wasm::WasmNebulaError::ProverBackendUnavailable { backend: "cuda", .. })
    ));
}

#[cfg(not(all(feature = "metal", target_vendor = "apple")))]
#[test]
fn explicit_metal_request_fails_when_metal_is_not_built() {
    assert!(matches!(
        WasmProver::metal(),
        Err(neo_wasm::WasmNebulaError::ProverBackendUnavailable { backend: "metal", .. })
    ));
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn explicit_metal_prover_reports_metal_without_fallback() {
    let prover = WasmProver::metal().expect("Metal prover");
    assert_eq!(prover.backend(), WasmProverBackend::Metal);
    assert_eq!(prover.fallback_reason(), None);
}
