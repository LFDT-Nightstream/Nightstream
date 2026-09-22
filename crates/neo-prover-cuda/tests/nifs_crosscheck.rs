#![cfg(all(feature = "cuda", feature = "legacy-adapter"))]

//! CUDA backend availability boundary.

use neo_fold_legacy::paper::nifs::Error;
use neo_prover_cuda::CudaNifsProver;

#[test]
fn cuda_rejects_selection_until_the_canonical_kernel_exists() {
    assert!(matches!(
        CudaNifsProver::new(),
        Err(Error::BackendUnavailable { backend: "cuda", .. })
    ));
}
