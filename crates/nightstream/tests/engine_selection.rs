//! An unavailable engine fails before the saved circuit loads: Metal needs
//! the `metal` feature, and CUDA needs the `cuda` feature.

use nightstream::{Engine, EngineError, Error, Prover};

#[test]
fn unavailable_engines_fail_before_loading_the_circuit() {
    let engines = [
        #[cfg(not(feature = "metal"))]
        Engine::Metal,
        #[cfg(not(feature = "cuda"))]
        Engine::Cuda,
    ];
    for engine in engines {
        let result = Prover::load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"),
            engine,
            114,
        );
        assert!(
            matches!(result, Err(Error::Engine(EngineError::Unavailable { engine: actual, .. })) if actual == engine)
        );
    }
}

#[test]
fn available_engines_reach_the_circuit_check() {
    let engines = [
        Engine::Optimized,
        #[cfg(feature = "metal")]
        Engine::Metal,
        #[cfg(feature = "cuda")]
        Engine::Cuda,
    ];
    for engine in engines {
        assert!(matches!(
            Prover::load(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"),
                engine,
                114
            ),
            Err(Error::Circuit("not a saved option-B application"))
        ));
    }
}
