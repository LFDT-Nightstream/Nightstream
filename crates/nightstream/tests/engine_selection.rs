use nightstream::{Engine, EngineError, Error, Prover};

#[test]
fn unavailable_engines_fail_before_loading_the_circuit() {
    let engines = [
        #[cfg(not(feature = "metal"))]
        Engine::Metal,
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
fn explicit_cpu_selection_keeps_circuit_validation() {
    for engine in [Engine::Optimized, Engine::PaperExact, Engine::Crosscheck] {
        assert!(matches!(
            Prover::load(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"),
                engine,
                114
            ),
            Err(Error::Package(_))
        ));
    }
}
