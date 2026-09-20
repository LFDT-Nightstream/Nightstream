use nightstream::{
    application::{Affine, ApplicationBuilder},
    Circuit, Engine, EngineError, Error,
};

fn identity() -> nightstream::application::ApplicationCircuit {
    let builder = ApplicationBuilder::new(0).unwrap();
    let state = builder.input_state().map(Affine::from);
    builder.finish(state).unwrap()
}

#[test]
fn unavailable_engines_fail_before_loading_the_circuit() {
    let engines = [
        #[cfg(not(feature = "metal"))]
        Engine::Metal,
        Engine::Cuda,
    ];
    for engine in engines {
        let result = Circuit::prepare_with_engine(b"not a circuit", identity(), engine);
        assert!(
            matches!(result, Err(Error::Engine(EngineError::Unavailable { engine: actual, .. })) if actual == engine)
        );
    }
}

#[test]
fn explicit_cpu_selection_keeps_circuit_validation() {
    assert!(matches!(
        Circuit::prepare_with_engine(b"not a circuit", identity(), Engine::Optimized),
        Err(Error::Assembly(_))
    ));
}
