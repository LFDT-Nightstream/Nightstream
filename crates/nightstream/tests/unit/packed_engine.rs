use super::*;

#[test]
fn packed_oracles_use_the_engine_selected_for_proving() {
    let cpu = Backend::new(Engine::Optimized).unwrap();
    let metal = Backend::new(Engine::Metal).unwrap();
    assert!(cpu.oracle(true).unwrap().backend().is_none());
    assert!(metal.oracle(true).unwrap().backend().is_some());
    assert!(matches!(metal.oracle(true).unwrap(), Oracle::Packed(_)));
}
