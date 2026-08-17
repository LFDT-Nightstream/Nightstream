use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::super::streaming_lifecycle::streaming_terminal_x_out_authority_audit;

#[test]
fn streaming_terminal_x_out_authority_rejects_each_changed_group() {
    let audit = streaming_terminal_x_out_authority_audit();
    let source = audit.source();
    let x_out_columns = audit.x_out_columns();
    let mut witness = source.witness().to_vec();
    assert!(
        source.is_satisfied(&witness),
        "honest terminal XOut authority must satisfy"
    );

    for (index, group) in [
        (0, "domain"),
        (1, "verifier key"),
        (5, "PiCCS header"),
        (9, "chunk counter"),
        (11, "step counter"),
        (13, "program counter"),
        (15, "boundary"),
        (19, "semantic state"),
        (23, "Construction-2 accumulator"),
        (27, "Nebula presence"),
        (28, "Nebula state"),
    ] {
        let column = x_out_columns[index];
        let original = witness[column];
        witness[column] = original + F::ONE;
        assert!(!source.is_satisfied(&witness), "changed {group} must reject");
        witness[column] = original;
        assert!(source.is_satisfied(&witness), "restored {group} must satisfy");
    }
}
