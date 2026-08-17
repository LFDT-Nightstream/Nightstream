//! Exact source-to-final placement test for the streaming terminal profile.

#[path = "../support/streaming_terminal_fixture.rs"]
mod streaming_terminal_fixture;

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use streaming_terminal_fixture::build_streaming_terminal_audit_fixture;

#[test]
#[ignore = "expensive exact lifecycle selective-lowering profile"]
fn terminal_profile_binds_the_trailing_fresh_after_state_to_final_rows() {
    let mut fixture = build_streaming_terminal_audit_fixture();
    let cases: [(usize, fn(F) -> F, &str); 6] = [
        (
            fixture.schedule_selector_column,
            |_| F::ZERO,
            "terminal schedule selector",
        ),
        (
            fixture.verifier_key_column,
            |value| value + F::ONE,
            "verifier-key digest",
        ),
        (
            fixture.program_binding_column,
            |value| value + F::ONE,
            "Nebula program binding",
        ),
        (
            fixture.delayed_payload_column,
            |value| F::ONE - value,
            "delayed payload",
        ),
        (fixture.fresh_adv_column, |value| value + F::ONE, "fresh adv opening"),
        (
            fixture.final_closed_lane_column,
            |_| F::from_u64(2),
            "final closed lane",
        ),
    ];
    for (column, change, label) in cases {
        let original = fixture.terminal.witness()[column];
        fixture.terminal.tamper_witness(column, change(original));
        assert!(!fixture.terminal.is_satisfied(), "changed {label} must reject");
        fixture.terminal.tamper_witness(column, original);
        assert!(fixture.terminal.is_satisfied(), "restored {label} must satisfy");
    }
}
