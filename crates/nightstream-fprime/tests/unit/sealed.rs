use super::*;
use crate::package::Segment;
use serde_json::json;

fn layout() -> Layout {
    Layout {
        row_count: 17,
        private_column_count: 20,
        constant_column: 20,
        public_column_count: 2,
        total_column_count: 23,
        private_segments: vec![
            Segment {
                role: APPLICATION_WITNESS_ROLE,
                start: 10,
                length: 2,
            },
            Segment {
                role: APPLICATION_LOCAL_ROLE,
                start: 12,
                length: 5,
            },
        ],
        public_segments: vec![],
    }
}

fn application_plan() -> Value {
    json!([
        APPLICATION_PLAN_SCHEMA,
        2,
        [1, 2, 3, 4],
        [10, 11],
        [5, 6, 7, 8],
        12,
        5,
        9,
        3,
        [],
        [],
        [],
        [],
        [[70]],
        [[80], [81]],
        [[90]]
    ])
}

#[test]
fn assignment_transport_accepts_schema4_and_rejects_legacy_shapes() {
    const PHYSICAL_WIDTH: usize = 60_000;
    const LOGICAL_PUBLIC_WIDTH: usize = 270;
    const PHI81_INVOCATIONS: usize = 52_326;
    const CHALLENGE_BITS: usize = 17 * 54 * 3;
    const DIGEST_START: usize = PHI81_INVOCATIONS + CHALLENGE_BITS;

    let logical_width = LOGICAL_PUBLIC_WIDTH + CHALLENGE_BITS + (4 + PHI81_INVOCATIONS) * 41;
    let transport = json!([
        4,
        [
            [0, CHALLENGE_BITS, [[PHI81_INVOCATIONS, 1, CHALLENGE_BITS]]],
            [2, 4, [[DIGEST_START, 1, 4]]],
            [2, PHI81_INVOCATIONS, [[PHYSICAL_WIDTH, 1, PHI81_INVOCATIONS]]]
        ],
        [
            [[17, 22, 1], [17, 5, 1], [17, 1, 2], [17, 14, 2]],
            [[0, 1, PHI81_INVOCATIONS]],
            [[PHI81_INVOCATIONS, 1, CHALLENGE_BITS]]
        ],
        [
            [0, DIGEST_START],
            [0, DIGEST_START + 1],
            [0, DIGEST_START + 2],
            [0, DIGEST_START + 3]
        ]
    ]);

    crate::package::assignment_transport::decode(&transport, PHYSICAL_WIDTH, LOGICAL_PUBLIC_WIDTH, logical_width)
        .expect("schema-4 assignment transport");

    for (pointer, value, expected) in [
        ("/0", json!(1), "assignment transport schema version"),
        ("/0", json!(2), "assignment transport schema version"),
        ("/0", json!(3), "assignment transport schema version"),
        ("/0", json!(5), "assignment transport schema version"),
        ("/1/0/0", json!(3), "assignment slot kind"),
        ("/1/0/1", json!(CHALLENGE_BITS + 1), "assignment source run coverage"),
        ("/1/0", json!([0, 0, 0, 0, []]), "wide retained block"),
        (
            "/1/0/2/0/0",
            json!(PHYSICAL_WIDTH + PHI81_INVOCATIONS),
            "assignment source domain bound",
        ),
        ("/2/0/0/1", json!(23), "wide quotient family shape"),
        (
            "/2/1/0/2",
            json!(PHI81_INVOCATIONS - 1),
            "assignment source run coverage",
        ),
        ("/2/2/0/2", json!(CHALLENGE_BITS - 1), "assignment source run coverage"),
        ("/2", json!([[], []]), "wide quotient recipe"),
        ("/3/0/1", json!(PHYSICAL_WIDTH), "assignment expression column bound"),
        ("/3", json!([[0, 0], [0, 1], [0, 2]]), "assignment expressions"),
    ] {
        let mut invalid = transport.clone();
        *invalid
            .pointer_mut(pointer)
            .expect("transport mutation field") = value;
        assert!(
            matches!(
                crate::package::assignment_transport::decode(
                    &invalid, PHYSICAL_WIDTH, LOGICAL_PUBLIC_WIDTH, logical_width,
                ),
                Err(PackageError::Invalid(actual)) if actual == expected
            ),
            "mutation {pointer}: expected {expected}"
        );
    }
    let mut missing_block = transport.clone();
    missing_block[1]
        .as_array_mut()
        .expect("assignment blocks")
        .pop();
    assert!(matches!(
        crate::package::assignment_transport::decode(
            &missing_block,
            PHYSICAL_WIDTH,
            LOGICAL_PUBLIC_WIDTH,
            logical_width,
        ),
        Err(PackageError::Invalid("wide assignment coordinate width"))
    ));
    let mut extra_fields = transport;
    extra_fields
        .as_array_mut()
        .expect("assignment transport")
        .push(json!([]));
    assert!(matches!(
        crate::package::assignment_transport::decode(
            &extra_fields,
            PHYSICAL_WIDTH,
            LOGICAL_PUBLIC_WIDTH,
            logical_width,
        ),
        Err(PackageError::Invalid("wide assignment transport plan"))
    ));
}

fn circuit_value() -> Value {
    json!([
        8,
        0,
        0,
        0,
        0,
        0,
        [],
        [],
        [],
        [],
        [[1], [70]],
        [[2], [80], [81]],
        [[3], [90], [12], [13], [14], [15], [16]],
        0
    ])
}

fn next_preimage_rows() -> std::ops::Range<usize> {
    decode_next_preimage_range(RawRowRange(12, 5), &layout()).expect("valid next preimage range")
}

fn decode_plan(plan: &Value, circuit: &Value) -> Result<LoadedApplicationPlan, PackageError> {
    decode_application_plan(plan, circuit, &layout(), &next_preimage_rows())
}

#[test]
fn application_plan_decodes_exact_lean_owned_ranges() {
    let circuit = circuit_value();
    let plan = decode_plan(&application_plan(), &circuit).expect("valid application plan");
    validate_next_preimage_assertion_suffix(&circuit, &next_preimage_rows()).expect("valid next preimage suffix");
    assert_eq!(plan.witness_word_count(), 2);
    assert_eq!(plan.input_columns(), [1, 2, 3, 4]);
    assert_eq!(plan.witness_columns(), [10, 11]);
    assert_eq!(plan.output_columns(), [5, 6, 7, 8]);
    assert_eq!(plan.private_range(), 12..17);
    assert_eq!(plan.row_range(), 9..12);
}

#[test]
fn application_message_is_input_and_application_local_is_generated() {
    assert!(!crate::package::v1_1::is_witness_role(APPLICATION_WITNESS_ROLE));
    assert!(crate::package::v1_1::is_witness_role(APPLICATION_LOCAL_ROLE));
}

#[test]
fn application_plan_rejects_a_row_not_present_in_the_package_suffix() {
    let mut plan = application_plan();
    plan[15][0] = json!([91]);
    assert!(matches!(
        decode_plan(&plan, &circuit_value()),
        Err(PackageError::Invalid("application plan package suffix"))
    ));
}

#[test]
fn application_plan_rejects_an_unowned_row_family() {
    let mut plan = application_plan();
    plan[11] = json!([[1]]);
    assert!(matches!(
        decode_plan(&plan, &circuit_value()),
        Err(PackageError::Invalid("application plan row family"))
    ));
}

#[test]
fn application_plan_rejects_changed_witness_ownership() {
    let mut plan = application_plan();
    plan[3] = json!([9, 10]);
    assert!(matches!(
        decode_plan(&plan, &circuit_value()),
        Err(PackageError::Invalid("application plan column ownership"))
    ));
}

#[test]
fn application_plan_rejects_a_wrong_row_count() {
    let mut plan = application_plan();
    plan[8] = json!(4);
    assert!(matches!(
        decode_plan(&plan, &circuit_value()),
        Err(PackageError::Invalid("application plan range"))
    ));
}

#[test]
fn next_preimage_range_rejects_changed_start_or_count() {
    for raw in [RawRowRange(11, 5), RawRowRange(13, 4)] {
        assert!(matches!(
            decode_next_preimage_range(raw, &layout()),
            Err(PackageError::Invalid("next preimage row range"))
        ));
    }
}

#[test]
fn next_preimage_suffix_rejects_changed_row_index() {
    let mut circuit = circuit_value();
    circuit[12][6][0] = json!(99);
    assert!(matches!(
        validate_next_preimage_assertion_suffix(&circuit, &next_preimage_rows()),
        Err(PackageError::Invalid("next preimage package suffix"))
    ));
}

#[test]
fn application_plan_rejects_assertion_not_immediately_before_next_preimage() {
    let mut circuit = circuit_value();
    circuit[12][1] = json!([91]);
    assert!(matches!(
        decode_plan(&application_plan(), &circuit),
        Err(PackageError::Invalid("application plan package suffix"))
    ));
}

#[test]
fn terminal_layout_retains_the_exact_outer_relation_shape() {
    let raw = json!([1, [0, 17, 16, 1]]);
    let layout = super::super::validate_terminal(raw.as_array().expect("terminal option"), 17)
        .expect("valid terminal option")
        .expect("present terminal layout");
    assert_eq!(layout.row_start(), 0);
    assert_eq!(layout.row_count(), 17);
    assert_eq!(layout.running_claim_count(), 16);
    assert_eq!(layout.fresh_claim_count(), 1);
}

#[test]
fn terminal_layout_rejects_each_changed_authoritative_field() {
    for index in 0..4 {
        let mut raw = json!([1, [0, 17, 16, 1]]);
        raw[1][index] = json!(raw[1][index].as_u64().expect("terminal word") + 1);
        assert!(matches!(
            super::super::validate_terminal(raw.as_array().expect("terminal option"), 17),
            Err(PackageError::Invalid("pilot terminal option"))
        ));
    }
}
