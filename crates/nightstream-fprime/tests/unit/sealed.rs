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
fn assignment_plan_accepts_only_the_lean_owned_order() {
    let canonical = (0..ASSIGNMENT_BLOCK_KIND_COUNT as u8).collect::<Vec<_>>();
    let plan =
        decode_assignment_plan(canonical.iter().copied().map(u64::from).collect()).expect("canonical assignment plan");
    assert_eq!(plan.kind_codes(), canonical.as_slice());

    let mut changed = canonical.into_iter().map(u64::from).collect::<Vec<_>>();
    changed[17] = 18;
    assert!(matches!(
        decode_assignment_plan(changed),
        Err(PackageError::Invalid("assignment transport plan"))
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
    let layout = super::super::validate_terminal(raw.as_array().expect("terminal option"), 8, 17)
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
            super::super::validate_terminal(raw.as_array().expect("terminal option"), 8, 17),
            Err(PackageError::Invalid("pilot terminal option"))
        ));
    }
}
