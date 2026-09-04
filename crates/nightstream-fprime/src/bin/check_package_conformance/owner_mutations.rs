//! Named-owner mutations checked by the independent raw-row evaluator.

use super::{
    event_row_count, events, expected_row, mul_mod,
    raw_assignment::{event_row_values, RawRowMutation},
    word, ColumnOwnerSpan, Event, MatrixSide, OwnerSpan, RawPackage, ReferenceLayout,
};

fn pilot_source_to_spartan(column: usize) -> usize {
    if column < 49_393 {
        column
    } else if column < 49_663 {
        14_722_239 + (column - 49_393)
    } else if column < 99_056 {
        49_393 + (column - 49_663)
    } else if column < 99_060 {
        14_722_509 + (column - 99_056)
    } else {
        98_786 + (column - 99_060)
    }
}

fn lift_pilot_column(column: usize) -> usize {
    if column < 98_786 {
        column
    } else if column < 14_722_238 {
        column + 29_288
    } else {
        29_336_446 + (column - 14_722_238)
    }
}

fn source_to_spartan(column: usize, layout: &ReferenceLayout) -> usize {
    assert!(column < 29_336_724, "Stage 1 source column");
    let prefix_column = if column < 14_722_512 {
        lift_pilot_column(pilot_source_to_spartan(column))
    } else if column < 14_722_516 {
        29_336_721 + (column - 14_722_512)
    } else if column < 14_751_804 {
        98_786 + (column - 14_722_516)
    } else {
        14_751_526 + (column - 14_751_804)
    };
    if prefix_column < 29_336_446 {
        prefix_column
    } else {
        prefix_column + (layout.unpadded_constant - 29_336_446)
    }
}

fn pilot_spartan_to_source(column: usize) -> Option<usize> {
    if column < 49_393 {
        Some(column)
    } else if column < 98_786 {
        Some(49_663 + (column - 49_393))
    } else if column < 14_722_238 {
        Some(99_060 + (column - 98_786))
    } else if column == 14_722_238 {
        None
    } else if column < 14_722_509 {
        Some(49_393 + (column - 14_722_239))
    } else if column < 14_722_513 {
        Some(99_056 + (column - 14_722_509))
    } else {
        None
    }
}

fn spartan_to_source(column: usize, layout: &ReferenceLayout) -> Option<usize> {
    let column = if column < 29_336_446 {
        column
    } else if column < layout.unpadded_constant {
        return None;
    } else {
        29_336_446 + (column - layout.unpadded_constant)
    };
    if column < 98_786 {
        pilot_spartan_to_source(column)
    } else if column < 128_074 {
        Some(14_722_516 + (column - 98_786))
    } else if column < 14_751_526 {
        pilot_spartan_to_source(column - 29_288)
    } else if column < 29_336_446 {
        Some(14_751_804 + (column - 14_751_526))
    } else if column == 29_336_446 {
        None
    } else if column < 29_336_721 {
        pilot_spartan_to_source(14_722_238 + (column - 29_336_446))
    } else if column < 29_336_725 {
        Some(14_722_512 + (column - 29_336_721))
    } else {
        None
    }
}

fn final_to_spartan(column: usize, layout: &ReferenceLayout) -> Option<usize> {
    if column < layout.unpadded_constant {
        Some(column)
    } else if column < layout.domain_size {
        None
    } else if column < layout.final_columns {
        Some(layout.unpadded_constant + (column - layout.domain_size))
    } else {
        None
    }
}

fn event_at<'a>(schedule: &[Event<'a>], raw: &RawPackage, row: usize) -> (Event<'a>, usize) {
    let index = schedule
        .partition_point(|event| event.row_start() <= row)
        .checked_sub(1)
        .expect("raw event before owner row");
    let event = schedule[index];
    let ordinal = row - event.row_start();
    assert!(ordinal < event_row_count(event, raw), "raw owner row coverage");
    (event, ordinal)
}

fn row_holds(values: [u64; 3]) -> bool {
    mul_mod(values[0], values[1]) == values[2]
}

pub(super) fn row_owner_mutation_checks(
    raw: &RawPackage,
    owners: &[OwnerSpan],
    private_values: &[u64],
    public_values: &[u64],
) -> usize {
    let schedule = events(raw);
    let mut checks = 0;
    for &owner in owners {
        let (event, ordinal) = event_at(&schedule, raw, owner.start);
        let actual = event_row_values(raw, event, ordinal, private_values, public_values, RawRowMutation::None);
        assert!(row_holds(actual), "{} raw owner row must hold", owner.name);

        let changed = event_row_values(
            raw,
            event,
            ordinal,
            private_values,
            public_values,
            RawRowMutation::CConstant(1),
        );
        assert!(!row_holds(changed), "{} raw row mutation must reject", owner.name);
        checks += 1;
    }
    checks
}

fn matching_column_rejects(
    raw: &RawPackage,
    schedule: &[Event<'_>],
    name: &str,
    rows: OwnerSpan,
    layout: &ReferenceLayout,
    private_values: &[u64],
    public_values: &[u64],
    mut owns: impl FnMut(usize) -> bool,
) -> bool {
    let first_event = schedule.partition_point(|&event| event.row_start() + event_row_count(event, raw) <= rows.start);
    for &event in &schedule[first_event..] {
        let event_start = event.row_start();
        if event_start >= rows.end {
            break;
        }
        let event_end = event_start + event_row_count(event, raw);
        let start = rows.start.max(event_start);
        let end = rows.end.min(event_end);
        for row in start..end {
            let ordinal = row - event_start;
            for side in [MatrixSide::A, MatrixSide::B, MatrixSide::C] {
                for &(final_column, _) in &expected_row(event, &raw.5, ordinal, side, layout) {
                    let Some(spartan_column) = final_to_spartan(final_column, layout) else {
                        continue;
                    };
                    if !owns(spartan_column) {
                        continue;
                    }
                    assert_ne!(spartan_column, layout.unpadded_constant, "{name} constant column");
                    let actual =
                        event_row_values(raw, event, ordinal, private_values, public_values, RawRowMutation::None);
                    assert!(row_holds(actual), "{name} source row must hold");
                    let changed = event_row_values(
                        raw,
                        event,
                        ordinal,
                        private_values,
                        public_values,
                        RawRowMutation::AssignmentColumn(spartan_column),
                    );
                    if !row_holds(changed) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn owned_column_rejects(
    raw: &RawPackage,
    schedule: &[Event<'_>],
    owner: ColumnOwnerSpan,
    layout: &ReferenceLayout,
    private_values: &[u64],
    public_values: &[u64],
) -> bool {
    matching_column_rejects(
        raw,
        schedule,
        owner.name,
        owner.rows,
        layout,
        private_values,
        public_values,
        |spartan_column| {
            let Some(source_column) = spartan_to_source(spartan_column, layout) else {
                return false;
            };
            if !(owner.columns.start <= source_column && source_column < owner.columns.end) {
                return false;
            }
            assert_eq!(
                source_to_spartan(source_column, layout),
                spartan_column,
                "{} round trip",
                owner.name,
            );
            true
        },
    )
}

pub(super) fn column_owner_mutation_checks(
    raw: &RawPackage,
    row_owners: &[OwnerSpan],
    column_owners: &[OwnerSpan],
    layout: &ReferenceLayout,
    private_values: &[u64],
    public_values: &[u64],
) -> usize {
    let schedule = events(raw);
    let phase_rows = OwnerSpan {
        name: "piccs",
        start: row_owners.first().expect("first PiCCS row owner").start,
        end: row_owners.last().expect("last PiCCS row owner").end,
    };
    let source_end = column_owners.last().expect("last PiCCS column owner").end;
    assert_ne!(source_end, 0, "nonempty PiCCS source-column inventory");
    assert_eq!(
        source_to_spartan(source_end - 1, layout) + 1,
        private_values.len(),
        "PiCCS owner columns cover the independently generated prefix",
    );

    let mut checks = 0;
    for columns in column_owners
        .iter()
        .copied()
        .filter(|owner| owner.start != owner.end)
    {
        let rows = match columns.name {
            "external" | "r1cs_intermediate" => phase_rows,
            child => *row_owners
                .iter()
                .find(|owner| owner.name == child)
                .expect("column owner has a row owner"),
        };
        let owner = ColumnOwnerSpan {
            name: columns.name,
            rows,
            columns,
        };
        assert!(owner.rows.start < owner.rows.end, "{} row interval", owner.name);
        assert!(
            owned_column_rejects(raw, &schedule, owner, layout, private_values, public_values),
            "{} has no raw-assignment-rejecting owned column",
            owner.name,
        );
        checks += 1;
    }
    checks
}

pub(super) fn public_segment_mutation_checks(
    raw: &RawPackage,
    row_owners: &[OwnerSpan],
    layout: &ReferenceLayout,
    private_values: &[u64],
    public_values: &[u64],
) -> usize {
    const PUBLIC_SEGMENTS: [(u64, &str); 3] = [
        (4, "prior_public_input"),
        (5, "output_digest"),
        (10, "verification_key"),
    ];

    assert_eq!(raw.3 .6.len(), PUBLIC_SEGMENTS.len(), "public segment count");
    let pi_ccs_rows = OwnerSpan {
        name: "piccs",
        start: row_owners.first().expect("first PiCCS row owner").start,
        end: row_owners.last().expect("last PiCCS row owner").end,
    };
    let pilot_rows = OwnerSpan {
        name: "pilot",
        start: 0,
        end: pi_ccs_rows.start,
    };
    let physical_end = word(raw.3 .4);
    let mut cursor = layout.unpadded_constant + 1;
    let schedule = events(raw);
    let mut checks = 0;
    for (segment, (expected_role, name)) in raw.3 .6.iter().zip(PUBLIC_SEGMENTS) {
        assert_eq!(segment.0, expected_role, "{name} public role");
        let start = word(segment.1);
        let count = word(segment.2);
        assert_eq!(start, cursor, "{name} public start");
        assert_ne!(count, 0, "{name} public count");
        let end = start.checked_add(count).expect("public segment end");
        assert!(end <= physical_end, "{name} public range");
        // The output digest is checked by Pilot. PiCCS consumes the output
        // preimage and must not claim ownership of this already-checked value.
        let rows = if expected_role == 5 { pilot_rows } else { pi_ccs_rows };
        assert!(
            matching_column_rejects(
                raw,
                &schedule,
                name,
                rows,
                layout,
                private_values,
                public_values,
                |column| start <= column && column < end,
            ),
            "{name} has no raw-assignment-rejecting column",
        );
        cursor = end;
        checks += 1;
    }
    assert_eq!(cursor, physical_end, "public segment coverage");
    checks
}
