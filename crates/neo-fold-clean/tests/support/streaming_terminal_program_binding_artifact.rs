//! Compact exact artifact for the full terminal Nebula program-binding family.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::STREAMING_TERMINAL_R1CS_FAMILY_NAMES;
use neo_fold_clean::paper::f_prime::nebula_lane_circuit::enforce_nebula_program_binding_digest_circuit;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::streaming_terminal_fixture::StreamingTerminalAuditFixture;
use super::{branch_constant_values, constant_row_output, relocated_terms};

const ARTIFACT_PATH: &str = "../../formal/nightstream-lean/Nightstream/Implementation/\
R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingTerminalFullProgramBinding.lean";

fn alloc_fixed(builder: &mut R1csBuilder, value: F) -> Var {
    let wire = builder.alloc(value);
    builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(value));
    wire
}

pub(super) fn artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(ARTIFACT_PATH)
}

fn reference_relation(
    full_source: &neo_fold_clean::engine::r1cs_circuit::R1csSnapshot,
    full_lane: [usize; 50],
    constant_values: &[u64],
) -> (neo_fold_clean::engine::r1cs_circuit::R1csSnapshot, [usize; 4]) {
    assert_eq!(constant_values.len(), 19);
    let mut builder = R1csBuilder::new();
    let carried = std::array::from_fn(|lane| builder.alloc(full_source.witness()[full_lane[lane]]));
    let initial_semantic = std::array::from_fn(|lane| alloc_fixed(&mut builder, F::from_u64(constant_values[lane])));
    let plan = std::array::from_fn(|lane| alloc_fixed(&mut builder, F::from_u64(constant_values[4 + lane])));
    let initial_memory = std::array::from_fn(|lane| alloc_fixed(&mut builder, F::from_u64(constant_values[8 + lane])));
    let computed = enforce_nebula_program_binding_digest_circuit(&mut builder, initial_semantic, plan, initial_memory);
    for lane in 0..4 {
        builder.enforce_eq(&Lc::from_var(carried[lane]), &Lc::from_var(computed[lane]));
    }
    let carried_columns = carried.map(Var::col);
    let source = builder.snapshot();
    assert!(source.is_satisfied(source.witness()));
    (source, carried_columns)
}

pub(super) fn render(fixture: StreamingTerminalAuditFixture) -> String {
    let family_name = STREAMING_TERMINAL_R1CS_FAMILY_NAMES[5];
    let ranges = fixture
        .terminal
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == family_name)
        .map(|range| range.row_start..range.row_end)
        .collect::<Vec<_>>();
    let [full_rows] = ranges.as_slice() else {
        panic!("terminal program-binding family must have one row range")
    };
    let full_lane = std::array::from_fn::<_, 50, _>(|index| fixture.source_binding_decoded_column_start + 32 + index);
    let full_source = fixture.terminal.into_snapshot();
    let full_constant_start = constant_row_output(&full_source, full_rows.start);
    let constant_values = branch_constant_values(&full_source, full_rows.start, full_constant_start, 19);
    let (reference, reference_carried) = reference_relation(&full_source, full_lane, &constant_values);
    assert_eq!(reference.rows(), full_rows.len());

    let reference_internal_start = constant_row_output(&reference, 0);
    let mut external_columns = BTreeMap::<usize, usize>::from([(0, 0)]);
    external_columns.extend(
        reference_carried
            .into_iter()
            .zip(full_lane[0..4].iter().copied()),
    );
    for (reference_row, full_row) in (0..reference.rows()).zip(full_rows.clone()) {
        assert_eq!(
            relocated_terms(
                reference.a_row(reference_row),
                &external_columns,
                reference_internal_start,
                full_constant_start,
            ),
            full_source.a_row(full_row),
        );
        assert_eq!(
            relocated_terms(
                reference.b_row(reference_row),
                &external_columns,
                reference_internal_start,
                full_constant_start,
            ),
            full_source.b_row(full_row),
        );
        assert_eq!(
            relocated_terms(
                reference.c_row(reference_row),
                &external_columns,
                reference_internal_start,
                full_constant_start,
            ),
            full_source.c_row(full_row),
        );
    }

    let constant_columns = (full_constant_start..full_constant_start + 19).collect::<Vec<_>>();
    let mut input_columns = constant_columns[12..19].to_vec();
    input_columns.extend_from_slice(&constant_columns[0..12]);
    let equality_row_start = full_rows.len() - 4;
    let carried_binding_columns = full_lane[0..4].to_vec();
    let hash_output_columns = (0..4)
        .map(|lane| {
            let carried = carried_binding_columns[lane];
            let columns = full_source
                .a_row(full_rows.start + equality_row_start + lane)
                .iter()
                .filter(|&&(column, _)| column != carried)
                .map(|&(column, _)| column)
                .collect::<Vec<_>>();
            let [output] = columns.as_slice() else {
                panic!("program-binding equality row must have one hash output")
            };
            *output
        })
        .collect::<Vec<_>>();
    assert_eq!(full_rows.len(), 3644);

    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalProgramBindingSchema\n\n\
         /-! Generated exact full-layout Rust terminal Nebula program-binding recipe.\n\n\
         Rust compares every row with a reference built by the production function.\n\
         The empty SHA field is legacy diagnostic structure and is not authority.\n\
         Emits constraints: no.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullProgramBinding\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProgramBinding.Artifact\n\n\
         def lifecycleScope : String := \"recursive-terminal-arm-435\"\n\n\
         def constantValues : List Nat := {:?}\n\n\
         def rawArtifact : RawArtifact :=\n  \
         {{ schemaVersion := 1,\n    \
            profileId := \"nightstream/goldilocks/streaming-terminal-full-program-binding/v1\",\n    \
            sourceIdentity := \"rust:nightstream/streaming-terminal-lifecycle/source-rows/v1\",\n    \
            sourceRowsSha256 := \"\", rowCount := {}, columnCount := {},\n    \
            sourceRowStart := {}, finalRowStart := {},\n    \
            constantValues := constantValues, constantStartColumn := {},\n    \
            inputColumns := {:?}, hashOutputColumns := {:?},\n    \
            carriedBindingColumns := {:?}, equalityRowStart := {} }}\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullProgramBinding\n",
        constant_values,
        full_rows.len(),
        full_source.cols(),
        full_rows.start,
        full_rows.start,
        full_constant_start,
        input_columns,
        hash_output_columns,
        carried_binding_columns,
        equality_row_start,
    )
}
