//! Standalone pilot assignment conformance on the canonical package rows.
//! The serialized pilot fixture supplies both preimages and the public values;
//! no PiCCS proof or later phase input is constructed.

use std::{fs, time::Instant};

use serde::Deserialize;

use super::pi_ccs_prefix_assignment_tests::{
    artifact_path, conformance_support, execute_pilot_prefix, inner_raw_package_bytes, logical_reference,
    PrefixExecutionCounts,
};
use crate::{load_poseidon2_hash_chain_v1_package, WitnessAssignment, PI_CCS_V1_1_STATE_PREIMAGE_WORDS};

// PilotProduction.physicalRowCountValue_eq gives 14,623,730 rows.
// The sealed Stage1 source map places its witness end at 14,751,526.
const PILOT_ROWS: usize = 14_623_730;
const PILOT_PRIVATE_END: usize = 14_751_526;
const PILOT_PUBLIC_WORDS: usize = 274;

// PerApplicationMatrixProgram orders pilotPoseidon before PiCCS, then
// pilotOrdinary and pilotDigestBinding after the PiCCS core. These are the
// complete pilot-owned ranges in that one final matrix program.
const PILOT_LOGICAL_RANGES: [(usize, usize); 3] = [(0, 2_321_800), (3_863_453, 3_864_783), (3_864_783, 3_864_791)];
const PILOT_BLOCK_ENDS: [usize; 7] = [
    1_160_900, 2_321_800, 3_036_576, 3_051_784, 3_863_453, 3_864_783, 3_864_791,
];
const PILOT_BLOCK_OPCODES: [usize; 7] = [2, 2, 2, 1, 0, 0, 1];
const PILOT_LOGICAL_FAMILIES: [(&str, usize, usize); 4] = [
    ("pilotPoseidon.prior", 0, PILOT_BLOCK_ENDS[0]),
    ("pilotPoseidon.output", PILOT_BLOCK_ENDS[0], PILOT_BLOCK_ENDS[1]),
    ("pilotOrdinary", PILOT_BLOCK_ENDS[4], PILOT_BLOCK_ENDS[5]),
    ("pilotDigestBinding", PILOT_BLOCK_ENDS[5], PILOT_BLOCK_ENDS[6]),
];

#[derive(Deserialize)]
struct RawPilotInput(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>);

#[derive(Deserialize)]
struct RawPilotResult(Vec<u64>, Vec<u64>, Vec<u64>, Vec<[u64; 3]>, Vec<u64>);

#[derive(Deserialize)]
struct RawPilotParity(u64, RawPilotInput, RawPilotResult);

fn check_pilot_preimage(words: &[u64], context: [u64; 4]) {
    assert_eq!(words.len(), PI_CCS_V1_1_STATE_PREIMAGE_WORDS);
    let domain: Vec<u64> = b"HyperNova/NIVC/state/v1"
        .iter()
        .copied()
        .map(u64::from)
        .collect();
    assert_eq!(&words[..domain.len()], domain);
    assert_eq!(words[domain.len()], 4);
    assert_eq!(&words[domain.len() + 1..domain.len() + 5], context);

    // The Lean PilotZeroRunning theorem supplies the bounded zero openings.
    // Check that this serialized input carries those exact running claims.
    let mut cursor = 39;
    for length in std::iter::once(56).chain((0..16).flat_map(|_| [1_188, 270, 1_620])) {
        assert_eq!(words[cursor], length as u64, "pilot running-field length");
        cursor += 1;
        assert!(
            words[cursor..cursor + length].iter().all(|word| *word == 0),
            "pilot zero-running values"
        );
        cursor += length;
    }
    assert_eq!(cursor + 1, words.len(), "pilot program-counter boundary");
}

fn check_logical_family_mutations(
    program: &logical_reference::matrix::MatrixProgram,
    sources: &logical_reference::source::SourcePackage,
    relation: &logical_reference::relation::Relation,
    first_public_padding: usize,
    value_at: &mut impl FnMut(usize) -> logical_reference::Result<logical_reference::Field>,
) -> logical_reference::Result<(usize, usize)> {
    use logical_reference::{evaluation::evaluate_row_with_result, Field, Form, MATRIX_COUNT};

    // The final encHash public prefix supplies a fixed one and fixed padding.
    // Column mutations change a decoded sparse entry to one of these columns.
    if value_at(0)? != Field::ONE || value_at(first_public_padding)? != Field::ZERO {
        return Err("pilot logical public one/padding values are not canonical".into());
    }
    let mut row_mutations = 0;
    let mut column_mutations = 0;
    for (name, start, end) in PILOT_LOGICAL_FAMILIES {
        let mut row_rejected = None;
        let mut column_rejected = None;
        for ordinal in start..end {
            let row = program.row(ordinal, sources)?;
            if relation.evaluate(&evaluate_row_with_result(&row, value_at)?) != Field::ZERO {
                return Err(format!("pilot family {name} original row {ordinal} does not hold"));
            }
            if row_rejected.is_none() {
                for matrix in 0..MATRIX_COUNT - 1 {
                    let mut changed = row.clone();
                    changed[matrix] = changed[matrix]
                        .clone()
                        .append(Form::singleton(0, Field::ONE));
                    if relation.evaluate(&evaluate_row_with_result(&changed, value_at)?) != Field::ZERO {
                        row_rejected = Some((ordinal, matrix));
                        break;
                    }
                }
            }
            if column_rejected.is_none() {
                'matrices: for (matrix, form) in row.iter().enumerate() {
                    for (entry_index, entry) in form.entries().iter().enumerate() {
                        for target in [0, first_public_padding] {
                            if entry.column == target {
                                continue;
                            }
                            let mut entries = form.entries().to_vec();
                            entries[entry_index].column = target;
                            let mut changed = row.clone();
                            changed[matrix] = Form::from_entries(entries);
                            if relation.evaluate(&evaluate_row_with_result(&changed, value_at)?) != Field::ZERO {
                                column_rejected = Some((ordinal, matrix, entry.column, target));
                                break 'matrices;
                            }
                        }
                    }
                }
            }
            if row_rejected.is_some() && column_rejected.is_some() {
                break;
            }
        }
        let (row, matrix) =
            row_rejected.ok_or_else(|| format!("pilot family {name} has no assignment-rejecting row mutation"))?;
        println!("pilot logical {name}: row {row} matrix {matrix} constant coefficient mutation rejected");
        let (row, matrix, source, target) = column_rejected
            .ok_or_else(|| format!("pilot family {name} has no assignment-rejecting column mutation"))?;
        println!("pilot logical {name}: row {row} matrix {matrix} column {source} -> {target} rejected");
        row_mutations += 1;
        column_mutations += 1;
    }
    Ok((row_mutations, column_mutations))
}

#[test]
#[ignore = "exact standalone pilot physical and logical assignment gate; run explicitly under the 300-second cap"]
fn sealed_package_checks_the_standalone_pilot_assignment() {
    let started = Instant::now();
    let sealed_bytes = fs::read(artifact_path("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"))
        .expect("sealed canonical Stage 1 package");
    let parity_bytes = fs::read(artifact_path("nightstream-fprime-stage1-pilot-parity-v1.json"))
        .expect("standalone Lean pilot fixture");
    let RawPilotParity(schema, RawPilotInput(prior, prior_public, output, output_digest), expected) =
        serde_json::from_slice(&parity_bytes).expect("standalone pilot parity decode");
    assert_eq!(schema, 1);
    assert_eq!(prior_public.len(), 270);
    assert_eq!(output_digest.len(), 4);
    assert!(prior
        .iter()
        .chain(&prior_public)
        .chain(&output)
        .chain(&output_digest)
        .all(|word| *word < super::super::GOLDILOCKS_MODULUS));
    assert_eq!(expected.0.len(), 4);
    assert_eq!(expected.1, output_digest);
    assert_eq!(expected.3, [[4, 0, 270], [5, 270, 4]]);
    assert_eq!(expected.4, [1, 1, 1, 1]);
    assert_ne!(expected.0, expected.1);

    let package = load_poseidon2_hash_chain_v1_package(&sealed_bytes).expect("verifier-owned production package");
    let context = package
        .production_verifier_binding()
        .expect("production verifier binding")
        .verifier_context()
        .digest();
    check_pilot_preimage(&prior, context);
    check_pilot_preimage(&output, context);
    let mut private_values = prior;
    private_values.extend(output);
    let mut public_values = prior_public;
    public_values.extend(output_digest);
    assert_eq!(public_values, expected.2, "exact serialized Lean pilot public vector");
    public_values.extend(context);
    println!("standalone pilot package and input decode: {:?}", started.elapsed());

    let assignment =
        execute_pilot_prefix(&package, &private_values, &public_values).expect("package IR pilot prefix assignment");
    assert_eq!(assignment.private_values.len(), PILOT_PRIVATE_END);
    assert_eq!(assignment.public_values, public_values);
    assert_eq!(
        assignment.counts,
        PrefixExecutionCounts {
            permutations: 24_700,
            compact_invocations: 0,
            witness_batches: 12,
            generic_instructions: 788,
        }
    );
    assert_eq!(package.circuit.hash_chains.len(), 2);
    for (chain, digest) in package
        .circuit
        .hash_chains
        .iter()
        .zip([&expected.0, &expected.1])
    {
        let start = chain.witness_start
            + chain.absorb_count * package.circuit.permutation.local_column_count
            + package.circuit.permutation.output_local_start;
        assert_eq!(
            &assignment.private_values[start..start + digest.len()],
            digest,
            "package-generated pilot digest equals the serialized Lean result"
        );
    }
    drop(private_values);
    println!("standalone pilot witness execution and result: {:?}", started.elapsed());

    let raw_bytes = inner_raw_package_bytes(&sealed_bytes);
    let report = conformance_support::evaluate_pilot_assignment(
        &raw_bytes,
        &assignment.private_values,
        &assignment.public_values,
    )
    .expect("every independent canonical pilot physical row holds");
    assert_eq!(report.rows, PILOT_ROWS);
    assert_eq!(report.public_mutations, PILOT_PUBLIC_WORDS);
    // For each hash owner: its first generated value and all four digest lanes.
    assert_eq!(report.generated_mutations, 2 * (1 + 4));
    // One effective decoded-row and column-reference mutation per hash owner.
    assert_eq!(report.row_mutations, 2);
    assert_eq!(report.column_mutations, 2);
    drop(raw_bytes);
    println!("standalone pilot physical rows and mutations: {:?}", started.elapsed());

    // The existing production transport takes a complete physical vector.
    // Extend only this test input with zeros; this is not a complete witness.
    // The independent view below keeps the proof gap and suffix unavailable,
    // so every logical coordinate used by a pilot row must decode without them.
    let mut transport_private = assignment.private_values.clone();
    transport_private.resize(package.circuit.layout.private_column_count, 0);
    let transport_input = WitnessAssignment {
        private_values: transport_private,
        public_values: assignment.public_values.clone(),
    };
    let production_logical_assignment = package
        .execute_logical_assignment(&transport_input)
        .expect("production logical transport on the zero-extended pilot input");
    drop(transport_input);
    drop(package);
    println!("standalone pilot production logical transport: {:?}", started.elapsed());

    let relation =
        logical_reference::relation::Relation::decode(&sealed_bytes).expect("independent final relation decoder");
    let artifact =
        logical_reference::source::SourcePackage::decode(&sealed_bytes).expect("independent sealed source decoder");
    let program = logical_reference::matrix::MatrixProgram::decode(
        &artifact.matrix_program,
        &artifact.sources,
        artifact.logical_columns,
        artifact.logical_rows,
    )
    .expect("independent final matrix-program decoder");
    assert_eq!(
        program
            .block_ends()
            .take(PILOT_BLOCK_ENDS.len())
            .collect::<Vec<_>>(),
        PILOT_BLOCK_ENDS
    );
    assert_eq!(
        program
            .block_opcodes()
            .take(PILOT_BLOCK_OPCODES.len())
            .collect::<Vec<_>>(),
        PILOT_BLOCK_OPCODES
    );
    let logical_assignment = logical_reference::assignment::PartialLogicalAssignment::decode_pilot(
        &sealed_bytes,
        &assignment.private_values,
        &assignment.public_values,
    )
    .expect("independent pilot logical assignment with unavailable non-pilot columns");
    assert_eq!(logical_assignment.len(), artifact.logical_columns);
    assert_eq!(production_logical_assignment.len(), artifact.logical_columns);
    println!("standalone pilot logical relation decode: {:?}", started.elapsed());

    // Compare each immutable production coordinate once, before recording it.
    // The bitset covers the exact package width; every row is still evaluated.
    let mut compared_columns = vec![0u64; artifact.logical_columns.div_ceil(u64::BITS as usize)];
    let mut checked_production_value = |column| {
        let production = production_logical_assignment
            .value(column)
            .map_err(|error| format!("production pilot logical column {column}: {error}"))?;
        let production = logical_reference::Field::checked(production, "production pilot logical value")?;
        let bit = 1u64 << (column % u64::BITS as usize);
        let compared = compared_columns
            .get_mut(column / u64::BITS as usize)
            .ok_or_else(|| format!("pilot logical column {column} exceeds the package width"))?;
        if *compared & bit == 0 {
            let independent = logical_assignment.value(column)?;
            if production != independent {
                return Err(format!(
                    "production pilot logical assignment differs at column {column}: {} != {}",
                    production.canonical(),
                    independent.canonical()
                ));
            }
            *compared |= bit;
        }
        Ok(production)
    };
    let mut logical_rows = 0;
    for (start, end) in PILOT_LOGICAL_RANGES {
        logical_rows += logical_reference::evaluation::verify_satisfaction_range_with(
            &program,
            &artifact.sources,
            &relation,
            start,
            end,
            &mut checked_production_value,
        )
        .expect("production logical values match the independent lift and satisfy every pilot-owned row");
        println!("standalone pilot logical rows {start}..{end}: {:?}", started.elapsed());
    }
    assert_eq!(logical_rows, 2_323_138);
    let (logical_row_mutations, logical_column_mutations) = check_logical_family_mutations(
        &program,
        &artifact.sources,
        &relation,
        1 + expected.1.len() * u64::BITS as usize,
        &mut checked_production_value,
    )
    .expect("every named pilot logical family has effective canonical row and column mutations");
    assert_eq!(logical_row_mutations, PILOT_LOGICAL_FAMILIES.len());
    assert_eq!(logical_column_mutations, PILOT_LOGICAL_FAMILIES.len());
    println!(
        "standalone pilot canonical row/column mutations: {:?}",
        started.elapsed()
    );
}
