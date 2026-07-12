//! Exact Lean artifact and adversarial vector for the production SIS field
//! encoding. The artifact includes the canonical shifted-base-3 decomposition
//! and one complete seeded Phi81 commitment coordinate.

#[path = "lean_artifact_support.rs"]
mod lean_artifact_support;

use std::fmt::Write as _;

use lean_artifact_support::{lean_nat_list, lean_rows, lean_witness, sha256_hex, SCHEMA_VERSION};
use neo_ajtai::commit_row_major_seeded;
use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, Var};
use neo_fold_clean::paper::reductions::accumulator_sis_circuit::{enforce_commit_fields, SisAccumulatorConfig};
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xA7; 32],
    kappa: 1,
    domain: 0x5349_5354_4553_5431,
};
const SAMPLE: u64 = F::ORDER_U64 / 2;
const ARTIFACT_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/ShiftedTernary/Generated/ShiftedTernaryArtifact.lean";

fn build(value: u64) -> (R1csBuilder, Var, Vec<Var>) {
    let mut builder = R1csBuilder::new();
    let field = builder.alloc(F::from_u64(value));
    let commitment = enforce_commit_fields(&mut builder, CONFIG, &[field]).expect("one-field SIS commitment");
    (builder, field, commitment.data)
}

fn forged() -> R1csBuilder {
    let (mut builder, field, commitment_columns) = build(SAMPLE);
    let modulus = F::ORDER_U64 as u128;
    let shift = (3u128.pow(41) - 1) / 2;
    let canonical_n = (SAMPLE as u128 + shift) % modulus;
    let mut remaining = canonical_n + modulus;
    assert!(remaining < 3u128.pow(41));
    let digits: [F; 41] = core::array::from_fn(|_| {
        let trit = remaining % 3;
        remaining /= 3;
        match trit {
            0 => -F::ONE,
            1 => F::ZERO,
            2 => F::ONE,
            _ => unreachable!("base-3 digit"),
        }
    });
    assert_eq!(remaining, 0);

    let digit_columns = builder
        .balanced_ternary_digit_columns(field)
        .expect("balanced ternary audit");
    let negative_start = digit_columns[40] + 1;
    let borrow_start = negative_start + 41;
    for (index, (&column, &digit)) in digit_columns.iter().zip(&digits).enumerate() {
        builder.tamper_witness(column, digit);
        builder.tamper_witness(negative_start + index, if digit == -F::ONE { F::ONE } else { F::ZERO });
    }

    let mut bound = F::ORDER_U64 - 1;
    let mut borrow = false;
    for (index, &digit) in digits.iter().enumerate() {
        let trit = if digit == -F::ONE {
            0
        } else if digit == F::ZERO {
            1
        } else {
            2
        };
        let next = trit + u64::from(borrow) > bound % 3;
        bound /= 3;
        if index + 1 < 41 {
            builder.tamper_witness(borrow_start + index, if next { F::ONE } else { F::ZERO });
        } else {
            assert!(next);
        }
        borrow = next;
    }

    let mut message = Mat::zero(D, 1, F::ZERO);
    for (row, digit) in digits.into_iter().enumerate() {
        message[(row, 0)] = digit;
    }
    let forged_commitment = commit_row_major_seeded(CONFIG.seed, D, CONFIG.kappa, 1, &message);
    for (column, value) in commitment_columns.iter().zip(forged_commitment.data) {
        builder.tamper_witness(column.col(), value);
    }
    builder
}

fn lean_seed_rows(rows: &[Vec<[u8; 32]>]) -> String {
    let rows = rows
        .iter()
        .map(|chunks| {
            let chunks = chunks
                .iter()
                .map(|seed| {
                    let bytes = seed.iter().map(u8::to_string).collect::<Vec<_>>();
                    format!("[{}]", bytes.join(", "))
                })
                .collect::<Vec<_>>();
            format!("[{}]", chunks.join(", "))
        })
        .collect::<Vec<_>>();
    format!("[{}]", rows.join(", "))
}

fn render(builder: &R1csBuilder, field: Var, commitment: &[Var], forged_witness: &[F]) -> String {
    let digit_columns = builder
        .balanced_ternary_digit_columns(field)
        .expect("balanced ternary audit");
    let negative_columns = (digit_columns[40] + 1..digit_columns[40] + 42).collect::<Vec<_>>();
    let borrow_columns = (digit_columns[40] + 42..digit_columns[40] + 82).collect::<Vec<_>>();
    let block = builder
        .seeded_phi81_a_blocks()
        .first()
        .expect("one seeded Phi81 block");
    assert_eq!(builder.seeded_phi81_a_blocks().len(), 1);
    assert_eq!(block.row_end(), builder.rows());
    let payload = format!(
        "def fieldCol : Nat := {}\n\
         def commitmentDCol : Nat := 2\n\
         def commitmentKappaCol : Nat := 3\n\
         def commitmentCols : List Nat := {}\n\
         def commitmentBlock : SeededPhi81.Block :=\n\
         \x20\x20{{ rowStart := {}\n\
         \x20\x20\x20\x20wordStarts := {}\n\
         \x20\x20\x20\x20wordWidth := {}\n\
         \x20\x20\x20\x20kappa := {}\n\
         \x20\x20\x20\x20messageCols := {}\n\
         \x20\x20\x20\x20outputColumns := {}\n\
         \x20\x20\x20\x20superneoTransformedColumns := {}\n\
         \x20\x20\x20\x20schedule :=\n\
         \x20\x20\x20\x20\x20\x20{{ chunkSize := {}\n\
         \x20\x20\x20\x20\x20\x20\x20\x20seedsByOutput := {}\n\
         \x20\x20\x20\x20\x20\x20\x20\x20rejectionFuel := 16 }} }}\n\
         def digitCols : List Nat := {}\n\
         def negativeCols : List Nat := {}\n\
         def borrowCols : List Nat := {}\n\
         def rowCount : Nat := {}\n\
         def colCount : Nat := {}\n\
         def rows : List Row :=\n  {}\n\n\
         {}\n\
         {}",
        field.col(),
        lean_nat_list(commitment.iter().map(|column| column.col())),
        block.row_start(),
        lean_nat_list(block.word_starts().iter().copied()),
        block.word_width(),
        block.kappa(),
        block.message_cols(),
        lean_nat_list(commitment.iter().map(|column| column.col())),
        block.has_superneo_transformed_columns(),
        block.chunk_size(),
        lean_seed_rows(block.chunk_seeds_by_row()),
        lean_nat_list(digit_columns),
        lean_nat_list(negative_columns),
        lean_nat_list(borrow_columns),
        builder.rows(),
        builder.cols(),
        lean_rows(builder),
        lean_witness("honestWitness", builder.witness()),
        lean_witness("forgedWitness", forged_witness),
    );
    let mut rendered = String::new();
    rendered.push_str("import Nightstream.Implementation.R1CS.Core.SeededPhi81\n\n");
    rendered.push_str("/-! Generated exact one-field shifted-base-3/SIS artifact. Do not hand-edit. -/\n\n");
    rendered.push_str("namespace Nightstream.Implementation.R1CS.ShiftedTernary\n\n");
    writeln!(rendered, "def schemaVersion : Nat := {SCHEMA_VERSION}").expect("render");
    writeln!(rendered, "def payloadSha256 : String := \"{}\"", sha256_hex(&payload)).expect("render");
    rendered.push_str(&payload);
    rendered.push_str("\n\nend Nightstream.Implementation.R1CS.ShiftedTernary\n");
    rendered
}

#[test]
fn honest_and_forged_boundaries_are_pinned() {
    let (builder, _, _) = build(SAMPLE);
    assert_eq!((builder.rows(), builder.cols()), (180, 180));
    assert!(builder.is_satisfied());
    assert!(builder.unconstrained_columns().is_empty());
    let forged = forged();
    assert_eq!(forged.first_unsatisfied_row(), Some(125));
}

#[test]
fn shifted_ternary_artifact_matches_committed_file() {
    let (builder, field, commitment) = build(SAMPLE);
    let forged = forged();
    let emitted = render(&builder, field, &commitment, forged.witness());
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_REL_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed != emitted {
        let expected = format!("{path}.expected");
        std::fs::write(&expected, emitted).expect("write shifted-ternary artifact");
        panic!("shifted-ternary Lean artifact drifted; wrote {expected}");
    }
}
