use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, Var};
use neo_fold_clean::paper::f_prime::r1cs::{enforce_public_bits_encode_digest, F_PRIME_ENC_INST_BITS};

use super::*;

struct EncodingEmbedding {
    column_map: Vec<usize>,
    row_start: usize,
    row_end: usize,
}

fn isolated_encoding() -> (R1csBuilder, [usize; 4], Vec<usize>) {
    let mut builder = R1csBuilder::new();
    let digest = std::array::from_fn(|_| builder.alloc(F::ZERO));
    let public_bits = (0..F_PRIME_ENC_INST_BITS)
        .map(|_| builder.alloc(F::ZERO))
        .collect::<Vec<_>>();
    enforce_public_bits_encode_digest(&mut builder, &public_bits, &digest).expect("emit isolated F' output encoding");
    (
        builder,
        digest.map(Var::col),
        public_bits.into_iter().map(Var::col).collect(),
    )
}

fn embedding(builder: &R1csBuilder, audit: &neo_fold_clean::engine::decider::FPrimeStepWireAudit) -> EncodingEmbedding {
    let (isolated, isolated_digest, isolated_public) = isolated_encoding();
    assert_eq!(isolated.rows(), 532, "canonical F' output encoding row count");
    assert_eq!(audit.x_out_bit_columns.len(), F_PRIME_ENC_INST_BITS);

    let mut column_map = vec![usize::MAX; isolated.cols()];
    column_map[0] = 0;
    for lane in 0..4 {
        column_map[isolated_digest[lane]] = audit.x_out_columns[lane];
    }
    for (local, &global) in isolated_public.iter().zip(&audit.x_out_bit_columns) {
        column_map[*local] = global;
    }

    let isolated_decompositions = isolated.canonical_u64_audits();
    let production_decompositions = builder.canonical_u64_audits();
    assert_eq!(isolated_decompositions.len(), 4);
    for (lane, local) in isolated_decompositions.iter().enumerate() {
        assert_eq!(local.field_col, isolated_digest[lane]);
        let global = production_decompositions
            .iter()
            .find(|candidate| candidate.field_col == audit.x_out_columns[lane])
            .unwrap_or_else(|| panic!("canonical decomposition for recursive x_out lane {lane}"));
        for (local_bit, global_bit) in local.bit_cols.iter().zip(global.bit_cols) {
            column_map[*local_bit] = global_bit;
        }
        column_map[local.bit_cols[63] + 1] = global.bit_cols[63] + 1;
        column_map[local.bit_cols[63] + 2] = global.bit_cols[63] + 2;
    }
    assert!(
        column_map.iter().all(|column| *column != usize::MAX),
        "every isolated output-encoding column must be embedded"
    );

    let row_end = audit.row_end;
    let row_start = row_end - isolated.rows();
    EncodingEmbedding {
        column_map,
        row_start,
        row_end,
    }
}

fn assert_exact_embedding(
    builder: &R1csBuilder,
    audit: &neo_fold_clean::engine::decider::FPrimeStepWireAudit,
    embedded: &EncodingEmbedding,
) {
    let (isolated, _, _) = isolated_encoding();
    let (expected_a, expected_b, expected_c) = isolated.sparse_triplets();
    let (actual_a, actual_b, actual_c) = builder.sparse_triplets();
    for (name, expected, actual) in [
        ("A", expected_a, actual_a),
        ("B", expected_b, actual_b),
        ("C", expected_c, actual_c),
    ] {
        let expected = expected
            .iter()
            .map(|&(row, column, coefficient)| (embedded.row_start + row, embedded.column_map[column], coefficient))
            .collect::<Vec<_>>();
        let actual = actual
            .iter()
            .copied()
            .filter(|(row, _, _)| embedded.row_start <= *row && *row < embedded.row_end)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "recursive output encoding {name} rows");
    }
    assert_eq!(embedded.row_end, audit.row_end);
}

pub fn render_output_encoding_artifact(
    builder: &R1csBuilder,
    audit: &neo_fold_clean::engine::decider::FPrimeStepWireAudit,
) -> String {
    let embedded = embedding(builder, audit);
    assert_exact_embedding(builder, audit, &embedded);
    format!(
        "import Nightstream.Implementation.R1CS.Ownership.FPrime.FPrimeEncodingArtifact\n\
         import Nightstream.Implementation.R1CS.Core.Relabel\n\n\
         /-! Generated exact recursive-output `enc_inst(x_out)` embedding. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncoding\n\n\
         open Nightstream.Implementation.R1CS\n\n\
         def rowStart : Nat := {}\n\
         def rowEnd : Nat := {}\n\
         def rowCount : Nat := {}\n\
         def columnMap : List Nat := {}\n\
         def rows : List Row := FPrimeEncoding.rows.map (Relabel.row columnMap)\n\n\
         theorem rows_length : rows.length = rowCount := by native_decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncoding\n",
        embedded.row_start,
        embedded.row_end,
        embedded.row_end - embedded.row_start,
        lean_nat_list(embedded.column_map),
    )
}
