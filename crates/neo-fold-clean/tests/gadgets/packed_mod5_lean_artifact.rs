//! Exact active packed-Mod-5 Rust-to-Lean artifact and drift gate.
//!
//! Owns: role-normalization of one production sampler's 64 identical Mod-5
//! source blocks, projected decoder LCs, eight emitted rows, and the 12-term
//! CCS polynomial specialization.
//!
//! Does not own: selector composition, the inactive selector row, transcript
//! authority, or the Lean semantic equivalence proof.
//!
//! Supported profile: the isolated one-rho, 64-chunk
//! `enforce_alphabet_sample_5_d` fixture. Full-F' contains 960 chunks and needs
//! a separate global placement/outer-image artifact.
//!
//! Emits constraints: no. It reads the production source and lowered CCS.
//!
//! Authority boundary: exact rows, decoder terms, matrices, and polynomial
//! terms are compared directly. The generated Lean file contains no authority
//! digest.
//!
//! | Artifact branch | Exact production evidence | Lean owner |
//! |---|---|---|
//! | `sourceRows` | 20 normalized R1CS rows, identical across 64 chunks | `PackedMod5Artifact` |
//! | `decoderDefinitions` | normalized `index`, high-bit, quotient LCs plus product chain | `PackedMod5Artifact` |
//! | `activeRows` | six low pairs, one high pair, one residue pair | `PackedMod5Artifact` |
//! | `polynomialTerms` | production-exported arity/roles and twelve exact terms | `PackedMod5Artifact` |

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use neo_ccs::{CcsMatrix, CscMat};
use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::enforce_alphabet_sample_5_d;
use neo_fold_clean::engine::r1cs_circuit::{Mod5TraceEntry, R1csBuilder, R1csSnapshot, TranscriptGadget, Var};
use neo_fold_clean::frontends::f_prime::gadget_native::{
    encode_r1cs_gadget_native, EncodedGadgetNativeR1cs, GadgetNativeCoordinateGateRoles,
};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

// Reuse the focused conformance fixture so stage boundaries and all five
// residues remain identical to the production-packed test census.
const APP: &[u8] = b"packed-mod5-gadget-native-test";
const CHUNKS: usize = 64;
const SOURCE_ROWS: usize = 20;
const SOURCE_COLUMNS: usize = 19;
const GATE_ARITY: usize = GadgetNativeCoordinateGateRoles::ARITY;
const SELECTOR_MATRIX: usize = GadgetNativeCoordinateGateRoles::SELECTOR;
const BIT_LEFT_MATRIX: usize = GadgetNativeCoordinateGateRoles::BOOLEAN_PAIR_LEFT;
const BIT_RIGHT_MATRIX: usize = GadgetNativeCoordinateGateRoles::BOOLEAN_PAIR_RIGHT;
const RESIDUE_LEFT_MATRIX: usize = GadgetNativeCoordinateGateRoles::MOD5_RESIDUE_LEFT;
const RESIDUE_RIGHT_MATRIX: usize = GadgetNativeCoordinateGateRoles::MOD5_RESIDUE_RIGHT;
const PACKED_MATRICES: [usize; 5] = [
    SELECTOR_MATRIX,
    BIT_LEFT_MATRIX,
    BIT_RIGHT_MATRIX,
    RESIDUE_LEFT_MATRIX,
    RESIDUE_RIGHT_MATRIX,
];
const ARTIFACT_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/PiRlcChallenge/Generated/PackedMod5ArtifactData.lean";

#[derive(Clone, Debug, PartialEq, Eq)]
struct RoleTerm {
    role: String,
    coefficient: i128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RoleRow {
    a: Vec<RoleTerm>,
    b: Vec<RoleTerm>,
    c: Vec<RoleTerm>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DecoderAudit {
    index: Vec<RoleTerm>,
    high: Vec<RoleTerm>,
    quotient: Vec<RoleTerm>,
    products: Vec<ProductDecoderAudit>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ProductDecoderAudit {
    left: Vec<RoleTerm>,
    right: Vec<RoleTerm>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PolynomialAuditTerm {
    coefficient: i128,
    powers: Vec<(&'static str, u32)>,
}

fn sampler_builder() -> R1csBuilder {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    builder.begin_encoding_stage("test.packed_mod5_artifact");
    let mut transcript = TranscriptGadget::new(&mut builder, APP);
    let _symbols = enforce_alphabet_sample_5_d(&mut builder, &mut transcript, 7);
    builder.begin_encoding_stage("complete");
    assert!(builder.is_satisfied(), "production sampler source relation");
    builder
}

fn signed(coefficient: F) -> i128 {
    let canonical = coefficient.as_canonical_u64() as i128;
    let modulus = F::ORDER_U64 as i128;
    if canonical > modulus / 2 {
        canonical - modulus
    } else {
        canonical
    }
}

fn source_role_map(event: &Mod5TraceEntry) -> BTreeMap<usize, String> {
    let mut roles = BTreeMap::from([(Var::ONE.col(), ".one".to_owned())]);
    for (index, variable) in event.chunk_bits.iter().enumerate() {
        assert!(roles
            .insert(variable.col(), format!("(SourceRole.chunkBit {index})"))
            .is_none());
    }
    let allocated = [
        (event.index, ".index".to_owned()),
        (event.quotient, ".quotient".to_owned()),
        (event.index_products[0], "(SourceRole.indexProduct 0)".to_owned()),
        (event.index_products[1], "(SourceRole.indexProduct 1)".to_owned()),
        (event.index_products[2], "(SourceRole.indexProduct 2)".to_owned()),
    ];
    for (variable, role) in allocated {
        assert!(roles.insert(variable.col(), role).is_none());
    }
    for (index, variable) in event.quotient_bits.iter().enumerate() {
        assert!(roles
            .insert(variable.col(), format!("(SourceRole.quotientBit {index})"))
            .is_none());
    }
    roles
}

fn role_terms(row: &[(usize, F)], roles: &BTreeMap<usize, String>) -> Vec<RoleTerm> {
    row.iter()
        .map(|&(column, coefficient)| RoleTerm {
            role: roles
                .get(&column)
                .unwrap_or_else(|| panic!("unowned source column {column}"))
                .clone(),
            coefficient: signed(coefficient),
        })
        .collect()
}

fn source_schema(source: &R1csSnapshot, event: &Mod5TraceEntry) -> Vec<RoleRow> {
    assert_eq!(event.source_rows.len(), SOURCE_ROWS);
    assert_eq!(event.allocated_columns.len(), SOURCE_COLUMNS);
    let roles = source_role_map(event);
    event
        .source_rows
        .clone()
        .map(|row| RoleRow {
            a: role_terms(source.a_row(row), &roles),
            b: role_terms(source.b_row(row), &roles),
            c: role_terms(source.c_row(row), &roles),
        })
        .collect()
}

fn decoder_atom_map(
    encoded: &EncodedGadgetNativeR1cs,
    event: &Mod5TraceEntry,
    chunk: usize,
) -> BTreeMap<usize, String> {
    let mut roles = BTreeMap::from([(0, ".source .one".to_owned())]);
    for (index, variable) in event.chunk_bits.iter().enumerate() {
        let range = encoded
            .plan
            .encoded_range_for_source_column(variable.col())
            .expect("isolated chunk input has one exact encoded cell");
        assert_eq!(range.len(), 1);
        assert!(roles
            .insert(range.start, format!(".source (.chunkBit {index})"))
            .is_none());
    }
    let low = encoded
        .plan
        .packed_mod5_low_bit_range(chunk)
        .expect("thirteen packed low bits");
    assert_eq!(low.len(), 13);
    for (index, column) in low.enumerate() {
        assert!(roles
            .insert(column, format!(".coordinate (.quotientLow {index})"))
            .is_none());
    }
    let residue = encoded
        .plan
        .packed_mod5_residue_range(chunk)
        .expect("two packed centered coordinates");
    assert_eq!(residue.len(), 2);
    assert!(roles
        .insert(residue.start, ".coordinate .residueLeft".to_owned())
        .is_none());
    assert!(roles
        .insert(residue.start + 1, ".coordinate .residueRight".to_owned())
        .is_none());
    roles
}

fn decoder_terms(terms: &[(usize, F)], roles: &BTreeMap<usize, String>) -> Vec<RoleTerm> {
    terms
        .iter()
        .map(|&(column, coefficient)| RoleTerm {
            role: roles
                .get(&column)
                .unwrap_or_else(|| panic!("unowned decoder coordinate {column}"))
                .clone(),
            coefficient: signed(coefficient),
        })
        .collect()
}

fn product_decoder_terms(terms: &[(usize, F)], constant: F, roles: &BTreeMap<usize, String>) -> Vec<RoleTerm> {
    let mut terms = terms
        .iter()
        .map(|&(column, coefficient)| RoleTerm {
            role: format!(
                ".source {}",
                roles
                    .get(&column)
                    .unwrap_or_else(|| panic!("unowned product-decoder source column {column}"))
            ),
            coefficient: signed(coefficient),
        })
        .collect::<Vec<_>>();
    if constant != F::ZERO {
        terms.push(RoleTerm {
            role: ".source .one".to_owned(),
            coefficient: signed(constant),
        });
    }
    terms
}

fn decoder_schema(encoded: &EncodedGadgetNativeR1cs, event: &Mod5TraceEntry, chunk: usize) -> DecoderAudit {
    let coordinate_roles = decoder_atom_map(encoded, event, chunk);
    let source_roles = source_role_map(event);
    let audit = encoded
        .plan
        .packed_mod5_decoder_audit(chunk)
        .expect("role-specific packed Mod-5 decoder audit");
    let products = audit
        .products
        .iter()
        .enumerate()
        .map(|(index, product)| {
            assert_eq!(product.output, event.index_products[index].col());
            ProductDecoderAudit {
                left: product_decoder_terms(product.left_terms, product.left_constant, &source_roles),
                right: product_decoder_terms(product.right_terms, product.right_constant, &source_roles),
            }
        })
        .collect();
    DecoderAudit {
        index: decoder_terms(audit.index, &coordinate_roles),
        high: decoder_terms(audit.high, &coordinate_roles),
        quotient: decoder_terms(audit.quotient, &coordinate_roles),
        products,
    }
}

fn csc_row(matrix: &CscMat<F>, row: usize) -> Vec<(usize, F)> {
    let mut terms = Vec::new();
    for column in 0..matrix.ncols {
        for entry in matrix.column_range(column) {
            if matrix.row_index(entry) == row {
                terms.push((column, matrix.vals[entry]));
            }
        }
    }
    terms
}

fn matrix_row(matrix: &CcsMatrix<F>, row: usize) -> Vec<(usize, F)> {
    match matrix {
        CcsMatrix::Csc(matrix) => csc_row(matrix, row),
        CcsMatrix::Identity { n } => {
            assert!(row < *n);
            vec![(row, F::ONE)]
        }
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            assert!(
                blocks.is_empty() && geometric_runs.is_empty(),
                "Mod-5 gate matrices are ordinary CSC"
            );
            csc_row(csc, row)
        }
        CcsMatrix::VerifierArtifact { .. } => {
            panic!("Mod-5 gate matrices cannot use a verifier artifact")
        }
    }
}

fn packed_rows(encoded: &EncodedGadgetNativeR1cs) -> Vec<usize> {
    let mut low_bit_columns = BTreeSet::new();
    let mut residue_left_columns = BTreeSet::new();
    for chunk in 0..CHUNKS {
        low_bit_columns.extend(
            encoded
                .plan
                .packed_mod5_low_bit_range(chunk)
                .expect("role-specific packed Mod-5 low-bit range"),
        );
        residue_left_columns.insert(
            encoded
                .plan
                .packed_mod5_residue_range(chunk)
                .expect("role-specific packed Mod-5 residue range")
                .start,
        );
    }
    let mut rows = BTreeSet::new();
    for row in 0..encoded.structure.n {
        let is_mod5_bit_pair = matrix_row(&encoded.structure.matrices[BIT_LEFT_MATRIX], row)
            .iter()
            .any(|(column, _)| low_bit_columns.contains(column));
        let is_mod5_residue = matrix_row(&encoded.structure.matrices[RESIDUE_LEFT_MATRIX], row)
            .iter()
            .any(|(column, _)| residue_left_columns.contains(column));
        if is_mod5_bit_pair || is_mod5_residue {
            rows.insert(row);
        }
    }
    rows.into_iter().collect()
}

fn assert_emitted_rows(encoded: &EncodedGadgetNativeR1cs, events: &[Mod5TraceEntry], rows: &[usize]) {
    assert_eq!(rows.len(), CHUNKS * 8);
    for &row in rows {
        for (matrix_index, matrix) in encoded.structure.matrices.iter().enumerate() {
            if !PACKED_MATRICES.contains(&matrix_index) {
                assert!(
                    matrix_row(matrix, row).is_empty(),
                    "unaccounted matrix {matrix_index} is nonzero on packed row {row}"
                );
            }
        }
    }
    for (chunk, (event, chunk_rows)) in events.iter().zip(rows.chunks_exact(8)).enumerate() {
        let low = encoded.plan.packed_mod5_low_bit_range(chunk).unwrap();
        let residue = encoded.plan.packed_mod5_residue_range(chunk).unwrap();
        let decoder = encoded.plan.packed_mod5_decoder_audit(chunk).unwrap();
        for pair in 0..6 {
            let row = chunk_rows[pair];
            assert_eq!(
                matrix_row(&encoded.structure.matrices[SELECTOR_MATRIX], row),
                vec![(0, F::ONE)]
            );
            assert_eq!(
                matrix_row(&encoded.structure.matrices[BIT_LEFT_MATRIX], row),
                vec![(low.start + 2 * pair, F::ONE)]
            );
            assert_eq!(
                matrix_row(&encoded.structure.matrices[BIT_RIGHT_MATRIX], row),
                vec![(low.start + 2 * pair + 1, F::ONE)]
            );
            assert!(matrix_row(&encoded.structure.matrices[RESIDUE_LEFT_MATRIX], row).is_empty());
            assert!(matrix_row(&encoded.structure.matrices[RESIDUE_RIGHT_MATRIX], row).is_empty());
        }
        let high_row = chunk_rows[6];
        assert_eq!(
            matrix_row(&encoded.structure.matrices[SELECTOR_MATRIX], high_row),
            vec![(0, F::ONE)]
        );
        assert_eq!(
            matrix_row(&encoded.structure.matrices[BIT_LEFT_MATRIX], high_row),
            vec![(low.start + 12, F::ONE)]
        );
        assert_eq!(
            matrix_row(&encoded.structure.matrices[BIT_RIGHT_MATRIX], high_row),
            decoder.high
        );
        assert!(matrix_row(&encoded.structure.matrices[RESIDUE_LEFT_MATRIX], high_row).is_empty());
        assert!(matrix_row(&encoded.structure.matrices[RESIDUE_RIGHT_MATRIX], high_row).is_empty());

        let residue_row = chunk_rows[7];
        assert_eq!(
            matrix_row(&encoded.structure.matrices[SELECTOR_MATRIX], residue_row),
            vec![(0, F::ONE)]
        );
        assert!(matrix_row(&encoded.structure.matrices[BIT_LEFT_MATRIX], residue_row).is_empty());
        assert!(matrix_row(&encoded.structure.matrices[BIT_RIGHT_MATRIX], residue_row).is_empty());
        assert_eq!(
            matrix_row(&encoded.structure.matrices[RESIDUE_LEFT_MATRIX], residue_row),
            vec![(residue.start, F::ONE)]
        );
        assert_eq!(
            matrix_row(&encoded.structure.matrices[RESIDUE_RIGHT_MATRIX], residue_row),
            vec![(residue.start + 1, F::ONE)]
        );

        assert_eq!(event.quotient_bits[12].col() + 1, event.quotient_bits[13].col());
    }
}

fn polynomial_schema(encoded: &EncodedGadgetNativeR1cs) -> Vec<PolynomialAuditTerm> {
    assert_eq!(encoded.structure.f.arity(), GATE_ARITY);
    let role = |index| match index {
        SELECTOR_MATRIX => "selector",
        BIT_LEFT_MATRIX => "bitLeft",
        BIT_RIGHT_MATRIX => "bitRight",
        RESIDUE_LEFT_MATRIX => "residueLeft",
        RESIDUE_RIGHT_MATRIX => "residueRight",
        _ => unreachable!(),
    };
    encoded
        .structure
        .f
        .terms()
        .iter()
        .filter_map(|term| {
            let active = term
                .exps
                .iter()
                .enumerate()
                .filter(|&(_, &power)| power != 0)
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            let touches_packed_payload = active.iter().any(|&index| {
                index == BIT_LEFT_MATRIX
                    || index == BIT_RIGHT_MATRIX
                    || index == RESIDUE_LEFT_MATRIX
                    || index == RESIDUE_RIGHT_MATRIX
            });
            if !touches_packed_payload {
                return None;
            }
            assert!(
                active.iter().all(|index| PACKED_MATRICES.contains(index)),
                "full CCS polynomial mixes a packed Mod-5 role with an unowned role"
            );
            Some(PolynomialAuditTerm {
                coefficient: signed(term.coeff),
                powers: active
                    .into_iter()
                    .map(|index| (role(index), term.exps[index]))
                    .collect(),
            })
        })
        .collect()
}

fn lean_terms(terms: &[RoleTerm]) -> String {
    format!(
        "[{}]",
        terms
            .iter()
            .map(|term| format!("⟨{}, {}⟩", term.role, term.coefficient))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_source_rows(rows: &[RoleRow]) -> String {
    let rows = rows
        .iter()
        .map(|row| {
            format!(
                "  ⟨{},\n    {},\n    {}⟩",
                lean_terms(&row.a),
                lean_terms(&row.b),
                lean_terms(&row.c)
            )
        })
        .collect::<Vec<_>>()
        .join("\n, ");
    format!("[\n{rows}\n]")
}

fn lean_decoder_lc(terms: &[RoleTerm]) -> String {
    lean_terms(terms)
}

fn lean_polynomial_terms(terms: &[PolynomialAuditTerm]) -> String {
    let terms = terms
        .iter()
        .map(|term| {
            let powers = term
                .powers
                .iter()
                .map(|(role, power)| format!("⟨.{role}, {power}⟩"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("  ⟨{}, [{}]⟩", term.coefficient, powers)
        })
        .collect::<Vec<_>>()
        .join("\n, ");
    format!("[\n{terms}\n]")
}

fn render(source_rows: &[RoleRow], decoder: &DecoderAudit, polynomial: &[PolynomialAuditTerm]) -> String {
    let mut out = String::new();
    out.push_str("import Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.PackedMod5Schema\n\n");
    out.push_str(
        "/-! Generated exact active packed-Mod-5 data; do not hand-edit.\n\n\
         Owns: one role-normalized production source block, its projected decoder,\n\
         the active row schedule, and the exact Mod-5 polynomial specialization.\n\n\
         Does not own: selector composition, inactive rows, or semantic authority.\n\n\
         Supported profile: isolated one-rho, 64-chunk sampler fixture. Full-F'\n\
         placement and outer-image conformance are separate obligations.\n\n\
         Emits constraints: no.\n\n\
         Authority boundary: Rust validates and compares equations directly. No digest\n\
         in this file authorizes a row or decoder.\n\n\
         | Data branch | Mathematical obligation | Production check |\n\
         |---|---|---|\n\
         | `sourceRows` | exact 20-row source language | all 64 trace schemas equal |\n\
         | `decoderDefinitions` | exact projected reconstruction | normalized production LCs |\n\
         | `activeRows` | exact 6 + 1 + 1 row schedule | materialized CCS matrices |\n\
         | `polynomialTerms` | exact packed residual expansion | production sparse polynomial |\n\
         -/\n\n",
    );
    out.push_str(
        "namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5.PackedMod5ArtifactData\n\n",
    );
    out.push_str("open PackedMod5Artifact\n\n");
    out.push_str("def schemaVersion : Nat := 1\n");
    out.push_str("def sourceInputOrder : List SourceRole :=\n  [");
    out.push_str(
        &(0..16)
            .map(|index| format!(".chunkBit {index}"))
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push_str("]\n");
    out.push_str(
        "def sourceAllocatedOrder : List SourceRole :=\n  [.index, .quotient, .indexProduct 0, .indexProduct 1, .indexProduct 2,\n",
    );
    out.push_str("   ");
    out.push_str(
        &(0..14)
            .map(|index| format!(".quotientBit {index}"))
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push_str("]\n");
    writeln!(
        out,
        "def sourceRows : List SourceRow :=\n{}",
        lean_source_rows(source_rows)
    )
    .unwrap();
    out.push_str("def coordinateOrder : List CoordinateRole :=\n  [");
    out.push_str(
        &(0..13)
            .map(|index| format!(".quotientLow {index}"))
            .chain([".residueLeft".to_owned(), ".residueRight".to_owned()])
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push_str("]\n");
    out.push_str("def decoderDefinitions : List DecoderDefinition :=\n  [");
    write!(
        out,
        ".linear .index {},\n   .linear (.quotientBit 13) {},\n   .linear .quotient {},\n",
        lean_decoder_lc(&decoder.index),
        lean_decoder_lc(&decoder.high),
        lean_decoder_lc(&decoder.quotient)
    )
    .unwrap();
    for (index, product) in decoder.products.iter().enumerate() {
        let close = if index + 1 == decoder.products.len() { "]" } else { "," };
        writeln!(
            out,
            "   .product (.indexProduct {index})\n     {}\n     {}{close}",
            lean_decoder_lc(&product.left),
            lean_decoder_lc(&product.right),
        )
        .unwrap();
    }
    writeln!(out, "def gateArity : Nat := {GATE_ARITY}").unwrap();
    writeln!(
        out,
        "def matrixBindings : List MatrixBinding :=\n  [ {{ role := .selector, index := {SELECTOR_MATRIX} }}\n\
         , {{ role := .bitLeft, index := {BIT_LEFT_MATRIX} }}\n\
         , {{ role := .bitRight, index := {BIT_RIGHT_MATRIX} }}\n\
         , {{ role := .residueLeft, index := {RESIDUE_LEFT_MATRIX} }}\n\
         , {{ role := .residueRight, index := {RESIDUE_RIGHT_MATRIX} }} ]"
    )
    .unwrap();
    out.push_str(
        "def activeRows : List ActiveRow :=\n  [ .bitPair (.quotientLow 0) (.quotientLow 1)\n\
         , .bitPair (.quotientLow 2) (.quotientLow 3)\n\
         , .bitPair (.quotientLow 4) (.quotientLow 5)\n\
         , .bitPair (.quotientLow 6) (.quotientLow 7)\n\
         , .bitPair (.quotientLow 8) (.quotientLow 9)\n\
         , .bitPair (.quotientLow 10) (.quotientLow 11)\n\
         , .bitPair (.quotientLow 12) .quotientHigh\n\
         , .residuePair ]\n",
    );
    writeln!(
        out,
        "def polynomialTerms : List PolynomialTerm :=\n{}",
        lean_polynomial_terms(polynomial)
    )
    .unwrap();
    out.push_str("\nend Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5.PackedMod5ArtifactData\n");
    out
}

#[test]
fn packed_mod5_lean_artifact_matches_exact_production() {
    let builder = sampler_builder();
    let source = builder.snapshot();
    let trace = builder.encoding_trace();
    assert_eq!(trace.mod5_chunks().len(), CHUNKS);

    let representative_source = source_schema(&source, &trace.mod5_chunks()[0]);
    for (chunk, event) in trace.mod5_chunks().iter().enumerate().skip(1) {
        assert_eq!(
            source_schema(&source, event),
            representative_source,
            "source role schema drift at chunk {chunk}"
        );
    }

    let chunk_inputs = trace
        .mod5_chunks()
        .iter()
        .flat_map(|event| event.chunk_bits)
        .map(Var::col)
        .collect::<Vec<_>>();
    let encoded = encode_r1cs_gadget_native(&source, trace, &chunk_inputs)
        .expect("exact packed Mod-5 lowering with the sixteen-bit leaf boundary exposed");
    assert!(encoded.is_satisfied());
    assert_eq!(encoded.decode_source().expect("exact inverse"), source.witness());
    let representative_decoder = decoder_schema(&encoded, &trace.mod5_chunks()[0], 0);
    for (chunk, event) in trace.mod5_chunks().iter().enumerate().skip(1) {
        assert_eq!(
            decoder_schema(&encoded, event, chunk),
            representative_decoder,
            "decoder role schema drift at chunk {chunk}"
        );
    }

    let rows = packed_rows(&encoded);
    assert_emitted_rows(&encoded, trace.mod5_chunks(), &rows);
    let polynomial = polynomial_schema(&encoded);
    assert_eq!(polynomial.len(), 12, "exact packed Mod-5 polynomial terms");

    let rendered = render(&representative_source, &representative_decoder, &polynomial);
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed != rendered {
        let expected = format!("{path}.expected");
        std::fs::create_dir_all(
            std::path::Path::new(&expected)
                .parent()
                .expect("generated artifact parent"),
        )
        .expect("create generated artifact directory");
        std::fs::write(&expected, rendered).expect("write packed Mod-5 Lean artifact .expected");
        panic!("packed Mod-5 Lean artifact drifted; inspect {expected} and deliberately promote it");
    }
}
