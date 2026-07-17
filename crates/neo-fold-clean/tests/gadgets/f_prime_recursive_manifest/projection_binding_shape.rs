//! Generated fixed-profile shape/count bridge for the production PiRLC binding.
//!
//! Owns: extraction of projection arity/roles from exact production identity
//! traces, padded width from exact zero-glue rows, and projection-SIS input
//! length from the seeded-Phi81 block emitted inside the binding stage.
//!
//! Does not own: the ordered binding labels, the ordered preimage field-source
//! map, native/circuit value equality, SIS security, or permission to remove
//! rows. In particular, a 3,616-word SIS block proves the production circuit's
//! input length; it does not prove byte-for-byte serializer conformance.
//!
//! Emits constraints: no. It regenerates and audits the production relation.
//!
//! Authority boundary: R1CS rows remain authoritative. Roles are diagnostic
//! profile evidence, and generated Lean data is a drift-checked count bridge.
//!
//! | Artifact branch | Mathematical obligation | Production evidence | Lean scope |
//! |---|---|---|---|
//! | `profile` | 15 inputs; 18 commitment, 5 X, 6 y_ring, 2 y_zcol identities; no adv | validated identity trace | numeric/profile instantiation |
//! | `active/quotient` | degree-54 active outputs and degree-53 quotients | every identity wire audit | shape equality only |
//! | `padding` | 10 zero tail lanes produce width 64 | exact y_ring/y_zcol glue rows | width equality only |
//! | `projection_sis` | long rank-2 block consumes 3,616 fields; rank-1 compression consumes 108 | seeded-Phi81 blocks inside exact stage window | input-count equality only |

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::ops::Range;

use neo_ccs::SeededPhi81LinearBlock;
use neo_fold_clean::engine::r1cs_circuit::builder::{ProjectionGlueRole, BALANCED_TERNARY_DIGITS};
use neo_fold_clean::engine::r1cs_circuit::projection_identity_trace::validate_projection_identity_traces;
use neo_fold_clean::engine::r1cs_circuit::{
    ProjectionIdentityRole, ProjectionIdentityTraceTestMutation, R1csBuilder, R1csEncodingTrace, R1csSnapshot,
};
use neo_fold_clean::paper::reductions::pi_rlc_circuit::stage;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::{build_recursive_program, repo_root};

const ARTIFACT_PATH: &str =
    "formal/superneo-lean/SuperNeo/FPrimeRecursiveVerifier/PiRlcAlgebra/Refinement/Generated/ProjectionBindingShapeArtifactData.lean";

#[derive(Clone, Debug, PartialEq, Eq)]
struct ProjectionBindingShapeArtifact {
    schema_version: usize,
    projection_sis_rows: Range<usize>,
    binding_block_rows: Range<usize>,
    compression_block_rows: Range<usize>,
    input_count: usize,
    active_degree: usize,
    quotient_degree: usize,
    commitment_lanes: usize,
    adv_commitment_lanes: usize,
    active_x_columns: usize,
    y_ring_rows: usize,
    extension_limbs: usize,
    y_zcol_limbs: usize,
    identity_count: usize,
    padding_tail: usize,
    padded_degree: usize,
    sis_block_count: usize,
    binding_preimage_fields: usize,
    digest_compression_fields: usize,
    binding_kappa: usize,
    digest_compression_kappa: usize,
    balanced_word_width: usize,
    roles: Vec<ProjectionIdentityRole>,
}

fn exact_stage_rows(trace: &R1csEncodingTrace, label: &str, next_label: &str) -> Result<Range<usize>, String> {
    let matches = trace
        .stages()
        .iter()
        .enumerate()
        .filter(|(_, checkpoint)| checkpoint.label == label)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!("expected one `{label}` checkpoint, got {}", matches.len()));
    }
    let (index, checkpoint) = matches[0];
    let next = trace
        .stages()
        .get(index + 1)
        .ok_or_else(|| format!("`{label}` has no closing checkpoint"))?;
    if next.label != next_label {
        return Err(format!("`{label}` closes at `{}`, expected `{next_label}`", next.label));
    }
    if checkpoint.row >= next.row {
        return Err(format!("`{label}` stage is empty or reversed"));
    }
    Ok(checkpoint.row..next.row)
}

fn consecutive(values: &BTreeSet<usize>) -> bool {
    values.iter().copied().eq(0..values.len())
}

fn derive_uniform_width(
    trace: &R1csEncodingTrace,
    width: impl Fn(&neo_fold_clean::engine::r1cs_circuit::ProjectionIdentityTraceEntry) -> usize,
    label: &str,
) -> Result<usize, String> {
    let mut widths = trace.projection_identities().iter().map(width);
    let first = widths
        .next()
        .ok_or_else(|| "production trace has no projection identities".to_owned())?;
    if first == 0 || !widths.all(|candidate| candidate == first) {
        return Err(format!("projection identities do not share one nonzero {label}"));
    }
    Ok(first)
}

fn derive_padding_tail(
    builder: &R1csBuilder,
    source: &R1csSnapshot,
    input_count: usize,
    extension_limbs: usize,
    y_ring_rows: usize,
) -> Result<usize, String> {
    let denominator = (input_count + 1)
        .checked_mul(extension_limbs)
        .ok_or_else(|| "projection padding denominator overflow".to_owned())?;
    if denominator == 0 {
        return Err("projection padding denominator must be nonzero".to_owned());
    }

    let y_zcol = builder
        .projection_glue_audits()
        .iter()
        .filter(|audit| audit.role == ProjectionGlueRole::YZColPaddingZero)
        .collect::<Vec<_>>();
    if y_zcol.len() != 1 {
        return Err(format!("expected one y_zcol padding owner, got {}", y_zcol.len()));
    }
    let mut pinned_columns = validate_zero_pin_rows(source, y_zcol[0].row_start..y_zcol[0].row_end, "y_zcol")?;
    let y_zcol_rows = y_zcol[0].row_end - y_zcol[0].row_start;
    if y_zcol_rows == 0 || y_zcol_rows % denominator != 0 {
        return Err("y_zcol padding rows do not encode a complete input/output tail".to_owned());
    }
    let tail = y_zcol_rows / denominator;

    let y_ring = builder
        .projection_glue_audits()
        .iter()
        .filter_map(|audit| match audit.role {
            ProjectionGlueRole::YRingPaddingZero { row } => Some((row, audit.row_end - audit.row_start)),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    if y_ring.len() != y_ring_rows || !y_ring.keys().copied().eq(0..y_ring_rows) {
        return Err("y_ring padding owners do not match the traced row roles".to_owned());
    }
    for (&row, &rows) in &y_ring {
        let owner = builder
            .projection_glue_audits()
            .iter()
            .find(|audit| audit.role == ProjectionGlueRole::YRingPaddingZero { row })
            .ok_or_else(|| format!("missing y_ring padding owner {row}"))?;
        if owner.row_end - owner.row_start != rows {
            return Err(format!("y_ring padding owner {row} changed while auditing"));
        }
        let row_columns = validate_zero_pin_rows(source, owner.row_start..owner.row_end, &format!("y_ring[{row}]"))?;
        if !pinned_columns.is_disjoint(&row_columns) {
            return Err(format!(
                "y_ring padding owner {row} repeats a column owned by another padded family"
            ));
        }
        pinned_columns.extend(row_columns);
    }
    if !y_ring.values().all(|&rows| rows == denominator * tail) {
        return Err("y_ring and y_zcol padding owners disagree on tail width".to_owned());
    }
    Ok(tail)
}

fn validate_zero_pin_rows(source: &R1csSnapshot, rows: Range<usize>, label: &str) -> Result<BTreeSet<usize>, String> {
    if rows.is_empty() {
        return Err(format!("{label} padding owner is empty"));
    }
    let mut columns = BTreeSet::new();
    for row in rows {
        let a = source.a_row(row);
        let b = source.b_row(row);
        let c = source.c_row(row);
        let [(column, coefficient)] = a else {
            return Err(format!("{label} padding row {row} is not a singleton A row"));
        };
        if *column == 0 || *coefficient != F::ONE || b != [(0, F::ONE)] || !c.is_empty() {
            return Err(format!("{label} padding row {row} is not `wire * 1 = 0`"));
        }
        if !columns.insert(*column) {
            return Err(format!("{label} padding column {column} is pinned twice"));
        }
    }
    Ok(columns)
}

fn validate_sis_words(builder: &R1csBuilder, block: &SeededPhi81LinearBlock, label: &str) -> Result<(), String> {
    if block.word_width() != BALANCED_TERNARY_DIGITS {
        return Err(format!(
            "{label} SIS word width {} is not the production balanced-ternary width {BALANCED_TERNARY_DIGITS}",
            block.word_width()
        ));
    }
    let audits = builder.balanced_ternary_audits();
    let by_start = audits
        .iter()
        .map(|audit| (audit.digit_cols[0], audit))
        .collect::<BTreeMap<_, _>>();
    if by_start.len() != audits.len() {
        return Err("balanced-ternary word starts are not globally unique".to_owned());
    }
    let mut source_fields = BTreeSet::new();
    for &start in block.word_starts() {
        let audit = by_start
            .get(&start)
            .ok_or_else(|| format!("{label} SIS word {start} has no field decomposition audit"))?;
        if !audit
            .digit_cols
            .iter()
            .copied()
            .eq(start..start + BALANCED_TERNARY_DIGITS)
        {
            return Err(format!(
                "{label} SIS word {start} is not one contiguous field decomposition"
            ));
        }
        if !source_fields.insert(audit.field_col) {
            return Err(format!("{label} SIS source field {} occurs twice", audit.field_col));
        }
    }
    Ok(())
}

fn extract_artifact(
    builder: &R1csBuilder,
    trace: &R1csEncodingTrace,
) -> Result<ProjectionBindingShapeArtifact, String> {
    let source = builder.snapshot();
    let validated = validate_projection_identity_traces(&source, trace)
        .map_err(|error| format!("projection trace validation: {error}"))?;
    let identities = trace.projection_identities();
    if validated.roles.as_slice()
        != identities
            .iter()
            .map(|identity| identity.role)
            .collect::<Vec<_>>()
    {
        return Err("validated role order disagrees with the production trace".to_owned());
    }

    let input_count = derive_uniform_width(trace, |identity| identity.input_columns.len(), "input arity")?;
    let rho_count = derive_uniform_width(trace, |identity| identity.rho_columns.len(), "rho arity")?;
    if input_count != rho_count {
        return Err("projection input and rho arities differ".to_owned());
    }
    let active_degree = derive_uniform_width(trace, |identity| identity.output_columns.len(), "active degree")?;
    let quotient_degree = derive_uniform_width(trace, |identity| identity.quotient_columns.len(), "quotient degree")?;

    let roles = validated.roles;
    let mut commitment = BTreeSet::new();
    let mut adv = Vec::new();
    let mut x = BTreeSet::new();
    let mut y_ring = BTreeMap::<usize, BTreeSet<usize>>::new();
    let mut y_zcol = BTreeSet::new();
    for role in &roles {
        match *role {
            ProjectionIdentityRole::CommitmentLane { lane } => {
                commitment.insert(lane);
            }
            ProjectionIdentityRole::NebulaCommitmentLane { coordinate, lane } => adv.push((coordinate, lane)),
            ProjectionIdentityRole::ActiveXColumn { column } => {
                x.insert(column);
            }
            ProjectionIdentityRole::YRingLimb { row, limb } => {
                y_ring.entry(row).or_default().insert(limb);
            }
            ProjectionIdentityRole::YZColLimb { limb } => {
                y_zcol.insert(limb);
            }
            ProjectionIdentityRole::Standalone => {
                return Err("fixed production profile contains a standalone projection role".to_owned())
            }
        }
    }
    if !consecutive(&commitment) || !consecutive(&x) || !consecutive(&y_zcol) {
        return Err("plain projection role indices are not canonical prefixes".to_owned());
    }
    let extension_limbs = y_zcol.len();
    if extension_limbs == 0 || !y_ring.keys().copied().eq(0..y_ring.len()) {
        return Err("y_ring/y_zcol role geometry is empty or noncanonical".to_owned());
    }
    if !y_ring
        .values()
        .all(|limbs| limbs.len() == extension_limbs && limbs.iter().copied().eq(0..extension_limbs))
    {
        return Err("y_ring rows do not share the y_zcol extension-limb shape".to_owned());
    }

    let padding_tail = derive_padding_tail(builder, &source, input_count, extension_limbs, y_ring.len())?;
    let padded_degree = active_degree
        .checked_add(padding_tail)
        .ok_or_else(|| "padded projection degree overflow".to_owned())?;

    let projection_sis_rows = exact_stage_rows(
        trace,
        stage::PROJECTION_BINDING_SIS_DIGEST,
        stage::PROJECTION_BINDING_TRANSCRIPT_BETA,
    )?;
    let mut sis_blocks = builder
        .seeded_phi81_a_blocks()
        .iter()
        .filter(|block| projection_sis_rows.start <= block.row_start() && block.row_end() <= projection_sis_rows.end)
        .collect::<Vec<_>>();
    sis_blocks.sort_by_key(|block| block.row_start());
    if sis_blocks.len() != 2 {
        return Err(format!(
            "expected binding and compression SIS blocks, got {}",
            sis_blocks.len()
        ));
    }
    let binding = sis_blocks[0];
    let compression = sis_blocks[1];
    if binding.word_starts().len() <= compression.word_starts().len() || binding.row_end() > compression.row_start() {
        return Err("projection SIS blocks are not ordered long-binding then compression".to_owned());
    }
    validate_sis_words(builder, binding, "binding")?;
    validate_sis_words(builder, compression, "compression")?;
    if binding.word_width() != compression.word_width() {
        return Err("projection SIS blocks disagree on field-word width".to_owned());
    }

    Ok(ProjectionBindingShapeArtifact {
        schema_version: 1,
        projection_sis_rows,
        binding_block_rows: binding.row_start()..binding.row_end(),
        compression_block_rows: compression.row_start()..compression.row_end(),
        input_count,
        active_degree,
        quotient_degree,
        commitment_lanes: commitment.len(),
        adv_commitment_lanes: adv.len(),
        active_x_columns: x.len(),
        y_ring_rows: y_ring.len(),
        extension_limbs,
        y_zcol_limbs: y_zcol.len(),
        identity_count: roles.len(),
        padding_tail,
        padded_degree,
        sis_block_count: sis_blocks.len(),
        binding_preimage_fields: binding.word_starts().len(),
        digest_compression_fields: compression.word_starts().len(),
        binding_kappa: binding.kappa(),
        digest_compression_kappa: compression.kappa(),
        balanced_word_width: binding.word_width(),
        roles,
    })
}

fn lean_role(role: ProjectionIdentityRole) -> String {
    match role {
        ProjectionIdentityRole::Standalone => ".standalone".to_owned(),
        ProjectionIdentityRole::CommitmentLane { lane } => format!(".commitmentLane {lane}"),
        ProjectionIdentityRole::NebulaCommitmentLane { coordinate, lane } => {
            let coordinate = match coordinate {
                neo_fold_clean::engine::r1cs_circuit::ProjectionNebulaCoordinate::Ops => "adviceOpsLane",
                neo_fold_clean::engine::r1cs_circuit::ProjectionNebulaCoordinate::Is => "adviceIsLane",
                neo_fold_clean::engine::r1cs_circuit::ProjectionNebulaCoordinate::Fs => "adviceFsLane",
            };
            format!(".{coordinate} {lane}")
        }
        ProjectionIdentityRole::ActiveXColumn { column } => format!(".activeXColumn {column}"),
        ProjectionIdentityRole::YRingLimb { row, limb } => format!(".yRingLimb {row} {limb}"),
        ProjectionIdentityRole::YZColLimb { limb } => format!(".yZcolLimb {limb}"),
    }
}

fn render_artifact(artifact: &ProjectionBindingShapeArtifact) -> String {
    let mut rendered = String::new();
    rendered.push_str(
        "import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Refinement.Generated.ProjectionIdentityCertificateData\n\n\
/-! Generated by `gadgets_f_prime_recursive_manifest`; do not hand-edit.\n\n\
Owns: fixed plain-profile numeric and role evidence extracted from the exact\n\
production projection trace, padding rows, and projection-SIS blocks.\n\n\
Does not own: ordered labels, ordered preimage field sources, native/circuit\n\
value equality, SIS security, or permission to remove rows. The 3,616-field\n\
count is not byte-for-byte serializer conformance.\n\n\
Emits constraints: no.\n\n\
| Data branch | Mathematical obligation | Rust source |\n\
|---|---|---|\n\
| `roles` | Exact 18 + 5 + 6 + 2 plain identity order, with no adv | validated production trace |\n\
| active/padded widths | 54 active, 53 quotient, 10 zero tail, 64 carrier | identity audits plus normalized single-variable zero rows |\n\
| projection SIS | 3,616-field long binding and 108-field compression | seeded-Phi81 blocks in the exact stage window |\n\
-/\n\n\
namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.ProjectionBindingShapeArtifactData\n\n\
open ProjectionIdentityCertificateData\n\n",
    );
    macro_rules! nat {
        ($name:literal, $value:expr) => {
            writeln!(rendered, "def {} : Nat := {}", $name, $value).expect("render artifact")
        };
    }
    nat!("schemaVersion", artifact.schema_version);
    nat!("projectionSisRowStart", artifact.projection_sis_rows.start);
    nat!("projectionSisRowEnd", artifact.projection_sis_rows.end);
    nat!("bindingBlockRowStart", artifact.binding_block_rows.start);
    nat!("bindingBlockRowEnd", artifact.binding_block_rows.end);
    nat!("compressionBlockRowStart", artifact.compression_block_rows.start);
    nat!("compressionBlockRowEnd", artifact.compression_block_rows.end);
    nat!("inputCount", artifact.input_count);
    nat!("activeDegree", artifact.active_degree);
    nat!("quotientDegree", artifact.quotient_degree);
    nat!("commitmentLanes", artifact.commitment_lanes);
    nat!("advCommitmentLanes", artifact.adv_commitment_lanes);
    nat!("activeXColumns", artifact.active_x_columns);
    nat!("yRingRows", artifact.y_ring_rows);
    nat!("extensionLimbs", artifact.extension_limbs);
    nat!("yZcolLimbs", artifact.y_zcol_limbs);
    nat!("identityCount", artifact.identity_count);
    nat!("paddingTail", artifact.padding_tail);
    nat!("paddedDegree", artifact.padded_degree);
    nat!("sisBlockCount", artifact.sis_block_count);
    nat!("bindingPreimageFields", artifact.binding_preimage_fields);
    nat!("digestCompressionFields", artifact.digest_compression_fields);
    nat!("bindingKappa", artifact.binding_kappa);
    nat!("digestCompressionKappa", artifact.digest_compression_kappa);
    nat!("balancedWordWidth", artifact.balanced_word_width);
    rendered.push_str("\ndef roles : List IdentityRole :=\n");
    for (index, &role) in artifact.roles.iter().enumerate() {
        let prefix = if index == 0 { "  [ " } else { "  , " };
        writeln!(rendered, "{prefix}{}", lean_role(role)).expect("render role");
    }
    rendered.push_str("  ]\n\nend SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.ProjectionBindingShapeArtifactData\n");
    rendered
}

#[test]
fn projection_binding_shape_artifact_matches_production_and_rejects_drift() {
    let builder = build_recursive_program();
    assert!(builder.is_satisfied(), "fixed plain recursive fixture");
    let artifact = extract_artifact(&builder, builder.encoding_trace()).expect("fixed projection-binding artifact");

    let mut drifted = artifact.clone();
    drifted.binding_preimage_fields += 1;
    assert_ne!(drifted, artifact, "preimage-length drift must change the artifact");
    let mut drifted = artifact.clone();
    drifted.roles.swap(0, 1);
    assert_ne!(drifted, artifact, "role-order drift must change the artifact");

    let mut corrupted_trace = builder.encoding_trace().clone();
    corrupted_trace.apply_projection_identity_trace_test_mutation(
        0,
        ProjectionIdentityTraceTestMutation::Role {
            role: ProjectionIdentityRole::CommitmentLane { lane: 1 },
        },
    );
    assert!(
        extract_artifact(&builder, &corrupted_trace).is_err(),
        "noncanonical production role drift must fail closed"
    );

    let rendered = render_artifact(&artifact);
    let path = repo_root().join(ARTIFACT_PATH);
    let committed = fs::read_to_string(&path).unwrap_or_default();
    if committed != rendered {
        let expected = path.with_extension("lean.expected");
        fs::create_dir_all(expected.parent().expect("artifact parent")).expect("create artifact parent");
        fs::write(&expected, &rendered).expect("write reviewed projection-binding artifact candidate");
    }
    assert_eq!(
        committed, rendered,
        "projection-binding shape artifact drifted; review the generated .expected file"
    );
}
