//! Exact final-assignment checks for the production claim-replay leaf.

#[path = "streaming_claim_replay_linked_overlay/conformance.rs"]
mod conformance;

use std::collections::BTreeMap;

use conformance::canonical_poseidon_call;
use neo_fold_clean::frontends::nebula::f_prime::{
    build_production_claim_replay_base_low_norm_r1cs, build_production_claim_replay_linked_overlay_low_norm_r1cs,
    production_claim_active_coordinate_overlay_base_kind_map,
    production_claim_active_coordinate_overlay_compact_layout_and_decoder_runs_for_ranges,
    production_claim_active_coordinate_overlay_links,
    production_claim_active_coordinate_overlay_nonseeded_row_projection,
    production_claim_active_coordinate_overlay_seeded_placements,
    production_claim_replay_base_compact_layout_and_decoder_runs_for_ranges, production_claim_replay_base_phase_kinds,
    production_claim_replay_base_retained_row_projection, production_claim_replay_base_semantic_row_projection,
    production_pi_rlc_family_body_projected_rows_with_source_provenance, NebulaFPrimeClaimCoordinateOverlaySynthesis,
    NebulaFPrimeClaimReplayArmKind, NebulaFPrimeClaimReplaySynthesis, NebulaFPrimeStreamingProgramAudit,
};
use neo_fold_clean::frontends::r1cs_f_prime::{
    LinkedOverlayLowNormR1cs, SelectiveEmittedRowFamily, SelectiveProjectedDecoderRunProvenance,
    SelectiveProjectedPort, SelectiveProjectedRowsAudit, SelectiveSourceRowDisposition,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const ACTIVE_CHUNKS: usize = 98;
const FULL_CHUNKS: usize = 97;

fn evaluate_decoder(terms: &[(usize, F)], assignment: &[F]) -> F {
    terms.iter().fold(F::ZERO, |sum, &(column, coefficient)| {
        sum + coefficient * assignment[column]
    })
}

fn evaluate_relation_row(relation: &LinkedOverlayLowNormR1cs, row: usize, assignment: &[F]) -> F {
    let point = relation
        .structure()
        .matrices
        .iter()
        .map(|matrix| {
            matrix
                .materialize_row(row)
                .expect("linked-overlay row is in range")
                .into_iter()
                .fold(F::ZERO, |sum, (column, coefficient)| {
                    sum + coefficient * assignment[column]
                })
        })
        .collect::<Vec<_>>();
    relation.structure().f.eval(&point)
}

fn direct_slot_terms(start: usize, width: usize) -> Vec<(usize, F)> {
    let radix = match width {
        41 => F::from_u64(3),
        23 => F::from_u64(7),
        1..=64 => F::from_u64(2),
        _ => panic!("unsupported direct decoder width {width}"),
    };
    let mut coefficient = F::ONE;
    (start..start + width)
        .map(|column| {
            let term = (column, coefficient);
            coefficient *= radix;
            term
        })
        .collect()
}

fn expand_projected_port(port: &SelectiveProjectedPort) -> Vec<(usize, F)> {
    assert!(
        port.seeded_blocks().is_empty(),
        "retained claim-replay rows must not contain seeded final ports"
    );
    let mut terms = BTreeMap::<usize, F>::new();
    for term in port.explicit() {
        *terms.entry(term.column()).or_insert(F::ZERO) += term.coefficient();
    }
    for run in port.geometric_runs() {
        let mut coefficient = run.initial();
        for column in run.column_start()..run.column_start() + run.length() {
            *terms.entry(column).or_insert(F::ZERO) += coefficient;
            coefficient *= run.ratio();
        }
    }
    terms.retain(|_, coefficient| *coefficient != F::ZERO);
    terms.into_iter().collect()
}

fn assert_retained_projection_matches_final_rows(
    relation: &LinkedOverlayLowNormR1cs,
    projection: &SelectiveProjectedRowsAudit,
) {
    assert_eq!(projection.rows(), relation.base_relation().structure().n);
    assert_eq!(projection.columns(), relation.base_relation().structure().m);
    let base_rows = relation.layout().base_rows();
    for compact in projection.row_artifacts() {
        let row = compact.emitted_row();
        assert!(base_rows.contains(&row));
        for (port, projected) in compact.ports().iter().enumerate() {
            let materialized = relation.structure().matrices[port]
                .materialize_row(row)
                .expect("retained row is present in the final linked relation");
            assert_eq!(
                expand_projected_port(projected),
                materialized,
                "retained base row {row}, port {port} must equal the exact final linked row"
            );
        }
    }
}

fn active_replay_decoders() -> Vec<SelectiveProjectedDecoderRunProvenance> {
    let requests = [(0, 2357..155957), (1, 2357..88157)];
    let (_, decoders) = production_claim_replay_base_compact_layout_and_decoder_runs_for_ranges(&requests)
        .expect("audit exact active replay-call decoder ranges");
    assert_eq!(decoders.len(), 2);
    let full_template = &decoders[0].repeated_templates()[0];
    let final_template = &decoders[1].repeated_templates()[0];
    assert_eq!(full_template.source_width(), 600);
    assert_eq!(final_template.source_width(), 600);
    assert_eq!(full_template.relative_runs(), final_template.relative_runs());
    assert_eq!(full_template.instances().len(), 1);
    assert_eq!(final_template.instances().len(), 1);
    assert!(decoders
        .iter()
        .all(|decoder| decoder.residual_strided_runs().is_empty()));
    decoders
}

#[test]
fn production_claim_replay_base_compiler_disposition_census_is_exact() {
    let relation =
        build_production_claim_replay_base_low_norm_r1cs().expect("build exact production claim-replay base");
    let audit = relation
        .selective_compiler_audit()
        .expect("production base retains its exact compiler audit");

    for (arm_index, arm) in audit.rows().arms().iter().enumerate() {
        let stages = &audit.source_arm_physical_stages()[arm_index];
        let mut source_cursor = 0;
        let mut census = vec![[0usize; 7]; stages.len()];
        for run in arm.source_runs() {
            let source_rows = run.source_rows();
            assert_eq!(source_rows.start, source_cursor);
            source_cursor = source_rows.end;
            let stage = run
                .stage_occurrence()
                .expect("every production-base source run has one physical stage");
            assert!(stage < stages.len());
            assert!(stages[stage].rows().start <= source_rows.start);
            assert!(source_rows.end <= stages[stage].rows().end);
            let disposition = match run.disposition() {
                SelectiveSourceRowDisposition::Retained => 0,
                SelectiveSourceRowDisposition::Poseidon2(_) => 1,
                SelectiveSourceRowDisposition::CenteredUnit(_) => 2,
                SelectiveSourceRowDisposition::ShiftedTernaryCanonical(_) => 3,
                SelectiveSourceRowDisposition::PolynomialEvaluation(_) => 4,
                SelectiveSourceRowDisposition::ProductSum(_) => 5,
                SelectiveSourceRowDisposition::LinearDefinition(_) => 6,
            };
            census[stage][disposition] += source_rows.len();
        }
        assert_eq!(
            source_cursor,
            stages
                .last()
                .expect("production base has stages")
                .rows()
                .end
        );
        eprintln!("claim-replay base arm {arm_index} source-row disposition census: {census:?}");
    }

    let requests = [(0, 2357..155957), (1, 2357..88157)];
    let (compact, decoders) = production_claim_replay_base_compact_layout_and_decoder_runs_for_ranges(&requests)
        .expect("audit exact active replay-call decoder ranges");
    assert_eq!(compact.rows(), audit.rows());
    assert_eq!(compact.selector_columns(), relation.selector_cols());
    assert_eq!(compact.final_columns(), relation.structure().m);
    assert_eq!(decoders.len(), requests.len());
    for ((arm, source_range), decoder) in requests.iter().zip(&decoders) {
        assert_eq!(decoder.arm(), *arm);
        assert_eq!(decoder.source_range(), source_range.clone());
        assert_eq!(decoder.final_columns(), relation.structure().m);
        eprintln!(
            "claim-replay base arm {arm} active decoder: contiguous_runs={}, strided_runs={}, templates={}, residual_runs={}, families={}",
            decoder.runs().len(),
            decoder.strided_runs().len(),
            decoder.repeated_templates().len(),
            decoder.residual_strided_runs().len(),
            decoder.source_families().len(),
        );
    }
    let full_template = &decoders[0].repeated_templates()[0];
    let final_template = &decoders[1].repeated_templates()[0];
    assert_eq!(full_template.source_width(), 600);
    assert_eq!(final_template.source_width(), 600);
    assert_eq!(full_template.relative_runs(), final_template.relative_runs());
    assert_eq!(full_template.instances().len(), 1);
    assert_eq!(final_template.instances().len(), 1);
    assert_eq!(full_template.instances()[0].count(), 256);
    assert_eq!(final_template.instances()[0].count(), 143);
    eprintln!(
        "claim-replay shared decoder template: relative_runs={}, full_instances={:?}, final_instances={:?}",
        full_template.relative_runs().len(),
        full_template.instances(),
        final_template.instances(),
    );
}

#[test]
fn production_claim_replay_base_semantic_projection_is_exact() {
    let relation = build_production_claim_replay_linked_overlay_low_norm_r1cs()
        .expect("build exact production claim-replay linked overlay");
    let decoders = active_replay_decoders();
    let reference_selected_rows = (74_375..74_547).collect::<Vec<_>>();
    let reference =
        production_pi_rlc_family_body_projected_rows_with_source_provenance(&reference_selected_rows, 0, &[], &[])
            .expect("project the proved direct and chained PiRLC Poseidon2 leaves");
    let reference_source = reference
        .source_provenance()
        .expect("PiRLC leaf reference has complete source provenance");
    let reference_rows = reference.row_artifacts().iter().collect::<Vec<_>>();
    let reference_direct = canonical_poseidon_call(
        &reference_source.poseidon2_sbox_steps()[..86],
        &reference_rows[..86],
        166_320,
        2_218_425,
        648,
        false,
    );
    let reference_chained = canonical_poseidon_call(
        &reference_source.poseidon2_sbox_steps()[86..172],
        &reference_rows[86..172],
        166_920,
        2_221_951,
        648,
        false,
    );
    for (arm, kind) in [
        NebulaFPrimeClaimReplayArmKind::Full,
        NebulaFPrimeClaimReplayArmKind::Final,
    ]
    .into_iter()
    .enumerate()
    {
        let projected = production_claim_replay_base_semantic_row_projection(kind)
            .expect("project exact production-base semantic rows");
        let source = projected
            .source_provenance()
            .expect("semantic row projection has complete source provenance");
        let retained = production_claim_replay_base_retained_row_projection(kind)
            .expect("project exact production-base retained rows");
        let retained_source = retained
            .source_provenance()
            .expect("retained row projection has complete source provenance");
        assert_retained_projection_matches_final_rows(&relation, &retained);
        assert_eq!(source.arm(), arm);
        assert_eq!(retained_source.arm(), arm);
        assert!(!projected.row_artifacts().is_empty());
        assert!(!source.retained_steps().is_empty());
        assert!(!source.poseidon2_sbox_steps().is_empty());
        assert!(
            source.poseidon2_output_steps().is_empty(),
            "production Poseidon2 outputs are exact compiler linear definitions"
        );
        assert_eq!(retained.row_artifacts().len(), source.retained_steps().len());
        assert_eq!(retained_source.retained_steps().len(), source.retained_steps().len());
        assert!(retained_source.poseidon2_sbox_steps().is_empty());
        assert!(retained_source.poseidon2_output_steps().is_empty());
        let expected = if arm == 0 {
            (22_178, 24_236, 23_208, 1_028, 22_016, 162, 180, 172, 8)
        } else {
            (12_469, 13_611, 13_038, 573, 12_298, 171, 177, 172, 5)
        };
        assert_eq!(
            (
                projected.row_artifacts().len(),
                source.source_columns().len(),
                source.retained_slots().len(),
                source.linear_definitions().len(),
                source.poseidon2_sbox_steps().len(),
                source.retained_steps().len(),
                retained_source.source_columns().len(),
                retained_source.retained_slots().len(),
                retained_source.linear_definitions().len(),
            ),
            expected
        );
        let poseidon_rows = projected
            .row_artifacts()
            .iter()
            .filter(|row| row.family() == SelectiveEmittedRowFamily::Poseidon2)
            .collect::<Vec<_>>();
        assert_eq!(poseidon_rows.len(), source.poseidon2_sbox_steps().len());
        let instances = decoders[arm].repeated_templates()[0].instances()[0];
        assert_eq!(source.poseidon2_sbox_steps().len(), instances.count() * 86);
        for call_index in 0..instances.count() {
            let source_start = instances.source_start() + call_index * instances.source_stride();
            let final_start = instances.final_start() + call_index * instances.final_stride();
            let steps = &source.poseidon2_sbox_steps()[call_index * 86..(call_index + 1) * 86];
            let rows = &poseidon_rows[call_index * 86..(call_index + 1) * 86];
            let actual = canonical_poseidon_call(steps, rows, source_start, final_start, 648 + arm, call_index == 0);
            let expected = if call_index == 0 {
                &reference_direct
            } else {
                &reference_chained
            };
            if &actual != expected {
                let step = actual
                    .0
                    .iter()
                    .zip(&expected.0)
                    .position(|(left, right)| left != right);
                let row = actual
                    .1
                    .iter()
                    .zip(&expected.1)
                    .position(|(left, right)| left != right);
                panic!(
                    "claim-replay arm {arm} Poseidon2 call {call_index} differs from the proved leaf: first_step={step:?} actual_step={:?} expected_step={:?}, first_row={row:?} actual_row={:?} expected_row={:?}",
                    step.map(|index| &actual.0[index]),
                    step.map(|index| &expected.0[index]),
                    row.map(|index| &actual.1[index]),
                    row.map(|index| &expected.1[index]),
                );
            }
        }
        eprintln!(
            "claim-replay base arm {arm} semantic projection: final_rows={}, source_columns={}, retained_slots={}, definitions={}, trace_eliminated={}, sboxes={}, outputs={}, rewrites={}, retained={}; retained-only: source_columns={}, retained_slots={}, definitions={}",
            projected.row_artifacts().len(),
            source.source_columns().len(),
            source.retained_slots().len(),
            source.linear_definitions().len(),
            source.trace_eliminated_columns().len(),
            source.poseidon2_sbox_steps().len(),
            source.poseidon2_output_steps().len(),
            source.rewrite_steps().len(),
            source.retained_steps().len(),
            retained_source.source_columns().len(),
            retained_source.retained_slots().len(),
            retained_source.linear_definitions().len(),
        );
    }
}

#[test]
fn production_claim_replay_active_overlay_compiler_disposition_census_is_exact() {
    let relation = build_production_claim_replay_linked_overlay_low_norm_r1cs()
        .expect("build exact production claim-replay linked overlay");
    let audit = relation
        .overlay_relation()
        .selective_compiler_audit()
        .expect("active overlay retains its exact compiler audit");
    for (arm_index, arm) in audit.rows().arms().iter().enumerate() {
        let mut source = [0usize; 7];
        for run in arm.source_runs() {
            let disposition = match run.disposition() {
                SelectiveSourceRowDisposition::Retained => 0,
                SelectiveSourceRowDisposition::Poseidon2(_) => 1,
                SelectiveSourceRowDisposition::CenteredUnit(_) => 2,
                SelectiveSourceRowDisposition::ShiftedTernaryCanonical(_) => 3,
                SelectiveSourceRowDisposition::PolynomialEvaluation(_) => 4,
                SelectiveSourceRowDisposition::ProductSum(_) => 5,
                SelectiveSourceRowDisposition::LinearDefinition(_) => 6,
            };
            source[disposition] += run.source_rows().len();
        }
        assert_eq!(
            source,
            match arm_index {
                0 => [298, 0, 0, 79_484, 0, 0, 652],
                61 | 69 => [298, 0, 0, 126_976, 0, 0, 328],
                97 => [149, 0, 0, 71_300, 0, 0, 326],
                _ => [149, 0, 0, 126_976, 0, 0, 326],
            },
            "active overlay source census for arm {arm_index}"
        );

        let rewrites = audit
            .rows()
            .rewrites()
            .iter()
            .filter(|rewrite| rewrite.arm() == arm_index)
            .collect::<Vec<_>>();
        assert!(rewrites.iter().all(|rewrite| {
            matches!(
                rewrite.kind(),
                neo_fold_clean::frontends::r1cs_f_prime::SelectiveRewriteKind::ShiftedTernaryCanonical
                    | neo_fold_clean::frontends::r1cs_f_prime::SelectiveRewriteKind::LinearDefinition
            )
        }));
        let shifted_rows = rewrites
            .iter()
            .filter(|rewrite| {
                rewrite.kind() == neo_fold_clean::frontends::r1cs_f_prime::SelectiveRewriteKind::ShiftedTernaryCanonical
            })
            .map(|rewrite| rewrite.emitted_rows().len())
            .sum::<usize>();
        assert_eq!(
            shifted_rows,
            match arm_index {
                0 => 13_461,
                97 => 12_075,
                _ => 21_504,
            },
            "active overlay shifted-ternary rows for arm {arm_index}"
        );
    }

    let requests = [0usize, 1, 61, 69, 97].map(|arm| {
        let source = NebulaFPrimeClaimCoordinateOverlaySynthesis::production_kind(arm + 1)
            .expect("representative active coordinate-overlay source arm");
        (arm, 1..source.columns())
    });
    let (compact, decoders) =
        production_claim_active_coordinate_overlay_compact_layout_and_decoder_runs_for_ranges(&requests)
            .expect("decode exact representative active overlay source arms");
    assert_eq!(compact.rows(), audit.rows());
    assert_eq!(compact.selector_columns(), relation.overlay_relation().selector_cols());
    assert_eq!(compact.final_columns(), relation.overlay_relation().structure().m);
    for decoder in decoders {
        eprintln!(
            "active overlay arm {} complete decoder {}..{}: runs={}, strided={}, templates={}, residual={}, families={}",
            decoder.arm(),
            decoder.source_range().start,
            decoder.source_range().end,
            decoder.runs().len(),
            decoder.strided_runs().len(),
            decoder.repeated_templates().len(),
            decoder.residual_strided_runs().len(),
            decoder.source_families().len(),
        );
    }
}

#[test]
fn production_claim_replay_first_active_overlay_nonseeded_row_projection_is_exact() {
    let projection = production_claim_active_coordinate_overlay_nonseeded_row_projection(0)
        .expect("project the exact non-seeded rows of the first active coordinate-overlay arm");
    let relation = build_production_claim_replay_linked_overlay_low_norm_r1cs()
        .expect("build exact production claim-replay linked overlay");
    assert_eq!(projection.rows(), relation.overlay_relation().structure().n);
    assert_eq!(projection.columns(), relation.overlay_relation().structure().m);
    assert_eq!(
        projection.selector_columns(),
        relation.overlay_relation().selector_cols()
    );
    assert!(projection.source_provenance().is_some());
    assert!(!projection.row_artifacts().is_empty());
    assert!(projection
        .row_artifacts()
        .iter()
        .flat_map(|row| row.ports())
        .all(|port| port.seeded_blocks().is_empty()));
}

#[test]
fn production_claim_replay_active_overlay_seeded_placements_are_exact() {
    let placements = production_claim_active_coordinate_overlay_seeded_placements()
        .expect("audit exact compact coordinate-overlay block placements");
    assert_eq!(placements.len(), 101);
    let mut counts = [0usize; ACTIVE_CHUNKS];
    let mut profile_counts = [0usize; 3];
    for placement in &placements {
        counts[placement.arm()] += 1;
        assert_eq!(placement.word_width(), 41);
        assert_eq!(placement.kappa(), 2);
        match (placement.word_count(), placement.message_columns()) {
            (28_672, 21_770) => profile_counts[0] += 1,
            (62_208, 47_232) => profile_counts[1] += 1,
            (8_640, 6_560) => profile_counts[2] += 1,
            profile => panic!("unexpected coordinate-overlay seeded profile {profile:?}"),
        }
        assert_eq!(
            placement
                .word_start_runs()
                .iter()
                .map(|run| run.count())
                .sum::<usize>(),
            placement.word_count()
        );
    }
    assert_eq!(profile_counts, [30, 62, 9]);
    assert_eq!(counts.iter().filter(|&&count| count == 2).count(), 3);
    assert_eq!(counts.iter().filter(|&&count| count == 1).count(), 95);
    eprintln!(
        "claim-replay active overlay compact seeded placements: blocks={}, runs={}, double_call_arms={:?}, first_selector={}, first_rows={}->{}, terminal_selector={}, terminal_rows={}->{}",
        placements.len(),
        placements
            .iter()
            .map(|placement| placement.word_start_runs().len())
            .sum::<usize>(),
        counts
            .iter()
            .enumerate()
            .filter_map(|(arm, &count)| (count == 2).then_some(arm))
            .collect::<Vec<_>>(),
        placements[0].selector_column(),
        placements[0].source_row_start(),
        placements[0].final_row_start(),
        placements[100].selector_column(),
        placements[100].source_row_start(),
        placements[100].final_row_start(),
    );
}

#[test]
fn production_claim_replay_linked_overlay_has_exact_assignments_and_links() {
    let relation = build_production_claim_replay_linked_overlay_low_norm_r1cs()
        .expect("build exact production claim-replay linked overlay");
    let base_phase_kinds = production_claim_replay_base_phase_kinds();
    let base_kind_map = production_claim_active_coordinate_overlay_base_kind_map();
    let links = production_claim_active_coordinate_overlay_links();
    conformance::assert_exact_final_row_embedding(&relation, &links);
    let layout = relation.layout();
    let retained = [
        production_claim_replay_base_retained_row_projection(NebulaFPrimeClaimReplayArmKind::Full)
            .expect("project exact retained full-arm rows"),
        production_claim_replay_base_retained_row_projection(NebulaFPrimeClaimReplayArmKind::Final)
            .expect("project exact retained final-arm rows"),
    ];
    let retained_decoders = retained
        .iter()
        .enumerate()
        .map(|(base_kind, projection)| {
            projection
                .source_provenance()
                .expect("retained projection has complete source provenance")
                .source_columns()
                .iter()
                .map(|&source_column| {
                    let terms = relation
                        .base_field_decoding_terms(base_kind, source_column)
                        .expect("retained base source field has an exact final decoder");
                    (source_column, terms)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    assert_eq!(base_phase_kinds, vec![3, 4]);
    assert_eq!(base_kind_map.len(), ACTIVE_CHUNKS);
    assert!(base_kind_map[..FULL_CHUNKS].iter().all(|&kind| kind == 0));
    assert_eq!(base_kind_map[FULL_CHUNKS], 1);
    assert_eq!(links.len(), ACTIVE_CHUNKS);
    assert_eq!(layout.base_phase_kinds(), base_phase_kinds);
    assert_eq!(layout.overlay_base_kinds(), base_kind_map);
    assert_eq!(layout.base_selector_columns().len(), 2);
    assert_eq!(layout.overlay_selector_columns().len(), ACTIVE_CHUNKS);
    assert_eq!(layout.base_rows(), 0..relation.base_relation().structure().n);
    assert_eq!(
        layout.overlay_rows(),
        layout.base_rows().end..layout.base_rows().end + relation.overlay_relation().structure().n
    );
    assert_eq!(layout.base_kind_equality_rows().len(), 2);
    assert_eq!(layout.overlay_activation_rows().len(), ACTIVE_CHUNKS);
    let base_pin_count = links
        .iter()
        .map(|contract| contract.base_pins.len())
        .sum::<usize>();
    assert_eq!(base_pin_count, 106);
    assert_eq!(layout.base_field_pin_rows().len(), base_pin_count);
    assert_eq!(
        layout.field_link_rows().len(),
        links
            .iter()
            .map(|contract| contract.fields.len())
            .sum::<usize>()
    );
    assert_eq!(layout.rows(), relation.structure().n);
    assert_eq!(layout.columns(), relation.structure().m);

    let mut final_assignment = None;
    let mut previous_link_end = layout.field_link_rows().start;
    let mut previous_pin_end = layout.base_field_pin_rows().start;
    let mut saw_affine_defined_link = false;
    let mut saw_zero_affine_link = false;
    for overlay_kind in 0..ACTIVE_CHUNKS {
        let base_kind = base_kind_map[overlay_kind];
        let base = if base_kind == 0 {
            NebulaFPrimeClaimReplaySynthesis::production_base_full(overlay_kind)
                .expect("active full production-base source")
        } else {
            NebulaFPrimeClaimReplaySynthesis::production_base_final()
        };
        let overlay = NebulaFPrimeClaimCoordinateOverlaySynthesis::production_kind(overlay_kind + 1)
            .expect("active coordinate-overlay source");
        assert!(base.is_satisfied(), "base source arm {overlay_kind} must accept");
        assert!(overlay.is_satisfied(), "overlay source arm {overlay_kind} must accept");

        let base_source = base
            .normalized_field_assignment_for_artifact()
            .expect("normalize production-base assignment");
        let overlay_source = overlay.normalized_field_assignment_for_artifact();
        let assignment = relation
            .encode(overlay_kind, &base_source, &overlay_source)
            .expect("encode one selected final assignment");
        assert_eq!(assignment.len(), relation.structure().m);
        let base_assignment = relation
            .base_relation()
            .encode(base_kind, &base_source)
            .expect("encode the selected production-base assignment");
        assert_eq!(
            &assignment[..base_assignment.len()],
            base_assignment.as_slice(),
            "the final assignment must retain the exact selected base assignment prefix"
        );
        for (source_column, terms) in &retained_decoders[base_kind] {
            assert_eq!(
                evaluate_decoder(terms, &assignment),
                base_source[*source_column],
                "retained base source column {source_column} must decode from the same final assignment"
            );
        }

        for (kind, &selector) in layout.base_selector_columns().iter().enumerate() {
            assert_eq!(assignment[selector], if kind == base_kind { F::ONE } else { F::ZERO });
        }
        for (kind, &selector) in layout.overlay_selector_columns().iter().enumerate() {
            assert_eq!(
                assignment[selector],
                if kind == overlay_kind { F::ONE } else { F::ZERO }
            );
        }

        let contract = &links[overlay_kind];
        assert_eq!(contract.overlay_kind, overlay_kind);
        assert_eq!(contract.phase_kind, base_phase_kinds[base_kind]);
        let link_rows = layout
            .field_link_rows_for_kind(overlay_kind)
            .expect("absolute final link rows for active kind");
        assert_eq!(link_rows.start, previous_link_end);
        assert_eq!(link_rows.len(), contract.fields.len());
        previous_link_end = link_rows.end;
        let pin_rows = layout
            .base_field_pin_rows_for_kind(overlay_kind)
            .expect("absolute final base-field pin rows for active kind");
        assert_eq!(pin_rows.start, previous_pin_end);
        assert_eq!(pin_rows.len(), contract.base_pins.len());
        previous_pin_end = pin_rows.end;
        let (pin, runtime_pins) = contract
            .base_pins
            .split_first()
            .expect("each active claim kind owns one program-cursor pin");
        assert_eq!(runtime_pins.len(), if overlay_kind == 0 { 8 } else { 0 });
        assert_eq!(
            pin.phase_field,
            base.normalized_before_program_cursor_column()
                .expect("normalized before-program-cursor field")
        );
        assert_eq!(
            pin.value,
            F::from_usize(NebulaFPrimeStreamingProgramAudit::production().first_claim_program_cursor() + overlay_kind)
        );
        let pin_terms = relation
            .base_field_decoding_terms(base_kind, pin.phase_field)
            .expect("base program-cursor decoder in final columns");
        assert_eq!(evaluate_decoder(&pin_terms, &assignment), pin.value);
        for (lane, runtime_pin) in runtime_pins.iter().enumerate() {
            assert_eq!(
                runtime_pin.phase_field,
                base.normalized_before_runtime_column(lane)
                    .expect("normalized before-runtime field")
            );
            assert_eq!(runtime_pin.value, F::ZERO);
            let runtime_terms = relation
                .base_field_decoding_terms(base_kind, runtime_pin.phase_field)
                .expect("base initial-runtime decoder in final columns");
            assert_eq!(evaluate_decoder(&runtime_terms, &assignment), F::ZERO);
        }
        if overlay_kind == 0 {
            let mut tampered = assignment.clone();
            let &(column, _) = pin_terms
                .first()
                .expect("program cursor has one direct decoder");
            tampered[column] += F::ONE;
            assert_ne!(
                evaluate_relation_row(&relation, pin_rows.start, &tampered),
                F::ZERO,
                "the exact first program-cursor pin row must reject a changed decoded value"
            );
        }

        for link in &contract.fields {
            let base_terms = relation
                .base_field_decoding_terms(base_kind, link.phase_field)
                .expect("base source field decoder in final columns");
            let overlay_terms = relation
                .overlay_field_decoding_terms(overlay_kind, link.overlay_field)
                .expect("overlay source field decoder in final columns");
            if let Some((base_start, base_width)) = relation
                .base_relation()
                .field_slot(base_kind, link.phase_field)
            {
                assert_eq!(base_terms, direct_slot_terms(base_start, base_width));
            } else {
                saw_affine_defined_link = true;
                saw_zero_affine_link |= base_terms.is_empty();
            }
            if let Some((overlay_start, overlay_width)) = relation
                .overlay_relation()
                .field_slot(overlay_kind, link.overlay_field)
            {
                let embedded_overlay_start = if overlay_start == 0 {
                    0
                } else {
                    layout.overlay_private_columns().start + overlay_start - 1
                };
                assert_eq!(overlay_terms, direct_slot_terms(embedded_overlay_start, overlay_width));
            } else {
                saw_affine_defined_link = true;
                saw_zero_affine_link |= overlay_terms.is_empty();
            }
            let decoded_base = evaluate_decoder(&base_terms, &assignment);
            let decoded_overlay = evaluate_decoder(&overlay_terms, &assignment);
            assert_eq!(decoded_base, base_source[link.phase_field]);
            assert_eq!(decoded_overlay, overlay_source[link.overlay_field]);
            assert_eq!(decoded_base, decoded_overlay);
        }

        if overlay_kind + 1 == ACTIVE_CHUNKS {
            final_assignment = Some(assignment);
        }
    }
    assert_eq!(previous_link_end, layout.field_link_rows().end);
    assert_eq!(previous_pin_end, layout.base_field_pin_rows().end);
    assert!(
        saw_affine_defined_link,
        "the exact link contract must retain affine-definition decoder provenance"
    );
    assert!(
        saw_zero_affine_link,
        "the exact link contract must retain zero affine-definition decoder provenance"
    );

    let final_assignment = final_assignment.expect("terminal claim assignment");
    assert_eq!(
        relation.first_unsatisfied_row(&final_assignment),
        None,
        "the exact selected terminal assignment must satisfy every final row"
    );
}
