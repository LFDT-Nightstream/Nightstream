//! Regressions for the active strict-PiDEC source and selective audits.
//!
//! The ordinary test covers the exact carrier, source equations, row census,
//! leaf ownership, and complete rewrite expansion. The final selective term
//! projection is a separate ignored diagnostic because the full fixed-point
//! emitter exceeds this repository's unconditional five-minute test cap.

#[path = "../support/mod.rs"]
mod support;

use std::collections::BTreeSet;

use neo_fold_clean::frontends::r1cs_f_prime::ivc::{R1csIvcBranch, R1csIvcPiDecSourceRowsAudit, R1csIvcRelation};
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::reductions::pi_dec_circuit::stage;
use support::r1cs_compiler_fixtures::{make_tiny_lifecycle_plan, one_product_r1cs, tiny_params};

fn assert_source_audit(params: &Params, audit: &R1csIvcPiDecSourceRowsAudit) {
    let strict = audit.strict();
    assert_eq!(strict.radix, 2);
    assert_eq!(strict.children.len(), 14);
    assert_eq!(strict.x_sign_traces.len(), 270);
    assert!(
        std::iter::once(&strict.parent)
            .chain(&strict.children)
            .all(|claim| claim.adv.is_none()),
        "the exact paper carrier has no product-commitment advice"
    );
    let active_source_rows = strict.row_end - strict.row_start;
    let expected_active_source_rows = 54 * params.kappa() as usize + 11_629;
    let prior_source_rows = 54 * params.kappa() as usize + 15_129;
    assert_eq!(
        active_source_rows, expected_active_source_rows,
        "active strict PiDEC source-field R1CS row census"
    );
    assert_eq!(
        prior_source_rows - active_source_rows,
        3_500,
        "canonical-X and semantic-prefix-y remove exactly 3,500 source rows"
    );
    assert_eq!(
        strict
            .x_sign_traces
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>()
            .len(),
        540,
        "all sign/product trace columns are distinct"
    );

    let leaves = audit.leaf_source_ranges();
    assert_eq!(
        leaves.iter().map(|range| range.name).collect::<Vec<_>>(),
        stage::LEAVES,
        "every strict PiDEC leaf occurs exactly once in emission order"
    );
    assert_eq!(
        leaves
            .iter()
            .map(|range| range.row_end - range.row_start)
            .collect::<Vec<_>>(),
        vec![
            54 * params.kappa() as usize,
            0,
            270,
            1_404,
            70,
            672,
            532,
            15,
            4_320,
            390,
            3_900,
            56,
        ],
        "exact source-field R1CS rows per active PiDEC leaf"
    );
    let mut cursor = strict.row_start;
    for range in leaves {
        assert_eq!(range.row_start, cursor, "PiDEC leaves form a gapless partition");
        assert!(range.row_end >= range.row_start);
        cursor = range.row_end;
    }
    assert_eq!(cursor, strict.row_end);

    assert!(
        (strict.row_start..strict.row_end).all(|row| audit.source_rows().binary_search(&row).is_ok()),
        "expanded source set retains every strict PiDEC row"
    );
    assert_eq!(audit.source_rows().len(), audit.source_row_artifacts().len());
    assert!(audit
        .source_rows()
        .iter()
        .copied()
        .eq(audit.source_row_artifacts().iter().map(|row| row.index())));
    for row in audit.source_row_artifacts() {
        for port in [row.a(), row.b(), row.c()] {
            assert!(port.windows(2).all(|pair| pair[0].0 < pair[1].0));
        }
    }

    let expanded = audit.source_rows().iter().copied().collect::<BTreeSet<_>>();
    let selected_rewrites = audit
        .fixed_point()
        .rows()
        .rewrites()
        .iter()
        .filter(|rewrite| {
            rewrite.arm() == R1csIvcBranch::Recursive as usize
                && rewrite
                    .source_rows()
                    .iter()
                    .flat_map(|range| range.clone())
                    .any(|row| expanded.contains(&row))
        })
        .collect::<Vec<_>>();
    for rewrite in &selected_rewrites {
        assert!(
            rewrite
                .source_rows()
                .iter()
                .flat_map(|range| range.clone())
                .all(|row| expanded.contains(&row)),
            "every selected rewrite contributes its complete source runs"
        );
    }
    eprintln!(
        "[pi-dec-source] kappa={} strict_source_rows={} expanded_source_rows={} source_ranges={} selected_rewrites={} rewrite_output_rows={}",
        params.kappa(),
        active_source_rows,
        audit.source_rows().len(),
        audit.source_row_ranges().len(),
        selected_rewrites.len(),
        selected_rewrites
            .iter()
            .map(|rewrite| rewrite.emitted_rows().len())
            .sum::<usize>(),
    );
}

#[test]
fn active_strict_pi_dec_source_rows_have_exact_layout_and_census() {
    let params = tiny_params();
    let app = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(app.m(), app.m_in);
    let audit = R1csIvcRelation::audit_fixed_point_pi_dec_source_rows(&params, &app.into(), &plan)
        .expect("audit active strict PiDEC source rows");
    assert_source_audit(&params, &audit);
}

#[test]
#[ignore = "full fixed-point selective term projection exceeds the unconditional five-minute test cap; use the source audit for ordinary regression"]
fn diagnostic_active_strict_pi_dec_selective_rows_have_exact_provenance() {
    let params = tiny_params();
    let app = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(app.m(), app.m_in);
    let audit = R1csIvcRelation::audit_fixed_point_pi_dec_rows(&params, &app.into(), &plan)
        .expect("project active strict PiDEC selective rows");
    assert_source_audit(&params, audit.source());

    let projected = audit.projected_rows();
    assert!(!projected.row_artifacts().is_empty());
    assert!(projected
        .row_artifacts()
        .windows(2)
        .all(|pair| pair[0].emitted_row() < pair[1].emitted_row()));
    let source = projected
        .source_provenance()
        .expect("PiDEC projection has exact source provenance");
    let decoder = projected
        .decoder_provenance()
        .expect("PiDEC projection has exact source decoder");
    assert_eq!(source.arm(), R1csIvcBranch::Recursive as usize);
    assert_eq!(decoder.arm(), R1csIvcBranch::Recursive as usize);
    assert!(source
        .source_columns()
        .iter()
        .copied()
        .eq(decoder.decoders().iter().map(|entry| entry.column())));
    let decoded = decoder
        .decoders()
        .iter()
        .map(|entry| entry.column())
        .collect::<BTreeSet<_>>();
    assert!(audit
        .strict()
        .x_sign_traces
        .iter()
        .flatten()
        .all(|column| decoded.contains(column)));
}
