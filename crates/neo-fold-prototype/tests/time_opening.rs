use neo_fold_prototype::core::opening::time::{prove_time_opening, verify_time_opening};
use neo_fold_prototype::core::opening::{OpeningClaim, OpeningDomain, OpeningSource, TimeOpeningProofSummary};
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

fn claim(
    source: OpeningSource,
    domain: OpeningDomain,
    point: Vec<K>,
    ordinal: u64,
    column_ids: Vec<u32>,
    byte: u8,
) -> OpeningClaim {
    OpeningClaim {
        source,
        domain,
        point,
        ordinal,
        column_ids,
        digest: [byte; 32],
    }
}

fn point(tag: u64) -> Vec<K> {
    vec![K::from(F::from_u64(tag))]
}

#[test]
fn time_opening_is_canonical_under_claim_reordering() {
    let main_lane = vec![
        claim(OpeningSource::MainLane, OpeningDomain::Cpu, point(0), 0, vec![0], 1),
        claim(OpeningSource::MainLane, OpeningDomain::Cpu, point(1), 0, vec![0], 2),
    ];
    let extension_claims = vec![
        claim(
            OpeningSource::ExtensionKernel,
            OpeningDomain::Mem,
            point(3),
            0,
            vec![1000],
            9,
        ),
        claim(
            OpeningSource::ExtensionRoot,
            OpeningDomain::Mem,
            point(7),
            0,
            vec![100, 101],
            4,
        ),
        claim(
            OpeningSource::ExtensionKernel,
            OpeningDomain::Mem,
            point(3),
            1,
            vec![1001],
            5,
        ),
        claim(
            OpeningSource::ExtensionRoot,
            OpeningDomain::Mem,
            point(8),
            0,
            vec![10000],
            6,
        ),
    ];

    let summary = prove_time_opening(&main_lane, &extension_claims).expect("time opening summary");
    let reordered = vec![
        extension_claims[3].clone(),
        extension_claims[1].clone(),
        extension_claims[0].clone(),
        extension_claims[2].clone(),
    ];

    verify_time_opening(&main_lane, &reordered, &Some(summary.clone()))
        .expect("canonical manifest/reduction should ignore input order");
    assert!(summary.groups.len() >= 4);
    assert!(!summary.unification.round_polys.is_empty());
    assert_eq!(summary.unification.r_unify.len(), summary.unification.round_polys.len());
    assert!(!summary.can_unify);
    assert_eq!(summary.unified_domain, OpeningDomain::Cpu);
    assert!(!summary.unified_point.is_empty());
}

#[test]
fn time_opening_rejects_tampered_unified_digest() {
    let main_lane = vec![claim(
        OpeningSource::MainLane,
        OpeningDomain::Cpu,
        point(0),
        0,
        vec![0],
        1,
    )];
    let extension_claims = vec![
        claim(
            OpeningSource::ExtensionKernel,
            OpeningDomain::Mem,
            point(3),
            0,
            vec![1000],
            9,
        ),
        claim(
            OpeningSource::ExtensionKernel,
            OpeningDomain::Mem,
            point(3),
            1,
            vec![1001],
            5,
        ),
    ];

    let mut summary = prove_time_opening(&main_lane, &extension_claims).expect("time opening summary");
    summary.unified_digest[0] ^= 1;

    let err = verify_time_opening(&main_lane, &extension_claims, &Some(summary))
        .expect_err("tampered unified digest must fail");
    assert!(format!("{err}").contains("unified digest"));
}

#[test]
fn time_opening_rejects_tampered_group_summary() {
    let main_lane = vec![claim(
        OpeningSource::MainLane,
        OpeningDomain::Cpu,
        point(0),
        0,
        vec![0],
        1,
    )];
    let extension_claims = vec![
        claim(
            OpeningSource::ExtensionKernel,
            OpeningDomain::Mem,
            point(4),
            0,
            vec![100, 101],
            8,
        ),
        claim(
            OpeningSource::ExtensionRoot,
            OpeningDomain::Mem,
            point(4),
            0,
            vec![10000],
            7,
        ),
    ];

    let mut summary: TimeOpeningProofSummary =
        prove_time_opening(&main_lane, &extension_claims).expect("time opening summary");
    summary.groups[0].group_digest[0] ^= 1;

    let err = verify_time_opening(&main_lane, &extension_claims, &Some(summary))
        .expect_err("tampered group summary must fail");
    assert!(format!("{err}").contains("group summary"));
}

#[test]
fn time_opening_rejects_tampered_group_coefficients() {
    let main_lane = vec![claim(
        OpeningSource::MainLane,
        OpeningDomain::Cpu,
        point(0),
        0,
        vec![0],
        1,
    )];
    let extension_claims = vec![
        claim(
            OpeningSource::ExtensionKernel,
            OpeningDomain::Mem,
            point(3),
            0,
            vec![1000],
            9,
        ),
        claim(
            OpeningSource::ExtensionKernel,
            OpeningDomain::Mem,
            point(3),
            1,
            vec![1001],
            5,
        ),
    ];

    let mut summary = prove_time_opening(&main_lane, &extension_claims).expect("time opening summary");
    summary.groups[0].coefficients[0] += K::ONE;

    let err = verify_time_opening(&main_lane, &extension_claims, &Some(summary))
        .expect_err("tampered group coefficients must fail");
    assert!(format!("{err}").contains("group summary"));
}

#[test]
fn time_opening_rejects_tampered_unification_rounds() {
    let main_lane = vec![claim(
        OpeningSource::MainLane,
        OpeningDomain::Cpu,
        point(0),
        0,
        vec![0],
        1,
    )];
    let extension_claims = vec![
        claim(
            OpeningSource::ExtensionKernel,
            OpeningDomain::Mem,
            point(3),
            0,
            vec![1000],
            9,
        ),
        claim(
            OpeningSource::ExtensionKernel,
            OpeningDomain::Mem,
            point(3),
            1,
            vec![1001],
            5,
        ),
    ];

    let mut summary = prove_time_opening(&main_lane, &extension_claims).expect("time opening summary");
    summary.unification.round_polys[0][0] += K::ONE;

    let err = verify_time_opening(&main_lane, &extension_claims, &Some(summary))
        .expect_err("tampered unification rounds must fail");
    assert!(format!("{err}").contains("unification"));
}

#[test]
fn time_opening_rejects_duplicate_claims() {
    let dup = claim(
        OpeningSource::ExtensionKernel,
        OpeningDomain::Mem,
        point(3),
        0,
        vec![1000],
        9,
    );
    let err = prove_time_opening(&[], &[dup.clone(), dup]).expect_err("duplicate claims must fail");
    assert!(format!("{err}").contains("duplicate claims"));
}

#[test]
fn time_opening_rejects_unsorted_column_ids() {
    let err = prove_time_opening(
        &[],
        &[claim(
            OpeningSource::ExtensionKernel,
            OpeningDomain::Mem,
            point(7),
            0,
            vec![101, 100],
            1,
        )],
    )
    .expect_err("unsorted column ids must fail");
    assert!(format!("{err}").contains("column_ids"));
}

#[test]
fn time_opening_keeps_groups_separate_across_sources_but_unifies_anchor() {
    let summary = prove_time_opening(
        &[],
        &[
            claim(
                OpeningSource::ExtensionKernel,
                OpeningDomain::Mem,
                point(9),
                0,
                vec![100, 101],
                1,
            ),
            claim(
                OpeningSource::ExtensionRoot,
                OpeningDomain::Mem,
                point(9),
                0,
                vec![1000],
                2,
            ),
        ],
    )
    .expect("time opening summary");

    assert_eq!(summary.groups.len(), 2);
    assert_eq!(summary.groups[0].sources, vec![OpeningSource::ExtensionKernel]);
    assert_eq!(summary.groups[1].sources, vec![OpeningSource::ExtensionRoot]);
    assert!(summary.can_unify);
    assert_eq!(summary.unified_domain, OpeningDomain::Mem);
    assert_eq!(summary.unified_point, point(9));
    assert_eq!(summary.unification.round_polys.len(), 1);
    assert_eq!(summary.unification.r_unify.len(), 1);
}

#[test]
fn time_opening_rejects_tampered_unified_point() {
    let summary = prove_time_opening(
        &[],
        &[
            claim(
                OpeningSource::ExtensionKernel,
                OpeningDomain::Mem,
                point(4),
                0,
                vec![100, 101],
                8,
            ),
            claim(
                OpeningSource::ExtensionRoot,
                OpeningDomain::Cpu,
                point(7),
                0,
                vec![10000],
                7,
            ),
        ],
    )
    .expect("time opening summary");
    let mut tampered = summary.clone();
    tampered.unified_point[0] += K::ONE;

    let err = verify_time_opening(
        &[],
        &[
            claim(
                OpeningSource::ExtensionKernel,
                OpeningDomain::Mem,
                point(4),
                0,
                vec![100, 101],
                8,
            ),
            claim(
                OpeningSource::ExtensionRoot,
                OpeningDomain::Cpu,
                point(7),
                0,
                vec![10000],
                7,
            ),
        ],
        &Some(tampered),
    )
    .expect_err("tampered unified point must fail");
    assert!(format!("{err}").contains("unified point"));
}
