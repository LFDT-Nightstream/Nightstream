//! Digest measurement for the campaign profile freeze (bar 2).
//!
//! The `#[ignore]` printer measures candidate profiles. The freeze document
//! and its drift test pin the chosen digests.

use std::time::Instant;

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_fold_clean::frontends::direct_ccs::ajtai;
use neo_fold_clean::frontends::nebula::f_prime::{NebulaFPrimeBranch, NebulaFPrimeRelation};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::{compile_combined_terminal_r1cs, TerminalR1csInput};
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::relations::{
    superneo_has_canonical_x_shape, superneo_public_x_cols, CcsInstance, CeClaim, WitnessMat,
};
use neo_math::{D, F, K};
use nightstream_constraint_exporter::{
    export_complete_nebula_problem, export_complete_terminal_problem, export_nebula_problem, nebula_family_census,
    ExportRequest,
};
use p3_field::PrimeCharacteristicRing;
use recursive_constraint_minimizer::Scope;

#[path = "../../../../crates/neo-fold-clean/tests/support/lean_manifest_fixture.rs"]
mod lean_manifest_fixture;
use lean_manifest_fixture::{combined_manifest, parse_combined, TEST_AJTAI_SEED};

const FREEZE_PLAN_SEED: [u8; 32] = [0xF5; 32];
const FREEZE_ROM: [u32; 1] = [7];

fn freeze_candidate_audit(
    params: &Params,
) -> neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimeConstraintSourceAudit {
    let memory = NebulaParams::new(0, 0, 1, 2, 1).expect("freeze memory profile");
    let plan = NebulaPlan::new(memory, FREEZE_ROM.to_vec(), FREEZE_PLAN_SEED, params.kappa() as usize)
        .expect("freeze Nebula plan");
    NebulaFPrimeRelation::audit_fixed_point_constraint_sources(params, &plan)
        .expect("discover freeze-candidate source arms")
}

#[test]
#[ignore = "measurement printer for the profile freeze; run with --ignored --nocapture"]
fn print_paper_b2_freeze_candidate_digests() {
    print_freeze_candidate("paper-b2", Params::goldilocks_paper_b2());
}

#[test]
#[ignore = "measurement printer for the profile freeze; run with --ignored --nocapture"]
fn print_paper_b2_lambda114_freeze_candidate_digests() {
    // Diagnostic only: paper B.2 with the security target lowered to the 114
    // bits this shape's census provides. The regime decision is not made here.
    let inner = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        neo_params::goldilocks_paper_b2::KAPPA,
        neo_params::goldilocks_paper_b2::M,
        neo_params::goldilocks_paper_b2::B_BASE,
        neo_params::goldilocks_paper_b2::K_RHO,
        neo_params::goldilocks_paper_b2::T,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        114,
    )
    .expect("paper B.2 shape with a 114-bit target");
    print_freeze_candidate("paper-b2-lambda114", Params::test_only_from_neo_params(inner));
}

fn campaign_audit(
    plan_seed: [u8; 32],
) -> neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimeConstraintSourceAudit {
    let params = nightstream_constraint_exporter::campaign_profile_params();
    let memory = NebulaParams::new(0, 0, 1, 2, 1).expect("campaign memory profile");
    let plan = NebulaPlan::new(memory, vec![7], plan_seed, params.kappa() as usize).expect("campaign Nebula plan");
    NebulaFPrimeRelation::audit_fixed_point_constraint_sources(&params, &plan).expect("discover campaign source arms")
}

fn compile_terminal_fixture() -> neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::CompiledTerminalR1cs {
    let manifest = parse_combined(&combined_manifest()).expect("valid combined manifest");
    let mut public = vec![F::ZERO; manifest.public_carrier_width()];
    public[0] = F::ONE;
    let emission = manifest
        .emit(&public, |_| Some(F::ZERO), &[F::ZERO])
        .expect("honest combined emission");
    assert!(emission.is_satisfied());

    let params = Params::goldilocks_paper_b2();
    let log = ajtai::setup_seeded(&params, emission.structure(), TEST_AJTAI_SEED);
    let fresh = CcsInstance::from_low_norm_assignment(
        &params,
        &log,
        emission.structure(),
        emission.assignment(),
        manifest.public_carrier_width(),
    )
    .expect("honest combined fresh instance");
    let zero_witness = Mat::zero(D, emission.structure().m / D, F::ZERO);
    let joint_row_variables = emission
        .structure()
        .n
        .max(emission.structure().m)
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;
    let zero_claim = CeClaim {
        c: Commitment::zeros(D, manifest.terminal_r1cs().verifier_rows()),
        X: {
            let x = Mat::zero(D, superneo_public_x_cols(manifest.public_carrier_width()), F::ZERO);
            assert!(superneo_has_canonical_x_shape(&x, manifest.public_carrier_width()));
            x
        },
        r: vec![K::ZERO; joint_row_variables],
        y_ring: vec![vec![K::ZERO; D.next_power_of_two()]; emission.structure().t() + 1],
        ct: vec![K::ZERO; emission.structure().t() + 1],
        m_in: manifest.public_carrier_width(),
        fold_digest: [0; 32],
        adv: None,
    };
    let running_claims = vec![zero_claim; 14];
    let running_witnesses: Vec<WitnessMat> = vec![zero_witness; 14];

    compile_combined_terminal_r1cs(
        &manifest,
        &log,
        TerminalR1csInput {
            running_claims: &running_claims,
            running_witnesses: &running_witnesses,
            fresh: &fresh,
        },
    )
    .expect("honest combined terminal R1CS")
}

#[test]
fn campaign_profile_v2_digests_are_frozen() {
    // PROFILE.md is the freeze document. These pins must match its table.
    const BASE_SOURCE_DIGEST: &str = "sha256:e5f31e44449fd9bdf41f742f0afd6a9cee93be2fe98b1dedfa4d27f6aa250570";
    const RECURSIVE_SOURCE_DIGEST: &str = "sha256:f06cd06435b8060f0c94adaddeb8349a24ba784b974a6bac7a06ca9e93163915";
    const FINAL_PLAN_DIGEST: &str = "sha256:42eb7385d90b1de44cb67a505ae5ba1634559f105c315031acb681401449b965";
    const TERMINAL_SOURCE_DIGEST: &str = "sha256:85b400cebcfaa8fac702072aff342d67c6acca87e4470199d86a935c98264461";
    const TERMINAL_DIAGNOSTIC_DIGEST: &str = "sha256:63664e95c3f91dcf35db99ad3e0dd235643d274e5ccfd9be6a18252eb8a12f98";

    let audit = campaign_audit([0xDA; 32]);
    for (branch, source_digest, rows, columns, families) in [
        (NebulaFPrimeBranch::Base, BASE_SOURCE_DIGEST, 39_949, 38_626, 6),
        (
            NebulaFPrimeBranch::Recursive,
            RECURSIVE_SOURCE_DIGEST,
            11_187_825,
            11_078_210,
            82,
        ),
    ] {
        let census = nebula_family_census(&audit, branch).expect("complete reviewed Nebula family ownership");
        assert_eq!(census.len(), families, "{branch:?} family count drifted");
        let arm = audit.arm(branch);
        let export = export_nebula_problem(
            &audit,
            branch,
            ExportRequest {
                profile: "campaign-profile-v1-freeze-gate".to_owned(),
                scope: Scope::Branch,
                public_input_count: arm.m_in,
                source_rows: vec![0],
                complete_families: Vec::new(),
            },
        )
        .expect("export one frozen-profile source row");
        let problem = export.problem();
        let binding = export.binding();
        assert_eq!(
            problem.source.artifact_digest, source_digest,
            "{branch:?} source digest drifted"
        );
        assert_eq!(problem.source.total_rows, rows, "{branch:?} source rows drifted");
        assert_eq!(problem.column_count, columns, "{branch:?} source columns drifted");
        assert_eq!(problem.public_input_count, 2_426, "{branch:?} public prefix drifted");
        assert_eq!(
            binding.final_plan_digest(),
            FINAL_PLAN_DIGEST,
            "{branch:?} final plan digest drifted"
        );
        assert_eq!(binding.final_rows(), 3_666_055, "{branch:?} final rows drifted");
        assert_eq!(binding.final_columns(), 13_314_834, "{branch:?} final columns drifted");
        assert_eq!(
            binding.final_public_input_count(),
            2_430,
            "{branch:?} final public columns drifted"
        );
    }

    let relation = compile_terminal_fixture();
    let terminal_audit = relation.constraint_audit();
    let export = export_complete_terminal_problem(&terminal_audit, "campaign-terminal-classification-v1")
        .expect("export the complete frozen terminal relation");
    let problem = export.problem();
    let binding = export.binding();
    assert_eq!(
        problem.source.artifact_digest, TERMINAL_SOURCE_DIGEST,
        "terminal source digest drifted"
    );
    assert_eq!(problem.source.total_rows, 58_593, "terminal source rows drifted");
    assert_eq!(problem.column_count, 58_592, "terminal source columns drifted");
    assert_eq!(problem.public_input_count, 48_871, "terminal public prefix drifted");
    assert_eq!(problem.complete_families.len(), 8, "terminal family count drifted");
    assert_eq!(
        binding.diagnostic_digest(),
        TERMINAL_DIAGNOSTIC_DIGEST,
        "terminal diagnostic digest drifted"
    );
    assert_eq!(binding.spartan_rows(), 65_536, "terminal Spartan rows drifted");
    assert_eq!(binding.spartan_columns(), 114_407, "terminal Spartan columns drifted");
}

#[test]
#[ignore = "y_ring slice measurement at the amended shape; run with --ignored --nocapture"]
fn probe_y_ring_slice_constants() {
    use nightstream_constraint_exporter::export_nebula_problem;
    use recursive_constraint_minimizer::{derive_scalar_certificate, Selection};
    use std::collections::BTreeSet;

    const CANDIDATE: &str = "nifs.pi_rlc.verify.padding.y_ring";
    const PI_CCS_SUPPORT: &str = "nifs.pi_ccs.padded_row.canonicality";
    const PI_DEC_SUPPORT: &str = "nifs.pi_dec.verify";

    let audit = campaign_audit([0xDA; 32]);
    let branch = NebulaFPrimeBranch::Recursive;
    let arm = audit.arm(branch);
    let census = nebula_family_census(&audit, branch).expect("census");
    let family = |name: &str| {
        census
            .iter()
            .find(|family| family.name() == name)
            .unwrap_or_else(|| panic!("missing {name}"))
    };
    let candidate_export = export_nebula_problem(
        &audit,
        branch,
        ExportRequest {
            profile: "y-ring-slice-probe".to_owned(),
            scope: Scope::Branch,
            public_input_count: arm.m_in,
            source_rows: family(CANDIDATE).source_rows().to_vec(),
            complete_families: vec![CANDIDATE.to_owned()],
        },
    )
    .expect("bind candidate rows");
    eprintln!(
        "y_ring: family_rows={} retained={} rewrites={} closure={} emitted={}",
        family(CANDIDATE).source_rows().len(),
        candidate_export.binding().retained_rows().len(),
        candidate_export.binding().rewrites().len(),
        candidate_export.binding().closure_source_rows().len(),
        candidate_export.binding().emitted_rows().len(),
    );
    let source_rows = [CANDIDATE, PI_CCS_SUPPORT, PI_DEC_SUPPORT]
        .into_iter()
        .flat_map(|name| family(name).source_rows().iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let export = export_nebula_problem(
        &audit,
        branch,
        ExportRequest {
            profile: "y-ring-slice-probe".to_owned(),
            scope: Scope::Branch,
            public_input_count: arm.m_in,
            source_rows,
            complete_families: vec![CANDIDATE.to_owned()],
        },
    )
    .expect("export candidate and supports");
    let certificate = derive_scalar_certificate(export.problem(), &Selection::Family(CANDIDATE.to_owned()))
        .expect("derive")
        .expect("certificate exists");
    let mut ccs = 0;
    let mut dec = 0;
    for row in &certificate.rows {
        let [support] = row.support.as_slice() else {
            panic!("one support")
        };
        let support_row = export
            .problem()
            .rows
            .binary_search_by_key(&support.source_index, |row| row.source_index)
            .map(|index| &export.problem().rows[index])
            .expect("support in slice");
        match support_row.family.as_str() {
            PI_CCS_SUPPORT => ccs += 1,
            PI_DEC_SUPPORT => dec += 1,
            other => panic!("unexpected support family {other}"),
        }
    }
    eprintln!(
        "y_ring certificate: rows={} pi_ccs_support={ccs} pi_dec_support={dec}",
        certificate.rows.len()
    );
}

#[test]
#[ignore = "column-ownership census: which families exclusively own columns; run with --ignored --nocapture"]
fn probe_recursive_column_ownership_census() {
    use neo_ccs::CcsMatrix;
    use nightstream_constraint_exporter::sparse_family_census;

    let audit = campaign_audit([0xDA; 32]);
    let arm = audit.arm(NebulaFPrimeBranch::Recursive);
    let census = sparse_family_census(arm).expect("complete reviewed ownership");
    let mut row_family = vec![u16::MAX; arm.n];
    for (family_index, family) in census.iter().enumerate() {
        for &row in family.source_rows() {
            row_family[row] = family_index as u16;
        }
    }
    // Ownership per column: untouched, one family, or shared.
    const UNUSED: u16 = u16::MAX;
    const SHARED: u16 = u16::MAX - 1;
    let mut owner = vec![UNUSED; arm.m];
    let mark = |row: usize, column: usize, owner: &mut Vec<u16>| {
        let family = row_family[row];
        owner[column] = match owner[column] {
            UNUSED => family,
            current if current == family => current,
            _ => SHARED,
        };
    };
    for matrix in [&arm.a, &arm.b, &arm.c] {
        match matrix {
            CcsMatrix::Csc(csc) => {
                for column in 0..csc.ncols {
                    for k in csc.column_range(column) {
                        mark(csc.row_index(k), column, &mut owner);
                    }
                }
            }
            CcsMatrix::CscWithSeededPhi81 { csc, blocks, .. } => {
                for column in 0..csc.ncols {
                    for k in csc.column_range(column) {
                        mark(csc.row_index(k), column, &mut owner);
                    }
                }
                for block in blocks {
                    for row in block.row_start()..block.row_end() {
                        for &start in block.word_starts() {
                            for column in start..start + block.word_width() {
                                mark(row, column, &mut owner);
                            }
                        }
                    }
                }
            }
            _ => panic!("unexpected matrix variant"),
        }
    }
    let mut exclusive = vec![0usize; census.len()];
    let mut shared = 0usize;
    let mut unused = 0usize;
    for &column_owner in &owner {
        match column_owner {
            UNUSED => unused += 1,
            SHARED => shared += 1,
            family => exclusive[family as usize] += 1,
        }
    }
    let mut ranked = census
        .iter()
        .enumerate()
        .map(|(index, family)| (exclusive[index], family.name(), family.source_rows().len()))
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|left, right| right.0.cmp(&left.0));
    eprintln!(
        "recursive column ownership: columns={} shared={shared} unused={unused}",
        arm.m
    );
    for (columns, name, rows) in ranked.iter().take(25) {
        eprintln!("exclusive_columns={columns} rows={rows} family={name}");
    }
    let exclusive_total = exclusive.iter().sum::<usize>();
    eprintln!("exclusive_total={exclusive_total}");
}

#[test]
#[ignore = "k_rho=12 digest probe for the bar-2 amendment; run with --ignored --nocapture"]
fn probe_k_rho_12_digests() {
    let inner = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        1,
        neo_params::goldilocks_paper_b2::M,
        neo_params::goldilocks_paper_b2::B_BASE,
        12,
        1,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        1,
    )
    .expect("k_rho=12 minimal parameters");
    let params = Params::test_only_from_neo_params(inner);
    let memory = NebulaParams::new(0, 0, 1, 2, 1).expect("campaign memory profile");
    let plan = NebulaPlan::new(memory, vec![7], [0xDA; 32], params.kappa() as usize).expect("campaign Nebula plan");
    let audit = NebulaFPrimeRelation::audit_fixed_point_constraint_sources(&params, &plan)
        .expect("discover k_rho=10 source arms");
    for branch in [NebulaFPrimeBranch::Base, NebulaFPrimeBranch::Recursive] {
        let arm = audit.arm(branch);
        let export = export_nebula_problem(
            &audit,
            branch,
            ExportRequest {
                profile: "k-rho-12-digest-probe".to_owned(),
                scope: Scope::Branch,
                public_input_count: arm.m_in,
                source_rows: vec![0],
                complete_families: Vec::new(),
            },
        )
        .expect("export one k_rho=10 source row");
        let census = nebula_family_census(&audit, branch).expect("census");
        eprintln!(
            "k_rho=12 branch={branch:?} n={} m={} m_in={} families={} digest={} final_rows={} final_cols={} final_plan_digest={}",
            export.problem().source.total_rows,
            export.problem().column_count,
            export.problem().public_input_count,
            census.len(),
            export.problem().source.artifact_digest,
            export.binding().final_rows(),
            export.binding().final_columns(),
            export.binding().final_plan_digest(),
        );
    }
}

#[test]
#[ignore = "k_rho=12 encoding-limit probe for the compact pipeline; run with --ignored --nocapture"]
fn probe_k_rho_12_encoding_limits() {
    use neo_ccs::CcsMatrix;
    use std::collections::HashSet;

    let inner = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        1,
        neo_params::goldilocks_paper_b2::M,
        neo_params::goldilocks_paper_b2::B_BASE,
        12,
        1,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        1,
    )
    .expect("k_rho=12 minimal parameters");
    let params = Params::test_only_from_neo_params(inner);
    let memory = NebulaParams::new(0, 0, 1, 2, 1).expect("campaign memory profile");
    let plan = NebulaPlan::new(memory, vec![7], [0xDA; 32], params.kappa() as usize).expect("campaign Nebula plan");
    let audit = NebulaFPrimeRelation::audit_fixed_point_constraint_sources(&params, &plan)
        .expect("discover k_rho=10 source arms");
    let arm = audit.arm(NebulaFPrimeBranch::Recursive);
    let mut values = HashSet::new();
    let mut nnz = 0usize;
    let mut row_terms = vec![0u32; arm.n];
    let mut blocks_total = 0usize;
    let mut geometric_total = 0usize;
    for matrix in [&arm.a, &arm.b, &arm.c] {
        match matrix {
            CcsMatrix::Csc(csc) => {
                nnz += csc.vals.len();
                for value in &csc.vals {
                    values.insert(p3_field::PrimeField64::as_canonical_u64(value));
                }
                for column in 0..csc.ncols {
                    for k in csc.column_range(column) {
                        row_terms[csc.row_index(k)] += 1;
                    }
                }
            }
            CcsMatrix::CscWithSeededPhi81 {
                csc,
                blocks,
                geometric_runs,
            } => {
                nnz += csc.vals.len();
                blocks_total += blocks.len();
                geometric_total += geometric_runs.len();
                for value in &csc.vals {
                    values.insert(p3_field::PrimeField64::as_canonical_u64(value));
                }
                for column in 0..csc.ncols {
                    for k in csc.column_range(column) {
                        row_terms[csc.row_index(k)] += 1;
                    }
                }
            }
            _ => panic!("unexpected matrix variant"),
        }
    }
    eprintln!(
        "k_rho=12 recursive encodings: nnz={nnz} distinct_values={} max_row_terms={} seeded_blocks={blocks_total} geometric_runs={geometric_total}",
        values.len(),
        row_terms.iter().max().copied().unwrap_or(0),
    );
}

#[test]
#[ignore = "k_rho=12 foldable-shape probe for the bar-2 amendment; run with --ignored --nocapture"]
fn probe_k_rho_12_minimal_shape_capture() {
    use neo_fold_clean::frontends::nebula::f_prime::{NebulaFPrimeChainBuilder, NebulaFPrimePreprocessing};
    use neo_fold_clean::frontends::nebula::trace::Memory;

    let inner = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        1,
        neo_params::goldilocks_paper_b2::M,
        neo_params::goldilocks_paper_b2::B_BASE,
        12,
        1,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        1,
    )
    .expect("k_rho=12 minimal parameters");
    let params = Params::test_only_from_neo_params(inner);
    let memory_params = NebulaParams::new(0, 0, 1, 2, 2).expect("two-step memory profile");
    let rom = [7];
    let plan = NebulaPlan::new(memory_params, rom.to_vec(), [0xDA; 32], params.kappa() as usize).expect("Nebula plan");

    let start = Instant::now();
    let audit = NebulaFPrimeRelation::audit_fixed_point_constraint_sources(&params, &plan)
        .expect("discover k_rho=10 source arms");
    eprintln!("k_rho=12 audit build: {} ms", start.elapsed().as_millis());
    for branch in [NebulaFPrimeBranch::Base, NebulaFPrimeBranch::Recursive] {
        let arm = audit.arm(branch);
        eprintln!("k_rho=12 branch={branch:?} n={} m={} m_in={}", arm.n, arm.m, arm.m_in);
    }

    let prep = NebulaFPrimePreprocessing::new_seeded(params, plan, 0xDA00_0001).expect("k_rho=12 Nebula preprocessing");
    let mut memory = Memory::new(memory_params, &rom).expect("memory");
    let mut chain = NebulaFPrimeChainBuilder::new(&prep);
    for index in 0..2usize {
        let step = Instant::now();
        let trace = {
            let mut segment = memory.begin_segment().expect("segment");
            segment.write(true, 0, 5 + index as u32).expect("RAM write");
            segment.finish().expect("accepted trace")
        };
        let witnesses = chain
            .append_segment_with_constraint_witness_audit(&trace)
            .expect("accepted k_rho=12 Nebula step");
        eprintln!(
            "k_rho=12 step {index}: branch={:?} assignment_len={} ms={}",
            witnesses[0].branch(),
            witnesses[0].source_assignment().len(),
            step.elapsed().as_millis(),
        );
    }
    eprintln!("k_rho=12 two-segment capture SUCCEEDED");
}

#[test]
#[ignore = "measurement printer for coefficient/value structure; run with --ignored --nocapture"]
fn print_campaign_profile_v1_recursive_value_census() {
    use neo_ccs::CcsMatrix;
    use std::collections::HashSet;

    let audit = campaign_audit([0xDA; 32]);
    let arm = audit.arm(NebulaFPrimeBranch::Recursive);
    let mut values = HashSet::new();
    let mut nnz = 0usize;
    let mut max_row_terms = vec![0usize; arm.n];
    for matrix in [&arm.a, &arm.b, &arm.c] {
        match matrix {
            CcsMatrix::Csc(csc) | CcsMatrix::CscWithSeededPhi81 { csc, .. } => {
                nnz += csc.vals.len();
                for value in &csc.vals {
                    values.insert(p3_field::PrimeField64::as_canonical_u64(value));
                }
                for column in 0..csc.ncols {
                    for k in csc.column_range(column) {
                        max_row_terms[csc.row_index(k)] += 1;
                    }
                }
            }
            _ => {}
        }
    }
    eprintln!(
        "recursive value census: nnz={nnz} distinct_values={} max_row_terms={}",
        values.len(),
        max_row_terms.iter().max().copied().unwrap_or(0),
    );
}

#[test]
#[ignore = "measurement printer for row-pattern repetition; run with --ignored --nocapture"]
fn print_campaign_profile_v1_recursive_row_pattern_census() {
    use neo_ccs::CcsMatrix;
    use std::collections::HashMap;

    let audit = campaign_audit([0xDA; 32]);
    let arm = audit.arm(NebulaFPrimeBranch::Recursive);

    // Per-row signature across A, B, C: term columns are anchored to the
    // row's minimum column so translated repetitions collapse.
    let rows = arm.n;
    let mut row_terms: Vec<[Vec<(usize, u64)>; 3]> = vec![[Vec::new(), Vec::new(), Vec::new()]; rows];
    for (matrix_index, matrix) in [&arm.a, &arm.b, &arm.c].into_iter().enumerate() {
        match matrix {
            CcsMatrix::Csc(csc) | CcsMatrix::CscWithSeededPhi81 { csc, .. } => {
                for column in 0..csc.ncols {
                    for k in csc.column_range(column) {
                        let row = csc.row_index(k);
                        row_terms[row][matrix_index]
                            .push((column, p3_field::PrimeField64::as_canonical_u64(&csc.vals[k])));
                    }
                }
            }
            _ => {}
        }
    }
    // Seeded rows are compact already; exclude them from the pattern census.
    let mut seeded = vec![false; rows];
    if let CcsMatrix::CscWithSeededPhi81 { blocks, .. } = &arm.a {
        for block in blocks {
            for row in block.row_start()..block.row_end() {
                seeded[row] = true;
            }
        }
    }

    let mut patterns: HashMap<Vec<Vec<(usize, u64)>>, u32> = HashMap::new();
    let mut pattern_terms = Vec::<usize>::new();
    let mut row_pattern = vec![(0u32, 0usize); rows];
    for row in 0..rows {
        if seeded[row] {
            continue;
        }
        let mut sorted: [Vec<(usize, u64)>; 3] = row_terms[row].clone();
        for terms in &mut sorted {
            terms.sort_unstable();
        }
        let anchor = sorted
            .iter()
            .flat_map(|terms| terms.first().map(|term| term.0))
            .min()
            .unwrap_or(0);
        let signature = sorted
            .iter()
            .map(|terms| {
                terms
                    .iter()
                    .map(|(column, value)| (column - anchor, *value))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let terms_len = signature.iter().map(Vec::len).sum::<usize>();
        let next_id = patterns.len() as u32;
        let id = *patterns.entry(signature).or_insert_with(|| {
            pattern_terms.push(terms_len);
            next_id
        });
        row_pattern[row] = (id, anchor);
    }

    // Compress placements: maximal runs of consecutive rows with one pattern
    // and a constant anchor stride.
    let mut runs = 0usize;
    let mut row = 0usize;
    while row < rows {
        if seeded[row] {
            row += 1;
            continue;
        }
        let (id, anchor) = row_pattern[row];
        let mut end = row + 1;
        let mut stride: Option<isize> = None;
        let mut previous_anchor = anchor as isize;
        while end < rows && !seeded[end] && row_pattern[end].0 == id {
            let next_anchor = row_pattern[end].1 as isize;
            let step = next_anchor - previous_anchor;
            match stride {
                None => stride = Some(step),
                Some(existing) if existing == step => {}
                _ => break,
            }
            previous_anchor = next_anchor;
            end += 1;
        }
        runs += 1;
        row = end;
    }
    let census_rows = seeded.iter().filter(|flag| !**flag).count();
    let table_terms = pattern_terms.iter().sum::<usize>();
    eprintln!(
        "recursive pattern census: rows={census_rows} seeded_rows={} distinct_patterns={} pattern_table_terms={table_terms} placement_runs={runs}",
        rows - census_rows,
        patterns.len(),
    );
}

#[test]
#[ignore = "measurement printer for the source-matrix encodings; run with --ignored --nocapture"]
fn print_campaign_profile_v1_source_matrix_encodings() {
    use neo_ccs::CcsMatrix;
    let audit = campaign_audit([0xDA; 32]);
    for branch in [NebulaFPrimeBranch::Base, NebulaFPrimeBranch::Recursive] {
        let arm = audit.arm(branch);
        for (matrix_name, matrix) in [("a", &arm.a), ("b", &arm.b), ("c", &arm.c)] {
            match matrix {
                CcsMatrix::Identity { n } => {
                    eprintln!("branch={branch:?} matrix={matrix_name} variant=identity n={n}");
                }
                CcsMatrix::Csc(csc) => {
                    eprintln!(
                        "branch={branch:?} matrix={matrix_name} variant=csc rows={} cols={} nnz={}",
                        csc.nrows,
                        csc.ncols,
                        csc.vals.len(),
                    );
                }
                CcsMatrix::CscWithSeededPhi81 {
                    csc,
                    blocks,
                    geometric_runs,
                } => {
                    let run_terms = geometric_runs.iter().map(|run| run.len()).sum::<usize>();
                    eprintln!(
                        "branch={branch:?} matrix={matrix_name} variant=csc+seeded rows={} cols={} csc_nnz={} blocks={} geometric_runs={} run_terms={}",
                        csc.nrows,
                        csc.ncols,
                        csc.vals.len(),
                        blocks.len(),
                        geometric_runs.len(),
                        run_terms,
                    );
                }
                CcsMatrix::VerifierArtifact { rows, cols } => {
                    eprintln!(
                        "branch={branch:?} matrix={matrix_name} variant=verifier-artifact rows={rows} cols={cols}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "measurement printer for the seeded-block geometry; run with --ignored --nocapture"]
fn print_campaign_profile_v1_seeded_block_geometry() {
    let audit = campaign_audit([0xDA; 32]);
    for branch in [NebulaFPrimeBranch::Base, NebulaFPrimeBranch::Recursive] {
        let arm = audit.arm(branch);
        for (matrix_name, matrix) in [("a", &arm.a), ("b", &arm.b), ("c", &arm.c)] {
            for block in matrix.seeded_phi81_blocks() {
                let seeds = block
                    .chunk_seeds_by_row()
                    .iter()
                    .map(|row| row.len())
                    .sum::<usize>();
                eprintln!(
                    "branch={branch:?} matrix={matrix_name} row_start={} row_end={} rows={} word_width={} kappa={} message_cols={} chunk_size={} seed_rows={} total_seeds={} superneo_transformed={}",
                    block.row_start(),
                    block.row_end(),
                    block.row_end() - block.row_start(),
                    block.word_width(),
                    block.kappa(),
                    block.message_cols(),
                    block.chunk_size(),
                    block.chunk_seeds_by_row().len(),
                    seeds,
                    block.has_superneo_transformed_columns(),
                );
            }
        }
    }
}

#[test]
#[ignore = "measurement printer for the profile freeze; run with --ignored --nocapture"]
fn print_campaign_profile_v1_digests() {
    for (label, seed) in [
        ("campaign-v1-mirror-shape", [0xDA; 32]),
        ("campaign-v1-census-shape", [0xD9; 32]),
    ] {
        let start = Instant::now();
        let audit = campaign_audit(seed);
        eprintln!("candidate={label} audit build: {} ms", start.elapsed().as_millis());
        for branch in [NebulaFPrimeBranch::Base, NebulaFPrimeBranch::Recursive] {
            // A one-row export carries the selection-independent source
            // artifact digest and final plan digest without the complete
            // recursive projection.
            let start = Instant::now();
            let arm = audit.arm(branch);
            let export = export_nebula_problem(
                &audit,
                branch,
                ExportRequest {
                    profile: label.to_owned(),
                    scope: Scope::Branch,
                    public_input_count: arm.m_in,
                    source_rows: vec![0],
                    complete_families: Vec::new(),
                },
            )
            .expect("export one campaign-candidate source row");
            let problem = export.problem();
            let binding = export.binding();
            eprintln!(
                "candidate={label} branch={branch:?} n={} m={} m_in={} digest={} final_rows={} final_cols={} final_public={} final_plan_digest={} export_ms={}",
                problem.source.total_rows,
                problem.column_count,
                problem.public_input_count,
                problem.source.artifact_digest,
                binding.final_rows(),
                binding.final_columns(),
                binding.final_public_input_count(),
                binding.final_plan_digest(),
                start.elapsed().as_millis(),
            );
        }
    }

    let start = Instant::now();
    let relation = compile_terminal_fixture();
    let audit = relation.constraint_audit();
    eprintln!(
        "candidate=campaign-v1-terminal fixture build: {} ms",
        start.elapsed().as_millis()
    );
    let start = Instant::now();
    let export = export_complete_terminal_problem(&audit, "campaign-terminal-classification-v1")
        .expect("export the complete campaign terminal relation");
    let problem = export.problem();
    let binding = export.binding();
    eprintln!(
        "candidate=campaign-v1-terminal n={} m={} m_in={} families={} digest={} spartan_rows={} spartan_cols={} diagnostic_digest={} export_ms={}",
        problem.source.total_rows,
        problem.column_count,
        problem.public_input_count,
        problem.complete_families.len(),
        problem.source.artifact_digest,
        binding.spartan_rows(),
        binding.spartan_columns(),
        binding.diagnostic_digest(),
        start.elapsed().as_millis(),
    );
}

fn print_freeze_candidate(label: &str, params: Params) {
    let start = Instant::now();
    let audit = freeze_candidate_audit(&params);
    eprintln!("candidate={label} audit build: {} ms", start.elapsed().as_millis());

    for branch in [NebulaFPrimeBranch::Base, NebulaFPrimeBranch::Recursive] {
        let start = Instant::now();
        let export =
            export_complete_nebula_problem(&audit, branch, label).expect("export the complete freeze-candidate branch");
        let problem = export.problem();
        let binding = export.binding();
        eprintln!(
            "candidate={label} branch={branch:?} n={} m={} m_in={} families={} digest={} final_rows={} final_cols={} final_plan_digest={} export_ms={}",
            problem.source.total_rows,
            problem.column_count,
            problem.public_input_count,
            problem.complete_families.len(),
            problem.source.artifact_digest,
            binding.final_rows(),
            binding.final_columns(),
            binding.final_plan_digest(),
            start.elapsed().as_millis(),
        );
    }
}

#[test]
#[ignore = "memory probe: quarter-slice complete export; extrapolate x4 for the v2 census budget"]
fn probe_v2_quarter_export_peak_memory() {
    fn vm_hwm_gb() -> f64 {
        let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
        let line = status.lines().find(|line| line.starts_with("VmHWM:")).expect("VmHWM line");
        let kb: f64 = line.split_whitespace().nth(1).expect("VmHWM value").parse().expect("VmHWM number");
        kb / 1048576.0
    }
    let audit = campaign_audit(nightstream_constraint_exporter::CAMPAIGN_PLAN_SEED);
    let arm = audit.arm(NebulaFPrimeBranch::Recursive);
    let census = nightstream_constraint_exporter::nebula_family_census(&audit, NebulaFPrimeBranch::Recursive)
        .expect("complete reviewed ownership");
    let before = vm_hwm_gb();
    let start = Instant::now();
    let quarter = arm.n / 4;
    let problem = nightstream_constraint_exporter::export_sparse_problem(
        arm,
        nightstream_constraint_exporter::ExportRequest {
            profile: "campaign-recursive-classification-v1".to_owned(),
            scope: recursive_constraint_minimizer::Scope::Branch,
            public_input_count: arm.m_in,
            source_rows: (0..quarter).collect(),
            complete_families: census
                .iter()
                .filter(|family| family.source_rows().iter().all(|&row| row < quarter))
                .map(|family| family.name().to_owned())
                .collect(),
        },
    )
    .expect("export the quarter slice");
    println!(
        "quarter export: rows {} in {:?}; VmHWM before {:.1} GB, after {:.1} GB (delta {:.1} GB, x4 extrapolation {:.1} GB)",
        problem.rows.len(),
        start.elapsed(),
        before,
        vm_hwm_gb(),
        vm_hwm_gb() - before,
        4.0 * (vm_hwm_gb() - before)
    );
}
