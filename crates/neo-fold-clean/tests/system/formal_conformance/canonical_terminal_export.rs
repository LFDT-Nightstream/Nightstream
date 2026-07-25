//! Shared-input Rust/Lean differential corpus for the one-slot F' terminal
//! verifier.
//!
//! Rust executes the production `verify_uncompressed` entry point. Separate
//! calls expose the exact terminal link, running CE, and latest CCS checks as
//! receipts; the final Rust result is never fed back into those receipts.

use neo_ccs::traits::SModuleHomomorphism;
use neo_fold_clean::frontends::r1cs_f_prime::ivc::{R1csIvc, R1csIvcPreprocessing};
use neo_fold_clean::paper::construction2::{LatestInstance, ProofState, RunningInstance, SemanticStateMode, State};
use neo_fold_clean::paper::digest::{
    digest32_as_fields, initial_boundary_digest, public_trace_seed_digest, state_x_out_digest_with_mode,
    AccumulatorHandle, StateXOutDigestMode,
};
use neo_fold_clean::paper::f_prime::r1cs::encode_f_prime_superneo_public_input;
use neo_fold_clean::paper::relations::{CcsClaim, CcsWitness};
use neo_fold_clean::{Preprocessing, Uncompressed};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::Serialize;

use super::support::r1cs_compiler_fixtures::{
    assignment_one_product, make_tiny_lifecycle_plan, one_product_r1cs, tiny_params,
};

#[path = "canonical_terminal_export/lean.rs"]
mod lean;

const SCHEMA: u32 = 1;

pub fn checked_canonical_terminal_cases() -> (String, String) {
    let corpus = build_corpus();
    assert_eq!(corpus.schema, SCHEMA);
    assert_eq!(
        corpus
            .cases
            .iter()
            .map(|case| case.name.as_str())
            .collect::<Vec<_>>(),
        [
            "honest_base",
            "honest_recursive",
            "base_endpoint_mutation",
            "recursive_pc_mutation",
            "recursive_prior_link_mutation",
            "recursive_running_relation_mutation",
            "recursive_fresh_relation_mutation",
        ]
    );
    assert_eq!(
        corpus
            .cases
            .iter()
            .map(|case| case.mapped.rust_accepted)
            .collect::<Vec<_>>(),
        [true, true, false, false, false, false, false]
    );

    let recursive = &corpus.cases[1..];
    assert_eq!(
        recursive
            .iter()
            .map(|case| {
                (
                    case.observed.link_accepted,
                    case.observed.running_relation_accepted,
                    case.observed.fresh_relation_accepted,
                )
            })
            .collect::<Vec<_>>(),
        [
            (Some(true), Some(true), Some(true)),
            (None, None, None),
            (Some(false), Some(true), Some(true)),
            (Some(false), Some(true), Some(true)),
            (Some(true), Some(false), Some(true)),
            (Some(true), Some(true), Some(false)),
        ]
    );

    let first = serde_json::to_string(&corpus).expect("serialize canonical-terminal corpus");
    let second = serde_json::to_string(&corpus).expect("serialize canonical-terminal corpus twice");
    assert_eq!(first, second, "canonical-terminal corpus must be deterministic");
    let json = format!("{first}\n");
    let lean = lean::render(&corpus);
    (json, lean)
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct Corpus {
    schema: u32,
    evidence_tier: &'static str,
    scope: &'static str,
    primitive_boundary: &'static str,
    carrier_preconditions: Vec<&'static str>,
    excluded_claims: Vec<&'static str>,
    profile: Profile,
    atoms: Atoms,
    pub(super) cases: Vec<Case>,
}

#[derive(Clone, Debug, Serialize)]
struct Profile {
    name: &'static str,
    relation_rows: usize,
    relation_columns: usize,
    matrix_count: usize,
    public_input_len: usize,
    semantic_mode: &'static str,
    terminal_induction: bool,
    recursive_link: bool,
    fresh_count: usize,
    verifier_key_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct Felt(u64);

fn felt(value: F) -> Felt {
    Felt(value.as_canonical_u64())
}

fn felts(values: &[F]) -> Vec<Felt> {
    values.iter().copied().map(felt).collect()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct DigestAtom {
    bytes: [u8; 32],
    fields: [Felt; 4],
}

#[derive(Clone, Debug, Serialize)]
struct RunningAtom {
    claim_count: usize,
    parent_authority_present: bool,
    pending_projection_present: bool,
}

#[derive(Clone, Debug, Serialize)]
struct RunningWitnessAtom {
    witness_count: usize,
    shapes: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, Serialize)]
struct FreshAtom {
    public_input_len: usize,
    public_input: Vec<Felt>,
    commitment_shape: (usize, usize),
}

#[derive(Clone, Debug, Serialize)]
struct FreshWitnessAtom {
    private_values: usize,
    packed_shape: (usize, usize),
}

#[derive(Clone, Debug, Default, Serialize)]
struct Atoms {
    keys: Vec<[u8; 32]>,
    digests: Vec<DigestAtom>,
    states: Vec<[u8; 32]>,
    running: Vec<RunningAtom>,
    running_witnesses: Vec<RunningWitnessAtom>,
    fresh: Vec<FreshAtom>,
    fresh_witnesses: Vec<FreshWitnessAtom>,
    encoded: Vec<Vec<Felt>>,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct Case {
    pub(super) name: String,
    mutation: &'static str,
    rust_input: RustInput,
    observed: Observed,
    pub(super) mapped: TerminalCaseMap,
}

#[derive(Clone, Debug, Serialize)]
struct RustInput {
    branch: &'static str,
    iteration: u64,
    step_count: u64,
    z0: u32,
    zi: u32,
    pc: u64,
    running: u32,
    running_witness: u32,
    fresh: u32,
    fresh_witness: u32,
    final_fold_present: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Observed {
    link_accepted: Option<bool>,
    running_relation_accepted: Option<bool>,
    fresh_relation_accepted: Option<bool>,
    rust_error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct TerminalCaseMap {
    pub(super) verifier_key: u32,
    pub(super) default_running: u32,
    pub(super) iteration: u64,
    pub(super) z0: u32,
    pub(super) zi: u32,
    pub(super) running: u32,
    pub(super) running_witness: u32,
    pub(super) fresh: u32,
    pub(super) fresh_witness: u32,
    pub(super) pc: u64,
    pub(super) trace: TerminalTraceMap,
    pub(super) rust_accepted: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct HashInputMap {
    pub(super) verifier_key: u32,
    pub(super) iteration: u64,
    pub(super) z0: u32,
    pub(super) current: u32,
    pub(super) running: u32,
    pub(super) pc: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct HashReceiptMap {
    pub(super) input: HashInputMap,
    pub(super) output: u32,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct FreshPublicMap {
    pub(super) input: u32,
    pub(super) output: u32,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct EncodeMap {
    pub(super) input: u32,
    pub(super) output: u32,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct RunningRelationMap {
    pub(super) key: u32,
    pub(super) value: u32,
    pub(super) witness: u32,
    pub(super) accepted: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct FreshRelationMap {
    pub(super) key: u32,
    pub(super) value: u32,
    pub(super) witness: u32,
    pub(super) accepted: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "branch", rename_all = "snake_case")]
pub(super) enum TerminalTraceMap {
    Base,
    Recursive {
        prior_hash: HashReceiptMap,
        fresh_public: FreshPublicMap,
        encode: EncodeMap,
        running_relation: RunningRelationMap,
        fresh_relation: FreshRelationMap,
    },
}

struct Fixture {
    prep: R1csIvcPreprocessing,
    base: Uncompressed,
    first_latest: LatestInstance,
    recursive: Uncompressed,
}

fn base_proof(prep: &Preprocessing) -> Uncompressed {
    let empty = AccumulatorHandle::empty().digest();
    let structure = prep.structure_digest();
    Uncompressed {
        state: State::base(
            initial_boundary_digest(structure, prep.public_input_len),
            public_trace_seed_digest(structure),
            empty,
            prep.initial_semantic_state_digest(),
        ),
        final_fold: None,
    }
}

fn build_fixture() -> Fixture {
    let app = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(app.m(), app.m_in);
    let prep = R1csIvcPreprocessing::new_seeded(tiny_params(), &app, plan, 0x1F15_C009)
        .expect("compile bounded terminal differential relation");
    assert!(prep.prep.enforces_terminal_induction());
    assert!(prep.prep.enforces_f_prime_recursive_link());
    assert!(prep.prep.nebula().is_none());

    let base = base_proof(&prep.prep);
    neo_fold_clean::verify_uncompressed(&prep.prep, &base).expect("bounded base proof verifies");

    let mut chain = R1csIvc::new(&prep);
    chain
        .extend(assignment_one_product(3, 7))
        .expect("bounded base application");
    let first_latest = match &chain.audit().expect("first audit").proof.state.proof {
        ProofState::Active { latest, .. } => latest.clone(),
        ProofState::Initial => panic!("first append must activate the chain"),
    };
    chain
        .extend(assignment_one_product(4, 9))
        .expect("bounded recursive application");
    let recursive = chain.finish().expect("bounded terminal proof");
    neo_fold_clean::verify_uncompressed(&prep.prep, &recursive).expect("bounded recursive proof verifies");

    Fixture {
        prep,
        base,
        first_latest,
        recursive,
    }
}

#[derive(Default)]
struct Builder {
    atoms: Atoms,
    running_keys: Vec<String>,
    running_witness_keys: Vec<String>,
    fresh_keys: Vec<String>,
    fresh_witness_keys: Vec<String>,
    cases: Vec<Case>,
}

fn intern<T: PartialEq>(values: &mut Vec<T>, value: T) -> u32 {
    if let Some(index) = values.iter().position(|found| found == &value) {
        return u32::try_from(index + 1).expect("atom index fits u32");
    }
    values.push(value);
    u32::try_from(values.len()).expect("atom table fits u32")
}

fn intern_keyed<T>(keys: &mut Vec<String>, values: &mut Vec<T>, key: String, value: T) -> u32 {
    if let Some(index) = keys.iter().position(|found| found == &key) {
        return u32::try_from(index + 1).expect("atom index fits u32");
    }
    keys.push(key);
    values.push(value);
    u32::try_from(keys.len()).expect("atom table fits u32")
}

impl Builder {
    fn key(&mut self, value: [u8; 32]) -> u32 {
        intern(&mut self.atoms.keys, value)
    }

    fn digest(&mut self, value: [u8; 32]) -> u32 {
        intern(
            &mut self.atoms.digests,
            DigestAtom {
                bytes: value,
                fields: digest32_as_fields(value).map(felt),
            },
        )
    }

    fn state(&mut self, value: [u8; 32]) -> u32 {
        intern(&mut self.atoms.states, value)
    }

    fn running(&mut self, value: &RunningInstance) -> u32 {
        let visible = value.claims_only();
        intern_keyed(
            &mut self.running_keys,
            &mut self.atoms.running,
            format!("{visible:#?}"),
            RunningAtom {
                claim_count: value.claims.len(),
                parent_authority_present: value.parent_authority.is_some(),
                pending_projection_present: value.pending_projection().is_some(),
            },
        )
    }

    fn running_witness(&mut self, value: &RunningInstance) -> u32 {
        intern_keyed(
            &mut self.running_witness_keys,
            &mut self.atoms.running_witnesses,
            format!("{:#?}", value.witnesses),
            RunningWitnessAtom {
                witness_count: value.witnesses.len(),
                shapes: value
                    .witnesses
                    .iter()
                    .map(|witness| (witness.rows(), witness.cols()))
                    .collect(),
            },
        )
    }

    fn fresh(&mut self, value: &CcsClaim) -> u32 {
        intern_keyed(
            &mut self.fresh_keys,
            &mut self.atoms.fresh,
            format!("{value:#?}"),
            FreshAtom {
                public_input_len: value.m_in,
                public_input: felts(&value.x),
                commitment_shape: (value.c.d, value.c.kappa),
            },
        )
    }

    fn fresh_witness(&mut self, value: &CcsWitness) -> u32 {
        intern_keyed(
            &mut self.fresh_witness_keys,
            &mut self.atoms.fresh_witnesses,
            format!("{value:#?}"),
            FreshWitnessAtom {
                private_values: value.w.len(),
                packed_shape: (value.Z.rows(), value.Z.cols()),
            },
        )
    }

    fn encoded(&mut self, value: &[F]) -> u32 {
        intern(&mut self.atoms.encoded, felts(value))
    }

    fn absent_running(&mut self) -> (u32, u32) {
        let empty = RunningInstance::default();
        (self.running(&empty), self.running_witness(&empty))
    }

    fn absent_fresh(&mut self) -> (u32, u32) {
        let value = intern_keyed(
            &mut self.fresh_keys,
            &mut self.atoms.fresh,
            "<base: no fresh value>".to_owned(),
            FreshAtom {
                public_input_len: 0,
                public_input: Vec::new(),
                commitment_shape: (0, 0),
            },
        );
        let witness = intern_keyed(
            &mut self.fresh_witness_keys,
            &mut self.atoms.fresh_witnesses,
            "<base: no fresh witness>".to_owned(),
            FreshWitnessAtom {
                private_values: 0,
                packed_shape: (0, 0),
            },
        );
        (value, witness)
    }
}

fn terminal_digest(prep: &Preprocessing, state: &State) -> [u8; 32] {
    let mode = match prep.semantic_state_mode() {
        SemanticStateMode::Stateless => StateXOutDigestMode::Stateless,
        SemanticStateMode::Stateful => StateXOutDigestMode::Stateful,
    };
    state_x_out_digest_with_mode(
        mode,
        prep.vk.digest(),
        prep.pi_ccs_header_bundle(),
        prep.structure_digest(),
        state.chunk_count,
        state.step_count,
        state.z_0,
        state.z_i,
        state.pc,
        state.semantic_state_digest,
        state.acc_digest,
        state.public_trace,
        None,
    )
}

fn active_parts(proof: &Uncompressed) -> (RunningInstance, &LatestInstance) {
    let ProofState::Active { running, latest } = &proof.state.proof else {
        panic!("recursive terminal case must be active")
    };
    (
        running
            .materialize()
            .expect("bounded terminal running state is materialized"),
        latest,
    )
}

fn add_base_case(
    builder: &mut Builder,
    prep: &Preprocessing,
    name: &'static str,
    mutation: &'static str,
    proof: Uncompressed,
) {
    let verifier_key = builder.key(prep.vk.digest());
    let (running, running_witness) = builder.absent_running();
    let (fresh, fresh_witness) = builder.absent_fresh();
    let rust_result = neo_fold_clean::verify_uncompressed(prep, &proof);
    let mapped = TerminalCaseMap {
        verifier_key,
        default_running: running,
        iteration: proof.state.chunk_count,
        z0: builder.state(proof.state.z_0),
        zi: builder.state(proof.state.z_i),
        running,
        running_witness,
        fresh,
        fresh_witness,
        pc: proof.state.pc,
        trace: TerminalTraceMap::Base,
        rust_accepted: rust_result.is_ok(),
    };
    builder.cases.push(Case {
        name: name.to_owned(),
        mutation,
        rust_input: RustInput {
            branch: "base",
            iteration: proof.state.chunk_count,
            step_count: proof.state.step_count,
            z0: mapped.z0,
            zi: mapped.zi,
            pc: proof.state.pc,
            running,
            running_witness,
            fresh,
            fresh_witness,
            final_fold_present: proof.final_fold.is_some(),
        },
        observed: Observed {
            link_accepted: None,
            running_relation_accepted: None,
            fresh_relation_accepted: None,
            rust_error: rust_result.err().map(|error| format!("{error:?}")),
        },
        mapped,
    });
}

fn add_recursive_case(
    builder: &mut Builder,
    prep: &Preprocessing,
    name: &'static str,
    mutation: &'static str,
    proof: Uncompressed,
) {
    let verifier_key = builder.key(prep.vk.digest());
    let default_running = builder.running(&RunningInstance::default());
    let (running_value, latest) = active_parts(&proof);
    assert_eq!(latest.instances.len(), 1, "bounded terminal profile is one-slot");
    let fresh_instance = &latest.instances[0];
    let running = builder.running(&running_value);
    let running_witness = builder.running_witness(&running_value);
    let fresh = builder.fresh(&fresh_instance.claim);
    let fresh_witness = builder.fresh_witness(&fresh_instance.witness);

    let digest = terminal_digest(prep, &proof.state);
    let digest_id = builder.digest(digest);
    let actual_public = builder.encoded(&fresh_instance.claim.x);
    let expected_public_values = encode_f_prime_superneo_public_input(digest32_as_fields(digest));
    let expected_public = builder.encoded(&expected_public_values);

    let link_accepted = neo_fold_clean::lifecycle::validate_terminal_latest_link(prep, &proof.state, latest).is_ok();
    assert_eq!(
        link_accepted,
        actual_public == expected_public,
        "terminal public-link quotient must match the production check"
    );
    let running_relation_accepted =
        neo_fold_clean::lifecycle::validate_final_witness_authority(prep, &running_value).is_ok();
    let fresh_relation_accepted = neo_fold_clean::lifecycle::validate_latest_witness_authority(prep, latest).is_ok();
    let rust_result = neo_fold_clean::verify_uncompressed(prep, &proof);

    let mapped = TerminalCaseMap {
        verifier_key,
        default_running,
        iteration: proof.state.chunk_count,
        z0: builder.state(proof.state.z_0),
        zi: builder.state(proof.state.z_i),
        running,
        running_witness,
        fresh,
        fresh_witness,
        pc: proof.state.pc,
        trace: TerminalTraceMap::Recursive {
            prior_hash: HashReceiptMap {
                input: HashInputMap {
                    verifier_key,
                    iteration: proof.state.chunk_count,
                    z0: builder.state(proof.state.z_0),
                    current: builder.state(proof.state.z_i),
                    running,
                    pc: proof.state.pc,
                },
                output: digest_id,
            },
            fresh_public: FreshPublicMap {
                input: fresh,
                output: actual_public,
            },
            encode: EncodeMap {
                input: digest_id,
                output: expected_public,
            },
            running_relation: RunningRelationMap {
                key: verifier_key,
                value: running,
                witness: running_witness,
                accepted: running_relation_accepted,
            },
            fresh_relation: FreshRelationMap {
                key: verifier_key,
                value: fresh,
                witness: fresh_witness,
                accepted: fresh_relation_accepted,
            },
        },
        rust_accepted: rust_result.is_ok(),
    };
    builder.cases.push(Case {
        name: name.to_owned(),
        mutation,
        rust_input: RustInput {
            branch: "recursive",
            iteration: proof.state.chunk_count,
            step_count: proof.state.step_count,
            z0: mapped.z0,
            zi: mapped.zi,
            pc: proof.state.pc,
            running,
            running_witness,
            fresh,
            fresh_witness,
            final_fold_present: proof.final_fold.is_some(),
        },
        observed: Observed {
            link_accepted: Some(link_accepted),
            running_relation_accepted: Some(running_relation_accepted),
            fresh_relation_accepted: Some(fresh_relation_accepted),
            rust_error: rust_result.err().map(|error| format!("{error:?}")),
        },
        mapped,
    });
}

fn build_corpus() -> Corpus {
    let fixture = build_fixture();
    let prep = &fixture.prep.prep;
    let mut builder = Builder::default();

    add_base_case(&mut builder, prep, "honest_base", "none", fixture.base.clone());
    add_recursive_case(
        &mut builder,
        prep,
        "honest_recursive",
        "none",
        fixture.recursive.clone(),
    );

    let mut bad_endpoint = fixture.base.clone();
    bad_endpoint.state.z_i[0] ^= 1;
    add_base_case(
        &mut builder,
        prep,
        "base_endpoint_mutation",
        "state.z_i[0] ^= 1",
        bad_endpoint,
    );

    let mut bad_pc = fixture.recursive.clone();
    bad_pc.state.pc = 2;
    add_recursive_case(&mut builder, prep, "recursive_pc_mutation", "state.pc := 2", bad_pc);

    let mut bad_link = fixture.recursive.clone();
    let ProofState::Active { latest, .. } = &mut bad_link.state.proof else {
        unreachable!("recursive fixture is active")
    };
    *latest = fixture.first_latest.clone();
    add_recursive_case(
        &mut builder,
        prep,
        "recursive_prior_link_mutation",
        "state.latest := prior honest latest",
        bad_link,
    );

    let mut bad_running = fixture.recursive.clone();
    let ProofState::Active { running, .. } = &mut bad_running.state.proof else {
        unreachable!("recursive fixture is active")
    };
    let running = running
        .as_materialized_mut()
        .expect("bounded terminal running state is materialized");
    running.witnesses[0].as_mut_slice()[0] += F::ONE;
    add_recursive_case(
        &mut builder,
        prep,
        "recursive_running_relation_mutation",
        "state.running.witnesses[0][0] += 1",
        bad_running,
    );

    let mut bad_fresh = fixture.recursive.clone();
    let ProofState::Active { latest, .. } = &mut bad_fresh.state.proof else {
        unreachable!("recursive fixture is active")
    };
    let instance = &mut latest.instances[0];
    let global_column = instance.claim.m_in;
    let packed_coordinate = (global_column % neo_math::D, global_column / neo_math::D);
    instance.witness.Z[packed_coordinate] = if instance.witness.Z[packed_coordinate] == F::ZERO {
        F::ONE
    } else {
        F::ZERO
    };
    instance.claim.c = prep.log.commit(&instance.witness.Z);
    add_recursive_case(
        &mut builder,
        prep,
        "recursive_fresh_relation_mutation",
        "latest.private[m_in] toggled and consistently recommitted",
        bad_fresh,
    );

    let profile = Profile {
        name: "r1cs_ivc_tiny_one_slot_terminal_v1",
        relation_rows: prep.structure().n,
        relation_columns: prep.structure().m,
        matrix_count: prep.structure().t(),
        public_input_len: prep.public_input_len.expect("fixed terminal public input"),
        semantic_mode: match prep.semantic_state_mode() {
            SemanticStateMode::Stateless => "stateless",
            SemanticStateMode::Stateful => "stateful",
        },
        terminal_induction: prep.enforces_terminal_induction(),
        recursive_link: prep.enforces_f_prime_recursive_link(),
        fresh_count: 1,
        verifier_key_digest: prep.vk.digest(),
    };
    Corpus {
        schema: SCHEMA,
        evidence_tier: "bounded Rust-conformant differential",
        scope: "one slot, base plus two-step plain HyperNova terminal profile",
        primitive_boundary:
            "link and relation outcomes are exact Rust checks exposed as receipts; the final Rust result is isolated",
        carrier_preconditions: vec![
            "verifier-owned terminal-induction preprocessing",
            "plain non-Nebula carrier with no final fold",
            "one latest F' instance on the recursive branch",
            "all non-terminal lifecycle anchors fixed by the bounded fixture except the named mutation",
        ],
        excluded_claims: vec![
            "general refinement over malformed raw Uncompressed carriers",
            "Poseidon2 or Ajtai internal correctness",
            "general running CE or fresh CCS relation refinement",
            "R1CS soundness or honest assignment completeness",
            "typed IR or physical row/column parity",
        ],
        profile,
        atoms: builder.atoms,
        cases: builder.cases,
    }
}
