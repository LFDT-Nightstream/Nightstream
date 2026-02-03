#![allow(dead_code)]

pub mod goldilocks_canonicality_relations;

use blake2b_simd::State as TranscriptHash;
use midnight_curves::Bls12;
use midnight_proofs::dev::cost_model::circuit_model;
use midnight_proofs::poly::kzg::params::ParamsKZG;
use midnight_proofs::utils::SerdeFormat;
use midnight_zk_stdlib::Relation;
use neo_math::{KExtensions, D, F, K};
use neo_midnight_bridge::k_field::KRepr;
use neo_midnight_bridge::relations::{
    PiCcsFeChunkAggSumcheckInstance, PiCcsFeChunkAggSumcheckRelation, PiCcsFeChunkAggSumcheckWitness,
    PiCcsFeChunkInstance, PiCcsFeChunkRelation, PiCcsFeChunkWitness, PiCcsNcChunkAggSumcheckInstance,
    PiCcsNcChunkAggSumcheckRelation, PiCcsNcChunkAggSumcheckWitness, PiCcsNcChunkInstance, PiCcsNcChunkRelation,
    PiCcsNcChunkWitness, SparsePolyRepr, SparsePolyTermRepr,
};
use neo_midnight_bridge::statement::{blake2b_256_domain, compute_step_bundle_digest_v2, digest32_to_u128_limbs_le};
use neo_reductions::engines::utils as reductions_utils;
use p3_field::PrimeCharacteristicRing;
use p3_field::PrimeField64;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

const PARAMS_DIGEST_DOMAIN: &[u8] = b"neo/midnight-bridge/params-digest/v1";
const ACC_DIGEST_DOMAIN: &[u8] = b"neo/midnight-bridge/acc-digest/v1";

fn kzg_params_testdata_path(k: u32) -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .join("testdata")
        .join("kzg_params")
        .join(format!("bls_midnight_2p{k}"))
}

fn kzg_params_unsafe_setup_deterministic(k: u32) -> ParamsKZG<Bls12> {
    let mut seed = [0u8; 32];
    seed[0] = 0xA5;
    seed[1] = (k & 0xFF) as u8;
    seed[2] = ((k >> 8) & 0xFF) as u8;
    seed[3] = ((k >> 16) & 0xFF) as u8;
    seed[4] = ((k >> 24) & 0xFF) as u8;
    ParamsKZG::unsafe_setup(k, ChaCha20Rng::from_seed(seed))
}

fn try_read_kzg_params_from_testdata(k: u32) -> Option<ParamsKZG<Bls12>> {
    let path = kzg_params_testdata_path(k);
    let mut f = fs::File::open(&path).ok()?;
    Some(
        ParamsKZG::<Bls12>::read_custom(&mut f, SerdeFormat::RawBytes)
            .unwrap_or_else(|e| panic!("read ParamsKZG from {path:?}: {e}")),
    )
}

pub fn test_kzg_params(k: u32) -> ParamsKZG<Bls12> {
    try_read_kzg_params_from_testdata(k).unwrap_or_else(|| kzg_params_unsafe_setup_deterministic(k))
}

fn params_digest32(params: &neo_params::NeoParams) -> [u8; 32] {
    let bytes = bincode::serialize(params).expect("NeoParams should be serializable");
    blake2b_256_domain(PARAMS_DIGEST_DOMAIN, &bytes)
}

fn ccs_digest32(s: &neo_ccs::CcsStructure<F>) -> [u8; 32] {
    let dig = reductions_utils::digest_ccs_matrices(s);
    assert_eq!(dig.len(), 4, "CCS digest must have len 4");
    let mut out = [0u8; 32];
    for (i, d) in dig.iter().enumerate() {
        out[i * 8..(i + 1) * 8].copy_from_slice(&d.as_canonical_u64().to_le_bytes());
    }
    out
}

fn acc_digest32_from_fold_digests(acc: &[neo_ccs::MeInstance<neo_ajtai::Commitment, F, K>]) -> [u8; 32] {
    let mut msg = Vec::new();
    msg.extend_from_slice(&(acc.len() as u32).to_le_bytes());
    for me in acc {
        msg.extend_from_slice(&me.fold_digest);
    }
    blake2b_256_domain(ACC_DIGEST_DOMAIN, &msg)
}

/// Cache KZG params by `k` so we don't pay `unsafe_setup()` repeatedly per proof.
///
/// `ParamsKZG::unsafe_setup` is expensive (generates the SRS). In these benchmark-style
/// bundle tests we generate many small proofs, so caching params drastically reduces
/// runtime while keeping runs deterministic.
#[derive(Default)]
struct KzgParamsCache {
    by_k: BTreeMap<u32, ParamsKZG<Bls12>>,
}

impl KzgParamsCache {
    fn get(&mut self, k: u32) -> &ParamsKZG<Bls12> {
        self.by_k.entry(k).or_insert_with(|| test_kzg_params(k))
    }
}

fn choose_max_count(
    max_count: usize,
    label: &str,
    mut model_for_count: impl FnMut(usize) -> (u32, usize),
) -> usize {
    let (k, rows) = model_for_count(max_count);
    println!("{label}: count={max_count} min_k={k} rows={rows}");
    max_count
}

fn k_to_repr(k: &neo_math::K) -> KRepr {
    let (c0, c1) = k.to_limbs_u64();
    KRepr { c0, c1 }
}

fn host_mle_eval(values: &[K], alpha: &[K]) -> K {
    assert_eq!(values.len(), 1usize << alpha.len());
    let mut cur = values.to_vec();
    for a in alpha {
        let next_len = cur.len() / 2;
        let mut next = Vec::with_capacity(next_len);
        for j in 0..next_len {
            let v0 = cur[2 * j];
            let v1 = cur[2 * j + 1];
            next.push(v0 + (*a) * (v1 - v0));
        }
        cur = next;
    }
    assert_eq!(cur.len(), 1);
    cur[0]
}

fn host_range_product(val: K, b: u32) -> K {
    let lo = -((b as i64) - 1);
    let hi = (b as i64) - 1;
    let mut prod = K::ONE;
    for t in lo..=hi {
        prod *= val - K::from(F::from_i64(t));
    }
    prod
}

fn host_eq_points(p: &[K], q: &[K]) -> K {
    assert_eq!(p.len(), q.len());
    let mut acc = K::ONE;
    for (pi, qi) in p.iter().copied().zip(q.iter().copied()) {
        acc *= (K::ONE - pi) * (K::ONE - qi) + pi * qi;
    }
    acc
}

fn host_chunk_sum(y_zcols: &[Vec<K>], alpha: &[K], gamma: K, b: u32, start_exp: usize) -> K {
    let mut g = K::ONE;
    for _ in 0..start_exp {
        g *= gamma;
    }
    let mut acc = K::ZERO;
    for yz in y_zcols {
        let val = host_mle_eval(yz, alpha);
        let rp = host_range_product(val, b);
        acc += g * rp;
        g *= gamma;
    }
    acc
}

fn host_fe_chunk_sum(
    y_rows_flat: &[Vec<K>],
    alpha_prime: &[K],
    gamma: K,
    k_total: usize,
    t: usize,
    start_idx: usize,
    count: usize,
) -> K {
    assert_eq!(y_rows_flat.len(), (k_total - 1) * t);
    assert!(start_idx + count <= y_rows_flat.len());

    // γ^i for i=0..k_total, and γ^k_total.
    let mut gamma_pows: Vec<K> = Vec::with_capacity(k_total + 1);
    gamma_pows.push(K::ONE);
    for i in 0..k_total {
        gamma_pows.push(gamma_pows[i] * gamma);
    }
    let gamma_to_k_total = gamma_pows[k_total];

    // (γ^k_total)^j for j=0..t-1.
    let mut gamma_k_pows: Vec<K> = Vec::with_capacity(t);
    gamma_k_pows.push(K::ONE);
    for j in 1..t {
        gamma_k_pows.push(gamma_k_pows[j - 1] * gamma_to_k_total);
    }

    let mut acc = K::ZERO;
    for flat in start_idx..start_idx + count {
        let out_idx = flat / t;
        let j = flat % t;
        let i_abs = out_idx + 1;
        let weight = gamma_pows[i_abs] * gamma_k_pows[j];
        let y_eval = host_mle_eval(&y_rows_flat[flat], alpha_prime);
        acc += weight * y_eval;
    }
    acc
}

pub fn prove_step1_nc_bundle_poseidon2_batch_40() {
    // Demonstrate that the NC terminal identity for step 1 (k_total=13) can be verified
    // using a bundle of small PLONK/KZG proofs, each fitting within Midnight's max-k cap
    // (k=14).
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join("../neo-fold/poseidon2-tests/poseidon2_ic_circuit_batch_40.json");
    let json = fs::read_to_string(&json_path).expect("read poseidon2 batch-40 json");
    let export = neo_fold::test_export::parse_test_export_json(&json).expect("parse test-export json");

    let target_folding_steps: usize = 2;
    let mut session = neo_fold::test_export::TestExportSession::new_from_circuit_json(&json).expect("session init");
    for i in 0..target_folding_steps {
        let z = &export.witness[i % export.witness.len()];
        session.add_step_witness_u64(z).expect("add witness step");
    }

    let (fold_run, _step_ms) = session
        .fold_and_prove_with_step_timings()
        .expect("fold_and_prove");
    assert_eq!(fold_run.steps.len(), target_folding_steps);
    assert!(session.verify(&fold_run).expect("verify"));

    let s = session.ccs();
    let m_pad = s.m.next_power_of_two().max(2);
    let ell_m = m_pad.trailing_zeros() as usize;
    let d_pad = D.next_power_of_two();
    let ell_d = d_pad.trailing_zeros() as usize;
    let params_b = session.params().b;

    let step1 = &fold_run.steps[1];
    let pi = &step1.fold.ccs_proof;
    let k_total = step1.fold.ccs_out.len();
    println!("Step 1 k_total={k_total} (expect 13 with k_rho=12)");

    let step_idx: u32 = 1;
    let params_digest32 = params_digest32(session.params());
    let ccs_digest32 = ccs_digest32(s);
    let initial_acc_digest32 = acc_digest32_from_fold_digests(&fold_run.steps[0].fold.dec_children);
    let final_acc_digest32 = acc_digest32_from_fold_digests(&fold_run.steps[1].fold.dec_children);
    let bundle_digest32 = compute_step_bundle_digest_v2(
        step_idx,
        params_digest32,
        ccs_digest32,
        initial_acc_digest32,
        final_acc_digest32,
    );
    let bundle_digest = digest32_to_u128_limbs_le(bundle_digest32);

    let want_nc_chals = ell_m + ell_d;
    assert_eq!(pi.sumcheck_challenges_nc.len(), want_nc_chals);
    let (s_col_prime, alpha_prime_nc) = pi.sumcheck_challenges_nc.split_at(ell_m);
    let gamma = pi.challenges_public.gamma;

    // Choose the largest chunk size (one proof per step if possible).
    let chunk_size = choose_max_count(k_total, "PiCcsNcChunkRelation", |count| {
        let rel_try = PiCcsNcChunkRelation {
            ell_d,
            b: params_b,
            start_exp: k_total, // worst-case exponent
            count,
        };
        let circuit_try = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel_try);
        let model_try = circuit_model::<_, 48, 32>(&circuit_try);
        (model_try.k, model_try.rows)
    });

    let y_zcols_all: Vec<Vec<K>> = step1
        .fold
        .ccs_out
        .iter()
        .map(|out| out.y_zcol.clone())
        .collect();
    assert_eq!(y_zcols_all.len(), k_total);
    for (i, yz) in y_zcols_all.iter().enumerate() {
        assert_eq!(yz.len(), 1usize << ell_d, "y_zcol[{i}] must be padded");
    }

    // Compute all chunk sums host-side first.
    let alpha_repr: Vec<KRepr> = alpha_prime_nc.iter().map(k_to_repr).collect();
    let gamma_repr = k_to_repr(&gamma);

    let mut chunk_instances: Vec<KRepr> = Vec::new();
    for start_i in (0..k_total).step_by(chunk_size) {
        let count = core::cmp::min(chunk_size, k_total - start_i);
        let start_exp = start_i + 1;
        let yz_slice = &y_zcols_all[start_i..start_i + count];
        let chunk_sum = host_chunk_sum(yz_slice, alpha_prime_nc, gamma, params_b, start_exp);
        chunk_instances.push(k_to_repr(&chunk_sum));
    }

    // Prove each chunk. Fold both the aggregate check and the NC sumcheck into the *last* chunk
    // proof to save bundle bytes.
    let agg_chunk_index = chunk_instances.len().saturating_sub(1);
    let beta_a_repr: Vec<KRepr> = pi.challenges_public.beta_a.iter().map(k_to_repr).collect();
    let beta_m_repr: Vec<KRepr> = pi.challenges_public.beta_m.iter().map(k_to_repr).collect();

    let n_rounds_nc = pi.sumcheck_rounds_nc.len();
    let poly_len_nc = pi.sumcheck_rounds_nc[0].len();
    let initial_sum_nc = pi
        .sc_initial_sum_nc
        .as_ref()
        .map(k_to_repr)
        .unwrap_or(KRepr::ZERO);

    let inst_agg = PiCcsNcChunkAggSumcheckInstance {
        bundle_digest,
        sumcheck_challenges: pi.sumcheck_challenges_nc.iter().map(k_to_repr).collect(),
        gamma: gamma_repr,
        beta_a: beta_a_repr.clone(),
        beta_m: beta_m_repr.clone(),
        chunk_sums: chunk_instances.clone(),
        initial_sum: initial_sum_nc,
        final_sum_nc: k_to_repr(&pi.sumcheck_final_nc),
    };

    let mut total_proof_bytes: usize = 0;
    let mut total_statement_bytes: usize = 0;
    let mut params_cache = KzgParamsCache::default();
    for (chunk_idx, start_i) in (0..k_total).step_by(chunk_size).enumerate() {
        let count = core::cmp::min(chunk_size, k_total - start_i);
        let start_exp = start_i + 1;
        let yz_slice = &y_zcols_all[start_i..start_i + count];

        if chunk_idx == agg_chunk_index {
            let rel = PiCcsNcChunkAggSumcheckRelation {
                n_rounds: n_rounds_nc,
                poly_len: poly_len_nc,
                ell_d,
                ell_m,
                b: params_b,
                start_exp,
                count,
                n_chunks: chunk_instances.len(),
                chunk_index: chunk_idx,
            };
            let witness = PiCcsNcChunkAggSumcheckWitness {
                rounds: pi
                    .sumcheck_rounds_nc
                    .iter()
                    .map(|r| r.iter().map(k_to_repr).collect())
                    .collect(),
                y_zcol: yz_slice
                    .iter()
                    .map(|yz| yz.iter().map(k_to_repr).collect::<Vec<_>>())
                    .collect(),
            };

            let circuit = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel);
            let model = circuit_model::<_, 48, 32>(&circuit);
            println!(
                "NC Chunk+Agg+Sumcheck {chunk_idx}: outputs=[{start_i}..{}), count={count}, start_exp={start_exp}, min_k={} rows={}",
                start_i + count,
                model.k,
                model.rows
            );

            total_statement_bytes += <PiCcsNcChunkAggSumcheckRelation as Relation>::format_instance(&inst_agg)
                .expect("format_instance")
                .len()
                * 32;

            let params = params_cache.get(model.k);
            let vk = midnight_zk_stdlib::setup_vk(params, &rel);
            let pk = midnight_zk_stdlib::setup_pk(&rel, &vk);
            let proof = midnight_zk_stdlib::prove::<_, TranscriptHash>(
                params,
                &pk,
                &rel,
                &inst_agg,
                witness,
                ChaCha20Rng::from_seed([71u8.wrapping_add(chunk_idx as u8); 32]),
            )
            .expect("prove chunk+agg");
            total_proof_bytes += proof.len();

            let params_v = params.verifier_params();
            midnight_zk_stdlib::verify::<PiCcsNcChunkAggSumcheckRelation, TranscriptHash>(
                &params_v, &vk, &inst_agg, None, &proof,
            )
            .expect("verify chunk+agg+sumcheck");
        } else {
            let rel = PiCcsNcChunkRelation {
                ell_d,
                b: params_b,
                start_exp,
                count,
            };
            let instance = PiCcsNcChunkInstance {
                bundle_digest,
                chunk_sum: chunk_instances[chunk_idx],
                alpha_prime: alpha_repr.clone(),
                gamma: gamma_repr,
            };
            let witness = PiCcsNcChunkWitness {
                y_zcol: yz_slice
                    .iter()
                    .map(|yz| yz.iter().map(k_to_repr).collect::<Vec<_>>())
                    .collect(),
            };

            let circuit = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel);
            let model = circuit_model::<_, 48, 32>(&circuit);
            println!(
                "NC Chunk {chunk_idx}: outputs=[{start_i}..{}), count={count}, start_exp={start_exp}, min_k={} rows={}",
                start_i + count,
                model.k,
                model.rows
            );

            let params = params_cache.get(model.k);
            let vk = midnight_zk_stdlib::setup_vk(params, &rel);
            let pk = midnight_zk_stdlib::setup_pk(&rel, &vk);
            let proof = midnight_zk_stdlib::prove::<_, TranscriptHash>(
                params,
                &pk,
                &rel,
                &instance,
                witness,
                ChaCha20Rng::from_seed([71u8.wrapping_add(chunk_idx as u8); 32]),
            )
            .expect("prove chunk");
            total_proof_bytes += proof.len();
            total_statement_bytes += <PiCcsNcChunkRelation as Relation>::format_instance(&instance)
                .expect("format_instance")
                .len()
                * 32;

            let params_v = params.verifier_params();
            midnight_zk_stdlib::verify::<PiCcsNcChunkRelation, TranscriptHash>(&params_v, &vk, &instance, None, &proof)
                .expect("verify chunk");
        }
    }

    // Aggregate: eq((α',s'),(β_a,β_m)) * Σ chunk_sum == final_sum_nc
    let alpha_k = alpha_prime_nc.to_vec();
    let beta_a_k = pi.challenges_public.beta_a.clone();
    let beta_m_k = pi.challenges_public.beta_m.clone();
    let s_col_k = s_col_prime.to_vec();
    let eq_apsp_beta = host_eq_points(&alpha_k, &beta_a_k) * host_eq_points(&s_col_k, &beta_m_k);

    let total_chunk_sum: K = chunk_instances.iter().fold(K::ZERO, |acc, cs| {
        acc + K::from_coeffs([F::from_u64(cs.c0), F::from_u64(cs.c1)])
    });
    let rhs = eq_apsp_beta * total_chunk_sum;
    assert_eq!(
        rhs, pi.sumcheck_final_nc,
        "host aggregate mismatch: expected rhs == sumcheck_final_nc"
    );

    println!("Total bundle proof bytes (chunks; last includes aggregate + sumcheck): {total_proof_bytes}");
    println!("Total bundle statement bytes (estimated): {total_statement_bytes}");
    println!(
        "Total bundle payload bytes (proof + statement): {}",
        total_proof_bytes + total_statement_bytes
    );
    assert!(total_proof_bytes < 50 * 1024, "expected total < 50KB bundle");
}

pub fn prove_step1_full_bundle_poseidon2_batch_40() {
    // Demonstrate (and currently quantify) the full Step-1 Pi-CCS verification payload:
    // - FE sumcheck + FE terminal identity (chunked)
    // - NC sumcheck + NC terminal identity (chunked, k_total=13)
    //
    // Goal is a *bundle* total < 50KB under Midnight's max-k cap (k=14).
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join("../neo-fold/poseidon2-tests/poseidon2_ic_circuit_batch_40.json");
    let json = fs::read_to_string(&json_path).expect("read poseidon2 batch-40 json");
    let export = neo_fold::test_export::parse_test_export_json(&json).expect("parse test-export json");

    let target_folding_steps: usize = 2;
    let mut session = neo_fold::test_export::TestExportSession::new_from_circuit_json(&json).expect("session init");
    for i in 0..target_folding_steps {
        let z = &export.witness[i % export.witness.len()];
        session.add_step_witness_u64(z).expect("add witness step");
    }

    let (fold_run, _step_ms) = session
        .fold_and_prove_with_step_timings()
        .expect("fold_and_prove");
    assert_eq!(fold_run.steps.len(), target_folding_steps);
    assert!(session.verify(&fold_run).expect("verify"));

    let s = session.ccs();
    let n_pad = s.n.next_power_of_two().max(2);
    let ell_n = n_pad.trailing_zeros() as usize;
    let m_pad = s.m.next_power_of_two().max(2);
    let ell_m = m_pad.trailing_zeros() as usize;
    let d_pad = D.next_power_of_two();
    let ell_d = d_pad.trailing_zeros() as usize;
    let params_b = session.params().b;

    let step1 = &fold_run.steps[1];
    let pi = &step1.fold.ccs_proof;
    let k_total = step1.fold.ccs_out.len();
    println!("Step 1 k_total={k_total} (expect 13 with k_rho=12)");

    let step_idx: u32 = 1;
    let params_digest32 = params_digest32(session.params());
    let ccs_digest32 = ccs_digest32(s);
    let initial_acc_digest32 = acc_digest32_from_fold_digests(&fold_run.steps[0].fold.dec_children);
    let final_acc_digest32 = acc_digest32_from_fold_digests(&fold_run.steps[1].fold.dec_children);
    let bundle_digest32 = compute_step_bundle_digest_v2(
        step_idx,
        params_digest32,
        ccs_digest32,
        initial_acc_digest32,
        final_acc_digest32,
    );
    let bundle_digest = digest32_to_u128_limbs_le(bundle_digest32);
    let mut params_cache = KzgParamsCache::default();

    // -----------------------------
    // NC bundle (chunked terminal + nc sumcheck)
    // -----------------------------
    let want_nc_chals = ell_m + ell_d;
    assert_eq!(pi.sumcheck_challenges_nc.len(), want_nc_chals);
    let (_s_col_prime, alpha_prime_nc) = pi.sumcheck_challenges_nc.split_at(ell_m);
    let gamma = pi.challenges_public.gamma;

    // Choose the largest chunk size (one proof per step if possible).
    let chunk_size = choose_max_count(k_total, "PiCcsNcChunkRelation", |count| {
        let rel_try = PiCcsNcChunkRelation {
            ell_d,
            b: params_b,
            start_exp: k_total, // worst-case exponent
            count,
        };
        let circuit_try = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel_try);
        let model_try = circuit_model::<_, 48, 32>(&circuit_try);
        (model_try.k, model_try.rows)
    });

    let y_zcols_all: Vec<Vec<K>> = step1
        .fold
        .ccs_out
        .iter()
        .map(|out| out.y_zcol.clone())
        .collect();
    assert_eq!(y_zcols_all.len(), k_total);
    for (i, yz) in y_zcols_all.iter().enumerate() {
        assert_eq!(yz.len(), 1usize << ell_d, "y_zcol[{i}] must be padded");
    }

    // Compute all chunk sums host-side first.
    let alpha_repr: Vec<KRepr> = alpha_prime_nc.iter().map(k_to_repr).collect();
    let gamma_repr = k_to_repr(&gamma);

    let mut chunk_instances: Vec<KRepr> = Vec::new();
    for start_i in (0..k_total).step_by(chunk_size) {
        let count = core::cmp::min(chunk_size, k_total - start_i);
        let start_exp = start_i + 1;
        let yz_slice = &y_zcols_all[start_i..start_i + count];
        let chunk_sum = host_chunk_sum(yz_slice, alpha_prime_nc, gamma, params_b, start_exp);
        chunk_instances.push(k_to_repr(&chunk_sum));
    }

    // Prove all chunks. Fold both the aggregate check and the NC sumcheck into the *last* chunk
    // proof to save bundle bytes.
    let agg_chunk_index = chunk_instances.len().saturating_sub(1);
    let beta_a_repr: Vec<KRepr> = pi.challenges_public.beta_a.iter().map(k_to_repr).collect();
    let beta_m_repr: Vec<KRepr> = pi.challenges_public.beta_m.iter().map(k_to_repr).collect();

    let n_rounds_nc = pi.sumcheck_rounds_nc.len();
    let poly_len_nc = pi.sumcheck_rounds_nc[0].len();
    let initial_sum_nc = pi
        .sc_initial_sum_nc
        .as_ref()
        .map(k_to_repr)
        .unwrap_or(KRepr::ZERO);
    let inst_nc_agg = PiCcsNcChunkAggSumcheckInstance {
        bundle_digest,
        sumcheck_challenges: pi.sumcheck_challenges_nc.iter().map(k_to_repr).collect(),
        gamma: gamma_repr,
        beta_a: beta_a_repr.clone(),
        beta_m: beta_m_repr.clone(),
        chunk_sums: chunk_instances.clone(),
        initial_sum: initial_sum_nc,
        final_sum_nc: k_to_repr(&pi.sumcheck_final_nc),
    };

    let mut nc_bundle_bytes: usize = 0;
    let mut nc_statement_bytes: usize = 0;
    for (chunk_idx, start_i) in (0..k_total).step_by(chunk_size).enumerate() {
        let count = core::cmp::min(chunk_size, k_total - start_i);
        let start_exp = start_i + 1;
        let yz_slice = &y_zcols_all[start_i..start_i + count];

        if chunk_idx == agg_chunk_index {
            let rel = PiCcsNcChunkAggSumcheckRelation {
                n_rounds: n_rounds_nc,
                poly_len: poly_len_nc,
                ell_d,
                ell_m,
                b: params_b,
                start_exp,
                count,
                n_chunks: chunk_instances.len(),
                chunk_index: chunk_idx,
            };
            let witness = PiCcsNcChunkAggSumcheckWitness {
                rounds: pi
                    .sumcheck_rounds_nc
                    .iter()
                    .map(|r| r.iter().map(k_to_repr).collect())
                    .collect(),
                y_zcol: yz_slice
                    .iter()
                    .map(|yz| yz.iter().map(k_to_repr).collect::<Vec<_>>())
                    .collect(),
            };

            let circuit = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel);
            let model = circuit_model::<_, 48, 32>(&circuit);
            println!(
                "NC Chunk+Agg+Sumcheck {chunk_idx}: outputs=[{start_i}..{}), count={count}, start_exp={start_exp}, min_k={} rows={}",
                start_i + count,
                model.k,
                model.rows
            );

            nc_statement_bytes += <PiCcsNcChunkAggSumcheckRelation as Relation>::format_instance(&inst_nc_agg)
                .expect("format_instance")
                .len()
                * 32;

            let params = params_cache.get(model.k);
            let vk = midnight_zk_stdlib::setup_vk(params, &rel);
            let pk = midnight_zk_stdlib::setup_pk(&rel, &vk);
            let proof = midnight_zk_stdlib::prove::<_, TranscriptHash>(
                params,
                &pk,
                &rel,
                &inst_nc_agg,
                witness,
                ChaCha20Rng::from_seed([80u8.wrapping_add(chunk_idx as u8); 32]),
            )
            .expect("prove chunk+agg");
            nc_bundle_bytes += proof.len();

            let params_v = params.verifier_params();
            midnight_zk_stdlib::verify::<PiCcsNcChunkAggSumcheckRelation, TranscriptHash>(
                &params_v,
                &vk,
                &inst_nc_agg,
                None,
                &proof,
            )
            .expect("verify chunk+agg+sumcheck");
        } else {
            let chunk_sum_repr = chunk_instances[chunk_idx];
            let rel = PiCcsNcChunkRelation {
                ell_d,
                b: params_b,
                start_exp,
                count,
            };
            let instance = PiCcsNcChunkInstance {
                bundle_digest,
                chunk_sum: chunk_sum_repr,
                alpha_prime: alpha_repr.clone(),
                gamma: gamma_repr,
            };
            let witness = PiCcsNcChunkWitness {
                y_zcol: yz_slice
                    .iter()
                    .map(|yz| yz.iter().map(k_to_repr).collect::<Vec<_>>())
                    .collect(),
            };

            let circuit = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel);
            let model = circuit_model::<_, 48, 32>(&circuit);
            println!(
                "NC Chunk {chunk_idx}: outputs=[{start_i}..{}), count={count}, start_exp={start_exp}, min_k={} rows={}",
                start_i + count,
                model.k,
                model.rows
            );

            let params = params_cache.get(model.k);
            let vk = midnight_zk_stdlib::setup_vk(params, &rel);
            let pk = midnight_zk_stdlib::setup_pk(&rel, &vk);
            let proof = midnight_zk_stdlib::prove::<_, TranscriptHash>(
                params,
                &pk,
                &rel,
                &instance,
                witness,
                ChaCha20Rng::from_seed([80u8.wrapping_add(chunk_idx as u8); 32]),
            )
            .expect("prove chunk");
            nc_bundle_bytes += proof.len();
            nc_statement_bytes += <PiCcsNcChunkRelation as Relation>::format_instance(&instance)
                .expect("format_instance")
                .len()
                * 32;

            let params_v = params.verifier_params();
            midnight_zk_stdlib::verify::<PiCcsNcChunkRelation, TranscriptHash>(&params_v, &vk, &instance, None, &proof)
                .expect("verify chunk");
        }
    }

    println!("NC bundle bytes (chunks; last includes aggregate + sumcheck): {nc_bundle_bytes}");
    println!("NC statement bytes (estimated): {nc_statement_bytes}");

    // -----------------------------
    // FE sumcheck (step1) params (proved together with FE terminal aggregate)
    // -----------------------------
    let n_rounds_fe = pi.sumcheck_rounds.len();
    let poly_len_fe = pi.sumcheck_rounds[0].len();
    let initial_sum_fe = pi
        .sc_initial_sum
        .as_ref()
        .map(k_to_repr)
        .unwrap_or(KRepr::ZERO);
    let final_sum_fe = k_to_repr(&pi.sumcheck_final);

    // -----------------------------
    // FE terminal identity (step1, k_total=13)
    // -----------------------------
    let want_fe_chals = ell_n + ell_d;
    assert_eq!(pi.sumcheck_challenges.len(), want_fe_chals);
    let (_r_prime_fe, alpha_prime_fe) = pi.sumcheck_challenges.split_at(ell_n);

    assert_eq!(pi.challenges_public.alpha.len(), ell_d);
    assert_eq!(pi.challenges_public.beta_a.len(), ell_d);
    assert_eq!(pi.challenges_public.beta_r.len(), ell_n);

    let poly_terms: Vec<SparsePolyTermRepr> =
        s.f.terms()
            .iter()
            .map(|t| SparsePolyTermRepr {
                coeff: t.coeff.as_canonical_u64(),
                exps: t.exps.clone(),
            })
            .collect();
    let poly = SparsePolyRepr {
        t: s.t(),
        terms: poly_terms,
    };

    // me_inputs_r is taken from the step-0 DEC children, which are step-1 ME inputs.
    assert!(
        !fold_run.steps[0].fold.dec_children.is_empty(),
        "expected step-0 dec_children non-empty"
    );
    let me_inputs_r = &fold_run.steps[0].fold.dec_children[0].r;
    assert_eq!(me_inputs_r.len(), ell_n);

    let out0 = &step1.fold.ccs_out[0];
    let t_fe = s.t();
    assert_eq!(out0.y_scalars.len(), t_fe);

    // Flatten FE digit rows into (k_total-1)*t items (out_idx-major, then j).
    let mut y_rows_flat: Vec<Vec<K>> = Vec::with_capacity((k_total - 1) * t_fe);
    let (mut total_y_entries, mut nonzero_c1) = (0usize, 0usize);
    for (i_abs, out) in step1.fold.ccs_out.iter().enumerate().skip(1) {
        assert_eq!(out.y.len(), t_fe, "ccs_out[{i_abs}].y len mismatch");
        for j in 0..t_fe {
            assert_eq!(
                out.y[j].len(),
                1usize << ell_d,
                "ccs_out[{i_abs}].y[{j}] must be padded"
            );
            total_y_entries += out.y[j].len();
            for v in &out.y[j] {
                let (_, c1) = v.to_limbs_u64();
                if c1 != 0 {
                    nonzero_c1 += 1;
                }
            }
            y_rows_flat.push(out.y[j].clone());
        }
    }
    println!("FE y entries: total={total_y_entries} nonzero_c1={nonzero_c1} (c1==0 means base-field digit)");
    assert_eq!(y_rows_flat.len(), (k_total - 1) * t_fe);

    // Choose the largest FE chunk size (in flattened (out_idx,j) pairs).
    let total_pairs = y_rows_flat.len();
    let fe_chunk_size = choose_max_count(total_pairs, "PiCcsFeChunkRelation", |count| {
        let rel_try = PiCcsFeChunkRelation {
            ell_d,
            k_total,
            t: t_fe,
            // start_idx affects the (out_idx,j) pattern and thus row count (some pairs skip
            // a weight multiplication when j==0). Use the "last chunk" position as a
            // pessimistic/default choice for sizing.
            start_idx: total_pairs.saturating_sub(count),
            count,
        };
        let circuit_try = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel_try);
        let model_try = circuit_model::<_, 48, 32>(&circuit_try);
        (model_try.k, model_try.rows)
    });

    // Host-compute chunk sums.
    let mut fe_chunk_instances: Vec<KRepr> = Vec::new();
    for start_idx in (0..total_pairs).step_by(fe_chunk_size) {
        let count = core::cmp::min(fe_chunk_size, total_pairs - start_idx);
        let cs = host_fe_chunk_sum(
            &y_rows_flat,
            alpha_prime_fe,
            pi.challenges_public.gamma,
            k_total,
            t_fe,
            start_idx,
            count,
        );
        fe_chunk_instances.push(k_to_repr(&cs));
    }

    let alpha_prime_repr: Vec<KRepr> = alpha_prime_fe.iter().map(k_to_repr).collect();
    let gamma_repr_fe = k_to_repr(&pi.challenges_public.gamma);

    // Prove all FE chunks (each binds one chunk_sum).
    let mut fe_chunk_ranges: Vec<(usize, usize)> = Vec::new();
    for start_idx in (0..total_pairs).step_by(fe_chunk_size) {
        let count = core::cmp::min(fe_chunk_size, total_pairs - start_idx);
        fe_chunk_ranges.push((start_idx, count));
    }
    assert_eq!(fe_chunk_ranges.len(), fe_chunk_instances.len());

    let n_chunks_fe = fe_chunk_instances.len();

    let mut fe_bundle_bytes: usize = 0;
    let mut fe_statement_bytes: usize = 0;
    let mut did_one_shot_fe = false;
    if n_chunks_fe == 1 {
        // If the entire Eval' table fits in a single chunk *and* the combined circuit
        // (chunk binding + FE sumcheck + FE aggregate) fits under Midnight's cap, prove it
        // all in one proof.
        let rel = PiCcsFeChunkAggSumcheckRelation {
            n_rounds: n_rounds_fe,
            poly_len: poly_len_fe,
            ell_n,
            ell_d,
            k_total,
            poly: poly.clone(),
            start_idx: 0,
            count: total_pairs,
            n_chunks: 1,
            chunk_index: 0,
        };

        let circuit = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel);
        let model = circuit_model::<_, 48, 32>(&circuit);
        println!(
            "FE Chunk+Agg+Sumcheck: pairs=[0..{total_pairs}), count={total_pairs}, min_k={} rows={}",
            model.k, model.rows
        );

        let instance = PiCcsFeChunkAggSumcheckInstance {
            bundle_digest,
            sumcheck_challenges: pi.sumcheck_challenges.iter().map(k_to_repr).collect(),
            gamma: gamma_repr_fe,
            alpha: pi.challenges_public.alpha.iter().map(k_to_repr).collect(),
            beta_a: pi.challenges_public.beta_a.iter().map(k_to_repr).collect(),
            beta_r: pi.challenges_public.beta_r.iter().map(k_to_repr).collect(),
            chunk_sums: fe_chunk_instances.clone(),
            initial_sum: initial_sum_fe,
            final_sum: final_sum_fe,
        };
        let witness = PiCcsFeChunkAggSumcheckWitness {
            rounds: pi
                .sumcheck_rounds
                .iter()
                .map(|r| r.iter().map(k_to_repr).collect())
                .collect(),
            me_inputs_r: me_inputs_r.iter().map(k_to_repr).collect(),
            y_scalars_0: out0.y_scalars.iter().map(k_to_repr).collect(),
            y_rows: y_rows_flat
                .iter()
                .map(|row| row.iter().map(k_to_repr).collect::<Vec<_>>())
                .collect(),
        };

        fe_statement_bytes += <PiCcsFeChunkAggSumcheckRelation as Relation>::format_instance(&instance)
            .expect("format_instance")
            .len()
            * 32;

        let params = params_cache.get(model.k);
        let vk = midnight_zk_stdlib::setup_vk(params, &rel);
        let pk = midnight_zk_stdlib::setup_pk(&rel, &vk);
        let proof = midnight_zk_stdlib::prove::<_, TranscriptHash>(
            params,
            &pk,
            &rel,
            &instance,
            witness,
            ChaCha20Rng::from_seed([141u8; 32]),
        )
        .expect("prove fe chunk+agg+sumcheck");
        fe_bundle_bytes += proof.len();

        let params_v = params.verifier_params();
        midnight_zk_stdlib::verify::<PiCcsFeChunkAggSumcheckRelation, TranscriptHash>(&params_v, &vk, &instance, None, &proof)
            .expect("verify fe chunk+agg+sumcheck");
        did_one_shot_fe = true;
    }

    if !did_one_shot_fe {
        // Prove all FE chunks (each binds one chunk_sum).
        for (chunk_idx, (start_idx, count)) in fe_chunk_ranges.iter().copied().enumerate() {
            let y_rows_slice = &y_rows_flat[start_idx..start_idx + count];

            let rel = PiCcsFeChunkRelation {
                ell_d,
                k_total,
                t: t_fe,
                start_idx,
                count,
            };
            let instance = PiCcsFeChunkInstance {
                bundle_digest,
                chunk_sum: fe_chunk_instances[chunk_idx],
                alpha_prime: alpha_prime_repr.clone(),
                gamma: gamma_repr_fe,
            };
            let witness = PiCcsFeChunkWitness {
                y_rows: y_rows_slice
                    .iter()
                    .map(|row| row.iter().map(k_to_repr).collect::<Vec<_>>())
                    .collect(),
            };

            let circuit = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel);
            let model = circuit_model::<_, 48, 32>(&circuit);
            println!(
                "FE Chunk {chunk_idx}: pairs=[{start_idx}..{}), count={count}, min_k={} rows={}",
                start_idx + count,
                model.k,
                model.rows
            );

            let params = params_cache.get(model.k);
            let vk = midnight_zk_stdlib::setup_vk(params, &rel);
            let pk = midnight_zk_stdlib::setup_pk(&rel, &vk);
            let proof = midnight_zk_stdlib::prove::<_, TranscriptHash>(
                params,
                &pk,
                &rel,
                &instance,
                witness,
                ChaCha20Rng::from_seed([121u8.wrapping_add(chunk_idx as u8); 32]),
            )
            .expect("prove fe chunk");
            fe_bundle_bytes += proof.len();
            fe_statement_bytes += <PiCcsFeChunkRelation as Relation>::format_instance(&instance)
                .expect("format_instance")
                .len()
                * 32;

            let params_v = params.verifier_params();
            midnight_zk_stdlib::verify::<PiCcsFeChunkRelation, TranscriptHash>(&params_v, &vk, &instance, None, &proof)
                .expect("verify fe chunk");
        }

        // One combined proof for: FE sumcheck + FE terminal identity aggregate (no chunk binding).
        let rel_fe_sc_agg = PiCcsFeChunkAggSumcheckRelation {
            n_rounds: n_rounds_fe,
            poly_len: poly_len_fe,
            ell_n,
            ell_d,
            k_total,
            poly: poly.clone(),
            start_idx: 0,
            count: 0,
            n_chunks: n_chunks_fe,
            chunk_index: 0,
        };
        let inst_fe_sc_agg = PiCcsFeChunkAggSumcheckInstance {
            bundle_digest,
            sumcheck_challenges: pi.sumcheck_challenges.iter().map(k_to_repr).collect(),
            gamma: gamma_repr_fe,
            alpha: pi.challenges_public.alpha.iter().map(k_to_repr).collect(),
            beta_a: pi.challenges_public.beta_a.iter().map(k_to_repr).collect(),
            beta_r: pi.challenges_public.beta_r.iter().map(k_to_repr).collect(),
            chunk_sums: fe_chunk_instances.clone(),
            initial_sum: initial_sum_fe,
            final_sum: final_sum_fe,
        };
        let wit_fe_sc_agg = PiCcsFeChunkAggSumcheckWitness {
            rounds: pi
                .sumcheck_rounds
                .iter()
                .map(|r| r.iter().map(k_to_repr).collect())
                .collect(),
            me_inputs_r: me_inputs_r.iter().map(k_to_repr).collect(),
            y_scalars_0: out0.y_scalars.iter().map(k_to_repr).collect(),
            y_rows: Vec::new(),
        };

        let circuit_fe_sc_agg = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel_fe_sc_agg);
        let model_fe_sc_agg = circuit_model::<_, 48, 32>(&circuit_fe_sc_agg);
        println!(
            "FE Sumcheck+Agg: n_chunks={n_chunks_fe} min_k={} rows={}",
            model_fe_sc_agg.k, model_fe_sc_agg.rows
        );

        let params_fe_sc_agg = params_cache.get(model_fe_sc_agg.k);
        let vk_fe_sc_agg = midnight_zk_stdlib::setup_vk(params_fe_sc_agg, &rel_fe_sc_agg);
        let pk_fe_sc_agg = midnight_zk_stdlib::setup_pk(&rel_fe_sc_agg, &vk_fe_sc_agg);
        let proof_fe_sc_agg = midnight_zk_stdlib::prove::<_, TranscriptHash>(
            params_fe_sc_agg,
            &pk_fe_sc_agg,
            &rel_fe_sc_agg,
            &inst_fe_sc_agg,
            wit_fe_sc_agg,
            ChaCha20Rng::from_seed([141u8; 32]),
        )
        .expect("prove fe sumcheck+agg");
        fe_bundle_bytes += proof_fe_sc_agg.len();
        fe_statement_bytes += <PiCcsFeChunkAggSumcheckRelation as Relation>::format_instance(&inst_fe_sc_agg)
            .expect("format_instance")
            .len()
            * 32;

        let params_v_fe_sc_agg = params_fe_sc_agg.verifier_params();
        midnight_zk_stdlib::verify::<PiCcsFeChunkAggSumcheckRelation, TranscriptHash>(
            &params_v_fe_sc_agg,
            &vk_fe_sc_agg,
            &inst_fe_sc_agg,
            None,
            &proof_fe_sc_agg,
        )
        .expect("verify fe sumcheck+agg");
    }

    println!("FE bundle bytes: {fe_bundle_bytes}");
    println!("FE statement bytes (estimated): {fe_statement_bytes}");

    let total_bundle_bytes = nc_bundle_bytes + fe_bundle_bytes;
    let total_statement_bytes = nc_statement_bytes + fe_statement_bytes;
    println!("Total step-1 bundle bytes (NC bundle + FE bundle): {total_bundle_bytes}");
    println!("Total step-1 statement bytes (estimated): {total_statement_bytes}");
    println!(
        "Total step-1 payload bytes (proof + statement): {}",
        total_bundle_bytes + total_statement_bytes
    );
    assert!(
        total_bundle_bytes < 50 * 1024,
        "expected full step-1 bundle < 50KB; need fewer/smaller chunks or combined circuits"
    );
}
