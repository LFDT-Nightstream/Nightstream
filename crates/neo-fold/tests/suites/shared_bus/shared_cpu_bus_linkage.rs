#![allow(non_snake_case)]

use std::marker::PhantomData;

use neo_ajtai::{AjtaiSModule, Commitment as Cmt};
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::{CcsClaim, CcsStructure, CcsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify, CommitMixers, StepProof};
use neo_fold::PiCcsError;
use neo_math::{F, K};
use neo_memory::plain::{LutTable, PlainLutTrace, PlainMemLayout, PlainMemTrace};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_memory::MemInit;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

use crate::suite::{default_mixers, setup_ajtai_committer};

fn create_identity_ccs(n: usize) -> CcsStructure<F> {
    let mat = Mat::identity(n);
    let f = SparsePoly::new(1, vec![]);
    CcsStructure::new(vec![mat], f).expect("CCS")
}

fn write_bits_le(out: &mut [F], mut x: u64, ell: usize) {
    for i in 0..ell {
        out[i] = if (x & 1) == 1 { F::ONE } else { F::ZERO };
        x >>= 1;
    }
}

fn bus_cols_shout(d: usize, ell: usize) -> usize {
    d * ell + 2
}

fn bus_cols_twist(d: usize, ell: usize) -> usize {
    2 * d * ell + 5
}

fn build_cpu_witness_with_bus(
    m: usize,
    bus_base: usize,
    chunk_size: usize,
    step_in_chunk: usize,
    lut_inst: &neo_memory::witness::LutInstance<Cmt, F>,
    lut_trace: &PlainLutTrace<F>,
    mem_inst: &neo_memory::witness::MemInstance<Cmt, F>,
    mem_trace: &PlainMemTrace<F>,
    tag: u64,
) -> Vec<F> {
    let mut z = vec![F::ZERO; m];
    if !z.is_empty() {
        z[0] = F::from_u64(tag);
    }

    let mut col_id = 0usize;

    // Shout: addr_bits, has_lookup, val
    {
        let ell_addr = lut_inst.d * lut_inst.ell;
        let mut bits = vec![F::ZERO; ell_addr];
        let addr = lut_trace.addr[step_in_chunk];
        let mut tmp = addr;
        for dim in 0..lut_inst.d {
            let comp = (tmp % (lut_inst.n_side as u64)) as u64;
            tmp /= lut_inst.n_side as u64;
            let offset = dim * lut_inst.ell;
            write_bits_le(&mut bits[offset..offset + lut_inst.ell], comp, lut_inst.ell);
        }
        for j in 0..ell_addr {
            z[bus_base + col_id * chunk_size + step_in_chunk] = bits[j];
            col_id += 1;
        }
        z[bus_base + col_id * chunk_size + step_in_chunk] = lut_trace.has_lookup[step_in_chunk];
        col_id += 1;
        z[bus_base + col_id * chunk_size + step_in_chunk] = lut_trace.val[step_in_chunk];
        col_id += 1;
    }

    // Twist: ra_bits, wa_bits, has_read, has_write, wv, rv, inc
    {
        let ell_addr = mem_inst.d * mem_inst.ell;
        let mut ra_bits = vec![F::ZERO; ell_addr];
        let mut wa_bits = vec![F::ZERO; ell_addr];

        let ra = mem_trace.read_addr[step_in_chunk];
        let wa = mem_trace.write_addr[step_in_chunk];

        let mut tmp = ra;
        for dim in 0..mem_inst.d {
            let comp = (tmp % (mem_inst.n_side as u64)) as u64;
            tmp /= mem_inst.n_side as u64;
            let offset = dim * mem_inst.ell;
            write_bits_le(&mut ra_bits[offset..offset + mem_inst.ell], comp, mem_inst.ell);
        }
        let mut tmp = wa;
        for dim in 0..mem_inst.d {
            let comp = (tmp % (mem_inst.n_side as u64)) as u64;
            tmp /= mem_inst.n_side as u64;
            let offset = dim * mem_inst.ell;
            write_bits_le(&mut wa_bits[offset..offset + mem_inst.ell], comp, mem_inst.ell);
        }

        for j in 0..ell_addr {
            z[bus_base + col_id * chunk_size + step_in_chunk] = ra_bits[j];
            col_id += 1;
        }
        for j in 0..ell_addr {
            z[bus_base + col_id * chunk_size + step_in_chunk] = wa_bits[j];
            col_id += 1;
        }

        z[bus_base + col_id * chunk_size + step_in_chunk] = mem_trace.has_read[step_in_chunk];
        col_id += 1;
        z[bus_base + col_id * chunk_size + step_in_chunk] = mem_trace.has_write[step_in_chunk];
        col_id += 1;
        z[bus_base + col_id * chunk_size + step_in_chunk] = mem_trace.write_val[step_in_chunk];
        col_id += 1;
        z[bus_base + col_id * chunk_size + step_in_chunk] = mem_trace.read_val[step_in_chunk];
        col_id += 1;
        z[bus_base + col_id * chunk_size + step_in_chunk] = mem_trace.inc_at_write_addr[step_in_chunk];
        col_id += 1;
    }

    debug_assert_eq!(
        col_id,
        bus_cols_shout(lut_inst.d, lut_inst.ell) + bus_cols_twist(mem_inst.d, mem_inst.ell),
        "bus col count mismatch"
    );

    z
}

fn build_time_columns_from_flattened_test_witness(
    z: &[F],
    m_in: usize,
    bus_base: usize,
    bus_cols: usize,
    chunk_size: usize,
) -> neo_memory::witness::TimeColumns<F> {
    let cpu_region_len = bus_base
        .checked_sub(m_in)
        .expect("bus_base must be >= m_in for flattened test witness");
    assert!(
        cpu_region_len % chunk_size == 0,
        "cpu region must be chunk-aligned (cpu_region_len={}, chunk_size={})",
        cpu_region_len,
        chunk_size
    );
    let cpu_cols_len_raw = cpu_region_len / chunk_size;
    let cpu_cols_len = cpu_cols_len_raw.max(21);

    let mut cpu_cols = vec![vec![F::ZERO; chunk_size]; cpu_cols_len];
    for col in 0..cpu_cols_len_raw {
        for j in 0..chunk_size {
            cpu_cols[col][j] = z[m_in + col * chunk_size + j];
        }
    }

    let mut mem_cols = vec![vec![F::ZERO; chunk_size]; bus_cols];
    for col in 0..bus_cols {
        for j in 0..chunk_size {
            mem_cols[col][j] = z[bus_base + col * chunk_size + j];
        }
    }

    let mut col_ids = Vec::with_capacity(cpu_cols_len + bus_cols);
    for id in 0..(cpu_cols_len + bus_cols) {
        col_ids.push(id);
    }

    neo_memory::witness::TimeColumns {
        t: chunk_size,
        cpu_cols,
        mem_cols,
        active_col: vec![F::ONE; chunk_size],
        col_ids,
    }
}

struct SharedBusFixture {
    params: NeoParams,
    ccs: CcsStructure<F>,
    l: AjtaiSModule,
    mixers: CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt>,
    steps_witness: Vec<StepWitnessBundle<Cmt, F, K>>,
    steps_instance: Vec<StepInstanceBundle<Cmt, F, K>>,
}

fn build_one_step_fixture(seed: u64) -> SharedBusFixture {
    let n = 32usize;
    let ccs = create_identity_ccs(n);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");
    params.k_rho = 16;
    let l = setup_ajtai_committer(&params, ccs.m);
    let mixers = default_mixers();

    let m_in = 5usize;

    // Geometry: k=2, d=1, n_side=2 (minimal).
    let mem_layout = PlainMemLayout {
        k: 2,
        d: 1,
        n_side: 2,
        lanes: 1,
    };
    let mem_init = MemInit::Zero;

    let write0 = F::from_u64(seed.wrapping_add(10));
    let mem_trace = PlainMemTrace {
        steps: 1,
        has_read: vec![F::ZERO],
        has_write: vec![F::ONE],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::ZERO],
        write_val: vec![write0],
        inc_at_write_addr: vec![write0],
    };

    // Shout table: k=2, d=1, n_side=2 (minimal).
    let lut_table = LutTable {
        table_id: 0,
        k: 2,
        d: 1,
        n_side: 2,
        content: vec![F::from_u64(11), F::from_u64(22)],
    };
    let lut_trace = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![1],
        val: vec![lut_table.content[1]],
    };

    let mem_ell = mem_layout.n_side.trailing_zeros() as usize;
    let lut_ell = lut_table.n_side.trailing_zeros() as usize;

    let mem_inst = neo_memory::witness::MemInstance::<Cmt, F> {
        mem_id: 0,
        comms: Vec::new(),
        k: mem_layout.k,
        d: mem_layout.d,
        n_side: mem_layout.n_side,
        steps: mem_trace.steps,
        lanes: mem_layout.lanes.max(1),
        ell: mem_ell,
        init: mem_init,
        init_digest: None,
    };
    let mem_wit = neo_memory::witness::MemWitness { mats: Vec::new() };

    let lut_inst = neo_memory::witness::LutInstance::<Cmt, F> {
        table_id: lut_table.table_id,
        comms: Vec::new(),
        k: lut_table.k,
        d: lut_table.d,
        n_side: lut_table.n_side,
        steps: lut_trace.has_lookup.len(),
        lanes: 1,
        ell: lut_ell,
        table_spec: None,
        table: lut_table.content.clone(),
        table_digest: None,
        addr_group: None,
        selector_group: None,
    };
    let lut_wit = neo_memory::witness::LutWitness { mats: Vec::new() };

    let bus_cols_total = bus_cols_shout(lut_inst.d, lut_inst.ell) + bus_cols_twist(mem_inst.d, mem_inst.ell);
    let chunk_size = 1usize;
    let bus_base = ccs.m - bus_cols_total * chunk_size;
    let z = build_cpu_witness_with_bus(
        ccs.m, bus_base, chunk_size, 0, &lut_inst, &lut_trace, &mem_inst, &mem_trace, seed,
    );
    let time_columns = build_time_columns_from_flattened_test_witness(&z, m_in, bus_base, bus_cols_total, chunk_size);
    let Z = neo_memory::ajtai::encode_vector_balanced_to_mat(&params, &z);
    let c = l.commit(&Z);
    let x = z[..m_in].to_vec();
    let w = z[m_in..].to_vec();
    let mcs = (CcsClaim { c, x, m_in }, CcsWitness { w, Z });

    let steps_witness = vec![StepWitnessBundle {
        mcs,
        lut_instances: vec![(lut_inst, lut_wit)],
        mem_instances: vec![(mem_inst, mem_wit)],
        time_columns,
        _phantom: PhantomData::<K>,
    }];
    let steps_instance = steps_witness.iter().map(StepInstanceBundle::from).collect();

    SharedBusFixture {
        params,
        ccs,
        l,
        mixers,
        steps_witness,
        steps_instance,
    }
}

fn prove_and_verify_shared(fx: &SharedBusFixture) -> Result<(), PiCcsError> {
    let mut tr = Poseidon2Transcript::new(b"shared-cpu-bus");
    let proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr,
        &fx.params,
        &fx.ccs,
        &fx.steps_witness,
        &[],
        &[],
        &fx.l,
        fx.mixers,
    )?;

    let mut tr_v = Poseidon2Transcript::new(b"shared-cpu-bus");
    let _outputs = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &fx.params,
        &fx.ccs,
        &fx.steps_instance,
        &[],
        &proof,
        fx.mixers,
    )?;
    Ok(())
}

fn first_materialized_step_mut(step: &mut StepProof) -> &mut StepProof {
    if step
        .compressed_substeps
        .as_ref()
        .is_some_and(|sub| !sub.is_empty())
    {
        return step
            .compressed_substeps
            .as_mut()
            .and_then(|sub| sub.first_mut())
            .expect("expected at least one compressed materialized proof step");
    }
    step
}

#[test]
fn shared_cpu_bus_happy_path_one_step() {
    let fx = build_one_step_fixture(7);
    prove_and_verify_shared(&fx).expect("shared-bus prove+verify should succeed");
}

#[test]
fn shared_cpu_bus_tamper_bus_opening_fails() {
    let fx = build_one_step_fixture(8);

    let mut tr = Poseidon2Transcript::new(b"shared-cpu-bus");
    let mut proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr,
        &fx.params,
        &fx.ccs,
        &fx.steps_witness,
        &[],
        &[],
        &fx.l,
        fx.mixers,
    )
    .expect("prove");

    // Tamper a shared-bus named opening at r_time.
    let container = proof
        .steps
        .first_mut()
        .expect("expected at least one proof step");
    let step0 = first_materialized_step_mut(container);
    let r_time = step0
        .fold
        .ccs_out
        .first()
        .expect("expected at least one CPU CE output")
        .r
        .clone();
    let cpu_cols_len = step0.fold.time_cpu_commitments.len();
    let bus_logical_col = *step0
        .fold
        .time_col_ids
        .get(cpu_cols_len)
        .expect("expected at least one committed bus column id");
    let open_idx = step0
        .fold
        .openings
        .iter()
        .position(|opening| opening.point == r_time && opening.col_ids.iter().any(|&c| c == bus_logical_col))
        .or_else(|| {
            step0
                .fold
                .openings
                .iter()
                .position(|opening| opening.col_ids.iter().any(|&c| c == bus_logical_col))
        })
        .or_else(|| {
            step0
                .fold
                .openings
                .iter()
                .position(|opening| opening.point == r_time)
        })
        .expect("expected r_time opening carrying shared-bus columns");
    let opening = &mut step0.fold.openings[open_idx];
    assert!(
        !opening.evals.is_empty(),
        "shared-bus named opening evals must be non-empty"
    );
    let eval_idx = opening
        .col_ids
        .iter()
        .position(|&c| c == bus_logical_col)
        .unwrap_or(0);
    assert!(
        eval_idx < opening.evals.len(),
        "shared-bus opening index must be in-bounds"
    );
    opening.evals[eval_idx] += K::ONE;

    let mut tr_v = Poseidon2Transcript::new(b"shared-cpu-bus");
    assert!(
        fold_shard_verify(
            FoldingMode::Optimized,
            &mut tr_v,
            &fx.params,
            &fx.ccs,
            &fx.steps_instance,
            &[],
            &proof,
            fx.mixers,
        )
        .is_err(),
        "tampering CPU bus opening must break verification in shared-bus mode"
    );
}

#[test]
fn shared_cpu_bus_missing_cpu_me_claim_val_fails() {
    let fx = build_one_step_fixture(9);

    let mut tr = Poseidon2Transcript::new(b"shared-cpu-bus");
    let mut proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr,
        &fx.params,
        &fx.ccs,
        &fx.steps_witness,
        &[],
        &[],
        &fx.l,
        fx.mixers,
    )
    .expect("prove");

    // Shared-bus mode expects CPU ME claims at r_val inside mem proof, so dropping them must fail.
    proof.steps[0].mem.val_me_claims.clear();

    let mut tr_v = Poseidon2Transcript::new(b"shared-cpu-bus");
    assert!(
        fold_shard_verify(
            FoldingMode::Optimized,
            &mut tr_v,
            &fx.params,
            &fx.ccs,
            &fx.steps_instance,
            &[],
            &proof,
            fx.mixers,
        )
        .is_err(),
        "missing CPU ME@r_val must fail in shared-bus mode"
    );
}

#[test]
fn shared_cpu_bus_missing_named_time_opening_fails() {
    let fx = build_one_step_fixture(10);

    let mut tr = Poseidon2Transcript::new(b"shared-cpu-bus");
    let mut proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr,
        &fx.params,
        &fx.ccs,
        &fx.steps_witness,
        &[],
        &[],
        &fx.l,
        fx.mixers,
    )
    .expect("prove");

    // In the in-place Route-A refactor, shared-bus verification requires explicit named time openings.
    proof.steps[0].fold.openings.clear();

    let mut tr_v = Poseidon2Transcript::new(b"shared-cpu-bus");
    assert!(
        fold_shard_verify(
            FoldingMode::Optimized,
            &mut tr_v,
            &fx.params,
            &fx.ccs,
            &fx.steps_instance,
            &[],
            &proof,
            fx.mixers,
        )
        .is_err(),
        "missing named time openings must fail in shared-bus mode"
    );
}

#[test]
fn shared_cpu_bus_stage8_tamper_matrix_fails() {
    let fx = build_one_step_fixture(11);

    let mut tr = Poseidon2Transcript::new(b"shared-cpu-bus");
    let proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr,
        &fx.params,
        &fx.ccs,
        &fx.steps_witness,
        &[],
        &[],
        &fx.l,
        fx.mixers,
    )
    .expect("prove");

    let verify = |candidate| {
        let mut tr_v = Poseidon2Transcript::new(b"shared-cpu-bus");
        fold_shard_verify(
            FoldingMode::Optimized,
            &mut tr_v,
            &fx.params,
            &fx.ccs,
            &fx.steps_instance,
            &[],
            candidate,
            fx.mixers,
        )
    };

    // Baseline must pass.
    let _ = verify(&proof).expect("baseline verify");
    assert!(
        proof.steps[0]
            .fold
            .joint_opening_lane
            .unified_fold
            .is_some(),
        "canonical Stage-8 must always emit unified_fold when groups are present"
    );
    let expected_stage8_fold_len = if proof.steps[0].fold.joint_opening_lane.groups.is_empty() {
        0usize
    } else {
        1usize
    };
    assert_eq!(
        proof.steps[0].stage8_fold.len(),
        expected_stage8_fold_len,
        "Stage-8 fold proof count must match canonical lane plan"
    );

    // 1) Manifest digest tamper must fail.
    let mut tampered_manifest = proof.clone();
    tampered_manifest.steps[0].fold.opening_manifest.digest[0] ^= 1;
    assert!(
        verify(&tampered_manifest).is_err(),
        "tampering opening manifest digest must fail verification"
    );

    // 2) Reduction group digest tamper must fail.
    let mut tampered_reduction = proof.clone();
    assert!(
        !tampered_reduction.steps[0]
            .fold
            .opening_reduction
            .groups
            .is_empty(),
        "fixture must have at least one Stage-8 reduction group"
    );
    tampered_reduction.steps[0].fold.opening_reduction.groups[0].group_digest[0] ^= 1;
    assert!(
        verify(&tampered_reduction).is_err(),
        "tampering reduction group digest must fail verification"
    );

    // 3) Unification proof tamper must fail.
    let mut tampered_unification = proof.clone();
    assert!(
        !tampered_unification.steps[0]
            .fold
            .opening_unification
            .round_polys
            .is_empty(),
        "fixture must have non-empty opening unification rounds"
    );
    tampered_unification.steps[0]
        .fold
        .opening_unification
        .round_polys[0][0] += K::ONE;
    assert!(
        verify(&tampered_unification).is_err(),
        "tampering opening unification sumcheck proof must fail verification"
    );

    // 4) Unified claim tamper must fail.
    let mut tampered_unified_claim = proof.clone();
    let unified = tampered_unified_claim.steps[0]
        .fold
        .joint_opening_lane
        .unified_fold
        .as_mut()
        .expect("unified fold present");
    unified.joint_claim += K::ONE;
    assert!(
        verify(&tampered_unified_claim).is_err(),
        "tampering Stage-8 unified claim must fail verification"
    );

    // 5) Missing stage8_fold proof with non-empty Stage-8 groups must fail.
    let mut tampered_stage8_lane = proof.clone();
    tampered_stage8_lane.steps[0].stage8_fold.clear();
    assert!(
        verify(&tampered_stage8_lane).is_err(),
        "missing Stage-8 fold proof must fail verification when Stage-8 groups exist"
    );
}
