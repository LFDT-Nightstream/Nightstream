#![allow(dead_code)]

use std::marker::PhantomData;
use std::sync::Arc;

use neo_ajtai::{setup as ajtai_setup, AjtaiSModule, Commitment as Cmt};
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::{CcsClaim, CcsStructure, CcsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_math::{D, F, K};
use neo_memory::ajtai::{commit_cols_for_ccs_m, encode_vector_for_ccs_m};
use neo_memory::plain::{LutTable, PlainLutTrace, PlainMemLayout, PlainMemTrace};
use neo_memory::riscv::trace::rv32_trace_lookup_n_vals_for_table_id;
use neo_memory::witness::{LutInstance, LutWitness, MemInstance, MemWitness, StepWitnessBundle};
use neo_memory::MemInit;
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub fn setup_ajtai_committer(m: usize, kappa: usize) -> AjtaiSModule {
    let m_commit = commit_cols_for_ccs_m(m);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let pp = ajtai_setup(&mut rng, D, kappa, m_commit).expect("Ajtai setup");
    AjtaiSModule::new(Arc::new(pp))
}

pub fn create_identity_ccs(n: usize) -> CcsStructure<F> {
    let mat = Mat::identity(n);
    let f = SparsePoly::new(1, vec![]);
    CcsStructure::new(vec![mat], f).expect("CCS")
}

pub fn create_mcs_from_z(
    params: &NeoParams,
    l: &AjtaiSModule,
    m_in: usize,
    z: Vec<F>,
) -> (CcsClaim<Cmt, F>, CcsWitness<F>) {
    let z_enc = encode_vector_for_ccs_m(params, z.len(), &z).expect("encode witness for CCS width");
    let c = l.commit(&z_enc);
    let x = z[..m_in].to_vec();
    let w = z[m_in..].to_vec();
    (CcsClaim { c, x, m_in }, CcsWitness { w, Z: z_enc })
}

pub fn make_twist_instance(
    mem_id: u32,
    layout: &PlainMemLayout,
    init: MemInit<F>,
    steps: usize,
) -> (MemInstance<Cmt, F>, MemWitness<F>) {
    let ell = layout.n_side.trailing_zeros() as usize;
    (
        MemInstance {
            mem_id,
            comms: Vec::new(),
            k: layout.k,
            d: layout.d,
            n_side: layout.n_side,
            steps,
            lanes: layout.lanes.max(1),
            ell,
            init,
            init_digest: None,
            guest_addr_remap: None,
        },
        MemWitness { mats: Vec::new() },
    )
}

pub fn make_shout_instance(
    table_id: u32,
    table: LutTable<F>,
    steps: usize,
    lanes: usize,
) -> (LutInstance<Cmt, F>, LutWitness<F>) {
    let ell = table.n_side.trailing_zeros() as usize;
    (
        LutInstance {
            table_id,
            comms: Vec::new(),
            k: table.k,
            d: table.d,
            n_side: table.n_side,
            steps,
            lanes: lanes.max(1),
            ell,
            table_spec: None,
            table: table.content,
            table_digest: None,
            addr_group: None,
            selector_group: None,
        },
        LutWitness { mats: Vec::new() },
    )
}

fn write_bits_le(out: &mut [F], mut x: u64, ell: usize) {
    for i in 0..ell {
        out[i] = if (x & 1) == 1 { F::ONE } else { F::ZERO };
        x >>= 1;
    }
}

fn flatten_cols(cols: &[Vec<F>]) -> Vec<F> {
    let total: usize = cols.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(total);
    for col in cols {
        out.extend_from_slice(col);
    }
    out
}

fn write_shout_lane_columns(inst: &LutInstance<Cmt, F>, trace: &PlainLutTrace<F>, cols: &mut Vec<Vec<F>>) {
    let t = trace.has_lookup.len();
    let ell_addr = inst.d * inst.ell;
    let mut addr_cols = vec![vec![F::ZERO; t]; ell_addr];
    let mut has_lookup = vec![F::ZERO; t];
    let mut val_cols = vec![vec![F::ZERO; t]; rv32_trace_lookup_n_vals_for_table_id(inst.table_id).max(1)];

    for j in 0..t {
        has_lookup[j] = trace.has_lookup[j];
        if trace.has_lookup[j] == F::ONE {
            let mut addr_bits = vec![F::ZERO; ell_addr];
            let mut tmp = trace.addr[j];
            for dim in 0..inst.d {
                let comp = tmp % (inst.n_side as u64);
                tmp /= inst.n_side as u64;
                let offset = dim * inst.ell;
                write_bits_le(&mut addr_bits[offset..offset + inst.ell], comp, inst.ell);
            }
            for (bit_idx, bit) in addr_bits.into_iter().enumerate() {
                addr_cols[bit_idx][j] = bit;
            }
            val_cols[0][j] = trace.val[j];
        }
    }

    cols.extend(addr_cols);
    cols.push(has_lookup);
    cols.extend(val_cols);
}

fn write_twist_lane_columns(inst: &MemInstance<Cmt, F>, trace: &PlainMemTrace<F>, cols: &mut Vec<Vec<F>>) {
    let t = trace.steps;
    let ell_addr = inst.d * inst.ell;
    let mut ra_cols = vec![vec![F::ZERO; t]; ell_addr];
    let mut wa_cols = vec![vec![F::ZERO; t]; ell_addr];
    let mut has_read = vec![F::ZERO; t];
    let mut has_write = vec![F::ZERO; t];
    let mut wv = vec![F::ZERO; t];
    let mut rv = vec![F::ZERO; t];
    let mut inc = vec![F::ZERO; t];

    for j in 0..t {
        has_read[j] = trace.has_read[j];
        has_write[j] = trace.has_write[j];
        if trace.has_read[j] == F::ONE {
            let mut bits = vec![F::ZERO; ell_addr];
            let mut tmp = trace.read_addr[j];
            for dim in 0..inst.d {
                let comp = tmp % (inst.n_side as u64);
                tmp /= inst.n_side as u64;
                let offset = dim * inst.ell;
                write_bits_le(&mut bits[offset..offset + inst.ell], comp, inst.ell);
            }
            for (bit_idx, bit) in bits.into_iter().enumerate() {
                ra_cols[bit_idx][j] = bit;
            }
            rv[j] = trace.read_val[j];
        }
        if trace.has_write[j] == F::ONE {
            let mut bits = vec![F::ZERO; ell_addr];
            let mut tmp = trace.write_addr[j];
            for dim in 0..inst.d {
                let comp = tmp % (inst.n_side as u64);
                tmp /= inst.n_side as u64;
                let offset = dim * inst.ell;
                write_bits_le(&mut bits[offset..offset + inst.ell], comp, inst.ell);
            }
            for (bit_idx, bit) in bits.into_iter().enumerate() {
                wa_cols[bit_idx][j] = bit;
            }
            wv[j] = trace.write_val[j];
            inc[j] = trace.inc_at_write_addr[j];
        }
    }

    cols.extend(ra_cols);
    cols.extend(wa_cols);
    cols.push(has_read);
    cols.push(has_write);
    cols.push(wv);
    cols.push(rv);
    cols.push(inc);
}

pub fn create_step_with_bus(
    params: &NeoParams,
    ccs: &CcsStructure<F>,
    l: &AjtaiSModule,
    m_in: usize,
    public_inputs: Vec<F>,
    cpu_cols: Vec<Vec<F>>,
    shout_instances: Vec<(LutInstance<Cmt, F>, LutWitness<F>, Vec<PlainLutTrace<F>>)>,
    mem_instances: Vec<(MemInstance<Cmt, F>, MemWitness<F>, Vec<PlainMemTrace<F>>)>,
) -> StepWitnessBundle<Cmt, F, K> {
    let t = cpu_cols
        .first()
        .map(Vec::len)
        .or_else(|| {
            shout_instances
                .first()
                .and_then(|(_, _, traces)| traces.first().map(|trace| trace.has_lookup.len()))
        })
        .or_else(|| {
            mem_instances
                .first()
                .and_then(|(_, _, traces)| traces.first().map(|trace| trace.steps))
        })
        .unwrap_or(1);

    assert_eq!(public_inputs.len(), m_in, "public_inputs len must equal m_in");
    for col in &cpu_cols {
        assert_eq!(col.len(), t, "cpu col length mismatch");
    }

    let mut mem_cols = Vec::new();
    for (inst, _, traces) in &shout_instances {
        assert_eq!(traces.len(), inst.lanes.max(1), "shout lane trace count mismatch");
        for trace in traces {
            assert_eq!(trace.has_lookup.len(), t, "shout trace length mismatch");
            write_shout_lane_columns(inst, trace, &mut mem_cols);
        }
    }
    for (inst, _, traces) in &mem_instances {
        assert_eq!(traces.len(), inst.lanes.max(1), "twist lane trace count mismatch");
        for trace in traces {
            assert_eq!(trace.steps, t, "twist trace length mismatch");
            write_twist_lane_columns(inst, trace, &mut mem_cols);
        }
    }

    let needed_m = m_in + (cpu_cols.len() + mem_cols.len()) * t;
    assert_eq!(
        ccs.m, needed_m,
        "CCS width mismatch: expected {needed_m}, got {}",
        ccs.m
    );

    let mut z = Vec::with_capacity(ccs.m);
    z.extend_from_slice(&public_inputs);
    z.extend(flatten_cols(&cpu_cols));
    z.extend(flatten_cols(&mem_cols));

    let (mcs, mcs_wit) = create_mcs_from_z(params, l, m_in, z);
    crate::common_setup::canonicalize_step_time_columns(StepWitnessBundle {
        mcs: (mcs, mcs_wit),
        lut_instances: shout_instances
            .into_iter()
            .map(|(inst, wit, _)| (inst, wit))
            .collect(),
        mem_instances: mem_instances
            .into_iter()
            .map(|(inst, wit, _)| (inst, wit))
            .collect(),
        time_columns: crate::common_setup::empty_time_columns(),
        _phantom: PhantomData::<K>,
    })
}
