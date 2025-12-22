use crate::builder::CpuArithmetization;
use crate::plain::LutTable;
use neo_ajtai::{decomp_b, DecompStyle};
use neo_ccs::matrix::Mat;
use neo_ccs::relations::{CcsStructure, McsInstance, McsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_params::NeoParams;
use neo_vm_trace::{StepTrace, VmTrace};
use p3_field::{PrimeCharacteristicRing, PrimeField, PrimeField64};
use p3_goldilocks::Goldilocks;
use std::collections::HashMap;
use std::marker::PhantomData;

/// Adapter that implements CpuArithmetization for a generic R1CS-based CPU.
///
/// It assumes the user provides:
/// 1. The CCS structure (wrapped R1CS matrices).
/// 2. A witness builder function that maps a `StepTrace` to the R1CS witness vector `w`.
/// 3. Shout tables to verify trace correctness.
pub struct R1csCpu<F, Cmt, L>
where
    F: PrimeField + PrimeField64,
    L: SModuleHomomorphism<F, Cmt>,
{
    pub ccs: CcsStructure<F>,
    pub params: NeoParams,
    pub committer: L,

    /// Number of public inputs m_in (prefix of z treated as x).
    /// The witness z is split as z = (x[0..m_in], w[m_in..]).
    pub m_in: usize,

    /// Cached Shout tables for verification: shout_id -> (key -> val)
    /// Used to ensure trace Shout lookups are valid against the table.
    pub shout_cache: HashMap<u32, HashMap<u64, F>>,

    /// Function to map a step trace to the full witness z = (x, w).
    /// The witness MUST satisfy the CCS relation.
    pub step_to_witness: Box<dyn Fn(&StepTrace<u64, u64>) -> Vec<F> + Send + Sync>,

    _phantom: PhantomData<Cmt>,
}

impl<F, Cmt, L> R1csCpu<F, Cmt, L>
where
    F: PrimeField + PrimeField64,
    L: SModuleHomomorphism<F, Cmt>,
{
    pub fn new(
        ccs: CcsStructure<F>,
        params: NeoParams,
        committer: L,
        m_in: usize,
        tables: &HashMap<u32, LutTable<F>>,
        step_to_witness: Box<dyn Fn(&StepTrace<u64, u64>) -> Vec<F> + Send + Sync>,
    ) -> Self {
        // Build fast Shout cache from tables
        let mut shout_cache = HashMap::new();
        for (id, table) in tables {
            // Assuming table content is dense (val = content[addr])
            // LutTable currently has `content: Vec<F>`.
            // We'll map key -> val.
            let mut map = HashMap::new();
            for (key, val) in table.content.iter().enumerate() {
                map.insert(key as u64, *val);
            }
            shout_cache.insert(*id, map);
        }

        Self {
            ccs,
            params,
            committer,
            m_in,
            shout_cache,
            step_to_witness,
            _phantom: PhantomData,
        }
    }
}

// R1csCpu implementation specifically for Goldilocks field because neo_ajtai::decomp_b uses Goldilocks
// The trait impl needs to be generic F, but decomp_b expects &[Fq] (Goldilocks).
// We need to cast F to Goldilocks if possible, or restrict F to Goldilocks.
// Given neo-ajtai is hardcoded to Goldilocks (Fq), we should restrict here.

impl<Cmt, L> CpuArithmetization<Goldilocks, Cmt> for R1csCpu<Goldilocks, Cmt, L>
where
    L: SModuleHomomorphism<Goldilocks, Cmt>,
{
    fn build_ccs_chunks(
        &self,
        trace: &VmTrace<u64, u64>,
        chunk_size: usize,
    ) -> Result<Vec<(McsInstance<Cmt, Goldilocks>, McsWitness<Goldilocks>)>, Self::Error> {
        if chunk_size != 1 {
            return Err(format!(
                "R1csCpu does not support chunk_size={} (expected 1)",
                chunk_size
            ));
        }
        self.build_ccs_steps(trace)
    }

    fn build_ccs_steps(
        &self,
        trace: &VmTrace<u64, u64>,
    ) -> Result<Vec<(McsInstance<Cmt, Goldilocks>, McsWitness<Goldilocks>)>, Self::Error> {
        let mut mcss = Vec::with_capacity(trace.steps.len());

        for step in &trace.steps {
            // 1. Verify Shout lookups in this step against the table cache
            for shout in &step.shout_events {
                if let Some(table) = self.shout_cache.get(&shout.shout_id.0) {
                    // Check val == table[key]
                    if let Some(&expected) = table.get(&shout.key) {
                        let trace_val = Goldilocks::from_u64(shout.value);
                        if trace_val != expected {
                            return Err(format!(
                                "Shout mismatch at step pc={:?}: shout {} key {} has {}, trace has {}",
                                step.pc_before,
                                shout.shout_id.0,
                                shout.key,
                                expected.as_canonical_u64(),
                                trace_val.as_canonical_u64()
                            ));
                        }
                    } else {
                        return Err(format!(
                            "Shout key {} not found in shout {}",
                            shout.key, shout.shout_id.0
                        ));
                    }
                } else {
                    return Err(format!("Missing Shout table for shout_id {}", shout.shout_id.0));
                }
            }

            // 2. Build witness z
            let z_vec = (self.step_to_witness)(step);

            // 3. Decompose z -> Z matrix
            let d = self.params.d as usize;
            let m = z_vec.len(); // This must match ccs.m

            if m != self.ccs.m {
                return Err(format!("Witness length {} does not match CCS width {}", m, self.ccs.m));
            }

            // Validate m_in
            let m_in = self.m_in;
            if m_in > m {
                return Err(format!("m_in={} exceeds witness length m={}", m_in, m));
            }

            // Decompose: Z is d x m
            let z_digits = decomp_b(&z_vec, self.params.b, d, DecompStyle::Balanced);

            // Convert to Mat (row-major d x m)
            // decomp_b returns digits "per element", i.e. column-major for the (d × m) matrix:
            // z_digits[c*d + r] = digit r (row) of value c (column).
            let mut mat_data = vec![Goldilocks::ZERO; d * m];
            for c in 0..m {
                for r in 0..d {
                    mat_data[r * m + c] = z_digits[c * d + r];
                }
            }
            let z_mat = Mat::from_row_major(d, m, mat_data);

            // 4. Commit to Z
            let c = self.committer.commit(&z_mat);

            // 5. Build Instance/Witness
            // Split z into public inputs x and private witness w
            let x = z_vec[..m_in].to_vec();
            let w = z_vec[m_in..].to_vec();

            mcss.push((McsInstance { c, x, m_in }, McsWitness { w, Z: z_mat }));
        }

        Ok(mcss)
    }
    type Error = String;
}
