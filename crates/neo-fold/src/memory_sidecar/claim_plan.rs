use core::ops::Range;

use neo_ajtai::Commitment as Cmt;
use neo_math::{F, K};
use neo_memory::witness::{LutInstance, MemInstance, StepInstanceBundle};

use crate::PiCcsError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimeClaimMeta {
    pub label: &'static [u8],
    pub degree_bound: usize,
    pub is_dynamic: bool,
}

#[derive(Clone, Debug)]
pub struct ShoutTimeClaimIdx {
    pub value: usize,
    pub adapter: usize,
    pub bitness_addr_bits: Range<usize>,
    pub bitness_has_lookup: usize,
    pub ell_addr: usize,
}

#[derive(Clone, Debug)]
pub struct TwistTimeClaimIdx {
    pub read_check: usize,
    pub write_check: usize,
    pub bitness_ra_bits: Range<usize>,
    pub bitness_wa_bits: Range<usize>,
    pub bitness_has_read: usize,
    pub bitness_has_write: usize,
    pub ell_addr: usize,
}

/// Deterministic claim schedule for Route A batched time claims (memory sidecar only).
///
/// This is a single source of truth for how indices into `batched_claimed_sums` /
/// `batched_final_values` map to each Shout/Twist instance.
#[derive(Clone, Debug)]
pub struct RouteATimeClaimPlan {
    pub claim_idx_start: usize,
    pub claim_idx_end: usize,
    pub shout: Vec<ShoutTimeClaimIdx>,
    pub twist: Vec<TwistTimeClaimIdx>,
}

impl RouteATimeClaimPlan {
    pub fn time_claim_metas_for_instances<'a, LI, MI>(
        lut_insts: LI,
        mem_insts: MI,
        ccs_time_degree_bound: usize,
    ) -> Vec<TimeClaimMeta>
    where
        LI: IntoIterator<Item = &'a LutInstance<Cmt, F>>,
        MI: IntoIterator<Item = &'a MemInstance<Cmt, F>>,
    {
        let mut out = Vec::new();

        out.push(TimeClaimMeta {
            label: b"ccs/time",
            degree_bound: ccs_time_degree_bound,
            is_dynamic: true,
        });

        for lut_inst in lut_insts {
            let ell_addr = lut_inst.d * lut_inst.ell;

            out.push(TimeClaimMeta {
                label: b"shout/value",
                degree_bound: 3,
                is_dynamic: true,
            });
            out.push(TimeClaimMeta {
                label: b"shout/adapter",
                degree_bound: 2 + ell_addr,
                is_dynamic: true,
            });

            for _ in 0..(ell_addr + 1) {
                out.push(TimeClaimMeta {
                    label: b"shout/bitness",
                    degree_bound: 3,
                    is_dynamic: false,
                });
            }
        }

        for mem_inst in mem_insts {
            let ell_addr = mem_inst.d * mem_inst.ell;

            out.push(TimeClaimMeta {
                label: b"twist/read_check",
                degree_bound: 3 + ell_addr,
                is_dynamic: true,
            });
            out.push(TimeClaimMeta {
                label: b"twist/write_check",
                degree_bound: 3 + ell_addr,
                is_dynamic: true,
            });

            for _ in 0..(2 * ell_addr + 2) {
                out.push(TimeClaimMeta {
                    label: b"twist/bitness",
                    degree_bound: 3,
                    is_dynamic: false,
                });
            }
        }

        out
    }

    /// Returns the full ordered metadata list for the Route A batched-time sumcheck.
    ///
    /// This is a single source of truth for claim ordering and expected degree bounds/labels.
    /// Claim indices returned by [`RouteATimeClaimPlan::build`] refer to the memory-only suffix
    /// of this list, starting at `claim_idx_start` (typically 1, after `ccs/time`).
    pub fn time_claim_metas_for_step(
        step: &StepInstanceBundle<Cmt, F, K>,
        ccs_time_degree_bound: usize,
    ) -> Vec<TimeClaimMeta> {
        Self::time_claim_metas_for_instances(step.lut_insts.iter(), step.mem_insts.iter(), ccs_time_degree_bound)
    }

    pub fn build(
        step: &StepInstanceBundle<Cmt, F, K>,
        claim_idx_start: usize,
    ) -> Result<RouteATimeClaimPlan, PiCcsError> {
        let mut idx = claim_idx_start;
        let mut shout = Vec::with_capacity(step.lut_insts.len());
        let mut twist = Vec::with_capacity(step.mem_insts.len());

        for lut_inst in &step.lut_insts {
            let ell_addr = lut_inst.d * lut_inst.ell;
            let value = idx;
            idx += 1;
            let adapter = idx;
            idx += 1;
            let bitness_addr_bits = idx..(idx + ell_addr);
            idx += ell_addr;
            let bitness_has_lookup = idx;
            idx += 1;

            shout.push(ShoutTimeClaimIdx {
                value,
                adapter,
                bitness_addr_bits,
                bitness_has_lookup,
                ell_addr,
            });
        }

        for mem_inst in &step.mem_insts {
            let ell_addr = mem_inst.d * mem_inst.ell;
            let read_check = idx;
            idx += 1;
            let write_check = idx;
            idx += 1;

            let bitness_ra_bits = idx..(idx + ell_addr);
            idx += ell_addr;
            let bitness_wa_bits = idx..(idx + ell_addr);
            idx += ell_addr;
            let bitness_has_read = idx;
            idx += 1;
            let bitness_has_write = idx;
            idx += 1;

            twist.push(TwistTimeClaimIdx {
                read_check,
                write_check,
                bitness_ra_bits,
                bitness_wa_bits,
                bitness_has_read,
                bitness_has_write,
                ell_addr,
            });
        }

        if idx < claim_idx_start {
            return Err(PiCcsError::ProtocolError("RouteATimeClaimPlan index underflow".into()));
        }

        Ok(RouteATimeClaimPlan {
            claim_idx_start,
            claim_idx_end: idx,
            shout,
            twist,
        })
    }
}

#[derive(Clone, Debug)]
pub struct TwistValEvalClaimPlan {
    pub has_prev: bool,
    pub claims_per_mem: usize,
    pub claim_count: usize,
    pub labels: Vec<&'static [u8]>,
    pub degree_bounds: Vec<usize>,
    pub bind_tags: Vec<u8>,
}

impl TwistValEvalClaimPlan {
    pub fn build<'a, I>(mem_insts: I, has_prev: bool) -> Self
    where
        I: IntoIterator<Item = &'a MemInstance<Cmt, F>>,
    {
        let mem_insts: Vec<&MemInstance<Cmt, F>> = mem_insts.into_iter().collect();
        let n_mem = mem_insts.len();
        let claims_per_mem = if has_prev { 3 } else { 2 };
        let claim_count = claims_per_mem * n_mem;

        let mut labels: Vec<&'static [u8]> = Vec::with_capacity(claim_count);
        let mut degree_bounds = Vec::with_capacity(claim_count);
        let mut bind_tags = Vec::with_capacity(claim_count);

        for inst in mem_insts {
            let ell_addr = inst.d * inst.ell;

            labels.push(b"twist/val_eval_lt".as_slice());
            degree_bounds.push(ell_addr + 3);
            bind_tags.push(0);

            labels.push(b"twist/val_eval_total".as_slice());
            degree_bounds.push(ell_addr + 2);
            bind_tags.push(1);

            if has_prev {
                labels.push(b"twist/rollover_prev_total".as_slice());
                degree_bounds.push(ell_addr + 2);
                bind_tags.push(2);
            }
        }

        Self {
            has_prev,
            claims_per_mem,
            claim_count,
            labels,
            degree_bounds,
            bind_tags,
        }
    }

    #[inline]
    pub fn base(&self, mem_idx: usize) -> usize {
        self.claims_per_mem * mem_idx
    }
}
