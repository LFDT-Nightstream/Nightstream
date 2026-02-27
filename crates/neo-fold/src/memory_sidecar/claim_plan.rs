use neo_ajtai::Commitment as Cmt;
use neo_math::{F, K};
use neo_memory::riscv::lookups::{
    RiscvOpcode, POSEIDON2_ABSORB_FUNCT7, POSEIDON2_CUSTOM_OPCODE, POSEIDON2_FINALIZE_FUNCT7, POSEIDON2_SQUEEZE_FUNCT7,
    PROG_ID, REG_ID,
};
use neo_memory::riscv::trace::{rv32_is_decode_lookup_table_id, rv32_is_width_lookup_table_id};
use neo_memory::witness::{LutInstance, LutTableSpec, MemInstance, StepInstanceBundle, StepWitnessBundle};
use p3_field::PrimeField64;

use crate::memory_sidecar::memory::W2_FIELDS_DEGREE_BOUND;
use crate::PiCcsError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeClaimMeta {
    pub label: &'static [u8],
    pub degree_bound: usize,
    pub is_dynamic: bool,
}

pub const POSEIDON_CYCLE_CLAIM_METAS: [TimeClaimMeta; 9] = [
    TimeClaimMeta {
        label: b"poseidon/io_link",
        degree_bound: 4,
        is_dynamic: false,
    },
    TimeClaimMeta {
        label: b"poseidon/bitness",
        degree_bound: 3,
        is_dynamic: false,
    },
    TimeClaimMeta {
        label: b"poseidon/canonical_u64",
        degree_bound: 6,
        is_dynamic: false,
    },
    TimeClaimMeta {
        label: b"poseidon/sidecar_link",
        degree_bound: 4,
        is_dynamic: false,
    },
    TimeClaimMeta {
        label: b"poseidon/mode",
        degree_bound: 3,
        is_dynamic: false,
    },
    TimeClaimMeta {
        label: b"poseidon/link_cycle_inv",
        degree_bound: 4,
        is_dynamic: false,
    },
    TimeClaimMeta {
        label: b"poseidon/link_cycle_sum",
        degree_bound: 3,
        is_dynamic: true,
    },
    TimeClaimMeta {
        label: b"poseidon/cont_inv",
        degree_bound: 6,
        is_dynamic: false,
    },
    TimeClaimMeta {
        label: b"poseidon/cont_sum",
        degree_bound: 3,
        is_dynamic: false,
    },
];

pub const POSEIDON_LOCAL_TIME_CLAIM_METAS: [TimeClaimMeta; 5] = [
    TimeClaimMeta {
        label: b"poseidon/round",
        degree_bound: 10,
        is_dynamic: false,
    },
    TimeClaimMeta {
        label: b"poseidon/transition",
        degree_bound: 4,
        is_dynamic: false,
    },
    TimeClaimMeta {
        label: b"poseidon/cycle_local_link",
        degree_bound: 8,
        is_dynamic: false,
    },
    TimeClaimMeta {
        label: b"poseidon/link_local_inv",
        degree_bound: 5,
        is_dynamic: false,
    },
    TimeClaimMeta {
        label: b"poseidon/link_local_sum",
        degree_bound: 3,
        is_dynamic: true,
    },
];

#[inline]
pub fn poseidon_cycle_claim_metas() -> &'static [TimeClaimMeta] {
    &POSEIDON_CYCLE_CLAIM_METAS
}

#[inline]
pub fn poseidon_local_time_claim_metas() -> &'static [TimeClaimMeta] {
    &POSEIDON_LOCAL_TIME_CLAIM_METAS
}

#[derive(Clone, Debug)]
pub struct ShoutLaneTimeClaimIdx {
    pub value: Option<usize>,
    pub adapter: Option<usize>,
    pub event_table_hash: Option<usize>,
    pub gamma_group: Option<usize>,
    pub transport_only: bool,
}

#[derive(Clone, Debug)]
pub struct ShoutTimeClaimIdx {
    pub lanes: Vec<ShoutLaneTimeClaimIdx>,
    pub bitness: Option<usize>,
    pub ell_addr: usize,
    pub transport_only: bool,
}

#[derive(Clone, Debug)]
pub struct ShoutGammaGroupLaneRef {
    pub flat_lane_idx: usize,
    pub inst_idx: usize,
    pub lane_idx: usize,
}

#[derive(Clone, Debug)]
pub struct ShoutGammaGroupSpec {
    pub key: u64,
    pub ell_addr: usize,
    pub lanes: Vec<ShoutGammaGroupLaneRef>,
}

#[derive(Clone, Debug)]
pub struct ShoutGammaGroupTimeClaimIdx {
    pub key: u64,
    pub ell_addr: usize,
    pub lanes: Vec<ShoutGammaGroupLaneRef>,
    pub value: usize,
    pub adapter: usize,
    pub bitness: usize,
}

#[derive(Clone, Debug)]
pub struct TwistTimeClaimIdx {
    pub read_check: usize,
    pub write_check: usize,
    pub bitness: usize,
    pub virtual_write_domain: Option<usize>,
    pub nonvirtual_arch_domain: Option<usize>,
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
    pub shout_gamma_groups: Vec<ShoutGammaGroupTimeClaimIdx>,
    pub shout_event_trace_hash: Option<usize>,
    pub twist: Vec<TwistTimeClaimIdx>,
    pub wb_bool: Option<usize>,
    pub wp_quiescence: Option<usize>,
    pub decode_fields: Option<usize>,
    pub decode_immediates: Option<usize>,
    pub width_bitness: Option<usize>,
    pub width_quiescence: Option<usize>,
    pub width_selector_linkage: Option<usize>,
    pub width_load_semantics: Option<usize>,
    pub width_store_semantics: Option<usize>,
    pub control_next_pc_linear: Option<usize>,
    pub control_next_pc_control: Option<usize>,
    pub control_branch_semantics: Option<usize>,
    pub control_writeback: Option<usize>,
    pub poseidon_io_link: Option<usize>,
    pub poseidon_bitness: Option<usize>,
    pub poseidon_canonical_u64: Option<usize>,
    pub poseidon_sidecar_link: Option<usize>,
    pub poseidon_mode: Option<usize>,
    pub poseidon_link_cycle_inv: Option<usize>,
    pub poseidon_link_cycle_sum: Option<usize>,
    pub poseidon_cont_inv: Option<usize>,
    pub poseidon_cont_sum: Option<usize>,
}

impl RouteATimeClaimPlan {
    #[inline]
    pub(crate) fn route_a_transport_only_shout_table(table_id: u32) -> bool {
        rv32_is_decode_lookup_table_id(table_id) || rv32_is_width_lookup_table_id(table_id)
    }

    fn is_poseidon_precompile_word(word: u32) -> bool {
        let opcode = word & 0x7f;
        if opcode != POSEIDON2_CUSTOM_OPCODE {
            return false;
        }
        let rd = ((word >> 7) & 0x1f) as u8;
        let funct3 = ((word >> 12) & 0x07) as u8;
        let rs1 = ((word >> 15) & 0x1f) as u8;
        let rs2 = ((word >> 20) & 0x1f) as u8;
        let funct7 = ((word >> 25) & 0x7f) as u8;
        match funct7 as u32 {
            POSEIDON2_ABSORB_FUNCT7 => funct3 == 0 && rd == 0,
            POSEIDON2_FINALIZE_FUNCT7 => funct3 == 0 && rd == 0 && rs1 == 0 && rs2 == 0,
            POSEIDON2_SQUEEZE_FUNCT7 => rs1 == 0 && rs2 == 0,
            _ => false,
        }
    }

    fn poseidon_stage_required_for_mem_instances<'a, I>(mem_insts: I) -> Result<bool, PiCcsError>
    where
        I: IntoIterator<Item = &'a MemInstance<Cmt, F>>,
    {
        let prog_inst = mem_insts.into_iter().find(|inst| inst.mem_id == PROG_ID.0);
        let Some(prog_inst) = prog_inst else {
            return Ok(false);
        };

        match &prog_inst.init {
            neo_memory::MemInit::Zero => Ok(false),
            neo_memory::MemInit::Sparse(pairs) => {
                for (addr, value) in pairs.iter() {
                    let word_u64 = value.as_canonical_u64();
                    if word_u64 > u32::MAX as u64 {
                        return Err(PiCcsError::ProtocolError(format!(
                            "poseidon stage probe: PROG init word does not fit u32 (addr={addr}, value={word_u64:#x})"
                        )));
                    }
                    let word = word_u64 as u32;
                    if Self::is_poseidon_precompile_word(word) {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
        }
    }

    pub fn poseidon_stage_required_for_step_instance(step: &StepInstanceBundle<Cmt, F, K>) -> Result<bool, PiCcsError> {
        Self::poseidon_stage_required_for_mem_instances(step.mem_insts.iter())
    }

    pub fn poseidon_stage_required_for_step_witness(step: &StepWitnessBundle<Cmt, F, K>) -> Result<bool, PiCcsError> {
        Self::poseidon_stage_required_for_mem_instances(step.mem_instances.iter().map(|(inst, _)| inst))
    }

    pub fn derive_shout_gamma_groups_for_instances<'a, LI>(lut_insts: LI) -> Vec<ShoutGammaGroupSpec>
    where
        LI: IntoIterator<Item = &'a LutInstance<Cmt, F>>,
    {
        let lut_insts: Vec<&LutInstance<Cmt, F>> = lut_insts.into_iter().collect();

        // Group all non-packed lookup families that share an address group.
        // The addr_group is carried on each LutInstance (set by the bus config for trace mode,
        // None when no lookup-family sharing is configured). This collapses per-column decode/width families into one
        // gamma-batched claim pair while keeping packed/event-table specs on their existing
        // per-lane schedule.
        let mut grouped: std::collections::BTreeMap<u64, Vec<ShoutGammaGroupLaneRef>> =
            std::collections::BTreeMap::new();
        let mut grouped_ell: std::collections::BTreeMap<u64, usize> = std::collections::BTreeMap::new();

        let mut flat_lane_idx = 0usize;
        for (inst_idx, lut_inst) in lut_insts.iter().enumerate() {
            if Self::route_a_transport_only_shout_table(lut_inst.table_id) {
                flat_lane_idx += lut_inst.lanes.max(1);
                continue;
            }
            let lanes = lut_inst.lanes.max(1);
            let ell_addr = lut_inst.d * lut_inst.ell;
            let is_packed = matches!(
                lut_inst.table_spec,
                Some(LutTableSpec::RiscvOpcodePacked { .. } | LutTableSpec::RiscvOpcodeEventTablePacked { .. })
            );
            let is_gamma_candidate = !is_packed && lut_inst.addr_group.is_some();
            for lane_idx in 0..lanes {
                if is_gamma_candidate {
                    if let Some(addr_group) = lut_inst.addr_group {
                        let key = (addr_group << 32) | lane_idx as u64;
                        grouped
                            .entry(key)
                            .or_default()
                            .push(ShoutGammaGroupLaneRef {
                                flat_lane_idx,
                                inst_idx,
                                lane_idx,
                            });
                        grouped_ell.entry(key).or_insert(ell_addr);
                    }
                }
                flat_lane_idx += 1;
            }
        }

        let mut out = Vec::new();
        for (key, lanes) in grouped.into_iter() {
            if lanes.len() <= 1 {
                continue;
            }
            if let Some(&ell_addr) = grouped_ell.get(&key) {
                out.push(ShoutGammaGroupSpec { key, ell_addr, lanes });
            }
        }
        out
    }

    pub fn time_claim_metas_for_instances<'a, LI, MI>(
        lut_insts: LI,
        mem_insts: MI,
        wb_enabled: bool,
        wp_enabled: bool,
        decode_stage_enabled: bool,
        width_stage_enabled: bool,
        control_stage_enabled: bool,
        poseidon_cycle_enabled: bool,
        ob_inc_total_degree_bound: Option<usize>,
    ) -> Vec<TimeClaimMeta>
    where
        LI: IntoIterator<Item = &'a LutInstance<Cmt, F>>,
        MI: IntoIterator<Item = &'a MemInstance<Cmt, F>>,
    {
        let lut_insts: Vec<&LutInstance<Cmt, F>> = lut_insts.into_iter().collect();
        let mem_insts: Vec<&MemInstance<Cmt, F>> = mem_insts.into_iter().collect();
        let shout_gamma_groups = Self::derive_shout_gamma_groups_for_instances(lut_insts.iter().copied());
        let mut lane_gamma_map: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
        for (g_idx, g) in shout_gamma_groups.iter().enumerate() {
            for lane in g.lanes.iter() {
                lane_gamma_map.insert((lane.inst_idx, lane.lane_idx), g_idx);
            }
        }
        let any_event_table_shout = lut_insts
            .iter()
            .any(|inst| matches!(inst.table_spec, Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. })));

        let mut out = Vec::new();
        let mut gamma_value_degree_bounds = vec![0usize; shout_gamma_groups.len()];
        let mut gamma_adapter_degree_bounds = vec![0usize; shout_gamma_groups.len()];

        for (inst_idx, lut_inst) in lut_insts.iter().enumerate() {
            if Self::route_a_transport_only_shout_table(lut_inst.table_id) {
                continue;
            }
            let ell_addr = lut_inst.d * lut_inst.ell;
            let lanes = lut_inst.lanes.max(1);
            let (packed_opcode, _packed_base_ell_addr) = match &lut_inst.table_spec {
                Some(LutTableSpec::RiscvOpcodePacked { opcode, xlen: 32 }) => (Some(*opcode), ell_addr),
                Some(LutTableSpec::RiscvOpcodeEventTablePacked {
                    opcode,
                    xlen: 32,
                    time_bits,
                }) => (Some(*opcode), ell_addr.saturating_sub(*time_bits)),
                _ => (None, ell_addr),
            };

            let (value_degree_bound, adapter_degree_bound) = match packed_opcode {
                Some(RiscvOpcode::And | RiscvOpcode::Andn | RiscvOpcode::Or | RiscvOpcode::Xor) => (8, 6),
                Some(RiscvOpcode::Add | RiscvOpcode::Sub) => (3, 2),
                Some(RiscvOpcode::Eq | RiscvOpcode::Neq) => (34, 3),
                Some(RiscvOpcode::Mul) => (4, 2),
                Some(RiscvOpcode::Mulh) => (4, 5),
                Some(RiscvOpcode::Mulhu) => (4, 2),
                Some(RiscvOpcode::Mulhsu) => (4, 4),
                Some(RiscvOpcode::Slt) => (3, 3),
                Some(RiscvOpcode::Divu | RiscvOpcode::Remu) => (5, 4),
                Some(RiscvOpcode::Div | RiscvOpcode::Rem) => (7, 6),
                Some(RiscvOpcode::Sll) => (8, 2),
                Some(RiscvOpcode::Srl | RiscvOpcode::Sra) => (8, 8),
                Some(RiscvOpcode::Sltu) => (3, 3),
                _ => (3, 2 + ell_addr),
            };

            let mut has_ungrouped_lane = false;
            for lane_idx in 0..lanes {
                if let Some(&g_idx) = lane_gamma_map.get(&(inst_idx, lane_idx)) {
                    gamma_value_degree_bounds[g_idx] = gamma_value_degree_bounds[g_idx].max(value_degree_bound);
                    gamma_adapter_degree_bounds[g_idx] = gamma_adapter_degree_bounds[g_idx].max(adapter_degree_bound);
                } else {
                    has_ungrouped_lane = true;
                    out.push(TimeClaimMeta {
                        label: b"shout/value",
                        degree_bound: value_degree_bound,
                        is_dynamic: true,
                    });
                    out.push(TimeClaimMeta {
                        label: b"shout/adapter",
                        degree_bound: adapter_degree_bound,
                        is_dynamic: true,
                    });
                }
                if let Some(LutTableSpec::RiscvOpcodeEventTablePacked { time_bits, .. }) = &lut_inst.table_spec {
                    out.push(TimeClaimMeta {
                        label: b"shout/event_table_hash",
                        degree_bound: 2 + *time_bits,
                        is_dynamic: true,
                    });
                }
            }

            if has_ungrouped_lane {
                out.push(TimeClaimMeta {
                    label: b"shout/bitness",
                    degree_bound: 3,
                    is_dynamic: false,
                });
            }
        }

        for (g_idx, _) in shout_gamma_groups.iter().enumerate() {
            out.push(TimeClaimMeta {
                label: b"shout/value",
                degree_bound: gamma_value_degree_bounds[g_idx],
                is_dynamic: true,
            });
            out.push(TimeClaimMeta {
                label: b"shout/adapter",
                degree_bound: gamma_adapter_degree_bounds[g_idx],
                is_dynamic: true,
            });
            out.push(TimeClaimMeta {
                label: b"shout/bitness",
                degree_bound: 3,
                is_dynamic: false,
            });
        }

        if any_event_table_shout {
            out.push(TimeClaimMeta {
                label: b"shout/event_trace_hash",
                degree_bound: 3,
                is_dynamic: true,
            });
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

            out.push(TimeClaimMeta {
                label: b"twist/bitness",
                degree_bound: 3,
                is_dynamic: false,
            });
            if decode_stage_enabled && mem_inst.mem_id == REG_ID.0 {
                out.push(TimeClaimMeta {
                    label: b"twist/virtual_write_domain",
                    degree_bound: 4,
                    is_dynamic: false,
                });
                out.push(TimeClaimMeta {
                    label: b"twist/nonvirtual_arch_domain",
                    degree_bound: 4,
                    is_dynamic: false,
                });
            }
        }

        if wb_enabled {
            out.push(TimeClaimMeta {
                label: b"wb/booleanity",
                degree_bound: 3,
                is_dynamic: false,
            });
        }

        if wp_enabled {
            out.push(TimeClaimMeta {
                label: b"wp/quiescence",
                degree_bound: 3,
                is_dynamic: false,
            });
        }

        if decode_stage_enabled {
            out.push(TimeClaimMeta {
                label: b"decode/fields",
                degree_bound: W2_FIELDS_DEGREE_BOUND,
                is_dynamic: false,
            });
            out.push(TimeClaimMeta {
                label: b"decode/immediates",
                degree_bound: 3,
                is_dynamic: false,
            });
        }

        if width_stage_enabled {
            out.push(TimeClaimMeta {
                label: b"width/bitness",
                degree_bound: 3,
                is_dynamic: false,
            });
            out.push(TimeClaimMeta {
                label: b"width/quiescence",
                degree_bound: 3,
                is_dynamic: false,
            });
            out.push(TimeClaimMeta {
                label: b"width/load_semantics",
                degree_bound: 5,
                is_dynamic: false,
            });
            out.push(TimeClaimMeta {
                label: b"width/store_semantics",
                degree_bound: 4,
                is_dynamic: false,
            });
        }

        if control_stage_enabled {
            out.push(TimeClaimMeta {
                label: b"control/next_pc_linear",
                degree_bound: 4,
                is_dynamic: false,
            });
            out.push(TimeClaimMeta {
                label: b"control/next_pc_control",
                degree_bound: 5,
                is_dynamic: false,
            });
            out.push(TimeClaimMeta {
                label: b"control/branch_semantics",
                degree_bound: 4,
                is_dynamic: false,
            });
            out.push(TimeClaimMeta {
                label: b"control/writeback",
                degree_bound: 5,
                is_dynamic: false,
            });
        }

        if poseidon_cycle_enabled {
            out.extend_from_slice(&POSEIDON_CYCLE_CLAIM_METAS);
        }

        if let Some(degree_bound) = ob_inc_total_degree_bound {
            out.push(TimeClaimMeta {
                label: crate::output_binding::OB_INC_TOTAL_LABEL,
                degree_bound,
                is_dynamic: true,
            });
        }

        out
    }

    /// Returns the full ordered metadata list for the Route A batched-time sumcheck.
    ///
    /// This is a single source of truth for claim ordering and expected degree bounds/labels.
    /// Claim indices returned by [`RouteATimeClaimPlan::build`] refer to the memory-only suffix
    /// of this list, starting at `claim_idx_start` (typically 0).
    pub fn time_claim_metas_for_step(
        step: &StepInstanceBundle<Cmt, F, K>,
        wb_enabled: bool,
        wp_enabled: bool,
        decode_stage_enabled: bool,
        width_stage_enabled: bool,
        control_stage_enabled: bool,
        poseidon_cycle_enabled: bool,
        ob_inc_total_degree_bound: Option<usize>,
    ) -> Vec<TimeClaimMeta> {
        Self::time_claim_metas_for_instances(
            step.lut_insts.iter(),
            step.mem_insts.iter(),
            wb_enabled,
            wp_enabled,
            decode_stage_enabled,
            width_stage_enabled,
            control_stage_enabled,
            poseidon_cycle_enabled,
            ob_inc_total_degree_bound,
        )
    }

    pub fn build(
        step: &StepInstanceBundle<Cmt, F, K>,
        claim_idx_start: usize,
        wb_enabled: bool,
        wp_enabled: bool,
        decode_stage_enabled: bool,
        width_stage_enabled: bool,
        control_stage_enabled: bool,
        poseidon_cycle_enabled: bool,
    ) -> Result<RouteATimeClaimPlan, PiCcsError> {
        let mut idx = claim_idx_start;
        let mut shout = Vec::with_capacity(step.lut_insts.len());
        let shout_gamma_specs = Self::derive_shout_gamma_groups_for_instances(step.lut_insts.iter());
        let mut lane_gamma_map: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
        for (g_idx, g) in shout_gamma_specs.iter().enumerate() {
            for lane in g.lanes.iter() {
                lane_gamma_map.insert((lane.inst_idx, lane.lane_idx), g_idx);
            }
        }
        let any_event_table_shout = step
            .lut_insts
            .iter()
            .filter(|inst| !Self::route_a_transport_only_shout_table(inst.table_id))
            .any(|inst| matches!(inst.table_spec, Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. })));
        let mut twist = Vec::with_capacity(step.mem_insts.len());

        for (inst_idx, lut_inst) in step.lut_insts.iter().enumerate() {
            let transport_only = Self::route_a_transport_only_shout_table(lut_inst.table_id);
            let ell_addr = lut_inst.d * lut_inst.ell;
            let lanes = lut_inst.lanes.max(1);
            let is_event_table = matches!(
                lut_inst.table_spec,
                Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. })
            );
            let mut lane_claims: Vec<ShoutLaneTimeClaimIdx> = Vec::with_capacity(lanes);
            let mut has_ungrouped_lane = false;
            for lane_idx in 0..lanes {
                let gamma_group = if transport_only {
                    None
                } else {
                    lane_gamma_map.get(&(inst_idx, lane_idx)).copied()
                };
                let (value, adapter) = if transport_only {
                    (None, None)
                } else if gamma_group.is_some() {
                    (None, None)
                } else {
                    has_ungrouped_lane = true;
                    let value = idx;
                    idx += 1;
                    let adapter = idx;
                    idx += 1;
                    (Some(value), Some(adapter))
                };
                let event_table_hash = if transport_only {
                    None
                } else if is_event_table {
                    let h = idx;
                    idx += 1;
                    Some(h)
                } else {
                    None
                };
                lane_claims.push(ShoutLaneTimeClaimIdx {
                    value,
                    adapter,
                    event_table_hash,
                    gamma_group,
                    transport_only,
                });
            }
            let bitness = if transport_only {
                None
            } else if has_ungrouped_lane {
                let out = idx;
                idx += 1;
                Some(out)
            } else {
                None
            };

            shout.push(ShoutTimeClaimIdx {
                lanes: lane_claims,
                bitness,
                ell_addr,
                transport_only,
            });
        }

        let mut shout_gamma_groups = Vec::with_capacity(shout_gamma_specs.len());
        for spec in shout_gamma_specs.into_iter() {
            let value = idx;
            idx += 1;
            let adapter = idx;
            idx += 1;
            let bitness = idx;
            idx += 1;
            shout_gamma_groups.push(ShoutGammaGroupTimeClaimIdx {
                key: spec.key,
                ell_addr: spec.ell_addr,
                lanes: spec.lanes,
                value,
                adapter,
                bitness,
            });
        }

        let shout_event_trace_hash = if any_event_table_shout {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };

        for mem_inst in &step.mem_insts {
            let ell_addr = mem_inst.d * mem_inst.ell;
            let read_check = idx;
            idx += 1;
            let write_check = idx;
            idx += 1;

            let bitness = idx;
            idx += 1;
            let virtual_write_domain = if decode_stage_enabled && mem_inst.mem_id == REG_ID.0 {
                let out = idx;
                idx += 1;
                Some(out)
            } else {
                None
            };
            let nonvirtual_arch_domain = if decode_stage_enabled && mem_inst.mem_id == REG_ID.0 {
                let out = idx;
                idx += 1;
                Some(out)
            } else {
                None
            };

            twist.push(TwistTimeClaimIdx {
                read_check,
                write_check,
                bitness,
                virtual_write_domain,
                nonvirtual_arch_domain,
                ell_addr,
            });
        }

        let wb_bool = if wb_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };

        let wp_quiescence = if wp_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };

        let decode_fields = if decode_stage_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };

        let decode_immediates = if decode_stage_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };

        let width_bitness = if width_stage_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };

        let width_quiescence = if width_stage_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };

        let width_selector_linkage = None;

        let width_load_semantics = if width_stage_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };

        let width_store_semantics = if width_stage_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };

        let control_next_pc_linear = if control_stage_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };

        let control_next_pc_control = if control_stage_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };

        let control_branch_semantics = if control_stage_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };

        let control_writeback = if control_stage_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };

        let poseidon_io_link = if poseidon_cycle_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };
        let poseidon_bitness = if poseidon_cycle_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };
        let poseidon_canonical_u64 = if poseidon_cycle_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };
        let poseidon_sidecar_link = if poseidon_cycle_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };
        let poseidon_mode = if poseidon_cycle_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };
        let poseidon_link_cycle_inv = if poseidon_cycle_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };
        let poseidon_link_cycle_sum = if poseidon_cycle_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };
        let poseidon_cont_inv = if poseidon_cycle_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };
        let poseidon_cont_sum = if poseidon_cycle_enabled {
            let out = idx;
            idx += 1;
            Some(out)
        } else {
            None
        };

        if idx < claim_idx_start {
            return Err(PiCcsError::ProtocolError("RouteATimeClaimPlan index underflow".into()));
        }

        Ok(RouteATimeClaimPlan {
            claim_idx_start,
            claim_idx_end: idx,
            shout,
            shout_gamma_groups,
            shout_event_trace_hash,
            twist,
            wb_bool,
            wp_quiescence,
            decode_fields,
            decode_immediates,
            width_bitness,
            width_quiescence,
            width_selector_linkage,
            width_load_semantics,
            width_store_semantics,
            control_next_pc_linear,
            control_next_pc_control,
            control_branch_semantics,
            control_writeback,
            poseidon_io_link,
            poseidon_bitness,
            poseidon_canonical_u64,
            poseidon_sidecar_link,
            poseidon_mode,
            poseidon_link_cycle_inv,
            poseidon_link_cycle_sum,
            poseidon_cont_inv,
            poseidon_cont_sum,
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
