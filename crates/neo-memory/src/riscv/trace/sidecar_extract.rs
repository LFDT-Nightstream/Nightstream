use std::collections::HashMap;

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use crate::riscv::exec_table::Rv32ExecTable;
use crate::riscv::instruction::{encode_lookup_key, operand_mode_keys_enabled, try_decode_lookup_operands};
use crate::riscv::lookups::{RiscvOpcode, RiscvShoutTables};

#[derive(Clone, Debug)]
pub struct TwistLaneOverTime {
    pub has_read: Vec<bool>,
    pub ra: Vec<u64>,
    pub rv: Vec<u64>,
    pub has_write: Vec<bool>,
    pub wa: Vec<u64>,
    pub wv: Vec<u64>,
    pub inc_at_write_addr: Vec<F>,
}

impl TwistLaneOverTime {
    fn new_zero(t: usize) -> Self {
        Self {
            has_read: vec![false; t],
            ra: vec![0; t],
            rv: vec![0; t],
            has_write: vec![false; t],
            wa: vec![0; t],
            wv: vec![0; t],
            inc_at_write_addr: vec![F::ZERO; t],
        }
    }
}

#[derive(Clone, Debug)]
pub struct TraceTwistLanesOverTime {
    pub prog: TwistLaneOverTime,
    pub reg_lane0: TwistLaneOverTime,
    pub reg_lane1: TwistLaneOverTime,
    pub ram: TwistLaneOverTime,
}

#[derive(Clone, Debug)]
pub struct ShoutLaneOverTime {
    pub has_lookup: Vec<bool>,
    pub key: Vec<u128>,
    pub value: Vec<u64>,
}

impl ShoutLaneOverTime {
    fn new_zero(t: usize) -> Self {
        Self {
            has_lookup: vec![false; t],
            key: vec![0; t],
            value: vec![0; t],
        }
    }
}

/// Extract fixed-lane Twist-style memories over time from `Rv32ExecTable`.
///
/// Layout/policy:
/// - PROG: lane0 read-only (exactly one read per active row)
/// - REG: lane0 reads rs1 + optional write rd, lane1 reads rs2 (exactly one read per active row)
/// - RAM: lane0 supports at most 1 read + 1 write per active row, both must share the same addr
///
/// `init_regs` and `init_ram` are used to compute `inc_at_write_addr` and to sanity-check read values.
pub fn extract_twist_lanes_over_time(
    exec: &Rv32ExecTable,
    init_regs: &HashMap<u64, u64>,
    init_ram: &HashMap<u64, u64>,
    ram_ell_addr: usize,
) -> Result<TraceTwistLanesOverTime, String> {
    let t = exec.rows.len();

    // Build REG state for `inc_at_write_addr`.
    let mut regs: HashMap<u64, u64> = HashMap::new();
    for (&addr, &value) in init_regs {
        if addr == 0 && value != 0 {
            return Err("trace extract: reg init must keep x0 == 0".into());
        }
        if value != 0 {
            regs.insert(addr, value);
        }
    }

    // Build RAM state for `inc_at_write_addr` and read-value checks.
    if ram_ell_addr > 64 {
        return Err(format!(
            "trace extract: RAM ell_addr too large for u64 addressing: ell_addr={ram_ell_addr}"
        ));
    }
    let mut ram: HashMap<u64, u64> = HashMap::new();
    for (&addr, &value) in init_ram {
        if ram_ell_addr < 64 && (addr >> ram_ell_addr) != 0 {
            return Err(format!(
                "trace extract: RAM init addr out of range for ell_addr={ram_ell_addr}: addr={addr}"
            ));
        }
        if value != 0 {
            ram.insert(addr, value);
        }
    }

    let mut prog = TwistLaneOverTime::new_zero(t);
    let mut reg0 = TwistLaneOverTime::new_zero(t);
    let mut reg1 = TwistLaneOverTime::new_zero(t);
    let mut ram_lane = TwistLaneOverTime::new_zero(t);

    for (row_idx, r) in exec.rows.iter().enumerate() {
        if !r.active {
            if r.prog_read.is_some()
                || r.reg_read_lane0.is_some()
                || r.reg_read_lane1.is_some()
                || r.reg_write_lane0.is_some()
                || !r.ram_events.is_empty()
                || !r.shout_events.is_empty()
            {
                return Err(format!("trace extract: inactive row has events at cycle {}", r.cycle));
            }
            continue;
        }

        // PROG: exactly one read
        let prog_read = r
            .prog_read
            .as_ref()
            .ok_or_else(|| format!("trace extract: active row missing prog_read at cycle {}", r.cycle))?;
        prog.has_read[row_idx] = true;
        prog.ra[row_idx] = prog_read.addr;
        prog.rv[row_idx] = prog_read.value;

        // REG: exactly one read per lane
        let rs1 = r
            .reg_read_lane0
            .as_ref()
            .ok_or_else(|| format!("trace extract: missing REG lane0 read at cycle {}", r.cycle))?;
        let rs2 = r
            .reg_read_lane1
            .as_ref()
            .ok_or_else(|| format!("trace extract: missing REG lane1 read at cycle {}", r.cycle))?;

        reg0.has_read[row_idx] = true;
        reg0.ra[row_idx] = rs1.addr;
        reg0.rv[row_idx] = rs1.value;

        reg1.has_read[row_idx] = true;
        reg1.ra[row_idx] = rs2.addr;
        reg1.rv[row_idx] = rs2.value;

        if let Some(wr) = &r.reg_write_lane0 {
            if wr.addr == 0 {
                return Err(format!("trace extract: unexpected x0 write at cycle {}", r.cycle));
            }
            let prev = regs.get(&wr.addr).copied().unwrap_or(0);
            if wr.value == 0 {
                regs.remove(&wr.addr);
            } else {
                regs.insert(wr.addr, wr.value);
            }
            regs.remove(&0);

            reg0.has_write[row_idx] = true;
            reg0.wa[row_idx] = wr.addr;
            reg0.wv[row_idx] = wr.value;
            reg0.inc_at_write_addr[row_idx] = F::from_u64(wr.value) - F::from_u64(prev);
        }

        // RAM (fixed-lane MVP): at most 1 read + 1 write per row
        let mut read: Option<(u64, u64)> = None;
        let mut write: Option<(u64, u64)> = None;
        for e in &r.ram_events {
            if e.lane.is_some() {
                return Err(format!(
                    "trace extract: unexpected RAM lane hint at cycle {}: lane={:?}",
                    r.cycle, e.lane
                ));
            }
            match e.kind {
                neo_vm_trace::TwistOpKind::Read => {
                    if read.is_some() {
                        return Err(format!("trace extract: multiple RAM reads at cycle {}", r.cycle));
                    }
                    read = Some((e.addr, e.value));
                }
                neo_vm_trace::TwistOpKind::Write => {
                    if write.is_some() {
                        return Err(format!("trace extract: multiple RAM writes at cycle {}", r.cycle));
                    }
                    write = Some((e.addr, e.value));
                }
            }
        }

        let has_read = read.is_some();
        let has_write = write.is_some();
        ram_lane.has_read[row_idx] = has_read;
        ram_lane.has_write[row_idx] = has_write;

        let (ra, rv) = match read {
            Some((a, v)) => (a, Some(v)),
            None => (0, None),
        };
        let (wa, wv) = match write {
            Some((a, v)) => (a, Some(v)),
            None => (0, None),
        };
        if has_read && has_write && ra != wa {
            return Err(format!(
                "trace extract: RAM read/write addr mismatch at cycle {}: ra={ra} wa={wa}",
                r.cycle
            ));
        }
        let addr = if has_read { ra } else { wa };

        if ram_ell_addr < 64 && (addr >> ram_ell_addr) != 0 {
            return Err(format!(
                "trace extract: RAM addr out of range for ell_addr={ram_ell_addr} at cycle {}: addr={addr}",
                r.cycle
            ));
        }

        if let Some(v) = rv {
            let prev = ram.get(&addr).copied().unwrap_or(0);
            if prev != v {
                return Err(format!(
                    "trace extract: RAM read value mismatch at cycle {} addr={addr}: got={v} expected_prev={prev}",
                    r.cycle
                ));
            }
            ram_lane.ra[row_idx] = addr;
            ram_lane.rv[row_idx] = v;
        }

        if let Some(v) = wv {
            let prev = ram.get(&addr).copied().unwrap_or(0);
            ram_lane.wa[row_idx] = addr;
            ram_lane.wv[row_idx] = v;
            ram_lane.inc_at_write_addr[row_idx] = F::from_u64(v) - F::from_u64(prev);

            if v == 0 {
                ram.remove(&addr);
            } else {
                ram.insert(addr, v);
            }
        }
    }

    Ok(TraceTwistLanesOverTime {
        prog,
        reg_lane0: reg0,
        reg_lane1: reg1,
        ram: ram_lane,
    })
}

/// Extract fixed-lane Shout lanes over time (one lane per `shout_table_ids` entry).
///
/// Policy:
/// - At most 1 Shout event per active row.
/// - Inactive rows must have no shout events.
pub fn extract_shout_lanes_over_time(
    exec: &Rv32ExecTable,
    shout_table_ids: &[u32],
) -> Result<Vec<ShoutLaneOverTime>, String> {
    let t = exec.rows.len();

    let mut table_id_to_idx: HashMap<u32, usize> = HashMap::new();
    for (idx, &id) in shout_table_ids.iter().enumerate() {
        if table_id_to_idx.insert(id, idx).is_some() {
            return Err(format!("trace extract: duplicate shout_table_id={id}"));
        }
    }

    let mut lanes: Vec<ShoutLaneOverTime> = (0..shout_table_ids.len())
        .map(|_| ShoutLaneOverTime::new_zero(t))
        .collect();

    for (row_idx, r) in exec.rows.iter().enumerate() {
        if !r.active {
            if !r.shout_events.is_empty() {
                return Err(format!(
                    "trace extract: inactive row has Shout events at cycle {}",
                    r.cycle
                ));
            }
            continue;
        }

        match r.shout_events.as_slice() {
            [] => {}
            [ev] => {
                let idx = table_id_to_idx
                    .get(&ev.shout_id.0)
                    .copied()
                    .ok_or_else(|| {
                        format!(
                            "trace extract: shout_id={} not provisioned (cycle {})",
                            ev.shout_id.0, r.cycle
                        )
                    })?;
                lanes[idx].has_lookup[row_idx] = true;
                let mut key = ev.key;
                if let Some(op) = RiscvShoutTables::new(/*xlen=*/ 32).id_to_opcode(ev.shout_id) {
                    // Canonicalize shift keys: RISC-V shifts use only the low 5 bits of `rhs`.
                    // This shrinks the key space and keeps trace/sidecar linkage stable across packed / bit-addressed encodings.
                    if matches!(op, RiscvOpcode::Sll | RiscvOpcode::Srl | RiscvOpcode::Sra) {
                        let fallback_lhs = r.reg_read_lane0.as_ref().map(|io| io.value).unwrap_or(0);
                        let fallback_rhs = r.reg_read_lane1.as_ref().map(|io| io.value).unwrap_or(0);
                        let (lhs, rhs) =
                            try_decode_lookup_operands(op, key, operand_mode_keys_enabled(), /*xlen=*/ 32)
                                .unwrap_or((fallback_lhs, fallback_rhs));
                        let rhs_masked = rhs & 0x1F;
                        key = encode_lookup_key(op, lhs, rhs_masked, /*xlen=*/ 32);
                    }
                }
                lanes[idx].key[row_idx] = key;
                lanes[idx].value[row_idx] = ev.value as u64;
            }
            _ => {
                return Err(format!(
                    "trace extract: multiple Shout events at cycle {} (fixed-lane policy supports 1)",
                    r.cycle
                ));
            }
        }
    }

    Ok(lanes)
}
