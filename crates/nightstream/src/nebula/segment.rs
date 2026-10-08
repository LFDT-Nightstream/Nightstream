//! Native execution of one memory segment and the witness words of its
//! invocations (spec §8–§12, segment-level prover input of decision D3).
//! The segment runs first, so the opening invocation knows the proposed ops
//! and FS roots. Each step's words satisfy `RowsHold` and `MachineRows` of
//! `Lifecycle/Nebula/StepRows.lean`.

use neo_math::K;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks as F;

use super::carry::{k_words, Carry, Products};
use super::machine::{data_port, exec, fetch, MachineState};
use super::plan::Plan;
use super::records::{ops_lane, pack, scan_lane, OpSlot, PortAccess, ScanSlot};
use super::transcript::{chain_link, chain_root, eta_challenges, header, plan_digest, state_words, Digest, Lane};
use super::words::{self, Layout};

#[derive(Debug, thiserror::Error)]
pub enum SegmentError {
    #[error("the plan must have two ports")]
    Ports,
    #[error("a segment needs exactly N step choices")]
    StepCount,
    #[error("a segment must start from a closed carry below S_max")]
    Carry,
    #[error("the program counter leaves the ROM")]
    ProgramCounter,
    #[error("a data address leaves the RAM")]
    DataAddress,
    #[error("a timestamp reaches 2^W_ts")]
    Timestamp,
    #[error("the segment's IS chain differs from the carried memory root")]
    MemoryRoot,
    #[error("the close checks of spec §11.2 fail")]
    Close,
}

/// One memory cell: its value and the stamp of its last access.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cell {
    pub value: u64,
    pub stamp: u64,
}

/// One step of the machine: no port, or the instruction at `pc`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Idle,
    Execute,
}

/// Plan constants that every invocation reads.
#[derive(Clone, Debug)]
pub struct Context {
    plan: Plan,
    plan_digest: Digest,
    header_ops: Digest,
    header_mem: Digest,
    d_init: Digest,
    layout: Layout,
}

impl Context {
    pub fn new(plan: Plan) -> Result<Self, SegmentError> {
        if plan.b_ops != 2 {
            return Err(SegmentError::Ports);
        }
        let plan_digest = plan_digest(&plan);
        let memory = initial_memory(&plan);
        let d_init = chain_root(Lane::Mem, &plan_digest, &memory_lanes(&plan, &memory));
        Ok(Self {
            header_ops: header(Lane::Ops, &plan_digest),
            header_mem: header(Lane::Mem, &plan_digest),
            layout: Layout::new(&plan),
            plan,
            plan_digest,
            d_init,
        })
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    pub fn witness_word_count(&self) -> usize {
        self.layout.count()
    }

    /// `D_init`: the IS root of the plan images (spec §9.2).
    pub fn d_init(&self) -> Digest {
        self.d_init
    }

    pub fn start_carry(&self) -> Carry {
        Carry::start(self.plan.n, self.d_init)
    }

    /// The Stage 1 state words of application words and a carry.
    pub fn state(&self, app: MachineState, carry: &Carry) -> Digest {
        state_words(&app_words(app), &carry.words())
    }
}

/// The plan images with every stamp zero (spec §5).
pub fn initial_memory(plan: &Plan) -> Vec<Cell> {
    (0..plan.cells())
        .map(|g| Cell {
            value: plan.image(g),
            stamp: 0,
        })
        .collect()
}

fn scan_of(plan: &Plan, memory: &[Cell], step: usize) -> Vec<ScanSlot> {
    (0..plan.b_scan)
        .map(|slot| {
            let cell = memory[step * plan.b_scan + slot];
            ScanSlot {
                value: cell.value,
                stamp: cell.stamp,
            }
        })
        .collect()
}

fn memory_lanes(plan: &Plan, memory: &[Cell]) -> Vec<Vec<F>> {
    (0..plan.n)
        .map(|step| pack(&scan_lane(plan, &scan_of(plan, memory, step))))
        .collect()
}

pub fn app_words(state: MachineState) -> [F; 2] {
    [F::from_u64(state.pc), F::from_u64(state.acc)]
}

/// The spec §8.2 fingerprint `g + η1·v + η1²·t − η2`.
fn fingerprint(eta: (K, K), t: u64, g: u64, v: u64) -> K {
    K::from(F::from_u64(g)) + eta.0 * K::from(F::from_u64(v)) + eta.0.square() * K::from(F::from_u64(t)) - eta.1
}

/// The native result of one step: its records and its states.
#[derive(Clone)]
struct Executed {
    idle: bool,
    state_in: MachineState,
    state_out: MachineState,
    ops: Vec<OpSlot>,
    write_stamps: Vec<u64>,
    active: u64,
}

/// One invocation's witness words, its output state, and its output carry.
#[derive(Clone, Debug)]
pub struct Invocation {
    pub words: Vec<F>,
    pub output: Digest,
    pub carry: Carry,
}

/// A proved-ready segment: its invocations and the values after it.
#[derive(Clone, Debug)]
pub struct Segment {
    pub invocations: Vec<Invocation>,
    pub carry: Carry,
    pub state: MachineState,
    pub memory: Vec<Cell>,
}

fn execute(
    plan: &Plan,
    state: MachineState,
    memory: &mut [Cell],
    ts: u64,
    step: Step,
) -> Result<Executed, SegmentError> {
    let ports: [Option<PortAccess>; 2];
    let state_out;
    match step {
        Step::Idle => {
            ports = [None, None];
            state_out = state;
        }
        Step::Execute => {
            let pc = usize::try_from(state.pc).map_err(|_| SegmentError::ProgramCounter)?;
            if pc >= plan.rom_size() {
                return Err(SegmentError::ProgramCounter);
            }
            let word = memory[pc].value;
            let data = match word % 4 {
                1 | 2 => {
                    let address = usize::try_from(word / 4).map_err(|_| SegmentError::DataAddress)?;
                    if address >= plan.ram_size() {
                        return Err(SegmentError::DataAddress);
                    }
                    memory[plan.rom_size() + address].value
                }
                _ => 0,
            };
            ports = [Some(fetch(state, word)), data_port(state, word, data)];
            state_out = exec(state, word, data);
        }
    }
    let mut ops = Vec::with_capacity(2);
    let mut write_stamps = Vec::with_capacity(2);
    let mut active = 0;
    for port in ports {
        match port {
            None => {
                ops.push(OpSlot::inactive());
                write_stamps.push(0);
            }
            Some(access) => {
                active += 1;
                let g = access.global_index(plan);
                let rt = memory[g].stamp;
                let wt = ts + active;
                if wt >= 1 << plan.w_ts {
                    return Err(SegmentError::Timestamp);
                }
                ops.push(OpSlot::active(access, rt));
                write_stamps.push(wt);
                memory[g] = Cell {
                    value: access.vw,
                    stamp: wt,
                };
            }
        }
    }
    Ok(Executed {
        idle: step == Step::Idle,
        state_in: state,
        state_out,
        ops,
        write_stamps,
        active,
    })
}

/// Execute the steps of one segment from `memory`; return the step records,
/// the memory after the segment, and the machine state after it.
fn execute_steps(
    plan: &Plan,
    state: MachineState,
    memory: &[Cell],
    ts: u64,
    steps: &[Step],
) -> Result<(Vec<Executed>, Vec<Cell>, MachineState), SegmentError> {
    let mut memory = memory.to_vec();
    let mut executed = Vec::with_capacity(steps.len());
    let mut machine = state;
    let mut ts = ts;
    for step in steps {
        let result = execute(plan, machine, &mut memory, ts, *step)?;
        machine = result.state_out;
        ts += result.active;
        executed.push(result);
    }
    Ok((executed, memory, machine))
}

fn set_bits(words: &mut [F], index: impl Fn(usize) -> usize, value: u64, width: usize) {
    for bit in 0..width {
        words[index(bit)] = F::from_u64((value >> bit) & 1);
    }
}

fn set_k(words: &mut [F], start: usize, value: K) {
    let [c0, c1] = k_words(value);
    words[start] = c0;
    words[start + 1] = c1;
}

impl Context {
    /// Run one segment from a closed carry: execute the `N` steps, compute the
    /// proposed ops and FS roots, then the witness words of each invocation.
    pub fn run_segment(
        &self,
        carry: Carry,
        state: MachineState,
        memory: &[Cell],
        steps: &[Step],
    ) -> Result<Segment, SegmentError> {
        let plan = &self.plan;
        if steps.len() != plan.n {
            return Err(SegmentError::StepCount);
        }
        if carry.idx != plan.n as u64 || carry.seg_idx >= plan.s_max as u64 {
            return Err(SegmentError::Carry);
        }
        if chain_root(Lane::Mem, &self.plan_digest, &memory_lanes(plan, memory)) != carry.mem_root {
            return Err(SegmentError::MemoryRoot);
        }
        let (executed, final_memory, machine) = execute_steps(plan, state, memory, carry.ts, steps)?;
        let proposal = self.proposal(&executed, &final_memory);
        let invocations = self.invocations(carry, proposal, &executed, memory, &final_memory);
        let last = invocations.last().ok_or(SegmentError::StepCount)?.carry;
        if last.seg_idx == carry.seg_idx {
            return Err(SegmentError::Close);
        }
        Ok(Segment {
            invocations,
            carry: last,
            state: machine,
            memory: final_memory,
        })
    }

    /// Spec §11.2 `open` proposals: the ops and FS roots of an executed
    /// segment.
    fn proposal(&self, executed: &[Executed], final_memory: &[Cell]) -> (Digest, Digest) {
        let ops: Vec<Vec<F>> = executed
            .iter()
            .map(|step| pack(&ops_lane(&self.plan, &step.ops)))
            .collect();
        (
            chain_root(Lane::Ops, &self.plan_digest, &ops),
            chain_root(Lane::Mem, &self.plan_digest, &memory_lanes(&self.plan, final_memory)),
        )
    }

    /// The invocations of an executed segment from `carry`, with `proposal` as
    /// the `open` proposals. It runs no host check, so a conformance test can
    /// build the invocations of a dishonest segment.
    fn invocations(
        &self,
        carry: Carry,
        proposal: (Digest, Digest),
        executed: &[Executed],
        start_memory: &[Cell],
        final_memory: &[Cell],
    ) -> Vec<Invocation> {
        let plan = &self.plan;
        let initial_packed = memory_lanes(plan, start_memory);
        let final_packed = memory_lanes(plan, final_memory);
        let mut current = carry;
        executed
            .iter()
            .enumerate()
            .map(|(k, step)| {
                let ops = pack(&ops_lane(plan, &step.ops));
                let (words, next) = self.invocation(
                    &current,
                    proposal,
                    step,
                    &scan_of(plan, start_memory, k),
                    &scan_of(plan, final_memory, k),
                    [&ops, &initial_packed[k], &final_packed[k]],
                );
                current = next;
                Invocation {
                    words,
                    output: state_words(&app_words(step.state_out), &next.words()),
                    carry: next,
                }
            })
            .collect()
    }

    /// The witness words of one invocation and its output carry.
    fn invocation(
        &self,
        input: &Carry,
        proposal: (Digest, Digest),
        step: &Executed,
        initial: &[ScanSlot],
        final_: &[ScanSlot],
        packed: [&Vec<F>; 3],
    ) -> (Vec<F>, Carry) {
        let plan = &self.plan;
        let layout = &self.layout;
        let n = plan.n as u64;
        let opens = input.idx == n;
        let fresh = eta_challenges(
            &self.plan_digest,
            F::from_u64(input.ts),
            &proposal.0,
            &input.mem_root,
            &proposal.1,
        );
        let effective = if opens {
            Carry {
                proposed: proposal,
                eta: fresh,
                products: Products::one(),
                seen: [self.header_ops, self.header_mem, self.header_mem],
                idx: 0,
                ..*input
            }
        } else {
            *input
        };
        let eta = effective.eta;
        let idx = effective.idx;
        let mut w = vec![F::ZERO; layout.count()];
        // Operation slots and the running read and write products.
        let mut read = effective.products.read;
        let mut write = effective.products.write;
        for (j, slot) in step.ops.iter().enumerate() {
            let base = |k: usize| layout.op(j, k);
            w[base(0)] = F::from_bool(slot.pad);
            w[base(1)] = F::from_bool(slot.is_write);
            w[base(2)] = F::from_bool(slot.is_ram);
            set_bits(&mut w, |k| base(layout.op_addr(k)), slot.addr, plan.mu);
            set_bits(&mut w, |k| base(layout.op_vr(k)), slot.vr, 32);
            set_bits(&mut w, |k| base(layout.op_vw(k)), slot.vw, 32);
            set_bits(&mut w, |k| base(layout.op_rt(k)), slot.rt, plan.w_ts);
            if !slot.pad {
                let wt = step.write_stamps[j];
                // A dishonest record with `rt ≥ wt` has no `W_ts`-bit word
                // here; O4 rejects it.
                set_bits(
                    &mut w,
                    |k| base(layout.op_diff(k)),
                    wt.wrapping_sub(slot.rt + 1),
                    plan.w_ts,
                );
                let access = PortAccess {
                    is_write: slot.is_write,
                    is_ram: slot.is_ram,
                    addr: slot.addr,
                    vr: slot.vr,
                    vw: slot.vw,
                };
                let g = access.global_index(plan) as u64;
                read *= fingerprint(eta, slot.rt, g, slot.vr);
                write *= fingerprint(eta, wt, g, slot.vw);
            }
            set_k(&mut w, layout.ops_products(j, 0), read);
            set_k(&mut w, layout.ops_products(j, 2), write);
        }
        // Scan slots and the running IS and FS products.
        let mut is_product = effective.products.initial;
        let mut fs_product = effective.products.final_;
        for j in 0..plan.b_scan {
            set_bits(&mut w, |k| layout.initial(j, k), initial[j].value, 32);
            set_bits(&mut w, |k| layout.initial(j, 32 + k), initial[j].stamp, plan.w_ts);
            set_bits(&mut w, |k| layout.final_(j, k), final_[j].value, 32);
            set_bits(&mut w, |k| layout.final_(j, 32 + k), final_[j].stamp, plan.w_ts);
            let g = idx * plan.b_scan as u64 + j as u64;
            is_product *= fingerprint(eta, initial[j].stamp, g, initial[j].value);
            fs_product *= fingerprint(eta, final_[j].stamp, g, final_[j].value);
            set_k(&mut w, layout.scan_products(j, 0), is_product);
            set_k(&mut w, layout.scan_products(j, 2), fs_product);
        }
        // Spec §11.2 step, then close exactly when idx reaches N.
        let advanced = Carry {
            seen: [
                chain_link(Lane::Ops, idx as usize, &effective.seen[0], packed[0]),
                chain_link(Lane::Mem, idx as usize, &effective.seen[1], packed[1]),
                chain_link(Lane::Mem, idx as usize, &effective.seen[2], packed[2]),
            ],
            products: Products {
                read,
                write,
                initial: is_product,
                final_: fs_product,
            },
            ts: effective.ts + step.active,
            idx: idx + 1,
            ..effective
        };
        let closes = advanced.idx == n;
        let products = &advanced.products;
        let close_holds = advanced.seen[0] == advanced.proposed.0
            && advanced.seen[1] == advanced.mem_root
            && advanced.seen[2] == advanced.proposed.1
            && products.initial * products.write == products.read * products.final_;
        let output = if closes && close_holds {
            Carry {
                mem_root: advanced.proposed.1,
                seg_idx: advanced.seg_idx + 1,
                ..advanced
            }
        } else {
            advanced
        };
        // Words.
        w[words::APP_IN..words::APP_IN + 2].copy_from_slice(&app_words(step.state_in));
        w[words::CARRY_IN..words::CARRY_IN + 39].copy_from_slice(&input.words());
        w[words::CARRY_OUT..words::CARRY_OUT + 39].copy_from_slice(&output.words());
        w[words::APP_OUT..words::APP_OUT + 2].copy_from_slice(&app_words(step.state_out));
        w[words::PROPOSAL..words::PROPOSAL + 4].copy_from_slice(&proposal.0);
        w[words::PROPOSAL + 4..words::PROPOSAL + 8].copy_from_slice(&proposal.1);
        w[words::IDLE] = F::from_bool(step.idle);
        w[words::IS_OPEN] = F::from_bool(opens);
        if !opens {
            w[words::OPEN_INVERSE] = (F::from_u64(input.idx) - F::from_u64(n)).inverse();
        }
        w[words::IS_CLOSE] = F::from_bool(closes);
        if !closes {
            w[words::CLOSE_INVERSE] = (F::from_u64(advanced.idx) - F::from_u64(n)).inverse();
        }
        set_k(&mut w, words::ETA_FRESH, fresh.0);
        set_k(&mut w, words::ETA_FRESH + 2, fresh.1);
        set_k(&mut w, words::ETA1_SQUARE, eta.0.square());
        for (lane, digest) in effective.seen.iter().enumerate() {
            w[words::SEEN_PREV + 4 * lane..words::SEEN_PREV + 4 * lane + 4].copy_from_slice(digest);
        }
        w[words::IDX_EFF] = F::from_u64(idx);
        set_bits(&mut w, |k| layout.ts_bits(k), output.ts, plan.w_ts);
        if opens {
            let room = plan.s_max as u64 - 1 - input.seg_idx;
            set_bits(&mut w, |k| layout.seg_bits(k), room, plan.seg_width());
        }
        (w, output)
    }
}

#[cfg(test)]
#[path = "../../tests/nebula/conformance.rs"]
mod conformance;
