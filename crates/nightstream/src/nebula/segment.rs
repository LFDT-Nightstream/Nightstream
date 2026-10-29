//! Native execution of one memory segment of an application and the witness
//! words of its invocations (spec §8–§12, segment-level prover input of
//! decision D3). The segment runs first, so the opening invocation knows the
//! proposed ops and FS roots. Each step's words satisfy `RowsHold` of
//! `Lifecycle/Nebula/StepRows.lean` and the application's rows.

use neo_math::K;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks as F;

use super::app::{Application, Memory};
use super::carry::{k_words, Carry, Products};
use super::geometry::Geometry;
use super::records::{ops_lane, pack, scan_lane, OpSlot, PortAccess, ScanSlot};
use super::transcript::{chain_link, chain_root, eta_challenges, header, plan_digest, state_words, Digest, Lane};
use super::words::{self, Layout};
use super::Program;

#[derive(Debug, thiserror::Error)]
pub enum SegmentError {
    #[error("a segment needs exactly N step choices")]
    StepCount,
    #[error("a segment must start from a closed carry below S_max")]
    Carry,
    #[error("the application has no step: {0}")]
    Application(Box<dyn std::error::Error + Send + Sync>),
    #[error("an application step needs B_ops ports, `width` state words, and `aux` private words")]
    Shape,
    #[error("a port access is not valid or reads another value than its cell holds")]
    Access,
    #[error("a timestamp reaches 2^W_ts")]
    Timestamp,
    #[error("the segment's IS chain differs from the carried memory root")]
    MemoryRoot,
    #[error("the close checks of spec §11.2 fail")]
    Close,
}

/// One memory cell: its value and the stamp of its last access.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Cell {
    pub value: u64,
    pub stamp: u64,
}

/// The application and the constants of a program that every invocation
/// reads.
#[derive(Clone, Debug)]
pub struct Context<A> {
    pub(super) app: A,
    program: Program,
    plan_digest: Digest,
    header_ops: Digest,
    header_mem: Digest,
    layout: Layout,
}

impl<A: Application> Context<A> {
    pub fn new(app: A, program: &Program) -> Self {
        let geometry = program.geometry;
        let plan_digest = plan_digest(&geometry);
        Self {
            layout: Layout::new(&geometry, app.width(), app.aux()),
            app,
            program: program.clone(),
            header_ops: header(Lane::Ops, &plan_digest),
            header_mem: header(Lane::Mem, &plan_digest),
            plan_digest,
        }
    }

    pub fn program(&self) -> &Program {
        &self.program
    }

    pub fn geometry(&self) -> Geometry {
        self.program.geometry
    }

    pub fn witness_word_count(&self) -> usize {
        self.layout.count()
    }

    pub fn start_carry(&self) -> Carry {
        Carry::start(self.geometry().n, self.program.d_init)
    }

    /// The Stage 1 state words of an application state and a carry.
    pub fn state(&self, state: &A::State, carry: &Carry) -> Digest {
        state_digest(&self.app.words(state), carry)
    }
}

/// The spec §11.1 state digest of application state words and a carry.
pub(super) fn state_digest(words: &[u64], carry: &Carry) -> Digest {
    state_words(&field_words(words), &carry.words())
}

fn field_words(words: &[u64]) -> Vec<F> {
    words.iter().map(|word| F::from_u64(*word)).collect()
}

/// The images with every stamp zero (spec §5): ROM cells, then RAM cells.
pub(crate) fn initial_memory(rom: &[u32], ram: &[u32]) -> Vec<Cell> {
    rom.iter()
        .chain(ram)
        .map(|word| Cell {
            value: u64::from(*word),
            stamp: 0,
        })
        .collect()
}

/// `D_init`: the IS root of the images (spec §9.2).
pub(super) fn initial_root(geometry: &Geometry, rom: &[u32], ram: &[u32]) -> Digest {
    let memory = initial_memory(rom, ram);
    chain_root(Lane::Mem, &plan_digest(geometry), &memory_lanes(geometry, &memory))
}

fn scan_of(geometry: &Geometry, memory: &[Cell], step: usize) -> Vec<ScanSlot> {
    (0..geometry.b_scan)
        .map(|slot| {
            let cell = memory[step * geometry.b_scan + slot];
            ScanSlot {
                value: cell.value,
                stamp: cell.stamp,
            }
        })
        .collect()
}

fn memory_lanes(geometry: &Geometry, memory: &[Cell]) -> Vec<Vec<F>> {
    (0..geometry.n)
        .map(|step| pack(&scan_lane(geometry, &scan_of(geometry, memory, step))))
        .collect()
}

/// The spec §8.2 fingerprint `g + η1·v + η1²·t − η2`.
fn fingerprint(eta: (K, K), t: u64, g: u64, v: u64) -> K {
    K::from(F::from_u64(g)) + eta.0 * K::from(F::from_u64(v)) + eta.0.square() * K::from(F::from_u64(t)) - eta.1
}

/// The native result of one step: its records, its state words, and its
/// private words.
#[derive(Clone)]
struct Executed {
    words_in: Vec<F>,
    words_out: Vec<F>,
    aux: Vec<F>,
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
pub(crate) struct Segment<S> {
    pub invocations: Vec<Invocation>,
    pub carry: Carry,
    pub state: S,
    pub memory: Vec<Cell>,
}

impl<A: Application> Context<A> {
    /// One step: the application's transition, then its active ports in slot
    /// order, each applied to memory as Lean `access` applies it.
    fn execute(
        &self,
        state: &A::State,
        memory: &mut [Cell],
        ts: u64,
        input: &A::Input,
    ) -> Result<(Executed, A::State), SegmentError> {
        let geometry = &self.geometry();
        let transition = self
            .app
            .step(state, input, &Memory::new(geometry, memory))
            .map_err(|error| SegmentError::Application(Box::new(error)))?;
        let (words_in, words_out) = (self.app.words(state), self.app.words(&transition.next));
        let width = self.layout.width();
        if transition.ports.len() != geometry.b_ops
            || words_in.len() != width
            || words_out.len() != width
            || transition.aux.len() != self.app.aux()
        {
            return Err(SegmentError::Shape);
        }
        let mut ops = Vec::with_capacity(geometry.b_ops);
        let mut write_stamps = Vec::with_capacity(geometry.b_ops);
        let mut active = 0;
        for port in &transition.ports {
            match port {
                None => {
                    ops.push(OpSlot::inactive());
                    write_stamps.push(0);
                }
                Some(access) => {
                    if !access.valid(geometry) {
                        return Err(SegmentError::Access);
                    }
                    let g = access.global_index(geometry);
                    if memory[g].value != access.vr {
                        return Err(SegmentError::Access);
                    }
                    active += 1;
                    let rt = memory[g].stamp;
                    let wt = ts + active;
                    if wt >= 1 << geometry.w_ts {
                        return Err(SegmentError::Timestamp);
                    }
                    ops.push(OpSlot::active(*access, rt));
                    write_stamps.push(wt);
                    memory[g] = Cell {
                        value: access.vw,
                        stamp: wt,
                    };
                }
            }
        }
        let executed = Executed {
            words_in: field_words(&words_in),
            words_out: field_words(&words_out),
            aux: transition.aux,
            ops,
            write_stamps,
            active,
        };
        Ok((executed, transition.next))
    }

    /// Execute the steps of one segment from `memory`; return the step
    /// records, the memory after the segment, and the state after it.
    fn execute_steps(
        &self,
        state: &A::State,
        memory: &[Cell],
        ts: u64,
        inputs: &[A::Input],
    ) -> Result<(Vec<Executed>, Vec<Cell>, A::State), SegmentError> {
        let mut memory = memory.to_vec();
        let mut executed = Vec::with_capacity(inputs.len());
        let mut state = state.clone();
        let mut ts = ts;
        for input in inputs {
            let (step, next) = self.execute(&state, &mut memory, ts, input)?;
            ts += step.active;
            executed.push(step);
            state = next;
        }
        Ok((executed, memory, state))
    }
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

impl<A: Application> Context<A> {
    /// Run one segment from a closed carry: execute the `N` steps, compute the
    /// proposed ops and FS roots, then the witness words of each invocation.
    pub(crate) fn run_segment(
        &self,
        carry: Carry,
        state: &A::State,
        memory: &[Cell],
        inputs: &[A::Input],
    ) -> Result<Segment<A::State>, SegmentError> {
        let geometry = &self.geometry();
        if inputs.len() != geometry.n {
            return Err(SegmentError::StepCount);
        }
        if carry.idx != geometry.n as u64 || carry.seg_idx >= geometry.s_max as u64 {
            return Err(SegmentError::Carry);
        }
        if chain_root(Lane::Mem, &self.plan_digest, &memory_lanes(geometry, memory)) != carry.mem_root {
            return Err(SegmentError::MemoryRoot);
        }
        let (executed, final_memory, state) = self.execute_steps(state, memory, carry.ts, inputs)?;
        let proposal = self.proposal(&executed, &final_memory);
        let invocations = self.invocations(carry, proposal, &executed, memory, &final_memory);
        let last = invocations.last().ok_or(SegmentError::StepCount)?.carry;
        if last.seg_idx == carry.seg_idx {
            return Err(SegmentError::Close);
        }
        Ok(Segment {
            invocations,
            carry: last,
            state,
            memory: final_memory,
        })
    }

    /// Spec §11.2 `open` proposals: the ops and FS roots of an executed
    /// segment.
    fn proposal(&self, executed: &[Executed], final_memory: &[Cell]) -> (Digest, Digest) {
        let ops: Vec<Vec<F>> = executed
            .iter()
            .map(|step| pack(&ops_lane(&self.geometry(), &step.ops)))
            .collect();
        (
            chain_root(Lane::Ops, &self.plan_digest, &ops),
            chain_root(
                Lane::Mem,
                &self.plan_digest,
                &memory_lanes(&self.geometry(), final_memory),
            ),
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
        let geometry = &self.geometry();
        let initial_packed = memory_lanes(geometry, start_memory);
        let final_packed = memory_lanes(geometry, final_memory);
        let mut current = carry;
        executed
            .iter()
            .enumerate()
            .map(|(k, step)| {
                let ops = pack(&ops_lane(geometry, &step.ops));
                let (words, next) = self.invocation(
                    &current,
                    proposal,
                    step,
                    &scan_of(geometry, start_memory, k),
                    &scan_of(geometry, final_memory, k),
                    [&ops, &initial_packed[k], &final_packed[k]],
                );
                current = next;
                Invocation {
                    words,
                    output: state_words(&step.words_out, &next.words()),
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
        let geometry = &self.geometry();
        let layout = &self.layout;
        let n = geometry.n as u64;
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
            set_bits(&mut w, |k| base(layout.op_addr(k)), slot.addr, geometry.mu);
            set_bits(&mut w, |k| base(layout.op_vr(k)), slot.vr, 32);
            set_bits(&mut w, |k| base(layout.op_vw(k)), slot.vw, 32);
            set_bits(&mut w, |k| base(layout.op_rt(k)), slot.rt, geometry.w_ts);
            if !slot.pad {
                let wt = step.write_stamps[j];
                // A dishonest record with `rt ≥ wt` has no `W_ts`-bit word
                // here; O4 rejects it.
                set_bits(
                    &mut w,
                    |k| base(layout.op_diff(k)),
                    wt.wrapping_sub(slot.rt + 1),
                    geometry.w_ts,
                );
                let access = PortAccess {
                    is_write: slot.is_write,
                    is_ram: slot.is_ram,
                    addr: slot.addr,
                    vr: slot.vr,
                    vw: slot.vw,
                };
                let g = access.global_index(geometry) as u64;
                read *= fingerprint(eta, slot.rt, g, slot.vr);
                write *= fingerprint(eta, wt, g, slot.vw);
            }
            set_k(&mut w, layout.ops_products(j, 0), read);
            set_k(&mut w, layout.ops_products(j, 2), write);
        }
        // Scan slots and the running IS and FS products.
        let mut is_product = effective.products.initial;
        let mut fs_product = effective.products.final_;
        for j in 0..geometry.b_scan {
            set_bits(&mut w, |k| layout.initial(j, k), initial[j].value, 32);
            set_bits(&mut w, |k| layout.initial(j, 32 + k), initial[j].stamp, geometry.w_ts);
            set_bits(&mut w, |k| layout.final_(j, k), final_[j].value, 32);
            set_bits(&mut w, |k| layout.final_(j, 32 + k), final_[j].stamp, geometry.w_ts);
            let g = idx * geometry.b_scan as u64 + j as u64;
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
        w[words::CARRY_IN..words::CARRY_IN + 39].copy_from_slice(&input.words());
        w[words::CARRY_OUT..words::CARRY_OUT + 39].copy_from_slice(&output.words());
        w[words::PROPOSAL..words::PROPOSAL + 4].copy_from_slice(&proposal.0);
        w[words::PROPOSAL + 4..words::PROPOSAL + 8].copy_from_slice(&proposal.1);
        for (k, word) in step.words_in.iter().enumerate() {
            w[layout.app_in(k)] = *word;
        }
        for (k, word) in step.words_out.iter().enumerate() {
            w[layout.app_out(k)] = *word;
        }
        for (k, word) in step.aux.iter().enumerate() {
            w[layout.aux(k)] = *word;
        }
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
        set_bits(&mut w, |k| layout.ts_bits(k), output.ts, geometry.w_ts);
        if opens {
            let room = geometry.s_max as u64 - 1 - input.seg_idx;
            set_bits(&mut w, |k| layout.seg_bits(k), room, geometry.seg_width());
        }
        (w, output)
    }
}

#[cfg(test)]
#[path = "../../tests/nebula/conformance.rs"]
mod conformance;
