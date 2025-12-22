use neo_vm_trace::{TwistOpKind, VmTrace};
use p3_field::PrimeField64;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct PlainMemLayout {
    pub k: usize,
    pub d: usize,
    pub n_side: usize,
}

#[derive(Clone, Debug)]
pub struct PlainMemTrace<F> {
    pub steps: usize,
    pub has_read: Vec<F>,
    pub has_write: Vec<F>,
    pub read_addr: Vec<u64>,
    pub write_addr: Vec<u64>,
    pub read_val: Vec<F>,
    pub write_val: Vec<F>,
    pub inc_at_write_addr: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct PlainLutTrace<F> {
    pub has_lookup: Vec<F>,
    pub addr: Vec<u64>,
    pub val: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct LutTable<F> {
    pub table_id: u32,
    pub k: usize,
    pub d: usize,
    pub n_side: usize,
    pub content: Vec<F>,
}

pub fn build_plain_mem_traces<F: PrimeField64>(
    trace: &VmTrace<u64, u64>,
    layouts: &HashMap<u32, PlainMemLayout>,
    initial_mem: &HashMap<(u32, u64), F>,
) -> HashMap<u32, PlainMemTrace<F>> {
    let mut results = HashMap::new();
    let steps_len = trace.steps.len();

    for (mem_id, layout) in layouts {
        let mut t = PlainMemTrace {
            steps: steps_len,
            has_read: vec![F::ZERO; steps_len],
            has_write: vec![F::ZERO; steps_len],
            read_addr: vec![0; steps_len],
            write_addr: vec![0; steps_len],
            read_val: vec![F::ZERO; steps_len],
            write_val: vec![F::ZERO; steps_len],
            inc_at_write_addr: vec![F::ZERO; steps_len],
        };

        // Track only non-zero cells to avoid materializing a dense `k`-vector.
        let mut current_mem: HashMap<u64, F> = HashMap::new();
        for ((init_mem_id, addr), &val) in initial_mem.iter() {
            if init_mem_id == mem_id && val != F::ZERO {
                current_mem.insert(*addr, val);
            }
        }

        for (j, step) in trace.steps.iter().enumerate() {
            // Using the TwistEvent structure (renamed from MemEvent)
            for event in &step.twist_events {
                if event.twist_id.0 == *mem_id {
                    match event.kind {
                        TwistOpKind::Read => {
                            // Ensure at most one read per step per memory
                            debug_assert_eq!(
                                t.has_read[j],
                                F::ZERO,
                                "Multiple reads for twist_id {} at step {}",
                                mem_id,
                                j
                            );
                            t.has_read[j] = F::ONE;
                            t.read_addr[j] = event.addr;
                            t.read_val[j] = F::from_u64(event.value);
                        }
                        TwistOpKind::Write => {
                            // Ensure at most one write per step per memory
                            debug_assert_eq!(
                                t.has_write[j],
                                F::ZERO,
                                "Multiple writes for twist_id {} at step {}",
                                mem_id,
                                j
                            );
                            t.has_write[j] = F::ONE;
                            t.write_addr[j] = event.addr;
                            t.write_val[j] = F::from_u64(event.value);

                            // Write increment at the write address: wv - prev.
                            let addr = event.addr;
                            if (addr as usize) < layout.k {
                                let old_val = current_mem.get(&addr).copied().unwrap_or(F::ZERO);
                                let new_val = F::from_u64(event.value);
                                t.inc_at_write_addr[j] = new_val - old_val;
                                if new_val == F::ZERO {
                                    current_mem.remove(&addr);
                                } else {
                                    current_mem.insert(addr, new_val);
                                }
                            }
                        }
                    }
                }
            }
        }

        results.insert(*mem_id, t);
    }

    results
}

pub fn build_plain_lut_traces<F: PrimeField64>(
    trace: &VmTrace<u64, u64>,
    table_sizes: &HashMap<u32, (usize, usize)>,
) -> HashMap<u32, PlainLutTrace<F>> {
    let mut results = HashMap::new();
    let steps_len = trace.steps.len();

    for (table_id, _) in table_sizes {
        results.insert(
            *table_id,
            PlainLutTrace {
                has_lookup: vec![F::ZERO; steps_len],
                addr: vec![0; steps_len],
                val: vec![F::ZERO; steps_len],
            },
        );
    }

    for (j, step) in trace.steps.iter().enumerate() {
        // Using ShoutEvent (renamed from LookupEvent)
        for shout in &step.shout_events {
            if let Some(t) = results.get_mut(&shout.shout_id.0) {
                // Ensure at most one lookup per step per table
                debug_assert_eq!(
                    t.has_lookup[j],
                    F::ZERO,
                    "Multiple shouts for shout_id {} at step {}",
                    shout.shout_id.0,
                    j
                );
                t.has_lookup[j] = F::ONE;
                t.addr[j] = shout.key;
                t.val[j] = F::from_u64(shout.value);
            }
        }
    }

    results
}
