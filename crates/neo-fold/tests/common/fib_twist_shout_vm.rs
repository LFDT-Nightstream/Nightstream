use std::collections::HashMap;

use neo_vm_trace::{Shout, ShoutId, StepMeta, Twist, TwistId, VmCpu};

#[derive(Clone, Debug, Default)]
pub struct MapTwist {
    mem: HashMap<(TwistId, u64), u64>,
}

impl Twist<u64, u64> for MapTwist {
    fn load(&mut self, id: TwistId, addr: u64) -> u64 {
        *self.mem.get(&(id, addr)).unwrap_or(&0)
    }

    fn store(&mut self, id: TwistId, addr: u64, val: u64) {
        self.mem.insert((id, addr), val);
    }
}

#[derive(Clone, Debug)]
pub struct MapShout {
    pub table: Vec<u64>,
}

impl Shout<u64> for MapShout {
    fn lookup(&mut self, id: ShoutId, key: u64) -> u64 {
        assert_eq!(id.0, 0, "this test only supports shout_id=0");
        self.table.get(key as usize).copied().unwrap_or(0)
    }
}

/// Fibonacci VM that (a) updates `(f_curr, f_next)` and (b) drives Twist+Shout every step.
///
/// - Shout: lookup table 0 at key=1 (value=1) to bind Shout into the CCS.
/// - Twist: RW at (twist_id=0, addr=0): read current `f_next`, then write `f_curr + f_next`.
pub struct FibTwistShoutVm {
    pub f_curr: u64,
    pub f_next: u64,
    pc: u64,
    step: u64,
    max_steps: u64,
    halted: bool,
    q: u64,
}

impl FibTwistShoutVm {
    pub fn new(max_steps: u64, q: u64) -> Self {
        Self {
            f_curr: 0,
            f_next: 1,
            pc: 0,
            step: 0,
            max_steps,
            halted: false,
            q,
        }
    }
}

impl VmCpu<u64, u64> for FibTwistShoutVm {
    type Error = String;

    fn snapshot_regs(&self) -> Vec<u64> {
        vec![self.f_curr, self.f_next]
    }

    fn pc(&self) -> u64 {
        self.pc
    }

    fn halted(&self) -> bool {
        self.halted
    }

    fn step<TW, SH>(&mut self, twist: &mut TW, shout: &mut SH) -> Result<StepMeta<u64>, Self::Error>
    where
        TW: Twist<u64, u64>,
        SH: Shout<u64>,
    {
        let _one = shout.lookup(ShoutId(0), 1);

        let mem_id = TwistId(0);
        let mem_next = twist.load(mem_id, 0);
        if mem_next != self.f_next {
            return Err(format!(
                "memory/state mismatch before step {}: f_next(reg)={} vs mem[0]={}",
                self.step, self.f_next, mem_next
            ));
        }

        // IMPORTANT: keep the VM's arithmetic consistent with Goldilocks field arithmetic by
        // computing Fibonacci mod q (not mod 2^64). This allows arbitrarily many steps without
        // overflow mismatches between u64 "register" values and field elements in the CCS.
        let f_new = add_mod_q(self.f_curr, self.f_next, self.q);
        self.f_curr = self.f_next;
        self.f_next = f_new;
        twist.store(mem_id, 0, self.f_next);

        self.step += 1;
        self.pc += 4;
        if self.step >= self.max_steps {
            self.halted = true;
        }

        Ok(StepMeta {
            pc_after: self.pc,
            opcode: 0xF1B0, // arbitrary "FIB" opcode for trace readability
        })
    }
}

pub fn add_mod_q(a: u64, b: u64, q: u64) -> u64 {
    debug_assert!(a < q);
    debug_assert!(b < q);
    let sum = (a as u128) + (b as u128);
    let q128 = q as u128;
    let reduced = if sum >= q128 { sum - q128 } else { sum };
    // a,b<q => sum<2q, so one subtraction is enough.
    reduced as u64
}

pub fn fib_mod_q_u64(n: usize, q: u64) -> u64 {
    // F_0=0, F_1=1, computed mod q.
    let (mut a, mut b) = (0u64, 1u64);
    for _ in 0..n {
        let c = add_mod_q(a, b, q);
        a = b;
        b = c;
    }
    a
}
