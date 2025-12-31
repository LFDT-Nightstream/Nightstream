use neo_vm_trace::{trace_program, Shout, ShoutId, StepMeta, Twist, TwistId, VmCpu};

struct HaltedCpu;

impl VmCpu<u64, u64> for HaltedCpu {
    type Error = String;

    fn snapshot_regs(&self) -> Vec<u64> {
        vec![0]
    }

    fn pc(&self) -> u64 {
        0
    }

    fn halted(&self) -> bool {
        true
    }

    fn step<T, S>(&mut self, _twist: &mut T, _shout: &mut S) -> Result<StepMeta<u64>, Self::Error>
    where
        T: Twist<u64, u64>,
        S: Shout<u64>,
    {
        Err("step should never be called for a CPU halted at entry".into())
    }
}

struct NoopTwist;

impl Twist<u64, u64> for NoopTwist {
    fn load(&mut self, _twist_id: TwistId, _addr: u64) -> u64 {
        0
    }

    fn store(&mut self, _twist_id: TwistId, _addr: u64, _value: u64) {}
}

struct NoopShout;

impl Shout<u64> for NoopShout {
    fn lookup(&mut self, _shout_id: ShoutId, _key: u64) -> u64 {
        0
    }
}

#[test]
fn trace_program_emits_snapshot_step_when_halted_at_entry() {
    let trace = trace_program(HaltedCpu, NoopTwist, NoopShout, 8).expect("trace_program failed");
    assert_eq!(trace.steps.len(), 1, "expected one snapshot step");
    assert!(trace.steps[0].halted, "snapshot step must be halted");
    assert_eq!(trace.steps[0].cycle, 0, "snapshot step must have cycle 0");
    assert!(trace.did_halt(), "trace should report halted");
}

#[test]
fn trace_program_zero_max_steps_still_returns_empty_trace() {
    let trace = trace_program(HaltedCpu, NoopTwist, NoopShout, 0).expect("trace_program failed");
    assert_eq!(trace.steps.len(), 0);
}

