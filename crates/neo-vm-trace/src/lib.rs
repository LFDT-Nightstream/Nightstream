//! # neo-vm-trace
//!
//! A thin abstraction layer for tracing VM execution. This crate sits between
//! a concrete VM implementation and the Neo proving stack (CCS + Twist/Shout).
//!
//! ## Overview
//!
//! The crate provides:
//! - **Trace format**: `VmTrace`, `StepTrace`, `TwistEvent`, `ShoutEvent`
//! - **Tracing wrappers**: `TracingTwist`, `TracingShout` that intercept operations
//! - **CPU trait**: `VmCpu` that any concrete VM can implement
//! - **Driver function**: `trace_program` that runs the VM and produces traces
//!
//! ## Terminology
//!
//! - **Twist**: Read/write memory (RAM, registers, stack). Proven via the Twist argument.
//! - **Shout**: Read-only lookup tables. Proven via the Shout argument.
//!
//! ## Usage
//!
//! ```ignore
//! // 1. Implement the traits for your VM
//! impl VmCpu<u64, u64> for MyCpu { ... }
//! impl Twist<u64, u64> for MyMemory { ... }
//! impl Shout<u64> for MyTables { ... }
//!
//! // 2. Run and trace
//! let trace = trace_program(my_cpu, my_twist, my_shout, 1000)?;
//!
//! // 3. Pass to neo-memory for witness generation
//! let steps = build_shard_witness_shared_cpu_bus(...);
//! ```
//!
//! ## Design Notes
//!
//! - **No field arithmetic**: This crate works in machine types (u64, etc.).
//!   Conversion to field elements happens in `neo-memory`.
//! - **Relational trace**: Only stores per-step deltas, not full state dumps.
//! - **Lane-bounded side effects**: A single step may contain multiple Twist/Shout events for an id,
//!   up to the number of access lanes provisioned downstream (and optionally pinned via `lane` hints).

use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// Identifiers
// ============================================================================

/// Identifies different Twist memories (e.g., 0=registers, 1=RAM, 2=heap).
///
/// Multiple memories allow tracking different address spaces separately,
/// which maps to different Twist instances in the proving layer.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct TwistId(pub u32);

impl fmt::Display for TwistId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "twist_{}", self.0)
    }
}

/// Identifies different Shout tables (read-only lookup tables).
///
/// Each table maps to a separate Shout instance in the proving layer.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ShoutId(pub u32);

impl fmt::Display for ShoutId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "shout_{}", self.0)
    }
}

// ============================================================================
// Twist Events (Read/Write Memory)
// ============================================================================

/// The kind of Twist (memory) operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TwistOpKind {
    Read,
    Write,
}

impl fmt::Display for TwistOpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TwistOpKind::Read => write!(f, "read"),
            TwistOpKind::Write => write!(f, "write"),
        }
    }
}

/// A single Twist (memory) read or write event.
///
/// Records which memory was accessed, the address, and the value read/written.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TwistEvent<Addr, Word> {
    /// Which Twist memory this event belongs to.
    pub twist_id: TwistId,
    /// Whether this is a read or write.
    pub kind: TwistOpKind,
    /// The address accessed.
    pub addr: Addr,
    /// The value read or written.
    pub value: Word,
    /// Optional lane hint for witness assignment (when multiple per-step access lanes exist).
    ///
    /// When present, downstream witness assignment should place this event in the specified
    /// lane index (0-based) rather than using a "first free lane" policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lane: Option<u32>,
}

impl<Addr: fmt::Debug, Word: fmt::Debug> fmt::Display for TwistEvent<Addr, Word> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{:?}", self.twist_id, self.addr)?;
        if let Some(lane) = self.lane {
            write!(f, ", lane={lane}")?;
        }
        write!(f, "] {} {:?}", self.kind, self.value)
    }
}

// ============================================================================
// Shout Events (Read-Only Lookups)
// ============================================================================

/// A Shout event: a lookup into a fixed read-only table.
///
/// Used for operations like byte decomposition, range checks, or any
/// precomputed function that the VM needs to query.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShoutEvent<Word> {
    /// Which Shout table was queried.
    pub shout_id: ShoutId,
    /// The lookup key (index or encoded tuple).
    pub key: Word,
    /// The value returned from the table.
    pub value: Word,
}

impl<Word: fmt::Debug> fmt::Display for ShoutEvent<Word> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{:?}] = {:?}", self.shout_id, self.key, self.value)
    }
}

// ============================================================================
// Step Trace
// ============================================================================

/// Per-step machine state and side effects.
///
/// Records everything that happened during a single CPU cycle:
/// - Program counter before/after
/// - Register state before/after
/// - All Twist (memory) operations
/// - All Shout (lookup) operations
/// - Whether the CPU halted
///
/// # Notes
///
/// The trace format allows multiple Twist/Shout events in a single step. Downstream
/// circuits may require you to provision enough access "lanes" (ports) and will
/// reject steps that exceed that capacity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepTrace<Addr, Word> {
    /// The cycle number (0-indexed).
    pub cycle: u64,
    /// Program counter before executing this instruction.
    pub pc_before: Addr,
    /// Program counter after executing this instruction.
    pub pc_after: Addr,
    /// The opcode executed (ISA-specific encoding).
    pub opcode: u32,
    /// Register state before execution.
    pub regs_before: Vec<Word>,
    /// Register state after execution.
    pub regs_after: Vec<Word>,
    /// Twist events (memory reads and writes) during this step.
    pub twist_events: Vec<TwistEvent<Addr, Word>>,
    /// Shout events (lookups) during this step.
    pub shout_events: Vec<ShoutEvent<Word>>,
    /// True if this step executed a halt instruction.
    pub halted: bool,
}

impl<Addr, Word> StepTrace<Addr, Word> {
    /// Returns the number of Twist reads in this step.
    pub fn num_reads(&self) -> usize {
        self.twist_events
            .iter()
            .filter(|e| matches!(e.kind, TwistOpKind::Read))
            .count()
    }

    /// Returns the number of Twist writes in this step.
    pub fn num_writes(&self) -> usize {
        self.twist_events
            .iter()
            .filter(|e| matches!(e.kind, TwistOpKind::Write))
            .count()
    }

    /// Returns the number of Shout lookups in this step.
    pub fn num_shouts(&self) -> usize {
        self.shout_events.len()
    }
}

// ============================================================================
// Full Trace
// ============================================================================

/// Full execution trace for a shard.
///
/// A shard is a contiguous sequence of VM steps (up to `max_steps`).
/// This trace is the input to the Neo proving pipeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VmTrace<Addr, Word> {
    /// The sequence of step traces.
    pub steps: Vec<StepTrace<Addr, Word>>,
}

impl<Addr, Word> VmTrace<Addr, Word> {
    /// Creates an empty trace.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Returns the number of steps in the trace.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns true if the trace has no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Returns true if the final step halted the CPU.
    pub fn did_halt(&self) -> bool {
        self.steps.last().map(|s| s.halted).unwrap_or(false)
    }

    /// Returns the total number of Twist events across all steps.
    pub fn total_twist_events(&self) -> usize {
        self.steps.iter().map(|s| s.twist_events.len()).sum()
    }

    /// Returns the total number of Shout events across all steps.
    pub fn total_shout_events(&self) -> usize {
        self.steps.iter().map(|s| s.shout_events.len()).sum()
    }
}

impl<Addr, Word> Default for VmTrace<Addr, Word> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Twist Trait and Tracing Wrapper (Read/Write Memory)
// ============================================================================

/// Abstraction for read/write memory (Twist).
///
/// The VM uses this trait for all memory operations. Different `TwistId` values
/// can represent different address spaces (registers, RAM, stack, etc.).
///
/// # Requirements
///
/// - `load` must return the current value at the address.
/// - `store` must update the value at the address.
/// - Implementations should be deterministic.
pub trait Twist<Addr, Word> {
    /// Load a value from memory.
    fn load(&mut self, twist_id: TwistId, addr: Addr) -> Word;
    /// Store a value to memory.
    fn store(&mut self, twist_id: TwistId, addr: Addr, value: Word);

    /// Load a value from memory, pinning the access to a specific lane.
    ///
    /// This is a *hint* for tracing/witness assignment; memory implementations typically ignore it.
    #[inline]
    fn load_lane(&mut self, twist_id: TwistId, addr: Addr, _lane: u32) -> Word {
        self.load(twist_id, addr)
    }

    /// Store a value to memory, pinning the access to a specific lane.
    ///
    /// This is a *hint* for tracing/witness assignment; memory implementations typically ignore it.
    #[inline]
    fn store_lane(&mut self, twist_id: TwistId, addr: Addr, value: Word, _lane: u32) {
        self.store(twist_id, addr, value)
    }

    /// Conditionally load a value from memory.
    ///
    /// If `cond` is false, returns `default` and does not record a Twist event in tracing wrappers.
    #[inline]
    fn load_if(&mut self, cond: bool, twist_id: TwistId, addr: Addr, default: Word) -> Word {
        if cond { self.load(twist_id, addr) } else { default }
    }

    /// Conditionally store a value to memory.
    ///
    /// If `cond` is false, no store occurs and no Twist event is recorded in tracing wrappers.
    #[inline]
    fn store_if(&mut self, cond: bool, twist_id: TwistId, addr: Addr, value: Word) {
        if cond {
            self.store(twist_id, addr, value);
        }
    }

    /// Conditionally load a value from memory, pinning the access to a specific lane.
    ///
    /// If `cond` is false, returns `default` and does not record a Twist event in tracing wrappers.
    #[inline]
    fn load_if_lane(&mut self, cond: bool, twist_id: TwistId, addr: Addr, default: Word, lane: u32) -> Word {
        if cond {
            self.load_lane(twist_id, addr, lane)
        } else {
            default
        }
    }

    /// Conditionally store a value to memory, pinning the access to a specific lane.
    ///
    /// If `cond` is false, no store occurs and no Twist event is recorded in tracing wrappers.
    #[inline]
    fn store_if_lane(&mut self, cond: bool, twist_id: TwistId, addr: Addr, value: Word, lane: u32) {
        if cond {
            self.store_lane(twist_id, addr, value, lane);
        }
    }
}

/// A tracing wrapper around any `Twist` implementation.
///
/// Intercepts all load/store calls and records them as `TwistEvent`s.
/// Use `take_events()` after each step to collect the events.
pub struct TracingTwist<T, Addr, Word> {
    /// The underlying Twist implementation.
    pub inner: T,
    /// Events accumulated during the current step.
    pub current_step_events: Vec<TwistEvent<Addr, Word>>,
}

impl<T, Addr, Word> TracingTwist<T, Addr, Word> {
    /// Wrap a Twist implementation with tracing.
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            current_step_events: Vec::new(),
        }
    }

    /// Take all events accumulated since the last call to `take_events`.
    ///
    /// Call this after each CPU step to collect Twist events for that step.
    pub fn take_events(&mut self) -> Vec<TwistEvent<Addr, Word>> {
        std::mem::take(&mut self.current_step_events)
    }

    /// Returns a reference to the underlying Twist.
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Returns a mutable reference to the underlying Twist.
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

impl<T, Addr, Word> Twist<Addr, Word> for TracingTwist<T, Addr, Word>
where
    T: Twist<Addr, Word>,
    Addr: Copy,
    Word: Copy,
{
    fn load(&mut self, twist_id: TwistId, addr: Addr) -> Word {
        let v = self.inner.load(twist_id, addr);
        self.current_step_events.push(TwistEvent {
            twist_id,
            kind: TwistOpKind::Read,
            addr,
            value: v,
            lane: None,
        });
        v
    }

    fn store(&mut self, twist_id: TwistId, addr: Addr, value: Word) {
        self.inner.store(twist_id, addr, value);
        self.current_step_events.push(TwistEvent {
            twist_id,
            kind: TwistOpKind::Write,
            addr,
            value,
            lane: None,
        });
    }

    fn load_lane(&mut self, twist_id: TwistId, addr: Addr, lane: u32) -> Word {
        let v = self.inner.load_lane(twist_id, addr, lane);
        self.current_step_events.push(TwistEvent {
            twist_id,
            kind: TwistOpKind::Read,
            addr,
            value: v,
            lane: Some(lane),
        });
        v
    }

    fn store_lane(&mut self, twist_id: TwistId, addr: Addr, value: Word, lane: u32) {
        self.inner.store_lane(twist_id, addr, value, lane);
        self.current_step_events.push(TwistEvent {
            twist_id,
            kind: TwistOpKind::Write,
            addr,
            value,
            lane: Some(lane),
        });
    }
}

// ============================================================================
// Shout Trait and Tracing Wrapper (Read-Only Lookups)
// ============================================================================

/// Abstraction for read-only lookup tables (Shout).
///
/// Used for precomputed functions like byte decomposition, range checks,
/// or any read-only data the VM needs to query.
///
/// # Requirements
///
/// - `lookup` must be deterministic: same key always returns same value.
/// - Tables are immutable during execution.
pub trait Shout<Word> {
    /// Look up a value in a table.
    fn lookup(&mut self, shout_id: ShoutId, key: Word) -> Word;
}

/// A tracing wrapper around any `Shout` implementation.
///
/// Intercepts all lookup calls and records them as `ShoutEvent`s.
/// Use `take_events()` after each step to collect the events.
pub struct TracingShout<S, Word> {
    /// The underlying Shout implementation.
    pub inner: S,
    /// Events accumulated during the current step.
    pub current_step_events: Vec<ShoutEvent<Word>>,
}

impl<S, Word> TracingShout<S, Word>
where
    Word: Copy,
{
    /// Wrap a Shout implementation with tracing.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            current_step_events: Vec::new(),
        }
    }

    /// Take all events accumulated since the last call to `take_events`.
    ///
    /// Call this after each CPU step to collect Shout events for that step.
    pub fn take_events(&mut self) -> Vec<ShoutEvent<Word>> {
        std::mem::take(&mut self.current_step_events)
    }

    /// Returns a reference to the underlying Shout.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Returns a mutable reference to the underlying Shout.
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }
}

impl<S, Word> Shout<Word> for TracingShout<S, Word>
where
    S: Shout<Word>,
    Word: Copy,
{
    fn lookup(&mut self, shout_id: ShoutId, key: Word) -> Word {
        let v = self.inner.lookup(shout_id, key);
        self.current_step_events.push(ShoutEvent {
            shout_id,
            key,
            value: v,
        });
        v
    }
}

// ============================================================================
// CPU Trait
// ============================================================================

/// Metadata returned from a single CPU step.
#[derive(Copy, Clone, Debug)]
pub struct StepMeta<Addr> {
    /// The program counter after executing the instruction.
    pub pc_after: Addr,
    /// The opcode that was executed.
    pub opcode: u32,
}

/// Abstraction for a CPU that can be traced.
///
/// Implement this trait for your concrete VM to enable tracing.
///
/// # Example
///
/// ```ignore
/// struct MyCpu {
///     pc: u64,
///     regs: [u64; 32],
///     halted: bool,
/// }
///
/// impl VmCpu<u64, u64> for MyCpu {
///     type Error = String;
///
///     fn snapshot_regs(&self) -> Vec<u64> {
///         self.regs.to_vec()
///     }
///
///     fn pc(&self) -> u64 {
///         self.pc
///     }
///
///     fn halted(&self) -> bool {
///         self.halted
///     }
///
///     fn step<T, S>(&mut self, twist: &mut T, shout: &mut S) -> Result<StepMeta<u64>, Self::Error>
///     where
///         T: Twist<u64, u64>,
///         S: Shout<u64>,
///     {
///         // Fetch, decode, execute...
///         Ok(StepMeta { pc_after: self.pc, opcode: 0 })
///     }
/// }
/// ```
pub trait VmCpu<Addr, Word> {
    /// The error type for step execution failures.
    type Error: std::fmt::Debug + std::fmt::Display;

    /// Snapshot the current register state.
    ///
    /// Called before and after each step to record register changes.
    fn snapshot_regs(&self) -> Vec<Word>;

    /// Get the current program counter.
    fn pc(&self) -> Addr;

    /// Check if the CPU has halted.
    fn halted(&self) -> bool;

    /// Execute one instruction.
    ///
    /// The CPU should:
    /// 1. Fetch the instruction at `pc()`
    /// 2. Decode it
    /// 3. Execute it, using `twist` for memory ops and `shout` for lookups
    /// 4. Update internal state (PC, registers, flags)
    /// 5. Return the new PC and opcode
    ///
    /// # Errors
    ///
    /// Return an error for illegal instructions, traps, or other failures.
    fn step<T, S>(&mut self, twist: &mut T, shout: &mut S) -> Result<StepMeta<Addr>, Self::Error>
    where
        T: Twist<Addr, Word>,
        S: Shout<Word>;
}

// ============================================================================
// Trace Driver
// ============================================================================

/// Run a VM and produce an execution trace.
///
/// This is the main entry point for tracing. It:
/// 1. Wraps the Twist and Shout implementations with tracing wrappers
/// 2. Runs the CPU for up to `max_steps` cycles
/// 3. Records all state changes and side effects
/// 4. Returns the complete trace
///
/// # Arguments
///
/// * `cpu` - The CPU implementation (consumed)
/// * `twist` - The Twist (memory) implementation (consumed)
/// * `shout` - The Shout (lookup tables) implementation (consumed)
/// * `max_steps` - Maximum number of steps to execute
///
/// # Returns
///
/// The complete execution trace, or an error if the CPU fails.
///
/// # Example
///
/// ```ignore
/// let trace = trace_program(my_cpu, my_twist, my_shout, 1000)?;
/// println!("Executed {} steps", trace.len());
/// if trace.did_halt() {
///     println!("CPU halted normally");
/// }
/// ```
pub fn trace_program<Cpu, Tw, Sh, Addr, Word>(
    mut cpu: Cpu,
    twist: Tw,
    shout: Sh,
    max_steps: usize,
) -> Result<VmTrace<Addr, Word>, Cpu::Error>
where
    Cpu: VmCpu<Addr, Word>,
    Tw: Twist<Addr, Word>,
    Sh: Shout<Word>,
    Addr: Copy,
    Word: Copy,
{
    let mut trace = VmTrace::new();
    let mut tracing_twist = TracingTwist::new(twist);
    let mut tracing_shout = TracingShout::new(shout);

    // If the CPU is already halted, emit a single no-op snapshot step so downstream
    // padding logic (fixed-N execution models) can still operate deterministically.
    if max_steps > 0 && cpu.halted() {
        let regs = cpu.snapshot_regs();
        let pc = cpu.pc();
        trace.steps.push(StepTrace {
            cycle: 0,
            pc_before: pc,
            pc_after: pc,
            opcode: 0,
            regs_before: regs.clone(),
            regs_after: regs,
            twist_events: Vec::new(),
            shout_events: Vec::new(),
            halted: true,
        });
        return Ok(trace);
    }

    for cycle in 0..max_steps {
        if cpu.halted() {
            break;
        }

        let regs_before = cpu.snapshot_regs();
        let pc_before = cpu.pc();

        let meta = cpu.step(&mut tracing_twist, &mut tracing_shout)?;

        let regs_after = cpu.snapshot_regs();
        let twist_events = tracing_twist.take_events();
        let shout_events = tracing_shout.take_events();
        let halted = cpu.halted();

        trace.steps.push(StepTrace {
            cycle: cycle as u64,
            pc_before,
            pc_after: meta.pc_after,
            opcode: meta.opcode,
            regs_before,
            regs_after,
            twist_events,
            shout_events,
            halted,
        });

        if halted {
            break;
        }
    }

    Ok(trace)
}

// ============================================================================
// Utility Implementations
// ============================================================================

/// Test utilities for neo-vm-trace.
///
/// Enable with the `test-utils` feature or in tests.
#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils {
    use super::*;
    use std::collections::HashMap;

    /// Simple hash-map based Twist (memory) for testing.
    pub struct SimpleTwist<Addr, Word> {
        data: HashMap<(TwistId, Addr), Word>,
        default: Word,
    }

    impl<Addr, Word> SimpleTwist<Addr, Word>
    where
        Addr: std::hash::Hash + Eq + Copy,
        Word: Copy,
    {
        /// Create a new Twist with the given default value.
        pub fn new(default: Word) -> Self {
            Self {
                data: HashMap::new(),
                default,
            }
        }

        /// Set an initial value.
        pub fn set(&mut self, twist_id: TwistId, addr: Addr, value: Word) {
            self.data.insert((twist_id, addr), value);
        }

        /// Get a value (for inspection, not traced).
        pub fn get(&self, twist_id: TwistId, addr: Addr) -> Word {
            self.data
                .get(&(twist_id, addr))
                .copied()
                .unwrap_or(self.default)
        }
    }

    impl<Addr, Word> Twist<Addr, Word> for SimpleTwist<Addr, Word>
    where
        Addr: std::hash::Hash + Eq + Copy,
        Word: Copy,
    {
        fn load(&mut self, twist_id: TwistId, addr: Addr) -> Word {
            self.data
                .get(&(twist_id, addr))
                .copied()
                .unwrap_or(self.default)
        }

        fn store(&mut self, twist_id: TwistId, addr: Addr, value: Word) {
            self.data.insert((twist_id, addr), value);
        }
    }

    /// Simple vector-based Shout (lookup tables) for testing.
    pub struct SimpleShout<Word> {
        tables: HashMap<ShoutId, Vec<Word>>,
        default: Word,
    }

    impl<Word> SimpleShout<Word>
    where
        Word: Copy,
    {
        /// Create new Shout tables with the given default value.
        pub fn new(default: Word) -> Self {
            Self {
                tables: HashMap::new(),
                default,
            }
        }

        /// Add a table.
        pub fn add_table(&mut self, shout_id: ShoutId, data: Vec<Word>) {
            self.tables.insert(shout_id, data);
        }
    }

    impl Shout<u64> for SimpleShout<u64> {
        fn lookup(&mut self, shout_id: ShoutId, key: u64) -> u64 {
            self.tables
                .get(&shout_id)
                .and_then(|t| t.get(key as usize).copied())
                .unwrap_or(self.default)
        }
    }

    /// A simple counter CPU for testing.
    ///
    /// This CPU just increments a counter register on each step.
    pub struct CounterCpu {
        pub pc: u64,
        pub counter: u64,
        pub max_count: u64,
        pub halted: bool,
    }

    impl CounterCpu {
        /// Create a new counter CPU that halts after `max_count` steps.
        pub fn new(max_count: u64) -> Self {
            Self {
                pc: 0,
                counter: 0,
                max_count,
                halted: false,
            }
        }
    }

    impl VmCpu<u64, u64> for CounterCpu {
        type Error = String;

        fn snapshot_regs(&self) -> Vec<u64> {
            vec![self.counter]
        }

        fn pc(&self) -> u64 {
            self.pc
        }

        fn halted(&self) -> bool {
            self.halted
        }

        fn step<T, S>(&mut self, _twist: &mut T, _shout: &mut S) -> Result<StepMeta<u64>, Self::Error>
        where
            T: Twist<u64, u64>,
            S: Shout<u64>,
        {
            self.counter += 1;
            self.pc += 4;
            if self.counter >= self.max_count {
                self.halted = true;
            }
            Ok(StepMeta {
                pc_after: self.pc,
                opcode: 1,
            })
        }
    }

    /// A CPU that performs Twist and Shout operations for testing.
    pub struct TwistShoutTestCpu {
        pub pc: u64,
        pub step_count: u64,
        pub max_steps: u64,
        pub halted: bool,
    }

    impl TwistShoutTestCpu {
        /// Create a new test CPU.
        pub fn new(max_steps: u64) -> Self {
            Self {
                pc: 0,
                step_count: 0,
                max_steps,
                halted: false,
            }
        }
    }

    impl VmCpu<u64, u64> for TwistShoutTestCpu {
        type Error = String;

        fn snapshot_regs(&self) -> Vec<u64> {
            vec![self.step_count]
        }

        fn pc(&self) -> u64 {
            self.pc
        }

        fn halted(&self) -> bool {
            self.halted
        }

        fn step<T, S>(&mut self, twist: &mut T, shout: &mut S) -> Result<StepMeta<u64>, Self::Error>
        where
            T: Twist<u64, u64>,
            S: Shout<u64>,
        {
            let ram = TwistId(1);
            let table = ShoutId(0);

            // Write step_count to address step_count
            twist.store(ram, self.step_count, self.step_count * 10);

            // Read it back
            let _val = twist.load(ram, self.step_count);

            // Do a Shout lookup
            let _lookup_val = shout.lookup(table, self.step_count % 256);

            self.step_count += 1;
            self.pc += 4;

            if self.step_count >= self.max_steps {
                self.halted = true;
            }

            Ok(StepMeta {
                pc_after: self.pc,
                opcode: 2,
            })
        }
    }
}
