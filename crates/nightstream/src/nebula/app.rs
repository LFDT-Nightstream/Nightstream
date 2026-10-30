//! The contract between Nebula memory checking and an application (spec §10),
//! the Rust mirror of Lean `PortApp` (`Lifecycle/Nebula/PortApp.lean`). The
//! memory rows own the operation slots and the carry. An application owns its
//! state words, its private words, its rows over a `View`, and its native
//! step. The segment runner applies the step's ports to memory, as Lean
//! `Executes` applies `accessAll`. This file owns no row and no application.

use std::fmt::Debug;

use p3_goldilocks::Goldilocks as F;

use super::expr::Expr;
use super::geometry::Geometry;
use super::records::PortAccess;
use super::rows::View;
use super::segment::Cell;

/// An application of the memory phase: the toy `Machine` now, and next the
/// WASM relation and the Starstream interleaving circuit. Lean proves the
/// memory rows for any `PortApp`; an application's own rows need its own Ob7
/// and Ob5 (`PortApp.sound`, `PortApp.complete`).
pub trait Application {
    /// The host's choice for one step.
    type Input;
    /// The state that a statement carries. `words` encodes it.
    type State: Clone + Debug + Eq;
    /// Why the application has no step from a state and an input.
    type Error: std::error::Error + Send + Sync + 'static;

    /// The number of state words (Lean `AppShape.width`).
    fn width(&self) -> usize;

    /// The number of private words (Lean `AppShape.aux`).
    fn aux(&self) -> usize;

    /// The number of ports of every step (Lean `PortApp.portsLength`). The
    /// application's rows must fix each of them. A memory circuit needs a
    /// geometry with this `b_ops`: a slot that no row fixes would let a step
    /// change memory.
    fn port_count(&self) -> usize;

    /// The `width` state words of `state`. The terminal checks reject a
    /// statement with a word at or above `q` (Ob1).
    fn words(&self, state: &Self::State) -> Vec<u64>;

    /// The application's rows over the step's view (Ob7). Each row is zero
    /// on an honest step.
    fn rows(&self, view: &View) -> Vec<Expr>;

    /// One native step from `state`. `memory` holds the cells before the
    /// step's accesses. The runner applies the active ports in slot order and
    /// rejects a port that is not valid or that reads another value than its
    /// cell holds.
    fn step(
        &self,
        state: &Self::State,
        input: &Self::Input,
        memory: &Memory,
    ) -> Result<Transition<Self::State>, Self::Error>;
}

/// One native step: the next state, one port for each operation slot
/// (`None` for an inactive slot), and the private words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transition<S> {
    pub next: S,
    pub ports: Vec<Option<PortAccess>>,
    pub aux: Vec<F>,
}

/// The memory before a step's accesses: the ROM cells, then the RAM cells.
pub struct Memory<'a> {
    geometry: &'a Geometry,
    cells: &'a [Cell],
}

impl<'a> Memory<'a> {
    pub(super) fn new(geometry: &'a Geometry, cells: &'a [Cell]) -> Self {
        Self { geometry, cells }
    }

    /// `ROM[addr]`, or `None` outside the ROM.
    pub fn rom(&self, addr: u64) -> Option<u64> {
        self.read(0, self.geometry.rom_size(), addr)
    }

    /// `RAM[addr]`, or `None` outside the RAM.
    pub fn ram(&self, addr: u64) -> Option<u64> {
        self.read(self.geometry.rom_size(), self.geometry.ram_size(), addr)
    }

    fn read(&self, start: usize, size: usize, addr: u64) -> Option<u64> {
        let addr = usize::try_from(addr).ok().filter(|addr| *addr < size)?;
        Some(self.cells[start + addr].value)
    }
}
