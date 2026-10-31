//! The block kinds and wires of a recorded program, as integers.

/// The Poseidon2 state width. `neo-spartan` checks it against the workspace permutation.
pub const WIDTH: usize = 16;
/// The sponge and duplex rate, and the digest length.
pub const RATE: usize = 12;
pub const DIGEST_LEN: usize = 4;
pub const SBOXES: usize = 150;
/// A permutation block: its inputs, the S-box outputs in round order, and its outputs.
pub const CELLS: usize = WIDTH + SBOXES + WIDTH;
/// A permutation block: one row per S-box and one per output.
pub const ROWS: usize = SBOXES + WIDTH;
/// The CCS matrices that a block row reads: `X, A_0, B_0, A_1, B_1, A_2, B_2, C`.
pub const MATRICES: usize = 8;

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Permutation,
}

pub const KINDS: [Kind; 1] = [Kind::Permutation];

impl Kind {
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn cells(self) -> usize {
        match self {
            Kind::Permutation => CELLS,
        }
    }

    pub fn rows(self) -> usize {
        match self {
            Kind::Permutation => ROWS,
        }
    }
}

/// A witness cell or a public word. `Public(0)` is the constant one.
#[derive(Clone, Copy, PartialEq)]
pub enum Wire {
    Glue(u32),
    /// Kind, block, cell.
    Block(Kind, u32, u16),
    Public(u32),
}
