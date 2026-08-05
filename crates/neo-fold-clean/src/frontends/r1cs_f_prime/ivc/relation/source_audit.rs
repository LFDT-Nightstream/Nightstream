//! Source-coordinate and source-row records exported by bounded relation audits.
//!
//! Owns compact audit data and read-only accessors. Relation discovery and
//! construction remain in the parent module.

/// Exact final-slot encoding used by one authoritative incoming running-X
/// coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum R1csIvcRawRunningEncodingAudit {
    /// The field element is stored directly in one signed-unit coordinate.
    CenteredScalar,
    /// The field element is reconstructed as `sum(digit[i] * 3^i)` from the
    /// canonical 41-coordinate signed balanced-ternary representation.
    BalancedTernary,
    /// The canonical field representative is reconstructed from little-endian
    /// Boolean coordinates.
    Binary,
}

/// Exact fixed-profile path from one authoritative incoming running-X
/// coordinate through the normalized source arm to its complete final
/// selective-assignment encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct R1csIvcRawRunningAssignmentAudit {
    pub(super) child: usize,
    pub(super) logical_column: usize,
    pub(super) source_column: usize,
    pub(super) final_start: usize,
    pub(super) width: usize,
    pub(super) encoding: R1csIvcRawRunningEncodingAudit,
}

impl R1csIvcRawRunningAssignmentAudit {
    pub fn child(self) -> usize {
        self.child
    }

    pub fn logical_column(self) -> usize {
        self.logical_column
    }

    pub fn source_column(self) -> usize {
        self.source_column
    }

    pub fn final_start(self) -> usize {
        self.final_start
    }

    pub fn width(self) -> usize {
        self.width
    }

    pub fn encoding(self) -> R1csIvcRawRunningEncodingAudit {
        self.encoding
    }
}
