pub(super) const RV32_XLEN: usize = 32;

// Canonical RV32 Shout table IDs (must match `RiscvShoutTables::opcode_to_id`).
pub(super) const AND_TABLE_ID: u32 = 0; // `RiscvOpcode::And`
pub(super) const XOR_TABLE_ID: u32 = 1; // `RiscvOpcode::Xor`
pub(super) const OR_TABLE_ID: u32 = 2; // `RiscvOpcode::Or`
pub(super) const ADD_TABLE_ID: u32 = 3; // `RiscvOpcode::Add`
pub(super) const SUB_TABLE_ID: u32 = 4; // `RiscvOpcode::Sub`
pub(super) const SLT_TABLE_ID: u32 = 5; // `RiscvOpcode::Slt`
pub(super) const SLTU_TABLE_ID: u32 = 6; // `RiscvOpcode::Sltu`
pub(super) const SLL_TABLE_ID: u32 = 7; // `RiscvOpcode::Sll`
pub(super) const SRL_TABLE_ID: u32 = 8; // `RiscvOpcode::Srl`
pub(super) const SRA_TABLE_ID: u32 = 9; // `RiscvOpcode::Sra`
pub(super) const EQ_TABLE_ID: u32 = 10; // `RiscvOpcode::Eq`
pub(super) const NEQ_TABLE_ID: u32 = 11; // `RiscvOpcode::Neq`

// RV32M (R-type, funct7 = 0b0000001).
pub(super) const MUL_TABLE_ID: u32 = 12; // `RiscvOpcode::Mul`
pub(super) const MULH_TABLE_ID: u32 = 13; // `RiscvOpcode::Mulh`
pub(super) const MULHU_TABLE_ID: u32 = 14; // `RiscvOpcode::Mulhu`
pub(super) const MULHSU_TABLE_ID: u32 = 15; // `RiscvOpcode::Mulhsu`
pub(super) const DIV_TABLE_ID: u32 = 16; // `RiscvOpcode::Div`
pub(super) const DIVU_TABLE_ID: u32 = 17; // `RiscvOpcode::Divu`
pub(super) const REM_TABLE_ID: u32 = 18; // `RiscvOpcode::Rem`
pub(super) const REMU_TABLE_ID: u32 = 19; // `RiscvOpcode::Remu`
