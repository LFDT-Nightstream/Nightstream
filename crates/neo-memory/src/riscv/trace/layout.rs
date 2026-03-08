#[derive(Clone, Debug)]
pub struct Rv32TraceLayout {
    pub cols: usize,

    // Core control / fetch.
    pub one: usize,
    pub active: usize,
    pub halted: usize,
    pub is_virtual: usize,
    pub virtual_sequence_remaining: usize,
    pub cycle: usize,
    pub pc_before: usize,
    pub pc_after: usize,
    pub instr_word: usize,

    // Regfile view (REG Twist).
    pub rs1_addr: usize,
    pub rs1_val: usize,
    pub rs2_addr: usize,
    pub rs2_val: usize,
    pub rd_addr: usize,
    pub rd_has_write: usize,
    pub rd_val: usize,

    // RAM view (RAM Twist, normalized to at most 1R + 1W per row).
    pub ram_addr: usize,
    pub ram_rv: usize,
    pub ram_wv: usize,

    // Shout view (single fixed-lane per row; output-only for now).
    pub shout_has_lookup: usize,
    pub shout_table_id: usize,
    pub shout_val: usize,
    pub shout_lhs: usize,
    pub shout_rhs: usize,
    pub shout_link_lhs: usize,
    pub shout_link_rhs: usize,
    pub shout_add_sub_key: usize,
    pub jalr_drop_bit: usize,
    /// `is_virtual(i) * (1 - is_virtual(i+1))` helper for transition constraints.
    pub virtual_transition: usize,
    /// `virtual_transition(i) * rd_has_write(i)` helper for virtual commit linkage.
    pub virtual_commit_link: usize,
    /// `virtual_commit_link(i-1)` exposed on row `i` for current-row helper-owned commit constraints.
    pub virtual_commit_from_prev: usize,
}

impl Rv32TraceLayout {
    pub fn new() -> Self {
        let mut next = 0usize;
        let mut take = || {
            let out = next;
            next += 1;
            out
        };

        let one = take();
        let active = take();
        let halted = take();
        let is_virtual = take();
        let virtual_sequence_remaining = take();
        let cycle = take();
        let pc_before = take();
        let pc_after = take();
        let instr_word = take();

        let rs1_addr = take();
        let rs1_val = take();
        let rs2_addr = take();
        let rs2_val = take();
        let rd_addr = take();
        let rd_has_write = take();
        let rd_val = take();

        let ram_addr = take();
        let ram_rv = take();
        let ram_wv = take();

        let shout_has_lookup = take();
        let shout_table_id = take();
        let shout_val = take();
        let shout_lhs = take();
        let shout_rhs = take();
        let shout_link_lhs = take();
        let shout_link_rhs = take();
        let shout_add_sub_key = take();
        let jalr_drop_bit = take();
        let virtual_transition = take();
        let virtual_commit_link = take();
        let virtual_commit_from_prev = take();

        debug_assert_eq!(next, 31, "RV32 trace width drift after virtual-step metadata columns");

        Self {
            cols: next,
            one,
            active,
            halted,
            is_virtual,
            virtual_sequence_remaining,
            cycle,
            pc_before,
            pc_after,
            instr_word,
            rs1_addr,
            rs1_val,
            rs2_addr,
            rs2_val,
            rd_addr,
            rd_has_write,
            rd_val,
            ram_addr,
            ram_rv,
            ram_wv,
            shout_has_lookup,
            shout_table_id,
            shout_val,
            shout_lhs,
            shout_rhs,
            shout_link_lhs,
            shout_link_rhs,
            shout_add_sub_key,
            jalr_drop_bit,
            virtual_transition,
            virtual_commit_link,
            virtual_commit_from_prev,
        }
    }
}
