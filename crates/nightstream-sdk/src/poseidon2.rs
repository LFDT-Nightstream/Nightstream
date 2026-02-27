//! Poseidon2 precompile helpers for Nightstream RISC-V guests.

/// Poseidon2 digest over Goldilocks field elements.
pub type GlDigest = [u64; 4];

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const CUSTOM0_OPCODE: u32 = 0x0B;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const P2_ABSORB_FUNCT7: u32 = 0x00;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const P2_FINALIZE_FUNCT7: u32 = 0x01;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const P2_SQUEEZE_FUNCT7: u32 = 0x02;

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const REG_X0: u32 = 0;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const REG_A0: u32 = 10;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const REG_A1: u32 = 11;

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const fn encode_r(funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    ((funct7 & 0x7f) << 25)
        | ((rs2 & 0x1f) << 20)
        | ((rs1 & 0x1f) << 15)
        | ((funct3 & 0x7) << 12)
        | ((rd & 0x1f) << 7)
        | (opcode & 0x7f)
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const P2_ABSORB_A0_A1: u32 = encode_r(P2_ABSORB_FUNCT7, REG_A1, REG_A0, 0, REG_X0, CUSTOM0_OPCODE);
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const P2_FINALIZE: u32 = encode_r(P2_FINALIZE_FUNCT7, REG_X0, REG_X0, 0, REG_X0, CUSTOM0_OPCODE);
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const P2_SQUEEZE_IDX0_A0: u32 = encode_r(P2_SQUEEZE_FUNCT7, REG_X0, REG_X0, 0, REG_A0, CUSTOM0_OPCODE);
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const P2_SQUEEZE_IDX1_A0: u32 = encode_r(P2_SQUEEZE_FUNCT7, REG_X0, REG_X0, 1, REG_A0, CUSTOM0_OPCODE);
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const P2_SQUEEZE_IDX2_A0: u32 = encode_r(P2_SQUEEZE_FUNCT7, REG_X0, REG_X0, 2, REG_A0, CUSTOM0_OPCODE);
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const P2_SQUEEZE_IDX3_A0: u32 = encode_r(P2_SQUEEZE_FUNCT7, REG_X0, REG_X0, 3, REG_A0, CUSTOM0_OPCODE);
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const P2_SQUEEZE_IDX4_A0: u32 = encode_r(P2_SQUEEZE_FUNCT7, REG_X0, REG_X0, 4, REG_A0, CUSTOM0_OPCODE);
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const P2_SQUEEZE_IDX5_A0: u32 = encode_r(P2_SQUEEZE_FUNCT7, REG_X0, REG_X0, 5, REG_A0, CUSTOM0_OPCODE);
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const P2_SQUEEZE_IDX6_A0: u32 = encode_r(P2_SQUEEZE_FUNCT7, REG_X0, REG_X0, 6, REG_A0, CUSTOM0_OPCODE);
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const P2_SQUEEZE_IDX7_A0: u32 = encode_r(P2_SQUEEZE_FUNCT7, REG_X0, REG_X0, 7, REG_A0, CUSTOM0_OPCODE);

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline]
pub fn poseidon2_absorb_elem(elem: u64) {
    let lo = elem as u32;
    let hi = (elem >> 32) as u32;
    unsafe {
        core::arch::asm!(
            ".word {insn}",
            insn = const P2_ABSORB_A0_A1,
            in("a0") lo,
            in("a1") hi,
            options(nostack),
        );
    }
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline]
pub fn poseidon2_finalize() {
    unsafe {
        core::arch::asm!(
            ".word {insn}",
            insn = const P2_FINALIZE,
            options(nostack),
        );
    }
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline]
fn squeeze_idx0() -> u32 {
    let out: u32;
    unsafe {
        core::arch::asm!(
            ".word {insn}",
            insn = const P2_SQUEEZE_IDX0_A0,
            lateout("a0") out,
            options(nostack),
        );
    }
    out
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline]
fn squeeze_idx1() -> u32 {
    let out: u32;
    unsafe {
        core::arch::asm!(
            ".word {insn}",
            insn = const P2_SQUEEZE_IDX1_A0,
            lateout("a0") out,
            options(nostack),
        );
    }
    out
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline]
fn squeeze_idx2() -> u32 {
    let out: u32;
    unsafe {
        core::arch::asm!(
            ".word {insn}",
            insn = const P2_SQUEEZE_IDX2_A0,
            lateout("a0") out,
            options(nostack),
        );
    }
    out
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline]
fn squeeze_idx3() -> u32 {
    let out: u32;
    unsafe {
        core::arch::asm!(
            ".word {insn}",
            insn = const P2_SQUEEZE_IDX3_A0,
            lateout("a0") out,
            options(nostack),
        );
    }
    out
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline]
fn squeeze_idx4() -> u32 {
    let out: u32;
    unsafe {
        core::arch::asm!(
            ".word {insn}",
            insn = const P2_SQUEEZE_IDX4_A0,
            lateout("a0") out,
            options(nostack),
        );
    }
    out
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline]
fn squeeze_idx5() -> u32 {
    let out: u32;
    unsafe {
        core::arch::asm!(
            ".word {insn}",
            insn = const P2_SQUEEZE_IDX5_A0,
            lateout("a0") out,
            options(nostack),
        );
    }
    out
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline]
fn squeeze_idx6() -> u32 {
    let out: u32;
    unsafe {
        core::arch::asm!(
            ".word {insn}",
            insn = const P2_SQUEEZE_IDX6_A0,
            lateout("a0") out,
            options(nostack),
        );
    }
    out
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline]
fn squeeze_idx7() -> u32 {
    let out: u32;
    unsafe {
        core::arch::asm!(
            ".word {insn}",
            insn = const P2_SQUEEZE_IDX7_A0,
            lateout("a0") out,
            options(nostack),
        );
    }
    out
}

/// Return one digest word (`idx` in 0..=7) from the finalized Poseidon2 state.
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline]
pub fn poseidon2_squeeze_word(idx: u8) -> u32 {
    match idx {
        0 => squeeze_idx0(),
        1 => squeeze_idx1(),
        2 => squeeze_idx2(),
        3 => squeeze_idx3(),
        4 => squeeze_idx4(),
        5 => squeeze_idx5(),
        6 => squeeze_idx6(),
        7 => squeeze_idx7(),
        _ => panic!("poseidon2_squeeze_word: idx out of range"),
    }
}

/// Hash a variable-length list of Goldilocks elements via precompile instructions.
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
pub fn poseidon2_hash(input: &[u64]) -> GlDigest {
    for &elem in input {
        poseidon2_absorb_elem(elem);
    }
    poseidon2_finalize();
    let mut words = [0u32; 8];
    for (idx, slot) in words.iter_mut().enumerate() {
        *slot = poseidon2_squeeze_word(idx as u8);
    }
    [
        (words[0] as u64) | ((words[1] as u64) << 32),
        (words[2] as u64) | ((words[3] as u64) << 32),
        (words[4] as u64) | ((words[5] as u64) << 32),
        (words[6] as u64) | ((words[7] as u64) << 32),
    ]
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
#[inline]
pub fn poseidon2_absorb_elem(_elem: u64) {
    unimplemented!("poseidon2 precompile is only available on RISC-V targets");
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
#[inline]
pub fn poseidon2_finalize() {
    unimplemented!("poseidon2 precompile is only available on RISC-V targets");
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
#[inline]
pub fn poseidon2_squeeze_word(_idx: u8) -> u32 {
    unimplemented!("poseidon2 precompile is only available on RISC-V targets");
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
pub fn poseidon2_hash(_input: &[u64]) -> GlDigest {
    unimplemented!("poseidon2 precompile is only available on RISC-V targets");
}
