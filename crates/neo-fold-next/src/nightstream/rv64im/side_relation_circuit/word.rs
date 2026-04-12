//! Owns exact `u32`/`u64` witness packing for side-relation transcript gadgets.

use bellpepper_core::{boolean::Boolean, num::AllocatedNum, ConstraintSystem, SynthesisError};
use spartan2::provider::goldi::F as SpartanF;

#[derive(Clone)]
pub struct U8Var {
    pub num: AllocatedNum<SpartanF>,
    pub value: u8,
}

#[derive(Clone)]
pub struct U16Var {
    pub num: AllocatedNum<SpartanF>,
    pub value: u16,
    pub lo8: U8Var,
    pub hi8: U8Var,
}

#[derive(Clone)]
pub struct U32Var {
    pub num: AllocatedNum<SpartanF>,
    pub value: u32,
    pub lo16: U16Var,
    pub hi16: U16Var,
}

#[derive(Clone)]
pub struct U64Var {
    pub lo: U32Var,
    pub hi: U32Var,
}

impl U32Var {
    pub fn field_value(&self) -> SpartanF {
        SpartanF::from_canonical_u64(self.value as u64)
    }

    pub fn limb16_vars(&self) -> [AllocatedNum<SpartanF>; 2] {
        [self.lo16.num.clone(), self.hi16.num.clone()]
    }

    pub fn limb16_values(&self) -> [SpartanF; 2] {
        [self.lo16.field_value(), self.hi16.field_value()]
    }

    pub fn byte_vars(&self) -> [AllocatedNum<SpartanF>; 4] {
        let [b0, b1] = self.lo16.byte_vars();
        let [b2, b3] = self.hi16.byte_vars();
        [b0, b1, b2, b3]
    }

    pub fn byte_values(&self) -> [SpartanF; 4] {
        let [b0, b1] = self.lo16.byte_values();
        let [b2, b3] = self.hi16.byte_values();
        [b0, b1, b2, b3]
    }
}

impl U64Var {
    pub fn half_vars(&self) -> [AllocatedNum<SpartanF>; 2] {
        [self.lo.num.clone(), self.hi.num.clone()]
    }

    pub fn half_values(&self) -> [SpartanF; 2] {
        [self.lo.field_value(), self.hi.field_value()]
    }

    pub fn limb16_vars(&self) -> [AllocatedNum<SpartanF>; 4] {
        let [lo0, lo1] = self.lo.limb16_vars();
        let [hi0, hi1] = self.hi.limb16_vars();
        [lo0, lo1, hi0, hi1]
    }

    pub fn limb16_values(&self) -> [SpartanF; 4] {
        let [lo0, lo1] = self.lo.limb16_values();
        let [hi0, hi1] = self.hi.limb16_values();
        [lo0, lo1, hi0, hi1]
    }

    pub fn byte_vars(&self) -> [AllocatedNum<SpartanF>; 8] {
        let [b0, b1, b2, b3] = self.lo.byte_vars();
        let [b4, b5, b6, b7] = self.hi.byte_vars();
        [b0, b1, b2, b3, b4, b5, b6, b7]
    }

    pub fn byte_values(&self) -> [SpartanF; 8] {
        let [b0, b1, b2, b3] = self.lo.byte_values();
        let [b4, b5, b6, b7] = self.hi.byte_values();
        [b0, b1, b2, b3, b4, b5, b6, b7]
    }
}

impl U16Var {
    pub fn field_value(&self) -> SpartanF {
        SpartanF::from_canonical_u64(self.value as u64)
    }

    pub fn byte_vars(&self) -> [AllocatedNum<SpartanF>; 2] {
        [self.lo8.num.clone(), self.hi8.num.clone()]
    }

    pub fn byte_values(&self) -> [SpartanF; 2] {
        [self.lo8.field_value(), self.hi8.field_value()]
    }
}

impl U8Var {
    pub fn field_value(&self) -> SpartanF {
        SpartanF::from_canonical_u64(self.value as u64)
    }
}

pub fn alloc_u32<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    value: u32,
    label: &str,
) -> Result<U32Var, SynthesisError> {
    let num = AllocatedNum::alloc(cs.namespace(|| format!("{label}_alloc")), || {
        Ok(SpartanF::from_canonical_u64(value as u64))
    })?;
    let bits = num.to_bits_le_strict(cs.namespace(|| format!("{label}_bits")))?;
    for (bit_idx, bit) in bits.iter().enumerate().skip(32) {
        Boolean::enforce_equal(
            cs.namespace(|| format!("{label}_high_zero_{bit_idx}")),
            bit,
            &Boolean::constant(false),
        )?;
    }
    let lo16 = alloc_u16(
        cs.namespace(|| format!("{label}_lo16")),
        value as u16,
        &format!("{label}_lo16"),
    )?;
    let hi16 = alloc_u16(
        cs.namespace(|| format!("{label}_hi16")),
        (value >> 16) as u16,
        &format!("{label}_hi16"),
    )?;
    cs.enforce(
        || format!("{label}_limb_recompose"),
        |lc| lc + lo16.num.get_variable() + (SpartanF::from_canonical_u64(1u64 << 16), hi16.num.get_variable()),
        |lc| lc + CS::one(),
        |lc| lc + num.get_variable(),
    );
    Ok(U32Var { num, value, lo16, hi16 })
}

pub fn alloc_u64<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    value: u64,
    label: &str,
) -> Result<U64Var, SynthesisError> {
    Ok(U64Var {
        lo: alloc_u32(
            cs.namespace(|| format!("{label}_lo")),
            value as u32,
            &format!("{label}_lo"),
        )?,
        hi: alloc_u32(
            cs.namespace(|| format!("{label}_hi")),
            (value >> 32) as u32,
            &format!("{label}_hi"),
        )?,
    })
}

fn alloc_u16<CS: ConstraintSystem<SpartanF>>(mut cs: CS, value: u16, label: &str) -> Result<U16Var, SynthesisError> {
    let num = AllocatedNum::alloc(cs.namespace(|| format!("{label}_alloc")), || {
        Ok(SpartanF::from_canonical_u64(value as u64))
    })?;
    let bits = num.to_bits_le_strict(cs.namespace(|| format!("{label}_bits")))?;
    for (bit_idx, bit) in bits.iter().enumerate().skip(16) {
        Boolean::enforce_equal(
            cs.namespace(|| format!("{label}_high_zero_{bit_idx}")),
            bit,
            &Boolean::constant(false),
        )?;
    }
    let lo8 = alloc_u8(
        cs.namespace(|| format!("{label}_lo8")),
        value as u8,
        &format!("{label}_lo8"),
    )?;
    let hi8 = alloc_u8(
        cs.namespace(|| format!("{label}_hi8")),
        (value >> 8) as u8,
        &format!("{label}_hi8"),
    )?;
    cs.enforce(
        || format!("{label}_byte_recompose"),
        |lc| lc + lo8.num.get_variable() + (SpartanF::from_canonical_u64(1u64 << 8), hi8.num.get_variable()),
        |lc| lc + CS::one(),
        |lc| lc + num.get_variable(),
    );
    Ok(U16Var { num, value, lo8, hi8 })
}

fn alloc_u8<CS: ConstraintSystem<SpartanF>>(mut cs: CS, value: u8, label: &str) -> Result<U8Var, SynthesisError> {
    let num = AllocatedNum::alloc(cs.namespace(|| format!("{label}_alloc")), || {
        Ok(SpartanF::from_canonical_u64(value as u64))
    })?;
    let bits = num.to_bits_le_strict(cs.namespace(|| format!("{label}_bits")))?;
    for (bit_idx, bit) in bits.iter().enumerate().skip(8) {
        Boolean::enforce_equal(
            cs.namespace(|| format!("{label}_high_zero_{bit_idx}")),
            bit,
            &Boolean::constant(false),
        )?;
    }
    Ok(U8Var { num, value })
}
