//! In-circuit Poseidon2 transcript (Goldilocks, WIDTH=8, RATE=4).
//!
//! This mirrors `neo_transcript::Poseidon2Transcript` semantics:
//! - `append_message(label, msg)`: absorb label bytes, msg.len, msg bytes
//! - `append_fields(label, fs)`: absorb label bytes, fs.len, fs elements
//! - `challenge_field(label)`: append_message("chal/label", label), absorb 1, permute, output st[0]
//! - `challenge_fields(label, n)`: same, but squeeze 4 elems per permute
//! - `digest32()`: absorb 1, permute, output st[0..4] (32 bytes as 4 u64 limbs)
//!
//! Note: constants are allocated once and constrained, then re-used (to keep the CS lean).

use std::collections::BTreeMap;

use bellpepper_core::num::AllocatedNum;
use bellpepper_core::{ConstraintSystem, SynthesisError, Variable};

use crate::gadgets::sponge::Poseidon2Sponge;
use crate::CircuitF;

const APP_DOMAIN: &[u8] = b"neo/transcript/v1|poseidon2-goldilocks-w8-r4";

#[derive(Clone)]
pub struct Poseidon2TranscriptVar {
    sponge: Poseidon2Sponge,
    const_cache: BTreeMap<u64, AllocatedNum<CircuitF>>,
    scope: String,
}

impl Poseidon2TranscriptVar {
    pub fn new<CS: ConstraintSystem<CircuitF>>(
        cs: &mut CS,
        app_label: &'static [u8],
        label: &str,
    ) -> Result<Self, SynthesisError> {
        let mut tr = Self {
            sponge: Poseidon2Sponge::new(cs, label)?,
            const_cache: BTreeMap::new(),
            scope: label.to_owned(),
        };
        tr.append_message(cs, APP_DOMAIN, app_label, "init")?;
        Ok(tr)
    }

    fn const_num<CS: ConstraintSystem<CircuitF>>(
        &mut self,
        cs: &mut CS,
        v: u64,
    ) -> Result<AllocatedNum<CircuitF>, SynthesisError> {
        if let Some(existing) = self.const_cache.get(&v) {
            return Ok(existing.clone());
        }

        let val = CircuitF::from(v);
        let num = AllocatedNum::alloc(cs.namespace(|| format!("{}_const_{}", self.scope, v)), || Ok(val))?;
        cs.enforce(
            || format!("{}_const_{}_eq", self.scope, v),
            |lc| lc + num.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + (val, CS::one()),
        );
        self.const_cache.insert(v, num.clone());
        Ok(num)
    }

    fn absorb_const_u64<CS: ConstraintSystem<CircuitF>>(
        &mut self,
        cs: &mut CS,
        v: u64,
    ) -> Result<(), SynthesisError> {
        let num = self.const_num(cs, v)?;
        self.sponge.absorb(cs, num)?;
        Ok(())
    }

    fn absorb_var_with_value<CS: ConstraintSystem<CircuitF>>(
        &mut self,
        cs: &mut CS,
        var: Variable,
        value: CircuitF,
        label: &str,
    ) -> Result<(), SynthesisError> {
        let num = AllocatedNum::alloc(cs.namespace(|| format!("{}_{}", self.scope, label)), || Ok(value))?;
        cs.enforce(
            || format!("{}_{}_eq", self.scope, label),
            |lc| lc + num.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + var,
        );
        self.sponge.absorb_slice_elem(cs, num)?;
        Ok(())
    }

    pub fn append_message<CS: ConstraintSystem<CircuitF>>(
        &mut self,
        cs: &mut CS,
        label: &'static [u8],
        msg: &[u8],
        _ctx: &str,
    ) -> Result<(), SynthesisError> {
        for &b in label {
            self.absorb_const_u64(cs, b as u64)?;
        }
        self.absorb_const_u64(cs, msg.len() as u64)?;
        for &b in msg {
            self.absorb_const_u64(cs, b as u64)?;
        }
        Ok(())
    }

    /// Append a message where `msg` is provided as **allocated field elements** (byte values).
    ///
    /// This is used for bindings like `me_fold_digest`, where the bytes are derived and
    /// constrained elsewhere in the circuit and must not become witness-dependent constants.
    pub fn append_message_bytes_allocated<CS: ConstraintSystem<CircuitF>>(
        &mut self,
        cs: &mut CS,
        label: &'static [u8],
        msg: &[AllocatedNum<CircuitF>],
        _ctx: &str,
    ) -> Result<(), SynthesisError> {
        for &b in label {
            self.absorb_const_u64(cs, b as u64)?;
        }
        self.absorb_const_u64(cs, msg.len() as u64)?;
        for byte in msg {
            self.sponge.absorb(cs, byte.clone())?;
        }
        Ok(())
    }

    pub fn append_u64s<CS: ConstraintSystem<CircuitF>>(
        &mut self,
        cs: &mut CS,
        label: &'static [u8],
        us: &[u64],
        _ctx: &str,
    ) -> Result<(), SynthesisError> {
        for &b in label {
            self.absorb_const_u64(cs, b as u64)?;
        }
        self.absorb_const_u64(cs, us.len() as u64)?;
        for &u in us {
            self.absorb_const_u64(cs, u)?;
        }
        Ok(())
    }

    pub fn append_fields_allocated<CS: ConstraintSystem<CircuitF>>(
        &mut self,
        cs: &mut CS,
        label: &'static [u8],
        fs: &[AllocatedNum<CircuitF>],
        _ctx: &str,
    ) -> Result<(), SynthesisError> {
        for &b in label {
            self.absorb_const_u64(cs, b as u64)?;
        }
        self.absorb_const_u64(cs, fs.len() as u64)?;
        for f in fs {
            self.sponge.absorb_slice_elem(cs, f.clone())?;
        }
        Ok(())
    }

    pub fn append_fields_vars<CS: ConstraintSystem<CircuitF>>(
        &mut self,
        cs: &mut CS,
        label: &'static [u8],
        vars: &[Variable],
        values: &[CircuitF],
        ctx: &str,
    ) -> Result<(), SynthesisError> {
        if vars.len() != values.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        for &b in label {
            self.absorb_const_u64(cs, b as u64)?;
        }
        self.absorb_const_u64(cs, vars.len() as u64)?;
        for (i, (var, val)) in vars.iter().zip(values.iter()).enumerate() {
            self.absorb_var_with_value(cs, *var, *val, &format!("{ctx}_field_{i}"))?;
        }
        Ok(())
    }

    /// Append base-field elements that are known circuit constants (append_fields semantics).
    pub fn append_fields_u64s<CS: ConstraintSystem<CircuitF>>(
        &mut self,
        cs: &mut CS,
        label: &'static [u8],
        fs: &[u64],
        _ctx: &str,
    ) -> Result<(), SynthesisError> {
        for &b in label {
            self.absorb_const_u64(cs, b as u64)?;
        }
        self.absorb_const_u64(cs, fs.len() as u64)?;
        for &f in fs {
            let num = self.const_num(cs, f)?;
            self.sponge.absorb_slice_elem(cs, num)?;
        }
        Ok(())
    }

    pub fn challenge_field<CS: ConstraintSystem<CircuitF>>(
        &mut self,
        cs: &mut CS,
        label: &'static [u8],
        ctx: &str,
    ) -> Result<AllocatedNum<CircuitF>, SynthesisError> {
        self.append_message(cs, b"chal/label", label, ctx)?;
        Ok(self.sponge.digest32(cs, &format!("{ctx}_challenge_field"))?[0].clone())
    }

    pub fn challenge_fields<CS: ConstraintSystem<CircuitF>>(
        &mut self,
        cs: &mut CS,
        label: &'static [u8],
        n: usize,
        ctx: &str,
    ) -> Result<Vec<AllocatedNum<CircuitF>>, SynthesisError> {
        self.append_message(cs, b"chal/label", label, ctx)?;
        let mut out = Vec::with_capacity(n);
        let mut squeeze_idx = 0usize;
        while out.len() < n {
            let limbs = self.sponge.digest32(cs, &format!("{ctx}_challenge_fields_{squeeze_idx}"))?;
            squeeze_idx += 1;
            for i in 0..core::cmp::min(4, n - out.len()) {
                out.push(limbs[i].clone());
            }
        }
        Ok(out)
    }

    pub fn digest32<CS: ConstraintSystem<CircuitF>>(
        &mut self,
        cs: &mut CS,
        ctx: &str,
    ) -> Result<[AllocatedNum<CircuitF>; 4], SynthesisError> {
        self.sponge.digest32(cs, &format!("{ctx}_digest32"))
    }

    /// Fork the transcript (does not mutate the parent), matching
    /// `neo_transcript::Poseidon2Transcript::fork`:
    ///
    /// `child = self.clone(); child.append_message(b"fork", scope);`
    pub fn fork<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        scope: &'static [u8],
        ctx: &str,
    ) -> Result<Self, SynthesisError> {
        let mut child = self.clone();

        // `bellpepper`'s `ShapeCS` requires unique namespaces across the whole circuit.
        // Since `fork` clones the underlying sponge counters, we must ensure the forked
        // transcript uses a distinct naming scope to avoid collisions.
        child.scope = format!("{}_fork_{}", self.scope, ctx);
        child.sponge.set_scope(child.scope.clone());

        child.append_message(cs, b"fork", scope, ctx)?;
        Ok(child)
    }
}
