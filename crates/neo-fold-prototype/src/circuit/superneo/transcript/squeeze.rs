//! Challenge and digest squeezing for the Poseidon2 transcript circuit.

use super::*;

impl Poseidon2TranscriptCircuit {
    pub fn challenge_fields<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        label: &'static [u8],
        n: usize,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        self.append_message(cs.namespace(|| "challenge_label"), b"chal/label", label)?;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            self.absorb_constant(cs.namespace(|| format!("challenge_gate_{}", out.len())), SpartanF::ONE)?;
            self.permute(cs.namespace(|| format!("challenge_permute_{}", out.len())))?;
            for i in 0..DIGEST_LEN.min(n - out.len()) {
                out.push(
                    self.state[i].allocate_canonical(cs.namespace(|| format!("chal_allocate_{}_{i}", out.len())))?,
                );
            }
        }
        Ok(out)
    }

    pub fn challenge_fields_raw<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
        n: usize,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            self.absorb_constant(cs.namespace(|| format!("challenge_gate_{}", out.len())), SpartanF::ONE)?;
            self.permute(cs.namespace(|| format!("challenge_permute_{}", out.len())))?;
            for i in 0..DIGEST_LEN.min(n - out.len()) {
                out.push(
                    self.state[i]
                        .allocate_canonical(cs.namespace(|| format!("chal_allocate_raw_{}_{i}", out.len())))?,
                );
            }
        }
        Ok(out)
    }

    pub fn digest32<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        cs: CS,
    ) -> Result<[AllocatedNum<SpartanF>; DIGEST_LEN], SynthesisError> {
        Ok(self.digest32_with_values(cs)?.0)
    }

    pub fn digest32_with_values<CS: ConstraintSystem<SpartanF>>(
        &mut self,
        mut cs: CS,
    ) -> Result<([AllocatedNum<SpartanF>; DIGEST_LEN], [SpartanF; DIGEST_LEN]), SynthesisError> {
        self.absorb_constant(cs.namespace(|| "digest_padding"), SpartanF::ONE)?;
        self.permute(cs.namespace(|| "digest_permute"))?;
        let values = core::array::from_fn(|i| self.state[i].value);
        let mut out = Vec::with_capacity(DIGEST_LEN);
        for i in 0..DIGEST_LEN {
            out.push(self.state[i].allocate_canonical(cs.namespace(|| format!("digest_allocate_{i}")))?);
        }
        Ok((out.try_into().map_err(|_| SynthesisError::Unsatisfiable)?, values))
    }
}
