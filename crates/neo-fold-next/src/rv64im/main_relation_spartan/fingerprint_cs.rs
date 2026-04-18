use bellpepper_core::{ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};
use spartan2::provider::goldi::F as SpartanF;

const FINGERPRINT_DOMAIN: &[u8] = b"neo.fold.next/rv64im/main_recursion_step_spartan/fingerprint_cs";
const ENFORCE_TAG: u64 = 1;
const LC_A_TAG: u64 = 2;
const LC_B_TAG: u64 = 3;
const LC_C_TAG: u64 = 4;
const STATE_SEEDS: [u64; 4] = [
    0x243f6a8885a308d3,
    0x13198a2e03707344,
    0xa4093822299f31d0,
    0x082efa98ec4e6c89,
];

#[derive(Clone)]
pub(crate) struct FingerprintCS {
    state: [u64; 4],
    inputs: usize,
    aux: usize,
    constraints: usize,
    namespace_depth: usize,
}

impl FingerprintCS {
    pub(crate) fn new() -> Self {
        let mut cs = Self {
            state: STATE_SEEDS,
            inputs: 1,
            aux: 0,
            constraints: 0,
            namespace_depth: 0,
        };
        for chunk in FINGERPRINT_DOMAIN.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            cs.mix_word(u64::from_le_bytes(word));
        }
        cs
    }

    pub(crate) fn public_input_count(&self, num_challenges: usize) -> usize {
        self.inputs.saturating_sub(1 + num_challenges)
    }

    pub(crate) fn num_aux(&self) -> usize {
        self.aux
    }

    pub(crate) fn num_constraints(&self) -> usize {
        self.constraints
    }

    pub(crate) fn finish_digest32(mut self, num_challenges: usize) -> [u8; 32] {
        self.mix_word(self.constraints as u64);
        self.mix_word(self.aux as u64);
        self.mix_word(self.public_input_count(num_challenges) as u64);
        self.mix_word(num_challenges as u64);
        let mut out = [0u8; 32];
        for (idx, lane) in self.state.iter_mut().enumerate() {
            *lane = splitmix64(*lane ^ ((idx as u64) << 32));
            out[idx * 8..(idx + 1) * 8].copy_from_slice(&lane.to_le_bytes());
        }
        out
    }

    fn mix_word(&mut self, word: u64) {
        let mixed = splitmix64(word ^ self.state[0]);
        self.state[0] = self.state[0].wrapping_add(mixed);
        self.state[1] ^= self.state[0].rotate_left(13);
        self.state[2] = self.state[2].wrapping_add(self.state[1].rotate_left(17)) ^ mixed.rotate_left(7);
        self.state[3] = self.state[3]
            .wrapping_mul(0x94d049bb133111eb)
            .wrapping_add(self.state[2] ^ 0x9e3779b97f4a7c15);
    }

    fn absorb_lc(&mut self, lc_tag: u64, lc: &LinearCombination<SpartanF>) {
        self.mix_word(lc_tag);
        let mut term_count = 0u64;
        for (variable, coeff) in lc.iter() {
            self.mix_word(match variable.get_unchecked() {
                Index::Input(_) => 0,
                Index::Aux(_) => 1,
            });
            self.mix_word(match variable.get_unchecked() {
                Index::Input(idx) | Index::Aux(idx) => idx as u64,
            });
            self.mix_word(coeff.to_canonical_u64());
            term_count += 1;
        }
        self.mix_word(term_count);
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

impl ConstraintSystem<SpartanF> for FingerprintCS {
    type Root = Self;

    fn new() -> Self {
        Self::new()
    }

    fn alloc<FN, A, AR>(&mut self, _annotation: A, value: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<SpartanF, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let _ = value()?;
        let var = Variable::new_unchecked(Index::Aux(self.aux));
        self.aux += 1;
        Ok(var)
    }

    fn alloc_input<FN, A, AR>(&mut self, _annotation: A, value: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<SpartanF, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let _ = value()?;
        let var = Variable::new_unchecked(Index::Input(self.inputs));
        self.inputs += 1;
        Ok(var)
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<SpartanF>) -> LinearCombination<SpartanF>,
        LB: FnOnce(LinearCombination<SpartanF>) -> LinearCombination<SpartanF>,
        LC: FnOnce(LinearCombination<SpartanF>) -> LinearCombination<SpartanF>,
    {
        let lc_a = a(LinearCombination::zero());
        let lc_b = b(LinearCombination::zero());
        let lc_c = c(LinearCombination::zero());
        self.constraints += 1;
        self.mix_word(ENFORCE_TAG);
        self.mix_word(self.constraints as u64);
        self.absorb_lc(LC_A_TAG, &lc_a);
        self.absorb_lc(LC_B_TAG, &lc_b);
        self.absorb_lc(LC_C_TAG, &lc_c);
    }

    fn push_namespace<NR, N>(&mut self, _name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        self.namespace_depth += 1;
    }

    fn pop_namespace(&mut self) {
        assert!(self.namespace_depth > 0);
        self.namespace_depth -= 1;
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}
