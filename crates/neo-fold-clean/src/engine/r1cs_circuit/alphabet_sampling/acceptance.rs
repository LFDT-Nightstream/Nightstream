//! Constraint leaf for the fixed sampler's enough-accepts condition.
//!
//! Owns: the exact lower bound on accepted chunks.
//!
//! Does not own: chunk acceptance bits or first-accepted selection.
//!
//! Emits constraints: yes.
//!
//! Authority boundary: the final prefix count comes from the checked chunk
//! recurrence; the four-bit slack cannot authorize that count by itself.
//!
//! | Stage path | Function | Equation | Multiplicity | Emitted rows/formula | Lowered gate | Lean theorem |
//! |---|---|---|---:|---|---|---|
//! | `challenge.sampler.acceptance_bound` | `enforce_enough_accepts` | `acceptedCount = 54 + slack` | one per rho | four bit rows plus two equalities | generic R1CS | `acceptanceArithmetic_exact` |

use neo_math::ring::D;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::engine::r1cs_circuit::boolean::enforce_bit;
use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder};

use super::chunk::ChunkRecord;
use super::pi_rlc_challenge_stage;

const SLACK_BITS: usize = 4;

pub(super) fn enforce_enough_accepts(builder: &mut R1csBuilder, chunks: &[ChunkRecord]) {
    builder.begin_encoding_stage(pi_rlc_challenge_stage::ACCEPTANCE_BOUND);
    let accepted = chunks.last().expect("fixed sampler has chunks").cumulative;
    let slack_value = builder.witness()[accepted.col()] - F::from_u64(D as u64);
    let slack = builder.alloc(slack_value);

    let slack_u64 = slack_value.as_canonical_u64();
    let mut decomposition = Lc::zero();
    let mut power = F::ONE;
    for offset in 0..SLACK_BITS {
        let bit = builder.alloc(F::from_u64((slack_u64 >> offset) & 1));
        enforce_bit(builder, bit);
        decomposition.add_term(bit, power);
        power += power;
    }
    builder.enforce_eq(&Lc::from_var(slack), &decomposition);

    let mut expected = Lc::from_var(slack);
    expected.add_constant(F::from_u64(D as u64));
    builder.enforce_eq(&Lc::from_var(accepted), &expected);
}
