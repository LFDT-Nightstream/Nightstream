//! The terminal statement over any backend: the canonical split of the
//! running public inputs, the state preimage and its hash, and the fresh
//! public input that encodes the hash (`lifecycle/inputs.rs` natively).

use neo_ccs::crypto::poseidon2_goldilocks::{RATE, WIDTH};
use neo_math::F;
use neo_spartan::{Backend, Error};
use nightstream_fprime::PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::words::FinalWords;

/// `HyperNova/NIVC/state/v2`, eight little-endian bytes per word, zero
/// words to one rate chunk.
const STATE_DOMAIN_TAG: &[u8] = b"HyperNova/NIVC/state/v2";
/// Three parent coordinates per preimage word.
const PACK_RADIX: u64 = 1 << 17;

/// The statement words: the iteration, `z0` and the current state.
pub(super) struct State<B: Backend> {
    pub(super) iteration: B::F,
    pub(super) z0: [B::F; 4],
    pub(super) current: [B::F; 4],
}

/// Check that the running public inputs are the canonical base-2 split of
/// their parent `Σ_j 2^j x_j`, and return the parent. A coordinate's digits
/// are `s·β_j` with `β_j ∈ {0, 1}` and one sign `s = ±1`.
pub(super) fn canonical_parent<B: Backend>(b: &mut B, digits: &[Vec<B::F>]) -> Result<Vec<B::F>, Error> {
    let mut parent = Vec::with_capacity(PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS);
    for coordinate in 0..PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS {
        let column: Vec<B::F> = digits.iter().map(|child| child[coordinate]).collect();
        let sign = b.hint(&column, 1, &|values| {
            let value = values
                .iter()
                .rev()
                .fold(F::ZERO, |acc, &digit| acc.double() + F::from_u64(digit));
            vec![u64::from(value.as_canonical_u64() > F::ORDER_U64 / 2)]
        })[0];
        let square = b.mul(sign, sign);
        b.assert_equal(square, sign, "parent sign bit")?;
        let twice = b.add(sign, sign);
        let one = b.constant(1);
        let unit = b.sub(one, twice);
        let mut value = b.constant(0);
        for (j, &digit) in column.iter().enumerate() {
            let offset = b.sub(digit, unit);
            let product = b.mul(digit, offset);
            b.assert_zero(product, "canonical child digit")?;
            let weighted = b.scale(digit, 1 << j);
            value = b.add(value, weighted);
        }
        parent.push(value);
    }
    Ok(parent)
}

/// The workspace `poseidon2_hash`: add each chunk into the rate lanes and
/// permute, then add one to lane 0, permute, and take four lanes.
fn hash<B: Backend>(b: &mut B, words: &[B::F]) -> [B::F; 4] {
    let zero = b.constant(0);
    let mut state = [zero; WIDTH];
    for chunk in words.chunks(RATE) {
        for (lane, &word) in chunk.iter().enumerate() {
            state[lane] = b.add(state[lane], word);
        }
        state = b.permute(state);
    }
    let one = b.constant(1);
    state[0] = b.add(state[0], one);
    let state = b.permute(state);
    std::array::from_fn(|lane| state[lane])
}

/// `stateHash` of the canonical preimage: the domain chunk, the running
/// commitments, `Eval_K`, `Eval_A` and point, the packed parent public
/// input, then the verifier context, `i`, `z0` and `zi`.
pub(super) fn state_hash<B: Backend>(
    b: &mut B,
    context: [u64; 4],
    state: &State<B>,
    words: &FinalWords<B>,
    parent: &[B::F],
) -> [B::F; 4] {
    let mut preimage = Vec::new();
    for chunk in 0..RATE {
        let bytes = STATE_DOMAIN_TAG.get(8 * chunk..).unwrap_or_default();
        let mut word = [0u8; 8];
        word[..bytes.len().min(8)].copy_from_slice(&bytes[..bytes.len().min(8)]);
        preimage.push(b.constant(u64::from_le_bytes(word)));
    }
    for commitment in &words.commitments {
        preimage.extend(commitment.iter().flatten());
    }
    for evaluations in &words.evaluations {
        preimage.extend(evaluations.eval_k.iter().flatten());
    }
    for evaluations in &words.evaluations {
        preimage.extend(evaluations.eval_a.iter().flatten().flatten());
    }
    preimage.extend(words.point.iter().flatten());
    for lanes in parent.chunks_exact(3) {
        let middle = b.scale(lanes[1], PACK_RADIX);
        let high = b.scale(lanes[2], PACK_RADIX * PACK_RADIX);
        let low = b.add(lanes[0], middle);
        preimage.push(b.add(low, high));
    }
    for word in context {
        preimage.push(b.constant(word));
    }
    preimage.push(state.iteration);
    preimage.extend(state.z0);
    preimage.extend(state.current);
    hash(b, &preimage)
}

/// Lean `encHash`: a marker one, the 256 little-endian digest bits, then
/// zeros, as one public input.
pub(super) fn encode<B: Backend>(b: &mut B, digest: [B::F; 4]) -> Result<Vec<B::F>, Error> {
    let mut words = vec![b.constant(1)];
    for word in digest {
        words.extend(b.bits(word, 64, "state digest bits")?);
    }
    let zero = b.constant(0);
    words.resize(PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS, zero);
    Ok(words)
}
