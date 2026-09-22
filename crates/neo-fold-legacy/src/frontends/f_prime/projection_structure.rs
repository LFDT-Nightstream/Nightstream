//! Projection-region layouts and local CCS semantics for the encoded `F'` cost model.
//!
//! Owns: projection batch configuration, typed projection lane slots, and local
//! evaluation, quotient, and product rows.
//!
//! Does not own: production NIFS wire binding, native trace filling, or the
//! complete `F'` relation.
//!
//! Emits constraints: yes, through [`emit_projection_semantic_rows`].
//!
//! Authority boundary: these rows constrain only their local projection image;
//! without binding that image to checked NIFS values they carry no protocol
//! authority.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Lane ownership | [`ProjectionLaneSlots`] | no | [`FPrimeImageLayout`] |
//! | Projection equations | [`emit_projection_semantic_rows`] | yes | Local committed lanes only |

use crate::engine::r1cs_circuit::ring_action::PROJECTION_QUOTIENT_LEN;
use crate::frontends::f_prime::image::FPrimeImageLayout;
use crate::frontends::f_prime::structure::{lane_terms, LaneSlot, MixedGateBuilder};
use crate::paper::f_prime::projection_trace::{
    LANE_BITS, PROJECTION_IDENTITY_LANES, PROJECTION_PAIR_LANES, PROJECTION_SHARED_LANES,
};
use neo_math::field::KExtensions;
use neo_math::ring::{D, PHI_MID_DEGREE};
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

const K_MUL_ROWS: usize = 5;

/// Projection region splices, from which individual canonical-u64
/// slots are derived.
#[derive(Clone, Debug, Default)]
pub struct ProjectionLaneSlots {
    shared_splice: Option<usize>,
    pair_splices: Vec<usize>,
    identity_splices: Vec<usize>,
}

impl ProjectionLaneSlots {
    pub fn total(&self) -> usize {
        self.shared_splice.map_or(0, |_| PROJECTION_SHARED_LANES)
            + self.pair_splices.len() * PROJECTION_PAIR_LANES
            + self.identity_splices.len() * PROJECTION_IDENTITY_LANES
    }
}

pub(crate) fn collect_projection_slots(layout: &FPrimeImageLayout) -> ProjectionLaneSlots {
    let pair_count: usize = layout.config.projection_batches.iter().sum();
    let identity_count = layout.config.projection_batches.len();
    assert_eq!(layout.projection_pair_splices.len(), pair_count);
    assert_eq!(layout.projection_identity_splices.len(), identity_count);
    ProjectionLaneSlots {
        shared_splice: if identity_count == 0 {
            None
        } else {
            Some(layout.projection_shared_splice)
        },
        pair_splices: layout.projection_pair_splices.clone(),
        identity_splices: layout.projection_identity_splices.clone(),
    }
}

pub(crate) fn projection_semantic_row_count(batches: &[usize]) -> usize {
    if batches.is_empty() {
        return 0;
    }
    let pair_count: usize = batches.iter().sum();
    let shared_rows = 2 + D * K_MUL_ROWS;
    let pair_rows = pair_count * (2 * eval_row_count(D) + K_MUL_ROWS);
    let identity_rows = batches.len() * (eval_row_count(D) + eval_row_count(PROJECTION_QUOTIENT_LEN) + K_MUL_ROWS + 2);
    shared_rows + pair_rows + identity_rows
}

pub(crate) fn emit_projection_semantic_rows(
    batches: &[usize],
    slots: &ProjectionLaneSlots,
    builder: &mut MixedGateBuilder,
) {
    if batches.is_empty() {
        assert!(slots.shared_splice.is_none());
        assert!(slots.pair_splices.is_empty());
        assert!(slots.identity_splices.is_empty());
        return;
    }

    let pair_count: usize = batches.iter().sum();
    assert!(batches.iter().all(|&n| n > 0));
    assert_eq!(slots.pair_splices.len(), pair_count);
    assert_eq!(slots.identity_splices.len(), batches.len());

    let start = builder.rows();
    let shared = ProjectionSharedSlots::new(slots.shared_splice.expect("projection shared region"));
    emit_shared_rows(&shared, builder);

    let mut pair_terms = Vec::with_capacity(pair_count);
    for &splice in &slots.pair_splices {
        let pair = ProjectionPairSlots::new(splice);
        emit_eval_rows(&pair.rho, &shared.powers, &pair.rho_eval, builder);
        emit_eval_rows(&pair.c, &shared.powers, &pair.c_eval, builder);
        emit_k_mul_rows(pair.rho_eval.out_terms(), pair.c_eval.out_terms(), pair.term, builder);
        pair_terms.push(pair.term.out);
    }

    let mut pair_cursor = 0usize;
    for (&batch_len, &splice) in batches.iter().zip(slots.identity_splices.iter()) {
        let identity = ProjectionIdentitySlots::new(splice);
        emit_eval_rows(&identity.out, &shared.powers, &identity.out_eval, builder);
        emit_eval_rows(&identity.quotient, &shared.powers, &identity.quotient_eval, builder);

        let phi_beta = KLcTerms {
            c0: concat_terms([
                slot_terms(shared.powers[D].c0),
                slot_terms(shared.powers[PHI_MID_DEGREE].c0),
                vec![(0, F::ONE)],
            ]),
            c1: concat_terms([
                slot_terms(shared.powers[D].c1),
                slot_terms(shared.powers[PHI_MID_DEGREE].c1),
            ]),
        };
        emit_k_mul_rows(identity.quotient_eval.out_terms(), phi_beta, identity.q_phi, builder);

        let consumed = &pair_terms[pair_cursor..pair_cursor + batch_len];
        emit_projection_identity_rows(consumed, &identity, builder);
        pair_cursor += batch_len;
    }

    debug_assert_eq!(pair_cursor, pair_count);
    debug_assert_eq!(
        builder.rows() - start,
        projection_semantic_row_count(batches),
        "projection semantic row count drifted"
    );
}

fn eval_row_count(coeff_len: usize) -> usize {
    2 * (coeff_len - 1) + 2
}

#[derive(Clone, Copy, Debug)]
struct KSlot {
    c0: LaneSlot,
    c1: LaneSlot,
}

impl KSlot {
    fn terms(self) -> KLcTerms {
        KLcTerms {
            c0: slot_terms(self.c0),
            c1: slot_terms(self.c1),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct KMulSlot {
    p: LaneSlot,
    q: LaneSlot,
    r: LaneSlot,
    out: KSlot,
}

#[derive(Clone, Debug)]
struct EvalSlots {
    partials: Vec<KSlot>,
    out: KSlot,
}

impl EvalSlots {
    fn out_terms(&self) -> KLcTerms {
        self.out.terms()
    }
}

#[derive(Clone, Debug)]
struct ProjectionSharedSlots {
    beta: KSlot,
    powers: Vec<KSlot>,
    kmuls: Vec<KMulSlot>,
}

impl ProjectionSharedSlots {
    fn new(base: usize) -> Self {
        let beta = k_slot(base, 0);
        let mut powers = Vec::with_capacity(D + 1);
        powers.push(k_slot(base, 2));
        let mut kmuls = Vec::with_capacity(D);
        for k in 1..=D {
            let slot_base = 4 + (k - 1) * 5;
            kmuls.push(k_mul_slot(base, slot_base));
            powers.push(k_slot(base, slot_base + 3));
        }
        Self { beta, powers, kmuls }
    }
}

#[derive(Clone, Debug)]
struct ProjectionPairSlots {
    rho: Vec<LaneSlot>,
    c: Vec<LaneSlot>,
    rho_eval: EvalSlots,
    c_eval: EvalSlots,
    term: KMulSlot,
}

impl ProjectionPairSlots {
    fn new(base: usize) -> Self {
        let rho = lane_range(base, 0, D);
        let c = lane_range(base, D, D);
        let rho_eval = eval_slots(base, 2 * D, D);
        let c_eval = eval_slots(base, 2 * D + eval_lanes(D), D);
        let term = k_mul_slot(base, 2 * D + 2 * eval_lanes(D));
        Self {
            rho,
            c,
            rho_eval,
            c_eval,
            term,
        }
    }
}

#[derive(Clone, Debug)]
struct ProjectionIdentitySlots {
    out: Vec<LaneSlot>,
    quotient: Vec<LaneSlot>,
    out_eval: EvalSlots,
    quotient_eval: EvalSlots,
    q_phi: KMulSlot,
}

impl ProjectionIdentitySlots {
    fn new(base: usize) -> Self {
        let out = lane_range(base, 0, D);
        let quotient = lane_range(base, D, PROJECTION_QUOTIENT_LEN);
        let out_eval_base = D + PROJECTION_QUOTIENT_LEN;
        let quotient_eval_base = out_eval_base + eval_lanes(D);
        let q_phi_base = quotient_eval_base + eval_lanes(PROJECTION_QUOTIENT_LEN);
        Self {
            out,
            quotient,
            out_eval: eval_slots(base, out_eval_base, D),
            quotient_eval: eval_slots(base, quotient_eval_base, PROJECTION_QUOTIENT_LEN),
            q_phi: k_mul_slot(base, q_phi_base),
        }
    }
}

#[derive(Clone, Debug)]
struct KLcTerms {
    c0: Vec<(usize, F)>,
    c1: Vec<(usize, F)>,
}

fn emit_shared_rows(shared: &ProjectionSharedSlots, builder: &mut MixedGateBuilder) {
    builder.linear(slot_terms(shared.powers[0].c0), vec![(0, F::ONE)]);
    builder.linear(slot_terms(shared.powers[0].c1), Vec::new());
    for k in 1..=D {
        emit_k_mul_rows(
            shared.powers[k - 1].terms(),
            shared.beta.terms(),
            shared.kmuls[k - 1],
            builder,
        );
    }
}

fn emit_eval_rows(coeffs: &[LaneSlot], powers: &[KSlot], eval: &EvalSlots, builder: &mut MixedGateBuilder) {
    assert_eq!(eval.partials.len(), coeffs.len() - 1);
    assert!(coeffs.len() <= powers.len());
    for (j, partial) in eval.partials.iter().enumerate() {
        let coeff = coeffs[j + 1];
        builder.product(slot_terms(coeff), slot_terms(powers[j + 1].c0), slot_terms(partial.c0));
        builder.product(slot_terms(coeff), slot_terms(powers[j + 1].c1), slot_terms(partial.c1));
    }

    let mut rhs_c0 = slot_terms(coeffs[0]);
    let mut rhs_c1 = Vec::new();
    for partial in &eval.partials {
        rhs_c0.extend(slot_terms(partial.c0));
        rhs_c1.extend(slot_terms(partial.c1));
    }
    builder.linear(slot_terms(eval.out.c0), rhs_c0);
    builder.linear(slot_terms(eval.out.c1), rhs_c1);
}

fn emit_k_mul_rows(a: KLcTerms, b: KLcTerms, slot: KMulSlot, builder: &mut MixedGateBuilder) {
    builder.product(a.c0.clone(), b.c0.clone(), slot_terms(slot.p));
    builder.product(a.c1.clone(), b.c1.clone(), slot_terms(slot.q));
    builder.product(
        concat_terms([a.c0, a.c1]),
        concat_terms([b.c0, b.c1]),
        slot_terms(slot.r),
    );

    let w = binomial_w();
    builder.linear(
        slot_terms(slot.out.c0),
        concat_terms([slot_terms(slot.p), scaled_slot_terms(slot.q, w)]),
    );
    builder.linear(
        slot_terms(slot.out.c1),
        concat_terms([
            slot_terms(slot.r),
            scaled_slot_terms(slot.p, F::ZERO - F::ONE),
            scaled_slot_terms(slot.q, F::ZERO - F::ONE),
        ]),
    );
}

fn emit_projection_identity_rows(
    pair_terms: &[KSlot],
    identity: &ProjectionIdentitySlots,
    builder: &mut MixedGateBuilder,
) {
    let mut lhs_c0 = Vec::new();
    let mut lhs_c1 = Vec::new();
    for pair in pair_terms {
        lhs_c0.extend(slot_terms(pair.c0));
        lhs_c1.extend(slot_terms(pair.c1));
    }

    let rhs_c0 = concat_terms([slot_terms(identity.out_eval.out.c0), slot_terms(identity.q_phi.out.c0)]);
    let rhs_c1 = concat_terms([slot_terms(identity.out_eval.out.c1), slot_terms(identity.q_phi.out.c1)]);
    builder.linear(lhs_c0, rhs_c0);
    builder.linear(lhs_c1, rhs_c1);
}

const fn eval_lanes(n: usize) -> usize {
    2 * (n - 1) + 2
}

fn eval_slots(base: usize, lane_base: usize, coeff_len: usize) -> EvalSlots {
    let partials = (0..coeff_len - 1)
        .map(|i| k_slot(base, lane_base + 2 * i))
        .collect();
    EvalSlots {
        partials,
        out: k_slot(base, lane_base + 2 * (coeff_len - 1)),
    }
}

fn lane_range(base: usize, lane_base: usize, count: usize) -> Vec<LaneSlot> {
    (0..count).map(|i| lane(base, lane_base + i)).collect()
}

fn lane(base: usize, lane_idx: usize) -> LaneSlot {
    LaneSlot {
        bit_start: base + lane_idx * LANE_BITS,
    }
}

fn k_slot(base: usize, lane_idx: usize) -> KSlot {
    KSlot {
        c0: lane(base, lane_idx),
        c1: lane(base, lane_idx + 1),
    }
}

fn k_mul_slot(base: usize, lane_idx: usize) -> KMulSlot {
    KMulSlot {
        p: lane(base, lane_idx),
        q: lane(base, lane_idx + 1),
        r: lane(base, lane_idx + 2),
        out: k_slot(base, lane_idx + 3),
    }
}

fn slot_terms(slot: LaneSlot) -> Vec<(usize, F)> {
    lane_terms(slot).collect()
}

fn scaled_slot_terms(slot: LaneSlot, coeff: F) -> Vec<(usize, F)> {
    lane_terms(slot).map(|(col, c)| (col, c * coeff)).collect()
}

fn concat_terms<const N: usize>(chunks: [Vec<(usize, F)>; N]) -> Vec<(usize, F)> {
    chunks.into_iter().flatten().collect()
}

fn binomial_w() -> F {
    let u = K::from_coeffs([F::ZERO, F::ONE]);
    let coeffs = (u * u).as_coeffs();
    debug_assert_eq!(coeffs[1], F::ZERO);
    coeffs[0]
}
