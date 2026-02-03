use crate::goldilocks::{
    assert_canonical_goldilocks, gl_mul_mod_check_with_quotient, host_mul_quotient_and_remainder, OuterScalar,
    GOLDILOCKS_P_U64,
};
use crate::k_field::{
    alloc_k_private, alloc_k_private_u64, alloc_k_public, assert_k_eq, k_add_mod_var, k_const, k_mle_fold_step,
    k_mul_mod_var, k_one, k_sub_mod_var, k_sum_mod_var, k_zero, KRepr, KVar, K_DELTA_U64,
};
use crate::sumcheck::{sumcheck_eval_horner, sumcheck_round_check};
use ff::PrimeField;
use midnight_circuits::{
    instructions::{AssignmentInstructions, PublicInputInstructions},
    types::AssignedNative,
};
use midnight_proofs::circuit::{Layouter, Value};
use midnight_proofs::plonk::Error;
use midnight_zk_stdlib::{Relation, ZkStdLib, ZkStdLibArch};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

fn write_relation_len_prefixed<W: Write, T: Serialize>(writer: &mut W, value: &T) -> std::io::Result<()> {
    let bytes =
        bincode::serialize(value).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let len = u32::try_from(bytes.len())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "relation too large"))?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&bytes)?;
    Ok(())
}

fn read_relation_len_prefixed<R: Read, T: for<'de> Deserialize<'de>>(reader: &mut R) -> std::io::Result<T> {
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    bincode::deserialize(&bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Public statement: `z = x * y (mod p)` over Goldilocks.
///
/// This is a minimal end-to-end “Option B” sanity check:
/// - Circuit field is Midnight’s `outer::Scalar` (BLS12-381 scalar).
/// - Goldilocks elements are represented as 64-bit integers with `< p` constraints.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoldilocksMulRelation;

/// Public instance (canonical u64 representatives).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoldilocksMulInstance {
    pub x: u64,
    pub y: u64,
    pub z: u64,
}

impl Relation for GoldilocksMulRelation {
    type Instance = GoldilocksMulInstance;
    type Witness = ();

    fn format_instance(instance: &Self::Instance) -> Result<Vec<OuterScalar>, Error> {
        Ok(vec![
            OuterScalar::from(instance.x),
            OuterScalar::from(instance.y),
            OuterScalar::from(instance.z),
        ])
    }

    fn write_relation<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write_relation_len_prefixed(writer, self)
    }

    fn read_relation<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        read_relation_len_prefixed(reader)
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<OuterScalar>,
        instance: Value<Self::Instance>,
        _witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        // Allocate public inputs x,y,z.
        let x_u64 = instance.as_ref().map(|i| i.x);
        let y_u64 = instance.as_ref().map(|i| i.y);
        let z_u64 = instance.as_ref().map(|i| i.z);

        let x: AssignedNative<OuterScalar> = std_lib.assign(layouter, x_u64.map(OuterScalar::from))?;
        std_lib.constrain_as_public_input(layouter, &x)?;

        let y: AssignedNative<OuterScalar> = std_lib.assign(layouter, y_u64.map(OuterScalar::from))?;
        std_lib.constrain_as_public_input(layouter, &y)?;

        let z: AssignedNative<OuterScalar> = std_lib.assign(layouter, z_u64.map(OuterScalar::from))?;
        std_lib.constrain_as_public_input(layouter, &z)?;

        // Enforce canonical Goldilocks encoding.
        assert_canonical_goldilocks(std_lib, layouter, &x)?;
        assert_canonical_goldilocks(std_lib, layouter, &y)?;
        assert_canonical_goldilocks(std_lib, layouter, &z)?;

        // Witness quotient k from the public values.
        let k = x_u64
            .zip(y_u64)
            .map(|(xu, yu)| host_mul_quotient_and_remainder(xu, yu).0);

        // Check z = x*y mod p using k.
        gl_mul_mod_check_with_quotient(std_lib, layouter, &x, &y, &z, k)?;
        Ok(())
    }
}

/// A single sumcheck round sanity-check over Neo's extension field K.
///
/// Checks:
/// - `p(0) + p(1) == claimed_sum`
/// - `p(challenge) == next_sum`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SumcheckSingleRoundRelation {
    pub n_coeffs: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SumcheckSingleRoundWitness {
    pub coeffs: Vec<KRepr>,
    pub challenge: KRepr,
    pub claimed_sum: KRepr,
    pub next_sum: KRepr,
}

impl Relation for SumcheckSingleRoundRelation {
    type Instance = ();
    type Witness = SumcheckSingleRoundWitness;

    fn format_instance(_instance: &Self::Instance) -> Result<Vec<OuterScalar>, Error> {
        Ok(vec![])
    }

    fn write_relation<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write_relation_len_prefixed(writer, self)
    }

    fn read_relation<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        read_relation_len_prefixed(reader)
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<OuterScalar>,
        _instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        if self.n_coeffs == 0 {
            return Err(Error::Synthesis(
                "SumcheckSingleRoundRelation requires n_coeffs > 0".into(),
            ));
        }

        // Allocate coefficients.
        let mut coeffs = Vec::with_capacity(self.n_coeffs);
        for i in 0..self.n_coeffs {
            let coeff_i = witness.as_ref().map(|w| w.coeffs[i]);
            coeffs.push(alloc_k_private(std_lib, layouter, coeff_i)?);
        }

        // Allocate round values.
        let challenge = alloc_k_private(std_lib, layouter, witness.as_ref().map(|w| w.challenge))?;
        let claimed_sum = alloc_k_private(std_lib, layouter, witness.as_ref().map(|w| w.claimed_sum))?;
        let next_sum = alloc_k_private(std_lib, layouter, witness.as_ref().map(|w| w.next_sum))?;

        // Enforce round binding and evaluation.
        sumcheck_round_check(std_lib, layouter, &coeffs, &claimed_sum)?;
        let eval = sumcheck_eval_horner(std_lib, layouter, &coeffs, &challenge, K_DELTA_U64)?;
        assert_k_eq(std_lib, layouter, &eval, &next_sum)?;
        Ok(())
    }
}

/// Multi-round sumcheck sanity-check over Neo's extension field K (δ=7).
///
/// This is the next incremental building block towards porting
/// `FoldRunCircuit::verify_sumcheck_rounds` into Midnight PLONK/KZG.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsSumcheckRelation {
    pub n_rounds: usize,
    pub poly_len: usize,
}

/// Public statement: the initial and final running sums of the sumcheck.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsSumcheckInstance {
    /// Step-bundle binding digest limbs (little-endian u128×2).
    pub bundle_digest: [u128; 2],
    pub initial_sum: KRepr,
    pub final_sum: KRepr,
    /// Sumcheck challenges (public).
    ///
    /// SECURITY: a deployable verifier must ensure these challenges are transcript-derived
    /// (Fiat–Shamir) and not prover-chosen. The circuit treats them as public inputs so they
    /// are never "free witness".
    pub challenges: Vec<KRepr>,
}

/// Witness: all round polynomials + challenges.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsSumcheckWitness {
    pub rounds: Vec<Vec<KRepr>>,
}

impl Relation for PiCcsSumcheckRelation {
    type Instance = PiCcsSumcheckInstance;
    type Witness = PiCcsSumcheckWitness;

    fn used_chips(&self) -> ZkStdLibArch {
        // This circuit is dominated by range checks (Goldilocks/K element canonicity).
        // Increasing the number of parallel pow2range columns significantly reduces the
        // required number of rows, and thus `min_k()`.
        let mut arch = ZkStdLibArch::default();
        // NOTE: midnight-zk-stdlib 1.0.0 slices pow2range columns as
        // `advice_columns[1..=nr_pow2range_cols]` while allocating only
        // `max(NB_ARITH_COLS, nr_pow2range_cols, ..)` advice columns. With
        // `NB_ARITH_COLS = 5`, the maximum safe value here is `4`.
        arch.nr_pow2range_cols = 4;
        arch
    }

    fn format_instance(instance: &Self::Instance) -> Result<Vec<OuterScalar>, Error> {
        let mut out = Vec::with_capacity(2 + 4 + 2 * instance.challenges.len());
        out.push(OuterScalar::from_u128(instance.bundle_digest[0]));
        out.push(OuterScalar::from_u128(instance.bundle_digest[1]));
        out.extend_from_slice(&[
            OuterScalar::from(instance.initial_sum.c0),
            OuterScalar::from(instance.initial_sum.c1),
            OuterScalar::from(instance.final_sum.c0),
            OuterScalar::from(instance.final_sum.c1),
        ]);
        for ch in &instance.challenges {
            out.push(OuterScalar::from(ch.c0));
            out.push(OuterScalar::from(ch.c1));
        }
        Ok(out)
    }

    fn write_relation<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write_relation_len_prefixed(writer, self)
    }

    fn read_relation<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        read_relation_len_prefixed(reader)
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<OuterScalar>,
        instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        if self.n_rounds == 0 {
            return Err(Error::Synthesis("PiCcsSumcheckRelation requires n_rounds > 0".into()));
        }
        if self.poly_len == 0 {
            return Err(Error::Synthesis("PiCcsSumcheckRelation requires poly_len > 0".into()));
        }

        // Public: bundle digest (anti mix-and-match across steps/runs).
        for limb_idx in 0..2 {
            let limb = instance.as_ref().map(|i| i.bundle_digest[limb_idx]);
            let assigned: AssignedNative<OuterScalar> = std_lib.assign(layouter, limb.map(OuterScalar::from_u128))?;
            std_lib.constrain_as_public_input(layouter, &assigned)?;
        }

        // Public inputs: (initial_sum, final_sum, challenges).
        let initial_sum = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.initial_sum))?;
        let final_sum = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.final_sum))?;

        // Allocate witness rounds.
        let mut rounds: Vec<Vec<_>> = Vec::with_capacity(self.n_rounds);
        for r in 0..self.n_rounds {
            let mut coeffs_r = Vec::with_capacity(self.poly_len);
            for j in 0..self.poly_len {
                let coeff = witness.as_ref().map(|w| w.rounds[r][j]);
                coeffs_r.push(alloc_k_private(std_lib, layouter, coeff)?);
            }
            rounds.push(coeffs_r);
        }

        let mut challenges = Vec::with_capacity(self.n_rounds);
        for r in 0..self.n_rounds {
            let ch = instance.as_ref().map(|i| i.challenges[r]);
            challenges.push(alloc_k_public(std_lib, layouter, ch)?);
        }

        // Verify all rounds.
        let mut running_sum = initial_sum;
        for r in 0..self.n_rounds {
            sumcheck_round_check(std_lib, layouter, &rounds[r], &running_sum)?;
            running_sum = sumcheck_eval_horner(std_lib, layouter, &rounds[r], &challenges[r], K_DELTA_U64)?;
        }

        assert_k_eq(std_lib, layouter, &running_sum, &final_sum)?;
        Ok(())
    }
}

/// Multi-round NC-only sumcheck sanity-check over Neo's extension field K (δ=7).
///
/// This mirrors the FE sumcheck relation, but is intended for the SplitNcV1
/// `sumcheck_rounds_nc` / `sumcheck_challenges_nc` channel.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsSumcheckNcRelation {
    pub n_rounds: usize,
    pub poly_len: usize,
}

impl Relation for PiCcsSumcheckNcRelation {
    type Instance = PiCcsSumcheckInstance;
    type Witness = PiCcsSumcheckWitness;

    fn used_chips(&self) -> ZkStdLibArch {
        // Same tuning as the FE sumcheck: we are range-check dominated.
        let mut arch = ZkStdLibArch::default();
        arch.nr_pow2range_cols = 4;
        arch
    }

    fn format_instance(instance: &Self::Instance) -> Result<Vec<OuterScalar>, Error> {
        let mut out = Vec::with_capacity(2 + 4 + 2 * instance.challenges.len());
        out.push(OuterScalar::from_u128(instance.bundle_digest[0]));
        out.push(OuterScalar::from_u128(instance.bundle_digest[1]));
        out.extend_from_slice(&[
            OuterScalar::from(instance.initial_sum.c0),
            OuterScalar::from(instance.initial_sum.c1),
            OuterScalar::from(instance.final_sum.c0),
            OuterScalar::from(instance.final_sum.c1),
        ]);
        for ch in &instance.challenges {
            out.push(OuterScalar::from(ch.c0));
            out.push(OuterScalar::from(ch.c1));
        }
        Ok(out)
    }

    fn write_relation<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write_relation_len_prefixed(writer, self)
    }

    fn read_relation<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        read_relation_len_prefixed(reader)
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<OuterScalar>,
        instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        if self.n_rounds == 0 {
            return Err(Error::Synthesis("PiCcsSumcheckNcRelation requires n_rounds > 0".into()));
        }
        if self.poly_len == 0 {
            return Err(Error::Synthesis("PiCcsSumcheckNcRelation requires poly_len > 0".into()));
        }

        // Public: bundle digest (anti mix-and-match across steps/runs).
        for limb_idx in 0..2 {
            let limb = instance.as_ref().map(|i| i.bundle_digest[limb_idx]);
            let assigned: AssignedNative<OuterScalar> = std_lib.assign(layouter, limb.map(OuterScalar::from_u128))?;
            std_lib.constrain_as_public_input(layouter, &assigned)?;
        }

        // Public inputs: (initial_sum, final_sum, challenges).
        let initial_sum = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.initial_sum))?;
        let final_sum = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.final_sum))?;

        // Allocate witness rounds.
        let mut rounds: Vec<Vec<_>> = Vec::with_capacity(self.n_rounds);
        for r in 0..self.n_rounds {
            let mut coeffs_r = Vec::with_capacity(self.poly_len);
            for j in 0..self.poly_len {
                let coeff = witness.as_ref().map(|w| w.rounds[r][j]);
                coeffs_r.push(alloc_k_private(std_lib, layouter, coeff)?);
            }
            rounds.push(coeffs_r);
        }

        let mut challenges = Vec::with_capacity(self.n_rounds);
        for r in 0..self.n_rounds {
            let ch = instance.as_ref().map(|i| i.challenges[r]);
            challenges.push(alloc_k_public(std_lib, layouter, ch)?);
        }

        // Verify all rounds.
        let mut running_sum = initial_sum;
        for r in 0..self.n_rounds {
            sumcheck_round_check(std_lib, layouter, &rounds[r], &running_sum)?;
            running_sum = sumcheck_eval_horner(std_lib, layouter, &rounds[r], &challenges[r], K_DELTA_U64)?;
        }

        assert_k_eq(std_lib, layouter, &running_sum, &final_sum)?;
        Ok(())
    }
}

/// Multi-round sumcheck with **public** rounds (coefficients) over Neo's extension field K (δ=7).
///
/// This is intended as a Fiat–Shamir “anchor” proof: by exposing the prover messages (round
/// polynomials) as public inputs, a verifier can recompute transcript-derived challenges and
/// check they match the public `challenges` vector used across the bundle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsSumcheckPublicRoundsRelation {
    pub n_rounds: usize,
    pub poly_len: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsSumcheckPublicRoundsInstance {
    /// Step-bundle binding digest limbs (little-endian u128×2).
    pub bundle_digest: [u128; 2],
    pub initial_sum: KRepr,
    pub final_sum: KRepr,
    /// Sumcheck challenges (public).
    pub challenges: Vec<KRepr>,
    /// Sumcheck round polynomials (public): `rounds[round_idx][coeff_idx]`.
    pub rounds: Vec<Vec<KRepr>>,
}

impl Relation for PiCcsSumcheckPublicRoundsRelation {
    type Instance = PiCcsSumcheckPublicRoundsInstance;
    type Witness = ();

    fn used_chips(&self) -> ZkStdLibArch {
        let mut arch = ZkStdLibArch::default();
        arch.nr_pow2range_cols = 4;
        arch
    }

    fn format_instance(instance: &Self::Instance) -> Result<Vec<OuterScalar>, Error> {
        let mut out = Vec::new();
        out.push(OuterScalar::from_u128(instance.bundle_digest[0]));
        out.push(OuterScalar::from_u128(instance.bundle_digest[1]));
        out.extend_from_slice(&[
            OuterScalar::from(instance.initial_sum.c0),
            OuterScalar::from(instance.initial_sum.c1),
            OuterScalar::from(instance.final_sum.c0),
            OuterScalar::from(instance.final_sum.c1),
        ]);
        for ch in &instance.challenges {
            out.push(OuterScalar::from(ch.c0));
            out.push(OuterScalar::from(ch.c1));
        }
        for round in &instance.rounds {
            for c in round {
                out.push(OuterScalar::from(c.c0));
                out.push(OuterScalar::from(c.c1));
            }
        }
        Ok(out)
    }

    fn write_relation<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write_relation_len_prefixed(writer, self)
    }

    fn read_relation<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        read_relation_len_prefixed(reader)
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<OuterScalar>,
        instance: Value<Self::Instance>,
        _witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        if self.n_rounds == 0 {
            return Err(Error::Synthesis(
                "PiCcsSumcheckPublicRoundsRelation requires n_rounds > 0".into(),
            ));
        }
        if self.poly_len == 0 {
            return Err(Error::Synthesis(
                "PiCcsSumcheckPublicRoundsRelation requires poly_len > 0".into(),
            ));
        }

        // Public: bundle digest (anti mix-and-match across steps/runs).
        for limb_idx in 0..2 {
            let limb = instance.as_ref().map(|i| i.bundle_digest[limb_idx]);
            let assigned: AssignedNative<OuterScalar> = std_lib.assign(layouter, limb.map(OuterScalar::from_u128))?;
            std_lib.constrain_as_public_input(layouter, &assigned)?;
        }

        let initial_sum = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.initial_sum))?;
        let final_sum = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.final_sum))?;

        let mut challenges = Vec::with_capacity(self.n_rounds);
        for r in 0..self.n_rounds {
            let ch = instance.as_ref().map(|i| i.challenges[r]);
            challenges.push(alloc_k_public(std_lib, layouter, ch)?);
        }

        let mut running_sum = initial_sum;
        for r in 0..self.n_rounds {
            // Public rounds.
            let mut coeffs_r = Vec::with_capacity(self.poly_len);
            for j in 0..self.poly_len {
                let coeff = instance.as_ref().map(|i| i.rounds[r][j]);
                coeffs_r.push(alloc_k_public(std_lib, layouter, coeff)?);
            }

            sumcheck_round_check(std_lib, layouter, &coeffs_r, &running_sum)?;
            running_sum = sumcheck_eval_horner(std_lib, layouter, &coeffs_r, &challenges[r], K_DELTA_U64)?;
        }

        assert_k_eq(std_lib, layouter, &running_sum, &final_sum)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparsePolyTermRepr {
    pub coeff: u64,
    pub exps: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparsePolyRepr {
    pub t: usize,
    pub terms: Vec<SparsePolyTermRepr>,
}

fn k_eq_points(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    p: &[KVar],
    q: &[KVar],
) -> Result<KVar, Error> {
    if p.len() != q.len() {
        return Err(Error::Synthesis("k_eq_points: length mismatch".into()));
    }
    let one = k_one(std, layouter)?;

    // eq(p,q) = Π_i ((1-p_i)(1-q_i) + p_i q_i)
    //         = Π_i (1 - p_i - q_i + 2 p_i q_i)
    let mut acc = one.clone();
    for (pi, qi) in p.iter().zip(q.iter()) {
        let pq = k_mul_mod_var(std, layouter, pi, qi, K_DELTA_U64)?;
        let two_pq = k_add_mod_var(std, layouter, &pq, &pq)?;

        let one_minus_p = k_sub_mod_var(std, layouter, &one, pi)?;
        let one_minus_p_minus_q = k_sub_mod_var(std, layouter, &one_minus_p, qi)?;
        let term = k_add_mod_var(std, layouter, &one_minus_p_minus_q, &two_pq)?;
        acc = k_mul_mod_var(std, layouter, &acc, &term, K_DELTA_U64)?;
    }
    Ok(acc)
}

fn k_eval_sparse_poly_in_ext(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    poly: &SparsePolyRepr,
    x: &[KVar],
) -> Result<KVar, Error> {
    if poly.t == 0 {
        return Err(Error::Synthesis("k_eval_sparse_poly_in_ext: t=0".into()));
    }
    if poly.terms.is_empty() {
        return Err(Error::Synthesis("k_eval_sparse_poly_in_ext: empty terms".into()));
    }
    if x.len() != poly.t {
        return Err(Error::Synthesis(format!(
            "k_eval_sparse_poly_in_ext: x.len()={} != t={}",
            x.len(),
            poly.t
        )));
    }

    let mut acc = k_zero(std, layouter)?;
    for term in &poly.terms {
        if term.exps.len() != poly.t {
            return Err(Error::Synthesis(format!(
                "k_eval_sparse_poly_in_ext: term.exps.len()={} != t={}",
                term.exps.len(),
                poly.t
            )));
        }

        // Lift coeff into K via (coeff, 0).
        let mut m = k_const(std, layouter, term.coeff, 0)?;

        for (xi, &pow) in x.iter().zip(term.exps.iter()) {
            match pow {
                0 => {}
                1 => {
                    m = k_mul_mod_var(std, layouter, &m, xi, K_DELTA_U64)?;
                }
                2 => {
                    let sq = k_mul_mod_var(std, layouter, xi, xi, K_DELTA_U64)?;
                    m = k_mul_mod_var(std, layouter, &m, &sq, K_DELTA_U64)?;
                }
                _ => {
                    // Exponents should be tiny for CCS (R1CS-style is degree 2),
                    // but keep this generic for debugging/prototyping.
                    let mut p = xi.clone();
                    for _ in 1..pow {
                        p = k_mul_mod_var(std, layouter, &p, xi, K_DELTA_U64)?;
                    }
                    m = k_mul_mod_var(std, layouter, &m, &p, K_DELTA_U64)?;
                }
            }
        }

        acc = k_add_mod_var(std, layouter, &acc, &m)?;
    }
    Ok(acc)
}

/// FE-only terminal identity check for the SplitNcV1 variant in the **k=1** (no ME inputs) case.
///
/// For step 0 of a `FoldRun` created without an initial accumulator, we have:
/// - `k_total = |out_me| = 1`
/// - `me_inputs_r_opt = None`
/// so the verifier RHS simplifies to:
///
/// `rhs = eq((α',r'), β) * F'(y_scalars)`
///
/// where `F' = f(y_scalars[0..t])` is the CCS polynomial evaluated in the extension field.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsFeTerminalK1Relation {
    pub ell_n: usize,
    pub ell_d: usize,
    pub poly: SparsePolyRepr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsFeTerminalK1Instance {
    pub final_sum: KRepr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsFeTerminalK1Witness {
    pub r_prime: Vec<KRepr>,
    pub alpha_prime: Vec<KRepr>,
    pub beta_a: Vec<KRepr>,
    pub beta_r: Vec<KRepr>,
    pub y_scalars: Vec<KRepr>,
}

impl Relation for PiCcsFeTerminalK1Relation {
    type Instance = PiCcsFeTerminalK1Instance;
    type Witness = PiCcsFeTerminalK1Witness;

    fn used_chips(&self) -> ZkStdLibArch {
        let mut arch = ZkStdLibArch::default();
        arch.nr_pow2range_cols = 4;
        arch
    }

    fn format_instance(instance: &Self::Instance) -> Result<Vec<OuterScalar>, Error> {
        Ok(vec![
            OuterScalar::from(instance.final_sum.c0),
            OuterScalar::from(instance.final_sum.c1),
        ])
    }

    fn write_relation<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write_relation_len_prefixed(writer, self)
    }

    fn read_relation<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        read_relation_len_prefixed(reader)
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<OuterScalar>,
        instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        if self.ell_n == 0 {
            return Err(Error::Synthesis("PiCcsFeTerminalK1Relation requires ell_n > 0".into()));
        }
        if self.ell_d == 0 {
            return Err(Error::Synthesis("PiCcsFeTerminalK1Relation requires ell_d > 0".into()));
        }
        if self.poly.t == 0 {
            return Err(Error::Synthesis("PiCcsFeTerminalK1Relation requires poly.t > 0".into()));
        }
        if self.poly.terms.is_empty() {
            return Err(Error::Synthesis(
                "PiCcsFeTerminalK1Relation requires poly.terms non-empty".into(),
            ));
        }

        let final_sum = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.final_sum))?;

        // Allocate witness points and challenges.
        let mut r_prime: Vec<KVar> = Vec::with_capacity(self.ell_n);
        for i in 0..self.ell_n {
            let v = witness.as_ref().map(|w| w.r_prime[i]);
            r_prime.push(alloc_k_private(std_lib, layouter, v)?);
        }

        let mut alpha_prime: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = witness.as_ref().map(|w| w.alpha_prime[i]);
            alpha_prime.push(alloc_k_private(std_lib, layouter, v)?);
        }

        let mut beta_a: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = witness.as_ref().map(|w| w.beta_a[i]);
            beta_a.push(alloc_k_private(std_lib, layouter, v)?);
        }

        let mut beta_r: Vec<KVar> = Vec::with_capacity(self.ell_n);
        for i in 0..self.ell_n {
            let v = witness.as_ref().map(|w| w.beta_r[i]);
            beta_r.push(alloc_k_private(std_lib, layouter, v)?);
        }

        let mut y_scalars: Vec<KVar> = Vec::with_capacity(self.poly.t);
        for j in 0..self.poly.t {
            let v = witness.as_ref().map(|w| w.y_scalars[j]);
            y_scalars.push(alloc_k_private(std_lib, layouter, v)?);
        }

        // eq((α',r'), β) = eq(α',β_a) * eq(r',β_r)
        let eq_a = k_eq_points(std_lib, layouter, &alpha_prime, &beta_a)?;
        let eq_r = k_eq_points(std_lib, layouter, &r_prime, &beta_r)?;
        let eq_aprp_beta = k_mul_mod_var(std_lib, layouter, &eq_a, &eq_r, K_DELTA_U64)?;

        // F' = f(y_scalars[0..t]) in K.
        let f_prime = k_eval_sparse_poly_in_ext(std_lib, layouter, &self.poly, &y_scalars)?;

        let rhs = k_mul_mod_var(std_lib, layouter, &eq_aprp_beta, &f_prime, K_DELTA_U64)?;
        assert_k_eq(std_lib, layouter, &rhs, &final_sum)?;
        Ok(())
    }
}

/// FE-only terminal identity check for the SplitNcV1 variant for `k_total >= 2`.
///
/// This implements `rhs_terminal_identity_fe_paper_exact`:
///
/// `rhs = eq((α',r'), β) * F' + eq((α',r'), (α,r)) * (γ^k_total * Eval')`
///
/// where:
/// - `F' = f(y_scalars[0..t])` from the first ME output
/// - `Eval'` is the γ-weighted sum of Ajtai MLE evaluations over the remaining outputs
/// - `k_total = |out_me|` (the number of Π-CCS outputs at this step)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsFeTerminalRelation {
    pub ell_n: usize,
    pub ell_d: usize,
    pub k_total: usize,
    pub poly: SparsePolyRepr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsFeTerminalInstance {
    pub final_sum: KRepr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsFeTerminalWitness {
    pub r_prime: Vec<KRepr>,
    pub alpha_prime: Vec<KRepr>,
    pub alpha: Vec<KRepr>,
    pub beta_a: Vec<KRepr>,
    pub beta_r: Vec<KRepr>,
    pub gamma: KRepr,
    pub me_inputs_r: Vec<KRepr>,
    pub y_scalars_0: Vec<KRepr>,
    /// Ajtai digit rows `y[i_abs][j][rho]` for outputs `i_abs = 1..k_total-1`.
    pub y: Vec<Vec<Vec<KRepr>>>,
}

impl Relation for PiCcsFeTerminalRelation {
    type Instance = PiCcsFeTerminalInstance;
    type Witness = PiCcsFeTerminalWitness;

    fn used_chips(&self) -> ZkStdLibArch {
        // This circuit is range-check dominated (Goldilocks/K element canonicity),
        // so increase pow2range columns to reduce rows.
        let mut arch = ZkStdLibArch::default();
        arch.nr_pow2range_cols = 4;
        arch
    }

    fn format_instance(instance: &Self::Instance) -> Result<Vec<OuterScalar>, Error> {
        Ok(vec![
            OuterScalar::from(instance.final_sum.c0),
            OuterScalar::from(instance.final_sum.c1),
        ])
    }

    fn write_relation<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write_relation_len_prefixed(writer, self)
    }

    fn read_relation<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        read_relation_len_prefixed(reader)
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<OuterScalar>,
        instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        if self.ell_n == 0 {
            return Err(Error::Synthesis("PiCcsFeTerminalRelation requires ell_n > 0".into()));
        }
        if self.ell_d == 0 {
            return Err(Error::Synthesis("PiCcsFeTerminalRelation requires ell_d > 0".into()));
        }
        if self.k_total < 2 {
            return Err(Error::Synthesis("PiCcsFeTerminalRelation requires k_total >= 2".into()));
        }
        if self.poly.t == 0 {
            return Err(Error::Synthesis("PiCcsFeTerminalRelation requires poly.t > 0".into()));
        }
        if self.poly.terms.is_empty() {
            return Err(Error::Synthesis(
                "PiCcsFeTerminalRelation requires poly.terms non-empty".into(),
            ));
        }

        let final_sum = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.final_sum))?;

        // Allocate witness points and challenges.
        let mut r_prime: Vec<KVar> = Vec::with_capacity(self.ell_n);
        for i in 0..self.ell_n {
            let v = witness.as_ref().map(|w| w.r_prime[i]);
            r_prime.push(alloc_k_private(std_lib, layouter, v)?);
        }

        let mut alpha_prime: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = witness.as_ref().map(|w| w.alpha_prime[i]);
            alpha_prime.push(alloc_k_private(std_lib, layouter, v)?);
        }

        let mut alpha: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = witness.as_ref().map(|w| w.alpha[i]);
            alpha.push(alloc_k_private(std_lib, layouter, v)?);
        }

        let mut beta_a: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = witness.as_ref().map(|w| w.beta_a[i]);
            beta_a.push(alloc_k_private(std_lib, layouter, v)?);
        }

        let mut beta_r: Vec<KVar> = Vec::with_capacity(self.ell_n);
        for i in 0..self.ell_n {
            let v = witness.as_ref().map(|w| w.beta_r[i]);
            beta_r.push(alloc_k_private(std_lib, layouter, v)?);
        }

        let gamma = alloc_k_private(std_lib, layouter, witness.as_ref().map(|w| w.gamma))?;

        let mut me_inputs_r: Vec<KVar> = Vec::with_capacity(self.ell_n);
        for i in 0..self.ell_n {
            let v = witness.as_ref().map(|w| w.me_inputs_r[i]);
            me_inputs_r.push(alloc_k_private(std_lib, layouter, v)?);
        }

        let mut y_scalars_0: Vec<KVar> = Vec::with_capacity(self.poly.t);
        for j in 0..self.poly.t {
            let v = witness.as_ref().map(|w| w.y_scalars_0[j]);
            y_scalars_0.push(alloc_k_private(std_lib, layouter, v)?);
        }

        // eq((α',r'), β) = eq(α',β_a) * eq(r',β_r)
        let eq_a_beta = k_eq_points(std_lib, layouter, &alpha_prime, &beta_a)?;
        let eq_r_beta = k_eq_points(std_lib, layouter, &r_prime, &beta_r)?;
        let eq_aprp_beta = k_mul_mod_var(std_lib, layouter, &eq_a_beta, &eq_r_beta, K_DELTA_U64)?;

        // eq((α',r'), (α,r)) = eq(α',α) * eq(r',r)
        let eq_a_alpha = k_eq_points(std_lib, layouter, &alpha_prime, &alpha)?;
        let eq_r_r = k_eq_points(std_lib, layouter, &r_prime, &me_inputs_r)?;
        let eq_aprp_ar = k_mul_mod_var(std_lib, layouter, &eq_a_alpha, &eq_r_r, K_DELTA_U64)?;

        // F' = f(y_scalars_0[0..t]) in K.
        let f_prime = k_eval_sparse_poly_in_ext(std_lib, layouter, &self.poly, &y_scalars_0)?;

        // Precompute powers γ^i for i=0..k_total and γ^k_total.
        let one = k_one(std_lib, layouter)?;
        let mut gamma_pows: Vec<KVar> = Vec::with_capacity(self.k_total + 1);
        gamma_pows.push(one.clone());
        for i in 0..self.k_total {
            let next = k_mul_mod_var(std_lib, layouter, &gamma_pows[i], &gamma, K_DELTA_U64)?;
            gamma_pows.push(next);
        }
        let gamma_to_k_total = gamma_pows.get(self.k_total).expect("len checked").clone();

        // Eval' = Σ_{j=0..t-1} Σ_{i_abs=1..k_total-1} (γ^{i_abs} * (γ^k_total)^j) * <y_{i_abs,j}, χ_{α'}>
        let d_pad = 1usize
            .checked_shl(self.ell_d as u32)
            .ok_or_else(|| Error::Synthesis("PiCcsFeTerminalRelation: 1<<ell_d overflow".into()))?;
        let mut eval_sum = k_zero(std_lib, layouter)?;

        // Allocate digit rows and accumulate Eval' on the fly to avoid storing all KVars.
        for out_idx in 0..(self.k_total - 1) {
            let i_abs = out_idx + 1;
            let mut weight_j = gamma_pows
                .get(i_abs)
                .expect("gamma_pows sized for i_abs")
                .clone(); // γ^{i_abs}

            for j in 0..self.poly.t {
                // y_row length must be d_pad.
                let mut y_row: Vec<KVar> = Vec::with_capacity(d_pad);
                for rho in 0..d_pad {
                    let v = witness.as_ref().map(|w| w.y[out_idx][j][rho]);
                    y_row.push(alloc_k_private_u64(std_lib, layouter, v)?);
                }

                // Evaluate y_eval = <y_row, χ_{α'}> via multilinear folding.
                let mut eval_vec = y_row;
                for a in &alpha_prime {
                    let next_len = eval_vec.len() / 2;
                    let mut next: Vec<KVar> = Vec::with_capacity(next_len);
                    for t_idx in 0..next_len {
                        let v0 = &eval_vec[2 * t_idx];
                        let v1 = &eval_vec[2 * t_idx + 1];
                        next.push(k_mle_fold_step(std_lib, layouter, v0, v1, a, K_DELTA_U64)?);
                    }
                    eval_vec = next;
                }
                if eval_vec.len() != 1 {
                    return Err(Error::Synthesis(format!(
                        "PiCcsFeTerminalRelation: eval_vec len {} != 1 (ell_d={})",
                        eval_vec.len(),
                        self.ell_d
                    )));
                }
                let y_eval = eval_vec.first().expect("len checked").clone();

                let term = k_mul_mod_var(std_lib, layouter, &weight_j, &y_eval, K_DELTA_U64)?;
                eval_sum = k_add_mod_var(std_lib, layouter, &eval_sum, &term)?;

                if j + 1 < self.poly.t {
                    weight_j = k_mul_mod_var(std_lib, layouter, &weight_j, &gamma_to_k_total, K_DELTA_U64)?;
                }
            }
        }

        // rhs = eq_aprp_beta * F' + eq_aprp_ar * (γ^k_total * Eval')
        let term_f = k_mul_mod_var(std_lib, layouter, &eq_aprp_beta, &f_prime, K_DELTA_U64)?;
        let scaled_eval = k_mul_mod_var(std_lib, layouter, &gamma_to_k_total, &eval_sum, K_DELTA_U64)?;
        let term_eval = k_mul_mod_var(std_lib, layouter, &eq_aprp_ar, &scaled_eval, K_DELTA_U64)?;
        let rhs = k_add_mod_var(std_lib, layouter, &term_f, &term_eval)?;
        assert_k_eq(std_lib, layouter, &rhs, &final_sum)?;
        Ok(())
    }
}

/// Chunked helper for the FE terminal identity: computes a partial sum of the Ajtai MLE evaluations.
///
/// This outputs a `chunk_sum` for a contiguous range of `(out_idx, j)` pairs (flattened):
/// - `out_idx ∈ [0..k_total-1)` indexes Π-CCS outputs `i_abs = out_idx+1`
/// - `j ∈ [0..t)` indexes CCS polynomial outputs
/// - `flat = out_idx * t + j`
///
/// The chunk sum is:
/// `chunk_sum = Σ_{flat=start_idx..start_idx+count-1} (γ^{i_abs} * (γ^{k_total})^j) * <y_{out_idx,j}, χ_{α'}>`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsFeChunkRelation {
    pub ell_d: usize,
    pub k_total: usize,
    pub t: usize,
    pub start_idx: usize,
    pub count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsFeChunkInstance {
    /// Step-bundle binding digest limbs (little-endian u128×2).
    pub bundle_digest: [u128; 2],
    pub chunk_sum: KRepr,
    /// Split (r', α') from the FE sumcheck challenges (public).
    pub alpha_prime: Vec<KRepr>,
    /// Fiat–Shamir challenge γ (public).
    pub gamma: KRepr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsFeChunkWitness {
    pub y_rows: Vec<Vec<KRepr>>,
}

impl Relation for PiCcsFeChunkRelation {
    type Instance = PiCcsFeChunkInstance;
    type Witness = PiCcsFeChunkWitness;

    fn used_chips(&self) -> ZkStdLibArch {
        let mut arch = ZkStdLibArch::default();
        arch.nr_pow2range_cols = 4;
        arch
    }

    fn format_instance(instance: &Self::Instance) -> Result<Vec<OuterScalar>, Error> {
        let mut out = Vec::with_capacity(6 + 2 * instance.alpha_prime.len() + 2);
        out.push(OuterScalar::from_u128(instance.bundle_digest[0]));
        out.push(OuterScalar::from_u128(instance.bundle_digest[1]));
        out.push(OuterScalar::from(instance.chunk_sum.c0));
        out.push(OuterScalar::from(instance.chunk_sum.c1));
        for a in &instance.alpha_prime {
            out.push(OuterScalar::from(a.c0));
            out.push(OuterScalar::from(a.c1));
        }
        out.push(OuterScalar::from(instance.gamma.c0));
        out.push(OuterScalar::from(instance.gamma.c1));
        Ok(out)
    }

    fn write_relation<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write_relation_len_prefixed(writer, self)
    }

    fn read_relation<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        read_relation_len_prefixed(reader)
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<OuterScalar>,
        instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        if self.ell_d == 0 {
            return Err(Error::Synthesis("PiCcsFeChunkRelation requires ell_d > 0".into()));
        }
        if self.k_total < 2 {
            return Err(Error::Synthesis("PiCcsFeChunkRelation requires k_total >= 2".into()));
        }
        if self.t == 0 {
            return Err(Error::Synthesis("PiCcsFeChunkRelation requires t > 0".into()));
        }
        if self.count == 0 {
            return Err(Error::Synthesis("PiCcsFeChunkRelation requires count > 0".into()));
        }

        let total_pairs = (self.k_total - 1)
            .checked_mul(self.t)
            .ok_or_else(|| Error::Synthesis("PiCcsFeChunkRelation: total_pairs overflow".into()))?;
        let end_idx = self
            .start_idx
            .checked_add(self.count)
            .ok_or_else(|| Error::Synthesis("PiCcsFeChunkRelation: start_idx+count overflow".into()))?;
        if end_idx > total_pairs {
            return Err(Error::Synthesis(format!(
                "PiCcsFeChunkRelation: chunk range [{}, {}) out of bounds (total_pairs={})",
                self.start_idx, end_idx, total_pairs
            )));
        }

        // Public: bundle digest (anti mix-and-match across steps/runs).
        for limb_idx in 0..2 {
            let limb = instance.as_ref().map(|i| i.bundle_digest[limb_idx]);
            let assigned: AssignedNative<OuterScalar> = std_lib.assign(layouter, limb.map(OuterScalar::from_u128))?;
            std_lib.constrain_as_public_input(layouter, &assigned)?;
        }

        let chunk_sum = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.chunk_sum))?;

        // Public: α' and γ (Fiat–Shamir challenges).
        let mut alpha_prime: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = instance.as_ref().map(|ins| ins.alpha_prime[i]);
            alpha_prime.push(alloc_k_public(std_lib, layouter, v)?);
        }
        let gamma = alloc_k_public(std_lib, layouter, instance.as_ref().map(|ins| ins.gamma))?;

        let d_pad = 1usize
            .checked_shl(self.ell_d as u32)
            .ok_or_else(|| Error::Synthesis("PiCcsFeChunkRelation: 1<<ell_d overflow".into()))?;

        // Precompute powers γ^i for i=0..k_total and γ^k_total.
        let one = k_one(std_lib, layouter)?;
        let mut gamma_pows: Vec<KVar> = Vec::with_capacity(self.k_total + 1);
        gamma_pows.push(one.clone());
        for i in 0..self.k_total {
            let next = k_mul_mod_var(std_lib, layouter, &gamma_pows[i], &gamma, K_DELTA_U64)?;
            gamma_pows.push(next);
        }
        let gamma_to_k_total = gamma_pows.get(self.k_total).expect("len checked").clone();

        // Precompute (γ^k_total)^j for j=0..t-1.
        let mut gamma_k_pows: Vec<KVar> = Vec::with_capacity(self.t);
        gamma_k_pows.push(one.clone()); // j=0
        for _ in 1..self.t {
            let next = k_mul_mod_var(
                std_lib,
                layouter,
                gamma_k_pows.last().expect("non-empty"),
                &gamma_to_k_total,
                K_DELTA_U64,
            )?;
            gamma_k_pows.push(next);
        }

        let mut terms: Vec<KVar> = Vec::with_capacity(self.count);
        for local_idx in 0..self.count {
            let flat = self.start_idx + local_idx;
            let out_idx = flat / self.t;
            let j = flat % self.t;
            let i_abs = out_idx + 1;

            let gamma_i = gamma_pows
                .get(i_abs)
                .ok_or_else(|| Error::Synthesis("PiCcsFeChunkRelation: gamma_pows index".into()))?
                .clone();
            let weight = if j == 0 {
                gamma_i.clone()
            } else {
                let gamma_k_j = gamma_k_pows
                    .get(j)
                    .ok_or_else(|| Error::Synthesis("PiCcsFeChunkRelation: gamma_k_pows index".into()))?
                    .clone();
                k_mul_mod_var(std_lib, layouter, &gamma_i, &gamma_k_j, K_DELTA_U64)?
            };

            // Allocate y_row (u64-only) for this (out_idx,j) pair.
            let mut y_row: Vec<KVar> = Vec::with_capacity(d_pad);
            for rho in 0..d_pad {
                let v = witness.as_ref().map(|w| w.y_rows[local_idx][rho]);
                y_row.push(alloc_k_private_u64(std_lib, layouter, v)?);
            }

            // Evaluate <y_row, χ_{α'}> via multilinear folding.
            let mut eval_vec = y_row;
            for a in &alpha_prime {
                let next_len = eval_vec.len() / 2;
                let mut next: Vec<KVar> = Vec::with_capacity(next_len);
                for t_idx in 0..next_len {
                    let v0 = &eval_vec[2 * t_idx];
                    let v1 = &eval_vec[2 * t_idx + 1];
                    next.push(k_mle_fold_step(std_lib, layouter, v0, v1, a, K_DELTA_U64)?);
                }
                eval_vec = next;
            }
            if eval_vec.len() != 1 {
                return Err(Error::Synthesis(format!(
                    "PiCcsFeChunkRelation: eval_vec len {} != 1 (ell_d={})",
                    eval_vec.len(),
                    self.ell_d
                )));
            }
            let y_eval = eval_vec.first().expect("len checked").clone();

            let term = k_mul_mod_var(std_lib, layouter, &weight, &y_eval, K_DELTA_U64)?;
            terms.push(term);
        }

        let acc = k_sum_mod_var(std_lib, layouter, &terms)?;
        assert_k_eq(std_lib, layouter, &acc, &chunk_sum)?;
        Ok(())
    }
}

/// Aggregates FE-terminal chunk sums and checks the full FE terminal identity against `final_sum`.
///
/// Checks:
/// `final_sum == eq((α',r'), β) * F' + eq((α',r'), (α,r)) * (γ^k_total * Σ chunk_sums[i])`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsFeTerminalAggregateRelation {
    pub ell_n: usize,
    pub ell_d: usize,
    pub k_total: usize,
    pub poly: SparsePolyRepr,
    pub n_chunks: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsFeTerminalAggregateInstance {
    pub chunk_sums: Vec<KRepr>,
    pub final_sum: KRepr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsFeTerminalAggregateWitness {
    pub r_prime: Vec<KRepr>,
    pub alpha_prime: Vec<KRepr>,
    pub alpha: Vec<KRepr>,
    pub beta_a: Vec<KRepr>,
    pub beta_r: Vec<KRepr>,
    pub gamma: KRepr,
    pub me_inputs_r: Vec<KRepr>,
    pub y_scalars_0: Vec<KRepr>,
}

impl Relation for PiCcsFeTerminalAggregateRelation {
    type Instance = PiCcsFeTerminalAggregateInstance;
    type Witness = PiCcsFeTerminalAggregateWitness;

    fn used_chips(&self) -> ZkStdLibArch {
        let mut arch = ZkStdLibArch::default();
        arch.nr_pow2range_cols = 4;
        arch
    }

    fn format_instance(instance: &Self::Instance) -> Result<Vec<OuterScalar>, Error> {
        let mut out = Vec::with_capacity(2 * (instance.chunk_sums.len() + 1));
        for cs in &instance.chunk_sums {
            out.push(OuterScalar::from(cs.c0));
            out.push(OuterScalar::from(cs.c1));
        }
        out.push(OuterScalar::from(instance.final_sum.c0));
        out.push(OuterScalar::from(instance.final_sum.c1));
        Ok(out)
    }

    fn write_relation<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write_relation_len_prefixed(writer, self)
    }

    fn read_relation<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        read_relation_len_prefixed(reader)
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<OuterScalar>,
        instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        if self.ell_n == 0 {
            return Err(Error::Synthesis(
                "PiCcsFeTerminalAggregateRelation requires ell_n > 0".into(),
            ));
        }
        if self.ell_d == 0 {
            return Err(Error::Synthesis(
                "PiCcsFeTerminalAggregateRelation requires ell_d > 0".into(),
            ));
        }
        if self.k_total < 2 {
            return Err(Error::Synthesis(
                "PiCcsFeTerminalAggregateRelation requires k_total >= 2".into(),
            ));
        }
        if self.poly.t == 0 {
            return Err(Error::Synthesis(
                "PiCcsFeTerminalAggregateRelation requires poly.t > 0".into(),
            ));
        }
        if self.poly.terms.is_empty() {
            return Err(Error::Synthesis(
                "PiCcsFeTerminalAggregateRelation requires poly.terms non-empty".into(),
            ));
        }
        if self.n_chunks == 0 {
            return Err(Error::Synthesis(
                "PiCcsFeTerminalAggregateRelation requires n_chunks > 0".into(),
            ));
        }

        // Public: chunk sums + final_sum.
        let mut chunk_sums: Vec<KVar> = Vec::with_capacity(self.n_chunks);
        for i in 0..self.n_chunks {
            let v = instance.as_ref().map(|ins| ins.chunk_sums[i]);
            chunk_sums.push(alloc_k_public(std_lib, layouter, v)?);
        }
        let final_sum = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.final_sum))?;

        // Allocate witness points and challenges.
        let mut r_prime: Vec<KVar> = Vec::with_capacity(self.ell_n);
        for i in 0..self.ell_n {
            let v = witness.as_ref().map(|w| w.r_prime[i]);
            r_prime.push(alloc_k_private(std_lib, layouter, v)?);
        }

        let mut alpha_prime: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = witness.as_ref().map(|w| w.alpha_prime[i]);
            alpha_prime.push(alloc_k_private(std_lib, layouter, v)?);
        }

        let mut alpha: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = witness.as_ref().map(|w| w.alpha[i]);
            alpha.push(alloc_k_private(std_lib, layouter, v)?);
        }

        let mut beta_a: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = witness.as_ref().map(|w| w.beta_a[i]);
            beta_a.push(alloc_k_private(std_lib, layouter, v)?);
        }

        let mut beta_r: Vec<KVar> = Vec::with_capacity(self.ell_n);
        for i in 0..self.ell_n {
            let v = witness.as_ref().map(|w| w.beta_r[i]);
            beta_r.push(alloc_k_private(std_lib, layouter, v)?);
        }

        let gamma = alloc_k_private(std_lib, layouter, witness.as_ref().map(|w| w.gamma))?;

        let mut me_inputs_r: Vec<KVar> = Vec::with_capacity(self.ell_n);
        for i in 0..self.ell_n {
            let v = witness.as_ref().map(|w| w.me_inputs_r[i]);
            me_inputs_r.push(alloc_k_private(std_lib, layouter, v)?);
        }

        let mut y_scalars_0: Vec<KVar> = Vec::with_capacity(self.poly.t);
        for j in 0..self.poly.t {
            let v = witness.as_ref().map(|w| w.y_scalars_0[j]);
            y_scalars_0.push(alloc_k_private(std_lib, layouter, v)?);
        }

        // eq((α',r'), β) = eq(α',β_a) * eq(r',β_r)
        let eq_a_beta = k_eq_points(std_lib, layouter, &alpha_prime, &beta_a)?;
        let eq_r_beta = k_eq_points(std_lib, layouter, &r_prime, &beta_r)?;
        let eq_aprp_beta = k_mul_mod_var(std_lib, layouter, &eq_a_beta, &eq_r_beta, K_DELTA_U64)?;

        // eq((α',r'), (α,r)) = eq(α',α) * eq(r',r)
        let eq_a_alpha = k_eq_points(std_lib, layouter, &alpha_prime, &alpha)?;
        let eq_r_r = k_eq_points(std_lib, layouter, &r_prime, &me_inputs_r)?;
        let eq_aprp_ar = k_mul_mod_var(std_lib, layouter, &eq_a_alpha, &eq_r_r, K_DELTA_U64)?;

        // F' = f(y_scalars_0[0..t]) in K.
        let f_prime = k_eval_sparse_poly_in_ext(std_lib, layouter, &self.poly, &y_scalars_0)?;

        // γ^k_total.
        let mut gamma_to_k_total = k_one(std_lib, layouter)?;
        for _ in 0..self.k_total {
            gamma_to_k_total = k_mul_mod_var(std_lib, layouter, &gamma_to_k_total, &gamma, K_DELTA_U64)?;
        }

        let mut eval_sum = k_zero(std_lib, layouter)?;
        for cs in &chunk_sums {
            eval_sum = k_add_mod_var(std_lib, layouter, &eval_sum, cs)?;
        }

        // rhs = eq_aprp_beta * F' + eq_aprp_ar * (γ^k_total * Eval')
        let term_f = k_mul_mod_var(std_lib, layouter, &eq_aprp_beta, &f_prime, K_DELTA_U64)?;
        let scaled_eval = k_mul_mod_var(std_lib, layouter, &gamma_to_k_total, &eval_sum, K_DELTA_U64)?;
        let term_eval = k_mul_mod_var(std_lib, layouter, &eq_aprp_ar, &scaled_eval, K_DELTA_U64)?;
        let rhs = k_add_mod_var(std_lib, layouter, &term_f, &term_eval)?;
        assert_k_eq(std_lib, layouter, &rhs, &final_sum)?;
        Ok(())
    }
}

/// Combines one FE-terminal chunk with:
/// - the FE sumcheck verification, and
/// - the FE terminal identity aggregate check.
///
/// This lets a bundle drop two proofs (sumcheck + aggregate) by proving them together, while
/// also binding one `chunk_sums[chunk_index]` via an in-circuit recomputation of that chunk.
///
/// Set `count=0` to skip the chunk-binding section (sumcheck + aggregate only).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsFeChunkAggSumcheckRelation {
    // Sumcheck parameters.
    pub n_rounds: usize,
    pub poly_len: usize,
    // Terminal identity parameters.
    pub ell_n: usize,
    pub ell_d: usize,
    pub k_total: usize,
    pub poly: SparsePolyRepr,
    // Chunk binding parameters.
    pub start_idx: usize,
    pub count: usize,
    pub n_chunks: usize,
    pub chunk_index: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsFeChunkAggSumcheckInstance {
    /// Step-bundle binding digest limbs (little-endian u128×2).
    pub bundle_digest: [u128; 2],
    /// FE sumcheck challenges (r' || α') (public).
    pub sumcheck_challenges: Vec<KRepr>,
    /// Fiat–Shamir challenge γ (public).
    pub gamma: KRepr,
    /// SplitNcV1 transcript-derived values (public).
    pub alpha: Vec<KRepr>,
    pub beta_a: Vec<KRepr>,
    pub beta_r: Vec<KRepr>,
    pub chunk_sums: Vec<KRepr>,
    pub initial_sum: KRepr,
    pub final_sum: KRepr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsFeChunkAggSumcheckWitness {
    /// FE sumcheck round polynomials (private): `rounds[round_idx][coeff_idx]`.
    pub rounds: Vec<Vec<KRepr>>,
    // ME input point r for eq(r',r).
    pub me_inputs_r: Vec<KRepr>,
    // CCS polynomial evaluation point for F'.
    pub y_scalars_0: Vec<KRepr>,
    // Digit rows for the designated chunk (length = count, each padded to 2^ell_d).
    pub y_rows: Vec<Vec<KRepr>>,
}

impl Relation for PiCcsFeChunkAggSumcheckRelation {
    type Instance = PiCcsFeChunkAggSumcheckInstance;
    type Witness = PiCcsFeChunkAggSumcheckWitness;

    fn used_chips(&self) -> ZkStdLibArch {
        let mut arch = ZkStdLibArch::default();
        arch.nr_pow2range_cols = 4;
        arch
    }

    fn format_instance(instance: &Self::Instance) -> Result<Vec<OuterScalar>, Error> {
        let mut out = Vec::with_capacity(
            2 + 2 * instance.sumcheck_challenges.len()
                + 2
                + 2 * (instance.alpha.len() + instance.beta_a.len() + instance.beta_r.len())
                + 2 * (instance.chunk_sums.len() + 2),
        );
        out.push(OuterScalar::from_u128(instance.bundle_digest[0]));
        out.push(OuterScalar::from_u128(instance.bundle_digest[1]));
        for ch in &instance.sumcheck_challenges {
            out.push(OuterScalar::from(ch.c0));
            out.push(OuterScalar::from(ch.c1));
        }
        out.push(OuterScalar::from(instance.gamma.c0));
        out.push(OuterScalar::from(instance.gamma.c1));
        for a in &instance.alpha {
            out.push(OuterScalar::from(a.c0));
            out.push(OuterScalar::from(a.c1));
        }
        for b in &instance.beta_a {
            out.push(OuterScalar::from(b.c0));
            out.push(OuterScalar::from(b.c1));
        }
        for b in &instance.beta_r {
            out.push(OuterScalar::from(b.c0));
            out.push(OuterScalar::from(b.c1));
        }
        for cs in &instance.chunk_sums {
            out.push(OuterScalar::from(cs.c0));
            out.push(OuterScalar::from(cs.c1));
        }
        out.push(OuterScalar::from(instance.initial_sum.c0));
        out.push(OuterScalar::from(instance.initial_sum.c1));
        out.push(OuterScalar::from(instance.final_sum.c0));
        out.push(OuterScalar::from(instance.final_sum.c1));
        Ok(out)
    }

    fn write_relation<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write_relation_len_prefixed(writer, self)
    }

    fn read_relation<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        read_relation_len_prefixed(reader)
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<OuterScalar>,
        instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        if self.n_rounds == 0 {
            return Err(Error::Synthesis(
                "PiCcsFeChunkAggSumcheckRelation requires n_rounds > 0".into(),
            ));
        }
        if self.poly_len == 0 {
            return Err(Error::Synthesis(
                "PiCcsFeChunkAggSumcheckRelation requires poly_len > 0".into(),
            ));
        }
        if self.ell_n == 0 {
            return Err(Error::Synthesis(
                "PiCcsFeChunkAggSumcheckRelation requires ell_n > 0".into(),
            ));
        }
        if self.ell_d == 0 {
            return Err(Error::Synthesis(
                "PiCcsFeChunkAggSumcheckRelation requires ell_d > 0".into(),
            ));
        }
        if self.k_total < 2 {
            return Err(Error::Synthesis(
                "PiCcsFeChunkAggSumcheckRelation requires k_total >= 2".into(),
            ));
        }
        if self.poly.t == 0 {
            return Err(Error::Synthesis(
                "PiCcsFeChunkAggSumcheckRelation requires poly.t > 0".into(),
            ));
        }
        if self.poly.terms.is_empty() {
            return Err(Error::Synthesis(
                "PiCcsFeChunkAggSumcheckRelation requires poly.terms non-empty".into(),
            ));
        }
        if self.n_chunks == 0 {
            return Err(Error::Synthesis(
                "PiCcsFeChunkAggSumcheckRelation requires n_chunks > 0".into(),
            ));
        }
        if self.chunk_index >= self.n_chunks {
            return Err(Error::Synthesis(
                "PiCcsFeChunkAggSumcheckRelation: chunk_index out of bounds".into(),
            ));
        }
        if self.n_rounds != self.ell_n + self.ell_d {
            return Err(Error::Synthesis(format!(
                "PiCcsFeChunkAggSumcheckRelation: n_rounds {} != ell_n+ell_d {}",
                self.n_rounds,
                self.ell_n + self.ell_d
            )));
        }

        let total_pairs = (self.k_total - 1)
            .checked_mul(self.poly.t)
            .ok_or_else(|| Error::Synthesis("PiCcsFeChunkAggSumcheckRelation: total_pairs overflow".into()))?;
        let end_idx = self
            .start_idx
            .checked_add(self.count)
            .ok_or_else(|| Error::Synthesis("PiCcsFeChunkAggSumcheckRelation: start_idx+count overflow".into()))?;
        if end_idx > total_pairs {
            return Err(Error::Synthesis(format!(
                "PiCcsFeChunkAggSumcheckRelation: chunk range [{}, {}) out of bounds (total_pairs={})",
                self.start_idx, end_idx, total_pairs
            )));
        }

        // Public: bundle digest (anti mix-and-match across steps/runs).
        for limb_idx in 0..2 {
            let limb = instance.as_ref().map(|i| i.bundle_digest[limb_idx]);
            let assigned: AssignedNative<OuterScalar> = std_lib.assign(layouter, limb.map(OuterScalar::from_u128))?;
            std_lib.constrain_as_public_input(layouter, &assigned)?;
        }

        // Public inputs (in the exact order of `format_instance`):
        // - bundle digest
        // - sumcheck challenges
        // - γ
        // - α, β_a, β_r
        // - all chunk sums
        // - (initial_sum, final_sum)
        let mut challenges: Vec<KVar> = Vec::with_capacity(self.n_rounds);
        for r in 0..self.n_rounds {
            let ch = instance.as_ref().map(|i| i.sumcheck_challenges[r]);
            challenges.push(alloc_k_public(std_lib, layouter, ch)?);
        }

        // Private witness: sumcheck rounds.
        let mut rounds: Vec<Vec<KVar>> = Vec::with_capacity(self.n_rounds);
        for r in 0..self.n_rounds {
            let mut coeffs_r = Vec::with_capacity(self.poly_len);
            for j in 0..self.poly_len {
                let coeff = witness.as_ref().map(|w| w.rounds[r][j]);
                coeffs_r.push(alloc_k_private(std_lib, layouter, coeff)?);
            }
            rounds.push(coeffs_r);
        }

        let gamma = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.gamma))?;

        let mut alpha: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = instance.as_ref().map(|ins| ins.alpha[i]);
            alpha.push(alloc_k_public(std_lib, layouter, v)?);
        }
        let mut beta_a: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = instance.as_ref().map(|ins| ins.beta_a[i]);
            beta_a.push(alloc_k_public(std_lib, layouter, v)?);
        }
        let mut beta_r: Vec<KVar> = Vec::with_capacity(self.ell_n);
        for i in 0..self.ell_n {
            let v = instance.as_ref().map(|ins| ins.beta_r[i]);
            beta_r.push(alloc_k_public(std_lib, layouter, v)?);
        }

        let mut chunk_sums: Vec<KVar> = Vec::with_capacity(self.n_chunks);
        for i in 0..self.n_chunks {
            let v = instance.as_ref().map(|ins| ins.chunk_sums[i]);
            chunk_sums.push(alloc_k_public(std_lib, layouter, v)?);
        }
        let initial_sum = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.initial_sum))?;
        let final_sum = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.final_sum))?;

        // --- Sumcheck ---
        let mut running_sum = initial_sum.clone();
        for r in 0..self.n_rounds {
            sumcheck_round_check(std_lib, layouter, &rounds[r], &running_sum)?;
            running_sum = sumcheck_eval_horner(std_lib, layouter, &rounds[r], &challenges[r], K_DELTA_U64)?;
        }
        assert_k_eq(std_lib, layouter, &running_sum, &final_sum)?;

        let (r_prime, alpha_prime) = challenges.split_at(self.ell_n);

        // --- Terminal identity aggregate ---
        let mut me_inputs_r: Vec<KVar> = Vec::with_capacity(self.ell_n);
        for i in 0..self.ell_n {
            let v = witness.as_ref().map(|w| w.me_inputs_r[i]);
            me_inputs_r.push(alloc_k_private(std_lib, layouter, v)?);
        }
        let mut y_scalars_0: Vec<KVar> = Vec::with_capacity(self.poly.t);
        for j in 0..self.poly.t {
            let v = witness.as_ref().map(|w| w.y_scalars_0[j]);
            y_scalars_0.push(alloc_k_private(std_lib, layouter, v)?);
        }

        let eq_a_beta = k_eq_points(std_lib, layouter, alpha_prime, &beta_a)?;
        let eq_r_beta = k_eq_points(std_lib, layouter, r_prime, &beta_r)?;
        let eq_aprp_beta = k_mul_mod_var(std_lib, layouter, &eq_a_beta, &eq_r_beta, K_DELTA_U64)?;

        let eq_a_alpha = k_eq_points(std_lib, layouter, alpha_prime, &alpha)?;
        let eq_r_r = k_eq_points(std_lib, layouter, r_prime, &me_inputs_r)?;
        let eq_aprp_ar = k_mul_mod_var(std_lib, layouter, &eq_a_alpha, &eq_r_r, K_DELTA_U64)?;

        let f_prime = k_eval_sparse_poly_in_ext(std_lib, layouter, &self.poly, &y_scalars_0)?;

        // Precompute powers γ^i for i=0..k_total and reuse γ^k_total both for the
        // terminal identity aggregate and for binding the designated chunk.
        //
        // This avoids duplicating the `k_total` multiplications that would be needed to compute
        // γ^k_total separately from γ^i.
        let one = k_one(std_lib, layouter)?;
        let mut gamma_pows: Vec<KVar> = Vec::with_capacity(self.k_total + 1);
        gamma_pows.push(one.clone());
        for i in 0..self.k_total {
            let next = k_mul_mod_var(std_lib, layouter, &gamma_pows[i], &gamma, K_DELTA_U64)?;
            gamma_pows.push(next);
        }
        let gamma_to_k_total = gamma_pows
            .get(self.k_total)
            .ok_or_else(|| Error::Synthesis("PiCcsFeChunkAggSumcheckRelation: gamma_pows index".into()))?
            .clone();

        let mut eval_sum = k_zero(std_lib, layouter)?;
        for cs in &chunk_sums {
            eval_sum = k_add_mod_var(std_lib, layouter, &eval_sum, cs)?;
        }

        let term_f = k_mul_mod_var(std_lib, layouter, &eq_aprp_beta, &f_prime, K_DELTA_U64)?;
        let scaled_eval = k_mul_mod_var(std_lib, layouter, &gamma_to_k_total, &eval_sum, K_DELTA_U64)?;
        let term_eval = k_mul_mod_var(std_lib, layouter, &eq_aprp_ar, &scaled_eval, K_DELTA_U64)?;
        let rhs = k_add_mod_var(std_lib, layouter, &term_f, &term_eval)?;
        assert_k_eq(std_lib, layouter, &rhs, &final_sum)?;

        // --- Bind one chunk sum (optional) ---
        if self.count == 0 {
            return Ok(());
        }

        let d_pad = 1usize
            .checked_shl(self.ell_d as u32)
            .ok_or_else(|| Error::Synthesis("PiCcsFeChunkAggSumcheckRelation: 1<<ell_d overflow".into()))?;

        let mut gamma_k_pows: Vec<KVar> = Vec::with_capacity(self.poly.t);
        gamma_k_pows.push(one.clone());
        for _ in 1..self.poly.t {
            let next = k_mul_mod_var(
                std_lib,
                layouter,
                gamma_k_pows.last().expect("non-empty"),
                &gamma_to_k_total,
                K_DELTA_U64,
            )?;
            gamma_k_pows.push(next);
        }

        let mut acc = k_zero(std_lib, layouter)?;
        for local_idx in 0..self.count {
            let flat = self.start_idx + local_idx;
            let out_idx = flat / self.poly.t;
            let j = flat % self.poly.t;
            let i_abs = out_idx + 1;

            let gamma_i = gamma_pows
                .get(i_abs)
                .ok_or_else(|| Error::Synthesis("PiCcsFeChunkAggSumcheckRelation: gamma_pows index".into()))?
                .clone();
            let weight = if j == 0 {
                gamma_i.clone()
            } else {
                let gamma_k_j = gamma_k_pows
                    .get(j)
                    .ok_or_else(|| Error::Synthesis("PiCcsFeChunkAggSumcheckRelation: gamma_k_pows index".into()))?
                    .clone();
                k_mul_mod_var(std_lib, layouter, &gamma_i, &gamma_k_j, K_DELTA_U64)?
            };

            let mut y_row: Vec<KVar> = Vec::with_capacity(d_pad);
            for rho in 0..d_pad {
                let v = witness.as_ref().map(|w| w.y_rows[local_idx][rho]);
                y_row.push(alloc_k_private_u64(std_lib, layouter, v)?);
            }

            let mut eval_vec = y_row;
            for a in alpha_prime {
                let next_len = eval_vec.len() / 2;
                let mut next: Vec<KVar> = Vec::with_capacity(next_len);
                for t_idx in 0..next_len {
                    let v0 = &eval_vec[2 * t_idx];
                    let v1 = &eval_vec[2 * t_idx + 1];
                    next.push(k_mle_fold_step(std_lib, layouter, v0, v1, a, K_DELTA_U64)?);
                }
                eval_vec = next;
            }
            if eval_vec.len() != 1 {
                return Err(Error::Synthesis(format!(
                    "PiCcsFeChunkAggSumcheckRelation: eval_vec len {} != 1 (ell_d={})",
                    eval_vec.len(),
                    self.ell_d
                )));
            }
            let y_eval = eval_vec.first().expect("len checked").clone();

            let term = k_mul_mod_var(std_lib, layouter, &weight, &y_eval, K_DELTA_U64)?;
            acc = k_add_mod_var(std_lib, layouter, &acc, &term)?;
        }

        let expected = chunk_sums
            .get(self.chunk_index)
            .ok_or_else(|| Error::Synthesis("PiCcsFeChunkAggSumcheckRelation: chunk_index".into()))?
            .clone();
        assert_k_eq(std_lib, layouter, &acc, &expected)?;

        Ok(())
    }
}

/// NC-only terminal identity check for the SplitNcV1 variant in the **k=1** case.
///
/// For step 0 of a `FoldRun` created without an initial accumulator, we have:
/// - `k_total = |out_me| = 1`
///
/// so the verifier RHS simplifies to:
///
/// `rhs = eq((α',s'), (β_a,β_m)) * γ * range_product( <y_zcol, χ_{α'}> )`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsNcTerminalK1Relation {
    pub ell_d: usize,
    pub ell_m: usize,
    pub b: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsNcTerminalK1Instance {
    pub final_sum_nc: KRepr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsNcTerminalK1Witness {
    pub s_col_prime: Vec<KRepr>,
    pub alpha_prime: Vec<KRepr>,
    pub beta_a: Vec<KRepr>,
    pub beta_m: Vec<KRepr>,
    pub gamma: KRepr,
    pub y_zcol: Vec<KRepr>,
}

impl Relation for PiCcsNcTerminalK1Relation {
    type Instance = PiCcsNcTerminalK1Instance;
    type Witness = PiCcsNcTerminalK1Witness;

    fn used_chips(&self) -> ZkStdLibArch {
        let mut arch = ZkStdLibArch::default();
        arch.nr_pow2range_cols = 4;
        arch
    }

    fn format_instance(instance: &Self::Instance) -> Result<Vec<OuterScalar>, Error> {
        Ok(vec![
            OuterScalar::from(instance.final_sum_nc.c0),
            OuterScalar::from(instance.final_sum_nc.c1),
        ])
    }

    fn write_relation<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write_relation_len_prefixed(writer, self)
    }

    fn read_relation<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        read_relation_len_prefixed(reader)
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<OuterScalar>,
        instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        if self.ell_d == 0 {
            return Err(Error::Synthesis("PiCcsNcTerminalK1Relation requires ell_d > 0".into()));
        }
        if self.ell_m == 0 {
            return Err(Error::Synthesis("PiCcsNcTerminalK1Relation requires ell_m > 0".into()));
        }
        if self.b < 2 {
            return Err(Error::Synthesis("PiCcsNcTerminalK1Relation requires b >= 2".into()));
        }

        let final_sum_nc = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.final_sum_nc))?;

        // Allocate witness vectors.
        let mut s_col_prime: Vec<KVar> = Vec::with_capacity(self.ell_m);
        for i in 0..self.ell_m {
            let v = witness.as_ref().map(|w| w.s_col_prime[i]);
            s_col_prime.push(alloc_k_private(std_lib, layouter, v)?);
        }
        let mut alpha_prime: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = witness.as_ref().map(|w| w.alpha_prime[i]);
            alpha_prime.push(alloc_k_private(std_lib, layouter, v)?);
        }
        let mut beta_a: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = witness.as_ref().map(|w| w.beta_a[i]);
            beta_a.push(alloc_k_private(std_lib, layouter, v)?);
        }
        let mut beta_m: Vec<KVar> = Vec::with_capacity(self.ell_m);
        for i in 0..self.ell_m {
            let v = witness.as_ref().map(|w| w.beta_m[i]);
            beta_m.push(alloc_k_private(std_lib, layouter, v)?);
        }
        let gamma = alloc_k_private(std_lib, layouter, witness.as_ref().map(|w| w.gamma))?;

        let d_pad = 1usize
            .checked_shl(self.ell_d as u32)
            .ok_or_else(|| Error::Synthesis("PiCcsNcTerminalK1Relation: 1<<ell_d overflow".into()))?;
        let mut y_zcol: Vec<KVar> = Vec::with_capacity(d_pad);
        for i in 0..d_pad {
            let v = witness.as_ref().map(|w| w.y_zcol[i]);
            y_zcol.push(alloc_k_private_u64(std_lib, layouter, v)?);
        }

        // eq((α',s'), (β_a,β_m)) = eq(α',β_a) * eq(s',β_m)
        let eq_a = k_eq_points(std_lib, layouter, &alpha_prime, &beta_a)?;
        let eq_s = k_eq_points(std_lib, layouter, &s_col_prime, &beta_m)?;
        let eq_apsp_beta = k_mul_mod_var(std_lib, layouter, &eq_a, &eq_s, K_DELTA_U64)?;

        // Evaluate y_eval = <y_zcol, χ_{α'}> using the standard multilinear evaluation
        // algorithm (in-place folding), which costs (2^ell_d - 1) K-multiplications.
        //
        // Bit order is LSB-first (alpha_prime[0] corresponds to the lowest index bit),
        // matching the `rho >> bit` convention in the paper-exact reference.
        let mut eval_vec = y_zcol;
        for a in &alpha_prime {
            let next_len = eval_vec.len() / 2;
            let mut next: Vec<KVar> = Vec::with_capacity(next_len);
            for j in 0..next_len {
                let v0 = &eval_vec[2 * j];
                let v1 = &eval_vec[2 * j + 1];
                next.push(k_mle_fold_step(std_lib, layouter, v0, v1, a, K_DELTA_U64)?);
            }
            eval_vec = next;
        }
        if eval_vec.len() != 1 {
            return Err(Error::Synthesis(format!(
                "PiCcsNcTerminalK1Relation: eval_vec len {} != 1 (ell_d={})",
                eval_vec.len(),
                self.ell_d
            )));
        }
        let y_eval = eval_vec.first().expect("len checked").clone();

        // range_product(y_eval) = ∏_{t=-(b-1)}^{b-1} (y_eval - t)
        let lo = -((self.b as i64) - 1);
        let hi = (self.b as i64) - 1;
        let one = k_one(std_lib, layouter)?;
        let mut range_prod = one.clone();
        for t in lo..=hi {
            let t_u64 = if t >= 0 {
                t as u64
            } else {
                GOLDILOCKS_P_U64
                    .checked_sub((-t) as u64)
                    .ok_or_else(|| Error::Synthesis("PiCcsNcTerminalK1Relation: t underflow".into()))?
            };
            let t_k = k_const(std_lib, layouter, t_u64, 0)?;
            let term = k_sub_mod_var(std_lib, layouter, &y_eval, &t_k)?;
            range_prod = k_mul_mod_var(std_lib, layouter, &range_prod, &term, K_DELTA_U64)?;
        }

        // rhs = eq_apsp_beta * gamma * range_product
        let gamma_times = k_mul_mod_var(std_lib, layouter, &gamma, &range_prod, K_DELTA_U64)?;
        let rhs = k_mul_mod_var(std_lib, layouter, &eq_apsp_beta, &gamma_times, K_DELTA_U64)?;
        assert_k_eq(std_lib, layouter, &rhs, &final_sum_nc)?;
        Ok(())
    }
}

/// NC-only terminal identity check for the SplitNcV1 variant for an arbitrary output count.
///
/// This checks:
/// `final_sum_nc == eq((α',s'), (β_a,β_m)) * Σ_{i=0..k_total-1} γ^{i+1} * range_product(<y_zcol_i, χ_{α'}>)`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsNcTerminalRelation {
    pub ell_d: usize,
    pub ell_m: usize,
    pub b: u32,
    pub k_total: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsNcTerminalWitness {
    pub s_col_prime: Vec<KRepr>,
    pub alpha_prime: Vec<KRepr>,
    pub beta_a: Vec<KRepr>,
    pub beta_m: Vec<KRepr>,
    pub gamma: KRepr,
    pub y_zcol: Vec<Vec<KRepr>>,
}

impl Relation for PiCcsNcTerminalRelation {
    type Instance = PiCcsNcTerminalK1Instance;
    type Witness = PiCcsNcTerminalWitness;

    fn used_chips(&self) -> ZkStdLibArch {
        let mut arch = ZkStdLibArch::default();
        arch.nr_pow2range_cols = 4;
        arch
    }

    fn format_instance(instance: &Self::Instance) -> Result<Vec<OuterScalar>, Error> {
        Ok(vec![
            OuterScalar::from(instance.final_sum_nc.c0),
            OuterScalar::from(instance.final_sum_nc.c1),
        ])
    }

    fn write_relation<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write_relation_len_prefixed(writer, self)
    }

    fn read_relation<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        read_relation_len_prefixed(reader)
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<OuterScalar>,
        instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        if self.ell_d == 0 {
            return Err(Error::Synthesis("PiCcsNcTerminalRelation requires ell_d > 0".into()));
        }
        if self.ell_m == 0 {
            return Err(Error::Synthesis("PiCcsNcTerminalRelation requires ell_m > 0".into()));
        }
        if self.b < 2 {
            return Err(Error::Synthesis("PiCcsNcTerminalRelation requires b >= 2".into()));
        }
        if self.k_total == 0 {
            return Err(Error::Synthesis("PiCcsNcTerminalRelation requires k_total > 0".into()));
        }

        let final_sum_nc = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.final_sum_nc))?;

        // Allocate witness vectors.
        let mut s_col_prime: Vec<KVar> = Vec::with_capacity(self.ell_m);
        for i in 0..self.ell_m {
            let v = witness.as_ref().map(|w| w.s_col_prime[i]);
            s_col_prime.push(alloc_k_private(std_lib, layouter, v)?);
        }
        let mut alpha_prime: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = witness.as_ref().map(|w| w.alpha_prime[i]);
            alpha_prime.push(alloc_k_private(std_lib, layouter, v)?);
        }
        let mut beta_a: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = witness.as_ref().map(|w| w.beta_a[i]);
            beta_a.push(alloc_k_private(std_lib, layouter, v)?);
        }
        let mut beta_m: Vec<KVar> = Vec::with_capacity(self.ell_m);
        for i in 0..self.ell_m {
            let v = witness.as_ref().map(|w| w.beta_m[i]);
            beta_m.push(alloc_k_private(std_lib, layouter, v)?);
        }
        let gamma = alloc_k_private(std_lib, layouter, witness.as_ref().map(|w| w.gamma))?;

        // eq((α',s'), (β_a,β_m)) = eq(α',β_a) * eq(s',β_m)
        let eq_a = k_eq_points(std_lib, layouter, &alpha_prime, &beta_a)?;
        let eq_s = k_eq_points(std_lib, layouter, &s_col_prime, &beta_m)?;
        let eq_apsp_beta = k_mul_mod_var(std_lib, layouter, &eq_a, &eq_s, K_DELTA_U64)?;

        let d_pad = 1usize
            .checked_shl(self.ell_d as u32)
            .ok_or_else(|| Error::Synthesis("PiCcsNcTerminalRelation: 1<<ell_d overflow".into()))?;

        // Accumulate Σ γ^{i+1} * range_product(<y_zcol_i, χ_{α'}>).
        let one = k_one(std_lib, layouter)?;
        let mut nc_sum = k_zero(std_lib, layouter)?;
        let mut g = gamma.clone(); // γ^1

        for out_idx in 0..self.k_total {
            // Allocate y_zcol for this output.
            let mut y_zcol: Vec<KVar> = Vec::with_capacity(d_pad);
            for i in 0..d_pad {
                let v = witness.as_ref().map(|w| w.y_zcol[out_idx][i]);
                y_zcol.push(alloc_k_private_u64(std_lib, layouter, v)?);
            }

            // Evaluate y_eval = <y_zcol, χ_{α'}> via multilinear folding.
            let mut eval_vec = y_zcol;
            for a in &alpha_prime {
                let next_len = eval_vec.len() / 2;
                let mut next: Vec<KVar> = Vec::with_capacity(next_len);
                for j in 0..next_len {
                    let v0 = &eval_vec[2 * j];
                    let v1 = &eval_vec[2 * j + 1];
                    next.push(k_mle_fold_step(std_lib, layouter, v0, v1, a, K_DELTA_U64)?);
                }
                eval_vec = next;
            }
            if eval_vec.len() != 1 {
                return Err(Error::Synthesis(format!(
                    "PiCcsNcTerminalRelation: eval_vec len {} != 1 (ell_d={})",
                    eval_vec.len(),
                    self.ell_d
                )));
            }
            let y_eval = eval_vec.first().expect("len checked").clone();

            // range_product(y_eval) = ∏_{t=-(b-1)}^{b-1} (y_eval - t)
            let lo = -((self.b as i64) - 1);
            let hi = (self.b as i64) - 1;
            let mut range_prod = one.clone();
            for t in lo..=hi {
                let t_u64 = if t >= 0 {
                    t as u64
                } else {
                    GOLDILOCKS_P_U64
                        .checked_sub((-t) as u64)
                        .ok_or_else(|| Error::Synthesis("PiCcsNcTerminalRelation: t underflow".into()))?
                };
                let t_k = k_const(std_lib, layouter, t_u64, 0)?;
                let term = k_sub_mod_var(std_lib, layouter, &y_eval, &t_k)?;
                range_prod = k_mul_mod_var(std_lib, layouter, &range_prod, &term, K_DELTA_U64)?;
            }

            let weighted = k_mul_mod_var(std_lib, layouter, &g, &range_prod, K_DELTA_U64)?;
            nc_sum = k_add_mod_var(std_lib, layouter, &nc_sum, &weighted)?;

            // Update γ^{i+2}.
            if out_idx + 1 < self.k_total {
                g = k_mul_mod_var(std_lib, layouter, &g, &gamma, K_DELTA_U64)?;
            }
        }

        let rhs = k_mul_mod_var(std_lib, layouter, &eq_apsp_beta, &nc_sum, K_DELTA_U64)?;
        assert_k_eq(std_lib, layouter, &rhs, &final_sum_nc)?;
        Ok(())
    }
}

/// Chunked NC-terminal helper: compute a partial sum over a subset of outputs.
///
/// The chunk sum is:
/// `chunk_sum = Σ_{j=0..count-1} γ^{start_exp+j} * range_product(<y_zcol_j, χ_{α'}>)`
///
/// where `start_exp` is the exponent of γ for the first output in the chunk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsNcChunkRelation {
    pub ell_d: usize,
    pub b: u32,
    pub start_exp: usize,
    pub count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsNcChunkInstance {
    /// Step-bundle binding digest limbs (little-endian u128×2).
    pub bundle_digest: [u128; 2],
    pub chunk_sum: KRepr,
    /// Split (s', α') from the NC sumcheck challenges (public).
    pub alpha_prime: Vec<KRepr>,
    /// Fiat–Shamir challenge γ (public).
    pub gamma: KRepr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsNcChunkWitness {
    pub y_zcol: Vec<Vec<KRepr>>,
}

impl Relation for PiCcsNcChunkRelation {
    type Instance = PiCcsNcChunkInstance;
    type Witness = PiCcsNcChunkWitness;

    fn used_chips(&self) -> ZkStdLibArch {
        let mut arch = ZkStdLibArch::default();
        arch.nr_pow2range_cols = 4;
        arch
    }

    fn format_instance(instance: &Self::Instance) -> Result<Vec<OuterScalar>, Error> {
        let mut out = Vec::with_capacity(6 + 2 * instance.alpha_prime.len() + 2);
        out.push(OuterScalar::from_u128(instance.bundle_digest[0]));
        out.push(OuterScalar::from_u128(instance.bundle_digest[1]));
        out.push(OuterScalar::from(instance.chunk_sum.c0));
        out.push(OuterScalar::from(instance.chunk_sum.c1));
        for a in &instance.alpha_prime {
            out.push(OuterScalar::from(a.c0));
            out.push(OuterScalar::from(a.c1));
        }
        out.push(OuterScalar::from(instance.gamma.c0));
        out.push(OuterScalar::from(instance.gamma.c1));
        Ok(out)
    }

    fn write_relation<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write_relation_len_prefixed(writer, self)
    }

    fn read_relation<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        read_relation_len_prefixed(reader)
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<OuterScalar>,
        instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        if self.ell_d == 0 {
            return Err(Error::Synthesis("PiCcsNcChunkRelation requires ell_d > 0".into()));
        }
        if self.b < 2 {
            return Err(Error::Synthesis("PiCcsNcChunkRelation requires b >= 2".into()));
        }
        if self.start_exp == 0 {
            return Err(Error::Synthesis("PiCcsNcChunkRelation requires start_exp > 0".into()));
        }
        if self.count == 0 {
            return Err(Error::Synthesis("PiCcsNcChunkRelation requires count > 0".into()));
        }

        // Public: bundle digest (anti mix-and-match across steps/runs).
        for limb_idx in 0..2 {
            let limb = instance.as_ref().map(|i| i.bundle_digest[limb_idx]);
            let assigned: AssignedNative<OuterScalar> = std_lib.assign(layouter, limb.map(OuterScalar::from_u128))?;
            std_lib.constrain_as_public_input(layouter, &assigned)?;
        }

        let chunk_sum = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.chunk_sum))?;

        // Public: α' and γ (Fiat–Shamir challenges).
        let mut alpha_prime: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = instance.as_ref().map(|ins| ins.alpha_prime[i]);
            alpha_prime.push(alloc_k_public(std_lib, layouter, v)?);
        }
        let gamma = alloc_k_public(std_lib, layouter, instance.as_ref().map(|ins| ins.gamma))?;

        let d_pad = 1usize
            .checked_shl(self.ell_d as u32)
            .ok_or_else(|| Error::Synthesis("PiCcsNcChunkRelation: 1<<ell_d overflow".into()))?;

        let one = k_one(std_lib, layouter)?;
        let mut acc = k_zero(std_lib, layouter)?;

        // g = γ^{start_exp}
        let mut g = one.clone();
        for _ in 0..self.start_exp {
            g = k_mul_mod_var(std_lib, layouter, &g, &gamma, K_DELTA_U64)?;
        }

        for out_idx in 0..self.count {
            // Allocate y_zcol for this output.
            let mut y_zcol: Vec<KVar> = Vec::with_capacity(d_pad);
            for i in 0..d_pad {
                let v = witness.as_ref().map(|w| w.y_zcol[out_idx][i]);
                y_zcol.push(alloc_k_private_u64(std_lib, layouter, v)?);
            }

            // Evaluate y_eval = <y_zcol, χ_{α'}> via multilinear folding.
            let mut eval_vec = y_zcol;
            for a in &alpha_prime {
                let next_len = eval_vec.len() / 2;
                let mut next: Vec<KVar> = Vec::with_capacity(next_len);
                for j in 0..next_len {
                    let v0 = &eval_vec[2 * j];
                    let v1 = &eval_vec[2 * j + 1];
                    next.push(k_mle_fold_step(std_lib, layouter, v0, v1, a, K_DELTA_U64)?);
                }
                eval_vec = next;
            }
            if eval_vec.len() != 1 {
                return Err(Error::Synthesis(format!(
                    "PiCcsNcChunkRelation: eval_vec len {} != 1 (ell_d={})",
                    eval_vec.len(),
                    self.ell_d
                )));
            }
            let y_eval = eval_vec.first().expect("len checked").clone();

            // range_product(y_eval) = ∏_{t=-(b-1)}^{b-1} (y_eval - t)
            let lo = -((self.b as i64) - 1);
            let hi = (self.b as i64) - 1;
            let mut range_prod = one.clone();
            for t in lo..=hi {
                let t_u64 = if t >= 0 {
                    t as u64
                } else {
                    GOLDILOCKS_P_U64
                        .checked_sub((-t) as u64)
                        .ok_or_else(|| Error::Synthesis("PiCcsNcChunkRelation: t underflow".into()))?
                };
                let t_k = k_const(std_lib, layouter, t_u64, 0)?;
                let term = k_sub_mod_var(std_lib, layouter, &y_eval, &t_k)?;
                range_prod = k_mul_mod_var(std_lib, layouter, &range_prod, &term, K_DELTA_U64)?;
            }

            let weighted = k_mul_mod_var(std_lib, layouter, &g, &range_prod, K_DELTA_U64)?;
            acc = k_add_mod_var(std_lib, layouter, &acc, &weighted)?;

            if out_idx + 1 < self.count {
                g = k_mul_mod_var(std_lib, layouter, &g, &gamma, K_DELTA_U64)?;
            }
        }

        assert_k_eq(std_lib, layouter, &acc, &chunk_sum)?;
        Ok(())
    }
}

/// Aggregates chunked NC-terminal sums, checking against the NC sumcheck final sum.
///
/// Checks:
/// `final_sum_nc == eq((α',s'), (β_a,β_m)) * Σ chunk_sums[i]`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsNcTerminalAggregateRelation {
    pub ell_d: usize,
    pub ell_m: usize,
    pub n_chunks: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsNcTerminalAggregateInstance {
    pub chunk_sums: Vec<KRepr>,
    pub final_sum_nc: KRepr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsNcTerminalAggregateWitness {
    pub s_col_prime: Vec<KRepr>,
    pub alpha_prime: Vec<KRepr>,
    pub beta_a: Vec<KRepr>,
    pub beta_m: Vec<KRepr>,
}

impl Relation for PiCcsNcTerminalAggregateRelation {
    type Instance = PiCcsNcTerminalAggregateInstance;
    type Witness = PiCcsNcTerminalAggregateWitness;

    fn used_chips(&self) -> ZkStdLibArch {
        let mut arch = ZkStdLibArch::default();
        arch.nr_pow2range_cols = 4;
        arch
    }

    fn format_instance(instance: &Self::Instance) -> Result<Vec<OuterScalar>, Error> {
        let mut out = Vec::with_capacity(2 * (instance.chunk_sums.len() + 1));
        for cs in &instance.chunk_sums {
            out.push(OuterScalar::from(cs.c0));
            out.push(OuterScalar::from(cs.c1));
        }
        out.push(OuterScalar::from(instance.final_sum_nc.c0));
        out.push(OuterScalar::from(instance.final_sum_nc.c1));
        Ok(out)
    }

    fn write_relation<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write_relation_len_prefixed(writer, self)
    }

    fn read_relation<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        read_relation_len_prefixed(reader)
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<OuterScalar>,
        instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        if self.ell_d == 0 {
            return Err(Error::Synthesis(
                "PiCcsNcTerminalAggregateRelation requires ell_d > 0".into(),
            ));
        }
        if self.ell_m == 0 {
            return Err(Error::Synthesis(
                "PiCcsNcTerminalAggregateRelation requires ell_m > 0".into(),
            ));
        }
        if self.n_chunks == 0 {
            return Err(Error::Synthesis(
                "PiCcsNcTerminalAggregateRelation requires n_chunks > 0".into(),
            ));
        }

        // Public: chunk sums + final_sum_nc.
        let mut chunk_sums: Vec<KVar> = Vec::with_capacity(self.n_chunks);
        for i in 0..self.n_chunks {
            let v = instance.as_ref().map(|ins| ins.chunk_sums[i]);
            chunk_sums.push(alloc_k_public(std_lib, layouter, v)?);
        }
        let final_sum_nc = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.final_sum_nc))?;

        // Witness: points/challenges to compute eq((α',s'),(β_a,β_m)).
        let mut s_col_prime: Vec<KVar> = Vec::with_capacity(self.ell_m);
        for i in 0..self.ell_m {
            let v = witness.as_ref().map(|w| w.s_col_prime[i]);
            s_col_prime.push(alloc_k_private(std_lib, layouter, v)?);
        }
        let mut alpha_prime: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = witness.as_ref().map(|w| w.alpha_prime[i]);
            alpha_prime.push(alloc_k_private(std_lib, layouter, v)?);
        }
        let mut beta_a: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = witness.as_ref().map(|w| w.beta_a[i]);
            beta_a.push(alloc_k_private(std_lib, layouter, v)?);
        }
        let mut beta_m: Vec<KVar> = Vec::with_capacity(self.ell_m);
        for i in 0..self.ell_m {
            let v = witness.as_ref().map(|w| w.beta_m[i]);
            beta_m.push(alloc_k_private(std_lib, layouter, v)?);
        }

        let eq_a = k_eq_points(std_lib, layouter, &alpha_prime, &beta_a)?;
        let eq_s = k_eq_points(std_lib, layouter, &s_col_prime, &beta_m)?;
        let eq_apsp_beta = k_mul_mod_var(std_lib, layouter, &eq_a, &eq_s, K_DELTA_U64)?;

        let mut sum = k_zero(std_lib, layouter)?;
        for cs in &chunk_sums {
            sum = k_add_mod_var(std_lib, layouter, &sum, cs)?;
        }

        let rhs = k_mul_mod_var(std_lib, layouter, &eq_apsp_beta, &sum, K_DELTA_U64)?;
        assert_k_eq(std_lib, layouter, &rhs, &final_sum_nc)?;
        Ok(())
    }
}

/// Combines:
/// - NC sumcheck verification, and
/// - (optionally) one NC terminal chunk proof that also performs the aggregate check.
///
/// This lets a bundle drop the standalone NC sumcheck proof.
///
/// Set `count=0` to skip the chunk-binding section (sumcheck + aggregate only).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsNcChunkAggSumcheckRelation {
    // Sumcheck parameters.
    pub n_rounds: usize,
    pub poly_len: usize,
    // NC terminal identity parameters.
    pub ell_d: usize,
    pub ell_m: usize,
    pub b: u32,
    // Chunk parameters.
    pub start_exp: usize,
    pub count: usize,
    pub n_chunks: usize,
    pub chunk_index: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsNcChunkAggSumcheckInstance {
    /// Step-bundle binding digest limbs (little-endian u128×2).
    pub bundle_digest: [u128; 2],
    /// NC sumcheck challenges (s' || α') (public).
    pub sumcheck_challenges: Vec<KRepr>,
    /// Fiat–Shamir challenge γ (public).
    pub gamma: KRepr,
    /// SplitNcV1 transcript-derived values (public).
    pub beta_a: Vec<KRepr>,
    pub beta_m: Vec<KRepr>,
    pub chunk_sums: Vec<KRepr>,
    pub initial_sum: KRepr,
    pub final_sum_nc: KRepr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsNcChunkAggSumcheckWitness {
    /// NC sumcheck round polynomials (private): `rounds[round_idx][coeff_idx]`.
    pub rounds: Vec<Vec<KRepr>>,
    // Digit columns for the designated chunk (length = count, each padded to 2^ell_d).
    //
    // If `count=0`, this must be empty and the chunk-binding section is skipped.
    pub y_zcol: Vec<Vec<KRepr>>,
}

impl Relation for PiCcsNcChunkAggSumcheckRelation {
    type Instance = PiCcsNcChunkAggSumcheckInstance;
    type Witness = PiCcsNcChunkAggSumcheckWitness;

    fn used_chips(&self) -> ZkStdLibArch {
        let mut arch = ZkStdLibArch::default();
        arch.nr_pow2range_cols = 4;
        arch
    }

    fn format_instance(instance: &Self::Instance) -> Result<Vec<OuterScalar>, Error> {
        let mut out = Vec::with_capacity(
            2 + 2 * instance.sumcheck_challenges.len()
                + 2
                + 2 * (instance.beta_a.len() + instance.beta_m.len())
                + 2 * (instance.chunk_sums.len() + 2),
        );
        out.push(OuterScalar::from_u128(instance.bundle_digest[0]));
        out.push(OuterScalar::from_u128(instance.bundle_digest[1]));
        for ch in &instance.sumcheck_challenges {
            out.push(OuterScalar::from(ch.c0));
            out.push(OuterScalar::from(ch.c1));
        }
        out.push(OuterScalar::from(instance.gamma.c0));
        out.push(OuterScalar::from(instance.gamma.c1));
        for b in &instance.beta_a {
            out.push(OuterScalar::from(b.c0));
            out.push(OuterScalar::from(b.c1));
        }
        for b in &instance.beta_m {
            out.push(OuterScalar::from(b.c0));
            out.push(OuterScalar::from(b.c1));
        }
        for cs in &instance.chunk_sums {
            out.push(OuterScalar::from(cs.c0));
            out.push(OuterScalar::from(cs.c1));
        }
        out.push(OuterScalar::from(instance.initial_sum.c0));
        out.push(OuterScalar::from(instance.initial_sum.c1));
        out.push(OuterScalar::from(instance.final_sum_nc.c0));
        out.push(OuterScalar::from(instance.final_sum_nc.c1));
        Ok(out)
    }

    fn write_relation<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write_relation_len_prefixed(writer, self)
    }

    fn read_relation<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        read_relation_len_prefixed(reader)
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<OuterScalar>,
        instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        if self.n_rounds == 0 {
            return Err(Error::Synthesis(
                "PiCcsNcChunkAggSumcheckRelation requires n_rounds > 0".into(),
            ));
        }
        if self.poly_len == 0 {
            return Err(Error::Synthesis(
                "PiCcsNcChunkAggSumcheckRelation requires poly_len > 0".into(),
            ));
        }
        if self.ell_d == 0 {
            return Err(Error::Synthesis(
                "PiCcsNcChunkAggSumcheckRelation requires ell_d > 0".into(),
            ));
        }
        if self.ell_m == 0 {
            return Err(Error::Synthesis(
                "PiCcsNcChunkAggSumcheckRelation requires ell_m > 0".into(),
            ));
        }
        if self.b < 2 {
            return Err(Error::Synthesis(
                "PiCcsNcChunkAggSumcheckRelation requires b >= 2".into(),
            ));
        }
        if self.start_exp == 0 {
            return Err(Error::Synthesis(
                "PiCcsNcChunkAggSumcheckRelation requires start_exp > 0".into(),
            ));
        }
        // `count=0` is allowed: skip chunk-binding and prove sumcheck + aggregate only.
        if self.n_chunks == 0 {
            return Err(Error::Synthesis(
                "PiCcsNcChunkAggSumcheckRelation requires n_chunks > 0".into(),
            ));
        }
        if self.chunk_index >= self.n_chunks {
            return Err(Error::Synthesis(
                "PiCcsNcChunkAggSumcheckRelation requires chunk_index < n_chunks".into(),
            ));
        }
        if self.n_rounds != self.ell_m + self.ell_d {
            return Err(Error::Synthesis(format!(
                "PiCcsNcChunkAggSumcheckRelation: n_rounds {} != ell_m+ell_d {}",
                self.n_rounds,
                self.ell_m + self.ell_d
            )));
        }

        // Public: bundle digest (anti mix-and-match across steps/runs).
        for limb_idx in 0..2 {
            let limb = instance.as_ref().map(|i| i.bundle_digest[limb_idx]);
            let assigned: AssignedNative<OuterScalar> = std_lib.assign(layouter, limb.map(OuterScalar::from_u128))?;
            std_lib.constrain_as_public_input(layouter, &assigned)?;
        }

        // Public inputs (in the exact order of `format_instance`):
        // - bundle digest
        // - sumcheck challenges
        // - γ
        // - β_a, β_m
        // - chunk sums
        // - (initial_sum, final_sum_nc)
        let mut challenges: Vec<KVar> = Vec::with_capacity(self.n_rounds);
        for r in 0..self.n_rounds {
            let ch = instance.as_ref().map(|i| i.sumcheck_challenges[r]);
            challenges.push(alloc_k_public(std_lib, layouter, ch)?);
        }

        // Private witness: sumcheck rounds.
        let mut rounds: Vec<Vec<KVar>> = Vec::with_capacity(self.n_rounds);
        for r in 0..self.n_rounds {
            let mut coeffs_r = Vec::with_capacity(self.poly_len);
            for j in 0..self.poly_len {
                let coeff = witness.as_ref().map(|w| w.rounds[r][j]);
                coeffs_r.push(alloc_k_private(std_lib, layouter, coeff)?);
            }
            rounds.push(coeffs_r);
        }

        let gamma = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.gamma))?;

        let mut beta_a: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = instance.as_ref().map(|ins| ins.beta_a[i]);
            beta_a.push(alloc_k_public(std_lib, layouter, v)?);
        }
        let mut beta_m: Vec<KVar> = Vec::with_capacity(self.ell_m);
        for i in 0..self.ell_m {
            let v = instance.as_ref().map(|ins| ins.beta_m[i]);
            beta_m.push(alloc_k_public(std_lib, layouter, v)?);
        }

        let mut chunk_sums: Vec<KVar> = Vec::with_capacity(self.n_chunks);
        for i in 0..self.n_chunks {
            let v = instance.as_ref().map(|ins| ins.chunk_sums[i]);
            chunk_sums.push(alloc_k_public(std_lib, layouter, v)?);
        }
        let initial_sum = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.initial_sum))?;
        let final_sum_nc = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.final_sum_nc))?;

        // --- Sumcheck ---
        let mut running_sum = initial_sum.clone();
        for r in 0..self.n_rounds {
            sumcheck_round_check(std_lib, layouter, &rounds[r], &running_sum)?;
            running_sum = sumcheck_eval_horner(std_lib, layouter, &rounds[r], &challenges[r], K_DELTA_U64)?;
        }
        assert_k_eq(std_lib, layouter, &running_sum, &final_sum_nc)?;

        // Split challenges into (s_col_prime, alpha_prime).
        let (s_col_prime, alpha_prime) = challenges.split_at(self.ell_m);

        // --- NC terminal: (optional) recompute designated chunk sum ---

        if self.count > 0 {
            let d_pad = 1usize
                .checked_shl(self.ell_d as u32)
                .ok_or_else(|| Error::Synthesis("PiCcsNcChunkAggSumcheckRelation: 1<<ell_d overflow".into()))?;

            // g = γ^{start_exp}
            let one = k_one(std_lib, layouter)?;
            let mut g = one.clone();
            for _ in 0..self.start_exp {
                g = k_mul_mod_var(std_lib, layouter, &g, &gamma, K_DELTA_U64)?;
            }

            let mut weighted_terms: Vec<KVar> = Vec::with_capacity(self.count);
            for out_idx in 0..self.count {
                let mut y_zcol: Vec<KVar> = Vec::with_capacity(d_pad);
                for i in 0..d_pad {
                    let v = witness.as_ref().map(|w| w.y_zcol[out_idx][i]);
                    y_zcol.push(alloc_k_private_u64(std_lib, layouter, v)?);
                }

                // y_eval = <y_zcol, χ_{α'}>.
                let mut eval_vec = y_zcol;
                for a in alpha_prime {
                    let next_len = eval_vec.len() / 2;
                    let mut next: Vec<KVar> = Vec::with_capacity(next_len);
                    for j in 0..next_len {
                        let v0 = &eval_vec[2 * j];
                        let v1 = &eval_vec[2 * j + 1];
                        next.push(k_mle_fold_step(std_lib, layouter, v0, v1, a, K_DELTA_U64)?);
                    }
                    eval_vec = next;
                }
                if eval_vec.len() != 1 {
                    return Err(Error::Synthesis(format!(
                        "PiCcsNcChunkAggSumcheckRelation: eval_vec len {} != 1 (ell_d={})",
                        eval_vec.len(),
                        self.ell_d
                    )));
                }
                let y_eval = eval_vec.first().expect("len checked").clone();

                // range_product(y_eval) = ∏_{t=-(b-1)}^{b-1} (y_eval - t)
                let lo = -((self.b as i64) - 1);
                let hi = (self.b as i64) - 1;
                let mut range_prod = one.clone();
                for t in lo..=hi {
                    let t_u64 = if t >= 0 {
                        t as u64
                    } else {
                        GOLDILOCKS_P_U64
                            .checked_sub((-t) as u64)
                            .ok_or_else(|| Error::Synthesis("PiCcsNcChunkAggSumcheckRelation: t underflow".into()))?
                    };
                    let t_k = k_const(std_lib, layouter, t_u64, 0)?;
                    let term = k_sub_mod_var(std_lib, layouter, &y_eval, &t_k)?;
                    range_prod = k_mul_mod_var(std_lib, layouter, &range_prod, &term, K_DELTA_U64)?;
                }

                let weighted = k_mul_mod_var(std_lib, layouter, &g, &range_prod, K_DELTA_U64)?;
                weighted_terms.push(weighted);

                if out_idx + 1 < self.count {
                    g = k_mul_mod_var(std_lib, layouter, &g, &gamma, K_DELTA_U64)?;
                }
            }

            let acc = k_sum_mod_var(std_lib, layouter, &weighted_terms)?;
            assert_k_eq(std_lib, layouter, &acc, &chunk_sums[self.chunk_index])?;
        }

        // Aggregate: final_sum_nc == eq((α',s'),(β_a,β_m)) * Σ chunk_sums.
        let eq_a = k_eq_points(std_lib, layouter, alpha_prime, &beta_a)?;
        let eq_s = k_eq_points(std_lib, layouter, s_col_prime, &beta_m)?;
        let eq_apsp_beta = k_mul_mod_var(std_lib, layouter, &eq_a, &eq_s, K_DELTA_U64)?;

        let sum = k_sum_mod_var(std_lib, layouter, &chunk_sums)?;
        let rhs = k_mul_mod_var(std_lib, layouter, &eq_apsp_beta, &sum, K_DELTA_U64)?;
        assert_k_eq(std_lib, layouter, &rhs, &final_sum_nc)?;
        Ok(())
    }
}

/// Combines one NC-terminal chunk with the aggregate check, so we can drop the standalone
/// aggregate proof and keep the bundle smaller.
///
/// Public instance includes all `chunk_sums` (for binding) and `final_sum_nc`.
/// This circuit:
/// 1) recomputes the designated chunk sum and checks it matches `chunk_sums[chunk_index]`
/// 2) checks `final_sum_nc == eq((α',s'),(β_a,β_m)) * Σ chunk_sums[i]`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsNcChunkAggregateRelation {
    pub ell_d: usize,
    pub ell_m: usize,
    pub b: u32,
    pub start_exp: usize,
    pub count: usize,
    pub n_chunks: usize,
    pub chunk_index: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiCcsNcChunkAggregateWitness {
    pub alpha_prime: Vec<KRepr>,
    pub gamma: KRepr,
    pub y_zcol: Vec<Vec<KRepr>>,
    pub s_col_prime: Vec<KRepr>,
    pub beta_a: Vec<KRepr>,
    pub beta_m: Vec<KRepr>,
}

impl Relation for PiCcsNcChunkAggregateRelation {
    type Instance = PiCcsNcTerminalAggregateInstance;
    type Witness = PiCcsNcChunkAggregateWitness;

    fn used_chips(&self) -> ZkStdLibArch {
        let mut arch = ZkStdLibArch::default();
        arch.nr_pow2range_cols = 4;
        arch
    }

    fn format_instance(instance: &Self::Instance) -> Result<Vec<OuterScalar>, Error> {
        let mut out = Vec::with_capacity(2 * (instance.chunk_sums.len() + 1));
        for cs in &instance.chunk_sums {
            out.push(OuterScalar::from(cs.c0));
            out.push(OuterScalar::from(cs.c1));
        }
        out.push(OuterScalar::from(instance.final_sum_nc.c0));
        out.push(OuterScalar::from(instance.final_sum_nc.c1));
        Ok(out)
    }

    fn write_relation<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write_relation_len_prefixed(writer, self)
    }

    fn read_relation<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        read_relation_len_prefixed(reader)
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<OuterScalar>,
        instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        if self.ell_d == 0 {
            return Err(Error::Synthesis(
                "PiCcsNcChunkAggregateRelation requires ell_d > 0".into(),
            ));
        }
        if self.ell_m == 0 {
            return Err(Error::Synthesis(
                "PiCcsNcChunkAggregateRelation requires ell_m > 0".into(),
            ));
        }
        if self.b < 2 {
            return Err(Error::Synthesis("PiCcsNcChunkAggregateRelation requires b >= 2".into()));
        }
        if self.start_exp == 0 {
            return Err(Error::Synthesis(
                "PiCcsNcChunkAggregateRelation requires start_exp > 0".into(),
            ));
        }
        if self.count == 0 {
            return Err(Error::Synthesis(
                "PiCcsNcChunkAggregateRelation requires count > 0".into(),
            ));
        }
        if self.n_chunks == 0 {
            return Err(Error::Synthesis(
                "PiCcsNcChunkAggregateRelation requires n_chunks > 0".into(),
            ));
        }
        if self.chunk_index >= self.n_chunks {
            return Err(Error::Synthesis(
                "PiCcsNcChunkAggregateRelation requires chunk_index < n_chunks".into(),
            ));
        }

        // Public: chunk sums + final_sum_nc.
        let mut chunk_sums: Vec<KVar> = Vec::with_capacity(self.n_chunks);
        for i in 0..self.n_chunks {
            let v = instance.as_ref().map(|ins| ins.chunk_sums[i]);
            chunk_sums.push(alloc_k_public(std_lib, layouter, v)?);
        }
        let final_sum_nc = alloc_k_public(std_lib, layouter, instance.as_ref().map(|i| i.final_sum_nc))?;

        // Witness vectors.
        let mut alpha_prime: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = witness.as_ref().map(|w| w.alpha_prime[i]);
            alpha_prime.push(alloc_k_private(std_lib, layouter, v)?);
        }
        let gamma = alloc_k_private(std_lib, layouter, witness.as_ref().map(|w| w.gamma))?;

        let d_pad = 1usize
            .checked_shl(self.ell_d as u32)
            .ok_or_else(|| Error::Synthesis("PiCcsNcChunkAggregateRelation: 1<<ell_d overflow".into()))?;

        let mut s_col_prime: Vec<KVar> = Vec::with_capacity(self.ell_m);
        for i in 0..self.ell_m {
            let v = witness.as_ref().map(|w| w.s_col_prime[i]);
            s_col_prime.push(alloc_k_private(std_lib, layouter, v)?);
        }
        let mut beta_a: Vec<KVar> = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let v = witness.as_ref().map(|w| w.beta_a[i]);
            beta_a.push(alloc_k_private(std_lib, layouter, v)?);
        }
        let mut beta_m: Vec<KVar> = Vec::with_capacity(self.ell_m);
        for i in 0..self.ell_m {
            let v = witness.as_ref().map(|w| w.beta_m[i]);
            beta_m.push(alloc_k_private(std_lib, layouter, v)?);
        }

        // Recompute the designated chunk sum and bind it to `chunk_sums[chunk_index]`.
        let one = k_one(std_lib, layouter)?;
        let mut acc = k_zero(std_lib, layouter)?;

        // g = γ^{start_exp}
        let mut g = one.clone();
        for _ in 0..self.start_exp {
            g = k_mul_mod_var(std_lib, layouter, &g, &gamma, K_DELTA_U64)?;
        }

        for out_idx in 0..self.count {
            // Allocate y_zcol for this output within the chunk.
            let mut y_zcol: Vec<KVar> = Vec::with_capacity(d_pad);
            for i in 0..d_pad {
                let v = witness.as_ref().map(|w| w.y_zcol[out_idx][i]);
                y_zcol.push(alloc_k_private_u64(std_lib, layouter, v)?);
            }

            // Evaluate y_eval = <y_zcol, χ_{α'}> via multilinear folding.
            let mut eval_vec = y_zcol;
            for a in &alpha_prime {
                let next_len = eval_vec.len() / 2;
                let mut next: Vec<KVar> = Vec::with_capacity(next_len);
                for j in 0..next_len {
                    let v0 = &eval_vec[2 * j];
                    let v1 = &eval_vec[2 * j + 1];
                    next.push(k_mle_fold_step(std_lib, layouter, v0, v1, a, K_DELTA_U64)?);
                }
                eval_vec = next;
            }
            if eval_vec.len() != 1 {
                return Err(Error::Synthesis(format!(
                    "PiCcsNcChunkAggregateRelation: eval_vec len {} != 1 (ell_d={})",
                    eval_vec.len(),
                    self.ell_d
                )));
            }
            let y_eval = eval_vec.first().expect("len checked").clone();

            // range_product(y_eval) = ∏_{t=-(b-1)}^{b-1} (y_eval - t)
            let lo = -((self.b as i64) - 1);
            let hi = (self.b as i64) - 1;
            let mut range_prod = one.clone();
            for t in lo..=hi {
                let t_u64 = if t >= 0 {
                    t as u64
                } else {
                    GOLDILOCKS_P_U64
                        .checked_sub((-t) as u64)
                        .ok_or_else(|| Error::Synthesis("PiCcsNcChunkAggregateRelation: t underflow".into()))?
                };
                let t_k = k_const(std_lib, layouter, t_u64, 0)?;
                let term = k_sub_mod_var(std_lib, layouter, &y_eval, &t_k)?;
                range_prod = k_mul_mod_var(std_lib, layouter, &range_prod, &term, K_DELTA_U64)?;
            }

            let weighted = k_mul_mod_var(std_lib, layouter, &g, &range_prod, K_DELTA_U64)?;
            acc = k_add_mod_var(std_lib, layouter, &acc, &weighted)?;

            if out_idx + 1 < self.count {
                g = k_mul_mod_var(std_lib, layouter, &g, &gamma, K_DELTA_U64)?;
            }
        }

        assert_k_eq(std_lib, layouter, &acc, &chunk_sums[self.chunk_index])?;

        // Aggregate check: final_sum_nc == eq((α',s'),(β_a,β_m)) * Σ chunk_sums.
        let eq_a = k_eq_points(std_lib, layouter, &alpha_prime, &beta_a)?;
        let eq_s = k_eq_points(std_lib, layouter, &s_col_prime, &beta_m)?;
        let eq_apsp_beta = k_mul_mod_var(std_lib, layouter, &eq_a, &eq_s, K_DELTA_U64)?;

        let mut sum = k_zero(std_lib, layouter)?;
        for cs in &chunk_sums {
            sum = k_add_mod_var(std_lib, layouter, &sum, cs)?;
        }
        let rhs = k_mul_mod_var(std_lib, layouter, &eq_apsp_beta, &sum, K_DELTA_U64)?;
        assert_k_eq(std_lib, layouter, &rhs, &final_sum_nc)?;
        Ok(())
    }
}
