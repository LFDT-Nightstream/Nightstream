//! This module implements the spartan SNARK protocol.
//! It provides the prover and verifier keys, as well as the SNARK itself.
use crate::{
  CommitmentKey,
  bellpepper::{
    r1cs::{PrecommittedState, SpartanShape, SpartanWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  },
  digest::{DigestComputer, SimpleDigestible},
  errors::SpartanError,
  math::Math,
  polys::{
    eq::EqPolynomial,
    multilinear::{MultilinearPolynomial, SparsePolynomial},
  },
  r1cs::{SparseMatrix, SplitR1CSInstance, SplitR1CSShape, SplitR1CSShapeDebugStats},
  start_span,
  sumcheck::SumcheckProof,
  traits::{
    Engine,
    circuit::SpartanCircuit,
    pcs::PCSEngineTrait,
    snark::{DigestHelperTrait, R1CSSNARKTrait, SpartanDigest},
    transcript::TranscriptEngineTrait,
  },
};
use ff::Field;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{info, info_span};

/// A type that represents the prover's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanProverKey<E: Engine> {
  ck: CommitmentKey<E>,
  S: SplitR1CSShape<E>,
  vk_digest: SpartanDigest, // digest of the verifier's key
}

impl<E: Engine> SpartanProverKey<E> {
  /// Returns sizes associated with the SplitR1CSShape.
  /// It returns an array of 10 elements containing:
  /// [num_cons_unpadded, num_shared_unpadded, num_precommitted_unpadded, num_rest_unpadded,
  ///  num_cons, num_shared, num_precommitted, num_rest,
  ///  num_public, num_challenges]
  pub fn sizes(&self) -> [usize; 10] {
    self.S.sizes()
  }

  /// Returns compact sparse-matrix stats for the carried split R1CS shape.
  pub fn shape_debug_stats(&self) -> SplitR1CSShapeDebugStats {
    self.S.debug_stats()
  }
}

/// A type that represents the verifier's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanVerifierKey<E: Engine> {
  vk_ee: <E::PCS as PCSEngineTrait<E>>::VerifierKey,
  S: SplitR1CSShape<E>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<SpartanDigest>,
}

impl<E: Engine> SimpleDigestible for SpartanVerifierKey<E> {}

impl<E: Engine> DigestHelperTrait<E> for SpartanVerifierKey<E> {
  /// Returns the digest of the verifier's key.
  fn digest(&self) -> Result<SpartanDigest, SpartanError> {
    self
      .digest
      .get_or_try_init(|| {
        let dc = DigestComputer::<_>::new(self);
        dc.digest()
      })
      .cloned()
      .map_err(|_| SpartanError::DigestError {
        reason: "Unable to compute digest for SpartanVerifierKey".to_string(),
      })
  }
}

/// Binds "row" variables of (A, B, C) matrices viewed as 2d multilinear polynomials
pub(crate) fn compute_eval_table_sparse<E: Engine>(
  S: &SplitR1CSShape<E>,
  rx: &[E::Scalar],
) -> (Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>) {
  assert_eq!(rx.len(), S.num_cons);

  let inner = |M: &SparseMatrix<E::Scalar>, M_evals: &mut Vec<E::Scalar>| {
    for (row_idx, ptrs) in M.indptr.windows(2).enumerate() {
      for (val, col_idx) in M.get_row_unchecked(ptrs.try_into().unwrap()) {
        M_evals[*col_idx] += rx[row_idx] * val;
      }
    }
  };

  let num_vars = S.num_shared + S.num_precommitted + S.num_rest;
  let (A_evals, (B_evals, C_evals)) = crate::parallel::join(
    || {
      let mut A_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * num_vars];
      inner(&S.A, &mut A_evals);
      A_evals
    },
    || {
      crate::parallel::join(
        || {
          let mut B_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * num_vars];
          inner(&S.B, &mut B_evals);
          B_evals
        },
        || {
          let mut C_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * num_vars];
          inner(&S.C, &mut C_evals);
          C_evals
        },
      )
    },
  );

  (A_evals, B_evals, C_evals)
}

/// A type that holds the pre-processed state for proving
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PrepSNARK<E: Engine> {
  ps: PrecommittedState<E>,
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSSNARK<E: Engine> {
  U: SplitR1CSInstance<E>,
  sc_proof_outer: SumcheckProof<E>,
  claims_outer: (E::Scalar, E::Scalar, E::Scalar),
  sc_proof_inner: SumcheckProof<E>,
  eval_W: E::Scalar,
  eval_arg: <E::PCS as PCSEngineTrait<E>>::EvaluationArgument,
  claim_inner_sum: E::Scalar, // ← NEW: correct inner sum-check claim
}

/// Serialized byte sizes for the major components of an `R1CSSNARK`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct R1CSSNarkSerializedSizeBreakdown {
  /// Serialized size of the full SNARK object.
  pub total: usize,
  /// Serialized size of the carried split R1CS instance.
  pub instance: usize,
  /// Serialized size of the outer sumcheck proof.
  pub outer_sumcheck: usize,
  /// Serialized size of the three outer claims tuple.
  pub outer_claims: usize,
  /// Serialized size of the inner sumcheck proof.
  pub inner_sumcheck: usize,
  /// Serialized size of the carried witness evaluation.
  pub eval_w: usize,
  /// Serialized size of the PCS evaluation argument.
  pub eval_arg: usize,
  /// Serialized size of the final inner sum claim.
  pub inner_sum_claim: usize,
}

impl R1CSSNarkSerializedSizeBreakdown {
  /// Returns the measured components as label/byte pairs for reporting.
  pub fn measured_components(&self) -> [(&'static str, usize); 7] {
    [
      ("instance", self.instance),
      ("outer_sumcheck", self.outer_sumcheck),
      ("outer_claims", self.outer_claims),
      ("inner_sumcheck", self.inner_sumcheck),
      ("eval_w", self.eval_w),
      ("eval_arg", self.eval_arg),
      ("inner_sum_claim", self.inner_sum_claim),
    ]
  }
}

/// Timing breakdown for the main proving phases of a direct R1CS Spartan proof.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpartanProvePerf {
  /// Time spent constructing the multilinear tau polynomial for the outer sumcheck.
  pub prepare_poly_tau_ms: f64,
  /// Time spent multiplying the split R1CS shape by the witness vector.
  pub matrix_vector_multiply_ms: f64,
  /// Time spent materializing the multilinear `Az`, `Bz`, and `Cz` polynomials.
  pub prepare_multilinear_polys_ms: f64,
  /// Time spent in the outer sumcheck.
  pub outer_sumcheck_ms: f64,
  /// Time spent preparing the inner-claim challenge and transcript fork anchor.
  pub prepare_inner_claims_ms: f64,
  /// Time spent evaluating the equality polynomial at the outer sumcheck point.
  pub compute_eval_rx_ms: f64,
  /// Time spent evaluating sparse matrix tables at the outer sumcheck point.
  pub compute_eval_table_sparse_ms: f64,
  /// Time spent preparing the combined `A + rB + r^2C` polynomial.
  pub prepare_poly_abc_ms: f64,
  /// Time spent preparing the carried `Z` polynomial for the inner sumcheck.
  pub prepare_poly_z_ms: f64,
  /// Time spent in the inner sumcheck.
  pub inner_sumcheck_ms: f64,
  /// Time spent producing the PCS opening proof.
  pub pcs_prove_ms: f64,
  /// End-to-end proving time for this direct Spartan proof call.
  pub total_ms: f64,
}

impl<E: Engine> R1CSSNARK<E> {
  /// Measures the serialized size of the SNARK and its major carried components.
  pub fn serialized_size_breakdown(
    &self,
  ) -> Result<R1CSSNarkSerializedSizeBreakdown, SpartanError> {
    let encode = |label: &'static str, bytes: Result<Vec<u8>, _>| -> Result<usize, SpartanError> {
      bytes
        .map(|bytes| bytes.len())
        .map_err(|err| SpartanError::InternalError {
          reason: format!("failed to serialize {label}: {err}"),
        })
    };

    Ok(R1CSSNarkSerializedSizeBreakdown {
      total: encode("snark", bincode::serialize(self))?,
      instance: encode("instance", bincode::serialize(&self.U))?,
      outer_sumcheck: encode("outer_sumcheck", bincode::serialize(&self.sc_proof_outer))?,
      outer_claims: encode("outer_claims", bincode::serialize(&self.claims_outer))?,
      inner_sumcheck: encode("inner_sumcheck", bincode::serialize(&self.sc_proof_inner))?,
      eval_w: encode("eval_w", bincode::serialize(&self.eval_W))?,
      eval_arg: encode("eval_arg", bincode::serialize(&self.eval_arg))?,
      inner_sum_claim: encode("inner_sum_claim", bincode::serialize(&self.claim_inner_sum))?,
    })
  }

  /// Produces a Spartan proof together with a direct phase timing breakdown.
  pub fn prove_with_perf<C: SpartanCircuit<E>>(
    pk: &SpartanProverKey<E>,
    circuit: C,
    prep_snark: &PrepSNARK<E>,
    is_small: bool,
  ) -> Result<(Self, SpartanProvePerf), SpartanError> {
    let total_started = std::time::Instant::now();
    let mut perf = SpartanProvePerf::default();
    let mut prep_snark = prep_snark.clone(); // make a copy so we can modify it

    let mut transcript = E::TE::new(b"R1CSSNARK");
    transcript.absorb(b"vk", &pk.vk_digest);

    let public_values = circuit
      .public_values()
      .map_err(|e| SpartanError::SynthesisError {
        reason: format!("Circuit does not provide public IO: {e}"),
      })?;

    // absorb the public values into the transcript
    transcript.absorb(b"public_values", &public_values.as_slice());

    let (U, W) = SatisfyingAssignment::r1cs_instance_and_witness(
      &mut prep_snark.ps,
      &pk.S,
      &pk.ck,
      &circuit,
      is_small,
      &mut transcript,
    )?;

    // Get U_regular for both matrix operations and Z polynomial construction
    let U_regular = U.to_regular_instance()?;

    let num_vars = pk.S.num_shared + pk.S.num_precommitted + pk.S.num_rest;
    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(pk.S.num_cons.ilog2()).unwrap(),
      (usize::try_from(num_vars.ilog2()).unwrap() + 1),
    );

    // Binary PCS engines commit directly to multilinear evaluation tables.
    let is_binary_mle_engine = E::PCS::width() == 2;

    debug_assert_eq!(W.W.len(), num_vars);
    debug_assert!(
      U_regular.X.len() <= num_vars,
      "X vector too long: {} > {}",
      U_regular.X.len(),
      num_vars
    );

    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<EqPolynomial<_>, SpartanError>>()?;

    let (_poly_tau_span, poly_tau_t) = start_span!("prepare_poly_tau");
    let mut poly_tau = MultilinearPolynomial::new(tau.evals());
    perf.prepare_poly_tau_ms = poly_tau_t.elapsed().as_secs_f64() * 1_000.0;
    info!(elapsed_ms = %poly_tau_t.elapsed().as_millis(), "prepare_poly_tau");

    let (_mv_span, mv_t) = start_span!("matrix_vector_multiply");
    let z = [
      W.W.clone(),
      vec![E::Scalar::ONE],
      U_regular.X.clone(),
      U.challenges.clone(),
    ]
    .concat();
    let (Az, Bz, Cz) = pk.S.multiply_vec(&z)?;
    perf.matrix_vector_multiply_ms = mv_t.elapsed().as_secs_f64() * 1_000.0;
    info!(
      elapsed_ms = %mv_t.elapsed().as_millis(),
      constraints = %pk.S.num_cons,
      vars = %num_vars,
      "matrix_vector_multiply"
    );

    let (_mp_span, mp_t) = start_span!("prepare_multilinear_polys");
    let (mut poly_Az, mut poly_Bz, mut poly_Cz) = (
      MultilinearPolynomial::new(Az),
      MultilinearPolynomial::new(Bz),
      MultilinearPolynomial::new(Cz),
    );
    perf.prepare_multilinear_polys_ms = mp_t.elapsed().as_secs_f64() * 1_000.0;
    info!(elapsed_ms = %mp_t.elapsed().as_millis(), "prepare_multilinear_polys");

    let (_sc_span, sc_t) = start_span!("outer_sumcheck");
    let comb_func_outer =
      |poly_A_comp: &E::Scalar,
       poly_B_comp: &E::Scalar,
       poly_C_comp: &E::Scalar,
       poly_D_comp: &E::Scalar|
       -> E::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };
    let (sc_proof_outer, r_x, claims_outer) = SumcheckProof::prove_cubic_with_additive_term(
      &E::Scalar::ZERO,
      num_rounds_x,
      &mut poly_tau,
      &mut poly_Az,
      &mut poly_Bz,
      &mut poly_Cz,
      comb_func_outer,
      &mut transcript,
    )?;
    let (claim_Az, claim_Bz, claim_Cz): (E::Scalar, E::Scalar, E::Scalar) =
      (claims_outer[1], claims_outer[2], claims_outer[3]);
    transcript.absorb(b"claims_outer", &[claim_Az, claim_Bz, claim_Cz].as_slice());
    perf.outer_sumcheck_ms = sc_t.elapsed().as_secs_f64() * 1_000.0;
    info!(elapsed_ms = %sc_t.elapsed().as_millis(), "outer_sumcheck");

    let (_r_span, r_t) = start_span!("prepare_inner_claims");
    let r = transcript.squeeze(b"r")?;
    perf.prepare_inner_claims_ms = r_t.elapsed().as_secs_f64() * 1_000.0;
    info!(elapsed_ms = %r_t.elapsed().as_millis(), "prepare_inner_claims");

    let anchor = transcript.squeeze(b"anchor/inner+pcs")?;
    let mut t_sc = E::TE::new(b"R1CSSNARK/inner");
    t_sc.absorb(b"anchor", &anchor);
    let mut t_pcs = E::TE::new(b"R1CSSNARK/pcs");
    t_pcs.absorb(b"anchor", &anchor);

    #[cfg(debug_assertions)]
    tracing::debug!("FS anchor created for transcript fork: {:?}", anchor);

    let (_eval_rx_span, eval_rx_t) = start_span!("compute_eval_rx");
    let evals_rx = EqPolynomial::evals_from_points(&r_x.clone());
    perf.compute_eval_rx_ms = eval_rx_t.elapsed().as_secs_f64() * 1_000.0;
    info!(elapsed_ms = %eval_rx_t.elapsed().as_millis(), "compute_eval_rx");

    let (_sparse_span, sparse_t) = start_span!("compute_eval_table_sparse");
    let (evals_A, evals_B, evals_C) = compute_eval_table_sparse(&pk.S, &evals_rx);
    perf.compute_eval_table_sparse_ms = sparse_t.elapsed().as_secs_f64() * 1_000.0;
    info!(elapsed_ms = %sparse_t.elapsed().as_millis(), "compute_eval_table_sparse");

    let (_abc_span, abc_t) = start_span!("prepare_poly_ABC");
    assert_eq!(evals_A.len(), evals_B.len());
    assert_eq!(evals_A.len(), evals_C.len());
    let poly_ABC = if crate::parallel::parallelism_enabled() {
      (0..evals_A.len())
        .into_par_iter()
        .map(|i| evals_A[i] + r * evals_B[i] + r * r * evals_C[i])
        .collect::<Vec<E::Scalar>>()
    } else {
      (0..evals_A.len())
        .map(|i| evals_A[i] + r * evals_B[i] + r * r * evals_C[i])
        .collect::<Vec<E::Scalar>>()
    };
    perf.prepare_poly_abc_ms = abc_t.elapsed().as_secs_f64() * 1_000.0;
    info!(elapsed_ms = %abc_t.elapsed().as_millis(), "prepare_poly_ABC");

    let (_z_span, z_t) = start_span!("prepare_poly_z");
    let poly_z = if is_binary_mle_engine {
      let mut z = Vec::with_capacity(2 * num_vars);

      let w_len = core::cmp::min(W.W.len(), num_vars);
      z.extend_from_slice(&W.W[..w_len]);
      if w_len < num_vars {
        z.resize(num_vars, E::Scalar::ZERO);
      }

      match U_regular.X.len() {
        0 => {
          z.resize(2 * num_vars, E::Scalar::ONE);
        }
        1 => {
          z.resize(2 * num_vars, U_regular.X[0]);
        }
        _ => {
          z.push(E::Scalar::ONE);
          let x_len = core::cmp::min(U_regular.X.len(), num_vars.saturating_sub(1));
          z.extend_from_slice(&U_regular.X[..x_len]);
          z.resize(2 * num_vars, E::Scalar::ZERO);
        }
      }

      z
    } else {
      let mut z = Vec::with_capacity(2 * num_vars);

      let w_len = core::cmp::min(W.W.len(), num_vars);
      z.extend_from_slice(&W.W[..w_len]);
      if w_len < num_vars {
        z.resize(num_vars, E::Scalar::ZERO);
      }

      z.push(E::Scalar::ONE);
      let x_len = core::cmp::min(U_regular.X.len(), num_vars.saturating_sub(1));
      z.extend_from_slice(&U_regular.X[..x_len]);
      z.resize(2 * num_vars, E::Scalar::ZERO);

      z
    };
    perf.prepare_poly_z_ms = z_t.elapsed().as_secs_f64() * 1_000.0;
    info!(elapsed_ms = %z_t.elapsed().as_millis(), "prepare_poly_z");

    let (_sc2_span, sc2_t) = start_span!("inner_sumcheck");
    let comb_func = |poly_A_comp: &E::Scalar, poly_B_comp: &E::Scalar| -> E::Scalar {
      *poly_A_comp * *poly_B_comp
    };
    let claim_inner_sum: E::Scalar = if crate::parallel::parallelism_enabled() {
      poly_ABC.par_iter().zip(&poly_z).map(|(a, z)| *a * *z).sum()
    } else {
      poly_ABC.iter().zip(&poly_z).map(|(a, z)| *a * *z).sum()
    };

    #[cfg(debug_assertions)]
    let _poly_z_for_debug = poly_z.clone();
    let (sc_proof_inner, r_y, _claims_inner) = SumcheckProof::prove_quad(
      &claim_inner_sum,
      num_rounds_y,
      &mut MultilinearPolynomial::new(poly_ABC),
      &mut MultilinearPolynomial::new(poly_z),
      comb_func,
      &mut t_sc,
    )?;
    perf.inner_sumcheck_ms = sc2_t.elapsed().as_secs_f64() * 1_000.0;
    info!(elapsed_ms = %sc2_t.elapsed().as_millis(), "inner_sumcheck");

    #[cfg(debug_assertions)]
    tracing::debug!(
      "Prover r_y.len(): {}, y0 (gate) = {:?}",
      r_y.len(),
      if r_y.is_empty() {
        E::Scalar::ZERO
      } else {
        r_y[0]
      }
    );

    let ry_no_gate: &[E::Scalar] = if r_y.is_empty() { &[] } else { &r_y[1..] };

    #[cfg(debug_assertions)]
    let gate: E::Scalar = if r_y.is_empty() {
      E::Scalar::ZERO
    } else {
      r_y[0]
    };

    #[cfg(debug_assertions)]
    {
      tracing::debug!("PROVER gate bit y0 = {:?}", gate);
      tracing::debug!("PROVER |ry_no_gate| = {}", ry_no_gate.len());
    }

    let (_pcs_span, pcs_t) = start_span!("pcs_prove");
    let (eval_W, eval_arg) = E::PCS::prove(
      &pk.ck,
      &mut t_pcs,
      &U_regular.comm_W,
      &W.W,
      &W.r_W,
      ry_no_gate,
    )?;
    perf.pcs_prove_ms = pcs_t.elapsed().as_secs_f64() * 1_000.0;
    info!(elapsed_ms = %pcs_t.elapsed().as_millis(), "pcs_prove");

    #[cfg(debug_assertions)]
    if is_binary_mle_engine && num_vars > 0 {
      let eval_X_check = {
        let mut X_full = vec![E::Scalar::ZERO; num_vars];
        match U_regular.X.len() {
          0 => X_full.fill(E::Scalar::ONE),
          1 => X_full.fill(U_regular.X[0]),
          _ => {
            X_full[0] = E::Scalar::ONE;
            for (i, xi) in U_regular.X.iter().cloned().enumerate() {
              if i + 1 < num_vars {
                X_full[i + 1] = xi;
              }
            }
          }
        }
        crate::polys::multilinear::MultilinearPolynomial::new(X_full).evaluate(ry_no_gate)
      };

      let eval_Z_expected = (E::Scalar::ONE - gate) * eval_W + gate * eval_X_check;
      let eval_Z_table =
        crate::polys::multilinear::MultilinearPolynomial::new(_poly_z_for_debug.clone())
          .evaluate(&r_y);

      debug_assert_eq!(
        eval_Z_expected, eval_Z_table,
        "Hash-MLE gated Z polynomial mismatch"
      );
    }

    perf.total_ms = total_started.elapsed().as_secs_f64() * 1_000.0;
    Ok((
      R1CSSNARK {
        U,
        sc_proof_outer,
        claims_outer: (claim_Az, claim_Bz, claim_Cz),
        sc_proof_inner,
        eval_W,
        eval_arg,
        claim_inner_sum,
      },
      perf,
    ))
  }
}

impl<E: Engine> R1CSSNARKTrait<E> for R1CSSNARK<E> {
  type ProverKey = SpartanProverKey<E>;
  type VerifierKey = SpartanVerifierKey<E>;
  type PrepSNARK = PrepSNARK<E>;

  fn setup<C: SpartanCircuit<E>>(
    circuit: C,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), SpartanError> {
    // Binary PCS engines use the direct multilinear-evaluation-table layout.
    let pcs_w = <E as Engine>::PCS::width();
    if pcs_w == 0 || !pcs_w.is_power_of_two() {
      return Err(SpartanError::InternalError {
        reason: format!("PCS width must be a non-zero power of two, got {pcs_w}"),
      });
    }

    let S = ShapeCS::r1cs_shape(&circuit)?;
    let (ck, vk_ee) = SplitR1CSShape::commitment_key(&[&S])?;

    let vk: SpartanVerifierKey<E> = SpartanVerifierKey {
      S: S.clone(),
      vk_ee,
      digest: OnceCell::new(),
    };
    let pk = Self::ProverKey {
      ck,
      S,
      vk_digest: vk.digest()?,
    };

    Ok((pk, vk))
  }

  /// Prepares the SNARK for proving
  fn prep_prove<C: SpartanCircuit<E>>(
    pk: &Self::ProverKey,
    circuit: C,
    is_small: bool, // do witness elements fit in machine words?
  ) -> Result<Self::PrepSNARK, SpartanError> {
    let mut ps = SatisfyingAssignment::shared_witness(&pk.S, &pk.ck, &circuit, is_small)?;
    SatisfyingAssignment::precommitted_witness(&mut ps, &pk.S, &pk.ck, &circuit, is_small)?;

    Ok(PrepSNARK { ps })
  }

  /// produces a succinct proof of satisfiability of an R1CS instance
  fn prove<C: SpartanCircuit<E>>(
    pk: &Self::ProverKey,
    circuit: C,
    prep_snark: &Self::PrepSNARK,
    is_small: bool,
  ) -> Result<Self, SpartanError> {
    Self::prove_with_perf(pk, circuit, prep_snark, is_small).map(|(proof, _)| proof)
  }

  /// verifies a proof of satisfiability of a `RelaxedR1CS` instance
  fn verify(&self, vk: &Self::VerifierKey) -> Result<Vec<E::Scalar>, SpartanError> {
    let (_verify_span, verify_t) = start_span!("r1cs_snark_verify");
    let mut transcript = E::TE::new(b"R1CSSNARK");

    // append the digest of R1CS matrices
    transcript.absorb(b"vk", &vk.digest()?);

    // validate the provided split R1CS instance and convert to regular instance
    self.U.validate(&vk.S, &mut transcript)?;
    let U_regular = self.U.to_regular_instance()?;

    let num_vars = vk.S.num_shared + vk.S.num_precommitted + vk.S.num_rest;

    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(vk.S.num_cons.ilog2()).unwrap(),
      (usize::try_from(num_vars.ilog2()).unwrap() + 1),
    );

    info!(
      "Verifying R1CS SNARK with {} rounds for outer sum-check and {} rounds for inner sum-check",
      num_rounds_x, num_rounds_y
    );

    // outer sum-check
    let (_tau_span, tau_t) = start_span!("compute_tau_verify");
    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<EqPolynomial<_>, SpartanError>>()?;
    info!(elapsed_ms = %tau_t.elapsed().as_millis(), "compute_tau_verify");

    let (_outer_sumcheck_span, outer_sumcheck_t) = start_span!("outer_sumcheck_verify");
    let (claim_outer_final, r_x) =
      self
        .sc_proof_outer
        .verify(E::Scalar::ZERO, num_rounds_x, 3, &mut transcript)?;

    // verify claim_outer_final
    let (claim_Az, claim_Bz, claim_Cz) = self.claims_outer;
    let taus_bound_rx = tau.evaluate(&r_x);
    let claim_outer_final_expected = taus_bound_rx * (claim_Az * claim_Bz - claim_Cz);
    if claim_outer_final != claim_outer_final_expected {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    info!(elapsed_ms = %outer_sumcheck_t.elapsed().as_millis(), "outer_sumcheck_verify");

    transcript.absorb(
      b"claims_outer",
      &[
        self.claims_outer.0,
        self.claims_outer.1,
        self.claims_outer.2,
      ]
      .as_slice(),
    );

    // inner sum-check
    let (_inner_sumcheck_span, inner_sumcheck_t) = start_span!("inner_sumcheck_verify");
    let r = transcript.squeeze(b"r")?;

    // Use the same anchoring and forks as in `prove`
    let anchor = transcript.squeeze(b"anchor/inner+pcs")?;
    let mut t_sc = E::TE::new(b"R1CSSNARK/inner");
    t_sc.absorb(b"anchor", &anchor);
    let mut t_pcs = E::TE::new(b"R1CSSNARK/pcs");
    t_pcs.absorb(b"anchor", &anchor);

    #[cfg(debug_assertions)]
    tracing::debug!("FS anchor verified for transcript fork: {:?}", anchor);

    let (claim_inner_final, r_y) =
      self
        .sc_proof_inner
        .verify(self.claim_inner_sum, num_rounds_y, 2, &mut t_sc)?;

    #[cfg(debug_assertions)]
    tracing::debug!(
      "Verifier r_y[0] (gating bit): {:?}, r_y.len(): {}",
      if r_y.is_empty() {
        E::Scalar::ZERO
      } else {
        r_y[0]
      },
      r_y.len()
    );

    let is_binary_mle_engine = E::PCS::width() == 2;

    // Gate is y0. Non-gate coordinates are y1..
    let (gate, ry_no_gate): (E::Scalar, &[E::Scalar]) = if r_y.is_empty() {
      (E::Scalar::ZERO, &[])
    } else {
      (r_y[0], &r_y[1..])
    };

    #[cfg(debug_assertions)]
    {
      tracing::debug!("VERIFIER gate bit y0 = {:?}", gate);
      tracing::debug!("VERIFIER |ry_no_gate| = {}", ry_no_gate.len());
    }

    // Compute eval_Z with gate=y0 and evaluate X on y1..
    let eval_Z = {
      let eval_X = if is_binary_mle_engine {
        if U_regular.X.is_empty() {
          E::Scalar::ONE
        } else {
          // num_vars = 2^{m-1} (the Y-table width without the gate bit)
          let mut x_full = vec![E::Scalar::ZERO; num_vars];
          match U_regular.X.len() {
            0 => {
              x_full.fill(E::Scalar::ONE); // not taken (guarded), kept for symmetry
            }
            1 => {
              x_full.fill(U_regular.X[0]); // broadcast-const
            }
            _ => {
              x_full[0] = E::Scalar::ONE; // (1, X...)
              for (i, xi) in U_regular.X.iter().cloned().enumerate() {
                if i + 1 < num_vars {
                  x_full[i + 1] = xi;
                }
              }
            }
          }
          let eval_x =
            crate::polys::multilinear::MultilinearPolynomial::new(x_full).evaluate(ry_no_gate);
          eval_x
        }
      } else {
        // Original logic for non-Hash-MLE engines
        let X = vec![E::Scalar::ONE]
          .into_iter()
          .chain(U_regular.X.iter().cloned())
          .collect::<Vec<E::Scalar>>();
        SparsePolynomial::new(num_vars.log_2(), X).evaluate(ry_no_gate)
      };

      let eval_z = (E::Scalar::ONE - gate) * self.eval_W + gate * eval_X;

      eval_z
    };

    // compute evaluations of R1CS matrices
    let (_matrix_eval_span, matrix_eval_t) = start_span!("matrix_evaluations");
    let multi_evaluate = |M_vec: &[&SparseMatrix<E::Scalar>],
                          r_x: &[E::Scalar],
                          r_y: &[E::Scalar]|
     -> Vec<E::Scalar> {
      let evaluate_with_table =
        |M: &SparseMatrix<E::Scalar>, T_x: &[E::Scalar], T_y: &[E::Scalar]| -> E::Scalar {
          if crate::parallel::parallelism_enabled() {
            M.indptr
              .par_windows(2)
              .enumerate()
              .map(|(row_idx, ptrs)| {
                M.get_row_unchecked(ptrs.try_into().unwrap())
                  .map(|(val, col_idx)| {
                    let prod = T_x[row_idx] * T_y[*col_idx];
                    if *val == E::Scalar::ONE {
                      prod
                    } else if *val == -E::Scalar::ONE {
                      -prod
                    } else {
                      prod * val
                    }
                  })
                  .sum::<E::Scalar>()
              })
              .sum()
          } else {
            M.indptr
              .windows(2)
              .enumerate()
              .map(|(row_idx, ptrs)| {
                M.get_row_unchecked(ptrs.try_into().unwrap())
                  .map(|(val, col_idx)| {
                    let prod = T_x[row_idx] * T_y[*col_idx];
                    if *val == E::Scalar::ONE {
                      prod
                    } else if *val == -E::Scalar::ONE {
                      -prod
                    } else {
                      prod * val
                    }
                  })
                  .sum::<E::Scalar>()
              })
              .sum()
          }
        };

      let (T_x, T_y) = crate::parallel::join(
        || EqPolynomial::evals_from_points(r_x),
        || EqPolynomial::evals_from_points(r_y),
      );

      if crate::parallel::parallelism_enabled() {
        (0..M_vec.len())
          .into_par_iter()
          .map(|i| evaluate_with_table(M_vec[i], &T_x, &T_y))
          .collect()
      } else {
        (0..M_vec.len())
          .map(|i| evaluate_with_table(M_vec[i], &T_x, &T_y))
          .collect()
      }
    };

    let evals = multi_evaluate(&[&vk.S.A, &vk.S.B, &vk.S.C], &r_x, &r_y);

    let claim_inner_final_expected = (evals[0] + r * evals[1] + r * r * evals[2]) * eval_Z;

    if claim_inner_final != claim_inner_final_expected {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    info!(elapsed_ms = %matrix_eval_t.elapsed().as_millis(), "matrix_evaluations");
    info!(elapsed_ms = %inner_sumcheck_t.elapsed().as_millis(), "inner_sumcheck_verify");

    // verify
    let (_pcs_verify_span, pcs_verify_t) = start_span!("pcs_verify");
    E::PCS::verify(
      &vk.vk_ee,
      &mut t_pcs,
      &U_regular.comm_W,
      ry_no_gate, // non-gate coordinates (y1..)
      &self.eval_W,
      &self.eval_arg,
    )?;
    info!(elapsed_ms = %pcs_verify_t.elapsed().as_millis(), "pcs_verify");

    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "r1cs_snark_verify");
    Ok(self.U.public_values.clone())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};
  use std::sync::Once;
  use tracing_subscriber::EnvFilter;

  #[derive(Clone, Debug, Default)]
  struct CubicCircuit {}

  impl<E: Engine> SpartanCircuit<E> for CubicCircuit {
    fn public_values(&self) -> Result<Vec<<E as Engine>::Scalar>, SynthesisError> {
      Ok(vec![E::Scalar::from(15u64)])
    }

    fn shared<CS: ConstraintSystem<E::Scalar>>(
      &self,
      _: &mut CS,
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
      // In this example, we do not have shared variables.
      Ok(vec![])
    }

    fn precommitted<CS: ConstraintSystem<<E as Engine>::Scalar>>(
      &self,
      _: &mut CS,
      _: &[AllocatedNum<E::Scalar>], // shared variables, if any
    ) -> Result<Vec<AllocatedNum<<E as Engine>::Scalar>>, SynthesisError> {
      // In this example, we do not have precommitted variables.
      Ok(vec![])
    }

    fn num_challenges(&self) -> usize {
      // In this example, we do not use challenges.
      0
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
      _: &[AllocatedNum<E::Scalar>],
      _: &[AllocatedNum<E::Scalar>],
      _: Option<&[E::Scalar]>,
    ) -> Result<(), SynthesisError> {
      // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
      let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(E::Scalar::ONE + E::Scalar::ONE))?;
      let x_sq = x.square(cs.namespace(|| "x_sq"))?;
      let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), &x)?;
      let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
        Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + E::Scalar::from(5u64))
      })?;

      cs.enforce(
        || "y = x^3 + x + 5",
        |lc| {
          lc + x_cu.get_variable()
            + x.get_variable()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
        },
        |lc| lc + CS::one(),
        |lc| lc + y.get_variable(),
      );

      let _ = y.inputize(cs.namespace(|| "output"));

      Ok(())
    }
  }

  fn init_test_tracing() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
      let _ = tracing_subscriber::fmt()
        .with_target(false)
        .with_ansi(true)
        .with_env_filter(EnvFilter::from_default_env())
        .try_init();
    });
  }

  #[test]
  fn test_snark() {
    init_test_tracing();

    type E = crate::provider::PallasHyraxEngine;
    type S = R1CSSNARK<E>;
    test_snark_with::<E, S>();

    type E2 = crate::provider::T256HyraxEngine;
    type S2 = R1CSSNARK<E2>;
    test_snark_with::<E2, S2>();
  }

  #[test]
  fn test_transcript_fork_consistency() {
    // Test that transcript forking prevents FS interleaving issues
    // This specifically exercises Hash-MLE engines where transcript interleaving
    // between inner sumcheck and PCS could cause different r_y values

    init_test_tracing();

    type E = crate::provider::GoldilocksMerkleMleEngine;
    type S = R1CSSNARK<E>;

    let circuit = CubicCircuit::default();
    let (pk, vk) = S::setup(circuit.clone()).unwrap();
    let prep_snark = S::prep_prove(&pk, circuit.clone(), false).unwrap();
    let proof = S::prove(&pk, circuit.clone(), &prep_snark, false).unwrap();

    // If transcript forking works correctly, verification should succeed
    // (previously would fail with InvalidSumcheckProof due to r_y mismatch)
    let res = proof.verify(&vk);
    if let Err(e) = &res {
      println!("❌ Verification failed with error: {:?}", e);
    }
    assert!(
      res.is_ok(),
      "Transcript fork should prevent FS interleaving issues: {:?}",
      res.err()
    );

    println!(
      "✅ Transcript fork consistency test passed - r_y values synchronized between prover/verifier"
    );
  }

  fn test_snark_with<E: Engine, S: R1CSSNARKTrait<E>>() {
    let circuit = CubicCircuit::default();

    // produce keys
    let (pk, vk) = S::setup(circuit.clone()).unwrap();

    // generate pre-processed state for proving
    let prep_snark = S::prep_prove(&pk, circuit.clone(), false).unwrap();

    // generate a witness and proof
    let res = S::prove(&pk, circuit.clone(), &prep_snark, false);
    assert!(res.is_ok());
    let snark = res.unwrap();

    // verify the SNARK
    let res = snark.verify(&vk);
    assert!(res.is_ok());
    assert_eq!(res.unwrap(), [<E as Engine>::Scalar::from(15u64)])
  }
}
