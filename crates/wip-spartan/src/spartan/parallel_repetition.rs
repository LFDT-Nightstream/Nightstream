//! Lockstep Fiat--Shamir amplification for the direct sparse R1CS proof.
//!
//! This module owns one rule: every round commits all three sum-check
//! messages before it derives any member challenge. It does not define a new
//! constraint system or polynomial commitment scheme. Its probability bound
//! is conditional on the Fiat--Shamir random-oracle model and WHIR binding.

use super::*;
use rayon::prelude::*;

const PARALLEL_DOMAIN: &[u8] = b"direct-r1cs/parallel-3/v1";
const TAU_LABELS: [&[u8]; 3] = [b"t/0", b"t/1", b"t/2"];
const CLAIM_LABELS: [&[u8]; 3] = [b"claims_outer/0", b"claims_outer/1", b"claims_outer/2"];
const INNER_MIX_LABELS: [&[u8]; 3] = [b"r/0", b"r/1", b"r/2"];
const PCS_ANCHOR_LABELS: [&[u8]; 3] = [b"pcs_anchor/0", b"pcs_anchor/1", b"pcs_anchor/2"];

impl<E: Engine> RepeatedR1CSSNARK<E> {
  /// Proves one direct sparse R1CS instance with three lockstep sum-check
  /// members.
  pub fn prove_direct(
    pk: &SpartanProverKey<E>,
    witness: &[E::Scalar],
    public_values: &[E::Scalar],
    is_small: bool,
  ) -> Result<Self, SpartanError> {
    ensure_parallel_sumcheck_bound(&pk.S)?;
    let (instance, witness) =
      R1CSSNARK::prepare_direct_instance(pk, witness, public_values, is_small)?;

    let mut transcript = E::TE::new(b"R1CSSNARK/parallel-3");
    transcript.dom_sep(PARALLEL_DOMAIN);
    transcript.absorb(b"vk", &pk.vk_digest);
    instance.validate(&pk.S, &mut transcript)?;

    prove_parallel(pk, instance, witness, transcript)
  }

  /// Verifies all lockstep members and returns their shared public values.
  pub fn verify(&self, vk: &SpartanVerifierKey<E>) -> Result<Vec<E::Scalar>, SpartanError> {
    verify_parallel(self, vk)
  }

  /// Accesses the three proof members for size reporting.
  pub fn proofs(&self) -> &[R1CSSNARK<E>] {
    &self.proofs
  }
}

fn prove_parallel<E: Engine>(
  pk: &SpartanProverKey<E>,
  instance: SplitR1CSInstance<E>,
  witness: R1CSWitness<E>,
  mut transcript: E::TE,
) -> Result<RepeatedR1CSSNARK<E>, SpartanError> {
  let regular = instance.to_regular_instance()?;
  let num_vars = pk.S.num_shared + pk.S.num_precommitted + pk.S.num_rest;
  let num_rounds_x =
    usize::try_from(pk.S.num_cons.ilog2()).map_err(|_| SpartanError::InternalError {
      reason: "outer sum-check round count does not fit usize".to_string(),
    })?;
  let num_rounds_y = usize::try_from(num_vars.ilog2())
    .map_err(|_| SpartanError::InternalError {
      reason: "inner sum-check round count does not fit usize".to_string(),
    })?
    .checked_add(1)
    .ok_or_else(|| SpartanError::InternalError {
      reason: "inner sum-check round count overflow".to_string(),
    })?;

  let mut tau_points: [Vec<E::Scalar>; 3] = std::array::from_fn(|_| Vec::new());
  for _round in 0..num_rounds_x {
    for member in 0..3 {
      tau_points[member].push(transcript.squeeze(TAU_LABELS[member])?);
    }
  }
  let mut poly_tau =
    tau_points.map(|points| MultilinearPolynomial::new(EqPolynomial::new(points).evals()));

  let z = [
    witness.W.clone(),
    vec![E::Scalar::ONE],
    regular.X.clone(),
    instance.challenges.clone(),
  ]
  .concat();
  let (az, bz, cz) = pk.S.multiply_vec(&z)?;
  let mut poly_az = std::array::from_fn(|_| MultilinearPolynomial::new(az.clone()));
  let mut poly_bz = std::array::from_fn(|_| MultilinearPolynomial::new(bz.clone()));
  let mut poly_cz = std::array::from_fn(|_| MultilinearPolynomial::new(cz.clone()));
  let comb_outer = |tau: &E::Scalar, a: &E::Scalar, b: &E::Scalar, c: &E::Scalar| -> E::Scalar {
    *tau * (*a * *b - *c)
  };
  let (outer_proofs, r_x, outer_evaluations) = SumcheckProof::prove_cubic_parallel_3(
    [E::Scalar::ZERO; 3],
    num_rounds_x,
    &mut poly_tau,
    &mut poly_az,
    &mut poly_bz,
    &mut poly_cz,
    comb_outer,
    &mut transcript,
  )?;
  let claims_outer: [(E::Scalar, E::Scalar, E::Scalar); 3] = std::array::from_fn(|member| {
    (
      outer_evaluations[member][1],
      outer_evaluations[member][2],
      outer_evaluations[member][3],
    )
  });

  for member in 0..3 {
    let values = [
      claims_outer[member].0,
      claims_outer[member].1,
      claims_outer[member].2,
    ];
    transcript.absorb(CLAIM_LABELS[member], &values.as_slice());
  }
  let inner_mix = [
    transcript.squeeze(INNER_MIX_LABELS[0])?,
    transcript.squeeze(INNER_MIX_LABELS[1])?,
    transcript.squeeze(INNER_MIX_LABELS[2])?,
  ];

  let z_table = build_z_table::<E>(&regular.X, &witness.W, num_vars);
  let mut poly_abc: [MultilinearPolynomial<E::Scalar>; 3] = std::array::from_fn(|member| {
    let evals_rx = EqPolynomial::evals_from_points(&r_x[member]);
    let (evals_a, evals_b, evals_c) = compute_eval_table_sparse(&pk.S, &evals_rx);
    let mix = inner_mix[member];
    MultilinearPolynomial::new(
      (0..evals_a.len())
        .map(|index| evals_a[index] + mix * evals_b[index] + mix * mix * evals_c[index])
        .collect(),
    )
  });
  let mut poly_z = std::array::from_fn(|_| MultilinearPolynomial::new(z_table.clone()));
  let inner_claims = std::array::from_fn(|member| {
    if crate::parallel::parallelism_enabled() {
      poly_abc[member]
        .Z
        .par_iter()
        .zip(&poly_z[member].Z)
        .map(|(a, z)| *a * *z)
        .sum()
    } else {
      poly_abc[member]
        .Z
        .iter()
        .zip(&poly_z[member].Z)
        .map(|(a, z)| *a * *z)
        .sum()
    }
  });

  transcript.dom_sep(b"direct-r1cs/parallel-3/inner");
  let comb_inner = |a: &E::Scalar, b: &E::Scalar| -> E::Scalar { *a * *b };
  let (inner_proofs, r_y, _inner_evaluations) = SumcheckProof::prove_quad_parallel_3(
    inner_claims,
    num_rounds_y,
    &mut poly_abc,
    &mut poly_z,
    comb_inner,
    &mut transcript,
  )?;

  let pcs_anchors = [
    transcript.squeeze(PCS_ANCHOR_LABELS[0])?,
    transcript.squeeze(PCS_ANCHOR_LABELS[1])?,
    transcript.squeeze(PCS_ANCHOR_LABELS[2])?,
  ];

  let mut proofs = Vec::with_capacity(3);
  for member in 0..3 {
    let mut pcs_transcript = E::TE::new(b"R1CSSNARK/parallel-3/pcs");
    pcs_transcript.absorb(b"anchor", &pcs_anchors[member]);
    let ry_no_gate = r_y[member].get(1..).unwrap_or(&[]);
    let (eval_w, eval_arg) = E::PCS::prove(
      &pk.ck,
      &mut pcs_transcript,
      &regular.comm_W,
      &witness.W,
      &witness.r_W,
      ry_no_gate,
    )?;
    proofs.push(R1CSSNARK {
      U: instance.clone(),
      parallel_member: u8::try_from(member + 1).expect("three member indices fit in u8"),
      sc_proof_outer: outer_proofs[member].clone(),
      claims_outer: claims_outer[member],
      sc_proof_inner: inner_proofs[member].clone(),
      eval_W: eval_w,
      eval_arg,
      claim_inner_sum: inner_claims[member],
    });
  }

  Ok(RepeatedR1CSSNARK { proofs })
}

fn verify_parallel<E: Engine>(
  proof: &RepeatedR1CSSNARK<E>,
  vk: &SpartanVerifierKey<E>,
) -> Result<Vec<E::Scalar>, SpartanError> {
  ensure_parallel_sumcheck_bound(&vk.S)?;
  if proof.proofs.len() != DIRECT_R1CS_REPETITIONS {
    return Err(SpartanError::InvalidSumcheckProof);
  }
  for member in 0..3 {
    let expected = u8::try_from(member + 1).expect("three member indices fit in u8");
    if proof.proofs[member].parallel_member != expected
      || proof.proofs[member].U != proof.proofs[0].U
    {
      return Err(SpartanError::InvalidSumcheckProof);
    }
  }

  let members = [&proof.proofs[0], &proof.proofs[1], &proof.proofs[2]];
  let mut transcript = E::TE::new(b"R1CSSNARK/parallel-3");
  transcript.dom_sep(PARALLEL_DOMAIN);
  transcript.absorb(b"vk", &vk.digest()?);
  members[0].U.validate(&vk.S, &mut transcript)?;
  let regular = members[0].U.to_regular_instance()?;
  let num_vars = vk.S.num_shared + vk.S.num_precommitted + vk.S.num_rest;
  let num_rounds_x =
    usize::try_from(vk.S.num_cons.ilog2()).map_err(|_| SpartanError::InternalError {
      reason: "outer sum-check round count does not fit usize".to_string(),
    })?;
  let num_rounds_y = usize::try_from(num_vars.ilog2())
    .map_err(|_| SpartanError::InternalError {
      reason: "inner sum-check round count does not fit usize".to_string(),
    })?
    .checked_add(1)
    .ok_or_else(|| SpartanError::InternalError {
      reason: "inner sum-check round count overflow".to_string(),
    })?;

  let mut tau_points: [Vec<E::Scalar>; 3] = std::array::from_fn(|_| Vec::new());
  for _round in 0..num_rounds_x {
    for member in 0..3 {
      tau_points[member].push(transcript.squeeze(TAU_LABELS[member])?);
    }
  }
  let (outer_finals, r_x) = SumcheckProof::verify_parallel_3(
    [
      &members[0].sc_proof_outer,
      &members[1].sc_proof_outer,
      &members[2].sc_proof_outer,
    ],
    [E::Scalar::ZERO; 3],
    num_rounds_x,
    3,
    &mut transcript,
  )?;
  for member in 0..3 {
    let (claim_a, claim_b, claim_c) = members[member].claims_outer;
    let expected = EqPolynomial::new(tau_points[member].clone()).evaluate(&r_x[member])
      * (claim_a * claim_b - claim_c);
    if outer_finals[member] != expected {
      return Err(SpartanError::InvalidSumcheckProof);
    }
  }

  for member in 0..3 {
    let values = [
      members[member].claims_outer.0,
      members[member].claims_outer.1,
      members[member].claims_outer.2,
    ];
    transcript.absorb(CLAIM_LABELS[member], &values.as_slice());
  }
  let inner_mix = [
    transcript.squeeze(INNER_MIX_LABELS[0])?,
    transcript.squeeze(INNER_MIX_LABELS[1])?,
    transcript.squeeze(INNER_MIX_LABELS[2])?,
  ];

  transcript.dom_sep(b"direct-r1cs/parallel-3/inner");
  let (inner_finals, r_y) = SumcheckProof::verify_parallel_3(
    [
      &members[0].sc_proof_inner,
      &members[1].sc_proof_inner,
      &members[2].sc_proof_inner,
    ],
    [
      members[0].claim_inner_sum,
      members[1].claim_inner_sum,
      members[2].claim_inner_sum,
    ],
    num_rounds_y,
    2,
    &mut transcript,
  )?;

  for member in 0..3 {
    let (gate, ry_no_gate) = r_y[member]
      .split_first()
      .map_or((E::Scalar::ZERO, &[][..]), |(gate, rest)| (*gate, rest));
    let eval_x = evaluate_public_table::<E>(&regular.X, num_vars, ry_no_gate);
    let eval_z = (E::Scalar::ONE - gate) * members[member].eval_W + gate * eval_x;
    let evaluations = evaluate_matrices(&vk.S, &r_x[member], &r_y[member]);
    let mix = inner_mix[member];
    let expected = (evaluations[0] + mix * evaluations[1] + mix * mix * evaluations[2]) * eval_z;
    if inner_finals[member] != expected {
      return Err(SpartanError::InvalidSumcheckProof);
    }
  }

  let pcs_anchors = [
    transcript.squeeze(PCS_ANCHOR_LABELS[0])?,
    transcript.squeeze(PCS_ANCHOR_LABELS[1])?,
    transcript.squeeze(PCS_ANCHOR_LABELS[2])?,
  ];
  for member in 0..3 {
    let mut pcs_transcript = E::TE::new(b"R1CSSNARK/parallel-3/pcs");
    pcs_transcript.absorb(b"anchor", &pcs_anchors[member]);
    let ry_no_gate = r_y[member].get(1..).unwrap_or(&[]);
    E::PCS::verify(
      &vk.vk_ee,
      &mut pcs_transcript,
      &regular.comm_W,
      ry_no_gate,
      &members[member].eval_W,
      &members[member].eval_arg,
    )?;
  }

  Ok(members[0].U.public_values.clone())
}

fn build_z_table<E: Engine>(
  public_values: &[E::Scalar],
  witness: &[E::Scalar],
  num_vars: usize,
) -> Vec<E::Scalar> {
  let mut z = Vec::with_capacity(2 * num_vars);
  z.extend_from_slice(&witness[..core::cmp::min(witness.len(), num_vars)]);
  z.resize(num_vars, E::Scalar::ZERO);

  if E::PCS::width() == 2 {
    match public_values.len() {
      0 => z.resize(2 * num_vars, E::Scalar::ONE),
      1 => z.resize(2 * num_vars, public_values[0]),
      _ => {
        z.push(E::Scalar::ONE);
        z.extend_from_slice(
          &public_values[..core::cmp::min(public_values.len(), num_vars.saturating_sub(1))],
        );
        z.resize(2 * num_vars, E::Scalar::ZERO);
      }
    }
  } else {
    z.push(E::Scalar::ONE);
    z.extend_from_slice(
      &public_values[..core::cmp::min(public_values.len(), num_vars.saturating_sub(1))],
    );
    z.resize(2 * num_vars, E::Scalar::ZERO);
  }
  z
}

fn evaluate_public_table<E: Engine>(
  public_values: &[E::Scalar],
  num_vars: usize,
  point: &[E::Scalar],
) -> E::Scalar {
  if E::PCS::width() != 2 {
    let values = vec![E::Scalar::ONE]
      .into_iter()
      .chain(public_values.iter().copied())
      .collect();
    return SparsePolynomial::new(num_vars.log_2(), values).evaluate(point);
  }

  let mut table = vec![E::Scalar::ZERO; num_vars];
  match public_values.len() {
    0 => table.fill(E::Scalar::ONE),
    1 => table.fill(public_values[0]),
    _ => {
      table[0] = E::Scalar::ONE;
      for (index, value) in public_values.iter().copied().enumerate() {
        if index + 1 < num_vars {
          table[index + 1] = value;
        }
      }
    }
  }
  MultilinearPolynomial::new(table).evaluate(point)
}

fn evaluate_matrices<E: Engine>(
  shape: &SplitR1CSShape<E>,
  r_x: &[E::Scalar],
  r_y: &[E::Scalar],
) -> [E::Scalar; 3] {
  let table_x = EqPolynomial::evals_from_points(r_x);
  let table_y = EqPolynomial::evals_from_points(r_y);
  let evaluate = |matrix: &SparseMatrix<E::Scalar>| {
    matrix
      .indptr
      .par_windows(2)
      .enumerate()
      .map(|(row, pointers)| {
        matrix
          .get_row_unchecked(pointers.try_into().expect("two CSR pointers"))
          .map(|(value, column)| table_x[row] * table_y[*column] * value)
          .sum::<E::Scalar>()
      })
      .sum::<E::Scalar>()
  };
  [evaluate(&shape.A), evaluate(&shape.B), evaluate(&shape.C)]
}
