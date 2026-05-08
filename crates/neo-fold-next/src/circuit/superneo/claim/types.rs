use super::*;

/// Circuit representation of one SuperNeo CE claim.
///
/// Paper-facing shape:
/// `CE(b, L) = (s; c, x, r, {y_j}; z)`, with `z` kept as the private
/// witness outside this struct. This type owns the claim-side variables inside
/// the recursive constraint circuit:
/// the commitment `c`, public/input projection `X`/`x`, evaluation point `r`,
/// matrix-output openings `{y_j}`, plus implementation channels used for norm
/// checks, side openings, and step/transcript binding.
///
/// Field convention: for each allocated field `foo`, the sibling `foo_values`
/// is the native assignment used for constants, digest preimages, and
/// shape/projection checks. The native mirror is not a separate authority.
#[derive(Clone)]
pub struct CircuitCeClaim {
    /// Paper commitment `c`.
    pub commitment: CircuitCeCommitment,

    /// Paper public/input projection `x`, represented by Rust as `X = L_x(Z)`.
    pub public_input: CircuitCePublicInput,

    /// Paper evaluation point and output openings.
    pub openings: CircuitCeOpenings,

    /// SuperNeo norm-check channel tying the private packed witness `Z` to the CE claim.
    pub norm_check: CircuitCeNormCheck,

    /// Implementation-side data that binds this claim to one folding step.
    ///
    /// These fields are not the paper CE tuple itself. They carry the step
    /// pre-commitment coordinates, transcript digest encoding, and witness-slice
    /// offsets needed by this circuit's link constraints.
    pub step_binding: CircuitCeStepBinding,
}

impl CircuitCeClaim {
    pub(super) fn from_parts(
        commitment: CircuitCeCommitment,
        public_input: CircuitCePublicInput,
        openings: CircuitCeOpenings,
        norm_check: CircuitCeNormCheck,
        step_binding: CircuitCeStepBinding,
    ) -> Self {
        Self {
            commitment,
            public_input,
            openings,
            norm_check,
            step_binding,
        }
    }
}

#[derive(Clone)]
pub struct CircuitCeCommitment {
    /// Allocated Ajtai commitment coordinates for the paper commitment `c in C`.
    ///
    /// This is the circuit form of `CeClaim.c.data`.
    pub data: Vec<AllocatedNum<SpartanF>>,
    /// Native assignment mirror for `data`.
    pub data_values: Vec<F>,
}

impl CircuitCeCommitment {
    pub(super) fn from_allocated(data: Vec<AllocatedNum<SpartanF>>, data_values: Vec<F>) -> Self {
        Self { data, data_values }
    }

    pub(super) fn values_only(data_values: Vec<F>) -> Self {
        Self::from_allocated(Vec::new(), data_values)
    }
}

#[derive(Clone)]
pub struct CircuitCePublicInput {
    /// Allocated public/input projection surface.
    ///
    /// In the paper this is the CE public input vector `x`. In the Rust CE
    /// model it is stored as `X = L_x(Z)`. Depending on the allocation surface,
    /// this may be the full `D x m_in` matrix or the canonical compact
    /// SuperNeo embedded vector; `rows`, `cols`, and `m_in` describe which.
    pub x: Vec<AllocatedNum<SpartanF>>,
    /// Native assignment mirror for `x`.
    pub x_values: Vec<F>,
    /// Row count for the represented `X` surface, normally `D` for full matrix form.
    pub rows: usize,
    /// Column count for the represented `X` surface, normally `m_in`.
    pub cols: usize,
    /// Number of public/input coordinates, matching paper `n_F,in` and Rust `m_in`.
    pub m_in: usize,
}

impl CircuitCePublicInput {
    pub(super) fn from_claim_parts(
        claim: &CeClaim<Commitment, F, K>,
        x: Vec<AllocatedNum<SpartanF>>,
        x_values: Vec<F>,
    ) -> Self {
        Self {
            x,
            x_values,
            rows: claim.X.rows(),
            cols: claim.X.cols(),
            m_in: claim.m_in,
        }
    }

    pub(super) fn values_only(claim: &CeClaim<Commitment, F, K>, x_values: Vec<F>) -> Self {
        Self::from_claim_parts(claim, Vec::new(), x_values)
    }
}

#[derive(Clone)]
pub struct CircuitCeOpenings {
    /// Allocated CE evaluation point `r`.
    ///
    /// This is the paper's `r` used when checking each matrix-output opening
    /// `y_j = M_j(z)(r)`.
    pub r: Vec<KNumVar>,
    /// Native assignment mirror for `r`.
    pub r_values: Vec<K>,

    /// Allocated matrix-output openings `{y_j}` from the paper CE claim.
    ///
    /// Each row represents one paper opening `y_j in R_K`, stored as a padded
    /// vector of `K` coefficients so the circuit can apply Pi_CCS, Pi_RLC, and
    /// Pi_DEC recomposition checks pointwise.
    pub y_ring: Vec<Vec<KNumVar>>,
    /// Native assignment mirror for `y_ring`.
    pub y_ring_values: Vec<Vec<K>>,

    /// Allocated scalar view of the core `y_ring` openings.
    ///
    /// In the SuperNeo embedding these are the constant-term/scalar openings
    /// associated with the paper `{y_j}` values.
    pub ct: Vec<KNumVar>,
    /// Native assignment mirror for `ct`.
    pub ct_values: Vec<K>,

    /// Allocated additional scalar openings carried with the CE claim.
    ///
    /// These are implementation sidecar openings, not a new paper CE component.
    /// RLC and DEC checks must carry/recompose them with the same binding
    /// discipline as the core CE openings.
    pub aux_openings: Vec<KNumVar>,
    /// Native assignment mirror for `aux_openings`.
    pub aux_openings_values: Vec<K>,
}

impl CircuitCeOpenings {
    pub(super) fn from_claim_parts(
        claim: &CeClaim<Commitment, F, K>,
        r: Vec<KNumVar>,
        r_values: Vec<K>,
        y_ring: Vec<Vec<KNumVar>>,
        ct: Vec<KNumVar>,
        aux_openings: Vec<KNumVar>,
    ) -> Self {
        Self {
            r,
            r_values,
            y_ring,
            y_ring_values: claim.y_ring.clone(),
            ct,
            ct_values: claim.ct.clone(),
            aux_openings,
            aux_openings_values: claim.aux_openings.clone(),
        }
    }

    pub(super) fn y_ring_only(
        claim: &CeClaim<Commitment, F, K>,
        r: Vec<KNumVar>,
        r_values: Vec<K>,
        y_ring: Vec<Vec<KNumVar>>,
    ) -> Self {
        Self::from_claim_parts(claim, r, r_values, y_ring, Vec::new(), Vec::new())
    }

    pub(super) fn point_only(claim: &CeClaim<Commitment, F, K>, r: Vec<KNumVar>, r_values: Vec<K>) -> Self {
        Self::from_claim_parts(claim, r, r_values, Vec::new(), Vec::new(), Vec::new())
    }
}

#[derive(Clone)]
pub struct CircuitCeNormCheck {
    /// Allocated column-domain point for the SuperNeo norm-check channel.
    ///
    /// This is implementation-side CE data used to check `y_zcol = Z * chi(s_col)`.
    /// It may be empty when the NC channel is not part of the allocated surface.
    pub s_col: Vec<KNumVar>,
    /// Native assignment mirror for `s_col`.
    pub s_col_values: Vec<K>,

    /// Allocated norm-check column opening `y_zcol = Z * chi(s_col)`.
    ///
    /// This ties the private CE witness `Z` to the digit-range/NC channel. It
    /// may be empty when that channel is intentionally absent.
    pub y_zcol: Vec<KNumVar>,
    /// Native assignment mirror for `y_zcol`.
    pub y_zcol_values: Vec<K>,
}

impl CircuitCeNormCheck {
    pub(super) fn from_claim_parts(
        claim: &CeClaim<Commitment, F, K>,
        s_col: Vec<KNumVar>,
        s_col_values: Vec<K>,
        y_zcol: Vec<KNumVar>,
    ) -> Self {
        Self {
            s_col,
            s_col_values,
            y_zcol,
            y_zcol_values: claim.y_zcol.clone(),
        }
    }

    pub(super) fn values_only(claim: &CeClaim<Commitment, F, K>) -> Self {
        Self::from_claim_parts(claim, Vec::new(), claim.s_col.clone(), Vec::new())
    }
}

#[derive(Clone)]
pub struct CircuitCeStepBinding {
    /// Allocated pre-commitment coordinates used by step link constraints.
    ///
    /// This is a Rust implementation field (`CeClaim.c_step_coords`), not part
    /// of the minimal paper CE tuple.
    pub c_step_coords: Vec<AllocatedNum<SpartanF>>,
    /// Native assignment mirror for `c_step_coords`.
    pub c_step_coords_values: Vec<F>,

    /// Allocated field encoding of the 32-byte folding transcript digest.
    ///
    /// This binds the CE claim to the Fiat-Shamir/folding transcript. It is a
    /// transcript-binding handle, not standalone proof authority.
    pub fold_digest_encoding: Vec<AllocatedNum<SpartanF>>,
    /// Native assignment mirror for `fold_digest_encoding`.
    pub fold_digest_encoding_values: Vec<SpartanF>,

    /// Offset of the rho-dependent witness slice used by step Pattern-A links.
    pub u_offset: usize,
    /// Length of the rho-dependent witness slice used by step Pattern-A links.
    pub u_len: usize,
}

impl CircuitCeStepBinding {
    pub(super) fn from_claim_parts(
        claim: &CeClaim<Commitment, F, K>,
        c_step_coords: Vec<AllocatedNum<SpartanF>>,
        fold_digest_encoding: Vec<AllocatedNum<SpartanF>>,
        fold_digest_encoding_values: Vec<SpartanF>,
    ) -> Self {
        Self {
            c_step_coords,
            c_step_coords_values: claim.c_step_coords.clone(),
            fold_digest_encoding,
            fold_digest_encoding_values,
            u_offset: claim.u_offset,
            u_len: claim.u_len,
        }
    }

    pub(super) fn without_fold_digest(
        claim: &CeClaim<Commitment, F, K>,
        c_step_coords: Vec<AllocatedNum<SpartanF>>,
    ) -> Self {
        Self::from_claim_parts(
            claim,
            c_step_coords,
            Vec::new(),
            packed_bytes_field_values(&claim.fold_digest),
        )
    }

    pub(super) fn values_only(claim: &CeClaim<Commitment, F, K>) -> Self {
        Self::without_fold_digest(claim, Vec::new())
    }
}
