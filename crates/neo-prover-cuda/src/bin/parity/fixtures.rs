//! Shared fixtures for the parity gates.
//!
//! Owns: deterministic sampling, the ring-matrix packing used by fresh
//! instances, the preprocessed identity-CCS fixture both provers run
//! against, and the construction of an internally consistent parent CE
//! claim for Π_DEC. Owns no protocol semantics — everything here delegates
//! to the canonical crates.

use neo_ccs::traits::SModuleHomomorphism as _;
use neo_ccs::{CcsMatrix, CcsStructure, Mat, SparsePoly};
use neo_fold_clean::{config, preprocess, CeClaim, Preprocessing, Structure};
use neo_math::{from_complex, D, F, K};
use neo_prover_cuda::kernels::goldilocks::GOLDILOCKS_MODULUS;
use p3_field::PrimeCharacteristicRing;
use rand::rngs::StdRng;
use rand::Rng;

pub use neo_prover_cuda::ring_layout::assignment_to_mat as pack_ring_matrix;

/// Run `f`, returning its result and wall time in milliseconds.
pub fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = std::time::Instant::now();
    let out = f();
    (out, start.elapsed().as_secs_f64() * 1e3)
}

pub fn rand_f(rng: &mut StdRng) -> F {
    F::from_u64(rng.random::<u64>() % GOLDILOCKS_MODULUS)
}

pub fn rand_k(rng: &mut StdRng) -> K {
    from_complex(rand_f(rng), rand_f(rng))
}

/// Uniform balanced field element with |v| < `bound`.
pub fn rand_bounded(rng: &mut StdRng, bound: u64) -> F {
    let value = F::from_u64(rng.random::<u64>() % bound);
    if rng.random::<bool>() {
        -value
    } else {
        value
    }
}

/// Install the deterministic seeded global Ajtai PP for `(kappa, cols)` if
/// this process has not done so yet.
pub fn install_seeded_global_pp(kappa: usize, cols: usize) {
    if neo_ajtai::has_global_pp_for_dims(D, cols) {
        return;
    }
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&0x4e45_4f46_4f4c_4431_u64.to_le_bytes());
    neo_ajtai::set_global_pp_seeded(D, kappa, cols, seed).expect("seeded Ajtai global setup");
}

/// The smallest real `Preprocessing` the gates can drive both provers
/// through: an identity CCS over n variables with production-core params
/// and a seeded global Ajtai PP.
pub struct Fixture {
    pub prep: Preprocessing,
    pub m_in: usize,
}

impl Fixture {
    /// `t` sparse identity matrices over `n = m` variables with a zero CCS
    /// polynomial. Sparse construction, so real workload sizes (m ≈ 450k)
    /// stay cheap to build.
    pub fn identity_ccs(n: usize, t: usize, m_in: usize) -> Self {
        Self::from_sparse_identity(n, t, SparsePoly::new(t, vec![]), m_in)
    }

    /// Three identity matrices with the R1CS polynomial `X0·X1 − X2`, for
    /// gates that need a nonzero `f` (the Π_CCS FE channel).
    pub fn r1cs_identity(n: usize, m_in: usize) -> Self {
        let f = SparsePoly::new(
            3,
            vec![
                neo_ccs::Term {
                    coeff: F::ONE,
                    exps: vec![1, 1, 0],
                },
                neo_ccs::Term {
                    coeff: F::ZERO - F::ONE,
                    exps: vec![0, 0, 1],
                },
            ],
        );
        Self::from_sparse_identity(n, 3, f, m_in)
    }

    fn from_sparse_identity(n: usize, t: usize, f: SparsePoly<F>, m_in: usize) -> Self {
        let matrices = vec![CcsMatrix::Identity { n }; t];
        let structure: Structure = CcsStructure::new_sparse(matrices, f).expect("identity CCS structure");
        let params = config::r1cs_params(structure.n, structure.m).expect("production-core params");
        install_seeded_global_pp(params.kappa() as usize, structure.m.div_ceil(D));
        let prep = preprocess(params, structure, Some(m_in)).expect("preprocess");
        Self { prep, m_in }
    }

    pub fn structure(&self) -> &Structure {
        self.prep.structure()
    }

    /// A random low-norm assignment (‖z‖∞ < b), valid fresh-instance input.
    pub fn low_norm_assignment(&self, rng: &mut StdRng) -> Vec<F> {
        let b = self.prep.params.b() as u64;
        (0..self.structure().m)
            .map(|_| rand_bounded(rng, b))
            .collect()
    }

    /// A fresh MCS witness built through the canonical constructor.
    pub fn fresh_witness(&self, rng: &mut StdRng) -> neo_fold_clean::CcsWitness {
        let z = self.low_norm_assignment(rng);
        self.instance_for(&z).witness
    }

    /// A fresh instance whose assignment is 0/1 — the satisfying witnesses
    /// of the `r1cs_identity` structure (z·z − z = 0 per row), as a full
    /// prove requires.
    pub fn satisfying_binary_instance(&self, rng: &mut StdRng) -> neo_fold_clean::CcsInstance {
        let z: Vec<F> = (0..self.structure().m)
            .map(|_| if rng.random::<bool>() { F::ONE } else { F::ZERO })
            .collect();
        self.instance_for(&z)
    }

    fn instance_for(&self, z: &[F]) -> neo_fold_clean::CcsInstance {
        neo_fold_clean::CcsInstance::from_low_norm_assignment(
            &self.prep.params,
            &self.prep.log,
            self.structure(),
            z,
            self.m_in,
        )
        .expect("fixture fresh instance")
    }

    /// A random parent witness for Π_DEC (‖Z‖∞ < b^k), in ring-matrix form.
    pub fn dec_parent_witness(&self, rng: &mut StdRng) -> Mat<F> {
        let k = self.prep.params.k_rho() as u32;
        let big_b = (self.prep.params.b() as u64).pow(k);
        let z: Vec<F> = (0..self.structure().m)
            .map(|_| rand_bounded(rng, big_b))
            .collect();
        pack_ring_matrix(&z, self.structure().m.div_ceil(D))
    }

    /// A parent CE claim honestly derived from `witness` at random fold
    /// points (row challenge + NC column channel), so that Π_DEC's
    /// reconstruction self-checks hold on both provers.
    pub fn consistent_parent_claim(&self, witness: &Mat<F>, rng: &mut StdRng) -> CeClaim {
        let s = self.structure();
        let ell_n = s.n.next_power_of_two().max(2).trailing_zeros() as usize;
        let ell_m = s.m.next_power_of_two().max(2).trailing_zeros() as usize;
        let r: Vec<K> = (0..ell_n).map(|_| rand_k(rng)).collect();
        let s_col: Vec<K> = (0..ell_m).map(|_| rand_k(rng)).collect();
        self.consistent_parent_claim_at(witness, &r, &s_col)
    }

    pub fn consistent_parent_claim_at(&self, witness: &Mat<F>, r: &[K], s_col: &[K]) -> CeClaim {
        let s = self.structure();
        let params = self.prep.params.inner();
        let d_pad = D.next_power_of_two();

        let chi_r = neo_ccs::utils::tensor_point_parallel::<K>(&r);
        let n_eff = core::cmp::min(s.n, 1usize << r.len());
        let forms = self
            .prep
            .optimized_cache()
            .superneo()
            .build_ring_linear_forms(&chi_r, n_eff);
        let z_blocks =
            neo_reductions::superneo_eval::SuperneoZBlocks::from_witness_mat(witness, s.m).expect("witness blocks");
        let y_ring: Vec<Vec<K>> =
            neo_reductions::superneo_eval::eval_ring_linear_forms_real_z_blocks(&forms, &z_blocks)
                .into_iter()
                .map(|coeffs| {
                    let mut row = coeffs.to_vec();
                    row.resize(d_pad, K::ZERO);
                    row
                })
                .collect();
        let ct = neo_reductions::common::ct_from_y_ring_for_ccs_m(&y_ring, params, s.m);

        let chi_s = neo_ccs::utils::tensor_point_parallel::<K>(&s_col);
        let y_zcol = neo_reductions::common::compute_y_zcol_from_witness(params, witness, s.m, &chi_s, d_pad)
            .expect("parent y_zcol");

        CeClaim {
            c_step_coords: vec![],
            u_offset: 0,
            u_len: 0,
            c: self.prep.log.commit(witness),
            X: neo_reductions::common::project_x_from_witness_mat(witness, s.m, self.m_in).expect("parent X"),
            r: r.to_vec(),
            s_col: s_col.to_vec(),
            y_ring,
            ct,
            aux_openings: vec![],
            y_zcol,
            m_in: self.m_in,
            fold_digest: [0u8; 32],
        }
    }
}
