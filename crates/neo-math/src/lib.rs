#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! neo-math: Fq (Goldilocks), K=F_{q^2}, R_q = F_q[X]/(Phi_eta), cf/cf^{-1}, and S-action.

pub mod balanced;
pub mod field;
pub mod ring;
pub mod s_action;
pub mod signed_sums;

pub use field::{from_complex, Fq, KExtensions, K};
pub use ring::{cf, cf_inv, ct, superneo_bar_block, superneo_bar_matrix, superneo_bar_vec, Rq, D, ETA};
pub use s_action::SAction;

pub use Fq as F;
