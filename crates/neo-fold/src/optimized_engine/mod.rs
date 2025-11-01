//! Optimized engine implementation for Π_CCS
//!
//! This module contains the optimized implementation of the CCS reduction protocol.
//! It has been refactored from the original `pi_ccs` module structure to allow
//! for better organization and testing against the paper-exact reference implementation.

pub mod context;         
pub mod transcript;      
pub mod precompute;      
pub mod checks;          
pub mod terminal;        
pub mod outputs;         
pub mod sumcheck_driver; 
pub mod oracle;          
pub mod transcript_replay; 

pub mod nc_core;         
pub mod nc_constraints;  
pub mod sparse_matrix;   
pub mod eq_weights;      

// Re-export commonly used public items
pub use oracle::GenericCcsOracle;
pub use transcript::Challenges;
pub use sparse_matrix::{Csr, to_csr};
pub use transcript_replay::{
    TranscriptTail,
    pi_ccs_derive_transcript_tail,
    pi_ccs_derive_transcript_tail_with_me_inputs,
    pi_ccs_derive_transcript_tail_with_me_inputs_and_label,
    pi_ccs_derive_transcript_tail_from_bound_transcript,
    pi_ccs_compute_terminal_claim_r1cs_or_ccs,
};

