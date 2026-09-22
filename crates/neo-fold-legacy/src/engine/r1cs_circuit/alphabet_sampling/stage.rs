//! Stable cost tree for Π_RLC transcript-derived alphabet sampling.
//!
//! Owns: protocol → phase → constraint-family paths and their immediate-child
//! hierarchy.
//!
//! Does not own: row emission, transcript authority, constraint semantics,
//! measured totals, or permission to remove rows.
//!
//! Emits constraints: no.
//!
//! Authority boundary: these paths are diagnostic metadata. The production
//! stage trace and decoded rows remain the physical evidence; Lean refinement
//! owns mathematical meaning separately.
//!
//! | Child | Mathematical obligation | Rust owner | Lean owner |
//! |---|---|---|---|
//! | `transcript` | bind Π_CCS output, domain-separate each rho, run eight digests, and derive exact candidates | `digest_rounds` and NIFS orchestration | `PiRlcChallenge.Transcript` |
//! | `sampler.chunk.accept.packed` | classify the rejected 65535 candidate | `gadget_native::acceptance` | `Sampler.Chunk.Acceptance.Aggregate` |
//! | `sampler.chunk.mod5.packed` | map accepted 16-bit candidates to centered Mod-5 symbols | `gadget_native::mod5` | `Sampler.Chunk.Mod5` |
//! | `sampler.acceptance_bound` | prove at least 54 of 64 candidates accept | `acceptance` | `Sampler.Selection.Acceptance` |
//! | `sampler.selection` | select exactly the first 54 accepted symbols | `selection` | `Sampler.Selection` |

pub const CHALLENGE: &str = "nifs.pi_rlc.challenge";

pub const TRANSCRIPT: &str = "nifs.pi_rlc.challenge.transcript";
pub const BIND_OUTPUTS_DIGEST: &str = "nifs.pi_rlc.challenge.transcript.bind_outputs_digest";
pub const RHO_DOMAIN_SEPARATOR: &str = "nifs.pi_rlc.challenge.transcript.rho_domain_separator";
pub const TRANSCRIPT_DIGEST: &str = "nifs.pi_rlc.challenge.transcript.digest_rounds";
pub const LANE_BIT_DECOMPOSITION: &str = "nifs.pi_rlc.challenge.transcript.lane_bit_decomposition";

pub const SAMPLER: &str = "nifs.pi_rlc.challenge.sampler";
pub const SAMPLE_INITIALIZE: &str = "nifs.pi_rlc.challenge.sampler.initialize";
pub const CHUNK: &str = "nifs.pi_rlc.challenge.sampler.chunk";
pub const CHUNK_ACCEPT: &str = "nifs.pi_rlc.challenge.sampler.chunk.accept";
pub const CHUNK_ACCEPT_PACKED: &str = "nifs.pi_rlc.challenge.sampler.chunk.accept.packed";
pub const ACCEPT_TREE_BIT_PAIRS: &str = "nifs.pi_rlc.challenge.sampler.chunk.accept.packed.tree_bit_pairs";
pub const ACCEPT_PRODUCT_AGGREGATE: &str = "nifs.pi_rlc.challenge.sampler.chunk.accept.packed.product_aggregate";
pub const ACCEPT_ROOT_BINDING: &str = "nifs.pi_rlc.challenge.sampler.chunk.accept.packed.root_binding";
pub const CHUNK_MOD5: &str = "nifs.pi_rlc.challenge.sampler.chunk.mod5";
pub const CHUNK_MOD5_PACKED: &str = "nifs.pi_rlc.challenge.sampler.chunk.mod5.packed";
pub const LOW_BIT_PAIRS: &str = "nifs.pi_rlc.challenge.sampler.chunk.mod5.packed.low_bit_pairs";
pub const HIGH_BIT_PAIR: &str = "nifs.pi_rlc.challenge.sampler.chunk.mod5.packed.high_bit_pair";
pub const RESIDUE_PAIR: &str = "nifs.pi_rlc.challenge.sampler.chunk.mod5.packed.residue_pair";
pub const CHUNK_SYMBOL_AND_PREFIX: &str = "nifs.pi_rlc.challenge.sampler.chunk.symbol_and_prefix";
pub const ACCEPTANCE_BOUND: &str = "nifs.pi_rlc.challenge.sampler.acceptance_bound";
pub const SELECTION: &str = "nifs.pi_rlc.challenge.sampler.selection";
pub const SELECT_INITIALIZE: &str = "nifs.pi_rlc.challenge.sampler.selection.initialize";
pub const SELECT_ONE_HOT: &str = "nifs.pi_rlc.challenge.sampler.selection.one_hot";
pub const SELECT_PRODUCTS: &str = "nifs.pi_rlc.challenge.sampler.selection.products";
pub const SELECT_BIND: &str = "nifs.pi_rlc.challenge.sampler.selection.bind";
pub const SELECT_BIND_ACCEPT: &str = "nifs.pi_rlc.challenge.sampler.selection.bind.accept";
pub const SELECT_BIND_PREFIX: &str = "nifs.pi_rlc.challenge.sampler.selection.bind.prefix";
pub const SELECT_BIND_SYMBOL: &str = "nifs.pi_rlc.challenge.sampler.selection.bind.symbol";

/// Every protocol, phase, and constraint-family node in the challenge tree.
pub const ALL: &[&str] = &[
    CHALLENGE,
    TRANSCRIPT,
    BIND_OUTPUTS_DIGEST,
    RHO_DOMAIN_SEPARATOR,
    TRANSCRIPT_DIGEST,
    LANE_BIT_DECOMPOSITION,
    SAMPLER,
    SAMPLE_INITIALIZE,
    CHUNK,
    CHUNK_ACCEPT,
    CHUNK_ACCEPT_PACKED,
    ACCEPT_TREE_BIT_PAIRS,
    ACCEPT_PRODUCT_AGGREGATE,
    ACCEPT_ROOT_BINDING,
    CHUNK_MOD5,
    CHUNK_MOD5_PACKED,
    LOW_BIT_PAIRS,
    HIGH_BIT_PAIR,
    RESIDUE_PAIR,
    CHUNK_SYMBOL_AND_PREFIX,
    ACCEPTANCE_BOUND,
    SELECTION,
    SELECT_INITIALIZE,
    SELECT_ONE_HOT,
    SELECT_PRODUCTS,
    SELECT_BIND,
    SELECT_BIND_ACCEPT,
    SELECT_BIND_PREFIX,
    SELECT_BIND_SYMBOL,
];

/// Immediate-child ownership used by the exact cost audit.
pub const HIERARCHY: &[(&str, &[&str])] = &[
    (CHALLENGE, &[TRANSCRIPT, SAMPLER]),
    (
        TRANSCRIPT,
        &[
            BIND_OUTPUTS_DIGEST,
            RHO_DOMAIN_SEPARATOR,
            TRANSCRIPT_DIGEST,
            LANE_BIT_DECOMPOSITION,
        ],
    ),
    (SAMPLER, &[SAMPLE_INITIALIZE, CHUNK, ACCEPTANCE_BOUND, SELECTION]),
    (CHUNK, &[CHUNK_ACCEPT, CHUNK_MOD5, CHUNK_SYMBOL_AND_PREFIX]),
    (CHUNK_ACCEPT, &[CHUNK_ACCEPT_PACKED]),
    (
        CHUNK_ACCEPT_PACKED,
        &[ACCEPT_TREE_BIT_PAIRS, ACCEPT_PRODUCT_AGGREGATE, ACCEPT_ROOT_BINDING],
    ),
    (CHUNK_MOD5, &[CHUNK_MOD5_PACKED]),
    (CHUNK_MOD5_PACKED, &[LOW_BIT_PAIRS, HIGH_BIT_PAIR, RESIDUE_PAIR]),
    (
        SELECTION,
        &[SELECT_INITIALIZE, SELECT_ONE_HOT, SELECT_PRODUCTS, SELECT_BIND],
    ),
    (
        SELECT_BIND,
        &[SELECT_BIND_ACCEPT, SELECT_BIND_PREFIX, SELECT_BIND_SYMBOL],
    ),
];
