"""Versioned ownership map for CUDA NVTX stage labels."""

TAXONOMY_VERSION = "gpuscope-taxonomy-v1"

STAGE_ORDER = """
session session.device session.kernels session.params session.structure session.buffers
fold fold.ingest fold.ingest.layout fold.ingest.fresh fold.ingest.running
fold.commit fold.commit.fresh fold.superneo fold.superneo.pi_ccs fold.superneo.pi_ccs.bind
fold.superneo.pi_ccs.challenge_alpha_gamma fold.superneo.pi_ccs.oracle
fold.superneo.pi_ccs.oracle.F fold.superneo.pi_ccs.oracle.NC fold.superneo.pi_ccs.oracle.Eval
fold.superneo.pi_ccs.oracle.upload fold.superneo.pi_ccs.oracle.Q
fold.superneo.pi_ccs.sumcheck fold.superneo.pi_ccs.sumcheck.fe fold.superneo.pi_ccs.sumcheck.nc
fold.superneo.pi_ccs.output fold.superneo.pi_ccs.output.y_prime
fold.superneo.pi_ccs.output.claims fold.superneo.pi_ccs.terminal_check
fold.superneo.pi_rlc fold.superneo.pi_rlc.challenge_rhos fold.superneo.pi_rlc.combine_claims
fold.superneo.pi_rlc.validate_in fold.superneo.pi_rlc.validate_cmb
fold.superneo.pi_rlc.mix_witness fold.superneo.pi_rlc.output fold.superneo.pi_rlc.output.X
fold.superneo.pi_rlc.output.k_surfaces fold.superneo.pi_rlc.output.k_surfaces.device
fold.superneo.pi_rlc.output.k_surfaces.host_claims fold.superneo.pi_rlc.output.y_ring
fold.superneo.pi_rlc.output.y_zcol fold.superneo.pi_dec
fold.superneo.pi_dec.split fold.superneo.pi_dec.commit_children fold.superneo.pi_dec.open_children
fold.superneo.pi_dec.open_children.forms fold.superneo.pi_dec.open_children.y_ring
fold.superneo.pi_dec.open_children.y_zcol fold.superneo.pi_dec.check_recompose
fold.superneo.pi_dec.emit fold.superneo.pi_dec.emit.download fold.superneo.pi_dec.emit.planes
fold.superneo.pi_dec.emit.assemble fold.superneo.pi_dec.emit.public_x fold.accumulate
fold.accumulate.running fold.accumulate.parent_authority fold.egress
fold.egress.retain_planes fold.egress.export finalize finalize.terminal_fold
finalize.proof_export
""".split()

KNOWN_STAGES = set(STAGE_ORDER)

SOURCE_ROWS = [
    ("session", "session.rs", ""),
    ("fold.ingest", "ingest.rs", ""),
    ("fold.commit", "commit.rs", "ajtai.rs"),
    ("fold.superneo.pi_ccs", "reduce/ccs/mod.rs", "pi_ccs_fe.rs pi_ccs_nc.rs csr.rs"),
    ("fold.superneo.pi_ccs.sumcheck.fe", "reduce/ccs/fe.rs", "pi_ccs_fe.rs"),
    ("fold.superneo.pi_ccs.sumcheck.nc", "reduce/ccs/nc.rs", "pi_ccs_nc.rs"),
    ("fold.superneo.pi_rlc", "reduce/rlc.rs", "goldilocks.rs"),
    ("fold.superneo.pi_rlc.output.X", "reduce/rlc.rs", "pi_rlc.rs"),
    ("fold.superneo.pi_rlc.output.k_surfaces", "reduce/rlc.rs", "pi_rlc.rs"),
    ("fold.superneo.pi_rlc.output.k_surfaces.device", "reduce/rlc.rs", "pi_rlc.rs"),
    ("fold.superneo.pi_rlc.output.k_surfaces.host_claims", "reduce/rlc.rs", "pi_rlc.rs"),
    ("fold.superneo.pi_rlc.output.y_ring", "reduce/rlc.rs", "pi_rlc.rs"),
    ("fold.superneo.pi_rlc.output.y_zcol", "reduce/rlc.rs", "pi_rlc.rs"),
    ("fold.superneo.pi_dec", "reduce/dec.rs", "pi_dec.rs ajtai.rs"),
    ("fold.superneo.pi_dec.open_children.forms", "ring_forms.rs", "csr.rs"),
    ("fold.accumulate", "adapter.rs", ""),
    ("fold.egress", "adapter.rs", ""),
    ("finalize", "adapter.rs", ""),
]

SOURCE_MAP = {
    stage: {
        "host": [f"crates/neo-prover-cuda/src/{path}" for path in host.split()],
        "kernels": [f"crates/neo-prover-cuda/src/kernels/{path}" for path in kernels.split()],
    }
    for stage, host, kernels in SOURCE_ROWS
}


def ancestors(stage_id):
    parts = stage_id.split(".")
    return [".".join(parts[:depth]) for depth in range(1, len(parts) + 1)]


def ordered_stage_ids(stage_ids):
    seen = set(stage_ids)
    ordered = [stage_id for stage_id in STAGE_ORDER if stage_id in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def source_for(stage_id):
    parts = ancestors(stage_id)
    for candidate in reversed(parts):
        if candidate in SOURCE_MAP:
            return SOURCE_MAP[candidate]
    return {"host": [], "kernels": []}
