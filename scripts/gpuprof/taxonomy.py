NUMERIC_COLS = """
gpu_ms launches h2d_mb h2d_ms h2d_copies d2h_mb d2h_ms d2h_copies
dtod_mb dtod_ms dtod_copies sync_ms sync_idle_ms syncs memset_mb memset_ms
memset_count idle_ms idle_sync_ms idle_api_ms idle_host_ms api_ms api_calls api_launch_ms api_launch_calls api_memcpy_h2d_ms
api_memcpy_h2d_calls api_memcpy_d2h_ms api_memcpy_d2h_calls api_memcpy_d2d_ms
api_memcpy_d2d_calls api_memset_ms api_memset_calls api_sync_ms api_sync_calls
api_memalloc_ms api_memalloc_calls api_memfree_ms api_memfree_calls
api_module_load_ms api_module_load_calls
""".split()

STAGE_ORDER = """
session session.device session.kernels session.params session.structure session.buffers
fold fold.ingest fold.ingest.layout fold.ingest.fresh fold.ingest.running
fold.commit fold.commit.fresh fold.superneo fold.superneo.pi_ccs fold.superneo.pi_ccs.bind
fold.superneo.pi_ccs.challenge_alpha_gamma fold.superneo.pi_ccs.oracle
fold.superneo.pi_ccs.oracle.F fold.superneo.pi_ccs.oracle.NC fold.superneo.pi_ccs.oracle.Eval
fold.superneo.pi_ccs.oracle.upload fold.superneo.pi_ccs.oracle.Q
fold.superneo.pi_ccs.sumcheck fold.superneo.pi_ccs.sumcheck.fe
fold.superneo.pi_ccs.sumcheck.fe.row_enqueue_loop
fold.superneo.pi_ccs.sumcheck.fe.row_download fold.superneo.pi_ccs.sumcheck.fe.row_decode
fold.superneo.pi_ccs.sumcheck.nc
fold.superneo.pi_ccs.output fold.superneo.pi_ccs.output.y_prime
fold.superneo.pi_ccs.output.claims fold.superneo.pi_ccs.terminal_check
fold.superneo.pi_rlc fold.superneo.pi_rlc.challenge_rhos fold.superneo.pi_rlc.combine_claims
fold.superneo.pi_rlc.validate_in fold.superneo.pi_rlc.validate_cmb
fold.superneo.pi_rlc.commit_mix fold.superneo.pi_rlc.claim_shell
    fold.superneo.pi_rlc.mix_witness fold.superneo.pi_rlc.output fold.superneo.pi_rlc.output.X
    fold.superneo.pi_rlc.output.k_surfaces fold.superneo.pi_rlc.output.k_surfaces.device
    fold.superneo.pi_rlc.output.k_surfaces.host_claims fold.superneo.pi_rlc.output.y_ring
    fold.superneo.pi_rlc.output.y_zcol fold.superneo.pi_dec
fold.superneo.pi_dec.split fold.superneo.pi_dec.commit_children fold.superneo.pi_dec.open_children
fold.superneo.pi_dec.open_children.forms fold.superneo.pi_dec.open_children.y_ring
fold.superneo.pi_dec.open_children.y_zcol fold.superneo.pi_dec.check_recompose
fold.superneo.pi_dec.emit fold.superneo.pi_dec.emit.download fold.superneo.pi_dec.emit.planes
fold.superneo.pi_dec.emit.assemble fold.superneo.pi_dec.emit.public_x fold.accumulate fold.accumulate.running
fold.accumulate.parent_authority fold.egress fold.egress.retain_planes fold.egress.export
finalize finalize.terminal_fold finalize.proof_export
""".split()

SOURCE_ROWS = [
    ("session", "session.rs", ""),
    ("fold.ingest", "ingest.rs", ""),
    ("fold.commit", "commit.rs", "ajtai.rs"),
    ("fold.superneo.pi_ccs", "reduce/ccs/mod.rs", "pi_ccs_fe.rs pi_ccs_nc.rs csr.rs"),
    ("fold.superneo.pi_ccs.sumcheck.fe", "reduce/ccs/fe.rs", "pi_ccs_fe.rs"),
    ("fold.superneo.pi_ccs.sumcheck.fe.row_enqueue_loop", "reduce/ccs/fe.rs", "pi_ccs_fe.rs"),
    ("fold.superneo.pi_ccs.sumcheck.fe.row_download", "reduce/ccs/fe.rs", ""),
    ("fold.superneo.pi_ccs.sumcheck.fe.row_decode", "reduce/ccs/fe.rs", ""),
    ("fold.superneo.pi_ccs.sumcheck.nc", "reduce/ccs/nc.rs", "pi_ccs_nc.rs"),
    ("fold.superneo.pi_rlc", "reduce/rlc.rs", "goldilocks.rs"),
    ("fold.superneo.pi_rlc.commit_mix", "adapter.rs", ""),
    ("fold.superneo.pi_rlc.claim_shell", "reduce/rlc.rs", ""),
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
        "host": [f"crates/neo-prover-cuda/src/{p}" for p in host.split()],
        **({"kernel": [f"crates/neo-prover-cuda/src/kernels/{p}" for p in ker.split()]} if ker else {}),
    }
    for stage, host, ker in SOURCE_ROWS
}

NCU_NAME_MAP = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "compute_memory_throughput_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "l2_throughput_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__occupancy_limit_registers": "occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem": "occupancy_limit_shared_mem",
    "launch__occupancy_limit_blocks": "occupancy_limit_blocks",
    "launch__shared_mem_per_block_static": "static_shared_memory_per_block",
    "launch__shared_mem_per_block_dynamic": "dynamic_shared_memory_per_block",
    "gpu__time_duration.sum": "kernel_duration_us",
    "derived__local_spilling_requests": "local_spilling_requests",
    "derived__local_spilling_requests_pct": "local_spilling_pct",
    "smsp__sass_average_branch_targets_threads_uniform.pct": "branch_uniformity_pct",
}
