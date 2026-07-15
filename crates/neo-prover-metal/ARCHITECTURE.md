# Metal prover architecture

## Scope and authority

`neo-prover-metal` accelerates the prover-side `NifsProverAdapter` path on Apple GPUs. It owns device resources, resident prover state, and phase execution. It does not own transcript order, claim algebra, proof formats, or verification; those remain in `neo-fold-clean`.

A Metal result is never verifier authority by itself. Deferred carriers eventually materialize an ordinary `NifsProof`, and the canonical verifier recomputes protocol-binding digests from claims.

## Public entry points

| Entry point | Location | Responsibility |
|---|---|---|
| `MetalNifsProver::new` | `src/adapter.rs` | Create one prover and its device session. |
| `MetalNifsProver::prepare_static` | `src/adapter.rs` | Prepare reusable Ajtai, lane, FE-oracle, and DEC-form plans. |
| `NifsProverAdapter::build_fresh_instances` | `src/adapter.rs` | Pack low-norm witnesses and construct canonical fresh claims. |
| `NifsProverAdapter::prove` | `src/adapter.rs` | Delegate one online fold to the phase flow. |
| `MetalNifsProver::last_profile` | `src/adapter.rs` | Expose measurements and routing decisions for the last fold. |
| `MetalSession` primitive methods | `src/session.rs` | Run arithmetic and hashing primitives used by tests and higher layers. |

## Module ownership

| Module | Owns | Does not own |
|---|---|---|
| `src/lib.rs` | Public types, errors, target selection | Device implementation details |
| `src/adapter.rs` | Adapter state, static-plan caching, fresh-instance construction | Online phase internals |
| `src/adapter/fold.rs` | Π_CCS → Π_RLC → Π_DEC ordering and handoffs | Kernel encoding or proof semantics |
| `src/adapter/dec.rs` | DEC projection, child materialization, resident child claims | Generic DEC buffer planning |
| `src/adapter/profile.rs` | Stable per-phase measurements | Profiling policy |
| `src/sumcheck/fe.rs` | FE backend and canonical fallback | NC state |
| `src/sumcheck/nc.rs` | NC backend, mask-native execution, canonical fallback | FE state |
| `src/sumcheck/mask_residency.rs` | Selection of host values versus signed masks | Sumcheck arithmetic |
| `src/session.rs` | Device, queues, pipelines, allocation, submission | Protocol phase ordering |
| `src/session/resident.rs` | Shared witness masks and transcript buffers | FE/NC-specific command plans |
| `src/session/resident/fe.rs` | FE resident plans and command encoding | NC plans |
| `src/session/resident/nc.rs` | NC resident plans and command encoding | FE plans |
| `src/session/dec/forms.rs` | Static and per-fold ring-form construction | Child witness lifecycle |
| `src/session/dec/split.rs` | Split, projection, validation, and child residency | Claim construction |
| `src/session/carrier.rs` | Device-resident running-child ownership | Public proof carrier |
| `src/fold_output.rs` | Shared deferred proof/running materialization state | GPU buffers |
| `src/unsupported.rs` | Type-compatible unavailable implementation | Apple execution |
| `shaders/*.metal` | Arithmetic kernels | Scheduling and protocol decisions |

## Online fold data flow

```text
fresh assignments
  └─ adapter: signed-unit masks + canonical commitments
       └─ Π_CCS
            FE: resident row state / Ajtai Y evaluation
            NC: the same witness masks, compact column state
            canonical transcript and proof assembly in neo-fold-clean
             └─ Π_RLC
                  resident random linear combination
                  canonical SIS/Ajtai + Poseidon2 projection digest
                  canonical Poseidon2 accumulator_ce_claim_digest
                   └─ Π_DEC
                        resident base-2 split
                        range + recomposition validation
                        child projections + commitments
                         └─ shared MetalFoldOutput
                              consistent Pi_RLC parent and Pi_DEC children
                              session ownership + generation id
                              deferred canonical proof
                              deferred resident running carrier
                              verifier-visible post-fold summary
```

The short orchestration is in `src/adapter/fold.rs`. Named phase results make ownership explicit: Π_CCS passes its NC backend into Π_RLC so shared masks remain usable, Π_RLC passes one resident mixed witness into Π_DEC, and Π_DEC returns the next running generation.

## Residency and transfers

The next Pi_CCS call uses virtual host witness shells for shape and reads the
carried assignment from session-owned resident sign masks. Fresh inputs remain
host-supplied. The backend-owned child witness planes remain resident for the
next recursive fold.

- `MetalSession` owns buffers and recycling slots; protocol code handles opaque plans rather than raw Metal objects.
- Fresh signed-unit masks are uploaded once and reused by FE, NC, Π_RLC, and lane commitments when shapes match.
- Π_RLC returns a resident witness, not a host matrix.
- Π_DEC retains child witness planes through carrier egress and returns an opaque generation id that the next fold can reuse.
- Public proof surfaces materialize at egress; private running witnesses materialize only at an explicit CPU or audit boundary. Cross-session, stale, and shape-mismatched carriers fail closed before virtual witness shells enter proving.
- Ajtai commitment plans are reused only when dimensions and the exact seed or materialized setup identity match.
- Activity counters report allocation, upload, download, dispatch, command-buffer, and wait totals.

## Authority and cache identity

The verifier never trusts a Metal-only digest, buffer, or generation id.
Every Metal-produced proof materializes as an ordinary `NifsProof` and is
checked by the canonical CPU verifier. The resident accumulator shortcut
carries exactly the Poseidon2 `accumulator_ce_claim_digest` of its Pi_RLC
parent; the SIS-backed `ce_claim_digest` is never substituted in that slot. The
post-fold summary derives from the same canonical parent digest. The v1 accumulator
CE-core digest omits `y_zcol`, and Metal residency does not close the known
old-point authority gap. Binding that delayed parent projection into
accumulator authority remains open.

The proof and running carriers share one materialization state with a
consistent Pi_RLC parent and Pi_DEC children. The proof reconstructs those
surfaces from the shared object rather than retaining duplicate claims; the
canonical NIFS verifier subsequently authenticates them. `MetalSession` owns
the current reusable device generation, while `MetalFoldOutput` retains an
immutable mask snapshot for explicit materialization. A stale or branched
generation cannot select the wrong buffer: it is rejected before the virtual
witness shells enter the prover. Terminal finalization consumes the same
carrier before materializing the final protocol object.

Fresh Ajtai plans bind dimensions and kappa to the exact setup seed or
materialized public-parameter allocation. FE and DEC plans bind cache identity
and matrix digest. Lane plans bind seeds and ranges, are prepared even for an
empty fresh batch, and are revalidated at Pi_DEC entry.

The primary and independent Metal queues are session-owned. Queue choice stays inside the device layer. Parallel proof chains use independent sessions; the two-chain WASM benchmark is the current integration check.

## Build and target selection

`build.rs` selects the Apple SDK from Cargo's target, compiles the shader entry source with Metal 3.0, links `nightstream-metal.metallib`, and emits canonical Poseidon2 constants. Apple builds with the Cargo `metal` feature and compiled shaders use `session`; feature-disabled or shaderless builds use `unsupported` while preserving the Rust API surface. Enabling `metal` on a non-Apple target is rejected at compile time.

There are no crate-owned runtime flags or environment variables. Xcode selection follows `xcode-select`, then the standard Xcode application path.

## Tests and performance gates

| Check | Location | Protects |
|---|---|---|
| Arithmetic parity | `tests/metal_arithmetic.rs` | Goldilocks, extension, Poseidon2, and primitive execution |
| NIFS parity and tamper rejection | `tests/nifs_adapter.rs` | Exact CPU/Metal authority and reduction-log parity, residency, canonical rejection |
| Compact NC reference checks | `tests/nc_compact_window.rs` | Window folding and signed-mask equivalence |
| Projection-digest parity | `tests/sis_digest.rs` | Metal SIS/Ajtai + Poseidon2 projection digest matches the canonical implementation |
| Lifecycle crossover/sustained gate | `../neo-prover-metal-bench/tests/benchmark_contract.rs` | End-to-end performance with proof validity |
| Authentic WASM + Nebula diagnostic | `../neo-prover-metal-bench/tests/wasm_nebula.rs` | Ignored integration benchmark from Wasmtime trace through Nebula proving and canonical verification |

Performance numbers are deliberately not embedded here because they are machine- and revision-specific. The benchmark JSON and raw ordered samples are the source of truth.

## Profiling workflow

Use the repository scripts under `scripts/`:

- `profile_for_ai.sh` for a quick CPU sample.
- `profile_xctrace.sh` with Metal System Trace, GPU Counters, or Allocations for CPU/GPU scheduling and transfer evidence.
- `profile_memory_deep.sh` for host allocation sites.

A performance change is complete only when exact proof parity still passes and a trace explains the timing change. Profile fields in `src/adapter/profile.rs` record which path actually ran; they should be checked alongside wall time so a fallback cannot masquerade as a GPU speedup.
