# Native application row storage review

Read-only source review, 2026-09-21. No build, test, large circuit, or memory measurement.

## Scope and assumptions

The contract is the current public `Circuit` lifecycle with Optimized or Metal, its selected key, and the native sealed-record path. These are necessary input boundaries, not new restrictions. The standalone F-prime loader and a caller's still-live `Affine` expression are different ownership boundaries. The selected profile remains b=2, k_rho=16. No proposed threshold, mode, or feature follows from this review.

## Accepted width

`crates/nightstream/artifacts/shared-verifier-v1.json:1` gives logical width `252,695,531 + 41*(W+L)`, where W is private inputs and L is generated locals. `assembly/manifest.rs:348-365` checks the ring-padded width against 2^28. This alone gives W+L <= 383,899 and V=8+W+L <= 383,907.

The public lifecycle has a tighter existing bound. `crates/nightstream/src/lifecycle/mod.rs:50-51,83-94` requires width <= `PRODUCTION_CARRIER_WIDTH`. `crates/neo-ajtai/src/nightstream_fprime_setup.rs:25-27` defines that as 4,685,394*54 = 253,011,276. Thus successful `Circuit::prepare_with_engine` requires W+L <= floor((253,011,276-252,695,531)/41) = 7,701 and V <= 7,709. This is an upper bound, not a claim that every shape below it is accepted. Assembly can temporarily reach the looser manifest bound before the lifecycle rejects a key-width mismatch.

## Live row representations

For one row let T be the sum of distinct nonzero terms in A/B/C. `package/native_application.rs:174-202` merges duplicate columns in three BTreeMaps and removes zero sums, so T <= 3V. Arbitrarily many raw duplicate terms do not increase this cardinality. They stay on disk and pass through bounded record cursors.

| Phase | Concrete live owners | Element payload at lifecycle bound T<=23,127 |
| --- | --- | --- |
| Normalize | At most three maps, then map nodes being consumed plus completed SparseTerm vectors | At most 23,127 map entries; completed vector payload <= 16T = 370,032 bytes, plus tree nodes/capacity |
| Source conversion | SparseRow and newly copied SourceRow | At most 32T = 740,064 bytes, plus Vec spare capacity |
| Projection | Original SourceRow and projected SourceRow | Two Entry sequences of at most T terms; collection capacity is separate |
| Compile | Projected SourceRow plus growing A/B/C run forms | Entry payload 16T plus run payload <= 32*(T+3) = 740,160 bytes; a Form merge can also keep its old vector while it allocates the replacement |
| MatrixWindow fill | RowForms plus the already allocated destination cache | Source/Entry rows and maps have gone; destination capacity is covered by MatrixWindow's existing workspace count |

Paths: `package/source_row.rs:24-31,158-191`; `package/matrix_program/mod.rs:310-327,506-530,559-582`; `matrix_program/form.rs:12-17,160-212`. The first three rows of this table are separate phases, not simultaneous owners. Form append can copy/reallocate runs, but it never retains all prior rows. Application substitution uses one length-41 geometric run per distinct variable; it does not expand each into 41 entries. Production `sealed.rs:457-479` borrows the run slices. Scalar `Form::entries` expansion is not called on this Optimized/Metal source path. Even the looser manifest bound has only T<=1,151,721, or 18,427,536 bytes per Sparse/Source payload; it does not provide a 16 GB single-row counterexample.

These are element-payload/cardinality counts. They are not exact allocator RSS bounds. This review did not assume a BTreeMap node layout or a Vec growth factor as a project memory policy.

## Finite width copies

`application/builder.rs:60-80,193-211,220-228` keeps W Variable IDs and converts them to shared Arc storage at finish. The conversion can temporarily keep both W-word copies. PreparedApplication has no full generated-column map (`native_application.rs:15-18,53-75`). Two LoadedApplicationPlan owners each retain a W-word witness-column vector: the shared PreparedApplication and `LoadedPerApplicationPackage.application` (`sealed.rs:104,613`). With the ApplicationCircuit IDs this is 24W <= 184,824 persistent bytes on the public lifecycle. Arc clones do not clone rows, recipes, or IDs.

Preparation also has the typed W-word application plan, JSON witness-column values, and transient decode/clone copies (`assembly/application.rs:12-17`, `assembly/mod.rs:73-87`, `sealed.rs:518-565,841-901`). Each has W entries, not rows or raw terms. During extend, the generated witness holds V field words and the private-input conversion holds W words (`circuit.rs:68-78`, `builder.rs:268-287`), at most 8V+8W = 123,280 bytes beyond retained IDs/plan columns. These values can remain live while the lifecycle call runs; no elimination by compiler lifetime shortening is assumed.

## Result and still-open boundary

No source-derived 16 GB counterexample remains in the reviewed single-row reconstruction or witness-column owners. They are bounded by existing accepted width, independent of application row count and raw duplicate-term count. An additional disk-backed normalized row implementation is not necessary on this evidence.

The universal RSS claim is still unproved: `neo-reductions/src/superneo_eval/matrix_window.rs:15-18,93-105` excludes borrowed-source scratch while its filled cache is live, and `nightstream/src/lifecycle/evaluation.rs:67-89` explicitly excludes package, allocator, and driver RSS. The finite row scratch above belongs to that excluded source term. This is an accounting boundary, not a measured failure or a claim that this scratch alone exceeds the 16 GiB guard. Proving total RSS needs this overlap and actual capacities/allocator/driver residency in the owner-level accounting or measurement; no new arbitrary circuit limit follows.
