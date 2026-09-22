# Current native engine memory accounting

Read-only, low-load source review, 2026-09-21. No build, test, proof, profiling, or large trace parsing. The active GPU run was not touched.

Result: no reproducible omitted O(rows)/O(width) allocation or over-limit lower bound was found in the current native Optimized/Metal lifecycle. The reviewed large owners are either part of the existing reserve or live in a separate phase with room under that reserve. This does not supply a new RSS measurement, and it does not justify another storage rewrite.

## Necessary input boundary

This review uses public `Circuit` preparation and locally produced envelopes, native sealed application records, b=2, k_rho=16, one fresh source and sixteen running sources. It does not count caller-retained extra circuits, duplicate envelopes, live Affine trees, the reference engines, or arbitrary dense witnesses supplied directly to lower-level reduction APIs. Those are separate owners; they are not assumptions about the native engine silently taking less memory.

The accepted geometry follows `nightstream/artifacts/shared-verifier-v1.json:1`, `assembly/manifest.rs:348-365`, and `lifecycle/mod.rs:50-51,83-94`:

- application rows <= 2^28 - 29,218,029 = 239,217,427;
- logical rows R <= 6,369,859 + 239,217,427 = 245,587,286;
- carrier C <= 4,685,394*54 = 253,011,276 (`neo-ajtai/src/nightstream_fprime_setup.rs:25-27`);
- minimum carrier is already above maximum R, so max(C,R)=C throughout this accepted profile;
- W+L <= 7,701. Raw application term count can be larger, but those terms and recipe nodes are sealed file records, not retained row vectors.

## Existing reserve, evaluated at the largest width

`nightstream/src/lifecycle/evaluation.rs:70-89` sets the matrix/value workspace A from the owner-authorized decimal 16 GB policy. Let B=C/54 and S=17. At B=4,685,394:

| Reserve | Formula | Bytes |
| --- | --- | ---: |
| Packed input masks plus CPU block masks/flags | S*B*(4*8+1) | 2,628,506,034 |
| Common vector or reusable ring opening form | 16*max(C,R) | 4,048,180,416 |
| Largest signed assignment transition | (S+1)*(2*ceil(C/2)+16*65,536) | 4,573,077,336 |
| Carried construction rows | 16*R | 3,929,396,576 |
| Matrix/value workspace | 16,000,000,000 - masks - common - max(assignment,carried rows) | 4,750,236,214 |

The assignment reserve dominates the carried-row reserve even at maximum accepted R. These are existing formulas, not new caps.

## CPU phase lifetimes

`optimized_engine/cpu_oracle.rs:100-124,279-316` constructs witness blocks, the empty/lazy application-table owner, and then the carried vector. `application.rs:41-73` does not allocate a resident value table in its constructor. Carried construction holds `16*C` result bytes, `16*R` matrix bytes, and one MatrixWindow. Its large array total, including input/block masks, is at most 15,356,319,240 bytes with A fully used. There is no resident fresh table overlapping this construction.

During SumCheck, ApplicationTables counts retained/replay values, matrix cache construction, original-row base values and prior weights, and its parallel worker arrays against A (`application.rs:97-195,239-310,355-447`). The assignments fold in place while encoded; their code vectors keep their original capacity. On the transition after alphabet sizes 3 -> 9 -> 81 -> 6,561, a K prefix has at most ceil(C/16) entries. Assignments convert sequentially, so at most S original C-byte code allocations plus one new C-byte K allocation overlap. The `(S+1)` reserve covers that overlap; 65,536 K values per reserved slot also exceed the actual 6,561-value alphabet and its small clone (`prefix.rs:120-199`). The common vector folds in place and retains its original capacity, which the common reserve explicitly covers. The result is the counted large CPU SumCheck payload <= masks + common + assignments + A = 16,000,000,000 bytes.

For output openings, assignments and application values are released before the common vector is moved into RingEvalScratch (`cpu_oracle.rs:258-274`). RingEvalScratch reuses the allocation for C K values and adds block flags plus active block indices (`superneo_eval/scratch.rs:7-39`). Those extra O(B) arrays are real, but the 4.573 GB assignment reserve is now unused; even an active index for every block is only 8B bytes before Vec spare capacity. It is not a simultaneous extra full-width K vector. Global row weights are charged per window (`window_eval.rs:69-79`). Terminal relation checks charge all fourteen F row-value tables per window (`window_eval.rs:133-146`, `authority.rs:86-108`). Their small worker points are fourteen fields each, not row-sized worker arrays.

## Metal phase lifetimes

`session/joint.rs:482-505` drops CPU source blocks after mask upload and before allocating common/application tables. During upload, packed input + CPU block masks/flags + GPU masks is about 3.90 GB at the full 17-source width; the common vector and application payload do not yet exist.

Carried construction holds GPU masks, the common buffer, row sums, and one matrix window (`joint.rs:197-214,290-302,349-395`). Including original packed input masks and A, the large payload bound is 2*(17*16B) + 16C + 16R + A = 15,276,667,542 bytes. MatrixWindow's Metal loader checks host cache, upload buffers, staging, and descriptors together (`joint/matrix_window.rs:87-125`); it does not count just the uploaded half.

For application replay, let G be live device masks/common/assignment-prefix payload, and let V be the base window payload. The existing reservation subtracts A before selecting V (`joint.rs:568-576,633-641`; `joint/application.rs:78-96`). For nontrivial power-of-two windows, replay selection requires 1.5V plus the final two-row output to fit 16GB-G-A (`application.rs:112-139`). Thus during matrix loading, packed host inputs + G + V + host/device metadata(A) is bounded by I + (2*16GB+G+A)/3, where I=17*16B. Across the signed-prefix states, G is largest at the first dense K state: I + 17*16*ceil(C/16) + 16*ceil(C/16) = 5,828,630,208 bytes. Substitution gives 15,467,382,642 bytes for these large arrays. Small equality/shape/alphabet buffers are additional fixed-profile terms. Fold replay drops matrix metadata before out-of-place K folding; its 1.5V peak is already in the replay check. Full initial residency uses the stricter base+next test; later promotion retains only the already folded prefix.

Metal norm partials and matrix construction run in separate command phases; the reservation uses their maximum (`joint.rs:633-641`). Common and assignment folds also run in sequence, with the old command released before the next prefix allocation (`joint.rs:789-842`). Opening plans measure their transpose/device metadata and evaluation work and shorten the matrix range until both fit (`joint/opening.rs:75-240,269-316`). In particular, the full Pad form is reserved before the first window; the previous minimum-row starvation issue is not present in this source.

## Other lifecycle phases

The oracle is dropped before output-claim construction and PiRLC (`optimized_engine/paper_joint.rs:529-541`). PiRLC borrows the 17 packed input matrices and allocates one dense F carrier (8C, not 16C); its packed branch reads masks directly (`optimized_engine/rlc.rs:17-55,99-124,374-399`). The caller drops the input witnesses before PiDEC (`nightstream/src/folding/compose.rs:20-27`; `engine/metal.rs:45-55`). Radix-two splitting allocates up to sixteen packed mask planes, and drops the dense parent before commitments/openings (`common.rs:96-132`, `folding/pi_dec.rs:80-95`). These owners do not overlap SumCheck's tables/workspace.

Commitment descriptors are finite width terms: at most one SignedBlock per nonzero column per witness. Metal releases the descriptor/position arrays before upload completes, then releases host mask arrays before device partial buffers (`session/production_commitment.rs:20-96`). CPU batch workers keep only per-witness ring sums and heap positions (`nightstream_fprime_setup.rs:337-402`). No full production key is built. Completion's physical assignment, byte logical assignment, and mask conversion run after the C/R/D fold, with the child masks retained (`lifecycle/complete.rs:139-183`). None is an omitted row-count-scaled owner.

## Payload claim versus measured RSS

The source supports a finite payload claim for these native phases: the large CPU SumCheck owners are covered by the 16,000,000,000-byte equation, matrix/value windows are checked, Metal live buffers are additionally checked at allocation, and row/source reconstruction no longer scales with total application rows. This review found no necessary engine reserve change.

Package/reference data, the bounded source scratch described in `next-memory-bound-review.md`, Vec/node/allocator overhead, thread stacks, completed-resource residency, and driver/runtime RSS are outside that equation. Their actual residency remains measured evidence, not a newly demanded formal proof. The previous full borrowed-source Metal benchmark recorded 13,848,379,392 bytes RSS under the existing 17,179,869,184-byte working guard (`matrix-window-borrowed-metal-benchmark.json`, exit 0, 291.8801 s). It used the prior borrowed-template image and its fixed fixture, so it does not measure the current streaming image or maximum-row circuit. The current run must supply its own result. No new synthetic workload or policy threshold is recommended solely because these measured overhead terms exist.
