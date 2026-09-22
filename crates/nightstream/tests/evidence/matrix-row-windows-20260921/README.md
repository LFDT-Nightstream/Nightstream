# Matrix row windows, 2026-09-21

CPU and Metal now consume original matrix rows through bounded local caches.
The package visitor stops at a complete row when the supplied workspace is full.
Column indexes and equality weights remain global. Opening contributions add
across windows; Pad is computed once. No Lean command or protocol change is used.

The first production check reached the ordinary 300-second cap. A separate
Time Profiler capture identified repeated row-window construction in the opening
phase. It was stopped after diagnosis and is not a completed proof. Its exact
saved native image UUID is `AA845931-2D45-342D-B92A-5A8F2AF6D885`.
All 5,090 recorded native addresses were resolved against that image. Window
construction accounts for 145.932 seconds of inclusive CPU sample weight out of
252.186 seconds with backtraces. These are sums across threads, not elapsed or
GPU times. The raw trace, complete sample export and source snapshot remain in
the private run directory recorded in `scope.json`.

The corrected retry calculation caps each rejected range at half its prior size.
It therefore avoids rebuilding almost the same range when fixed metadata costs
dominate. Scalar coefficients use the existing compact explicit representation;
only longer runs use geometric kernels. This preserves the parallel and tiled
opening paths for columns shared by many rows. CPU table construction and folding
use disjoint slices, and each bounded window shares one row-weight vector across
its matrices.

The workspace measures owned cache capacities, construction counts, and each
caller's explicit row-value reservation. Metal also accounts for device metadata,
upload overlap and opening scratch before each window is used. This does not
prove a process RSS bound: eager application/assembly storage, external inputs,
allocator overhead and driver ownership remain separate. No large circuit was
allocated to test the source-derived application storage counterexample.

`scope.json` records the current validation state. The prior 7.62× matched
lifecycle result applies to the earlier saved executable, not this implementation.

The matched Time Profiler comparison now completes on saved image
`5841982D-F5C1-35FA-903D-66E7053B74CB`: CPU 1,330.702711417 s versus Metal
291.840242833 s, or **4.5597×**. Both runs use identical inputs and return the
same final state, circuit identity and profile. Both traces report Nominal
thermal state throughout. This comparison is below the 5× full lifecycle target.
CPU recording and finalization finish in 1,461.62 s, within the approved
1,800-second Instruments cap. Its observed lifetime RSS peak is
16,790,093,824 bytes; the final post-exit peak is unavailable.

Later sealed-record and template-row changes are not part of this saved image.
Their performance must be measured separately.
