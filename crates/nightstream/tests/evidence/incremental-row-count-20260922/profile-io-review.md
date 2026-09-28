# Profile I/O review

This review uses the completed Metal Time Profiler capture from saved image
`8B18A367-06BA-3915-8254-40C42C9C0620`, PID 1066. It does not measure the newer
incremental-count image. The [function table](../sealed-application-records-20260921/streamed-template-metal-lifecycle-profile-functions.json)
and [export receipt](../sealed-application-records-20260921/streamed-template-profile-export-receipt.json)
identify the exact image and retained backtrace data.

`execute_stage1_v1_1_witness` has 7.756 seconds of inclusive CPU sample weight
and 0.261 seconds of self weight. Within it, `storage::private_file` has
4.333 seconds inclusive and 0.002 seconds self. Every sampled `private_file`
call is below `execute_stage1_v1_1_witness` → `complete_step` → `Circuit::extend`.

The following self weights are restricted to that witness-execution scope:

| Sampled syscall | CPU sample seconds |
|---|---:|
| `__open` | 1.595 |
| `mkdir` | 1.329 |
| `__rmdir` | 0.766 |
| `__unlink` | 0.614 |
| `write` | 0.742 |
| `close` | 0.276 |
| `__lseek` | 0.159 |
| `pread` | 0.153 |
| `read` | 0.120 |

Wrapper frames such as `open` share samples with `__open`; their inclusive
weights must not be added. Release inlining leaves no separate function total
for `evaluate_recipe`, `RecipeStack`, or `execute_recipes`.

The bound application has 7,696 recipes. The saved implementation creates a new
disk stack for each recipe. Each stack creates a private directory, opens writer
and reader handles, unlinks the file, and removes the directory. Reusing one
stack per batch would remove 7,695 such file lifecycles per batch. No change or
performance result for stack reuse is part of this review.

These weights sum sampled CPU work across threads. The trace excludes waiting
threads, so it does not measure blocked I/O. The samples establish repeated
file work; they do not measure added profiler overhead or explain the full
raw-versus-profile elapsed-time difference.

The sampled images and functions contain no MetalTools, capture, or validation
frames. The trace has no Metal/GPU-specific instrument schema. This is not a
complete inventory of loaded images, and it does not establish layer injection.
Apple documents CPU overhead from [Metal API validation](https://developer.apple.com/documentation/xcode/validating-your-apps-metal-api-usage)
and GPU/compiler overhead from [Shader Validation](https://developer.apple.com/documentation/xcode/validating-your-apps-metal-shader-usage).
These documents do not establish that Time Profiler launch enables those layers
or that attach disables them.

Apple recommends [Deferred recording to reduce overhead](https://developer.apple.com/videos/play/wwdc2025/308/);
the saved captures already use it. Apple also documents
[suspended process creation for profilers](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/posix_spawnattr_setflags.3.html),
which could support an external attach harness before user code runs. Such a
method would still need validation and the same configuration for both engines.
The project’s 1800-second profiling cap must cover setup, execution and trace
finalization. No attach run was made in this review.
