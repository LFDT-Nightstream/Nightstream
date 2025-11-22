# Profiling Scripts

This directory contains scripts for profiling Rust tests and benchmarks.

## Quick Start

### Profile the Starstream test specifically:

```bash
./scripts/profile_starstream_test.sh flamegraph
```

### Profile any test:

```bash
./scripts/profile_any_test.sh <package> <test_name> [mode]

# Examples:
./scripts/profile_any_test.sh neo-fold test_starstream_tx_valid_optimized flamegraph
./scripts/profile_any_test.sh neo-ajtai basic_commit_tests instruments
```

## Profiling Modes

### 1. Flamegraph (Recommended for CPU profiling)

```bash
./scripts/profile_starstream_test.sh flamegraph
```

**What it does:**
- Creates an interactive SVG visualization showing where CPU time is spent
- Each "flame" represents a function call stack
- Width shows how much time was spent in that function
- Click to zoom, search for function names

**Best for:**
- Finding hot spots in your code
- Understanding call hierarchies
- Seeing which functions dominate runtime

**Output:** `flamegraph-*.svg` - Open in any browser

**Requires:** sudo access (uses DTrace on macOS)

---

### 2. Instruments (macOS Native Profiler)

```bash
./scripts/profile_starstream_test.sh instruments
```

**What it does:**
- Uses Apple's professional profiling tool
- Beautiful GUI with timeline view
- Shows detailed CPU usage per thread
- Can track allocations, system calls, etc.

**Best for:**
- Detailed performance analysis
- Thread-level profiling
- Professional presentation-quality reports

**Output:** `*.trace` file - Opens in Instruments app

**Requires:** Xcode Command Line Tools

---

### 3. Samply (Modern Web-Based Profiler)

```bash
./scripts/profile_starstream_test.sh samply
```

**What it does:**
- Modern profiler that opens Firefox Profiler in browser
- Interactive timeline view
- Flame charts, call trees, and more
- Very user-friendly interface

**Best for:**
- Interactive exploration
- Sharing results (exports JSON)
- Modern, familiar web UI

**Output:** Opens automatically in browser

**Requires:** `cargo install samply`

---

### 4. Chrome Tracing

```bash
./scripts/profile_starstream_test.sh chrome
```

**What it does:**
- Manual instrumentation-based profiling
- Shows exact spans you instrument
- Timeline view in Chrome

**Best for:**
- When you want precise control over what's measured
- Understanding specific code paths
- Debugging performance issues in known areas

**Requires:** 
- Add to `Cargo.toml`:
  ```toml
  [dev-dependencies]
  tracing = "0.1"
  tracing-subscriber = "0.3"
  tracing-chrome = "0.7"
  ```
- Instrument your code:
  ```rust
  #[tracing::instrument]
  fn my_function() { ... }
  ```

**Output:** `trace-*.json` - Open in `chrome://tracing`

---

## Comparison

| Mode        | Setup Effort | Detail Level | Ease of Use | Best For |
|-------------|--------------|--------------|-------------|----------|
| Flamegraph  | Low          | High         | Easy        | Finding hotspots |
| Instruments | Low          | Very High    | Medium      | Deep analysis |
| Samply      | Low          | High         | Very Easy   | Quick exploration |
| Chrome      | High         | Custom       | Easy        | Specific spans |

## Installation

Most tools will auto-install, but you can pre-install:

```bash
# Flamegraph
cargo install flamegraph

# Samply
cargo install samply

# Instruments (comes with Xcode)
xcode-select --install
```

## Tips

### Getting Better Flamegraphs
- Make sure to build in release mode (scripts do this automatically)
- Look for wide bars - those are your hot spots
- Ignore `__pthread_*` and other system internals unless debugging threading

### Understanding Instruments
- Use the "Time Profiler" template for CPU profiling
- Check "High Frequency" in settings for more detail
- Filter to your crate's code using the search box

### Interpreting Results
- **CPU Time**: Total time function was on CPU
- **Self Time**: Time in function excluding children
- **Call Count**: How many times function was called
- Look for:
  - Unexpectedly wide/tall flames
  - Functions called more often than expected
  - Allocations in hot paths

## Troubleshooting

### "Permission denied" on macOS
Flamegraph needs sudo for DTrace. The script uses `sudo` automatically.

### "jq: command not found"
Install jq: `brew install jq`

### Test binary not found
Make sure package and test names are correct:
```bash
cargo test --package neo-fold --list | grep test_
```

### Profile looks empty
1. Make sure test actually runs long enough (>100ms)
2. Try increasing test workload
3. Check that release mode is being used

## Examples

### Profile and compare two implementations

```bash
# Profile version A
git checkout feature-a
./scripts/profile_any_test.sh neo-fold my_test flamegraph
mv flamegraph-*.svg flamegraph-a.svg

# Profile version B
git checkout feature-b
./scripts/profile_any_test.sh neo-fold my_test flamegraph
mv flamegraph-*.svg flamegraph-b.svg

# Compare in browser
open flamegraph-a.svg flamegraph-b.svg
```

### Quick performance check

```bash
# Run test 5 times and eyeball the timing
./scripts/profile_any_test.sh neo-fold test_starstream_tx_valid_optimized bench
```

### Deep dive with Instruments

```bash
# Get the full professional view
./scripts/profile_starstream_test.sh instruments
# Then in Instruments:
# - Switch to "Call Tree" view
# - Click "Invert Call Tree" to see bottom-up
# - Hide system libraries
```

## Adding Custom Profiling to Your Tests

You can add manual instrumentation for finer control:

```rust
use std::time::Instant;

#[test]
fn my_test() {
    let start = Instant::now();
    expensive_operation();
    println!("Operation took: {:?}", start.elapsed());
}
```

Or use the `tracing` crate for structured profiling:

```rust
use tracing::{info, instrument};

#[instrument]
fn expensive_operation() {
    // Automatically tracked when using chrome tracing
}
```

## Further Reading

- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Flamegraph.pl on GitHub](https://github.com/brendangregg/FlameGraph)
- [Firefox Profiler Docs](https://profiler.firefox.com/docs/)
- [Instruments User Guide](https://help.apple.com/instruments/)

