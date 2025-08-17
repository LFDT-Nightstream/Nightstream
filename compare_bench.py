import os
import subprocess
import re


def simple_table(data, headers):
    """Simple table formatter to replace tabulate dependency"""
    if not data:
        return ""
    
    # Calculate column widths
    all_rows = [headers] + data
    col_widths = []
    for col in range(len(headers)):
        max_width = max(len(str(row[col])) for row in all_rows)
        col_widths.append(max_width)
    
    # Create format string
    row_format = " | ".join(f"{{:<{width}}}" for width in col_widths)
    
    # Build table
    lines = []
    # Header
    lines.append(row_format.format(*headers))
    # Separator
    lines.append("-+-".join("-" * width for width in col_widths))
    # Data rows
    for row in data:
        lines.append(row_format.format(*row))
    
    return "\n".join(lines)


def parse_rust(path: str):
    with open(path, 'r') as f:
        text = f.read()
    benches = [
        'ring_mul_n64',
        'commit_small',
        'multilinear_sumcheck_N1024',
    ]
    times = {}
    for bench in benches:
        # Look for benchmark name followed by time on same line or next few lines
        pattern = rf"{bench}.*?time:\s*\[\s*([0-9.]+)\s*([µmun]s)"
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if match:
            val = float(match.group(1))
            unit = match.group(2)
            if unit == 'ns':
                val /= 1_000_000
            elif unit in ('µs', 'us'):
                val /= 1_000
            times[bench] = val
    return times


def parse_sim(path: str):
    times = {}
    with open(path, 'r') as f:
        for line in f:
            m = re.match(r'([^:]+):\s*([0-9.]+) ms', line)
            if m:
                times[m.group(1)] = float(m.group(2))
    return times


def main():
    subprocess.run(['bash', 'bench_rust.sh'], check=False)
    subprocess.run(['python', 'bench_sim.py'], check=False)

    rust_times = parse_rust('rust_bench.txt') if os.path.exists('rust_bench.txt') else {}
    sim_times = parse_sim('sim_bench.txt') if os.path.exists('sim_bench.txt') else {}

    benchmarks = sorted(set(rust_times) | set(sim_times))
    headers = ["Benchmark", "Rust (ms)", "Sim (ms)", "Diff (%)"]
    data = []
    for b in benchmarks:
        r = rust_times.get(b)
        s = sim_times.get(b)
        
        # Format values, showing "N/A" for missing data
        r_str = f"{r:.3f}" if r is not None else "N/A"
        s_str = f"{s:.3f}" if s is not None else "N/A"
        
        # Calculate diff only if both values exist
        if r is not None and s is not None and s > 0:
            diff = ((r - s) / s * 100)
            diff_str = f"{diff:.1f}"
        else:
            diff_str = "N/A"
        
        data.append([b, r_str, s_str, diff_str])
    print(simple_table(data, headers))


if __name__ == '__main__':
    main()
