import Foundation
import Metal

let device = MTLCreateSystemDefaultDevice()!
let library = try device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
let names = ["production_ajtai_partials", "ajtai_reduce_columns", "dec_sparse_ring_partials", "dec_sparse_ring_sum_chunks", "dec_add_geometric_ring_forms", "dec_bar_ring_forms_in_place"]
let pipelines: [[String: Any]] = try names.map { name in
    let state = try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
    return ["function": name, "thread_execution_width": state.threadExecutionWidth,
            "max_threads_per_threadgroup": state.maxTotalThreadsPerThreadgroup,
            "static_threadgroup_bytes": state.staticThreadgroupMemoryLength]
}
let counters: [[String: Any]] = (device.counterSets ?? []).map { set in
    ["name": set.name, "counters": set.counters.map { $0.name }]
}
let result: [String: Any] = ["device": device.name, "max_threadgroup_bytes": device.maxThreadgroupMemoryLength,
    "dispatch_boundary_sampling": device.supportsCounterSampling(.atDispatchBoundary),
    "stage_boundary_sampling": device.supportsCounterSampling(.atStageBoundary),
    "counter_sets": counters, "pipelines": pipelines]
let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
print(String(data: data, encoding: .utf8)!)
