use neo_wasm::{build_wasm_relation_layout, named_column_family, plan_wasm_memory_slots};

fn main() {
    let plan = plan_wasm_memory_slots(build_wasm_relation_layout().auxiliary.memory.entries());
    let max_load = plan.support_loads.values().copied().max().unwrap_or(0);
    println!("logical ports: {}", plan.logical_ports);
    println!("physical slots (greedy): {}", plan.slots.len());
    println!("maximum support load: {max_load}");
    println!("conservative singleton slots: {}", plan.conservative_singletons);
    for (&atom, &load) in &plan.support_loads {
        if load == max_load {
            let name = named_column_family(atom).map_or("unknown", |family| family.name);
            println!("busiest support atom: {name} (column {atom})");
        }
    }
}
