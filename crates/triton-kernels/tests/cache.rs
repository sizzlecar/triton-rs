//! Structural assertions for the KV-cache kernels.

use triton_kernels::prelude::*;

#[test]
fn kv_cache_append_loads_pos_then_scatters_kv() {
    let text = kv_cache_append_f32::<128>::mlir();
    assert!(text.contains("tt.func @kv_cache_append_f32("));
    // Mod / div from `bh / num_heads` and `bh % num_heads`.
    assert!(text.contains("\"arith.divsi\""), "missing divsi for batch decode:\n{text}");
    assert!(text.contains("\"arith.remsi\""), "missing remsi for head decode:\n{text}");
    // 3 loads (cache_idx, new_keys, new_values), 2 stores (k_cache, v_cache).
    assert_eq!(text.matches("\"tt.load\"").count(), 3);
    assert_eq!(text.matches("\"tt.store\"").count(), 2);
    // 5 pointer-typed function args: 4× f32 (k_cache, v_cache, new_keys,
    // new_values) + 1× i32 (cache_idx).
    let header = &text[..text.find(") {").unwrap()];
    assert_eq!(header.matches("!tt.ptr<f32>").count(), 4,
               "header should declare 4 f32 pointers:\n{header}");
    assert!(header.contains("!tt.ptr<i32>"), "missing i32 cache_idx pointer:\n{header}");
}
