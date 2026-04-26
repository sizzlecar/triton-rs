//! Structural assertions for embedding_lookup.

use triton_kernels::prelude::*;

#[test]
fn embedding_lookup_loads_index_then_gathers_row() {
    let text = embedding_lookup_f32::<1024>::mlir();
    assert!(text.contains("tt.func @embedding_lookup_f32("));
    // Two pointer types in the signature: f32 (embeddings, output) and i32 (indices).
    assert!(text.contains("!tt.ptr<f32>"));
    assert!(text.contains("!tt.ptr<i32>"));
    // 2 loads: scalar load of indices[tok], tile load of embeddings[idx].
    assert_eq!(text.matches("\"tt.load\"").count(), 2);
    // 1 store of the row to output.
    assert_eq!(text.matches("\"tt.store\"").count(), 1);
}
