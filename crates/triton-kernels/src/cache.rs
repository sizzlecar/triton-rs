//! KV-cache management — appending new key/value rows to a paged or
//! contiguous cache during decode.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

/// Append a single new K/V row per (batch, head) into the cache at
/// `cache_idx[batch]`. One block per `(batch * num_heads + head)` slot.
/// Mirrors ferrum's `kv_cache_append.cu` pattern: gather the destination
/// position from `cache_idx`, scatter the new row into the cache.
///
/// Layout:
///   - `new_keys`, `new_values`:   `[batch, num_heads, head_dim]`
///   - `k_cache`, `v_cache`:       `[max_seq, num_heads, head_dim]`
///   - `cache_idx`:                `[batch]` — write-position per batch row
///
/// Requires `head_dim <= BLOCK`.
#[triton_kernel]
pub fn kv_cache_append_f32<const BLOCK: usize>(
    new_keys: Ptr<f32>,
    new_values: Ptr<f32>,
    k_cache: Ptr<f32>,
    v_cache: Ptr<f32>,
    cache_idx: Ptr<i32>,
    num_heads: i32,
    head_dim: i32,
) {
    let bh = program_id(0);
    let b = bh / num_heads;
    let h = bh % num_heads;
    let pos = load(cache_idx + b);

    let cols = make_range(0, BLOCK as i32);
    let mask = cols < head_dim;

    // src: [batch=b, head=h, :head_dim]
    let src_off = b * num_heads * head_dim + h * head_dim + cols;
    // dst: [seq=pos, head=h, :head_dim]
    let dst_off = pos * num_heads * head_dim + h * head_dim + cols;

    let k = load(new_keys + src_off, mask);
    let v = load(new_values + src_off, mask);
    store(k_cache + dst_off, k, mask);
    store(v_cache + dst_off, v, mask);
}
