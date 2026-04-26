//! Vocab embedding lookup: gather one row out of the embedding table
//! per token in the sequence.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

/// `output[tok, c] = embeddings[indices[tok], c]`.
///
/// Launch: grid = (num_tokens, 1, 1). One block per token: each block
/// loads its row index, then copies that row from the embedding table.
/// Requires `hidden_size <= BLOCK`.
///
/// Mirrors the prefill-time embedding gather every transformer does
/// before the first attention block.
#[triton_kernel]
pub fn embedding_lookup_f32<const BLOCK: usize>(
    embeddings: Ptr<f32>,
    indices: Ptr<i32>,
    output: Ptr<f32>,
    hidden_size: i32,
) {
    let tok = program_id(0);
    let idx = load(indices + tok);

    let cols = make_range(0, BLOCK as i32);
    let mask = cols < hidden_size;

    let src_off = idx * hidden_size + cols;
    let dst_off = tok * hidden_size + cols;
    let v = load(embeddings + src_off, mask);
    store(output + dst_off, v, mask);
}

/// f16 embedding lookup. Pure-copy kernel — no compute, just gather +
/// store. Same shape as the f32 variant.
#[triton_kernel]
pub fn embedding_lookup_f16<const BLOCK: usize>(
    embeddings: Ptr<f16>,
    indices: Ptr<i32>,
    output: Ptr<f16>,
    hidden_size: i32,
) {
    let tok = program_id(0);
    let idx = load(indices + tok);

    let cols = make_range(0, BLOCK as i32);
    let mask = cols < hidden_size;

    let src_off = idx * hidden_size + cols;
    let dst_off = tok * hidden_size + cols;
    let v = load(embeddings + src_off, mask);
    store(output + dst_off, v, mask);
}
