//! Reduction-driven kernels: softmax and cross-entropy.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

/// Numerically-stable softmax over the last dim:
/// `out[r, c] = exp(x[r, c] - max(x[r])) / sum(exp(x[r, :] - max(x[r])))`.
/// One block per row. Requires `cols <= BLOCK`.
#[triton_kernel]
pub fn softmax_f32<const BLOCK: usize>(
    input: Ptr<f32>,
    output: Ptr<f32>,
    rows: i32,
    cols: i32,
) {
    let _ = rows;
    let row = program_id(0);
    let row_off = row * cols;
    let col_idx = make_range(0, BLOCK as i32);
    let mask = col_idx < cols;
    let abs_off = row_off + col_idx;

    let xv = load(input + abs_off, mask);

    let row_max = reduce(xv, 0, |a, b| max(a, b));
    let shifted = xv - row_max;
    let exp_v = exp(shifted);
    let sum_e = reduce(exp_v, 0, |a, b| a + b);
    let result = exp_v / sum_e;

    store(output + abs_off, result, mask);
}

/// f16 softmax. Compute happens in f32 (exp / sum / div); only loads
/// and final store are f16. Matches Python Triton's standard pattern
/// for stable f16 softmax.
#[triton_kernel]
pub fn softmax_f16<const BLOCK: usize>(
    input: Ptr<f16>,
    output: Ptr<f16>,
    rows: i32,
    cols: i32,
) {
    let _ = rows;
    let row = program_id(0);
    let row_off = row * cols;
    let col_idx = make_range(0, BLOCK as i32);
    let mask = col_idx < cols;
    let abs_off = row_off + col_idx;

    let xv = to_f32(load(input + abs_off, mask));

    let row_max = reduce(xv, 0, |a, b| max(a, b));
    let shifted = xv - row_max;
    let exp_v = exp(shifted);
    let sum_e = reduce(exp_v, 0, |a, b| a + b);
    let result = exp_v / sum_e;

    store(output + abs_off, to_f16(result), mask);
}

/// Cross-entropy forward (training utility, but useful as a shipped
/// ready-to-use kernel when ferrum is extended for fine-tuning):
/// `loss[r] = log(sum(exp(logits[r, :] - max))) + max - logits[r, label[r]]`.
///
/// Numerically stable via the log-sum-exp trick. One block per row.
/// Requires `num_classes <= BLOCK`.
#[triton_kernel]
pub fn cross_entropy_forward_f32<const BLOCK: usize>(
    logits: Ptr<f32>,
    labels: Ptr<i32>,
    losses: Ptr<f32>,
    num_classes: i32,
) {
    let row = program_id(0);
    let cols = make_range(0, BLOCK as i32);
    let mask = cols < num_classes;
    let row_off = row * num_classes;

    let logits_v = load(logits + row_off + cols, mask);

    // Numerically stable log-sum-exp.
    let row_max = reduce(logits_v, 0, |a, b| max(a, b));
    let shifted = logits_v - row_max;
    let exp_v = exp(shifted);
    let sum_e = reduce(exp_v, 0, |a, b| a + b);
    let log_sum_exp = log(sum_e) + row_max;

    // Pull out the logit at the ground-truth label index.
    let label = load(labels + row);
    let label_logit = load(logits + row_off + label);

    let loss = log_sum_exp - label_logit;
    store(losses + row, loss);
}
