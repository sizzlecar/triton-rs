//! `tt` (Triton) dialect — the tile-level GPU primitives.
//!
//! Only the subset needed by the Phase 2 vector_add demo is implemented.
//! Add more ops (dot, expand_dims, reshape, broadcast, ...) as kernels need
//! them.

use crate::attr::Attr;
use crate::op::OpSpec;
use crate::ty::Type;
use crate::value::Value;

/// Program-axis enum accepted by `tt.get_program_id`.
#[derive(Debug, Clone, Copy)]
pub enum Axis {
    /// X axis (axis = 0).
    X,
    /// Y axis (axis = 1).
    Y,
    /// Z axis (axis = 2).
    Z,
}

impl From<i32> for Axis {
    fn from(v: i32) -> Self {
        match v {
            0 => Axis::X,
            1 => Axis::Y,
            2 => Axis::Z,
            _ => panic!("axis must be 0, 1, or 2 (got {v})"),
        }
    }
}

impl Axis {
    /// MLIR `axis` attribute integer (0/1/2).
    pub fn as_i32(&self) -> i32 {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }
}

/// `tt.get_program_id { axis = N : i32 } : i32`.
pub fn get_program_id(axis: impl Into<Axis>) -> OpSpec {
    let axis = axis.into();
    OpSpec::new("tt.get_program_id")
        .with_result(Type::I32)
        .with_attr("axis", Attr::i32(axis.as_i32()))
}

/// `tt.make_range { start = S, end = E } : tensor<NxLENxi32>` where N = E - S.
pub fn make_range(start: i32, end: i32) -> OpSpec {
    assert!(end > start, "make_range requires end > start (got {start}..{end})");
    let len = (end - start) as i64;
    OpSpec::new("tt.make_range")
        .with_result(Type::tensor([len], Type::I32))
        .with_attr("start", Attr::i32(start))
        .with_attr("end", Attr::i32(end))
}

/// `tt.splat %scalar : SCALAR_T -> tensor<...x SCALAR_T>`. Result shape is
/// supplied explicitly because splat targets are kernel-defined.
pub fn splat(scalar: Value, target_shape: Vec<i64>) -> OpSpec {
    let elem_ty = scalar.ty().clone();
    let result_ty = Type::tensor(target_shape, elem_ty);
    OpSpec::new("tt.splat")
        .with_operand(scalar)
        .with_result(result_ty)
}

/// `tt.addptr %ptr_tensor, %offset_tensor`. Result type matches `ptrs`.
pub fn addptr(ptrs: Value, offsets: Value) -> OpSpec {
    let ty = ptrs.ty().clone();
    OpSpec::new("tt.addptr")
        .with_operand(ptrs)
        .with_operand(offsets)
        .with_result(ty)
}

/// `tt.load %ptrs` with optional mask. Result element type = pointee, result
/// shape mirrors `ptrs` (tensor of pointers → tensor of values).
///
/// The op has three variadic operand groups (`ptr`, `mask`, `other`), so
/// generic-form printing requires the `operandSegmentSizes` attribute to
/// tell the parser how to slice the operand list. We don't expose `other`
/// (the default fill value) yet — kernels that need it use the standard
/// pattern of `tt.splat` + a separate `tt.load` argument.
pub fn load(ptrs: Value, mask: Option<Value>) -> OpSpec {
    let result_ty = match ptrs.ty() {
        Type::Tensor { shape, elem } => match elem.as_ref() {
            Type::Ptr(pointee) => Type::tensor(shape.clone(), (**pointee).clone()),
            _ => panic!("tt.load expected tensor of !tt.ptr<T>, got {}", ptrs.ty()),
        },
        Type::Ptr(pointee) => (**pointee).clone(),
        other => panic!("tt.load expected pointer or tensor of pointers, got {other}"),
    };

    let has_mask = mask.is_some();
    let mut spec = OpSpec::new("tt.load")
        .with_operand(ptrs)
        .with_result(result_ty);
    if let Some(m) = mask {
        spec = spec.with_operand(m);
    }
    spec.with_attr(
        "operandSegmentSizes",
        Attr::DenseI32Array(vec![1, has_mask as i32, 0]),
    )
}

/// `tt.store %ptrs, %values` with optional mask. No SSA results.
///
/// `tt.store` operand groups: `ptr`, `value`, `mask`. Same `operandSegmentSizes`
/// requirement as `tt.load`.
pub fn store(ptrs: Value, values: Value, mask: Option<Value>) -> OpSpec {
    let has_mask = mask.is_some();
    let mut spec = OpSpec::new("tt.store")
        .with_operand(ptrs)
        .with_operand(values);
    if let Some(m) = mask {
        spec = spec.with_operand(m);
    }
    spec.with_attr(
        "operandSegmentSizes",
        Attr::DenseI32Array(vec![1, 1, has_mask as i32]),
    )
}

/// `tt.return` — terminator for kernel functions, no SSA result.
pub fn return_() -> OpSpec {
    OpSpec::new("tt.return")
}

// ── tile-level matrix and shape ops ─────────────────────────────────────

/// `tt.dot %a, %b, %c` — block-level matmul: `a` (`M×K`) * `b` (`K×N`)
/// accumulated into `c` (`M×N`). Result type matches `c`. Common dtypes:
/// inputs `f16`/`bf16`/`tf32`, accumulator `f32`. The op's default
/// attributes (`inputPrecision = ieee`, etc.) are inferred by the parser
/// when omitted.
pub fn dot(a: Value, b: Value, c: Value) -> OpSpec {
    let result_ty = c.ty().clone();
    OpSpec::new("tt.dot")
        .with_operand(a)
        .with_operand(b)
        .with_operand(c)
        .with_result(result_ty)
}

/// `tt.broadcast %x : tensor<...> -> tensor<target_shape>`. Input must be a
/// tensor whose shape, after singleton-dim expansion, matches the target.
/// Element type is preserved.
pub fn broadcast(input: Value, target_shape: Vec<i64>) -> OpSpec {
    let elem_ty = match input.ty() {
        Type::Tensor { elem, .. } => (**elem).clone(),
        other => panic!("tt.broadcast expected a tensor, got {}", other),
    };
    OpSpec::new("tt.broadcast")
        .with_operand(input)
        .with_result(Type::tensor(target_shape, elem_ty))
}

/// `tt.expand_dims %x {axis = N : i32}` — insert a singleton dimension at
/// position `axis` in `input`'s shape. Negative axes count from the back.
pub fn expand_dims(input: Value, axis: i32) -> OpSpec {
    let result_ty = match input.ty() {
        Type::Tensor { shape, elem } => {
            let rank = shape.len() as i32;
            let axis_idx = if axis < 0 {
                (rank + 1 + axis) as usize
            } else {
                axis as usize
            };
            let mut new_shape = shape.clone();
            new_shape.insert(axis_idx, 1);
            Type::tensor(new_shape, (**elem).clone())
        }
        other => panic!("tt.expand_dims expected a tensor, got {}", other),
    };
    OpSpec::new("tt.expand_dims")
        .with_operand(input)
        .with_result(result_ty)
        .with_attr("axis", crate::attr::Attr::i32(axis))
}

/// `tt.reshape %x : tensor<...> -> tensor<new_shape>`. Element count must
/// be preserved (caller's responsibility — the IR builder doesn't check).
pub fn reshape(input: Value, new_shape: Vec<i64>) -> OpSpec {
    let elem_ty = match input.ty() {
        Type::Tensor { elem, .. } => (**elem).clone(),
        other => panic!("tt.reshape expected a tensor, got {}", other),
    };
    OpSpec::new("tt.reshape")
        .with_operand(input)
        .with_result(Type::tensor(new_shape, elem_ty))
}

/// `tt.reduce.return` — terminator for the reduction body region. Always
/// `void`; the body's combined value flows through this op back into the
/// enclosing `tt.reduce`.
pub fn reduce_return(value: Value) -> OpSpec {
    OpSpec::new("tt.reduce.return").with_operand(value)
}

// ── element-wise math intrinsics (Triton dialect) ───────────────────────
//
// All of these are 1-input → 1-output, preserve shape and element type.
// Triton lowers them to the appropriate libdevice / PTX special-function
// instructions at codegen. Element type must be one of the float kinds.

fn unary_elementwise(name: &'static str, x: Value) -> OpSpec {
    let ty = x.ty().clone();
    OpSpec::new(name).with_operand(x).with_result(ty)
}

/// `tt.exp` — natural exponential, element-wise.
pub fn exp(x: Value) -> OpSpec {
    unary_elementwise("tt.exp", x)
}
/// `tt.exp2` — base-2 exponential, element-wise.
pub fn exp2(x: Value) -> OpSpec {
    unary_elementwise("tt.exp2", x)
}
/// `tt.log` — natural logarithm, element-wise.
pub fn log(x: Value) -> OpSpec {
    unary_elementwise("tt.log", x)
}
/// `tt.log2` — base-2 logarithm, element-wise.
pub fn log2(x: Value) -> OpSpec {
    unary_elementwise("tt.log2", x)
}
/// `tt.sqrt` — square root, element-wise.
pub fn sqrt(x: Value) -> OpSpec {
    unary_elementwise("tt.sqrt", x)
}
/// `tt.sin` — sine, element-wise.
pub fn sin(x: Value) -> OpSpec {
    unary_elementwise("tt.sin", x)
}
/// `tt.cos` — cosine, element-wise.
pub fn cos(x: Value) -> OpSpec {
    unary_elementwise("tt.cos", x)
}
/// `tt.abs` — absolute value, element-wise. Works on int and float.
pub fn abs(x: Value) -> OpSpec {
    unary_elementwise("tt.abs", x)
}
