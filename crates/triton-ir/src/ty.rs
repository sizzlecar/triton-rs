//! MLIR types relevant to the Triton dialect.
//!
//! Printing rules follow the MLIR textual format:
//!   - scalars: `i32`, `f16`, `bf16`, ...
//!   - tensors: `tensor<NxMxELEM>`
//!   - Triton pointers: `!tt.ptr<ELEM>`
//!   - tensor of pointers: `tensor<Nx!tt.ptr<f32>>`

use std::fmt;

/// MLIR / Triton type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    /// 1-bit integer (boolean in IR).
    I1,
    /// 8-bit signed integer.
    I8,
    /// 16-bit signed integer.
    I16,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// IEEE half-precision float.
    F16,
    /// IEEE single-precision float.
    F32,
    /// IEEE double-precision float.
    F64,
    /// bfloat16.
    BF16,
    /// Architecture-dependent integer (`index` dialect).
    Index,
    /// Ranked tensor `tensor<DIMSxELEM>`.
    Tensor {
        /// Dimensions; static sizes only for now (no `?` dynamic dims).
        shape: Vec<i64>,
        /// Element type. Boxed to break the recursive size cycle.
        elem: Box<Type>,
    },
    /// Triton pointer type `!tt.ptr<ELEM>`.
    Ptr(Box<Type>),
}

impl Type {
    /// `i1`.
    pub fn i1() -> Self {
        Type::I1
    }
    /// `i32`.
    pub fn i32() -> Self {
        Type::I32
    }
    /// `i64`.
    pub fn i64() -> Self {
        Type::I64
    }
    /// `f16`.
    pub fn f16() -> Self {
        Type::F16
    }
    /// `f32`.
    pub fn f32() -> Self {
        Type::F32
    }
    /// `bf16`.
    pub fn bf16() -> Self {
        Type::BF16
    }
    /// `tensor<...xELEM>`.
    pub fn tensor(shape: impl Into<Vec<i64>>, elem: Type) -> Self {
        Type::Tensor {
            shape: shape.into(),
            elem: Box::new(elem),
        }
    }
    /// `!tt.ptr<ELEM>`.
    pub fn ptr(pointee: Type) -> Self {
        Type::Ptr(Box::new(pointee))
    }

    /// `true` if this is a tensor type.
    pub fn is_tensor(&self) -> bool {
        matches!(self, Type::Tensor { .. })
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::I1 => f.write_str("i1"),
            Type::I8 => f.write_str("i8"),
            Type::I16 => f.write_str("i16"),
            Type::I32 => f.write_str("i32"),
            Type::I64 => f.write_str("i64"),
            Type::F16 => f.write_str("f16"),
            Type::F32 => f.write_str("f32"),
            Type::F64 => f.write_str("f64"),
            Type::BF16 => f.write_str("bf16"),
            Type::Index => f.write_str("index"),
            Type::Tensor { shape, elem } => {
                f.write_str("tensor<")?;
                for d in shape {
                    write!(f, "{}x", d)?;
                }
                write!(f, "{}>", elem)
            }
            Type::Ptr(t) => write!(f, "!tt.ptr<{}>", t),
        }
    }
}

/// Mapping from a Rust marker type (e.g. `f32`, `f16`, `i32`) to its
/// MLIR-side [`Type`]. Lets dtype-generic kernels written like
/// `pub fn vec_add<T: TritonElem, const BLOCK: usize>(...)` resolve `T`
/// to the right [`Type`] at IR-build time without the proc-macro
/// having to bake the dtype into the function name.
///
/// Implementations exist for the float types (`f32`, `f16`, `bf16`)
/// and the integer types we expose in the DSL (`i1`, `i8`, `i16`,
/// `i32`, `i64`). To use a custom marker type, implement this trait.
pub trait TritonElem {
    /// The MLIR element type for `Self`.
    fn ir_type() -> Type;
}

/// Marker for IEEE half-precision (f16). Sized to one byte for
/// `Ptr<f16>` to make sense in a kernel signature.
#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct f16;
/// Marker for bfloat16.
#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct bf16;

impl TritonElem for f32 { fn ir_type() -> Type { Type::F32 } }
impl TritonElem for f64 { fn ir_type() -> Type { Type::F64 } }
impl TritonElem for f16 { fn ir_type() -> Type { Type::F16 } }
impl TritonElem for bf16 { fn ir_type() -> Type { Type::BF16 } }
impl TritonElem for i8 { fn ir_type() -> Type { Type::I8 } }
impl TritonElem for i16 { fn ir_type() -> Type { Type::I16 } }
impl TritonElem for i32 { fn ir_type() -> Type { Type::I32 } }
impl TritonElem for i64 { fn ir_type() -> Type { Type::I64 } }
impl TritonElem for bool { fn ir_type() -> Type { Type::I1 } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_types() {
        assert_eq!(Type::i32().to_string(), "i32");
        assert_eq!(Type::f16().to_string(), "f16");
        assert_eq!(Type::bf16().to_string(), "bf16");
    }

    #[test]
    fn tensor_types() {
        assert_eq!(Type::tensor([1024], Type::i32()).to_string(), "tensor<1024xi32>");
        assert_eq!(
            Type::tensor([128, 256], Type::f32()).to_string(),
            "tensor<128x256xf32>"
        );
    }

    #[test]
    fn ptr_types() {
        assert_eq!(Type::ptr(Type::f32()).to_string(), "!tt.ptr<f32>");
        assert_eq!(
            Type::tensor([1024], Type::ptr(Type::f32())).to_string(),
            "tensor<1024x!tt.ptr<f32>>"
        );
    }
}
