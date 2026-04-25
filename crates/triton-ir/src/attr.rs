//! MLIR attributes (op-level metadata).
//!
//! Attributes carry constant data attached to ops: integer/float/string
//! literals, type references, arrays, and nested dictionaries. Printing
//! follows the MLIR textual format (`42 : i32`, `"foo"`, `[1, 2, 3]`,
//! `{key = val, ...}`).

use crate::ty::Type;
use std::fmt;

/// A single MLIR attribute value.
#[derive(Debug, Clone, PartialEq)]
pub enum Attr {
    /// Boolean literal: `true` / `false`.
    Bool(bool),
    /// Typed integer literal (printed as `N : iX`).
    Int(i64, Type),
    /// Typed float literal (printed as `N.N : fX`).
    Float(f64, Type),
    /// Quoted string literal.
    String(String),
    /// Bare type reference (`i32`, `f16`, `tensor<...>`).
    Type(Type),
    /// Array of attributes: `[a, b, c]`.
    Array(Vec<Attr>),
    /// Dictionary: `{k1 = v1, k2 = v2}`.
    Dict(Vec<(String, Attr)>),
    /// MLIR `DenseI32ArrayAttr`, printed as `array<i32: 1, 2, 3>`. Used for
    /// `operandSegmentSizes` and similar metadata on variadic-operand ops.
    DenseI32Array(Vec<i32>),
}

impl Attr {
    /// Shorthand for `Attr::Int(value, Type::I32)`.
    pub fn i32(value: i32) -> Self {
        Attr::Int(value as i64, Type::I32)
    }
    /// Shorthand for `Attr::Int(value, Type::I64)`.
    pub fn i64(value: i64) -> Self {
        Attr::Int(value, Type::I64)
    }
    /// Shorthand for `Attr::Float(value, Type::F32)`.
    pub fn f32(value: f32) -> Self {
        Attr::Float(value as f64, Type::F32)
    }
}

impl fmt::Display for Attr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Attr::Bool(b) => write!(f, "{}", b),
            Attr::Int(v, t) => write!(f, "{} : {}", v, t),
            Attr::Float(v, t) => write!(f, "{:?} : {}", v, t),
            Attr::String(s) => write!(f, "\"{}\"", s.escape_default()),
            Attr::Type(t) => write!(f, "{}", t),
            Attr::Array(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Attr::Dict(items) => {
                write!(f, "{{")?;
                for (i, (k, v)) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{} = {}", k, v)?;
                }
                write!(f, "}}")
            }
            Attr::DenseI32Array(items) => {
                write!(f, "array<i32")?;
                if items.is_empty() {
                    write!(f, ">")
                } else {
                    write!(f, ": ")?;
                    for (i, v) in items.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", v)?;
                    }
                    write!(f, ">")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_attr() {
        assert_eq!(Attr::i32(42).to_string(), "42 : i32");
        assert_eq!(Attr::i64(1024).to_string(), "1024 : i64");
    }

    #[test]
    fn bool_attr() {
        assert_eq!(Attr::Bool(true).to_string(), "true");
    }

    #[test]
    fn array_attr() {
        let a = Attr::Array(vec![Attr::i32(1), Attr::i32(2)]);
        assert_eq!(a.to_string(), "[1 : i32, 2 : i32]");
    }

    #[test]
    fn dict_attr() {
        let d = Attr::Dict(vec![
            ("start".into(), Attr::i32(0)),
            ("end".into(), Attr::i32(1024)),
        ]);
        assert_eq!(d.to_string(), "{start = 0 : i32, end = 1024 : i32}");
    }
}
