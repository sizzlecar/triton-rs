//! SSA values and the per-function ID counter.
//!
//! For now we use auto-numbered `%0`, `%1`, ... names. Named values
//! (`%pid`, `%offset`) come later as a polish pass.

use crate::ty::Type;
use std::fmt;

/// An SSA value: an ID plus its type. Cheap to clone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Value {
    id: u32,
    ty: Type,
}

impl Value {
    /// Internal: construct a value from an already-allocated ID.
    /// Public users should go through [`SsaCounter::fresh`] or builder methods.
    pub(crate) fn new(id: u32, ty: Type) -> Self {
        Value { id, ty }
    }

    /// SSA name (`%0`, `%1`, ...).
    pub fn name(&self) -> String {
        format!("%{}", self.id)
    }

    /// Numeric ID (without the leading `%`).
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Type of this value.
    pub fn ty(&self) -> &Type {
        &self.ty
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.id)
    }
}

/// Per-function counter that hands out fresh SSA IDs.
#[derive(Debug, Default)]
pub struct SsaCounter {
    next: u32,
}

impl SsaCounter {
    /// New counter starting at `%0`.
    pub fn new() -> Self {
        SsaCounter { next: 0 }
    }

    /// Allocate a fresh value of the given type.
    pub fn fresh(&mut self, ty: Type) -> Value {
        let id = self.next;
        self.next += 1;
        Value::new(id, ty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_ids() {
        let mut c = SsaCounter::new();
        let a = c.fresh(Type::i32());
        let b = c.fresh(Type::f32());
        assert_eq!(a.name(), "%0");
        assert_eq!(b.name(), "%1");
        assert_eq!(a.ty(), &Type::I32);
    }
}
