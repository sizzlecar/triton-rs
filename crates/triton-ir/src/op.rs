//! Generic Op data structure + Region/Block (which live inside ops).
//!
//! An `Op` is the universal MLIR representation:
//!   - name (e.g. `arith.addi`, `tt.load`)
//!   - operands (referenced SSA values)
//!   - results (newly-bound SSA values)
//!   - attributes (constant metadata)
//!   - regions (nested op trees, e.g. `scf.for` body)
//!
//! The printer ([`crate::printer`]) emits the **MLIR generic form**:
//! `%r = "op.name"(%a, %b) {attr = ...} : (Ta, Tb) -> Tr`. This is more
//! verbose than custom assembly but works for every op without per-op
//! printer code, which is exactly the right tradeoff for Phase 2.

use crate::attr::Attr;
use crate::value::Value;

/// A single MLIR operation.
#[derive(Debug, Clone)]
pub struct Op {
    /// Fully-qualified op name, e.g. `"arith.addi"` or `"tt.load"`.
    pub name: String,
    /// SSA operands consumed by the op.
    pub operands: Vec<Value>,
    /// SSA results produced by the op.
    pub results: Vec<Value>,
    /// Named attributes.
    pub attrs: Vec<(String, Attr)>,
    /// Nested regions (zero or more).
    pub regions: Vec<Region>,
}

/// A region contained inside an op (body of a function, branches of `scf.if`,
/// loop body of `scf.for`, etc.). Holds an ordered list of basic blocks.
#[derive(Debug, Clone, Default)]
pub struct Region {
    /// Ordered list of blocks. The first is the entry block.
    pub blocks: Vec<Block>,
}

impl Region {
    /// Empty region.
    pub fn new() -> Self {
        Region::default()
    }
}

/// A basic block: an ordered list of ops, optionally taking SSA arguments
/// (used by region entry blocks for function/loop parameters).
#[derive(Debug, Clone, Default)]
pub struct Block {
    /// Block arguments. The entry block of a function region holds the
    /// function's parameters here.
    pub args: Vec<Value>,
    /// Ops in execution order.
    pub ops: Vec<Op>,
}

impl Block {
    /// Empty block with no args.
    pub fn new() -> Self {
        Block::default()
    }

    /// Block whose entry args are the supplied SSA values (e.g. function
    /// parameters bound at entry).
    pub fn with_args(args: Vec<Value>) -> Self {
        Block { args, ops: Vec::new() }
    }
}

/// Builder description of an op: types of results to bind + op contents.
///
/// User-facing dialect helpers (e.g. [`crate::dialect::tt::get_program_id`])
/// return an `OpSpec`. The function/block builder allocates fresh result
/// values from its [`crate::value::SsaCounter`] and assembles the final
/// [`Op`].
#[derive(Debug, Clone)]
pub struct OpSpec {
    /// Op name.
    pub name: String,
    /// Operands referenced.
    pub operands: Vec<Value>,
    /// Result types to bind (lengths must match number of results desired).
    pub result_types: Vec<crate::ty::Type>,
    /// Attributes.
    pub attrs: Vec<(String, Attr)>,
    /// Nested regions.
    pub regions: Vec<Region>,
}

impl OpSpec {
    /// Convenience constructor: zero-result, no attrs, no regions.
    pub fn new(name: impl Into<String>) -> Self {
        OpSpec {
            name: name.into(),
            operands: Vec::new(),
            result_types: Vec::new(),
            attrs: Vec::new(),
            regions: Vec::new(),
        }
    }

    /// Add an operand.
    pub fn with_operand(mut self, v: Value) -> Self {
        self.operands.push(v);
        self
    }

    /// Add many operands.
    pub fn with_operands(mut self, vs: impl IntoIterator<Item = Value>) -> Self {
        self.operands.extend(vs);
        self
    }

    /// Add a result type.
    pub fn with_result(mut self, t: crate::ty::Type) -> Self {
        self.result_types.push(t);
        self
    }

    /// Add a named attribute.
    pub fn with_attr(mut self, key: impl Into<String>, value: Attr) -> Self {
        self.attrs.push((key.into(), value));
        self
    }

    /// Add a region.
    pub fn with_region(mut self, r: Region) -> Self {
        self.regions.push(r);
        self
    }
}
