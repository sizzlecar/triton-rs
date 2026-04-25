//! Top-level `Module`, `Func`, and the user-facing `FuncBuilder`.
//!
//! A `Module` holds zero or more `Func`s. Each `Func` owns a single body
//! `Region` whose entry block's arguments are the function parameters.
//!
//! `FuncBuilder` is the main authoring API: it allocates SSA IDs, appends
//! ops to the entry block, and exposes the function arguments as `Value`s.

use crate::op::{Block, Op, OpSpec, Region};
use crate::ty::Type;
use crate::value::{SsaCounter, Value};

/// MLIR `builtin.module` — top-level container.
#[derive(Debug, Clone, Default)]
pub struct Module {
    /// Functions defined in this module.
    pub funcs: Vec<Func>,
}

impl Module {
    /// Empty module.
    pub fn new() -> Self {
        Module::default()
    }

    /// Start a new `tt.func`. Returns a builder that owns the function until
    /// `.finish()` is called.
    pub fn func(&mut self, name: impl Into<String>) -> FuncBuilder<'_> {
        FuncBuilder {
            module: self,
            name: name.into(),
            visibility: Visibility::Public,
            params: Vec::new(),
            return_types: Vec::new(),
            body: Region::new(),
            counter: SsaCounter::new(),
            committed: false,
        }
    }
}

/// Function visibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// `public` (the default for kernels).
    Public,
    /// `private`.
    Private,
}

impl Visibility {
    /// Keyword form for the MLIR printer.
    pub fn as_str(&self) -> &'static str {
        match self {
            Visibility::Public => "public",
            Visibility::Private => "private",
        }
    }
}

/// A `tt.func` definition: name, signature, body region.
#[derive(Debug, Clone)]
pub struct Func {
    /// Function symbol name (printed as `@name`).
    pub name: String,
    /// Visibility (`public`/`private`).
    pub visibility: Visibility,
    /// Parameter (name, type) list. Names are documentation-only; the
    /// printer uses the entry block's SSA IDs.
    pub params: Vec<(String, Type)>,
    /// Return types (Triton kernels usually return `()`).
    pub return_types: Vec<Type>,
    /// Body region. Entry block's args correspond to `params`.
    pub body: Region,
}

/// Authoring-time builder for a function body.
pub struct FuncBuilder<'m> {
    module: &'m mut Module,
    name: String,
    visibility: Visibility,
    params: Vec<(String, Type)>,
    return_types: Vec<Type>,
    body: Region,
    counter: SsaCounter,
    committed: bool,
}

impl<'m> FuncBuilder<'m> {
    /// Set the visibility (default: public).
    pub fn visibility(mut self, v: Visibility) -> Self {
        self.visibility = v;
        self
    }

    /// Add a parameter; returns the SSA value bound to it inside the body.
    pub fn arg(&mut self, name: impl Into<String>, ty: Type) -> Value {
        let v = self.counter.fresh(ty.clone());
        self.params.push((name.into(), ty));
        // Bind to the entry block's args list. Lazily create the entry block
        // on first call.
        if self.body.blocks.is_empty() {
            self.body.blocks.push(Block::new());
        }
        self.body.blocks[0].args.push(v.clone());
        v
    }

    /// Set return types (default: empty / `()`).
    pub fn returns(mut self, types: Vec<Type>) -> Self {
        self.return_types = types;
        self
    }

    /// Append an op described by `spec` to the entry block. Allocates fresh
    /// SSA values for each declared result type and returns them.
    pub fn op(&mut self, spec: OpSpec) -> Vec<Value> {
        let results: Vec<Value> = spec
            .result_types
            .iter()
            .map(|t| self.counter.fresh(t.clone()))
            .collect();
        if self.body.blocks.is_empty() {
            self.body.blocks.push(Block::new());
        }
        self.body.blocks[0].ops.push(Op {
            name: spec.name,
            operands: spec.operands,
            results: results.clone(),
            attrs: spec.attrs,
            regions: spec.regions,
        });
        results
    }

    /// Append an op with exactly one result. Panics if `spec` declares any
    /// other result count — this is a programming error in the dialect helper.
    pub fn op_one(&mut self, spec: OpSpec) -> Value {
        let mut r = self.op(spec);
        assert_eq!(
            r.len(),
            1,
            "op_one called on op `{}` with {} results",
            r.first().map(|_| "<unknown>").unwrap_or("<none>"),
            r.len()
        );
        r.pop().unwrap()
    }

    /// Append an op with no SSA results (e.g. `tt.store`, `tt.return`).
    pub fn op_void(&mut self, spec: OpSpec) {
        let r = self.op(spec);
        assert!(r.is_empty(), "op_void called on op with {} results", r.len());
    }

    /// Allocate a new region whose entry block carries the supplied argument
    /// types. Returns the empty region and the freshly-bound SSA values for
    /// the entry-block args (typically loop-induction variable + iter_args
    /// for `scf.for`, or branch params for `scf.if`).
    ///
    /// SSA IDs come from the same counter as the rest of the function, so
    /// values are unique across the whole function (MLIR's "isolated from
    /// above" rule for function bodies).
    pub fn new_region(&mut self, arg_types: Vec<Type>) -> (Region, Vec<Value>) {
        let entry_args: Vec<Value> = arg_types
            .into_iter()
            .map(|t| self.counter.fresh(t))
            .collect();
        let region = Region {
            blocks: vec![Block::with_args(entry_args.clone())],
        };
        (region, entry_args)
    }

    /// Append an op into a freshly-built (or otherwise existing) region's
    /// entry block. Use this together with [`Self::new_region`] to construct
    /// the bodies of region-having ops like `scf.for`/`scf.if` before
    /// passing the region to their op constructor.
    pub fn append_to_region(&mut self, region: &mut Region, spec: OpSpec) -> Vec<Value> {
        let results: Vec<Value> = spec
            .result_types
            .iter()
            .map(|t| self.counter.fresh(t.clone()))
            .collect();
        if region.blocks.is_empty() {
            region.blocks.push(Block::new());
        }
        let block = region.blocks.first_mut().expect("region has at least one block");
        block.ops.push(crate::op::Op {
            name: spec.name,
            operands: spec.operands,
            results: results.clone(),
            attrs: spec.attrs,
            regions: spec.regions,
        });
        results
    }

    /// Like [`Self::append_to_region`] but asserts the op produces exactly
    /// one result.
    pub fn append_to_region_one(&mut self, region: &mut Region, spec: OpSpec) -> Value {
        let mut r = self.append_to_region(region, spec);
        assert_eq!(r.len(), 1, "append_to_region_one called on op with {} results", r.len());
        r.pop().unwrap()
    }

    /// Like [`Self::append_to_region`] but asserts the op produces no results.
    pub fn append_to_region_void(&mut self, region: &mut Region, spec: OpSpec) {
        let r = self.append_to_region(region, spec);
        assert!(r.is_empty(), "append_to_region_void called on op with {} results", r.len());
    }

    /// Commit the function to its parent module.
    pub fn finish(mut self) {
        self.committed = true;
        self.module.funcs.push(Func {
            name: std::mem::take(&mut self.name),
            visibility: self.visibility,
            params: std::mem::take(&mut self.params),
            return_types: std::mem::take(&mut self.return_types),
            body: std::mem::take(&mut self.body),
        });
    }
}

impl<'m> Drop for FuncBuilder<'m> {
    fn drop(&mut self) {
        // Help users notice forgotten `.finish()` calls in debug builds.
        if !self.committed && !std::thread::panicking() {
            debug_assert!(
                self.committed,
                "FuncBuilder for `{}` dropped without .finish()",
                self.name
            );
        }
    }
}
