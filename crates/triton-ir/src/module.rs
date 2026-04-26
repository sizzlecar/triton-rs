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
            region_stack: Vec::new(),
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
///
/// `region_stack` makes [`Self::op`] / [`Self::op_one`] / [`Self::op_void`]
/// context-aware: when empty, ops append to the function body; when
/// non-empty, ops go to the top region (the body of the most-recently
/// entered region-having op like `scf.for`). This lets [`Self::for_loop_with`]
/// expose a closure-based API where the user writes the loop body using the
/// same `op_one` calls as outside the loop.
pub struct FuncBuilder<'m> {
    module: &'m mut Module,
    name: String,
    visibility: Visibility,
    params: Vec<(String, Type)>,
    return_types: Vec<Type>,
    body: Region,
    counter: SsaCounter,
    region_stack: Vec<Region>,
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

    /// Append an op described by `spec` to the **current** target region —
    /// which is the topmost entry on `region_stack` if any, otherwise the
    /// function body. Allocates fresh SSA values for each declared result
    /// type and returns them.
    pub fn op(&mut self, spec: OpSpec) -> Vec<Value> {
        let results: Vec<Value> = spec
            .result_types
            .iter()
            .map(|t| self.counter.fresh(t.clone()))
            .collect();
        let target_region: &mut Region = if let Some(top) = self.region_stack.last_mut() {
            top
        } else {
            &mut self.body
        };
        if target_region.blocks.is_empty() {
            target_region.blocks.push(Block::new());
        }
        target_region.blocks[0].ops.push(Op {
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

    /// High-level `scf.for` construction. Pushes a fresh region for the loop
    /// body, runs `body_fn` (which can use `self.op_one` etc. — those calls
    /// will land in the loop body, not the function body), appends an
    /// `scf.yield` of the values returned by the closure, then pops the
    /// region and constructs the `scf.for` op.
    ///
    /// `lb`, `ub`, `step` must share the same scalar integer type (typically
    /// `i32`); the loop induction variable will have that same type.
    /// `iter_args` is the list of values threaded through the loop. The
    /// closure receives the induction variable and a snapshot of the
    /// iter_args bound to the entry block of the body region; it must
    /// return a `Vec<Value>` of matching length (these become the
    /// `scf.yield` operands).
    ///
    /// Nested `for_loop_with` calls work transparently — the region stack
    /// keeps track.
    pub fn for_loop_with<F>(
        &mut self,
        lb: Value,
        ub: Value,
        step: Value,
        iter_args: Vec<Value>,
        body_fn: F,
    ) -> Vec<Value>
    where
        F: FnOnce(&mut FuncBuilder<'m>, Value, Vec<Value>) -> Vec<Value>,
    {
        // Allocate entry-block SSA values for the induction var + iter_args
        // viewed from inside the body.
        let induction_ty = lb.ty().clone();
        let body_induction = self.counter.fresh(induction_ty);
        let body_iter_args: Vec<Value> = iter_args
            .iter()
            .map(|v| self.counter.fresh(v.ty().clone()))
            .collect();

        // Build the entry block carrying (induction, iter_args...) as args.
        let mut entry_block_args = vec![body_induction.clone()];
        entry_block_args.extend(body_iter_args.iter().cloned());
        let region = Region {
            blocks: vec![Block::with_args(entry_block_args)],
        };

        // Push, run body, append scf.yield, pop.
        self.region_stack.push(region);
        let yield_values = body_fn(self, body_induction, body_iter_args);
        self.op_void(crate::dialect::scf::yield_(yield_values));
        let body_region = self
            .region_stack
            .pop()
            .expect("for_loop_with stack underflow — region was popped twice");

        // Construct the scf.for op and append it to the outer region.
        self.op(crate::dialect::scf::for_loop(
            lb, ub, step, iter_args, body_region,
        ))
    }

    /// High-level `tt.reduce` construction. Pushes a fresh region for the
    /// reducer body, runs `body_fn(lhs, rhs)` to produce the combined value,
    /// appends a `tt.reduce.return`, then pops and constructs the
    /// `tt.reduce` op against the outer region.
    ///
    /// `axis` is the dimension being reduced (negative axes count from the
    /// back). The result type drops that dimension from `input`'s shape;
    /// when only one dim remains and it's the reduced one, the result
    /// becomes a scalar (the element type).
    pub fn reduce_with<F>(&mut self, input: Value, axis: i32, body_fn: F) -> Value
    where
        F: FnOnce(&mut FuncBuilder<'m>, Value, Value) -> Value,
    {
        let (in_shape, elem_ty) = match input.ty() {
            Type::Tensor { shape, elem } => (shape.clone(), (**elem).clone()),
            other => panic!("tt.reduce expected a tensor input, got {}", other),
        };

        // Body region: entry block has 2 args (lhs, rhs), both elem type.
        let lhs = self.counter.fresh(elem_ty.clone());
        let rhs = self.counter.fresh(elem_ty.clone());
        let region = Region {
            blocks: vec![Block::with_args(vec![lhs.clone(), rhs.clone()])],
        };

        self.region_stack.push(region);
        let yielded = body_fn(self, lhs, rhs);
        self.op_void(crate::dialect::tt::reduce_return(yielded));
        let body_region = self
            .region_stack
            .pop()
            .expect("reduce_with stack underflow");

        // Compute result shape by dropping the reduced axis.
        let rank = in_shape.len() as i32;
        let axis_idx = if axis < 0 {
            (rank + axis) as usize
        } else {
            axis as usize
        };
        let mut out_shape = in_shape;
        out_shape.remove(axis_idx);
        let result_ty = if out_shape.is_empty() {
            elem_ty
        } else {
            Type::tensor(out_shape, elem_ty)
        };

        let results = self.op(
            OpSpec::new("tt.reduce")
                .with_operand(input)
                .with_result(result_ty)
                .with_attr("axis", crate::attr::Attr::i32(axis))
                .with_region(body_region),
        );
        results.into_iter().next().expect("tt.reduce produces one result")
    }

    // ── DSL helper methods (delegate to crate::ops) ────────────────────
    //
    // These exist as inherent methods so the proc-macro can emit
    // `__triton_f.add(a, b)` and Rust's method-call auto-reborrow handles
    // both scopes — top-level where __triton_f is a FuncBuilder value, and
    // closure bodies where __triton_f is `&mut FuncBuilder`. Free
    // `crate::ops::add(&mut __triton_f, ...)` would need explicit reborrow
    // at every call site inside a closure (`&mut *__triton_f`), which we'd
    // rather not bake into the proc-macro's emit code paths.
    //
    // All binary ops auto-broadcast a scalar operand to match a tensor one
    // via `tt.splat`, so users can write `tensor + scalar` (or `ptr + i32`
    // — even works for tt.addptr) without sprinkling splat_1d everywhere.

    /// Coerce a (scalar | tensor) pair into matching shapes AND elem types
    /// for element-wise ops:
    ///
    /// 1. **Scalar + tensor** — cast the scalar's elem to the tensor's elem
    ///    (truncf / extf / sitofp / etc. via `cast_with_elem`), then splat
    ///    to the tensor's shape. Mirrors Python @triton.jit's implicit
    ///    behavior: `f16_tensor * 2.0_f32` → the f32 literal silently
    ///    truncates to f16 to match the tensor.
    /// 2. **Tensor + tensor with same shape and elem type** — pass through.
    /// 3. **Tensor + tensor with singleton-broadcastable shapes** — emit
    ///    `tt.broadcast` on each side to the common max-shape (e.g.
    ///    `[32, 1]` and `[1, 128]` broadcast to `[32, 128]`). Required by
    ///    the attention kernel's outer-add of `pos_range[:, None] +
    ///    dim_range[None, :]`.
    /// 4. **Tensor + tensor with mismatched non-1 dims** — panic; this is
    ///    almost certainly a kernel bug, not an intended broadcast.
    fn coerce_elemwise(&mut self, a: Value, b: Value) -> (Value, Value) {
        let a_t = matches!(a.ty(), Type::Tensor { .. });
        let b_t = matches!(b.ty(), Type::Tensor { .. });
        match (a_t, b_t) {
            (true, false) => {
                let (shape, a_elem) = if let Type::Tensor { shape, elem } = a.ty() {
                    (shape.clone(), (**elem).clone())
                } else {
                    unreachable!()
                };
                // Only cast when both sides are arithmetic — for pointer
                // tensors (`tensor<!tt.ptr<T>>`), the scalar is an offset
                // (i32) that tt.addptr handles after splat without a cast.
                let b_cast = if is_arith_elem(&a_elem) && is_arith_elem(&b.ty()) {
                    self.cast_with_elem(b, a_elem)
                } else {
                    b
                };
                let b_splat = self.op_one(crate::dialect::tt::splat(b_cast, shape));
                (a, b_splat)
            }
            (false, true) => {
                let (shape, b_elem) = if let Type::Tensor { shape, elem } = b.ty() {
                    (shape.clone(), (**elem).clone())
                } else {
                    unreachable!()
                };
                let a_cast = if is_arith_elem(&b_elem) && is_arith_elem(&a.ty()) {
                    self.cast_with_elem(a, b_elem)
                } else {
                    a
                };
                let a_splat = self.op_one(crate::dialect::tt::splat(a_cast, shape));
                (a_splat, b)
            }
            (true, true) => {
                let (sa, sb) = match (a.ty(), b.ty()) {
                    (Type::Tensor { shape: sa, .. }, Type::Tensor { shape: sb, .. }) => {
                        (sa.clone(), sb.clone())
                    }
                    _ => unreachable!(),
                };
                if sa == sb {
                    return (a, b);
                }
                // Singleton-broadcast: ranks must match (caller used
                // expand_dims to align). For each dim, both sides must be
                // either equal or one of them must be 1; broadcast the 1.
                if sa.len() != sb.len() {
                    panic!(
                        "elemwise rank mismatch: {} vs {} (lift a singleton dim with `expand_dims` first)",
                        a.ty(),
                        b.ty()
                    );
                }
                let target: Vec<i64> = sa
                    .iter()
                    .zip(sb.iter())
                    .map(|(&da, &db)| match (da, db) {
                        (x, y) if x == y => x,
                        (1, y) => y,
                        (x, 1) => x,
                        _ => panic!(
                            "elemwise non-broadcastable shapes: {} vs {}",
                            a.ty(),
                            b.ty()
                        ),
                    })
                    .collect();
                let a_b = if sa == target {
                    a
                } else {
                    self.op_one(crate::dialect::tt::broadcast(a, target.clone()))
                };
                let b_b = if sb == target {
                    b
                } else {
                    self.op_one(crate::dialect::tt::broadcast(b, target))
                };
                (a_b, b_b)
            }
            _ => (a, b),
        }
    }

    /// Type-dispatched `+`. Auto-broadcasts scalar → tensor (and ptr-tensor
    /// + scalar-i32 etc. when chained through splat). See [`crate::ops::add`].
    pub fn add(&mut self, a: Value, b: Value) -> Value {
        let (a, b) = self.coerce_elemwise(a, b);
        crate::ops::add(self, a, b)
    }
    /// Type-dispatched `-`. See [`crate::ops::sub`].
    pub fn sub(&mut self, a: Value, b: Value) -> Value {
        let (a, b) = self.coerce_elemwise(a, b);
        crate::ops::sub(self, a, b)
    }
    /// Type-dispatched `*`. See [`crate::ops::mul`].
    pub fn mul(&mut self, a: Value, b: Value) -> Value {
        let (a, b) = self.coerce_elemwise(a, b);
        crate::ops::mul(self, a, b)
    }
    /// Type-dispatched `/`. See [`crate::ops::div`].
    pub fn div(&mut self, a: Value, b: Value) -> Value {
        let (a, b) = self.coerce_elemwise(a, b);
        crate::ops::div(self, a, b)
    }
    /// Type-dispatched `%`. See [`crate::ops::rem`].
    pub fn rem(&mut self, a: Value, b: Value) -> Value {
        let (a, b) = self.coerce_elemwise(a, b);
        crate::ops::rem(self, a, b)
    }
    /// Type-dispatched `max`. See [`crate::ops::max`].
    pub fn max(&mut self, a: Value, b: Value) -> Value {
        let (a, b) = self.coerce_elemwise(a, b);
        crate::ops::max(self, a, b)
    }
    /// Type-dispatched `min`. See [`crate::ops::min`].
    pub fn min(&mut self, a: Value, b: Value) -> Value {
        let (a, b) = self.coerce_elemwise(a, b);
        crate::ops::min(self, a, b)
    }
    /// Type-dispatched `<`. See [`crate::ops::lt`].
    pub fn lt(&mut self, a: Value, b: Value) -> Value {
        let (a, b) = self.coerce_elemwise(a, b);
        crate::ops::lt(self, a, b)
    }
    /// Type-dispatched `<=`. See [`crate::ops::le`].
    pub fn le(&mut self, a: Value, b: Value) -> Value {
        let (a, b) = self.coerce_elemwise(a, b);
        crate::ops::le(self, a, b)
    }
    /// Type-dispatched `>`. See [`crate::ops::gt`].
    pub fn gt(&mut self, a: Value, b: Value) -> Value {
        let (a, b) = self.coerce_elemwise(a, b);
        crate::ops::gt(self, a, b)
    }
    /// Type-dispatched `>=`. See [`crate::ops::ge`].
    pub fn ge(&mut self, a: Value, b: Value) -> Value {
        let (a, b) = self.coerce_elemwise(a, b);
        crate::ops::ge(self, a, b)
    }
    /// Type-dispatched `==`. See [`crate::ops::eq`].
    pub fn eq(&mut self, a: Value, b: Value) -> Value {
        let (a, b) = self.coerce_elemwise(a, b);
        crate::ops::eq(self, a, b)
    }
    /// Type-dispatched `!=`. See [`crate::ops::ne`].
    pub fn ne(&mut self, a: Value, b: Value) -> Value {
        let (a, b) = self.coerce_elemwise(a, b);
        crate::ops::ne(self, a, b)
    }

    /// Build a Value from a Rust integer literal that matches `sample`'s
    /// element type. Used by the DSL's auto-promotion: when the proc-macro
    /// sees `pid * 1024`, it lifts the literal to a Value via this helper
    /// before calling `mul`. The element type of the result follows
    /// `sample`'s element type (`I32` / `I64` / `F32`); shape coercion to
    /// tensor happens in the binary op via `coerce_elemwise`.
    pub fn lit_i64(&mut self, sample: &Value, lit: i64) -> Value {
        let elem = match sample.ty() {
            Type::Tensor { elem, .. } => (**elem).clone(),
            other => other.clone(),
        };
        match elem {
            Type::I32 => self.op_one(crate::dialect::arith::constant_i32(lit as i32)),
            Type::I64 => self.op_one(crate::dialect::arith::constant_i64(lit)),
            Type::F32 => self.op_one(crate::dialect::arith::constant_f32(lit as f32)),
            // Pointer + int literal → the literal is the offset for
            // tt.addptr, which always wants i32. Don't try to lift the
            // literal to be a pointer-typed value.
            Type::Ptr(_) => self.op_one(crate::dialect::arith::constant_i32(lit as i32)),
            other => panic!(
                "lit_i64: cannot lift integer literal to type {} (sample type was {})",
                other,
                sample.ty()
            ),
        }
    }

    /// Element-type cast dispatcher used by `to_f32` / `to_f16` / `to_i32`.
    ///
    /// Dispatches on the input's element type vs the requested target:
    /// `int → float` → `arith.sitofp`, `float → int` → `arith.fptosi`,
    /// `float → float` widening/narrowing → `arith.extf` / `arith.truncf`,
    /// `int → int` widening/narrowing → `arith.extsi` / `arith.trunci`.
    /// Identity casts (input already at the target elem type) pass through.
    /// Tensor inputs preserve shape.
    fn cast_with_elem(&mut self, x: Value, target_elem: Type) -> Value {
        let in_elem = match x.ty() {
            Type::Tensor { elem, .. } => (**elem).clone(),
            other => other.clone(),
        };
        if in_elem == target_elem {
            return x;
        }
        let target_ty = match x.ty() {
            Type::Tensor { shape, .. } => Type::tensor(shape.clone(), target_elem.clone()),
            _ => target_elem.clone(),
        };
        let in_is_int = is_int_elem(&in_elem);
        let in_is_float = is_float_elem(&in_elem);
        let out_is_int = is_int_elem(&target_elem);
        let out_is_float = is_float_elem(&target_elem);

        let in_w = elem_bit_width(&in_elem);
        let out_w = elem_bit_width(&target_elem);

        let spec = if in_is_int && out_is_float {
            crate::dialect::arith::sitofp(x, target_ty)
        } else if in_is_float && out_is_int {
            crate::dialect::arith::fptosi(x, target_ty)
        } else if in_is_float && out_is_float {
            if out_w > in_w {
                crate::dialect::arith::extf(x, target_ty)
            } else {
                crate::dialect::arith::truncf(x, target_ty)
            }
        } else if in_is_int && out_is_int {
            if out_w > in_w {
                crate::dialect::arith::extsi(x, target_ty)
            } else {
                crate::dialect::arith::trunci(x, target_ty)
            }
        } else {
            panic!(
                "cast_with_elem: unsupported cast {} → {}",
                in_elem, target_elem
            );
        };
        self.op_one(spec)
    }

    /// Cast a value's element type to `f32`. Tensor inputs preserve shape.
    /// No-op if already `f32`.
    pub fn to_f32(&mut self, x: Value) -> Value {
        self.cast_with_elem(x, Type::F32)
    }
    /// Cast a value's element type to `f16`. Tensor inputs preserve shape.
    /// No-op if already `f16`.
    pub fn to_f16(&mut self, x: Value) -> Value {
        self.cast_with_elem(x, Type::F16)
    }
    /// Cast a value's element type to `i32` (truncating for floats / wider
    /// integers). Tensor inputs preserve shape. No-op if already `i32`.
    pub fn to_i32(&mut self, x: Value) -> Value {
        self.cast_with_elem(x, Type::I32)
    }

    /// Build a Value from a Rust float literal matching `sample`'s
    /// element type. Float literals can only land in float types.
    pub fn lit_f64(&mut self, sample: &Value, lit: f64) -> Value {
        let elem = match sample.ty() {
            Type::Tensor { elem, .. } => (**elem).clone(),
            other => other.clone(),
        };
        match elem {
            Type::F32 => self.op_one(crate::dialect::arith::constant_f32(lit as f32)),
            // For f16/bf16: build a f32 constant then truncf — `arith.constant`
            // can take f16/bf16 attributes but our printer's Attr::f32 helper
            // is f32-only, and the truncf pass-through is what Python Triton
            // does anyway for literal lifting in mixed-precision kernels.
            Type::F16 => {
                let c = self.op_one(crate::dialect::arith::constant_f32(lit as f32));
                self.op_one(crate::dialect::arith::truncf(c, Type::F16))
            }
            Type::BF16 => {
                let c = self.op_one(crate::dialect::arith::constant_f32(lit as f32));
                self.op_one(crate::dialect::arith::truncf(c, Type::BF16))
            }
            other => panic!(
                "lit_f64: cannot lift float literal to type {} (sample type was {})",
                other,
                sample.ty()
            ),
        }
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

/// True when `t` (or its tensor element) is an arithmetic dtype that
/// `cast_with_elem` knows how to convert. Pointers, indexes, and other
/// non-numeric types return false — those skip the auto-cast that
/// `coerce_elemwise` does on scalar+tensor pairs (which would otherwise
/// mis-cast offsets in pointer arithmetic).
fn is_arith_elem(t: &Type) -> bool {
    is_int_elem(t) || is_float_elem(t)
}

fn is_int_elem(t: &Type) -> bool {
    matches!(t, Type::I1 | Type::I8 | Type::I16 | Type::I32 | Type::I64)
}

fn is_float_elem(t: &Type) -> bool {
    matches!(t, Type::F16 | Type::F32 | Type::F64 | Type::BF16)
}

fn elem_bit_width(t: &Type) -> u32 {
    match t {
        Type::I1 => 1,
        Type::I8 => 8,
        Type::I16 | Type::F16 | Type::BF16 => 16,
        Type::I32 | Type::F32 => 32,
        Type::I64 | Type::F64 => 64,
        _ => 0,
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
