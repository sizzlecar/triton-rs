# triton-rs Architecture

> Last updated: 2026-04-26
> Status: **Phase 1 done — zero-Python compile + runtime path verified end-to-end on RTX 5070 Ti** (max_err=0 on vec_add, sm_89). The C ABI shim (`crates/triton-sys/`) drives Triton's MLIR pass pipeline directly from C++; cubin loads via cudarc; no Python in build or runtime path. See `crates/triton-sys/SPIKE.md` for the orchestrator map.
>
> Pre-built LLVM tarball pinned at commit `86b69c31` (Triton v3.2.0). Bumping to v3.6.0 is the next milestone.

## 1. 项目愿景

triton-rs 是一个 **Rust 原生的 GPU 算子 DSL + 编译/运行时框架**。它复用 Triton 项目的 MLIR 编译器后端（生成 PTX/AMDGPU/SPIR-V），但**前端、运行时和工具链全部用 Rust 实现**，对运行环境**零 Python 依赖**。

**第一服务对象**：sibling `ferrum-infer-rs`（Rust LLM 推理引擎），但 DSL/runtime 设计为可独立使用。

**长期愿景**：成为 Rust 生态的 GPU 算子事实标准，在类型安全、人体工学、IDE 支持上**全面超越** Python Triton。

---

## 2. 核心架构决策（不可逆）

### 2.1 ABI 边界 = MLIR 文本字符串

```
┌───────────────────────────────────────────┐
│  Rust DSL: #[triton_kernel] fn vec_add()  │
└─────────────────────┬─────────────────────┘
                      │ proc-macro 编译期生成
                      ▼
┌───────────────────────────────────────────┐
│  MLIR 文本字符串                            │ ◄── 稳定 ABI 边界
│  "module { tt.func @kernel(...) ... }"    │
└─────────────────────┬─────────────────────┘
                      │ FFI: triton_compile_mlir(text, target_arch)
                      ▼
┌───────────────────────────────────────────┐
│  C ABI shim (~300 行 C++)                  │
└─────────────────────┬─────────────────────┘
                      │ C++ API 调用
                      ▼
┌───────────────────────────────────────────┐
│  Triton C++ libs (vendor, pin v3.6.0)      │
│  Triton IR → TritonGPU IR → LLVM IR → PTX │
└─────────────────────┬─────────────────────┘
                      │
                      ▼
        cubin / hsaco / SPIR-V binary
                      │
                      ▼ load + launch
┌───────────────────────────────────────────┐
│  triton-runtime (Rust, cudarc/hip/ze)     │
└───────────────────────────────────────────┘
```

**为什么以 MLIR 文本作 ABI**：
- Triton 自己的回归测试就是 `.mlir` 文件（`test/Triton/*.mlir`），上游不敢乱改文本语法
- MLIR 项目对 op 的 printer/parser 对称性有保证
- 文本是字符串，跨语言、跨进程、跨版本最稳
- 不需要绑 C++ dialect builder，避开 name mangling、template、虚函数等坑

**反例（不采用）**：直接绑 Triton C++ 的 `tt::LoadOp` builder。Triton 内部 C++ API 不稳定，每次升级都得跟。

### 2.2 Triton 版本 pin 策略

- **当前 pin: `v3.6.0`**（2025-01-21 release）
- 以 git submodule 形式 vendor 在 `crates/triton-sys/vendor/triton/`
- **不 fork、不 patch**——只用上游 tag，避免分叉维护
- 每 6 个月评估一次升级（跟 Triton release 节奏）

### 2.3 C ABI Shim 极薄原则

shim 只暴露 **5 个稳定 C 函数**，永远不暴露 dialect op builder：

```c
TritonContext* triton_context_create(void);
void           triton_context_destroy(TritonContext*);
TritonResult*  triton_compile_mlir(TritonContext*, const char* mlir_text,
                                    const TritonCompileOptions*);
void           triton_result_destroy(TritonResult*);
const char*    triton_get_version(void);
```

- 表面积越小，跟 Triton 上游升级越稳
- 错误用 `TritonResult` 结构体传，不 throw 跨 FFI

### 2.4 多后端通过 target_arch 字符串延续

Triton 后端支持 NVIDIA / AMD / Intel XPU。Rust 端**不重做这套 dispatch**，只把 `target_arch` 透传给 shim：

| 后端 | target_arch 示例 | Runtime 实现 crate |
|---|---|---|
| NVIDIA | `"sm_80"`, `"sm_89"`, `"sm_90a"` | `triton-runtime` (cudarc) — v0.1 |
| AMD | `"gfx90a"`, `"gfx942"` | `triton-runtime` (hip-rs) — v0.5 |
| Intel XPU | `"xelp"`, `"pvc"` | `triton-runtime` (level-zero) — v0.6 |

`triton-runtime` 内 `DeviceRuntime` trait 抽象，先实现 CUDA。

### 2.5 编译期/运行期 Python 依赖：双零

- **编译期**：`cargo build` 不需要 Python（shim cmake 编 Triton C++ libs，全程 C++）
- **运行期**：单二进制部署，零 Python

这与 ferrum-infer-rs 的 "Single binary, no Python" 一致。

---

## 3. DSL 设计目标（核心差异化）

Rust DSL 不是 "Rust 版 @triton.jit"，而是**修正 Python Triton 已知不足、利用 Rust 类型系统/工具链/借用检查的下一代 GPU DSL**。

### 3.1 类别 A：类型系统（最大改进）

| Python Triton 不足 | Rust DSL 解法 | 优先级 |
|---|---|---|
| `tl.constexpr` 是装饰约定，运行时报错 | `const` 泛型：`fn k<const BLOCK: usize>(...)` | v0.1 |
| Tensor shape 是 string `"BLOCK_SIZE"`，编译期不验证 | `Tensor<f32, BLOCK>` const generics | v0.1 |
| Launcher 类型签名 `"*fp16:16, i32"` 是 string | `add::launch(stream, x, y, n)`，rustc 检查 | v0.1 |
| `tl.float16` vs `tl.fp16` 拼写易错 | Rust type alias，拼错编不过 | v0.1 |
| 隐式类型转换混乱 | 显式 `cast::<f32>()` + `From`/`Into` | v0.1 |

### 3.2 类别 B：错误信息和调试

| 不足 | 解法 | 优先级 |
|---|---|---|
| 编译错误指 IR，源码行号丢 | proc-macro `Span` 精准映射 | v0.1 |
| Shape mismatch GPU 上才挂 | 编译期 shape check | v0.1 |
| 无 host-side 调试 | **CPU emulation backend**（dry-run IR + cargo test） | v0.1（CI 强依赖） |
| `tl.device_print` 输出受限 | 接 tracing，结构化 log | v0.2 |

### 3.3 类别 C：IDE / 工具链

| 不足 | 解法 | 优先级 |
|---|---|---|
| `@triton.jit` 内代码 pyright 不分析 | `#[triton_kernel]` 内合法 Rust，rust-analyzer 全功能 | v0.1（自动获得） |
| 自动补全差 | LSP go-to-def / rename 全 work | v0.1（自动获得） |
| 重构跨 kernel 容易漏 | Cargo + rust-analyzer 跨 crate 重构 | v0.1（自动获得） |

### 3.4 类别 D：Kernel 编写人体工学

| 不足 | 解法 | 优先级 |
|---|---|---|
| Boundary mask 到处手写 | `Tile::load(ptr).with_auto_mask(N)` | v0.2 |
| Pointer 算术模板代码多 | 重载 `+`，`x_ptr + offsets` 直接生效 | v0.2 |
| GEMM GROUP_M 重排手算 | `tile_swizzle::<GROUP_M>(pid)` 标准库 | v0.2 |
| `cdiv` / 整除手算 | `usize::div_ceil` Rust 内置 | v0.1（自动） |
| Mask 写法重复 | `with mask(...) { ... }` 上下文 | v0.2 |

### 3.5 类别 E：Autotuning

| 不足 | 解法 | 优先级 |
|---|---|---|
| `configs=[Config({...}), ...]` 手写一堆 | `#[autotune(BLOCK = 64..=1024 step 64, NUM_WARPS = [4,8])]` | v0.2 |
| Cache key 函数手写 | proc-macro 自动派生 | v0.2 |
| 不能继承 config | `#[autotune(preset = gemm_preset())]` | v0.3 |
| Autotune 输出无格式 | 结构化结果 + Criterion 风格报告 | v0.2 |

### 3.6 类别 F：模块化和复用

| 不足 | 解法 | 优先级 |
|---|---|---|
| Kernel 间互调限制多 | Rust 函数互调，trait/泛型完整 | v0.3 |
| 没有跨文件 kernel 模板复用 | crate.io 生态 | v0.3 |
| 第三方算子库碎片化 | Cargo 标准化依赖 | v0.3 |

### 3.7 类别 G：内存安全（Rust 独有）

| 不足 | 解法 | 优先级 |
|---|---|---|
| Shared memory 生命周期手管 | `SharedMem<'_, f32, 1024>` 借用检查 | v1.0 |
| Mutable aliasing 不防范 | `&mut Tile` 借用挡掉 | v1.0 |
| 共享内存超额编不知道 | 编译期累加 SMEM 用量 + lint | v0.3 |

### 3.8 类别 H：部署和运行时

| 不足 | 解法 | 优先级 |
|---|---|---|
| 部署带 Python+Triton+LLVM ~500MB+ | 静态链接，单二进制 | v0.1（架构决策） |
| 启动慢（Py import + Triton init） | Rust 毫秒级启动 | v0.1（自动） |
| GIL 限制多线程编译 | tokio + rayon 并行 | v0.2 |

### 3.9 类别 I：可观测性

| 不足 | 解法 | 优先级 |
|---|---|---|
| Bank conflict 不警告 | proc-macro lint pass | v1.0 |
| SMEM 用量看 PTX 才知道 | 编译期算 + cargo metric | v0.3 |
| Occupancy 估算工具弱 | 内置 calculator，autotune 输出 | v0.3 |

### 3.10 Rust 独有的全新特性

#### 可组合算子代数（v1.0）
```rust
let result = tile.load(x)
    .map(|v| v * scale)
    .dot(tile.load(y))
    .with_mask(boundary)
    .store(out);
// proc-macro 把整条链 fuse 成单个 kernel
```

#### 类型化 Pipeline（v1.0）
```rust
let buf_a = pipeline.async_load(a_ptr).await;
let buf_b = pipeline.async_load(b_ptr).await;
let acc = matmul(buf_a, buf_b);
```

#### Backend-conditional 优化（v0.3）
```rust
#[triton_kernel]
fn matmul(...) {
    #[cfg(target_arch = "sm_90a")]
    use_wgmma();
    #[cfg(target_arch = "sm_80")]
    use_mma_m16n8k16();
}
```

#### Kernel 当 crate 发布（v0.3+）
```toml
[dependencies]
flash-attn-rs = "1.0"
```

---

## 4. Workspace 结构

```
triton-rs/
├── Cargo.toml                        # workspace
├── ARCHITECTURE.md                   # 本文档
├── README.md
├── .github/workflows/
│   ├── ci.yml                        # Layer 1+2: free runner + cuda docker
│   └── gpu-e2e.yml                   # Layer 3: self-hosted GPU
├── crates/
│   ├── triton-sys/                   # FFI 绑定 + cmake 编 Triton + shim
│   │   ├── vendor/triton/            # git submodule, pin v3.6.0
│   │   ├── shim/
│   │   │   ├── triton_c.h            # 5 个 C entry point
│   │   │   └── triton_c.cpp          # ~300 行 C++ wrapper
│   │   ├── build.rs
│   │   └── src/lib.rs
│   ├── triton-ir/                    # MLIR 文本 builder（纯 Rust，零 FFI）
│   │   └── src/
│   │       ├── module.rs             # Module / Func / Block / Region
│   │       ├── op.rs                 # Op trait + builder
│   │       ├── ty.rs                 # 类型系统
│   │       ├── value.rs              # SSA 值管理
│   │       └── dialect/
│   │           ├── arith.rs          # arith dialect
│   │           ├── tt.rs             # Triton dialect
│   │           └── scf.rs            # scf dialect
│   ├── triton-runtime/               # cubin 加载 + launch + DeviceRuntime trait
│   │   └── src/
│   │       ├── runtime.rs            # DeviceRuntime trait
│   │       ├── cuda.rs               # CudaRuntime (cudarc)
│   │       ├── kernel.rs             # Kernel handle
│   │       └── buffer.rs             # DeviceBuffer<T>
│   ├── triton-dsl/                   # proc-macro: #[triton_kernel]
│   │   ├── src/lib.rs                # proc-macro 入口
│   │   └── codegen/
│   │       ├── parse.rs              # syn AST 解析
│   │       ├── infer.rs              # 类型推断
│   │       └── emit.rs               # → triton-ir builder calls
│   └── triton-core/                  # 用户 facade，re-export 常用 API
│       └── src/lib.rs
├── examples/
│   └── vector_add/                   # 端到端 PoC: hello-world kernel
└── tests/
    └── e2e/                          # GPU 端到端测试
```

**Crate 依赖关系**：
```
triton-core ─► triton-dsl, triton-runtime, triton-ir
triton-dsl  ─► triton-ir
triton-ir   (纯 Rust，无外部依赖)
triton-sys  (cmake + bindgen, 独立)
triton-runtime ─► triton-sys, cudarc
```

**为什么 `triton-ir` 独立**：proc-macro 只生成 IR builder calls，不直接接触 FFI。这样 IR builder 能在 GitHub free runner 上单独测试（不需要编 Triton）。

---

## 5. CI 策略（分层）

### Layer 1: GitHub free runner（每个 PR 都跑）

```yaml
jobs:
  cpu-quality-gate:    # ubuntu-latest
    - cargo fmt --check
    - cargo clippy --workspace
    - cargo check --workspace
    - cargo test --workspace --no-default-features
      # 跑 IR generation 测试 + proc-macro 单元测试 + CPU emulation kernel 测试
```

### Layer 2: CUDA build-only（每个 PR）

```yaml
  cuda-build:          # nvidia/cuda docker container
    - cargo check --workspace --features cuda
    - cargo build -p triton-sys --features cuda  # 编 Triton + shim（漫长）
```

### Layer 3: Self-hosted GPU runner（标 `gpu` label / merge 前）

```yaml
  cuda-e2e:            # self-hosted GPU (RTX PRO 6000 Blackwell)
    - cargo test --workspace --features cuda --test '*_e2e'
    - cargo bench --workspace --features cuda
```

### Layer 4: nightly / 手动

- Autotune sweep
- 多 GPU arch 矩阵（sm_80 / sm_89 / sm_90a）
- Stress test

### Test 分类约定

```rust
#[test]
fn ir_generation_correct() { ... }   // CPU，free runner

#[test]
#[cfg(feature = "cuda")]
fn vector_add_e2e() { ... }          // GPU，self-hosted only
```

**关键含义**：`triton-dsl` 的 **CPU emulation backend 是 v0.1 必须**——没有它，Layer 1 CI 只能测 IR generation，无法验证 kernel 语义正确性。

---

## 6. 路线图

| Phase | 名称 | 估时 | 交付物 |
|---|---|---|---|
| **0** | **项目骨架（本阶段）** | 1 周 | workspace + ARCHITECTURE + 5 crate 空壳 |
| 1 | C ABI Shim + Runtime | 4-6 周 | shim 编出 cubin、cudarc 加载跑通 vector_add |
| 2 | Rust IR Builder | 4-6 周 | 手写 IR 能产出和 Python Triton 等价的 kernel |
| 3 | proc-macro DSL（v0.1） | 6-8 周 | `#[triton_kernel]` 可用，CPU emulation backend |
| 4 | Autotuning + 人体工学（v0.2） | 4-6 周 | `#[autotune]`、自动 mask、tile 库 |
| 5 | 多后端 AMD/Intel | 6-8 周 | `target_arch="gfx942"` 通路 |
| 6 | v1.0：可组合算子 + 借用检查 | 持续 | 算子代数、内存安全、生态 |

---

## 7. 不做的事（明确边界）

- ❌ **不重写 Triton 编译器后端**（vendor + pin 即可）
- ❌ **不绑 Triton C++ dialect builder**（走 MLIR 文本）
- ❌ **不依赖 Python**（编译期/运行期都不要）
- ❌ **不做 PyTorch 集成**（需要的话另起 crate，不在主项目）
- ❌ **不追 Triton main 分支**（pin LTS-like release，6 月升级一次）

---

## 8. 参考资料

- Triton 项目: https://github.com/triton-lang/triton
- MLIR 项目: https://mlir.llvm.org/
- ferrum-infer-rs: 同级目录，第一服务对象，CI 模式参考
- cudarc: https://github.com/coreylowman/cudarc
