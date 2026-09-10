# Phono

**A high-throughput, bare-metal 3D quantum wave-function simulation engine.**

Phono integrates the Time-Dependent Schrödinger Equation on a uniform Cartesian
grid using the **implicit Crank–Nicolson method**, factored into per-axis Cayley
transforms and solved by the Thomas algorithm. Orchestration, sharding and
lifetime management live in Go; every floating-point operation on the stepping
path executes in optimised C++17. The FFI layer is not hand-written — the
`extern "C"` ABI and the cgo wrapper are generated ahead of time from the C++
AST by the [`cppgo`](https://github.com/ekyboi/cppgo) toolchain, so the boundary
cannot drift out of sync with the core.

The engine allocates **zero bytes on the Go heap per timestep**, spawns no
threads of its own, and conserves probability to rounding error rather than to
truncation order.

```
64³ grid, Apple M-series, 10 cores, go1.27 / Apple clang 21
  serial   15.06 ms/step
  10-way    4.10 ms/step   (3.7×)
  norm drift over 400 steps:  -6.9e-13
  allocations per step:        0
```

---

## Table of Contents

- [Systems Architecture & Memory Topology](#systems-architecture--memory-topology)
- [Theoretical Physics & Calculus Foundations](#theoretical-physics--calculus-foundations)
- [Directory Layout & Compilation Pipeline](#directory-layout--compilation-pipeline)
- [Quick Start](#quick-start)
- [Regenerating the Bridge](#regenerating-the-bridge)
- [Verification](#verification)
- [Performance Notes](#performance-notes)
- [Known Limitations](#known-limitations)

---

## Systems Architecture & Memory Topology

### The wavefunction never enters the Go heap

The C++ object is allocated on the native heap and reached exclusively through
an opaque, pointer-sized token:

```c
typedef void* PhonoEngineHandle;
```

Go holds that token and nothing else. The collector therefore never scans,
traces into, or relocates a single byte of simulation state — there are no Go
pointers inside it to scan, and the memory is not part of any Go span.

State lives in six `std::vector<double>` containers sized `N = Width·Height·Depth`
(`psi_real`, `psi_imag`, `potential`, `density`, and a precomputed
`cos`/`sin` phase table). They are allocated once, in `Initialize()`, and never
resized, which is what makes their `.data()` addresses stable for the lifetime
of the engine.

### The escape hatch: `unsafe.Slice` over unmanaged memory

Those addresses cross the ABI and are re-materialised as Go slice headers that
alias the C++ storage directly:

```go
func view(addr uint, n int) []float64 {
	if addr == 0 || n <= 0 {
		return nil
	}
	ptr := C.phono_addr_to_ptr(C.size_t(addr))
	return unsafe.Slice((*float64)(unsafe.Pointer(ptr)), n)
}
```

`unsafe.Slice` fabricates the three-word header in the caller's frame; the
backing store is the C++ vector itself. Reading or writing `eng.PsiReal()`
touches native memory with no copy, no bounds-check indirection, and no
allocation.

> **Why the address crosses as `size_t` rather than `double*`.** `cppgo`'s type
> registry has no mapping whose Go surface is a pointer, and both of the
> generator's pointer-shaped paths are structurally closed: `goZeroValue` would
> emit `unsafe.Pointer(0)` on the closed-handle guard, which is not a legal
> conversion in Go, and the default return path would emit `*float64(...)`,
> which does not parse. Registering a `double*` mapping is not sufficient — it
> would require a new `TypeKind` inside the generator. Crossing as `size_t`
> uses a mapping already present in the default registry and survives both
> emitters untouched.
>
> The integer-to-pointer conversion is then performed **in C**, inside the cgo
> preamble, so the construct `unsafe.Pointer(uintptr(x))` never appears in Go
> and `go vet`'s `unsafeptr` analyser stays quiet. That analyser is right to be
> suspicious in general — the collector may move what a `uintptr` referred to.
> Here it cannot: the memory belongs to a C++ vector the Go runtime has never
> heard of.

### Lifetime safety across the FFI boundary

Two mechanisms, addressing two genuinely different failure modes:

**`sync.RWMutex`** guards the handle field. Every generated method takes
`RLock`; `Close()` takes the write lock, destroys the native object, nils the
handle, and retracts the finalizer. Readers proceed concurrently — which is
exactly what the sharded driver needs, since all workers call engine methods
simultaneously — while a `Close` racing an in-flight call is serialised into
either "before" (method sees a live handle) or "after" (method sees `nil` and
returns the zero value). This is what makes `Close()` idempotent and prevents a
double `delete`.

**`runtime.KeepAlive(recv)`** addresses a subtler hazard, and it is worth being
precise about what it actually does. It does *not* stop the GC from reclaiming
the native allocation — the GC has no knowledge of that memory. It keeps the Go
*wrapper struct* reachable across the cgo call. Without it, the compiler may
observe that the receiver's last use is the handle load at the top of the
method; the wrapper becomes unreachable at that instant, the finalizer
installed by the constructor becomes eligible to run, and `delete` executes on
the C++ object **while C++ is still inside the call**. Every instance method
therefore carries:

```go
defer runtime.KeepAlive(p)
```

### Cache topology and the unit-stride axis

The 3D grid is flattened with **x as the unit-stride axis**:

```
index = (z · Height + y) · Width + x
```

Consecutive `x` are consecutive `double`s in memory. What that buys differs by
phase, and the distinction matters:

| Phase | Access pattern | Where the parallelism comes from |
|---|---|---|
| Potential half-step, density | Flat sequential over `N` | Trivially vectorised; pure streaming, prefetcher-ideal |
| **X sweep** | One contiguous line per `(y,z)` | **Not SIMD** — the Thomas recurrence is serial along the line. Contiguity buys prefetch and full cache-line utilisation; throughput comes from independent lines across shards |
| **Y / Z sweeps** | Recurrence along a strided axis | **SIMD** — `kLaneWidth = 8` adjacent `x` values are carried through the recurrence together, so the inner loop over lanes is unit-stride and dependency-free |

The naive reading — "x innermost, therefore everything vectorises" — is wrong
for exactly the phases that dominate the runtime. A tridiagonal solve is a
loop-carried dependency; no compiler can vectorise *along* the direction of the
sweep. Vector parallelism has to be harvested from the **orthogonal** axis, and
the layout is what makes that possible: when sweeping y or z, the eight
neighbouring `x` values needed for a vector register are already contiguous.
Solving one strided line at a time would instead touch one `double` per
64-byte cache line and waste 87.5% of every fetch.

Boundary rows are peeled out of the interior loop via template parameters
(`RhsRow<HasPrev, HasNext>`) so the hot body contains no conditionals at all:

```cpp
template <bool HasPrev, bool HasNext>
inline void RhsRow(const double* __restrict u, const double* __restrict v, ...) {
    for (int l = 0; l < lanes; ++l) {
        const double um = HasPrev ? u[idx + l - stride] : 0.0;
        ...
    }
}
```

On arm64 this lowers to NEON; on x86-64 to AVX2/AVX-512 depending on `-march`.

### False sharing

Each worker receives a private scratch arena for the Thomas forward sweep,
allocated with over-aligned nothrow `operator new` and **padded to 128 bytes**
(the larger of the x86-64 and Apple-silicon cache-line sizes):

```cpp
scratch_ = static_cast<double*>(::operator new(
    total, std::align_val_t(kCacheLineBytes), std::nothrow));
```

`std::vector` guarantees no alignment beyond `alignof(double)`, so the arenas
deliberately do not use it. Worker `i` is the only thread that ever touches
arena `i`. Without the padding, two arenas could share one line and every write
by one core would invalidate the other's copy — the classic silent 10×
slowdown that profiles as memory stalls with no obvious cause.

### Concurrency model

The C++ core **spawns no threads**. It exposes each phase of a timestep as a
range-sharded entry point and leaves scheduling to Go:

```
PrepareStep(dt)                    ← serial; rebuilds factor + phase tables
ApplyPotentialHalfStep(begin, end) ← sharded over [0, Size)
SweepX(worker, begin, end)         ← sharded over [0, XLineCount)
SweepY(worker, begin, end)         ← sharded over [0, YBlockCount)
SweepZ(worker, begin, end)         ← sharded over [0, ZBlockCount)
```

Within a phase, units of work are provably disjoint — two x-lines share no grid
point. Across phases they are not: the x sweep reads what the potential
half-step wrote. **Every phase boundary is a full barrier**, and there are seven
of them per step. `Driver` maintains a persistent goroutine pool parked on
channel receives, so a step costs seven send-and-join rounds rather than
`7 × workers` goroutine spawns, and precomputes all shard boundaries at
construction — hence zero allocations per step on the Go side as well.

---

## Theoretical Physics & Calculus Foundations

### The equation

```
                ∂Ψ         ħ²
           iħ ───── =  − ────── ∇²Ψ  +  V(x,y,z)·Ψ  ≡  ĤΨ
                ∂t         2m
```

`Ĥ` is Hermitian, so the exact propagator `U(dt) = exp(−iĤ·dt/ħ)` is **unitary**
and `⟨Ψ|Ψ⟩` is conserved. Since `|Ψ|²` is a probability density, a scheme that
leaks norm is reporting that the particle is ceasing to exist. Any acceptable
integrator must inherit unitarity.

### Why explicit methods are unusable here

Forward Euler gives `Ψⁿ⁺¹ = (1 − iĤ·dt/ħ)Ψⁿ`. For an eigenvalue `E` of `Ĥ`, the
amplification factor is

```
|1 − iE·dt/ħ|  =  √(1 + (E·dt/ħ)²)  >  1     for every E ≠ 0, every dt
```

Every mode grows. This is **unconditionally unstable** — unlike a CFL condition,
no reduction of `dt` rescues it, because the defect is in the modulus, not the
step size.

### Crank–Nicolson: the trapezoid rule on the time derivative

Averaging `Ĥ` across the step:

```
     Ψⁿ⁺¹  =  Ψⁿ  −  (i·dt / 2ħ) · Ĥ (Ψⁿ + Ψⁿ⁺¹)
```

Collecting unknowns on the left yields the **Cayley transform**:

```
     ( I  +  (i·dt/2ħ)·Ĥ ) Ψⁿ⁺¹   =   ( I  −  (i·dt/2ħ)·Ĥ ) Ψⁿ        (★)
```

Its amplification factor is

```
     1 − iE·dt/2ħ
     ─────────────        a complex number over its own conjugate
     1 + iE·dt/2ħ
```

whose modulus is **exactly 1** for every `E` and every `dt`. The scheme is
unconditionally stable *and* exactly unitary — norm is preserved to rounding
error, not merely to truncation order. Expanding both sides against
`exp(−iĤ·dt/ħ)` shows agreement through `dt²`, so it is second-order accurate
in time.

The cost is the word *implicit*: (★) is a linear **system**. In 3D the operator
`I + (i·dt/2ħ)Ĥ` is `N × N` with `N = W·H·D` and a seven-point stencil.

### Splitting a 3D implicit solve into 1D tridiagonal solves

Decompose the Hamiltonian along the axes:

```
     Ĥ  =  V̂  +  T̂ₓ  +  T̂_y  +  T̂_z        T̂_d  =  −(ħ²/2m) ∂²/∂d²
```

Each piece is separately Hermitian, so each Cayley transform is separately
unitary. Compose them **symmetrically** (Strang splitting):

```
     U(dt)  ≈  V(dt/2) · X(dt/2) · Y(dt/2) · Z(dt) · Y(dt/2) · X(dt/2) · V(dt/2)
```

The palindrome is load-bearing. A palindromic composition of second-order
factors is itself second-order: the first-order commutator error `[A,B]dt²/2`
generated by the forward half cancels against the one generated by the reverse
half. Applying X, Y, Z once each in fixed order — the naive ADI arrangement —
leaves that term uncancelled and **silently degrades the scheme to first
order**. Because every factor is unitary, the product is unitary: the splitting
costs accuracy, never stability and never norm.

The payoff is that `T̂_d` involves one axis only, so its Cayley system decouples
into independent tridiagonal systems — one per grid line along that axis.

### Spatial discretisation: central finite differences

The second derivative is approximated by the central difference on a uniform
grid of spacing `h`:

```
     ∂²Ψ │       Ψⱼ₋₁ − 2Ψⱼ + Ψⱼ₊₁
     ────│    =  ────────────────────  +  O(h²)
     ∂x² │ⱼ              h²
```

Substituting into (★) with sub-step `τ` and defining

```
     r  =  τ · ħ / (4 · m · h²)
```

row `j` of the system becomes

```
   −ir·Ψ*ⱼ₋₁  +  (1 + 2ir)·Ψ*ⱼ  −  ir·Ψ*ⱼ₊₁   =   Ψⱼ + ir(Ψⱼ₋₁ − 2Ψⱼ + Ψⱼ₊₁)
   └──────────────── implicit LHS ────────────┘   └────── explicit RHS ──────┘
```

A **constant-coefficient complex tridiagonal system**: `a = c = −ir`,
`b = 1 + 2ir`. Dirichlet walls place zero ghost values at `j = −1` and `j = n`,
so all `n` points are unknowns. Splitting into real and imaginary parts
(`u`, `v`) with `Lu_j = u_{j−1} − 2u_j + u_{j+1}`:

```
     d_re,j  =  u_j  −  r · Lv_j
     d_im,j  =  v_j  +  r · Lu_j
```

### Thomas, with the factorisation hoisted out of the loop

```
  forward:   w₀ = 1/b,              cp₀ = c·w₀,           dp₀ = d₀·w₀
             wⱼ = 1/(b − a·cpⱼ₋₁),  cpⱼ = c·wⱼ,           dpⱼ = (dⱼ − a·dpⱼ₋₁)·wⱼ
  backward:  xₙ₋₁ = dpₙ₋₁,          xⱼ = dpⱼ − cpⱼ·xⱼ₊₁
```

Because the grid is uniform, `a`, `b`, `c` depend on neither `j` nor which line
is being solved. The elimination factors `{wⱼ, cpⱼ}` therefore depend **only on
the axis length and `r`** — computed once per `PrepareStep()` and reused across
all `H·D` lines of that axis. The per-line cost collapses to two multiply-add
passes.

### The diagonal factor

`V̂` is diagonal, so its propagator is an exact pointwise phase rotation
`exp(−iV·τ/ħ)` — no linear solve, no Cayley approximation, exactly unitary. For
a static potential and fixed `dt` the angle is constant in time, so `cos` and
`sin` are tabulated in `PrepareStep()` and the half-step reduces to a 2×2 plane
rotation per grid point with **zero transcendentals on the hot path**.

### On "conserved forever"

Unitarity means there is **no systematic drift** — no secular decay or growth,
regardless of `dt`. The residual is floating-point rounding, which accumulates
as a random walk (`~√steps`), not as a trend. Measured drift is `−6.9e-13` over
400 steps at 32³. This is a strictly stronger guarantee than "stable": a
dissipative-but-stable scheme would converge to zero without ever blowing up.

---

## Directory Layout & Compilation Pipeline

```
Phono/
├── PhonoEngine.h          ── hand-written C++17 core: class declaration,
├── PhonoEngine.cpp           full Crank–Nicolson derivation, Thomas solves,
│                             blocked SIMD sweeps, cache-aligned arenas
│
├── bridge.h               ── GENERATED by cppgo — flattened extern "C" ABI,
├── bridge.cpp                opaque handle typedef, nothrow trampolines
├── bindings.go            ── GENERATED by cppgo — cgo wrapper struct,
│                             RWMutex, finalizer, KeepAlive on every method
│
├── telemetry.go           ── hand-written — unsafe.Slice views over native
│                             memory; package-wide CXXFLAGS
├── driver.go              ── hand-written — persistent worker pool, shard
│                             tables, seven-phase barriered timestep
│
├── phono_test.go          ── correctness, concurrency, allocation, regression
├── go.mod                 ── module github.com/ekyboi/phono, package phono
├── .gitignore             ── C++ build products from the static-library route
└── README.md
```

> **The three generated files carry a `DO NOT EDIT` banner and are rewritten in
> full by `cppgo generate`.** Anything hand-authored inside them is lost on the
> next regeneration — which is precisely why `telemetry.go` and `driver.go`
> exist as separate files rather than as additions to `bindings.go`. Package-wide
> `#cgo` directives also live in `telemetry.go` for the same reason: cgo
> concatenates the directives of every file in a package, so flags placed there
> survive regeneration.

### Pipeline

```
   PhonoEngine.h
        │
        │  clang -Xclang -ast-dump=json      (cppgo frontend)
        ▼
   CppClass meta-tree
        │
        │  type registry: C++ type → C ABI type → cgo type → Go type
        ▼
   ┌────────────┬─────────────┬──────────────┐
   │  bridge.h  │ bridge.cpp  │  bindings.go │
   └────────────┴─────────────┴──────────────┘
        │
        │  go build  (cgo invokes clang++ on every .cpp in the package)
        ▼
   linked test binary / application
```

### Default build — in-tree cgo compilation

`cgo` compiles every `.cpp` in the package directory itself. There is no
separate library step and no `LDFLAGS` to configure:

```bash
go build ./...
go test ./...
```

The go tool prepends `-O2 -g` to every cgo C++ compile on its own; `telemetry.go`
raises that to `-O3 -DNDEBUG`. Confirm what actually reaches the compiler with:

```bash
go build -a -x -o /dev/null . 2>&1 | grep PhonoEngine.cpp
# ... c++ -I . -fPIC -arch arm64 ... -O2 -g -std=c++17 -O3 -DNDEBUG ... -c PhonoEngine.cpp
```

`-ffast-math` is **deliberately excluded**. It licenses reassociation of
floating-point sums, which would break the bit-for-bit agreement between the
serial and sharded step paths that `TestDriverMatchesSerial` asserts, and buys
nothing in the tridiagonal solves that dominate runtime. (cgo enforces a flag
allowlist regardless — most `-f` options, `-fno-math-errno` among them, are
rejected outright.)

### Alternative build — prebuilt static archive

Useful when the C++ core is shared with a non-Go consumer, or to keep clang out
of the Go build cycle. **The `.cpp` files must be moved out of the package
directory**, or their symbols will be defined twice:

```
phono/
├── native/           PhonoEngine.{h,cpp}, bridge.{h,cpp}
├── libphono.a
└── *.go
```

```bash
cd native
c++ -std=c++17 -O3 -DNDEBUG -fPIC -c PhonoEngine.cpp bridge.cpp
ar rcs ../libphono.a PhonoEngine.o bridge.o
ranlib ../libphono.a
```

Then, in `telemetry.go`'s preamble (not `bindings.go` — it is regenerated):

```go
/*
#cgo CPPFLAGS: -I${SRCDIR}/native
#cgo LDFLAGS:  -L${SRCDIR} -lphono -lc++
#include <stddef.h>
#include "bridge.h"
*/
import "C"
```

Two details that are easy to get wrong here:

- **`-lc++` becomes mandatory.** The go tool appends the platform C++ runtime
  automatically *only* for packages containing `.cpp` files. Once they are moved
  out, nothing pulls in libc++ and the link fails on `std::nothrow`,
  `std::terminate()` and the `std::length_error` machinery. On a GNU toolchain
  the equivalent flag is `-lstdc++`.
- **`go build ./...` will not catch a link error** in a library package, because
  it never runs the linker. Verify with something that produces a binary:
  `go test -c -o /dev/null .`

For a shared object, substitute `-shared -o libphono.so` (`.dylib` on macOS) and
ensure the loader can find it at runtime (`-Wl,-rpath,${SRCDIR}`).

---

## Quick Start

```bash
go get github.com/ekyboi/phono
```

```go
package main

import (
	"fmt"
	"log"

	"github.com/ekyboi/phono"
)

func main() {
	const n = 64
	const h = 0.3

	eng, err := phono.NewPhonoEngine(n, n, n, h, h, h, 1.0, 1.0, 16)
	if err != nil {
		log.Fatal(err)
	}
	defer eng.Close()

	// The C++ constructor is allocation-free and noexcept; every buffer is
	// acquired here. Until this returns true the engine no-ops silently.
	if !eng.Initialize() {
		log.Fatal("phono: native buffers could not be allocated")
	}

	// Write the harmonic trap straight into the native potential buffer.
	c := float64(n+1) * h / 2
	v := eng.Potential()
	for z := 0; z < n; z++ {
		for y := 0; y < n; y++ {
			for x := 0; x < n; x++ {
				dx := float64(x+1)*h - c
				dy := float64(y+1)*h - c
				dz := float64(z+1)*h - c
				v[eng.Index(x, y, z)] = 0.5 * (dx*dx + dy*dy + dz*dz)
			}
		}
	}
	eng.MarkPotentialDirty() // invalidates the cached phase table

	eng.SetGaussianPacket(c, c, c, 1.0, 1.0, 1.0, 2.0, 0, 0)

	drv, err := phono.NewDriver(eng, 0) // 0 selects GOMAXPROCS
	if err != nil {
		log.Fatal(err)
	}
	defer drv.Close() // LIFO: the driver stops before the engine is freed

	fmt.Printf("norm before : %.15f\n", eng.Norm())
	drv.Advance(0.002, 500)
	fmt.Printf("norm after  : %.15f\n", eng.Norm())

	drv.ComputeDensity()
	d := eng.Density()
	fmt.Printf("centre |psi|^2 = %.6e\n", d[eng.Index(n/2, n/2, n/2)])
}
```

```
norm before : 0.999999999999960
norm after  : 0.999999999999766
centre |psi|^2 = 1.784026e-02
```

### Lifetime contract for native views

The slices returned by `PsiReal()`, `PsiImag()`, `Potential()` and `Density()`
alias C++ memory. They are **not** garbage collected and **not** tied to the
wrapper's lifetime. A view becomes a dangling pointer the instant `Close()` runs
— or the instant the finalizer runs on a dropped wrapper — and writing through a
stale view corrupts the heap silently.

> **Rule:** take the view, use it, drop it. Never cache one in a long-lived
> struct, and never let a view outlive the engine that produced it.

### Units

Natural units are not assumed; `ħ` and `m` are constructor parameters. Grid
point `i` on an axis sits at `(i+1)·h`, placing the Dirichlet walls in the ghost
slots at `−1` and `n`, so the physical box spans `[0, (n+1)·h]`.

---

## Regenerating the Bridge

After any change to the public surface of `PhonoEngine.h`:

```bash
cppgo generate \
  --header PhonoEngine.h \
  --outdir . \
  --class PhonoEngine \
  --package phono \
  --module github.com/ekyboi/phono \
  --include-as PhonoEngine.h \
  --strict
```

`--strict` is not optional in practice. Without it, a method whose signature the
type registry cannot flatten is **skipped with a warning** and simply vanishes
from the Go surface — a failure mode that surfaces later as a confusing
"undefined method" rather than at generation time.

Current status: **31 methods wrappable, 0 skipped, 33 exported C symbols.**
Regeneration is idempotent — it reproduces all three artifacts byte-for-byte.

### Type-registry constraints worth knowing

| Concern | Behaviour |
|---|---|
| Handle typedef | `c.Name + "Handle"` → `PhonoEngineHandle`. The namespace prefix applies to *function symbols*, not the typedef |
| Symbol naming | Only the namespace is snake-cased; method names pass through verbatim → `quantum_physics_PhonoEngine_StepTime` |
| `std::size_t` | **Not registered.** Only unqualified `size_t` resolves. Spelling it `std::size_t` silently skips the method |
| `bool` | Crosses as `unsigned char` (0/1), not `_Bool` |
| Reserved Go names | `Close`, `Valid`, `finalize`, `handle`, `mu` are reserved; a C++ method named `Valid()` would be renamed `Valid2()` in Go — hence the core spells it `Ready()` |

---

## Verification

```bash
go test ./...                                    # correctness + concurrency
go test -race ./...                              # Go-side synchronisation
go test -run xxx -bench . -benchtime 30x         # throughput
```

| Test | What it establishes |
|---|---|
| **Exact eigenmode** | A product of sine modes is an *exact* eigenvector of the discrete Dirichlet Laplacian, so the propagated state must equal the initial state times an analytically-known complex amplification factor. Agreement: **8.5e-15** — machine precision — with anisotropic spacing and non-unit `ħ`/`m` |
| `TestNormConservation` | Drift `−6.9e-13` over 400 steps: unitarity, not slow decay |
| `TestDriverMatchesSerial` | 8 workers reproduce 50 serial steps **bit-for-bit**. Any mismatch would mean two workers touched the same grid point or scratch arena |
| `TestStepDoesNotAllocate` | `testing.AllocsPerRun == 0` on both the serial and sharded paths |
| `TestViewsAliasNativeMemory` | A write through the Go slice changes a norm computed entirely in C++ |
| `TestPotentialTrapsPacket` | A harmonic well confines a packet that otherwise disperses (rms 0.24 vs 0.62) |
| `TestCloseIsIdempotent` | Triple `Close()`, then method calls on a closed wrapper return zero values instead of dereferencing `nil` |
| `TestDegenerateAndPartialLaneDimensions` | Nine shapes including partial-lane widths and single-cell axes |

The race detector instruments Go only — it validates the channel, `WaitGroup`
and `RWMutex` layer. The C++ side's disjointness is what the bit-for-bit test
covers, and memory safety is checked separately:

```bash
c++ -std=c++17 -O1 -g -fsanitize=address,undefined <harness>.cpp PhonoEngine.cpp -o t && ./t
```

> ASan earned its place here. The blocked y/z sweeps peel the first row out of
> the interior loop as `RhsRow<false, true>` — but when an axis has length 1 the
> first row is *also* the last, and the successor read runs off the end of the
> vector. `SweepX` guarded this; the blocked sweeps did not. It fires on any
> `W × H × 1` grid — i.e. **every 2D run**. Fixed, and pinned by
> `TestDegenerateAndPartialLaneDimensions`.

---

## Performance Notes

**Memory footprint** is `48 · N` bytes — six `float64` arrays over
`N = W·H·D`:

| Grid | Cells | Resident |
|---|---|---|
| 64³ | 262,144 | 12 MiB |
| 128³ | 2,097,152 | 96 MiB |
| 256³ | 16,777,216 | 768 MiB |

Per-worker scratch is `padded(max(W,H,D) · 8 lanes · 2) · 8` bytes — 16 KiB per
worker at 128³, negligible against the grid.

**Scaling** is bounded by memory bandwidth, not arithmetic. Each step streams
the full state seven times, and the tridiagonal solves are two multiply-add
passes per line — a low arithmetic intensity that puts the engine firmly on the
bandwidth side of the roofline. The observed 3.7× on 10 cores reflects that
ceiling, along with the seven hard barriers per step. Expect the marginal return
on additional cores to fall off well before core count is exhausted.

Practical consequences:

- `PrepareStep` is `O(N)` but runs its transcendental loop **only** when `dt`
  changes or the potential is marked dirty. Holding `dt` fixed across a run
  reduces it to a pair of comparisons. Varying `dt` every step re-tabulates
  `cos`/`sin` over the whole grid and will dominate.
- Shard remainders are distributed one unit at a time across the leading
  workers, so the longest shard exceeds the shortest by at most one unit. With a
  barrier per phase, a phase costs as much as its slowest worker.
- `Driver` oversubscription is rejected at construction (`ErrDriverWorkers`)
  rather than silently aliasing scratch arenas: worker ids index slabs sized at
  engine construction, so `NewDriver` cannot exceed `MaxWorkers`.

---

## Known Limitations

- **Static potentials only.** `V` is assumed constant in time; the phase table
  is cached across steps. A time-dependent `V(t)` requires calling
  `MarkPotentialDirty()` every step, which re-tabulates `cos`/`sin` over the
  entire grid and forfeits most of the optimisation.
- **Hard-wall boundaries.** Dirichlet walls reflect. A packet reaching the
  boundary interferes with its own reflection; there is no absorbing or
  perfectly-matched layer. Size the box so the wavefunction stays away from the
  edges for the duration of the run.
- **Second-order in space and time.** Discrete dispersion is real and visible:
  the finite-difference group velocity is `(ħ/mh)·sin(kh)` rather than `ħk/m`.
  At `k = 2`, `h = 0.25` that is a 4% deficit — physical for the discretisation,
  not a defect, but it must be accounted for when comparing against continuum
  analytics.
- **`Norm()` and `Normalize()` are single-threaded** and `O(N)`; they are
  diagnostics, not step-path operations.
- **The two-step construction is a footgun.** `NewPhonoEngine` returns a wrapper
  whose C++ object is constructed but *not* initialised. Every method guards on
  a `ready_` flag and silently no-ops until `Initialize()` returns true. This
  shape is forced by the allocation-free `noexcept` constructor that keeps
  `std::bad_alloc` from unwinding through the `extern "C"` frame into the Go
  runtime — an exception crossing that boundary is undefined behaviour.
- **A `PhonoEngine` value must never be copied.** A copy duplicates ownership of
  a handle destroyed exactly once. Always pass the pointer the constructor
  returns.

---

## License

No `LICENSE` file is present in this repository yet. Until one is added, the
work is under exclusive copyright by default — no redistribution or derivative
rights are granted, which is worth resolving before the code is shared or
vendored. The `cppgo` toolchain is licensed separately in
[its own repository](https://github.com/ekyboi/cppgo).
