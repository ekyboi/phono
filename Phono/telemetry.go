// telemetry.go -- zero-allocation views onto the native wavefunction.
//
// HAND-WRITTEN. This file is deliberately NOT part of the cppgo output.
//
// bindings.go carries a "DO NOT EDIT" banner and is rewritten in full on
// every `cppgo generate`, so anything hand-authored there is lost on the
// next regeneration. It also cannot express what this file does: the
// toolchain's type registry has no mapping whose Go surface is a pointer.
// Both of the generator's pointer-shaped paths are closed --
// gobind.goZeroValue would emit `unsafe.Pointer(0)` on the closed-handle
// guard, which is not a legal conversion in Go, and the default return
// path would emit `*float64(...)`, which does not parse. Registering a
// `double*` mapping is therefore not enough; it would need a new TypeKind
// in the generator itself.
//
// So the buffer accessors cross the ABI as `size_t` -- a mapping that is
// already in the default registry and survives both emitters untouched --
// and this file turns those addresses into Go slice headers.
//
// ---------------------------------------------------------------------
// LIFETIME CONTRACT -- read this before using any slice from this file
// ---------------------------------------------------------------------
//
// The returned slices alias memory owned by the C++ std::vectors. They
// are NOT garbage collected, NOT copied, and NOT bounds-tied to the
// wrapper's lifetime. A slice obtained here becomes a dangling pointer
// the instant PhonoEngine.Close() runs, or the instant the finalizer
// runs on a dropped wrapper. Writing through a stale slice corrupts the
// heap silently.
//
// The rule: never let a view outlive the engine, and keep the engine
// reachable for as long as any view derived from it is in use. In
// practice, take the view, use it, drop it -- do not cache it in a
// long-lived struct.

package phono

/*
// Optimisation flags for the whole package.
//
// cgo concatenates the #cgo directives of every file in a package, so
// this block applies to PhonoEngine.cpp and bridge.cpp as well. It lives
// here rather than in bindings.go because cppgo regenerates that file
// wholesale and emits only `-std=c++17`.
//
// Note what the go tool already does on its own: it prepends `-O2 -g` to
// every cgo C++ compile, so the engine is optimised by default and this
// line is a modest bump to -O3 plus the assert removal, not a rescue from
// -O0. Verify with `go build -a -x` and read the actual clang command.
//
// The deliberate omission is -ffast-math. It permits reassociation of
// floating-point sums, which would break the bit-for-bit agreement
// between the serial and sharded step paths that TestDriverMatchesSerial
// depends on, and it buys nothing in the tridiagonal solves that dominate
// the runtime. cgo enforces a flag allowlist here in any case -- most -f
// options, -fno-math-errno among them, are rejected outright.
#cgo CXXFLAGS: -O3 -DNDEBUG

#include <stddef.h>
#include "bridge.h"

// Turns the integer address the bridge returns back into a double*.
//
// This exists so that the integer-to-pointer conversion happens in C,
// where it is well defined, rather than in Go as unsafe.Pointer(uintptr(x))
// -- a construct go vet's unsafeptr analyser flags, because in general the
// collector may have moved whatever the uintptr referred to. Here it never
// can: the memory belongs to a C++ std::vector on the native heap, which
// the Go collector does not know about, never scans and never relocates.
static inline double* phono_addr_to_ptr(size_t addr) {
	return (double*)addr;
}
*/
import "C"

import (
	"runtime"
	"unsafe"
)

// view is the common path: turn a bridge-reported address plus an element
// count into a Go slice header aliasing the native buffer.
//
// No allocation occurs. unsafe.Slice fabricates the header in the
// caller's frame; the backing store is the C++ vector itself, so the Go
// collector never sees, scans, or copies a single one of the Size()
// doubles.
func view(addr uint, n int) []float64 {
	if addr == 0 || n <= 0 {
		return nil
	}
	ptr := C.phono_addr_to_ptr(C.size_t(addr))
	return unsafe.Slice((*float64)(unsafe.Pointer(ptr)), n)
}

// PsiReal returns the real part of the wavefunction as a slice aliasing
// native memory, laid out as index = (z*Height + y)*Width + x.
//
// Writes through the slice are immediately visible to the C++ engine --
// that is the point. Subject to the lifetime contract at the top of this
// file.
func (p *PhonoEngine) PsiReal() []float64 {
	s := view(p.PsiRealAddress(), p.Size())
	runtime.KeepAlive(p)
	return s
}

// PsiImag returns the imaginary part of the wavefunction, aliasing native
// memory. Same layout and lifetime rules as PsiReal.
func (p *PhonoEngine) PsiImag() []float64 {
	s := view(p.PsiImagAddress(), p.Size())
	runtime.KeepAlive(p)
	return s
}

// Potential returns the static potential V, aliasing native memory.
//
// After writing to it, call MarkPotentialDirty so that the next
// PrepareStep rebuilds the cached phase table; otherwise the engine keeps
// propagating under the old potential.
func (p *PhonoEngine) Potential() []float64 {
	s := view(p.PotentialAddress(), p.Size())
	runtime.KeepAlive(p)
	return s
}

// Density returns the |Psi|^2 buffer, aliasing native memory. Its
// contents are whatever the last ComputeDensity call left there.
func (p *PhonoEngine) Density() []float64 {
	s := view(p.DensityAddress(), p.Size())
	runtime.KeepAlive(p)
	return s
}

// Index converts grid coordinates to the flat offset used by every buffer
// this file returns. Bounds are the caller's responsibility.
func (p *PhonoEngine) Index(x, y, z int) int {
	return (z*p.Height()+y)*p.Width() + x
}
