// PhonoEngine.h -- 3D time-dependent Schrodinger propagator.
//
// Hand-written C++ core. This file is the *input* to cppgo; bridge.h,
// bridge.cpp and bindings.go are generated from the AST description in
// cppgo_gen.go and must not be edited by hand.
//
// ---------------------------------------------------------------------------
// Physics
// ---------------------------------------------------------------------------
//
// The engine integrates the time-dependent Schrodinger equation
//
//     i*hbar dPsi/dt = H Psi,     H = -(hbar^2 / 2m) Laplacian + V(x,y,z)
//
// on a uniform Cartesian grid with hard (Dirichlet) walls, using the
// Crank-Nicolson / Cayley scheme in a symmetric Strang splitting. See
// PhonoEngine.cpp for the full derivation; the short version is that each
// factor of the composition is *unitary*, so the discrete norm
// sum |Psi|^2 dV is conserved to rounding error for any dt.
//
// ---------------------------------------------------------------------------
// Memory model
// ---------------------------------------------------------------------------
//
// Every buffer is allocated once, in Initialize(). Nothing on the stepping
// path allocates, frees, throws, or takes a lock. The wavefunction lives in
// two parallel std::vector<double> (real and imaginary parts, structure of
// arrays, not interleaved) flattened as
//
//     index = (z * Height + y) * Width + x
//
// so x is the unit-stride axis. Go reaches these buffers through their raw
// .data() addresses and never copies them.
//
// ---------------------------------------------------------------------------
// Threading
// ---------------------------------------------------------------------------
//
// The engine spawns no threads. It exposes each phase of a timestep as a
// range-sharded entry point, so the caller (Go, over GOMAXPROCS workers)
// owns the scheduling and the barriers. Each worker passes a distinct
// worker id and gets a 64-byte-aligned private scratch arena, so two cores
// never write to the same cache line.

#ifndef PHONO_PHONOENGINE_H
#define PHONO_PHONOENGINE_H

#include <stddef.h>

#include <cstddef>
#include <vector>

namespace Quantum {
namespace Physics {

// Number of unit-stride x-elements a blocked y/z sweep carries through the
// tridiagonal recurrence at once. The recurrence is serial along the swept
// axis, so vector parallelism has to come from the *other* axis; eight
// doubles is two AVX2 registers or four NEON registers.
inline constexpr int kLaneWidth = 8;

// Cache line on x86-64 and Apple silicon respectively is 64 and 128 bytes.
// Padding to the larger value makes per-worker scratch false-sharing-proof
// on both.
inline constexpr std::size_t kCacheLineBytes = 128;

// One direction's Crank-Nicolson tridiagonal factorisation.
//
// The linear system produced by a Cayley step along a single axis has
// *constant* coefficients (the grid is uniform), so the forward elimination
// factors depend only on the axis length -- not on the line being solved.
// They are computed once per PrepareStep and reused by every one of the
// Height*Depth lines of that axis.
struct TridiagFactors {
    std::vector<double> cp_re;  // c-prime: super-diagonal after elimination
    std::vector<double> cp_im;
    std::vector<double> w_re;   // reciprocal of the pivot at row j
    std::vector<double> w_im;
    double r = 0.0;             // r = tau*hbar / (4*m*h^2)
};

class PhonoEngine {
public:
    // Constructs the engine descriptor. Deliberately allocation-free and
    // noexcept: every buffer is acquired in Initialize(), so a heap
    // exhaustion is reported as a return value rather than as a
    // std::bad_alloc unwinding through the extern "C" bridge frame into the
    // Go runtime (which is undefined behaviour).
    PhonoEngine(int width, int height, int depth,
                double dx, double dy, double dz,
                double hbar, double mass,
                int max_workers) noexcept;

    ~PhonoEngine();

    PhonoEngine(const PhonoEngine&) = delete;
    PhonoEngine& operator=(const PhonoEngine&) = delete;

    // Acquires every buffer the engine will ever use. Returns false if any
    // allocation failed, in which case the object must be destroyed and not
    // used. Idempotent.
    bool Initialize() noexcept;

    // True once Initialize() has succeeded.
    bool Ready() const noexcept;

    // ---- geometry -------------------------------------------------------
    int Width() const noexcept;
    int Height() const noexcept;
    int Depth() const noexcept;
    int Size() const noexcept;        // Width*Height*Depth
    int MaxWorkers() const noexcept;
    double Dx() const noexcept;
    double Dy() const noexcept;
    double Dz() const noexcept;
    double Hbar() const noexcept;
    double Mass() const noexcept;
    double CellVolume() const noexcept;

    // ---- raw buffer addresses ------------------------------------------
    //
    // Returned as integer addresses rather than double*, because that is
    // what survives the cppgo type registry unchanged (see cppgo_gen.go).
    // Each is the .data() of a std::vector<double> of exactly Size()
    // elements, valid until the engine is destroyed. The vectors are never
    // resized after Initialize(), so the addresses are stable.
    size_t PsiRealAddress() const noexcept;
    size_t PsiImagAddress() const noexcept;
    size_t PotentialAddress() const noexcept;
    size_t DensityAddress() const noexcept;

    // ---- shard counts ---------------------------------------------------
    //
    // Each phase of a timestep is partitioned into independent units of
    // work. A caller shards [0, Count) across its workers and joins before
    // starting the next phase.
    int XLineCount() const noexcept;   // Height*Depth lines, one per (y,z)
    int YBlockCount() const noexcept;  // Depth * ceil(Width/kLaneWidth)
    int ZBlockCount() const noexcept;  // Height * ceil(Width/kLaneWidth)

    // ---- stepping -------------------------------------------------------

    // Rebuilds the tridiagonal factorisations and the potential phase table
    // for this dt. Cheap and idempotent when dt is unchanged and the
    // potential is clean. MUST be called, from a single thread, before any
    // of the sharded phases below and after any write to the potential.
    void PrepareStep(double dt) noexcept;

    // Call after writing through PotentialAddress() so the next
    // PrepareStep() rebuilds the cached phase table.
    void MarkPotentialDirty() noexcept;

    // Phase 1 and 7: the diagonal half-step exp(-i V dt / 2hbar).
    // Operates on the flat index range [begin, end) of [0, Size()).
    void ApplyPotentialHalfStep(int begin, int end) noexcept;

    // Phases 2-6: Cayley solves along one axis, over the unit range
    // [begin, end) of the matching *BlockCount()/XLineCount(). worker must
    // be in [0, MaxWorkers()) and distinct per concurrent caller.
    void SweepX(int worker, int begin, int end) noexcept;
    void SweepY(int worker, int begin, int end) noexcept;
    void SweepZ(int worker, int begin, int end) noexcept;

    // Single-threaded convenience: PrepareStep plus the whole seven-phase
    // composition, on worker 0. Equivalent to what a correctly barriered
    // multi-worker driver computes.
    void StepTime(double dt) noexcept;

    // ---- observables ----------------------------------------------------

    // Fills the density buffer with |Psi|^2 over the flat range
    // [begin, end). Sharded like ApplyPotentialHalfStep.
    void ComputeDensity(int begin, int end) noexcept;

    // Total probability, sum |Psi|^2 dV. Single-threaded; O(Size()).
    double Norm() const noexcept;

    // Rescales Psi so that Norm() == 1. No-op on an all-zero state.
    void Normalize() noexcept;

    // Seeds Psi with a normalised Gaussian wave packet centred at
    // (x0,y0,z0) with widths (sx,sy,sz) and mean wavevector (kx,ky,kz):
    //
    //   Psi = exp(-(x-x0)^2/2sx^2 - ...) * exp(i(kx*x + ky*y + kz*z))
    //
    // Grid point i on an axis sits at (i+1)*h, so the Dirichlet walls at
    // index -1 and index n are at 0 and (n+1)*h.
    void SetGaussianPacket(double x0, double y0, double z0,
                           double sx, double sy, double sz,
                           double kx, double ky, double kz) noexcept;

private:
    // Recomputes one axis' constant-coefficient Thomas factorisation.
    void BuildFactors(TridiagFactors& f, int n, double h, double tau) noexcept;

    // Scratch arena base for a worker, already cache-line isolated.
    double* WorkerScratch(int worker) const noexcept;

    int width_;
    int height_;
    int depth_;
    int size_;
    int max_workers_;

    double dx_;
    double dy_;
    double dz_;
    double hbar_;
    double mass_;

    // Real and imaginary parts of Psi, the static potential, and |Psi|^2.
    // Flattened 3D, index = (z*height_ + y)*width_ + x.
    std::vector<double> psi_real_;
    std::vector<double> psi_imag_;
    std::vector<double> potential_;
    std::vector<double> density_;

    // cos/sin of the half-step phase -theta = -V*dt/(2*hbar), precomputed
    // so the diagonal half-step costs no transcendentals on the hot path.
    std::vector<double> phase_cos_;
    std::vector<double> phase_sin_;

    TridiagFactors fx_;  // x axis, tau = dt/2
    TridiagFactors fy_;  // y axis, tau = dt/2
    TridiagFactors fz_;  // z axis, tau = dt

    // Per-worker scratch for the Thomas forward sweep, one padded slab per
    // worker. Raw, over-aligned, nothrow-allocated: std::vector guarantees
    // no alignment beyond alignof(double).
    double* scratch_ = nullptr;
    std::size_t scratch_stride_ = 0;  // doubles between worker slabs

    double prepared_dt_ = 0.0;
    bool potential_dirty_ = true;
    bool ready_ = false;
};

}  // namespace Physics
}  // namespace Quantum

#endif  // PHONO_PHONOENGINE_H
