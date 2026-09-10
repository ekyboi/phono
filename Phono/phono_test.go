package phono

import (
	"math"
	"testing"
)

// open is the two-step construction the generated API requires: the C++
// constructor is allocation-free and noexcept, so every buffer is acquired
// separately by Initialize.
func open(t testing.TB, w, h, d int, sp, hbar, mass float64, workers int) *PhonoEngine {
	t.Helper()
	e, err := NewPhonoEngine(w, h, d, sp, sp, sp, hbar, mass, workers)
	if err != nil {
		t.Fatalf("NewPhonoEngine: %v", err)
	}
	if !e.Initialize() {
		e.Close()
		t.Fatal("Initialize returned false")
	}
	return e
}

// TestViewsAliasNativeMemory proves the slices are windows onto the C++
// vectors and not copies: a write through the Go slice has to be visible
// to a value computed entirely on the C++ side.
func TestViewsAliasNativeMemory(t *testing.T) {
	e := open(t, 8, 8, 8, 0.5, 1.0, 1.0, 1)
	defer e.Close()

	re := e.PsiReal()
	im := e.PsiImag()
	if len(re) != e.Size() || len(im) != e.Size() {
		t.Fatalf("view length = %d/%d, want %d", len(re), len(im), e.Size())
	}

	// Two distinct calls must hand back the same backing array.
	if &e.PsiReal()[0] != &re[0] {
		t.Fatal("PsiReal returned a different backing array on the second call")
	}

	for i := range re {
		re[i] = 0
		im[i] = 0
	}
	re[e.Index(3, 4, 5)] = 2.0

	// Norm is computed in C++ over its own vector. dV = 0.5^3 = 0.125,
	// |Psi|^2 = 4, so the norm must be exactly 0.5.
	if got := e.Norm(); math.Abs(got-0.5) > 1e-15 {
		t.Fatalf("Norm through C++ = %v, want 0.5 -- the view is not aliasing", got)
	}
}

// TestNormConservation is the physics acceptance test. Every factor of the
// Crank-Nicolson/Strang composition is unitary, so the norm must hold to
// rounding error over a long run -- not merely drift slowly.
func TestNormConservation(t *testing.T) {
	e := open(t, 32, 32, 32, 0.4, 1.0, 1.0, 1)
	defer e.Close()

	c := 33 * 0.4 / 2
	e.SetGaussianPacket(c, c, c, 1.2, 1.2, 1.2, 3.0, -1.5, 0.75)

	start := e.Norm()
	if math.Abs(start-1.0) > 1e-12 {
		t.Fatalf("packet norm after seeding = %v, want 1", start)
	}
	for i := 0; i < 400; i++ {
		e.StepTime(0.004)
	}
	end := e.Norm()
	if math.Abs(end-1.0) > 1e-10 {
		t.Fatalf("norm drifted to %v over 400 steps (delta %.3e)", end, end-1)
	}
	t.Logf("norm drift over 400 steps: %.3e", end-1)
}

// TestPotentialTrapsPacket checks that the potential is actually coupled
// in: a steep well should hold a packet that would otherwise disperse.
func TestPotentialTrapsPacket(t *testing.T) {
	const n = 24
	const sp = 0.4
	c := float64(n+1) * sp / 2

	spread := func(useWell bool) float64 {
		e := open(t, n, n, n, sp, 1.0, 1.0, 1)
		defer e.Close()
		if useWell {
			v := e.Potential()
			for z := 0; z < n; z++ {
				for y := 0; y < n; y++ {
					for x := 0; x < n; x++ {
						dx := float64(x+1)*sp - c
						dy := float64(y+1)*sp - c
						dz := float64(z+1)*sp - c
						v[e.Index(x, y, z)] = 12.0 * (dx*dx + dy*dy + dz*dz)
					}
				}
			}
			e.MarkPotentialDirty()
		}
		e.SetGaussianPacket(c, c, c, 0.7, 0.7, 0.7, 0, 0, 0)
		for i := 0; i < 200; i++ {
			e.StepTime(0.002)
		}
		e.ComputeDensity(0, e.Size())
		d := e.Density()

		var mass, m2 float64
		for z := 0; z < n; z++ {
			for y := 0; y < n; y++ {
				for x := 0; x < n; x++ {
					p := d[e.Index(x, y, z)]
					dx := float64(x+1)*sp - c
					mass += p
					m2 += p * dx * dx
				}
			}
		}
		return math.Sqrt(m2 / mass)
	}

	free := spread(false)
	trapped := spread(true)
	t.Logf("rms width after 200 steps: free=%.4f trapped=%.4f", free, trapped)
	if !(trapped < free) {
		t.Fatalf("harmonic well did not confine the packet: free=%v trapped=%v", free, trapped)
	}
}

// TestDriverMatchesSerial is the concurrency correctness test. Each unit of
// work performs an identical, independent sequence of operations whichever
// worker runs it, so a sharded step must agree with the serial one
// bit-for-bit -- not approximately. Any mismatch means two workers touched
// the same grid point or the same scratch arena.
func TestDriverMatchesSerial(t *testing.T) {
	const n = 20
	c := float64(n+1) * 0.35 / 2

	serial := open(t, n, n, n, 0.35, 1.1, 0.9, 1)
	defer serial.Close()
	serial.SetGaussianPacket(c, c, c, 0.9, 0.9, 0.9, 2.0, 1.0, -1.0)

	par := open(t, n, n, n, 0.35, 1.1, 0.9, 8)
	defer par.Close()
	par.SetGaussianPacket(c, c, c, 0.9, 0.9, 0.9, 2.0, 1.0, -1.0)

	d, err := NewDriver(par, 8)
	if err != nil {
		t.Fatalf("NewDriver: %v", err)
	}
	defer d.Close()

	for i := 0; i < 50; i++ {
		serial.StepTime(0.003)
		d.Step(0.003)
	}

	sr, si := serial.PsiReal(), serial.PsiImag()
	pr, pi := par.PsiReal(), par.PsiImag()
	for i := range sr {
		if sr[i] != pr[i] || si[i] != pi[i] {
			t.Fatalf("divergence at %d: serial=(%v,%v) parallel=(%v,%v)",
				i, sr[i], si[i], pr[i], pi[i])
		}
	}
	t.Logf("8 workers reproduced 50 serial steps bit-for-bit over %d points", len(sr))
}

// TestDriverRejectsOversubscription guards the scratch-arena invariant:
// worker ids index cache-line-isolated slabs sized at construction.
func TestDriverRejectsOversubscription(t *testing.T) {
	e := open(t, 8, 8, 8, 0.5, 1.0, 1.0, 2)
	defer e.Close()
	if _, err := NewDriver(e, 4); err != ErrDriverWorkers {
		t.Fatalf("NewDriver(4) over MaxWorkers(2) = %v, want ErrDriverWorkers", err)
	}
}

// TestCloseIsIdempotent exercises the RWMutex double-free guard.
func TestCloseIsIdempotent(t *testing.T) {
	e := open(t, 8, 8, 8, 0.5, 1.0, 1.0, 1)
	if !e.Valid() {
		t.Fatal("engine not valid after Initialize")
	}
	e.Close()
	e.Close()
	e.Close()
	if e.Valid() {
		t.Fatal("engine still valid after Close")
	}
	// Every generated method guards on a nil handle and returns the zero
	// value rather than dereferencing it.
	if got := e.Size(); got != 0 {
		t.Fatalf("Size() after Close = %d, want 0", got)
	}
	if got := e.PsiReal(); got != nil {
		t.Fatalf("PsiReal() after Close = %v, want nil", got)
	}
	e.StepTime(0.01) // must not crash
}

// TestStepDoesNotAllocate holds the no-allocation-on-the-hot-path line.
func TestStepDoesNotAllocate(t *testing.T) {
	e := open(t, 16, 16, 16, 0.5, 1.0, 1.0, 1)
	defer e.Close()
	c := 17 * 0.5 / 2
	e.SetGaussianPacket(c, c, c, 1.0, 1.0, 1.0, 1, 0, 0)

	if n := testing.AllocsPerRun(50, func() { e.StepTime(0.002) }); n != 0 {
		t.Fatalf("StepTime allocated %v times per run, want 0", n)
	}
}

func TestDriverStepDoesNotAllocate(t *testing.T) {
	e := open(t, 16, 16, 16, 0.5, 1.0, 1.0, 4)
	defer e.Close()
	c := 17 * 0.5 / 2
	e.SetGaussianPacket(c, c, c, 1.0, 1.0, 1.0, 1, 0, 0)

	d, err := NewDriver(e, 4)
	if err != nil {
		t.Fatalf("NewDriver: %v", err)
	}
	defer d.Close()

	if n := testing.AllocsPerRun(50, func() { d.Step(0.002) }); n != 0 {
		t.Fatalf("Driver.Step allocated %v times per run, want 0", n)
	}
}

func BenchmarkStepSerial(b *testing.B) {
	e := open(b, 64, 64, 64, 0.3, 1.0, 1.0, 1)
	defer e.Close()
	c := 65 * 0.3 / 2
	e.SetGaussianPacket(c, c, c, 2.0, 2.0, 2.0, 1, 0, 0)
	e.PrepareStep(0.002)
	b.SetBytes(int64(e.Size()) * 16)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.StepTime(0.002)
	}
}

func BenchmarkStepParallel(b *testing.B) {
	e := open(b, 64, 64, 64, 0.3, 1.0, 1.0, 16)
	defer e.Close()
	c := 65 * 0.3 / 2
	e.SetGaussianPacket(c, c, c, 2.0, 2.0, 2.0, 1, 0, 0)
	d, err := NewDriver(e, 0)
	if err != nil {
		b.Fatalf("NewDriver: %v", err)
	}
	defer d.Close()
	e.PrepareStep(0.002)
	b.SetBytes(int64(e.Size()) * 16)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		d.Step(0.002)
	}
}

// TestDegenerateAndPartialLaneDimensions is a regression test.
//
// The blocked y/z sweeps carry kLaneWidth adjacent x-values through the
// recurrence at once, and peel the first and last rows out of the
// interior loop. Two shapes break the naive version of that:
//
//   - a width that is not a multiple of the lane width, which leaves a
//     partial trailing block;
//   - an axis of length 1, where the first row is ALSO the last row, so
//     reading a successor neighbour runs off the end of the vector.
//     (The original code guarded this in SweepX but not in SweepY/SweepZ,
//     and ASan caught it on a W x H x 1 grid -- i.e. any 2D run.)
//
// Norm conservation is the probe: a sweep that reads or writes out of
// bounds stops being unitary.
func TestDegenerateAndPartialLaneDimensions(t *testing.T) {
	shapes := [][3]int{
		{21, 20, 16}, // width not a multiple of the lane width
		{9, 9, 9},    // partial trailing block on every plane
		{3, 17, 5},   // width smaller than one lane
		{8, 1, 1},    // 1D along x
		{1, 8, 1},    // 1D along y
		{1, 1, 8},    // 1D along z
		{16, 16, 1},  // 2D slab -- the shape that exposed the bug
		{1, 1, 1},    // single cell
		{2, 2, 2},
	}
	for _, s := range shapes {
		w, h, d := s[0], s[1], s[2]
		e := open(t, w, h, d, 0.4, 1.0, 1.0, 1)

		// Seed something non-trivial that respects the Dirichlet walls.
		re, im := e.PsiReal(), e.PsiImag()
		for z := 0; z < d; z++ {
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					i := e.Index(x, y, z)
					re[i] = math.Sin(math.Pi*float64(x+1)/float64(w+1)) *
						math.Sin(math.Pi*float64(y+1)/float64(h+1)) *
						math.Sin(math.Pi*float64(z+1)/float64(d+1))
					im[i] = 0.3 * re[i]
				}
			}
		}
		e.Normalize()

		before := e.Norm()
		for i := 0; i < 25; i++ {
			e.StepTime(0.003)
		}
		after := e.Norm()
		if math.Abs(after-before) > 1e-11 {
			t.Errorf("%dx%dx%d: norm %v -> %v (delta %.3e)", w, h, d, before, after, after-before)
		}
		e.Close()
	}
}
