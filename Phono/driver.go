// driver.go -- multi-core sharded stepping.
//
// HAND-WRITTEN, like telemetry.go, and for the same reason: bindings.go
// is regenerated wholesale by cppgo.
//
// The C++ engine spawns no threads of its own. It exposes each phase of a
// timestep as a range-sharded entry point and leaves scheduling to the
// caller, so the parallel decomposition lives here, in Go, where the
// goroutine scheduler already is.
//
// ---------------------------------------------------------------------
// Why the phases cannot simply all run at once
// ---------------------------------------------------------------------
//
// Within one phase, the units of work are genuinely independent: two
// x-lines share no grid point, so two workers sweeping different lines
// touch disjoint memory. Across phases they are not. The x sweep reads
// the values the potential half-step wrote, and the y sweep reads what
// the x sweep wrote. Every phase boundary is therefore a full barrier,
// and there are seven phases in the Strang palindrome:
//
//     V(dt/2) X(dt/2) Y(dt/2) Z(dt) Y(dt/2) X(dt/2) V(dt/2)
//
// PrepareStep is the one genuinely serial part; it is O(Size()) but runs
// only when dt changes or the potential is marked dirty, so in a steady
// run it is a pair of compares.
//
// ---------------------------------------------------------------------
// Why the workers are persistent
// ---------------------------------------------------------------------
//
// Spawning goroutines per phase would mean 7*workers goroutine creations
// per timestep. Instead the pool is created once and parked on a channel
// receive; a phase is a send plus a WaitGroup join. Shard boundaries are
// precomputed at construction, so a step performs no allocation at all --
// consistent with the C++ side, which does none either.

package phono

import (
	"errors"
	"runtime"
	"sync"
)

// ErrDriverWorkers reports a worker count the engine cannot serve.
var ErrDriverWorkers = errors.New("phono: worker count exceeds the engine's MaxWorkers")

// phaseKind selects which sharded entry point a worker invokes.
type phaseKind uint8

const (
	phasePotential phaseKind = iota
	phaseSweepX
	phaseSweepY
	phaseSweepZ
	phaseDensity
	phaseStop
)

// bounds is one worker's half-open slice of a phase's unit range.
type bounds struct {
	begin int
	end   int
}

// Driver owns a persistent worker pool bound to one engine and executes
// the seven-phase timestep across it.
//
// A Driver is not safe for concurrent use by multiple goroutines; it is
// the thing that creates the concurrency. Close it before closing the
// engine it wraps.
type Driver struct {
	eng     *PhonoEngine
	workers int

	// Precomputed shard tables, one entry per worker, indexed by phase
	// family. Computed once so that a step allocates nothing.
	flat []bounds // over Size(): potential half-step and density
	xs   []bounds // over XLineCount()
	ys   []bounds // over YBlockCount()
	zs   []bounds // over ZBlockCount()

	tasks []chan phaseKind
	wg    sync.WaitGroup

	closeOnce sync.Once
}

// shard splits [0, n) into w contiguous, near-equal, half-open ranges.
//
// The remainder is spread one unit at a time across the leading workers
// rather than dumped on the last, so the longest shard exceeds the
// shortest by at most one unit. With a barrier at the end of every phase,
// the phase costs as much as its slowest worker, so that imbalance is the
// whole story.
func shard(n, w int) []bounds {
	out := make([]bounds, w)
	if w <= 0 {
		return out
	}
	base := n / w
	rem := n % w
	pos := 0
	for i := 0; i < w; i++ {
		size := base
		if i < rem {
			size++
		}
		out[i] = bounds{begin: pos, end: pos + size}
		pos += size
	}
	return out
}

// NewDriver builds a worker pool over eng.
//
// workers <= 0 selects runtime.GOMAXPROCS(0), clamped to the engine's
// MaxWorkers -- the engine sized exactly that many cache-line-isolated
// scratch arenas at construction, and a worker id outside that range has
// nowhere private to write.
func NewDriver(eng *PhonoEngine, workers int) (*Driver, error) {
	if eng == nil || !eng.Valid() {
		return nil, errors.New("phono: driver needs a live engine")
	}
	max := eng.MaxWorkers()
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if workers > max {
		return nil, ErrDriverWorkers
	}
	if workers < 1 {
		workers = 1
	}

	d := &Driver{
		eng:     eng,
		workers: workers,
		flat:    shard(eng.Size(), workers),
		xs:      shard(eng.XLineCount(), workers),
		ys:      shard(eng.YBlockCount(), workers),
		zs:      shard(eng.ZBlockCount(), workers),
		tasks:   make([]chan phaseKind, workers),
	}

	for i := 0; i < workers; i++ {
		// Unbuffered: the send in run() is itself the release edge that
		// publishes the previous phase's writes to this worker.
		d.tasks[i] = make(chan phaseKind)
		go d.loop(i, d.tasks[i])
	}
	return d, nil
}

// Workers reports the pool size.
func (d *Driver) Workers() int { return d.workers }

// loop is one worker. It parks on its channel and never allocates.
//
// The worker id doubles as the engine's scratch-arena index, so worker i
// is the only thread that ever touches arena i -- and the arenas are
// padded to a cache line, so two workers never contend for ownership of
// one line. That is the false-sharing guarantee; without the padding,
// two arenas could land in one 128-byte line and every write by one core
// would invalidate the other's copy.
func (d *Driver) loop(id int, tasks <-chan phaseKind) {
	for k := range tasks {
		switch k {
		case phasePotential:
			b := d.flat[id]
			d.eng.ApplyPotentialHalfStep(b.begin, b.end)
		case phaseSweepX:
			b := d.xs[id]
			d.eng.SweepX(id, b.begin, b.end)
		case phaseSweepY:
			b := d.ys[id]
			d.eng.SweepY(id, b.begin, b.end)
		case phaseSweepZ:
			b := d.zs[id]
			d.eng.SweepZ(id, b.begin, b.end)
		case phaseDensity:
			b := d.flat[id]
			d.eng.ComputeDensity(b.begin, b.end)
		case phaseStop:
			d.wg.Done()
			return
		}
		d.wg.Done()
	}
}

// run executes one phase across every worker and joins. The join is the
// barrier that makes the next phase safe.
func (d *Driver) run(k phaseKind) {
	d.wg.Add(d.workers)
	for i := range d.tasks {
		d.tasks[i] <- k
	}
	d.wg.Wait()
}

// Step advances the wavefunction by dt across the whole pool.
//
// The composition is the symmetric Strang palindrome derived in
// PhonoEngine.cpp. Every factor is unitary, so the norm is conserved
// exactly regardless of how the work was sharded -- and because each unit
// of work performs an identical, independent sequence of floating-point
// operations no matter which worker runs it, the result is bit-for-bit
// identical to the serial PhonoEngine.StepTime.
func (d *Driver) Step(dt float64) {
	// Serial, and must precede the parallel phases: it rewrites the
	// factor tables and the phase table that every worker then reads.
	d.eng.PrepareStep(dt)

	d.run(phasePotential) // V(dt/2)
	d.run(phaseSweepX)    // X(dt/2)
	d.run(phaseSweepY)    // Y(dt/2)
	d.run(phaseSweepZ)    // Z(dt)
	d.run(phaseSweepY)    // Y(dt/2)
	d.run(phaseSweepX)    // X(dt/2)
	d.run(phasePotential) // V(dt/2)
}

// Advance runs n timesteps of size dt.
func (d *Driver) Advance(dt float64, n int) {
	for i := 0; i < n; i++ {
		d.Step(dt)
	}
}

// ComputeDensity fills the |Psi|^2 buffer across the pool. Read the
// result through PhonoEngine.Density.
func (d *Driver) ComputeDensity() {
	d.run(phaseDensity)
}

// Close stops the worker pool. It does not close the engine. Idempotent.
func (d *Driver) Close() {
	d.closeOnce.Do(func() {
		d.wg.Add(d.workers)
		for i := range d.tasks {
			d.tasks[i] <- phaseStop
		}
		d.wg.Wait()
		for i := range d.tasks {
			close(d.tasks[i])
		}
	})
}
