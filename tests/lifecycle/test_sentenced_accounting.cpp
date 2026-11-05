// Executable lifecycle test for the dev/tc-memory-ports accounting change.
// Uses the repo's real DelayedDestructionRing header and a faithful mirror of
// the ported TickFrame/DeleteImage/MakeDormant accounting.
//
// Physical model, matching production ownership: an image's bytes become
// driver-visible at REGISTRATION (allocation), stay visible while the image
// sits in the deletion ring (move TRANSFERS residency - the move adds nothing
// and the moved-from husk owns nothing), and stop being visible when the ring
// destroys the image. Unbacked images are modeled with 0 resident bytes: the
// driver never counted them.
//
// COUNTER INVARIANT (friend-review requirement: establish what the counter
// means before fixing). total_used_memory is a per-frame driver-anchored
// HEURISTIC of live texture-cache memory pressure:
//   - TickFrame re-anchors it to (driver-used - sentenced-pending), i.e. the
//     driver unit anchor discounted for bytes already sentenced.
//   - Between refreshes it is adjusted by BOOKED ESTIMATE deltas:
//     +host at RegisterImage, +scaled at first ScaleUp, -scaled/-host at
//     DeleteImage, -host/+host across the dormant round trip.
//   - The anchor and the deltas are different units (driver bytes vs
//     estimates, which deliberately need not match): per-image divergence is
//     absorbed at the next refresh. Within a frame the counter is therefore
//     an approximation, never a physical truth.
//   - The SUBTRACTIONS must never wrap: a wrapped value reads as
//     over-critical to every GC threshold for the rest of the frame
//     (eviction-storm class). Saturating a subtraction at 0 is containment of
//     the unit-mixing mismatch, not correction: it is SURFACED (production
//     LOG_WARNING; here the real CollectorSubtract's return is counted), and
//     the absorbed excess biases the counter toward under-collection for the
//     rest of the frame. The next refresh re-derives the anchor; it does not
//     recover the absorbed divergence. Additions remain unchecked (out of
//     scope): booked estimates are logical values, not physically bounded.
//   - Estimate path (no driver reporting): booked-estimate ledger, exact
//     per image ONLY for the modeled symmetric operations (register/delete,
//     first-ScaleUp, dormant round trip); no global exactness is claimed.
//
// This test executes the ACTUAL production subtraction arithmetic
// (collector_accounting.h); the surrounding frame flow is a line-mirrored
// model (no Vulkan device), so it is arithmetic evidence, not production
// execution.
//
// Shared-value model (friend-review rework): production has ONE collector
// variable, refreshed every TickFrame and adjusted in between; this mirror
// uses one shared collector_total the same way, so refresh-then-mutate
// interactions are exercised (cases 12-17), including below/equal/above the
// refreshed total, scaled images, and multiple deletions in one frame.
// --old runs the pre-fix production semantics (plain wrapping u64
// subtraction) to demonstrate the defect the fix addresses.
//
// NOT covered: the real TextureCache integration (needs a Vulkan device);
// the mirror is verified line-by-line against the ported code but executes no
// production DeleteImage/TickFrame/SlotVector/destructor code.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>
#include "../../src/video_core/delayed_destruction_ring.h"
#include "../../src/video_core/texture_cache/collector_accounting.h" // actual production arithmetic

using u64 = std::uint64_t;

static bool g_saturating = true; // false = pre-fix production semantics (wrapping subtraction)
static u64 g_saturation_events = 0; // fixed-mode count of saturating steps (surfaces the mismatch)
static bool g_driver_reporting = true; // false = CanReportMemoryUsage() false regime (pure ledger)

static u64 driver_usage = 0; // simulated runtime.GetDeviceMemoryUsage()

struct TrackedImage {
    u64 resident;   // driver-visible bytes (0 while unbacked)
    u64 estimate;   // booked host estimate (RegisterImage/DeleteImage accounting)
    u64 scaled_est; // scaled estimate (booked at first ScaleUp, subtracted at DeleteImage)
    bool gated;     // unbacked or sparse: estimates excluded from the discount
    bool has_scaled = false; // mirror of HasScaled(): set by the first ScaleUp
    TrackedImage(u64 r, u64 e, bool g, u64 sc = 0)
        : resident(r), estimate(e), scaled_est(sc), gated(g) {
        driver_usage += resident; // registration: the allocation becomes visible
    }
    // Move transfers residency: no new driver bytes, and the husk owns none.
    TrackedImage(TrackedImage&& o) noexcept
        : resident(std::exchange(o.resident, 0)), estimate(o.estimate),
          scaled_est(o.scaled_est), gated(o.gated), has_scaled(o.has_scaled) {}
    TrackedImage(const TrackedImage&) = delete;
    TrackedImage& operator=(const TrackedImage&) = delete;
    ~TrackedImage() { driver_usage -= resident; } // destruction frees what it still owns
};

// --- mirror of the ported members (texture_cache_base.h) ---
static constexpr size_t TICKS_TO_DESTROY = 8;
VideoCommon::DelayedDestructionRing<TrackedImage, TICKS_TO_DESTROY> sentenced_images;
std::array<u64, TICKS_TO_DESTROY> sentenced_image_bytes{};
size_t sentenced_image_index = 0;

// --- the shared collector value: mirror of production total_used_memory ---
static u64 collector_total = 0;
static u64 dormant_bytes = 0;
static u64 driver_used_memory = 0; // mirror of the raw member: telemetry reads this

// Oracle: what a TickFrame refresh would produce right now. Also updates the
// last-read raw figure (driver_used_memory), as production's TickFrame does.
static u64 effective_total_driver_path() {
    driver_used_memory = driver_usage;
    // Mirror of the ported clamp-at-driver aggregation (never wraps; invariant:
    // pending <= driver_used_memory at every step).
    u64 sentenced_pending_bytes = 0;
    for (const u64 bytes : sentenced_image_bytes) {
        sentenced_pending_bytes += std::min(bytes, driver_used_memory - sentenced_pending_bytes);
    }
    return driver_used_memory - sentenced_pending_bytes;
}
// Production subtraction from the shared total, both semantics. The fixed mode
// calls the ACTUAL production arithmetic (collector_accounting.h).
static void sub_collector(u64 x) {
    if (g_saturating) {
        if (VideoCommon::CollectorSubtract(collector_total, x)) {
            ++g_saturation_events; // mismatch surfaced, not hidden (production: LOG_WARNING)
        }
    } else {
        collector_total -= x; // pre-fix production: wraps
    }
}
// Production TickFrame order (:320-337 -> :472-479): refresh the total from
// the driver figure; then GC, dormant eviction and unbacked retries run
// (modeled as explicit mutations between refresh_total() and advance_ring());
// the ring tick happens at the END of the frame.
static void refresh_total() {
    if (g_driver_reporting) {
        collector_total = effective_total_driver_path();
    }
}
static void advance_ring() {
    // Ring tick: destroys the wrapped slot (physically freeing those bytes),
    // then index advance + slot zeroing.
    sentenced_images.Tick();
    sentenced_image_index = (sentenced_image_index + 1) % TICKS_TO_DESTROY;
    sentenced_image_bytes[sentenced_image_index] = 0;
}
static void tick_frame() { // full frame with no GC/dormant activity in between
    refresh_total();
    advance_ring();
}
// --- mirror of RegisterImage/DeleteImage/MakeDormant/EvictDormantImages ---
static std::vector<TrackedImage> registry; // live images; deleted ones stay as empty husks
static size_t register_image(u64 resident, u64 estimate, bool gated = false, u64 scaled = 0) {
    collector_total += estimate; // host estimate booked at registration
    registry.emplace_back(resident, estimate, gated, scaled);
    return registry.size() - 1;
}
static void scale_up(size_t idx) { // mirror of ScaleUp: books scaled ONLY on first acquisition
    TrackedImage& img = registry[idx];
    if (img.has_scaled || img.scaled_est == 0) {
        return; // production: !has_copy && HasScaled() gates the add
    }
    img.has_scaled = true;
    collector_total += img.scaled_est; // unchecked add, as in production
}
static void delete_image(size_t idx) {
    TrackedImage& img = registry[idx];
    // Production subtracts scaled only when HasScaled (M5), then host unconditionally (M6);
    // the unbacked/sparse gate below only affects the discount booking.
    if (img.has_scaled) {
        sub_collector(img.scaled_est);
    }
    sub_collector(img.estimate);
    // Ported gate: unbacked images were never backed; sparse images commit only populated
    // pages. Neither estimate tracks driver bytes, so both stay out of the discount.
    // Mirror of the ported checked scaled+host sum: overflow drops the discount entirely.
    const u64 scaled = img.has_scaled ? img.scaled_est : 0; // production: 0 when !HasScaled
    const u64 freed_bytes =
        img.gated ? 0
        : scaled > ~0ULL - img.estimate ? 0
                                        : scaled + img.estimate;
    sentenced_images.Push(std::move(img)); // transfer residency into the ring
    // Mirror of the ported checked add: on u64 overflow keep the previous total
    // (that delete's discount is dropped), never wrap.
    u64& slot_bytes = sentenced_image_bytes[sentenced_image_index];
    if (freed_bytes <= ~0ULL - slot_bytes) {
        slot_bytes += freed_bytes;
    }
}
static void set_dormant(size_t idx) { // mirror of MakeDormant
    sub_collector(registry[idx].estimate);
    dormant_bytes += registry[idx].estimate;
}
static void evict_dormant(size_t idx) { // mirror of EvictDormantImages
    dormant_bytes -= registry[idx].estimate;
    collector_total += registry[idx].estimate; // DeleteImage takes it off again
    delete_image(idx);
}
static void reset_state() {
    registry.clear(); // destroy husks (all own 0 by then)
    // drain the ring fully (8 ticks) so no object leaks between cases
    const bool saved = g_driver_reporting;
    g_driver_reporting = false;
    for (size_t i = 0; i < TICKS_TO_DESTROY; ++i) tick_frame();
    g_driver_reporting = saved;
    sentenced_image_bytes.fill(0);
    sentenced_image_index = 0;
    driver_usage = 0;
    collector_total = 0;
    dormant_bytes = 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--old") == 0) {
        g_saturating = false; // replicate pre-fix production subtraction semantics
    }
    // 1. Ring destruction delay: a deleted image's bytes stay driver-visible
    //    (physically resident) for exactly 8 ticks after the delete.
    {
        reset_state();
        const size_t a = register_image(400, 400);
        assert(driver_usage == 400 && "registration must be driver-visible");
        delete_image(a);
        assert(driver_usage == 400 && "delete must not free: residency transfers to the ring");
        for (int i = 0; i < 7; ++i) { tick_frame(); assert(driver_usage == 400); }
        tick_frame();
        assert(driver_usage == 0 && "must be destroyed and freed at wrap");
    }
    // 2. Accounting invariant across frames with two staggered deletes and
    //    unrelated live usage kept distinct.
    {
        reset_state();
        register_image(600, 600);      // unrelated, never deleted
        const size_t a = register_image(400, 400);
        const size_t b = register_image(300, 300);
        delete_image(a);               // slot0
        tick_frame();
        delete_image(b);               // slot1
        for (int frame = 0; frame < 7; ++frame) {
            assert(driver_usage == 600 + 400 + 300 && "residency changed early");
            assert(effective_total_driver_path() == 600 && "ring bytes not discounted");
            assert(collector_total == 600 && "shared value diverged from the oracle");
            tick_frame();
        }
        // 8 ticks after the first push: a destroyed (physically freed), b still resident.
        assert(driver_usage == 600 + 300);
        assert(effective_total_driver_path() == 600 && "stale slot not zeroed at wrap");
        assert(collector_total == 600 && "shared value diverged at wrap");
        tick_frame();
        assert(driver_usage == 600);
        assert(effective_total_driver_path() == 600 && "residual discount after all frees");
        assert(collector_total == 600 && "shared value diverged after all frees");
    }
    // 3. Clamp: ring bytes larger than reported usage never underflows.
    //    Other driver-side frees drop the report below the ring's booked bytes;
    //    fresh allocations raise it again before the ring destroys the image,
    //    so the model never destroys more than is currently visible.
    {
        reset_state();
        const size_t a = register_image(400, 400);
        delete_image(a);               // resident 400, discount 400
        driver_usage = 100;            // driver-side frees outside the cache's view
        assert(effective_total_driver_path() == 0 && "clamp failed");
        driver_usage = 400;            // new allocations before destruction
        for (size_t i = 0; i < TICKS_TO_DESTROY; ++i) tick_frame();
        assert(driver_usage == 0 && "ring must free exactly what it owns");
    }
    // 4. Accumulation exactness at physical extremes + clamp. Overflow itself is
    //    boundary-tested in case 9.
    {
        reset_state();
        constexpr u64 TiB = 1ULL << 40;
        const size_t a = register_image(TiB, TiB);
        const size_t b = register_image(TiB, TiB);
        delete_image(a);
        delete_image(b); // same slot: bytes accumulate exactly
        assert(sentenced_image_bytes[sentenced_image_index] == 2 * TiB && "sum inexact");
        driver_usage += TiB; // an unrelated TiB allocation
        assert(effective_total_driver_path() == TiB && "exact subtraction failed");
        driver_usage -= 2 * TiB; // driver-side frees drop the report below the ring bytes
        assert(effective_total_driver_path() == 0 && "clamp failed");
        driver_usage += 2 * TiB; // fresh allocations before the ring destroys anything
    }
    // 5. Ring rollover: pushes interleaved with ticks age one slot per frame.
    //    After 8 push+tick pairs only the FIRST-pushed image has been destroyed
    //    (aged 8 ticks); pushes 2..8 are still resident, aged 7..1 ticks.
    {
        reset_state();
        register_image(8000, 0); // unrelated resident usage
        std::vector<size_t> imgs;
        for (int i = 0; i < 8; ++i) { imgs.push_back(register_image(100, 100)); delete_image(imgs.back()); tick_frame(); }
        assert(driver_usage == 8000 + 700 && "only the first push may be destroyed");
        assert(effective_total_driver_path() == 8000 && "wrong pending total after rollover");
        assert(collector_total == 8000 && "shared value diverged after rollover");
        // Core invariant: a register+delete pair raises physical residency AND the
        // discount by the same amount, so the collector's effective total must
        // not move on the deleting frame.
        const u64 before = effective_total_driver_path();
        const size_t c = register_image(1000, 1000);
        delete_image(c);
        assert(effective_total_driver_path() == before && "delete moved the effective total");
    }
    // 6. Non-driver regime: the shared value is a pure running ledger (never
    //    refreshed from a driver figure) and is never discounted by the ring.
    {
        reset_state();
        g_driver_reporting = false; // CanReportMemoryUsage() == false regime
        const size_t a = register_image(400, 400); // estimate 400 booked at register
        collector_total = 1000;                    // plus unrelated booked estimate
        delete_image(a);                           // estimate subtracts at once
        assert(collector_total == 600 && "ledger must subtract at delete");
        tick_frame(); tick_frame();
        assert(collector_total == 600 && "ledger must not be re-discounted");
        g_driver_reporting = true;
    }
    // 7. Gated images (unbacked/sparse) stay out of the discount: deleting one
    //    changes neither the discount nor the effective total. An unbacked
    //    image is modeled with 0 resident bytes - the driver never saw it.
    {
        reset_state();
        register_image(700, 700);                        // unrelated
        const size_t u = register_image(0, 500, true);   // unbacked: no driver bytes
        delete_image(u);
        assert(sentenced_image_bytes[sentenced_image_index] == 0 && "gated bytes booked");
        assert(driver_usage == 700 && "unbacked image counted by the driver");
        assert(effective_total_driver_path() == 700 && "gate changed the effective total");
        tick_frame();
        const size_t n = register_image(400, 400);       // normal image, next slot
        delete_image(n);
        assert(effective_total_driver_path() == 700 && "normal delete moved effective total");
        tick_frame();
        assert(driver_usage == 1100 && "residency wrong");
        assert(effective_total_driver_path() == 700 && "discount did not stick");
    }
    // 8. Raw driver figure stays separate from the collector figure while a
    //    discount is pending (telemetry honesty split).
    {
        reset_state();
        register_image(900, 0); // unrelated resident usage
        const size_t a = register_image(300, 300);
        delete_image(a);
        const u64 collector = effective_total_driver_path();
        assert(driver_used_memory == 1200 && "raw figure lost");
        assert(collector == 900 && "discount missing from collector figure");
        assert(driver_used_memory - collector == 300 && "split mismatch");
    }
    // 9. Checked accumulation boundary: when the slot total would overflow u64,
    //    the add is dropped (slot keeps its prior total) instead of wrapping.
    {
        reset_state();
        const size_t a = register_image(100, 100);
        delete_image(a);
        sentenced_image_bytes[sentenced_image_index] = ~0ULL - 10; // force near-overflow
        const size_t b = register_image(100, 100);
        delete_image(b); // 100 > 10 headroom: add must be dropped, not wrapped
        assert(sentenced_image_bytes[sentenced_image_index] == ~0ULL - 10 &&
               "overflow wrapped the slot total");
        // The kept total still discounts correctly (clamped at the driver figure).
        assert(effective_total_driver_path() == 0 && "kept total misapplied");
    }
    // 10. Aggregate boundary: multiple individually valid near-max slots cannot wrap
    //     the running sum; the result must equal min(driver, exact sum) in all regions.
    {
        reset_state();
        constexpr u64 BIG = ~0ULL / 2; // 2*BIG = u64max-1 (no wrap); 3*BIG wraps u64
        sentenced_image_bytes[0] = BIG;
        sentenced_image_bytes[1] = BIG;
        sentenced_image_bytes[2] = BIG; // exact sum 3*BIG wraps u64 (to BIG-2)
        driver_usage = 1000;            // driver below one slot: full clamp
        assert(effective_total_driver_path() == 0 && "aggregate wrapped past the clamp");
        driver_usage = BIG + 500;       // driver above one slot, below the wrapping sum
        assert(effective_total_driver_path() == 0 && "clamp-at-driver mismatch vs exact min");
        sentenced_image_bytes[1] = 0;   // back below u64: single slot, partial discount
        sentenced_image_bytes[2] = 0;
        assert(effective_total_driver_path() == 500 && "partial discount wrong");
    }
    // 11. Prebooking boundary: the scaled+host sum is checked BEFORE booking; on
    //     overflow the whole discount for that image is dropped (slot unchanged).
    {
        reset_state();
        register_image(700, 700); // unrelated
        const size_t o = register_image(0, 100, false, ~0ULL - 10); // scaled + host overflows
        scale_up(o); // book scaled so M5's HasScaled path applies (the unchecked add wraps
                     // the collector total in the model, as production's M2 would; this case
                     // asserts only the discount-booking overflow guard)
        delete_image(o);
        assert(sentenced_image_bytes[sentenced_image_index] == 0 && "overflowed sum booked");
        tick_frame();
        const size_t n = register_image(400, 300, false, 100); // scaled 100 + host 300 = 400
        scale_up(n);                                            // ScaleUp books the scaled add
        delete_image(n);
        assert(sentenced_image_bytes[sentenced_image_index] == 400 && "split sum wrong");
        assert(effective_total_driver_path() == 700 && "split discount misapplied");
    }
    // 12. Friend-review case: the refreshed (driver-derived, discounted) total can
    //     be below a deleted image's estimate. driver 100, pending 90 -> collector 10;
    //     an image whose estimate was booked before the refresh (20) is then deleted:
    //     the fixed production saturates; the pre-fix production wraps to ~2^64, which
    //     every threshold read sees as over-critical until the next refresh.
    {
        reset_state();
        constexpr u64 CRITICAL = 1000; // collector threshold stand-in
        register_image(10, 10);                    // unrelated live: driver 10
        const size_t x = register_image(90, 90);   // driver 100
        delete_image(x);                           // pending 90 booked
        const size_t y = register_image(0, 20, true); // gated unbacked: estimate 20, no resident bytes
        tick_frame(); // refresh: collector = 100 - 90 = 10 (the +20 estimate is overwritten)
        assert(collector_total == 10 && "setup must reproduce the friend's 10");
        g_saturation_events = 0;
        delete_image(y); // gated: subtracts estimates but books no discount
        if (g_saturating) {
            assert(g_saturation_events == 1 && "the friend's case saturates exactly once");
            assert(collector_total == 0 && "saturated delete must leave 0");
            assert(collector_total < CRITICAL && "saturated total must not read over-critical");
        } else {
            assert(collector_total == ~0ULL - 9 && "pre-fix production must wrap (10-20)");
            assert(collector_total >= CRITICAL && "wrapped total reads over-critical: the defect");
        }
    }
    // 13. Dormant path: MakeDormant subtracts the same estimate from the shared
    //     total; eviction re-adds it before DeleteImage, balancing post-fix.
    {
        reset_state();
        register_image(10, 10);
        const size_t x = register_image(90, 90);
        delete_image(x);
        tick_frame(); // collector 10
        const size_t y = register_image(0, 20, true);
        tick_frame(); // refresh overwrites the +20: collector 10
        g_saturation_events = 0;
        set_dormant(y);
        if (g_saturating) {
            assert(g_saturation_events == 1 && "dormant sentence saturates exactly once");
            assert(collector_total == 0 && "saturated dormant subtract must leave 0");
        } else {
            assert(collector_total == ~0ULL - 9 && "pre-fix production must wrap on dormant");
        }
        assert(dormant_bytes == 20 && "dormant ledger wrong");
        evict_dormant(y); // dormant_bytes -> 0, total += 20, delete subtracts 20
        if (g_saturating) {
            assert(g_saturation_events == 1 && "balanced evict re-add must not saturate again");
            assert(collector_total == 0 && "evict re-add + delete must balance post-fix");
        } else {
            assert(collector_total == ~0ULL - 9 && "pre-fix wrap persists through evict");
        }
        assert(dormant_bytes == 0 && "dormant ledger must drain");
    }
    // 14. Scaled+host two-step: the scaled subtraction can leave the total below
    //     the host estimate; both steps saturate independently post-fix.
    {
        reset_state();
        register_image(15, 0);                          // driver 15
        const size_t y = register_image(0, 10, true, 10); // estimate 10 + scaled 10
        scale_up(y);                                       // collector 20
        tick_frame();                                      // refresh: 15 - 0 = 15
        assert(collector_total == 15);
        g_saturation_events = 0;
        delete_image(y); // scaled 10: 15 -> 5 (exact); host 10: pre-fix 5-10 wraps, fixed saturates
        if (g_saturating) {
            assert(g_saturation_events == 1 && "only the host step saturates");
            assert(collector_total == 0 && "saturated host step must leave 0");
        } else {
            assert(collector_total == ~0ULL - 4 && "pre-fix production must wrap (5-10)");
        }
    }
    // 15. Boundary - equal: adjusted total exactly equals the deletion estimate.
    //     Exact 0 in BOTH semantics: the defect is strictly the below-total class.
    {
        reset_state();
        register_image(10, 10);
        const size_t x = register_image(90, 90);
        delete_image(x);
        tick_frame(); // collector 10
        const size_t y = register_image(0, 10, true);
        tick_frame(); // refresh overwrites the +10: collector 10
        g_saturation_events = 0;
        delete_image(y);
        assert(collector_total == 0 && "equal case must land exactly on 0 in both semantics");
        if (g_saturating) {
            assert(g_saturation_events == 0 && "equal case must not saturate");
        }
    }
    // 16. Boundary - above: adjusted total exceeds the deletion estimate; the
    //     remainder must be exact in both semantics (no containment involved).
    {
        reset_state();
        register_image(30, 30);
        const size_t x = register_image(90, 90);
        delete_image(x);
        tick_frame(); // collector 120 - 90 = 30
        assert(collector_total == 30 && "above setup wrong");
        const size_t y = register_image(0, 20, true);
        tick_frame(); // refresh overwrites the +20: collector 30
        g_saturation_events = 0;
        delete_image(y);
        assert(collector_total == 10 && "above case must leave an exact remainder");
        if (g_saturating) {
            assert(g_saturation_events == 0 && "above case must not saturate");
        }
    }
    // 17. Multiple deletions within one frame after one refresh: the first
    //     subtraction is exact, the second crosses below the remaining total.
    {
        reset_state();
        constexpr u64 CRITICAL = 1000;
        register_image(10, 10);
        const size_t x = register_image(90, 90);
        delete_image(x);
        tick_frame(); // collector 10
        const size_t a = register_image(0, 5, true);
        const size_t b = register_image(0, 20, true);
        tick_frame(); // refresh overwrites both adds: collector 10
        g_saturation_events = 0;
        delete_image(a); // exact: 10 - 5 = 5
        assert(collector_total == 5 && "first delete of the frame must be exact");
        delete_image(b); // 5 - 20: pre-fix wraps, fixed saturates
        if (g_saturating) {
            assert(collector_total == 0 && "second delete must saturate");
            assert(g_saturation_events == 1 && "exactly one saturating step expected");
        } else {
            assert(collector_total == ~0ULL - 14 && "pre-fix production must wrap (5-20)");
            assert(collector_total >= CRITICAL && "wrapped total reads over-critical: the defect");
        }
    }
    // 18. Ordinary BACKED image with estimate above the refreshed total: the delete
    //     mutates the shared total AND books the discount while the raw driver figure
    //     is unchanged, then the next refresh and the ring rollover run (editor gap 3).
    {
        reset_state();
        const size_t y = register_image(30, 45); // driver commits 30, estimate booked 45
        refresh_total(); // frame N refresh: collector = 30 - 0 = 30
        assert(collector_total == 30 && "backed setup wrong");
        g_saturation_events = 0;
        delete_image(y); // GC step of frame N: subtract 45, book 45 pending, driver still 30
        assert(sentenced_image_bytes[sentenced_image_index] == 45 && "discount not booked");
        assert(driver_usage == 30 && "raw driver figure must be unchanged");
        if (g_saturating) {
            assert(collector_total == 0 && "saturated backed delete must leave 0");
            assert(g_saturation_events == 1 && "one saturating step");
        } else {
            assert(collector_total == ~0ULL - 14 && "pre-fix production must wrap (30-45)");
        }
        advance_ring(); // end of frame N (image not yet destroyed: 8-tick ring)
        refresh_total(); // frame N+1 refresh: min-clamped pending 30, collector 0
        assert(collector_total == 0 && "post-delete refresh mismatch");
        for (size_t i = 0; i < TICKS_TO_DESTROY; ++i) advance_ring(); // ring rollover destroys it
        assert(driver_usage == 0 && "ring destruction must free the driver bytes");
        refresh_total();
        assert(collector_total == 0 && "post-destruction refresh mismatch");
    }
    // 19. Scaled-first-step underflow: the SCALED subtraction (M5), not the host step,
    //     crosses below the refreshed total; both steps then saturate independently.
    {
        reset_state();
        register_image(10, 0);
        const size_t y = register_image(0, 5, true, 15); // host est 5, scaled est 15
        scale_up(y);                                     // collector 20 booked
        refresh_total();                                 // driver 10, pending 0 -> 10
        assert(collector_total == 10);
        g_saturation_events = 0;
        delete_image(y); // scaled step: 10 vs 15; host step: 0 vs 5
        if (g_saturating) {
            assert(collector_total == 0 && "both steps must leave 0");
            assert(g_saturation_events == 2 && "scaled step AND host step both saturate");
        } else {
            assert(collector_total == ~0ULL - 9 && "pre-fix production must wrap (10-15-5)");
        }
    }
    // 20. Scaled equal/above: exact arithmetic in both semantics, no saturation.
    {
        reset_state();
        register_image(40, 0);
        const size_t y = register_image(0, 20, true, 10);
        scale_up(y);       // collector 30
        refresh_total();   // driver 40 -> collector 40
        g_saturation_events = 0;
        delete_image(y);   // 40 - 10 - 20 = 10 exact (above class, with scaled)
        assert(collector_total == 10 && "scaled above must leave exact remainder");
        // equal: refreshed total 30 exactly equals scaled 20 + host 10
        const size_t z = register_image(0, 10, true, 20);
        scale_up(z);
        driver_usage -= 10; // driver-side free outside the cache's view: anchor drops to 30
        refresh_total();    // collector 30
        delete_image(z);    // 30 - 20 - 10 = 0 exact
        assert(collector_total == 0 && "scaled equal must land exactly on 0");
        driver_usage += 10; // driver-side re-allocation: tracked residency is still 40,
                            // so the reported figure must return to 40 before the drain
        if (g_saturating) {
            assert(g_saturation_events == 0 && "exact cases must not saturate");
        }
    }
    // 21. Multiple DISCOUNTED (backed) deletes in one frame: exact both semantics,
    //     pending accumulates with the running-min clamp, ring rollover frees.
    {
        reset_state();
        const size_t a = register_image(50, 50);
        const size_t b = register_image(40, 40);
        refresh_total(); // collector 90
        g_saturation_events = 0;
        delete_image(a); // 90 - 50 = 40 exact, pending 50
        delete_image(b); // 40 - 40 = 0 exact, pending min-clamped to 90
        assert(collector_total == 0 && "discounted multi-delete must be exact");
        assert(g_saturation_events == 0 && "no saturation on the exact path");
        refresh_total(); // driver 90 - pending 90 = 0
        assert(collector_total == 0 && "post-delete refresh mismatch");
        for (size_t i = 0; i < TICKS_TO_DESTROY; ++i) advance_ring();
        assert(driver_usage == 0 && "ring rollover must free both");
        refresh_total();
        assert(collector_total == 0);
    }
    // 22. Dormant across a refresh: the driver keeps counting the dormant image's
    //     bytes, so the refresh re-anchors the total UP while dormant_bytes holds the
    //     estimate; the evict re-add + delete nets zero and books the discount.
    {
        reset_state();
        const size_t y = register_image(100, 100);
        refresh_total(); // collector 100
        set_dormant(y);  // -100 -> 0, dormant_bytes 100 (exact, no saturation)
        assert(collector_total == 0 && dormant_bytes == 100);
        tick_frame();    // refresh: driver still counts the 100 -> collector 100 again
        assert(collector_total == 100 && "refresh must re-anchor over the dormant allocation");
        g_saturation_events = 0;
        evict_dormant(y); // dormant_bytes -> 0, +100 then -100 inside the same step, books 100
        assert(dormant_bytes == 0);
        assert(collector_total == 100 && "evict re-add + delete must net zero");
        assert(g_saturation_events == 0 && "dormant evict must not saturate");
        refresh_total(); // driver 100 - pending 100 = 0
        assert(collector_total == 0);
    }
    // 23. Dormant revival across a refresh: revive + re-register double-counts the
    //     allocation transiently (driver anchor + booked estimate) until the next
    //     refresh re-anchors. Documented behavior, asserted explicitly.
    {
        reset_state();
        const size_t y = register_image(100, 100);
        refresh_total();
        set_dormant(y);
        tick_frame(); // refresh re-anchors to 100 (driver still counts it)
        // Revive: dormant_bytes drains; re-register books the estimate again.
        dormant_bytes -= registry[y].estimate; // mirror of ReviveDormantImage's subtract
        collector_total += registry[y].estimate; // mirror of RegisterImage's re-add
        assert(dormant_bytes == 0);
        assert(collector_total == 200 && "revive transiently double-counts until refresh");
        refresh_total();
        assert(collector_total == 100 && "refresh re-anchors after revival");
    }
    // 24. Not-yet-scaled-up BACKED image carrying a scaled estimate: DeleteImage
    //     must subtract and book ONLY the host estimate (production gates both on
    //     HasScaled). A wrong gate would subtract the 500 (wrap/saturation) and
    //     book 510 into the discount instead of 10.
    {
        reset_state();
        const size_t y = register_image(10, 10, false, 500); // scaled_est present, never scale_up
        refresh_total(); // collector = driver 10
        g_saturation_events = 0;
        delete_image(y); // host only: 10 - 10 = 0 exact; scaled 500 untouched
        assert(collector_total == 0 && "unscaled image must subtract host only");
        assert(sentenced_image_bytes[sentenced_image_index] == 10 &&
               "discount must book the host estimate only (wrong gate would book 510)");
        assert(g_saturation_events == 0 && "exact case must not saturate");
    }
    // 25. Backed multi-delete WITH mismatch: first delete exact, second crosses
    //     below the remaining total; both booked discounts still apply.
    {
        reset_state();
        constexpr u64 CRITICAL = 1000;
        const size_t a = register_image(30, 50); // driver 30, est 50
        const size_t b = register_image(30, 40); // driver 60, est 40
        refresh_total(); // collector 60
        g_saturation_events = 0;
        delete_image(a); // 60 - 50 = 10 exact, pending 50
        assert(collector_total == 10);
        delete_image(b); // 10 - 40: pre-fix wraps, fixed saturates; pending 50+40
        if (g_saturating) {
            assert(collector_total == 0 && "second delete must saturate");
            assert(g_saturation_events == 1 && "exactly one saturating step");
        } else {
            assert(collector_total == ~0ULL - 29 && "pre-fix production must wrap (10-40)");
            assert(collector_total >= CRITICAL && "wrapped total reads over-critical: the defect");
        }
        assert(sentenced_image_bytes[sentenced_image_index] == 90 && "both discounts booked");
        refresh_total(); // pending min-clamped to driver 60 -> collector 0
        assert(collector_total == 0);
    }
    if (g_saturating) {
        std::puts("all lifecycle accounting tests passed");
    } else {
        std::puts("pre-fix semantics: underflow wrap reproduced in cases 12-14,17-19,25 (defect evidence)");
    }
    return 0;
}
