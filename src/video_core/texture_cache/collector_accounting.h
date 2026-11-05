// SPDX-License-Identifier: GPL-3.0-or-later
// Pure collector-total arithmetic shared by the texture cache and the
// lifecycle accounting test (which includes this header directly, no Vulkan
// device needed).
//
// Counter invariant: total_used_memory is a per-frame driver-anchored
// heuristic of live texture-cache memory pressure. TickFrame re-anchors it to
// (driver-used - sentenced-pending); between refreshes it is adjusted by
// booked estimate deltas. Anchor and deltas are different units (driver bytes
// vs estimates, which deliberately need not match), so an estimate delta can
// exceed the current total within a frame. A SUBTRACTION must never wrap the
// counter: a wrapped value reads as over-critical to every garbage-collector
// threshold for the rest of the frame (the eviction-storm class). Additions
// stay unchecked (booked estimates are logical values; no physical bound is
// claimed). Saturating at 0
// contains that mismatch and biases the within-frame error toward
// under-collection; the next refresh re-derives the anchor (it does not
// recover the absorbed divergence). Saturation is always
// surfaced to the caller (return value; production logs a warning).
#pragma once

#include <cstdint>

namespace VideoCommon {

// Subtracts a booked estimate from the collector total, saturating at 0.
// Returns true when the estimate exceeded the total (saturation engaged).
[[nodiscard]] inline bool CollectorSubtract(std::uint64_t& total, std::uint64_t estimate) {
    if (estimate > total) [[unlikely]] {
        total = 0;
        return true;
    }
    total -= estimate;
    return false;
}

} // namespace VideoCommon
