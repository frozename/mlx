// Copyright © 2026 Apple Inc.
//
// Unit tests for the per-stream Metal back-pressure gate added on top of
// the exception-safety patch (#2670). These tests target the scheduler
// surface directly and do not require a Metal device — they use fresh
// CPU streams so the test indices are guaranteed not to collide with any
// other stream in the process.

#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>

#include "doctest/doctest.h"

#include "mlx/mlx.h"
#include "mlx/scheduler.h"

using namespace mlx::core;
using namespace std::chrono_literals;

// Build a detached Stream with an explicit index and generation so a test can
// craft two incarnations that share a backing index. Bypasses new_stream().
static Stream make_incarnation(int index, uint64_t generation) {
  Stream s{index, Device::cpu};
  s.generation = generation;
  return s;
}

TEST_CASE("backpressure stream gate blocks at limit" * doctest::test_suite("[backpressure]")) {
  auto s = new_stream(Device::cpu);

  // Thread A acquires the only slot.
  scheduler::acquire_stream_slot(s, /*limit=*/1, /*timeout_secs=*/30);

  // Thread B tries to acquire and must block.
  std::atomic<bool> b_done{false};
  auto fut = std::async(std::launch::async, [&] {
    scheduler::acquire_stream_slot(s, /*limit=*/1, /*timeout_secs=*/30);
    b_done.store(true);
  });

  // B must NOT have returned within 50 ms.
  CHECK(fut.wait_for(50ms) == std::future_status::timeout);
  CHECK_FALSE(b_done.load());

  // Release A's slot — B should unblock.
  scheduler::release_stream_slot(s);
  CHECK(fut.wait_for(200ms) == std::future_status::ready);
  CHECK(b_done.load());

  // Clean up B's slot so the stream's in-flight count returns to zero.
  scheduler::release_stream_slot(s);
}

TEST_CASE("backpressure fast path (INT_MAX limit) has no measurable overhead" * doctest::test_suite("[backpressure]")) {
  auto s = new_stream(Device::cpu);

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 10000; ++i) {
    scheduler::acquire_stream_slot(s, std::numeric_limits<int>::max());
    scheduler::release_stream_slot(s);
  }
  auto t1 = std::chrono::steady_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                .count();

  // Should comfortably fit in a few ms; a regression that adds a cv wait
  // to the fast path would blow past this by orders of magnitude. The 50 ms
  // ceiling is loose enough to absorb CI scheduling jitter without hiding
  // a real regression.
  CHECK(ms < 50);
}

TEST_CASE("backpressure timeout stashes error AND still holds the slot" * doctest::test_suite("[backpressure]")) {
  auto s = new_stream(Device::cpu);

  // Thread A holds the only slot.
  scheduler::acquire_stream_slot(s, /*limit=*/1, /*timeout_secs=*/30);

  // Thread B asks for a slot with a 1-second timeout — it must fail to
  // acquire in time and publish a backpressure error via notify_stream_error.
  auto t0 = std::chrono::steady_clock::now();
  scheduler::acquire_stream_slot(s, /*limit=*/1, /*timeout_secs=*/1);
  auto t1 = std::chrono::steady_clock::now();
  auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  // The timeout call returns around the 1-second mark.
  CHECK(elapsed_ms >= 900);
  CHECK(elapsed_ms < 5000);

  // The caller's next throw_if_stream_error waitpoint rethrows the stashed
  // runtime_error with the expected message.
  bool threw = false;
  try {
    scheduler::throw_if_stream_error(s);
  } catch (const std::runtime_error& e) {
    threw = true;
    std::string msg = e.what();
    CHECK(msg.find("backpressure timeout") != std::string::npos);
  }
  CHECK(threw);

  // Option (b): the timed-out acquire STILL increments the in-flight count
  // (the caller commits the command buffer regardless, and its completion
  // handler will call release_stream_slot — incrementing here keeps the two
  // balanced and prevents underflow). So the stream now holds TWO slots:
  // A's plus B's timed-out one. Proof: after releasing once, a fresh
  // limit-1 acquire must STILL block. (Under the old "no increment on
  // timeout" behavior, one release would drop to zero and this would
  // acquire immediately.)
  scheduler::release_stream_slot(s); // 2 -> 1
  std::atomic<bool> acquired{false};
  auto fut = std::async(std::launch::async, [&] {
    scheduler::acquire_stream_slot(s, /*limit=*/1, /*timeout_secs=*/30);
    acquired.store(true);
  });
  CHECK(fut.wait_for(100ms) == std::future_status::timeout);
  CHECK_FALSE(acquired.load());

  // Release the second slot — the waiter unblocks.
  scheduler::release_stream_slot(s); // 1 -> 0, waiter acquires -> 1
  CHECK(fut.wait_for(500ms) == std::future_status::ready);
  CHECK(acquired.load());

  // Drain the waiter's slot so the stream returns to a clean state.
  scheduler::release_stream_slot(s);
}

TEST_CASE("release_stream_slot ignores a stale-generation completion" * doctest::test_suite("[backpressure]")) {
  // Regression for the release ABA: a delayed completion handler from a
  // torn-down stream (old generation) must NOT decrement the in-flight count
  // of a new incarnation that reused the same index. Before the fix the
  // counter was keyed on index only, so the stale release granted the new
  // stream a free slot beyond its cap.
  constexpr int kTestIndex = 8'000'019;
  auto old_s = make_incarnation(kTestIndex, /*generation=*/100);
  auto new_s = make_incarnation(kTestIndex, /*generation=*/101);

  // New incarnation acquires its single slot (live count -> 1).
  scheduler::acquire_stream_slot(new_s, /*limit=*/1, /*timeout_secs=*/30);

  // A stale completion handler from the OLD incarnation fires its release.
  // It shares the index but carries the old generation, so it must be a
  // no-op against the live count.
  scheduler::release_stream_slot(old_s);

  // The new incarnation's slot must still be held: a second limit-1 acquire
  // must block. (If the stale release wrongly decremented to zero, this would
  // acquire immediately and the test would fail.)
  std::atomic<bool> acquired{false};
  auto fut = std::async(std::launch::async, [&] {
    scheduler::acquire_stream_slot(new_s, /*limit=*/1, /*timeout_secs=*/30);
    acquired.store(true);
  });
  CHECK(fut.wait_for(100ms) == std::future_status::timeout);
  CHECK_FALSE(acquired.load());

  // Releasing the new incarnation's real slot unblocks the waiter.
  scheduler::release_stream_slot(new_s);
  CHECK(fut.wait_for(500ms) == std::future_status::ready);
  CHECK(acquired.load());

  // Drain the waiter's slot to leave the index clean.
  scheduler::release_stream_slot(new_s);
}
