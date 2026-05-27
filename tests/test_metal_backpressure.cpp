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
#include <future>
#include <stdexcept>
#include <string>
#include <thread>

#include "doctest/doctest.h"

#include "mlx/mlx.h"
#include "mlx/scheduler.h"

using namespace mlx::core;
using namespace std::chrono_literals;

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

TEST_CASE("backpressure timeout routes through stream-error stash" * doctest::test_suite("[backpressure]")) {
  auto s = new_stream(Device::cpu);

  // Thread A holds the only slot.
  scheduler::acquire_stream_slot(s, /*limit=*/1, /*timeout_secs=*/30);

  // Thread B asks for a slot with a 1-second timeout — it must fail
  // to acquire and publish a backpressure error via notify_stream_error.
  auto t0 = std::chrono::steady_clock::now();
  scheduler::acquire_stream_slot(s, /*limit=*/1, /*timeout_secs=*/1);
  auto t1 = std::chrono::steady_clock::now();
  auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  // The timeout call should have returned around the 1-second mark and
  // not held the slot (no increment on timeout).
  CHECK(elapsed_ms >= 900);
  CHECK(elapsed_ms < 5000);

  // The caller's next throw_if_stream_error waitpoint should rethrow
  // the stashed runtime_error with the expected message.
  bool threw = false;
  try {
    scheduler::throw_if_stream_error(s);
  } catch (const std::runtime_error& e) {
    threw = true;
    std::string msg = e.what();
    CHECK(msg.find("backpressure timeout") != std::string::npos);
  }
  CHECK(threw);

  // Release A's slot so the stream returns to a clean state.
  scheduler::release_stream_slot(s);
}
