// Copyright © 2026 Apple Inc.
//
// Unit tests for the opt-in per-stream Metal back-pressure gate
// (acquire_stream_slot / release_stream_slot), layered on top of the #2670
// error-stash patch. They target the scheduler surface directly and need no
// Metal device. Each case drains the slots it takes so it leaves the stream's
// in-flight count at zero for the next test.

#include <atomic>
#include <chrono>
#include <future>
#include <limits>

#include "doctest/doctest.h"

#include "mlx/mlx.h"
#include "mlx/scheduler.h"

using namespace mlx::core;
using namespace std::chrono_literals;

TEST_CASE(
    "back-pressure gate blocks at the limit and unblocks on release" *
    doctest::test_suite("[backpressure]")) {
  auto s = new_stream(Device::cpu);

  // Thread A takes the only slot (limit 1).
  scheduler::acquire_stream_slot(s, /*limit=*/1, /*timeout_secs=*/30);

  // Thread B tries to take a second slot and must block.
  std::atomic<bool> b_acquired{false};
  auto fut = std::async(std::launch::async, [&] {
    scheduler::acquire_stream_slot(s, /*limit=*/1, /*timeout_secs=*/30);
    b_acquired.store(true);
  });
  CHECK(fut.wait_for(50ms) == std::future_status::timeout);
  CHECK_FALSE(b_acquired.load());

  // Releasing A's slot lets B through.
  scheduler::release_stream_slot(s);
  CHECK(fut.wait_for(5s) == std::future_status::ready);
  CHECK(b_acquired.load());

  scheduler::release_stream_slot(s); // drain B's slot
}

TEST_CASE(
    "back-pressure disabled (INT_MAX) never blocks or tracks" *
    doctest::test_suite("[backpressure]")) {
  auto s = new_stream(Device::cpu);

  // A disabled limit is a pure no-op: it neither blocks nor tracks. Exercise it
  // many times to make sure it stays side-effect-free. (Wall-clock timing of
  // this loop is deliberately not asserted here — that belongs in a benchmark,
  // not a correctness test that would flake on a loaded CI runner.)
  for (int i = 0; i < 1000; ++i) {
    scheduler::acquire_stream_slot(s, std::numeric_limits<int>::max());
  }

  // Because the disabled acquires tracked nothing, a subsequent limit-1 acquire
  // proceeds immediately rather than seeing a phantom backlog.
  scheduler::release_stream_slot(s); // no-op (nothing tracked)
  std::atomic<bool> acquired{false};
  auto fut = std::async(std::launch::async, [&] {
    scheduler::acquire_stream_slot(s, /*limit=*/1, /*timeout_secs=*/30);
    acquired.store(true);
  });
  CHECK(fut.wait_for(5s) == std::future_status::ready);
  CHECK(acquired.load());

  scheduler::release_stream_slot(s); // drain
}

TEST_CASE(
    "back-pressure timeout proceeds without failing the stream" *
    doctest::test_suite("[backpressure]")) {
  auto s = new_stream(Device::cpu);

  // A holds the only slot.
  scheduler::acquire_stream_slot(s, /*limit=*/1, /*timeout_secs=*/30);

  // B asks for a slot with a 1 s deadline. It cannot get one, so it proceeds
  // anyway (the cap is advisory, not a hard kill) after ~1 s.
  auto t0 = std::chrono::steady_clock::now();
  scheduler::acquire_stream_slot(s, /*limit=*/1, /*timeout_secs=*/1);
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
  CHECK(elapsed >= 900);
  CHECK(elapsed < 5000);

  // Crucially, the timed-out acquire must NOT publish a stream error — a slow
  // stream is not a failed stream.
  bool threw = false;
  try {
    scheduler::throw_if_stream_error(s);
  } catch (...) {
    threw = true;
  }
  CHECK_FALSE(threw);

  // It still counted the slot (so it balances the completion handler's release
  // that the committed buffer will issue): the stream now holds two. One
  // release therefore leaves it still at the limit.
  scheduler::release_stream_slot(s); // 2 -> 1
  std::atomic<bool> acquired{false};
  auto fut = std::async(std::launch::async, [&] {
    scheduler::acquire_stream_slot(s, /*limit=*/1, /*timeout_secs=*/30);
    acquired.store(true);
  });
  CHECK(fut.wait_for(100ms) == std::future_status::timeout);
  CHECK_FALSE(acquired.load());

  scheduler::release_stream_slot(s); // 1 -> 0, waiter acquires -> 1
  CHECK(fut.wait_for(5s) == std::future_status::ready);
  CHECK(acquired.load());

  scheduler::release_stream_slot(s); // drain
}
