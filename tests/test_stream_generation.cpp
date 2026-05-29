// Copyright © 2026 Apple Inc.
//
// Unit tests for the Stream generation counter that prevents stale
// completion-handler errors from being attributed to a new stream that
// reuses the same index. Does not require a Metal device — test streams
// use explicit high-numbered indices that cannot collide with streams
// allocated by other tests in the same process.

#include <stdexcept>
#include <string>

#include "doctest/doctest.h"

#include "mlx/mlx.h"
#include "mlx/scheduler.h"

using namespace mlx::core;
using namespace mlx::core::scheduler;

// ── helpers ─────────────────────────────────────────────────────────────────

// Build a detached Stream with an explicit index and generation.
// Bypasses new_stream() so we can craft two incarnations that share an index.
static Stream make_incarnation(int index, uint64_t generation) {
  Stream s{index, Device::cpu};
  s.generation = generation;
  return s;
}

// ── test ────────────────────────────────────────────────────────────────────

TEST_CASE(
    "stream generation counter prevents stale error attribution" *
    doctest::test_suite("[stream-generation]")) {
  // Use a high-numbered index to avoid collisions with streams allocated
  // elsewhere in the test suite (new_stream() allocates sequentially from 0).
  constexpr int kTestIndex = 8'000'001;
  constexpr uint64_t kOldGen = 7;
  constexpr uint64_t kNewGen = 8;

  auto old_s = make_incarnation(kTestIndex, kOldGen);
  auto new_s = make_incarnation(kTestIndex, kNewGen);

  // --- step 1: stash an error under the old generation ---

  scheduler::notify_stream_error(
      old_s,
      std::make_exception_ptr(
          std::runtime_error("stale error from old incarnation")));

  // --- step 2: new stream (same index, later generation) must NOT see it ---

  bool threw = false;
  try {
    scheduler::throw_if_stream_error(new_s);
  } catch (...) {
    threw = true;
  }
  CHECK_FALSE(threw);

  // --- step 3: the stale slot was consumed; now stash a real error ---

  // throw_if_stream_error erased the slot unconditionally, so new_s has no
  // pending error and any_stream_error_ was lowered.  A fresh notify must go
  // through.
  scheduler::notify_stream_error(
      new_s,
      std::make_exception_ptr(
          std::runtime_error("real error on new incarnation")));

  // --- step 4: new stream's matching generation MUST surface the error ---

  bool threw2 = false;
  std::string msg;
  try {
    scheduler::throw_if_stream_error(new_s);
  } catch (const std::runtime_error& e) {
    threw2 = true;
    msg = e.what();
  }
  CHECK(threw2);
  CHECK(msg.find("real error on new incarnation") != std::string::npos);

  // --- step 5: baseline — a notify + throw on old incarnation still works ---
  // (Confirms the generation-match branch is not always skipping.)

  // Slot was consumed in step 4; stash against old_s now.
  scheduler::notify_stream_error(
      old_s,
      std::make_exception_ptr(std::runtime_error("error attributed to old")));

  bool threw3 = false;
  std::string msg3;
  try {
    scheduler::throw_if_stream_error(old_s);
  } catch (const std::runtime_error& e) {
    threw3 = true;
    msg3 = e.what();
  }
  CHECK(threw3);
  CHECK(msg3.find("error attributed to old") != std::string::npos);
}

TEST_CASE(
    "newer-generation error supersedes a stranded stale error in notify" *
    doctest::test_suite("[stream-generation]")) {
  // Regression for the notify_stream_error ABA race. Timeline:
  //   1. A stream (index i, generation N) is torn down; its in-flight Metal
  //      completion handler is still queued on libdispatch.
  //   2. A new stream reuses index i as generation N+1.
  //   3. The OLD handler finally runs and stashes an error against index i.
  //   4. The new incarnation hits its OWN error and calls notify.
  // Before the fix, step 4 saw an occupied slot (first-error-wins) and
  // dropped the live error; throw_if_stream_error then discarded the stale
  // step-3 entry on generation mismatch, so the real failure vanished and the
  // stream looked healthy. The fix lets the higher generation supersede.
  constexpr int kTestIndex = 8'000'017;
  constexpr uint64_t kOldGen = 41;
  constexpr uint64_t kNewGen = 42;
  auto old_s = make_incarnation(kTestIndex, kOldGen);
  auto new_s = make_incarnation(kTestIndex, kNewGen);

  // Stale error from the old incarnation lands first and occupies the slot.
  scheduler::notify_stream_error(
      old_s,
      std::make_exception_ptr(std::runtime_error("stale old-gen error")));

  // The new incarnation's real error arrives via notify WITHOUT an
  // intervening throw_if_stream_error — that ordering is what makes this an
  // ABA rather than the already-covered discard-on-throw path.
  scheduler::notify_stream_error(
      new_s,
      std::make_exception_ptr(std::runtime_error("live new-gen error")));

  // The new incarnation MUST observe its own error, not silence.
  bool threw = false;
  std::string msg;
  try {
    scheduler::throw_if_stream_error(new_s);
  } catch (const std::runtime_error& e) {
    threw = true;
    msg = e.what();
  }
  CHECK(threw);
  CHECK(msg.find("live new-gen error") != std::string::npos);

  // Inverse direction: a late duplicate from the OLD incarnation arriving
  // after the supersede must NOT surface on the new incarnation. (The slot
  // was consumed by the throw above, so this stash occupies it again under
  // the old generation; a new-gen throw discards it on mismatch.)
  scheduler::notify_stream_error(
      old_s,
      std::make_exception_ptr(std::runtime_error("late old-gen dup")));
  bool threw2 = false;
  try {
    scheduler::throw_if_stream_error(new_s);
  } catch (...) {
    threw2 = true;
  }
  CHECK_FALSE(threw2);
}
