// Copyright © 2023 Apple Inc.

#pragma once

#include <atomic>
#include <exception>
#include <future>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <unordered_map>

#include "mlx/api.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace mlx::core::scheduler {

struct StreamThread {
  std::mutex mtx;
  std::queue<std::function<void()>> q;
  std::condition_variable cond;
  bool stop;
  std::thread thread;

  StreamThread() : stop(false), thread(&StreamThread::thread_fn, this) {}

  ~StreamThread() {
    {
      std::lock_guard<std::mutex> lk(mtx);
      stop = true;
    }
    cond.notify_one();
    thread.join();
  }

  void thread_fn() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lk(mtx);
        cond.wait(lk, [this] { return !this->q.empty() || this->stop; });
        if (q.empty() && stop) {
          return;
        }
        task = std::move(q.front());
        q.pop();
      }

      task();
    }
  }

  void enqueue(std::function<void()> f) {
    {
      std::lock_guard<std::mutex> lk(mtx);
      if (stop) {
        throw std::runtime_error(
            "Cannot enqueue work after stream is stopped.");
      }
      q.emplace(std::move(f));
    }
    cond.notify_one();
  }
};

class MLX_API Scheduler {
 public:
  Scheduler();
  ~Scheduler();

  // Not copyable or moveable
  Scheduler(const Scheduler&) = delete;
  Scheduler(Scheduler&&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;
  Scheduler& operator=(Scheduler&&) = delete;

  void enqueue(Stream s, std::function<void()> task);

  void notify_new_task(const Stream& stream) {
    {
      std::lock_guard<std::mutex> lk(mtx);
      n_active_tasks_++;
    }
    completion_cv.notify_all();
  }

  void notify_task_completion(const Stream& stream) {
    {
      std::lock_guard<std::mutex> lk(mtx);
      n_active_tasks_--;
    }
    completion_cv.notify_all();
  }

  int n_active_tasks() const {
    return n_active_tasks_;
  }

  // Stash an exception raised on a stream from a background thread (e.g. a
  // Metal completion handler running on libdispatch). A C++ exception thrown
  // across a libdispatch block aborts the process unrecoverably, so the error
  // is recorded here and re-thrown on the caller thread at the next
  // synchronous waitpoint instead — see mlx-explore/mlx#2670. First error
  // wins per stream: later errors are dropped until the pending one is
  // consumed by throw_if_stream_error.
  //
  // The sentinel transition runs INSIDE `stream_error_mtx_` so it cannot race a
  // concurrent throw_if_stream_error lowering the sentinel after observing an
  // empty map.
  void stash_stream_error(const Stream& stream, std::exception_ptr eptr) {
    if (!eptr) {
      return;
    }
    std::lock_guard<std::mutex> lk(stream_error_mtx_);
    auto& err_slot = stream_errors_[stream.index];
    if (err_slot) {
      // First-error-wins: keep the earliest unconsumed error for this stream.
      return;
    }
    err_slot = std::move(eptr);
    has_pending_stream_error_.store(true, std::memory_order_release);
  }

  // If the given stream has a stashed error, clear and re-throw it on the
  // calling thread. Must be called from a synchronous waitpoint where the
  // exception can be caught (e.g. at the entry of eval/finalize/synchronize).
  //
  // Hot path: the atomic sentinel makes the common (no-error) case a single
  // acquire-load with no mutex acquisition, so this is safe to call on every
  // eval() without measurable throughput cost. The sentinel is mutated only
  // while holding `stream_error_mtx_`.
  //
  // Contention note: `has_pending_stream_error_` is process-global, not
  // per-stream. While ANY stream has an unconsumed error, error-free streams
  // fall off the fast path and briefly take `stream_error_mtx_` (a find-miss
  // no-op). The window normally ends at the erroring stream's next waitpoint,
  // which drains its entry and lowers the sentinel. It does NOT self-clear for
  // a stream that errors and is then abandoned (never waited on again and not
  // explicitly cleared): the entry persists and the sentinel stays set for the
  // process lifetime, so callers that can leak streams should clear_stream_error
  // them. A per-stream lock-free flag was rejected: stream indices are
  // unbounded, so any per-stream check needs the map (and lock) anyway; the
  // conservative global gate keeps the design simple and correct.
  void throw_if_stream_error(const Stream& stream) {
    if (!has_pending_stream_error_.load(std::memory_order_acquire)) {
      return;
    }
    std::exception_ptr eptr;
    {
      std::lock_guard<std::mutex> lk(stream_error_mtx_);
      auto it = stream_errors_.find(stream.index);
      if (it != stream_errors_.end()) {
        eptr = std::move(it->second);
        stream_errors_.erase(it);
      }
      // Sentinel transition INSIDE the lock so a concurrent
      // stash_stream_error cannot insert after we observed the map empty but
      // before we cleared the flag.
      if (stream_errors_.empty()) {
        has_pending_stream_error_.store(false, std::memory_order_release);
      }
    }
    if (eptr) {
      std::rethrow_exception(eptr);
    }
  }

  // Drop any stashed error for `stream` without rethrowing. Called from stream
  // teardown (clear_streams) to sweep out an error left by a completion handler
  // that fired during shutdown, so it neither leaks nor pins the sentinel on
  // the slow path.
  void clear_stream_error(const Stream& stream) {
    std::lock_guard<std::mutex> lk(stream_error_mtx_);
    stream_errors_.erase(stream.index);
    if (stream_errors_.empty()) {
      has_pending_stream_error_.store(false, std::memory_order_release);
    }
  }

  void wait_for_one() {
    std::unique_lock<std::mutex> lk(mtx);
    int n_tasks_old = n_active_tasks();
    if (n_tasks_old > 1) {
      completion_cv.wait(lk, [this, n_tasks_old] {
        return this->n_active_tasks() < n_tasks_old;
      });
    }
  }

 private:
  friend Stream mlx::core::new_stream(Device d);

  int n_active_tasks_{0};
  std::unordered_map<int, std::unique_ptr<StreamThread>> threads_;
  std::shared_mutex threads_mtx_;
  std::condition_variable completion_cv;
  std::mutex mtx;

  // Per-stream stash for exceptions raised on background threads (Metal
  // completion handlers), drained and rethrown at synchronous waitpoints.
  // Keyed by stream index; guarded by `stream_error_mtx_`.
  std::unordered_map<int, std::exception_ptr> stream_errors_;
  std::mutex stream_error_mtx_;
  // Hot-path sentinel: true iff at least one stream has a stashed error. Lets
  // `throw_if_stream_error` short-circuit on the common no-error path without
  // touching the mutex. Mutated only under `stream_error_mtx_`.
  std::atomic<bool> has_pending_stream_error_{false};
};

MLX_API Scheduler& scheduler();

template <typename F>
void enqueue(const Stream& stream, F&& f) {
  scheduler().enqueue(stream, std::forward<F>(f));
}

inline int n_active_tasks() {
  return scheduler().n_active_tasks();
}

inline void notify_new_task(const Stream& stream) {
  scheduler().notify_new_task(stream);
}

inline void notify_task_completion(const Stream& stream) {
  scheduler().notify_task_completion(stream);
}

inline void stash_stream_error(
    const Stream& stream,
    std::exception_ptr eptr) {
  scheduler().stash_stream_error(stream, std::move(eptr));
}

inline void throw_if_stream_error(const Stream& stream) {
  scheduler().throw_if_stream_error(stream);
}

inline void clear_stream_error(const Stream& stream) {
  scheduler().clear_stream_error(stream);
}

inline void wait_for_one() {
  scheduler().wait_for_one();
}

} // namespace mlx::core::scheduler
