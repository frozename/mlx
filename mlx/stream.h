// Copyright © 2023 Apple Inc.

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "mlx/api.h"
#include "mlx/device.h"

namespace mlx::core {

struct MLX_API Stream {
  int index;
  Device device;
  // Optional advisory tag used by higher layers (oMLX, mlx-lm) to record
  // model affinity. The backend MAY use this to key per-stream resources
  // such as command queues (#3491) or residency sets (#3492); when unset
  // (the default) the backend falls back to its global resources, so
  // existing call sites compile and behave unchanged. Not consulted by
  // equality / ordering: two streams with the same {index, device} are
  // considered identical regardless of tag.
  std::optional<std::string> tag = std::nullopt;
  explicit Stream(int index, Device device) : index(index), device(device) {}
  Stream(int index, Device device, std::string tag)
      : index(index), device(device), tag(std::move(tag)) {}

  bool operator==(const Stream& rhs) const {
    return index == rhs.index && device == rhs.device;
  }
  bool operator<(const Stream& rhs) const {
    return device < rhs.device || index < rhs.index;
  }
};

struct MLX_API ThreadLocalStream : public Stream {
  using Stream::Stream;
};

/** Get the default stream of current thread for the given device. */
MLX_API Stream default_stream(Device d);

/** Make the stream the default for its device on current thread. */
MLX_API void set_default_stream(Stream s);

/** Make a new stream on the given device. */
MLX_API Stream new_stream(Device d);

/** Make a new stream that will be unique per thread. */
MLX_API ThreadLocalStream new_thread_local_stream(Device d);

/** Get the stream for current thread from ThreadLocalStream. */
MLX_API Stream stream_from_thread_local_stream(ThreadLocalStream tls);

/** Get all available streams. */
MLX_API std::vector<Stream> get_streams();

/* Synchronize with the default stream. */
MLX_API void synchronize();

/* Synchronize with the provided stream. */
MLX_API void synchronize(Stream);

/* Synchronize with the stream corresponding to the current thread. */
MLX_API void synchronize(ThreadLocalStream);

/* Destroy all streams created in current thread. */
MLX_API void clear_streams();

} // namespace mlx::core
