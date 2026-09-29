#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif

extern "C" {

EXPORT double score_translation(
    const std::uint8_t* a,
    int aw,
    int ah,
    const std::uint8_t* b,
    int bw,
    int bh,
    int dx,
    int dy,
    double sigma,
    int alpha_threshold,
    int normaliser) {
  if (!a || !b || aw <= 0 || ah <= 0 || bw <= 0 || bh <= 0 || sigma <= 0.0) {
    return -std::numeric_limits<double>::infinity();
  }

  const int x0 = std::max(0, dx);
  const int y0 = std::max(0, dy);
  const int x1 = std::min(aw, dx + bw);
  const int y1 = std::min(ah, dy + bh);
  if (x1 <= x0 || y1 <= y0) {
    return -std::numeric_limits<double>::infinity();
  }

  const double inv_sigma = 1.0 / sigma;
  double total = 0.0;

  for (int y = y0; y < y1; ++y) {
    const int by = y - dy;
    for (int x = x0; x < x1; ++x) {
      const int bx = x - dx;
      const std::uint8_t* pa = a + (static_cast<std::size_t>(y) * aw + x) * 4u;
      const std::uint8_t* pb = b + (static_cast<std::size_t>(by) * bw + bx) * 4u;
      if (pa[3] <= alpha_threshold && pb[3] <= alpha_threshold) continue;

      const double dist = (
          std::abs(static_cast<int>(pa[0]) - static_cast<int>(pb[0])) +
          std::abs(static_cast<int>(pa[1]) - static_cast<int>(pb[1])) +
          std::abs(static_cast<int>(pa[2]) - static_cast<int>(pb[2])) +
          std::abs(static_cast<int>(pa[3]) - static_cast<int>(pb[3]))) * 0.25;
      const double z = dist * inv_sigma;
      total += std::exp(-(z * z));
    }
  }

  return total / static_cast<double>(std::max(normaliser, 1));
}

}
