# Design notes

## Pipeline

```
frame ─► undistort (optional) ─► mask extractor ─► warp to bird's-eye ─► lane search ─► polyfit
                                  (colour | Sobel,                       (windows, or      │
                                   morphology, ROI)                       around last fit)  ▼
overlay ◄─ unwarp lane polygon ◄─ curvature + offset ◄─ tracker (sanity checks, smoothing, fallback)
```

| Stage | Class / function | Module |
|---|---|---|
| Lens undistortion (optional) | `Undistorter`, `calibrate_from_chessboard` | `camera.py` |
| Binary lane mask | `ThresholdMaskExtractor` | `preprocess.py` |
| Bird's-eye view | `PerspectiveTransform` | `perspective.py` |
| Lane pixel search | `SlidingWindowFinder`, `search_around_fits` | `lanefit.py` |
| Polynomial fit | `fit_lane`, `LaneFit` | `lanefit.py` |
| Metric geometry | `LaneGeometry` | `lanefit.py` |
| Temporal tracking | `LaneTracker` | `tracker.py` |
| Orchestration | `LanePipeline` → `FrameResult` | `pipeline.py` |
| Baseline | `HoughLaneDetector` → `FrameResult` | `hough.py` |
| Drawing | `draw_lanes`, `debug_panel` | `overlay.py` |
| Metrics / evaluation | `score_frame`, `aggregate`, `evaluate` | `metrics.py`, `evaluation.py` |

### Class responsibilities

Each stage has one job and a small interface, so stages can be swapped:

- **Mask extractor**: any callable `frame -> uint8 mask (0/255)` of the frame's size. `LanePipeline`
  takes it as `mask_extractor=`, which is where a CNN segmentation model would plug in. Everything
  downstream stays the same.
- **`PerspectiveTransform`** owns the matrix and its inverse. It warps images and points, and turns a
  warped-space polynomial into a camera-space polyline (`unwarp_curve`).
- **`SlidingWindowFinder`** only finds pixels. Fitting is a separate function so the search can be
  replaced (e.g. by `search_around_fits` on later frames).
- **`LaneGeometry`** holds the pixel-to-metre scale and the camera position, and does all metric
  calculations.
- **`LaneTracker`** knows nothing about images. It takes fits in and gives fits out, which makes it
  easy to unit-test with hand-made polynomials.
- **Detectors** (`LanePipeline`, `HoughLaneDetector`) share `process(frame) -> FrameResult` and
  `reset()`. `FrameResult` always carries camera-space polylines, so drawing and evaluation do not
  depend on which detector produced them.

## Stage details and parameter choices

All points in the config are fractions of the frame size, so one config works at any resolution.
Window sizes are in warped pixels and were chosen for 1280x720 (TuSimple's resolution).

### Mask

| Parameter | Default | Why |
|---|---|---|
| `s_thresh` | 100–255 | Yellow paint is strongly saturated in HLS; asphalt and white paint are not. |
| `s_min_lightness` | 70 | In HLS, saturation is unstable for dark pixels, so shadows show up as "saturated". Requiring some lightness removes most of that noise. |
| `l_thresh` | 200–255 | White paint is the brightest thing on the road surface. |
| `sobel_thresh` | 40–255 | Absolute x-gradient of lightness. It is scaled by the **kernel's maximum possible response**, not by the frame's maximum. Scaling by the frame maximum (the common tutorial approach) stretches pure noise up to 255 when there are no lanes in view. |
| `morph_close` | 3 | Joins the two edges Sobel gives for each painted line. |
| `morph_open` | 0 (off) | An opening big enough to remove specks also erases far-away lines, which are only a few pixels wide. |

### Setting the perspective points

The `src` trapezoid's bottom corners should sit on the ego-lane lines in a frame where the road is
straight and flat; the top corners further up the same lines. Check by warping that frame: the lines
should come out vertical and parallel. `dst` puts the lane between 25% and 75% of the width, which
leaves room on both sides for curves and lateral drift.

The defaults are a starting point for a camera mounted on the centre-line looking straight ahead,
with the horizon at about 40% of the image height. They are not fitted to any real camera (on
TuSimple they score 35.6%). For each new camera:

1. Pick a straight-road frame from the development split.
2. Read off the four corner pixels and divide by width/height.
3. Warp it (`PerspectiveTransform.warp`, or the notebook's section 3) and adjust until the lines are vertical.
4. Set `geometry.view_length_m` by counting dashes in the warped image. US highway dashes are 3 m
   (10 ft) long with 9 m (30 ft) gaps, so a 12 m period gives the vertical scale.

**With labelled data, skip the manual steps.** `kaggle/tusimple_eval.py` derives the trapezoid from
the labels themselves. It fits a straight line to the near half of each ego lane in every training
frame, takes the median left and right lines (on average the car drives centred on a straight road),
and puts the corners on them. That produced [configs/tusimple.yaml](../configs/tusimple.yaml).

**`perspective.extend_to`.** The fitted curve only exists between the trapezoid top and the image
bottom. Lanes visible further up (TuSimple labels them to about a third of the image height) are
continued in a straight line from the curve's top quarter (`perspective.extend_up`). In the camera
image a straight lane is a straight line, so this is exact on straight roads and an approximation on
curves. Extending in the image avoids extrapolating the polynomial in the warped view, where
distant rows are stretched so much that a quadratic would quickly diverge. `null` (the default) keeps the
old behaviour.

### Lane search

- **Histogram**: column sums of the bottom half of the warped mask; the peak in each half is a lane
  base. An empty half means "no lane on that side", not a guess.
- **Sliding windows** (9 windows, half-width 100 px, re-centre after 50 px): robust from a cold start.
- **Search around the previous fit** (band of ±80 px): used once the tracker has history. It is faster
  and less likely to jump to a neighbouring lane. If it fails for either lane, the same frame is
  re-searched with windows.

### Fit and geometry

`numpy.polyfit` of `x = A*y^2 + B*y + C` per lane, in warped pixels, needing at least 150 pixels over
3 or more rows. For metres, `A_m = A * xm / ym^2` and `B_m = B * xm / ym`, with
`xm = 3.7 m / lane width in warped px` and `ym = view_length_m / warped height`.

- Radius at the row next to the car: `R = (1 + (2 A_m y + B_m)^2)^1.5 / |2 A_m|`. A straight line
  (`A = 0`) returns `inf`, not a division error. The two lanes are combined as the harmonic mean of the
  radii (the mean curvature), so one straight lane does not produce an infinite average.
- Offset: the camera is assumed to be on the image's vertical centre-line. That point is mapped into
  the warped image and compared with the lane centre at the bottom row. Positive means the car is
  right of centre.

### Tracking

A new pair of fits is accepted only if:

1. both lanes were found,
2. the lines do not cross,
3. the lane width next to the car is 2.8–4.8 m,
4. the width varies by less than 0.5 m (std) along the lane, so the lines are roughly parallel,
5. neither line moved more than 1.0 m from the smoothed estimate since the last frame.

Accepted fits go into a 5-frame history, and the output is their mean (or an EMA). A rejected frame
outputs the smoothed estimate instead. After 5 consecutive rejections the history is cleared so the
next frame starts a fresh window search. This is what lets the tracker follow a real lane change
instead of holding on to the old lane for ever. With no history, a single visible lane is passed through
(no width can be checked) and an implausible pair is dropped.

## Hough baseline

Gray → Gaussian blur → Canny → same ROI → `HoughLinesP`. Segments are split by slope sign and side of
the image, and one straight line `x = m*y + c` is fitted per side, weighted by segment length. It
reports curvature as `inf`, since a straight line cannot represent a curve.

## Evaluation choices

- **Metric**: TuSimple-style, as specified in the plan: a point is correct if it is within 20 px of the
  ground truth at the same row; a lane matches if at least 85% of its points are correct. GT lanes are
  matched one-to-one with predictions (best point accuracy first). Accuracy, FP and FN are pooled
  over all frames (points and lanes, not per-frame averages).
- **Ego lanes only** (default): the detectors only look for the two lines bordering the ego lane, while
  TuSimple labels up to 5 lanes. Scoring against every lane would mostly measure that design choice.
  `--all-lanes` scores against everything for comparison with published numbers. These are **not**
  directly comparable to official TuSimple leaderboard results, which use all lanes and an
  angle-adjusted threshold.
- **Temporal context**: TuSimple labels only the 20th frame of each 20-frame clip. The smoothed
  variant runs on frames 1–20 and is scored on frame 20; per-frame variants see only frame 20.
- **FPS**: detector time only (no disk reading or drawing), averaged over every processed frame,
  single-threaded Python on CPU.
- **Dev / test split**: tune on TuSimple's training clips (`label_data_*.json`), report on
  `test_label.json`. Do not tune on the frames you report.

## Known failure modes

These are what the design predicts. `docs/results.md` shows the ones observed on TuSimple: faint
markings, pavement texture and seams, construction zones, vehicle shadows and overpass shadows.

| Situation | Why it fails | Mitigation |
|---|---|---|
| Strong shadows / tree shade | Shadow edges pass the Sobel test; dark noisy pixels pass the S test. | `s_min_lightness`; tracker fallback; CNN mask. |
| Worn or faded paint | Too few pixels for the fit, or windows wander. | `min_fit_pixels`; search-around-fit; tracker. |
| Sharp curves | Lane leaves the warped view or the window margin; a quadratic is too simple. | Wider `dst` margin; more windows. |
| Hills / camera pitch | The flat-road homography no longer holds, so lines are not parallel when warped and `view_length_m` is wrong. | Per-frame pitch estimate (out of scope). |
| Lane change | The jump check rejects the new lane until the tracker resets. | `max_failures` reset (5 frames). |
| Night, glare, rain | The colour thresholds are fixed. | Adaptive thresholds or a learned mask. |
| Adjacent lanes, cars, road edges | Histogram peak lands on the wrong object. | ROI, base search limited to each image half. |
| A different camera | Perspective points are camera-specific. | Re-tune `perspective.src` per camera (above). |

## Synthetic data

`lanedet.synthetic` paints lanes with known polynomials in the bird's-eye view and projects them into
the camera view with the same transform the pipeline uses. Unit tests can then compare against exact
answers. Scenarios: straight, curve, shadow, worn paint, and dropout (right line disappears). The
roads are flat, clean and uncluttered, so synthetic scores show the code is correct, not that the
detector works on real roads.
