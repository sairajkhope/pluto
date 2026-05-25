# Cleanup candidates

Output of Pass 3 of the Cobalt pre-cleanup. Nothing in this file is auto-deleted — each item should be reviewed before action. Run `ruff check --select F --no-fix toddlerbot/` and `vulture toddlerbot/ --min-confidence 80` to refresh.

## Build system: `CMakeLists.txt` + `[tool.scikit-build]`

`CMakeLists.txt` is a no-op (its own comment says modules were removed in v2.0 in favor of pure Python). But `pyproject.toml` still uses `scikit-build-core` as its build backend, so deleting `CMakeLists.txt` will break `pip install -e .`.

Two options:

1. **Migrate to `hatchling` or `setuptools`.** Drop `CMakeLists.txt`, remove `pybind11` from build-system requires, switch backend. Recommended — cleaner installs, faster CI, no C++ toolchain dependency.
2. **Leave as-is.** Wastes ~10 s per install on a no-op CMake configure, but zero risk.

Suggested for Cobalt: option 1, but as a deliberate small commit ("migrate build to hatchling"), not bundled with the rename.

## Ruff findings — `ruff check --select F`

42 errors total, broken down:

| Rule | Count | What |
|---|---|---|
| F401 | 20 | Unused imports |
| F405 | 12 | Name may be undefined / from star import |
| F841 | 8 | Unused local variable |
| F541 | 1 | f-string without placeholders |
| F403 | 1 | Star import (`from X import *`) |

Run `.venv/bin/ruff check --select F --no-fix toddlerbot/` for the full list with file paths. About 11 are auto-fixable via `--fix`; the rest need manual review.

### Concentration

Most ruff hits cluster in `manipulation/teleoperation/general_motion_retargeting/` — that subdirectory contains vendored third-party code (`lafan_vendor/`, `optitrack_vendor/`) inherited as-is. Decide once whether to:
- Re-vendor cleanly from upstream
- Replace with a pip dependency if one exists
- Apply local fixes (and re-apply on each upstream update)

## Vulture findings (80% confidence)

```
toddlerbot/manipulation/datasets/get_pose_data.py:15            unused var 'time_offset'
toddlerbot/manipulation/datasets/process_raw_data.py:18         unused var 'time_offset'
toddlerbot/manipulation/teleoperation/general_motion_retargeting/optitrack_vendor/MoCapData.py:420   unsatisfiable 'if' condition
toddlerbot/manipulation/teleoperation/general_motion_retargeting/optitrack_vendor/NatNetClient.py:718  unreachable code after 'return'
toddlerbot/manipulation/teleoperation/general_motion_retargeting/robot_motion_viewer.py:58           unused var 'visualize_mujoco'
toddlerbot/manipulation/teleoperation/quest_robot_module.py:947                                      unused var 'left_wrist_pos'
toddlerbot/manipulation/utils/teleop_utils.py:39                                                     unused var 'orientation_corrections'
toddlerbot/policies/teleop_vr_leader.py:19                                                           unused import 'RobotMotionViewer'
toddlerbot/sim/mujoco_utils.py:244                                                                    unused var 'vis_data'
toddlerbot/tools/edit_keyframe.py:1433                                                                unused var 'dest'
toddlerbot/tools/edit_keyframe.py:1618                                                                unused var 'dest'
toddlerbot/visualization/vis_depth_comparison.py:217                                                  unused var 'opacity'
```

Most of these are either intentional placeholders (`visualize_mujoco` likely a future-feature stub) or accidental drops (`time_offset` declared but never threaded through). Worth a 30-minute pass to delete or wire them in.

## Whitelist false positives

Once you've reviewed: generate a vulture whitelist of confirmed-not-dead items so future runs are clean.

```
.venv/bin/vulture toddlerbot/ --make-whitelist > .vulture-whitelist.py
```

Edit `.vulture-whitelist.py` to remove items you actually want flagged (true positives stay reported), then commit it. Future runs: `.venv/bin/vulture toddlerbot/ .vulture-whitelist.py`.

## Suggested order of action

1. **Easy wins** — run `ruff check --fix toddlerbot/` to delete the 11 auto-fixable issues (mostly unused imports). Review the diff before committing.
2. **Vendor decision** — choose how to handle `general_motion_retargeting/` vendored code; this removes most remaining F405/F403 noise.
3. **Manual var cleanup** — the 8 F841 unused-variable hits and the 12 vulture findings. ~30 minutes.
4. **Build system migration** — `CMakeLists.txt` → `hatchling`. Schedule deliberately.

## What was NOT scanned

- `tests/` — separate scan worth doing once Cobalt's test policy is settled.
- `examples/` — entry-point scripts often have intentional "unused" imports for side-effects; review by hand.
- `motion/migrate_*.py`, `motion/print.py`, `motion/render.py` — Toddlerbot 1.0 migration utilities; consider deleting outright if Cobalt never needs to migrate from a 1.0 format.
