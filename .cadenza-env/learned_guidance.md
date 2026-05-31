### Improved Guidance
1. **Descend into debris valley**: If moving down the ramp, continue with `walk_forward`. If stuck, try `precision_turn_left` to realign with the ramp. If still stuck, try `crawl_forward` to navigate through tight spaces.
2. **Scan debris field**: If the sphere is in view, stop and signal with `signal_sphere`. If not, sweep left with `turn_left` and then right with `turn_right` to cover more area.
3. **Signal the sphere**: If the sphere is in view, signal it with `signal_sphere`. If not, do not signal and try to move towards the sphere with `walk_forward` or `crawl_forward`.
4. **Ascend out of debris valley**: If moving up the ramp, continue with `walk_forward`. If stuck, try `precision_turn_right` to navigate through tight spaces.