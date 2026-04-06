import sys
sys.path.append(r"c:\Users\Obada\OneDrive\Desktop\AlabiHippocampalModel-1\HippocampalModel-Tests\AlabiHippocampalModel")
import pickle
import torch
import numpy as np

def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

base = r"c:\Users\Obada\OneDrive\Desktop\AlabiHippocampalModel-1\HippocampalModel-Tests\AlabiHippocampalModel\webots\controllers\msg_controller\pkl\environment_6"

# ============================================================
# SECTION 1: goal_associations.pkl
# ============================================================
print("=" * 70)
print("SECTION 1: goal_associations.pkl")
print("=" * 70)

ga_path = base + r"\networks\multi_goal_rewards\goal_associations.pkl"
with open(ga_path, "rb") as f:
    ga = pickle.load(f)

print("\n--- All keys ---")
for k in sorted(ga.keys()):
    print(f"  {k}: {type(ga[k]).__name__}")

print("\n--- checkpoint_pc_groups ---")
if "checkpoint_pc_groups" in ga:
    cpg = ga["checkpoint_pc_groups"]
    for cp_idx, groups in cpg.items():
        if isinstance(groups, dict):
            ga_count = len(groups.get("group_a", []))
            gb_count = len(groups.get("group_b", []))
            print(f"  cp{cp_idx}: group_a={ga_count} PCs, group_b={gb_count} PCs")
        else:
            print(f"  cp{cp_idx}: {type(groups).__name__} — {groups}")
else:
    print("  NOT FOUND")

print("\n--- room_partition / _room_partition_data ---")
for key in ga.keys():
    if "room" in key.lower() or "partition" in key.lower():
        val = ga[key]
        print(f"  {key}: {type(val).__name__}")
        if isinstance(val, dict):
            for k2, v2 in val.items():
                if isinstance(v2, (np.ndarray, torch.Tensor)):
                    print(f"    {k2}: shape={v2.shape}, dtype={v2.dtype}")
                elif isinstance(v2, (list, tuple)) and len(v2) > 10:
                    print(f"    {k2}: len={len(v2)}")
                else:
                    print(f"    {k2}: {v2}")
        elif isinstance(val, (np.ndarray, torch.Tensor)):
            print(f"    shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"    {val}")

# Check for any room/partition fields we might have missed
found_room = False
for key in ga.keys():
    if "room" in key.lower() or "partition" in key.lower():
        found_room = True
if not found_room:
    print("  No keys containing 'room' or 'partition' found.")

print("\n--- checkpoint_visit_counts ---")
if "checkpoint_visit_counts" in ga:
    print(f"  {ga['checkpoint_visit_counts']}")
else:
    print("  NOT FOUND")

print("\n--- checkpoint_crossing_counts ---")
if "checkpoint_crossing_counts" in ga:
    print(f"  {ga['checkpoint_crossing_counts']}")
else:
    print("  NOT FOUND")

# Print any other checkpoint-related keys
print("\n--- Other checkpoint/reward related data ---")
for key in sorted(ga.keys()):
    if any(kw in key.lower() for kw in ["checkpoint", "reward", "replay", "source", "goal", "visit", "cross"]):
        if key not in ["checkpoint_pc_groups", "checkpoint_visit_counts", "checkpoint_crossing_counts"]:
            val = ga[key]
            if isinstance(val, (np.ndarray, torch.Tensor)):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}, min={val.min():.4f}, max={val.max():.4f}")
            elif isinstance(val, dict):
                print(f"  {key}: dict with {len(val)} keys: {list(val.keys())[:20]}")
            elif isinstance(val, (list, tuple)):
                print(f"  {key}: {type(val).__name__} len={len(val)}")
            else:
                print(f"  {key}: {val}")


# ============================================================
# SECTION 2: unified_rcn_goal_goal.pkl
# ============================================================
print("\n" + "=" * 70)
print("SECTION 2: unified_rcn_goal_goal.pkl")
print("=" * 70)

rcn_path = base + r"\networks\multi_goal_rewards\unified_rcn_goal_goal.pkl"
with open(rcn_path, "rb") as f:
    rcn = pickle.load(f)

print(f"\nType: {type(rcn).__name__}")

# List all attributes
all_attrs = [a for a in dir(rcn) if not a.startswith("__")]
print(f"\nTotal attributes/methods: {len(all_attrs)}")

# Check w_in
if hasattr(rcn, "w_in"):
    w = rcn.w_in
    if isinstance(w, (np.ndarray, torch.Tensor)):
        w_np = to_np(w)
        print(f"\nw_in: shape={w.shape}, dtype={w.dtype}")
        print(f"  min={w_np.min():.6f}, max={w_np.max():.6f}, mean={w_np.mean():.6f}")
        print(f"  nonzero count: {np.count_nonzero(w_np)}/{w_np.size}")
    else:
        print(f"\nw_in: {type(w).__name__}")
else:
    print("\nw_in: NOT FOUND")

# Check reward_weights or similar
print("\n--- Reward-related attributes ---")
for attr in all_attrs:
    if any(kw in attr.lower() for kw in ["reward", "weight"]):
        val = getattr(rcn, attr)
        if isinstance(val, (np.ndarray, torch.Tensor)):
            v_np = to_np(val) if isinstance(val, (torch.Tensor, np.ndarray)) else val  # noqa
            print(f"  {attr}: shape={val.shape}, min={v_np.min():.6f}, max={v_np.max():.6f}, mean={v_np.mean():.6f}")
        elif callable(val):
            continue
        else:
            print(f"  {attr}: {type(val).__name__} = {val}")

# Attributes containing key terms
print("\n--- Attributes with key terms (room/mask/replay/checkpoint/source) ---")
for attr in all_attrs:
    if any(kw in attr.lower() for kw in ["room", "mask", "replay", "checkpoint", "source", "anchor", "local"]):
        val = getattr(rcn, attr)
        if callable(val) and not isinstance(val, (np.ndarray, torch.Tensor)):
            continue
        if isinstance(val, (np.ndarray, torch.Tensor)):
            v_np = to_np(val) if isinstance(val, (torch.Tensor, np.ndarray)) else val  # noqa
            print(f"  {attr}: shape={val.shape}, dtype={val.dtype}, min={v_np.min():.4f}, max={v_np.max():.4f}")
        elif isinstance(val, dict):
            print(f"  {attr}: dict with {len(val)} keys: {list(val.keys())[:20]}")
        elif isinstance(val, (list, tuple)):
            print(f"  {attr}: {type(val).__name__} len={len(val)}")
        else:
            print(f"  {attr}: {type(val).__name__} = {val}")

# goal_map_local_anchor_weights
if hasattr(rcn, "goal_map_local_anchor_weights"):
    glaw = rcn.goal_map_local_anchor_weights
    print(f"\n--- goal_map_local_anchor_weights ---")
    if isinstance(glaw, dict):
        for k, v in glaw.items():
            if isinstance(v, (np.ndarray, torch.Tensor)):
                v_np = to_np(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v  # noqa
                print(f"  {k}: shape={v.shape}, min={v_np.min():.4f}, max={v_np.max():.4f}, nonzero={np.count_nonzero(v_np)}")
            else:
                print(f"  {k}: {type(v).__name__}")
    elif isinstance(glaw, (np.ndarray, torch.Tensor)):
        v_np = glawto_np(glaw)
        print(f"  shape={glaw.shape}, min={v_np.min():.4f}, max={v_np.max():.4f}")
    else:
        print(f"  type={type(glaw).__name__}, value={glaw}")

# Print per-source/per-checkpoint replay info
print("\n--- Per-source / per-checkpoint data ---")
for attr in all_attrs:
    if any(kw in attr.lower() for kw in ["per_source", "per_checkpoint", "source_weights", "source_mask", "replay_mask"]):
        val = getattr(rcn, attr)
        if isinstance(val, (np.ndarray, torch.Tensor)):
            v_np = to_np(val) if isinstance(val, (torch.Tensor, np.ndarray)) else val  # noqa
            print(f"  {attr}: shape={val.shape}, min={v_np.min():.4f}, max={v_np.max():.4f}")
        elif isinstance(val, dict):
            print(f"  {attr}: dict keys={list(val.keys())[:20]}")
            for k, v in list(val.items())[:5]:
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    v_np = to_np(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v  # noqa
                    print(f"    {k}: shape={v.shape}, min={v_np.min():.4f}, max={v_np.max():.4f}")
        else:
            print(f"  {attr}: {type(val).__name__} = {val}")

# Also print ALL attribute names for reference
print("\n--- All non-dunder attribute names ---")
for attr in sorted(all_attrs):
    val = getattr(rcn, attr)
    if callable(val) and not isinstance(val, (np.ndarray, torch.Tensor)):
        continue
    if isinstance(val, (np.ndarray, torch.Tensor)):
        print(f"  {attr}: {type(val).__name__} shape={val.shape}")
    elif isinstance(val, dict):
        print(f"  {attr}: dict ({len(val)} keys)")
    elif isinstance(val, (list, tuple)):
        print(f"  {attr}: {type(val).__name__} len={len(val)}")
    elif isinstance(val, (int, float, bool, str)):
        print(f"  {attr}: {val}")
    else:
        print(f"  {attr}: {type(val).__name__}")


# ============================================================
# SECTION 3: unified_pcn.pkl
# ============================================================
print("\n" + "=" * 70)
print("SECTION 3: unified_pcn.pkl")
print("=" * 70)

pcn_path = base + r"\networks\unified_pcn.pkl"
with open(pcn_path, "rb") as f:
    pcn = pickle.load(f)

print(f"\nType: {type(pcn).__name__}")

# Basic info
for attr in ["num_pc_total", "num_scales", "scale_boundaries"]:
    if hasattr(pcn, attr):
        print(f"  {attr}: {getattr(pcn, attr)}")
    else:
        print(f"  {attr}: NOT FOUND")

# w_rec_unified
if hasattr(pcn, "w_rec_unified"):
    w = pcn.w_rec_unified
    w_np = to_np(w)
    print(f"\nw_rec_unified: shape={w_np.shape}, dtype={w_np.dtype}")
    print(f"  overall min={w_np.min():.6f}, max={w_np.max():.6f}, mean={w_np.mean():.6f}")

    # Per-scale block analysis
    if hasattr(pcn, "scale_boundaries"):
        sb = pcn.scale_boundaries
        n_scales = len(sb)
        print(f"\n  Per-scale block analysis (scale_boundaries={sb}):")
        for i in range(n_scales):
            s_i = sb[i][0] if isinstance(sb[i], (list, tuple)) else sb[i]
            e_i = sb[i+1][0] if i+1 < n_scales and isinstance(sb[i+1], (list, tuple)) else (sb[i+1] if i+1 < n_scales else w_np.shape[0])
            if i+1 >= n_scales:
                e_i = w_np.shape[0]
            else:
                e_i = sb[i+1][0] if isinstance(sb[i+1], (list, tuple)) else sb[i+1]
            s_i_val = s_i
            e_i_val = e_i

            block = w_np[..., s_i_val:e_i_val, s_i_val:e_i_val] if w_np.ndim == 3 else w_np[s_i_val:e_i_val, s_i_val:e_i_val]
            if block.size == 0:
                print(f"    Scale {i} diagonal block [{s_i_val}:{e_i_val}]: EMPTY")
                continue
            print(f"    Scale {i} diagonal block [{s_i_val}:{e_i_val}]: max={block.max():.6f}, mean={block.mean():.6f}, nonzero={np.count_nonzero(block)}")

            # Cross-scale
            for j in range(n_scales):
                if j == i:
                    continue
                s_j = sb[j][0] if isinstance(sb[j], (list, tuple)) else sb[j]
                e_j = sb[j+1][0] if j+1 < n_scales and isinstance(sb[j+1], (list, tuple)) else (sb[j+1] if j+1 < n_scales else w_np.shape[1])
                if j+1 >= n_scales:
                    e_j = w_np.shape[1]
                else:
                    e_j = sb[j+1][0] if isinstance(sb[j+1], (list, tuple)) else sb[j+1]
                cross = w_np[..., s_i_val:e_i_val, s_j:e_j] if w_np.ndim == 3 else w_np[s_i_val:e_i_val, s_j:e_j]
                if cross.size > 0 and cross.max() > 0:
                    print(f"    Cross scale {i}->{j} [{s_i_val}:{e_i_val}, {s_j}:{e_j}]: max={cross.max():.6f}, nonzero={np.count_nonzero(cross)}")
else:
    print("\nw_rec_unified: NOT FOUND")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
