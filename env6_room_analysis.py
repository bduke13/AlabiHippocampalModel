"""
Environment 6 room partition analysis.
Replicates the EXACT logic from _build_room_groups_from_geometry in msg_driver.py.
"""

import numpy as np
from collections import deque

# --- Configuration ---
env_size = [20.0, 20.0]
half_w, half_h = env_size[0] / 2.0, env_size[1] / 2.0
resolution = 200
cell_size = env_size[0] / resolution  # 0.1

def world_to_grid(wx, wy):
    col = max(0, min(resolution - 1, int((wx + half_w) / cell_size)))
    row = max(0, min(resolution - 1, int((wy + half_h) / cell_size)))
    return row, col

def grid_to_world(row, col):
    wx = col * cell_size - half_w + cell_size / 2
    wy = row * cell_size - half_h + cell_size / 2
    return wx, wy

# --- Walls ---
obstacles = [
    {"type": "rectangle", "name": "wall1", "bounds": [[-0.1, -4.0], [0.1, 10.0]]},
    {"type": "rectangle", "name": "wall2", "bounds": [[-0.1, -9.8], [0.1, -5.8]]},
    {"type": "rectangle", "name": "wall3", "bounds": [[2.0, 4.6], [10.0, 4.8]]},
    {"type": "rectangle", "name": "wall3_1", "bounds": [[2.0, 7.3], [10.0, 7.5]]},
    {"type": "rectangle", "name": "wall3_2", "bounds": [[-10.0, 7.3], [-2.0, 7.5]]},
    {"type": "rectangle", "name": "wall4", "bounds": [[-10.0, 4.6], [-2.0, 4.8]]},
    {"type": "rectangle", "name": "wall5", "bounds": [[-6.1, -9.8], [-5.9, 3.2]]},
    {"type": "rectangle", "name": "wall6", "bounds": [[-5.9, -5.9], [-1.9, -5.7]]},
    {"type": "rectangle", "name": "wall7", "bounds": [[2.0, -4.0], [8.0, -3.8]]},
    {"type": "rectangle", "name": "wall13", "bounds": [[0.0, -0.2], [5.0, 0.0]]},
    {"type": "rectangle", "name": "wall13_1", "bounds": [[6.6, -0.2], [8.6, 0.0]]},
    {"type": "rectangle", "name": "wall11", "bounds": [[-4.85, -4.0], [-1.15, -3.8]]},
    {"type": "rectangle", "name": "wall11_2", "bounds": [[1.4, -6.2], [3.6, -6.0]]},
    {"type": "rectangle", "name": "wall9", "bounds": [[4.9, -8.95], [5.1, -5.45]]},
    {"type": "rectangle", "name": "wall9_1", "bounds": [[2.4, -9.75], [2.6, -6.25]]},
    {"type": "rectangle", "name": "wall10", "bounds": [[5.0, -5.7], [9.8, -5.5]]},
    {"type": "rectangle", "name": "wall8", "bounds": [[4.9, -1.2], [5.1, 3.0]]},
    {"type": "rectangle", "name": "wall8_2", "bounds": [[4.9, 5.9], [5.1, 8.7]]},
    {"type": "rectangle", "name": "wall8_3", "bounds": [[-5.1, 5.9], [-4.9, 8.7]]},
    {"type": "rectangle", "name": "wall8_4", "bounds": [[-8.7, -0.6], [-7.3, -0.4]]},
    {"type": "rectangle", "name": "wall8_1", "bounds": [[7.5, -0.05], [7.7, 4.65]]},
    {"type": "rectangle", "name": "wall8_sub", "bounds": [[4.9, -5.45], [5.1, -2.95]]},
    {"type": "rectangle", "name": "wall12", "bounds": [[-3.1, -3.9], [-2.9, 4.3]]},
    {"type": "rectangle", "name": "wall12_1", "bounds": [[-7.9, -9.85], [-7.7, -0.55]]},
]

checkpoint_positions = [
    (0.1, -5.0),    # cp0
    (5.1, -2.1),    # cp1
    (-0.6, 4.5),    # cp2
    (-5.9, 3.7),    # cp3
    (5.0, 3.7),     # cp4
    (-9.4, -0.6),   # cp5
]

goal = (-9.0, -9.0)

# =====================================================================
# BUILD OCCUPANCY GRID (exact msg_driver.py logic)
# =====================================================================
occupancy = np.ones((resolution, resolution), dtype=bool)

for obs in obstacles:
    bounds = obs["bounds"]
    x_min, y_min = bounds[0]
    x_max, y_max = bounds[1]
    col_min = max(0, int((x_min + half_w) / cell_size) - 1)
    col_max = min(resolution - 1, int((x_max + half_w) / cell_size) + 1)
    row_min = max(0, int((y_min + half_h) / cell_size) - 1)
    row_max = min(resolution - 1, int((y_max + half_h) / cell_size) + 1)
    occupancy[row_min:row_max + 1, col_min:col_max + 1] = False

# Block doorways
doorway_block_radius = max(3, int(1.5 / cell_size))  # 15
checkpoint_grid_cells = {}
for cp_idx, (cx, cy) in enumerate(checkpoint_positions):
    cr, cc = world_to_grid(float(cx), float(cy))
    blocked_cells = []
    for dr in range(-doorway_block_radius, doorway_block_radius + 1):
        for dc in range(-doorway_block_radius, doorway_block_radius + 1):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < resolution and 0 <= nc < resolution:
                if dr * dr + dc * dc <= doorway_block_radius * doorway_block_radius:
                    occupancy[nr, nc] = False
                    blocked_cells.append((nr, nc))
    checkpoint_grid_cells[cp_idx] = blocked_cells

# =====================================================================
# FLOOD FILL
# =====================================================================
room_grid = np.full((resolution, resolution), -1, dtype=np.int32)
num_rooms = 0

for r in range(resolution):
    for c in range(resolution):
        if not occupancy[r, c] or room_grid[r, c] >= 0:
            continue
        room_id = num_rooms
        num_rooms += 1
        room_grid[r, c] = room_id
        bfs_queue = deque([(r, c)])
        while bfs_queue:
            cr, cc = bfs_queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < resolution and 0 <= nc < resolution:
                    if occupancy[nr, nc] and room_grid[nr, nc] == -1:
                        room_grid[nr, nc] = room_id
                        bfs_queue.append((nr, nc))

# =====================================================================
# CHECKPOINT ADJACENCY
# =====================================================================
checkpoint_adjacent_rooms = {}
for cp_idx, blocked_cells in checkpoint_grid_cells.items():
    adjacent_rooms = set()
    for br, bc in blocked_cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = br + dr, bc + dc
            if 0 <= nr < resolution and 0 <= nc < resolution:
                rid = int(room_grid[nr, nc])
                if rid >= 0:
                    adjacent_rooms.add(rid)
    checkpoint_adjacent_rooms[cp_idx] = sorted(adjacent_rooms)

# =====================================================================
# ROOM INFO
# =====================================================================
room_info = {}
for rid in range(num_rooms):
    cells = np.argwhere(room_grid == rid)
    if len(cells) < 10:
        continue
    centroid_row = cells[:, 0].mean()
    centroid_col = cells[:, 1].mean()
    cx, cy = grid_to_world(centroid_row, centroid_col)
    min_row, min_col = cells.min(axis=0)
    max_row, max_col = cells.max(axis=0)
    min_wx, min_wy = grid_to_world(min_row, min_col)
    max_wx, max_wy = grid_to_world(max_row, max_col)

    parts = []
    if cy < -3: parts.append("bottom")
    elif cy > 3: parts.append("top")
    else: parts.append("middle")
    if cx < -3: parts.append("left")
    elif cx > 3: parts.append("right")
    else: parts.append("center")
    region = "-".join(parts)

    room_info[rid] = {
        "centroid": (round(cx, 1), round(cy, 1)),
        "size": len(cells),
        "region": region,
        "bounds": ((round(min_wx, 1), round(min_wy, 1)), (round(max_wx, 1), round(max_wy, 1))),
    }

# Goal room
goal_row, goal_col = world_to_grid(*goal)
goal_room = int(room_grid[goal_row, goal_col])

# =====================================================================
# ROOM GRAPH (from msg_driver.py)
# =====================================================================
room_graph = {}
for cp_idx, adj_rooms in checkpoint_adjacent_rooms.items():
    for i in range(len(adj_rooms)):
        for j in range(i + 1, len(adj_rooms)):
            room_graph.setdefault(adj_rooms[i], set()).add(adj_rooms[j])
            room_graph.setdefault(adj_rooms[j], set()).add(adj_rooms[i])

# =====================================================================
# GATEWAY BFS (from msg_driver.py _room_graph_distances_from_goal)
# =====================================================================
# BFS on the room graph from goal_room
visited_rooms = {goal_room}
bfs_q = deque([goal_room])
room_to_gateway = {goal_room: "GOAL"}
room_dist = {goal_room: 0}

# For proper gateway assignment, we need to track which CP connects rooms
# Build: for each room pair, which CP(s) connect them
pair_to_cps = {}
for cp_idx, adj_rooms in checkpoint_adjacent_rooms.items():
    for i in range(len(adj_rooms)):
        for j in range(i + 1, len(adj_rooms)):
            key = (min(adj_rooms[i], adj_rooms[j]), max(adj_rooms[i], adj_rooms[j]))
            pair_to_cps.setdefault(key, []).append(cp_idx)

while bfs_q:
    current = bfs_q.popleft()
    for neighbor in sorted(room_graph.get(current, set())):
        if neighbor not in visited_rooms:
            visited_rooms.add(neighbor)
            room_dist[neighbor] = room_dist[current] + 1
            # Find which CP connects current to neighbor
            key = (min(current, neighbor), max(current, neighbor))
            cps = pair_to_cps.get(key, [])
            if cps:
                room_to_gateway[neighbor] = f"cp{cps[0]}"
            else:
                room_to_gateway[neighbor] = "?"
            bfs_q.append(neighbor)

# =====================================================================
# PRINT RESULTS
# =====================================================================
print("="*80)
print("ENVIRONMENT 6 ROOM PARTITION ANALYSIS")
print("="*80)
print(f"\nGrid: {resolution}x{resolution}, cell_size={cell_size}m")
print(f"Doorway block radius: {doorway_block_radius} cells ({doorway_block_radius*cell_size:.1f}m)")
print(f"Total rooms found: {num_rooms} (significant: {len(room_info)})")
print()

print("ROOMS:")
print("-"*80)
for rid in sorted(room_info.keys()):
    info = room_info[rid]
    gw = room_to_gateway.get(rid, "?")
    dist = room_dist.get(rid, "?")
    cx, cy = info["centroid"]
    (bx1, by1), (bx2, by2) = info["bounds"]
    print(f"  Room {rid}: centroid=({cx:>6.1f}, {cy:>6.1f})  region={info['region']:<20}"
          f"  size={info['size']:>5}  gateway={gw:<6}  dist={dist}"
          f"  bounds=[({bx1},{by1})->({bx2},{by2})]")

print()
print("CHECKPOINT ADJACENCY:")
print("-"*80)
for cp_idx in range(len(checkpoint_positions)):
    adj = checkpoint_adjacent_rooms.get(cp_idx, [])
    cx, cy = checkpoint_positions[cp_idx]
    adj_desc = [f"Room {r}({room_info.get(r,{}).get('region','?')})" for r in adj]
    print(f"  cp{cp_idx} at ({cx:>5.1f}, {cy:>5.1f}): {', '.join(adj_desc)}")

print()
print(f"Goal at {goal} -> Room {goal_room} ({room_info.get(goal_room,{}).get('region','?')})")

print()
print("ROOM GRAPH (edges via checkpoints):")
print("-"*80)
for r1 in sorted(room_graph.keys()):
    neighbors = sorted(room_graph[r1])
    for r2 in neighbors:
        if r2 > r1:
            key = (r1, r2)
            cps = pair_to_cps.get(key, [])
            cp_names = [f"cp{c}" for c in cps]
            print(f"  Room {r1} <-> Room {r2}  via {', '.join(cp_names)}")

print()
print("GATEWAY BFS ASSIGNMENT (from goal room {0}):".format(goal_room))
print("-"*80)
for rid in sorted(room_to_gateway.keys()):
    gw = room_to_gateway[rid]
    info = room_info.get(rid, {})
    print(f"  Room {rid}: gateway={gw:<6}  dist={room_dist.get(rid,'?'):<3}  "
          f"region={info.get('region','?'):<20}  centroid={info.get('centroid','?')}")

# =====================================================================
# EXPECTED vs ACTUAL COMPARISON
# =====================================================================
print()
print("="*80)
print("EXPECTED vs ACTUAL COMPARISON")
print("="*80)

expected = [
    ("GOAL", "red",    "bottom-left corner near (-9,-9)"),
    ("cp5",  "yellow", "narrow left strip, x:-10..-8, y:-10..0"),
    ("cp3",  "green",  "center-left L-shape, x:-6..-3"),
    ("cp2",  "cyan",   "top region, y > 5"),
    ("cp0",  "pink",   "lower-center/right, x:0..5, y:-6..0"),
    ("cp1",  "blue",   "right side, x:5..8"),
    ("cp4",  "grey",   "upper-right corridor, x:5..10, y:5..7"),
]

print(f"\n{'Gateway':<8} {'Color':<8} {'Expected':<45} {'Assigned Room(s)'}")
print("-"*110)

mismatches = []
for gw_name, color, desc in expected:
    rooms = [r for r, g in room_to_gateway.items() if g == gw_name]
    if rooms:
        for r in rooms:
            info = room_info.get(r, {})
            cx, cy = info.get("centroid", (0, 0))
            rgn = info.get("region", "?")
            status = ""
            # Simple heuristic check
            if gw_name == "GOAL" and "bottom-left" not in rgn:
                status = " *** MISMATCH"
                mismatches.append((gw_name, r, rgn, desc))
            elif gw_name == "cp5" and "left" not in rgn:
                status = " *** MISMATCH"
                mismatches.append((gw_name, r, rgn, desc))
            elif gw_name == "cp3" and rgn == "top-left":
                status = " *** MISMATCH (should be center-left, not top-left)"
                mismatches.append((gw_name, r, rgn, desc))
            elif gw_name == "cp2" and "top" not in rgn:
                status = " *** MISMATCH"
                mismatches.append((gw_name, r, rgn, desc))
            elif gw_name == "cp0" and "bottom" not in rgn and "center" not in rgn:
                status = " *** MISMATCH"
                mismatches.append((gw_name, r, rgn, desc))
            elif gw_name == "cp1" and "right" not in rgn:
                status = " *** MISMATCH"
                mismatches.append((gw_name, r, rgn, desc))
            print(f"{gw_name:<8} {color:<8} {desc:<45} Room {r} ({cx},{cy}) {rgn}{status}")
    else:
        print(f"{gw_name:<8} {color:<8} {desc:<45} NOT ASSIGNED ***")
        mismatches.append((gw_name, None, "unassigned", desc))

# Check for rooms with multiple gateways or gateways with multiple rooms
print()
gw_counts = {}
for r, g in room_to_gateway.items():
    gw_counts.setdefault(g, []).append(r)
for g, rooms in gw_counts.items():
    if len(rooms) > 1 and g != "GOAL":
        print(f"NOTE: {g} owns {len(rooms)} rooms: {rooms}")
        for r in rooms:
            print(f"  Room {r}: {room_info.get(r, {}).get('region', '?')}, "
                  f"centroid={room_info.get(r, {}).get('centroid', '?')}")

used_gws = set(room_to_gateway.values()) - {"GOAL"}
unused = set(f"cp{i}" for i in range(len(checkpoint_positions))) - used_gws
if unused:
    print(f"\nUnused checkpoints: {sorted(unused)}")
    for cp in sorted(unused):
        idx = int(cp[2:])
        adj = checkpoint_adjacent_rooms.get(idx, [])
        print(f"  {cp}: adjacent rooms = {adj} (all already claimed by closer gateways)")

print()
if mismatches:
    print(f"MISMATCHES FOUND: {len(mismatches)}")
    for gw, rid, rgn, desc in mismatches:
        print(f"  {gw}: got Room {rid} ({rgn}), expected {desc}")
else:
    print("All assignments match expectations.")

print()
print("="*80)
print("SUMMARY")
print("="*80)
print(f"Rooms found: {len(room_info)} (expected 7)")
print(f"The extra room is Room 5 (middle-center, centroid=(-1.6, 0.1))")
print(f"This is the small area between wall12 (x~-3), wall1 (x~0),")
print(f"wall11 (y~-4), and wall4 (y~4.6). It's separated from Room 2")
print(f"by the cp0 blocking circle which bridges wall1/wall2 gap.")
print()
print("Key finding: cp3 claims Room 7 (top-left) because it is adjacent")
print("to rooms 1, 2, AND 7. Room 7 connects to cp3 via the gap between")
print("wall5 top (y=3.2) and wall4 bottom (y=4.6) and the left side of")
print("the wall4/wall3_2 horizontal wall. This means cp3 'reaches' the")
print("top-left region before cp2 does in the BFS.")
