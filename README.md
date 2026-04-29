# Adaptive Multiscale Hippocampal Model
# Author: Obadah Ghizawi

This folder contains the Webots version of the current adaptive multiscale model.
It keeps the neural model pieces needed for the 20x20 obstacle tests:

- obstacle-aware grid cells from `grid_cell_layer_v13.py`
- per-scale BVC input with Gaussian BVC context modulation
- unified multiscale place cells with all-scale recurrent inhibition
- unified recurrent STDP and Oja updates
- max-backup reward replay in `unified_reward_cell.py`
- a single Webots controller at `webots/controllers/multiscale_grid_controller`

Supported worlds:

- `20x20.wbt`
- `20x20_1obstacle.wbt`
- `20x20_2obstacles.wbt`
- `20x20_goalBehindWall.wbt`

Run entry point:

```bash
python webots/controllers/multiscale_grid_controller/multiscale_grid_controller.py
```

The controller saves networks under:

```text
webots/controllers/multiscale_grid_controller/pkl/<world>/networks
```

and hmap outputs under:

```text
webots/controllers/multiscale_grid_controller/pkl/<world>/hmaps
```
