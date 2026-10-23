import sys
import time
from collections import Counter
import numpy as np

from classical.feature_extractor import NavigationFeatureExtractor
from simulation.obstacles import ObstacleField, CircularObstacle, RectangularObstacle, MovingObstacle
from simulation.collision import check_boundary_violation
from config import DEFAULT_CONFIG

def create_10_synthetic_environments(env_id: int, rng: np.random.Generator, map_size: float = 50.0):
    """Generate 10 diverse procedural topologies for training without test leakage."""
    env_type = env_id % 10
    obs_list = []
    moving_list = []

    if env_type == 0:
        # 1. Open space: 5-8 small scattered circular obstacles
        desc = "Open Space"
        n = rng.integers(5, 9)
        for i in range(n):
            obs_list.append(CircularObstacle(i, rng.uniform(8.0, 42.0), rng.uniform(8.0, 42.0), rng.uniform(1.5, 2.8)))

    elif env_type == 1:
        # 2. Random obstacle clusters: 12-16 circular obstacles
        desc = "Random Obstacle Clusters"
        n = rng.integers(12, 17)
        for i in range(n):
            obs_list.append(CircularObstacle(i, rng.uniform(6.0, 44.0), rng.uniform(6.0, 44.0), rng.uniform(1.8, 3.2)))

    elif env_type == 2:
        # 3. Narrow corridors
        desc = "Narrow Corridors"
        cx = rng.choice([18.0, 25.0, 32.0])
        gap = rng.uniform(5.0, 6.5)
        obs_list.append(RectangularObstacle(1, 0.0, 15.0, cx - gap / 2.0, 35.0))
        obs_list.append(RectangularObstacle(2, cx + gap / 2.0, 15.0, map_size, 35.0))

    elif env_type == 3:
        # 4. Slalom: staggered horizontal/vertical barriers
        desc = "Slalom Course"
        y_levels = [14.0, 26.0, 38.0]
        for idx, y in enumerate(y_levels):
            x_c = 18.0 if idx % 2 == 0 else 32.0
            obs_list.append(RectangularObstacle(idx, x_c - 9.0, y - 2.0, x_c + 9.0, y + 2.0))

    elif env_type == 4:
        # 5. Chokepoints: central divider with narrow passage
        desc = "Chokepoints"
        cy = rng.uniform(22.0, 28.0)
        gap_x = rng.uniform(20.0, 30.0)
        gap_w = rng.uniform(5.0, 6.5)
        obs_list.append(RectangularObstacle(1, 0.0, cy - 2.5, gap_x - gap_w / 2.0, cy + 2.5))
        obs_list.append(RectangularObstacle(2, gap_x + gap_w / 2.0, cy - 2.5, map_size, cy + 2.5))
        obs_list.append(CircularObstacle(3, rng.uniform(10.0, 40.0), rng.uniform(10.0, 18.0), 2.5))

    elif env_type == 5:
        # 6. Perimeter traps: margin obstacles forcing inward flight
        desc = "Perimeter Traps"
        for i in range(5):
            obs_list.append(CircularObstacle(i, rng.uniform(6.0, 12.0), rng.uniform(8.0, 42.0), 2.5))
            obs_list.append(CircularObstacle(10 + i, rng.uniform(38.0, 44.0), rng.uniform(8.0, 42.0), 2.5))
        obs_list.append(RectangularObstacle(20, 22.0, 22.0, 28.0, 28.0))

    elif env_type == 6:
        # 7. Concave obstacle arrangements (U-shaped pocket)
        desc = "Concave U-Shaped Arrangement"
        u_x = rng.uniform(20.0, 30.0)
        u_y = rng.uniform(20.0, 30.0)
        # bottom
        obs_list.append(RectangularObstacle(1, u_x - 7.0, u_y - 6.0, u_x + 7.0, u_y - 3.0))
        # left wall
        obs_list.append(RectangularObstacle(2, u_x - 7.0, u_y - 3.0, u_x - 4.0, u_y + 6.0))
        # right wall
        obs_list.append(RectangularObstacle(3, u_x + 4.0, u_y - 3.0, u_x + 7.0, u_y + 6.0))

    elif env_type == 7:
        # 8. Multiple obstacle clusters (4 clusters)
        desc = "Multiple Obstacle Clusters"
        centers = [(14.0, 14.0), (36.0, 14.0), (14.0, 36.0), (36.0, 36.0)]
        oid = 0
        for cx, cy in centers:
            for _ in range(rng.integers(2, 4)):
                obs_list.append(CircularObstacle(oid, np.clip(cx + rng.uniform(-3.0, 3.0), 5.0, 45.0),
                                                 np.clip(cy + rng.uniform(-3.0, 3.0), 5.0, 45.0), rng.uniform(1.5, 2.6)))
                oid += 1

    elif env_type == 8:
        # 9. Dynamic obstacle situations
        desc = "Dynamic Obstacle Situations"
        for i in range(4):
            obs_list.append(CircularObstacle(i, rng.uniform(12.0, 38.0), rng.uniform(12.0, 38.0), rng.uniform(2.0, 3.0)))
        m_obs = MovingObstacle(
            obs_id=201,
            position=[float(rng.uniform(12.0, 25.0)), float(rng.uniform(15.0, 35.0))],
            radius=2.0,
            pattern="linear",
            velocity=[float(rng.uniform(0.6, 1.2)), float(rng.uniform(-0.4, 0.4))],
            bounds=(5.0, 5.0, 45.0, 45.0),
        )
        moving_list.append(m_obs)

    else:
        # 10. Dead-end-like configurations (box with single open side)
        desc = "Dead-End Configuration"
        bx = rng.uniform(20.0, 30.0)
        by = rng.uniform(20.0, 30.0)
        obs_list.append(RectangularObstacle(1, bx - 6.0, by - 6.0, bx + 6.0, by - 4.0)) # bottom
        obs_list.append(RectangularObstacle(2, bx - 6.0, by - 4.0, bx - 4.0, by + 6.0)) # left
        obs_list.append(RectangularObstacle(3, bx + 4.0, by - 4.0, bx + 6.0, by + 6.0)) # right
        # open at top (by + 6.0)

    return ObstacleField(static_obstacles=obs_list, moving_obstacles=moving_list), moving_list, desc

print("Loaded 10 environment generator.")
