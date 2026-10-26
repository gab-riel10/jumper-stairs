"""MDP terms for jumper.stairs (specific to this task).

The staircase is `terrain.py`; everything else reads it through the command
term in `commands.py`. `jumper.stairs_sighted` mounts the same terms.
"""

from . import commands, curriculum, events, observations, rewards, terminations, terrain

__all__ = ["commands", "curriculum", "events", "observations", "rewards", "terminations", "terrain"]
