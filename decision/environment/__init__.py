from collections import namedtuple

Timestep = namedtuple("Timestep", "t observation action reward next_observation is_terminal info")

from environment.EnvironmentReckonMMDP import EnvironmentReckonMMDP
from environment.GarbageCollectionMMDP import GarbageCollectionMMDP
from environment.PanicButtonsMMDP import SmallPanicButtons, MediumPanicButtons, LargePanicButtons
