from environment.PanicButtonsMMDP import PanicButtonsMMDP


class PanicButtons(PanicButtonsMMDP):

    def __init__(self, n, teammate, config, render=False):
        initial_position = (0, 0, n-1, 0)
        if config == 1: panic_location = [0, n-1, n-1, n-1]
        elif config == 2: panic_location = [int((n-1) / 2), n-1, n-1, 0]
        elif config == 3: panic_location = [0, 0, int((n-1) / 2), n-1]
        else: raise ValueError("Invalid config (pick 1, 2 or 3)")
        super(PanicButtons, self).__init__(n, n, initial_position, panic_location, teammate, render)

SmallPanicButtons = lambda teammate, config: PanicButtons(3, teammate, config)
MediumPanicButtons = lambda teammate, config: PanicButtons(4, teammate, config)
LargePanicButtons = lambda teammate, config: PanicButtons(5, teammate, config)
