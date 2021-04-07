import math


class Scheduler:
    def __init__(self, n_epoch):
        self.n_epoch = n_epoch


class ExponentialScheduler(Scheduler):
    def __init__(self, x_init, x_final, n_epoch):
        Scheduler.__init__(self, n_epoch)
        self.step_factor = math.exp(math.log(x_final / x_init) / n_epoch)

    def step(self, x):
        return x * self.step_factor
