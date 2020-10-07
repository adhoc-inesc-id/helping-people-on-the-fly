from abc import ABC, abstractmethod


class Observer(ABC):

    def __init__(self, name):
        super().__init__()
        self._name = name
        #self._timestep_queue = Queue(maxsize=100)

    @property
    def name(self):
        return self._name

    def __call__(self, timestep):
        #self._timestep_queue.put(timestep)
        #timestep = self._timestep_queue.get()
        # TODO - Think how to asynchronously process data
        return self._process(timestep)

    @abstractmethod
    def _process(self, timestep):
        raise NotImplementedError()


class Heartbeat(Observer):

    def __init__(self):
        super().__init__("Heartbeat")

    def _process(self, timestep):
        print(timestep, flush=True)


class Metric(Observer, ABC):

    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    def result(self):
        raise NotImplementedError()
