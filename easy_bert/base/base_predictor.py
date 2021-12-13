import abc


class BasePredictor(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        pass
