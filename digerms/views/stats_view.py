# coding: utf-8
from .base import Group


class StatsView(Group):
    def __init__(self, stats, width, height, mode="full", debug=False):
        super(StatsView, self).__init__()
        self.stats = stats
        self.width = width
        self.height = height
        self.mode = mode
        self.debug = debug

    def add_to_batch(self, batch=None, parent=None):
        pass

    def update(self):
        pass
