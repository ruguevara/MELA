#!/usr/bin/env python
# coding: utf-8

import argparse
from glob import glob
from random import choice
import numpy as np
import sys

from environment import Environment, FOOD_INIT_PROB, Walls
from messia import PolulationWithMessia
from population import Population
from settings import POPULATION_SIZE, GRID_SCALE
from statistics import Statistics
from views.application import Application
from views.env_view import EnvironmentView, ExperimentMode
from views.stats_view import StatsView

DEFAULTS = dict(
    debug=True,
    fullscreen=False,
    size='1200x600',
    fps=60,
    show_fps=True,
    paused=True,
    random_maze=False,
    screen=0
)

def main():
    parser = argparse.ArgumentParser(description="Digital germs %(prog)s", prog="digerms")

    parser.add_argument("--fps", "-f",  type=int, help="Frames per second")
    parser.add_argument("--fullscreen", "-F", action="store_true", help="Fullscreen mode")
    parser.add_argument("--screen", "-S", action="store", help="Screen number")
    parser.add_argument("--show-fps",    action="store_true", help="Display FPS count")
    parser.add_argument("--paused",      action="store_true", help="Start paused")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--size", "-s", type=str, help="Field size")
    group.add_argument("--random-maze", "-r", action="store_true", help="Random maze")
    group.add_argument("--maze", "-m", type=argparse.FileType('r'), help="Maze name or path to a file")
    parser.set_defaults(**DEFAULTS)
    options = parser.parse_args()

    params = dict(fullscreen=options.fullscreen,
                  visible=False,
                  vsync=False,
                  fps=options.fps,
                  show_fps=options.show_fps,
                  screen=int(options.screen))

    maze = None
    grid_size = None

    # guessing dimensions:
    # from fullscreen dimensions
    # from maze
    # from size parameter
    if options.maze:
        maze = Walls.load(options.maze)
        grid_size = maze.shape
    elif options.random_maze:
        maze = Walls.load(choice(list(glob('mazes/*.npy'))))
        grid_size = maze.shape
    if options.fullscreen:
        app = Application(**params)
    else:
        # TODO понять, как определять высоту из высоты лабиринта и высоты статистики
        if maze is None:
            size = map(int, options.size.split('x'))
            grid_size = (size[1] // GRID_SCALE, size[0] // GRID_SCALE)
        width = grid_size[1] * GRID_SCALE
        height = grid_size[0] * GRID_SCALE
        params.update(dict(width=width, height=height+160,))
        app = Application(**params)

    # population = PolulationWithMessia.random(POPULATION_SIZE)
    population = Population.random(POPULATION_SIZE)
    env_view = ExperimentMode(app.window, population, maze, stats_height=160, debug=options.debug)
    env_view.paused = options.paused
    app.set_mode(env_view, 1./2000)
    app.run()

if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=120, suppress=False)
    # np.set_printoptions(precision=2, linewidth=120, suppress=True)
    main()

