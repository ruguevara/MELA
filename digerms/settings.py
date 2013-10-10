# coding: utf-8
import math

POPULATION_SIZE = 500
SELECT_RATIO = 0.95
# RANDOM_RATIO = 0
RANDOM_RATIO = 0.01

ROT_SPEED = 0.5
# масштаб сетки
GRID_SCALE = 8

MAX_HEALTH = 50000
MIN_HEALTH = 500
# сколько энергии тратится на ход (умножается на скорость движения)
ENERGY_PER_TURN = 1
# сколько энергии тратится на атаку
ATTACK_ENERGY = 1

# чтобы было невыгодно есть своих детей BIRTH_KPD должно быть больше HUNT_KPD
#HUNT_KPD = 0.9
#BIRTH_KPD = 1
HUNT_KPD = 0.3
BIRTH_KPD = 0.3

# food
FOOD_GROWTH = 0.01 #0.00001 # how quickly does food grow on a square?
FOOD_INTAKE = 200. # how much does every agent consume?
FOOD_MAX = 500. # how much food per cell can there be at max?
FOOD_INIT_PROB = 0.3 # at what prob new food appears at game start
FOOD_APPEAR_PROB = 0.001 # at what prob new food appears
FOOD_APPEAR_AMM = float(FOOD_MAX) / 2

# calculated
MAX_SPEED = GRID_SCALE * math.sqrt(2) / 2
CARN_SPEED_HANDICAP = MAX_SPEED / 2
ATTACK_RADIUS = GRID_SCALE * math.sqrt(2) * 2
ATTACK_RADIUS_SQ = ATTACK_RADIUS * ATTACK_RADIUS

class Settings(object):
    """
    dynamic settings
    """
    # TODO
    SELECT_RATIO = 0.95
    RANDOM_RATIO = 0.01
    ROT_SPEED = 0.5
    MAX_HEALTH = 50000
    MIN_HEALTH = 500
    # сколько энергии тратится на ход (умножается на скорость движения)
    ENERGY_PER_TURN = 1
    # сколько энергии тратится на атаку
    ATTACK_ENERGY = 1
    # чтобы было невыгодно есть своих детей BIRTH_KPD должно быть больше HUNT_KPD
    HUNT_KPD = 0.99
    BIRTH_KPD = 0.5

    # food
    FOOD_GROWTH = 0.01 #0.00001 # how quickly does food grow on a square?
    FOOD_INTAKE = 200. # how much does every agent consume?
    FOOD_MAX = 500. # how much food per cell can there be at max?
    FOOD_INIT_PROB = 0.3 # at what prob new food appears at game start
    FOOD_APPEAR_PROB = 0.0001 # at what prob new food appears
    FOOD_APPEAR_AMM = float(FOOD_MAX) / 2

    # calculated
    MAX_SPEED = GRID_SCALE * math.sqrt(2) / 2
    CARN_SPEED_HANDICAP = MAX_SPEED / 2
    ATTACK_RADIUS = GRID_SCALE * math.sqrt(2) * 2

    @property
    def ATTACK_RADIUS_SQ(self):
        return self.ATTACK_RADIUS * self.ATTACK_RADIUS


