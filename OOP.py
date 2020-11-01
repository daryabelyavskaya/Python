"""Предисловие

Океан представляется двумерным массивом ячеек.

Каждая явейка может быть:

    пустой
    со скалой
    с рыбой
    с креветкой

Ячейки являются соседними, если имеют хотя бы по одной общей точке с данной ячейкой. Все ячейки за границами игрового поля считаются пустыми.

Правила обновления ячеек:

    ячейки со скалами не меняются во времени
    если какой-то рыбе слишком тесно (от 4 и более соседей-рыб), то рыба погибает
    если рыбе слишком одиноко (0 или 1 соседей-рыб), то рыба погибает
    если у рыбы 2 или 3 соседа-рыбы, то она просто продолжает жить
    соседи-скалы и соседи-креветки никак не влияют на жизнь рыб
    креветки существуют по аналогичным правилам (учитывая только соседей креветок)
    если какая-то ячейка океана была пуста и имела ровно 3-х соседей рыб, то в следующий момент времени в ней рождается рыба. Иначе если у ячейки было ровно три соседа-креветки, в ней рождается креветка
    изменение всех ячеек океана происходит одновременно, учитывая только состояния ячеек в предыдущий момент времени

В каждый квант времени ячейки последовательно обрабатываются.
Условие

Вам нужно в файле life_game.py реализовать класс LifeGame.

    Инициализируется начальным состоянием океана - прямоугольным списком списков (формируя тем самым матрицу), каждый элемент которого это число. 0 - если ячейка пустая, 1 - со скалой, 2 - с рыбой, 3 - с креветкой
    Содержит метод get_next_generation, который обновляет состояние океана и возвращает его содержимое
    get_next_generation должен быть единственный публичным методом в классе
    Вам нужно подумать, как поделить все на небольшие логические методы, которые, в отличие от get_next_generation пометить "приватными", то есть через underscore.

Например, вы хотите создать метод, который извлекает соседей для клетки"""

from copy import deepcopy
import numpy as np


class LifeGame(object):
    EMPTY = 0
    ROCK = 1
    FISH = 2
    SHRIMP = 3

    def __init__(self, field):
        self.field = field

    def get_next_generation(self):
        field2 = deepcopy(self.field)
        for i in range(len(self.field)):
            for j in range(len(self.field[i])):
                self._process_cell(field2, i, j)
        self.field = field2
        return np.array(self.field).reshape(len(self.field), len(self.field[0]))

    def _process_cell(self, field2, i, j):
        count = self._get_neighbours(i, j)
        if self.field[i][j] == 0:
            field2[i][j] = self._process_empty(count)
        if self.field[i][j] == 2:
            field2[i][j] = self._process_fish(count)
        if self.field[i][j] == 3:
            field2[i][j] = self._process_shrimp(count)

    @staticmethod
    def _process_empty(neighbours):
        if neighbours[0] == 3:
            return 2
        elif neighbours[1] == 3:
            return 3
        return 0

    @staticmethod
    def _process_fish(neighbours):
        if neighbours[0] >= 4 or neighbours[0] <= 1:
            return 0
        else:
            return 2

    @staticmethod
    def _process_shrimp(neighbours):
        if neighbours[1] >= 4 or neighbours[1] <= 1:
            return 0
        else:
            return 3

    def isCorrect(self, i, j):
        if i > -1 and j > -1 and i < len(self.field) and j < len(self.field[0]):
            return True
        else:
            return False

    def _get_neighbours(self, i, j):
        fish = 0
        shrimp = 0
        for k in range(-1, 2):
            for l in range(-1, 2):
                if not k == l == 0 and self.isCorrect(i + k, l + j):
                    if self.field[k + i][l + j] == 2:
                        fish += 1
                    if self.field[k + i][l + j] == 3:
                        shrimp += 1
        return [fish, shrimp]


life_game = LifeGame([[1, 2, 3, 0],
                      [2, 0, 0, 2],
                      [1, 2, 0, 3],
                      [3, 3, 2, 0]])
life_game1 = LifeGame([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
])

print(life_game.get_next_generation())
print(life_game1.get_next_generation())
board = LifeGame([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
])
for i in range(8):
    board.get_next_generation()
print(board.get_next_generation())
