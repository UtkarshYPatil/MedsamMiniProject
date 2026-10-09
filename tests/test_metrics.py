import numpy as np
from src.evaluate.metrics import dice, iou


def test_dice_identical():
    a = np.zeros((4, 4), bool); a[1:3, 1:3] = True
    assert dice(a, a) == 1.0


def test_dice_disjoint():
    a = np.zeros((4, 4), bool); a[0, 0] = True
    b = np.zeros((4, 4), bool); b[3, 3] = True
    assert dice(a, b) == 0.0


def test_iou_half():
    a = np.zeros((1, 4), bool); a[0, :2] = True
    b = np.zeros((1, 4), bool); b[0, 1:3] = True
    assert abs(iou(a, b) - 1/3) < 1e-9
