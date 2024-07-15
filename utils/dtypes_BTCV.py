#-*- coding:utf-8 -*-
from enum import IntEnum, Enum

class LabelEnum(IntEnum):
    BACKGROUND = 0  # background
    SPLEEN = 1  # spleen
    RKID = 2  # right kidney
    LKID = 3  # left kidney
    GALL = 4  # gallbladder
    ESO = 5  # esophagus
    LIVER = 6  # liver
    STO = 7  # stomach
    AORTA = 8  # aorta
    IVC = 9  # inferior vena cava
    VEINS = 10  # veins
    PANCREAS = 11  # pancreas
    RAD = 12  # right adrenal gland
    LAD = 13  # left adrenal gland

class FilterMethods(Enum):
    CUBIC = "CUBIC"
    LANCZOS2 = "LANCZOS2"
    LANCZOS3 = "LANCZOS3"
    BOX = "BOX"
    LINEAR = "LINEAR"
