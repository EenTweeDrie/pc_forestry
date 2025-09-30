import os
from classes.PCD import PCD
from classes.PCD_AREA import PCD_AREA
from classes.PCD_UTILS import PCD_UTILS
from classes.CELL import CELL
from classes.VOR_TES import VOR_TES
from settings.coord_settings import CS
import numpy as np
import pandas as pd
import circle_fit as cf
import statistics
import math
from tqdm import tqdm

from .pipeline import CoordinatesPipeline


def coordinates(intensity_cut_make, cs):
    pipeline = CoordinatesPipeline().set_params({
        "intensity_cut_make": intensity_cut_make,
        "cs": cs,
    })
    return pipeline.run()


if __name__ == "__main__":
    cs = CS()
    yml_path = "settings\settings.yaml"
    cs.set(yml_path)
    intensity_cut_make = 7000
    coordinates(intensity_cut_make=intensity_cut_make, cs=cs)
