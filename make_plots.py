import os, glob, time, sys, tomopy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from scipy import signal

HOME_PATH = '/home/etsai/BNL/Research/GIWAXS_tomo_2020C3/RLi/waxs/'
GI_TOMO_PATH = HOME_PATH+'GI_tomo/'
GI_TOMO_PATH in sys.path or sys.path.append(GI_TOMO_PATH)
import analysis.peaks as peaks
import analysis.tomo as tomo
import analysis.seg as seg
import analysis.util as util
import analysis.io as io

INDEX_PATH = '/home/etsai/BNL/Users/software/Indexing/'
INDEX_PATH in sys.path or sys.path.append(INDEX_PATH)
import fun_index as index

# C8BTBT
lattice = (5.83, 7.88, 29.18, 90/180*np.pi, 99.4/180*np.pi, 90/180*np.pi)
x1 = index.angle_interplane(1,0,0,  2,1,0,lattice)
x2 = index.angle_interplane(1,0,0,  1,1,0,lattice)
x3 = index.angle_interplane(1,0,0,  1,2,0,lattice)
x4 = index.angle_interplane(-1,0,0,  0,1,5,lattice)
x5 = index.angle_interplane(-1,0,0,  1,1,5,lattice)
x6 = index.angle_interplane(-1,0,0,  -1,1,5,lattice)
angles_deg = [0, x1, x2, x3, x6, 90, x4, 180-x3, x5, 180-x2, 180-x1, 180]

tomo.plot_angles(angles_deg, fignum=100, color='c', labels= ['-100', '-210', '-110', '-120','-115', '010', '015', '120', '115', '110', '210', '100'])
