#!/usr/bin/env python

from datetime import datetime

import numpy as np
import scipy

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.utils import iers
from photutils import IRAFStarFinder


print("This had better force astropy to download leapseconds file...")
