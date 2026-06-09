# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from skycam_utils.alcor import build_alcor_badpix_mask


def test_build_badpix_mask_channel_multiplicity():
    # flat zero background; isolated spikes spaced > kernel apart
    cube = np.zeros((3, 15, 15), dtype=float)
    cube[0, 2, 2] = 1000.0                      # 1-channel spike -> flagged in R only
    cube[0, 7, 7] = 1000.0; cube[1, 7, 7] = 1000.0   # 2-channel -> flagged R and G
    cube[:, 12, 12] = 1000.0                    # 3-channel -> real source, excluded

    mask = build_alcor_badpix_mask(cube, ksize=5, z_thresh=25)

    assert mask.shape == (3, 15, 15)
    assert mask.dtype == bool
    # 1-channel
    assert mask[0, 2, 2] and not mask[1, 2, 2] and not mask[2, 2, 2]
    # 2-channel
    assert mask[0, 7, 7] and mask[1, 7, 7] and not mask[2, 7, 7]
    # 3-channel excluded everywhere
    assert not mask[:, 12, 12].any()
    # nothing else flagged
    assert mask.sum() == 3
