# -*- coding: utf-8 -*-
#
# Helper functions for CellViT
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
import sys
from pathlib import Path
from typing import List, Union

import cupy as cp
import numpy as np
import pandas as pd

from cupyx.scipy import ndimage as ndimage_cp
from scipy import ndimage


def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def remove_small_objects_cp(pred: cp.ndarray, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided.

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel.

    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage_cp.generate_binary_structure(pred.ndim, connectivity)
        ccs = cp.zeros_like(pred, dtype=cp.int32)
        ndimage_cp.label(
            pred, selem, output=ccs
        )  # https://docs.rapids.ai/api/cucim/stable/api/#cucim.skimage.measure.label
    else:
        ccs = out

    try:
        component_sizes = cp.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided.

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel.

    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(
            pred.ndim, connectivity
        )  # numpy function
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)  # generate_binary_structure
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dictionary and insert the sep to seperate keys

    Args:
        d (dict): dict to flatten
        parent_key (str, optional): parent key name. Defaults to ''.
        sep (str, optional): Seperator. Defaults to '.'.

    Returns:
        dict: Flattened dict
    """
    items = []
    for k, v in d.items():
        if type(k) != str:
            k = str(k)
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = ".") -> dict:
    """Unflatten a flattened dictionary (created a nested dictionary)

    Args:
        d (dict): Dict to be nested
        sep (str, optional): Seperator of flattened keys. Defaults to '.'.

    Returns:
        dict: Nested dict
    """
    output_dict = {}
    for key, value in d.items():
        keys = key.split(sep)
        d = output_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    return output_dict


def get_size_of_dict(d: dict) -> int:
    size = sys.getsizeof(d)
    for key, value in d.items():
        size += sys.getsizeof(key)
        size += sys.getsizeof(value)
    return size


def load_wsi_files_from_csv(csv_path: Union[Path, str], wsi_extension: str) -> List:
    """Load filenames from csv file with column name "Filename"

    Args:
        csv_path (Union[Path, str]): Path to csv file
        wsi_extension (str): WSI file ending (suffix)

    Returns:
        List: List of WSI
    """
    wsi_filelist = pd.read_csv(csv_path)
    wsi_filelist = wsi_filelist["Filename"].to_list()
    wsi_filelist = [f for f in wsi_filelist if Path(f).suffix == f".{wsi_extension}"]

    return wsi_filelist


def close_logger(logger: logging.Logger) -> None:
    """Closing a logger savely

    Args:
        logger (logging.Logger): Logger to close
    """
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()

    logger.handlers.clear()
    logging.shutdown()


def remove_parameter_tag(d: dict, sep: str = ".") -> dict:
    """Remove all paramter tags from dictionary

    Args:
        d (dict): Dict must be flattened with defined seperator
        sep (str, optional): Seperator used during flattening. Defaults to ".".

    Returns:
        dict: Dict with parameter tag removed
    """
    param_dict = {}
    for k, _ in d.items():
        unflattened_keys = k.split(sep)
        new_keys = []
        max_num_insert = len(unflattened_keys) - 1
        for i, k in enumerate(unflattened_keys):
            if i < max_num_insert and k != "parameters":
                new_keys.append(k)
        joined_key = sep.join(new_keys)
        param_dict[joined_key] = {}
    print(param_dict)
    for k, v in d.items():
        unflattened_keys = k.split(sep)
        new_keys = []
        max_num_insert = len(unflattened_keys) - 1
        for i, k in enumerate(unflattened_keys):
            if i < max_num_insert and k != "parameters":
                new_keys.append(k)
        joined_key = sep.join(new_keys)
        param_dict[joined_key][unflattened_keys[-1]] = v

    return param_dict
