"""Detection Dataset from LST file."""
from __future__ import absolute_import

__all__ = ['RecordFileDetection', 'LstDetection', 'LstDcmDetection']

import os

import mxnet as mx
import numpy as np
import pydicom
import re
from gluoncv.data.recordio.detection import _transform_label
from mxnet import nd, gluon
from mxnet.gluon.data import Dataset


class RecordFileDetection(gluon.data.vision.ImageRecordDataset):
    def __init__(self, filename, coord_normalized=True, flag=1):
        super(RecordFileDetection, self).__init__(filename, flag)
        self._coord_normalized = coord_normalized

    def __getitem__(self, idx):
        img, label = super(RecordFileDetection, self).__getitem__(idx)
        h, w, _ = img.shape
        if self._coord_normalized:
            label = _transform_label(label, h, w)
        else:
            label = _transform_label(label)
        return img, label


class LstDetection(Dataset):
    """Detection dataset loaded from LST file and raw images.
    LST file is a pure text file but with special label format.

    Checkout :ref:`lst_record_dataset` for tutorial of how to prepare this file.

    Parameters
    ----------
    filename : type
        Description of parameter `filename`.
    root : str
        Relative image root folder for filenames in LST file.
    flag : int, default is 1
        Use 1 for color images, and 0 for gray images.
    coord_normalized : boolean
        Indicate whether bounding box coordinates haved been normalized to (0, 1) in labels.
        If so, we will rescale back to absolute coordinates by multiplying width or height.

    """

    def __init__(self, filename, root='', flag=1, is_dicom=False, coord_normalized=True):
        self._is_dicom = is_dicom
        self._flag = flag
        self._coord_normalized = coord_normalized
        self._items = []
        self._labels = []
        full_path = os.path.expanduser(filename)
        with open(full_path) as fin:
            for line in iter(fin.readline, ''):
                line = line.strip().split('\t')
                label = np.array(line[1:-1]).astype('float')
                im_path = os.path.join(root, line[-1])
                self._items.append(im_path)
                self._labels.append(label)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        im_path = self._items[idx]
        if not self._is_dicom:
            img = mx.image.imread(im_path, self._flag)
        else:
            img = nd.expand_dims(nd.array(pydicom.dcmread(im_path).pixel_array), axis=-1)
            # img = nd.array(pydicom.dcmread(im_path).pixel_array)
        h, w, _ = img.shape
        label = self._labels[idx]
        if self._coord_normalized:
            label = _transform_label(label, h, w)
        else:
            label = _transform_label(label)
        return img, label


def _get_sample_fname(folder, instance_num, delta):
    files = os.listdir(folder)
    sample_fname = range(instance_num - delta, instance_num + delta + 1)
    for f in files:
        sp = int(filter(str.isdigit, re.split('\.|_|-', f))[-1])
        for idx in range(len(sample_fname)):
            if sp == sample_fname[idx]:
                sample_fname[idx] = os.path.join(folder, f)
                break
    return sample_fname


class LstDcmDetection(Dataset):
    """Detection dataset loaded from LST file and raw images.
    LST file is a pure text file but with special label format.

    Checkout :ref:`lst_record_dataset` for tutorial of how to prepare this file.

    Parameters
    ----------
    filename : type
        Description of parameter `filename`.
    root : str
        Relative image root folder for filenames in LST file.
    flag : int, default is 1
        Use 1 for color images, and 0 for gray images.
    coord_normalized : boolean
        Indicate whether bounding box coordinates haved been normalized to (0, 1) in labels.
        If so, we will rescale back to absolute coordinates by multiplying width or height.

    """

    def __init__(self, filename, root='', multi_slices=False, slice_width=0, coord_normalized=True):
        self._multi_slices = multi_slices
        self._delta = slice_width // 2
        self._coord_normalized = coord_normalized
        self._items = []
        self._labels = []
        self._root = str(root)
        full_path = os.path.expanduser(filename)
        with open(full_path) as fin:
            for line in iter(fin.readline, ''):
                line = line.strip().split('\t')
                label = np.array(line[1:-1]).astype('float')
                im_path = line[-1]
                self._items.append(im_path)
                self._labels.append(label)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        if self._multi_slices:
            split_p = self._items[idx].rfind('_')
            bname = self._items[idx][:split_p]
            instance_num = int(self._items[idx][split_p + 1:])
            im_path = os.path.join(self._root, bname)
            dcm_slices = list()
            sample_fnames = _get_sample_fname(im_path, instance_num, self._delta)
            for f in sample_fnames:
                one_slice = pydicom.dcmread(f).pixel_array
                dcm_slices.append(np.expand_dims(one_slice, axis=0))
            img = np.concatenate(dcm_slices, axis=0)
            img = nd.array(np.transpose(img, (1, 2, 0)))
        else:
            im_path = os.path.join(self._root, self._items[idx])
            img = nd.expand_dims(nd.array(pydicom.dcmread(im_path).pixel_array), axis=-1)
        h, w, _ = img.shape
        label = self._labels[idx]
        if self._coord_normalized:
            label = _transform_label(label, h, w)
        else:
            label = _transform_label(label)
        return img, label
