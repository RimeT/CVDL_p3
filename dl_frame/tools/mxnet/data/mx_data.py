import os

from gluoncv.data import RecordFileDetection
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import ImageRecordDataset, ImageFolderDataset, transforms as gdt

from . import LstDetection

DB_EXTENSIONS = {
    'rec': ['.REC'],
    'lst': ['.LST'],
    'txt': ['.TXT'],
    'file': ['.JPG', '.JPEG', '.PNG'],
    'json': ['.JSON'],
}
IMAGE_SUFFIX = ('.JPG', '.JPEG', '.PNG', '.TIF', '.DCM')

IMAGE_TYPE_FOLDER = 'imgfolder'
IMAGE_TYPE_REC = 'rec'
IMAGE_TYPE_LST = 'lst'
IMAGE_TYPE_JSON = 'json'


def get_backend_of_source(db_path):
    # If a directory is given, we include all its contents. Otherwise it's just the one file.
    if os.path.isdir(db_path):
        files_in_path = [fn for fn in os.listdir(db_path) if not fn.startswith('.')]
    else:
        files_in_path = [db_path]

    # Keep the below priority ordering
    for db_fmt in DB_EXTENSIONS:
        ext_list = DB_EXTENSIONS[db_fmt]
        for ext in ext_list:
            if any(ext in os.path.splitext(fn)[1].upper() for fn in files_in_path):
                return db_fmt

    # if we got a image folder
    imgfolder_num = 0
    for roots, dirs, files in os.walk(db_path):
        image_num = 0
        for f in files:
            if f.upper().endswith(IMAGE_SUFFIX):
                dir = roots.split('/')[-1]
                if roots == db_path + '/' + dir or roots == db_path + dir:
                    if image_num > 6:
                        imgfolder_num += 1
                        break
                    image_num += 1

    # imgfolder_num indicates num of folders contains images
    if imgfolder_num > 1:
        return IMAGE_TYPE_FOLDER

    raise ValueError("Unknown dataset backend.")


class LoaderFactory(object):
    def __init__(self, db_path, labels, data_format, **kwargs) -> None:
        self._dataset = None
        self.db_path = db_path
        self.classes = labels
        self.data_format = data_format
        self.channel_flag = 1
        self.channels = 1
        self.is_dicom = False
        if data_format == 'dicom':
            self.channel_flag = 0
            self.is_dicom = True
        self.data_volume = 0
        self.batch_size = None
        self.batch_loader = None
        self.niters = 0

    @staticmethod
    def set_source(ml_type, db_path, labels, data_format, **kwargs):
        backend = get_backend_of_source(db_path)
        if ml_type == 'classification':
            if backend == IMAGE_TYPE_REC:
                return ClassRecLoader(db_path, labels=labels, data_format=data_format, **kwargs)
            if backend == IMAGE_TYPE_FOLDER:
                return ClassFolderLoader(db_path, labels=labels, data_format=data_format, **kwargs)
        if ml_type == 'object-detection':
            if backend == IMAGE_TYPE_REC:
                return DetectRecLoader(db_path, labels=labels, data_format=data_format, **kwargs)
            elif backend == IMAGE_TYPE_LST:
                return DetectLstLoader(db_path, labels=labels, data_format=data_format, **kwargs)
        raise ValueError("Machine Learning type %s with %s not supported" % (ml_type, backend))

    def setup(self, batch_size, shuffle, fn, **kwargs):
        raise NotImplementedError

    def get_batch_loader(self):
        return self.batch_loader


class ClassLoader(LoaderFactory):

    def __init__(self, db_path, **kwargs) -> None:
        super(ClassLoader, self).__init__(db_path, **kwargs)

    def setup(self, batch_size, shuffle, fn=None, **kwargs):
        self.batch_size = batch_size
        # TODO: Check whether dataset[0][0] in range(0, 255).
        #  If not, do window transformation and rescale
        window_center = 0 if 'window_center' not in kwargs else kwargs['window_center']
        window_width = 0 if 'window_width' not in kwargs else kwargs['window_width']

        transformer = gdt.Compose([
            fn,  # transformation and augmentation
            # dicom window transformation
            gdt.ToTensor(),
        ])
        self.batch_loader = DataLoader(dataset=self._dataset.transform_first(fn=transformer),
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=8,
                                       last_batch='rollover',
                                       pin_memory=True)
        self.niters = len(self.batch_loader)


class ClassFolderLoader(ClassLoader):

    def __init__(self, db_path, **kwargs) -> None:
        super(ClassFolderLoader, self).__init__(db_path, **kwargs)
        self._dataset = ImageFolderDataset(self.db_path,
                                           flag=self.channel_flag,
                                           transform=None)
        self.data_volume = len(self._dataset)
        self.channels = self._dataset[0][0].shape[-1]


class ClassRecLoader(ClassLoader):

    def __init__(self, db_path, **kwargs) -> None:
        super(ClassRecLoader, self).__init__(db_path, **kwargs)
        self._dataset = ImageRecordDataset(db_path, self.channel_flag)
        self.data_volume = len(self._dataset)
        self.channels = self._dataset[0][0].shape[-1]


class DetectionLoader(LoaderFactory):

    def __init__(self, db_path, **kwargs) -> None:
        super(DetectionLoader, self).__init__(db_path, **kwargs)

    def setup(self, batch_size, shuffle, fn, **kwargs):
        self.batch_size = batch_size
        window_center = 0 if 'window_center' not in kwargs else kwargs['window_center']
        window_width = 0 if 'window_width' not in kwargs else kwargs['window_width']
        batchify_fn = kwargs['batchify_fn']
        self.batch_loader = DataLoader(self._dataset.transform(fn),
                                       batch_size, shuffle,
                                       batchify_fn=batchify_fn,
                                       last_batch='rollover',
                                       # num_workers=4,
                                       )
        self.niters = len(self.batch_loader)


class DetectLstLoader(DetectionLoader):

    def __init__(self, db_path, root, **kwargs) -> None:
        super(DetectLstLoader, self).__init__(db_path, **kwargs)
        self._dataset = LstDetection(db_path, root=root, flag=self.channel_flag, is_dicom=self.is_dicom)
        self.data_volume = len(self._dataset)
        self.channels = self._dataset[0][0].shape[-1]


class DetectRecLoader(DetectionLoader):

    def __init__(self, db_path, **kwargs) -> None:
        super(DetectRecLoader, self).__init__(db_path, **kwargs)
        self._dataset = RecordFileDetection(db_path, coord_normalized=True)
        self.data_volume = len(self._dataset)
        self.channels = self._dataset[0][0].shape[-1]
