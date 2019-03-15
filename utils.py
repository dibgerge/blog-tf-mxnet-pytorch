import numpy as np
from os.path import basename, splitext, isfile
import pathlib
import xml.etree.ElementTree as ET
from glob import glob
import pickle
import json
import tqdm
from PIL import Image

import torch.utils.data
from torchvision import transforms

from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.utils import Sequence

import mxnet as mx
import gluoncv


class Data:
    """
    This is a generic data class which other framework specific data loaders will extend. This
    class knows where the data is, and how to convert the validation data labels.
    """
    DATA_PATH = pathlib.Path('/home/gerges/Documents/Datasets/ILSVRC')
    VAL_IMAGES_PATH = DATA_PATH/'Data/CLS-LOC/val'

    # folder of xml files containing annotations for the validation images
    VAL_ANNOT_PATH = DATA_PATH/'Annotations/CLS-LOC/val'

    # File which maps synset word net id (wnid) to neural network index. This is obtained from
    # https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
    CLASS_INDEX = DATA_PATH/'imagenet_class_index.json'

    # File to store mapping from validation image file name to label index
    VAL_LABELS_FILE = DATA_PATH/'image_maps.json'

    IMG_WIDTH = 224
    IMG_HEIGHT = 224

    def __init__(self):
        self.labels_map = self.make_labels()
        self.val_image_files = glob(str(self.VAL_IMAGES_PATH/'*.JPEG'))

    def make_labels(self, rewrite=False):
        """
        Obtains the label index for each of the validation images in the Imagenet dataset.
        Writes the results to a file for faster access later.

        Parameters
        ----------
        rewrite : bool
            If True, write the labels to a file, even if the file already exists.
        """
        if isfile(self.VAL_LABELS_FILE) and not rewrite:
            with open(self.VAL_LABELS_FILE, 'rb') as fid:
                return pickle.load(fid)

        with open(self.CLASS_INDEX) as fid:
            class_index = json.load(fid)
        indices, synsets = zip(*class_index.items())
        indices = list(map(int, indices))
        synsets = list(zip(*synsets))[0]
        synset_to_index = dict(zip(synsets, indices))

        labels = {}
        val_annot_files = glob(str(self.VAL_ANNOT_PATH/'*.xml'))

        for file_name in tqdm.tqdm(val_annot_files):
            file_name_base = splitext(basename(file_name))[0]
            tree = ET.parse(file_name)
            root = tree.getroot()
            synset_label = root.find('object').find('name').text
            labels[file_name_base] = synset_to_index[synset_label]

        with open(self.VAL_LABELS_FILE, 'wb') as fid:
            pickle.dump(labels, fid)

        return labels

    def get(self, file_name):
        file_name_base = splitext(basename(file_name))[0]
        image = Image.open(file_name).convert('RGB')
        label = self.labels_map[file_name_base]
        return image, label


class TorchDataset(torch.utils.data.Dataset, Data):
    """
    Data loader to be used with PyTorch models.
    """
    def __init__(self, batch_size=10, num_workers=4):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.val_image_files)

    def __getitem__(self, item):
        image_name = self.val_image_files[item]
        image, label = self.get(image_name)
        image = self.transforms(image)
        return image, label

    def __iter__(self):
        data_loader = torch.utils.data.DataLoader(self,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.num_workers)
        for images, labels in data_loader:
            yield images, labels


class TensorflowDataset(Sequence):
    """
    Data loader to be used with Tensorflow/Keras models.
    """
    def __init__(self, batch_size=10, preprocess=None):
        self.data = Data()
        super().__init__()
        self.batch_size = batch_size
        self.preprocess = preprocess

    def __getitem__(self, item):
        image_names = self.data.val_image_files[item*self.batch_size:(item+1)*self.batch_size]
        images, labels = [], []
        for file_name in image_names:
            image, label = self.data.get(file_name)
            image = self.transform(image)
            images.append(image)
            labels.append(label)
        return np.stack(images), np.array(labels)

    def __len__(self):
        return int(np.ceil(len(self.data.val_image_files) / self.batch_size))

    # def stream(self):
    #     for i in range(self.num_iterations):
    #         batch_image_files = self.val_image_files[i * self.batch_size:(i + 1) * self.batch_size]
    #         images, labels = [], []
    #         for file_name in batch_image_files:
    #             image, label = self.get(file_name)
    #             image = self.transform(image)
    #             images.append(image)
    #             labels.append(label)
    #         yield np.stack(images), np.array(labels)

    # def __iter__(self):
    #     return tf.data.Dataset().batch(self.batch_size).from_generator(
    #         self.stream(),
    #         output_types=(tf.float32, tf.int64),
    #         output_shapes=(tf.TensorShape([None, 224, 224, 3]), tf.TensorShape([None]))
    #     )

    def transform(self, image):
        """
        Applies the transformations required for feeding into the CNN.
        """
        if image.height < image.width:
            new_height = 256
            new_width = int(256 * image.width/image.height)
        else:
            new_width = 256
            new_height = int(256 * image.height/image.width)

        height_crop = new_height - self.data.IMG_HEIGHT
        width_crop = new_width - self.data.IMG_WIDTH

        top_crop = height_crop // 2
        bot_crop = new_height - int(np.ceil(height_crop / 2))
        right_crop = new_width - int(np.ceil(width_crop / 2))
        left_crop = width_crop // 2

        image = image.resize((new_width, new_height))
        image = image.crop((left_crop, top_crop, right_crop, bot_crop))
        image = img_to_array(image)

        if self.preprocess:
            image = self.preprocess(image)
        return image

    # @property
    # def num_iterations(self):
    #     return int(np.ceil(len(self.data.val_image_files) / self.batch_size))


class MxnetDataset(Data):
    """
    Data loader to be used with MxNet models.
    """
    def __init__(self, batch_size=10, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = gluoncv.data.transforms.presets.imagenet.transform_eval

    def __len__(self):
        return len(self.val_image_files)

    def __getitem__(self, item):
        image_name = self.val_image_files[item]
        image, label = self.get(image_name)
        image = self.transforms(mx.nd.array(image))
        return image.squeeze(), label

    def __iter__(self):
        data_loader = mx.gluon.data.DataLoader(self,
                                               batch_size=self.batch_size,
                                               shuffle=False,
                                               num_workers=self.num_workers,
                                               prefetch=self.num_workers*4)
        for images, labels in data_loader:
            yield images, labels

