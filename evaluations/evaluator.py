import io
import os
import random
import warnings
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterable, Optional, Tuple

import numpy as np
import requests
import tensorflow.compat.v1 as tf
from scipy import linalg
from tqdm.auto import tqdm

INCEPTION_V3_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
INCEPTION_V3_PATH = "classify_image_graph_def.pb"

FID_POOL_NAME = "pool_3:0"
FID_SPATIAL_NAME = "mixed_6/conv:0"


class InvalidFIDException(Exception):
    pass


class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        """
        Compute the Frechet distance between two sets of statistics.
        """
        # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L132
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), f"Training and test mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
        assert (
            sigma1.shape == sigma2.shape
        ), f"Training and test covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; adding %s to diagonal of cov estimates"
                % eps
            )
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class Evaluator:
    def __init__(
        self,
        session,
        batch_size=32,
        softmax_batch_size=512,
    ):
        self.sess = session
        self.batch_size = batch_size
        self.softmax_batch_size = softmax_batch_size
        with self.sess.graph.as_default():
            self.image_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.softmax_input = tf.placeholder(tf.float32, shape=[None, 2048])
            self.pool_features, self.spatial_features = _create_feature_graph(self.image_input)
            self.softmax = _create_softmax_graph(self.softmax_input)

    def read_activations(self, npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
        with open_npz_array(npz_path, "arr_0") as reader:
            return self.compute_activations(reader.read_batches(self.batch_size))

    def compute_activations(self, batches: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute image features for downstream evals.

        :param batches: a iterator over NHWC numpy arrays in [0, 255].
        :return: a tuple of numpy arrays of shape [N x X], where X is a feature
                 dimension. The tuple is (pool_3, spatial).
        """
        preds = []
        spatial_preds = []
        for batch in tqdm(batches):
            batch = batch.astype(np.float32)
            pred, spatial_pred = self.sess.run(
                [self.pool_features, self.spatial_features], {self.image_input: batch}
            )
            preds.append(pred.reshape([pred.shape[0], -1]))
            spatial_preds.append(spatial_pred.reshape([spatial_pred.shape[0], -1]))
        return (
            np.concatenate(preds, axis=0),
            np.concatenate(spatial_preds, axis=0),
        )

    def read_statistics(
        self, npz_path: str, activations: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[FIDStatistics, FIDStatistics]:
        obj = np.load(npz_path)
        if "mu" in list(obj.keys()):
            return FIDStatistics(obj["mu"], obj["sigma"]), FIDStatistics(
                obj["mu_s"], obj["sigma_s"]
            )
        return tuple(self.compute_statistics(x) for x in activations)

    def compute_statistics(self, activations: np.ndarray) -> FIDStatistics:
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return FIDStatistics(mu, sigma)

    def compute_inception_score(self, activations: np.ndarray, split_size: int = 5000) -> float:
        softmax_out = []
        for i in range(0, len(activations), self.softmax_batch_size):
            acts = activations[i : i + self.softmax_batch_size]
            softmax_out.append(self.sess.run(self.softmax, feed_dict={self.softmax_input: acts}))
        preds = np.concatenate(softmax_out, axis=0)
        # https://github.com/openai/improved-gan/blob/4f5d1ec5c16a7eceb206f42bfc652693601e1d5c/inception_score/model.py#L46
        scores = []
        for i in range(0, len(preds), split_size):
            part = preds[i : i + split_size]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return float(np.mean(scores))


class NpzArrayReader(ABC):
    @abstractmethod
    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        pass

    def read_batches(self, batch_size: int) -> Iterable[np.ndarray]:
        while True:
            batch = self.read_batch(batch_size)
            if batch is None:
                break
            yield batch


class StreamingNpzArrayReader(NpzArrayReader):
    def __init__(self, arr_f, shape, dtype):
        self.arr_f = arr_f
        self.shape = shape
        self.dtype = dtype
        self.idx = 0

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.shape[0]:
            return None

        bs = min(batch_size, self.shape[0] - self.idx)
        self.idx += bs

        if self.dtype.itemsize == 0:
            return np.ndarray([bs, *self.shape[1:]], dtype=self.dtype)

        read_count = bs * np.prod(self.shape[1:])
        read_size = int(read_count * self.dtype.itemsize)
        data = _read_bytes(self.arr_f, read_size, "array data")
        return np.frombuffer(data, dtype=self.dtype).reshape([bs, *self.shape[1:]])


class MemoryNpzArrayReader(NpzArrayReader):
    def __init__(self, arr):
        self.arr = arr
        self.idx = 0

    @classmethod
    def load(cls, path: str, arr_name: str):
        with open(path, "rb") as f:
            arr = np.load(f)[arr_name]
        return cls(arr)

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.arr.shape[0]:
            return None

        res = self.arr[self.idx : self.idx + batch_size]
        self.idx += batch_size
        return res


@contextmanager
def open_npz_array(path: str, arr_name: str) -> NpzArrayReader:
    with _open_npy_file(path, arr_name) as arr_f:
        version = np.lib.format.read_magic(arr_f)
        if version == (1, 0):
            header = np.lib.format.read_array_header_1_0(arr_f)
        elif version == (2, 0):
            header = np.lib.format.read_array_header_2_0(arr_f)
        else:
            yield MemoryNpzArrayReader.load(path, arr_name)
            return
        shape, fortran, dtype = header
        if fortran or dtype.hasobject:
            yield MemoryNpzArrayReader.load(path, arr_name)
        else:
            yield StreamingNpzArrayReader(arr_f, shape, dtype)


def _read_bytes(fp, size, error_template="ran out of data"):
    """
    Copied from: https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/lib/format.py#L788-L886

    Read from file-like object until size bytes are read.
    Raises ValueError if not EOF is encountered before size bytes are read.
    Non-blocking objects only supported if they derive from io objects.
    Required as e.g. ZipExtFile in python 2.6 can return less data than
    requested.
    """
    data = bytes()
    while True:
        # io files (default in python3) return None or raise on
        # would-block, python2 file will truncate, probably nothing can be
        # done about that.  note that regular files can't be non-blocking
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        msg = "EOF: reading %s, expected %d bytes got %d"
        raise ValueError(msg % (error_template, size, len(data)))
    else:
        return data


@contextmanager
def _open_npy_file(path: str, arr_name: str):
    with open(path, "rb") as f:
        with zipfile.ZipFile(f, "r") as zip_f:
            if f"{arr_name}.npy" not in zip_f.namelist():
                raise ValueError(f"missing {arr_name} in npz file")
            with zip_f.open(f"{arr_name}.npy", "r") as arr_f:
                yield arr_f


def _download_inception_model():
    if os.path.exists(INCEPTION_V3_PATH):
        return
    print("downloading InceptionV3 model...")
    with requests.get(INCEPTION_V3_URL, stream=True) as r:
        r.raise_for_status()
        tmp_path = INCEPTION_V3_PATH + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
        os.rename(tmp_path, INCEPTION_V3_PATH)


def _create_feature_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    pool3, spatial = tf.import_graph_def(
        graph_def,
        input_map={f"ExpandDims:0": input_batch},
        return_elements=[FID_POOL_NAME, FID_SPATIAL_NAME],
        name=prefix,
    )
    _update_shapes(pool3)
    spatial = spatial[..., :7]
    return pool3, spatial


def _create_softmax_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    (matmul,) = tf.import_graph_def(
        graph_def, return_elements=[f"softmax/logits/MatMul"], name=prefix
    )
    w = matmul.inputs[1]
    logits = tf.matmul(input_batch, w)
    return tf.nn.softmax(logits)


def _update_shapes(pool3):
    # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L50-L63
    ops = pool3.graph.get_operations()
    for op in ops:
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:  # pylint: disable=protected-access
                # shape = [s.value for s in shape] TF 1.x
                shape = [s for s in shape]  # TF 2.x
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__["_shape_val"] = tf.TensorShape(new_shape)
    return pool3
