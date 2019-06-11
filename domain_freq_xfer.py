from dlcliche.utils import *

from lib_fat2019 import *

def to_cls_idxs(one_labels, classes, delimiter=','):
    one_cls_idxs = [classes.index(l) for l in one_labels.split(delimiter)]
    return one_cls_idxs

def decompose_class_indexes(all_labels, classes, delimiter=','):
    cls_idxs = [to_cls_idxs(labels, classes, delimiter=delimiter) for labels in all_labels]
    return cls_idxs

def freq_enveropes(X):
    return np.mean([np.mean(x[:, :, 0], axis=1) for x in X], axis=0)

def to_cls_freq_envs(X, labels, classes, delimiter):
    cls_idxs = decompose_class_indexes(labels, classes, delimiter=delimiter)
    freq_envs = [freq_enveropes([X[i] for i in idxs])
                 for idxs in cls_idxs]
    return np.array(freq_envs)

def to_cls_envelope_map(srcs, dests, clip_min_src=255/10, clip_max_map=2.):
    env_map = []
    for src, dest in zip(srcs, dests):
        src = np.clip(src, clip_min_src, np.max(src))
        cls_env_map = np.clip(dest / src, 0, clip_max_map)
        env_map.append(cls_env_map)
    return np.array(env_map)

def mix_envelope_map(maps, cls_idxs):
    return np.mean([maps[ci] for ci in cls_idxs], axis=0)


def apply_envelope_map_one(x_one_plane, _map):
    x1 = x_one_plane * _map
    #x1 = (x1 / x1.max() * 255.)
    return x1.astype(np.uint8)

def apply_envelope_map(x, _map):
    _xA = apply_envelope_map_one(x[..., 0], _map)
    _xP = x[..., 1]
    _xD = apply_envelope_map_one(x[..., 2], _map)
    return np.stack([_xA, _xP, _xD], axis=-1)

class DomainFreqTransfer:
    def __init__(self, classes, clip_min_src=255/10, clip_max_map=2.):
        self.classes = classes
        self.clip_min_src = clip_min_src
        self.clip_max_map = clip_max_map

    def fit(self, dest_domain_X, dest_domain_ml_y, delimiter=',',
            src_domain_X=None, src_domain_ml_y=None):
        self.delimiter = delimiter
        # Class mean freq envelope
        self.dest_freq_envs = to_cls_freq_envs(dest_domain_X, dest_domain_ml_y,
                                               self.classes, delimiter=delimiter)
        if src_domain_X is not None:
            self.src_freq_envs = to_cls_freq_envs(src_domain_X, src_domain_ml_y,
                                                  self.classes, delimiter=delimiter)
        # Freq envelope map from src to dest
        #self.maps = to_cls_envelope_map(self.src_freq_envs, self.dest_freq_envs,
        #                clip_min_src=self.clip_min_src, clip_max_map=self.clip_max_map)

    #def map_by_labels(self, one_y):
    #    cls_idxs = to_cls_idxs(one_y, self.classes, delimiter=self.delimiter)
    #    return mix_envelope_map(self.maps, cls_idxs).reshape((self.maps[0].shape[0], 1))

    def __call__(self, X, y, show_samples=20):
        """Domain transfer samples in X inplace."""
        assert len(X[0].shape) == 3
        for i, (_x, _y) in enumerate(zip(X, y)):
            # Old design: single envelope mapping per class
            #_map = self.map_by_labels(_y)

            # New design: sample-wise envelope mapping
            src = freq_enveropes([_x])
            dest = self.dest_freq_envs[self.classes.index(_y)]
            _map = (dest/np.maximum(src, 1)).reshape((_x.shape[0], 1))

            X[i] = apply_envelope_map(_x, _map)
        
            if i < show_samples:
                mapped = freq_enveropes([X[i]])
                plt.plot(src, label='src')
                plt.plot(dest, label='dest')
                plt.plot(mapped, label='mapped')
                plt.title(_y)
                plt.legend()
                plt.show()