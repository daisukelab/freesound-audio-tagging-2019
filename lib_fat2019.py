import librosa
import librosa.display
#from dlcliche.utils import *
import PIL
import cv2
import random

from fastai import *
from fastai.callbacks import *
from fastai.vision import *
from fastai.vision.data import *
from fastai.callbacks.hooks import *

from audio_utils import *
from private_models import *

from sklearn.model_selection import StratifiedKFold

# Determine if this is running in Jupyter notebook or not
from IPython import get_ipython
ipython = get_ipython()
if ipython:
    running_in_notebook = ipython.has_trait('kernel')
else:
    running_in_notebook = False


if not running_in_notebook:
    #print('Switched matplotlib output to agg.')
    plt.switch_backend('agg')


# https://github.com/daisukelab/dl-cliche/blob/master/dlcliche/math.py
def geometric_mean_preds(list_preds):
    """Calculate geometric mean of prediction results.
    Prediction result is expected to be probability value of ML model's
    softmax outputs in 2d array.

    Arguments:
        list_preds: List of 2d numpy array prediction results.

    Returns:
        Geometric mean of input list of 2d numpy arrays.
    """
    preds = list_preds[0].copy()
    for next_preds in list_preds[1:]:
        preds = np.multiply(preds, next_preds)
    return np.power(preds, 1/len(list_preds))


def arithmetic_mean_preds(list_preds):
    """Calculate arithmetic mean of prediction results.
    Prediction result is expected to be probability value of ML model's
    softmax outputs in 2d array.

    Arguments:
        list_preds: List of 2d numpy array prediction results.

    Returns:
        Arithmetic mean of input list of 2d numpy arrays.
    """
    preds = list_preds[0].copy()
    for next_preds in list_preds[1:]:
        preds = np.add(preds, next_preds)
    return preds / len(list_preds)


def pick_confident_preds_by_rows(list_preds, k=5):
    """This picks top k confident predictions out of list.
    CAUTION: It is either effective or not, depends on the problem.
    Not for FAT2019.
    """
    top_k_rows = []
    for r in range(list_preds[0].shape[0]): # -> size(n_rows, k, n_classes)
        ordered = sorted([(np.max(preds[r]), preds[r]) for preds in list_preds],
                         key=lambda x: x[0], reverse=True)
        top_k_rows.append([row_preds for p, row_preds in ordered[:k]])
    relisted_preds = []
    for ik in range(k): # -> size(k, n_rows, n_classes)
        relisted_preds.append([rows[ik] for rows in top_k_rows])
    return np.array(relisted_preds)


def get_co_occurrence(multi_labels, delimiter=',', T=1.0, eps=1e-12):
    """Calculates co-occurrence probability of each single label from multi-labels set.
    Returns:
        Probability K x K matrix.
    """
    classes = sorted(list(set(flatten_list([ml.split(delimiter) for ml in multi_labels]))))
    c = len(classes)
    # Once get one hot labels
    train_one_hots = np.array([np.sum([np.eye(c)[classes.index(l)]
                                       for l in labels.split(delimiter)], axis=0)
                               for labels in multi_labels])
    # Calculate frequency of label occurances per classes
    n_co_occurrence = np.array([np.sum(train_one_hots[np.where(train_one_hots[:, i])], axis=0)
                              for i in range(c)])
    n_co_occurrence = np.power(n_co_occurrence, 1./T)
    n_co_oc_sum = np.clip(n_co_occurrence.sum(axis=1).reshape(-1, 1), eps, 1e12)
    # Calculate prior probability distribution of label co-occurence P(L)
    p_co_occurrence = n_co_occurrence / n_co_oc_sum
    # Adjust co-occuerence probability:
    # - Higher alpha than 1. will make probabilities harder
    # - Alpha in range (0., 1.) will make them softer
    #  ---> 1.0 is the only confirmed to be effective, others didn't work
    return p_co_occurrence


fat2019_data_conf = {
    'X': None,
    'fnames': [],
    'rand_crop': True,
    'crop_level': 1,
    'tfms': None,
    'tta_mode': 0,
    'part': 0,
    'n_parts': 1,
    'base_dim': 0,
    'fname_map': [],
}

def get_aug_for_mixmatch(conf):
    # print('Importing Albumentations...')
    import albumentations as A
    if conf.AUG_LEVEL == 0:
        augs = [
            # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=2., p=0.75),
            A.RandomBrightnessContrast(brightness_limit=0.02, contrast_limit=0.02, p=0.5),
            A.Cutout(max_h_size=conf.SIZE//40, max_w_size=conf.SIZE//40, p=0.5),
        ]
    elif conf.AUG_LEVEL == 1:
        augs = [
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=1., p=0.75),
            A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5),
            A.Cutout(max_h_size=conf.SIZE//20, max_w_size=conf.SIZE//20, p=0.5),
        ]
    elif conf.AUG_LEVEL == 2:
        augs = [
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=2., p=0.75),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.Cutout(max_h_size=conf.SIZE//20, max_w_size=conf.SIZE//20, p=0.5),
        ]
    elif conf.AUG_LEVEL == 3:
        augs = [
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=2., p=0.75),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Cutout(max_h_size=conf.SIZE//10, max_w_size=conf.SIZE//10, p=0.5),
        ]
    else:
        raise Exception(f'unknown conf.AUG_LEVEL {conf.AUG_LEVEL}')
    tfms = A.Compose(augs)
    def call_tfms(instance):
        return tfms(image=instance)['image']
    return call_tfms

def _get_fat2019_image(X, idx, rand_crop=True, crop_level=0, tfms=None, tta_mode=0, part=0, n_parts=1)->Image:
    # Load
    x = PIL.Image.fromarray(X[idx])

    # Crop
    time_dim, base_dim = x.size
    if rand_crop:
        if crop_level == 0:
            crop_x = random.randint(0 - base_dim//40, time_dim - base_dim)
            crop_y = random.randint(-base_dim//40, 0)
        elif crop_level == 1:
            crop_x = random.randint(0 - base_dim//20, time_dim - base_dim)
            crop_y = random.randint(-base_dim//10, 0)
        elif crop_level >= 2:
            crop_x = random.randint(0 - base_dim//10, time_dim - base_dim)
            crop_y = random.randint(-base_dim//3, 0)
        else:
            raise Exception(f'unknown crop_level {crop_level}')
    else:
        d = max(time_dim - base_dim, 0) // n_parts
        crop_x = d * part
        crop_y = 0
        if tta_mode > 0:
            if part > 0:
                delta = (tta_mode * base_dim) // 20
                crop_x += random.randint(-delta, delta)
                crop_y = random.randint(-delta, 0)
    x = x.crop([crop_x, crop_y, crop_x+base_dim, crop_y + base_dim])

    # Augment
    if tfms is not None:
        x = PIL.Image.fromarray(tfms(np.array(x)))

    # normalize
    return Image(pil2tensor(x, np.float32).div_(255)).data


def set_fat2019_data(conf, X, fnames, rand_crop=True, tfms=None, tta_mode=0, part=0, n_parts=1):
    fat2019_data_conf['X'] = X
    fat2019_data_conf['fnames'] = fnames
    fat2019_data_conf['rand_crop'] = rand_crop
    fat2019_data_conf['crop_level'] = conf.AUG_LEVEL
    fat2019_data_conf['tfms'] = tfms
    fat2019_data_conf['tta_mode'] = tta_mode
    fat2019_data_conf['part'] = part
    fat2019_data_conf['n_parts'] = n_parts
    fat2019_data_conf['base_dim'] = conf.SIZE


def get_fat2019_image(idx)->Image:
    if fat2019_data_conf['X'] is None:
        X = [np.zeros((fat2019_data_conf['base_dim'], fat2019_data_conf['base_dim'], 3))
             .astype(np.uint8)]
        idx = 0 # dummy
    else:
        X = fat2019_data_conf['X']

    return _get_fat2019_image(
        X=X,
        idx=idx,
        rand_crop=fat2019_data_conf['rand_crop'],
        crop_level=fat2019_data_conf['crop_level'],
        tfms=fat2019_data_conf['tfms'],
        tta_mode=fat2019_data_conf['tta_mode'],
        part=fat2019_data_conf['part'],
        n_parts=fat2019_data_conf['n_parts'],
    )


class ImageListFat2019(ImageList):
    def __init__(self, items, **kwargs):
        super().__init__(items, **kwargs)
        #setattr(self, 'label_from_df', self.label_from_df)
        self.index_map = None

    def get(self, i):
        if self.index_map is None:
            fnames = fat2019_data_conf['fnames']
            runtime_fnames = [fn.split('/')[-1] for fn in self.items]
            self.index_map = [fnames.index(fn) for fn in runtime_fnames]
        res = get_fat2019_image(self.index_map[i])
        self.sizes[i] = res.size
        return res


def get_image_for_mixmatch(X, idx, tfms=None, rand_crop=False, part=0, n_parts=1):
    x = _get_fat2019_image(X, idx, rand_crop=rand_crop, tfms=tfms,
                            tta_mode=0, part=part, n_parts=n_parts)
    return x


class SimpleFat2019Dataset(torch.utils.data.Dataset):

    def __init__(self, X, ys, device, rand_crop, tfms, tta_mode=0, part=0, n_parts=1):
        self.X, self.ys, self.device = X, ys, device
        self.set_params(rand_crop, tfms, tta_mode, part, n_parts)

    def set_params(self, rand_crop, tfms, tta_mode, part, n_parts):
        self.rand_crop, self.tfms, self.tta_mode, self.part, self.n_parts = rand_crop, tfms, tta_mode, part, n_parts

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        x = _get_fat2019_image(self.X, idx, rand_crop=self.rand_crop, tfms=self.tfms,
                               tta_mode=self.tta_mode, part=self.part, n_parts=self.n_parts)
        return x.data, self.ys[idx]


class MixMatchCallback(MixUpCallback):
    """MixMatch implementation _INCOMPLETE_

    Unfortunately just be used as super version of Mixup.
    """

    eps = 1e-12
    GUESS_EPOCH_SPAN = 4
    GUESS_PREFIX_MIX = 0.3
    EWA_COEF = 0.5

    def __init__(self, learn:Learner, conf=None, y_prefix_onehots=None, X_unlabeled=None, K=5,
                 pseudo_labeling=True, mixmatch_alpha=0., T=0.5, U=64, lambda_U=75/5, activ=nn.Sigmoid(), super_mixup=False,
                 alpha:float=0.75, stack_x:bool=False, stack_y:bool=True):
        super().__init__(learn, alpha, stack_x, stack_y)
        self.y_prefix_onehots = torch.tensor(y_prefix_onehots).float()
        self.X_unlabeled, self.K = X_unlabeled, K
        self.pseudo_labeling, self.mixmatch_alpha, self.T = pseudo_labeling, mixmatch_alpha, T
        self.guessed_ys = None
        self.U, self.lambda_U, self.activ = U, lambda_U, activ
        self.tfms = get_aug_for_mixmatch(conf)
        self.rand_crop = False
        if super_mixup:
            self.mixmatch_alpha = self.alpha
            self.U = 0
            self.pseudo_labeling = False
            self.guessed_ys = self.y_prefix_onehots
            self.rand_crop = True

    def on_train_begin(self, **kwargs):
        self.learn.loss_func = MixMatchLoss(self.learn.loss_func, self.U, self.lambda_U,
                                            self.learn.data.batch_size, self.activ)

    def on_batch_begin(self, last_input, last_target, train, num_batch, **kwargs):
        "Applies mixup to `last_input` and `last_target` if `train`."
        self.learn.loss_func.set_train(train)
        if not train: return

        # Usual Mixup or MixMatch
        if not self.unlabeled_blending_ready():
            ret = super().on_batch_begin(last_input, last_target, train, **kwargs)
            new_target, new_input = ret['last_target'], ret['last_input']
        else:
            new_target, new_input = last_target, last_input

        not_used_input = np.array(list(range(len(new_input))))

        # MixMatch batch building, only when batch size is enough
        if self.unlabeled_blending_ready() and self.learn.data.batch_size == len(new_input) and self.U > 0:
            shuffle_input = random.sample(range(len(new_input)), len(new_input)-self.U)
            not_used_input = np.array([i for i in range(len(new_input)) if i not in shuffle_input])
            shuffle_noisy = random.sample(range(len(self.X_unlabeled)), self.U)
            xL = new_input[shuffle_input]
            xU = torch.stack([get_image_for_mixmatch(self.X_unlabeled, idx, tfms=self.tfms, rand_crop=self.rand_crop).data.cuda() for idx in shuffle_noisy])
            x1 = torch.cat([xL, xU])
            y1 = torch.cat([new_target[shuffle_input], torch.tensor(self.guessed_ys[shuffle_noisy]).cuda()])
            new_target, new_input = y1, x1
            #if num_batch == 0: print(f'batch={len(new_input)} U={self.U}')

        # MixMatch's MixUp
        if self.unlabeled_blending_ready():
            lambd = np.random.beta(self.mixmatch_alpha, self.mixmatch_alpha, len(new_input))
            lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
            lambd = new_input.new(lambd)
            shuffle_noisy = random.sample(range(len(self.X_unlabeled)), len(new_input))
            xU = torch.stack([get_image_for_mixmatch(self.X_unlabeled, idx, tfms=self.tfms, rand_crop=self.rand_crop).data.cuda() for idx in shuffle_noisy])
            yU = torch.tensor(self.guessed_ys[shuffle_noisy]).cuda()
            xL = last_input[not_used_input]
            yL = last_target[not_used_input]
            shuffle_mix = random.sample(range(len(xL) + len(xU)), len(new_input))
            x1 = torch.cat([xL, xU])[shuffle_mix]
            y1 = torch.cat([yL, yU])[shuffle_mix]
            new_input = (new_input * lambd.view(lambd.size(0),1,1,1) + x1 * (1-lambd).view(lambd.size(0),1,1,1))
            if len(last_target.shape) == 2:
                lambd = lambd.unsqueeze(1).float()
            new_target = new_target.float() *  lambd + y1.float() * (1-lambd)
            del x1, y1
        elif self.pseudo_labeling and num_batch == 0:
            pass #print('First epoch: Skip blending unlabeled samples.')

        return {'last_input': new_input, 'last_target': new_target}
    
    def on_epoch_end(self, epoch, pbar, **kwargs):
        if not self.pseudo_labeling: return
        assert not self.rand_crop, 'Label guessing cannot come with rand_crop'
        if (epoch % self.GUESS_EPOCH_SPAN) != 0: return
        
        this_guessed_ys = guess_labels(self.learn, X=self.X_unlabeled, prefix_y=self.y_prefix_onehots,
                                       tfms=self.tfms, pbar=pbar, K=self.K, T=self.T, 
                                       PREFIX_MIX=self.GUESS_PREFIX_MIX, eps=self.eps)
        #EWA
        if self.guessed_ys is None:
            self.guessed_ys =  this_guessed_ys
        else:
            self.guessed_ys = (1 - self.EWA_COEF) * self.guessed_ys + self.EWA_COEF * this_guessed_ys
        # Following print will help us understanding how confident guessed_ys is.
        print(f'guessed {self.guessed_ys.max(1)[0]} {self.guessed_ys.min(1)[0]}')

    def on_train_end(self, **kwargs):
        self.learn.loss_func = self.learn.loss_func.get_old()

    def unlabeled_blending_ready(self):
        return self.mixmatch_alpha > 0 and self.guessed_ys is not None


def guess_labels(learn, X, prefix_y, tfms, pbar, K=5, T=0.5, PREFIX_MIX=0.5, eps=1e-13):
    classes = learn.data.classes
    assert isinstance(X, list)
    X_selected = X * K
    ys = np.zeros((len(X_selected), len(classes)))
    dl = torch.utils.data.DataLoader(SimpleFat2019Dataset(X_selected, ys, device=learn.data.device,
                                                          rand_crop=False, tfms=tfms),
                                     batch_size=learn.data.batch_size, shuffle=False)
    dl = DeviceDataLoader(dl, learn.data.device)
    with torch.no_grad():
        preds, _ = get_preds(learn.model, dl, activ=torch.sigmoid, pbar=pbar)
        #print(f'1 {torch.isnan(preds).sum()} {preds.shape}')
        preds = torch.mean(preds.view(K, len(X), len(classes)), dim=0)
        if prefix_y is not None:
            prefix_y = torch.tensor(prefix_y).float()
            preds = PREFIX_MIX * prefix_y + (1 - PREFIX_MIX) * preds
        #print(f'2 {torch.isnan(preds).sum()} {preds.shape}')
        preds = preds.pow(1/T)
        #print(f'3 {torch.isnan(preds).sum()} {preds.shape}, zeros={(0 == preds).sum()}')
        preds_sum = preds.sum(1).unsqueeze(1).clamp(min=eps)
        #print(f'4 {torch.isnan(preds_sum).sum()} {preds_sum.shape}, zeros={(0 == preds_sum).sum()}')
        guessed_ys = preds / preds_sum
    return guessed_ys


class MixMatchLoss(nn.Module):
    "Adapt the loss function `crit` to go with MixMatch."
    
    def __init__(self, crit, U, lambda_U, batch_size, activ=nn.Sigmoid(), reduction='mean'):
        super().__init__()
        self.U, self.lambda_U, self.batch_size, self.activ = U, lambda_U, batch_size, activ
        self.crit, self.reduction = crit, reduction
        self.l2 = nn.MSELoss(reduction=self.reduction)
        self.train = True

    def set_train(self, train):
        self.train = train

    def use_original_loss(self, cur_batch_size=None):
        if not self.train: return True # In validation
        if self.U == 0: return True    # Not using unlabeled
        if cur_batch_size and self.batch_size > cur_batch_size: return True
        return False

    def forward(self, output, target):
        if self.use_original_loss(cur_batch_size=len(output)): # incomplete batch
            return self.crit(output, target)
        L = len(output) - self.U
        dL = self.crit(output[:L], target[:L])
        ul_out, ul_targ = self.activ(output[L:]), self.activ(target[L:])
        dU = self.l2(ul_out, ul_targ)
        d = dL + self.lambda_U * dU
        #print(f'o={nan_o} t={nan_t} {dL} {dU} {d}')
        return d

    def get_old(self):
        return self.crit


def load_fat2019_X(conf, filename):
    print(f'load_fat2019_X: {filename}')
    # Cache
    if filename in conf.cur_X_train_cache:
        cache = conf.cur_X_train_cache[filename]
        if (cache['DATAMODE'] == conf.DATAMODE
            and cache['DATA_BG'] == conf.DATA_BG
            and cache['DURATION'] == conf.DURATION):
            print(f'Using cached {filename} - {cache["DATAMODE"]},{cache["DATA_BG"]},{cache["DURATION"]}')
            return cache['X']
        del cache['X']

    # Create new
    X = pickle.load(open(filename, 'rb'))
    # Data mode conversion
    if conf.DATAMODE == 'AAA': # Amplitude only
        print(' remaking as AAA')
        for x in X:
            x[..., 1:] = x[..., :1] 
    elif conf.DATAMODE == 'AAP': # Amplitude x 2 + Phoase
        for x in X:
            x[:] = np.stack([x[..., 0], x[..., 0], x[..., 1]], axis=-1)
    elif conf.DATAMODE == 'APX':
        print(' remaking as APX')
        for x in X:
            x[:] = np.stack([x[..., 0], x[..., 1], x[..., 0]*x[..., 1]], axis=-1)
    elif conf.DATAMODE == 'APD':
        pass
    else:
        raise Exception(f'unknowm mode:{conf.DATAMODE}')
    # Data BG Modification
    if conf.DATA_BG == 'SUB_MEAN':
        print('X -> subtracting stationary BG noise')
        for i in range(len(X)):
            real_width = np.argmax(np.sum(X[i], axis=(0, 2)) == 0)
            real_width = real_width or X[i].shape[1] # real_width is zero if no 0-filled
            x_mean = np.mean(X[i][:, :real_width, :], axis=(0, 1), keepdims=True).astype(np.uint8)
            X[i][X[i] < x_mean] = 0

    # Data length conversion
    if conf.raw_duration != conf.DURATION:
        width_to_be_one_sec = conf.DURATION*conf.SIZE
        for i in range(len(X)):
            # Pad if too short
            if X[i].shape[1] < width_to_be_one_sec:
                X[i] = np.pad(X[i], ((0, 0), (0, width_to_be_one_sec - X[i].shape[1]), (0, 0)),
                              mode=conf.PADMODE)
            # Resize
            h, w, _ = X[i].shape
            X[i] = cv2.resize(X[i], (int(w/width_to_be_one_sec*conf.SIZE), h))

    # Save cache
    conf.cur_X_train_cache[filename] = {
        'DATAMODE': conf.DATAMODE,
        'DATA_BG': conf.DATA_BG,
        'DURATION': conf.DURATION,
        'X': X
    }
    return X


def fat2019_get_model(conf):
    if conf.MODEL == 'resnet34':
        this_model = models.resnet34
    elif conf.MODEL == 'simple1':
        this_model = simple1
    elif conf.MODEL == 'senet18':
        this_model = se_resnet18
    elif conf.MODEL == 'senet9':
        this_model = se_resnet9
    elif conf.MODEL == 'vgg16_bn':
        this_model = models.vgg16_bn
    elif conf.MODEL == 'vgg19_bn':
        this_model = models.vgg19_bn
    elif conf.MODEL == 'alexnet':
        this_model = models.alexnet
    elif conf.MODEL == 'specgram':
        this_model = specgram_net
    elif conf.MODEL == 'densenet121':
        this_model = models.densenet121
    elif conf.MODEL == 'densenet169':
        this_model = models.densenet169
    else:
        raise Exception('unknown model')
    print(f'Model {conf.MODEL}')
    return this_model


def single_label_dataset(classes, df, X=None):
    cls_idxs = list_class_indexes(classes, df)
    flat_classes, flat_idxs = [], []
    for i, idxs in enumerate(cls_idxs):
        flat_classes.extend([classes[i]] * len(idxs))
        flat_idxs.extend(idxs)
    df = df.loc[flat_idxs].copy()
    if X is not None:
        X = [X[i] for i in flat_idxs]
    # relabel as single label
    df['org_labels'] = df.labels
    df.labels = flat_classes
    df.reset_index(drop=True, inplace=True)
    return df, X


def add_one_hot(df, classes):
    for c in classes: df[c] = 0
    for i, row in df.iterrows():
        labels = row.labels.split(',')
        for l in labels:
            df.loc[i, l] = 1
    return df


def coocfy_one_hots(df, classes, prior_df, T):
    cooc_probs = get_co_occurrence(prior_df.labels, T=T)
    target = df[classes].values
    cooc = np.dot(target, cooc_probs)
    cooc /= cooc.max(axis=-1)[:, np.newaxis] # Normalized to range [0., 1.]
    df[classes] = cooc
    return df


def _get_noisy_data_def(conf):
    _def = [
        ['NOISY_FULL', conf.MELSP_TRN_NOISY, conf.MELSP_TRN_NOISY_DX, conf.CSV_TRN_NOISY_CV],
        ['NOISY_50S', conf.MELSP_TRN_NOISY_BEST50S, conf.MELSP_TRN_NOISY_DX_BEST50S, conf.CSV_TRN_NOISY_BEST50S],
        ['NOISY_SINGLE', conf.MELSP_TRN_NOISY_SINGLE, conf.MELSP_TRN_NOISY_DX_SINGLE, conf.CSV_TRN_NOISY_SINGLE],
    ]
    name_list = [str(d[0]) for d in _def]
    if conf.NOISY_DATA in name_list:
        return _def, name_list.index(conf.NOISY_DATA)
    raise Exception(f'Unknown noisy data (conf.NOISY_DATA): {name}')
def noisy_mels_data(conf):
    data_def, idx = _get_noisy_data_def(conf)
    return data_def[idx][2] if conf.DATA_DOMAIN_XFR else data_def[idx][1]
def noisy_csv_data(conf):
    data_def, idx = _get_noisy_data_def(conf)
    return data_def[idx][3]


def list_class_indexes(classes, df):
    escaped_classes = [re.escape(c) for c in classes]
    return [list(df[df.labels.str.contains(c)].index) for c in escaped_classes]


def filter_by_OKs(classes, OKs, dists):
    return [d if c in OKs else 0 for c, d in zip(classes, dists)]


def list_sample_sizes(classes, dict_sizes, df=None):
    if dict_sizes is None:
        # Auto adjust size for complete flat distribution
        cls_idxs = list_class_indexes(classes, df)
        max_items = int(np.max([len(idxs) for idxs in cls_idxs]))
        return [max_items - len(idxs) for idxs in cls_idxs]
    return [dict_sizes[c] if c in dict_sizes else 0 for c in classes]


def resample_by_class_sizes(conf, classes, train_df, trn_idxs):
    cls_idxs = list_class_indexes(classes, train_df.iloc[trn_idxs])
    max_items = int(np.max([len(idxs) for idxs in cls_idxs]))
    sizes = list_sample_sizes(classes, conf.RESAMPLE_SIZES, train_df.iloc[trn_idxs])
    print(' resampling detail', sizes)
    # Undersampling - edit cls_idxs/sizes directly
    for i, (idxs, sz) in enumerate(zip(cls_idxs, sizes)):
        if sz >= 0: continue
        idxs
        resampled_idxs = list(random.sample(idxs, len(idxs) + sz))
        cls_idxs[i] = resampled_idxs
        sizes[i] = 0
    # Oversampling
    resampled_cls_idxs = [idxs + list(random.sample(idxs, sz))
                          for idxs, sz in zip(cls_idxs, sizes)]
    idxs = flatten_list(resampled_cls_idxs)
    print(f'Resampled {len(trn_idxs)} -> {len(idxs)}')
    return idxs


def select_noisy_samples(conf, classes, df_cur_train, df_noisy):
    # Calculate number of adding sample size per class
    if conf.NOISY_SAMPLE_SIZES is None:
        # Fill with noisy samples according to current training distributions
        cur_cls_idxs = list_class_indexes(classes, df_cur_train)
        cur_dists = [len(idxs) for idxs in cur_cls_idxs]
        max_items = int(np.max(cur_dists))

        num_of_cls_item_to_add = [max_items - cur for cur in cur_dists]
    else:
        # Add noisy samples according to NOISY_SAMPLE_SIZES
        num_of_cls_item_to_add = list_sample_sizes(classes, conf.NOISY_SAMPLE_SIZES)
    num_of_cls_item_to_add = filter_by_OKs(classes, conf.noisy_OKs, num_of_cls_item_to_add)
    print(' adding noisy items', num_of_cls_item_to_add)

    cls_idxs = list_class_indexes(classes, df_noisy)
    resampled_cls_idxs = [random.sample(idxs, size) if idxs else []
                          for idxs, size in zip(cls_idxs, num_of_cls_item_to_add)]
    idxs = flatten_list(resampled_cls_idxs)
    print(f' selected {len(idxs)} noisy samples')
    return idxs


def fat2019_initialize_training(conf, for_what='train'):
    update_conf(conf, size=conf.SIZE, raw_padmode=conf.raw_padmode, raw_trim_silence=conf.raw_trim_silence)
    classes = get_classes(conf)

    # Basic data loading
    if for_what == 'train':
        X_curated_raw = load_fat2019_X(conf, conf.MELSP_TRN_CURATED)
    else:
        X_curated_raw = None
    if for_what == 'test':
        X_noisy_raw = None
    else:
        X_noisy_raw = load_fat2019_X(conf, noisy_mels_data(conf))
    df = pd.read_csv(conf.CSV_TRN_CV)
    df_noisy = pd.read_csv(noisy_csv_data(conf))

    if conf.FORCE_SINGLE_LABEL:
        before_ns = len(df), len(df_noisy)
        df, X_curated_raw = single_label_dataset(classes, df, X_curated_raw)
        df_noisy, X_noisy_raw = single_label_dataset(classes, df_noisy, X_noisy_raw)
        print(f'Forced labels to be single: curated {before_ns[0]}->{len(df)}, noisy {before_ns[1]}->{len(df_noisy)}')

    if conf.NOISY_REMOVE_NG and for_what != 'pretrain':
        ng_idxs = flatten_list(list_class_indexes(conf.noisy_NGs, df_noisy))
        if X_noisy_raw is not None:
            X_noisy_raw = [x for i, x in enumerate(X_noisy_raw) if i not in ng_idxs]
        df_noisy = df_noisy[~df_noisy.index.isin(ng_idxs)].reset_index(drop=True)

    # More data extension
    if for_what == 'pretrain':
        df = df_noisy
        train_idxes, valid_idxes = np.where(df.train)[0], np.where(~df.train)[0]
        X_train = X_noisy_raw
        tfms = get_aug_for_mixmatch(conf)
        print(f'Using train noisy data')
    elif for_what == 'test':
        train_idxes, valid_idxes = list(range(len(df) - 1)), [len(df) - 1] # dummy
        X_train = None
        tfms = None
        print(f'Using no data for test')
    elif for_what == 'train':
        print(f'Using train curated data')

        # Basic: Resample & CV
        X_train, df, train_idxes, valid_idxes = fat2019_train_valid_split(conf, X_curated_raw, df, classes)
        if conf.RESAMPLE:
            train_idxes = resample_by_class_sizes(conf, classes, train_df=df, trn_idxs=train_idxes)

        # Mixed split
        if conf.USE_NOISY:
            idxs = select_noisy_samples(conf, classes, df.iloc[train_idxes], df_noisy)
            X_addon = [X_noisy_raw[i] for i in idxs]
            df_addon = df_noisy.iloc[idxs]

            n = len(df)
            X_train = X_train + X_addon
            df = pd.concat([df, df_addon], ignore_index=True)
            train_idxes = list(train_idxes) + list(range(n, len(df)))
            print(f' adding noisy sample: {conf.NOISY_DATA}')

        tfms = get_aug_for_mixmatch(conf)
    else:
        raise Exception(f'Unknown purpose: {for_what}')

    df = df[['fname', 'labels']]
    add_one_hot(df, classes)
    if conf.RELABEL == 'COOC_PROB':
        print(f'Co-occurrence probability soft label. T={conf.COOC_PROB_T}')
        coocfy_one_hots(df, classes, prior_df=pd.read_csv(conf.CSV_TRN_CURATED), T=conf.COOC_PROB_T)

    df.to_csv(conf.WORK/'working_train.csv', index=False)
    print(f'Saved training metadata as {conf.WORK/"working_train.csv"}')
    print(f'Data split: train) n={len(train_idxes)} {np.min(train_idxes)}...{np.max(train_idxes)}, '
        + f'valid) n={len(valid_idxes)} {np.min(valid_idxes)}...{np.max(valid_idxes)}')

    # Model
    this_model = fat2019_get_model(conf)
    print(f'Configured as {conf.NAME}')
    
    set_fat2019_data(conf, X_train, list(df.fname.values), rand_crop=True, tfms=tfms)

    data = (ImageListFat2019.from_csv(conf.WORK, 'working_train.csv', folder='trn_curated')
            .split_by_idxs(train_idxes, valid_idxes)
            .label_from_df(cols=classes, one_hot=True)
            #.transform(tfms) # not using fast.ai's
            .databunch(bs=conf.BS, no_check=True)
    )
    conf.cur_train_indexes = train_idxes
    conf.cur_valid_indexes = valid_idxes

    learn = cnn_learner(data, this_model, pretrained=False, metrics=[lwlrap])

    if conf.PRETRAINED_MODEL:
        print(f'Loading {conf.PRETRAINED_MODEL}...')
        load_fastai_learner_model(learn, conf.PRETRAINED_MODEL)
        learn.freeze()
    else:
        learn.unfreeze()

    if conf.LOSS == 'BCEL':
        print('Using BCEWithLogitsLoss()')
        learn.loss_func = nn.BCEWithLogitsLoss()
    else:
        raise Exception('unknown loss')

    if conf.MIXUP_ALPHA > 0 and for_what != 'test':
        print(f'Mixup alpha={conf.MIXUP_ALPHA} MixMatch alpha={conf.MIXMATCH_ALPHA}')
        X_mixmatch = X_noisy_raw
        df_mixmatch = df_noisy.copy()
        add_one_hot(df_mixmatch, classes)
        learn.callback_fns.append(partial(MixMatchCallback, conf=conf,
                                          y_prefix_onehots=df_mixmatch[classes].values,
                                          X_unlabeled=X_mixmatch,
                                          K=3, pseudo_labeling=False, mixmatch_alpha=conf.MIXMATCH_ALPHA,
                                          T=0.5, U=0, lambda_U=75/3, activ=nn.Sigmoid(), super_mixup=conf.SUPER_MIXUP,
                                          alpha=conf.MIXUP_ALPHA, stack_x=False, stack_y=False))
    return df, X_train, learn, data


def predict_fat2019_test(learn, WORK, CSV_SUBMISSION, X_test, test_df,
                         n_oversample=1, mean_fn=arithmetic_mean_preds, tta_mode=2):
    sampled_preds = []
    for p in range(n_oversample):
        if p == 0:
            set_fat2019_data(conf, X=X_test, fnames=list(test_df.fname.values))
            learn.data.add_test(test_df.fname.values)
        set_fat2019_data(conf, X=X_test, fnames=list(test_df.fname.values), rand_crop=False,
                         tfms=None, tta_mode=2, n_parts=n_oversample, part=p)
        preds, _ = learn.get_preds(ds_type=DatasetType.Test)
        if preds.min() < 0 or 1 < preds.max():
            preds.sigmoid_()
        sampled_preds.append(to_np(preds))
    preds = mean_fn(sampled_preds)
    return preds


class conf:
    # Preprocessing settings
    sampling_rate = 44100
    raw_duration = 1
    hop_length = 276*raw_duration # 347 for 128, 197 for 224, 276 for 160
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 160
    n_fft = n_mels * 20
    raw_padmode = 'repeat'#, 'constant' # 'symmetric'
    raw_trim_silence = False
    samples = sampling_rate * raw_duration
    
    # Basic things
    DATA = Path('../input/freesound-audio-tagging-2019')
    ROOT = Path('../input/fat2019_files')
    WORK = Path('work')

    # Training settings
    SIZE = 160
    MODEL = 'densenet121'
    PRETRAINED_MODEL = WORK/'base_4_APX_best.pth'
    PADMODE = 'constant' # 'repeat' #'symmetric'
    FLIPMODE = False
    AUG_LEVEL = 0
    DATAMODE = 'APD'
    BS = 128
    DURATION = 1
    RESAMPLE = False
    USE_NOISY = True
    NOISY_DATA = 'NOISY_SINGLE'
    DATA_DOMAIN_XFR = True
    LOSS = 'BCEL' 
    DATA_BG = '' # 'SUB_MEAN'
    CV_SPLIT = 3
    CV = 0
    MIXUP_ALPHA = 0.75
    MIXMATCH_ALPHA = 0.75
    REAL_MIXMATCH = False
    RELABEL = 'COOC_PROB'
    COOC_PROB_T = 1.
    SAVE_CRITERIA = 'lwlrap' # 'loss'
    RESAMPLE_SIZES = None
    NOISY_SAMPLE_SIZES = None
    NOISY_REMOVE_NG = True
    FORCE_SINGLE_LABEL = True
    DATA_ADJUST_SET = -1
    SUPER_MIXUP = False

    # Data use definitions
    noisy_OKs = ['Fart', 'Chewing_and_mastication', 'Cutlery_and_silverware', 'Burping_and_eructation', 'Writing',
    'Dishes_and_pots_and_pans', 'Sneeze', 'Squeak', 'Child_speech_and_kid_speaking', 'Knock',
    'Cupboard_open_or_close', 'Clapping', 'Chink_and_clink', 'Drawer_open_or_close', 'Bass_guitar',
    'Bark', 'Acoustic_guitar', 'Meow', 'Chirp_and_tweet', 'Hiss']

    noisy_NGs = ['Race_car_and_auto_racing', 'Gong', 'Traffic_noise_and_roadway_noise', 'Waves_and_surf']

    curated_NG_but__s = ['Bus',  'Church_bell',  'Cricket',  'Electric_guitar',  'Frying_(food)',
    'Harmonica',  'Male_singing',  'Male_speech_and_man_speaking',  'Run',  'Slam',
    'Traffic_noise_and_roadway_noise']

    curated_OKs = ['Finger_snapping', 'Tick-tock', 'Sigh', 'Skateboard', 'Gong',
    'Glockenspiel', 'Scissors', 'Marimba_and_xylophone', 'Computer_keyboard', 'Bicycle_bell',
    'Accordion', 'Drip', 'Raindrop', 'Motorcycle', 'Car_passing_by',
    'Water_tap_and_faucet', 'Toilet_flush', 'Fill_(with_liquid)', 'Shatter', 'Strum',
    'Sink_(filling_or_washing)', 'Yell', 'Race_car_and_auto_racing', 'Printer', 'Zipper_(clothing)',
    'Applause', 'Gurgling', 'Waves_and_surf', 'Cheering', 'Accelerating_and_revving_and_vroom',
    'Bathtub_(filling_or_washing)', 'Female_speech_and_woman_speaking', 'Screaming', 'Mechanical_fan', 'Keys_jangling',
    'Microwave_oven', 'Trickle_and_dribble', 'Buzz', 'Gasp', 'Female_singing',
    'Stream', 'Purr', 'Whispering', 'Hi-hat', 'Crackle',
    'Crowd', 'Bass_drum', 'Tap', 'Walk_and_footsteps']

    # Runtime
    cur_train_indexes = []
    cur_valid_indexes = []
    cur_X_train_cache = {}


def one_second_samples(size=160):
    return (347 if size == 128 else
            276 if size == 160 else
            197 if size == 224 else
            115 if size == 384 else
            0) # error


def padmode_letter(padmode, trim_silence):
    silence = '' if trim_silence else 'z'
    return (silence + 'S' if padmode == 'symmetric' else
            silence + 'R' if padmode == 'repeat' else
            silence + 'Z' if padmode == 'constant' else 'PADMODE_ERROR')


def update_conf(conf, size=160, raw_duration=1, raw_padmode='constant', raw_trim_silence=False, keep_filenames=False):
    conf.raw_duration = raw_duration
    conf.n_mels = conf.SIZE = size
    conf.raw_padmode = raw_padmode
    conf.raw_trim_silence = raw_trim_silence
    conf.hop_length = raw_duration * one_second_samples(size)
    conf.n_fft = conf.n_mels * 20
    conf.samples = conf.sampling_rate * raw_duration

    _SZ = str(conf.n_mels) + padmode_letter(conf.raw_padmode, conf.raw_trim_silence)
    _SA = str(conf.SIZE) + padmode_letter(conf.PADMODE, conf.raw_trim_silence)
    if conf.PRETRAINED_MODEL:
        t = str(Path(conf.PRETRAINED_MODEL).name).split('_')
        t = t[1] if len(t) > 1 and t[1].isnumeric() else 'x'
    _M = ('_M1' if conf.MODEL == 'resnet34' else
          '_M2' if conf.MODEL == 'vgg16_bn' else
          '_M3' if conf.MODEL == 'simple1' else
          '_M4' if conf.MODEL == 'specgram' else
          '_M5' if conf.MODEL == 'densenet121' else
          '_M6' if conf.MODEL == 'densenet169' else
          '_M7' if conf.MODEL == 'senet9' else
          '_M8' if conf.MODEL == 'simple1' else '_Mx')
    _DU = f'{conf.raw_duration}s' if conf.raw_duration != 1 else ''
    _DA = f'_{conf.DURATION}s'
    _CV = f'_sp{conf.CV}'
    _PT = '_Pnone' if not conf.PRETRAINED_MODEL else '_P'+t
    _RE = '_RE' if conf.RESAMPLE else ''
    _tN = ('' if not conf.USE_NOISY else
           '_tNa' if conf.NOISY_DATA == 'NOISY_FULL' else
           '_tN5' if conf.NOISY_DATA == 'NOISY_50S' else
           '_tNs' if conf.NOISY_DATA == 'NOISY_SINGLE' else '_tNunk')
    _BG = '_BGs' if conf.DATA_BG == 'SUB_MEAN' else ''
    _FL = '_NF' if conf.FLIPMODE == False else '' # No Flip
    _AU = '' if conf.AUG_LEVEL == 0 else f'_A{conf.AUG_LEVEL}'
    _RL = f'_l{conf.COOC_PROB_T}' if conf.RELABEL == 'COOC_PROB' else '_lx'
    _MU = f'_m{conf.MIXUP_ALPHA}-{conf.MIXMATCH_ALPHA}{"-S" if conf.SUPER_MIXUP else ""}'
    _MM = '_MM' if conf.REAL_MIXMATCH else ''
    _SC = '_Sl' if conf.SAVE_CRITERIA == 'loss' else ''
    _LS = '_Ls' if conf.FORCE_SINGLE_LABEL else ''
    _ADJ = f'_da{conf.DATA_ADJUST_SET}' if conf.DATA_ADJUST_SET >= 0 else ''
    FOOTER = f'{_BG}{_RE}{_tN}{_FL}{_AU}{_RL}{_MU}{_MM}{_SC}{_LS}{_ADJ}'
    conf.NAME = f'app{_M}{_DA}{_CV}{_PT}_sz{_SA}_bs{conf.BS}_{conf.DATAMODE}{FOOTER}'

    # File names
    if keep_filenames: return
    Path(conf.WORK).mkdir(exist_ok=True, parents=True)
    sys.path.append(str(conf.ROOT))

    conf.CSV_TRN_CURATED = conf.DATA/'train_curated.csv'
    conf.CSV_TRN_NOISY = conf.DATA/'train_noisy.csv'
    conf.CSV_SUBMISSION = conf.DATA/'sample_submission.csv'
    conf.CSV_TRN_CV = conf.ROOT/'trn_curated_cv.csv'
    conf.CSV_TRN_NOISY_CV = conf.ROOT/'train_noisy_cv.csv'

    conf.MELSP_TRN_CURATED = conf.WORK/f'melsp{_DU}{_SZ}_train_curated.pkl'
    conf.MELSP_TRN_NOISY = conf.WORK/f'melsp{_DU}{_SZ}_train_noisy.pkl'
    conf.MELSP_TEST = conf.WORK/f'melsp{_DU}{_SZ}_test.pkl'
    
    conf.CSV_TRN_NOISY_SINGLE = conf.ROOT/'trn_noisy_single.csv'
    conf.MELSP_TRN_NOISY_SINGLE = conf.WORK/f'melsp{_DU}{_SZ}_train_noisy_single.pkl'
    conf.CSV_TRN_NOISY_BEST50S = conf.ROOT/f'trn_noisy_best50s.csv'
    conf.MELSP_TRN_NOISY_BEST50S = conf.WORK/f'melsp{_DU}{_SZ}_trn_noisy_best50s.pkl'

    conf.MELSP_TRN_NOISY_DX = conf.WORK/f'melsp{_DU}{_SZ}_train_noisy_dx.pkl'
    conf.MELSP_TRN_NOISY_DX_SINGLE = conf.WORK/f'melsp{_DU}{_SZ}_train_noisy_dx_single.pkl'
    conf.MELSP_TRN_NOISY_DX_BEST50S = conf.WORK/f'melsp{_DU}{_SZ}_trn_noisy_dx_best50s.pkl'

    conf.TRN_CURATED = conf.DATA/'train_curated'
    conf.TRN_NOISY = conf.DATA/'train_noisy'
    conf.TEST = conf.DATA/'test'


update_conf(conf)


def set_fastai_random_seed(seed=42):
    # https://docs.fast.ai/dev/test.html#getting-reproducible-results

    # python RNG
    random.seed(seed)

    # pytorch RNGs
    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # numpy RNG
    import numpy as np
    np.random.seed(seed)


def disable_fastai_progress():
    import fastprogress
    import fastai

    fastprogress.fastprogress.NO_BAR = True
    master_bar, progress_bar = fastprogress.force_console_behavior()
    fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar


def get_saver_callbacks(conf, learn, name):
    if conf.SAVE_CRITERIA == 'loss':
        return [SaveModelCallback(learn, name=name)] # valid_loss/auto by default
    return [SaveModelCallback(learn, monitor='lwlrap', mode='max', name=name)]


def visualize_cnn_by_cam(learn, data_index):
    # Thanks to https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson6-pets-more.ipynb

    x, _y = learn.data.valid_ds[data_index]
    y = _y.data
    if not isinstance(y, (list, np.ndarray)): # single label -> one hot encoding
        y = np.eye(learn.data.valid_ds.c)[y]

    m = learn.model.eval()
    xb,_ = learn.data.one_item(x)
    xb_im = Image(learn.data.denorm(xb)[0])
    xb = xb.cuda()

    def hooked_backward(cat):
        with hook_output(m[0]) as hook_a: 
            with hook_output(m[0], grad=True) as hook_g:
                preds = m(xb)
                preds[0,int(cat)].backward()
        return hook_a,hook_g
    def show_heatmap(img, hm, label):
        _,axs = plt.subplots(1, 2)
        axs[0].set_title(label)
        img.show(axs[0])
        axs[1].set_title(f'CAM of {label}')
        img.show(axs[1])
        axs[1].imshow(hm, alpha=0.6, extent=(0,img.shape[1],img.shape[1],0),
                      interpolation='bilinear', cmap='magma');
        plt.show()

    for y_i in np.where(y > 0)[0]:
        hook_a,hook_g = hooked_backward(cat=y_i)
        acts = hook_a.stored[0].cpu()
        grad = hook_g.stored[0][0].cpu()
        grad_chan = grad.mean(1).mean(1)
        mult = (acts*grad_chan[...,None,None]).mean(0)
        show_heatmap(img=xb_im, hm=mult, label=str(learn.data.valid_ds.y[data_index]))
        
# https://discuss.pytorch.org/t/how-to-visualize-the-actual-convolution-filters-in-cnn/13850
from sklearn.preprocessing import minmax_scale

def visualize_first_layer(learn, save_name=None):
    conv1 = list(learn.model.children())[0][0]
    if isinstance(conv1, torch.nn.modules.container.Sequential):
        conv1 = conv1[0] # for some models, 1 layer inside
    weights = conv1.weight.data.cpu().numpy()
    weights_shape = weights.shape
    weights = minmax_scale(weights.ravel()).reshape(weights_shape)
    fig, axes = plt.subplots(8, 8, figsize=(8,8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.rollaxis(weights[i], 0, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if save_name:
        fig.savefig(str(save_name))


def force_load_state_dict(model, state_dict, layer='classifier'):
    new_dict = OrderedDict()
    state_dict_weights = list(state_dict.values())
    # Rename all items in state_dict as the name from model's state_dict
    for i, (k, v) in enumerate(model.state_dict().items()):
        if k.find(layer) < 0:
            new_dict[k] = state_dict_weights[i]
    model.load_state_dict(new_dict, strict=False)

def load_fastai_learner_model(learn, filename):
    state_dict = torch.load(filename)
    force_load_state_dict(learn.model, state_dict['model'] if 'model' in state_dict else state_dict)

def load_fat2019_model_weight(conf, learn, learner_pkl_file):
    learner_pkl_file = str(learner_pkl_file)
    learner_pkl_file = (learner_pkl_file[:-4] if learner_pkl_file[-4:] == '.pth'
                        else learner_pkl_file)
    def load_or_skip(filepath):
        filepath = Path(filepath)
        if filepath.is_file():
            print(f'Loading {filepath}')
            load_fastai_learner_model(learn, filepath)
            return True
        return False
    if load_or_skip(f'{learner_pkl_file}'): return
    if load_or_skip(f'{learner_pkl_file}.pth'): return
    if load_or_skip(conf.WORK/f'models/{learner_pkl_file}.pth'): return
    learn.load(learner_pkl_file)


from itertools import chain
import re

def flatten_list(lists):
    return list(chain.from_iterable(lists))

def get_classes(conf=conf):
    csvfile=conf.CSV_SUBMISSION
    classes = list(pd.read_csv(csvfile).columns[1:])
    return classes


import pickle

def save_as_pkl_binary(obj, filename):
    """Save object as pickle binary file.
    Thanks to https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file/32216025
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pkl(filename):
    """Load pickle object from file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def split_train_valid_as_idx(conf, trn_df, classes, cv, cv_split):
    assert cv_split == 3, 'Not implemented'
    
    trn_idxs = np.where(trn_df.valid != cv)[0]
    val_idxs = np.where(trn_df.valid == cv)[0]

    return trn_idxs, val_idxs


def fat2019_train_valid_split(conf, X_trn, trn_df, classes):
    print(f'Valid set={conf.CV}/{conf.CV_SPLIT}')
    trn_idxs, val_idxs = split_train_valid_as_idx(conf, trn_df, classes,
                                                  cv=conf.CV, cv_split=conf.CV_SPLIT)
    print(f'Train split: train/valid = {len(trn_idxs)}/{len(val_idxs)}')
    return X_trn, trn_df, trn_idxs, val_idxs


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def flatten_y_if_onehot(y):
    """De-one-hot y, i.e. [0,1,0,0,...] to 1 for all y."""
    return y if len(np.array(y).shape) == 1 else np.argmax(y, axis = -1)

def get_class_distribution(y):
    """Calculate number of samples per class."""
    # y_cls can be one of [OH label, index of class, class label name string]
    # convert OH to index of class
    y_cls = flatten_y_if_onehot(y)
    # y_cls can be one of [index of class, class label name]
    classset = sorted(list(set(y_cls)))
    sample_distribution = {cur_cls:len([one for one in y_cls if one == cur_cls]) for cur_cls in classset}
    return sample_distribution

def get_class_distribution_list(y, num_classes):
    """Calculate number of samples per class as list"""
    dist = get_class_distribution(y)
    assert(y[0].__class__ != str) # class index or class OH label only
    list_dist = np.zeros((num_classes))
    for i in range(num_classes):
        if i in dist:
            list_dist[i] = dist[i]
    return list_dist


# from official code https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8#scrollTo=cRCaCIb9oguU
def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class


def lwlrap(scores, truth, **kwargs):
    if scores.min() < 0 or 1 < scores.max():
        scores.sigmoid_()
    score, weight = calculate_per_class_lwlrap(to_np(truth), to_np(scores))
    return torch.Tensor([(score * weight).sum()])


def np_lwlrap(scores, truth):
    if scores.min() < 0 or 1 < scores.max():
        scores = to_np(torch.Tensor(scores).sigmoid_())
    score, weight = calculate_per_class_lwlrap(truth, scores)
    return (score * weight).sum()


def df_lwlrap_sum(df):
    return (df.lwlrap*df.weight).sum()


### Building App

APD_PRETRAINED_MODELS = {
    #'senet9': 'base_models/base_13_senet9_APD_best.pth',
    'senet9': 'base_models/base_13_160_senet9_XP2_best.pth',
    'resnet34': 'base_models/base_24_resnet34_best.pth',
    'vgg16_bn': 'base_models/base_26_vgg16_bn_best.pth',
    'densenet121': 'base_models/base_27_densenet121_best2.pth', # 'base_models/base_28_densenet121_dx_best.pth', 
    'specgram': 'base_models/base_16_specgram_best.pth'
}
APX_PRETRAINED_MODELS = {
    'resnet34': 'base_models/base_4_APX_best.pth',
    'vgg16_bn': 'base_models/base_7_vgg_XP2_best.pth',
    'densenet121': 'base_models/base_17_densenet121_best.pth',
    'densenet169': 'base_models/base_19_densenet169_best.pth',
    'senet9': 'base_models/base_13_160_senet9_XP2_best.pth',
    'simple1': 'base_models/base_15_simple1_best.pth',
}
AAA_PRETRAINED_MODELS = {
    'densenet121': 'base_models/base_18_densenet121_AAA_best2.pth',
    'densenet169': 'base_models/base_20_densenet169_AAA_best2.pth',
}
TUNE_STAGE1_CYCLES = {
    'resnet34': [(10, 1e-1), (10, 1e-2), (10, 1e-2), (10, 1e-2), (10, 1e-2)],
    'vgg16_bn': [(10, 1e-1), (10, 1e-2), (10, 1e-2), (10, 1e-2), (10, 1e-2)],
    'densenet121': [(10, 1e-1), (5, 1e-2), (5, 1e-2), (10, 1e-2)],
    'densenet169': [(10, 1e-1), (10, 1e-2), (10, 1e-2), (10, 1e-2)],
    'senet9': [(10, 1e-1), (10, 1e-2), (10, 1e-2), (10, 1e-2), (10, 1e-2)],
    'simple1': [(10, 1e-1), (10, 1e-2), (10, 1e-2), (10, 1e-2), (10, 1e-2)],
}
TUNE_STAGE2_CYCLES = {
    'resnet34': [(50, slice(1e-3, 1e-2)), (30, slice(3e-4, 3e-3)), (30, slice(3e-4, 3e-3)), (30, slice(1e-4, 1e-3)), (20, slice(1e-4, 1e-3))],
    'vgg16_bn': [(50, slice(1e-3, 1e-2)), (30, slice(3e-4, 3e-3)), (30, slice(3e-4, 3e-3)), (30, slice(1e-4, 1e-3)), (20, slice(1e-4, 1e-3))],
    # long 'densenet121': [(50, slice(1e-3, 1e-2)), (30, slice(3e-4, 3e-3)), (30, slice(3e-4, 3e-3)), (20, slice(1e-4, 1e-3)), (20, slice(1e-4, 1e-3))],
    'densenet121': [(30, slice(1e-3, 1e-2)), (20, slice(3e-4, 3e-3)), (10, slice(3e-5, 3e-4))],
    'densenet169': [(50, slice(1e-3, 1e-2)), (30, slice(3e-4, 3e-3)), (30, slice(3e-4, 3e-3)), (20, slice(1e-4, 1e-3)), (20, slice(1e-4, 1e-3))],
    'senet9': [(50, slice(1e-3, 1e-2)), (30, slice(3e-4, 3e-3)), (30, slice(3e-4, 3e-3)), (30, slice(1e-4, 1e-3)), (20, slice(1e-4, 1e-3))],
    'simple1': [(50, slice(1e-3, 1e-2)), (30, slice(3e-4, 3e-3)), (30, slice(3e-4, 3e-3)), (30, slice(1e-4, 1e-3)), (20, slice(1e-4, 1e-3))],
}

def get_pretrained_model(conf):
    if conf.DATAMODE == 'APX' and conf.MODEL in APX_PRETRAINED_MODELS:
        return APX_PRETRAINED_MODELS[conf.MODEL]
    if conf.DATAMODE == 'AAA' and conf.MODEL in AAA_PRETRAINED_MODELS:
        return AAA_PRETRAINED_MODELS[conf.MODEL]
    return APD_PRETRAINED_MODELS[conf.MODEL]


def set_app_conf(app_conf):
    for k in app_conf:
        if k == 'MODEL':    conf.MODEL = app_conf[k]
        if k == 'DATAMODE': conf.DATAMODE = app_conf[k]
        if k == 'FLIPMODE': conf.FLIPMODE = app_conf[k]
        if k == 'DURATION': conf.DURATION = app_conf[k]
        if k == 'LOSS':     conf.LOSS = app_conf[k]
        if k == 'RESAMPLE': conf.RESAMPLE = app_conf[k]
        if k == 'DATA_BG':  conf.DATA_BG = app_conf[k]
        if k == 'CV':       conf.CV = app_conf[k]
        if k == 'MIXUP':    conf.MIXUP_ALPHA = app_conf[k]
        if k == 'MIXMATCH': conf.MIXMATCH_ALPHA = app_conf[k]
        if k == 'RELABEL':  conf.RELABEL = app_conf[k]


## Evaluating App

def prm_get_model(prm):
    return ['resnet34', 'vgg16_bn', 'simple1', 'specgram', 'densenet121', 'densenet169', 'senet9', 'simple1'][int(prm[1][1:]) - 1]
def prm_get_seconds(prm):
    return int(prm[2][:-1])
def prm_get_data_bg(prm):
    return 'SUB_MEAN' if 'BGs' in prm else ''
def prm_get_padmode(prm):
    padmode = prm[5][-1]
    return ('symmetric' if padmode == 'S' else
            'repeat' if padmode == 'R' else
            'constant' if padmode == 'Z' else 'PADMODE_ERROR')
def prm_get_raw_trim_silence(prm):
    padmode = prm[5][-2]
    return (padmode != 'z')
def prm_get_split(prm):
    return int(prm[3][2:]) # _sp0
def prm_get_resample(prm):
    for p in prm:
        if p == 'RE': return True
    return False
def prm_get_relabel(prm):
    for p in prm:
        if p == 'lx': return False
    return True

def create_learner_by_weight(conf, weight_file, name_as=None, work='.', for_what='test'):
    prm = str(Path(name_as or weight_file).name).split('_')
    conf.WORK = work
    conf.MODEL = prm_get_model(prm)
    conf.DURATION = prm_get_seconds(prm)
    conf.DATAMODE = prm[7]
    conf.PADMODE = prm_get_padmode(prm)
    conf.DATA_BG = prm_get_data_bg(prm)
    conf.raw_trim_silence = prm_get_raw_trim_silence(prm)
    conf.CV = prm_get_split(prm)
    conf.RESAMPLE = prm_get_resample(prm)
    conf.RELABEL = prm_get_relabel(prm)
    conf.PRETRAINED_MODEL = ''
    update_conf(conf, raw_trim_silence=conf.raw_trim_silence)
    
    print(weight_file, conf.MODEL, conf.DURATION, conf.PADMODE, conf.DATAMODE, conf.DATA_BG)

    df, X_train, learn, data = fat2019_initialize_training(conf, for_what=for_what)
    load_fat2019_model_weight(conf, learn, weight_file)

    return df, X_train, learn


def label_to_one_hot(classes, labels):
    c = len(classes)
    oh = np.sum([np.eye(c)[classes.index(l)]
                 for l in labels if l in classes], axis=0)
    if not isinstance(oh, Iterable):
        oh = np.zeros((c))
    return oh


def evaluate_valid_preds(classes, preds, valid_df):
    one_hot_gts = np.array([label_to_one_hot(classes, labels.split(','))
                            for labels in valid_df.labels.values])
    lwlrap = np_lwlrap(preds, one_hot_gts)
    return lwlrap


def evaluate_weight(conf, weight, name_as=None, for_what='train', K=3, df_only=True, with_eval_only=True):
    conf.FORCE_SINGLE_LABEL = True
    conf.USE_NOISY = False
    conf.RESAMPLE = True
    conf.RESAMPLE_SIZES = None
    conf.RELABEL = False

    df, X_train, learn = create_learner_by_weight(conf, weight, name_as=name_as, work=conf.WORK, for_what=for_what)

    if with_eval_only:
        list_vreds = []
        for _ in range(K):
            vreds = learn.get_preds(DatasetType.Valid)
            list_vreds.append(to_np(vreds[0].sigmoid_()))
            gts = to_np(vreds[1])
        preds = np.mean(list_vreds, axis=0)
    else:
        # All training samples
        indexes = np.r_[conf.cur_train_indexes, conf.cur_valid_indexes]
        df = df.loc[indexes]
        X = [X_train[i] for i in indexes]
        preds = predict_fat2019_test(learn, conf.WORK, conf.CSV_SUBMISSION, X, df, n_oversample=K)
        classes = get_classes(conf)
        add_one_hot(df, classes)
        gts = df[classes].values

    score, cls_weight = calculate_per_class_lwlrap(gts, preds)
    df = (pd.DataFrame({'class': get_classes(conf), 'lwlrap': score, 'weight': cls_weight})
          .set_index('class').sort_values('lwlrap'))

    if df_only:
        return df
    return df, gts, preds


def np_softmax(z):
    """Numpy version softmax.
    Thanjs to https://stackoverflow.com/a/39558290/6528729
    """
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


def model_class_weighted_mean_preds(list_preds, list_lwlraps):
    all_preds = np.array(list_preds)
    all_lwlraps = np.array(list_lwlraps)

    # Class-wise softmax over models
    weight = np_softmax(all_lwlraps.T).T

    weighted_preds = all_preds * weight.reshape((all_preds.shape[0], 1, -1))
    weighted_mean_preds = weighted_preds.sum(axis=0)

    return weighted_mean_preds


def model_class_weighted_mean_preds_by_df(list_preds, pred_score_dfs):
    list_preds = np.array(list_preds)
    list_lwlraps = [df.lwlrap.values for df in pred_score_dfs]

    return model_class_weighted_mean_preds(list_preds, list_lwlraps)

