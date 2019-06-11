#from dlcliche.utils import *
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython
from tqdm import tqdm


def _spectrogram_phase(y=None, S=None, n_fft=2048, hop_length=512, power=1,
                       win_length=None, window='hann', center=True, pad_mode='reflect'):
    """
    Thanks to https://github.com/librosa/librosa/blob/master/librosa/feature/spectral.py
    Extension to the original _spectrogram, phase is also retrieved.
    """
    if S is not None:
        n_fft = 2 * (S.shape[0] - 1)
    else:
        D = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length,
                              win_length=win_length, center=center,
                              window=window, pad_mode=pad_mode)
        S = np.abs(D)**power
        P = np.angle(D)

    return S, P, n_fft

def melspectrogram_phase(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                         win_length=None, window='hann', center=True, pad_mode='reflect',
                         power=2.0, **kwargs):
    """
    Thanks to https://github.com/librosa/librosa/blob/master/librosa/feature/spectral.py
    Extension to the original librosa.feature.melspectrogram, phase is also retrieved.
    """
    S, P, n_fft = _spectrogram_phase(y=y, S=S, n_fft=n_fft, hop_length=hop_length,
                                     power=power, win_length=win_length, window=window,
                                     center=center, pad_mode=pad_mode)

    mel_basis = librosa.filters.mel(sr, n_fft, **kwargs)

    return np.dot(mel_basis, S), np.dot(mel_basis, P)


def read_audio(conf, pathname, trim_long_data, pad_mode, trim_silence=True):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    # trim silence
    if trim_silence:
        if 0 < len(y): # workaround: 0 length causes error
            y, _ = librosa.effects.trim(y) # trim, top_db=default(60)
    # make it unified length to conf.samples
    if len(y) > conf.samples: # long enough
        if trim_long_data:
            y = y[0:0+conf.samples]
    else: # pad blank
        padding = conf.samples - len(y)    # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), pad_mode)
    return y


def audio_to_melsphase(conf, audio):
    spectrogram, phase = melspectrogram_phase(audio, 
                                              sr=conf.sampling_rate,
                                              n_mels=conf.n_mels,
                                              hop_length=conf.hop_length,
                                              n_fft=conf.n_fft,
                                              fmin=conf.fmin,
                                              fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram, phase


def audio_to_melspectrogram(conf, audio):
    spectrogram, phase = audio_to_melsphase(conf, audio)
    return spectrogram


def show_melsphase(conf, mels, melp, title='Log-frequency power spectrogram'):
    _, axs = plt.subplots(1, 2)
    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 
                             sr=conf.sampling_rate, hop_length=conf.hop_length,
                             fmin=conf.fmin, fmax=conf.fmax, ax=axs[0])
    #axs[0].colorbar(format='%+2.0f dB')
    axs[0].set_title(title)
    axs[1].imshow(melp)
    plt.show()


def show_melspectrogram(conf, mels, title='Log-frequency power spectrogram'):
    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 
                             sr=conf.sampling_rate, hop_length=conf.hop_length,
                            fmin=conf.fmin, fmax=conf.fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()


def read_as_melsphase(conf, pathname, trim_long_data, pad_mode, trim_silence=True, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data, pad_mode, trim_silence=trim_silence)
    mels, melp = audio_to_melsphase(conf, x)
    if debug_display:
        IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))
        show_melsphase(conf, mels, melp)
    return mels, melp


def read_as_melspectrogram(conf, pathname, trim_long_data, pad_mode, trim_silence=True, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data, pad_mode, trim_silence=trim_silence)
    mels = audio_to_melspectrogram(conf, x)
    if debug_display:
        IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))
        show_melspectrogram(conf, mels)
    return mels


def normalize_mels1d(X1, eps=1e-12):
    norm_min, norm_max = X1.min(), X1.max()
    if (norm_max - norm_min) > eps:
        # Normalize to [0, 255]
        V = X1
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
    else:
        # Just zero
        V = np.zeros_like(X1)
    return V.astype(np.uint8)


def mono_to_APX(X, P, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-12):
    X = np.stack([X, P, X*P], axis=-1)

    # Standardize
    mean = mean or X.mean(axis=(0,1), keepdims=True)
    std = std or X.std(axis=(0,1), keepdims=True)
    Xstd = (X - mean) / (std + eps)

    # Normalize
    return np.stack([normalize_mels1d(Xstd[..., ch], eps) for ch in range(3)], axis=-1)


def convert_wav_to_APX(conf, df, source, pad_mode, trim_silence):
    X = []
    for i, row in tqdm(df.iterrows()):
        x, p = read_as_melsphase(conf, source/str(row.fname), trim_long_data=False,
                                 pad_mode=pad_mode, trim_silence=trim_silence)
        x_color = mono_to_APX(x, p)
        X.append(x_color)
    return X


def mono_to_APD(X, P, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-12):
    X = np.stack([X, P, librosa.feature.delta(X)], axis=-1)

    # Standardize
    mean = mean or X.mean(axis=(0,1), keepdims=True)
    std = std or X.std(axis=(0,1), keepdims=True)
    Xstd = (X - mean) / (std + eps)

    # Normalize
    return np.stack([normalize_mels1d(Xstd[..., ch], eps) for ch in range(3)], axis=-1)


def convert_wav_to_APD(conf, df, source, pad_mode, trim_silence):
    X = []
    for i, row in tqdm(df.iterrows()):
        x, p = read_as_melsphase(conf, source/str(row.fname), trim_long_data=False,
                                 pad_mode=pad_mode, trim_silence=trim_silence)
        x_color = mono_to_APD(x, p)
        X.append(x_color)
    return X
