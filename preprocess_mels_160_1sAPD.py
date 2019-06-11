from dlcliche.utils import *
from lib_fat2019 import *

conf.DATA = Path('/mnt/dataset/freesound-audio-tagging-2019')
conf.ROOT = Path('/mnt/dataset/fat2019_files')
conf.WORK = Path('/mnt/dataset/work/fat2019')

update_conf(conf, size=160, raw_duration=1, raw_padmode='constant', raw_trim_silence=False)

trn_curated_df = pd.read_csv(conf.CSV_TRN_CURATED)
test_df = pd.read_csv(conf.CSV_SUBMISSION)
trn_noisy_50s_df = pd.read_csv(conf.CSV_TRN_NOISY_BEST50S)
trn_noisy_single_df = pd.read_csv(conf.CSV_TRN_NOISY_SINGLE)
trn_noisy_df = pd.read_csv(conf.CSV_TRN_NOISY)

classes = get_classes(conf)


def convert_dataset(df, source_folder, filename):
    print(f'Creating {filename}...')
    X = convert_wav_to_APD(conf, df, source=source_folder,
                           pad_mode=conf.raw_padmode, trim_silence=conf.raw_trim_silence)
    save_as_pkl_binary(X, filename)
    return X


# Preprocess samples into Mel-spectrogram
convert_dataset(trn_curated_df, conf.TRN_CURATED, conf.MELSP_TRN_CURATED);
convert_dataset(test_df, conf.TEST, conf.MELSP_TEST);
convert_dataset(trn_noisy_50s_df, conf.TRN_NOISY, conf.MELSP_TRN_NOISY_BEST50S);
convert_dataset(trn_noisy_single_df, conf.TRN_NOISY, conf.MELSP_TRN_NOISY_SINGLE);
convert_dataset(trn_noisy_df, conf.TRN_NOISY, conf.MELSP_TRN_NOISY);


# Domain transfer
from domain_freq_xfer import *

Xdest = load_pkl(conf.MELSP_TRN_CURATED)
ydest = pd.read_csv(conf.CSV_TRN_CURATED).labels
classes = get_classes(conf)

dom_freq_xfer = DomainFreqTransfer(classes)
dom_freq_xfer.fit(Xdest, ydest)

print(f'Creating {conf.MELSP_TRN_NOISY_DX_BEST50S}...')
Xsrc = load_pkl(conf.MELSP_TRN_NOISY_BEST50S)
ysrc = pd.read_csv(conf.CSV_TRN_NOISY_BEST50S).labels
dom_freq_xfer(Xsrc, ysrc, show_samples=0)
save_as_pkl_binary(Xsrc, conf.MELSP_TRN_NOISY_DX_BEST50S)

print(f'Creating {conf.MELSP_TRN_NOISY_DX_SINGLE}...')
Xsrc = load_pkl(conf.MELSP_TRN_NOISY_SINGLE)
ysrc = pd.read_csv(conf.CSV_TRN_NOISY_SINGLE).labels
dom_freq_xfer(Xsrc, ysrc, show_samples=0)
save_as_pkl_binary(Xsrc, conf.MELSP_TRN_NOISY_DX_SINGLE)

print(f'Done.')
