from dlcliche.utils import *
from lib_fat2019 import *

BASE = 'base_36_specgram_6s_l1'

conf.DATA = Path('/mnt/dataset/freesound-audio-tagging-2019')
conf.PREPROCESSED = Path('/mnt/dataset/fat2019_private2')
conf.WORK = Path('/mnt/dataset/work/fat2019')
conf.MODEL = 'specgram'
conf.PRETRAINED_MODEL = ''
conf.SIZE = 160
conf.BS = 128
conf.DURATION = 6

conf.MIXMATCH_ALPHA = 0.
conf.CV = 0
conf.AUG_LEVEL = 3
conf.RELABEL = 'COOC_PROB'
conf.NOISY_DATA = 'NOISY_FULL'
conf.DATA_DOMAIN_XFR = False
conf.FORCE_SINGLE_LABEL = False

update_conf(conf, raw_duration=conf.DURATION, raw_padmode='constant')

set_fastai_random_seed()

df, X_train, learn, data = fat2019_initialize_training(conf, for_what='pretrain')
print(len(df), len(X_train))
print(conf.NAME)

learn.callback_fns[1].keywords['pseudo_labeling'] = False
learn.callback_fns[1].keywords['mixmatch_alpha'] = 0.
                        
learn.unfreeze()

print('stage 1')
learn.fit_one_cycle(3, slice(1e-2, 1e-1))
learn.fit_one_cycle(10, slice(1e-3, 1e-2))
learn.fit_one_cycle(100, 3e-3)
print('stage 2')
learn.fit_one_cycle(200, 1e-3, callbacks=get_saver_callbacks(conf, learn, f'{conf.WORK/BASE}_best'))
print('stage 3')
learn.fit_one_cycle(200, 1e-4, callbacks=get_saver_callbacks(conf, learn, f'{conf.WORK/BASE}_best2'))
visualize_first_layer(learn, conf.WORK/f'{BASE}.png')
learn.save(conf.WORK/f'{BASE}_last')

filename = f'{conf.WORK/BASE}_best2'
df = evaluate_weight(conf, filename, for_what='pretrain',
                     name_as='app_M5_6s_sp0_P37_sz160zZ_bs128_APD_tNa_NF_A3_l1.0_m0.75-0.0', K=3)
df.to_csv(Path(filename).with_suffix('.csv'))
