from dlcliche.utils import *
from lib_fat2019 import *

BASE = 'base_16_specgram'

conf.DATA = Path('/mnt/dataset/freesound-audio-tagging-2019')
conf.ROOT = Path('/mnt/dataset/fat2019_files')
conf.WORK = Path('/mnt/dataset/work/fat2019')
conf.MODEL = 'specgram'
conf.PRETRAINED_MODEL = ''
conf.MIXMATCH_ALPHA = 0.
conf.SIZE = 160
conf.BS = 128
conf.DURATION = 1
conf.CV = 0
conf.AUG_LEVEL = 2
conf.RELABEL = True
conf.NOISY_DATA = 'NOISY_FULL'
conf.DATA_DOMAIN_XFR = False
conf.FORCE_SINGLE_LABEL = False

update_conf(conf, raw_duration=1, raw_padmode='constant')

set_fastai_random_seed()

df, X_train, learn, data = fat2019_initialize_training(conf, for_what='pretrain')
print(len(df), len(X_train))
print(conf.NAME)

learn.callback_fns[1].keywords['pseudo_labeling'] = False
learn.callback_fns[1].keywords['mixmatch_alpha'] = 0.

if True:
    learn.unfreeze()

    print('stage 1')
    learn.fit_one_cycle(3, slice(1e-2, 1e-1))
    learn.fit_one_cycle(10, slice(1e-3, 1e-2))
    learn.fit_one_cycle(100, 3e-3)
    print('stage 2')
    #learn.fit_one_cycle(200, 1e-3)
    #print('stage 3')
    learn.fit_one_cycle(200, 1e-4, callbacks=get_saver_callbacks(conf, learn, f'{conf.WORK/BASE}_best'))
    visualize_first_layer(learn, conf.WORK/f'{BASE}.png')
    learn.save(conf.WORK/f'{BASE}_last')

print(f'Writing {conf.WORK/BASE}_best.csv ...')
df = evaluate_weight(conf, f'{conf.WORK/BASE}_best', for_what='pretrain',
                     name_as='app_M4_1s_sp0_P16_sz160zZ_bs128_APD_tNa_NF_A2_lx_m0.75-0.0')
df.to_csv(f'{conf.WORK/BASE}_best.csv')
print(df[:10])
    
    
