from dlcliche.utils import *
from lib_fat2019 import *

BASE = 'base_27_densenet121'

conf.DATA = Path('/mnt/dataset/freesound-audio-tagging-2019')
conf.PREPROCESSED = Path('/mnt/dataset/fat2019_private2')
conf.WORK = Path('/mnt/dataset/work/fat2019')
conf.MODEL = 'densenet121'
conf.PRETRAINED_MODEL = ''
conf.MIXMATCH_ALPHA = 0.
conf.SIZE = 160
conf.BS = 128
conf.DURATION = 1

update_conf(conf, raw_duration=1, raw_padmode='constant')

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
