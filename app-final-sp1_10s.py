from dlcliche.image import *
from lib_fat2019 import *

APPNAME = 'final'

conf.DURATION = 10
conf.AUG_LEVEL = 2
conf.CV = 1
conf.RELABEL = 'COOC_PROB'

conf.DATA = Path('/mnt/dataset/freesound-audio-tagging-2019')
conf.ROOT = Path('/mnt/dataset/fat2019_files')
conf.WORK = Path('/mnt/dataset/work/fat2019')

conf.MODEL = 'densenet121'
conf.PRETRAINED_MODEL = 'base_models/base_37_densenet121_10s_l1_best2.pth' # get_pretrained_model(conf)

conf.RESAMPLE = False
conf.RESAMPLE_SIZES = None

conf.USE_NOISY = True
conf.NOISY_DATA = 'NOISY_SINGLE'
conf.NOISY_SAMPLE_SIZES = None

conf.NOISY_REMOVE_NG = ['Bus', 'Run',
 'Male_speech_and_man_speaking',
 'Cricket',
 'Race_car_and_auto_racing',
 'Printer',
 'Church_bell',
 'Crowd',
 'Gong',
 'Mechanical_fan',
 'Traffic_noise_and_roadway_noise',
 'Waves_and_surf']

conf.noisy_OKs = ['Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Applause', 'Bark',
 'Bass_drum', 'Bass_guitar', 'Bathtub_(filling_or_washing)', 'Bicycle_bell', 'Burping_and_eructation',
 'Buzz', 'Car_passing_by', 'Cheering', 'Chewing_and_mastication', 'Child_speech_and_kid_speaking',
 'Chink_and_clink', 'Chirp_and_tweet', 'Clapping', 'Computer_keyboard', 'Crackle',
 'Cupboard_open_or_close', 'Cutlery_and_silverware','Dishes_and_pots_and_pans', 'Drawer_open_or_close', 'Drip',
 'Electric_guitar', 'Fart', 'Female_singing', 'Female_speech_and_woman_speaking', 'Fill_(with_liquid)',
 'Finger_snapping', 'Frying_(food)', 'Gasp', 'Glockenspiel', 'Gurgling',
 'Harmonica', 'Hi-hat', 'Hiss', 'Keys_jangling', 'Knock',
 'Male_singing', 'Marimba_and_xylophone', 'Meow', 'Microwave_oven', 'Motorcycle',
 'Purr', 'Raindrop', 'Scissors', 'Screaming', 'Shatter',
 'Sigh', 'Sink_(filling_or_washing)', 'Skateboard', 'Slam', 'Sneeze',
 'Squeak', 'Stream', 'Strum', 'Tap', 'Tick-tock',
 'Toilet_flush', 'Trickle_and_dribble', 'Walk_and_footsteps', 'Water_tap_and_faucet', 'Whispering',
 'Writing', 'Yell', 'Zipper_(clothing)']

conf.FORCE_SINGLE_LABEL = True
conf.SUPER_MIXUP = True

update_conf(conf)
set_fastai_random_seed()

best_weight = f'{conf.NAME}_{APPNAME}'

df, X_train, learn, data = fat2019_initialize_training(conf)

learn.fit_one_cycle(8, 1e-1)
learn.fit_one_cycle(10, 1e-2)
learn.unfreeze()
learn.fit_one_cycle(20, 1e-2)
learn.fit_one_cycle(20, 1e-2)
learn.fit_one_cycle(20, 1e-2)

learn.fit_one_cycle(30, slice(1e-3, 1e-2))
learn.fit_one_cycle(30, slice(1e-3, 1e-2))
learn.fit_one_cycle(30, slice(1e-3, 1e-2))
learn.fit_one_cycle(300, slice(1e-4, 1e-2), callbacks=get_saver_callbacks(conf, learn, best_weight))

print(f'Writing {best_weight}.csv ...')
df = evaluate_weight(conf, f'{best_weight}', for_what='train')
df.to_csv(f'work/models/{best_weight}.csv')
print('lwlrap', df_lwlrap_sum(df))
print(df[:10])
