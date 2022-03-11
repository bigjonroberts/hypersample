# prepare the environment
import jukebox
import torch as t
import librosa
import os
import gc
# from IPython.display import Audio
from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.sample import sample_single_window, _sample, \
                           sample_partial_window, upsample, \
                           load_prompts
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.torch_utils import empty_cache
rank, local_rank, device = setup_dist_from_mpi()


gc.collect()

# Sample from the 5B or 1B Lyrics Model

model = '5b_lyrics' # or '5b' or '1b_lyrics'
hps = Hyperparams()
hps.sr = 44100
hps.n_samples = 3 if model in ('5b', '5b_lyrics') else 8
# Specifies the directory to save the sample in.
# We set this to the Google Drive mount point.
hps.name = '/datadrive/samples'
chunk_size = 16 if model in ('5b', '5b_lyrics') else 32
max_batch_size = 3 if model in ('5b', '5b_lyrics') else 16
hps.levels = 3
hps.hop_fraction = [.5,.5,.125]

vqvae, *priors = MODELS[model]
vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = 1048576)), device)
top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)



# Select mode
# Run one of these cells to select the desired mode.

# The default mode of operation.
# Creates songs based on artist and genre conditioning.
# mode = 'ancestral'
# codes_file=None
# audio_file=None
# prompt_length_in_seconds=None

# Prime song creation using an arbitrary audio sample.
mode = 'primed'
codes_file=None
# Specify an audio file here.
audio_file = '/datadrive/primer.wav'
# Specify how many seconds of audio to prime on.
prompt_length_in_seconds=12

# Run this cell to automatically resume from the latest checkpoint file, but only if the checkpoint file exists.
# This will override the selected mode.
# We will assume the existance of a checkpoint means generation is complete and it's time for upsamping to occur.
# if os.path.exists(hps.name):
#   # Identify the lowest level generated and continue from there.
#   for level in [1, 2]:
#     data = f"{hps.name}/level_{level}/data.pth.tar"
#     if os.path.isfile(data):
#       mode = 'upsample'
#       codes_file = data
#       print('Upsampling from level '+str(level))
#       break
# print('mode is now '+mode)


# Run the cell below regardless of which mode you chose.
sample_hps = Hyperparams(dict(mode=mode, codes_file=codes_file, audio_file=audio_file, prompt_length_in_seconds=prompt_length_in_seconds))


# Specify your choice of artist, genre, lyrics, and length of musical sample. 
# IMPORTANT: The sample length is crucial for how long your sample takes to generate. Generating a shorter sample takes less time. You are limited to 12 hours on the Google Colab free tier. A 50 second sample should be short enough to fully generate after 12 hours of processing. 

sample_length_in_seconds = 50          # Full length of musical sample to generate - we find songs in the 1 to 4 minute
                                       # range work well, with generation time proportional to sample length.  
                                       # This total length affects how quickly the model 
                                       # progresses through lyrics (model also generates differently
                                       # depending on if it thinks it's in the beginning, middle, or end of sample)
hps.sample_length = (int(sample_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
assert hps.sample_length >= top_prior.n_ctx*top_prior.raw_to_tokens, f'Please choose a larger sampling rate'

# Note: Metas can contain different prompts per sample.
# By default, all samples use the same prompt.
metas = [dict(artist = "regina spektor",
            genre = "pop-folk",
            total_length = hps.sample_length,
            offset = 0,
            lyrics = """This is just to confess
I have eaten the delicious plumbs
So sweet and so cold
I read of thousands dying
Then carried on with life
I’ll scoop less coffee grounds
Stretch the ritual thin
 I put my smartphone down
Watched the sunset
Thought to sacrifice a tear
For the lives lost
The wild garlic doesn’t know why I cry
The color white moves like an animal
Unaffected by the death of others
If I could bottle peace
I wouldn’t add salt
I would mail it to my grandmother
With a note saying
Use generously""",
            ),
          ] * hps.n_samples
labels = [None, None, top_prior.labeller.get_batch_labels(metas, 'cuda')]

# Generate 3 options for the start of the song
# Initial generation is set to be 4 seconds long, but feel free to change this

def seconds_to_tokens(sec, sr, prior, chunk_size):
  tokens = sec * hps.sr // prior.raw_to_tokens
  tokens = ((tokens // chunk_size) + 1) * chunk_size
  assert tokens <= prior.n_ctx, 'Choose a shorter generation length to stay within the top prior context'
  return tokens

initial_generation_in_seconds = 4
tokens_to_sample = seconds_to_tokens(initial_generation_in_seconds, hps.sr, top_prior, chunk_size)


# found this solution here to freeing up memory: https://github.com/pytorch/pytorch/issues/16417#issuecomment-669229108
import gc
gc.collect()



# Change the sampling temperature if you like (higher is more random).  Our favorite is in the range .98 to .995

sampling_temperature = .98

lower_batch_size = 16
max_batch_size = 3 if model in ('5b', '5b_lyrics') else 16
lower_level_chunk_size = 32
chunk_size = 16 if model in ('5b', '5b_lyrics') else 32
sampling_kwargs = [dict(temp=.99, fp16=True, max_batch_size=lower_batch_size,
                        chunk_size=lower_level_chunk_size),
                    dict(temp=0.99, fp16=True, max_batch_size=lower_batch_size,
                         chunk_size=lower_level_chunk_size),
                    dict(temp=sampling_temperature, fp16=True, 
                         max_batch_size=max_batch_size, chunk_size=chunk_size)]


# Now we're ready to sample from the model. We'll generate the top level (2) first, followed by the first upsampling (level 1), and the second upsampling (0). In this CoLab we load the top prior separately from the upsamplers, because of memory concerns on the hosted runtimes. If you are using a local machine, you can also load all models directly with make_models, and then use sample.py's ancestral_sampling to put this all in one step.
# After each level, we decode to raw audio and save the audio files.
# This next cell will take a while (approximately 10 minutes per 20 seconds of music sample)

if sample_hps.mode == 'ancestral':
  zs = [t.zeros(hps.n_samples,0,dtype=t.long, device='cuda') for _ in range(len(priors))]
  zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
elif sample_hps.mode == 'upsample':
  assert sample_hps.codes_file is not None
  # Load codes.
  data = t.load(sample_hps.codes_file, map_location='cpu')
  zs = [z.cuda() for z in data['zs']]
  assert zs[-1].shape[0] == hps.n_samples, f"Expected bs = {hps.n_samples}, got {zs[-1].shape[0]}"
  del data
  print('Falling through to the upsample step later in the notebook.')
elif sample_hps.mode == 'primed':
  assert sample_hps.audio_file is not None
  audio_files = sample_hps.audio_file.split(',')
  duration = (int(sample_hps.prompt_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
  x = load_prompts(audio_files, duration, hps)
  zs = top_prior.encode(x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0])
  zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
else:
  raise ValueError(f'Unknown sample mode {sample_hps.mode}.')
