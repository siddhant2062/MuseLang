# MuseLang

Python library for text-to-music generation and lyrics transcription (HeartMuLa + HeartCodec).

---

## 🧭 TODOs

- ⏳ Release scripts for inference acceleration and streaming inference. The current inference speed is around RTF $\approx 1.0$.
- ⏳ Support **reference audio conditioning**, **fine-grained controllable music generation**, **hot song generation**.
- ⏳ Release the **HeartMuLa-oss-7B** version.
- ✅ Release inference code and pretrained checkpoints of  
  **HeartCodec-oss, HeartMuLa-oss-3B, and HeartTranscriptor-oss**.

---

## 🛠️ Local Deployment

### ⚙️ Environment Setup

We recommend using `python=3.10` for local deployment.

Clone this repo and install locally.

```
git clone https://github.com/siddhant2062/MuseLang.git
cd MuseLang
pip install -e .
```

Download pretrained checkpoints from huggingface or modelscope using the following command:

```
# if you are using huggingface
hf download --local-dir './ckpt' 'HeartMuLa/HeartMuLaGen'

## To use version released on 20260123 (recommended)
hf download --local-dir './ckpt/HeartMuLa-oss-3B' 'HeartMuLa/HeartMuLa-RL-oss-3B-20260123'
hf download --local-dir './ckpt/HeartCodec-oss' HeartMuLa/HeartCodec-oss-20260123

## To use oss-3B version
hf download --local-dir './ckpt/HeartMuLa-oss-3B' 'HeartMuLa/HeartMuLa-oss-3B'
hf download --local-dir './ckpt/HeartCodec-oss' 'HeartMuLa/HeartCodec-oss'

# if you are using modelscope
modelscope download --model 'HeartMuLa/HeartMuLaGen' --local_dir './ckpt'

## To use version released on 20260123 (recommended)
modelscope download --model 'HeartMuLa/HeartMuLa-RL-oss-3B-20260123' --local_dir './ckpt/HeartMuLa-oss-3B'
modelscope download --model 'HeartMuLa/HeartCodec-oss-20260123' --local_dir './ckpt/HeartCodec-oss'

## To use oss-3B version
modelscope download --model 'HeartMuLa/HeartMuLa-oss-3B' --local_dir './ckpt/HeartMuLa-oss-3B'
modelscope download --model 'HeartMuLa/HeartCodec-oss' --local_dir './ckpt/HeartCodec-oss'
```

After downloading, the `./ckpt` subfolder should structure like this:
```
./ckpt/
├── HeartCodec-oss/
├── HeartMuLa-oss-3B/
├── gen_config.json
└── tokenizer.json
```

#### Workaround for Mac Users

This library is primarily built for CUDA (NVIDIA) devices. On Mac (including Apple Silicon) you can run without changes.

**Community solution (e.g. Riolutail):** A two-part approach is often suggested — (1) build Triton from source, and (2) change device targets from `cuda` to `cpu` or `mps` in `examples/run_music_generation.py` and `src/heartlib/heartcodec/modeling_heartcodec.py`. **This repo already implements part 2:** device is chosen automatically and passed through the pipeline, so you do **not** need to edit those files. HeartCodec uses `self.device` (set when the model is loaded), and the example script uses `--mula_device` / `--codec_device` (default: cuda → mps → cpu). No manual device edits or indentation changes are required.

- **Out of the box:** This repo auto-selects device: **cuda** (if available) → **mps** (Apple Silicon) → **cpu**. A Triton stub is included so dependencies that optionally import `triton` do not fail; you do **not** need to install or build Triton to run on Mac.
- **Optional – build Triton from source:** Some community members try building Triton from source on Mac (requires cmake and can take several minutes):

  ```bash
  git clone https://github.com/triton-lang/triton.git
  cd triton
  pip install -r python/requirements.txt
  pip install -e .
  ```

  **Builds often fail on Mac** (e.g. LLVM download timeout, or CMake error: `llvm-tblgen` not found because the LLVM tarball is incomplete). If that happens, **use the no-build setup above**: device auto-select + built-in Triton stub. You do not need real Triton to run on Mac. If you retry the build, use a stable network and clear the cache first: `rm -rf ~/.triton/llvm` then run `pip install -e .` again.

  To force **cpu** or **mps** explicitly, use:

  ```bash
  python ./examples/run_music_generation.py --model_path=./ckpt --version="3B" --mula_device cpu --codec_device cpu
  ```

  or `--mula_device mps --codec_device mps` for Apple Silicon GPU.

### ▶️ Example Usage

To generate music, run:

```
python ./examples/run_music_generation.py --model_path=./ckpt --version="3B"
```

By default this command will generate a piece of music conditioned on lyrics and tags provided in `./assets` folder. The output music will be saved at `./assets/output.mp3`.

#### FAQs

1. How to specify lyrics and tags?

    The model will load lyrics from the txt file `--lyrics` link to (by default `./assets/lyrics.txt`). If you would like to use your own lyrics, just modify the content in `./assets/lyrics.txt`. If you would like to save your lyrics to another path, e.g. `my_awesome_lyrics.txt`, remember to input arguments `--lyrics my_awesome_lyrics.txt`.

    For tags it's basically the same.

2. CUDA out of memory?

    If you have multi-GPUs (e.g. 2 4090s), we recommend placing the params of HeartMuLa and HeartCodec separately on different devices. You can do it by typing `--mula_device cuda:0 --codec_device cuda:1`

    If you are running on a single GPU, use `--lazy_load true` so that modules will be loaded on demand and deleted once inference completed to save GPU memory.

All parameters:

- `--model_path` (required): Path to the pretrained model checkpoint
- `--lyrics`: Path to lyrics file (default: `./assets/lyrics.txt`)
- `--tags`: Path to tags file (default: `./assets/tags.txt`)
- `--save_path`: Output audio file path (default: `./assets/output.mp3`)
- `--max_audio_length_ms`: Maximum audio length in milliseconds (default: 240000)
- `--topk`: Top-k sampling parameter for generation (default: 50)
- `--temperature`: Sampling temperature for generation (default: 1.0)
- `--cfg_scale`: Classifier-free guidance scale (default: 1.5)
- `--version`: The version of HeartMuLa, choose between [`3B`, `7B`]. (default: `3B`) # `7B` version not released yet.
- `--mula_device/--codec_device`: The device where params will be placed. Default: `cuda` if available, else `mps` (Apple Silicon), else `cpu`. You can use `--mula_device cuda:0 --codec_device cuda:1` to place modules on different devices.
- `--mula_dtype/--codec_dtype`: Inference dtype. By default is `bf16` for HeartMuLa and `fp32` for HeartCodec. Setting `bf16` for HeartCodec may result in the degradation of audio quality.
- `--lazy_load`: Whether or not to use lazy loading (default: false). If turned on, modules will be loaded on demand to save GPU usage. 
Recommended format of lyrics and tags:
```txt
[Intro]

[Verse]
The sun creeps in across the floor
I hear the traffic outside the door
The coffee pot begins to hiss
It is another morning just like this

[Prechorus]
The world keeps spinning round and round
Feet are planted on the ground
I find my rhythm in the sound

[Chorus]
Every day the light returns
Every day the fire burns
We keep on walking down this street
Moving to the same steady beat
It is the ordinary magic that we meet

[Verse]
The hours tick deeply into noon
Chasing shadows,chasing the moon
Work is done and the lights go low
Watching the city start to glow

[Bridge]
It is not always easy,not always bright
Sometimes we wrestle with the night
But we make it to the morning light

[Chorus]
Every day the light returns
Every day the fire burns
We keep on walking down this street
Moving to the same steady beat

[Outro]
Just another day
Every single day
```

Tags are comma-separated without spaces as illustrated below:
```txt
piano,happy,wedding,synthesizer,romantic
```

