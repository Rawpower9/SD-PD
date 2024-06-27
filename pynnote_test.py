from pyannote.audio import Pipeline 
from sd_pyannote import SD_Modified
# from pyannote.audio.pipelines import SpeakerDiarization
import time
from typing import Union,Text
from pathlib import Path
from pyannote.audio.core.model import CACHE_DIR
from huggingface_hub import hf_hub_download
import os
from huggingface_hub.utils import RepositoryNotFoundError
# from pyannote.core.utils.helper import get_class_by_name
from pyannote.audio.utils.version import check_version
import yaml
import torch
import numpy as np
from dotenv import load_dotenv
# from pyannote.database import FileFinder

PIPELINE_PARAMS_NAME = "config.yaml"
__version__ = "3.2.0"


def from_pretrained(
    checkpoint_path: Union[Text, Path],
    hparams_file: Union[Text, Path] = None,
    use_auth_token: Union[Text, None] = None,
    cache_dir: Union[Path, Text] = CACHE_DIR,
) -> "Pipeline":
    """Load pretrained pipeline

    Parameters
    ----------
    checkpoint_path : Path or str
        Path to pipeline checkpoint, or a remote URL,
        or a pipeline identifier from the huggingface.co model hub.
    hparams_file: Path or str, optional
    use_auth_token : str, optional
        When loading a private huggingface.co pipeline, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`
    cache_dir: Path or str, optional
        Path to model cache directory. Defauorch/pyannote" when unset.
    """

    checkpoint_path = str(checkpoint_path)

    if os.path.isfile(checkpoint_path):
        config_yml = checkpoint_path

    else:
        if "@" in checkpoint_path:
            model_id = checkpoint_path.split("@")[0]
            revision = checkpoint_path.split("@")[1]
        else:
            model_id = checkpoint_path
            revision = None

        try:
            config_yml = hf_hub_download(
                model_id,
                PIPELINE_PARAMS_NAME,
                repo_type="model",
                revision=revision,
                library_name="pyannote",
                library_version=__version__,
                cache_dir=cache_dir,
                # force_download=False,
                # proxies=None,
                # etag_timeout=10,
                # resume_download=False,
                use_auth_token=use_auth_token,
                # local_files_only=False,
                # legacy_cache_layout=False,
            )

        except RepositoryNotFoundError:
            print(
                f"""
Could not download '{model_id}' pipeline.
It might be because the pipeline is private or gated so make
sure to authenticate. Visit https://hf.co/settings/tokens to
create your access token and retry with:

>>> Pipeline.from_pretrained('{model_id}',
...                          use_auth_token=YOUR_AUTH_TOKEN)

If this still does not work, it might be because the pipeline is gated:
visit https://hf.co/{model_id} to accept the user conditions."""
            )
            return None

    with open(config_yml, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    if "version" in config:
        check_version(
            "pyannote.audio", config["version"], __version__, what="Pipeline"
        )
    params = config["pipeline"].get("params", {})
    params.setdefault("use_auth_token", use_auth_token)
    pipeline = SD_Modified(**params)

    # freeze  parameters
    if "freeze" in config:
        print("freeze in config")
        params = config["freeze"]
        pipeline.freeze(params)

    if "params" in config:
        pipeline.instantiate(config["params"])

    if hparams_file is not None:
        pipeline.load_params(hparams_file)

    # if "preprocessors" in config:
    #     print("proprocessors are in config :()")
    #     preprocessors = {}
    #     for key, preprocessor in config.get("preprocessors", {}).items():
    #         # preprocessors:
    #         #    key:
    #         #       name: package.module.ClassName
    #         #       params:
    #         #          param1: value1
    #         #          param2: value2
    #         if isinstance(preprocessor, dict):
    #             params = preprocessor.get("params", {})
    #             preprocessors[key] = SpeakerDiarization(**params)
    #             continue

    #         try:
    #             # preprocessors:
    #             #    key: /path/to/database.yml
    #             preprocessors[key] = FileFinder(database_yml=preprocessor)

    #         except FileNotFoundError:
    #             # preprocessors:
    #             #    key: /path/to/{uri}.wav
    #             template = preprocessor
    #             preprocessors[key] = template

    #     pipeline.preprocessors = preprocessors

    # send pipeline to specified device
    if "device" in config:
        device = torch.device(config["device"])
        try:
            pipeline.to(device)
        except RuntimeError as e:
            print(e)

    return pipeline

load_dotenv()
AUDIO_FILE_NAME = "10second-gap"
CENTROID_FILE_NAME = "jane_doe.txt"

if AUDIO_FILE_NAME.endswith(".wav"):
    AUDIO_FILE_NAME = AUDIO_FILE_NAME.removesuffix(".wav")

def hook(step_name: str, step_artefact=None, file=None, completed=None, total=None):
    print(step_name, step_artefact, file, completed, total)


pipeline = from_pretrained(
    checkpoint_path="pyannote/speaker-diarization-3.1",
    use_auth_token=os.getenv("HF_TOKEN"),
)

pipeline.set_file_name(CENTROID_FILE_NAME)

start_time = time.time()

# diarization = pipeline(
#     os.path.join("audio", AUDIO_FILE_NAME + ".wav"),
#     hook=hook,
# )


pipeline.initialize()
waveform, sample_rate = pipeline._segmentation.model.audio(
    os.path.join("audio", AUDIO_FILE_NAME + ".wav")
)
pipeline.add_audio(waveform)

end_time = time.time()

# print("duration", end_time - start_time)
# with open(os.path.join("transcript",AUDIO_FILE_NAME+".txt"), "w") as f:
#     diarization.write_rttm(f)

# 12,589,3,6988/21_204,
# 3525161 -> 220 second -> 212 segments -> 1.038 ratio
# 332510 -> 20 second -> 12 segments -> 1.667 ratio
# 6988,106559
# num_chunks= 11 has_last_chunk= True window_size= 160000 num_samples= 332510
# num_chunks= 211 has_last_chunk= True window_size= 160000 num_samples= 3525161,16_628.1179245283

# num_chunks= 20 has_last_chunk= True window_size= 160000 num_samples= 476160 -> 29 seconds
# num_chunks= 4 has_last_chunk= True window_size= 160000 num_samples= 209238 -> 13 seconds
# num_chunks= 19 has_last_chunk= True window_size= 160000 num_samples= 449536 -> 28 seconds
# num_chunks= 1 has_last_chunk= True window_size= 160000 num_samples= 165547 -> 10 seconds
