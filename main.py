import logging
import sys

import deepspeech
from deepspeech import Model
from tabulate import tabulate
import numpy as np
import wave
import whisper
from googletrans import Translator
import os

# robinhad/voice-recognition-ua
# pretrained language model
UKRAINIAN_MODEL_ROBINHAD: str = "Neural Network Models/Ukrainian/DeepSpeech/robinhad_voice-recognition-ua-v-0-04/uk.pbmm"
# external language model that assists with transcription
UKRAINIAN_SCORER_ROBINHAD: str = "Neural Network Models/Ukrainian/DeepSpeech/robinhad_voice-recognition-ua-v-0-04/kenlm.scorer"

# github.com/egorsmkv/speech-recognition-uk/tree/master/tts-demos
UKRAINIAN_AUDIO_SAMPLE_NEON: str = "Audio Demo/audio_submission_ylqyya.mp3.wav"
UKRAINIAN_AUDIO_SAMPLE_SILERO: str = "Audio Demo/tts-demos_silero_tts.wav"
EXPECTED_OUTPUT_UKRAINIAN_AUDIO_SAMPLE: str = """Кам'янець-Подільський - місто в Хмельницькій області України, центр Кам'янець-Подільської
міської об'єднаної територіальної громади і Кам'янець-Подільського району."""


class File:
    rate: int = None
    frames: int = None
    buffer: bytes = None
    name: str = None

    def __init__(self, file_location: str):
        with wave.open(file_location, 'rb') as wave_read:
            self.rate = wave_read.getframerate()
            self.frames = wave_read.getnframes()
            self.buffer = wave_read.readframes(self.frames)

            if file_location.find('/') != -1:
                self.name = file_location.rsplit('/', 1)[1]
            else:
                self.name = file_location


class STTModel:
    file: File = None
    model = None
    engine: str = None
    model_name: str = None
    file_name: str = None
    language: str = None
    actual_stt: str = None
    expected_stt: str = None
    tabular_data: [str] = None

    def __init__(self, file: File, model, engine: str, model_name: str, file_name: str, language: str,
                 actual_stt: str, expected_stt: str):
        self.file = file
        self.model = model
        self.engine = engine
        self.model_name = model_name
        self.file_name = file_name
        self.language = language
        self.actual_stt = actual_stt
        self.expected_stt = expected_stt
        self.tabular_data = [self.engine, self.model_name, self.file_name, self.language,
                             self.actual_stt, self.expected_stt]


def configure_deepspeech_model(model_path: str, config: dict):
    # value listed on release page for model
    beam_width = int(config.get("beam_width"))
    lm_alpha = float(config.get("lm_alpha"))
    lm_beta = float(config.get("lm_beta"))

    deepspeech_model = Model(model_path)
    deepspeech_model.enableExternalScorer(UKRAINIAN_SCORER_ROBINHAD)
    deepspeech_model.setScorerAlphaBeta(lm_alpha, lm_beta)
    deepspeech_model.setBeamWidth(beam_width)
    return deepspeech_model


def configure_openai_model(model_type: str):
    openai_model = whisper.load_model(model_type)
    return openai_model


class Model:
    model_engine: str = None
    pretrained_model_location: str = None
    scorer_location: str = None
    config: dict = None

    def __init__(self, model_type: str, config: dict, pretrained_model_location: str = None,
                 scorer_location: str = None):
        self.model_engine = model_type
        self.config = config
        self.pretrained_model_location = pretrained_model_location
        self.scorer_location = scorer_location


def retrieve_pretrained_model(model: Model):
    if model.model_engine == 'DeepSpeech':
        lm_alpha: float = float(model.config.get("lm_alpha"))
        lm_beta: float = float(model.config.get("lm_beta"))
        beam_width: int = int(model.config.get("beam_width"))

        pretrained_model = deepspeech.Model(model.pretrained_model_location)
        pretrained_model.enableExternalScorer(model.scorer_location)
        pretrained_model.setScorerAlphaBeta(lm_alpha, lm_beta)
        pretrained_model.setBeamWidth(beam_width)

        return pretrained_model
    if model.model_engine == "OpenAI":
        pretrained_model = whisper.load_model(model.config.get("quality"))

        return pretrained_model


def transcribe_batch(model, file):
    data16 = np.frombuffer(file.buffer, dtype=np.int16)
    stt = ''
    if isinstance(model, deepspeech.Model):
        # stt = model.stt(data16)
        pass
    if isinstance(model, whisper.Whisper):
        result = model.transcribe("Audio/Demo/" + file.name)
        stt = result["text"]
        return stt

    return stt


def run_benchmarks():
    files: [File] = []


def get_file_locations(directory: str):
    file_paths: [str] = []
    for file in os.scandir(directory):
        if file.is_file():
            file_paths.append(directory + "/" + file.name)

    return file_paths


def main(configs: [dict], headers: [dict]):
    DEEPSPEECH_MODEL: str = "Neural Network Models/Ukrainian/DeepSpeech/robinhad_voice-recognition-ua-v-0-04/uk.pbmm"
    DEEPSPEECH_SCORER: str = "Neural Network Models/Ukrainian/DeepSpeech/robinhad_voice-recognition-ua-v-0-04/kenlm.scorer"
    CONFIGS: [] = []
    CONFIGS.append({"engine": "DeepSpeech",
                    "beam_width": "100",
                    "lm_alpha": "0.7200873732640549",
                    "lm_beta": "1.3010463457623596"})
    CONFIGS.append({"engine": "OpenAI",
                    "quality": "base"})
    models: [] = None
    demo_files = get_file_locations("Audio/Demo")
    live_files = get_file_locations("Audio/Live")
    deepspeech_model = Model(CONFIGS[0].get("engine"), configs[0], DEEPSPEECH_MODEL, DEEPSPEECH_SCORER)
    openai_model = Model(CONFIGS[1].get("engine"), configs[1])

    for config in CONFIGS:
        if config.get("engine") == 'DeepSpeech':
            model = Model('DeepSpeech', config, DEEPSPEECH_MODEL, DEEPSPEECH_SCORER)
            pretrained_model = retrieve_pretrained_model(model)
            for non_cvt_file in demo_files:
                file = File(non_cvt_file)
                transcribe_batch(pretrained_model, file)
        if config.get("engine") == 'OpenAI':
            model = Model('OpenAI', config)
            pretrained_model = retrieve_pretrained_model(model)
            for non_cvt_file in demo_files:
                file = File(non_cvt_file)
                transcribe_batch(pretrained_model, file)

            #stt_model = STTModel(file, model, config.get('engine'), config.get('model'), file.name,
                                 #config.get('name'),
                                 #'', transcribe_batch(pretrained_model, file))
            #print(tabulate([stt_model.tabular_data],
                           #headers=['Engine', 'Model', 'File', 'Language', 'Actual Speech-to-Text',
                                    #'Expected Speech-to-Text'], tablefmt="fancy_grid"))

    # models: [] = [Model('DeepSpeech', DEEPSPEECH_MODEL, DEEPSPEECH_SCORER), Model('OpenAI')]

    # deepspeech_model = retrieve_pretrained_model(Model("OpenAI", ))

    # models.append()

    # models: [] = [configure_deepspeech_model(UKRAINIAN_MODEL_ROBINHAD, configs[0]),
    # configure_openai_model(headers[1].get("Model"))]

    # translator = Translator(service_urls=['translate.googleapis.com'])

    # i = 0

    # for model in models:
    # for file in files:
    # transcribed = transcribe_batch(model, file)
    # print(translator.detect(transcribed))
    # translated = translator.translate(transcribed, lang_tgt="en").text
    # print(translated)
    # stt_model = STTModel(file, model, headers[i].get("Engine"), headers[i].get("Model"), file.name,
    # headers[i].get("Language"),
    # transcribed, EXPECTED_OUTPUT_UKRAINIAN_AUDIO_SAMPLE)
    # print(tabulate([stt_model.tabular_data],
    # headers=['Engine', 'Model', 'File', 'Language', 'Actual Speech-to-Text',
    # 'Expected Speech-to-Text'], tablefmt="fancy_grid"))
    # i = i + 1

    # print(tabulate(tabular_data_array, headers=['Engine', 'Model', 'File', 'Language', 'Actual Speech-to-Text',
    # 'Expected Speech-to-Text'], tablefmt="fancy_grid"))

    return 0


if __name__ == '__main__':
    stt_configs: [dict] = []
    voice_recognition_ua_v_0_04_config = {
        "beam_width": "100",
        "lm_alpha": "0.7200873732640549",
        "lm_beta": "1.3010463457623596"
    }
    voice_recognition_openai_config = {
        "detect_language": "false"
    }
    stt_headers: [dict] = []
    voice_recognition_ua_v_0_04_headers = {
        'Engine': "DeepSpeech",
        'Model': 'robinhad v0.4',
        'Language': 'Ukrainian',
    }
    voice_recognition_openai_headers = {
        'Engine': "OpenAI",
        'Model': 'medium',
        'Language': 'Various',
    }
    stt_configs.append(voice_recognition_ua_v_0_04_config)
    stt_configs.append(voice_recognition_openai_config)
    stt_headers.append(voice_recognition_ua_v_0_04_headers)
    stt_headers.append(voice_recognition_openai_headers)

    exit(main(stt_configs, stt_headers))
