import logging
import sys
from configparser import ConfigParser
import deepspeech
from deepspeech import Model
from tabulate import tabulate
import numpy as np
import wave
import whisper
from googletrans import Translator
import os

configs = ConfigParser()
configs.read("config.ini")

class File:
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
    # deepspeech_model.enableExternalScorer(UKRAINIAN_SCORER_ROBINHAD)
    deepspeech_model.setScorerAlphaBeta(lm_alpha, lm_beta)
    deepspeech_model.setBeamWidth(beam_width)
    return deepspeech_model


def configure_openai_model(model_type: str):
    openai_model = whisper.load_model(model_type)
    return openai_model


class Model:
    def __init__(self, model_engine: str, config: dict, pretrained_model_location: str = None,
                 scorer_location: str = None):
        self.model_engine = model_engine
        self.config = config
        self.pretrained_model_location = pretrained_model_location
        self.scorer_location = scorer_location


def retrieve_pretrained_model(model: Model):
    print(model.config)
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
        return model.stt(data16)

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


def main():
    models: [] = []

    demo_files = get_file_locations("Audio/Demo")
    DEMO_EXPECTED_RESULT = """Кам'янець-Подільський - місто в Хмельницькій області України, цeентр 
    Кам'янець-Подільської міської об'єднаної територіальної громади і Кам'янець-Подільського району."""

    live_files = get_file_locations("Audio/Live")

    translator = Translator(service_urls=['translate.googleapis.com'])

    logging.debug('RUNNING DEMO')

    engine_config: [] = []
    deepspeech_model: Model = None
    openai_model: Model = None

    for section in configs.sections():

        if section == "ENGINE_DEEPSPEECH":
            deepspeech_model = Model("DeepSpeech", configs[section], configs[section].get("model_path"), configs[section].get("scorer_path"))
            models.append(deepspeech_model)
            pretrained_model = retrieve_pretrained_model(deepspeech_model)
            deepspeech_config: dict = {}

            for (key, val) in configs.items(section):
                deepspeech_config[key] = val
            engine_config.append(deepspeech_config)

            for non_cvt_file in demo_files:
                file = File(non_cvt_file)
                transcribed_text = transcribe_batch(pretrained_model, file)
                translated_text = translator.translate(transcribed_text, lang_tgt="en").text
                stt_model = STTModel(file, deepspeech_model, deepspeech_config.get('name'), 'N/A',
                                     file.name,
                                     translated_text, transcribed_text, DEMO_EXPECTED_RESULT)
                print(tabulate([stt_model.tabular_data],
                               headers=['Engine', 'Model', 'File', 'Translated Speech', 'Original Speech',
                                        'Expected Speech-to-Text'], tablefmt="fancy_grid"))

        if section == "ENGINE_OPENAI":
            openai_model = Model("OpenAI", configs[section])
            models.append(deepspeech_model)

            openai_config: dict = {}
            for (key, val) in configs.items(section):
                openai_config[key] = val

            engine_config.append(openai_config)

            pretrained_model = retrieve_pretrained_model(openai_model)
            openai_config['model'] = 'Whisper'

            for non_cvt_file in demo_files:
                file = File(non_cvt_file)
                transcribed_text = transcribe_batch(pretrained_model, file)
                translated_text = translator.translate(transcribed_text, lang_tgt="en").text

                stt_model = STTModel(file, openai_model, openai_config.get('name'), openai_config.get('quality'), file.name,
                                     translated_text, transcribed_text, DEMO_EXPECTED_RESULT)
                print(tabulate([stt_model.tabular_data],
                               headers=['Engine', 'Model', 'File', 'Translated Speech', 'Original Speech',
                                        'Expected Speech-to-Text'], tablefmt="fancy_grid"))

    return 0


if __name__ == '__main__':
    exit(main())
