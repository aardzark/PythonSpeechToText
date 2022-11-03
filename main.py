import logging
import sys

from deepspeech import Model
from tabulate import tabulate
import numpy as np
import wave

# robinhad/voice-recognition-ua
# pre-trained language model on 1230 hours of Ukrainian
UKRAINIAN_MODEL_ROBINHAD: str = "Neural Network Models/Ukrainian/DeepSpeech/robinhad_voice-recognition-ua-v-0-04/uk.pbmm"
# outside language model that assists with transcription
UKRAINIAN_SCORER_ROBINHAD: str = "Neural Network Models/Ukrainian/DeepSpeech/robinhad_voice-recognition-ua-v-0-04/kenlm.scorer"

# github.com/egorsmkv/speech-recognition-uk/tree/master/tts-demos
UKRAINIAN_AUDIO_SAMPLE_NEON: str = "Audio Sample/tts-demos_neon_tts.wav"
UKRAINIAN_AUDIO_SAMPLE_SILERO: str = "Audio Sample/tts-demos_silero_tts.wav"
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
    model: Model = None
    engine: str = None
    model_name: str = None
    file_name: str = None
    language: str = None
    actual_stt: str = None
    expected_stt: str = None
    tabular_data: [str] = None

    def __init__(self, file: File, model: Model, engine: str, model_name: str, file_name: str, language: str,
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


def setup_deepspeech(model_path: str, config: dict):
    # value listed on release page for model
    beam_width = int(config.get("beam_width"))
    lm_alpha = float(config.get("lm_alpha"))
    lm_beta = float(config.get("lm_beta"))

    model = Model(model_path)
    model.enableExternalScorer(UKRAINIAN_SCORER_ROBINHAD)
    model.setScorerAlphaBeta(lm_alpha, lm_beta)
    model.setBeamWidth(beam_width)
    return model


def transcribe_batch(model, file):
    data16 = np.frombuffer(file.buffer, dtype=np.int16)
    return model.stt(data16)


def main(configs: [dict], headers: [dict]):
    try:
        files: [File] = [File(UKRAINIAN_AUDIO_SAMPLE_NEON), File(UKRAINIAN_AUDIO_SAMPLE_SILERO)]
        models: [Model] = [setup_deepspeech(UKRAINIAN_MODEL_ROBINHAD, configs[0])]
        tabular_data_array: [[str]] = []

        for model in models:
            for file in files:
                stt_model = STTModel(file, model, headers[0].get("Engine"), headers[0].get("Model"), file.name,
                                     headers[0].get("Language"),
                                     transcribe_batch(model, file), EXPECTED_OUTPUT_UKRAINIAN_AUDIO_SAMPLE)
                tabular_data_array.append(stt_model.tabular_data)

        print(tabulate(tabular_data_array, headers=['Engine', 'Model', 'File', 'Language', 'Actual Speech-to-Text',
                                                    'Expected Speech-to-Text'], tablefmt="fancy_grid"))

        return 0
    except:
        logging.ERROR('An error occured.')
        return 1


if __name__ == '__main__':
    stt_configs: [dict] = []
    voice_recognition_ua_v_0_04_config = {
        "beam_width": "100",
        "lm_alpha": "0.7200873732640549",
        "lm_beta": "1.3010463457623596"
    }
    stt_headers: [dict] = []
    voice_recognition_ua_v_0_04_headers = {
        'Engine': "DeepSpeech",
        'Model': 'robinhad v0.4',
        'Language': 'Ukrainian',
    }
    stt_configs.append(voice_recognition_ua_v_0_04_config)
    stt_headers.append(voice_recognition_ua_v_0_04_headers)

    exit(main(stt_configs, stt_headers))
