from deepspeech import Model
from tabulate import tabulate
import numpy as np
import wave

# robinhad/voice-recognition-ua
# pre-trained language model on 1230 hours of Ukrainian
UKRAINIAN_MODEL_ROBINHAD = "Neural Network Models/Ukrainian/DeepSpeech/robinhad_voice-recognition-ua-v-0-04/uk.pbmm"
# outside language model that assists with transcription
UKRAINIAN_SCORER_ROBINHAD = "Neural Network Models/Ukrainian/DeepSpeech/robinhad_voice-recognition-ua-v-0-04/kenlm.scorer"

# github.com/egorsmkv/speech-recognition-uk/tree/master/tts-demos
UKRAINIAN_AUDIO_SAMPLE_NEON = "Audio Sample/tts-demos_neon_tts.wav"
UKRAINIAN_AUDIO_SAMPLE_SILERO = "Audio Sample/tts-demos_silero_tts.wav"
EXPECTED_OUTPUT_UKRAINIAN_AUDIO_SAMPLE = """Кам'янець-Подільський - місто в Хмельницькій області України, центр Кам'янець-Подільської
міської об'єднаної територіальної громади і Кам'янець-Подільського району."""


def setup_deepspeech(model_path, ukrainian_config):
    beam_width = 100

    # value listed on release page for model
    lm_alpha = float(ukrainian_config.get("lm_alpha"))
    lm_beta = float(ukrainian_config.get("lm_beta"))

    model = Model(model_path)
    model.enableExternalScorer(UKRAINIAN_SCORER_ROBINHAD)
    model.setScorerAlphaBeta(lm_alpha, lm_beta)
    model.setBeamWidth(beam_width)
    return model


def read_wav_file(filename):
    with wave.open(filename, 'rb') as w:
        rate = w.getframerate()
        frames = w.getnframes()
        buffer = w.readframes(frames)
        print('Rate:', rate)
        print('Frames:', frames)
        print('Buffer Len:', len(buffer))

    return buffer, rate


def transcribe_batch(model, audio_file):
    buffer, rate = read_wav_file(audio_file)
    data16 = np.frombuffer(buffer, dtype=np.int16)
    return model.stt(data16)


if __name__ == '__main__':
    ukrainian_config = {
        "lm_alpha": "0.7200873732640549",
        "lm_beta": "1.3010463457623596"
    }

    ukrainian_model = setup_deepspeech(UKRAINIAN_MODEL_ROBINHAD, ukrainian_config)
    tabular_data = [['DeepSpeech', 'robinhad v0.4', 'tts-demos_neon_tts.wav', 'Ukrainian',
                     transcribe_batch(ukrainian_model, UKRAINIAN_AUDIO_SAMPLE_NEON),
                     EXPECTED_OUTPUT_UKRAINIAN_AUDIO_SAMPLE],
                    ['DeepSpeech', 'robinhad v0.4', 'tts-demos_silero_tts.wav', 'Ukrainian',
                     transcribe_batch(ukrainian_model, UKRAINIAN_AUDIO_SAMPLE_SILERO),
                     EXPECTED_OUTPUT_UKRAINIAN_AUDIO_SAMPLE]]

    print(tabulate(tabular_data,
                   headers=['Engine', 'Model', 'File', 'Language', 'Actual Speech-to-Text', 'Expected Speech-to-Text'],
                   tablefmt="fancy_grid"))
