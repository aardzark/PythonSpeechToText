from deepspeech import Model
import numpy as np
import wave

# robinhad/voice-recognition-ua
# pre-trained language model on 1230 hours of Ukrainian
UKRAINIAN_MODEL_ROBINHAD = "Neural Network Models/ukrainian_robinhad.pbmm"
# outside language model that assists with transcription
UKRAINIAN_SCORER_ROBINHAD = "Neural Network Models/kenlm_robinhad.scorer"

UKRAINIAN_AUDIO_SAMPLE = "Audio Sample/ukrainian_demo.wav"


def setup_deepspeech(model_path,ukrainian_config):
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
    print(transcribe_batch(ukrainian_model, UKRAINIAN_AUDIO_SAMPLE))
