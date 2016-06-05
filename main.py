import wave
import numpy as np
from scipy.fftpack import fft
import math as m
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

types = {
    1: np.int8,
    2: np.int16,
    3: np.int32
}

T = 0.15 # 150 ms
SampleRate = 44100
N = int(SampleRate * T)


def hemming(i):
    return 0.53836 - 0.46164 * m.cos((6.28 * i) / (N - 1))


def fourier_freq(i, T):
    return int(i / T)


mel_num = 100 # number of intervals
mel_delta = 25 # size of interval in mels


def get_mel_window_value(mel, i):
    mel_dif = mel - i * mel_delta

    value = 2 - 4 * abs((mel_dif - mel_delta / 2) / mel_delta)

    return value


def get_mel_energy(spectre):
    mel_energy = []

    for i in range(0, mel_num):
        energy = 0.0
        for j in range(0, len(spectre)):
            mel = 1125 * m.log(1 + float(fourier_freq(j, T)) / 700)

            if mel <= i * mel_delta:
                continue

            if mel >= (i + 1) * mel_delta:
                break

            energy = energy + pow(spectre[j] * get_mel_window_value(mel, i), 2)

        mel_energy.append(energy)

    return mel_energy


def make_vectors_for_wav(frames, person):
    vect = []
    for frame in frames:
        support = []
        x = np.array(frame)

        fft_arr = np.absolute(fft(x))
        fft_arr /= np.linalg.norm(fft_arr)

        energy = get_mel_energy(fft_arr)

        support.append(energy)
        support.append(person)
        vect.append(support)

    return vect


def add_wav_to_db(wav_name, person, sss):
    wav = wave.open(wav_name, mode="r")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
    content = wav.readframes(nframes)
    samples = np.fromstring(content, dtype=types[sampwidth])

    i = 0
    frames = []
    while i < len(samples) - N:
        ar = []

        for j in xrange(i, i + N):
            ar.append(float(samples[j]) * hemming(j - i))
        fr = np.array(ar)
        frames.append(fr)
        i += N / 2

    sss.append(make_vectors_for_wav(frames, person))


def predict_wav(name, cls, cls2):
    wav = wave.open(name, mode="r")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
    content = wav.readframes(nframes)
    print len(content)
    samples = np.fromstring(content, dtype=types[sampwidth])

    i = 0
    frames = []
    while i < len(samples) - N:
        ar = []

        for t in samples[i: i + N]:
            ar.append(float(t) * hemming(i))

        fr = np.array(ar)
        frames.append(fr)
        i += N / 2

    ff = make_vectors_for_wav(frames, i)
    a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for f in ff:
        res = cls.predict_proba(f[0])
        res2 = cls2.predict_proba(f[0])

        ans = np.argmax(res[0], axis=0)
        ans2 = np.argmax(res2[0], axis=0)
        #ans = cls.predict(f[0])
        #ans2 = cls2.predict(f[0])

        if ans != ans2:
            continue

        a[ans2] += max(res[0]) * np.linalg.norm(f[0])

    return a


def validate(cls, cls2):
    test_list = ["SashaTest.wav", "TaniaTest.wav", "ZheniaTest.wav", "Zhenia2Test.wav", "Tania2Test.wav", "DashaTest.wav", "Sasha2Test.wav", "VikaTest.wav", "MaxTest.wav", "PashaTest.wav"]

    indexes = []
    b = []
    for wav in test_list:
        a = predict_wav(wav, cls, cls2)
        indexes.append(a.index(max(a)))
        b.append(float(max(a)) / float(sum(a)))
    print "indexes: " + str(indexes) + " ? " + str(indexes == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print b


def trainModel():
    sss = []
    train_list = [["SashaTrain.wav", 0], ["TaniaTrain.wav", 1], ["ZheniaTrain.wav", 2], ["Zhenia2Train.wav", 3], ["Tania2Train.wav", 4], ["DashaTrain.wav", 5], ["Sasha2Train.wav", 6], ["VikaTrain.wav", 7], ["MaxTrain.wav", 8], ["PashaTrain.wav", 9]]

    for wav_name in train_list:
        add_wav_to_db(wav_name[0], wav_name[1], sss)

    data = []
    ans = []
    i = 0
    for index in xrange(len(sss)):
        for v in sss[index]:
            data.append(v[0])
            ans.append(v[1])

    clfNeural = MLPClassifier()
    clfNeural.fit(data, ans)

    clfForest = DecisionTreeClassifier(max_depth=250)
    clfForest.fit(data, ans)

    joblib.dump(clfNeural, 'model.pkl')
    joblib.dump(clfForest, 'forest.pkl')


def loadAndTest():
    clfNeural = joblib.load('model.pkl')
    clfForest = joblib.load('forest.pkl')
    validate(clfNeural, clfForest)


def train():
    trainModel()

import time

startTime = time.time()

train()
loadAndTest()

endTime = time.time()

print "execution time: " + str(endTime - startTime)