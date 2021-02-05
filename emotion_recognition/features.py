import librosa
import librosa.display
import math
import numpy as np
import essentia
import essentia.streaming
from essentia.standard import *


def calculateAmpSignal(rawSignal, frameSize):
    ampSignal = np.zeros(shape = (int(frameSize/2),))
    fftSignal = np.fft.fft(rawSignal)
    
    for i in range(int(frameSize/2)):
        ampSignal[i] = math.sqrt(
                math.pow(fftSignal[i].real, 2) +
                math.pow(fftSignal[i].imag, 2)
                )
        
    return ampSignal


def MFCC(y,sr,window_size,n_mfcc):
    S = librosa.feature.melspectrogram(y=y, hop_length = window_size+1,n_mels = 512, sr=sr)
    log_S = librosa.amplitude_to_db(S, ref = np.max)
    mfcc = librosa.feature.mfcc(S=log_S,sr=sr, n_mfcc=n_mfcc)
    
    return mfcc

def delta_MFCC(mfcc):
    return librosa.feature.delta(mfcc)

def delta2_MFCC(mfcc):
    return librosa.feature.delta(mfcc, order=2)

def LPC(y,p,window_size):
    N = window_size-1
    m = p
    
    R =np.zeros(m+1, dtype=np.float64)
    Ak = np.zeros(m+1,dtype=np.float64)
    
    for i in range(m+1):
        for j in range(N-i+1):
            R[i] = R[i] + (y[j]*y[j+i])
    
    Ak[0] = 1.0
    Ek = R[0]
    
    for k in range(m):
        lamb = 0.0
        for j in range(k+1):
            lamb = lamb-Ak[j]*R[k+1-j]
        lamb = lamb/Ek
        
        for x in range(int((k+1)/2+1)):
            temp = Ak[k+1-x]+lamb*Ak[x]
            Ak[x] = Ak[x] + lamb*Ak[k+1-x]
            Ak[k+1-x]=temp
        
        Ek = Ek*(1.0-lamb*lamb)
    
    Ak = np.delete(Ak,[0])
    return Ak
        
def autocorrelation(p,length,y):
    C =0.0
    E1 =0; E2 =0
    
    for i in range(length -p):
        C = C + y[i] * y[i+p]
    
    return C

def nomalized_autocorrelation(p,length,y):
    C = 0.0
    E1 =0.0; E2 = 0.0
    
    for j in range(length-p+1):
        C = C+ y[j]*y[j+p]
    
    for j in range(length-p+1):
        E1 = E1 + y[j]*y[j]
        E2 = E2 + y[j+p]*y[j+p]
    
    C = C/math.sqrt(E1*E2)
    
    return C

def PITCH(y,sr,window_size):
    result = 0.0
    MAX = 0.0
    idx =0
    
    for j in range(int((sr/1000.0)*2),int((sr/1000.0)*20)):
        result = autocorrelation(j,window_size,y)
        if (abs(result) > MAX):
            MAX = result
            idx = j
    
    return sr/idx

def spectral_centroid(y_sub,sr,window_size):
    S, phase = librosa.magphase(librosa.stft(y=y_sub,n_fft=window_size,hop_length=window_size+1))
    centroid = librosa.feature.spectral_centroid(S=S,n_fft=window_size)
    
    return centroid[0][0]

def spectral_bandwidth(y_sub,sr,window_size):
    S, phase = librosa.magphase(librosa.stft(y=y_sub,n_fft=window_size,hop_length=window_size+1))
    bandwidth = librosa.feature.spectral_bandwidth(S=S)
    
    return bandwidth[0][0]

def spectral_contrast(y_sub,sr,window_size):
    S = np.abs(librosa.stft(y_sub,n_fft=window_size,hop_length=window_size+1))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    
    return contrast

def spectral_flatness(y_sub,sr,window_size):
    S, phase = librosa.magphase(librosa.stft(y=y_sub,n_fft=window_size,hop_length=window_size+1))
    flatness = librosa.feature.spectral_flatness(S=S)
    
    return flatness[0][0]

def spectral_rolloff(y_sub,sr,window_size):
    S, phase = librosa.magphase(librosa.stft(y=y_sub,n_fft=window_size,hop_length=window_size+1))
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    
    return rolloff[0][0]

def spectral_slope(window_y,sr,window_size):
    ampSignal = calculateAmpSignal(window_y, window_size)
    
    ampSum = 0
    freqSum = 0
    powFreqSum = 0
    ampFreqSum = 0
    freqs = np.zeros(shape = (len(ampSignal),))
    
    for i in range(len(ampSignal)):
        ampSum += ampSignal[i]
        curFreq = i * sr / window_size
        freqs[i] = curFreq
        powFreqSum += curFreq * curFreq;
        freqSum += curFreq;
        ampFreqSum += curFreq * ampSignal[i];
    
    return -(len(ampSignal) * ampFreqSum - freqSum * ampSum) / (ampSum * (powFreqSum - math.pow(freqSum, 2)))


def chroma(y_sub,sr,window_size):
    chroma = librosa.feature.chroma_stft(y=y_sub, sr=sr,n_fft=window_size,hop_length=window_size+1)
    
    return chroma


def inharmonicity(harfreq, harmag):
    #Module Initialization for Essentia
    inharmonicity = Inharmonicity()
    
    return inharmonicity(harfreq, harmag)

def tristimulus(harfreq, harmag):
    #Module Initialization for Essentia
    tristimulus = Tristimulus()
    
    tristimulus1_ = tristimulus(harfreq, harmag)[0]
    tristimulus2_ = tristimulus(harfreq, harmag)[1]
    tristimulus3_ = tristimulus(harfreq, harmag)[2]
    
    return tristimulus1_, tristimulus2_, tristimulus3_


def harEnergy(harfreq, harmag):
    harEnergySum = 0
    for i in range(len(harfreq)):
        harEnergySum += harmag[i] ** 2
    harEnergy = harEnergySum
    
    return harEnergy

def noiseEnergy(spec_y,har_energy):
    energy = Energy()
    totEnergy = energy(spec_y)
    
    return totEnergy-har_energy

def noiseness(spec_y,har_energy):
    energy = Energy()
    totEnergy = energy(spec_y)
    
    if totEnergy==0:
        return 0
    
    return (totEnergy-har_energy)/totEnergy

