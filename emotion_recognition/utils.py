import wave
import struct
import numpy as np

def loadwav(filename):
    waveFile = wave.open(filename,'rb')
    nchannel = waveFile.getnchannels()
    length = waveFile.getnframes()
    nBytes = nchannel * length
    format = "<" +str(nBytes) + "h"

    waveData = waveFile.readframes(length)
    data_a = struct.unpack(format,waveData)

    waveFile.close()

    wave_channel1 =list(data_a)
    y = np.array(wave_channel1,np.float32)
    sr = waveFile.getframerate()
    
    return y,sr

def IAV(data,frame_size):
    value=0.0;

    for i in range (frame_size):
        value += abs(data[i])
    
    return value

def get_IAV_threshold(y,length,frame_size):
    IAV_data_size= 0
    Max_length = 0
    Min_length = 0
    threshold = 0.0
    Max_avg = 0.0
    Min_avg = 0.0
    
    IAV_data_size = int(length / frame_size)
    Max_length = int(IAV_data_size * 0.2)
    Min_length = int(IAV_data_size * 0.2)
    
    IAV_data = np.zeros(IAV_data_size, dtype=np.float32)
    Max = np.zeros(Max_length, dtype=np.float32)
    Min = np.zeros(Min_length, dtype=np.float32)
    
    j = 0
    
    for i in range(0,length,frame_size):
        if (j >= IAV_data_size):
            break
            
        data_sub = y[i:i+frame_size]
        IAV_data[j] = IAV(data_sub,frame_size)
        j = j + 1
        
    sort_indices = np.argsort(IAV_data)[::-1]
    IAV_data[:] = IAV_data[sort_indices]
        
    for i in range(Max_length):
        Max[i] = IAV_data[i]
        
    IAV_data.sort()
    
    for i in range(Min_length):
        Min[i] = IAV_data[i]
                          
    threshold = 0.0
    
    Max_avg = np.mean(Max)
    Min_avg = np.mean(Min)
    
    if(Max_avg*0.7 > Min_avg):
        threshold = (Max_avg - Min_avg) * 0.1
        threshold = threshold + Min_avg        
    else:
        threshold = Max_avg*0.2
        
    return threshold

def search_voicearea(y,frame_size,length,data_idx,th,IAV_th):
    i = 0
    j = 0
    res = 0.0
    start_point = 0
    end_point = 0
    temp = False

    i = data_idx
    while(i < length):        
        i = i + 1
        
        if(i+frame_size >= length):
            i = length
            break
        
        data_sub = y[i:i+frame_size]

        res = IAV(data_sub,frame_size)
        if( res < IAV_th ):
            i = i + (frame_size-1)
        else:
            for j in range(i,(i+frame_size)):
                if (abs(y[j])>th):
                    start_point = j
                    temp = True
                    break
            if(temp == True):
                break;

                
    if (i >= length):
        end_point = i
        return -1,-1

    temp = False
    temp2 = False
    count = 0

    for j in range(start_point,length,frame_size):
        if(j+frame_size >= length):
            end_point = length
            break
        
        data_sub = y[j:j+frame_size]

        res = IAV(data_sub,frame_size)
        if( res < IAV_th):
            if (count == 0):
                end_point = j
                count = count + 1
                temp = True
                temp2 = True
            elif (count < 4):
                count = count + 1
            else:
                break   
        else:
            if(temp2 == True):
                count = 0
                temp2 = False

    if (temp != True):
        end_point = length
    
    return start_point, end_point

def Remove_unvoice(y):
    frame_size = 500
    length = len(y)
    idx = 0
    
    IAV_th = get_IAV_threshold(y,length,frame_size)
    th = IAV_th/frame_size*2
    
    temp = []
    
    for i in range(0,length,frame_size):
        start_point, end_point = search_voicearea(y,frame_size,length,idx,th,IAV_th)
        
        if start_point == -1 :
            break
        
        y_sub = y[start_point:end_point].copy()
    
        temp.append(y_sub)
        idx = end_point
    
    temp = np.concatenate(temp)
    return temp

def normalization(y):
    windowsize = 500
    nor_coef = 700000.0
    
    IAV_avg = 0
    cnt = 0
    
    temp = Remove_unvoice(y)
    
    for i in range(0,len(temp)-windowsize,windowsize):
        IAV_avg += IAV(temp[i:i+windowsize], windowsize)
        cnt += 1
    
    IAV_avg = IAV_avg / cnt
    coef = nor_coef / IAV_avg
    
    y = y * coef
    
    return y    