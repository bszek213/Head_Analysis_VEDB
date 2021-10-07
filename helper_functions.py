#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:10:03 2021

@author: bszekely

Functions for biomechanical analyses of head movement data
from VEDB 
"""
import numpy as np
import yaml
from scipy.signal import find_peaks,variation
from scipy.fft import rfft, rfftfreq

def change_odo_timestamp(odometry,odometry_time_stamp,world_time_stamp):
    new_odo = np.zeros((len(world_time_stamp),3))
    for i in range(len(world_time_stamp)):   
        odo_index = np.argmin(np.abs((odometry_time_stamp - world_time_stamp[i]).astype(float)))  
        new_odo[i] = odometry.position[odo_index,:]
        return new_odo

def get_start_end(doc,session_num,fps):
    with open(r'/home/bszekely/Desktop/ProjectsResearch/biomechanics_head_vedb/slippage_sessions_list.yaml') as file:
        documents = yaml.full_load(file)
        for item, doc in documents.items():
            slow_start = doc[session_num]['slow_start']
            slow_end = doc[session_num]['slow_end']
            med_start = doc[session_num]['medium_start']
            med_end = doc[session_num]['medium_end']
            fast_start = doc[session_num]['fast_start']
            fast_end = doc[session_num]['fast_end']

    start_i_slow = (slow_start[0][0] * 60 + slow_start[0][1]) * fps
    end_i_slow = (slow_end[0][0] * 60 + slow_end[0][1]) * fps

    start_i_med = (med_start[0][0] * 60 + med_start[0][1]) * fps
    end_i_med = (med_end[0][0] * 60 + med_end[0][1]) * fps
    
    start_i_fast = (fast_start[0][0] * 60 + fast_start[0][1]) * fps 
    end_i_fast = (fast_end[0][0] * 60 + fast_end[0][1]) * fps
    
    return start_i_slow, end_i_slow, start_i_med, end_i_med, start_i_fast, end_i_fast

def gait_var(new_odo,fps,start_i_slow, end_i_slow, start_i_med,end_i_med, start_i_fast, end_i_fast):
    slow = new_odo[start_i_slow:end_i_slow,:]
    med = new_odo[start_i_med:end_i_med,:]
    fast = new_odo[start_i_fast:end_i_fast,:]
    
    peaks_slow, _= find_peaks(slow[:,1],distance=fps / 3)
    peaks_med, _= find_peaks(med[:,1],distance=fps / 3)
    peaks_fast, _= find_peaks(fast[:,1],distance=fps / 3)
    
    step_time_slo = np.zeros(len(peaks_slow) - 1)
    for i in range(len(peaks_slow) - 1):
        step_time_slo[i] = (peaks_slow[i+1] - peaks_slow[i]) * (1/fps)
        
    step_time_med = np.zeros(len(peaks_med) - 1)
    for i in range(len(peaks_med) - 1):
        step_time_med[i] = (peaks_med[i+1] - peaks_med[i]) * (1/fps)
        
    step_time_fas = np.zeros(len(peaks_fast) - 1)
    for i in range(len(peaks_fast) - 1):
        step_time_fas[i] = (peaks_fast[i+1] - peaks_fast[i]) * (1/fps)

    # CV of the time  variability
    cv_slow = variation(np.abs(np.diff(step_time_slo)))
    cv_med = variation(np.abs(np.diff(step_time_med)))
    cv_fast = variation(np.abs(np.diff(step_time_fas)))
    return cv_slow, cv_med, cv_fast
    # print(variation(np.abs(np.diff(step_time_slo))))
    # print(variation(np.abs(np.diff(step_time_med))))
    # print(variation(np.abs(np.diff(step_time_fas))))
    
def fft_analysis(sig,fps):
    N = len(sig[:,1])
    yf = rfft(sig[:,1])
    xf = rfftfreq(N, 1 / fps)
    new_yf = np.delete(yf, 0)
    new_xf = np.delete(xf, 0)
    real_power = np.abs(new_yf)
    max_power = np.max(real_power)
    ind = np.where(real_power == max_power)
    res_freq = new_xf[ind]
    return res_freq