import os
import re
import csv
import warnings
import numpy as np
import scipy.io
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import plotly.plotly as py
import plotly.graph_objs as go

import mne
from mne import create_info, EpochsArray
from mne.preprocessing import ICA
from mne.time_frequency.stft import stft, stftfreq, stft_norm2
from mne.time_frequency import tfr_morlet, tfr_multitaper, psd_multitaper
from mne.connectivity import spectral_connectivity, phase_slope_index
from mne.viz import circular_layout, plot_connectivity_circle
import pyedflib
import pyeeg
from biosppy.signals import eeg

from scipy import signal
from scipy.stats import spearmanr, pearsonr
from pywt import wavedec
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, KernelDensity
from sklearn.cluster import KMeans, SpectralClustering, spectral_clustering
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, classification_report, cohen_kappa_score, roc_curve, roc_auc_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure, fowlkes_mallows_score, silhouette_score, calinski_harabaz_score
from sklearn.model_selection import cross_val_score

# patients = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06', 'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb12', 'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18', 'chb19', 'chb20', 'chb21', 'chb22', 'chb23', 'chb24']
channel_names = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8-1']
channel_names_car = ['FP1', 'F7', 'T7', 'P7', 'O1', 'P3', 'C3', 'F3', 'FP2', 'F8', 'T8', 'P8', 'O2', 'P4', 'C4', 'F4', 'FT9', 'FT10']
channel_proj_mat = np.linalg.inv([
    [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
con_window_size = 2
con_window_shift = 0.2
freq_limit_low = (4, 8, 14)
freq_limit_high = (7, 13, 30)
cwt_freqs = np.arange(4, 30, 1)
cwt_n_cycles = cwt_freqs / 20.


# perform preprocessing on raw data
def preprocess(raw):
    # extract the data
    eeg_data = raw.get_data()
    eeg_data_left = eeg_data[0:7, :]
    eeg_data_right = eeg_data[8:15, :]
    eeg_data_cross = eeg_data[19:22, :]
    eeg_data_lr = np.append(eeg_data_left, eeg_data_right, 0)
    eeg_data_lrc = np.append(eeg_data_lr, eeg_data_cross, 0)
    eeg_data_corrected = np.append(eeg_data_lrc, np.zeros((1, eeg_data.shape[1])), 0)

    # project into common average reference
    eeg_data_reref = np.matmul(channel_proj_mat, eeg_data_corrected)
    info = mne.create_info(ch_names=channel_names_car, sfreq=raw.info['sfreq'], ch_types='eeg')
    raw = mne.io.RawArray(eeg_data_reref, info, verbose='Warning')

    # bandpass filtering between 1-45 hz
    raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin')

    # # run ICA
    # ica = ICA(n_components=0.99, method='fastica')
    # ica.fit(raw, verbose='Warning')
    # raw_sources = ica.get_sources(raw)

    # extract the data
    eeg_data = raw.get_data()
    # eeg_data = raw_sources.get_data()

    return eeg_data


# update figure
def coherence_updatefig(idx):
    con_img.set_array(con_data[idx])

    return [con_img]


# visualize the connectivity matrix
def visualize_connectivity(connectivity_matrix):
    global con_data, con_img
    con_data = np.transpose(connectivity_matrix, (0, 2, 1))
    # fig = plt.figure()
    # con_img = plt.imshow(con_data[0], interpolation='nearest', aspect='auto', cmap=plt.get_cmap('jet'), vmin=0, vmax=1)
    # ani = animation.FuncAnimation(fig, coherence_updatefig, frames=range(con_data.shape[0]), interval=200, blit=True, repeat=False)
    plt.imshow(np.transpose(con_data[:,:,15],(1,0)), interpolation='nearest', aspect='auto', cmap=plt.get_cmap('jet'), vmin=0, vmax=1)
    plt.show()


# extract data from the sample's files
def preictal_extract(addr, session_addr):
    print(re.sub('.edf', '', re.search('\chb[0-9]{2}_[0-9]{2}.edf', addr).group(0)))

    # read the info file
    subject_addr = re.sub('\chb[0-9]{2}_[0-9]{2}.edf', '', addr)
    for file in os.listdir(subject_addr):
        if file.endswith('.txt'):
            info_file_addr = os.path.join(subject_addr, file)
    f = open(info_file_addr)
    info_file_content = f.read()

    # extract the seizure timing from file
    file_name = re.search('chb[0-9]{2}_[0-9]{2}.edf', addr).group(0)
    info_pattern = 'File Name: ' + file_name + '\nFile Start Time: [0-9]+:[0-9]+:[0-9]+\nFile End Time: [0-9]+:[0-9]+:[0-9]+\nNumber of Seizures in File: 1\nSeizure [1 ]*Start Time: [0-9]+ seconds\nSeizure [1 ]*End Time: [0-9]+ seconds\n'
    info = re.search(info_pattern, info_file_content).group(0)
    seizure_start_pattern = 'Start Time: [0-9]+ seconds'
    seizure_end_pattern = 'End Time: [0-9]+ seconds'
    seizure_start_text = re.search(seizure_start_pattern, info).group(0)
    seizure_end_text = re.search(seizure_end_pattern, info).group(0)
    seizure_start_time = np.int(re.search('[0-9]+', seizure_start_text).group(0))
    seizure_end_time = np.int(re.search('[0-9]+', seizure_end_text).group(0))

    # read the edf file
    raw = mne.io.read_raw_edf(addr, preload=True, verbose='ERROR')

    # check channel names, sampling freq, and preictal timing
    if raw.ch_names != channel_names or raw.info['sfreq'] != 256.0:
        error('Invalid Sample!')

    # calculate the connectivity
    eeg_data = preprocess(raw)
    preictal_con = []
    window_center = np.int((seizure_start_time - con_window_size / 2) * raw.info['sfreq'])
    while (window_center - np.int((con_window_size / 2) * raw.info['sfreq'])) >= 0:
        window_start = window_center - np.int((con_window_size / 2) * raw.info['sfreq'])
        window_end = window_center + np.int((con_window_size / 2) * raw.info['sfreq'])
        window_eeg = np.expand_dims(eeg_data[:, window_start:window_end], axis=0)
        try:
            con, freqs, times, _, _ = spectral_connectivity(window_eeg, method='coh', mode='multitaper', sfreq=raw.info['sfreq'], fmin=freq_limit_low, fmax=freq_limit_high, faverage=True, n_jobs=1, verbose='Warning')
            # con, freqs, times, _, _ = phase_slope_index(window_eeg, mode='multitaper', sfreq=raw.info['sfreq'], fmin=freq_limit_low, fmax=freq_limit_high, n_jobs=1, verbose='Warning')
        except:
            error('Wrong Connectivity!')

        # remove zero elements
        del_idx = []
        for row in range(con.shape[0]):
            for col in range(con.shape[1]):
                if row <= col:
                    del_idx.append(row * con.shape[1] + col)
        con = np.reshape(con, (con.shape[0] * con.shape[1], con.shape[2]))
        con = np.delete(con, del_idx, 0)
        preictal_con.append(con)
        window_center = window_center - np.int(con_window_shift * raw.info['sfreq'])
    preictal_con = np.asarray(preictal_con)

    # check if the data has nan
    if np.isnan(preictal_con).any():
        return

    # visualize the connectivity matrix
    # visualize_connectivity(preictal_con)

    # save to file
    np.save(session_addr, preictal_con)


# extract data from the sample's files
def interictal_extract(addr, session_addr):
    print(re.sub('.edf', '', re.search('\chb[0-9]{2}_[0-9]{2}.edf', addr).group(0)))

    # read the edf file
    raw = mne.io.read_raw_edf(addr, preload=True, verbose='ERROR')

    # check channel names, sampling freq, and preictal timing
    if raw.ch_names != channel_names or raw.info['sfreq'] != 256.0:
        error('Invalid Sample!')

    # calculate the connectivity
    eeg_data = preprocess(raw)
    interictal_con = []
    window_center = np.int((con_window_size / 2) * raw.info['sfreq'])
    while window_center + np.int((con_window_size / 2) * raw.info['sfreq']) < 300 * raw.info['sfreq']: # eeg_data.shape[1]:
        window_start = window_center - np.int((con_window_size / 2) * raw.info['sfreq'])
        window_end = window_center + np.int((con_window_size / 2) * raw.info['sfreq'])
        window_eeg = np.expand_dims(eeg_data[:, window_start:window_end], axis=0)
        try:
            con, freqs, times, n_epochs, n_tapers = spectral_connectivity(window_eeg, method='coh', mode='multitaper', sfreq=raw.info['sfreq'], fmin=freq_limit_low, fmax=freq_limit_high, faverage=False, n_jobs=1, verbose='Warning')
            # con, freqs, times, _, _ = phase_slope_index(window_eeg, mode='multitaper', sfreq=raw.info['sfreq'], fmin=freq_limit_low, fmax=freq_limit_high, n_jobs=1, verbose='Warning')
        except:
            error('Wrong Connectivity!')

        # remove zero elements
        del_idx = []
        for row in range(con.shape[0]):
            for col in range(con.shape[1]):
                if row <= col:
                    del_idx.append(row * con.shape[1] + col)
        con = np.reshape(con, (con.shape[0] * con.shape[1], con.shape[2]))
        con = np.delete(con, del_idx, 0)
        interictal_con.append(con)

        # plt.imshow(np.transpose(con, (1, 0)), interpolation='nearest', aspect='auto', cmap=plt.get_cmap('jet'), vmin=0, vmax=1)
        # plt.ylim(30, 4)
        # plt.show()

        window_center = window_center + np.int(con_window_shift * raw.info['sfreq'])
    interictal_con = np.asarray(interictal_con)

    # check if the data has nan
    if np.isnan(interictal_con).any():
        return

    # visualize the connectivity matrix
    visualize_connectivity(interictal_con)

    # save to file
    np.save(session_addr, interictal_con)


# reload cons
def reload_data(reload, pids):
    # dataset addr
    base_addr = 'F:\Saeed\Programming\Dataset\MIT\\'
    for pid in pids:

        # extract preictal data
        if reload=='preictal' or reload=='both':
            with open('RECORDS-WITH-SEIZURES.txt') as f:
                content = f.readlines()
            seizure_records = [x.strip() for x in content]
            for record_addr in seizure_records:
                addr = base_addr + record_addr.replace('/', '\\')
                if pid in record_addr:
                    session_addr = 'dataset\\' + record_addr.replace('/', '\\preictal\\').replace('.edf', '.npy')
                    preictal_extract(addr, session_addr)

        # extract interictal data
        if reload=='interictal' or reload=='both':
            with open('RECORDS-WITHOUT-SEIZURES.txt') as f:
                content = f.readlines()
            nonseizure_records = [x.strip() for x in content]
            for record_addr in nonseizure_records:
                addr = base_addr + record_addr.replace('/', '\\')
                if pid in record_addr:
                    session_addr = 'dataset\\' + record_addr.replace('/', '\\interictal\\').replace('.edf', '.npy')
                    interictal_extract(addr, session_addr)


# normalize the con
def normalize_con(con):
    temp_con = []
    for channel_pair in range(0, con.shape[1]):
        features = con[:, channel_pair, :]
        features = normalize(features, norm='l1', axis=1, copy=True, return_norm=False)
        temp_con.append(np.transpose(features), (1, 0))
    con = np.transpose(np.asarray(temp_con), (2, 0, 1))

    return con


# extract wavelet features
def wavelet_decomposition(con):
    wavelet_family = 'db1'
    wavelet_level = 2
    temp_con = []
    for channel_pair in range(0, con.shape[1]):
        freq_con = []
        for freq in range(0, con.shape[2]):
            signal = con[:, channel_pair, freq]
            features = np.hstack(wavedec(signal, wavelet=wavelet_family, mode='symmetric', level=wavelet_level))
            freq_con.append(features)
        temp_con.append(freq_con)
    con = np.transpose(np.asarray(temp_con), (2, 0, 1))

    return con


# load data
def load_data(pid, window_size, preictal_threshold):
    # read the preictal file addresses
    addr = 'dataset\\' + pid + '\\preictal\\'
    session_addrs = []
    for (dirpath, dirnames, filenames) in os.walk(addr):
        for file in filenames:
            session_addrs.append(dirpath + file)
    preictal_samples = []
    preictal_sessions = []
    for session_addr in session_addrs:
        preictal_con = np.load(session_addr)

        # normalize the con
        # preictal_con = normalize_con(preictal_con)

        # extract wavelet features
        # preictal_con = wavelet_decomposition(preictal_con)

        # visualize the connectivity matrix
        # visualize_connectivity(preictal_con)

        # extract preictal samples
        window_center = np.int(window_size / 2)
        while (window_center + np.int(window_size / 2)) < preictal_con.shape[0] and (window_center + np.int(window_size / 2)) < np.int(preictal_threshold / con_window_shift):
            window_start = window_center - np.int(window_size / 2)
            window_end = window_center + np.int(window_size / 2) + 1
            window_con = preictal_con[window_start:window_end, :, :]
            #window_con = -np.log(1-np.power(0.999999*preictal_con[window_start:window_end, :, :],2))
            preictal_samples.append(window_con[::-1, :, :])
            preictal_sessions.append(session_addr)
            window_center = window_center + max(np.int(preictal_window_shift * window_size), 1)
    preictal_samples = np.asarray(preictal_samples)

    # read the interictal file addresses
    addr = 'dataset\\' + pid + '\\interictal\\'
    session_addrs = []
    for (dirpath, dirnames, filenames) in os.walk(addr):
        for file in filenames:
            session_addrs.append(dirpath + file)
    interictal_samples = []
    interictal_sessions = []
    for session_addr in session_addrs:
        interictal_con = np.load(session_addr)

        # normalize the con
        # interictal_con = normalize_con(interictal_con)

        # extract wavelet features
        # interictal_con = wavelet_decomposition(interictal_con)

        # visualize the connectivity matrix
        # visualize_connectivity(interictal_con)

        # extract interictal samples
        window_center = np.int(window_size / 2)
        while (window_center + np.int(window_size / 2)) < interictal_con.shape[0]:
            window_start = window_center - np.int(window_size / 2)
            window_end = window_center + np.int(window_size / 2) + 1
            window_con = interictal_con[window_start:window_end, :, :]
            interictal_samples.append(window_con)
            interictal_sessions.append(session_addr)
            window_center = window_center + max(np.int(interictal_window_shift * window_size), 1)
    interictal_samples = np.asarray(interictal_samples)

    return preictal_samples, interictal_samples, preictal_sessions, interictal_sessions


# split the train and test data, then balance the classes by either upsampling preictal or downsampling interictal
def train_test_split(preictal_samples, interictal_samples, preictal_labels, interictal_labels):

    # parameters
    test_proportion = 0.2

    # split the data into training/testing sets
    preictal_n_test = np.int(test_proportion * preictal_samples.shape[0])
    indices = np.random.permutation(preictal_samples.shape[0])
    train_idx = indices[:-preictal_n_test]
    test_idx = indices[-preictal_n_test:]
    preictal_samples_train = preictal_samples[train_idx, :]
    preictal_samples_test = preictal_samples[test_idx, :]
    preictal_labels_train = preictal_labels[train_idx]
    preictal_labels_test = preictal_labels[test_idx]

    interictal_n_test = np.int(test_proportion * interictal_samples.shape[0])
    indices = np.random.permutation(interictal_samples.shape[0])
    train_idx = indices[:-interictal_n_test]
    test_idx = indices[-interictal_n_test:]
    interictal_samples_train = interictal_samples[train_idx, :]
    interictal_samples_test = interictal_samples[test_idx, :]
    interictal_labels_train = interictal_labels[train_idx]
    interictal_labels_test = interictal_labels[test_idx]

    # upsample the train data
    if balance == 'train_up':
        temp_preictal_samples_train = []
        temp_preictal_labels_train = []
        for repeat_idx in range(np.int(interictal_samples_train.shape[0] / preictal_samples_train.shape[0])):
            [temp_preictal_samples_train.append(preictal_samples_train[sample_idx]) for sample_idx in range(preictal_samples_train.shape[0])]
            [temp_preictal_labels_train.append(preictal_labels_train[sample_idx]) for sample_idx in range(preictal_labels_train.shape[0])]
        preictal_samples_train = np.asarray(temp_preictal_samples_train)
        preictal_labels_train = np.asarray(temp_preictal_labels_train)

    # downsample the train data
    elif balance == 'train_down':
        n_down = preictal_samples_train.shape[0]
        interictal_indices_train = np.random.permutation(interictal_samples_train.shape[0])
        interictal_indices_train = interictal_indices_train[0:n_down]
        interictal_samples_train = interictal_samples_train[interictal_indices_train]
        interictal_labels_train = interictal_labels_train[interictal_indices_train]

    # downsample both the train and test data
    elif balance == 'all_down':
        n_down = preictal_samples_train.shape[0]
        interictal_indices_train = np.random.permutation(interictal_samples_train.shape[0])
        interictal_indices_train = interictal_indices_train[0:n_down]
        interictal_samples_train = interictal_samples_train[interictal_indices_train]
        interictal_labels_train = interictal_labels_train[interictal_indices_train]
        n_down = preictal_samples_test.shape[0]
        interictal_indices_test = np.random.permutation(interictal_samples_test.shape[0])
        interictal_indices_test = interictal_indices_test[0:n_down]
        interictal_samples_test = interictal_samples_test[interictal_indices_test]
        interictal_labels_test = interictal_labels_test[interictal_indices_test]

    return preictal_samples_train, preictal_samples_test, interictal_samples_train, interictal_samples_test, preictal_labels_train, preictal_labels_test, interictal_labels_train, interictal_labels_test


# evaluate the classification result
def evaluate_classification(labels_tests, labels_preds, probas_preds):
    labels_tests = np.asarray(labels_tests)
    labels_preds = np.asarray(labels_preds)
    probas_preds = np.asarray(probas_preds)
    probas_preds = probas_preds[:, 1]
    cnf = confusion_matrix(labels_tests, labels_preds)
    interictal_accuracy = cnf[0,0] / (cnf[0,0] + cnf[0,1])
    preictal_accuracy = cnf[1,1] / (cnf[1,0] + cnf[1,1])
    print(cnf, '\nInterictal: %0.2f, Preictal: %0.2f, Avg: %0.2f' % (interictal_accuracy, preictal_accuracy, accuracy_score(labels_tests, labels_preds)))
    target_names = ['interictal', 'preictal']
    # print(classification_report(labels_tests, labels_preds, target_names=target_names))

    # auc = roc_auc_score(labels_tests, probas_preds)
    # fpr, tpr, _ = roc_curve(labels_tests, probas_preds)
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([-0.02, 1.0])
    # plt.ylim([0.0, 1.03])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.show()

    return preictal_accuracy, interictal_accuracy


def classification(pid, preictal_samples, interictal_samples, preictal_sessions, interictal_sessions):
    # reshape
    sample_shape = preictal_samples.shape[1:]
    preictal_samples = preictal_samples.reshape((preictal_samples.shape[0], preictal_samples.shape[1] * preictal_samples.shape[2] * preictal_samples.shape[3]))
    interictal_samples = interictal_samples.reshape((interictal_samples.shape[0], interictal_samples.shape[1] * interictal_samples.shape[2] * interictal_samples.shape[3]))

    # generate labels
    # preictal_labels = np.asarray(LabelEncoder().fit_transform(preictal_sessions))
    # interictal_labels = LabelEncoder().fit_transform(interictal_sessions)
    # offset = max(preictal_labels)+1
    # temp_interictal_labels = []
    # [temp_interictal_labels.append(old_label+offset) for old_label in interictal_labels]
    # interictal_labels = np.asarray(temp_interictal_labels)
    preictal_labels = np.array([1 for preictal_sample in preictal_samples])
    interictal_labels = np.array([0 for interictal_sample in interictal_samples])

    n_repeat = 5
    labels_tests = []
    labels_preds = []
    probas_preds = []
    betas = []
    for test_idx in range(n_repeat):
        print(test_idx)

        # split and balance the classes
        preictal_samples_train, preictal_samples_test, interictal_samples_train, interictal_samples_test, preictal_labels_train, preictal_labels_test, interictal_labels_train, interictal_labels_test = train_test_split(preictal_samples, interictal_samples, preictal_labels, interictal_labels)

        samples_train = np.concatenate((preictal_samples_train, interictal_samples_train), axis=0)
        samples_test = np.concatenate((preictal_samples_test, interictal_samples_test), axis=0)

        labels_train = np.concatenate((preictal_labels_train, interictal_labels_train), axis=0)
        labels_test = np.concatenate((preictal_labels_test, interictal_labels_test), axis=0)
        labels_tests.extend(labels_test)

        # fit the classifier
        # clf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample')
        # clf = SVC(probability=True)
        # clf = GaussianNB()
        # clf = LinearDiscriminantAnalysis()
        # clf = QuadraticDiscriminantAnalysis()
        # clf = KNeighborsClassifier(n_neighbors=5)
        # clf = MLPClassifier(hidden_layer_sizes=(25,5,))
        # clf = LogisticRegression(penalty='l1', solver='liblinear', C=1e5, class_weight='balanced')
        clf = LogisticRegressionCV(Cs=[1e-5,1e-4,1e-3,1e0,1e3,1e4,1e5], cv=2, refit=True, penalty='l2', solver='liblinear', class_weight='balanced')
        # clf = RidgeClassifier(tol=1e-2, solver="lsqr")
        # clf = PassiveAggressiveClassifier(n_iter=50)
        # clf = SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")
        # clf = MultinomialNB(alpha=.01)
        # clf = KernelRidge(alpha=1.0)
        # kde = KernelDensity(bandwidth=0.04, metric='haversine', kernel='gaussian', algorithm='ball_tree')
        # kde.fit(cons_train[labels_train == 0])
        # kde.score_samples(cons_train)

        clf.fit(samples_train, labels_train)
        labels_pred = clf.predict(samples_test)
        labels_preds.extend(labels_pred)
        probas_pred = clf.predict_proba(samples_test)
        probas_preds.extend(probas_pred)
        # betas.append(np.reshape(clf.coef_,(sample_shape)))

    # betas = np.asarray(betas)
    # betas_mean = np.absolute(np.mean(betas, axis=0))
    # betas_std = np.std(betas, axis=0)
    # print('min: %0.2f, max: %0.2f' % (betas_mean.min(), betas_mean.max()))
    # plt.imshow(np.transpose(betas_mean[0,:,:],(1,0)), interpolation='nearest', aspect='auto', cmap=plt.get_cmap('jet'), vmin=betas_mean.min(), vmax=betas_mean.max())
    # plt.show()
    # print('min: %0.2f, max: %0.2f' % (betas_std.min(), betas_std.max()))
    # plt.imshow(np.transpose(betas_std[0,:,:], (1, 0)), interpolation='nearest', aspect='auto', cmap=plt.get_cmap('jet'), vmin=betas_std.min(), vmax=betas_std.max())
    # plt.show()

    # evaluate the result
    preictal_accuracy, interictal_accuracy = evaluate_classification(labels_tests, labels_preds, probas_preds)

    return preictal_accuracy, interictal_accuracy


# visualize the tsne trajectory
def visualize_tsne(samples_tsne, labels):
    scatter = plt.scatter(samples_tsne[:, 0], samples_tsne[:, 1], c=labels, alpha=1.0)
    plt.colorbar(scatter)
    plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(samples_tsne[:,0], samples_tsne[:,1], samples_tsne[:,2], c=colors_tsne, marker='o')
    # plt.show()

    # trace = go.Scatter3d(x=samples_tsne[:,0], y=samples_tsne[:,1], z=samples_tsne[:,2], mode='markers', marker=dict(size=12,color=labels,colorscale='Viridis',opacity=0.8))
    # data = [trace]
    # layout = go.Layout(margin=dict(l=0,r=0,b=0,t=0))
    # fig = go.Figure(data=data, layout=layout)
    # py.iplot(fig, filename='3d visualization')

    # global tsne_data, tsne_img
    # con_data = np.transpose(connectivity_matrix, (0, 2, 1))
    # fig = plt.figure()
    # con_img = plt.imshow(con_data[0], interpolation='nearest', aspect='auto', cmap=plt.get_cmap('jet'), vmin=0, vmax=1)
    # ani = animation.FuncAnimation(fig, coherence_updatefig, frames=range(con_data.shape[0]), interval=200, blit=True, repeat=False)
    # plt.show()


def clustering(pid, preictal_samples, interictal_samples, preictal_sessions, interictal_sessions):
    # reshape
    preictal_samples = preictal_samples.reshape((preictal_samples.shape[0], preictal_samples.shape[1] * preictal_samples.shape[2] * preictal_samples.shape[3]))
    interictal_samples = interictal_samples.reshape((interictal_samples.shape[0], interictal_samples.shape[1] * interictal_samples.shape[2] * interictal_samples.shape[3]))

    # generate labels
    preictal_labels = np.asarray(LabelEncoder().fit_transform(preictal_sessions))
    interictal_labels = LabelEncoder().fit_transform(interictal_sessions)
    offset = max(preictal_labels)+1
    temp_interictal_labels = []
    [temp_interictal_labels.append(old_label+offset) for old_label in interictal_labels]
    interictal_labels = np.asarray(temp_interictal_labels)
    # preictal_labels = np.array([1 for preictal_sample in preictal_samples])
    # interictal_labels = np.array([0 for interictal_sample in interictal_samples])

    # pca = PCA(n_components=200)
    # samples_pca = pca.fit_transform(samples)
    # print(sum(pca.explained_variance_ratio_))
    # plt.scatter(samples_pca[:,0], samples_pca[:,1], c=labels, alpha=0.5)
    # plt.show()

    # select data points
    # preictal_indices = np.random.randint(low=0, high=preictal_samples.shape[0], size=n_sample)
    # interictal_indices = np.random.randint(low=0, high=interictal_samples.shape[0], size=n_sample)
    # preictal_indices = range(0,n_sample)
    # interictal_indices = range(0,n_sample)
    n_sample = 2000
    preictal_indices = []
    counter = np.zeros((max(preictal_labels) + 1))
    for idx, label in enumerate(preictal_labels):
        if counter[label] <= n_sample:
            preictal_indices.append(idx)
            counter[label] = counter[label] + 1
    interictal_indices = []
    counter = np.zeros((max(interictal_labels) + 1))
    for idx, label in enumerate(interictal_labels):
        if counter[label] <= n_sample:
            interictal_indices.append(idx)
            counter[label] = counter[label] + 1
    samples = np.concatenate((preictal_samples[preictal_indices,:], interictal_samples[interictal_indices,:]), axis=0)
    labels = np.concatenate((preictal_labels[preictal_indices], interictal_labels[interictal_indices]), axis=0)
    colors = []
    for label in labels:
        if label == 0: # or label == 1 or label == 2:
            colors.append('g')
        elif label == 1:
            colors.append('c')
        elif label == 2:
            colors.append('b')
        elif label == 3: # or label == 4 or label == 5:
            colors.append('r')
        elif label == 4:
            colors.append('m')
        elif label == 5:
            colors.append('y')
        else:
            colors.append('k')
    colors_tsne = np.asarray(colors)
    tsne = TSNE(n_components=2, perplexity=50, early_exaggeration=12, learning_rate=100, n_iter=500, verbose=1)
    samples_tsne = tsne.fit_transform(samples)
    visualize_tsne(samples_tsne, labels)

    # fit the clustering algorithm
    clus = KMeans(n_clusters=2, init='k-means++')
    # clus = SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity='nearest_neighbors', n_neighbors=5)
    labels_pred = clus.fit_predict(samples_tsne)

    # evaluate the result
    ari = adjusted_rand_score(labels, labels_pred)
    ami = adjusted_mutual_info_score(labels, labels_pred)
    h, c, v = homogeneity_completeness_v_measure(labels, labels_pred)
    # fmi = fowlkes_mallows_score(labels, labels_pred)
    sil = silhouette_score(samples_tsne, labels_pred, metric='euclidean')
    cal = calinski_harabaz_score(samples_tsne, labels_pred)
    print('ari=%.2f, ' % ari, 'ami=%.2f, ' % ami, 'h=%.2f, ' % h, 'c=%.2f, ' % c, 'v=%.2f, ' % v, 'sil=%.2f, ' % sil, 'cal=%.2f' % cal)

def preictal_clustering(pid, preictal_samples, preictal_sessions):
    # reshape
    preictal_samples = preictal_samples.reshape((preictal_samples.shape[0], preictal_samples.shape[1] * preictal_samples.shape[2] * preictal_samples.shape[3]))

    # generate labels
    preictal_labels = np.array([idx for idx in range(preictal_samples.shape[0])])

    n_sample = 15000
    # samples = np.concatenate((preictal_samples[0:n_sample,:], preictal_samples[-n_sample:,:]), axis=0)
    # labels = np.concatenate((preictal_labels[0:n_sample], preictal_labels[-n_sample:]), axis=0)
    samples = preictal_samples[0:n_sample,:]
    labels = preictal_labels[0:n_sample] * con_window_shift
    tsne = TSNE(n_components=2, perplexity=100, early_exaggeration=12, learning_rate=100, n_iter=500, verbose=1)
    samples_tsne = tsne.fit_transform(samples)
    visualize_tsne(samples_tsne, labels)

    # fit the clustering algorithm
    clus = KMeans(n_clusters=2, init='k-means++')
    # clus = SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity='nearest_neighbors', n_neighbors=5)
    labels_pred = clus.fit_predict(samples_tsne)

    # evaluate the result
    ari = adjusted_rand_score(labels, labels_pred)
    ami = adjusted_mutual_info_score(labels, labels_pred)
    h, c, v = homogeneity_completeness_v_measure(labels, labels_pred)
    # fmi = fowlkes_mallows_score(labels, labels_pred)
    sil = silhouette_score(samples_tsne, labels_pred, metric='euclidean')
    cal = calinski_harabaz_score(samples_tsne, labels_pred)
    print('ari=%.2f, ' % ari, 'ami=%.2f, ' % ami, 'h=%.2f, ' % h, 'c=%.2f, ' % c, 'v=%.2f, ' % v, 'sil=%.2f, ' % sil, 'cal=%.2f' % cal)


def interictal_clustering(pid, interictal_samples):
    # reshape
    interictal_samples = interictal_samples.reshape((interictal_samples.shape[0], interictal_samples.shape[1] * interictal_samples.shape[2] * interictal_samples.shape[3]))

    # generate labels
    interictal_labels = np.array([idx for idx in range(interictal_samples.shape[0])]) * con_window_shift

    n_sample = 5000
    samples = np.concatenate((interictal_samples[0:n_sample,:], interictal_samples[-n_sample:,:]), axis=0)
    labels = np.concatenate((interictal_labels[0:n_sample], interictal_labels[-n_sample:]), axis=0)
    # samples = preictal_samples[0:n_sample,:]
    # labels = preictal_labels[0:n_sample] * con_window_shift
    tsne = TSNE(n_components=2, perplexity=50, early_exaggeration=12, learning_rate=100, n_iter=500, verbose=1)
    samples_tsne = tsne.fit_transform(samples)
    visualize_tsne(samples_tsne, labels)


# main
if __name__ == '__main__':

    # parameters
    reload = 'none' # 'none', 'preictal', 'interictal', 'both'
    window_sizes = [25]
    preictal_thresholds = [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
    preictal_window_shift = 2
    interictal_window_shift = 2
    balance = 'train_down'

    # load cons from file or reload cons
    if reload!='none':
        pids = ['chb10', 'chb01', 'chb02', 'chb03', 'chb05', 'chb07', 'chb08']
        reload_data(reload, pids)

    else:
        pid = 'chb08'
        print(pid)
        preictal_accuracies = {}
        interictal_accuracies = {}
        for window_size in window_sizes:
            for preictal_threshold in preictal_thresholds:
                preictal_samples, interictal_samples, preictal_sessions, interictal_sessions = load_data(pid, window_size, preictal_threshold)
                preictal_accuracy, interictal_accuracy = classification(pid, preictal_samples, interictal_samples, preictal_sessions, interictal_sessions)
                preictal_accuracies[(window_size, preictal_threshold)] = preictal_accuracy
                interictal_accuracies[(window_size, preictal_threshold)] = interictal_accuracy
        preictal_accuracies = np.fromiter(preictal_accuracies.values(), dtype=float)
        interictal_accuracies = np.fromiter(interictal_accuracies.values(), dtype=float)
        plt.figure()
        plt.plot(preictal_thresholds, preictal_accuracies, color='darkorange', lw=2, linestyle='--', label='Preictal Accuracy')
        plt.plot(preictal_thresholds, interictal_accuracies, color='darkblue', lw=2, linestyle='--', label='Interictal Accuracy')
        plt.xlim([0.0, max(preictal_thresholds)+10])
        plt.ylim([0.0, 1.03])
        plt.xlabel('Preictal Threshold')
        plt.ylabel('Accuracy')
        plt.legend(loc="lower right")
        plt.show()

        # clustering(pid, preictal_samples, interictal_samples, preictal_sessions, interictal_sessions)
        # preictal_clustering(pid, preictal_samples, preictal_sessions)
        # interictal_clustering(pid, interictal_samples)