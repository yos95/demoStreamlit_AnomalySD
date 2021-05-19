

def logMelSpectrogram(audio, fe, dt):
    stfts = np.abs(librosa.stft(audio,n_fft = int(dt*fe),hop_length = int(dt*fe),center = True)).T
    num_spectrogram_bins = stfts.shape[-1]
    # MEL filter
    linear_to_mel_weight_matrix = librosa.filters.mel(sr=fe,n_fft=int(dt*fe) + 1,n_mels=num_spectrogram_bins,).T
    # Apply the filter to the spectrogram
    mel_spectrograms = np.tensordot(stfts,linear_to_mel_weight_matrix,1)
    return np.log(mel_spectrograms + 1e-6)