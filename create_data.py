import os
from itertools import product

import numpy as np
import scipy as sp

import noise
import streaking_calculation as sc


if __name__=='__main__':
    # Streaking Specifications
    xuv_pulse = 105 # [eV]

    # NOTE: You you can change/add the parameters of the streaking
    #       Be careful though since the simulation time will scale fast
    binding_energies_ls = [[6.7, 65]] # Mehrere Atome mit 2 Energie Niveas hinzufügenm
    time_delays_ls      = [[0, 48e-18]] 
    amplitudes_ls       = [[1, 0.9]] 
    spectral_widths_ls  = [[3.87, 3.87]]
    chirps_ls           = [[2.86, 18.8]]

    data_x = [] # Spektogramm / Unabhängiege Trainingsdaten X 
    data_y = [] # Parameter / Abhängiege Trainingsdaten Y

    for binding_energies, time_delays, amplitudes, spectral_widths, chirps in product(binding_energies_ls,
                                                                                      time_delays_ls,
                                                                                      amplitudes_ls,
                                                                                      spectral_widths_ls,
                                                                                      chirps_ls):

        E_axis, tau_axis, spectogram = sc.create_spectogram(xuv_pulse,
                                                            binding_energies,
                                                            time_delays,
                                                            amplitudes,
                                                            spectral_widths,
                                                            chirps,
                                                            x_axis=np.linspace(-4, 4, 40),
                                                            y_axis=np.linspace(30, 120, 800))

        # This part creates a map between SNRs and Eps
        eps_map_ls=np.linspace(0.01, 7, 100)

        snr_map_ls=np.zeros_like(eps_map_ls)
        for idx, eps in enumerate(eps_map_ls):
            snr_safe_ls=[]
            for i in range(0, 5):
                noise_spectrogram, bg_spectrogram = noise.as_deep_learning(tau_axis, E_axis, spectogram, eps, re_with_bg=True)

                snr_safe_ls.append(noise.signal_to_noise(noise_spectrogram, bg_spectrogram))

            snr_map_ls[idx]=np.mean(snr_safe_ls)

        interp_mask=np.argsort(snr_map_ls)
        interp=sp.interpolate.splrep(snr_map_ls[interp_mask], eps_map_ls[interp_mask], s=1)

        ## Here the data is created
        SNR_ls=np.linspace(0.4, 8, 10)

        for SNR_idx, SNR in enumerate(SNR_ls):
            
            for i in range(0, 3): # the amount of specs per SNR can be changed depending on what exactly you want
                filename='spectrogram_'+str(i)

                # Speicher Anpassung
                noise_spectrogram = noise.as_deep_learning(tau_axis, E_axis, spectogram, sp.interpolate.splev(SNR, interp))
                params = np.array([binding_energies,time_delays,amplitudes,spectral_widths,chirps])


                data_x.append(noise_spectrogram.astype(np.float32)) 
                data_y.append(params.flatten().astype(np.float32))


final_x =np.array(data_x)
final_y = np.array(data_y)

#---
# VON KI KOPIERT !!! (start)

# Speichern als eine einzige komprimierte Datei
save_path = os.path.join(os.getcwd(), 'data', 'training_data.npz')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
np.savez_compressed(save_path, x=final_x,y=final_y) # Speichert mehrere Arrays in einer Datei

# VON KI KOPIERT !!! (ende)
#---