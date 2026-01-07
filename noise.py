import numpy as np
import scipy as sp


def signal_to_noise(spectogram, fit):
    """
    Calculates the SNR in dB of a spectrogram with the 'fit' spectrogram. But that is not necesarily bound to the
    fitted spectrogram. The 'fit' just needs to be the 'ideal spectrogram'.
    """
    return 10*np.log10(np.sum(fit)/np.sum(np.abs(fit-spectogram)))



# Note: Shirley-Backrogund ermittelt den Hintergrund der durch Energieverluste der Elektronen beim Verlassen eines Gitters auftreten. 
# $$I(E)_{\text{Ideal}} \approx I(E)_{\text{Real}} - B_{\text{Shirley}}(E)$$

def shirley_vegh_salvi_castle(energy, spectrum, k, sigma_weight, **kwargs):
    """
    Shirley-Vegh-Salvi-Castle implementation for background simulation as described in 'Practical methods for 
    background subtraction in photoemission spectra' (DOI 10.1002/sia.5453). To use that for simulating background
    has been presented in the supplement to 'Deep learning in attosecond metrology' (DOR 10.1364/oe.452108).
    
    :param energy: The energy-interval of the simulated data. np.ndarray.
    :param spectrum: The spectogram-crosssection to add the background to. np.ndarray.
    :param k: Parameter to control the background-level/amplitude. Float.
    :param sigma_weight: Parameter to control in which area under the peak to apply the background. Float.
    :param kwargs: Keyword-Arguments in case the peak positions (peak_pos : Array) is already known.
    """
    if 'peak_pos' not in kwargs:
        def gaussian(x, a, mu, sig): return a*np.exp(-1/2*((x-mu)/sig)**2)
        
        peak_idx=sp.signal.find_peaks(spectrum, prominence=0.01)[0]
        params_ls=[]
        for peak_id in peak_idx:
            mask=np.where(np.logical_and(energy[peak_id]-5<energy, energy<energy[peak_id]+5))[0]
            params, _ = sp.optimize.curve_fit(gaussian, energy[mask], spectrum[mask], p0=[spectrum[peak_id], energy[peak_id], 1])
            params_ls.append(params)

        b=[]
        for params in params_ls:
            mask=np.where(np.logical_and(params[1]-sigma_weight*np.abs(params[2])<energy, energy<params[1]+sigma_weight*np.abs(params[2])))[0]
            peak_background=-k*np.flip(sp.integrate.cumulative_trapezoid(np.flip(spectrum[mask]), np.flip(energy[mask]), initial=0))
            low_mask=np.where(energy<params[1]-3*params[2])[0]
            high_mask=np.where(params[1]+3*params[2]<energy)[0]
            background=np.zeros_like(energy)
            background[low_mask]=peak_background[0]
            background[high_mask]=peak_background[-1]
            background[mask]=peak_background
            b.append(background)
    else:
        for peak_idx, peak_pos in enumerate(kwargs['peak_pos']):
            mask=np.where(np.logical_and(peak_pos-3*np.abs(kwargs['peak_sigma'][peak_id])<energy, energy<peak_pos+sigma_weight*np.abs(kwargs['peak_sigma'][peak_idx])))[0]
            peak_background=-k*np.flip(sp.integrate.cumulative_trapezoid(np.flip(spectrum[mask]), np.flip(energy[mask]), initial=0))
            low_mask=np.where(energy<peak_pos-sigma_weight*kwargs['peak_sigma'][peak_idx])[0]
            high_mask=np.where(peak_pos+sigma_weight*kwargs['peak_sigma'][peak_idx]<energy)[0]
            background=np.zeros_like(energy)
            background[low_mask]=peak_background[0]
            background[high_mask]=peak_background[-1]
            background[mask]=peak_background
            b.append(background)
        
    return b


def slope(energy, spectrum, k):
    """
    Slope-background implementation for background simulation as described in 'The slope-background for the near-peak 
    regimen of photoemission spectra' (DOI 10.1016/j.elspec.2013.07.006). To use that for simulating background
    has been presented in the supplement to 'Deep learning in attosecond metrology' (DOR 10.1364/oe.452108).
    
    :param energy: The energy-interval of the simulated data. np.ndarray.
    :param spectrum: The spectogram-crosssection of the simulation. np.ndarray.
    :param k: Parameter to control the background-level/amplitude. Float.
    """
    return k*np.flip(sp.integrate.cumulative_trapezoid(sp.integrate.cumulative_trapezoid(np.flip(spectrum), np.flip(energy), initial=0), np.flip(energy), initial=0))


def shirley_vegh_salvi_castle_bg_to_spect(time_delay, energy, spectrogram, k, step_width):
    """
    Docstring für shirley_vegh_salvi_castle_bg_to_spect
    
    :param time_delay: Time delay axis of the spectrogram. np.ndarray.
    :param energy: Energy axis of the spectrogram. np.ndarray.
    :param spectrogram: Spectrogram to add the background to. np.ndarray.
    :param k: Parameter to control the background-level/amplitude. Float.
    :param step_width: Parameter to control in which area under the peak to apply the background. Float.
    """
    for i in range(0, len(time_delay)):
        svsc_bg=np.sum(shirley_vegh_salvi_castle(energy, spectrogram[i, :], k, step_width), axis=0)
        spectrogram[i, :]+=svsc_bg


def slope_bg_to_spect(time_delay, energy, spectrogram, k):
    """
    Function to add a slope background to the spectrogram.
    
    :param time_delay: Time delay axis of the spectrogram. np.ndarray.
    :param energy: Energy axis of the spectrogram. np.ndarray.
    :param spectrogram: Spectrogram to add the background to. np.ndarray.
    :param k: Parameter to control the background-level/amplitude. Float.
    """
    for i in range(0, len(time_delay)):
        slope_bg=slope(energy, spectrogram[i, :], k)
        spectrogram[i, :]+=slope_bg


def gaussian_laser_fluctuations_to_spect(time_delay, spectrogram, sigma):
    """
    Function to add noise stemming from laser fluctuations to the spectrogram.
    The noise is modled via a gaussian distribution.

    Parameters
    ----------
    :param time_delay: Time delay axis of the spectrogram. np.ndarray.
    :param spectrogram: Spectrogram to add the background to. np.ndarray.
    :param sigma: Sigma of the underlying gaussian distribution. Float.
    """
    for i in range(0, len(time_delay)):
        spectrogram[i, :]*=np.random.normal(1, sigma)


def generic_gaussian_multiplicative_noise_to_spect(spectrogram, sigma):
    """

    Parameters
    ----------
    :param spectogram: Beschreibung
    :param sigma: Beschreibung
    """
    spectrogram*=np.random.normal(1, sigma, spectrogram.shape)


def generic_poisson_additive_noise_to_spect(spectrogram, prefactor, l):
    """

    Parameters
    ----------
    :param spectrogram: Beschreibung
    :param prefactor: Beschreibung
    :param l: Beschreibung
    """
    spectrogram+=prefactor*np.random.poisson(l, spectrogram.shape)


def as_deep_learning(time_delay, energy, spectrogram, eps, SVSC={'k':0.002, 'sigma':0.01}, 
                     slope={'k':0.000035, 'sigma':0.01}, re_with_bg=False):
    """
    Parameters
    ----------    
    :param time_delay: Beschreibung
    :param energy: Beschreibung
    :param spectrogram: Beschreibung
    :param eps: Beschreibung
    :param SVSC: Beschreibung
    :param slope: Beschreibung
    :param re_with_bg: Beschreibung
    """
    spect = spectrogram.copy()

    shirley_vegh_salvi_castle_bg_to_spect(time_delay, energy, spect, SVSC['k']*np.random.normal(1, SVSC['sigma']), 3)
    slope_bg_to_spect(time_delay, energy, spect, slope['k']*np.random.normal(1, slope['sigma']))

    bg_spect = spect.copy()

    gaussian_laser_fluctuations_to_spect(time_delay, spect, 0.1)
    generic_gaussian_multiplicative_noise_to_spect(spect, 0.2*eps)
    generic_poisson_additive_noise_to_spect(spect, 0.04/7*eps, 7)

    spect /= np.max(spect)

    spect = np.where(spect < 0, 0, spect)

    if re_with_bg:
        return spect, bg_spect
    else:
        return spect
