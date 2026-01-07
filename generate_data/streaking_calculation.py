import numpy as np
from numpy.fft import ifft, fftshift, ifftshift, fftfreq
from scipy.integrate import cumulative_trapezoid

E_h = 27.211385 # eV
t0  = 2.418884326505e-17 # sec

def energy_and_time(E_max, N):
    """
    Function to create Fourier pair of energy and time axis that are 
    used to calculate the spectrogram

    Parameters
    ----------
    :param E_max: Maximal energy for the energy axis [eV]
    :param N: Number of values for the energy axis 
    """
    omega = np.linspace(-E_max, E_max, N)/E_h
        
    t=fftshift(fftfreq(len(omega), np.abs(omega[1]-omega[0])/(2.*np.pi)))
    
    return omega, t


def E_x(omega, a,  w0, dw, delta, beta):
    """
    Function to get the XUV pulse / wave packet in time, synthesised from frequency domain.
    
    Parameters
    ----------
    :param omega: energy/freq. axis
    :param a: amplitude
    :param w0: central energy
    :param dw: spectral width
    :param delta: delay
    :param beta: chirp
    """
    out = fftshift(ifft(
            ifftshift(E_w(omega, a,  w0, dw, delta, beta)), norm = "ortho"
            ))
    return out


def E_w(omega, a,  w0, dw, delta, beta):
    """
    Function to get the XUV pulse / wave packet in frequency domain.
    
    Parameters
    ----------
    :param omega: energy/freq. axis
    :param a: amplitude
    :param w0: central energy
    :param dw: spectral width
    :param delta: delay
    :param beta: chirp
    """
    return a * np.exp(-4.*np.log(2)*(omega - w0)**2 / dw**2) * np.exp(-1.j * delta * (omega - w0))* np.exp(-.5j * beta * (omega - w0)**2)


def A_l(t, a,  omega, tau, beta, phi):
    """
    Function to get a Gaussian laser pulse in time domain.
    
    :param t: time axis
    :param a: amplitude
    :param omega: carrier frequency
    :param tau: duration
    :param beta: chirp
    :param phi: CE phase
    """
    return np.abs(a) * np.exp(-4.*np.log(2) * (t / tau) ** 2) * np.sin(omega * t + .5 * beta * t**2 + phi)


def SFASpectrogramDiscreteXUV(t,d,E, NIRfunc, NIRparams, XUV, I_p=0.):
    """
    Calculates the SPA spectrogram from NIRfunc discrete values for XUV based on the formula from
    Quantum Theory of Attosecond XUV Pulse Measurement by Laser Dressed Photoionization
    Markus Kitzler, Nenad Milosevic, Armin Scrinzi, Ferenc Krausz, and Thomas Brabec*
    Institut für Photonik, Technische Universität Wien, Gusshausstrasse 27/387, A-1040 Wien, Austria
    
    :param t: time axis
    :param d: delay axis
    :param E: Energy axis
    :param NIRfunc: NIR pulse function
    :param NIRparams: parameters for NIR pulse function
    :param XUV: XUV pulse train (discrete vaules, e.g. when ansatz is made in spectral domain). Has to be real.
    :param I_p: Ionization potential. defaults to 0.
    """
    spect = np.zeros((len(d), len(E)), dtype='complex64')
    for i, E_i in enumerate(E):
        for j, d_j in enumerate(d):
            #integrand=pulsetrain(t_fft-d_j, w0_XUV, tau_XUV, I_XUV, **harmonics_params)*exponential
            vphase       = .5 * (np.sqrt(2. * E_i) - NIRfunc(t - d_j, **NIRparams))**2
            integ_vphase = cumulative_trapezoid(vphase, t, initial=0)
            exponential  = np.exp(-1.j * integ_vphase + 1.j*I_p*t)
            integrand    = XUV * exponential
            spect[j][i]  = 1.j * np.trapz(integrand, t)
    return spect


def create_spectogram(xuv_pulse_energy : float,
                      binding_energies : list,
                      time_delays : list,
                      amplitudes : list, 
                      spectral_widths : list,
                      chirps : list, 
                      x_axis : np.ndarray = None,
                      y_axis : np.ndarray = None):
    """
    Docstring für create_spectogram
    
    :param xuv_pulse_energy: Energy of the XUV pulse [eV]
    :type xuv_pulse_energy: float
    :param binding_energies: Binding energies of the target [eV]
    :type binding_energies: list
    :param time_delays: Delays between the outgoing wavepackets [s]
    :type time_delays: list
    :param amplitudes: (Normed) amplitudes of the outgoing wavepackets [s]
    :type amplitudes: list
    :param spectral_widths: Spectral width of the outgoing wavepackets [eV]
    :type spectral_widths: list
    :param chirps: Chirp of the outgoing wavepackets [s-2]
    :type chirps: list
    :param x_axis: X-axis (time delay axis) for which to plot the spectrogram [fs]
    :type x_axis: np.ndarray
    :param y_axis: Y-axis (energy axis) for which to plot the spectrogram [eV]
    :type y_axis: np.ndarray
    """
    
    if y_axis is not None:
        E_axis=y_axis
    else:
        E_axis=np.linspace(20, 140, 800)
    if x_axis is not None:
        tau_axis=x_axis
    else:
        tau_axis=np.linspace(-3, 3, 40)
    
    w_xuv=xuv_pulse_energy/E_h
    omega, t=energy_and_time(220, 1000)

    xuv_pulses=[]
    for binding_energy, time_delay, amplitude, spectral_width, chirp in zip(binding_energies, 
                                                                            time_delays, 
                                                                            amplitudes, 
                                                                            spectral_widths, 
                                                                            chirps):
        xuv_pulses.append(E_x(omega, amplitude, w_xuv-binding_energy/E_h, spectral_width/E_h, time_delay/t0, chirp))
    xuv=np.sum(np.array(xuv_pulses), axis=0)

    NIR=dict(a=0.05, omega=1.76/E_h, tau=4.97e-15/t0, beta=0, phi=7.46e-01*np.pi)

    spectrogram=np.abs(SFASpectrogramDiscreteXUV(t, tau_axis*1e-15/t0, E_axis/E_h, A_l, NIR, xuv))**2

    return E_axis, tau_axis, spectrogram/np.max(spectrogram)