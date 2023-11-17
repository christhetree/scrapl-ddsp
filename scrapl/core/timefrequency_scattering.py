import numpy as np
from kymatio.scattering1d.core.timefrequency_scattering import \
    time_averaging, frequency_averaging, time_formatting


def jtfs_singlepath(U_0, backend, filters, log2_stride, average_local,
        filters_fr, log2_stride_fr, average_local_fr, n2, n_fr):
    """
    Parameters
    ----------
    U_0 : array indexed by (batch, time)
    backend : module
    filters : [phi, psi1, psi2] list of dictionaries. same as scattering1d
    log2_stride : int >=0
        Yields coefficients with a temporal stride equal to 2**log2_stride.
    average_local : boolean
        whether to locally average the result by means of a low-pass filter phi.
    filters_fr : [phi, psis] list where
     * phi is a dictionary describing the low-pass filter of width F, used
       to average S1 and S2 in frequency if and only if average_local_fr.
     * psis is a list of dictionaries, each describing a low-pass or band-pass
       filter indexed by n_fr. The first element, n_fr=0, corresponds
       to a low-pass filter of width 2**J_fr and satisfies xi=0, i.e, spin=0.
       Other elements, such that n_fr>0, correspond to "spinned" band-pass
       filter, where spin denotes the sign of the center frequency xi.
    log2_stride_fr : int >=0
        yields coefficients with a frequential stride of 2**log2_stride_fr
        if average_local_fr (see below), otherwise 2**j_fr
    average_local_fr : boolean
        whether the result will be locally averaged with phi after this function
    n2 : int
        index of the temporal filter psi2
    n_fr : int
        index of the frequential filter psi_fr

    Yields
    ------
    path : dict with fields
        - 'coef': tensor indexed by (batch, n1[log2_T], time[log2_F]),
        complex-valued, where n1 has been zero-padded to size N_fr
        before convolution with psi_fr.
        - n, j, n_fr, j_fr, spin, n1_max, n1_stride: integers
        as in TimeFrequencyScattering
    """
    # compute the Fourier transform
    U_0_hat = backend.rfft(U_0)

    # First order
    psi1 = filters[1]
    U_1_hats = []
    for n1 in range(len(psi1)):
        # Convolution + downsampling
        j1 = psi1[n1]['j']
        k1 = min(j1, log2_stride) if average_local else j1
        U_1_c = backend.cdgmm(U_0_hat, psi1[n1]['levels'][0])
        U_1_hat = backend.subsample_fourier(U_1_c, 2 ** k1)
        U_1_c = backend.ifft(U_1_hat)

        # Take the modulus
        U_1_m = backend.modulus(U_1_c)
        U_1_hat = backend.rfft(U_1_m)
        
        # Width-first algorithm: store U1 in anticipation of next layer
        U_1_hats.append({'coef': U_1_hat, 'j': (j1,), 'n': (n1,)})

    # Second order.
    psi2 = filters[2]
    j2 = psi2[n2]['j']
    Y_2_list = []

    # Time scattering
    for U_1_hat in U_1_hats:
        j1 = U_1_hat['j'][0]
        k1 = min(j1, log2_stride) if average_local else j1

        if j2 > j1:
            k2 = min(j2-k1, log2_stride) if average_local else (j2-k1)
            U_2_c = backend.cdgmm(U_1_hat['coef'], psi2[n2]['levels'][k1])
            U_2_hat = backend.subsample_fourier(U_2_c, 2 ** k2)
            U_2_c = backend.ifft(U_2_hat)
            Y_2_list.append(U_2_c)

    # Stack Y_2_list along the n1 axis
    Y_2 = backend.stack(Y_2_list)
    n1_max = len(Y_2_list)

    # Swap time and frequency axis
    Y_2_T = backend.swap_time_frequency(Y_2)

    # Zero-pad frequency domain
    phi, psis = filters_fr
    pad_right = phi['N'] - n1_max
    Y_2_pad = backend.pad_frequency(Y_2_T, pad_right)

    # Frequency scattering
    Y_2_hat = backend.cfft(Y_2_pad)
    psi = psis[n_fr]
    j_fr = psi['j']
    spin = np.sign(psi['xi'])
    k_fr = min(j_fr, log2_stride_fr) if average_local_fr else j_fr
    Y_fr_hat = backend.cdgmm(Y_2_hat, psi['levels'][0])
    Y_fr_sub = backend.subsample_fourier(Y_fr_hat, 2 ** k_fr)
    Y_fr_T = backend.ifft(Y_fr_sub)
    Y_fr = backend.swap_time_frequency(Y_fr_T)

    return {'coef': Y_fr, 'n': (n2, n_fr), 'j': (-1, j2),
        'n_fr': (n_fr,), 'j_fr': (j_fr,), 'spin': spin,
        'n1_max': n1_max, 'n1_stride': (2**j_fr)}


def jtfs_singlepath_average_and_format(path, backend, phi_f, log2_stride,
        average, phi_fr_f, oversampling_fr, average_fr):

    # "The phase of the integrand must be set to a constant. This
    # freedom in setting the stationary phase to an arbitrary constant
    # value suggests the existence of a gauge boson" — Glinsky
    path['coef'] = backend.modulus(path['coef'])

    # Temporal averaging. Switch cases:
    # 1. If averaging is global, no need for unpadding at all.
    # 2. If averaging is local, averaging depends on order:
    #     2a. at order 1, U_gen yields
    #               Y_1_fr = S_1 * psi_{n_fr}
    #         no need for further averaging
    #     2b. at order 2, U_gen yields
    #               Y_2_fr = U_1 * psi_{n2} * psi_{n_fr}
    #         average with phi
    # (for simplicity, we assume oversampling=0 in the rationale above,
    #  but the implementation below works for any value of oversampling)
    if average == 'global':
        # Case 1.
        path['coef'] = backend.average_global(path['coef'])
    elif average == 'local' and len(path['n']) > 1:
        # Case 2b.
        path = time_averaging(path, backend, phi_f, log2_stride)

    # Frequential averaging. NB. if spin==0, U_gen is already averaged.
    # Hence, we only average if spin!=0, i.e. psi_{n_fr} in path.
    if average_fr and not path['spin'] == 0:
        path = frequency_averaging(path, backend,
            phi_fr_f, oversampling_fr, average_fr)

    # Splitting and reshaping
    return {**path, 'order': len(path['n'])}