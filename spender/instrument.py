import os

import numpy as np
import torch
from torch import nn

class BaseInstrument(nn.Module):
    """Base class for instruments

    Container for wavelength vector, LSF and calibration functions.

    CAUTION:
    Don't base concrete implementations on this class, use :class:`Instrument` instead!

    Parameters
    ----------
    wave_obs: `torch.tensor`
        Observed wavelengths
    lsf: :class:`LSF`
        (optional) Line spread function model
    calibration: callable
        (optional) function to calibrate the observed spectrum
    """

    def __init__(
        self,
        wave_obs,
        lsf=None,
        calibration=None,
    ):

        super(BaseInstrument, self).__init__()

        self.calibration = calibration
        if lsf is not None:
            assert isinstance(lsf, (LSF, torch.Tensor))
            if isinstance(lsf, LSF):
                self.lsf = lsf
            else:
                self.lsf = LSF(lsf)
        else:
            self.lsf = None

        # register wavelength tensors on the same device as the entire model
        self.register_buffer("wave_obs", wave_obs)

    @property
    def name(self):
        return self.__class__.__name__


class LSF(nn.Conv1d):
    def __init__(self, kernel, requires_grad=True):
        super(LSF, self).__init__(1, 1, len(kernel), bias=False, padding="same")
        # if LSF should be fit, set `requires_grad=True`
        self.weight = nn.Parameter(
            kernel.flip(0).reshape(1, 1, -1), requires_grad=requires_grad
        )

    def forward(self, x):
        # convolution with flux preservation
        return super(LSF, self).forward(x) / self.weight.sum()

def _load_emission_lines():
    this_dir, _ = os.path.split(__file__)
    fn = os.path.join(this_dir, "data", "emission-lines.txt")
    sky = np.genfromtxt(fn,
                        names=["wavelength","intensity","name","status"],
                        dtype=None, encoding=None)
    sky["wavelength"] *= 10.0          # nm → Å
    return sky

_EMISSION_LINES = _load_emission_lines()

def get_emission_mask(
    wave_obs: torch.Tensor,
    z,
    min_intensity: float = 2.0,
    mask_size: float = 5.0,
) -> torch.Tensor:
    """
    Build a boolean mask that flags the expected positions of bright
    emission lines in *wave_obs*.

    Parameters
    ----------
    wave_obs : torch.Tensor, shape (Npix,)
        Wavelength grid in Å.
    z : torch.Tensor or float
        Redshift(s).  If a scalar, the mask is 1‑D (single spectrum).
        If an array of shape (Nspectra,), the mask is 2‑D
        (Nspectra, Npix).
    min_intensity : float
        Only lines brighter than this are considered.
    mask_size : float
        Minimum half‑width (in Å) to mask for a line with intensity == min_intensity.
        The width grows logarithmically with the line’s relative brightness.

    Returns
    -------
    torch.Tensor
        Boolean mask.  Shape matches the broadcasted shape of
        ``wave_obs`` and ``z`` (see notes below).
    """
    if isinstance(z, float) or (isinstance(z, torch.Tensor) and z.ndim == 0):
        # single spectrum → keep mask 1‑D
        mask = torch.zeros_like(wave_obs, dtype=torch.bool)
    else:
        # many spectra → 2‑D mask
        z = z.to(wave_obs.device)
        mask = torch.zeros((z.numel(), wave_obs.size(0)), dtype=torch.bool)

    lines = _EMISSION_LINES[_EMISSION_LINES["intensity"] > min_intensity]

    # Wave‑obs is 1‑D, but we can broadcast it against (Nspectra, Npix)
    # by unsqueezing the first axis if needed.
    wave_grid = wave_obs
    if mask.ndim == 2:                     # (Nspectra, Npix)
        wave_grid = wave_obs.unsqueeze(0)   # shape (1, Npix)

    for line in lines:
        # Size of the mask grows with intensity
        size = mask_size * (1.0 + np.log10(line["intensity"] / min_intensity))

        # Observed wavelength of the line for each redshift
        obs_wl = line["wavelength"] * (1.0 + z)   # shape: (Nspectra,) or scalar

        # If mask is 1‑D → obs_wl must be a scalar
        if mask.ndim == 1:
            # shape (Npix,)
            diff = torch.abs(wave_grid - obs_wl)   # scalar –> broadcast to (Npix,)
            mask |= diff < size
        else:
            # shape (Nspectra, Npix)
            diff = torch.abs(wave_grid - obs_wl[:, None])   # (Nspectra, Npix)
            mask |= diff < size

    return mask


def get_skyline_mask(wave_obs, min_intensity=2, mask_size=5):
    """Return vector that masks the major skylines

    For ever line in the skyline list in the file `data/sky-lines.txt` that is brighter
    than a threshold, this method creates a mask whose size scales logarithmically with
    line brightness.

    Parameter
    ---------
    wave_obs: `torch.tensor`
        Observed wavelengths
    min_intensity: float
        Intensity threshold
    mask_size: float
        Number of spectral elements to mask on either side of the line. This number
        is the minmum size for lines with `min_intensity`.
    Returns
    -------
    mask, `torch.tensor` of dtype `bool` with same shape as `wave_obs`
    """
    this_dir, this_filename = os.path.split(__file__)
    filename = os.path.join(this_dir, "data", "sky-lines.txt")
    skylines = np.genfromtxt(
        filename,
        names=["wavelength", "intensity", "name", "status"],
        dtype=None,
        encoding=None,
    )
    # wavelength in nm, need A
    skylines["wavelength"] *= 10

    mask = torch.zeros(len(wave_obs), dtype=torch.bool)
    for line in skylines[skylines["intensity"] > min_intensity]:
        # increase masking area with intensity
        mask_size_ = mask_size * (1 + np.log10(line["intensity"] / min_intensity))
        mask |= (wave_obs - line["wavelength"]).abs() < mask_size_
    return mask


# allow registry of new instruments
# see https://effectivepython.com/2015/02/02/register-class-existence-with-metaclasses
instrument_register = {}


def register_class(target_class):
    instrument_register[target_class.__name__] = target_class


class Meta(type):
    """Meta class to enable registration of instruments"""

    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        # remove those that are directly derived from the base class
        if BaseInstrument not in bases:
            register_class(cls)
        return cls


class Instrument(BaseInstrument, metaclass=Meta):
    """Instrument class

    Container for wavelength vector, LSF and calibration functions.

    See `spender.instrument.instrument_register` for all known classes that derive from
    :class:`Instrument`.
    """

    pass
