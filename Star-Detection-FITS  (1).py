#!/usr/bin/env python
# coding: utf-8

# In[55]:


import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
from astropy.stats import sigma_clipped_stats
from photutils import aperture_photometry, CircularAperture
import numpy as np

# Load the FITS file
hdu = fits.open('hst_10342_03_acs_wfc_f555w_drz.fits')[1]
data = hdu.data

# Set a clean and professional style
plt.style.use('seaborn-whitegrid')

# Basic visualization
fig, ax = plt.subplots(figsize=(12, 10))
norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())
im = ax.imshow(data, norm=norm, cmap='twilight', origin='lower')
ax.set_title('M31 - Andromeda Galaxy')
cbar = plt.colorbar(im, label='Flux')
cbar.ax.yaxis.set_tick_params(color='black')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')
cbar.set_label('Flux (counts/s)', color='black')

# Radial profile
y, x = np.indices(data.shape)
center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
r_int = r.astype(int)
tbin = np.bincount(r_int.ravel(), data.ravel())
nr = np.bincount(r_int.ravel())
radialprofile = tbin / nr

plt.figure(figsize=(10, 6))
plt.plot(radialprofile, color='navy')
plt.title('Radial Profile of M31')
plt.xlabel('Radius (pixels)')
plt.ylabel('Average Flux')

# Simple photometry example
mean, median, std = sigma_clipped_stats(data, sigma=3.0)
threshold = median + (5.0 * std)
peaks = np.array(np.where(data > threshold)).T
apertures = CircularAperture(peaks, r=5.0)
phot_table = aperture_photometry(data, apertures)

fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(data, cmap='ocean', origin='lower', norm=norm)
ax.plot(peaks[:, 1], peaks[:, 0], 'r.', markersize=3)
ax.set_title('Detected Sources in M31')

plt.tight_layout()
plt.show()


# In[ ]:


#Add WCS(world Coordinate System)
fig = plt.figure()
ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], wcs=wcs)
fig.add_axes(ax)

# Apply LogNorm and 'ocean' colormap
norm = LogNorm()
ax.imshow(img, origin='lower', cmap='gray')
im = ax.imshow(img, origin='lower', norm=norm, cmap='ocean')

ax.set_xlabel('RA')
ax.set_ylabel('Dec')
ax.set_title('HST Image')

# Add colorbar with the same normalization and colormap
plt.colorbar(im, ax=ax)

plt.show()


# In[30]:


import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ZScaleInterval, AsinhStretch
from photutils.aperture import CircularAperture

# Remove seaborn styling
# sns.set_theme(style="darkgrid")

# Set up the plot style
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

def create_interactive_plot(fig, ax, title):
    ax.set_title(title)
    plt.tight_layout()

# Star detection
mean, median, std = sigma_clipped_stats(data, sigma=3.0)
daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
sources = daofind(data - median)

# Plot detected stars
fig, ax = plt.subplots(figsize=(12, 10))

# Create a mask for the actual image data
mask = np.isfinite(data)

# Create an RGBA image
rgba_img = np.zeros((data.shape[0], data.shape[1], 4))

# Normalize the data for display
interval = ZScaleInterval()
vmin, vmax = interval.get_limits(data)
norm_data = (data - vmin) / (vmax - vmin)

# Set the RGB channels to a blue colormap
rgba_img[..., 0] = 0  # Red channel
rgba_img[..., 1] = norm_data * 0.5  # Green channel (reduced intensity)
rgba_img[..., 2] = norm_data  # Blue channel
rgba_img[..., 3] = mask  # Alpha channel (1 where there's data, 0 otherwise)

# Display the image
ax.imshow(rgba_img, origin='lower')

# Plot detected stars
positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=4.)
apertures.plot(color='red', lw=1.5, alpha=0.7, ax=ax)

create_interactive_plot(fig, ax, f'Detected Stars in M31 (Total: {len(sources)})')
plt.show()

# The rest of your code (magnitude histogram, statistics) remains the same


# In[23]:


from matplotlib.colors import LogNorm


# In[53]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Generate sample data
def generate_sample_data(t):
    return np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.1 * np.random.randn(len(t))

# Time array
duration = 10  # seconds
sample_rate = 1000  # Hz
t = np.linspace(0, duration, duration * sample_rate, endpoint=False)

# Generate signal
signal = generate_sample_data(t)

# Perform FFT
n = len(t)
fft_result = fft(signal)
frequencies = fftfreq(n, 1 / sample_rate)

# Compute magnitude spectrum
magnitude_spectrum = np.abs(fft_result) / n

# Plot time domain signal
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Time Domain Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot frequency domain (magnitude spectrum)
plt.subplot(2, 1, 2)
plt.plot(frequencies[:n//2], magnitude_spectrum[:n//2])
plt.title('Frequency Domain - Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 50)  # Limit x-axis to 0-50 Hz for better visualization

plt.tight_layout()
plt.show()

# Find dominant frequencies
threshold = 0.1  # Adjust this threshold as needed
peaks = frequencies[np.where(magnitude_spectrum > threshold)]
print("Dominant frequencies:")
for peak in peaks:
    if peak > 0:  # Only print positive frequencies
        print(f"{peak:.2f} Hz")


# In[52]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# Set the plot style to a more traditional scientific look
#plt.style.use('seaborn-whitegrid')

# Function to load spectral data (replace this with your data loading method)
def load_spectral_data(filename):
    # This is a placeholder. In reality, you'd load your data from a file.
    # For this example, we'll generate synthetic IR data
    wavenumbers = np.linspace(4000, 400, 1000)
    transmittance = 100 - 5*np.exp(-((wavenumbers-3300)/50)**2) - \
                    10*np.exp(-((wavenumbers-1700)/30)**2) - \
                    15*np.exp(-((wavenumbers-1100)/40)**2)
    return wavenumbers, transmittance

# Load spectral data
wavenumbers, transmittance = load_spectral_data("your_spectrum_file.txt")

# Detect peaks (absorption bands)
peaks, _ = find_peaks(-transmittance, height=-90, distance=50)

# Create interpolation function for more precise peak location
interp_func = interp1d(wavenumbers, -transmittance, kind='quadratic')

# Refine peak locations
refined_peaks = []
for peak in peaks:
    x_range = np.linspace(wavenumbers[peak-5], wavenumbers[peak+5], 1000)
    y_range = interp_func(x_range)
    refined_peak = x_range[np.argmax(y_range)]
    refined_peaks.append(refined_peak)

# Plot the spectrum
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(wavenumbers, transmittance, color='navy')
ax.plot(wavenumbers[peaks], transmittance[peaks], "x", color='red')
ax.invert_xaxis()  # Invert x-axis as per IR convention
ax.set_xlabel('Wavenumber (cm⁻¹)')
ax.set_ylabel('Transmittance (%)')
ax.set_title('IR Spectrum with Detected Peaks')

# Annotate peaks
for peak in refined_peaks:
    ax.annotate(f'{peak:.0f}', xy=(peak, interp_func(peak)),
                 xytext=(0, 10), textcoords='offset points', ha='center',
                 color='red', fontweight='bold')

# Customize the plot
ax.set_facecolor('white')
fig.patch.set_facecolor('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# Print peak information
print("Detected absorption bands:")
for peak in refined_peaks:
    print(f"Wavenumber: {peak:.2f} cm⁻¹")

# Basic interpretation (very simplified)
def interpret_peak(wavenumber):
    if 3600 > wavenumber > 3200:
        return "O-H stretch"
    elif 3000 > wavenumber > 2850:
        return "C-H stretch"
    elif 1750 > wavenumber > 1650:
        return "C=O stretch"
    elif 1650 > wavenumber > 1450:
        return "C=C stretch"
    else:
        return "Other"

print("\nBasic interpretation:")
for peak in refined_peaks:
    print(f"{peak:.2f} cm⁻¹: {interpret_peak(peak)}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




