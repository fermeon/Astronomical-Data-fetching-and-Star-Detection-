#!/usr/bin/env python
# coding: utf-8

# #Reading FITS File

# In[1]:


#IMPORT "astropy.io" to work with astronomy data
from astropy.io import fits
import numpy as np


# In[2]:


# open and get information about the downloaded FITS file. 
hst=fits.open('hst_08597_11_wfpc2_f606w_pc_drz.fits')
hst.info()


# In[3]:


#read data from hst
img = hst[1].data
np.shape(img)


# In[4]:


#get header information
header=hst[1].header
header


# In[5]:


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.figure()
plt.imshow(img,norm= LogNorm(),cmap='twilight')
plt.colorbar()
plt.title('')
plt.xlabel('x-pixel')
plt.ylabel('y-pixel')
plt.show()


# In[6]:


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.figure()
plt.imshow(img,norm= LogNorm(),cmap='jet',origin='lower')
plt.colorbar()
plt.xlabel('x-pixel')
plt.ylabel('y-pixel')
plt.show()


# In[9]:


from astropy.wcs import WCS
wcs=WCS(hst[1].header)
wcs


# In[51]:


fig = plt.figure()
ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], wcs=wcs)
fig.add_axes(ax)

ax.imshow(img, origin='lower', cmap='gray')
plt.imshow(img,norm= LogNorm(),cmap='ocean',origin='lower')
ax.set_xlabel('RA')
ax.set_ylabel('Dec')
ax.set_title('HST Image')
plt.colorbar(im, ax=ax)
plt.show()


# In[59]:


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


# In[42]:


import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
from matplotlib.colors import LogNorm

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Create a normalization object with AsinhStretch
norm = ImageNormalize(img, interval=ZScaleInterval(), stretch=AsinhStretch(a=0.1))

# Plot the image with LogNorm and 'twilight' colormap
im = ax.imshow(img, norm=LogNorm(), cmap='twilight', origin='lower')

# Set the background color to black
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Set labels and title with white color
ax.set_xlabel('Pixel', color='white')
ax.set_ylabel('Pixel', color='white')
ax.set_title('Messier 31', color='white')

# Add a colorbar with the same colormap as the main image
cbar = plt.colorbar(im, ax=ax, label='Flux (counts/s)')
cbar.ax.yaxis.set_tick_params(color='white')
cbar.outline.set_edgecolor('white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
cbar.set_label('Flux (counts/s)', color='white')

# Set tick colors to white
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Add a white grid
ax.grid(color='white', ls='solid', alpha=0.3)

# Adjust layout and display
plt.tight_layout()
plt.show()


# In[49]:


from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# Create the divider for the existing axis
divider = make_axes_locatable(ax)

# Append a new axes for the histogram
ax_histx = divider.append_axes("top", size="20%", pad=0.1)

# Plot the histogram
hist_data = img.flatten()
hist_data = hist_data[np.isfinite(hist_data)]  # Remove any non-finite values
counts, bins, _ = ax_histx.hist(hist_data, bins=100, color='white', alpha=0.5)

# Set background color
ax_histx.set_facecolor('black')

# Set labels and ticks
ax_histx.set_ylabel('Counts')
ax_histx.yaxis.set_label_position("right")
ax_histx.tick_params(axis='x', labelbottom=False)
ax_histx.tick_params(colors='white')

# Use log scale if the data has a large dynamic range
ax_histx.set_yscale('log')

# Remove the x-axis label as it's redundant with the main plot
ax_histx.set_xlabel('')

# Ensure the x-axis limits match the main plot
ax_histx.set_xlim(ax.get_xlim())

# Add grid to histogram
ax_histx.grid(color='white', linestyle=':', alpha=0.3)

# Adjust the main plot's position
plt.subplots_adjust(hspace=0.1)
plt.figure()
plt.hist(img.flatten())
plt.yscale('log')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




