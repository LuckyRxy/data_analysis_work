"""
This is the source code for volume visualization

Computer Vision Center
Universitat Autonoma de Barcelona

__author__ = "Debora Gil, Guillermo Torres, Carles Sanchez, Pau Cano"
__license__ = "GPL"
__email__ = "debora,gtorres,csanchez,pcano@cvc.uab.es"
__year__ = "2023"
"""

import os
import matplotlib

from public.SegmentationQualityScores import VOE, DICE, RelVolDiff

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import distance_transform_edt as bwdist
from skimage.measure import find_contours as contour
from skimage.filters import threshold_otsu
from skimage import morphology as Morpho
from scipy.ndimage import gaussian_filter as gfilt
from scipy.ndimage import median_filter as mfilt
from public.NiftyIO import readNifty
from public.VolumeCutBrowser import VolumeCutBrowser

######## LOAD DATA
SessionDataFolder=r"D:\Project\PythonProject\data_analysis\resources"
os.chdir(SessionDataFolder)

CaseFolder='VOIs'
NiiFile='LIDC-IDRI-0001_R_1.nii.gz'

NiiFile=os.path.join(SessionDataFolder,CaseFolder,'image',NiiFile)
niiROI,niimetada=readNifty(NiiFile)

######## SEGMENTATION PIPELINE

### 1. PRE-PROCESSING
# 1.1 Gaussian Filtering
sig=1
# niiROIGauss = gfilt.gaussian_filter(niiROI, sigma=sig)
niiROIGauss = gfilt(niiROI, sigma=sig)
# 1.2 MedFilter
sze=3
# niiROIMed = mfilt.median_filter(niiROI, sze)
niiROIMed = mfilt(niiROI, sze)

VolumeCutBrowser(niiROIGauss)

### 2. BINARIZATION (TH is Threshold)
Th = threshold_otsu(niiROIGauss)
niiROISeg=niiROIGauss>Th

VolumeCutBrowser(niiROISeg)

# ROI Histogram
fig,ax=plt.subplots()
ax.hist(niiROIGauss.flatten(),bins=50,edgecolor='k')
# Visualize Lesion Segmentation_Validation
VolumeCutBrowser(niiROI,IMSSeg=niiROISeg)


### 3.POST-PROCESSING

# 3.1  Opening
szeOp=4
se=Morpho.cube(szeOp)
niiROISegOpen = Morpho.binary_opening(niiROISeg, se)

# 3.2  Closing
szeCl=3
se=Morpho.cube(szeCl)
niiROISegClose = Morpho.binary_closing(niiROISeg, se)

VolumeCutBrowser(niiROISegOpen)

#  Volumetric Measures
print('---------------------------------Validation----------------------------------------')
SegVOE = VOE(niiROISegOpen, niiROISeg)
SegDICE = DICE(niiROISegOpen, niiROISeg)
SegRelDiff = RelVolDiff(niiROISegOpen, niiROISeg)

k = int(niiROIGauss.shape[2] / 2)  # Cut at the middle of the volume. Change k to get other cuts
SA = niiROIGauss[:, :, k]
SAGT = niiROISegOpen[:, :, k]
SASeg = niiROISeg[:, :, k]

SegVOE_SA = VOE(SASeg, SAGT)
SegDICE_SA = DICE(SASeg, SAGT)
SegRelDiff_SA = RelVolDiff(SASeg, niiROISeg)
print(f"SegVOE:{SegVOE}\nSegDICE:{SegDICE}\nSegRelDiff:{SegRelDiff}")
print(f"SegVOE_SA:{SegVOE_SA}\nSegDICE_SA:{SegDICE_SA}\nSegRelDiff_SA:{SegRelDiff_SA}")


# Distance Measures
print('------------------------------Distance Measures-------------------------------------')
# 1.Distance Map to Otsu segmentation SA cut
DistSegInt = bwdist(SASeg)  # Distance Map inside segmentation
DistSegExt = bwdist(1 - SASeg)  # Distance Map outside segmentation
DistSeg = np.maximum(DistSegInt, DistSegExt)  # Distance Map at all points

# 2.Distance from GT to Otsu segmentation
# GT Mask boundary points
BorderGT = contour(SAGT, 0.5)
i = BorderGT[0][:, 0].astype(int)
j = BorderGT[0][:, 1].astype(int)

# Show histogram
fig = plt.figure()
plt.hist(DistSeg[i, j], bins=50, edgecolor='k')
plt.title('Distance Measures')
plt.show()

# 3.3.3 Distance Scores
AvgDist, MxDist = DistScores(SASeg, SAGT)
print(f"AvgDist:{AvgDist}\nMxDist:{MxDist}")