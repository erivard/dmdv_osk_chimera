#=================================================================================
#import

import numpy as np
import matplotlib.pyplot as plt 
import cv2
from skimage import filters
import os
from scipy.optimize import curve_fit 
import seaborn as sns
from itertools import combinations
from tifffile import imread
from scipy.ndimage import binary_erosion



#=================================================================================
#Define functions
def quantify_radial_intensity(IMG, Mask):
    dz = 2.0
    dx = 0.325
    osk = np.copy(IMG)

    # ---------- use mask as binary mask ----------
    BI = Mask.astype(int)

    # ---------- cropping using bounding box + fixed padding ----------
    ys, xs = np.where(BI > 0)

    buffer = 50  # adjust this to avoid clipping poles
    c = max(np.min(ys) - buffer, 0)
    d = min(np.max(ys) + buffer, IMG.shape[1])

    a_ = max(np.min(xs) - buffer, 0)
    b = min(np.max(xs) + buffer, IMG.shape[2])

    # ---------- crop ----------
    Osk_ = osk[:, c:d, a_:b]

    # ---------- rescale Z ----------
    nz, ny, nx = Osk_.shape
    nz_ = int(nz * (dz / dx))
    Osk = np.zeros((nz_, ny, nx), dtype=np.uint16)

    for i in range(ny):
        Osk[:, i, :] = cv2.resize(Osk_[:, i, :], dsize=(nx, nz_), interpolation=cv2.INTER_LINEAR)

    # ---------- radial projection ----------
    YZ = np.mean(Osk, axis=2)

    # ---------- threshold and mask ----------
    th = filters.threshold_otsu(YZ)
    B = np.zeros(YZ.shape)
    B[YZ > th] = 1

    # ---------- erosion + mean intensity ----------
    Vals = []
    for i in range(20):
        B = binary_erosion(B, iterations=2)
        if np.sum(B) == 0:
            Vals.append(np.nan)
        else:
            val = np.sum(B * YZ) / np.sum(B)
            Vals.append(val)

    # ---------- plot for QC ----------
    fig, ax = plt.subplots(ncols=2)
    ax[0].matshow(np.mean(Osk, axis=0))
    ax[1].matshow(YZ.T)
    plt.show()

    return YZ, B, Vals


def st_line(x,m,c):
    return (m*x + c)


def fit_st_line(x,y):
    popt, pcov = curve_fit(st_line,x,y)
    m,c = popt
    y_fit = m*x + c
    return(m,c,y_fit)

def sampwithrepl(dataset1, dataset2, samplenumb=1000):
    mu1 = np.mean(dataset1)
    mu2 = np.mean(dataset2)
    mudiff = mu1 - mu2
    mu_diff_sample_array = np.zeros(samplenumb)
    dataset1_len = len(dataset1)
    dataset2_len = len(dataset2)

    for i in range(samplenumb):
        sample1 = np.random.choice(dataset1, size=dataset1_len, replace=True)
        sample2 = np.random.choice(dataset2, size=dataset2_len, replace=True)
        mu_diff_sample_array[i] = np.mean(sample1) - np.mean(sample2)
    return mu_diff_sample_array, mudiff

def prettypvalue(pvalue):
    pvalstring = str(pvalue)
    splitatdot = pvalstring.split(".")
    if "e" in pvalstring:
        splitate = splitatdot[1].split("e")
        onlytwovalues = splitate[0][0:2]
        pvaluereport = splitatdot[0] + "." + onlytwovalues + "x10^" + splitate[1]
    else:
        onlytwovalues = splitatdot[1][0:5]
        pvaluereport = splitatdot[0] + "." + onlytwovalues
    return pvaluereport

def sampwithrepl_plot(dataset1, dataset2, samplenumb=100000):
    from scipy.stats import norm
    import matplotlib.pyplot as plt

    mu_diff_sample_array, mudiff = sampwithrepl(dataset1, dataset2, samplenumb)
    mu_mu_diff_sample_array = np.mean(mu_diff_sample_array)
    stdev = np.std(mu_diff_sample_array)
    conf_right = mu_mu_diff_sample_array + (2 * stdev)
    conf_left = mu_mu_diff_sample_array - (2 * stdev)

    # Two-tailed p-value
    cdf_zero = norm.cdf(0, loc=mu_mu_diff_sample_array, scale=stdev)
    pvalue = 1 - cdf_zero if mudiff <= 0 else cdf_zero

    # Plot histogram
    y, x = np.histogram(mu_diff_sample_array, bins=30)
    x_centers = 0.5 * (x[1:] + x[:-1])
    fig, ax = plt.subplots()
    ax.plot(x_centers, y)
    ax.axvline(x=mudiff, color='rebeccapurple', label='Observed Δμ')
    ax.axvline(x=conf_right, color='pink', linestyle='--', label='95% CI')
    ax.axvline(x=conf_left, color='pink', linestyle='--')
    ax.legend()
    plt.show()

    pvaluereport = prettypvalue(pvalue)
    print('_' * 50)
    print('    Δμ      CI95%(L)      CI95%(R)      p-value')
    print('_' * 50)
    print(f"  {np.round(mudiff,2)}    {np.round(conf_left,2)}    {np.round(conf_right,2)}    {pvaluereport}")
    print('_' * 50)
    return mudiff, conf_right, conf_left, pvalue

#=====================================================================================
#Set directory
base = 'embryos_HAloc_20250130/'
Files = os.listdir(base)
genotype_tags = ["Dm1234", "Dv1234", "OreR", "Dvir", "Dm123Dv4"]


# Define genotype categories and initialize output containers
genotype_outputs = {}

#Call the quantify radial intensity function
for tag in genotype_tags:
    matching_files = [f for f in Files if tag in f and f.endswith('.tif')]
    output = np.zeros((len(matching_files), 20))
    
    for i, tif_file in enumerate(matching_files):
        print(f"{tag} | {i+1}/{len(matching_files)}: {tif_file}")
        try:
            img_path = os.path.join(base, tif_file)
            mask_path = img_path.replace('.tif', '.npy')

            Img = imread(img_path)
            Mask = np.load(mask_path)

            YZ, B, Vals = quantify_radial_intensity(Img, Mask)
            output[i, :] = Vals
        except Exception as e:
            print(f"⚠️ Skipping {tif_file}: {e}")
    
    genotype_outputs[tag] = output
    np.save(f'{tag}_Out_erosion.npy', output)

#=====================================================================================
# Load data
slopes_dict = {}
data_dict = {}
genotype_tags = ["Dm1234", "Dv1234", "OreR", "Dvir", "Dm123Dv4"]


for tag in genotype_tags:
    data = np.load(f'{tag}_Out_erosion.npy')
    data_dict[tag] = data
    slopes = []

    X = np.arange(data.shape[1])
    for i in range(data.shape[0]):
        Y = data[i, :]
        if np.all(np.isnan(Y)) or np.nanmax(Y) == 0:
            print(f"⚠️ Skipping row {i} — empty or zero data")
            continue
        Y = Y - np.nanmin(Y)
        Y = Y / np.nanmax(Y)
        y = Y[np.isfinite(Y)]
        x = X[np.isfinite(Y)]
        m, c, y_ = fit_st_line(x, y)
        slopes.append(m)

    slopes_dict[tag] = slopes

#=====================================================================================
# Plot radial profiles
fig, axs = plt.subplots(ncols=len(genotype_tags), sharey=True, figsize=(10, 4))

for i, tag in enumerate(genotype_tags):
    ax = axs[i]
    ax.set_title(tag, fontsize=18)
    ax.set_xlabel('iter', fontsize=14)
    if i == 0:
        ax.set_ylabel('mean intensity', fontsize=14)
    data = data_dict[tag]
    X = np.arange(data.shape[1])
    for row in data:
        Y = row - np.nanmin(row)
        Y = Y / np.nanmax(Y)
        ax.plot(X, Y, '.-', alpha=0.6)

plt.tight_layout()
plt.show()

#=====================================================================================
# Plot slopes
sns.set_style("white")
fig, ax = plt.subplots(figsize=(4, 4))
# Boxplot
data_for_box = [slopes_dict[tag] for tag in genotype_tags]
ax.boxplot(data_for_box,
           widths=0.3,
           patch_artist=True,
           boxprops={'facecolor': 'white', 'edgecolor': 'black'},
           whiskerprops={'color': 'black'},
           capprops={'color': 'black'},
           medianprops={'color': 'black'})

# Jittered points
for i, tag in enumerate(genotype_tags, start=1):
    x_jitter = np.random.normal(i, 0.05, len(slopes_dict[tag]))
    ax.plot(x_jitter, slopes_dict[tag], '.', color='black')

# Formatting
ax.axhline(0, linestyle='--', color='silver')
ax.set_xticks(np.arange(1, len(genotype_tags)+1))
ax.set_xticklabels(genotype_tags, fontsize=14)
ax.set_ylabel('slope', fontsize=14)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1)

plt.savefig('Dmel_Dvir_OreR_Dm123Dv4_radial.png', bbox_inches='tight', dpi=600)
plt.show()

#=================================================================================
#Statistical analysis -- Pairwise bootstrap comparisons
genotype_tags = ["Dm1234", "Dv1234", "OreR", "Dvir", "Dm123Dv4"]
pairs = list(combinations(genotype_tags, 2))

for g1, g2 in pairs:
    print(f"\nBootstrapping comparison: {g1} vs {g2}")
    sampwithrepl_plot(slopes_dict[g1], slopes_dict[g2], samplenumb=100000)
    
    
