"""
Generate a fully synthetic test dataset for PulpBrain.

Nothing here is derived from real mill, pulp or spectrometer measurements:
every number is drawn from hand-written, literature-plausible ranges using a
fixed random seed. The files only reproduce the *structure* the importer
expects, so the whole pipeline (import -> MySQL -> prediction) can be tested.

Outputs (next to this script):
    raman/LRyy-nnn_YYMMDD-HHMMSS.csv   Wasatch WP785 (BarrelWise API) spectra:
                                       headerless, 1024 pixel intensities,
                                       3 replicate scans per sample
    pulpeye_sample.csv                 PulpEye-style export (positional
                                       columns, data from row 13)

Usage:
    python generate_sample_data.py [--samples 20] [--seed 42]
"""

import argparse
import csv
import datetime as dt
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

N_PIXELS = 1024
N_PE_COLS = 468                 # importer reads up to column 467
LEVELS = (50, 100, 150)         # refining levels per sample (kWh/t, in the sample name)
SCANS_PER_SAMPLE = 3
YEAR = 24
START = dt.datetime(2024, 6, 3, 9, 0, 0)

# Wasatch WP785 wavelength calibration (nm = C0 + C1*px + C2*px^2 + C3*px^3),
# the same constants the importer stores for BarrelWise files.
CCD = (801.451477051, 0.142020002007, -5.66349990549e-06, -8.55110027231e-09)
LASER_NM = 785.0

# Column order of PulpEye columns 6..44 (= the `targets` table after the two ID columns)
TARGET_COLS = ['Refining_Energy', 'CSF', 'Caliper', 'Bulk', 'Density', 'Air_resistance',
               'Smoothness', 'Brightness', 'Yellowness', 'Opacity', 'Scattering',
               'Absorption', 'L*', 'a*', 'b*', 'c*', 'h*', 'Tear', 'Burst', 'BL',
               'Stretch', 'Stiffness', 'TEA', 'TensileIndex', 'ZSBLwet', 'ZSBLdry',
               'artFL', 'LWFL', 'Coarseness_Ref', 'Fines_Arth', 'Fines_LW',
               'Curl_Arithmetic', 'Curl_LW', 'KinkIndex', 'TotKinkAngle', 'Kinkspermm',
               'Viscosity', 'WRV', 'Acetone_Extract']
FIBERSUMMARY_COLS = ['FiberOk', 'AnalyzedFibers', 'ValidFibers', 'NumOfFrames',
                     'FibersPerFrame', 'FiberConc', 'ArithmLen', 'LWeightLen', 'ArithmWid',
                     'LWeightWid', 'ArithmCurl', 'LWeightCurl', 'FinesShare'] \
    + [f'LenFrac{i}' for i in range(1, 6)] + [f'WidFrac{i}' for i in range(1, 6)] \
    + [f'CurlFrac{i}' for i in range(1, 6)] + ['Coarseness']
FWTDIST_COLS = ['MeasMode', 'SegmCounter', 'ImageCounter', 'FwtAvg', 'SegmWidthAvg',
                'SegmLengthAvg', 'WidFwtRatioAvg', 'CoarsenessAvg', 'IntensAvg'] \
    + [f'FwtDist{i}' for i in range(1, 31)] + ['FibrillAngle']
ARR_COLS = [f'ArrIndex{i}{s}' for s in ('', '_2', '_3', '_4') for i in range(50)]


def raman_shift_axis():
    """Raman shift (cm^-1) of every detector pixel."""
    px = np.arange(N_PIXELS)
    nm = CCD[0] + CCD[1] * px + CCD[2] * px ** 2 + CCD[3] * px ** 3
    return 1e7 * (1 / LASER_NM - 1 / nm)


def histogram(rng, n_bins, centre, width, jitter=0.04):
    """Smooth unimodal distribution that sums to 100 (percent)."""
    i = np.arange(n_bins)
    h = np.exp(-0.5 * ((i - centre) / width) ** 2) * rng.normal(1, jitter, n_bins).clip(0.5)
    return 100 * h / h.sum()


def fractions(rng, base):
    """Five size-class fractions that sum to 100."""
    f = np.asarray(base, float) * rng.normal(1, 0.05, len(base)).clip(0.5)
    return 100 * f / f.sum()


# --- Raman ---------------------------------------------------------------------------

# (Raman shift cm^-1, relative height, FWHM cm^-1): generic cellulose bands plus a
# lignin aromatic-ring band near 1600 cm^-1 (textbook positions, not fitted to any data).
CELLULOSE_BANDS = [(380, 1.00, 28), (437, 0.35, 20), (458, 0.20, 18), (520, 0.18, 26),
                   (898, 0.30, 26), (1095, 0.85, 24), (1121, 0.75, 20), (1150, 0.25, 22),
                   (1337, 0.30, 26), (1378, 0.45, 22), (1470, 0.20, 26)]
LIGNIN_BANDS = [(1270, 0.10, 30), (1600, 0.45, 32)]


def make_spectrum(rng, shift, lignin, fluorescence, scale):
    """One synthetic raw WP785 spectrum (ADC counts)."""
    x = np.linspace(0, 1, N_PIXELS)
    baseline = 900 + fluorescence * (0.45 + 0.9 * x - 0.55 * x ** 2)
    signal = np.zeros(N_PIXELS)
    for pos, h, fwhm in CELLULOSE_BANDS:
        signal += h * np.exp(-0.5 * ((shift - pos) / (fwhm / 2.355)) ** 2)
    for pos, h, fwhm in LIGNIN_BANDS:
        signal += lignin * h * np.exp(-0.5 * ((shift - pos) / (fwhm / 2.355)) ** 2)
    counts = baseline + scale * signal
    counts = counts + rng.normal(0, 1, N_PIXELS) * np.sqrt(counts) * 0.9 + rng.normal(0, 12, N_PIXELS)
    counts[:49] = rng.normal(250, 30, 49)       # dead pixels (the importer discards them)
    return counts


# --- PulpEye-style row ---------------------------------------------------------------

def excel_serial(t):
    return (t - dt.datetime(1899, 12, 30)).total_seconds() / 86400


def pulpeye_row(rng, sample_no, level, u, a, k, t):
    """Build one positional PulpEye row. u: refining (-1..1), a: fibre quality, k: lignin."""
    n = lambda mu, sd: float(rng.normal(mu, sd))                              # noqa: E731
    row = [''] * N_PE_COLS
    row[1] = f'LR{YEAR}-{sample_no:03d}-{level}'
    row[2] = 'DemoMill'

    # Paper properties (cols 6..44): generic, loosely physical trends only.
    bulk = n(1.58 - 0.08 * u + 0.015 * a, 0.01)
    bright = n(86 - 0.5 * u - 1.2 * k, 0.4)
    L = n(95.4 - 0.3 * k, 0.15)
    # Units follow what the bundled models predict (caliper in mm, density in g/cm3, ...)
    # so the Predicted-vs-Measured table is comparable.
    a_, b_ = n(-0.65, 0.04), n(1.55 + 0.15 * k, 0.05)
    t_ = dict(
        Refining_Energy=level, CSF=n(480 - 120 * u + 25 * a, 10),
        Caliper=bulk * 60 / 1000, Bulk=bulk, Density=1 / bulk,
        Air_resistance=max(1.0, n(25 * np.exp(0.9 * u), 2)), Smoothness=n(1200 - 250 * u, 40),
        Brightness=bright, Yellowness=n(2.3 + 0.4 * k, 0.1), Opacity=n(71 - 2 * u, 0.5),
        Scattering=n(30 - 2 * u, 0.4), Absorption=n(0.22 + 0.01 * k, 0.005),
        **{'L*': L, 'a*': a_, 'b*': b_}, **{'c*': float(np.hypot(a_, b_)),
                                          'h*': float(np.degrees(np.arctan2(b_, a_)) % 360)},
        Tear=n(18 - 2 * u + 0.6 * a, 0.5), Burst=n(4.5 + 0.9 * u + 0.2 * a, 0.1),
        BL=n(7 + 1.5 * u + 0.4 * a, 0.15), Stretch=n(3.3 + 0.3 * u, 0.06),
        Stiffness=n(7.0 - 0.5 * u, 0.3), TEA=n(1400 + 350 * u + 40 * a, 40),
        TensileIndex=n(70 + 12 * u + 3 * a, 1.2), ZSBLwet=n(12.3 + 0.2 * u, 0.2),
        ZSBLdry=n(13.2 + 0.4 * u + 0.2 * a, 0.2), artFL=n(2.4 - 0.05 * u, 0.03),
        LWFL=n(2.6 - 0.05 * u, 0.03), Coarseness_Ref=n(0.17, 0.005),
        Fines_Arth=n(12 + 4 * u, 0.5), Fines_LW=n(3 + 1 * u, 0.2),
        Curl_Arithmetic=n(0.12 + 0.01 * u, 0.004), Curl_LW=n(0.15 + 0.01 * u, 0.004),
        KinkIndex=n(1.1, 0.05), TotKinkAngle=n(45, 1.5), Kinkspermm=n(0.25, 0.01),
        Viscosity=n(800 - 40 * k, 12), WRV=n(1.2 + 0.1 * u, 0.02),
        Acetone_Extract=n(0.2, 0.02))
    for j, name in enumerate(TARGET_COLS):
        row[6 + j] = round(t_[name], 4)

    # PulpEye summary (cols 48..52 and 61..65)
    row[48] = 'DEMO sample'
    row[49] = 0
    row[50] = 'SYNTH-SW'
    row[51] = round(bright, 3)
    row[52] = round(t_['CSF'] + n(0, 6), 2)             # PulpEye CSF
    row[61:66] = [round(n(1.0, 0.05), 4), round(n(0.5, 0.03), 4), round(n(0.3, 0.02), 4),
                  round(n(28 + 4 * k, 0.5), 3), round(n(10, 0.4), 3)]
    row[71] = round(excel_serial(t), 6)

    # Kinks (175..178)
    row[175:179] = [round(n(0.6 + 0.05 * u, 0.02), 4), round(n(0.25, 0.01), 4),
                    round(n(0.9, 0.03), 4), round(n(45, 1.5), 3)]
    # Fibre width distribution (179..198)
    row[179:199] = np.round(histogram(rng, 20, 9 + a, 4), 4)
    # FWT results (199..238)
    fwt = [1, int(n(15000, 800)), int(n(300, 20)), n(2.8 - 0.1 * u, 0.05), n(28, 0.6),
           n(110, 3), n(10, 0.4), n(0.17, 0.005), n(150, 5)]
    fwt += list(histogram(rng, 30, 10 + a - 1.2 * u, 5)) + [n(10, 0.5)]
    row[199:239] = np.round(fwt, 4)
    # Fibre summary (239..267)
    fs = [1, int(n(20000, 500)), int(n(18000, 500)), int(n(400, 15)), int(n(45, 2)),
          n(1.0, 0.05), n(1.1 - 0.03 * u, 0.02), n(2.2 - 0.04 * u + 0.05 * a, 0.03),
          n(28, 0.5), n(30, 0.5), n(0.14 + 0.01 * u, 0.004), n(0.15 + 0.01 * u, 0.004),
          n(25 + 5 * u, 1.0)]
    fs += list(fractions(rng, [20 + 4 * u, 25, 25, 20 - 2 * u, 10 - 2 * u]))
    fs += list(fractions(rng, [15, 30, 30, 15, 10]))
    fs += list(fractions(rng, [40 - 3 * u, 30, 15, 10, 5 + 3 * u]))
    fs += [n(0.17, 0.005)]
    row[239:268] = np.round(fs, 4)
    # Fibre distributions (268..467): four 50-bin histograms
    for b, (c, w) in enumerate([(12, 7), (14, 8), (10, 6), (20, 10)]):
        row[268 + 50 * b:268 + 50 * (b + 1)] = np.round(histogram(rng, 50, c + 1.5 * a - 1.0 * u, w), 4)
    return row


def titles():
    t = [f'unused_{i}' for i in range(N_PE_COLS)]
    t[1], t[2] = 'SampleName', 'Mill'
    for j, name in enumerate(TARGET_COLS):
        t[6 + j] = name
    t[48:53] = ['SampleDesc', 'ErrData', 'PulpName', 'BrightValue', 'CsfAvg']
    t[61:66] = ['Sum1', 'Sum2', 'Sum3', 'Kappa', 'Cpa']
    t[71] = 'SampleTime_ExcelSerial'
    t[175:179] = ['KinksPerFiber', 'KinksPerLen', 'KinksSegmLen', 'KinksAngleAvg']
    t[179:199] = [f'WidthDist{i}' for i in range(1, 21)]
    t[199:239] = FWTDIST_COLS
    t[239:268] = FIBERSUMMARY_COLS
    t[268:468] = ARR_COLS
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--samples', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    raman_dir = os.path.join(HERE, 'raman')
    os.makedirs(raman_dir, exist_ok=True)
    for f in os.listdir(raman_dir):
        if f.endswith('.csv'):
            os.remove(os.path.join(raman_dir, f))

    shift = raman_shift_axis()
    pe_rows = []
    for s in range(1, args.samples + 1):
        a, k = rng.normal(), rng.normal()                        # fibre quality, lignin content
        t0 = START + dt.timedelta(days=(s - 1) * 1.3, minutes=int(rng.integers(0, 90)))

        # Raman: replicate scans a few seconds apart (the importer averages them)
        lignin = 0.55 + 0.25 * k
        fluor = 22000 * (1 + 0.25 * k) * (1 + 0.1 * rng.normal())
        scale = 4200 * (1 + 0.1 * rng.normal())
        for r in range(SCANS_PER_SAMPLE):
            ts = t0 + dt.timedelta(seconds=8 * r)
            spec = make_spectrum(rng, shift, max(lignin, 0.05), fluor, scale)
            name = f'LR{YEAR}-{s:03d}_{ts:%y%m%d-%H%M%S}.csv'
            np.savetxt(os.path.join(raman_dir, name), spec, fmt='%.2f')

        # PulpEye: one row per refining level, measured later the same day
        for li, level in enumerate(LEVELS):
            u = li - 1
            pe_rows.append(pulpeye_row(rng, s, level, u, a, k, t0 + dt.timedelta(hours=5, minutes=20 * li)))

    header = [[''] * N_PE_COLS for _ in range(11)]
    header[0][0] = 'SYNTHETIC PulpEye-style export - dummy data for testing PulpBrain, not real measurements'
    header[1][0] = 'Data rows start at row 13; column 2 holds LRyy-nnn-<refining level>'
    header[2][0] = 'Row 12 holds column titles (matching the PulpBrain database column names)'
    with open(os.path.join(HERE, 'pulpeye_sample.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerows(header)
        w.writerow(titles())
        w.writerows(pe_rows)
    print(f'Wrote {args.samples * SCANS_PER_SAMPLE} spectra to {raman_dir}')
    print(f'Wrote {len(pe_rows)} PulpEye rows to pulpeye_sample.csv')


if __name__ == '__main__':
    main()
