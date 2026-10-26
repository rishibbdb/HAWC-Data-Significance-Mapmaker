# HealpixSigFluxMap (standalone)

`HealpixSigFluxMap.py` is a single-file port of
`aerie/src/aerie-apps/analysis/map-making/HealpixSigFluxMap.cc`. It needs no AERIE,
ROOT, or install step. Run it with Python in the `astro_hal` conda environment.
It uses numpy, scipy, numba, healpy, astropy and uproot; hawc_hal is only needed
for HAL `.hd5` inputs.

```bash
source /Users/rishi/Documents/Analysis/miniconda3/bin/activate
conda activate astro_hal
cd /Users/rishi/Documents/Analysis/Sources/website/hawc/hawc_sigfluxmap
```

## Run directly (same options as `aerie-apps-HealpixSigFluxMap`)

```bash
python HealpixSigFluxMap.py \
    -i /Users/rishi/Documents/Analysis/data/nncoutmaps \
    -b B4C0Ef B5C0Ef B6C0Ef B6C0Eg B7C0Eg \
    -d /Users/rishi/Documents/Analysis/data/detRes-nn-pass5f_ML-refit.root \
    --index 2.63 --pivot 7 --extension 0.5 \
    --window 83.63 22.01 3 3 --negFlux --negSignif \
    -o crab_map.fits.gz
```

`python HealpixSigFluxMap.py -h` lists all options.

`-i` takes one of:

- a list of per-bin FITS files;
- a directory plus `-b`;
- a HAL map tree (`.root` or `.hd5`).

## From Python / threeML

```python
import sys
sys.path.insert(0, "/Users/rishi/Documents/Analysis/Sources/website/hawc/hawc_sigfluxmap")
from HealpixSigFluxMap import make_sig_flux_map

res = make_sig_flux_map(
    inputs="/Users/rishi/Documents/Analysis/data/nncoutmaps",
    bins=["B4C0Ef", "B5C0Ef", "B6C0Ef", "B6C0Eg", "B7C0Eg"],
    response="/Users/rishi/Documents/Analysis/data/detRes-nn-pass5f_ML-refit.root",
    index=2.63, pivot=7.0,          # or spectrum=<astromodels function>
    window=(83.63, 22.01, 3, 3),
    allow_negative_ts=True, allow_negative_flux=True,
)
print(res.value_at(83.63, 22.01, "significance"))
res.write("map.fits.gz")
```

If you pass an astromodels function as `spectrum`, the flux map is that function's
`K` in TeV⁻¹ cm⁻² s⁻¹. Remember astromodels' conventions: the index is negative and
`piv` is in keV.

The script's header docstring lists the differences from the C++ code.
