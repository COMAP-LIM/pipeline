import numpy as np, h5py, time
from pixell import enmap, utils
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u


def read_patch_file(file):
    patch_data = {}
    with open(file, "r") as infile:
        for line in infile.readlines()[:-1]:
            cols = line.split()
            if "x" in cols:
                continue
            else: 
                patch_data[cols[0]] = {
                    "ra": eval(cols[1]),
                    "dec": eval(cols[2]),
                    "image_radius": eval(cols[4]),
                    "resolution": eval(cols[5]),
                }
                
    return patch_data    

def make_standard_geometries(patch_data):

    for name, patch in patch_data.items():
        Nside_ra, Nside_dec = 120, 90
        field_center = np.array([patch["ra"], patch["dec"]])
        pix_res = patch["resolution"] / 60
        resolution = (np.array([
                                pix_res / np.abs(np.cos(np.radians(field_center[1]))),
                                pix_res, 
                            ]) 
                            )
        print(resolution, patch["resolution"])
        boundaries = np.array([[field_center[1] - Nside_dec / 2.0 * resolution[1], field_center[0] + Nside_ra / 2.0 * resolution[0]], 
                            [field_center[1] + Nside_dec / 2.0 * resolution[1], field_center[0] - Nside_ra / 2.0 * resolution[0]]])
        shape, wcs = enmap.geometry(boundaries * utils.degree, res = resolution[::-1] * utils.degree, proj = "car", force = True)

        omap = enmap.zeros(shape, wcs, np.float32)

        enmap.write_map(f"/mn/stornext/d5/data/nilsoles/nils/standard_geometries/{name}_standard_geometry_rectangular.fits", omap)


if __name__ == "__main__":
    patch_file = "/mn/stornext/d16/cmbco/comap/data/aux_data/patches_celestial.txt"

    patch_data = read_patch_file(patch_file)
    make_standard_geometries(patch_data)
