import h5py
import numpy as np
import numpy.typing as npt
from typing import Dict
import os
import sys
import random
import matplotlib.pyplot as plt
import ctypes
import time
import warnings

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


def read_map(feed: int) -> Dict[str, npt.ArrayLike]:
    # l2path = "/mn/stornext/d22/cmbco/comap/protodir/level2/Ka/co7/"
    # l2files = os.listdir(l2path)
    # path = l2path + random.choice(l2files)

    path = "/mn/stornext/d22/cmbco/comap/protodir/level2/Ka/co6/co6_002083603.h5"
    with h5py.File(path, "r") as infile:
        pixels = infile["pixels"][()]
        pixels = np.where(pixels == feed)[0][0]

        ra = infile["point_cel"][pixels, :, 0]
        dec = infile["point_cel"][pixels, :, 1]
        tod = infile["tod"][pixels, :, :, :]
        sigma = infile["sigma0"][pixels, :, :]

    tod[(0, 1), :, :] = tod[(0, 1), ::-1, :]
    sigma[(0, 1), :] = sigma[(0, 1), ::-1]

    return {
        "ra": ra,
        "dec": dec,
        "tod": tod,
        "sigma": sigma,
    }


def read_map_allfeed() -> Dict[str, npt.ArrayLike]:
    path = "/mn/stornext/d22/cmbco/comap/protodir/level2/Ka/co6/co6_002083603.h5"
    with h5py.File(path, "r") as infile:
        pixels = infile["pixels"][()] - 1

        _ra = infile["point_cel"][:, :, 0]
        _dec = infile["point_cel"][:, :, 1]
        _tod = infile["tod"][()]
        _sigma = infile["sigma0"][()]
        _freqmask = infile["freqmask"][()]

    _tod[:, (0, 1), :, :] = _tod[:, (0, 1), ::-1, :]
    _sigma[:, (0, 1), :] = _sigma[:, (0, 1), ::-1]
    _freqmask[:, (0, 1), :] = _freqmask[:, (0, 1), ::-1]

    tod = np.zeros((20, 4, 64, _tod.shape[-1]))
    sigma = np.zeros((20, 4, 64))
    freqmask = np.zeros((20, 4, 64))
    ra = np.zeros((20, _ra.shape[-1]))
    dec = np.zeros((20, _dec.shape[-1]))

    tod[pixels] = _tod
    sigma[pixels] = _sigma
    freqmask[pixels] = _freqmask
    ra[pixels] = _ra
    dec[pixels] = _dec
    print(tod.shape, _tod.shape)
    return {
        "ra": ra,
        "dec": dec,
        "tod": tod,
        "sigma": sigma,
        "freqmask": freqmask,
    }


def get_pointing_matrix(ra: npt.ArrayLike, dec: npt.ArrayLike) -> npt.ArrayLike:
    # Read these from file in future
    FIELDCENT = [226.00, 55.00]
    DPIX = 2 / 60
    NSIDE = 120
    NPIX = NSIDE * NSIDE

    # RA/Dec grid
    RA = np.zeros(NSIDE)
    DEC = np.zeros(NSIDE)
    dRA = DPIX / np.abs(np.cos(np.radians(FIELDCENT[1])))
    dDEC = DPIX

    # Min values in RA/Dec. directions
    if NSIDE % 2 == 0:
        RA_min = FIELDCENT[0] - dRA * NSIDE / 2.0
        DEC_min = FIELDCENT[1] - dDEC * NSIDE / 2.0

    else:
        RA_min = FIELDCENT[0] - dRA * NSIDE / 2.0 - dRA / 2.0
        DEC_min = FIELDCENT[1] - dDEC * NSIDE / 2.0 - dDEC / 2.0

    print(RA_min, DEC_min)

    # Defining piRAel centers
    RA[0] = RA_min + dRA / 2
    DEC[0] = DEC_min + dDEC / 2

    for i in range(1, NSIDE):
        RA[i] = RA[i - 1] + dRA
        DEC[i] = DEC[i - 1] + dDEC

    RA_min, DEC_min = RA[0], DEC[0]

    idx_ra = np.round((ra - RA_min) / dRA).astype(np.int32)
    idx_dec = np.round((dec - DEC_min) / dDEC).astype(np.int32)

    idx_ra_allfeed = np.round((ra - RA_min) / dRA).astype(np.int32)
    idx_dec_allfeed = np.round((dec - DEC_min) / dDEC).astype(np.int32)
    NFEED = idx_ra_allfeed.shape[0]

    idx_pix = idx_dec_allfeed * NSIDE + idx_ra_allfeed

    print(np.any(idx_pix < 0), np.any(idx_pix > NPIX))
    idx_allfeed = NSIDE**2 * np.arange(NFEED)[:, None] + idx_pix

    # sys.exit()
    # idx_allfeed = (
    #     NFEED * (idx_dec_allfeed * NSIDE + idx_ra_allfeed) + np.arange(NFEED)[:, None]
    # )

    # idx_3d = (
    #     256 * (idx_dec[:, None] * NSIDE + idx_ra[:, None]) + np.arange(256)[None, :]
    # )

    idx = idx_dec * NSIDE + idx_ra

    mask = ~np.logical_or(idx < 0, idx >= NPIX)
    mask = np.where(mask)[0]

    # mask_3d = ~np.logical_or(idx_3d < 0, idx_3d >= NPIX * 256)
    # mask_3d = np.where(mask_3d)[0]
    return idx, mask, idx_ra, idx_dec, idx_ra_allfeed, idx_dec_allfeed, idx_allfeed
    # return idx, idx_3d, mask, mask_3d


def bin_map(data, idx, mask, NSIDE):
    tod = data["tod"][..., mask]
    sigma = data["sigma"]

    inv_var = np.ones_like(tod) / sigma[..., None] ** 2
    nanmask = ~np.isfinite(inv_var)

    tod[nanmask] = 0.0
    inv_var[nanmask] = 0.0

    sidebands, channels, _samples = tod.shape

    numerator = np.zeros((sidebands, channels, NSIDE * NSIDE))
    denominator = np.zeros_like(numerator)
    hits = np.ones(denominator.shape, dtype=np.int32)

    for sb in range(sidebands):
        for freq in range(channels):
            hits[sb, freq] = np.bincount(idx, minlength=NSIDE * NSIDE)

            numerator[sb, freq, :] = np.bincount(
                idx,
                minlength=NSIDE * NSIDE,
                weights=tod[sb, freq, ...] * inv_var[sb, freq, ...],
            )

            denominator[sb, freq, :] = np.bincount(
                idx, minlength=NSIDE * NSIDE, weights=inv_var[sb, freq, ...]
            )

    return numerator, denominator, hits


def bin_map_vectorized(data, idx, mask, NSIDE, NFREQ):
    NSAMP = data["tod"].shape[-1]
    idx = idx.flatten()

    tod = data["tod"].transpose(2, 0, 1)

    sigma = data["sigma"]

    inv_var = np.ones_like(tod) / sigma[None, ...] ** 2
    nanmask = ~np.isfinite(inv_var)

    tod[nanmask] = 0.0
    inv_var[nanmask] = 0.0

    tod = tod.reshape(NFREQ * NSAMP)
    inv_var = inv_var.reshape(NFREQ * NSAMP)

    tod[mask] = 0
    inv_var[mask] = 0

    hits = np.bincount(idx, minlength=NSIDE * NSIDE * NFREQ)

    numerator = np.bincount(
        idx,
        minlength=NSIDE * NSIDE * NFREQ,
        weights=tod * inv_var,
    )

    denominator = np.bincount(idx, minlength=NSIDE * NSIDE * NFREQ, weights=inv_var)

    return numerator, denominator, hits


def bin_map_c(tod, sigma, freqmask, idx, NSIDE, nthread):
    # t0 = time.perf_counter()

    float32_array4 = np.ctypeslib.ndpointer(
        dtype=ctypes.c_float, ndim=4, flags="contiguous"
    )  # 4D array 32-bit float pointer object.
    float32_array3 = np.ctypeslib.ndpointer(
        dtype=ctypes.c_float, ndim=3, flags="contiguous"
    )  # 4D array 32-bit float pointer object.
    float32_array2 = np.ctypeslib.ndpointer(
        dtype=ctypes.c_float, ndim=2, flags="contiguous"
    )  # 4D array 32-bit float pointer object.
    float32_array1 = np.ctypeslib.ndpointer(
        dtype=ctypes.c_float, ndim=1, flags="contiguous"
    )  # 4D array 32-bit float pointer object.

    int32_array4 = np.ctypeslib.ndpointer(
        dtype=ctypes.c_int, ndim=4, flags="contiguous"
    )  # 4D array 32-bit integer pointer object.
    # int32_array3 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=3, flags="contiguous")       # 4D array 32-bit integer pointer object.
    int32_array2 = np.ctypeslib.ndpointer(
        dtype=ctypes.c_int, ndim=2, flags="contiguous"
    )  # 4D array 32-bit integer pointer object.
    # int32_array1 = np.ctypeslib.ndpointer(
    #     dtype=ctypes.c_int, ndim=1, flags="contiguous"
    # )  # 4D array 32-bit integer pointer object.

    # mapbinner.bin_freq_map.argtypes = [
    # mapbinner.add_tod.argtypes = [
    mapbinner.add_tod2map.argtypes = [
        float32_array3,  # tod
        float32_array2,  # sigma
        float32_array2,  # sigma
        int32_array2,  # idx_pix
        int32_array4,  # nhit map
        float32_array4,  # numerator map
        float32_array4,  # denominator map
        ctypes.c_int,  # nfreq
        ctypes.c_int,  # nsamp
        ctypes.c_int,  # nside
        ctypes.c_int,  # nfeed
        ctypes.c_int,  # nthread
    ]

    NFEED, NSAMP, NFREQ = tod.shape
    # print("Time ctypes pointer setup", time.perf_counter() - t0)
    # t0 = time.perf_counter()

    # nhit = np.zeros((NSIDE, NSIDE, NFEED, NFREQ), dtype=ctypes.c_int)
    # numerator = np.zeros((NSIDE, NSIDE, NFEED, NFREQ), dtype=ctypes.c_float)
    # denominator = np.zeros((NSIDE, NSIDE, NFEED, NFREQ), dtype=ctypes.c_float)
    nhit = np.zeros((NFEED, NSIDE, NSIDE, NFREQ), dtype=ctypes.c_int)
    numerator = np.zeros((NFEED, NSIDE, NSIDE, NFREQ), dtype=ctypes.c_float)
    denominator = np.zeros((NFEED, NSIDE, NSIDE, NFREQ), dtype=ctypes.c_float)
    # print("Time matrix setup", time.perf_counter() - t0)
    # t0 = time.perf_counter()

    # print(NSAMP, NFEED, NFREQ)
    # print(tod.shape)
    # print(sigma.shape)
    # print(idx.shape)
    # print(nhit.size)

    # mapbinner.bin_freq_map(
    # mapbinner.add_tod(
    mapbinner.add_tod2map(
        tod,
        sigma,
        freqmask,
        idx,
        nhit,
        numerator,
        denominator,
        NFREQ,
        NFEED,
        NSAMP,
        NSIDE,
        nthread,
    )
    # print("Time C call", time.perf_counter() - t0)
    # t0 = time.perf_counter()
    # print(np.all(_nhit == 0))
    # print(np.all(nhit == 0))
    # plotdata = numerator / denominator
    # # print(np.where(np.isfinite(plotdata)))
    # fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    # ax[0].imshow(
    #     np.nanmean(plotdata[..., 4], axis=0),
    #     # plotdata[0, ..., 4],
    #     origin="lower",
    #     vmin=-np.nanstd(np.nanmean(plotdata[..., 4], axis=0)),
    #     vmax=np.nanstd(np.nanmean(plotdata[..., 4], axis=0)),
    #     # vmin=-np.nanstd(plotdata[0, ..., 4]),
    #     # vmax=np.nanstd(plotdata[0, ..., 4]),
    #     cmap="RdBu_r",
    #     interpolation="none",
    # )
    # img = ax[1].imshow(
    #     np.nanmean(plotdata[..., 249], axis=0),
    #     # plotdata[0, ..., 249],
    #     origin="lower",
    #     vmin=-np.nanstd(np.nanmean(plotdata[..., 249], axis=0)),
    #     vmax=np.nanstd(np.nanmean(plotdata[..., 249], axis=0)),
    #     # vmin=-np.nanstd(plotdata[0, ..., 249]),
    #     # vmax=np.nanstd(plotdata[0, ..., 249]),
    #     cmap="RdBu_r",
    #     interpolation="none",
    # )
    # plt.colorbar(img, ax=ax[1])
    # plt.show()
    # sys.exit()
    return nhit, numerator, denominator


def main():
    import timeit

    repetitions = 50
    # repetitions = 1

    # runtime = timeit.repeat(
    #     "read_map(10)", number=100, repeat=repetitions, globals=globals()
    # )

    # runtime = np.array(runtime) / 100

    # print("Reading map data:")
    # print(f"Runtime: mean = {runtime.mean()} s ± {runtime.std()} s \n")

    # runtime = timeit.repeat(
    #     "read_map_allfeed()", number=100, repeat=repetitions, globals=globals()
    # )

    # runtime = np.array(runtime) / 100

    # print("Reading map data all feed:")
    # print(f"Runtime: mean = {runtime.mean()} s ± {runtime.std()} s \n")

    # runtime = timeit.repeat(
    #     time_pointing_matrix,
    #     number=100,
    #     globals=globals(),
    # )

    # runtime = np.array(runtime) / 100

    # print("Pointing matrix setup:")
    # print(f"Runtime: mean = {runtime.mean()} s ± {runtime.std()} s \n")

    # runtime = timeit.repeat(
    #     time_bin_map,
    #     number=100,
    #     repeat=repetitions,
    #     globals=globals(),
    # )

    # runtime = np.array(runtime) / 100

    # print("Binning maps for all frequencies:")
    # print(f"Runtime: mean = {runtime.mean()} s ± {runtime.std()} s \n")

    # runtime = np.array(runtime) / (100 * 256)

    # print("Binning maps per frequencies:")
    # print(f"Runtime: mean = {runtime.mean()} s ± {runtime.std()} s \n")

    # runtime = timeit.repeat(
    #     time_bin_map_vectorized,
    #     number=100,
    #     repeat=repetitions,
    #     globals=globals(),
    # )

    # runtime = np.array(runtime) / 100

    # print("Binning maps for all frequencies (vecorized):")
    # print(f"Runtime: mean = {runtime.mean()} s ± {runtime.std()} s \n")
    times = []
    time_error = []
    # threads = np.arange(1, 50, 1)
    threads = [1]
    global thread
    for thread in threads:
        # os.environ["OMP_NUM_THREADS"] = f"{thread}"
        print("OMP_NUM_THREADS: ", thread)  # , os.environ["OMP_NUM_THREADS"])
        runtime = timeit.repeat(
            time_bin_map_c,
            number=1,
            repeat=repetitions,
            globals=globals(),
        )

        runtime = np.array(runtime)
        times.append(runtime.mean())
        time_error.append(runtime.std())

        print("Binning maps for all frequencies (c):")
        print(f"Runtime: mean = {runtime.mean()} s ± {runtime.std()} s \n")

    # fig, ax = plt.subplots(figsize=(10, 8))
    # ax.errorbar(threads, times, time_error, fmt="")
    # ax.scatter(threads, times)

    # ax.plot(threads, times[0] / threads, "r--")

    # ax.set_xlabel("OMP_NUM_THREADS")
    # ax.set_ylabel("runtime [s]")
    # plt.show()


if __name__ == "__main__":
    mapbinner = ctypes.cdll.LoadLibrary("mapbinner.so.1")

    # data = read_map(10)
    data_allfeeds = read_map_allfeed()

    # print("Setup pointing")
    # idx, idx_3d, mask, mask_3d = get_pointing_matrix(data["ra"], data["dec"])
    (
        idx,
        mask,
        idx_ra,
        idx_dec,
        idx_ra_allfeeds,
        idx_dec_allfeeds,
        idx_allfeeds,
    ) = get_pointing_matrix(data_allfeeds["ra"], data_allfeeds["dec"])

    # print("Binning 3d map")
    # # bin_map(data, idx, mask, 120)
    # # bin_map_vectorized(data, idx_3d, mask_3d, 120, 256)

    # def time_pointing_matrix():
    #     get_pointing_matrix(data["ra"], data["dec"])

    # def time_bin_map():
    #     bin_map(data, idx, mask, 120)

    # def time_bin_map_vectorized():
    #     bin_map_vectorized(data, idx_3d, mask_3d, 120, 256)

    # tod = data_allfeeds["tod"]
    # sigma = data_allfeeds["sigma"]
    # freqmask = data_allfeeds["freqmask"]
    # nfeed, nsb, nfreq, nsamp = tod.shape
    # tod = tod.transpose(3, 2, 0, 1)
    # tod = tod.reshape(nsamp, nfeed, nsb * nfreq)
    # sigma = sigma.reshape(nfeed, nsb * nfreq)
    # freqmask = freqmask.reshape(nfeed, nsb * nfreq)
    # idx_allfeeds = idx_allfeeds.T

    tod = data_allfeeds["tod"]

    sigma = data_allfeeds["sigma"]
    freqmask = data_allfeeds["freqmask"]
    nfeed, nsb, nfreq, nsamp = tod.shape
    tod = tod.transpose(
        0,
        3,
        1,
        2,
    )
    tod = tod.reshape(nfeed, nsamp, nsb * nfreq)
    sigma = sigma.reshape(nfeed, nsb * nfreq).T
    freqmask = freqmask.reshape(nfeed, nsb * nfreq).T
    idx_allfeeds = idx_allfeeds

    tod = np.ascontiguousarray(tod, dtype=np.float32)
    sigma = np.ascontiguousarray(sigma, dtype=np.float32)
    freqmask = np.ascontiguousarray(freqmask, dtype=np.float32)
    idx_allfeeds = np.ascontiguousarray(idx_allfeeds, dtype=np.int32)

    #### PRE-COMP MASKING ####

    sigma = 1 / sigma**2
    sigma[freqmask == 0] = 0
    print(np.sum(freqmask == 0) / freqmask.size)
    # mask = ~np.logical_and(idx_allfeeds < 0, idx_allfeeds > 120 * 120 * nfeed)
    # mask_feed, mask_samp = np.where(mask)
    # tod[mask_feed, mask_samp, :] = 0
    # print(tod[mask_feed, mask_samp, :].shape)
    # sys.exit()
    print(np.any(idx_allfeeds < 0))
    # ##########################

    # thread = 24
    # t0 = time.perf_counter()
    # bin_map_c(tod, sigma, freqmask, idx_allfeeds, 120, thread)
    # print("Time python call of ctypes func", time.perf_counter() - t0)

    def time_bin_map_c():
        bin_map_c(tod, sigma, freqmask, idx_allfeeds, 120, thread)

    main()

    # np.random.seed(12345)
    # idx_ra = np.random.randint(0, 12, 15)
    # idx_dec = np.random.randint(0, 12, 15)
    # print(idx_dec, idx_ra)
    # idx_nu = np.arange(5)

    # idx_2d = 12 * idx_dec + idx_ra
    # idx_3d = 5 * idx_2d[:, None] + idx_nu[None, :]

    # # for i in idx_ra:
    # #     # for i in range(15):
    # #     for j in idx_dec:
    # #         # for j in range(15):
    # #         for k in range(5):
    # #             print(5 * (15 * i + j) + k)

    # # print(idx_3d)

    # fig, ax = plt.subplots()
    # ax.plot(idx_ra[0], idx_dec[0], "ro")
    # ax.plot(idx_ra[-1], idx_dec[-1], "bo")
    # ax.plot(idx_ra, idx_dec)

    # hits = np.bincount(idx_3d.flatten(), minlength=(5 * 12**2))
    # print(hits.size / (12 * 5))
    # hits = hits.reshape(12, 12, 5, order="C")

    # fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    # print(ax)
    # ax[0].imshow(hits[..., 0], cmap="RdBu_r", origin="lower")
    # img = ax[1].imshow(hits[..., -1], cmap="RdBu_r", origin="lower")

    # # plt.colorbar(img, ax=ax[1])

    # ax[0].plot(idx_ra[0], idx_dec[0], "ro")
    # ax[0].plot(idx_ra[-1], idx_dec[-1], "bo")
    # ax[0].plot(idx_ra, idx_dec, "g")

    # ax[1].plot(idx_ra[0], idx_dec[0], "ro")
    # ax[1].plot(idx_ra[-1], idx_dec[-1], "bo")
    # ax[1].plot(idx_ra, idx_dec, "g")

    # plt.show()
