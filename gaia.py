import healpy
import numpy as np
from polyopticx.tools import cached_download, get_cache_dir
import tqdm
import gzip
import os
from tqdm import tqdm
import os

md5sum = cached_download("https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/_MD5SUM.txt")

columns_of_interest = ['source_id', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'phot_g_mean_flux', 'phot_g_mean_flux_error', 'phot_bp_mean_flux', 'phot_bp_mean_flux_error', 'phot_rp_mean_flux', 'phot_rp_mean_flux_error', 'radial_velocity', 'radial_velocity_error']

def parse_md5sum(fn):
    """Use md5sum file to get the full list of files.

    :param str fn: md5sum file from https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/
    :return int list bins: first pixels
    :return str list ranges: list of pixel range used in filenames
    """
    with open(fn) as fid:
        lines = fid.readlines()
    ranges = [
        l.split("GaiaSource_")[-1].split(".csv.gz")[0] for l in lines if l.endswith('.gz\n')
    ]
    bins = [int(b.split("-")[0]) for b in ranges]
    return bins, ranges


def get_pixlist(ras, decs, level=8):
    """Return list of healpix pixel matching ra, dec coordinate of a
    given catalog.

    :param arrays:  ras, decs coordinates of the 4 corners of the field
    :param int level: Gaia spectra are registered with healpix level 8 (nside=256)
    :return list pixlist: list of healpix pixels (in [0, 786431])
    """
    pixlist = []
    nside = healpy.order2nside(level)
    pixlist = list(np.unique([healpy.ang2pix(nside, ra, dec, lonlat=True, nest=True) for ra, dec in zip(ras, decs)]))
    return pixlist

def get_pix_range(ra, dec):
    """Return a list of pixel ranges matching indexing of gaia 36XX files.
    :param list ra:  ra
    :param list dec: dec
    :return str list ranges: list of pixel range covered by sky corrdinates
    """
    pixlist = get_pixlist(ra, dec)

    # todo span range
    bins, ranges = parse_md5sum(md5sum)
    range_index = np.digitize(pixlist, bins) - 1
    range_index = np.unique(range_index)
    return [ranges[i] for i in range_index]

def retrieve_gaia_data(pixel_range):
    """Load Gaia spectra.

    We drop the slow to load csv for compressed binary format with a
    selection of columns.
    
    :param str pixel_range: pixel range
    :param str gaia_dir: location of calibrated spectra
    :return dataframe calibrated_spectra: source_id, flux, flux_error
    :return array sampling: wavelength in angstrom

    """
    fname = os.path.join(get_cache_dir(), f'GaiaSource_{pixel_range}.npy')
    if os.path.exists(fname):
        res = np.load(fname)
        return res
    else:
        rawname = f'GaiaSource_{pixel_range}.csv.gz'
        sampled = cached_download(f"https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/{rawname}")
        sources = read_gaia(sampled)
        np.save(fname, sources)
        os.remove(sampled)
        return sources

def read_gaia(fname):
    fid = gzip.GzipFile(fname)
    lines = tqdm(fid.readlines())
    flux = []
    def process(line):
        vals = line.replace(b'[', b'').replace(b']', b'').replace(b'"', b'').split(b',')
        return [float(vals[i].replace(b'null', b'nan')) for i in keep_index]
    for line in lines:
        if line[0] == 35:
            continue
        if line[0] == 115:
            keys = [k.strip().decode() for k in line.split(b',')]
            keep_index = [keys.index(k) for k in keys if k in columns_of_interest]
            keys = [keys[i] for i in keep_index]
            assert len(keys) == len(columns_of_interest), 'missing column'
            continue
        flux.append(process(line))
    return np.rec.fromrecords(flux, names=keys)


def haversine(ra1, dec1, ra2, dec2):
    _ra1, _dec1, _ra2, _dec2 = np.radians(ra1), np.radians(dec1), np.radians(ra2), np.radians(dec2)
    dlambda = np.array(_ra1 - _ra2)
    return np.degrees(np.arccos(np.sin(_dec1) * np.sin(_dec2) + np.cos(_dec1) * np.cos(_dec2) * np.cos(dlambda)))

def query(ra_deg, dec_deg, radius_arcmin=30):
    rad = radius_arcmin / 60
    dra, ddec = np.meshgrid(np.linspace(ra_deg-rad, ra_deg + rad, 5),
                            np.linspace(dec_deg - rad, dec_deg + rad, 5))
    
    pranges = get_pix_range(dra, ddec)
    all_sources = []
    for pixel in pranges:
        sources = retrieve_gaia_data(pixel)
        in_radius = haversine(sources['ra'], sources['dec'], ra_deg, dec_deg) < rad
        all_sources.append(sources[in_radius])
    return np.hstack(all_sources)

def get_gaia_in_frame(params, transfo, im_shape=(1024, 1024), clip=1, N=500, radius_arcmin=20, **kwargs):
    ''' Retrieve an exerpt of the Gaia catalog given an instrument configuration.

    '''
    from polyopticx import celestial_coordinates
    stars = query(ra_deg=params['ra_center'], dec_deg=params['dec_center'],
                  radius_arcmin=radius_arcmin,
                 )
    stars = celestial_coordinates.gaia_to_topocentric(stars, **kwargs)
    stars['phot_g_mean_flux'] = np.nan_to_num(stars['phot_g_mean_flux'])
    # Select all stars in the sensor frame
    ra, dec = stars['ra_apparent_deg'], stars['dec_apparent_deg']
    in_frame, x, y = celestial_coordinates.select_stars_in_frame(transfo, params, ra, dec, im_shape=im_shape, clip=clip)
    stars = stars[in_frame]
    
    if N is not None:
        # Select the N brightest object (sorted according to Gaia G mag)
        level = np.percentile(stars['phot_g_mean_flux'], (1-N/len(stars))*100)
        above_level = stars['phot_g_mean_flux']>level
        stars = stars[above_level]
        x, y, = x[above_level], y[above_level]

    ra, dec = stars['ra_apparent_deg'], stars['dec_apparent_deg']

    return ra, dec, x, y, stars


if __name__ == '__main__':
    # test by querying the location of G191B2B
    sources = query(76.37757540733, 52.83108869489)
