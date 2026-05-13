"""Coordinate transformations and topocentric conversions."""

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    AltAz,
    Distance,
    EarthLocation,
    GCRS,
    SkyCoord,
)
from astropy.time import Time

from .cache import join


def airmass_pickering(h):
    """Compute the airmass from the apparent altitude according to Pickering (2002)

    According to wikipedia, Pickering (2002) formula is the better fit to obs.

    Parameters
    ----------
    h (float, or array): the apparent altitude in radian.

    Returns
    -------
       Airmass
    """
    hdeg = h * 180.0 / np.pi

    return 1 / np.sin((hdeg + 244 / (165 + 47 * hdeg**1.1)) * np.pi / 180.0)


def conform_coordinates(ra, dec):
    """Transform celestial coordinates to the normal convention (0<ra<360, -90<dec<90)

    Parameters
    ----------
    ra: float
        right ascension in degrees
    dec: float
        declination in degrees

    Returns
    -------
        Original coordinates following the standard convention
    """
    ra = ra % 360
    dec = (dec + 180) % 360 - 180
    if np.isscalar(ra):
        if dec > 90:
            ra -= 180
            dec = 180 - dec
        elif dec < -90:
            ra -= 180
            dec = -180 - dec
    # dec = (dec + 180) % 360 - 180
    ra = ra % 360
    return ra, dec


def gaia_to_topocentric(
    catalog,
    mjd=None,
    lon_deg=5.712637,
    lat_deg=43.932615,
    height_m=640.0,
    pressure_hPa=980.0,
    temperature_C=10.0,
    wavelength_nm=550.0,
):
    """Convert a Gaia DR3 catalog (ICRS J2016.0) to apparent/topocentric coordinates.

    Parameters
    ----------
    catalog : np.recarray
        Must contain columns: 'ra', 'dec' (degrees), and optionally
        'parallax' (mas), 'pmra' (mas/yr), 'pmdec' (mas/yr),
        'radial_velocity' (km/s).
    mjd : float, optional
        Modified Julian Date of observation (UTC). Defaults to now.
    lon_deg : float, optional
        Observatory longitude (east positive) in degrees.
    lat_deg : float, optional
        Observatory latitude in degrees.
    height_m : float, optional
        Observatory altitude in metres.
    pressure_hPa : float, optional
        Atmospheric pressure for refraction correction.
    temperature_C : float, optional
        Temperature for refraction correction.
    wavelength_nm : float, optional
        Observation wavelength for refraction correction.

    Returns
    -------
    np.recarray
        Original columns plus: alt_deg, az_deg, zenith_distance_deg,
        hour_angle_deg, airmass, ra_apparent_deg, dec_apparent_deg.
    """

    def fill(x):
        return np.nan_to_num(x)

    parallax = fill(catalog["parallax"]) * u.mas
    parallax[parallax < 0] = 0

    coord_j2016 = SkyCoord(
        ra=catalog["ra"] * u.deg,
        dec=catalog["dec"] * u.deg,
        pm_ra_cosdec=fill(catalog["pmra"]) * u.mas / u.yr,
        pm_dec=fill(catalog["pmdec"]) * u.mas / u.yr,
        distance=Distance(parallax=parallax),
        radial_velocity=fill(catalog["radial_velocity"]) * u.km / u.s,
        frame="icrs",
        obstime=Time(2016.0, format="jyear", scale="tcb"),
    )

    if mjd is not None:
        obstime = Time(mjd, format="mjd", scale="utc")
    else:
        obstime = Time.now()

    coord_date = coord_j2016.apply_space_motion(obstime)

    location = EarthLocation(
        lon=lon_deg * u.deg,
        lat=lat_deg * u.deg,
        height=height_m * u.m,
    )

    altaz_frame = AltAz(
        obstime=obstime,
        location=location,
        pressure=pressure_hPa * u.hPa,
        temperature=temperature_C * u.deg_C,
        relative_humidity=50,
        obswl=wavelength_nm * u.nm,
    )

    altaz = coord_date.transform_to(altaz_frame)

    lst = obstime.sidereal_time("mean", longitude=lon_deg * u.deg)
    ha = (lst - coord_date.ra).wrap_at(180 * u.deg)

    zd = 90 * u.deg - altaz.alt
    airmass = airmass_pickering(np.radians(altaz.alt.value))

    apparent = coord_date.transform_to(GCRS(obstime=obstime))

    sup = np.rec.fromarrays(
        [
            altaz.alt.deg,
            altaz.az.deg,
            zd.deg,
            ha.deg,
            airmass,
            apparent.ra.deg,
            apparent.dec.deg,
        ],
        names=[
            "alt_deg",
            "az_deg",
            "zenith_distance_deg",
            "hour_angle_deg",
            "airmass",
            "ra_apparent_deg",
            "dec_apparent_deg",
        ],
    )
    return join(catalog, sup)


def center_at_date(ra, dec, mjd, refepoch=2016.0):
    """Compute apparent RA/Dec of a sky position at a given observation date.

    Parameters
    ----------
    ra : float
        Right ascension in degrees (ICRS J2016.0).
    dec : float
        Declination in degrees (ICRS J2016.0).
    mjd : float
        Modified Julian Date of observation (UTC).
    refepoch : float, optional
        Reference epoch in Julian years (default: 2016.0).

    Returns
    -------
    ra_deg : float
        Apparent right ascension in degrees.
    dec_deg : float
        Apparent declination in degrees.
    """
    center_j2016 = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        frame="icrs",
        obstime=Time(refepoch, format="jyear", scale="tcb"),
    )
    center_apparent = center_j2016.transform_to(
        GCRS(obstime=Time(mjd, format="mjd", scale="utc"))
    )
    return center_apparent.ra.deg, center_apparent.dec.deg
