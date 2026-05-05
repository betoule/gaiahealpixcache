''' A collection of tools to transform celestial coordinates into instruments ray vectors
'''
import jax.numpy as jnp
from astropy.coordinates import (SkyCoord, EarthLocation, AltAz, Distance, 
                                 GCRS, ITRS, get_body_barycentric)
from astropy import units as u
from astropy.time import Time
import astropy.table
import numpy as np

def radec_to_ray(
    ra_deg, dec_deg,          # target coordinates (degrees)
    ra0_deg, dec0_deg,        # optical axis / boresight (degrees)
    theta_deg=0.0             # field rotation angle (degrees), default: North=+Y, East=+X
):
    """
    Convert RA/Dec (in decimal degrees) to unit ray direction vectors
    in the camera frame using JAX.

    Parameters
    ----------
    ra_deg, dec_deg : float or jax array
        Target right ascension and declination in decimal degrees.
    ra0_deg, dec0_deg : float
        Boresight (optical axis) RA/Dec in decimal degrees.
    theta_deg : float, optional
        Field rotation angle in degrees (counter-clockwise).
        theta_deg = 0°  → +Y = celestial north, +X = celestial east
        theta_deg = 90° → +Y = celestial east, +X = celestial south

    Returns
    -------
    ray : jax array of shape (..., 3)
        Unit vectors (x, y, z) in the camera coordinate system.
        z points along the optical axis.
    """
    # Convert everything to radians once
    to_rad = jnp.pi / 180.0
    ra, dec       = ra_deg * to_rad, dec_deg * to_rad
    ra0, dec0     = ra0_deg * to_rad, dec0_deg * to_rad
    theta         = theta_deg * to_rad

    dalpha = ra - ra0

    # Pre-compute trig functions
    cd, sd     = jnp.cos(dec), jnp.sin(dec)
    cd0, sd0   = jnp.cos(dec0), jnp.sin(dec0)
    cda        = jnp.cos(dalpha)

    # Unrotated coordinates in the tangent plane (standard gnomonic projection)
    x0 = -cd * jnp.sin(dalpha)
    y0 = -sd * cd0 + cd * sd0 * cda
    z  =  sd * sd0 + cd * cd0 * cda

    # Apply field rotation around the optical axis
    ct, st = jnp.cos(theta), jnp.sin(theta)
    x = x0 * ct - y0 * st
    y = x0 * st + y0 * ct

    # Normalize (should be very close to 1, but safe for autodiff)
    norm = jnp.sqrt(x**2 + y**2 + z**2)
    x, y, z = x / norm, y / norm, z / norm

    return jnp.stack([x, y, z], axis=-1)

def gaia_to_topocentric(catalog, mjd=None, lon_deg=5.712637, lat_deg=43.932615, height_m=640.0,
                        pressure_hPa=980., temperature_C=10., wavelength_nm=550., 
                        ):
    """
    Convert a Gaia DR3 catalog (ICRS J2016.0) to apparent/topocentric coordinates.
    
    Parameters
    ----------
    catalog : pandas DataFrame
        Must contain columns: 'ra', 'dec' (in degrees), and optionally
        'parallax' (mas), 'pmra' (mas/yr), 'pmdec' (mas/yr), 'radial_velocity' (km/s)
    mjd : float
        Modified Julian Date of observation (UTC)
    lon_deg, lat_deg : float
        Observatory longitude (east positive) and latitude in degrees
    height_m : float, optional
        Altitude of observatory in metres (default 0)
    pressure_hPa, temperature_C : float, optional
        For accurate refraction correction (AltAz frame)
    return_apparent_radec : bool, optional
        If True, also return apparent RA/Dec (including proper motion, parallax,
        radial velocity, light deflection, aberration, and precession/nutation)
    
    Returns
    -------
    df_out : pandas DataFrame
        Original columns + new columns:
        - alt_deg, az_deg
        - zenith_distance_deg
        - hour_angle_deg
        - airmass (X ≈ sec(z) simple approximation)
        and optionally:
        - ra_apparent_deg, dec_apparent_deg
    """
    def fill(x):
        return np.nan_to_num(x)
    # 1. Build SkyCoord in ICRS at J2016.0 (Gaia frame)
    parallax = fill(catalog['parallax'] * u.mas)
    # avoid noisy negative parallaxes 
    parallax[parallax<0] = 0
    coord_j2016 = SkyCoord(ra=catalog['ra']* u.deg,
                           dec=catalog['dec'] * u.deg,
                           pm_ra_cosdec=fill(catalog['pmra'])*u.mas/u.yr,
                           pm_dec=fill(catalog['pmdec'])*u.mas/u.yr,
                           distance=Distance(parallax=parallax),
                           radial_velocity=fill(catalog['radial_velocity'])*u.km/u.s,
                           frame='icrs',
                           obstime=Time(2016.0, format='jyear', scale='tcb'))
    
    # 2. Apply proper motion, parallax, radial velocity → position at date
    if mjd is not None:
        obstime = Time(mjd, format='mjd', scale='utc')
    else:
        obstime = Time.now()
    
    
    # This does full space motion + light deflection + aberration
    coord_date = coord_j2016.apply_space_motion(obstime)
    
    # 3. Observatory location
    location = EarthLocation(lon=lon_deg * u.deg,
                             lat=lat_deg * u.deg,
                             height=height_m * u.m)
    
    # 4. Alt/Az frame with refraction
    altaz_frame = AltAz(obstime=obstime,
                        location=location,
                        pressure=pressure_hPa * u.hPa,
                        temperature=temperature_C * u.deg_C,
                        relative_humidity=50,          # doesn't matter much
                        obswl=550 * u.nm)               # optical
    
    altaz = coord_date.transform_to(altaz_frame)
    
    # 5. Hour angle
    lst = obstime.sidereal_time('mean', longitude=lon_deg * u.deg)
    ha = (lst - coord_date.ra).wrap_at(180*u.deg)   # in degrees, -180..+180
    
    # 6. Airmass (simple plane-parallel approximation)
    zd = 90*u.deg - altaz.alt
    airmass = 1.0 / jnp.cos(jnp.radians(zd.clip(max=87*u.deg)))   # avoid division by zero

    apparent = coord_date.transform_to(GCRS(obstime=obstime))
    sup = np.rec.fromarrays([altaz.alt.deg,
                             altaz.az.deg,
                             zd.deg,
                             ha.deg,
                             airmass,
                             apparent.ra.deg,
                             apparent.dec.deg],
                            names=['alt_deg', 'az_deg', 'zenith_distance_deg', 'hour_angle_deg',
                                   'airmass', 'ra_apparent_deg', 'dec_apparent_deg']
                            )
    return join(catalog, sup)

def center_at_date(ra, dec, mjd, refepoch=2016.0):    
    center_j2016 = SkyCoord(ra=ra* u.deg,
                            dec=dec * u.deg,
                            frame='icrs',
                            obstime=Time(refepoch, format='jyear', scale='tcb'))
    center_apparent = center_j2016.transform_to(GCRS(obstime=Time(mjd, format='mjd', scale='utc')))
    return center_apparent.ra.deg, center_apparent.dec.deg


