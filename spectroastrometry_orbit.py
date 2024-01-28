"""
Created on Sat Feb 25 17:39:14 2023.

@author: Quirijn B. van Woerkom
Class that simulates Keplerian orbits with a set of auxiliary functions.
"""
# Standard imports
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
import time
import tqdm

# Non-standard imports
import os


# Plotstyle changes
# Increase the matplotlib font size
plt.rcParams.update({"font.size": 22})
# Set layout to tight
plt.rcParams.update({"figure.autolayout": True})
# Set grid to true
plt.rcParams.update({"axes.grid": True})
# Set the plotting style to interactive (autoshows plots)
plt.ion()


# %% Initialise the required values for the 'cordic' method in solve_Kepler
# based on initialisation routine given in Zechmeister (2018).
divtwopi = 1/(2*np.pi)
cordic_consts = np.array([(a, np.cos(a), np.sin(a)) for a in
                          [np.pi/2**i for i in range(1, 60)]])
# %% Auxiliary functions/code for the Orbit class


def E0_guess(M, e):
    """Calculate a guess for the eccentric anomaly.

    Calculate an initial guess for the eccentric anomaly. Based off the
    approach by K.F. Wakker in "Fundamentals of Astrodynamics, Jan. 2015".

    Parameters
    ----------
    M (float):
        Mean anomaly.
    e (float):
        Eccentricity.

    Returns
    -------
    E0 (float):
        Guess for the eccentric anomaly.
    """
    M = M % (2*np.pi)
    E0 = M + e*(1-e*e/8 + e*e*e*e/192)*np.sin(M) + e*e*(1/2 - e*e/6) *\
        np.sin(2*M) + e*e*e*(3/8 - 27/128)*np.sin(3*M) + \
        e*e*e*e/3*np.sin(4*M) + 125/384*e*e*e*e*e*np.sin(5*M)
    return E0


@u.quantity_input
def solve_Kepler(M, e, acc=1e-10, method='markley', cordic_n=55):
    """Solve Kepler's equation for E up to a desired accuracy.

    Solves Kepler's equation:
    M = E - e*sin(E)
    for E for a given pair of M and e. Only suited for elliptical orbits.

    Parameters
    ----------
    M (float):
        Mean anomaly of the orbit.
    e (float):
        Eccentricity of the orbit.
    method : str
        Whether to use the version based on Markley (1995) ('markley'),
        the CORDIC-like method by Zechmeister (2018) ('cordic') or a more
        naive Newton's method solver ('newton'). Overall, Markley performs best
        and should be preferred for high-accuracy applications.
    acc : float
        Accuracy to employ in the iterative method. For the Markley method,
        this is irrelevant.
    cordic_n : int
        Number of iterations to use with the CORDIC-like method; 29 yields
        single precision and 55 yields double precision.

    Returns
    -------
    E (float):
        Eccentric anomaly of the orbit.
    sinE (float):
        Sine of E; returned only for the CORDIC-like method as it comes for
        free in that calculation.
    cosE (float):
        Cosine of E; returned only for the CORDIC-like method as it comes for
        free in that calculation.
    """
    # Check that the orbit is elliptical
    if e >= 1.:
        raise ValueError("solve_Kepler does not support eccentricities "
                         "greater than 1 (only elliptical orbits).")
    # Immediately return the trivial cases
    if e == 0:
        return M
    if M == 0:
        return M
    if method == 'newton':
        # Otherwise, solve iteratively
        # Based off treatment in "Fundamentals of Astrodynamics, Jan. 2015" by
        # K.F. Wakker
        # Initialise the iterative estimator
        E = E0_guess(M, e)
        diff = acc + 1
        it = 0
        while diff > acc:
            it += 1
            E_old = E
            E = E - (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
            diff = np.abs(E - E_old)
        # After the iteration stops, return the found value
        return E
    elif method == 'markley':
        # Procedure based on 'Kepler Equation Solver' by F. Landis Markley
        # (1995).
        # First calculate E1
        # To do this, we must first calculate some auxiliary quantities
        # and reduce M to a value in [-pi, pi]
        while abs(M) > np.pi:
            M = M - np.sign(M)*2*np.pi
        alpha = (3*np.pi*np.pi + 1.6*np.pi*(np.pi - abs(M))/(1+e)
                 )/(np.pi*np.pi - 6)
        d = 3*(1 - e) + alpha*e
        q = 2*alpha*d*(1 - e) - M*M
        r = 3*alpha*d*(d - 1 + e)*M + M*M*M
        w = (abs(r)+np.sqrt(q*q*q+r*r))**(2/3)
        E1 = (2*r*w/(w*w+w*q+q*q) + M)/d
        # E1 must then be refined
        # Compute the trig functions so those need only be evaluated once
        esinE1 = e*np.sin(E1)
        ecosE1 = e*np.cos(E1)
        # Compute the derivatives of f(E1)=E1 - esinE1 - M
        if e > .5 and E1 < 1:
            numerator = - 1.7454287843856404e-6*E1**6 \
                + 4.1584640418181644e-4*E1**4 \
                - 3.0956446448551138e-2*E1*E1 + 1
            denom = 1.7804367119519884e-8*E1**8 \
                + 5.9727613731070647e-6*E1**6 \
                + 1.0652873476684142e-3*E1**4 \
                + 1.1426132130869317e-1*E1*E1 + 6
            Mstar = (1-e)*E1 + e*E1*E1*E1*numerator/denom
        else:
            Mstar = E1 - esinE1
        fE = Mstar - M
        fE1 = 1 - ecosE1
        fE2 = E1 - Mstar
        fE3 = 1 - fE1
        fE4 = -fE2
        # Compute the Halley corrections
        d3 = -fE/(fE1 - .5*fE*fE2/fE1)
        d4 = -fE/(fE1 + .5*d3*fE2 + d3*d3*fE3/6)
        d5 = -fE/(fE1 + .5*d4*fE2 + d4*d4*fE3/6 + d4*d4*d4*fE4/24)
        E = E1 + d5
        return E
    elif method == 'cordic':
        # Minimum rotation (one-sided) variant of Zechmeister (2018)
        E = 2*np.pi*np.floor(M*divtwopi+.5)
        cosE = 1.
        sinE = 0.
        for alpha, cosalpha, sinalpha in cordic_consts[:cordic_n+1, :]:
            sinE_pot = sinE*cosalpha + cosE*sinalpha
            E_pot = E + alpha
            if E_pot - e*sinE_pot < M:
                E = E_pot
                cosE = cosE*cosalpha - sinE*sinalpha
                sinE = sinE_pot
        return E, sinE, cosE


@u.quantity_input
def theta_from_E(E: u.rad, e):
    """Calculate the true anomaly.

    Calculate the true anomaly from a given eccentric anomaly and eccentricity.

    Parameters
    ----------
    E (astropy.Quantity):
        Eccentric anomaly.
    e (float):
        Eccentricity.

    Returns
    -------
    theta (astropy.Quantity):
        True anomaly.
    """
    E = E.to(u.rad).value
    e_factor = np.sqrt((1+e)/(1-e))
    theta = 2*np.arctan2((e_factor*np.tan(E/2)), 1) % (2*np.pi)
    return theta*u.rad


def xi_hat(t0P, TP, e, acc=1e-10):
    """
    Calculate the in-plane average position of a satellite over a timespan.

    Parameters
    ----------
    t0P : float
        Initial time since periapse as fraction of the total period.
    TP : float
        Duration over which the average is measured as fraction of the total
        period.
    e : float
        Eccentricity of the orbit.
    acc : float
        Accuracy to pass to the solve_Kepler call.

    Returns
    -------
    xi : (2,)-array of floats
        Average in-orbital plane position of the satellite, expressed in units
        of a/d.
    """
    # Reduce t0P down to a number in [0, 1]
    t0P = t0P - np.floor(t0P)
    if TP <= 1e-10:
        # If the time difference is zero (or close enough that round-off error
        # might cause issues) instead return the value for no movement
        # i.e. the exact position at time t0
        M0 = 2*np.pi*t0P
        E0 = solve_Kepler(M0, e, acc=acc)
        theta0 = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E0/2))
        r_norm = (1 - e*np.cos(E0))
        xi_x = r_norm*np.cos(theta0)
        xi_y = r_norm*np.sin(theta0)
        xi = np.array([xi_x, xi_y])
        return xi
    # Calculate the mean anomaly at the start and end of the timespan
    M0 = 2*np.pi*t0P
    M1 = 2*np.pi*(t0P + TP)
    # Calculate the corresponding eccentric anomalies
    E0 = solve_Kepler(M0, e, acc=acc)
    E1 = solve_Kepler(M1, e, acc=acc)
    # Calculate the relevant quantities for the calculation
    E_avg = (E1 + E0)/2
    E_diff = (E1 - E0)/2
    # Compute the x- and y-components of xi
    xi_x = np.sin(E_diff)/(np.pi*TP)*((1-e*e/2)*np.cos(E_avg) -
                                      e/2*np.cos(E_diff)*np.cos(2*E_avg)
                                      ) - 3*e/2
    xi_y = np.sqrt(1-e*e)*np.sin(E_diff)/(np.pi*TP)*(np.sin(E_avg) -
                                                     e/2*np.cos(
                                                         E_diff
    )*np.sin(2*E_avg))
    xi = np.array([xi_x, xi_y])
    return xi


# %% Orbit class


class Orbit:
    """Abstraction of Keplerian orbits.

    Class that contains orbit information, allows orbit propagation and orbit
    plotting and the like. For now only represents Keplerian orbits.

    Attributes
    ----------
    a (astropy.Quantity):
        Semi-major axis of the orbit.
    e (float):
        Eccentricity of the orbit; constant with time.
    i (astropy.Quantity):
        Inclination of the orbit w.r.t. a reference plane. We shall leave the
        reference plane open for now, as this will depend on the object the
        orbit is used for; we shall often find it convenient to describe the
        inclination of objects with respect to the observation vector, so this
        will be how we shall implement it in the majority of cases (i.e. in
        agreement with astronomical convention).
        Constant with time.
    Omg (astropy.Quantity):
        Right ascension of the ascending node (RAAN) of the orbit. Again,
        what this is referenced again we shall leave open in this class, but it
        will be useful to note for now that it will be referenced again an
        on-sky angle in the final implementation. Will instead be equal to the
        longitude of perigee for an equatorial orbit. Constant with time. Note
        that unless specific measures are taken, this deviates from the
        standard practice in astronomy, where the RAAN is measured from the
        direction of North; to obtain the astronomical RAAN (assuming that the
        y-direction is North), simply subtract 90 degrees.
    omg (astropy.Quantity):
        Argument of periapsis of the orbit. Measured along the plane of the
        orbit from the intersection of the RAAN and the reference plane for the
        inclination. Will instead be equal to the longitude of perigee for an
        equatorial orbit. Constant with time.
    M0 (astropy.Quantity):
        Mass of the system. The mass of the primary body will often be a good
        approximation.
    tau (astropy.Quantity):
        Time of pericenter passage. Constant with time (that is to say, it
        defines a time (degenerate up to integer orbital periods of the object)
        when the object last passed pericenter or will next pass it. For
        near-circular orbits this simply refers to the time when the object
        last passed the reference angle. If set to None, use Mref instead.
    Mref (astropy.Quantity):
        Mean anomaly at reference epoch (t=0). If set to None, use tau instead.
    th (astropy.Quantity):
        True anomaly; defines the location of the object in its orbit. This
        will of course vary with time. For near-circular orbits, we set this
        equal to the argument of latitude; for near-equatorial orbits, we set
        this equal to the true longitude. For near-equatorial, near-circular
        orbits we set this equal to the true longitude.
    M (astropy.Quantity):
        Mean anomaly.
    E (astropy.Quantity):
        Eccentric anomaly.
    omg_tilde (astropy.Quantity):
        Longitude of perigee. Equal to the sum of Omg and omg, and useful in
        near-equatorial orbits, where Omg and omg are individually degenerate.
    lambda_m (astropy.Quantity):
        Mean longitude; equal to the sum of omg_tilde and M, the mean anomaly,
        which is useful for near-equatorial orbits.
    lambda_t (astropy.Quantity):
        True longitude; equal to the sum of omg_tilde and th, the true
        longitude, for use with near-equatorial orbits.
    lambda_e (astropy.Quantity):
        Eccentric longitude; equal to the sum of omg_tilde and E, the eccentric
        anomaly. Useful for near-equatorial orbits.
    u (astropy.Quantity):
        Argument of latitude, the sum of omg and theta. Useful for
        near-circular orbits.
    n (astropy.Quantity):
        Mean motion of the body. Note that this is constant!
    P (astropy.Quantity):
        Period of the body.
    r (astropy.Quantity):
        Distance of the body to its primary.
    t (astropy.Quantity):
        Current time.
    H (astropy.Quantity):
        Angular momentum of the object in its orbit.
    dist (astropy.Quantity):
        Distance of the central body to the observer. Allows setting of correct
        labels on the angular size of the axes.


    """

    @u.quantity_input
    def __init__(self,
                 dist: u.pc = 10*u.pc,
                 a: u.AU = 1*u.AU,
                 e=0.,
                 i: u.rad = 0*u.rad,
                 Omg: u.rad = 0*u.rad,
                 omg: u.rad = 0*u.rad,
                 tau: u.yr = None,
                 Mref: u.rad = None,
                 M0: u.Msun = 1*u.Msun):
        """Initialise the Orbit.

        Initialises the Orbit with six Keplerian elements. Automatically
        calculates any degeneracy-breaking elements, too. For degenerate
        orbits, set the ambiguous orbital elements such that their sums line
        up with the desired value (e.g. for an equatorial orbit, set the
        argument of periapsis and RAAN to be such that their sum is the
        desired longitude of periapsis). Only the Cartesian->Keplerian case
        should run into issues transforming this way, which we will handle if
        we ever need to.

        Parameters
        ----------
        dist (astropy.Quantity):
            Distance of the central body to the observer.
        a (astropy.Quantity):
            Semi-major axis.
        e (float):
            Eccentricity.
        i (astropy.Quantity):
            Inclination i.e. angle between observer-ward direction and the
            angular momentum vector of the body.
        Omg (astropy.Quantity):
            Right-ascension of the ascending node (RAAN).
        omg (astropy.Quantity):
            Argument of pericentre.
        tau (astropy.Quantity or None):
            Time of pericentre passage. If set to None, use Mref instead.
        Mref (astropy.Quantity or None):
            Mean anomaly at reference epoch. If set to None, use tau instead.
        M0 (astropy.Quantity):
            Mass of the system.
        """
        # Insert the given parameters
        self.dist = dist.to(u.pc)
        self.a = a
        self.e = e
        self.i = i.to(u.rad)
        self.Omg = Omg.to(u.rad)
        self.omg = omg.to(u.rad)
        self.tau = tau
        self.Mref = Mref
        self.M0 = M0

        # Calculate any dependent parameters
        self.omg_tilde = self.Omg + self.omg
        self.n = np.sqrt(const.G*self.M0/(self.a*self.a*self.a)).to(
            u.rad/u.s, equivalencies=u.dimensionless_angles())
        self.P = (2*np.pi/self.n).to(u.yr,
                                     equivalencies=u.dimensionless_angles())
        self.H = np.sqrt(const.G*self.M0*self.a*(1-self.e*self.e))

        # Solve Kepler's equation to find the initial value of th at t=0
        self.t = 0*u.yr
        if ((Mref is None) and (tau is None)):
            raise ValueError("One of mref and tau must be set to a value.")
        if ((Mref is not None) and (tau is not None)):
            raise ValueError("Only one of mref and tau must be set!")
        if Mref is None:
            self.Mref = -tau*self.n
        if tau is None:
            self.tau = -Mref/self.n
        self.M = np.copy(self.Mref)
        self.E = solve_Kepler(self.M.to(u.rad).value, self.e)*u.rad
        self.th = theta_from_E(self.E, self.e)
        self.r = self.a*(1-self.e*self.e)/(1 + self.e*np.cos(self.th))

        # Define the other degeneracy-breaking components
        self.omg_tilde = self.Omg + self.omg
        self.lambda_m = self.omg_tilde + self.M
        self.lambda_t = self.omg_tilde + self.th
        self.lambda_e = self.omg_tilde + self.E
        self.u = self.omg + self.th

    @u.quantity_input
    def Kepler_at_time(self, t: u.yr = 0*u.yr):
        """Calculate the orbital state at time t.

        Caluclate the orbital state at time t as described by the angle
        theta (or, equivalently, the angle u i.e. the argument of latitude
        for near-circular orbits or the true longitude for near-equatorial,
        near-circular orbits). Whichever the case, we simply return that
        angle as for solving of Kepler's equation it makes no difference. Note
        that we explicitly will always use Keplerian orbits in this function.

        Parameters
        ----------
        t (astropy.Quantity):
            Time at which to find the position.

        Returns
        -------
        th (astropy.Quantity):
            Position of the object in orbit with respect to the reference
            angle for the given orbit. N.B.: does not set this angle,
            only returns it!
        """
        # Take the time modulo the period of the orbit
        t = t % self.P
        M = self.n*(t - self.tau)
        E = solve_Kepler(M.to(u.rad).value, self.e)*u.rad
        th = theta_from_E(E, self.e)
        return th

    @u.quantity_input
    def propagate(self, delta_t: u.yr = 1*u.d):
        """Propagate the orbit by a time delta_t.

        Propagate the orbit by a time delta_t; for now only uses unperturbed
        Keplerian orbits, such that the resulting orbit will have the same
        orbital elements. In the future, it may be worthwhile adding precessing
        orbits and such.

        Parameters
        ----------
        delta_t (astropy.Quantity):
            Time by which to propagate the orbit.
        """
        # Simply solve Kepler's equation as we did initially
        self.t += delta_t
        self.M = self.n*(self.t - self.tau)
        self.E = solve_Kepler(self.M.to(u.rad).value, self.e)*u.rad
        self.th = theta_from_E(self.E, self.e)
        self.r = self.a*(1-self.e*self.e)/(1 + self.e*np.cos(self.th))

        # Define the degeneracy-breaking components
        self.omg_tilde = self.Omg + self.omg
        self.lambda_m = self.omg_tilde + self.M
        self.lambda_t = self.omg_tilde + self.th
        self.lambda_e = self.omg_tilde + self.E
        self.u = self.omg + self.th

    @u.quantity_input
    def pos(self, t: u.yr = 0*u.yr):
        """Calculate the position of the object in the observation frame.

        Calculate the position of the object in a rectangular reference
        frame with the x-axis corresponding to a westward direction on the sky,
        the y-axis to the northward direction on the sky and the z-direction
        corresponding to the observation vector i.e. pointing toward the
        observer such that the coordinate frame is right-handed.
        Here we assume that the RAAN is referenced against the x-axis, and the
        inclination against the z-axis (corresponding to astronomical
        convention).

        Returns
        -------
        xvec (np.array of astropy.Quantity):
            x, y and z-positions of the object.
        """
        if t == 0.:
            th = self.th
        else:
            th = self.Kepler_at_time(t=t)
        # Calculat the new radius
        r = self.a*(1-self.e*self.e)/(1 + self.e*np.cos(th))
        # Save some useful values to forego double computations
        sinomg_th = np.sin(self.omg + th)
        cosomg_th = np.cos(self.omg + th)

        cosOmg = np.cos(self.Omg)
        sinOmg = np.sin(self.Omg)

        cosi = np.cos(self.i)
        sini = np.sin(self.i)
        # Perform the commputations
        xvec = np.zeros((3,))*u.AU
        xvec[0] = r*(cosomg_th*cosOmg - sinomg_th*cosi*sinOmg)
        xvec[1] = r*(sinomg_th*cosi*cosOmg + cosomg_th*sinOmg)
        xvec[2] = r*sinomg_th*sini
        return xvec

    def calc_orbit_pos(self, n):
        """Calculate n sample positions equally spaced over one period.

        Calculate n sample positions equally spaced over one period; useful
        for plotting purposes.

        Parameters
        ----------
        n (int):
            Number of samples to take.
        """
        # Compute the epochs at which to calculate the position
        t_arr = np.linspace(self.t.to(u.yr).value,
                            self.t.to(u.yr).value + self.P.to(
                                u.yr).value, n,
                            endpoint=True)*u.yr
        # Preallocate an array for the orbit
        xvec_arr = np.zeros((3, n))*u.AU
        # Calculate the positions
        for t_idx, t in enumerate(t_arr):
            xvec_arr[:, t_idx] = self.pos(t=t)
        return xvec_arr

    @u.quantity_input
    def plot(self, n=100, dist_label="AU", view_3D=True):
        """Plot the orbit and the object's current position.

        Plot the orbit of the object an the object's current position in that
        orbit, using n samples of the orbit over a full period.

        Parameters
        ----------
        n (int):
            Number of samples to use to plot the orbit.
        dist_label (str):
            If set to "arcsec", set the axes to be labelled in arcseconds. If
            set to "AU" instead, mark the axes in AU.
        view_3D (bool):
            If set to False, projects the orbit onto the sky; if set to True,
            explore the orbit in 3D (note that the arcsec labels are then no
            longer accurate!).
        """
        xvec_arr = self.calc_orbit_pos(n)
        # Set up the figure
        fig = plt.figure()
        # Calculate the angular distances if necessary
        if dist_label == "arcsec":
            plot_arr = ((xvec_arr/self.dist).to(
                u.dimensionless_unscaled)*u.rad).to(u.arcsec)
        if dist_label == "AU":
            plot_arr = xvec_arr
        # Determine whether to plot a 3D or 2D figure
        if view_3D:
            ax = fig.add_subplot(projection="3d")
            ax.plot(plot_arr[0, :], plot_arr[1, :], plot_arr[2, :],
                    linestyle="solid", alpha=.3,
                    color="grey",
                    label="Keplerian orbit")
            ax.plot(plot_arr[0, 0], plot_arr[1, 0], plot_arr[2, 0],
                    marker="o", color="grey",
                    label="Current position",
                    linestyle="None")
            ax.plot(0., 0., 0.,
                    marker="*", color="orange",
                    linestyle="None", markersize=20)
            ax.set_zlabel(f"$\\Delta z$ [{dist_label}]")
        else:
            ax = fig.add_subplot()
            ax.plot(plot_arr[0, :], plot_arr[1, :],
                    linestyle="solid", alpha=.3,
                    color="grey",
                    label="Keplerian orbit")
            ax.plot(plot_arr[0, 0], plot_arr[1, 0],
                    marker="o", color="grey",
                    label="Current position",
                    linestyle="None")
            ax.plot(0., 0.,
                    marker="*", color="orange",
                    linestyle="None", markersize=20)
        ax.set_aspect('equal', 'box')
        ax.set_ylabel(f"$\\Delta y$ [{dist_label}]")
        ax.set_xlabel(f"$\\Delta x$ [{dist_label}]")
        ax.legend(loc="upper right")
