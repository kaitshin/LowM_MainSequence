# getdata.py
# Code to get M*, SFR data from Illustris-TNG-1 simulations
#
# Written by M. de los Reyes
# Portions of code from Illustris-TNG Web API tutorial:
#  http://www.tng-project.org/data/docs/api/
#
# Note that code requires an API key (need Illustris account)
#############################################################

import requests
import h5py
import numpy as np
import os
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value

FULL_PATH = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'

baseUrl = 'http://www.tng-project.org/api/'
api = os.environ["ILLUSTRIS_API"]
headers = {"api-key":api} # API key saved as environment variable for security purposes

# Helper function from Illustris-TNG documentation
def get(path, params=None):

	# make HTTP GET request to path
	r = requests.get(path, params=params, headers=headers)

	# raise exception if response code is not HTTP SUCCESS (200)
	r.raise_for_status()

	if r.headers['content-type'] == 'application/json':
		return r.json() # parse json responses automatically

	if 'content-disposition' in r.headers:
		filename = FULL_PATH+'illustrisdata/'
		filename += r.headers['content-disposition'].split("filename=")[1]
		with open(filename, 'wb') as f:
			f.write(r.content)
		return filename # return the filename string

	return r

# First get IDs for every subhalo with the right mass & SFR
def get_ids(minmass, maxmass, snapshotnum):

	# first convert log solar masses into group catalog units
	mass_min = 10**minmass / 1e10 * 0.704
	mass_max = 10**maxmass / 1e10 * 0.704

	# form the search_query string by hand for once
	search_query = "?mass_stars__gt=" + str(mass_min) + "&mass_stars__lt=" + str(mass_max) + "&sfr__gt=0.07079"

	# form the url and make the request
	url = baseUrl + "TNG100-1/snapshots/" + str(snapshotnum) + "/subhalos/" + search_query
	subhalos = get( url, {'limit':10000})
	print(subhalos['count'])

	ids = [ subhalo['id'] for subhalo in subhalos['results'] ]
	print(len(ids))
	return ids

# Then get current stellar masses and initial stellar masses for each subhalo
def get_masses_sfrs(ids, snapshotnum, a_current, a_10Myr, startid=0, outputfile="output.csv"):

	if startid==0:
		with open(outputfile, 'w+') as outfile:
			outfile.write('Mstar (log[Msun]), SFR (log[Msun/yr])\n')

	# Initialize some variables before looping
	grnr_old = '' 		# Variable to check if we need to save a new cutout
	saved_filename = ''	# Name of cutout
	first = True		# Variable to check if this is the first iteration of the code

	# If starting from the middle of the run, get correct ID
	if startid != 0:
		startidx = np.where(np.asarray(ids)==startid)[0][0] + 1
	else:
		startidx = 0

	for i in range(startidx, len(ids)):
		url = "http://www.tng-project.org/api/TNG100-1/snapshots/"+str(snapshotnum)+"/subhalos/" + str(ids[i])
		sub = get(url)

		# If halo is too close to the boundary, just skip it
		if ((sub['pos_x'] + 2.*sub['halfmassrad_stars']) > 75000.) or ((sub['pos_x'] - 2.*sub['halfmassrad_stars']) < 0.):
			continue
		if ((sub['pos_y'] + 2.*sub['halfmassrad_stars']) > 75000.) or ((sub['pos_y'] - 2.*sub['halfmassrad_stars']) < 0.):
			continue
		if ((sub['pos_z'] + 2.*sub['halfmassrad_stars']) > 75000.) or ((sub['pos_z'] - 2.*sub['halfmassrad_stars']) < 0.):
			continue

		# If we need to, save a new cutout
		if sub['grnr'] != grnr_old:

			# If it's not the first iteration, remove the old cutout
			if first:
				first = False
			else:
				os.remove(saved_filename)

			# Get cutout
			params = {'stars':'GFM_StellarFormationTime,GFM_InitialMass,Masses,Coordinates'}
			saved_filename = get(sub['cutouts']['parent_halo'],params)

			# Update check of new cutout
			grnr_old = sub['grnr']

		print(sub['id'], sub['grnr'], saved_filename[61:])

		try:
			# Get data
			with h5py.File(saved_filename) as f:

				# Pick particles within 2*(halfmassrad_stars) from the center of the halo
				dx = f['PartType4']['Coordinates'][:,0] - sub['pos_x']
				dy = f['PartType4']['Coordinates'][:,1] - sub['pos_y']
				dz = f['PartType4']['Coordinates'][:,2] - sub['pos_z']
				rr = np.sqrt(dx**2 + dy**2 + dz**2)

				# Get masses of stars born over the last 10 Myr
				ages = f['PartType4']['GFM_StellarFormationTime'][:]
				star_last_10Myr = np.where((ages > a_10Myr) & (ages < a_current) & (rr < sub['halfmassrad_stars']))[0].tolist()
				recentstars = f['PartType4']['GFM_InitialMass'][star_last_10Myr]

				# Note: have to convert Illustris mass units (10^10 Msun / h, where h = 0.704) to Msun

				# Compute SFR over last 10 Myr
				m_init = np.sum(recentstars)*1e10/0.704
				sfr = np.log10(m_init/(10.**7.)) # Log(Solar masses per year)

				# Current stellar mass
				m_star = np.log10(sub['mass_stars']*1e10/0.704) # Log(solar masses)

				print(m_star, sfr)

			with open(outputfile, 'a') as outfile:
				outfile.write(str(m_star)+', '+str(sfr)+'\n')

		# Note: will get local error if there are no recently-formed stars in the halo, so just skip halo if this happens
		except Exception as e:
			print(repr(e))
			continue

	return

# Helper function to compute scale factors
def compute_scales(z_current):

	cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

	# Age of universe at current z
	age = cosmo.age(z_current)

	# Compute redshift when universe was 10 Myr younger than current z
	z_10Myr = z_at_value(cosmo.age, cosmo.age(z_current) - (10.*u.Myr))

	# Compute scalefactors
	a_current = 1./(1.+z_current)
	a_10Myr = 1./(1.+z_10Myr)

	return a_current, a_10Myr

def main():
	# snapshot = 72
	# z = 0.4
	# outputfile = 'output_z04.csv'

	# snapshots can be from 67--93
	r = get(baseUrl)
	names = [sim['name'] for sim in r['simulations']]
	i = names.index('TNG100-1')
	sim = get( r['simulations'][i]['url'] )
	snaps = get( sim['snapshots'] )
	redshifts = np.array([snap['redshift'] for snap in snaps])
	min_snapshot_num = min(range(len(redshifts)), key=lambda i: abs(redshifts[i]-0.4984))
	max_snapshot_num = min(range(len(redshifts)), key=lambda i: abs(redshifts[i]-0.0677))

	for snapshot in np.arange(min_snapshot_num, max_snapshot_num-1)[::-1]:
		print('### Snapshot number', snapshot)
		ids = get_ids(6.0,10.5, snapshot)

		z = snaps[snapshot]['redshift']
		a_current, a_10Myr = compute_scales(z)

		outputfile = FULL_PATH+'illustrisdata/output_snap'+str(snapshot)+'_z%.4f'%z+'.csv'
		get_masses_sfrs(ids, snapshot, a_current, a_10Myr, outputfile=outputfile, startid=0)

if __name__ == "__main__":
	main()