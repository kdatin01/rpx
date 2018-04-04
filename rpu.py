#!/usr/bin/env python
import os
import subprocess
import linecache
import shlex
import operator
import numpy as np
import math
import sys
import ssio
import pdb

N_DIM = 3
X = 0
Y = 1
Z = 2
NUM_COLORS = 256

class RUBBLE_PILE(object):
#Its so hard, being this cute, I'm a little baby, I don't know what to do.
	def __init__(self):
			# traditional quantities #
		self.mass = 0.0
		self.radius = 0.0
		self.pos = np.zeros((1,3))
		self.pos = self.pos[0]
		self.vel = np.zeros((1,3))
		self.vel = self.vel[0]
		self.spin = np.zeros((1,3))
		self.spin = self.spin[0]
		self.color = 0
		#special quantities */
		self.agg_id = 0
		self.inertia = np.zeros((3,3))
		self.moments = np.zeros((1,3))
		self.moments = self.moments[0]
		self.mom_ord= np.zeros((1,3))
		self.mom_ord = self.mom_ord[0]
		self.axes = np.zeros((3,3)) #principal axes are ROWS of this matrix, i.e. applying this matrix will convert from space coords to body coords */
		self.ang_mom = np.zeros((1,3))
		self.ang_mom = self.ang_mom[0]
		self.ang_mom_mag = None
		self.kin_energy = None
		self.dyn_inertia = None
		self.eff_spin = None
		self.rot_idx = None
		self.axis_len = np.zeros((1,3)) # semi-axes *//*DEBUG! rename so it's OBVIOUS these are semi-axes!!!*/
		self.axis_len = self.axis_len[0]
		self.axis_ord = None;
		self.density = None
		self.packing = None
		# particle data */
		self.n_particles = None
		self.data = []
	
	def setN_Particles(self, n_particles):
		self.n_particles = n_particles
	def getN_Particles(self):
		return self.n_particles
	
	def setData(self, ssdata):
		self.data = (ssdata)
	def getData(self):
		return self.data
		
	def setPos(self, pos):
		self.pos = pos[0]
	def setVel(self, vel):
		self.vel = vel[0]
	def setSpin(self, spin):
		self.spin = spin[0]
	def setMoments(self, moments):
		self.moments = moments[0]
	def setMomOrd(self, mom_ord):
		self.mom_ord = mom_ord[0]
	def setAngMom(self, ang_mom):
		self.ang_mom = ang_mom[0]
	def setAxisLen(self, axis_len):
		self.axis_len = axis_len[0]
		
class SSDATA(object):
	
	def __init__(self, n_particles):
		self.mass = None
		self.radius = None
		self.pos = np.zeros((n_particles,3))
		self.vel = np.zeros((n_particles,3))
		self.spin = np.zeros((n_particles,3))
		self.color = None
		self.org_idx = None
	
	def setMass(self, mass):
		self.mass = mass
	def getMass(self):
		return self.mass
	
	def setRadius(self, radius):
		self.radius	= radius
	def getRadius(self):
		return self.radius
		
	def setPos(self, pos):
		self.pos = pos[0]
	def getPos(self):
		return self.pos
		
	def setVel(self, vel):
		self.vel = vel[0]
	def getVel(self):
		return self.vel
	
	def setSpin(self, spin):
		self.spin = spin[0]
	def getSpin(self):
		return self.spin
		
	def setColor(self, color):
		self.color = color
	def getColor(self):
		return self.color
	
	def setOrigIdx(self, orig_idx):
		self.orig_idx = orig_idx
	def getOrigIdx(self):
		return self.orig_idx
		
class SSHEAD(object):
	def __init__(self):
		self.time = None
		self.n_data = None
		self.iMagicNumber = None
	def setTime(self, time):
		self.time = time
	def getTime(self):
		return self.time
	
	def setNData(self, n_data):
		self.n_data = n_data
	def getNData(self):
		return self.n_data
		
	def setMagNum(self, iMagicNumber):
		self.iMagicNumber = iMagicNumber
	def getMagNum(self):
		return self.iMagicNumber		
	
def rpCalcMass(rp):
	#Calculates total mass of rubble pile
	
	assert(rp != None)
	assert(rp.n_particles != None)
	assert rp.data
	for i in range(0,rp.n_particles):
		rp.mass = rp.mass + rp.data.mass[i]

def rpCalcPos(rp):
	#Sets rp->pos to center-of-mass position. Note rp->mass must be
	#the total mass. Call rpuCalcMass() to compute this if necessary.
	
	assert(rp != None)
	assert(rp.n_particles != None)
	assert rp.data
	assert(rp.mass >= 0.0)
	rp.pos = np.zeros((1,3))
	rp.pos = rp.pos[0]
	for i in range(0,rp.n_particles):
		r = rp.data.pos[i]
		r = r * rp.data.mass[i]
		rp.pos = rp.pos + r
	if rp.mass > 0.0:
		norm = 1.0/rp.mass
		rp.pos = rp.pos * norm
	else:
		norm = 1.0/rp.n_particles
		rp.pos = rp.pos * norm
	#rp.pos = rp.pos[0]
		
def rpCalcVel(rp):
	#Sets rp->vel to center-of-mass velocity -- see rpuCalcPos()
	
	assert(rp != None)
	assert(rp.n_particles != None)
	assert rp.data
	assert(rp.mass >= 0.0)
	
	rp.vel = np.zeros((1,3))
	rp.vel = rp.vel[0]
	for i in range(0, rp.n_particles):
		v = rp.data.vel[i]
		v = v * rp.data.mass[i]
		rp.vel = rp.vel + v
	if (rp.mass > 0.0):
		norm = 1.0/rp.mass
		rp.vel = rp.vel * norm
	else:
		norm = 1.0/rp.n_particles
		rp.vel = rp.vel * norm
	
def	rpCalcRadius(rp):
	assert(rp != None)
	assert(rp.n_particles != None)
	assert rp.data
	
	rp.radius = 0.0
	for i in range(rp.n_particles):
		v1 = rp.data.pos[i]
		v2 = rp.pos
		r = np.subtract(v1, v2)
		r_mag = np.linalg.norm(r) + rp.data.radius[i]
		if r_mag > rp.radius:
			rp.radius = r_mag

def	rpCalcInertia(rp):
	#calculates inertia tensor of rp (requires rp.pos to be populated
	#NOTE: The inertia tensor consists of the point mass distribution PLUS the
	#component spheres.  This allos us then to set L = Iw to solve for the bulk
	#spin w(on the assumption that each component sphere has the same spin w).
	
	assert(rp != None)
	assert(rp.n_particles != None)
	assert rp.data
	
	rp.inertia = np.zeros((3,3))
	for i in range(0,rp.n_particles):
		r = np.subtract(rp.data.pos[i], rp.pos)
		for j in range(0,N_DIM):
			for k in range(0,N_DIM):
				trVal = 0.4*(rp.data.radius[i]*rp.data.radius[i])+ rpDOT(r,r)
				flVal = 0
				rp.inertia[j,k] = rp.inertia[j,k] + rp.data.mass[i]*((trVal if j == k else flVal) - r[j]*r[k])

def	rpCalcAxes(rp):
	#Calculates moments(eigenvalues) and axes(eigenvectors) of rp.inertia.  The
	#principal axes are the ROWS of rp.axes.  This routine automatically calls
	#rpuCalcAxisLengths().
	
	assert(rp != None)
	
	#START HERE
	m = rp.inertia
	rpJacobi(m, rp.moments, rp.axes)
	rp.axes = rp.axes.transpose()
	for k in range(0, N_DIM):
		x = np.linalg.norm(rp.axes[k])
		assert(x > 0.0)
		norm = 1.0/x
		rp.axes[k] = rp.axes[k]*norm
	
	#Determine moment order (smallest to largest)
	x = rp.moments[X]
	y = rp.moments[Y]
	z = rp.moments[Z]
	if (x < y and x < z):
		if (y < z):
			rp.mom_ord = [X, Y, Z]
		else:
			rp.mom_ord = [X, Z, Y]
	elif (y < z):
		if (x < z):
			rp.mom_ord = [Y, X, Z]
		else:
			rp.mom_ord = [Y, Z, X]
	else:
		if (x < y):
			rp.mom_ord = [Z, X, Y]
		else:
			rp.mom_ord = [Z, Y, X]
	
	rpCalcAxisLengths(rp)
	
def rpCalcAxisLengths(rp):	
	#Calculates the lengths of the rubble pile principal semi-axes. The
	#matrix rp->axes must contain the (normalized) axis orientations
	#(see rpuCalcAxes()). Optionally, rp->axes can be set to the unit
	#matrix to get body dimensions along the Cartesian axes.
	#NOTE: It is very hard to calculate the size of the best-fitting
	#ellipsoid exactly, so we use an iterative procedure to converge
	#on a reasonable guess. Note the lengths include the finite size of
	#the particles and assume spatial symmetry wrt the center of mass.
	
	#DEBUG! updated to use center of figure, not center of mass, and to NOT
	#use rpuInEllipsoid()*
	assert rp
	assert(rp.n_particles != None)
	assert rp.data
	#get max dimensions along body axes
	rp.axis_len = np.zeros((1,3))
	rp.axis_len = rp.axis_len[0]
	rp.axis_ord = [X, Y, Z]
	vMin = np.zeros((1, 3))
	vMin = vMin[0]
	vMax = np.zeros((1, 3))
	vMax = vMax[0]
	for i in range(0, rp.n_particles):
		d = rp.data
		r = np.subtract(d.pos[i], rp.pos)
		s = rp.axes.dot(r)
		for k in range(0, N_DIM):
			if (s[k] - d.radius[i] < vMin[k]):
				vMin[k] = s[k] - d.radius[i]
			if (s[k] + d.radius[i] > vMax[k]):
				vMax[k] = s[k] + d.radius[i]
	for k in range(0, N_DIM):
		rp.axis_len[k] = 0.5*(vMax[k] - vMin[k])
	
	x = rp.axis_len[X]
	y = rp.axis_len[Y]
	z = rp.axis_len[Z]
	
	if (x < y and x < z):
		if (y < z):
			rp.axis_ord = [X, Y, Z]
		else:
			rp.axis_ord = [X, Z, Y]
	elif (y < z):
		if (x < z):
			rp.axis_ord = [Y, X, Z]
		else:
			rp.axis_ord = [Y, Z, X]
	else:
		if (x < y):
			rp.axis_ord = [Z, X, Y]
		else:
			rp.axis_ord = [Z, Y, X]		

def rpRot(a, s, tau, i, j, k, l):
	#a is MATRIX
	#s is double
	#tau is double
	#i, j, k, l are integers
	
	#based on struct Jacobi from NR3 11.1
	g = a[i][j]
	h = a[k][l]
	a[i][j] = g-s*(h+g*tau)
	a[k][l] = h+s*(g-h*tau)
	
def rpJacobi(a, d, v):
	#a is MATRIX
	#d is VECTOR
	#v is MATRIX
	#based on struct jacobi() from NR3 11.1
	n = N_DIM #hardcoded as 3D
	EPS = 1.1102e-16
	b = np.zeros((1,3))
	b = b[0]
	z = np.zeros((1,3))
	z = z[0]
	for ip in range(0, n):#Creates unit matrix
		for iq in range(0, n):
			v[ip][iq] = 0.0
		v[ip][ip] = 1.0
	for ip in range(0, n):
		b[ip] = d[ip]=a[ip][ip]
		z[ip] = 0.0
	for i in range(1, 50):
		sm = 0.0
		for ip in range(0,n-1):
			for iq in range(ip+1,n):
				sm = sm + abs(a[ip][iq])
		if sm == 0.0:
			rpEigsrt(d,v)
			return
		if i > 4:
			tresh = 0.2*sm/(n*n)
		else:
			tresh = 0.0
		for ip in range(0, n-1):
			for iq in range(ip+1, n):
				g = 100.0*abs(a[ip][iq])
				if i > 4 and g <= EPS*abs(d[ip]) and g <=EPS*abs(d[iq]):
					a[ip][iq] = 0.0
				elif abs(a[ip][iq]) > tresh:
					h = d[iq] - d[ip]
					if g <= EPS*abs(h):
						t = (a[ip][iq])/h
					else:
						theta = 0.5*h/(a[ip][iq])
						t = 1.0/(abs(theta) + math.sqrt(1.0 + theta*theta))
						if (theta < 0.0):
							t = -t
					c = 1.0/math.sqrt(1+t*t)
					s = t*c
					tau = s/(1.0+c)
					h = t*a[ip][iq]
					z[ip] = z[ip] - h
					z[iq] = z[iq] + h
					d[ip] = d[ip] - h
					d[iq] = d[iq] + h
					a[ip][iq] = 0.0
					for j in range(0, ip):
						rpRot(a, s, tau, j, ip, j, iq)
					for j in range(ip+1, iq):
						rpRot(a, s, tau, ip, j, j, iq)
					for j in range(iq+1, n):
						rpRot(a, s, tau, ip, j, iq, j)
					for j in range(0, n):
						rpRot(v, s, tau, j, ip, j, iq)
		for ip in range(0, n):
			b[ip] = b[ip] + z[ip]
			d[ip] = b[ip]
			z[ip] = 0.0
	print "jacobi(): Too many interactions"
	sys.exit(1)

def rpEigsrt(d, v):
	#d is VECTOR, v is MATRIX
	#based on eigsrt(), NR3 11.1
	n = N_DIM
	print 'Got in Eigsrt'
	for i in range(0, n-1):
		p = d[i]
		k = i
		for j in range(i, n):
			if d[j] >= p:
				p = d[j]
				k = j
		if (k != i):
			d[k] = d[i]
			d[i] = p
			for j in range(0, n):
				p = v[j][i]
				v[j][i] = v[j][k]
				v[j][k] = p

def	rpCalcSpin(rp):
	#Calculates spin (and angular momentum) of rubble pile. Note the
	#inertia tensor and axis moments are required in advance -- see
	#rpuCalcInertia() and rpuCalcAxes().
	
	assert rp
	assert(rp.n_particles != None)
	assert rp.data
	assert(rp.mass >= 0.0)
	TOLERANCE = 1.0e-12
	rp.ang_mom = np.zeros((1,3))
	rp.ang_mom = rp.ang_mom[0]
	if (rp.mass == 0.0):
		rp.spin = np.zeros((1,3))
		rp.spin = rp.spin[0]
	elif (rp.n_particles == 1):
		d = rp.data
		Isph = 0.4*d.mass[i]*(d.radius[i]*d.radius[i])
		rp.ang_mom = d.spin[i]
		rp.ang_mom = rp.ang_mom*Isph
		rp.inertia = np.eye(3, dtype=int)
		rp.inertia = rp.inertia * Isph
		rp.spin = d.spin[i]
	else:
		for i in range(0, rp.n_particles):
			d = rp.data
			Isph = 0.4*d.mass[i]*(d.radius[i]*d.radius[i])
			r = np.subtract(d.pos[i], rp.pos)
			v = np.subtract(d.vel[i], rp.vel)
			h = np.cross(r, v)
			h = h * d.mass[i]
			rp.ang_mom = h + rp.ang_mom
			h = d.spin[i]
			h = h * Isph
			rp.ang_mom = rp.ang_mom + h
		a = rp.inertia
		try:
			a = np.linalg.inv(a)
		except np.linalg.LinAlgError:
			pass
		else:
			rp.spin = a.dot(rp.ang_mom)
	rp.ang_mom_mag = np.linalg.norm(rp.ang_mom)
	rp.kin_energy = 0.5*np.dot(rp.spin, rp.ang_mom)
	if (rp.ang_mom_mag == 0.0):
		rp.eff_spin = 0.0
	else:
		rp.eff_spin = 2 * rp.kin_energy/rp.ang_mom_mag
	if (rp.kin_energy == 0.0):
		rp.dyn_inertia = 0.0
	else:
		rp.dyn_inertia = 0.5*(rp.ang_mom_mag*rp.ang_mom_mag)/rp.kin_energy
	#Problem here, not in the right order
	Ix = rp.moments[rp.mom_ord[X]]
	Iy = rp.moments[rp.mom_ord[Y]]
	Iz = rp.moments[rp.mom_ord[Z]]
	
	assert(Ix <= Iy <= Iz)
	Id = rp.dyn_inertia
	if (Id == 0.0):
		rp.rot_idx = 0.0
		return
	if (Id < Ix  and abs(Id - Ix) <= TOLERANCE):
		Id = Ix
	if (Id > Iz and abs(Id - Iz) <= TOLERANCE):
		Id = Iz
	assert (Id >= Ix and Id <= Iz) 
	
	if (Ix == Iy and Ix == Iz): #spherical
		rp.rot_idx = 1.0
	elif (Ix == Iy and Ix < Iz): #oblate
		rp.rot_idx = (Id-Ix)/(Iz - Ix)
	elif (Iz == Iy and Ix < Iz): #prolate
		rp.rot_idx = (Id - Iz)/(Iz - Ix)
	elif (Id >= Iy):
		rp.rot_idx == (Id - Iy)/(Iz - Iy)
	else:
		rp.rot_idx = (Id - Iy)/(Iy - Ix)
	if (rp.rot_idx > 1.0 and abs(rp.rot_idx - 1.0) <= TOLERANCE):
		rp.rot_idx = 1.0
	if (abs(rp.rot_idx) <= TOLERANCE):
		rp.rot_idx = 0.0
	if (rp.rot_idx < -1.0 and abs(rp.rot_ind +1) <= TOLERANCE):
		rp.rot_idx = -1.0
	assert(rp.rot_idx >= -1.0 and rp.rot_idx <= 1.0)
			
def	rpCalcColor(rp):
	#Sets rubble pile color to most dominant particle color
	assert rp
	assert(rp.n_particles != None)
	assert rp.data
	#n = np.empty(NUM_COLORS, dtype=object)
	n = np.empty(NUM_COLORS)
	for i in range(0, NUM_COLORS):
		n[i] = 0
	
	for i in range(0, rp.n_particles):
		c = rp.data.color[i]
		if c < 0:
			c = 0
		elif (c >= NUM_COLORS):
			c = NUM_COLORS - 1
		n[c] = n[c] + 1
	
	rp.color = n_max = 0
	
	for i in range(0, NUM_COLORS):
		if (n[i] > n_max):
			rp.color = i
			n_max = n[i]
	
	assert(rp.color >= 0 and rp.color < NUM_COLORS)

def	rpCalcAggID(rp):
	#Determines whether all particles in rubble pile share same
	#(negative) original index (org_idx).  If so, the aggregate
	#ID is computed as -1 - org_idx; otherwise it is set to -1.
	
	assert rp
	assert(rp.n_particles != None)
	assert rp.data
	rp.agg_id = -1
	for i in range(0, rp.n_particles):
		agg_id = -1 - (rp.data.orig_idx[i])#AGG_IDX
		if (agg_id < 0):
			return
		if (agg_id != rp.agg_id):
			if (rp.agg_id < 0):
				rp.agg_id = agg_id
			else:
				rp.agg_id = -1
				return
		
def	rpCalcDensity(rp):
	assert rp
	rp.density = rp.mass/rpVolEll(rp.axis_len)

def rpCalcPacking(rp):
	assert rp
	assert(rp.n_particles != None)
	assert rp.data
	volume = 0
	for i in range(0, rp.n_particles):
		volume = volume + rpVolSph(rp.data.radius[i])
	
	rp.packing = volume/(rpVolEll(rp.axis_len))

def rpVolSph(r):
	#r is a double
	return 4.0/3*math.pi*(r*r*r)
	
def rpVolEll(a):
	#a is a VECTOR
	return 4.0/3*math.pi*a[X]*a[Y]*a[Z]

def rpDOT(v1, v2):
	return (v1[X]*v2[X] + v1[Y]*v2[Y] + v1[Z]*v2[Z])



#####FOR TESTING, DELETE#####	
	
	
def readSSFile(fileName):
	##reads in SS file from pkdgrav and creates SSIO object, used to create rubble pile object
	header, ssioData = ssio.read_SS(fileName, 'yes') #read ss data and get header info
	head = transformSSHead(header)
	ssData = ssioToSSData(ssioData, head)
	
	return ssData, head

def ssioToSSData(ssioData, h):
	#ssioData is a vstack of 14 columns and n_particles rows
	#Use data in ssioData vstack to populate ssData object
	ssData = SSDATA(h.n_data)
	ssData.setMass(ssioData[2])
	ssData.setRadius(ssioData[3])
	ssData.setPos(np.dstack((ssioData[4],ssioData[5],ssioData[6])))
	ssData.setVel(np.dstack((ssioData[7],ssioData[8],ssioData[9])))
	ssData.setSpin(np.dstack((ssioData[10],ssioData[11],ssioData[12])))
	ssData.setColor(ssioData[13])
	ssData.setOrigIdx(ssioData[0])
	return ssData

def transformSSHead(header):
	##read and return header object contained in ssio data
	h = SSHEAD()
	h.setTime(header[0])
	h.setNData(header[1])
	h.setMagNum(header[2])
	return h

def rpAnalyze(rp):
	rpCalcMass(rp)
	rpCalcPos(rp)
	rpCalcVel(rp)
	rpCalcRadius(rp)
	rpCalcInertia(rp)
	rpCalcAxes(rp)
	rpCalcSpin(rp)
	rpCalcColor(rp)
	rpCalcAggID(rp)
	rpCalcDensity(rp)
	rpCalcPacking(rp)
	
	print rp.axes[rp.axis_ord[X]][X]
	print rp.axes[rp.axis_ord[X]][Y]
	print rp.axes[rp.axis_ord[X]][Z]
	
def setRBParams(rp, ssData, head):
	rp.setN_Particles(head.getNData())
	rp.setData(ssData)
	rpAnalyze(rp)
	
#####################################

def main():
	rp = RUBBLE_PILE()
	file = '/mnt/c/Users/kdatin01/Documents/pkdgrav_current-master/test/soft_sphere/rpg1.ss'
	ssData, head = readSSFile(file)
	setRBParams(rp, ssData, head)
	
if __name__=='__main__':
	main()