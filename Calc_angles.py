import os, glob, time, sys
from math import *
import numpy as np
import pandas as pd

INDEX_PATH = '/home/etsai/BNL/Research/GIWAXS_index/Indexing/'
INDEX_PATH in sys.path or sys.path.append(INDEX_PATH)

from fun_index import *

def calc_ratation(q, energy=13.5, verbose=0):
  theta_x = QtoTheta(Q=q[0], wavelength=12.4/energy)
  theta_z = QtoTheta(Q=q[1], wavelength=12.4/energy)
  if verbose>1: 
      print('2theta = {:.2f}, {:.2f}'.format(theta_x*2, theta_z*2))

  alpha = np.arctan((1-cos(2*theta_x/180*np.pi)*cos(2*theta_z/180*np.pi)) / (sin(2*theta_x/180*np.pi)*cos(2*theta_z/180*np.pi)))
  alpha = alpha/np.pi*180
  if verbose>0:
      print('2alpha = {:.2f}'.format(alpha*2))

  return alpha

def calc_ratation_theta(theta_x, theta_z, verbose=0):
  if verbose>1: 
      print('2theta = {:.2f}, {:.2f}'.format(theta_x*2, theta_z*2))

  alpha = np.arctan((1-cos(2*theta_x/180*np.pi)*cos(2*theta_z/180*np.pi)) / (sin(2*theta_x/180*np.pi)*cos(2*theta_z/180*np.pi)))
  alpha = alpha/np.pi*180
  if verbose>0:
      print('2alpha = {:.2f}'.format(alpha*2))

  return alpha

def calc_psi(alpha, beta, thetaB):
  f1 = np.cos(alpha/180*np.pi)**2 + np.cos(beta/180*np.pi)**2 - 4*np.sin(thetaB/180*np.pi)**2
  f2 = 2*np.cos(alpha/180*np.pi)*np.cos(beta/180*np.pi)
  #print(f1)
  #print(f2)
  psi = np.arccos(f1 / f2 )
  psi = psi/np.pi*180

  return psi

def get_theta(lattice, hkl=[1,1,0], ori_hkl=[0,0,1], energy=13.5, spacegroup=14, verbose = 0):
    [a,b,c,alp_deg,beta_deg,gam_deg] = lattice
    [h,k,l] = hkl

    alp = radians(alp_deg)
    beta = radians(beta_deg)
    gam = radians(gam_deg)
    wavelength = 12.4/energy  
    
    if verbose:
        print('ori_hkl = {}, spacegroup = {}'.format(ori_hkl,spacegroup))
        print('(a,b,c)=({:.2f}, {:.2f}, {:.2f}) A, (alp,beta,gam)=({:.2f}, {:.2f}, {:.2f}) deg'.format(a,b,c,alp_deg,beta_deg,gam_deg))
        #print('lambda={:.2f}A, Inci={}deg'.format(lambda_A, kwargs['inc_theta_deg'][0]))

    #Lattice calculation
    lattice=[a,b,c,alp,beta,gam]
    V=a*b*c*sqrt(1+2*cos(alp)*cos(beta)*cos(gam)-cos(alp)**2-cos(beta)**2-cos(gam)**2)

    #reciprocal lattice
    ar=2*pi*b*c*sin(alp)/V
    br=2*pi*a*c*sin(beta)/V
    cr=2*pi*a*b*sin(gam)/V

    alpr=acos((cos(gam)*cos(beta)-cos(alp))/abs(sin(gam)*sin(beta)))
    betar=acos((cos(alp)*cos(gam)-cos(beta))/abs(sin(alp)*sin(gam)))
    gamr=acos((cos(alp)*cos(beta)-cos(gam))/abs(sin(alp)*sin(beta)))

    #rint('reciprocal lattice:\n a = {}  b = {} c = {} \n {}  {}  {}'.format(ar,br,cr,degrees(alpr), degrees(betar), degrees(gamr)))
    #reciprocal space vector
    As=np.array([ar, 0, 0]).reshape(3,1)
    Bs=np.array([br*cos(gamr), br*sin(gamr), 0]).reshape(3,1)
    Cs=np.array([cr*cos(betar), (-1)*cr*sin(betar)*cos(alp), 2*pi/c]).reshape(3,1)
    #As=np.array([ar, br*cos(gamr), cr*cos(betar)]).reshape(3,1)
    #Bs=np.array([0, br*sin(gamr), (-1)*cr*sin(betar)*cos(alp)]).reshape(3,1)
    #Cs=np.array([0, 0, 2*pi/c]).reshape(3,1)
    
    #rint('reciprocal space vector :\n {} \n {} \n {} '.format(As, Bs, Cs))

    #preferential reflections //to surface normal as z
    H = ori_hkl[0]
    K = ori_hkl[1]
    L = ori_hkl[2]
    G=H*As+K*Bs+L*Cs
    tol=0.01
    for index, item in enumerate(G):
        if abs(item) < tol:
            G[index] = 0

    phi=atan2(G[1],G[0])
    chi=acos(G[2]/sqrt(G[0]**2+G[1]**2+G[2]**2))

    #R is the rotation matrix from reciprocal to surface normal
    R1 = np.array([[cos(-chi),0,sin(-chi)], [0, 1, 0], [-sin(-chi),0,cos(-chi)]])
    R2 = np.array([[cos(-phi),-sin(-phi),0],[sin(-phi),cos(-phi),0],[0,0,1]])
    R = np.matmul(R1, R2)
    #print(R)

    #rotated reciprocal lattice vectors
    AR=np.matmul(R, As)
    BR=np.matmul(R, Bs)
    CR=np.matmul(R, Cs)        
    
    ##############
    #check_ref(0,0,1,spacegroup=spacegroup)
    
    ##############

    Qxy_list=[]
    Qz_list=[]
    phi_list=[]
    nu_list=[]
    h_list=[]
    k_list=[]
    l_list=[]
    d_list = []
    
    q_data = pd.DataFrame()

    count = 0

    if 1: #check_ref(h, k, l, spacegroup) and temp:
        
        d = d_rule(lattice=spacegroup, a=a, h=h, k=k, l=l)
        q = 2*np.pi/d
        if count==0:
            q0 = q
            count = 1
        q_ratio = q/q0
            
        #hkl = str(str(h) + str(k) + str(l))                  
        #data = [{'hkl': hkl,'d':d, 'q':q, 'q_ratio':q_ratio}]
        #df = pds.DataFrame(data)
        #q_data = q_data.append(df, ignore_index=True)
        
        #Q meet the reciprocal space
        Q=h*AR+k*BR+l*CR

        #plot Qxy and Qz position
        Qxy=sqrt(Q[0]**2+Q[1]**2)
        Qz=Q[2]

        theta_xy = QtoTheta(Q=Qxy, wavelength=wavelength)
        theta_z = QtoTheta(Q=Qz, wavelength=wavelength)

        #transfer to space geometry
        if False:
            beta=asin(Qz/2/pi*lambda_A-sin(Inci))

            beta_n=asin(sqrt(sin(beta)**2+sin(Inci_c)**2))
            theta=asin(Qxy/4/pi*lambda_A)
            if (cos(Inci)**2+cos(beta)**2-4*sin(theta)**2)>(2*cos(Inci)*cos(beta)):
                phi = acos(1)
            else:
                phi=acos((cos(Inci)**2+cos(beta)**2-4*sin(theta)**2)/(2*cos(Inci)*cos(beta)))
            phi_n=atan(tan(phi)/cos(Inci))
            nu_n=beta_n+atan(tan(Inci)*cos(phi_n))
            #save data
            phi_list.append(phi_n)
            nu_list.append(nu_n)
                    
    return theta_xy, theta_z   


# =============================================================================
# Calculate theoretical angles between projections
# =============================================================================
wavelength = 12.4/13.5

peaks = []
#peaks = [[0,1,7]]
#peaks = [[0,1,1]] #, [0,1,5], [0,1,7]]

if 0:
    for ii in [1]:
      peaks.append([1,1,ii])
    peaks.append([0,2,0])
    peaks.append([1,2,0]); 
    for ii in [0]:
        peaks.append([2,0,ii])
    peaks.append([2,1,1])

else:
    peaks = [[0,1,7]]
    for ii in [14]:
      peaks.append([1,1,ii])
    #peaks.append([0,2,1])
    #peaks.append([1,2,2]); 
    for ii in [8]:
        peaks.append([2,0,ii])
    #peaks.append([2,1,9])


peaks_full = []
for p in peaks:
  if p[0]>0 and (p[1]!=0 or p[2]!=0):
    peaks_full.append([-p[0],p[1],p[2]])
peaks_full = peaks_full+ peaks

N = len(peaks_full); print('N={}'.format(N))
x = {}

#lattice_deg= np.asarray([5.93, 7.88, 29.18, 90, 92.4, 90])
#lattice_deg= np.asarray([5.83, 7.88, 29.18, 90, 99.4, 90])
lattice_deg= np.asarray([5.86, 7.72, 33.7, 90, 93.2, 90])
lattice = lattice_deg.copy()
lattice[3:] = lattice[3:]/180*np.pi

jj = 0
for peak in peaks_full:
  peakname = '{}{}{}'.format(peak[0], peak[1], peak[2])
  #print(peakname)
  angle = angle_interplane(peak[0], peak[1], 0,    1, 0, 0,   lattice);
  theta_x, theta_z = get_theta(lattice_deg, hkl=peak, ori_hkl=[0,0,1], energy=13.5)
  qx = TwoThetatoQ(TwoTheta=theta_x*2, wavelength=wavelength)
  qz = TwoThetatoQ(TwoTheta=theta_z*2, wavelength=wavelength)
  print('{}: {:.2f}{}'.format(peakname,qx, qz))

  peak_z0 = [peak[0], peak[1], 0]
  [theta_x0, theta_z0] = get_theta(lattice_deg, hkl=peak_z0, ori_hkl=[0,0,1], energy=13.5)

  add_rot = calc_ratation_theta(theta_x, theta_z) - calc_ratation_theta(theta_x0, theta_z0)
  if peak[2] != 0:
    angle = angle + add_rot
    
  angle = np.round(angle, decimals=3) - 90
  #print('# rot = {:.2}'.format(2*calc_ratation_theta(theta_x, theta_z)))
  x[jj] = pd.DataFrame([[angle, peakname]], columns=['angle','peak']); jj=jj+1
  x[jj] = pd.DataFrame([[angle+180, peakname+'*']], columns=['angle','peak']); jj=jj+1

  angle_left = angle - 2*calc_ratation_theta(theta_x, theta_z)
  x[jj] = pd.DataFrame([[angle_left, peakname+'b']], columns=['angle','peak']); jj=jj+1
  x[jj] = pd.DataFrame([[angle_left+180, peakname+'b*']], columns=['angle','peak']); jj=jj+1  
  
  
temp_list = pd.concat(x)
print(temp_list)

# =============================================================================
# Plot
# =============================================================================
labels = np.asarray(temp_list['peak'])
angles_deg = np.asarray(temp_list['angle'])
angles_deg = np.round(angles_deg,1)
angles_rad =  temp_list['angle']/180*np.pi
ones = np.ones(len(angles_rad))

plt.figure(1, figsize=[8,8]); plt.clf()
ax = plt.subplot(111, projection='polar')
ax.bar(angles_rad, ones*0.8, width=ones*0.01, color='b', alpha=0.9)
ax.set_rticks([]) 
ax.set_xticklabels([])
plt.axis('off')
FS = 11; FW = 'bold'; FW1='bold'
green = [0, 0.7, 0]

for ii, angle in enumerate(angles_rad):
        label = labels[ii]
        if label[0]=='-':
          s = 0
        else:
          s = 0
               
        ## Label angles
        #if label[s+0]=='0' or label[s+1]=='0':
         #   ax.bar(angle, ones*0.8, width=ones*0.01, color='g', alpha=0.8)
        #    tt = ax.text(angle, 0.92, str(angles_deg[ii]), color='k', fontsize=FS-1,fontweight=FW1, ha='center', va='center')
        if label[-1]=='b' or label[-2]=='b':
            ax.bar(angle, ones*0.85, width=ones*0.01, color=green, alpha=0.9)
            tt = ax.text(angle, 0.87, str(angles_deg[ii]), color='k', fontsize=FS-1,fontweight=FW1, ha='center', va='center')
        else:
            tt = ax.text(angle, 0.85, str(angles_deg[ii]), color='k', fontsize=FS-1, fontweight=FW1, ha='center',va='center')
            
        if np.cos(angle)<=0.01: rotate=angle+np.pi
        else: rotate = angle
        tt.set_rotation(rotate/np.pi*180)
     
        ## Label peak
        if len(labels)>0:
          #if label[s+0]=='0' or label[s+1]=='0':
           #   tt2 = ax.text(angle, 1.05+0.0*np.mod(ii,2), label[s:], color='g', fontsize=FS, fontweight='bold', ha='center',va='center')
          if label[-1]=='b' or label[-2]=='b':
              tt2 = ax.text(angle, 1.1+0.0*np.mod(ii,2), label[s:], color=green, fontsize=FS, fontweight='bold', ha='center',va='center')
          else:
              tt2 = ax.text(angle, 0.98+0.0*np.mod(ii,2), label[s:], color='b',fontsize=FS, fontweight='bold', ha='center', va='center')



















