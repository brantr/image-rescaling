#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sep
import matplotlib.colors as cl
import matplotlib as mpl
mpl.use('Agg')

def rescale_dash(data_dash):
    a = -1.5
    b = 0.5
    c = b+0.05
    d = 1.0

    d_min_sub = -0.085
    d_min = -0.08
    d_max = 15.
    d_b_cut = 3.1638243
    data_dash_clip = np.clip(data_dash.copy(),d_min,d_max)-d_min_sub

    idx = np.where( data_dash_clip < d_b_cut)
    z_alt =  data_dash_clip.copy()
    z_alt[idx] = (np.log10(data_dash_clip[idx])**d)
    idx = np.where( data_dash_clip > d_b_cut)
    z_alt[idx] = (c-b)*(data_dash_clip[idx]-d_b_cut)/(data_dash_clip[idx].max()-d_b_cut) + b
    z_dash_alt = np.clip(z_alt,a,c)/(c-a)

    return z_dash_alt

def beautify_dash(imgd):

    dbkg = sep.Background(imgd, bw=128, bh=128, fw=3, fh=3)


    mclip = 0. #for pl
    #mclip = 1.0e-3 #noisy
    #mclip = 1.0e-2 #bit noisy
    #mclip = 5.0e-2 #close, pl better
    #mclip = 1.0e-1 #too aggressive
    d = np.clip(imgd-dbkg,mclip,1)

    d = d**0.5

    return d


def rescale_F160W(data_F160W):

    a = -1.25
    b = 0.5
    c = b+0.5
    d = 1.0

    d_min_sub = -0.085
    d_min = -0.08
    d_max = 15.
    d_b_cut = 3.1638243
    data_F160W_clip = np.clip(data_F160W.copy(),d_min,d_max)-d_min_sub

    idx = np.where( data_F160W_clip < d_b_cut)
    z_alt =  data_F160W_clip.copy()
    z_alt[idx] = (np.log10(data_F160W_clip[idx])**d)
    idx = np.where( data_F160W_clip > d_b_cut)
    z_alt[idx] = (c-b)*(data_F160W_clip[idx]-d_b_cut)/(data_F160W_clip[idx].max()-d_b_cut) + b

    a_F160W = a
    b_F160W = b
    c_F160W = c
    d_F160W = d

    z_alt_F160W = np.clip(z_alt,a,c)
    return z_alt_F160W,a_F160W,c_F160W


def rescale_F814W(data_F814W):

    a = -1.25
    b = 0.5
    c = b+0.05
    d = 1.0

    d_min_sub = -0.085
    d_min = -0.08
    d_max = 15.
    d_b_cut = 3.1638243
    data_F814W_clip = np.clip(data_F814W.copy(),d_min,d_max)-d_min_sub

    idx = np.where( data_F814W_clip < d_b_cut)
    z_alt =  data_F814W_clip.copy()
    z_alt[idx] = (np.log10(data_F814W_clip[idx])**d)
    idx = np.where( data_F814W_clip > d_b_cut)
    z_alt[idx] = (c-b)*(data_F814W_clip[idx]-d_b_cut)/(data_F814W_clip[idx].max()-d_b_cut) + b
    a_F814W = a
    b_F814W = b
    c_F814W = c
    d_F814W = d
    z_alt_F814W = np.clip(z_alt,a,c)

    return z_alt_F814W,a_F814W,c_F814W
 
def rescale_F606W(data_F606W):

    a = -1.25
    b = 0.5
    c = b+0.05
    d = 1.0

    d_min_sub = -0.085
    d_min = -0.08
    d_max = 15.
    d_b_cut = 3.1638243
    data_F606W_clip = np.clip(data_F606W.copy(),d_min,d_max)-d_min_sub

    idx = np.where( data_F606W_clip < d_b_cut)
    z_alt =  data_F606W_clip.copy()

    z_alt[idx] = (np.log10(data_F606W_clip[idx])**d)
    idx = np.where( data_F606W_clip > d_b_cut)
    z_alt[idx] = (c-b)*(data_F606W_clip[idx]-d_b_cut)/(data_F606W_clip[idx].max()-d_b_cut) + b

    a_F606W = a
    b_F606W = b
    c_F606W = c
    d_F606W = d
    z_alt_F606W= np.clip(z_alt,a,c)
    return z_alt_F606W,a_F606W,c_F606W

def rescale_cosmos(data_F160W,data_F814W,data_F606W):

    #rescale F160W
    z_F160W, a_F160W, c_F160W = rescale_F160W(data_F160W)

    #rescale F814W
    z_F814W, a_F814W, c_F814W  = rescale_F814W(data_F814W)

    #rescale F606W
    z_F606W, a_F606W, c_F606W  = rescale_F606W(data_F606W)

    rgb = np.zeros((z_F606W.shape[0],z_F606W.shape[1],3))
    l_clip_F160W = 0.0
    l_clip_F814W = 0.0
    l_clip_F606W = 0.0
    u_clip_F160W = 1.0
    u_clip_F814W = 1.0
    u_clip_F606W = 1.0

    r = np.clip((z_F160W-a_F160W)/(c_F160W - a_F160W),l_clip_F160W,u_clip_F160W)#/(u_clip_F160W-l_clip_F160W)
    g = np.clip((z_F814W-a_F814W)/(c_F814W - a_F814W),l_clip_F814W,u_clip_F814W)#/(u_clip_F814W-l_clip_F814W)
    b = np.clip((z_F606W-a_F606W)/(c_F606W - a_F606W),l_clip_F606W,u_clip_F606W)#/(u_clip_F606W-l_clip_F606W)


    rbkg = sep.Background(r, bw=128, bh=128, fw=3, fh=3)
    gbkg = sep.Background(g, bw=128, bh=128, fw=3, fh=3)
    bbkg = sep.Background(b, bw=128, bh=128, fw=3, fh=3)


    mclip = 0. #for pl
    #mclip = 1.0e-3 #noisy
    #mclip = 1.0e-2 #bit noisy
    #mclip = 5.0e-2 #close, pl better
    #mclip = 1.0e-1 #too aggressive
    r = np.clip(r-rbkg,mclip,1)
    g = np.clip(g-gbkg,mclip,1)
    b = np.clip(b-bbkg,mclip,1)

    #r = (np.log10(r)-np.log10(mclip))/(-1*np.log10(mclip))
    #g = (np.log10(g)-np.log10(mclip))/(-1*np.log10(mclip))
    #b = (np.log10(b)-np.log10(mclip))/(-1*np.log10(mclip))

    #gamma = 0.8 #ok
    gamma = 0.5 #pretty good as power law

  
    rgb[:,:,0] = r**gamma
    rgb[:,:,1] = g**gamma
    rgb[:,:,2] = b**gamma
    return rgb


def sigmoid(x):
    return 1./(1.+np.exp(-1*x))

def rescale_thumbnail_test(rgb, gamma=1.75):
    
    def sigmoid(x):
        return 1./(1.+np.exp(-1.*x))

    hsv = cl.rgb_to_hsv(rgb)

    
    scale = 5.**(1./gamma)

    
    v = hsv[:,:,2]


    v*=0.5
    print(v.max())

    vcut = 1.0e-3
    idx = np.where(v>vcut)
    v[idx] = vcut + (1.-vcut)*(v[idx]-vcut)/(v[idx].max()-vcut)

    print(v.min(),v.max())
    v*=scale/v.max()
    
    v = v**gamma
    
    v = 2*(sigmoid(v)-0.5)
    
    #print(v.min(),v.max())
    
    v/=v.max()
    
    hsv[:,:,2] = np.clip(v,0,1.0)
    
    return cl.hsv_to_rgb(hsv)
    
def rescale_thumbnail(rgb, gamma=1.75):

    def sigmoid(x):
        return 1./(1.+np.exp(-1.*x))

    hsv = cl.rgb_to_hsv(rgb)


    scale = 5.**(1./gamma)


    v = hsv[:,:,2]
    print(v.min(),v.max())
    v*=scale/v.max()

    v = v**gamma

    v = 2*(sigmoid(v)-0.5)

    #print(v.min(),v.max())

    v/=v.max()

    hsv[:,:,2] = np.clip(v,0,1.0)

    return cl.hsv_to_rgb(hsv)

def main():

    hdu = fits.open("DASH-Snippet.fits")
    #hdu_F160W = fits.open("COSMOS-F160W-Snippet.fits")
    #hdu_F814W = fits.open("COSMOS-F814W-Snippet.fits")
    #hdu_F606W = fits.open("COSMOS-F606W-Snippet.fits")
    #hdu_F160W = fits.open("hlsp_candels_hst_wfc3_cos-tot_f160w_v1.0_drz.fits")
    #hdu_F814W = fits.open("hlsp_candels_hst_acs_cos-tot_f814w_v1.0_drz.fits")
    #hdu_F606W = fits.open("hlsp_candels_hst_acs_cos-tot_f606w_v1.0_drz.fits")
#    hdu = fits.open("reprojected_COSMOS_flux_160w-606w-814w.fits")
    #hdu = fits.open("./extract.fits")

    hdu.info()

    data = hdu[0].data

    #get a rescaled dash image
    img_dash = rescale_dash(data)

    #save dash img
    plt.imsave("dash.init.png",1-img_dash,cmap="gray")

    #img_dash_b = beautify_dash(img_dash)

    #plt.imsave("dash.png",1-img_dash_b,cmap="gray")


###########################################
#
# run the main program
#
###########################################
if __name__ == "__main__":
        main()
