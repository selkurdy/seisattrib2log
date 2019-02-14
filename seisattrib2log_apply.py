"""
Designed to:
    read well data,
    one segy
    generate spectral attributes for each trace resulting in 10 attributes
    build a dataframe of attributes at each slice
    build a dataframe for all wells for that slice
    append all dataframes to generate one big dataframe of wells with attributes
    fit ML model to that final dataframe.


python seisattrib2log_apply.py allsgy.txt SWAttrib_GR.csv allsgy_cbr.json
"""

import os.path
import argparse
from datetime import datetime
import numpy as np
import pickle
import scipy.signal as sg
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import scipy.stats as sts
import segyio as sg
from shutil import copyfile
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import griddata,Rbf,LinearNDInterpolator,CloughTocher2DInterpolator
from matplotlib.backends.backend_pdf import PdfPages
import itertools as it
from scipy.signal import savgol_filter
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import NuSVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import mixture
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import model_from_json
import pickle

try:
    from catboost import CatBoostRegressor
except ImportError:
    print('***Warning:CatBoost is not installed')


def process_segylist(segyflist):
    segylist = []
    with open(segyflist,'r') as f:
        for line in f:
            segylist.append(line.rstrip())
    return segylist

def get_onetrace(fname,tracenum,sstart=None,send=None):
    """Get one trace from one file."""
    with sg.open(fname,'r',ignore_geometry=True) as srcp:
        tri = srcp.trace[tracenum]
        return tri[sstart:send]

def collect_traces(segyflist,trcnum,sstart=None,send=None):
    trclst = list()
    for fn in segyflist:
        trclst.append(get_onetrace(fn,trcnum,sstart,send))
    if trcnum % 10000 == 0:
        print(f'Trace: {trcnum}')
    trca = np.array(trclst)
    Xpred = trca.T
    # trace array transposed
    return Xpred


def zero_segy(fname):
    with sg.open(fname,'r+',ignore_geometry= True) as srcp:
        for trnum,tr in enumerate(srcp.trace):
            srcp.trace[trnum] = tr * 0

def get_samplerate(fname):
    with sg.open(fname,'r',ignore_geometry= True) as srcp:
        hdrdict = dict(enumerate(srcp.header[1].items()))
    return hdrdict[39][1]/1000

def gensamples(datain,targetin,
    ncomponents=2,
    nsamples=10,
    kind='r',
    func=None):
    """
    Generate data using GMM.

    Reads in features array and target column
    uses GMM to generate samples after scaling target col
    scales target col back to original values
    saves data to csv file
    returns augmented data features array and target col

    newfeatures,newtarget =gensamples(wf,codecol,kind='c',func='cbr')
    """
    d0 = datain
    t0 = targetin
    d0t0 = np.concatenate((d0,t0.values.reshape(1,-1).T),axis=1)
    sc = StandardScaler()
    t0min,t0max = t0.min(),t0.max()
    t0s = sc.fit_transform(t0.values.reshape(1,-1))
    d0t0s = np.concatenate((d0,t0s.T),axis=1)
    gmm = mixture.GaussianMixture(n_components=ncomponents,covariance_type='spherical', max_iter=500, random_state=0)
    gmm.fit(d0t0s)
    d0sampled = gmm.sample(nsamples)[0]
    d1sampled = d0sampled[:,:-1]
    targetunscaled = d0sampled[:,-1].reshape(1,-1)
    scmm = MinMaxScaler((t0min,t0max))
    if kind == 'c':
        targetscaledback = np.floor(scmm.fit_transform(targetunscaled.T))
    else:
        targetscaledback = scmm.fit_transform(targetunscaled.T)
    d1t1 = np.concatenate((d1sampled,targetscaledback),axis=1)
    d1 = np.concatenate((d0t0,d1t1))
    print(d1.shape)
    fname = 'gensamples_' + func + '.csv'
    np.savetxt(fname,d1,fmt='%.3f',delimiter=',')
    return d1[:,:-1],d1[:,-1]


def modelpredict(model,Xpred,scalelog=True,logmin=0,logmax=150):
    ypred = model.predict(Xpred)
    # grmin,grmax = wdf.iloc[:,gr].min(),wdf.iloc[:,gr].max()
    if scalelog:
        sc = MinMaxScaler((logmin,logmax))
        # wdf['LIsc'] = sc.fit_transform(wdf['LI'].values.reshape(-1,1))
        ypredsc = sc.fit_transform(ypred.reshape(-1,1))
        ypred = ypredsc.reshape(-1)
    return ypred




def  getcommandline():
    parser = argparse.ArgumentParser(description='ML to convert seismic to logs')
    parser.add_argument('segyfileslist',help='File that lists all segy files to be used for attributes')
    parser.add_argument('sattribwellscsv',
        help='csv file with seismic attributes at wells generated from seisattrib2log_build.py, e.g. SWAttrib_XXX.csv ')
    parser.add_argument('MLmodelname',default=None,help='ML model name.default=none')
    parser.add_argument('--modeltype',choices=['cbr','linreg','knn','svr','ann','sgdr'],default='cbr',
        help='cbr, linreg, knn default= cbr')
    parser.add_argument('--segyxhdr',type=int,default=73,help='xcoord header.default=73')
    parser.add_argument('--segyyhdr',type=int,default=77,help='ycoord header. default=77')
    parser.add_argument('--xyscalerhdr',type=int,default=71,help='hdr of xy scaler to divide by.default=71')
    parser.add_argument('--startendinterval',type=int,nargs=2,
        default=[1000,2000],help='Start end interval in depth/time. default= 1000 2000')
    parser.add_argument('--intime',action='store_true',default=False,
        help='processing domain. default= True for depth')
    parser.add_argument('--donotscalelog',action='store_false',default=True,
        help='do not apply min max scaler to computed Psuedo GR. default apply scaling')
    parser.add_argument('--logscalemm',nargs=2,type=float,default=(0,150),
        help='Min Max values to scale output computed Psuedo GR trace. default 0 150')

    parser.add_argument('--outdir',default=None,help='output directory,default= same dir as input')

    result=parser.parse_args()
    if not result.segyfileslist:
        parser.print_help()
        exit()
    else:
        return result



def main():
    cmdl = getcommandline()
    # csv file generated from _build without prediction column
    allwdfsa = pd.read_csv(cmdl.sattribwellscsv)
    # need to extraqct by well to find depth increment
    wlst = allwdfsa.WELL.unique().tolist()
    wdf0 = allwdfsa[allwdfsa['WELL'] == wlst[0]]
    dz = np.diff(wdf0[wdf0.columns[1]])[2]
    print(f'Well Vertical increment {dz}')
    sstart = int(cmdl.startendinterval[0] // dz)
    send = int(cmdl.startendinterval[1] // dz)
    logname = allwdfsa.columns[-1]
    print(f'Curve Name: {logname} Sample start: {sstart}  Sample end: {send}')

    if cmdl.segyfileslist:
        sflist = list()
        sflist = process_segylist(cmdl.segyfileslist)

        dirsplit,fextsplit= os.path.split(cmdl.segyfileslist)
        fname,fextn= os.path.splitext(fextsplit)
        if cmdl.outdir:
            outfsegy = os.path.join(cmdl.outdir,fname) + f"_p{logname}.sgy"
        else:
            outfsegy = os.path.join(dirsplit,fname) + f"_p{logname}.sgy"

        print('Copying file, please wait ........')
        start_copy = datetime.now()
        copyfile(sflist[0], outfsegy)
        end_copy = datetime.now()
        print(f'Duration of copying: {(end_copy - start_copy)}')

        sr = get_samplerate(outfsegy)
        print(f'Seismic Sample Rate: {sr}')

        print('Zeroing segy file, please wait ........')
        start_zero = datetime.now()
        zero_segy(outfsegy)
        end_zero = datetime.now()
        print(f'Duration of zeroing: {(end_zero - start_zero)}')


        scols = list()
        for f in sflist:
            dirsplit,fextsplit= os.path.split(f)
            fname,fextn= os.path.splitext(fextsplit)
            scols.append(fname)
        # sstart = cmdl.startendinterval[0]
        # send = cmdl.startendinterval[1]
        start_process = datetime.now()
        if cmdl.modeltype == 'cbr':
            inmodel = CatBoostRegressor()
            inmodel.load_model(cmdl.MLmodelname)
            # inmodel = pickle.load(open(cmdl.MLmodelname, 'rb'))
        elif cmdl.modeltype == 'linreg':
            inmodel = pickle.load(open(cmdl.MLmodelname, 'rb'))
            # result = loaded_model.score(X_test, Y_test)

        elif cmdl.modeltype == 'knn':
            inmodel = pickle.load(open(cmdl.MLmodelname, 'rb'))
            # result = loaded_model.score(X_test, Y_test)
        elif cmdl.modeltype == 'svr':
            inmodel = pickle.load(open(cmdl.MLmodelname, 'rb'))
            # result = loaded_model.score(X_test, Y_test)
        elif cmdl.modeltype == 'ann':
            anndirsplit,annfextsplit= os.path.split(cmdl.segyfileslist)
            annfname,annfextn= os.path.splitext(annfextsplit)
            annwtsfname = os.path.join(anndirsplit,annfname) + '.h5'
        elif cmdl.modeltype == 'sgdr':
            inmodel = pickle.load(open(cmdl.MLmodelname, 'rb'))

            json_file = open(cmdl.MLmodelname, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            inmodel = model_from_json(loaded_model_json)
            # load weights into new model
            inmodel.load_weights(annwtsfname)
            print("Loaded model from disk")
            inmodel.compile(loss='mean_squared_error', optimizer='adam')

        with sg.open(outfsegy, "r+") as srcp:
            # numoftraces = len(srcp.trace)
            for trnum,tr in enumerate(srcp.trace):
                Xpred = collect_traces(sflist,trnum,sstart=sstart,send=send)
                # print(Xpred.shape)
                trpred = modelpredict(inmodel,Xpred,
                    scalelog=cmdl.donotscalelog,
                    logmin=cmdl.logscalemm[0],
                    logmax=cmdl.logscalemm[1])

                tr[sstart: send] = trpred
                srcp.trace[trnum] = tr
        print(f'Successfully generated {outfsegy}')
        end_process = datetime.now()
        print(f'Duration: {end_process - start_process}')

if __name__ == '__main__':
    main()
