"""
Tune previously generated model by seisattrib2log_build.

Designed to:
    read well data,

python seisattrib2log_build.py -h
python seisattrib2log_build.py allsgy.txt allgr.csv --startendslice 500 1000

"""

import os.path
import argparse
# from datetime import datetime
import numpy as np
import scipy.signal as sg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.stats as sts
import segyio as sg
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import griddata,Rbf,LinearNDInterpolator,CloughTocher2DInterpolator
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import mixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score

try:
    from catboost import CatBoostRegressor
except ImportError:
    print('***Warning:CatBoost is not installed')




def process_segylist(segyflist):
    """Process list of segy attribute files."""
    segylist = []
    with open(segyflist,'r') as f:
        for line in f:
            segylist.append(line.rstrip())
    return segylist

def get_slice(fname,slicenum):
    slc = list()
    with sg.open(fname,'r',ignore_geometry= True) as srcp:
        for trnum,tr in enumerate(srcp.trace):
            slc.append(tr[slicenum])
    return slc

def get_onetrace(fname,tracenum,sstart=None,send=None):
    """Get one trace from one file."""
    with sg.open(fname,'r',ignore_geometry=True) as srcp:
        tri = srcp.trace[tracenum]
        yield tri[sstart:send]

def collect_traces(segyflist,numoftraces,trcnum,sstart=None,send=None):
    """Collect the same trace from all attribute segy files."""
    for trn in range(numoftraces):
        trclst = list()
        for fn in segyflist:
            trclst.append(get_onetrace(fn,trn,sstart,send))
        trca = np.array(trclst)
        Xpred = trca.T
        # trace array transposed
        yield Xpred


def zero_segy(fname):
    """."""
    with sg.open(fname,'r+',ignore_geometry=True) as srcp:
        for trnum,tr in enumerate(srcp.trace):
            srcp.trace[trnum] = tr * 0

def get_samplerate(fname):
    """."""
    with sg.open(fname,'r',ignore_geometry=True) as srcp:
        hdrdict = dict(enumerate(srcp.header[1].items()))
    return hdrdict[39][1] / 1000


def plotwells(wdf,hideplots=True):
    """."""
    wlst = wdf[wdf.columns[0]].unique().tolist()
    print(wlst)
    pdfcl = "AllWellsPsuedoLogs.pdf"
    with PdfPages(pdfcl) as pdf:
        for wname in wlst:
            wndf = wdf[wdf[wdf.columns[0]] == wname]
            fig,ax = plt.subplots(figsize=(4,7))
            ax.invert_yaxis()
            ax.plot(wndf[wndf.columns[-2]],wndf[wndf.columns[1]],label='Actual')
            ax.plot(wndf[wndf.columns[-1]],wndf[wndf.columns[1]],label='Predicted')
            ax.set_xlabel(wndf.columns[-1])
            ax.set_ylabel(wndf.columns[1])
            ax.set_title(wname)
            ax.legend()
            fig.tight_layout()
            pdf.savefig()
            if not hideplots:
                fig.show()
            plt.close()

    pdfclh = "AllWellshistograms.pdf"
    with PdfPages(pdfclh) as pdf:
        for wname in wlst:
            wndf = wdf[wdf[wdf.columns[0]] == wname]
            plt.figure(figsize=(6,6))
            sns.distplot(wndf.GR,kde=True,
                hist_kws={"histtype": "step", "linewidth": 2,"alpha": 1, "color": "m","label": "GR"},
                kde_kws={"color": "m", "lw": 2, "label": "GR"})
            sns.distplot(wndf.GRPRED,kde=True,
                hist_kws={"histtype": "step", "linewidth": 2,"alpha": 1, "color": "g","label": "GRPRED"},
                kde_kws={"color": "g", "lw": 2, "label": "GRPRED"})
            plt.title(wname)
            pdf.savefig()
            if not hideplots:
                plt.show()
            plt.close()




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

def modelpredict(model,Xpred,fromfile=False,scalelog=True,logmin=0,logmax=150):
    """."""
    if fromfile:
        modelin = CatBoostRegressor()
        # Load the model from JSON
        modelin.load_model(model)
    else:
        modelin = model
    ypred = modelin.predict(Xpred)
    # grmin,grmax = wdf.iloc[:,gr].min(),wdf.iloc[:,gr].max()
    if scalelog:
        sc = MinMaxScaler((logmin,logmax))
        # wdf['LIsc'] = sc.fit_transform(wdf['LI'].values.reshape(-1,1))
        ypredsc = sc.fit_transform(ypred.reshape(-1,1))
        return ypredsc
    else:
        return ypred

def model_xvalidate(wdfsa,
    cv=5,
    testsize=.3,
    cbrlearningrate=None,
    cbriterations=None,
    cbrdepth=None,
    generatesamples=False,
    generatensamples=10,
    generatencomponents=2,
    overfittingdetection=True,
    odpval=0.005):
    """Cross validate CBR."""
    X = wdfsa.iloc[:,4: -1]
    y = wdfsa.iloc[:,-1]
    if y.size > 2 and generatesamples:
        X,y = gensamples(X,y,
            nsamples=generatensamples,
            ncomponents=generatencomponents,
            kind='r',func='cbr')
# *******************************
    model = CatBoostRegressor(iterations=cbriterations, learning_rate=cbrlearningrate,
                depth=cbrdepth,loss_function='RMSE',random_seed=42,logging_level='Silent')
    cvscore = cross_val_score(model,X,y,cv=cv,scoring='r2')
    # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
    print(f'{cv} fold Cross validation results:')
    print("Mean Score R2: %10.4f" % (np.mean(cvscore[0])))
    # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
    cvscore = cross_val_score(model,X,y,cv=cv,scoring='neg_mean_squared_error')
    print("Mean MSE: %10.4f" % (np.mean(np.abs(cvscore[1]))))
    print(model.get_params())
    xvalmdlfname = 'xvalidation_CBRmodel.mdl'
    model.save_model()
    print(f'Successfully generated {xvalmdlfname} ')

    # does not work: problem with eval_set variable Mar 1 2018
    if overfittingdetection:
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=testsize, random_state=42)
        evalset = (Xtest,ytest)
        model = CatBoostRegressor(iterations=cbriterations,
                    learning_rate=cbrlearningrate,
                    depth=cbrdepth,
                    loss_function='RMSE',
                    use_best_model=True,
                    od_type='IncToDec',
                    od_pval=odpval,
                    eval_metric='RMSE',
                    random_seed=42,logging_level='Silent')

        # Fit model
        model.fit(Xtrain, ytrain,eval_set=evalset)
        # Get predictions
        ytestpred = model.predict(Xtest)
        # Calculating Mean Squared Error
        mse = np.mean((ytestpred - ytest)**2)
        print(f'Metrics on test data set:{testsize} of the full data set ')
        print('MSE: %.4f' % (mse))
        r2 = r2_score(ytest,ytestpred)
        print('R2 : %10.3f' % r2)
        cctsplt = sts.pearsonr(ytest,ytestpred)
        print('Train-Test-Split Pearsonr : %10.3f  %10.4f ' % (cctsplt[0],cctsplt[1]))
        print(model.get_params())
        ttsplitmdlfname = 'train_test_split_CBRmodel.mdl'
        model.save_model()
        print(f'Successfully generated {ttsplitmdlfname} ')


def model_create(wdfsa,
    cbrlearningrate=None,
    cbriterations=None,
    cbrdepth=None,
    generatesamples=False,
    generatensamples=10,
    generatencomponents=2,savemodelname=None):
    """Read in welldata and generate model."""
    X = wdfsa.iloc[:,4: -1]
    y = wdfsa.iloc[:,-1]
    if y.size > 2 and generatesamples:
        X,y = gensamples(X,y,
            nsamples=generatensamples,
            ncomponents=generatencomponents,
            kind='r',func='cbr')
    model = CatBoostRegressor(iterations=cbriterations,
        learning_rate=cbrlearningrate,
        depth=cbrdepth,
        loss_function='RMSE',
        random_seed=42,
        logging_level='Silent')
    model.fit(X, y)

    model.save_model(savemodelname,format="json",export_parameters=None)
    print(f'Successfully saved CatBoostRegressor Model {savemodelname}')

    return model


def apply_model_towells(allwdfsa,cbrmodel,
                        logname=None,scalelog=True,
                        generatesamples=False,
                        generatensamples=10,
                        generatencomponents=2,
                        plotfname=None,
                        dirsplit=None,
                        hideplots=False,
                        outdir=None):
    """apply previously generated model."""
    X = allwdfsa.iloc[:,4 : -1]
    y = allwdfsa.iloc[:,-1]
    inshape = y.size
    # print( f"size of y: {inshape}")
    if y.size > 2 and generatesamples:
        X,y = gensamples(X,y,
            nsamples=generatensamples,
            ncomponents=generatencomponents,
            kind='r',func='cbr')
    # Get predictions
    ypred = cbrmodel.predict(X)
    if scalelog:
        logmin = allwdfsa.iloc[:,-1].min()
        logmax = allwdfsa.iloc[:,-1].max()
        print(logmin,logmax)
        sc = MinMaxScaler((logmin,logmax))
        ypredsc = sc.fit_transform(ypred.reshape(-1,1))
        ypred = ypredsc.reshape(-1)
    predcolname = logname +'PRED'
    allwdfsa[predcolname] = ypred
    # Calculating Mean Squared Error
    mse = np.mean((ypred - y)**2)
    print('Metrics on input data: ')
    print('MSE: %.4f' %(mse))
    r2 = r2_score(y,ypred)
    print('R2 : %10.3f' % r2)

    ccmdl = sts.pearsonr(y,ypred)
    qc0 = np.polyfit(y,ypred,1)
    xrngmin,xrngmax = y.min(),y.max()
    xvi = np.linspace(xrngmin,xrngmax)
    yvi0 = np.polyval(qc0,xvi)

    fig,ax = plt.subplots()

    plt.scatter(y,ypred,alpha=0.5,c='b',s=15,label='Model Predicted')
    if generatesamples:
        ax.scatter(y[inshape:],ypred[inshape:],c='r', marker='X',s=25,label='Generated Samples')

    plt.plot(xvi,yvi0,c='k',lw=2)



    ax.annotate('Model = %-.*f * Actual + %-.*f' %
            (2,qc0[0],2,qc0[1]),xy =(xvi[0],yvi0[0]),xytext=(0.14,0.85),
            textcoords='figure fraction', fontsize=10)
    ax.annotate ('Model Pearson cc = %-.*f   Pearson p = %-.*f' %
            (2,ccmdl[0],3,ccmdl[1]),xy =(xvi[0],yvi0[0]),xytext=(0.14,0.81),
            textcoords='figure fraction', fontsize=10)
    ax.set_title(f'CBR for all Pseudo {logname}' )
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    fig.savefig(plotfname)
    if not hideplots:
        plt.show()
    return allwdfsa


def getcommandline():
    """Get command line options."""
    parser = argparse.ArgumentParser(description='Build and tume one ML model to convert seismic to logs')
    parser.add_argument('sattribwellscsv',
        help='csv file with seismic attributes at wells depth devx devy log. Output from seisattrib2log_build.py ')
    # parser.add_argument('--logname',default='GR',help='Log curve name. default=GR')
    parser.add_argument('--cbriterations',type=int,default=500,help='Learning Iterations, default =500')
    parser.add_argument('--cbrlearningrate',type=float,default=0.01,help='learning_rate. default=0.01')
    parser.add_argument('--cbrdepth',type=int,default=6,help='depth of trees. default=6')
    parser.add_argument('--cv',type=int,default=None,help='Cross Validate. default=None.')
    parser.add_argument('--testsize',type=float,default=0.3,help='Validation. default=0.3')
    parser.add_argument('--overfittingdetection',action='store_true',default=False,
        help='Over Fitting Detection.default= False')
    parser.add_argument('--odpval',type=float,default=0.005,
        help='ranges from 10e-10 to 10e-2. Used with overfittingdetection')

    parser.add_argument('--generatesamples',action='store_true',default=False,help='Generate Samples.default=False')
    parser.add_argument('--generatensamples',type=int,default=10,help='# of sample to generate. default= 10')
    parser.add_argument('--generatencomponents',type=int,default=2,help='# of clusters for GMM.default=2')
    parser.add_argument('--intime',action='store_true',default=False,
        help='processing domain. default= True for depth')
    parser.add_argument('--donotscalelog',action='store_true',default=False,
        help='apply min max scaler to computed Psuedo GR. default do not apply')
    parser.add_argument('--logscalemm',nargs=2,type=float,default=(0,150),
        help='Min Max values to scale output computed Psuedo GR trace. default 0 150')

    parser.add_argument('--outdir',help='output directory,default= same dir as input')
    parser.add_argument('--hideplots',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    result = parser.parse_args()
    if not result.segyfileslist:
        parser.print_help()
        exit()
    else:
        return result



def main():
    """main program."""
    cmdl = getcommandline()
    if cmdl.seiswellscsv:
        allwdfsa = pd.read_csv(cmdl.sattribwellscsv)
        dz = np.diff(allwdfsa[allwdfsa.columns[1]])[2]
        print(f'Well Vertical increment {dz}')
        logname = allwdfsa.columns[-1]
        print(f'Curve Name: {logname} ')

        dirsplit,fextsplit = os.path.split(cmdl.seiswellscsv)
        fname,fextn = os.path.splitext(fextsplit)

        if cmdl.outdir:
            cbrmodelname = os.path.join(cmdl.outdir,fname) + f'_cbr{cmdl.cbrlearningrate}_{cmdl.cbriterations}_{cmdl.cbrdepth}.json'
            wsdfpred = os.path.join(cmdl.outdir,fname) + f'_cbr{cmdl.cbrlearningrate}_{cmdl.cbriterations}_{cmdl.cbrdepth}.csv'
            pdfxplot = os.path.join(cmdl.outdir,cmdl.logname) + 'xplt.pdf'
        else:
            cbrmodelname = os.path.join(dirsplit,fname) + f'_cbr{cmdl.cbrlearningrate}_{cmdl.cbriterations}_{cmdl.cbrdepth}.json'
            wsdfpred = os.path.join(dirsplit,fname) + f'_cbr{cmdl.cbrlearningrate}_{cmdl.cbriterations}_{cmdl.cbrdepth}.csv'
            pdfxplot = os.path.join(dirsplit,cmdl.logname) + 'xplt.pdf'

        if cmdl.cv:
            model_xvalidate(allwdfsa,cv=cmdl.cv,
                testsize=cmdl.testsize,
                cbrlearningrate=cmdl.cbrlearningrate,
                cbriterations=cmdl.cbriterations,
                cbrdepth=cmdl.cbrdepth,
                overfittingdetection=cmdl.overfittingdetection,
                odpval=cmdl.odpval)
        else:
            cbrmodel = model_create(allwdfsa,savemodelname=cbrmodelname,
                cbrlearningrate=cmdl.cbrlearningrate,
                cbriterations=cmdl.cbriterations,
                cbrdepth=cmdl.cbrdepth)
            allwdfsapred = apply_model_towells(allwdfsa,cbrmodel,
                scalelog=cmdl.scalelog,
                logname=cmdl.logname,
                dirsplit=dirsplit,
                plotfname=pdfxplot,
                hideplots=cmdl.hideplots,
                outdir=cmdl.outdir)
            allwdfsapred.to_csv(wsdfpred,index=False)
            print(f'Successfully generated all attributes with wells prediction {wsdfpred}')
            plotwells(allwdfsapred)
if __name__ == '__main__':
    main()
