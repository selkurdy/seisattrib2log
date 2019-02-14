'''
python seisattrib2log_fitmodel.py linreg SWAttrib_GR.csv --wcolsrange 4 8
python seisattrib2log_fitmodel.py linreg SWAttrib_GR.csv --wcolsrange 4 8  --minmaxscale
python seisattrib2log_fitmodel.py KNN SWAttrib_GR.csv --wcolsrange 4 8  --minmaxscale
python seisattrib2log_fitmodel.py NuSVR SWAttrib_GR.csv --wcolsrange 4 8  --errpenalty 0.8 --nu 0.8 --minmaxscale
python seisattrib2log_fitmodel.py CatBoostRegressor SWAttrib_GR.csv --wcolsrange 4 8
python seisattrib2log_fitmodel.py CatBoostRegressor SWAttrib_GR.csv --wcolsrange 4 8  --minmaxscale
python seisattrib2log_fitmodel.py SGDR SWAttrib_GR.csv --wcolsrange 4 8 --minmaxscale


'''
import os.path
import argparse
import shlex
# import datetime
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as sts
import pickle
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import griddata,Rbf,LinearNDInterpolator,CloughTocher2DInterpolator
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import seaborn as sns
from collections import Counter
import itertools


from pandas.tools.plotting import scatter_matrix
# ->deprecated
# from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler,QuantileTransformer
# from sklearn import cross_validation
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC,NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import mixture
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import NuSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import r2_score
# Coefficient of Determination
from sklearn.cluster import KMeans,DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_samples
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import RandomizedLasso
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.manifold import TSNE
# from sklearn.svm import SVR
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.semi_supervised import LabelSpreading
try:
    from catboost import CatBoostRegressor
    from catboost import CatBoostClassifier
except ImportError:
    print('***Warning:CatBoost is not installed')

try:
    from imblearn.over_sampling import SMOTE, ADASYN,RandomOverSampler
except ImportError:
    print('***Warning:imblearn is not installed')

try:
    import umap
except ImportError:
    print('***Warning: umap is not installed')

def module_exists(module_name):
    """Check for module installation."""
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True

def plot_classifier(classifier, X, y,xcol0=0,xcol1=1):
    """Plot Classification curves: ROC/AUC."""
    # define ranges to plot the figure
    x_min, x_max = min(X[:, xcol0]) - 1.0, max(X[:, xcol0]) + 1.0
    y_min, y_max = min(X[:, xcol1]) - 1.0, max(X[:, xcol1]) + 1.0

    # denotes the step size that will be used in the mesh grid
    step_size = 0.01

    # define the mesh grid
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # compute the classifier output
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

    # reshape the array
    mesh_output = mesh_output.reshape(x_values.shape)

    # Plot the output using a colored plot
    plt.figure()

    # choose a color scheme you can find all the options
    # here: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)

    # Overlay the training points on the plot
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # specify the boundaries of the figure
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    # specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(min(X[:, 0]) - 1), int(max(X[:, 0]) + 1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1]) - 1), int(max(X[:, 1]) + 1), 1.0)))
    plt.show()


def qcattributes(dfname,pdffname=None,deg=1,dp=3,scattermatrix=False,cmdlsample=None):
    """
    Establish linear fit relationships between each predictor and singular target.

    The second variable changes it from 1=linear to >1=Polynomial Fit

    """
    with PdfPages(pdffname) as pdf:
        for i in range((dfname.shape[1]) - 1):
            xv = dfname.iloc[:,i].values
            yv = dfname.iloc[:,-1].values
            xtitle = dfname.columns[i]
            ytitle = dfname.columns[-1]
            xrngmin,xrngmax = xv.min(),xv.max()
            # print(xrngmin,xrngmax)
            xvi = np.linspace(xrngmin,xrngmax)
            # print(xrng)
            qc = np.polyfit(xv,yv,deg)
            if deg == 1:
                print('Slope: %5.3f, Intercept: %5.3f' % (qc[0],qc[1]))
            else:
                print(qc)
            yvi = np.polyval(qc,xvi)
            plt.scatter(xv,yv,alpha=0.5)
            plt.plot(xvi,yvi,c='red')
            plt.xlabel(xtitle)
            plt.ylabel(ytitle)
            # commenting out annotation : only shows on last plot!!
            if deg == 1:
                plt.annotate('%s = %-.*f   + %-.*f * %s' % (ytitle,dp,qc[0],dp,qc[1],xtitle),
                    xy=(yv[4],xv[4]),xytext=(0.25,0.80),textcoords='figure fraction')

            # plt.show()
            pdf.savefig()
            plt.close()

        if scattermatrix:
            dfnamex = dfname.sample(frac=cmdlsample).copy()
            scatter_matrix(dfnamex)
            pdf.savefig()
            plt.show()
            plt.close()

def savefiles(seisf=None,sdf=None,sxydf=None,
    wellf=None, wdf=None, wxydf=None,
    outdir=None,ssuffix='',wsuffix='',name2merge=None):
    """Generic function to save csv & txt files."""
    if seisf:
        dirsplit,fextsplit = os.path.split(seisf)
        fname1,fextn = os.path.splitext(fextsplit)

        if name2merge:
            dirsplit2,fextsplit2 = os.path.split(name2merge)
            fname2,fextn2 = os.path.splitext(fextsplit2)
            fname = fname1 + '_' + fname2
        else:
            fname = fname1

        if outdir:
            slgrf = os.path.join(outdir,fname) + ssuffix + ".csv"
        else:
            slgrf = os.path.join(dirsplit,fname) + ssuffix + ".csv"
        # if not sdf.empty:
        if isinstance(sdf,pd.DataFrame):
            sdf.to_csv(slgrf,index=False)
            print('Successfully generated %s file' % slgrf)

        if outdir:
            slgrftxt = os.path.join(outdir,fname) + ssuffix + ".txt"
        else:
            slgrftxt = os.path.join(dirsplit,fname) + ssuffix + ".txt"
        if isinstance(sxydf,pd.DataFrame):
            sxydf.to_csv(slgrftxt,sep=' ',index=False)
            print('Successfully generated %s file' % slgrftxt)

    if wellf:
        dirsplit,fextsplit = os.path.split(wellf)
        fname1,fextn = os.path.splitext(fextsplit)

        if name2merge:
            dirsplit2,fextsplit2 = os.path.split(name2merge)
            fname2,fextn2 = os.path.splitext(fextsplit2)
            fname = fname1 + '_' + fname2
        else:
            fname = fname1

        if outdir:
            wlgrf = os.path.join(outdir,fname) + wsuffix + ".csv"
        else:
            wlgrf = os.path.join(dirsplit,fname) + wsuffix + ".csv"
        if isinstance(wdf,pd.DataFrame):
            wdf.to_csv(wlgrf,index=False)
            print('Successfully generated %s file' % wlgrf)

        if outdir:
            wlgrftxt = os.path.join(outdir,fname) + wsuffix + ".txt"
        else:
            wlgrftxt = os.path.join(dirsplit,fname) + wsuffix + ".txt"
        if isinstance(wxydf,pd.DataFrame):
            wxydf.to_csv(wlgrftxt,sep=' ',index=False)
            print('Successfully generated %s file' % wlgrftxt)

def listfiles(flst):
    """Print the list file."""
    for fl in flst:
        print(fl)

def plot_confusion_matrix(cm, classes,
    normalize=False,
    title='Confusion Matrix',
    cmap=plt.cm.Set2,
    hideplot=False):
    """
    Print and plot the confusion matrix.

    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig,ax = plt.subplots()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_roc_curve(y_test,
    preds,
    poslbl,
    hideplot=False):
    """Calculate the fpr and tpr for all thresholds of the classification."""
    fpr, tpr, threshold = roc_curve(y_test, preds,pos_label=poslbl)
    roc_auc = auc(fpr, tpr)
    plt.title('ROC Curve  %1d' % poslbl)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

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

def plotwellslog(wdf,hideplots=True):
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


def plotwellshist(wdf,hideplots=True):
    """."""
    wlst = wdf[wdf.columns[0]].unique().tolist()
    # print(wlst)

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



def process_dropcols(csvfile,cmdlcols2drop=None,cmdloutdir=None):
    """Drop columns."""
    allattrib = pd.read_csv(csvfile)
    # print(allattrib.head(5))
    # cols = allattrib.columns.tolist()
    allattrib.drop(allattrib.columns[[cmdlcols2drop]],axis=1,inplace=True)
    # print(allattrib.head(5))
    # cols = allattrib.columns.tolist()

    savefiles(seisf=csvfile,
        sdf=allattrib,
        outdir=cmdloutdir,
        ssuffix='_drpc')


def process_listcsvcols(csvfile):
    """List enumerate csv columns."""
    data = pd.read_csv(csvfile)
    clist = list(enumerate(data.columns))
    print(clist)


def process_PCAanalysis(cmdlallattribcsv,cmdlacolsrange=None,
    cmdlanalysiscols=None,cmdlhideplot=False):
    """Filter out components."""
    swa = pd.read_csv(cmdlallattribcsv)
    if cmdlacolsrange:
        print('From col# %d to col %d' % (cmdlacolsrange[0],cmdlacolsrange[1]))
        X = swa[swa.columns[cmdlacolsrange[0]: cmdlacolsrange[1] + 1]].values
    else:
        print('analysis cols',cmdlanalysiscols)
        X = swa[swa.columns[cmdlanalysiscols]].values
    # Create scaler: scaler
    scaler = StandardScaler()

    # Create a PCA instance: pca
    pca = PCA()

    # Create pipeline: pipeline
    pipeline = make_pipeline(scaler,pca)

    # Fit the pipeline to well data
    pipeline.fit(X)
    dirsplit,fextsplit = os.path.split(cmdlallattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    pdfsave = os.path.join(dirsplit,fname) + "_pca.pdf"

    # Plot the explained variances
    features = range(pca.n_components_)
    with PdfPages(pdfsave) as pdf:
        plt.figure(figsize=(8,8))
        plt.bar(features, pca.explained_variance_)
        plt.xlabel('PCA feature')
        plt.ylabel('variance')
        plt.xticks(features)
        plt.title('Elbow Plot')
        pdf.savefig()
        if not cmdlhideplot:
            plt.show()
        plt.close()

def process_PCAfilter(cmdlallattribcsv,
    cmdlacolsrange=None,
    cmdlanalysiscols=None,
    cmdltargetcol=None,
    cmdlncomponents=None,
    cmdloutdir=None,
    cmdlcols2addback=None):
    """PCA keep selected components only."""
    swa = pd.read_csv(cmdlallattribcsv)
    if cmdlacolsrange:
        print('From col# %d to col %d' % (cmdlacolsrange[0],cmdlacolsrange[1]))
        X = swa[swa.columns[cmdlacolsrange[0]: cmdlacolsrange[1] + 1]].values
        colnames = swa[swa.columns[cmdlacolsrange[0]: cmdlacolsrange[1] + 1]].columns
    else:
        print('analysis cols',cmdlanalysiscols)
        X = swa[swa.columns[cmdlanalysiscols]].values
        colnames = swa[swa.columns[cmdlanalysiscols]].columns
    if cmdltargetcol:
        targetname = swa.columns[cmdltargetcol]
        print(colnames,targetname)

    # Create scaler: scaler
    # scaler = StandardScaler()

    # Create a PCA instance: pca
    if not cmdlncomponents:
        pca = PCA(X.shape[1])
        colnames = list()
        # [colnames.append('PCA%d'%i) for i in range(X.shape[1] -1)]
        [colnames.append('PCA%d' % i) for i in range(X.shape[1])]
    else:
        pca = PCA(cmdlncomponents)
        colnames = list()
        # [colnames.append('PCA%d'%i) for i in range(cmdl.ncomponents -1)]
        [colnames.append('PCA%d' % i) for i in range(cmdlncomponents)]

    if cmdltargetcol:
        colnames.append(targetname)
    # Create pipeline: pipeline
    # pipeline = make_pipeline(scaler,pca)

    # Fit the pipeline to well data
    # CX = pipeline.fit_transform(X)
    CX = pca.fit_transform(X)
    print('cx shape',CX.shape,'ncolumns ',len(colnames))
    swa0 = swa[swa.columns[cmdlcols2addback]]
    cxdf = pd.DataFrame(CX,columns=colnames)
    if cmdltargetcol:
        cxdf[targetname] = swa[swa.columns[cmdltargetcol]]
    cxdfall = pd.concat([swa0,cxdf],axis=1)

    savefiles(seisf=cmdlallattribcsv,
        sdf=cxdfall, sxydf=cxdfall,
        outdir=cmdloutdir,
        ssuffix='_pca')

def process_scattermatrix(cmdlallattribcsv,cmdlwellxyzcols=None,cmdlsample=None):
    """Plot scatter matrix for all attributes."""
    swa = pd.read_csv(cmdlallattribcsv)
    # print(swa.sample(5))
    swax = swa.drop(swa.columns[cmdlwellxyzcols],axis=1)
    swaxx = swax.sample(frac=cmdlsample).copy()
    scatter_matrix(swaxx)
    plt.show()

def process_eda(cmdlallattribcsv,
    cmdlxyzcols=None,
    cmdlpolydeg=None,
    cmdlsample=None,
    cmdlhideplot=False,
    cmdlplotoption=None,
    cmdloutdir=None):
    """Generate Exploratroy Data Analyses plots."""
    plt.style.use('seaborn-whitegrid')
    swa = pd.read_csv(cmdlallattribcsv)
    swax = swa.drop(swa.columns[cmdlxyzcols],axis=1)
    if cmdlsample:
        swax = swax.sample(frac=cmdlsample).copy()
        print('**********Data has been resampled by %.2f' % cmdlsample)
    dirsplit,fextsplit = os.path.split(cmdlallattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    def pltundefined():
        pass

    def pltheatmap():
        if cmdloutdir:
            pdfheat = os.path.join(cmdloutdir,fname) + "_heat.pdf"
        else:
            pdfheat = os.path.join(dirsplit,fname) + "_heat.pdf"

        plt.figure(figsize=(8,8))
        mask = np.zeros_like(swax.corr(), dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        ht = sns.heatmap(swax.corr(),
                    vmin=-1, vmax=1,
                    square=True,
                    cmap='RdBu_r',
                    mask=mask,
                    linewidths=.5)
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)

        if not cmdlhideplot:
            plt.show()
        fig = ht.get_figure()
        fig.savefig(pdfheat)

    def pltxplots():
        # decimal places for display
        dp = 3
        ytitle = swax.columns[-1]
        # assume the xplots are for attributes vs target
        # that assumption is not valid for other plots
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_xplots.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_xplots.pdf"
        with PdfPages(pdfcl) as pdf:
            for i in range((swax.shape[1]) - 1):

                xtitle = swax.columns[i]

                xv = swax.iloc[:,i].values
                yv = swax.iloc[:,-1].values
                xrngmin,xrngmax = xv.min(),xv.max()
                # print(xrngmin,xrngmax)
                xvi = np.linspace(xrngmin,xrngmax)
                # print(xrng)
                qc = np.polyfit(xv,yv,cmdlpolydeg)
                if cmdlpolydeg == 1:
                    print('%s  vs %s  Slope: %5.3f, Intercept: %5.3f'% (xtitle,ytitle,qc[0],qc[1]))
                else:
                    print('%s  vs %s ' % (xtitle,ytitle),qc)
                yvi = np.polyval(qc,xvi)
                plt.scatter(xv,yv,alpha=0.5)
                plt.plot(xvi,yvi,c='red')
                plt.xlabel(xtitle)
                plt.ylabel(ytitle)
                # commenting out annotation : only shows on last plot!!
                if cmdlpolydeg == 1:
                    plt.annotate('%s = %-.*f   + %-.*f * %s' % (ytitle,dp,qc[0],dp,qc[1],xtitle),
                        xy=(yv[4],xv[4]),xytext=(0.25,0.80),textcoords='figure fraction')

                # if not cmdlhideplot:
                    # plt.show()
                # fig = p0.get_figure()
                # fig.savefig(pdfcl)
                # plt.close()
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

    def pltbox():
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_box.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_box.pdf"
        with PdfPages(pdfcl) as pdf:
            for i in range((swax.shape[1])):
                # xtitle = swax.columns[i]
                sns.boxplot(x=swax.iloc[:,i])
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()
            ax = sns.boxplot(data=swax)
            labels = ax.get_xticklabels()
            plt.setp(labels, rotation=30, fontsize=10)
            # ax.set_xticklabels(xticklabels, rotation = 45)
            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

    def pltdistribution():
        plt.style.use('seaborn-whitegrid')
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_dist.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_dist.pdf"
        with PdfPages(pdfcl) as pdf:
            for i in range(swax.shape[1]):
                sns.distplot(swax.iloc[:,i])
                # title = swax.columns[i]

                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

    def pltscattermatrix():
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_scatter.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_scatter.pdf"
        scatter_matrix(swax, alpha=0.2, figsize=(8, 8), diagonal='kde')
        if not cmdlhideplot:
            plt.show()
        plt.savefig(pdfcl)
        plt.close()

    plotoptlist = {'xplots': pltxplots,'heatmap': pltheatmap,
                'box':pltbox,
                'distribution':pltdistribution,
                # 'distribution':lambda: pltdistribution(swax),
                'scattermatrix':pltscattermatrix}
    plotoptlist.get(cmdlplotoption,pltundefined)()


def process_linreg(cmdlwellattribcsv,
    cmdlwcolsrange=None,
    cmdlwpredictorcols=None,
    cmdlwtargetcol=None,
    cmdlminmaxscale=None,
    cmdloutdir=None,
    cmdlhideplot=False):
    """Perform linear fitting and prediction."""
    swa = pd.read_csv(cmdlwellattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1]]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]


    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)
    # plt.style.use('seaborn-whitegrid')
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) + "_linregxplt.pdf"
        xyplt = os.path.join(cmdloutdir,fname) + "_linregxplt.csv"
        boxpfname = os.path.join(cmdloutdir,fname) + '_linregbox.pdf'
        modelfname = os.path.join(cmdloutdir,fname) + '_linregmdl.sav'
        predfname = os.path.join(cmdloutdir,fname) + '_linreg.csv'
    else:
        pdfcl = os.path.join(dirsplit,fname) + "_linregxplt.pdf"
        xyplt = os.path.join(dirsplit,fname) + "_linregxplt.csv"
        boxpfname = os.path.join(dirsplit,fname) + "_linregbox.pdf"
        modelfname = os.path.join(dirsplit,fname) + '_linregmdl.sav'
        predfname = os.path.join(dirsplit,fname) + '_linreg.csv'


    lm = LinearRegression()
    lm.fit(X, y)
    # Fitting all predictors 'X' to the target 'y' using linear fit model
    ypred = lm.predict(X)
    if cmdlminmaxscale:
        ymin,ymax = y.min(), y.max()
        mmscale = MinMaxScaler((ymin,ymax))
        # mmscale.fit(pred)
        pred1 = ypred.reshape(-1,1)
        ypred = mmscale.fit_transform(pred1).reshape(-1)

    # Print intercept and coefficients
    print('Intercept: ',lm.intercept_)
    print('Coefficients: ',lm.coef_)
    print('R2 Score:',lm.score(X, y))

    # Calculating coefficients
    cflst = lm.coef_.tolist()
    # cflst.append(lm.intercept_)
    cflst.insert(0,lm.intercept_)
    cnameslst = colnames.tolist()
    # cnameslst.append('Intercept')
    cnameslst.insert(0,'Intercept')
    coeff = pd.DataFrame(cnameslst,columns=['Attribute'])
    coeff['Coefficient Estimate'] = pd.Series(cflst)

    # Calculating Mean Squared Error
    mse = np.mean((ypred - y)**2)
    print('Metrics on input data: ')
    print('MSE: %.4f' % (mse))
    print('R2 Score: %.4f' % (lm.score(X,y)))
    ccmdl = sts.pearsonr(y,ypred)
    print('Pearsonr : %10.3f  %10.4f ' % (ccmdl[0],ccmdl[1]))

    qc0 = np.polyfit(y,ypred,1)
    xrngmin,xrngmax = y.min(),y.max()
    xvi = np.linspace(xrngmin,xrngmax)
    predcol = swa.columns[cmdlwtargetcol] + 'PRED'
    swa[predcol] = ypred

    fig = plt.figure()
    ax = fig.add_subplot(111)

    yvi0 = np.polyval(qc0,xvi)
    plt.scatter(y,ypred,alpha=0.5,c='b',label='Model Predicted')
    plt.plot(xvi,yvi0,c='k',lw=2)

    ax.annotate('Model = %-.*f * Actual + %-.*f' %
        (2,qc0[0],2,qc0[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.85),
        textcoords='figure fraction', fontsize=10)
    ax.annotate('Model Pearson cc = %-.*f   Pearson p = %-.*f' %
        (2,ccmdl[0],3,ccmdl[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.81),
        textcoords='figure fraction', fontsize=10)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'LinReg {swa.columns[cmdlwtargetcol]}' )

    if not cmdlhideplot:
        plt.show()

    fig = ax.get_figure()
    fig.savefig(pdfcl)
    plt.boxplot([y,ypred],labels=['Actual','Model'],showmeans=True,notch=True)
    plt.title(f'LinReg for {swa.columns[cmdlwtargetcol]}')
    if not cmdlhideplot:
        plt.show()
    fig = ax.get_figure()
    fig.savefig(boxpfname)


    pickle.dump(lm, open(modelfname, 'wb'))
    print(f'Sucessfully saved model {modelfname}')
    swa.to_csv(predfname,index=False)
    print(f'Sucessfully saved predicted file {predfname}')

def process_featureranking(cmdlallattribcsv,
    cmdlwtargetcol=None,
    cmdlwcolsrange=None,
    cmdlwpredictorcols=None,
    cmdltestfeatures=None,
    cmdllassoalpha=None,
    cmdlfeatures2keep=None,
    cmdlcv=None,
    cmdltraintestsplit=None):
    """Rank features with different approaches."""
    swa = pd.read_csv(cmdlallattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    if cmdltestfeatures == 'rlasso':
        rlasso = RandomizedLasso(alpha=cmdllassoalpha)
        rlasso.fit(X, y)
        print("Features sorted by their score:")
        print(sorted(zip(map(lambda x: round(x, 4), rlasso.scores_),colnames), reverse=True))
        # print (sorted(zip(rlasso.scores_,colnames), reverse=True))

    elif cmdltestfeatures == 'rfe':
        # rank all features, i.e continue the elimination until the last one
        lm = LinearRegression()
        rfe = RFE(lm, n_features_to_select=cmdlfeatures2keep)
        rfe.fit(X,y)
        # print ("Features sorted by their rank:")
        # print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), colnames)))

        scores = []
        for i in range(X.shape[1]):
            score = cross_val_score(lm, X[:, i:i + 1], y, scoring="r2",
                cv=ShuffleSplit(len(X), cmdlcv, cmdltraintestsplit))
        # scores.append(round(np.mean(score), 3))
        scores.append(np.mean(score))
        # print (sorted(scores, reverse=True))
        r2fr = pd.DataFrame(sorted(zip(scores, colnames),reverse=True),columns=['R2 Score ','Attribute'])
        print('Feature Ranking by R2 scoring: ')
        print(r2fr)

    elif cmdltestfeatures == 'svrcv':
        # rank all features, i.e continue the elimination until the last one

        estimator = SVR(kernel="linear")
        selector = RFECV(estimator, step=1, cv=cmdlcv)
        selector = selector.fit(X, y)
        fr = pd.DataFrame(sorted(zip(selector.ranking_, colnames)),columns=['Importance','Attribute'])
        print('Feature Ranking with Cross Validated Recursive Feature Elimination Using SVR: ')
        print(fr)

    elif cmdltestfeatures == 'svr':
        estimator = SVR(kernel="linear")
        selector = RFE(estimator, cmdlfeatures2keep, step=1)
        selector = selector.fit(X, y)
        fr = pd.DataFrame(sorted(zip(selector.ranking_, colnames)),columns=['Importance','Attribute'])
        print('Feature Ranking with Recursive Feature Elimination Using SVR: ')
        print(fr)

    elif cmdltestfeatures == 'rfregressor':
        rf = RandomForestRegressor(n_estimators=20, max_depth=4)
        rf.fit(X,y)
        fi = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), colnames),
            reverse=True),columns=['Importance','Attribute'])
        print(fi)
        # print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), colnames)))
        # print(rf.feature_importances_)
        scores = []

        for i in range(X.shape[1]):
            score = cross_val_score(rf, X[:, i:i + 1], y, scoring="r2",
                  cv=ShuffleSplit(len(X), cmdlcv, cmdltraintestsplit))
            scores.append((round(np.mean(score), 3), colnames[i]))
        cvscoredf = pd.DataFrame(sorted(scores,reverse=True),columns=['Partial R2','Attribute'])
        print('\nCross Validation:')
        print(cvscoredf)

    elif cmdltestfeatures == 'decisiontree':
        regressor = DecisionTreeRegressor(random_state=0)
        # cross_val_score(regressor, X, y, cv=3)
        regressor.fit(X,y)
        # print(regressor.feature_importances_)
        fr = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), regressor.feature_importances_),
            colnames), reverse=True),columns=['Importance','Attribute'])
        print('Feature Ranking with Decision Tree Regressor: ')
        print(fr)

def process_KNNtest(cmdlallattribcsv,
    cmdlsample=1.0,
    cmdlwcolsrange=None,
    cmdlwpredictorcols=None,
    cmdlwtargetcol=None,
    cmdlcv=None,
    cmdlhideplot=None,
    cmdloutdir=None):
    """Test for # of neighbors for KNN regression."""
    swa = pd.read_csv(cmdlallattribcsv)
    swa = swa.sample(frac=cmdlsample).copy()

    if cmdlwcolsrange:
        print('Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        # colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    k_values = np.array([n for n in range(1,21)])
    # print('kvalues:',k_values)
    mselist = []
    stdlist = []
    for k in k_values:
        # kfold = KFold(n_splits=10, random_state=7)
        kfold = KFold(n_splits=cmdlcv, random_state=7)
        KNNmodel = KNeighborsRegressor(n_neighbors=k)
        scoring = 'neg_mean_squared_error'
        results = cross_val_score(KNNmodel, X, y, cv=kfold, scoring=scoring)
        print("K value: %2d  MSE: %.3f (%.3f)" % (k,results.mean(), results.std()))
        mselist.append(results.mean())
        stdlist.append(results.std())

    dirsplit,fextsplit = os.path.split(cmdlallattribcsv)
    fname,fextn = os.path.splitext(fextsplit)
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) + "_knn.pdf"
    else:
        pdfcl = os.path.join(dirsplit,fname) + "_knn.pdf"
    plt.plot(k_values,mselist)
    plt.xlabel('# of clusters')
    plt.ylabel('Neg Mean Sqr Error')
    plt.savefig(pdfcl)
    if not cmdlhideplot:
        plt.show()

def process_KNN(cmdlwellattribcsv,
    cmdlwcolsrange=None,
    cmdlwpredictorcols=None,
    cmdlwtargetcol=None,
    cmdlkneighbors=None,
    cmdlminmaxscale=None,
    cmdloutdir=None,
    cmdlgeneratesamples=None,
    cmdlhideplot=False):
    """Use K Nearest Neigbors to fit regression."""
    swa = pd.read_csv(cmdlwellattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1]]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        # colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    if cmdlgeneratesamples:
        # Xoriginal = X.copy()
        # yoriginal = y.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='r',func='svr')


    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)
    plt.style.use('seaborn-whitegrid')
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) + f"_knnr{cmdlkneighbors}.pdf"
        xyplt = os.path.join(cmdloutdir,fname) + f"_knnrxplt{cmdlkneighbors}.csv"
        boxpfname = os.path.join(cmdloutdir,fname) + '_knnrbox.pdf'
        modelfname = os.path.join(cmdloutdir,fname) + f'_knnrmdl{cmdlkneighbors}.sav'
        predfname = os.path.join(cmdloutdir,fname) + f'_knnr{cmdlkneighbors}.csv'
    else:
        pdfcl = os.path.join(dirsplit,fname) + f"_knnr{cmdlkneighbors}.pdf"
        xyplt = os.path.join(dirsplit,fname) + f"_knnrxplt{cmdlkneighbors}.csv"
        boxpfname = os.path.join(dirsplit,fname) + "_knnrbox.pdf"
        modelfname = os.path.join(dirsplit,fname) + f'_knnrmdl{cmdlkneighbors}.sav'
        predfname = os.path.join(dirsplit,fname) + f'_knnr{cmdlkneighbors}.csv'

    KNNmodel = KNeighborsRegressor(n_neighbors=cmdlkneighbors)
    KNNmodel.fit(X, y)

    ypred = KNNmodel.predict(X)
    if cmdlminmaxscale:
        ymin,ymax = y.min(), y.max()
        mmscale = MinMaxScaler((ymin,ymax))
        # mmscale.fit(pred)
        pred1 = ypred.reshape(-1,1)
        ypred = mmscale.fit_transform(pred1).reshape(-1)

    # Calculating Mean Squared Error
    mse = np.mean((ypred - y)**2)
    print('Metrics on input data: ')
    print('MSE: %.4f' % (mse))
    print('R2 Score: %.4f' % (KNNmodel.score(X,y)))
    ccmdl = sts.pearsonr(y,ypred)
    print('Pearsonr : %10.3f  %10.4f ' % (ccmdl[0],ccmdl[1]))

    qc0 = np.polyfit(y,ypred,1)
    xrngmin,xrngmax = y.min(),y.max()
    xvi = np.linspace(xrngmin,xrngmax)

    predcol = swa.columns[cmdlwtargetcol] + 'PRED'
    swa[predcol] = ypred

    fig = plt.figure()
    ax = fig.add_subplot(111)

    yvi0 = np.polyval(qc0,xvi)
    plt.scatter(y,ypred,alpha=0.5,c='b',label='Model Predicted')
    plt.plot(xvi,yvi0,c='k',lw=2)

    ax.annotate('Model = %-.*f * Actual + %-.*f' %
        (2,qc0[0],2,qc0[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.85),
        textcoords='figure fraction', fontsize=10)
    ax.annotate('Model Pearson cc = %-.*f   Pearson p = %-.*f' %
        (2,ccmdl[0],3,ccmdl[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.81),
        textcoords='figure fraction', fontsize=10)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'KNN Regressor {swa.columns[cmdlwtargetcol]} {cmdlkneighbors}')

    if not cmdlhideplot:
        plt.show()

    fig = ax.get_figure()
    fig.savefig(pdfcl)
    plt.boxplot([y,ypred],labels=['Actual','Model'],showmeans=True,notch=True)
    plt.title(f'KNN for {swa.columns[cmdlwtargetcol]}{cmdlkneighbors}')
    if not cmdlhideplot:
        plt.show()
    fig = ax.get_figure()
    fig.savefig(boxpfname)


    if not cmdlgeneratesamples:
        # xpltcols = ['Actual','Predicted']
        xpltdf = swa.iloc[:,0].copy()
        # copy well x y
        xpltdf['Actual'] = y
        xpltdf['Predicted'] = ypred
        xpltdf.to_csv(xyplt,index=False)
        print('Sucessfully generated xplot file %s' % xyplt)

    pickle.dump(KNNmodel, open(modelfname, 'wb'))
    print(f'Sucessfully pickled model {modelfname}')
    swa.to_csv(predfname,index=False)
    print(f'Sucessfully saved predicted file {predfname}')


def process_CatBoostRegressor(cmdlwellattribcsv,
    cmdlwcolsrange=None,
    cmdlwpredictorcols=None,
    cmdlwtargetcol=None,
    cmdlminmaxscale=None,
    cmdloutdir=None,
    cmdliterations=None,
    cmdllearningrate=None,
    cmdldepth=None,
    cmdlcv=None,
    cmdlscaleminmaxvalues=None,
    cmdlfeatureimportance=None,
    cmdlimportancelevel=None,
    cmdlgeneratesamples=None,
    cmdlhideplot=False,
    cmdlvalsize=0.3,
    cmdloverfittingdetection=False,
    cmdlodpval=None):
    """CatBoostRegression."""
    swa = pd.read_csv(cmdlwellattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    if cmdlgeneratesamples:
        # Xoriginal = X.copy()
        # yoriginal = y.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='r',func='cbr')


    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) + f"I{cmdliterations}_LR{cmdllearningrate:.3f}_D{cmdldepth}" + "_cbr.pdf"
        xyplt = os.path.join(cmdloutdir,fname) + f"I{cmdliterations}_LR{cmdllearningrate:.3f}_D{cmdldepth}"  + "_cbrxplt.csv"
        boxpfname = os.path.join(cmdloutdir,fname) + f"I{cmdliterations}_LR{cmdllearningrate:.3f}_D{cmdldepth}" + '_cbrbox.pdf'
        modelfname = os.path.join(cmdloutdir,fname) + f"I{cmdliterations}_LR{cmdllearningrate:.3f}_D{cmdldepth}" + '_cbrpklmdl.sav'
        savemodelfname = os.path.join(cmdloutdir,fname) + f"I{cmdliterations}_LR{cmdllearningrate:.3f}_D{cmdldepth}" + '_cbrmdl.sav'
        predfname = os.path.join(cmdloutdir,fname) + f"I{cmdliterations}_LR{cmdllearningrate:.3f}_D{cmdldepth}" + '_cbr.csv'
    else:
        pdfcl = os.path.join(dirsplit,fname) + f"I{cmdliterations}_LR{cmdllearningrate:.3f}_D{cmdldepth}" + "_cbr.pdf"
        xyplt = os.path.join(dirsplit,fname) + f"I{cmdliterations}_LR{cmdllearningrate:.3f}_D{cmdldepth}"  + "_cbrxplt.csv"
        boxpfname = os.path.join(dirsplit,fname) + f"I{cmdliterations}_LR{cmdllearningrate:.3f}_D{cmdldepth}" + '_cbrbox.pdf'
        modelfname = os.path.join(dirsplit,fname) + f"I{cmdliterations}_LR{cmdllearningrate:.3f}_D{cmdldepth}" + '_cbrbox.pdf'
        savemodelfname = os.path.join(dirsplit,fname) + f"I{cmdliterations}_LR{cmdllearningrate:.3f}_D{cmdldepth}" + '_cbrmdl.sav'
        predfname = os.path.join(dirsplit,fname) + f"I{cmdliterations}_LR{cmdllearningrate:.3f}_D{cmdldepth}" + '_cbr.csv'


    if cmdlfeatureimportance:
        model = CatBoostRegressor(iterations=cmdliterations, learning_rate=cmdllearningrate,
                    depth=cmdldepth,loss_function='RMSE',calc_feature_importance=True,
                    random_seed=42,logging_level='Silent')
        model.fit(X, y)
        fr = pd.DataFrame(sorted(zip(model.get_feature_importance(X,y), colnames),reverse=True),columns=['Importance','Attribute'])
        if cmdloutdir:
            fifname = os.path.join(cmdloutdir,fname) + f"_fi{cmdlimportancelevel:.1f}.csv"
            fislctfname = os.path.join(cmdloutdir,fname) + f"_fislct{cmdlimportancelevel:.1f}.csv"
        else:
            fifname = os.path.join(dirsplit,fname) + f"_fi{cmdlimportancelevel:.1f}.csv"
            fislctfname = os.path.join(dirsplit,fname) + f"_fislct{cmdlimportancelevel:.1f}.csv"

        # fr.to_csv(fifname,index=False)
        fr.to_csv(fifname)
        # intentionally save feature importance file with column #
        print(f'Successfully generated Feature Importance file {fifname}')
        frx = fr[fr['Importance'] > cmdlimportancelevel]
        frxcols = swa.columns[:5].tolist() + frx.Attribute.tolist() + [swa.columns[-1]]
        # Well X Y Z + selected attributes + target col
        print(f'# of features retained: {len(frx.Attribute.tolist())}')
        swafi = swa[frxcols].copy()
        # targetname = swa.columns[-1]
        # swafi[targetname] = y
        swafi.to_csv(fislctfname,index=False)
        print(f'Successfully generated selected features file {fislctfname}')
        # print('Feature Ranking with CatBoostRegressor: ')
        # print(fr)

        plt.style.use('seaborn-whitegrid')
        ax = frx['Importance'].plot(kind='bar', figsize=(12,8))
        ax.set_xticklabels(frx['Attribute'],rotation=45)
        ax.set_ylabel('Feature Importance')
        ax.set_title('CatBoostRegressor Feature Importance of %s' % cmdlwellattribcsv)
        if not cmdlhideplot:
            plt.show()
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + f"_cbrfi{cmdlimportancelevel:.1f}.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + f"_cbrfi{cmdlimportancelevel:.1f}.pdf"
        fig = ax.get_figure()
        fig.savefig(pdfcl)

    elif cmdlcv:
        model = CatBoostRegressor(iterations=cmdliterations, learning_rate=cmdllearningrate,
                    depth=cmdldepth,loss_function='RMSE',random_seed=42,logging_level='Silent')
        cvscore = cross_val_score(model,X,y,cv=cmdlcv,scoring='r2')
        # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        print("Mean Score R2: %10.4f" % (np.mean(cvscore[0])))
        # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        cvscore = cross_val_score(model,X,y,cv=cmdlcv,scoring='neg_mean_squared_error')
        print("Mean MSE: %10.4f" % (np.mean(np.abs(cvscore[1]))))
        print('No files will be generated. Re-run without cross validation')

    # does not work: problem with eval_set variable Mar 1 2018
    elif cmdloverfittingdetection:
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)
        evalset = (Xtest,ytest)
        model = CatBoostRegressor(iterations=cmdliterations,
                    learning_rate=cmdllearningrate,
                    depth=cmdldepth,
                    loss_function='RMSE',
                    use_best_model=True,
                    od_type='IncToDec',
                    od_pval=cmdlodpval,
                    eval_metric='RMSE',
                    random_seed=42,logging_level='Silent')

        # Fit model
        model.fit(X, y,eval_set=evalset)
        # Get predictions
        ypred = model.predict(X)
        # Calculating Mean Squared Error
        mse = np.mean((ypred - y)**2)
        print('Metrics on input data: ')
        print('MSE: %.4f' % (mse))
        r2 = r2_score(y,ypred)
        print('R2 : %10.3f' % r2)
        print(model.get_params())
        # model.save_model('CBRmodel.mdl')
    else:
        model = CatBoostRegressor(iterations=cmdliterations, learning_rate=cmdllearningrate,
                depth=cmdldepth,loss_function='RMSE',random_seed=42,logging_level='Silent')
        # Fit model
        model.fit(X, y)
        # Get predictions
        ypred = model.predict(X)

        if cmdlminmaxscale:
            ymin,ymax = y.min(), y.max()
            mmscale = MinMaxScaler((ymin,ymax))
            # mmscale.fit(pred)
            pred1 = ypred.reshape(-1,1)
            ypred = mmscale.fit_transform(pred1).reshape(-1)

        # Calculating Mean Squared Error
        mse = np.mean((ypred - y)**2)
        print('Metrics on input data: ')
        print('MSE: %.4f' % (mse))
        r2 = r2_score(y,ypred)
        print('R2 : %10.3f' % r2)
        ccmdl = sts.pearsonr(y,ypred)
        print('Train-Test-Split Pearsonr : %10.3f  %10.4f ' % (ccmdl[0],ccmdl[1]))
        qc0 = np.polyfit(y,ypred,1)
        xrngmin,xrngmax = y.min(),y.max()
        xvi = np.linspace(xrngmin,xrngmax)


        Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
            random_state=42)
        model.fit(Xtrain, ytrain)
        print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))

        yvalpred = model.predict(Xval)

        # Calculating Mean Squared Error
        msev = np.mean((yvalpred - yval)**2)
        print('Metrics on Train-Test-Split data: ')
        print('Train-Test-Split MSE: %.4f' % (msev))
        r2v = r2_score(yval,yvalpred)
        print('Train-Test-Split R2 : %10.3f' % r2v)
        ccxv = sts.pearsonr(yval,yvalpred)
        print('Train-Test-Split Pearsonr : %10.3f  %10.4f ' % (ccxv[0],ccxv[1]))

        predcol = swa.columns[cmdlwtargetcol] + 'PRED'
        swa[predcol] = ypred
        plotwellslog(swa,cmdlhideplot)
        plotwellshist(swa,cmdlhideplot)

        fig = plt.figure()

        ax = fig.add_subplot(111)
        # qc0 = np.polyfit(y,ypred,1) #has already been calculated above
        yvi0 = np.polyval(qc0,xvi)
        plt.scatter(y,ypred,alpha=0.5,c='b',label='Model Predicted')
        plt.plot(xvi,yvi0,c='k',lw=2)

        ax.annotate('Model = %-.*f * Actual + %-.*f' %
            (2,qc0[0],2,qc0[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.85),
            textcoords='figure fraction', fontsize=10)
        ax.annotate ('Model Pearson cc = %-.*f   Pearson p = %-.*f' %
            (2,ccmdl[0],3,ccmdl[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.81),
            textcoords='figure fraction', fontsize=10)
        plt.title(f'CatBoostRegressor for {swa.columns[cmdlwtargetcol]} I {cmdliterations} LR {cmdllearningrate} D {cmdldepth}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        if not cmdlhideplot:
            plt.show()
        fig = ax.get_figure()
        fig.savefig(pdfcl)

        plt.boxplot([y,ypred],labels=['Actual','Model'],showmeans=True,notch=True)
        plt.title(f'CatBoostRegressor for {swa.columns[cmdlwtargetcol]} I {cmdliterations} LR {cmdllearningrate} D {cmdldepth}')
        if not cmdlhideplot:
            plt.show()
        fig = ax.get_figure()
        fig.savefig(boxpfname)


        if not cmdlgeneratesamples:
            # xpltcols = ['Actual','Predicted']
            xpltdf = swa.iloc[:,0].copy()
            # copy well name only
            xpltdf['Actual'] = y
            xpltdf['Predicted'] = ypred
            xpltdf.to_csv(xyplt,index=False)
            print('Sucessfully generated xplot file %s' % xyplt)
        # save model both as pickle and via catboost
        pickle.dump(model, open(modelfname, 'wb'))
        print(f'Sucessfully saved/pickled model {modelfname}')
        model.save_model(savemodelfname,format="json",export_parameters=None)
        print(f'Successfully saved CatBoostRegressor Model {savemodelfname}')

        swa.to_csv(predfname,index=False)
        print(f'Sucessfully saved predicted file {predfname}')



# **********NuSVR support vector regresssion: uses nusvr
def process_NuSVR(cmdlwellattribcsv,
    cmdlwcolsrange=None,
    cmdlwpredictorcols=None,
    cmdlwtargetcol=None,
    cmdlminmaxscale=None,
    cmdlscaleminmaxvalues=None,
    cmdloutdir=None,
    cmdlnu=None,
    cmdlerrpenalty=None,
    cmdlcv=None,
    cmdlhideplot=False,
    cmdlvalsize=0.3,
    cmdlgeneratesamples=None):
    """Nu SVR Support Vector Machine Regression."""
    swa = pd.read_csv(cmdlwellattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        # colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    if cmdlgeneratesamples:
        # Xoriginal = X.copy()
        # yoriginal = y.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='r',func='svr')

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) + f"C{cmdlerrpenalty}_nu{cmdlnu}svr.pdf"
        xyplt = os.path.join(cmdloutdir,fname) + f"C{cmdlerrpenalty}_nu{cmdlnu}_svrxplt.csv"
        boxpfname = os.path.join(cmdloutdir,fname) + f"C{cmdlerrpenalty}_nu{cmdlnu}_svrbox.pdf"
        modelfname = os.path.join(cmdloutdir,fname) + f"C{cmdlerrpenalty}_nu{cmdlnu}_svr.sav"
        predfname = os.path.join(cmdloutdir,fname) + f"C{cmdlerrpenalty}_nu{cmdlnu}_svr.csv"
    else:
        pdfcl = os.path.join(dirsplit,fname) + f"C{cmdlerrpenalty}_nu{cmdlnu}svr.pdf"
        xyplt = os.path.join(dirsplit,fname) + f"C{cmdlerrpenalty}_nu{cmdlnu}_svrxplt.csv"
        boxpfname = os.path.join(dirsplit,fname) + f"C{cmdlerrpenalty}_nu{cmdlnu}_svrbox.pdf"
        modelfname = os.path.join(dirsplit,fname) + f"C{cmdlerrpenalty}_nu{cmdlnu}_svr.sav"
        predfname = os.path.join(dirsplit,fname) + f"C{cmdlerrpenalty}_nu{cmdlnu}_svr.csv"

    if cmdlcv:
        model = NuSVR(C=cmdlerrpenalty, nu=cmdlnu)
        cvscore = cross_val_score(model,X,y,cv=cmdlcv,scoring='r2')
        # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        print("Mean Score R2: %10.4f" % (np.mean(cvscore[0])))
        # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        cvscore = cross_val_score(model,X,y,cv=cmdlcv,scoring='neg_mean_squared_error')
        print("Mean MSE: %10.4f" % (np.mean(np.abs(cvscore[1]))))
        print('No files will be generated. Re-run without cross validation')

    else:
        model = NuSVR(C=cmdlerrpenalty, nu=cmdlnu)
        # Fit model
        model.fit(X, y)
        # Get predictions
        ypred = model.predict(X)

        if cmdlminmaxscale:
            ymin,ymax = y.min(), y.max()
            mmscale = MinMaxScaler((ymin,ymax))
            # mmscale.fit(pred)
            pred1 = ypred.reshape(-1,1)
            ypred = mmscale.fit_transform(pred1).reshape(-1)

        # Calculating Mean Squared Error
        mse = np.mean((ypred - y)**2)
        print('Metrics on input data: ')
        print('MSE: %.4f' % (mse))
        r2 = r2_score(y,ypred)
        print('R2 : %10.3f' % r2)
        ccmdl = sts.pearsonr(y,ypred)
        print('Train-Test-Split Pearsonr : %10.3f  %10.4f ' % (ccmdl[0],ccmdl[1]))

        Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
            random_state=42)
        model.fit(Xtrain, ytrain)
        print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))
        qc0 = np.polyfit(y,ypred,1)
        xrngmin,xrngmax = y.min(),y.max()
        xvi = np.linspace(xrngmin,xrngmax)

        yvalpred = model.predict(Xval)
        # Calculating Mean Squared Error
        msev = np.mean((yvalpred - yval)**2)
        print('Metrics on Train-Test-Split data: ')
        print('Train-Test-Split MSE: %.4f' % (msev))
        r2v = r2_score(yval,yvalpred)
        print('Train-Test-Split R2 : %10.3f' % r2v)

        predcol = swa.columns[cmdlwtargetcol] + 'PRED'
        swa[predcol] = ypred

        fig = plt.figure()
        ax = fig.add_subplot(111)

        yvi0 = np.polyval(qc0,xvi)
        plt.scatter(y,ypred,alpha=0.5,c='b',label='Model Predicted')
        plt.plot(xvi,yvi0,c='k',lw=2)

        ax.annotate('Model = %-.*f * Actual + %-.*f' %
            (2,qc0[0],2,qc0[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.85),
            textcoords='figure fraction', fontsize=10)
        ax.annotate('Model Pearson cc = %-.*f   Pearson p = %-.*f' %
            (2,ccmdl[0],3,ccmdl[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.81),
            textcoords='figure fraction', fontsize=10)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'NuSVR {swa.columns[cmdlwtargetcol]}  C{cmdlerrpenalty}  nu{cmdlnu}')

        if not cmdlhideplot:
            plt.show()
        fig = ax.get_figure()
        fig.savefig(pdfcl)
        plt.boxplot([y,ypred],labels=['Actual','Model'],showmeans=True,notch=True)
        plt.title(f'NuSVR for {swa.columns[cmdlwtargetcol]}  C{cmdlerrpenalty}  nu{cmdlnu}')
        if not cmdlhideplot:
            plt.show()
        fig = ax.get_figure()
        fig.savefig(boxpfname)

        if not cmdlgeneratesamples:
            # xpltcols = ['Actual','Predicted']
            xpltdf = swa.iloc[:,0].copy()
            # copy well name only
            xpltdf['Actual'] = y
            xpltdf['Predicted'] = ypred
            xpltdf.to_csv(xyplt,index=False)
            print('Sucessfully generated xplot file %s' % xyplt)

        # save model both as pickle and via catboost
        pickle.dump(model, open(modelfname, 'wb'))
        print(f'Sucessfully saved/pickled model {modelfname}')

        swa.to_csv(predfname,index=False)
        print(f'Sucessfully saved predicted file {predfname}')


# **********ANN Regressor
def process_ANNRegressor(cmdlwellattribcsv,
    cmdlwcolsrange=None,
    cmdlwpredictorcols=None,
    cmdlwtargetcol=None,
    cmdlminmaxscale=None,
    # min max scaler to scale predicted to input range.
    cmdlscaleminmaxvalues=None,
    cmdloutdir=None,
    cmdlcv=None,
    # same number as num layers
    cmdlnodes=None,
    # same numberof codes as num layers
    cmdlactivation=None,
    # one number
    cmdlepochs=None,
    # one number
    cmdlbatch=None,
    cmdlhideplot=False,
    cmdlvalsize=0.3,
    cmdlgeneratesamples=None):
    """**********ANNRegressor."""
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasRegressor
    from keras.models import model_from_json


    swa = pd.read_csv(cmdlwellattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1]+1]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        # colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    if cmdlgeneratesamples:
        # Xoriginal = X.copy()
        # yoriginal = y.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='r',func='ann')

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    print('cmdlactivation',cmdlactivation)
    cmdllayers = len(cmdlnodes)
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) + f"_L{cmdllayers} " + "_ann.pdf"
        xyplt = os.path.join(cmdloutdir,fname) + f"_L{cmdllayers}"  + "_annxplt.csv"
        boxpfname = os.path.join(cmdloutdir,fname) + f"_L{cmdllayers}" + '_annbox.pdf'
        # modelfname = os.path.join(cmdloutdir,fname) + f"L{cmdllayers}" + '_annpklmdl.sav'
        savemodelfname = os.path.join(cmdloutdir,fname) + f"_L{cmdllayers}" + '_annmdl.json'
        modelwtsfname = os.path.join(cmdloutdir,fname) + f"_L{cmdllayers}" + '_annmdl.h5'
        predfname = os.path.join(cmdloutdir,fname) + f"_L{cmdllayers}" + '_ann.csv'
    else:
        pdfcl = os.path.join(dirsplit,fname) + f"_L{cmdllayers}" + "_ann.pdf"
        xyplt = os.path.join(dirsplit,fname) + f"_L{cmdllayers}"  + "_annxplt.csv"
        boxpfname = os.path.join(dirsplit,fname) + f"_L{cmdllayers}" + '_annbox.pdf'
        # modelfname = os.path.join(dirsplit,fname) + f"_L{cmdllayers}" + '_annbox.pdf'
        savemodelfname = os.path.join(dirsplit,fname) + f"_L{cmdllayers}" + '_annmdl.json'
        modelwtsfname = os.path.join(dirsplit,fname) + f"_L{cmdllayers}" + '_annmdl.h5'
        predfname = os.path.join(dirsplit,fname) + f"_L{cmdllayers}" + '_ann.csv'


    def build_model():
        indim = cmdlwcolsrange[1] - cmdlwcolsrange[0] + 1
        model = Sequential()
        model.add(Dense(cmdlnodes[0], input_dim=indim, kernel_initializer='normal', activation=cmdlactivation[0]))
        for i in range(1,cmdllayers):
            model.add(Dense(cmdlnodes[i], kernel_initializer='normal', activation=cmdlactivation[i]))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model


    if cmdlcv:
        # kfold = KFold(n_splits=cmdlcv, random_state=42)
        estimator = KerasRegressor(build_fn=build_model,
                    epochs=cmdlepochs,
                    batch_size=cmdlbatch,
                    verbose=0)
        cvscore = cross_val_score(estimator,X,y,cv=cmdlcv,scoring='r2')
        # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        print("Mean Score R2: %10.4f" % (np.mean(cvscore[0])))
        # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        cvscore = cross_val_score(estimator,X,y,cv=cmdlcv,scoring='neg_mean_squared_error')
        print("Mean MSE: %10.4f" % (np.mean(np.abs(cvscore[1]))))
        print('No files will be generated. Re-run without cross validation')

    else:
        estimator = KerasRegressor(build_fn=build_model,
                    epochs=cmdlepochs,
                    batch_size=cmdlbatch,
                    verbose=0)
        estimator.fit(X, y)
        # Get predictions
        ypred = estimator.predict(X)

        if cmdlminmaxscale:
            if cmdlscaleminmaxvalues:
                ymin,ymax = cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]
                print('****Manual Scaling of output data to min: %10.4f,  max: %10.4f'
                    % (cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]))
            else:
                ymin,ymax = y.min(), y.max()
            mmscale = MinMaxScaler((ymin,ymax))
            pred1 = ypred.reshape(-1,1)
            ypred = mmscale.fit_transform(pred1).reshape(-1)


        # Calculating Mean Squared Error
        mse = np.mean((ypred - y)**2)
        print('Metrics on input data: ')
        print('MSE: %.4f' % (mse))
        r2 = r2_score(y,ypred)
        print('R2 : %10.3f' % r2)

        Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
            random_state=42)
        estimator.fit(Xtrain, ytrain)
        print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))
        ccmdl = sts.pearsonr(y,ypred)
        qc0 = np.polyfit(y,ypred,1)
        xrngmin,xrngmax = y.min(),y.max()
        xvi = np.linspace(xrngmin,xrngmax)

        yvalpred = estimator.predict(Xval)
        # Calculating Mean Squared Error
        msev = np.mean((yvalpred - yval)**2)
        print('Metrics on Train-Test-Split data: ')
        print('Train-Test-Split MSE: %.4f' % (msev))
        r2v = r2_score(yval,yvalpred)
        print('Train-Test-Split R2 : %10.3f' % r2v)

        predcol = swa.columns[cmdlwtargetcol] + 'PRED'
        swa[predcol] = ypred

        fig = plt.figure()
        ax = fig.add_subplot(111)

        yvi0 = np.polyval(qc0,xvi)
        plt.scatter(y,ypred,alpha=0.5,c='b',label='Model Predicted')
        plt.plot(xvi,yvi0,c='k',lw=2)

        ax.annotate('Model = %-.*f * Actual + %-.*f' %
            (2,qc0[0],2,qc0[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.85),
            textcoords='figure fraction', fontsize=10)
        ax.annotate('Model Pearson cc = %-.*f   Pearson p = %-.*f' %
            (2,ccmdl[0],3,ccmdl[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.81),
            textcoords='figure fraction', fontsize=10)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'ANNRegressor {swa.columns[cmdlwtargetcol]}')

        if not cmdlhideplot:
            plt.show()
        fig = ax.get_figure()
        fig.savefig(pdfcl)

        fig.savefig(pdfcl)
        plt.boxplot([y,ypred],labels=['Actual','Model'],showmeans=True,notch=True)
        plt.title(f'ANN for {swa.columns[cmdlwtargetcol]}  L {cmdllayers}')
        if not cmdlhideplot:
            plt.show()
        fig = ax.get_figure()
        fig.savefig(boxpfname)


        if not cmdlgeneratesamples:
            # xpltcols =['Actual','Predicted']
            xpltdf = swa.iloc[:,0].copy()
            # copy well x y
            xpltdf['Actual'] = y
            xpltdf['Predicted'] = ypred
            xpltdf.to_csv(xyplt,index=False)
            print('Sucessfully generated xplot file %s' % xyplt)

        estimator_json = estimator.model.to_json()
        with open(savemodelfname, "w") as json_file:
            json_file.write(estimator_json)
        # serialize weights to HDF5
        estimator.model.save_weights(modelwtsfname)
        print(f'Sucessfully saved model as {savemodelfname}  and model weights as  {modelwtsfname}')
        swa.to_csv(predfname,index=False)
        print(f'Sucessfully saved predicted file {predfname}')

def process_SGDR(cmdlwellattribcsv,
    cmdlwcolsrange=None,
    cmdlwpredictorcols=None,
    cmdlwtargetcol=None,
    cmdlminmaxscale=None,
    cmdlscaleminmaxvalues=None,
    cmdloutdir=None,
    cmdlloss=None,
    # ‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’
    cmdlpenalty='l2',
    # options: l1,l2,elasticnet,none
    cmdll1ratio=0.15,
    # elastic net mixing: 0 (l2)to 1 (l1)
    cmdlcv=None,
    cmdlhideplot=False,
    cmdlvalsize=0.3,
    cmdlgeneratesamples=None):
    """Stochastic Gradient Descent Regressor OLS/L1/L2 regresssion."""
    swa = pd.read_csv(cmdlwellattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        # colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    if cmdlgeneratesamples:
        # Xoriginal = X.copy()
        # yoriginal = y.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='r',func='svr')

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) + f"L{cmdlloss}_L1R{cmdll1ratio}_P{cmdlpenalty}" + "_sgdr.pdf"
        xyplt = os.path.join(cmdloutdir,fname) + f"L{cmdlloss}_L1R{cmdll1ratio}_P{cmdlpenalty}"   + "_sgdrxplt.csv"
        boxpfname = os.path.join(cmdloutdir,fname) + f"L{cmdlloss}_L1R{cmdll1ratio}_P{cmdlpenalty}"  + '_sgdrbox.pdf'
        modelfname = os.path.join(cmdloutdir,fname) + f"L{cmdlloss}_L1R{cmdll1ratio}_P{cmdlpenalty}"  + '_sgdrpklmdl.sav'
        predfname = os.path.join(cmdloutdir,fname) + f"L{cmdlloss}_L1R{cmdll1ratio}_P{cmdlpenalty}"  + '_sgdr.csv'
    else:
        pdfcl = os.path.join(dirsplit,fname) + f"L{cmdlloss}_L1R{cmdll1ratio}_P{cmdlpenalty}" + "_sgdr.pdf"
        xyplt = os.path.join(dirsplit,fname) + f"L{cmdlloss}_L1R{cmdll1ratio}_P{cmdlpenalty}"  + "_sgdrxplt.csv"
        boxpfname = os.path.join(dirsplit,fname) + f"L{cmdlloss}_L1R{cmdll1ratio}_P{cmdlpenalty}" + '_sgdrbox.pdf'
        modelfname = os.path.join(dirsplit,fname) + f"L{cmdlloss}_L1R{cmdll1ratio}_P{cmdlpenalty}" + '_sgdrpklmdl.sav'
        predfname = os.path.join(dirsplit,fname) + f"L{cmdlloss}_L1R{cmdll1ratio}_P{cmdlpenalty}"  + '_sgdr.csv'

    if cmdlcv:
        model = SGDRegressor(loss=cmdlloss,penalty=cmdlpenalty,l1_ratio=cmdll1ratio)
        cvscore = cross_val_score(model,X,y,cv=cmdlcv,scoring='r2')
        # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        print("Mean Score R2: %10.4f" % (np.mean(cvscore[0])))
        # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        cvscore = cross_val_score(model,X,y,cv=cmdlcv,scoring='neg_mean_squared_error')
        print("Mean MSE: %10.4f" % (np.mean(np.abs(cvscore[1]))))
        print('No files will be generated. Re-run without cross validation')

    else:
        model = SGDRegressor(loss=cmdlloss,penalty=cmdlpenalty,l1_ratio=cmdll1ratio)
        # Fit model
        model.fit(X, y)
        # Get predictions
        ypred = model.predict(X)

        if cmdlminmaxscale:
            ymin,ymax = y.min(), y.max()
            mmscale = MinMaxScaler((ymin,ymax))
            # mmscale.fit(pred)
            pred1 = ypred.reshape(-1,1)
            ypred = mmscale.fit_transform(pred1).reshape(-1)


        # Calculating Mean Squared Error
        mse = np.mean((ypred - y)**2)
        print('Metrics on input data: ')
        print('MSE: %.4f' % (mse))
        r2 = r2_score(y,ypred)
        print('R2 : %10.3f' % r2)
        ccmdl = sts.pearsonr(y,ypred)
        print('Train-Test-Split Pearsonr : %10.3f  %10.4f ' % (ccmdl[0],ccmdl[1]))

        Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
            random_state=42)
        model.fit(Xtrain, ytrain)
        print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))
        qc0 = np.polyfit(y,ypred,1)
        xrngmin,xrngmax = y.min(),y.max()
        xvi = np.linspace(xrngmin,xrngmax)

        yvalpred = model.predict(Xval)
        # Calculating Mean Squared Error
        msev = np.mean((yvalpred - yval)**2)
        print('Metrics on Train-Test-Split data: ')
        print('Train-Test-Split MSE: %.4f' % (msev))
        r2v = r2_score(yval,yvalpred)
        print('Train-Test-Split R2 : %10.3f' % r2v)

        predcol = swa.columns[cmdlwtargetcol] + 'PRED'
        swa[predcol] = ypred

        fig = plt.figure()
        ax = fig.add_subplot(111)

        yvi0 = np.polyval(qc0,xvi)
        plt.scatter(y,ypred,alpha=0.5,c='b',label='Model Predicted')
        plt.plot(xvi,yvi0,c='k',lw=2)

        ax.annotate('Model = %-.*f * Actual + %-.*f' %
            (2,qc0[0],2,qc0[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.85),
            textcoords='figure fraction', fontsize=10)
        ax.annotate('Model Pearson cc = %-.*f   Pearson p = %-.*f' %
            (2,ccmdl[0],3,ccmdl[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.81),
            textcoords='figure fraction', fontsize=10)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('SGDR %s  Loss=%s  Penalty=%s l1ratio=%.1f' % (swa.columns[cmdlwtargetcol],cmdlloss,cmdlpenalty,cmdll1ratio))

        if not cmdlhideplot:
            plt.show()
        fig = ax.get_figure()
        fig.savefig(pdfcl)

        plt.boxplot([y,ypred],labels=['Actual','Model'],showmeans=True,notch=True)
        plt.title(f'SGDR for {swa.columns[cmdlwtargetcol]} L {cmdlloss} L1R {cmdll1ratio} P {cmdlpenalty}')
        if not cmdlhideplot:
            plt.show()
        fig = ax.get_figure()
        fig.savefig(boxpfname)

        if not cmdlgeneratesamples:
            # xpltcols = ['Actual','Predicted']
            xpltdf = swa.iloc[:,:3].copy()
            # copy well x y
            xpltdf['Actual'] = y
            xpltdf['Predicted'] = ypred
            xpltdf.to_csv(xyplt,index=False)
            print('Sucessfully generated xplot file %s' % xyplt)

        pickle.dump(model, open(modelfname, 'wb'))
        print(f'Sucessfully saved/pickled model {modelfname}')

        swa.to_csv(predfname,index=False)
        print(f'Sucessfully saved predicted file {predfname}')

def process_testCmodels(cmdlwellattribcsv,
    cmdlwcolsrange=None,
    cmdlwanalysiscols=None,
    cmdlwtargetcol=None,
    cmdlqcut=None,
    cmdlcv=None,
    cmdloutdir=None,
    cmdlhideplot=None):
    """test various classification models."""
    swa = pd.read_csv(cmdlwellattribcsv)
    if cmdlwcolsrange:
        print('Well From col# %d to col %d' %(cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1]]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1]]].columns
    else:
        print('Well analysis cols',cmdlwanalysiscols)
        X = swa[swa.columns[cmdlwanalysiscols]].values
        # colnames = swa[swa.columns[cmdlwanalysiscols]].columns

    probacolnames = ['Class%d' % i for i in range(cmdlqcut)]
    swa['qa'],qbins = pd.qcut(swa[swa.columns[cmdlwtargetcol]],cmdlqcut,labels=probacolnames,retbins=True)
    swa['qcodes'] = swa['qa'].cat.codes
    y = swa['qcodes'].values
    print('Quantile bins: ',qbins)
    qcount = Counter(y)
    print(qcount)

    models = []
    models.append((' LR ', LogisticRegression()))
    models.append((' LDA ', LinearDiscriminantAnalysis()))
    models.append((' KNN ', KNeighborsClassifier()))
    models.append((' CART ', DecisionTreeClassifier()))
    models.append((' NB ', GaussianNB()))
    models.append((' SVM ', SVC()))
    # evaluate each model in turn
    results = []
    names = []
    resultsmean = []
    resultsstd = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = KFold(n_splits=cmdlcv, random_state=7)
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "Model: %s: Mean Accuracy: %0.4f Std: (%0.4f)" % (name, cv_results.mean(), cv_results.std())
        # print (msg)
        resultsmean.append(cv_results.mean())
        resultsstd.append(cv_results.std())
    modeltest = pd.DataFrame(list(zip(names,resultsmean,resultsstd)),columns=['Model','Model Mean Accuracy','Accuracy STD'])
    print(modeltest)

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) + "_testcm.pdf"
    else:
        pdfcl = os.path.join(dirsplit,fname) + "_testcm.pdf"

    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle( ' Classification Algorithm Comparison ' )
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.savefig(pdfcl)
    if not cmdlhideplot:
        plt.show()

def process_CatBoostClassifier(cmdlwellattribcsv,
    cmdlwcolsrange=None,
    cmdlwpredictorcols=None,
    cmdlwtargetcol=None,
    cmdlcoded=None,
    cmdlminmaxscale=None,
    cmdloutdir=None,
    cmdliterations=None,
    cmdllearningrate=None,
    cmdldepth=None,cmdlqcut=None,
    cmdlimportancelevel=None,
    cmdlcv=None,
    cmdlfeatureimportance=False,
    cmdlgeneratesamples=None,
    cmdlbalancetype=None,
    cmdlnneighbors=None,
    cmdlvalsize=0.3,cmdlhideplot=False):
    """***************CatBoostClassifier."""
    cvalmin = 20
    # skip cross validation if data is less than cvalmin
    swa = pd.read_csv(cmdlwellattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) + f"_I{cmdliterations}_LR{cmdllearningrate:.3f}_D{cmdldepth}" + "_cbc.pdf"
        savemodelfname = os.path.join(cmdloutdir,fname) + f"_I{cmdliterations}_LR{cmdllearningrate:.3f}_D{cmdldepth}" + '_cbcmdl.json'
        predfname = os.path.join(cmdloutdir,fname) + f"_I{cmdliterations}_LR{cmdllearningrate:.3f}_D{cmdldepth}" + '_cbc.csv'
    else:
        pdfcl = os.path.join(dirsplit,fname) + f"_I{cmdliterations}_LR{cmdllearningrate:.3f}_D{cmdldepth}" + "_cbc.pdf"
        savemodelfname = os.path.join(dirsplit,fname) + f"_I{cmdliterations}_LR{cmdllearningrate:.3f}_D{cmdldepth}" + '_cbcmdl.json'
        predfname = os.path.join(dirsplit,fname) + f"_I{cmdliterations}_LR{cmdllearningrate:.3f}_D{cmdldepth}" + '_cbc.csv'


    if cmdlcoded:
        y = swa[swa.columns[cmdlwtargetcol]]
        # swxyz['qcodes'] = swa[swa.columns[cmdlwtargetcol]]
        qcount = len(Counter(y).items())
        probacolnames = ['Class%d' % i for i in range(qcount)]
        nclasses = qcount
        print('In coded:',probacolnames,nclasses)

    else:
        # use cmdlqcut
        probacolnames = ['Class%d' % i for i in range(cmdlqcut)]
        nclasses = cmdlqcut
        print('In qcut:',probacolnames,nclasses)
        swa['qa'],qbins = pd.qcut(swa[swa.columns[cmdlwtargetcol]],cmdlqcut,labels=probacolnames,retbins=True)
        swa['qcodes'] = swa['qa'].cat.codes
        y = swa['qcodes'].values
        print(qbins)
        print(swa.qa.value_counts())

    if cmdlgeneratesamples:
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='c',func='svc')

    resampled = False
    if cmdlbalancetype == 'ros':
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'smote':
        X_resampled, y_resampled = SMOTE(k_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'adasyn':
        X_resampled, y_resampled = ADASYN(n_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True

    if cmdlfeatureimportance:
        clf = CatBoostClassifier(iterations=cmdliterations, learning_rate=cmdllearningrate,
                    depth=cmdldepth,loss_function='MultiClass',calc_feature_importance=True,
                    random_seed=42,logging_level='Silent')
        clf.fit(X, y)
        fr = pd.DataFrame(sorted(zip(clf.get_feature_importance(X,y), colnames)),columns=['Importance','Attribute'])
        if cmdloutdir:
            fifname = os.path.join(cmdloutdir,fname) + f"_fi{cmdlimportancelevel:.1f}.csv"
            fislctfname = os.path.join(cmdloutdir,fname) + f"_fislct{cmdlimportancelevel:.1f}.csv"
        else:
            fifname = os.path.join(dirsplit,fname) + f"_fi{cmdlimportancelevel:.1f}.csv"
            fislctfname = os.path.join(dirsplit,fname) + f"_fislct{cmdlimportancelevel:.1f}.csv"

        # fr.to_csv(fifname,index=False)
        fr.to_csv(fifname)
        # intentionally save feature importance file with column #
        print(f'Successfully generated Feature Importance file {fifname}')
        frx = fr[fr['Importance'] > cmdlimportancelevel]
        frxcols = swa.columns[:5].tolist() + frx.Attribute.tolist() + [swa.columns[-1]]
        # Well X Y Z + selected attributes + target col
        print(f'# of features retained: {len(frx.Attribute.tolist())}')
        swafi = swa[frxcols].copy()
        # targetname = swa.columns[-1]
        # swafi[targetname] = y
        swafi.to_csv(fislctfname,index=False)
        print(f'Successfully generated selected features file {fislctfname}')

        # print('Feature Ranking with CatBoostClassifier: ')
        # print(fr)

    elif cmdlcv:
        clf = CatBoostClassifier(iterations=cmdliterations, learning_rate=cmdllearningrate,
                    depth=cmdldepth,loss_function='MultiClass',random_seed=42,logging_level='Silent')
        cvscore = cross_val_score(clf,X,y,cv=cmdlcv)
        print("Accuracy: %.3f%% (%.3f%%)" % (cvscore.mean() * 100.0, cvscore.std() * 100.0))
        # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        # print("Mean Score: %10.4f" %(np.mean(cvscore)))
        print('No files will be generated. Re-run without cross validation')

    else:
        clf = CatBoostClassifier(iterations=cmdliterations, learning_rate=cmdllearningrate,
                    depth=cmdldepth,loss_function='MultiClass',random_seed=42,logging_level='Silent')
        if resampled:
            clf.fit(X_resampled, y_resampled)
        else:
            clf.fit(X, y)
        # Get predictions
        if y.size >= cvalmin:
            Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
                random_state=42)
            clf.fit(Xtrain, ytrain)

            yvalpred = clf.predict(Xval)
            yvalproba = clf.predict_proba(Xval)
            print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))
            print('Validation Data Accuracy Score: %10.4f' % accuracy_score(yval,yvalpred))
            print('Validation Data Log Loss: %10.4f' % log_loss(yval,yvalproba))
            ydf = pd.DataFrame({'A':yval.ravel(),'P':yvalpred.ravel()})
            print(pd.crosstab(ydf['A'],ydf['P'],rownames=['Actuall'], colnames=['Predicted']))
            # print(confusion_matrix(yval.ravel(),yvalpred.ravel()))
            print(classification_report(yval.ravel(),yvalpred.ravel()))
            cnfmat = confusion_matrix(yval.ravel(),yvalpred.ravel())
            if cmdloutdir:
                pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_cbccvcnfmtx.pdf"
            else:
                pdfcnfmtx = os.path.join(dirsplit,fname) + "_cbccvcnfmtx.pdf"
            with PdfPages(pdfcnfmtx) as pdf:
                plot_confusion_matrix(cnfmat,probacolnames,title='Validation CatBoost Train Confusion Matrix',
                    hideplot=cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

                plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='Validation CatBoost Train Confusion Matrix Normalized',
                    hideplot=cmdlhideplot)

                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

        else:
            print('********No cross validation will be done due to low # of data points************')
        # if cmdlcoded:
        #     nclasses = qcount
        # else:
        #     nclasses = cmdlqcut
        # for i in range(cmdlqcut):
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_cbcroc.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_cbcroc.pdf"
        with PdfPages(pdfcl) as pdf:
            for i in range(nclasses):
                yw_prob = clf.predict_proba(X)[:,i]
                plot_roc_curve(y,yw_prob,i,cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

        yw = clf.predict(X)
        ywproba = clf.predict_proba(X)
        print('Full Data size: %5d' % len(yw))
        print('Full Data Accuracy Score: %10.4f' % accuracy_score(y,yw))
        print('Full Data Log Loss: %10.4f' % log_loss(y,ywproba))
        ywdf = pd.DataFrame({'A':y.ravel(),'P':yw.ravel()})
        print(pd.crosstab(ywdf['A'],ywdf['P'],rownames=['Actuall'], colnames=['Predicted']))
        print(classification_report(y.ravel(),yw.ravel()))
        cnfmat = confusion_matrix(y.ravel(),yw.ravel())
        if cmdloutdir:
            pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_cbccnfmtx.pdf"
        else:
            pdfcnfmtx = os.path.join(dirsplit,fname) + "_cbccnfmtx.pdf"
        with PdfPages(pdfcnfmtx) as pdf:
            plot_confusion_matrix(cnfmat,probacolnames,title='Full Data CatBoost Confusion Matrix',
                hideplot=cmdlhideplot)
            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

            plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='Full Data CatBoost Confusion Matrix Normalized',
                hideplot=cmdlhideplot)

            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

        if not cmdlgeneratesamples:
            swa['predqcodes'] = yw
            # print(f'ywproba len: {ywproba.shape}')
            for i,pclass in enumerate(probacolnames):
                swa[pclass] = ywproba[:,i]
            # swa['predproba'] = ywproba
            # pdfbar = os.path.join(dirsplit,fname) +"_gnbbar.pdf"

        else:
            print('******Will not generate well bar plots because of generate samples option')

        clf.save_model(savemodelfname,format="json",export_parameters=None)
        print(f'Successfully saved CatBoostRegressor Model {savemodelfname}')

        swa.to_csv(predfname,index=False)
        print(f'Sucessfully saved predicted file {predfname}')

def process_TuneCatBoostClassifier(cmdlwellattribcsv,cmdlseisattribcsv,
                cmdlwcolsrange=None,cmdlwpredictorcols=None,
                cmdlwtargetcol=None,cmdlsaxyzcols=None,
                cmdlscolsrange=None,cmdlspredictorcols=None,
                cmdlwellsxyzcols=None,cmdlcoded=None,
                cmdlminmaxscale=None,cmdloutdir=None,cmdliterations=None,
                cmdllearningrate=None,cmdldepth=None,cmdlqcut=None,cmdlcv=None,
                cmdlhideplot=False):
    """Tuning hyperparameters for CatBoost Classifier."""
    swa = pd.read_csv(cmdlwellattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        # colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    probacolnames = ['Class%d' % i for i in range(cmdlqcut)]

    if not cmdlcoded:
        swa['qa'],qbins = pd.qcut(swa[swa.columns[cmdlwtargetcol]],cmdlqcut,labels=probacolnames,retbins=True)
        swa['qcodes'] = swa['qa'].cat.codes
        y = swa['qcodes'].values
        print(qbins)

        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qa'] = swa['qa']
        swxyz['qcodes'] = swa['qcodes']

    else:
        y = swa[swa.columns[cmdlwtargetcol]]
        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qcodes'] = swa[swa.columns[cmdlwtargetcol]]
    qcount = Counter(y)
    print(qcount)

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    if cmdlscolsrange:
        print('Seismic Predictors From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1] + 1))
        Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
        # colnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].columns
    else:
        print('Seismic Predictor cols',cmdlspredictorcols)
        Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
        # colnames = swa[swa.columns[cmdlspredictorcols]].columns

    params = {'iterations': cmdliterations,
        'learning_rate': cmdllearningrate,
        'depth': cmdldepth}
    grdcv = GridSearchCV(CatBoostClassifier(loss_function='MultiClass'),params,cv=cmdlcv)

    # Fit model
    grdcv.fit(X, y)
    print(grdcv.best_params_)
    clf = grdcv.best_estimator_
    # Get predictions
    wpred = clf.predict(X)
    y_clf = clf.predict(Xpred, prediction_type='Class')
    # ypred = clf.predict(Xpred, prediction_type='RawFormulaVal')
    allprob = clf.predict_proba(Xpred)
    wproba = clf.predict_proba(X)
    print('All Data Accuracy Score: %10.4f' % accuracy_score(y,wpred))
    print('Log Loss: %10.4f' % log_loss(y,wproba))

    # [(ssa[probacolnames[i]] = allprob[:,i]) for i in range(cmd.qcut)]
    # add class column before probabilities
    ssa['TunedCatBoost'] = y_clf
    ssxyz['TunedCatBoost'] = y_clf
    for i in range(cmdlqcut):
        ssa[probacolnames[i]] = allprob[:,i]
        ssxyz[probacolnames[i]] = allprob[:,i]

    yw = clf.predict(X)

    ywproba = clf.predict_proba(X)
    print('Full Data size: %5d' % len(yw))
    print('Full Data Accuracy Score: %10.4f' % accuracy_score(y,yw))
    print('Full Data Log Loss: %10.4f' % log_loss(y,ywproba))
    ywdf = pd.DataFrame({'A':y.ravel(),'P':yw.ravel()})
    print(pd.crosstab(ywdf['A'],ywdf['P'],rownames=['Actuall'], colnames=['Predicted']))
    print(classification_report(y.ravel(),yw.ravel()))

    swxyz['predqcodes'] = yw
    swa['predqcodes'] = yw
    # pdfbar = os.path.join(dirsplit,fname) +"_gnbbar.pdf"

    if cmdlcoded:
        swxyz1 = swxyz[~swxyz['Well'].str.contains('PW')]
    else:
        swxyz1 = swxyz.copy()

    swxyz1.set_index('Well',inplace=True)

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    # plot 20 wells per bar graph
    for i in range(0,swxyz1.shape[0],20):
        swtemp = swxyz1.iloc[i:i + 20,:]
        ax = swtemp[['qcodes','predqcodes']].plot(kind='bar',rot=45,figsize=(15,10))
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_tcbccode%1d.pdf" % i
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_tcbccode%1d.pdf" % i
        if not cmdlhideplot:
            plt.show()
        fig = ax.get_figure()
        fig.savefig(pdfcl)

    savefiles(seisf=cmdlseisattribcsv,
        sdf=ssa, sxydf=ssxyz,
        wellf=cmdlwellattribcsv,
        wdf=swa, wxydf=swxyz,
        outdir=cmdloutdir,
        ssuffix='_stcbc',
        wsuffix='_wtcbc',name2merge=cmdlwellattribcsv)

def process_logisticreg(cmdlwellattribcsv,cmdlseisattribcsv,
        cmdlwcolsrange=None,cmdlwanalysiscols=None,
        cmdlwtargetcol=None,cmdlwellsxyzcols=None,
        cmdlsaxyzcols=None,cmdlscolsrange=None,
        cmdlspredictorcols=None,cmdlqcut=None,
        cmdlcoded=None,cmdlclassweight=False,
        cmdloutdir=None,cmdlcv=None,cmdlvalsize=0.3,
        cmdlgeneratesamples=None,
        cmdlbalancetype=None,
        cmdlnneighbors=None,
        cmdlhideplot=False):
    """Logistic Regression -> Classification."""
    cvalmin = 20
    # skip cross validation if data is less than cvalmin
    swa = pd.read_csv(cmdlwellattribcsv)
    if cmdlwcolsrange:
        print('Well From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well analysis cols',cmdlwanalysiscols)
        X = swa[swa.columns[cmdlwanalysiscols]].values
        # colnames = swa[swa.columns[cmdlwanalysiscols]].columns

    if cmdlcoded:
        y = swa[swa.columns[cmdlwtargetcol]]
        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qcodes'] = swa[swa.columns[cmdlwtargetcol]]
        qcount = len(Counter(y).items())
        print('qcount:',qcount)
        probacolnames = ['Class%d' % i for i in range(qcount)]
        nclasses = qcount
        print('In coded:',probacolnames,nclasses)
        # print(qbins)
    else:
        # use cmdlqcut
        probacolnames = ['Class%d' % i for i in range(cmdlqcut)]
        nclasses = cmdlqcut
        print('In qcut:',probacolnames,nclasses)
        swa['qa'],qbins = pd.qcut(swa[swa.columns[cmdlwtargetcol]],cmdlqcut,labels=probacolnames,retbins=True)
        swa['qcodes'] = swa['qa'].cat.codes
        y = swa['qcodes'].values
        print(qbins)

        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qa'] = swa['qa']
        swxyz['qcodes'] = swa['qcodes']

    if cmdlgeneratesamples:
        # Xoriginal = X.copy()
        # yoriginal = y.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='c',func='svc')

    resampled = False
    if cmdlbalancetype == 'ros':
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'smote':
        X_resampled, y_resampled = SMOTE(k_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'adasyn':
        X_resampled, y_resampled = ADASYN(n_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    if cmdlscolsrange:
        print('Seismic From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
        Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
        predcolnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].columns
    else:
        print('Seismic analysis cols',cmdlspredictorcols)
        Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
        # predcolnames = ssa[ssa.columns[cmdlspredictorcols]].columns

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    if cmdlcv:
        seed = 42
        kfold = KFold(n_splits=cmdlcv, random_state=seed)
        if cmdlclassweight:
            clf = LogisticRegression(class_weight='balanced')
            print('Class weight balanced')
        else:
            clf = LogisticRegression()
        results = cross_val_score(clf, X, y, cv=kfold)
        print("Logistic Regression Accuracy: %.3f%% (%.3f%%)"  % (results.mean() * 100.0, results.std() * 100.0))
        print('No files will be generated. Re-run without cv option')
    else:
        if cmdlclassweight:
            clf = LogisticRegression(class_weight='balanced')
            print('Class weight balanced')
        else:
            clf = LogisticRegression()
        if resampled:
            clf.fit(X_resampled, y_resampled)
        else:
            clf.fit(X, y)
        y_clf = clf.predict(Xpred)
        allprob = clf.predict_proba(Xpred)

        if y.size >= cvalmin:
            Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
                random_state=42)
            clf.fit(Xtrain, ytrain)

            yvalpred = clf.predict(Xval)
            yvalproba = clf.predict_proba(Xval)
            print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))
            print('Validation Data Accuracy Score: %10.4f' % accuracy_score(yval,yvalpred))
            print('Validation Data Log Loss: %10.4f' % log_loss(yval,yvalproba))
            ydf = pd.DataFrame({'A':yval.ravel(),'P':yvalpred.ravel()})
            print(pd.crosstab(ydf['A'],ydf['P'],rownames=['Actuall'], colnames=['Predicted']))
            print(classification_report(yval.ravel(),yvalpred.ravel()))

            cnfmat = confusion_matrix(yval.ravel(),yvalpred.ravel())
            if cmdloutdir:
                pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_lgrcvcnfmtx.pdf"
            else:
                pdfcnfmtx = os.path.join(dirsplit,fname) + "_lgrcvcnfmtx.pdf"
            with PdfPages(pdfcnfmtx) as pdf:
                plot_confusion_matrix(cnfmat,probacolnames,title='Logistic Reg Train Confusion Matrix',
                    hideplot=cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

                plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='Logistic Reg Train Confusion Matrix Normalized',
                    hideplot=cmdlhideplot)

                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()
        else:
            print('********No cross validation will be done due to low # of data points************')

        ssa['LRClass'] = y_clf
        ssxyz['LRClass'] = y_clf
        for i in range(nclasses):
            ssa[probacolnames[i]] = allprob[:,i]
            ssxyz[probacolnames[i]] = allprob[:,i]

        # for i in range(cmdlqcut):
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_lgrroc.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_lgrroc.pdf"
        with PdfPages(pdfcl) as pdf:
            for i in range(nclasses):
                yw_prob = clf.predict_proba(X)[:,i]
                plot_roc_curve(y,yw_prob,i,cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

        yw = clf.predict(X)
        ywproba = clf.predict_proba(X)
        print('Full Data size: %5d' % len(yw))
        print('Full Data Accuracy Score: %10.4f' % accuracy_score(y,yw))
        print('Full Data Log Loss: %10.4f' % log_loss(y,ywproba))
        ywdf = pd.DataFrame({'A':y.ravel(),'P':yw.ravel()})
        print(pd.crosstab(ywdf['A'],ywdf['P'],rownames=['Actuall'], colnames=['Predicted']))
        print(classification_report(y.ravel(),yw.ravel()))
        cnfmat = confusion_matrix(y.ravel(),yw.ravel())
        if cmdloutdir:
            pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_lgrcnfmtx.pdf"
        else:
            pdfcnfmtx = os.path.join(dirsplit,fname) + "_lgrcnfmtx.pdf"
        with PdfPages(pdfcnfmtx) as pdf:
            plot_confusion_matrix(cnfmat,probacolnames,title='Logistic Regression Confusion Matrix',
                hideplot=cmdlhideplot)
            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

            plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='Logistic Regression Confusion Matrix Normalized',
                hideplot=cmdlhideplot)

            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

        if not cmdlgeneratesamples:
            swxyz['predqcodes'] = yw
            swa['predqcodes'] = yw
            # pdfbar = os.path.join(dirsplit,fname) +"_lgrbar.pdf"

            if cmdlcoded:
                swxyz1 = swxyz[~swxyz['Well'].str.contains('PW')]
            else:
                swxyz1 = swxyz.copy()

            swxyz1.set_index('Well',inplace=True)
            # plot 20 wells per bar graph
            if cmdloutdir:
                pdfcl = os.path.join(cmdloutdir,fname) + "_lgrbar.pdf"
            else:
                pdfcl = os.path.join(dirsplit,fname) + "_lgrbar.pdf"
            with PdfPages(pdfcl) as pdf:
                for i in range(0,swxyz1.shape[0],20):
                    swtemp = swxyz1.iloc[i:i + 20,:]
                    # ax =swtemp[['qcodes','predqcodes']].plot(kind='bar',rot=45,figsize=(15,10))
                    swtemp[['qcodes','predqcodes']].plot(kind='bar',rot=45)
                    pdf.savefig()
                    if not cmdlhideplot:
                        plt.show()
                    plt.close()

            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                wellf=cmdlwellattribcsv,
                wdf=swa, wxydf=swxyz,
                outdir=cmdloutdir,
                ssuffix='_slgrg',
                wsuffix='_wlgrg',name2merge=cmdlwellattribcsv)
        else:
            print('******Will not generate well bar plots because of generate samples option')

            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                outdir=cmdloutdir,
                ssuffix='_slgrg',
                name2merge=cmdlwellattribcsv)

def process_GaussianNaiveBayes(cmdlwellattribcsv,cmdlseisattribcsv,
        cmdlwcolsrange=None,
        cmdlwanalysiscols=None,
        cmdlqcut=None,
        cmdlwellsxyzcols=None,
        cmdlsaxyzcols=None,
        cmdlscolsrange=None,
        cmdlspredictorcols=None,
        cmdlwtargetcol=None,
        cmdlcoded=None,
        cmdloutdir=None,
        cmdlcv=None,
        cmdlgeneratesamples=None,
        cmdlbalancetype=None,
        cmdlnneighbors=None,
        cmdlvalsize=0.3,
        cmdlhideplot=False):
    """Gaussian Naive Bayes Classification."""
    cvalmin = 20
    # skip cross validation if data is less than cvalmin
    swa = pd.read_csv(cmdlwellattribcsv)
    if cmdlwcolsrange:
        print('Well From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well analysis cols',cmdlwanalysiscols)
        X = swa[swa.columns[cmdlwanalysiscols]].values
        # colnames = swa[swa.columns[cmdlwanalysiscols]].columns

    if cmdlcoded:
        y = swa[swa.columns[cmdlwtargetcol]]
        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qcodes'] = swa[swa.columns[cmdlwtargetcol]]
        qcount = len(Counter(y).items())
        print('qcount:',qcount)
        probacolnames = ['Class%d' % i for i in range(qcount)]
        nclasses = qcount
        print('In coded:',probacolnames,nclasses)

    else:
        probacolnames = ['Class%d' % i for i in range(cmdlqcut)]
        nclasses = cmdlqcut
        print('In qcut:',probacolnames,nclasses)
        swa['qa'],qbins = pd.qcut(swa[swa.columns[cmdlwtargetcol]],cmdlqcut,labels=probacolnames,retbins=True)
        swa['qcodes'] = swa['qa'].cat.codes
        y = swa['qcodes'].values
        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qa'] = swa['qa']
        swxyz['qcodes'] = swa['qcodes']
        print(qbins)

    if cmdlgeneratesamples:
        # Xoriginal = X.copy()
        # yoriginal = y.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='c',func='svc')

    resampled = False
    if cmdlbalancetype == 'ros':
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'smote':
        X_resampled, y_resampled = SMOTE(k_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'adasyn':
        X_resampled, y_resampled = ADASYN(n_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    if cmdlscolsrange:
        print('Seismic From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
        Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
        # predcolnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].columns
    else:
        print('Seismic analysis cols',cmdlspredictorcols)
        Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
        # predcolnames = ssa[ssa.columns[cmdlspredictorcols]].columns

    if cmdlcv:
        seed = 42
        kfold = KFold(n_splits=cmdlcv, random_state=seed)
        clf = GaussianNB()
        results = cross_val_score(clf, X, y, cv=kfold)
        print("Gaussian Naive Bayes Accuracy: %.3f%% (%.3f%%)" % (results.mean() * 100.0, results.std() * 100.0))
        print('No files will be generated. Re-run without cv option')
    else:
        clf = GaussianNB()
        if resampled:
            clf.fit(X_resampled, y_resampled)
        else:
            clf.fit(X, y)
        y_clf = clf.predict(Xpred)
        allprob = clf.predict_proba(Xpred)

        if y.size >= cvalmin:
            Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
                random_state=42)
            clf.fit(Xtrain, ytrain)

            yvalpred = clf.predict(Xval)
            yvalproba = clf.predict_proba(Xval)
            print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))
            print('Validation Data Accuracy Score: %10.4f' % accuracy_score(yval,yvalpred))
            print('Validation Data Log Loss: %10.4f' % log_loss(yval,yvalproba))
            ydf = pd.DataFrame({'A':yval.ravel(),'P':yvalpred.ravel()})
            print(pd.crosstab(ydf['A'],ydf['P'],rownames=['Actuall'], colnames=['Predicted']))
            # print(confusion_matrix(yval.ravel(),yvalpred.ravel()))
            print(classification_report(yval.ravel(),yvalpred.ravel()))
            cnfmat =confusion_matrix(yval.ravel(),yvalpred.ravel())
            if cmdloutdir:
                pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_gnbcvcnfmtx.pdf"
            else:
                pdfcnfmtx = os.path.join(dirsplit,fname) + "_gnbcvcnfmtx.pdf"
            with PdfPages(pdfcnfmtx) as pdf:
                plot_confusion_matrix(cnfmat,probacolnames,title='Gaussian NB Train Confusion Matrix',
                    hideplot=cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

                plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='Gaussian NB Train Confusion Matrix Normalized',
                    hideplot=cmdlhideplot)

                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()
        else:
            print('********No cross validation will be done due to low # of data points************')

        # add class column before probabilities
        ssa['GNBClass'] = y_clf
        ssxyz['GNBClass'] = y_clf

        # [(ssa[probacolnames[i]] = allprob[:,i]) for i in range(cmd.qcut)]
        for i in range(cmdlqcut):
            ssa[probacolnames[i]] = allprob[:,i]
            ssxyz[probacolnames[i]] = allprob[:,i]

        # for i in range(cmdlqcut):
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_gnbroc.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_gnbroc.pdf"
        with PdfPages(pdfcl) as pdf:
            for i in range(nclasses):
                yw_prob = clf.predict_proba(X)[:,i]
                plot_roc_curve(y,yw_prob,i,cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

        yw = clf.predict(X)
        ywproba = clf.predict_proba(X)
        print('Full Data size: %5d' % len(yw))
        print('Full Data Accuracy Score: %10.4f' % accuracy_score(y,yw))
        print('Full Data Log Loss: %10.4f' % log_loss(y,ywproba))
        ywdf = pd.DataFrame({'A':y.ravel(),'P':yw.ravel()})
        print(pd.crosstab(ywdf['A'],ywdf['P'],rownames=['Actuall'], colnames=['Predicted']))
        print(classification_report(y.ravel(),yw.ravel()))
        cnfmat = confusion_matrix(y.ravel(),yw.ravel())
        if cmdloutdir:
            pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_gnbcnfmtx.pdf"
        else:
            pdfcnfmtx = os.path.join(dirsplit,fname) + "_gnbcnfmtx.pdf"
        with PdfPages(pdfcnfmtx) as pdf:
            plot_confusion_matrix(cnfmat,probacolnames,title='Gaussian NB Confusion Matrix',
                hideplot=cmdlhideplot)
            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

            plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='Gaussian NB Confusion Matrix Normalized',
                hideplot=cmdlhideplot)

            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

        if not cmdlgeneratesamples:
            swxyz['predqcodes'] = yw
            swa['predqcodes'] = yw

            if cmdlcoded:
                swxyz1 = swxyz[~swxyz['Well'].str.contains('PW')]
            else:
                swxyz1 = swxyz.copy()

            swxyz1.set_index('Well',inplace=True)
            # plot 20 wells per bar graph
            if cmdloutdir:
                pdfcl = os.path.join(cmdloutdir,fname) + "_gnbbar.pdf"
            else:
                pdfcl = os.path.join(dirsplit,fname) + "_gnbbar.pdf"
            with PdfPages(pdfcl) as pdf:
                for i in range(0,swxyz1.shape[0],20):
                    swtemp = swxyz1.iloc[i:i + 20,:]
                    swtemp[['qcodes','predqcodes']].plot(kind='bar',rot=45)
                    pdf.savefig()
                    if not cmdlhideplot:
                        plt.show()
                    plt.close()

            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                wellf=cmdlwellattribcsv,
                wdf=swa, wxydf=swxyz,
                outdir=cmdloutdir,
                ssuffix='_sgnb',
                wsuffix='_wgnb',name2merge=cmdlwellattribcsv)

        else:
            print('******Will not generate well bar plots because of generate samples option')

            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                outdir=cmdloutdir,
                ssuffix='_sgnb',
                name2merge=cmdlwellattribcsv)

def process_QuadraticDiscriminantAnalysis(cmdlwellattribcsv,cmdlseisattribcsv,
        cmdlwcolsrange=None,
        cmdlwanalysiscols=None,
        cmdlqcut=None,
        cmdlwellsxyzcols=None,
        cmdlsaxyzcols=None,
        cmdlscolsrange=None,
        cmdlspredictorcols=None,
        cmdlwtargetcol=None,
        cmdlcoded=None,
        cmdloutdir=None,
        cmdlcv=None,
        cmdlgeneratesamples=None,
        cmdlbalancetype=None,
        cmdlnneighbors=None,
        cmdlvalsize=0.3,
        cmdlhideplot=False):
    """Quadratic Discriminant Anlalysis Classification."""
    cvalmin = 20
    # skip cross validation if data is less than cvalmin
    swa = pd.read_csv(cmdlwellattribcsv)
    if cmdlwcolsrange:
        print('Well From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well analysis cols',cmdlwanalysiscols)
        X = swa[swa.columns[cmdlwanalysiscols]].values
        # colnames = swa[swa.columns[cmdlwanalysiscols]].columns

    if cmdlcoded:
        y = swa[swa.columns[cmdlwtargetcol]]
        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qcodes'] = swa[swa.columns[cmdlwtargetcol]]
        qcount = len(Counter(y).items())
        print('qcount:',qcount)
        probacolnames = ['Class%d' % i for i in range(qcount)]
        nclasses = qcount
        print('In coded:',probacolnames,nclasses)

    else:
        probacolnames = ['Class%d' % i for i in range(cmdlqcut)]
        nclasses = cmdlqcut
        print('In qcut:',probacolnames,nclasses)
        swa['qa'],qbins = pd.qcut(swa[swa.columns[cmdlwtargetcol]],cmdlqcut,labels=probacolnames,retbins=True)
        swa['qcodes'] = swa['qa'].cat.codes
        y = swa['qcodes'].values
        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qa'] = swa['qa']
        swxyz['qcodes'] = swa['qcodes']
        print(qbins)

    if cmdlgeneratesamples:
        # Xoriginal = X.copy()
        # yoriginal = y.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='c',func='svc')

    resampled = False
    if cmdlbalancetype == 'ros':
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'smote':
        X_resampled, y_resampled = SMOTE(k_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'adasyn':
        X_resampled, y_resampled = ADASYN(n_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    if cmdlscolsrange:
        print('Seismic From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
        # Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
        # predcolnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].columns
    else:
        print('Seismic analysis cols',cmdlspredictorcols)
        Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
        # predcolnames = ssa[ssa.columns[cmdlspredictorcols]].columns

    if cmdlcv:
        seed = 42
        kfold = KFold(n_splits=cmdlcv, random_state=seed)
        clf = QuadraticDiscriminantAnalysis()
        results = cross_val_score(clf, X, y, cv=kfold)
        print("Quadratic Discriminant Analysis Accuracy: %.3f%% (%.3f%%)" % (results.mean() * 100.0, results.std() * 100.0))
        print('No files will be generated. Re-run without cv option')
    else:
        clf = QuadraticDiscriminantAnalysis()
        if resampled:
            clf.fit(X_resampled, y_resampled)
        else:
            clf.fit(X, y)
        y_clf = clf.predict(Xpred)
        allprob = clf.predict_proba(Xpred)

        if y.size >= cvalmin:
            Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
                random_state=42)
            clf.fit(Xtrain, ytrain)

            yvalpred = clf.predict(Xval)
            yvalproba = clf.predict_proba(Xval)
            print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))
            print('Validation Data Accuracy Score: %10.4f' % accuracy_score(yval,yvalpred))
            print('Validation Data Log Loss: %10.4f' % log_loss(yval,yvalproba))
            ydf = pd.DataFrame({'A':yval.ravel(),'P':yvalpred.ravel()})
            print(pd.crosstab(ydf['A'],ydf['P'],rownames=['Actuall'], colnames=['Predicted']))
            # print(confusion_matrix(yval.ravel(),yvalpred.ravel()))
            print(classification_report(yval.ravel(),yvalpred.ravel()))

            cnfmat = confusion_matrix(yval.ravel(),yvalpred.ravel())
            if cmdloutdir:
                pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_qdacvcnfmtx.pdf"
            else:
                pdfcnfmtx = os.path.join(dirsplit,fname) + "_qdacvcnfmtx.pdf"
            with PdfPages(pdfcnfmtx) as pdf:
                plot_confusion_matrix(cnfmat,probacolnames,title='QDA Train Confusion Matrix',
                    hideplot=cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

                plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='QDA Train Confusion Matrix Normalized',
                    hideplot=cmdlhideplot)

                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

        else:
            print('********No cross validation will be done due to low # of data points************')

        # add class column before probabilities
        ssa['QDAClass'] = y_clf
        ssxyz['QDAClass'] = y_clf

        # [(ssa[probacolnames[i]] = allprob[:,i]) for i in range(cmd.qcut)]
        for i in range(cmdlqcut):
            ssa[probacolnames[i]] = allprob[:,i]
            ssxyz[probacolnames[i]] = allprob[:,i]

        # for i in range(cmdlqcut):
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_qdaroc.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_qdaroc.pdf"
        with PdfPages(pdfcl) as pdf:
            for i in range(nclasses):
                yw_prob = clf.predict_proba(X)[:,i]
                plot_roc_curve(y,yw_prob,i,cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

        yw = clf.predict(X)
        ywproba = clf.predict_proba(X)
        print('Full Data size: %5d' % len(yw))
        print('Full Data Accuracy Score: %10.4f' % accuracy_score(y,yw))
        print('Full Data Log Loss: %10.4f' % log_loss(y,ywproba))
        ywdf = pd.DataFrame({'A':y.ravel(),'P':yw.ravel()})
        print(pd.crosstab(ywdf['A'],ywdf['P'],rownames=['Actuall'], colnames=['Predicted']))
        print(classification_report(y.ravel(),yw.ravel()))

        cnfmat = confusion_matrix(y.ravel(),yw.ravel())
        if cmdloutdir:
            pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_qdacnfmtx.pdf"
        else:
            pdfcnfmtx = os.path.join(dirsplit,fname) + "_qdacnfmtx.pdf"
        with PdfPages(pdfcnfmtx) as pdf:
            plot_confusion_matrix(cnfmat,probacolnames,title='QDA Confusion Matrix',
                hideplot=cmdlhideplot)
            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

            plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='QDA Confusion Matrix Normalized',
                hideplot=cmdlhideplot)

            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

        if not cmdlgeneratesamples:
            swxyz['predqcodes'] = yw
            swa['predqcodes'] = yw
            # pdfbar = os.path.join(dirsplit,fname) +"_gnbbar.pdf"

            if cmdlcoded:
                swxyz1 = swxyz[~swxyz['Well'].str.contains('PW')]
            else:
                swxyz1 = swxyz.copy()

            swxyz1.set_index('Well',inplace=True)
            # plot 20 wells per bar graph
            if cmdloutdir:
                pdfcl = os.path.join(cmdloutdir,fname) + "_qdabar.pdf"
            else:
                pdfcl = os.path.join(dirsplit,fname) + "_qdabar.pdf"
            with PdfPages(pdfcl) as pdf:
                for i in range(0,swxyz1.shape[0],20):
                    swtemp = swxyz1.iloc[i:i + 20,:]
                    swtemp[['qcodes','predqcodes']].plot(kind='bar',rot=45)
                    # ax.set_ylim(-1.0,2.0)
                    pdf.savefig()
                    if not cmdlhideplot:
                        plt.show()
                    plt.close()

            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                wellf=cmdlwellattribcsv,
                wdf=swa, wxydf=swxyz,
                outdir=cmdloutdir,
                ssuffix='_sqda',
                wsuffix='_wqda',name2merge=cmdlwellattribcsv)

        else:
            print('******Will not generate well bar plots because of generate samples option')

            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                outdir=cmdloutdir,
                ssuffix='_sqda',
                name2merge=cmdlwellattribcsv)

def process_NuSVC(cmdlwellattribcsv,cmdlseisattribcsv,
        cmdlwcolsrange=None,
        cmdlwanalysiscols=None,
        cmdlqcut=None,
        cmdlwellsxyzcols=None,
        cmdlsaxyzcols=None,
        cmdlscolsrange=None,
        cmdlspredictorcols=None,
        cmdlwtargetcol=None,
        cmdlcoded=None,
        cmdloutdir=None,cmdlcv=None,
        cmdlvalsize=0.3,
        cmdlnu=None,
        cmdlbalancetype=None,
        cmdlnneighbors=None,
        cmdlgeneratesamples=None,
        cmdlhideplot=False):
    """Support Vector Machine Classification."""
    cvalmin = 20
    # skip cross validation if data is less than cvalmin
    swa = pd.read_csv(cmdlwellattribcsv)
    if cmdlwcolsrange:
        print('Well From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well analysis cols',cmdlwanalysiscols)
        X = swa[swa.columns[cmdlwanalysiscols]].values
        # colnames = swa[swa.columns[cmdlwanalysiscols]].columns

    if cmdlcoded:
        y = swa[swa.columns[cmdlwtargetcol]]
        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qcodes'] = swa[swa.columns[cmdlwtargetcol]]
        qcount = len(Counter(y).items())
        print('qcount:',qcount)
        probacolnames = ['Class%d' % i for i in range(qcount)]
        nclasses = qcount
        # print('In coded:',probacolnames,nclasses)
    else:
        probacolnames = ['Class%d' % i for i in range(cmdlqcut)]
        nclasses = cmdlqcut
        # print('In qcut:',probacolnames,nclasses)
        swa['qa'],qbins = pd.qcut(swa[swa.columns[cmdlwtargetcol]],cmdlqcut,labels=probacolnames,retbins=True)
        swa['qcodes'] = swa['qa'].cat.codes
        y = swa['qcodes'].values
        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qa'] = swa['qa']
        swxyz['qcodes'] = swa['qcodes']
        print(qbins)

    if cmdlgeneratesamples:
        # Xoriginal = X.copy()
        # yoriginal = y.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='c',func='svc')

    resampled = False
    if cmdlbalancetype == 'ros':
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'smote':
        X_resampled, y_resampled = SMOTE(k_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'adasyn':
        X_resampled, y_resampled = ADASYN(n_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    if cmdlscolsrange:
        print('Seismic From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
        Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
        # predcolnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].columns
    else:
        print('Seismic analysis cols',cmdlspredictorcols)
        Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
        # predcolnames = ssa[ssa.columns[cmdlspredictorcols]].columns

    if cmdlcv:
        seed = 42
        kfold = KFold(n_splits=cmdlcv, random_state=seed)
        clf = NuSVC(nu=cmdlnu,probability=True)
        results = cross_val_score(clf, X, y, cv=kfold)
        print("NuSVC Accuracy: %.3f%% (%.3f%%)" % (results.mean() * 100.0, results.std() * 100.0))
        print('No files will be generated. Re-run without cv option')
    else:
        clf = NuSVC(nu=cmdlnu,probability=True)
        if resampled:
            clf.fit(X_resampled, y_resampled)
        else:
            clf.fit(X, y)
        y_clf = clf.predict(Xpred)
        allprob = clf.predict_proba(Xpred)

        if y.size >= cvalmin:
            Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
                random_state=42)
            clf.fit(Xtrain, ytrain)

            yvalpred = clf.predict(Xval)
            yvalproba = clf.predict_proba(Xval)
            print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))
            print('Validation Data Accuracy Score: %10.4f' % accuracy_score(yval,yvalpred))
            print('Validation Data Log Loss: %10.4f' % log_loss(yval,yvalproba))
            ydf = pd.DataFrame({'A':yval.ravel(),'P':yvalpred.ravel()})
            print(pd.crosstab(ydf['A'],ydf['P'],rownames=['Actuall'], colnames=['Predicted']))
            # print(confusion_matrix(yval.ravel(),yvalpred.ravel()))
            print(classification_report(yval.ravel(),yvalpred.ravel()))

            cnfmat = confusion_matrix(yval.ravel(),yvalpred.ravel())
            if cmdloutdir:
                pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_nsvccvcnfmtx.pdf"
            else:
                pdfcnfmtx = os.path.join(dirsplit,fname) + "_nsvccvcnfmtx.pdf"
            with PdfPages(pdfcnfmtx) as pdf:
                plot_confusion_matrix(cnfmat,probacolnames,title='NuSVC Train Confusion Matrix',
                    hideplot=cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

                plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='NuSVC Train Confusion Matrix Normalized',
                    hideplot=cmdlhideplot)

                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

        else:
            print('********No cross validation will be done due to low # of data points************')

        # add class column before probabilities
        ssa['NuSVCClass'] = y_clf
        ssxyz['NuSVCClass'] = y_clf

        # [(ssa[probacolnames[i]] = allprob[:,i]) for i in range(cmd.qcut)]
        # for i in range(cmdlqcut):

        # ************need to adjust this for all other classifications
        for i in range(nclasses):
            ssa[probacolnames[i]] = allprob[:,i]
            ssxyz[probacolnames[i]] = allprob[:,i]

        # for i in range(cmdlqcut):
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_nusvcroc.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_nusvcroc.pdf"
        with PdfPages(pdfcl) as pdf:
            for i in range(nclasses):
                yw_prob = clf.predict_proba(X)[:,i]
                plot_roc_curve(y,yw_prob,i,cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

        yw = clf.predict(X)
        ywproba = clf.predict_proba(X)
        print('Full Data size: %5d' % len(yw))
        print('Full Data Accuracy Score: %10.4f' % accuracy_score(y,yw))
        print('Full Data Log Loss: %10.4f' % log_loss(y,ywproba))
        ywdf = pd.DataFrame({'A':y.ravel(),'P':yw.ravel()})
        print(pd.crosstab(ywdf['A'],ywdf['P'],rownames=['Actuall'], colnames=['Predicted']))
        print(classification_report(y.ravel(),yw.ravel()))

        cnfmat = confusion_matrix(y.ravel(),yw.ravel())
        if cmdloutdir:
            pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_nsvccnfmtx.pdf"
        else:
            pdfcnfmtx = os.path.join(dirsplit,fname) + "_nsvccnfmtx.pdf"
        with PdfPages(pdfcnfmtx) as pdf:
            plot_confusion_matrix(cnfmat,probacolnames,title='NuSVC Confusion Matrix',
                hideplot=cmdlhideplot)
            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

            plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='NuSVC Confusion Matrix Normalized',
                hideplot=cmdlhideplot)

            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

        if not cmdlgeneratesamples:
            swxyz['predqcodes'] = yw
            swa['predqcodes'] = yw
            # pdfbar = os.path.join(dirsplit,fname) +"_gnbbar.pdf"

            if cmdlcoded:
                swxyz1 = swxyz[~swxyz['Well'].str.contains('PW')]
            else:
                swxyz1 = swxyz.copy()

            swxyz1.set_index('Well',inplace=True)
            # plot 20 wells per bar graph
            if cmdloutdir:
                pdfcl = os.path.join(cmdloutdir,fname) + "_nusvcbar.pdf"
            else:
                pdfcl = os.path.join(dirsplit,fname) + "_nusvcbar.pdf"
            with PdfPages(pdfcl) as pdf:
                for i in range(0,swxyz1.shape[0],20):
                    swtemp = swxyz1.iloc[i:i + 20,:]
                    swtemp[['qcodes','predqcodes']].plot(kind='bar',rot=45)
                    # ax.set_ylim(-1.0,2.0)
                    pdf.savefig()
                    if not cmdlhideplot:
                        plt.show()
                    plt.close()

            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                wellf=cmdlwellattribcsv,
                wdf=swa, wxydf=swxyz,
                outdir=cmdloutdir,
                ssuffix='_snusvc',
                wsuffix='_wnusvc',name2merge=cmdlwellattribcsv)
        else:
            print('******Will not generate well bar plots because of generate samples option')

            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                outdir=cmdloutdir,
                ssuffix='_snusvc',
                name2merge=cmdlwellattribcsv)

def process_GaussianMixtureModel(cmdlwellattribcsv,cmdlseisattribcsv,
        cmdlbayesian=False,
        cmdlwcolsrange=None,
        cmdlwanalysiscols=None,
        cmdlwtargetcol=None,
        cmdlwellsxyzcols=None,
        cmdlsaxyzcols=None,
        cmdlscolsrange=None,
        cmdlcatcol=None,
        cmdlspredictorcols=None,
        cmdlncomponents=None,
        cmdloutdir=None,
        cmdlhideplot=False):
    """Gaussian Mixture Model Classification.

    This can be used as a clustering process by supplying only the welld data.
    The saved file will have the probabilities of the specified classes

    If seismic attributes are supplied the model generated from the well data
    will be used to predict the probabilities of the seismic.
    """
    swa = pd.read_csv(cmdlwellattribcsv)
    if cmdlcatcol:
        collabels = pd.get_dummies(swa.iloc[:,cmdlcatcol],drop_first=True)
        swa.drop(swa.columns[cmdlcatcol],axis=1,inplace=True)
        swa = pd.concat([swa,collabels],axis=1)
        cmdlwcolsrange[1] += collabels.shape[1]
    if cmdlwcolsrange:
        print('Well From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        if cmdlcatcol:
            # need to find a way to add to list of columns the list of dummy cols
            pass
        print('Well analysis cols',cmdlwanalysiscols)
        X = swa[swa.columns[cmdlwanalysiscols]].values
        # colnames = swa[swa.columns[cmdlwanalysiscols]].columns
    # if cmdlwtargetcol:
        # inclasses = swa[swa.columns[cmdlwtargetcol]].unique().tolist()
        # probacolnames = ['Class%d' % i for i in inclasses]

    probaclassnames = ['GMMClass%d' % i for i in range(cmdlncomponents)]
    swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    pltname = ''
    if cmdlbayesian:
        gmm = mixture.BayesianGaussianMixture(n_components=cmdlncomponents,
                covariance_type='spherical',
                max_iter=500,
                random_state=0).fit(X)
        pltname = 'bayes'
    else:
        gmm = mixture.GaussianMixture(n_components=cmdlncomponents,
                covariance_type='spherical',
                max_iter=500,
                random_state=0).fit(X)
    xpdf = np.linspace(-4, 3, 1000)
    _,ax = plt.subplots()
    for i in range(gmm.n_components):
        pdf = gmm.weights_[i] * sts.norm(gmm.means_[i, 0],np.sqrt(gmm.covariances_[i])).pdf(xpdf)
        ax.fill(xpdf, pdf, edgecolor='none', alpha=0.3,label='%s' % probaclassnames[i])
    ax.legend()
    if not cmdlhideplot:
        plt.show()
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) + "_gmm%d%s.pdf" % (cmdlncomponents,pltname)
    else:
        pdfcl = os.path.join(dirsplit,fname) + "_gmm%d%s.pdf" % (cmdlncomponents,pltname)
    fig = ax.get_figure()
    fig.savefig(pdfcl)

    # yw_gmm = gmm.predict(X)
    allwprob = gmm.predict_proba(X)
    for i in range(cmdlncomponents):
        swa[probaclassnames[i]] = allwprob[:,i]
        swxyz[probaclassnames[i]] = allwprob[:,i]

    if cmdlseisattribcsv:
        ssa = pd.read_csv(cmdlseisattribcsv)
        ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

        if cmdlscolsrange:
            print('Seismic From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
            Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
            # predcolnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].columns
        else:
            print('Seismic analysis cols',cmdlspredictorcols)
            Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
            # predcolnames = ssa[ssa.columns[cmdlspredictorcols]].columns

        ys_gmm = gmm.predict(Xpred)
        allsprob = gmm.predict_proba(Xpred)
        ssa['GMMclass'] = ys_gmm
        ssxyz['GMMclass'] = ys_gmm

        for i in range(cmdlncomponents):
            ssa[probaclassnames[i]] = np.around(allsprob[:,i],decimals=3)
            ssxyz[probaclassnames[i]] = np.around(allsprob[:,i],decimals=3)

        savefiles(seisf=cmdlseisattribcsv,
            sdf=ssa, sxydf=ssxyz,
            wellf=cmdlwellattribcsv,
            wdf=swa, wxydf=swxyz,
            outdir=cmdloutdir,
            ssuffix='_gmm%d' % cmdlncomponents,
            wsuffix='_gmm%d' % cmdlncomponents,name2merge=cmdlwellattribcsv)

    else:
        savefiles(seisf=cmdlwellattribcsv,
            sdf=swa, sxydf=swxyz,
            outdir=cmdloutdir,
            ssuffix='_gmm%d%s' % (cmdlncomponents,pltname))

def process_clustertest(cmdlallattribcsv,
        cmdlcolsrange=None,
        cmdlcols2cluster=None,
        cmdlsample=None,
        cmdloutdir=None,
        cmdlhideplot=False):
    """Test for optimum # of clusters for KMeans Clustering."""
    swa = pd.read_csv(cmdlallattribcsv)
    #print(swa.sample(5))

    if cmdlcolsrange:
        print('Well From col# %d to col %d' %(cmdlcolsrange[0],cmdlcolsrange[1]))
        swax = swa[swa.columns[cmdlcolsrange[0]: cmdlcolsrange[1] + 1]]
    else:
        swax = swa[swa.columns[cmdlcols2cluster]]
    swax = swax.sample(frac=cmdlsample).copy()


    dirsplit,fextsplit = os.path.split(cmdlallattribcsv)
    fname,fextn = os.path.splitext(fextsplit)
    if cmdloutdir:
        pdfcl = os.path.join(cmdl.outdir,fname) + "_cla.pdf"
    else:
        pdfcl = os.path.join(dirsplit,fname) + "_cla.pdf"
    inertia = list()
    delta_inertia = list()
    for k in range(1,21):
        clustering = KMeans(n_clusters=k, n_init=10,random_state= 1)
        clustering.fit(swax)
        if inertia:
            delta_inertia.append(inertia[-1] - clustering.inertia_)
        inertia.append(clustering.inertia_)
    with PdfPages(pdfcl) as pdf:
        plt.figure(figsize=(8,8))
        plt.plot([k for k in range(2,21)], delta_inertia,'ko-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Rate of Change of Intertia')
        plt.title('KMeans Cluster Analysis')
        pdf.savefig()
        if not cmdlhideplot:
            plt.show()
        plt.close()
        print('Successfully generated %s file'  % pdfcl)

def process_clustering(cmdlallattribcsv,cmdlcolsrange=None,
        cmdlcols2cluster=None,
        cmdlnclusters=None,
        cmdlplotsilhouette=False,
        cmdlsample=None,
        cmdlxyzcols=None,
        cmdladdclass=None,
        cmdloutdir=None,
        cmdlhideplot=False):
    """Cluster once csv."""
    swa = pd.read_csv(cmdlallattribcsv)
    if cmdlcolsrange:
        print('Well From col# %d to col %d' % (cmdlcolsrange[0],cmdlcolsrange[1]))
        swax = swa[swa.columns[cmdlcolsrange[0]: cmdlcolsrange[1] + 1]]
    else:
        swax = swa[swa.columns[cmdlcols2cluster]]

    clustering = KMeans(n_clusters=cmdlnclusters,
        n_init=5,
        max_iter=300,
        tol=1e-04,
        random_state=1)
    ylabels = clustering.fit_predict(swax)
    nlabels = np.unique(ylabels)
    print('nlabels',nlabels)

    if cmdladdclass == 'labels':
        swa['Cluster'] = ylabels
    elif cmdladdclass == 'dummies':
        classlabels = pd.Series(ylabels)
        classdummies = pd.get_dummies(classlabels,prefix='Cluster')
        swa = pd.concat([swa,classdummies],axis=1)
    print(swa.shape)

    swatxt = swa[swa.columns[cmdlxyzcols]].copy()
    if cmdladdclass == 'labels':
        swatxt['Class'] = ylabels
    elif cmdladdclass == 'dummies':
        classlabels = pd.Series(ylabels)
        classdummies = pd.get_dummies(classlabels,prefix='Cluster')
        swatxt = pd.concat([swatxt,classdummies],axis=1)

    if cmdladdclass == 'labels':
        savefiles(seisf=cmdlallattribcsv,
            sdf=swa, sxydf=swatxt,
            outdir=cmdloutdir,
            ssuffix='_cl')
    else:
        savefiles(seisf=cmdlallattribcsv,
            sdf=swa, sxydf=swatxt,
            outdir=cmdloutdir,
            ssuffix='_cld')

    '''
    Warning: Do not use sample to enable plot silhouette and add labels or dummies
    Better make a seperate run for silhouette plot on sampled data then use full data
    to add labels
    '''

    if cmdlplotsilhouette:
        dirsplit,fextsplit = os.path.split(cmdlallattribcsv)
        fname,fextn = os.path.splitext(fextsplit)
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_silcl.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_silcl.pdf"
        # only resample data if plotting silhouette
        swax = swax.sample(frac=cmdlsample).copy()
        ylabels = clustering.fit_predict(swax)
        n_clusters = ylabels.shape[0]
        silhouette_vals = silhouette_samples(swax, ylabels, metric='euclidean')
        y_ax_lower, y_ax_upper = 0, 0
        yticks = []
        for i, c in enumerate(nlabels):
            c_silhouette_vals = silhouette_vals[ylabels == c]
            c_silhouette_vals.sort()
            y_ax_upper += len(c_silhouette_vals)
            color = cm.jet(i / n_clusters)
            plt.barh(range(y_ax_lower, y_ax_upper),c_silhouette_vals,height=1.0,edgecolor='none',color=color)
            yticks.append((y_ax_lower + y_ax_upper) / 2)
            y_ax_lower += len(c_silhouette_vals)
            silhouette_avg = np.mean(silhouette_vals)
            plt.axvline(silhouette_avg,
                color="red",
                linestyle="--")
            plt.yticks(yticks, ylabels + 1)
            plt.ylabel('Cluster')
            plt.xlabel('Silhouette coefficient')
        plt.savefig(pdfcl)
        if not cmdlhideplot:
            plt.show()

def process_dbscan(cmdlallattribcsv,
    cmdlcolsrange=None,
                cmdlcols2cluster=None,
                cmdlsample=None,
                cmdlxyzcols=None,
                cmdlminsamples=None,
                cmdladdclass=None,
                cmdleps=None,
                cmdloutdir=None):
    """DBSCAN CLUSTERING."""
    swa = pd.read_csv(cmdlallattribcsv)
    if cmdlcolsrange:
        print('Well From col# %d to col %d' % (cmdlcolsrange[0],cmdlcolsrange[1]))
        swax = swa[swa.columns[cmdlcolsrange[0]: cmdlcolsrange[1] + 1]]
    else:
        swax = swa[swa.columns[cmdlcols2cluster]]

    dbscan = DBSCAN(eps=cmdleps, metric='euclidean', min_samples=cmdlminsamples)
    ylabels = dbscan.fit_predict(swax)
    print('Labels count per class:',list(Counter(ylabels).items()))

    # n_clusters = len(set(ylabels)) - (1 if -1 in ylabels else 0)
    n_clusters = len(set(ylabels))
    print('Estimated number of clusters: %d' % n_clusters)

    if cmdladdclass == 'labels':
        swa['Class'] = ylabels
    elif cmdladdclass == 'dummies':
        classlabels = pd.Series(ylabels)
        classdummies = pd.get_dummies(classlabels,prefix='Class')
        swa = pd.concat([swa,classdummies],axis=1)
    print(swa.shape)

    swatxt = swa[swa.columns[cmdlxyzcols]].copy()
    if cmdladdclass == 'labels':
        swatxt['Class'] = ylabels
    elif cmdladdclass == 'dummies':
        classlabels = pd.Series(ylabels)
        classdummies = pd.get_dummies(classlabels,prefix='Class')
        swatxt = pd.concat([swatxt,classdummies],axis=1)

    if cmdladdclass == 'labels':
        savefiles(seisf=cmdlallattribcsv,
            sdf=swa, sxydf=swatxt,
            outdir=cmdloutdir,
            ssuffix='_dbscn')
    else:
        savefiles(seisf=cmdlallattribcsv,
            sdf=swa, sxydf=swatxt,
            outdir=cmdloutdir,
            ssuffix='_dbscnd')

def process_tSNE(cmdlallattribcsv,
        cmdlcolsrange=None,
        cmdlcols2cluster=None,
        cmdlsample=None,
        cmdlxyzcols=None,
        cmdllearningrate=None,
        cmdlscalefeatures=True,
        cmdloutdir=None,
        cmdlhideplot=None):
    """Student t stochastic neighborhood embedding."""
    swa = pd.read_csv(cmdlallattribcsv)
    swaxx = swa.sample(frac=cmdlsample).copy()
    if cmdlcolsrange:
        print('Attrib From col# %d to col %d' % (cmdlcolsrange[0],cmdlcolsrange[1]))
        swax = swaxx[swaxx.columns[cmdlcolsrange[0]: cmdlcolsrange[1] + 1]]
    else:
        swax = swaxx[swaxx.columns[cmdlcols2cluster]]

    xyzc = swaxx[swaxx.columns[cmdlxyzcols]].copy()

    clustering = TSNE(n_components=2,
                learning_rate=cmdllearningrate)
    start_time = datetime.now()
    tsne_features = clustering.fit_transform(swax)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

    dirsplit,fextsplit = os.path.split(cmdlallattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    xs = tsne_features[:,0]
    ys = tsne_features[:,1]

    # colvals = [dt.hour for dt in datashuf[MIN:MAX].index]
    # for i in range(len(cmdlcolorby)):
    for i in range(swax.shape[1]):
        colvals = swax.iloc[:,i].values
        minima = min(colvals)
        maxima = max(colvals)
        matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        # mapper = cm.ScalarMappable(norm=norm, cmap=cm.Blues_r)
        # mapper = cm.ScalarMappable(norm=norm, cmap=cm.PiYG)
        # mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
        # mycolors = [mapper.to_rgba(v) for v in colvals]
        # clmp=cm.get_cmap('rainbow_r')
        clmp = cm.get_cmap('hsv')

        fig, ax = plt.subplots(figsize=(8,6))
        scatter = ax.scatter(xs,ys,
                 # c=mycolors,
                 # cmap=plt.cm.hsv,
                 c=colvals,
                 cmap=clmp,
                 s=10,
                 alpha=0.5)
        # cbar = plt.colorbar(scatter,mapper)
        plt.colorbar(scatter)
        plt.title('tSNE Colored by: %s' % swax.columns[i])
        if not cmdlhideplot:
            plt.show()
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "c%s" % swax.columns[i] + "_tsne.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "c%s" % swax.columns[i] + "_tsne.pdf"
        fig.savefig(pdfcl)
        # plt.savefig(pdfcl)

    tsnescaled = StandardScaler().fit_transform(tsne_features)
    if cmdlscalefeatures:
        swaxx['tSNE0s'] = tsnescaled[:,0]
        swaxx['tSNE1s'] = tsnescaled[:,1]
        xyzc['tSNE0s'] = tsnescaled[:,0]
        xyzc['tSNE1s'] = tsnescaled[:,1]
    else:
        swaxx['tSNE0'] = tsne_features[:,0]
        swaxx['tSNE1'] = tsne_features[:,1]
        xyzc['tSNE0'] = tsne_features[:,0]
        xyzc['tSNE1'] = tsne_features[:,1]

    savefiles(seisf=cmdlallattribcsv,
        sdf=swaxx, sxydf=xyzc,
        outdir=cmdloutdir,
        ssuffix='_tsne')

def process_tSNE2(cmdlwellattribcsv,
        cmdlseisattribcsv,
        cmdlwcolsrange=None,
        cmdlwpredictorcols=None,
        cmdlwtargetcol=None,
        cmdlsxyzcols=None,
        cmdlscolsrange=None,
        cmdlspredictorcols=None,
        cmdlsample=None,
        cmdlwxyzcols=None,
        cmdllearningrate=None,
        cmdlscalefeatures=True,
        cmdloutdir=None,
        cmdlhideplot=None):
    """Student t stochastic neighborhood embedding Using seismic and well csv's."""
    swa = pd.read_csv(cmdlwellattribcsv)
    if cmdlwcolsrange:
        print('Attrib From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        swax = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]]
    else:
        swax = swa[swa.columns[cmdlwpredictorcols]]

    wxyzc = swax[swax.columns[cmdlwxyzcols]].copy()
    y = swa[swa.columns[cmdlwtargetcol]].copy()
    targetcname = swa.columns[cmdlwtargetcol]
    print('Target col: %s %d' % (targetcname,y.shape[0]))
    print(swa.columns)
    swa.drop(swa.columns[cmdlwtargetcol],axis=1,inplace=True)

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssa = ssa.sample(frac=cmdlsample)
    ssxyz = ssa[ssa.columns[cmdlsxyzcols]].copy()

    if cmdlscolsrange:
        print('Seismic Predictors From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
        sspred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]]
        # colnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1]+1]].columns
    else:
        print('Seismic Predictor cols',cmdlspredictorcols)
        sspred = ssa[ssa.columns[cmdlspredictorcols]]
        # colnames = swa[swa.columns[cmdlspredictorcols]].columns

    alldata = pd.concat([swax,sspred])
    wrows = swax.shape[0]

    clustering = TSNE(n_components=2,
                learning_rate=cmdllearningrate)
    start_time = datetime.now()
    tsne_features = clustering.fit_transform(alldata)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    tsnescaled = StandardScaler().fit_transform(tsne_features)
    if cmdlscalefeatures:
        wxs = tsnescaled[:wrows,0]
        wys = tsnescaled[:wrows,1]
        ssxs = tsnescaled[wrows:,0]
        ssys = tsnescaled[wrows:,1]
    else:
        wxs = tsne_features[:wrows,0]
        wys = tsne_features[:wrows,1]
        ssxs = tsne_features[wrows:,0]
        ssys = tsne_features[wrows:,1]

    swa['tSNE0s'] = wxs
    swa['tSNE1s'] = wys
    swa[targetcname] = y
    wxyzc['tSNE0s'] = wxs
    wxyzc['tSNE1s'] = wys
    wxyzc[targetcname] = y

    ssa['tSNE0s'] = ssxs
    ssa['tSNE1s'] = ssys
    ssxyz['tSNE0s'] = ssxs
    ssxyz['tSNE1s'] = ssys

    # colvals = [dt.hour for dt in datashuf[MIN:MAX].index]
    # for i in range(len(cmdlcolorby)):
    for i in range(alldata.shape[1]):
        colvals = alldata.iloc[:,i].values
        minima = min(colvals)
        maxima = max(colvals)
        matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        # mapper = cm.ScalarMappable(norm=norm, cmap=cm.Blues_r)
        # mapper = cm.ScalarMappable(norm=norm, cmap=cm.PiYG)
        # mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
        # mycolors = [mapper.to_rgba(v) for v in colvals]
        # clmp=cm.get_cmap('rainbow_r')
        clmp = cm.get_cmap('hsv')

        fig, ax = plt.subplots(figsize=(8,6))
        scatter = ax.scatter(tsne_features[:,0],tsne_features[:,1],
                 # c=mycolors,
                 # cmap=plt.cm.hsv,
                 c=colvals,
                 cmap=clmp,
                 s=10,
                 alpha=0.5)
        # cbar = plt.colorbar(scatter,mapper)
        plt.colorbar(scatter)
        plt.title('tSNE Colored by: %s' % swax.columns[i])
        if not cmdlhideplot:
            plt.show()
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "c%s" % swax.columns[i] + "_tsne.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "c%s" % swax.columns[i] + "_tsne.pdf"
        fig.savefig(pdfcl)
        # plt.savefig(pdfcl)

    savefiles(seisf=cmdlseisattribcsv,
        sdf=ssa, sxydf=ssxyz,
        wellf=cmdlwellattribcsv,
        wdf=swa, wxydf=wxyzc,
        outdir=cmdloutdir,
        ssuffix='_tsne',
        name2merge=cmdlwellattribcsv)

def process_umap(cmdlallattribcsv,
                cmdlcolsrange=None,
                cmdlcols2cluster=None,
                cmdlsample=None,
                cmdlxyzcols=None,
                cmdlnneighbors=None,
                cmdlmindistance=0.3,
                cmdlncomponents=3,
                cmdlscalefeatures=False,
                cmdloutdir=None,
                cmdlhideplot=None):
    """Uniform Manifold Approximation Projection Clustering."""
    swa = pd.read_csv(cmdlallattribcsv)
    swaxx = swa.sample(frac=cmdlsample).copy()
    print('# of components {:2d}'.format(cmdlncomponents))
    if cmdlcolsrange:
        print('Attrib From col# %d to col %d' % (cmdlcolsrange[0],cmdlcolsrange[1]))
        swax = swaxx[swaxx.columns[cmdlcolsrange[0]: cmdlcolsrange[1] + 1]]
    else:
        swax = swaxx[swaxx.columns[cmdlcols2cluster]]

    xyzc = swaxx[swaxx.columns[cmdlxyzcols]].copy()

    clustering = umap.UMAP(n_neighbors=cmdlnneighbors, min_dist=cmdlmindistance, n_components=cmdlncomponents)

    start_time = datetime.now()
    umap_features = clustering.fit_transform(swax)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

    dirsplit,fextsplit = os.path.split(cmdlallattribcsv)
    fname,fextn = os.path.splitext(fextsplit)
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) + "_umapnc%d.pdf" % (cmdlncomponents)
    else:
        pdfcl = os.path.join(dirsplit,fname) + "_umapnc%d.pdf" % (cmdlncomponents)
    fig, ax = plt.subplots(figsize=(8,6))

    nclst = [i for i in range(cmdlncomponents)]
    pltvar = itertools.combinations(nclst,2)
    pltvarlst = list(pltvar)
    for i in range(len(pltvarlst)):
        ftr0 = pltvarlst[i][0]
        ftr1 = pltvarlst[i][1]
        print('umap feature #: {}, umap feature #: {}'.format(ftr0,ftr1))
        # ax.scatter(umap_features[:,pltvarlst[i][0]],umap_features[:,pltvarlst[i][1]],s=2,alpha=.2)
        ax.scatter(umap_features[:,ftr0],umap_features[:,ftr1],s=2,alpha=.2)

        # ax.scatter(umap_features[:,0],umap_features[:,1],s=2,alpha=.2)
        # ax.scatter(umap_features[:,1],umap_features[:,2],s=2,alpha=.2)
        # ax.scatter(umap_features[:,2],umap_features[:,0],s=2,alpha=.2)

    if not cmdlhideplot:
        plt.show()
    fig.savefig(pdfcl)

    if cmdlscalefeatures:
        umapscaled = StandardScaler().fit_transform(umap_features)
        for i in range(cmdlncomponents):
            cname = 'umap' + str(i) + 's'
            swaxx[cname] = umapscaled[:,i]
            xyzc[cname] = umapscaled[:,i]

            # swaxx['umap0s'] = umapscaled[:,0]
            # swaxx['umap1s'] = umapscaled[:,1]
            # swaxx['umap2s'] = umapscaled[:,2]
            # xyzc['umap0s'] =   umapscaled[:,0]
            # xyzc['umap1s'] =   umapscaled[:,1]
            # xyzc['umap2s'] =   umapscaled[:,2]
    else:
        for i in range(cmdlncomponents):
            cname = 'umap' + str(i)
            swaxx[cname] = umap_features[:,i]
            xyzc[cname] = umap_features[:,i]

            # swaxx['umap0'] = umap_features[:,0]
            # swaxx['umap1'] = umap_features[:,1]
            # swaxx['umap2'] = umap_features[:,2]
            # xyzc['umap0'] =  umap_features[:,0]
            # xyzc['umap1'] =  umap_features[:,1]
            # xyzc['umap2'] =  umap_features[:,2]

    savefiles(seisf=cmdlallattribcsv,
        sdf=swaxx, sxydf=xyzc,
        outdir=cmdloutdir,
        ssuffix='_umapnc%d' % (cmdlncomponents))

def process_semisupervised(wfname,sfname,
        cmdlwcolsrange=None,
        cmdlwtargetcol=None,
        cmdlwellsxyzcols=None,
        cmdlsample=0.005,
        cmdlkernel='knn',
        cmdlnneighbors=7,
        cmdlcol2drop=None,
        cmdloutdir=None):
    """Semi supervised: creating extra data from existing data. Regression."""
    i4w = pd.read_csv(wfname)
    if cmdlcol2drop:
        i4w.drop(i4w.columns[cmdlcol2drop],axis=1,inplace=True)
    dirsplitw,fextsplit = os.path.split(wfname)
    fnamew,fextn = os.path.splitext(fextsplit)
    if cmdloutdir:
        ppsave = os.path.join(cmdloutdir,fnamew) + "_pw.csv"
    else:
        ppsave = os.path.join(dirsplitw,fnamew) + "_pw.csv"
    # print('target col',cmdlwtargetcol)
    # print(i4w[i4w.columns[cmdlwtargetcol]],i4w.columns[cmdlwtargetcol])
    # coln = i4w.columns[cmdlwtargetcol]
    # print('coln:',coln)
    if cmdlcol2drop:
        cmdlwtargetcol -= 1
    i4w['qa'],qbins = pd.qcut(i4w[i4w.columns[cmdlwtargetcol]],3,labels=['Low','Medium','High'],retbins=True)

    i4w['qcodes'] = i4w['qa'].cat.codes
    print('codes: ',i4w['qcodes'].unique())

    i4s = pd.read_csv(sfname)

    # i4w.drop(['Av_PHIT', 'qa'],axis=1,inplace=True)
    i4w.drop(i4w.columns[[cmdlwtargetcol,cmdlwtargetcol + 1]],axis=1,inplace=True)
    i4sx = i4s.sample(frac=cmdlsample,random_state=42)
    i4sxi = i4sx.reset_index()
    i4sxi.drop('index',axis=1,inplace=True)
    i4sxi['Well1'] = ['PW%d' % i for i in i4sxi.index]
    i4sxi.insert(0,'Well',i4sxi['Well1'])
    i4sxi.drop('Well1',axis=1,inplace=True)
    i4sxi['qcodes'] = [(-1) for i in i4sxi.index]
    wcols = list(i4w.columns)
    i4sxi.columns = wcols
    i4 = pd.concat([i4w,i4sxi],axis=0)
    X = i4[i4.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
    y = i4[i4.columns[cmdlwtargetcol]].values
    print(Counter(list(y)).items())
    lblsprd = LabelSpreading(kernel=cmdlkernel,n_neighbors=cmdlnneighbors)
    lblsprd.fit(X,y)
    ynew = lblsprd.predict(X)
    print(Counter(list(ynew)).items())
    i4['qcodespred'] = ynew
    i4.drop(i4.columns[cmdlwtargetcol],axis=1,inplace=True)
    i4.to_csv(ppsave,index=False)
    print('Successfully generated %s file' % ppsave)
    i4xy = i4.copy()
    i4xy.drop(i4xy.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1],axis=1,inplace=True)
    if cmdloutdir:
        ppxysave = os.path.join(cmdloutdir,fnamew) + "_pw.txt"
    else:
        ppxysave = os.path.join(dirsplitw,fnamew) + "_pw.txt"
    i4xy.to_csv(ppxysave,sep=' ',index=False)
    print('Successfully generated %s file' % ppxysave)

def cmnt(line):
    """Check if a line is comment."""
    if '#' in line:
        return True
    else:
        return False

class ClassificationMetrics:
    """Compute Classification Metrics."""

    def __init__(self,actual,predicted,tolist=True,tocsv=False):
        """Initializer for class."""
        self.actual = actual
        self.predicted = predicted
        self.tolist = tolist
        self.tocsv = tocsv

    def comp_confusion(self):
        """Compute confusion matrix."""
        if self.tolist:
            print('Confusion Report: ')
            print(pd.crosstab(self.actual,self.predicted,rownames=['Actual'], colnames =['Predicted']))

    def comp_accuracy(self):
        """Compute accuracy."""
        if self.tolist:
            print('Accuracy Score: ',accuracy_score(self.actual,self.predicted))

    def comp_clfreport(self):
        """Generate report."""
        if self.tolist:
            print('Classification Report: ')
            print(classification_report(self.actual,self.predicted))

def getcommandline(*oneline):
    """Process command line interface."""
    allcommands = ['workflow','dropcols','listcsvcols','PCAanalysis','PCAfilter','scattermatrix',
                'EDA','featureranking','linreg','KNNtest','KNN','CatBoostRegressor',
                'CatBoostClassifier','testCmodels','logisticreg','GaussianNaiveBayes','clustertest','clustering',
                'tSNE','tSNE2','TuneCatBoostClassifier','DBSCAN','wscaletarget','semisupervised',
                'GaussianMixtureModel','ANNRegressor','NuSVR','NuSVC','SGDR','QuadraticDiscriminantAnalysis','umap']

    mainparser = argparse.ArgumentParser(description='Seismic and Well Attributes Modeling.')
    mainparser.set_defaults(which=None)
    subparser = mainparser.add_subparsers(help='File name listing all attribute grids')

    wrkflowparser = subparser.add_parser('workflow',help='Workflow file instead of manual steps')
    wrkflowparser.set_defaults(which='workflow')
    wrkflowparser.add_argument('commandfile',help='File listing workflow')
    wrkflowparser.add_argument('--startline',type=int,default =0,help='Line in file to start flow from. default=0')
    # wrkflowparser.add_argument('--stopat',type=int,default =None,help='Line in file to end flow after. default=none')


    dropparser = subparser.add_parser('dropcols',help='csv drop columns')
    dropparser.set_defaults(which='dropcols')
    dropparser.add_argument('csvfile',help='csv file to drop columns')
    dropparser.add_argument('--cols2drop',type=int,nargs='*',default=None,help='default=none')
    dropparser.add_argument('--outdir',help='output directory,default= same dir as input')

    listcolparser = subparser.add_parser('listcsvcols',help='List header row of any csv')
    listcolparser.set_defaults(which='listcsvcols')
    listcolparser.add_argument('csvfile',help='csv file name')

    pcaaparser = subparser.add_parser('PCAanalysis',help='PCA analysis')
    pcaaparser.set_defaults(which='PCAanalysis')
    pcaaparser.add_argument('allattribcsv',help='csv file with all previously scaled predictors with target column')
    pcaaparser.add_argument('--analysiscols',type=int,nargs='+',help='Predictor column #s, no default')
    pcaaparser.add_argument('--acolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    pcaaparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    pcafparser = subparser.add_parser('PCAfilter',help='PCA filter')
    pcafparser.set_defaults(which='PCAfilter')
    pcafparser.add_argument('allattribcsv',help='csv file with all previously scaled predictors with target column')
    pcafparser.add_argument('--analysiscols',type=int,nargs='+',help='Predictor column #s, no default')
    pcafparser.add_argument('--acolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    pcafparser.add_argument('--ncomponents',type=int,default=None,help='# of components to keep,default =none')
    pcafparser.add_argument('--targetcol',type=int,default=None,
        help='Target column # to add back. You do not have to add a target default = none')
    pcafparser.add_argument('--cols2addback',type=int,nargs='+',default=[0,1,2],
        help='Columns, e.g. well x y z  to remove from fitting model, but addback to saved file. default= 0 1 2 3 ')
    pcafparser.add_argument('--outdir',help='output directory,default= same dir as input')

    sctrmparser = subparser.add_parser('scattermatrix',help='Scatter matrix of all predictors and target')
    sctrmparser.set_defaults(which='scattermatrix')
    sctrmparser.add_argument('allattribcsv',help='csv file with all predictors and target column')
    sctrmparser.add_argument('--wellxyzcols',type=int,nargs='+',default=[0,1,2,3],
        help='Columns well x y z  to remove from fitting model. default= 0 1 2 3 ')
    sctrmparser.add_argument('--sample',type=float,default=.5,help='fraction of data of sample.default=0.5')

    # *************EDA
    edaparser = subparser.add_parser('EDA',help='Exploratory Data Analysis')
    edaparser.set_defaults(which='EDA')
    edaparser.add_argument('allattribcsv',help='csv file with all predictors and target column')
    edaparser.add_argument('--xyzcols',type=int,nargs='+',help='Any # of columns to remove before analysis,e.g. x y z')
    edaparser.add_argument('--polydeg',type=int, default=1,help='degree of polynomial to fit data in xplots choice. default = 1, i.e. st line')
    edaparser.add_argument('--sample',type=float,default=.5,help='fraction of data of sample for ScatterMatrix Plot.default=0.5')
    edaparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')
    edaparser.add_argument('--plotoption',
        choices=['xplots','heatmap','box','distribution','scattermatrix'],help='choices: xplots,heatmap,box,distribution,scattermatrix',default='heatmap')
    edaparser.add_argument('--outdir',help='output directory,default= same dir as input')


    frparser = subparser.add_parser('featureranking',help='Ranking of attributes')
    frparser.set_defaults(which='featureranking')
    frparser.add_argument('allattribcsv',help='csv file with all predictors and target column')
    frparser.add_argument('--wtargetcol',type=int, default=-1,help='Target column #.default = last col')
    frparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    frparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    frparser.add_argument('--testfeatures',choices=['rfe','rlasso','svr','svrcv','rfregressor','decisiontree'],default='rfregressor',
        help='Test for features significance: Randomized Lasso, recursive feature elimination #,default= rfregressor')
    # lassalpha is used with randomized lasso only
    frparser.add_argument('--lassoalpha',type=float,default=0.025,help='alpha = 0 is OLS. default=0.005')
    # features2keep is used with svr only
    frparser.add_argument('--features2keep',type=int,default=5,help='#of features to keep in rfe.default=5')
    # following 2 are used with any cross validation e.g. random forest regressor, svrcv
    frparser.add_argument('--cv',type=int,default=3,help='Cross Validation nfold. default=3')
    frparser.add_argument('--traintestsplit',type=float,default=.3,help='Train Test split. default = 0.3')

    # *************linreg linear regression
    lfpparser = subparser.add_parser('linreg',help='Linear Regression fit and predict on one data set  ')
    lfpparser.set_defaults(which='linreg')
    lfpparser.add_argument('allattribcsv',help='csv file of all attributes at well locations to fit model')
    lfpparser.add_argument('--wtargetcol',type=int, default=-1,help='Target column #.default = last col')
    lfpparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    lfpparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    lfpparser.add_argument('--minmaxscale',action='store_true',default=False,
        help='Apply min max scaler to scale predicted to input range. default= False i.e. no scaling')
    lfpparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    lfpparser.add_argument('--outdir',help='output directory,default= same dir as input')

    knntparser = subparser.add_parser('KNNtest',help='Test number of nearest neighbors for KNN')
    knntparser.set_defaults(which='KNNtest')
    knntparser.add_argument('allattribcsv',help='csv file with all previously scaled predictors with target column')
    knntparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column #.default = last col')
    knntparser.add_argument('--predictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    knntparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='KNN test min max col #')
    # knntparser.add_argument('--traintestsplit',type=float,default=.2,
    #     help='train test split ratio, default= 0.2, i.e. keep 20%% for test')
    knntparser.add_argument('--cv',type=int,default=3,help='Cross Validation nfold. default=3')
    knntparser.add_argument('--sample',type=float,default=1.0,help='fraction of data of sample.default=1, i.e. all data')
    knntparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')
    knntparser.add_argument('--outdir',help='output directory,default= same dir as input')

    knnfparser = subparser.add_parser('KNN',help='KNN fit on one data set and predicting on another ')
    knnfparser.set_defaults(which='KNN')
    knnfparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    knnfparser.add_argument('--wtargetcol',type=int, default=-1,help='Target column #.default = last col')
    knnfparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    knnfparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    knnfparser.add_argument('--minmaxscale',action='store_true',default=False,
        help='Apply min max scaler to scale predicted to input range. default= False i.e. no scaling')
    knnfparser.add_argument('--kneighbors',type=int,default=10,help='# of nearest neighbors. default = 10')
    knnfparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    knnfparser.add_argument('--outdir',help='output directory,default= same dir as input')
    knnfparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')


    # *************SVR support vector regresssion: uses nusvr
    nsvrparser = subparser.add_parser('NuSVR',help='Nu Support Vector Machine Regressor')
    nsvrparser.set_defaults(which='NuSVR')
    nsvrparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    nsvrparser.add_argument('--wtargetcol',type=int, default=-1,help='Target column #.default = last col')
    nsvrparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    nsvrparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    nsvrparser.add_argument('--minmaxscale',action='store_true',default=False,
        help='Apply min max scaler to scale predicted to input range. default= False i.e. no scaling')
    nsvrparser.add_argument('--scaleminmaxvalues',type=float,nargs=2,
                            help='Min Max scale limits. default=use input data limits ')

    nsvrparser.add_argument('--outdir',help='output directory,default= same dir as input')
    nsvrparser.add_argument('--nu',type=float,default=0.5,help='upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. value between 0 1, default =0.5')
    nsvrparser.add_argument('--errpenalty',type=float,default=1.0,help='error penalty. default=1.0')
    nsvrparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    nsvrparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    nsvrparser.add_argument('--cv',type=int,default=None,help='Cross Validate. default=None.')
    nsvrparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    # *************SGD Regressor
    sgdrparser = subparser.add_parser('SGDR',help='Stochastic Gradient Descent Regressor: OLS/Lasso/Ridge/ElasticNet')
    sgdrparser.set_defaults(which='SGDR')
    sgdrparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    sgdrparser.add_argument('--wtargetcol',type=int, default=-1,help='Target column #.default = last col')
    sgdrparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    sgdrparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    sgdrparser.add_argument('--minmaxscale',action='store_true',default=False,
        help='Apply min max scaler to scale predicted to input range. default= False i.e. no scaling')
    sgdrparser.add_argument('--scaleminmaxvalues',type=float,nargs=2,
        help='Min Max scale limits. default=use input data limits ')
    sgdrparser.add_argument('--outdir',help='output directory,default= same dir as input')
    sgdrparser.add_argument('--loss',choices=['squared_loss','huber','epsilon_insensitive','squared_epsilon_insensitive'],default='squared_loss',
        help='default= squared_loss')
    sgdrparser.add_argument('--penalty',choices=['l1','l2','elasticnet','none'],default='l2',help='default=l2')
    sgdrparser.add_argument('--l1ratio',type=float,default=0.15,help='elastic net mixing: 0 (l2)to 1 (l1), default =0.15')
    sgdrparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    sgdrparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    sgdrparser.add_argument('--cv',type=int,default=None,help='Cross Validate. default=None.')
    sgdrparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    # *************CatBoostRegressor
    cbrparser = subparser.add_parser('CatBoostRegressor',help='CatBoost Regressor')
    cbrparser.set_defaults(which='CatBoostRegressor')
    cbrparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    cbrparser.add_argument('--wtargetcol',type=int, default=-1,help='Target column #.default = last col')
    cbrparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    cbrparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    cbrparser.add_argument('--minmaxscale',action='store_true',default=False,
        help='Apply min max scaler to scale predicted to input range. default= False i.e. no scaling')
    cbrparser.add_argument('--scaleminmaxvalues',type=float,nargs=2,
        help='Min Max scale limits. default=use input data limits ')
    cbrparser.add_argument('--outdir',help='output directory,default= same dir as input')
    cbrparser.add_argument('--iterations',type=int,default=500,help='Learning Iterations, default =500')
    cbrparser.add_argument('--learningrate',type=float,default=0.01,help='learning_rate. default=0.01')
    cbrparser.add_argument('--depth',type=int,default=6,help='depth of trees. default=6')
    cbrparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    cbrparser.add_argument('--cv',type=int,default=None,help='Cross Validate. default=None.')
    cbrparser.add_argument('--featureimportance',action='store_true',default=False,
        help='List feature importance.default= False')
    cbrparser.add_argument('--importancelevel',type=float,default=0.0,
        help='Select features with higher importance level.default=0 i.e. keep all features')
    cbrparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    cbrparser.add_argument('--overfittingdetection',action='store_true',default=False,
        help='Over Fitting Detection.default= False')
    cbrparser.add_argument('--odpval',type=float,default=0.005,
        help='ranges from 10e-10 to 10e-2. Used with overfittingdetection')
    cbrparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')

    # *************ANNRegressor
    annrparser = subparser.add_parser('ANNRegressor',help='Artificial Neural Network')
    annrparser.set_defaults(which='ANNRegressor')
    annrparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    annrparser.add_argument('--wtargetcol',type=int, default=-1,help='Target column #.default = last col')
    annrparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    annrparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    annrparser.add_argument('--minmaxscale',action='store_true',default=False,
        help='Apply min max scaler to scale predicted to input range. default= False i.e. no scaling')
    annrparser.add_argument('--scaleminmaxvalues',type=float,nargs=2,
        help='Min Max scale limits. default=use input data limits ')
    annrparser.add_argument('--outdir',help='output directory,default= same dir as input')
    annrparser.add_argument('--nodes',type=int,nargs='+',help='# of nodes in each layer. no defaults')
    annrparser.add_argument('--activation',choices=['relu','sigmoid'],nargs='+',
        help='activation per layer.choices: relu or sigmoid. no default, repeat for number of layers')
    annrparser.add_argument('--epochs',type=int,default=100,help='depth of trees. default=100')
    annrparser.add_argument('--batch',type=int,default=5,help='depth of trees. default=5')
    annrparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    annrparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    annrparser.add_argument('--cv',type=int,default=None,help='Cross Validate. default=None.')
    annrparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    tcmdlparser = subparser.add_parser('testCmodels',help='Test Classification models')
    tcmdlparser.set_defaults(which='testCmodels')
    tcmdlparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    tcmdlparser.add_argument('--wanalysiscols',type=int,nargs='+',help='Predictor column #s, no default')
    tcmdlparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    tcmdlparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column # in well csv file. default =-1')
    tcmdlparser.add_argument('--qcut',type=int,default=3,help='Divisions to input target data set. default = 3')
    tcmdlparser.add_argument('--cv',type=int,default=3,help='Cross Validation nfold. default=3')
    tcmdlparser.add_argument('--outdir',help='output directory,default= same dir as input')
    tcmdlparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    lgrparser = subparser.add_parser('logisticreg',help='Apply Logistic Regression Classification')
    lgrparser.set_defaults(which='logisticreg')
    lgrparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    lgrparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    lgrparser.add_argument('--wanalysiscols',type=int,nargs='+',help='Predictor column #s, no default')
    lgrparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    lgrparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column # in well csv file. default = last col')
    lgrparser.add_argument('--coded',action='store_true',default=False,
        help='Target col is already coded-> output from semisupervised. default =False target is not coded')
    lgrparser.add_argument('--classweight',action='store_true',default=False,
        help='Balance classes by proportional weighting. default =False -> no balancing')
    lgrparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    lgrparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    lgrparser.add_argument('--qcut',type=int,default=3,help='Divisions to input target data set. default = 3')
    lgrparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns  . default= 0 1 2 ')
    lgrparser.add_argument('--wellsxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols,default= 0 1 2 3')
    lgrparser.add_argument('--outdir',help='output directory,default= same dir as input')
    lgrparser.add_argument('--cv',type=int,help='Cross Validation default=None')
    lgrparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    lgrparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    lgrparser.add_argument('--balancetype',choices=['ros','smote','adasyn'],default=None,
        help='Random Oversampling,Synthetic Minority Oversampling,Adaptive Synthetic Sampling,default=None')
    lgrparser.add_argument('--nneighbors',type=int,default=1,help='use only for smote and adasyn')
    lgrparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    nbparser = subparser.add_parser('GaussianNaiveBayes',help='Apply Gaussian Naive Bayes Classification')
    nbparser.set_defaults(which='GaussianNaiveBayes')
    nbparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    nbparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    nbparser.add_argument('--wanalysiscols',type=int,nargs='+',help='Predictor column #s, no default')
    nbparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    nbparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column # in well csv file. default =last col')
    nbparser.add_argument('--coded',action='store_true',default=False,
        help='Target col is already coded-> output from semisupervised. default =False target is not coded')
    nbparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    nbparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    nbparser.add_argument('--qcut',type=int,default=3,help='Divisions to input target data set. default = 3')
    nbparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns  . default= 0 1 2 ')
    nbparser.add_argument('--wellsxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols')
    nbparser.add_argument('--outdir',help='output directory,default= same dir as input')
    nbparser.add_argument('--cv',type=int,help='Cross Validation default=None.')
    nbparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    nbparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    nbparser.add_argument('--balancetype',choices=['ros','smote','adasyn'],default=None,
        help='Random Oversampling,Synthetic Minority Oversampling,Adaptive Synthetic Sampling,default=None')
    nbparser.add_argument('--nneighbors',type=int,default=1,help='use only for smote and adasyn')
    nbparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    qdaparser = subparser.add_parser('QuadraticDiscriminantAnalysis',help='Apply Quadratic Discriminant Analysis Classification')
    qdaparser.set_defaults(which='QuadraticDiscriminantAnalysis')
    qdaparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    qdaparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    qdaparser.add_argument('--wanalysiscols',type=int,nargs='+',help='Predictor column #s, no default')
    qdaparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    qdaparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column # in well csv file. default =last col')
    qdaparser.add_argument('--coded',action='store_true',default=False,
        help='Target col is already coded-> output from semisupervised. default =False target is not coded')
    qdaparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    qdaparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    qdaparser.add_argument('--qcut',type=int,default=3,help='Divisions to input target data set. default = 3')
    qdaparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns  . default= 0 1 2 ')
    qdaparser.add_argument('--wellsxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols')
    qdaparser.add_argument('--outdir',help='output directory,default= same dir as input')
    qdaparser.add_argument('--cv',type=int,help='Cross Validation default=None.')
    qdaparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    qdaparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    qdaparser.add_argument('--balancetype',choices=['ros','smote','adasyn'],default=None,
        help='Random Oversampling,Synthetic Minority Oversampling,Adaptive Synthetic Sampling,default=None')
    qdaparser.add_argument('--nneighbors',type=int,default=1,help='use only for smote and adasyn')
    qdaparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    nsvcparser = subparser.add_parser('NuSVC',help='Apply Nu Support Vector Machine Classification')
    nsvcparser.set_defaults(which='NuSVC')
    nsvcparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    nsvcparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    nsvcparser.add_argument('--wanalysiscols',type=int,nargs='+',help='Predictor column #s, no default')
    nsvcparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    nsvcparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column # in well csv file. default =last col')
    nsvcparser.add_argument('--coded',action='store_true',default=False,
        help='Target col is already coded-> output from semisupervised. default =False target is not coded')
    nsvcparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    nsvcparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    nsvcparser.add_argument('--qcut',type=int,default=3,help='Divisions to input target data set. default = 3')
    nsvcparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns  . default= 0 1 2 ')
    nsvcparser.add_argument('--wellsxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols')
    nsvcparser.add_argument('--nu',type=float,default=0.5,
            help='upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. value between 0 1, default =0.5')
    nsvcparser.add_argument('--outdir',help='output directory,default= same dir as input')
    nsvcparser.add_argument('--cv',type=int,help='Cross Validation default=None.')
    nsvcparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    nsvcparser.add_argument('--balancetype',choices=['ros','smote','adasyn'],default=None,
        help='Random Oversampling,Synthetic Minority Oversampling,Adaptive Synthetic Sampling,default=None')
    nsvcparser.add_argument('--nneighbors',type=int,default=1,help='use only for smote and adasyn')
    nsvcparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    nsvcparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    cbcparser = subparser.add_parser('CatBoostClassifier',help='Apply CatBoost Classification - Multi Class')
    cbcparser.set_defaults(which='CatBoostClassifier')
    cbcparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    cbcparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    cbcparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    cbcparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column #.default = last col')
    cbcparser.add_argument('--coded',action='store_true',default=False,
        help='Target col is already coded-> output from semisupervised. default =False target is not coded')
    cbcparser.add_argument('--qcut',type=int,default=3,help='Divisions to input target data set. default = 3')
    cbcparser.add_argument('--outdir',help='output directory,default= same dir as input')
    cbcparser.add_argument('--iterations',type=int,default=500,help='Learning Iterations, default =500')
    cbcparser.add_argument('--learningrate',type=float,default=0.3,help='learning_rate. default=0.3')
    cbcparser.add_argument('--depth',type=int,default=2,help='depth of trees. default=2')
    cbcparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    cbcparser.add_argument('--cv',type=int,default=None,help='Cross Validate. default=None.')
    cbcparser.add_argument('--featureimportance',action='store_true',default=False,
        help='List feature importance.default= False')
    cbcparser.add_argument('--importancelevel',type=float,default=0.0,
        help='Select features with higher importance level.default=0 i.e. keep all features')
    cbcparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    cbcparser.add_argument('--balancetype',choices=['ros','smote','adasyn'],default=None,
        help='Random Oversampling,Synthetic Minority Oversampling,Adaptive Synthetic Sampling,default=None')
    cbcparser.add_argument('--nneighbors',type=int,default=1,help='use only for smote and adasyn')
    cbcparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    tcbcparser = subparser.add_parser('TuneCatBoostClassifier',help='Hyper Parameter Tuning of CatBoost Classification - Multi Class')
    tcbcparser.set_defaults(which='TuneCatBoostClassifier')
    tcbcparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    tcbcparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    tcbcparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    tcbcparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    tcbcparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column # in well csv file. default =-1')
    tcbcparser.add_argument('--coded',action='store_true',default=False,
        help='Target col is already coded-> output from semisupervised. default =False target is not coded')
    tcbcparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    tcbcparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    tcbcparser.add_argument('--qcut',type=int,default=3,help='Divisions to input target data set. default = 3')
    tcbcparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns  . default= 0 1 2 ')
    tcbcparser.add_argument('--wellsxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols,default= 0 1 2 3')
    tcbcparser.add_argument('--outdir',help='output directory,default= same dir as input')
    tcbcparser.add_argument('--iterations',type=int,nargs='+',default=[10,500,1000,5000],
        help='Learning Iterations, default =[10,500,1000,5000]')
    tcbcparser.add_argument('--learningrate',type=float,nargs='+', default=[0.01,0.03,0.1],
        help='learning_rate. default=[0.01,0.03,0.1]')
    tcbcparser.add_argument('--depth',type=int,nargs='+',default=[2,4,6,8],help='depth of trees. default=[2,4,6,8]')
    tcbcparser.add_argument('--cv',type=int,default=3,help='Cross Validation default=3')
    tcbcparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    clparser = subparser.add_parser('clustertest',help='Testing of KMeans # of clusters using elbow plot')
    clparser.set_defaults(which='clustertest')
    clparser.add_argument('allattribcsv',help='csv file with all previously scaled predictors ')
    clparser.add_argument('--cols2cluster',type=int,nargs='+',default=[3,4,5],
        help='Columns to use for clustering. default= 3 4 5 ')
    clparser.add_argument('--colsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    clparser.add_argument('--sample',type=float,default=1.0,help='fraction of data of sample.default=1, i.e. all data')
    clparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')
    clparser.add_argument('--outdir',help='output directory,default= same dir as input')

    cl1parser = subparser.add_parser('clustering',help='Apply KMeans clustering')
    cl1parser.set_defaults(which='clustering')
    cl1parser.add_argument('allattribcsv',help='csv file will all attributes')
    cl1parser.add_argument('--cols2cluster',type=int,nargs='+',default=[3,4,5],
        help='Columns to use for clustering. default= 3 4 5 ')
    cl1parser.add_argument('--colsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    cl1parser.add_argument('--nclusters',type=int,default=5,help='# of clusters. default = 5')
    cl1parser.add_argument('--plotsilhouette',action='store_true',default=False,help='Plot Silhouete. default=False')
    cl1parser.add_argument('--sample',type=float,default=1.0,
        help='fraction of data of sample.default=1, i.e. all data. Use with plotsilhouette')
    cl1parser.add_argument('--xyzcols',type=int,nargs='+',default=[0,1,2],help='Attribute csv x y z columns.default= 0 1 2 ')
    cl1parser.add_argument('--addclass',choices=['labels','dummies'],default='labels',
        help='add cluster labels or binary dummies.default=labels')
    cl1parser.add_argument('--outdir',help='output directory,default= same dir as input')
    cl1parser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')

    gmmparser = subparser.add_parser('GaussianMixtureModel',
        help='Gaussian Mixture Model. model well csv apply to seismic csv')
    gmmparser.set_defaults(which='GaussianMixtureModel')
    gmmparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    gmmparser.add_argument('--bayesian',action='store_true',default=False,
        help='Bayesian Gauusian Mixture Model. default= use Gaussian Mixture Model')
    gmmparser.add_argument('--seisattribcsv',help='csv file of seismic attributes to predict at')
    gmmparser.add_argument('--wanalysiscols',type=int,nargs='+',help='Predictor column #s, no default')
    gmmparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    gmmparser.add_argument('--wtargetcol',type=int,help='Target column # in well csv file. no default ')
    gmmparser.add_argument('--ncomponents',type=int,default=4,help='# of clusters.default=4')
    gmmparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    gmmparser.add_argument('--catcol',type=int,default=None,
        help='Column num to convert from categories to dummies.Only one column is allowed. default=None')
    gmmparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    gmmparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns  . default= 0 1 2 ')
    gmmparser.add_argument('--wellsxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols,default= 0 1 2 3')
    gmmparser.add_argument('--outdir',help='output directory,default= same dir as input')
    gmmparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    tsneparser = subparser.add_parser('tSNE',
        help='Apply tSNE (t distribution Stochastic Neighbor Embedding) clustering to one csv')
    tsneparser.set_defaults(which='tSNE')
    tsneparser.add_argument('allattribcsv',help='csv file will all attributes')
    tsneparser.add_argument('--cols2cluster',type=int,nargs='+',default=[3,4,5],
        help='Columns to use for clustering. default= 3 4 5 ')
    tsneparser.add_argument('--colsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    tsneparser.add_argument('--xyzcols',type=int,nargs='+',default=[0,1,2],help='Attribute csv x y z columns  . default= 0 1 2 ')
    # tsneparser.add_argument('--targetcol',type=int,default = None,
    # help='Target column # to add back. You do not have to add a target default = none')
    tsneparser.add_argument('--learningrate',type=int,default=200,help='Learning rate. default=200')
    tsneparser.add_argument('--sample',type=float,default=0.2,help='fraction of data of sample.default=0.2')
    tsneparser.add_argument('--scalefeatures',action='store_false',default=True,
        help='Do not scale tSNE feature. default = to scale featues')
    tsneparser.add_argument('--outdir',help='output directory,default= same dir as input')
    tsneparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    tsne2parser = subparser.add_parser('tSNE2',
        help='Apply tSNE (t distribution Stochastic Neighbor Embedding) clustering to both well and seismic csv')
    tsne2parser.set_defaults(which='tSNE2')
    tsne2parser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    tsne2parser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    tsne2parser.add_argument('--wtargetcol',type=int, default=-1,help='Target column #.default = last col')
    tsne2parser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    tsne2parser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    tsne2parser.add_argument('--wxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols,default= 0 1 2 3')
    tsne2parser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    tsne2parser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='Seismic Predictor min max col #')
    tsne2parser.add_argument('--sxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns.default= 0 1 2 ')
    tsne2parser.add_argument('--learningrate',type=int,default=200,help='Learning rate. default=200')
    tsne2parser.add_argument('--sample',type=float,default=0.2,help='fraction of data of sample.default=0.2')
    tsne2parser.add_argument('--scalefeatures',action='store_false',default=True,
        help='Do not scale tSNE feature. default = to scale featues')
    tsne2parser.add_argument('--outdir',help='output directory,default= same dir as input')
    tsne2parser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    umapparser = subparser.add_parser('umap',help='Clustering using UMAP (Uniform Manifold Approximation & Projection) to one csv')
    umapparser.set_defaults(which='umap')
    umapparser.add_argument('allattribcsv',help='csv file will all attributes')
    umapparser.add_argument('--cols2cluster',type=int,nargs='+',default=[3,4,5],help='Columns to use for clustering. default= 3 4 5 ')
    umapparser.add_argument('--colsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    umapparser.add_argument('--xyzcols',type=int,nargs='+',default=[0,1,2],help='Attribute csv x y z columns  . default= 0 1 2 ')
    umapparser.add_argument('--nneighbors',type=int,default=5,help='Nearest neighbors. default=5')
    umapparser.add_argument('--mindistance',type=float,default=0.3,help='Min distantce for clustering. default=0.3')
    umapparser.add_argument('--ncomponents',type=int,default=3,help='Projection axes. default=3')
    umapparser.add_argument('--sample',type=float,default=1,help='fraction of data of sample 0 -> 1.default=1, no sampling')
    umapparser.add_argument('--scalefeatures',action='store_true',default=False,
        help='Do not scale umap features. default = not to scale featues')
    umapparser.add_argument('--outdir',help='output directory,default= same dir as input')
    umapparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    dbsnparser = subparser.add_parser('DBSCAN',help='Apply DBSCAN (Density Based Spatial Aanalysis with Noise) clustering')
    dbsnparser.set_defaults(which='DBSCAN')
    dbsnparser.add_argument('allattribcsv',help='csv file will all attributes')
    dbsnparser.add_argument('--cols2cluster',type=int,nargs='+',default=[3,4,5],
        help='Columns to use for clustering. default= 3 4 5 ')
    dbsnparser.add_argument('--colsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    dbsnparser.add_argument('--xyzcols',type=int,nargs='+',default=[0,1,2],help='Attribute csv x y z columns  . default= 0 1 2 ')
    dbsnparser.add_argument('--targetcol',type=int,default=None,
        help='Target column # to add back. You do not have to add a target default = none')
    dbsnparser.add_argument('--eps',type=float,default=0.5,help='eps. default=0.5')
    dbsnparser.add_argument('--minsamples',type=int,default=10,help='minsamples. default=10')
    dbsnparser.add_argument('--addclass',choices=['labels','dummies'],default='labels',
        help='add cluster labels or binary dummies.default=labels')
    dbsnparser.add_argument('--outdir',help='output directory,default= same dir as input')

    sspparser = subparser.add_parser('semisupervised',help='Apply semi supervised Class prediction ')
    sspparser.set_defaults(which='semisupervised')
    sspparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    sspparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    sspparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    sspparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column # in well csv file. default = last column')
    sspparser.add_argument('--wellsxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols,default= 0 1 2 3')
    sspparser.add_argument('--col2drop',type=int,default=None,help='drop column in case of scaled target.default=None')
    # sspparser.add_argument('--qcut',type=int,default=3,help='Divisions to input target data set. default = 3')
    sspparser.add_argument('--sample',type=float,default=.005,help='fraction of data of sample.default=0.005')
    sspparser.add_argument('--outdir',help='output directory,default= same dir as input')
    sspparser.add_argument('--nneighbors',type=int,default=7,help='Used with knn to classify data.default=7')
    sspparser.add_argument('--kernel',choices=['knn','rbf'],default='knn',
        help='Kernel for semi supervised classification.default= knn')

    if not oneline:
        result = mainparser.parse_args()
    else:
        result = mainparser.parse_args(oneline)

    # result = mainparser.parse_args()
    if result.which not in allcommands:
        mainparser.print_help()
        exit()
    else:
        return result

def main():
    """Main program."""
    sns.set()
    warnings.filterwarnings("ignore")

    def process_commands():
        """Command line processing."""
        print(cmdl.which)

        if cmdl.which == 'dropcols':
            process_dropcols(cmdl.csvfile,
                cmdlcols2drop=cmdl.cols2drop,
                cmdloutdir=cmdl.outdir)


        elif cmdl.which == 'listcsvcols':
            process_listcsvcols(cmdl.csvfile)

        elif cmdl.which == 'PCAanalysis':
            process_PCAanalysis(cmdl.allattribcsv,
                cmdlacolsrange=cmdl.acolsrange,
                cmdlanalysiscols=cmdl.analysiscols,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'PCAfilter':
            process_PCAfilter(cmdl.allattribcsv,
                cmdlacolsrange=cmdl.acolsrange,
                cmdlanalysiscols=cmdl.analysiscols,
                cmdltargetcol=cmdl.targetcol,
                cmdlncomponents=cmdl.ncomponents,
                cmdloutdir=cmdl.outdir,
                cmdlcols2addback=cmdl.cols2addback)

        elif cmdl.which == 'scattermatrix':
            process_scattermatrix(cmdl.allattribcsv,
                cmdlwellxyzcols=cmdl.wellxyzcols,
                cmdlsample=cmdl.sample)

        elif cmdl.which == 'EDA':
            process_eda(cmdl.allattribcsv,
                cmdlxyzcols=cmdl.xyzcols,
                cmdlpolydeg=cmdl.polydeg,
                cmdlsample=cmdl.sample,
                cmdlhideplot=cmdl.hideplot,
                cmdlplotoption=cmdl.plotoption,
                cmdloutdir=cmdl.outdir)

        elif cmdl.which == 'featureranking':
            process_featureranking(cmdl.allattribcsv,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdltestfeatures=cmdl.testfeatures,
                cmdllassoalpha=cmdl.lassoalpha,
                cmdlfeatures2keep=cmdl.features2keep,
                cmdlcv=cmdl.cv,
                cmdltraintestsplit=cmdl.traintestsplit)

        elif cmdl.which == 'linreg':
            process_linreg(cmdl.allattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlminmaxscale=cmdl.minmaxscale,
                cmdloutdir=cmdl.outdir,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'KNNtest':
            process_KNNtest(cmdl.allattribcsv,
                cmdlsample=cmdl.sample,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlcv=cmdl.cv,
                cmdlhideplot=cmdl.hideplot,
                cmdloutdir=cmdl.outdir)

        elif cmdl.which == 'KNN':
            process_KNN(cmdl.wellattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlkneighbors=cmdl.kneighbors,
                cmdlminmaxscale=cmdl.minmaxscale,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdloutdir=cmdl.outdir,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'CatBoostRegressor':
            process_CatBoostRegressor(cmdl.wellattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlminmaxscale=cmdl.minmaxscale,
                cmdloutdir=cmdl.outdir,
                cmdliterations=cmdl.iterations,
                cmdllearningrate=cmdl.learningrate,
                cmdlscaleminmaxvalues=cmdl.scaleminmaxvalues,
                cmdldepth=cmdl.depth,
                cmdlcv=cmdl.cv,
                cmdlfeatureimportance=cmdl.featureimportance,
                cmdlimportancelevel=cmdl.importancelevel,
                cmdlhideplot=cmdl.hideplot,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdlvalsize=cmdl.valsize,
                cmdloverfittingdetection=cmdl.overfittingdetection,
                cmdlodpval=cmdl.odpval)

        elif cmdl.which == 'ANNRegressor':
            process_ANNRegressor(cmdl.wellattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlminmaxscale=cmdl.minmaxscale,
                cmdlscaleminmaxvalues=cmdl.scaleminmaxvalues,
                cmdloutdir=cmdl.outdir,
                cmdlnodes=cmdl.nodes,
                cmdlactivation=cmdl.activation,
                cmdlepochs=cmdl.epochs,
                cmdlbatch=cmdl.batch,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdlcv=cmdl.cv,
                cmdlhideplot=cmdl.hideplot,
                cmdlvalsize=cmdl.valsize)

        elif cmdl.which == 'NuSVR':
            process_NuSVR(cmdl.wellattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlminmaxscale=cmdl.minmaxscale,
                cmdlscaleminmaxvalues=cmdl.scaleminmaxvalues,
                cmdloutdir=cmdl.outdir,
                cmdlerrpenalty=cmdl.errpenalty,
                cmdlnu=cmdl.nu,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdlcv=cmdl.cv,
                cmdlhideplot=cmdl.hideplot,
                cmdlvalsize=cmdl.valsize)

        elif cmdl.which == 'SGDR':
            process_SGDR(cmdl.wellattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlminmaxscale=cmdl.minmaxscale,
                cmdlscaleminmaxvalues=cmdl.scaleminmaxvalues,
                cmdloutdir=cmdl.outdir,
                cmdlloss=cmdl.loss,
                cmdlpenalty=cmdl.penalty,
                cmdll1ratio=cmdl.l1ratio,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdlcv=cmdl.cv,
                cmdlhideplot=cmdl.hideplot,
                cmdlvalsize=cmdl.valsize)

        elif cmdl.which == 'CatBoostClassifier':
            process_CatBoostClassifier(cmdl.wellattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdloutdir=cmdl.outdir,
                cmdliterations=cmdl.iterations,
                cmdllearningrate=cmdl.learningrate,
                cmdlcoded=cmdl.coded,
                cmdldepth=cmdl.depth,
                cmdlqcut=cmdl.qcut,
                cmdlcv=cmdl.cv,
                cmdlvalsize=cmdl.valsize,
                cmdlfeatureimportance=cmdl.featureimportance,
                cmdlimportancelevel=cmdl.importancelevel,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdlbalancetype=cmdl.balancetype,
                cmdlnneighbors=cmdl.nneighbors,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'TuneCatBoostClassifier':
            process_TuneCatBoostClassifier(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlwellsxyzcols=cmdl.wellsxyzcols,
                cmdlcoded=cmdl.coded,
                cmdloutdir=cmdl.outdir,
                cmdliterations=cmdl.iterations,
                cmdllearningrate=cmdl.learningrate,
                cmdldepth=cmdl.depth,
                cmdlqcut=cmdl.qcut,
                cmdlcv=cmdl.cv,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'testCmodels':
            process_testCmodels(cmdl.wellattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwanalysiscols=cmdl.wanalysiscols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlqcut=cmdl.qcut,
                cmdlcv=cmdl.cv,
                cmdloutdir=cmdl.outdir,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'logisticreg':
            process_logisticreg(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwanalysiscols=cmdl.wanalysiscols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlwellsxyzcols=cmdl.wellsxyzcols,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlqcut=cmdl.qcut,cmdlcoded=cmdl.coded,
                cmdloutdir=cmdl.outdir,
                cmdlcv=cmdl.cv,
                cmdlvalsize=cmdl.valsize,
                cmdlhideplot=cmdl.hideplot,
                cmdlclassweight=cmdl.classweight,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdlbalancetype=cmdl.balancetype,
                cmdlnneighbors=cmdl.nneighbors)

        elif cmdl.which == 'GaussianNaiveBayes':
            process_GaussianNaiveBayes(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwanalysiscols=cmdl.wanalysiscols,
                cmdlqcut=cmdl.qcut,
                cmdlwellsxyzcols=cmdl.wellsxyzcols,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlcoded=cmdl.coded,
                cmdloutdir=cmdl.outdir,
                cmdlcv=cmdl.cv,
                cmdlvalsize=cmdl.valsize,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdlbalancetype=cmdl.balancetype,
                cmdlnneighbors=cmdl.nneighbors,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'QuadraticDiscriminantAnalysis':
            process_QuadraticDiscriminantAnalysis(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwanalysiscols=cmdl.wanalysiscols,
                cmdlqcut=cmdl.qcut,
                cmdlwellsxyzcols=cmdl.wellsxyzcols,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlcoded=cmdl.coded,
                cmdloutdir=cmdl.outdir,
                cmdlcv=cmdl.cv,
                cmdlvalsize=cmdl.valsize,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdlbalancetype=cmdl.balancetype,
                cmdlnneighbors=cmdl.nneighbors,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'NuSVC':
            process_NuSVC(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwanalysiscols=cmdl.wanalysiscols,
                cmdlqcut=cmdl.qcut,
                cmdlwellsxyzcols=cmdl.wellsxyzcols,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlcoded=cmdl.coded,
                cmdloutdir=cmdl.outdir,
                cmdlcv=cmdl.cv,
                cmdlvalsize=cmdl.valsize,
                cmdlnu=cmdl.nu,
                cmdlbalancetype=cmdl.balancetype,
                cmdlnneighbors=cmdl.nneighbors,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'GaussianMixtureModel':
            process_GaussianMixtureModel(cmdl.wellattribcsv,
                cmdlseisattribcsv=cmdl.seisattribcsv,
                cmdlbayesian=cmdl.bayesian,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwanalysiscols=cmdl.wanalysiscols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlwellsxyzcols=cmdl.wellsxyzcols,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlcatcol=cmdl.catcol,
                cmdlncomponents=cmdl.ncomponents,
                cmdloutdir=cmdl.outdir,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'clustertest':
            process_clustertest(cmdl.allattribcsv,
                cmdlcolsrange=cmdl.colsrange,
                cmdlcols2cluster=cmdl.cols2cluster,
                cmdlsample=cmdl.sample,
                cmdloutdir=cmdl.outdir,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'clustering':
            process_clustering(cmdl.allattribcsv,
                cmdlcolsrange=cmdl.colsrange,
                cmdlcols2cluster=cmdl.cols2cluster,
                cmdlnclusters=cmdl.nclusters,
                cmdlplotsilhouette=cmdl.plotsilhouette,
                cmdlsample=cmdl.sample,
                cmdlxyzcols=cmdl.xyzcols,
                cmdladdclass=cmdl.addclass,
                cmdloutdir=cmdl.outdir,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'DBSCAN':
            process_dbscan(cmdl.allattribcsv,
                cmdlcolsrange=cmdl.colsrange,
                cmdlcols2cluster=cmdl.cols2cluster,
                cmdlxyzcols=cmdl.xyzcols,
                cmdlminsamples=cmdl.minsamples,
                cmdladdclass=cmdl.addclass,
                cmdleps=cmdl.eps,
                cmdloutdir=cmdl.outdir)

        elif cmdl.which == 'tSNE':
            process_tSNE(cmdl.allattribcsv,
                cmdlcolsrange=cmdl.colsrange,
                cmdlcols2cluster=cmdl.cols2cluster,
                cmdlsample=cmdl.sample,
                cmdlxyzcols=cmdl.xyzcols,
                cmdllearningrate=cmdl.learningrate,
                cmdlscalefeatures=cmdl.scalefeatures,
                cmdloutdir=cmdl.outdir,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'tSNE2':
            process_tSNE2(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlwxyzcols=cmdl.wxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlsxyzcols=cmdl.sxyzcols,
                cmdlsample=cmdl.sample,
                cmdllearningrate=cmdl.learningrate,
                cmdlscalefeatures=cmdl.scalefeatures,
                cmdloutdir=cmdl.outdir,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'umap':
            process_umap(cmdl.allattribcsv,
                cmdlcolsrange=cmdl.colsrange,
                cmdlcols2cluster=cmdl.cols2cluster,
                cmdlsample=cmdl.sample,
                cmdlxyzcols=cmdl.xyzcols,
                cmdlnneighbors=cmdl.nneighbors,
                cmdlmindistance=cmdl.mindistance,
                cmdlncomponents=cmdl.ncomponents,
                cmdlscalefeatures=cmdl.scalefeatures,
                cmdloutdir=cmdl.outdir,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'semisupervised':
            process_semisupervised(cmdl.wellattribcsv,
                cmdl.seisattribcsv,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwellsxyzcols=cmdl.wellsxyzcols,
                cmdlsample=cmdl.sample,
                cmdlnneighbors=cmdl.nneighbors,
                cmdlkernel=cmdl.kernel,
                cmdlcol2drop=cmdl.col2drop,
                cmdloutdir=cmdl.outdir)

    # print(__doc__)
    cmdl = getcommandline()
    if cmdl.which == 'workflow':
        lnum = 0
        startline = cmdl.startline
        with open(cmdl.commandfile,'r') as cmdlfile:
            for line in cmdlfile:
                lnum += 1
                print()
                print('%00d:>' % lnum,line)
                if lnum >= startline:
                    parsedline = shlex.split(line)[2:]
                    # if len(parsedline) >=1:
                    if len(parsedline) >= 1 and not cmnt(line):
                        cmdl = getcommandline(*parsedline)
                        process_commands()
                else:
                    print('Skip line:%00d' % lnum,line)
    else:
        process_commands()


if __name__=='__main__':
	main()
