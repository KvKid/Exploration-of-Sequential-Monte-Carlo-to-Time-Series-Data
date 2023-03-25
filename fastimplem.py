import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy.stats
import scipy
import math
import random
from scipy.stats import binom_test
from scipy.stats import kstest
from scipy.stats import norm
import sys 
from scipy.stats import binom
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from scipy.stats import poisson
import random
import numpy as np
from numpy.random import random
import matplotlib.dates as mdates
import analysisimport as ai

class MonteCarlo:
    def __init__(self,analysis):
        self.hi = 2
        self.df = analysis.df
        self.analysis = analysis
        self.SMCstates = {}
        self.smoothdf = pd.DataFrame()

    # state space model - Mutation
    def StateSpace_iplus1_mean(self,xt,v):
        dummy = xt+np.random.normal(0,scale =np.sqrt(v))
        # print('oub: ',dummy)
        while dummy<0:
            dummy = xt+np.random.normal(0,scale =np.sqrt(v))
        # print("fixed")
        return(dummy)
        #return(Xt_1 *dnp.random.lognormal(mean=0.0, sigma=1.0)
        #return(Xt_1*np.random.exponential(scale = 1))
    def StateSpace_iplus1_var(self,v,pt):
        return(v*np.random.lognormal(mean = 0,sigma = pt))

    def transitionkernels(self,x,statevar1,statevar2,covv):
        # dummy = x+np.random.normal(0,scale =np.sqrt(v))
        covv = np.sqrt(statevar1) * np.sqrt(statevar2)*covv
        cov_mat = [[statevar1,covv],[covv,statevar2]]
        dummy = x + np.random.multivariate_normal([0,0],cov_mat)
        # print('oub: ',dummy)
        while dummy[0]<0 or dummy[1]<0:
            dummy = x + np.random.multivariate_normal([0,0],cov_mat)
        # print("fixed")
        return(dummy)
    def logtransitionkernels(self,X,Y,Sdx,Sdy,Corr):
        LX = np.log(X)
        LY = np.log(Y)
        cov_mat = [[Sdx**2,Sdx*Sdy*Corr],[Sdx*Sdy*Corr,Sdy**2]]
        dummy = [LX,LY] + np.random.multivariate_normal([0,0],cov_mat)
        return(dummy)

    def StateSpace_iplus1corr(self,xt):
        wt = xt + np.random.normal(0,0.05)
        while wt>1 or wt<-1:
            wt = xt + np.random.normal(0,0.05)
        #print(wt)
        return(wt)
            
    #Mutation
    def hiddenmodel_iplus1(self,obs1,mu1,obs2,mu2):
        return(poisson.pmf(obs1,mu1)*poisson.pmf(obs2,mu2))
    
    def hiddenmodel_i_plus_1_var(self,observation,statemean,statevar):
        return(scipy.stats.norm.pdf(observation,statemean,np.sqrt(statevar)))

    def general_filter(self,obs1,obs2,N,param):
        ParticlesSdx = np.zeros((N,len(obs1)))
        ParticlesSdy =  np.zeros((N,len(obs1)))
        ParticlesCorr =  np.zeros((N,len(obs1)))
        ParticlesX =  np.zeros((N,len(obs1)))
        ParticlesY =  np.zeros((N,len(obs1)))
        ParticlesIndicies =  np.zeros((N,len(obs1)))

        obs1 = np.array(obs1)
        obs2 = np.array(obs2)

        Sdx = np.zeros(N)
        Sdy = np.zeros((N))
        Corr= np.zeros((N))
        X= np.zeros((N))
        Y= np.zeros((N))
        # Initial prior distribution for our unobservable quantity of interest
        Sdx = [np.random.uniform(0,3) for i in range(0,N)]
        Sdy = [np.random.uniform(0,3) for i in range(0,N)]
        Corr = [np.random.uniform(-1,1) for elem in range(N)]
        X = [np.random.uniform(0,10) for elem in range(N)]
        Y = [np.random.uniform(0,10) for elem in range(N)]
        Sdx = np.array(Sdx)
        Sdy = np.array(Sdy)
        Corr = np.array(Corr)
        X= np.array(X)
        Y = np.array(Y)

        for i in range(0,len(obs1)):
            #Give the parameter for our state space model and updated rate 
            # having ovserved our data point Y[i]
            [Sdx,Sdy,Corr,X,Y,I] = self.Filter(Sdx,Sdy,Corr,X,Y,obs1[i],obs2[i],N,param)
            ParticlesSdx[:,i] = Sdx
            ParticlesSdy[:,i] = Sdy
            ParticlesCorr[:,i] = Corr
            ParticlesX[:,i] = X
            ParticlesY[:,i] = Y
            ParticlesIndicies[:,i] = I
            print(i)

        #print(RoundedIndicies.shape)
        return(ParticlesSdx,ParticlesSdy,ParticlesCorr,ParticlesX,ParticlesY,ParticlesIndicies)

    def Filter(self,Sdx,Sdy,Corr,X,Y,Xobs,Yobs,N,param):
        Weights = np.zeros(N)
        #For each particle in our filter
        #Particles are a 2xN object
        #Draw a sample from our 'proposal distribution' i.e. for computing predictive step
        for i in range(0,N):
            Sdx[i] = self.StateSpace_iplus1_var(Sdx[i],param)
            Sdy[i] = self.StateSpace_iplus1_var(Sdy[i],param)
            Corr[i] = self.StateSpace_iplus1corr(Corr[i])

            [LX,LY] = self.logtransitionkernels(X[i],Y[i],Sdx[i],Sdy[i],Corr[i])
            X[i] = np.exp(LX)
            Y[i] =np.exp(LY)

            Weights[i] = self.hiddenmodel_iplus1(Xobs,X[i],Yobs,Y[i])


        #Normalise weights
        sumW = sum(Weights)
        for i in range(N):
            Weights[i] = Weights[i]/sumW
        #Resample from these weights    

        I = np.random.choice(list(range(N)),replace = True, size = N, p = Weights)
        # print(I)
        Sdx = Sdx[I]
        Sdy = Sdy[I]
        Corr = Corr[I]
        X = X[I]
        Y = Y[I]
                
        #Convert to state space vectors
        return(Sdx,Sdy,Corr,X,Y,I)
    
    def movingaveragemontecarlo(self,ID1,ID2,K,N,pt,pltshow = True):
        p_est_dict = {}
        vals = self.df[ID1]
        df_dummy = self.df.copy()
        [ParticlesSdx,ParticlesSdy,ParticlesCorr,ParticlesX,ParticlesY,ParticlesIndicies] = self.general_filter(self.df[ID1],self.df[ID2],N,pt)
        #self.SMCstates[ID] = Indicies
        #np.savetxt("States.txt",Indicies , delimiter=",")
        variances = np.mean(ParticlesSdx,axis = 0)
        #dfp.rolling(7, center=True).mean()
        hi = pd.DataFrame(vals)
        dude = pd.DataFrame(hi).rolling(K, center=True).mean()        
        dudevar = pd.DataFrame(hi).rolling(30, center=True).var()  
        
        df_dummy['Plottingrate'] = np.mean(ParticlesX,axis = 0)
        if pltshow == True:
            ID=ID1
            figure(figsize=(5, 20), dpi=320)
            fig, ax = plt.subplots(nrows = 2, ncols  =1, sharex = True, sharey = False)
            plt.subplots_adjust(hspace = 0)
            ax[0].plot(self.df.index.values,df_dummy['Plottingrate'], color = 'red')
            ax[0].bar(self.df.index.values, dude[ID] ,width = 0.3)
            ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%b')) # Only plot the month
            ax[0].set_xlabel('Date')
            ax[0].title.set_text(ID)
            ax[0].set_ylabel('Frequency')
            
            #ax2 = ax.twinx()
            #widthintervals = [0.1 + 0.2*(np.log10(changeinvar[i])) for i in range(0,255)]#np.log(cinvar[-1])#initialwidth*np.abs(cinvar[-1]-1)
            ax[1].plot(self.df.index.values,variances,label = 'widths',color = plt.cm.viridis(.9))
            ax[1].plot(self.df.index.values,dudevar[ID1],label = 'widths',color = plt.cm.viridis(0))
            ax[1].set_ylabel('Variance')
            #plt.savefig('./Plots/SMC_dyn_interval/'+ID)
            #ax[1].set_ylim(top = 1)
            plt.show()
        print("ID2")
        vals = self.df[ID2]
        print("1")
        df_dummy = self.df.copy()
        variances = np.mean(ParticlesSdy,axis = 0)
        #dfp.rolling(7, center=True).mean()
        hi = pd.DataFrame(vals)
        dude = pd.DataFrame(hi).rolling(K, center=True).mean()        
        dudevar = pd.DataFrame(hi).rolling(30, center=True).var()  
        # print("2")

        
        df_dummy['Plottingrate'] = np.mean(ParticlesY,axis = 0)
        if pltshow == True:
            ID=ID2
            # print("3")

            figure(figsize=(5, 20), dpi=320)
            fig, ax = plt.subplots(nrows = 2, ncols  =1, sharex = True, sharey = False)
            plt.subplots_adjust(hspace = 0)
            ax[0].plot(self.df.index.values,df_dummy['Plottingrate'], color = 'red')
            ax[0].bar(self.df.index.values, dude[ID] ,width = 0.3)
            # print("4")
            
            ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%b')) # Only plot the month
            ax[0].set_xlabel('Date')
            ax[0].title.set_text(ID)
            ax[0].set_ylabel('Frequency')
            # print("1")
            
            #ax2 = ax.twinx()
            #widthintervals = [0.1 + 0.2*(np.log10(changeinvar[i])) for i in range(0,255)]#np.log(cinvar[-1])#initialwidth*np.abs(cinvar[-1]-1)
            ax[1].plot(self.df.index.values,variances,label = 'widths',color = plt.cm.viridis(.9))
            ax[1].plot(self.df.index.values,dudevar[ID2],label = 'widths',color = plt.cm.viridis(0))
            ax[1].set_ylabel('Variance')
            # print("1")

            #plt.savefig('./Plots/SMC_dyn_interval/'+ID)
            #ax[1].set_ylim(top = 1)
            plt.show()
        Corr = np.mean(ParticlesCorr,axis = 0)
        plt.plot(self.df.index.values,Corr)
        plt.show()
        # return(MeanStates1,VarStates1,Imean,Ivar)


def get_ancestry(States,Indicies):
    trajectory_ind = np.zeros(Indicies.shape[1])
    trajectory_states = np.zeros(Indicies.shape[1])
    index =0
    
    for i in reversed(range(0,Indicies.shape[1])):
        #index before:
        trajectory_states[i] = States[index,i]

        index = int(Indicies[index,i])
        trajectory_ind[i] = index
        print(index)
    return(trajectory_states, trajectory_ind)


import pickle

with open('./smoothingdata/CovarianceMeasureparam1/SdyStates.pickle', 'rb') as meanfile:
    state_dict_Sdy = pickle.load(meanfile)
with open('./smoothingdata/CovarianceMeasureparam1/CorrStates.pickle', 'rb') as meanfile:
    state_dict_Corr = pickle.load(meanfile)
with open('./smoothingdata/CovarianceMeasureparam1/XStates.pickle', 'rb') as meanfile:
    state_dict_X = pickle.load(meanfile)
with open('./smoothingdata/CovarianceMeasureparam1/YStates.pickle', 'rb') as meanfile:
    state_dict_Y = pickle.load(meanfile)

import pickle
analysis3 = ai.Model()
df = analysis3.df
Accuracy = 20
N = 3000
IDs = list(analysis3.df)
K = 2
state_dict_Sdx = {}
# state_dict_Sdy = {}
# state_dict_Corr = {}
# state_dict_X = {}
# state_dict_Y = {}
# hi = MonteCarlo()
cols = list(df)[:-1]#.reverse()
cols.reverse()
print(cols)
for ii in range(len(cols)):
    ID1 = cols[ii]
    for jj in range(0,ii):
        ID2 = cols[jj]
        IDNOM = ID1+ID2
        if IDNOM in state_dict_Corr:
            continue
        print('Smoother for ', ID1, " and ",ID2)
        try:
            Ancestry_matrixSdx = np.zeros((Accuracy,len(analysis3.df[ID1])))
            Ancestry_matrixSdy = np.zeros((Accuracy,len(analysis3.df[ID1])))
            Ancestry_matrixCorr = np.zeros((Accuracy,len(analysis3.df[ID1])))
            Ancestry_matrixX = np.zeros((Accuracy,len(analysis3.df[ID1])))
            Ancestry_matrixY = np.zeros((Accuracy,len(analysis3.df[ID1])))
            # Ancestry_matrix_var = np.zeros((Accuracy,len(analysis3.df[ID])))
            hi = MonteCarlo(analysis3)
            pt = 0.05
            for i in range(Accuracy):
                hi = MonteCarlo(analysis3)
                #[rate,ParticleStates,changeinvar,p,q] = self.general_filter(self.df[ID],N)
                #[,IndiciesVar] = hi.general_filter(analysis3.df[ID],2000,t1 = (Accuracy), t2 = 100*i/(Accuracy))
                [ParticlesSdx,ParticlesSdy,ParticlesCorr,ParticlesX,ParticlesY,Indicies] = hi.general_filter(df[ID1],df[ID2],N,pt)
                #Accuracy *len(IDs) for global percentage counter
                [Ancestry_matrixSdx[i,:],h] = get_ancestry(ParticlesSdx,Indicies)
                [Ancestry_matrixSdy[i,:],h] = get_ancestry(ParticlesSdy,Indicies)
                [Ancestry_matrixCorr[i,:],h] = get_ancestry(ParticlesCorr,Indicies)
                [Ancestry_matrixX[i,:],h] = get_ancestry(ParticlesX,Indicies)
                [Ancestry_matrixY[i,:],h] = get_ancestry(ParticlesY,Indicies)
                print('\n', int(100*(i+1)/Accuracy),'% SMCs.')
            
                
                state_dict_Sdx[IDNOM] = Ancestry_matrixSdx
                state_dict_Sdy[IDNOM] = Ancestry_matrixSdy
                state_dict_Corr[IDNOM] = Ancestry_matrixCorr
                state_dict_X[IDNOM] = Ancestry_matrixX
                state_dict_Y[IDNOM] = Ancestry_matrixY

                with open('./smoothingdata/CovarianceMeasureparam1/SdxStates.pickle', 'wb') as meanfile:
                    pickle.dump(state_dict_Sdx,meanfile,protocol = pickle.HIGHEST_PROTOCOL)
                with open('./smoothingdata/CovarianceMeasureparam1/SdyStates.pickle', 'wb') as meanfile:
                    pickle.dump(state_dict_Sdy,meanfile,protocol = pickle.HIGHEST_PROTOCOL)
                with open('./smoothingdata/CovarianceMeasureparam1/CorrStates.pickle', 'wb') as meanfile:
                    pickle.dump(state_dict_Corr,meanfile,protocol = pickle.HIGHEST_PROTOCOL)
                with open('./smoothingdata/CovarianceMeasureparam1/XStates.pickle', 'wb') as meanfile:
                    pickle.dump(state_dict_X,meanfile,protocol = pickle.HIGHEST_PROTOCOL)
                with open('./smoothingdata/CovarianceMeasureparam1/YStates.pickle', 'wb') as meanfile:
                    pickle.dump(state_dict_Y,meanfile,protocol = pickle.HIGHEST_PROTOCOL)
                pltshow = True
                p_est_dict = {}
                vals = df[ID1]
                df_dummy = df.copy()
                
                #self.SMCstates[ID] = Indicies
                #np.savetxt("States.txt",Indicies , delimiter=",")
                variances = np.mean(ParticlesSdx,axis = 0)
                #dfp.rolling(7, center=True).mean()
                hi = pd.DataFrame(vals)
                dude = pd.DataFrame(hi).rolling(K, center=True).mean()        
                dudevar = pd.DataFrame(hi).rolling(30, center=True).var()  
                
                df_dummy['Plottingrate'] = np.mean(ParticlesX,axis = 0)
                if pltshow == True:
                    ID=ID1
                    figure(figsize=(5, 20), dpi=320)
                    fig, ax = plt.subplots(nrows = 2, ncols  =1, sharex = True, sharey = False)
                    plt.subplots_adjust(hspace = 0)
                    ax[0].plot(df.index.values,df_dummy['Plottingrate'], color = 'red')
                    ax[0].bar(df.index.values, dude[ID] ,width = 0.3)
                    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%b')) # Only plot the month
                    ax[0].set_xlabel('Date')
                    ax[0].title.set_text(ID)
                    ax[0].set_ylabel('Frequency')
                    
                    #ax2 = ax.twinx()
                    #widthintervals = [0.1 + 0.2*(np.log10(changeinvar[i])) for i in range(0,255)]#np.log(cinvar[-1])#initialwidth*np.abs(cinvar[-1]-1)
                    ax[1].plot(df.index.values,variances,label = 'widths',color = plt.cm.viridis(.9))
                    ax[1].plot(df.index.values,dudevar[ID1],label = 'widths',color = plt.cm.viridis(0))
                    ax[1].set_ylabel('Variance')
                    plt.savefig('./Plots/SmoothedSMC/CovPlotsq/Obs/'+ID)
                    #ax[1].set_ylim(top = 1)
                    plt.show()
                print("ID2")
                vals = df[ID2]
                print("1")
                df_dummy = df.copy()
                variances = np.mean(ParticlesSdy,axis = 0)
                #dfp.rolling(7, center=True).mean()
                hi = pd.DataFrame(vals)
                dude = pd.DataFrame(hi).rolling(K, center=True).mean()        
                dudevar = pd.DataFrame(hi).rolling(30, center=True).var()  
                # print("2")

                
                df_dummy['Plottingrate'] = np.mean(ParticlesY,axis = 0)
                if pltshow == True:
                    ID=ID2
                    # print("3")

                    figure(figsize=(5, 20), dpi=320)
                    fig, ax = plt.subplots(nrows = 2, ncols  =1, sharex = True, sharey = False)
                    plt.subplots_adjust(hspace = 0)
                    ax[0].plot(df.index.values,df_dummy['Plottingrate'], color = 'red')
                    ax[0].bar(df.index.values, dude[ID] ,width = 0.3)
                    # print("4")
                    
                    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%b')) # Only plot the month
                    ax[0].set_xlabel('Date')
                    ax[0].title.set_text(ID)
                    ax[0].set_ylabel('Frequency')
                    # print("1")
                    
                    #ax2 = ax.twinx()
                    #widthintervals = [0.1 + 0.2*(np.log10(changeinvar[i])) for i in range(0,255)]#np.log(cinvar[-1])#initialwidth*np.abs(cinvar[-1]-1)
                    ax[1].plot(df.index.values,variances,label = 'widths',color = plt.cm.viridis(.9))
                    ax[1].plot(df.index.values,dudevar[ID2],label = 'widths',color = plt.cm.viridis(0))
                    ax[1].set_ylabel('Variance')
                    # print("1")

                    #plt.savefig('./Plots/SMC_dyn_interval/'+ID)
                    plt.savefig('./Plots/SmoothedSMC/CovPlotsq/Obs/'+ID)

                    #ax[1].set_ylim(top = 1)
                    plt.show()
                fig, ax = plt.subplots(nrows = 1, ncols  =1, sharex = True, sharey = False)
                plt.subplots_adjust(hspace = 0)
                Corr = np.mean(ParticlesCorr,axis = 0)
                ax.plot(df.index.values,Corr)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b')) # Only plot the month
                plt.title(str(ID1) + " against " + str(ID2))
                plt.xlabel("Month")
                plt.ylabel("Covariance")
                plt.savefig("./Plots/SmoothedSMC/CovPlotsq/"+str(IDNOM))
                plt.show()
            # return(MeanStates1,VarStates1,Imean,Ivar)
            # input()
        except Exception as e:
            print(e)
            continue