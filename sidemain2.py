import seaborn as sns
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
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima

class Model:
    def __init__(self):
        self.df = self.readindata() # processes our data into one dataframe
        self.cumprobdf = self.generatecumdata() # generates a cumulative probability dataframe
        self.binomialestimates = self.generatefinalbinomest() # returns final binomial estimates given observed all the data

    def readindata(self):
        k1 = pd.read_csv('keywordlist1.txt',sep = ';',header=None)
        keys = k1[0]
        keys1 = keys.to_list()
        k2 = pd.read_csv('keywordlist1.txt',sep = ';',header=None)
        keys2 = k2[0].to_list()
        
        df1 = pd.read_csv('combined.csv', sep = ';', header = None,on_bad_lines='skip',names = ['Link'] + keys1 + ['Date','a26'])
        df1 = df1.dropna(subset = 'Date')
        
        df2 = pd.read_csv('set2_combined.csv', sep = ';', header = None,on_bad_lines='skip',names = ['Link'] + keys2 + ['Date','b26'])
        df2 = df2.dropna(subset = 'Date')

        #Perform outer join
        df = pd.merge(df1,df2,how = 'outer')
        df['Date'] = [elem.split(sep='T')[0] for elem in df['Date']]
        df['Date'] = [datetime.strptime(elem,"%Y-%m-%d") for elem in df['Date']]
        #df['Date'] = [elem.strftime('%Y-%m') for elem in df['Date']]
        #Convert date format

        #Create a counter of the number of articles so when resampling, we attain the number of articles per day
        df['NumOfArticles'] = [1 for i in range(len(df['Date']))]

        #Drop columns a26 and b26 and the list of links
        del df['a26']
        del df['b26']
        del df['Link']

        #Set index to equal date and get rid of that column
        df.index = df['Date']
        del df['Date']

        #For each row, if the element appears in the article set equal to 1 else 0
        for column in list(df):
            df[column] = [1 if elem>0 else 0 for elem in df[column]]

        #Group by day
        df = df.resample('D').sum()

        #Select start and end dates
        df =df.loc['2022-02-02':'2023-2-10']
        print("Process our DataFrame:                   Successfully parsed and formatted our data")
        df.to_csv('./processeddata.csv')
        return(df) # returns processed dataframe

    def generatecumdata(self):
        #Create a dataframe that computes the cumulative probability up until that date of a keyword occuring
        cumprob = self.df.copy()

        #Generates cumulative sum
        for column in list(cumprob)[:-1]:
            cumprob[column] = [sum(cumprob[column][:i+1])/sum(cumprob['NumOfArticles'][:i+1]) for i in range(0,len(cumprob[column]))]
        cumprob['NumOfArticles'] =np.cumsum(cumprob['NumOfArticles'])

        #Compute Binomial Estimates
        binomestimates = [sum(self.df[column])/sum(self.df['NumOfArticles']) for column in list(self.df)[:-1]]

        #Check that estimates line up with cumulative probaiblities at the end of the dataframe
        for i in range(0,len(cumprob.iloc[-1][:-1])):
            if (cumprob.iloc[-1][:-1][i] != binomestimates[i]):
                print(False)
                break
            if i == len(cumprob.iloc[-1][:-1])-1:
                print("""Process our Cumulative Probabilities DataFrame:     TEST-
                    Binomial Probabilities is the same as last row of cumulative probabilities""")
                #If test completes successfully
        return(cumprob) # returns a dataframe containing cumulative probabilities
        
    def generatefinalbinomest(self):
        return([sum(self.df[column])/sum(self.df['NumOfArticles']) for column in list(self.df)[:-1]])
    
    def MA_mean_df(self,K):
        #Generate moving average of observations
        movingaveragedf = self.df.copy()
        return(pd.DataFrame(self.df).rolling(K,center = True)).mean()

    def MA_sd_df(self,K):
        #Generate moving average of standard deviations
        movingsddf = self.df.copy()
        return(pd.DataFrame(self.df).rolling(K,center=True).std())

    def MA_binomest_CI_plot(self,column= None, K=30,alpha=0.05):
        #Generates plot for a given column
        rollingbinomestimatesdf = self.cumprobdf

        rolling_binom_estimate = list(rollingbinomestimatesdf[column])

        CI_lower = []
        CI_upper = []

        for k in range(len(rolling_binom_estimate)):
            total = list(self.cumprobdf['NumOfArticles'])[k]
            successes = total*rolling_binom_estimate[k]
            lo, hi = clopper_pearson(successes, total)
            CI_lower.append(lo)
            CI_upper.append(hi)
        
        figure(figsize=(10, 10), dpi=160)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        #Plot binomestiamtes
        ax1.plot(self.df.index.values, rolling_binom_estimate, label = "line 1")
        #Plot upper and lower CI
        ax1.plot(self.df.index.values,CI_lower , label = "line 2")
        ax1.plot(self.df.index.values,CI_upper , label = "line 3")
        #Plot mean of our data
        ax2.bar(self.df.index.values, self.MA_mean_df(30)[column] ,width = 0.3)
        plt.show()
    
    def CI_of_estimates_plot(self):
        lot = []
        hit = []
        # For each of the final binomial estimates
        for prob in self.binomialestimates:
            successes = prob*self.cumprobdf['NumOfArticles'][-1]
            total = self.cumprobdf['NumOfArticles'][-1]
            lo, hi = clopper_pearson(successes, total)
            lot.append(lo)
            hit.append(hi)

        y=[]
        for i in range(len(self.binomialestimates)):
            y.append((lot[i],hit[i]))
        x = [i for i in range(len(self.binomialestimates))]
        plt.scatter(x,self.binomialestimates, s = 4)
        plt.plot((x,x),([i for (i,j) in y], [j for (i,j) in y]),c='black')
        plt.xlabel("Category")
        plt.ylabel('Probability')
        plt.show()
        #remember to define p first!

    #remember to define p first!

    def binplot2(self):
        columns = list(self.df)[0:-1]
        k = 0 
        column = columns
        for column in columns:
            #Set up plots with two axis
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            #Compute histogram with weights i.e. the pdf
            #weights = np.ones_like(self.df[column]) / list(self.cumprobdf["NumOfArticles"])[-1]
            labels, counts = np.unique(self.df[column], return_counts=True)
            counts = counts/list(self.cumprobdf["NumOfArticles"])[-1]
            ax1.bar(labels, counts, align='center',width = 0.4,alpha = 0.7)
            #ax1.gca().set_xticks(labels)
            #ax1.hist(self.df[column], weights=weights, align = 'center')

            #Compute the binomial pdf
            p_hat = np.mean(self.df[column])/list(self.df["NumOfArticles"])[-1]
            # n is fixed
            n = list(self.df["NumOfArticles"])[-1]
            # defining list of r values
            r_values = list(range(0,max(self.df[column])+1))
            # list of pmf values
            dist = [binom.pmf(r, n, p_hat) for r in r_values ]
            # plotting the graphs
            ax2.plot(r_values, dist,"r-")
            
            ax1.set_title(column)
            ax1.set_ylabel("Histogram Density")
            ax2.set_ylabel("Probability Density")
            ax1_ylims = ax1.axes.get_ylim()           # Find y-axis limits set by the plotter
            ax1_yratio = ax1_ylims[0] / ax1_ylims[1]  # Calculate ratio of lowest limit to highest limit

            ax2_ylims = ax2.axes.get_ylim()           # Find y-axis limits set by the plotter
            ax2_yratio = ax2_ylims[0] / ax2_ylims[1]  # Calculate ratio of lowest limit to highest limit


            # If the plot limits ratio of plot 1 is smaller than plot 2, the first data set has
            # a wider range range than the second data set. Calculate a new low limit for the
            # second data set to obtain a similar ratio to the first data set.
            # Else, do it the other way around

            if ax1_yratio < ax2_yratio: 
                ax2.set_ylim(bottom = ax2_ylims[1]*ax1_yratio)
            else:
                ax1.set_ylim(bottom = ax1_ylims[1]*ax2_yratio)

            plt.tight_layout()
            plt.show()


class MonteCarlo:
    def __init__(self,analysis):
        self.hi = 2
        self.df = analysis.df
        self.analysis = analysis
        self.SMCstates = {}

    # state space model - Mutation
    def StateSpace_iplus1(self,xt):
        return(xt*np.random.uniform(low = 0.9, high = 1.1))
        #return(Xt_1 *dnp.random.lognormal(mean=0.0, sigma=1.0)
        #return(Xt_1*np.random.exponential(scale = 1))

    #Conditional probability of Xt given Xt+1
    def backwardkernelfunc(self,xt,xt_1):
        return(scipy.stats.uniform.pdf(xt_1/xt,loc = 0.8,scale = 0.4))

    
    #Mutation
    def hiddenmodel_iplus1(self,observation,state):
        return(poisson.pmf(observation, mu=state))

    def general_filter(self,obs,N):
        ParticleStates = [0 for z in range(len(obs))]
        # N is the number of particles for filter
        # oba is our observation vector
        obs = np.array(obs)

        # Initial prior distribution for our unobservable quantity of interest
        initialstate = [np.mean(obs) for i in range(0,N)]
        X = initialstate
        ParticleStates[0] = initialstate
        #Store the final rate paramater
        rate =np.zeros(len(obs))

        for i in range(0,len(obs)):
            #Give the parameter for our state space model and updated rate 
            # having ovserved our data point Y[i]
            [X,X_est] = self.Filter(X,obs[i],N)
            ParticleStates[i] = X
            rate[i] = X_est
        return(rate,ParticleStates)

    def Filter(self,Particles,Yobs,N):
        Weights = [0 for i in range(N)]
        #For each particle in our filter

        #Draw a sample from our 'proposal distribution' i.e. for computing predictive step
        for i in range(0,N):
            #Mutate firstly
            
            Particles[i] = self.StateSpace_iplus1(Particles[i])
            #Compute weights by computing the current likelihood of a particle
            Weights[i] = self.hiddenmodel_iplus1(Yobs,Particles[i])

        #Normalise weights
        sumW = sum(Weights)
        for i in range(N):
            Weights[i] = Weights[i]/sumW
            
        #Resample from these weights    

        Particles = np.random.choice(Particles,replace = True, size = N, p = Weights)
        
        #Convert to state space vectors
        Particles_est = sum(Particles)/N
        return(Particles,Particles_est)

    def movingaveragemontecarlo(self,ID,K,N):
        p_est_dict = {}
        vals = self.df[ID]
        df_dummy = self.df.copy()
        [rate,ParticleStates] = self.general_filter(self.df[ID],N)
        self.SMCstates[ID] = ParticleStates
        
        #dfp.rolling(7, center=True).mean()
        hi = pd.DataFrame(vals)
        dude = pd.DataFrame(hi).rolling(K, center=True).mean()        
        
        df_dummy['Plottingrate'] = rate
        
        figure(figsize=(10, 20), dpi=320)
        fig, ax = plt.subplots()
        plt.plot(self.df.index.values,df_dummy['Plottingrate'], color = 'red')
        ax.bar(self.df.index.values, dude[ID] ,width = 0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b')) # Only plot the month
        plt.title(ID)
        plt.savefig('./Plots/SMC/'+ID)
        plt.show()


    def general_smoother_filter(self,obs,N,ParticleStates):
        SmoothedParticleStates = [0 for z in range(len(obs))]
        # N is the number of particles for filter
        # oba is our observation vector
        obs = np.array(obs)

        # Initial prior distribution for our unobservable quantity of interest
        initialstate = [np.mean(obs) for i in range(0,N)]
        X = initialstate
        X_est = np.mean(X)
        SmoothedParticleStates[0] = initialstate
        #Store the final rate paramater
        rate =np.zeros(len(obs))

        for t in range(1,len(obs)):
            #Give the parameter for our state space model and updated rate 
            # having ovserved our data point Y[i]
            [X,X_est] = self.SmoothFilter(ParticleStates,X_est,N,t)
            SmoothedParticleStates[t] = X
            rate[t] = X_est
        return(rate,SmoothedParticleStates)

    def SmoothFilter(self,ParticleStates,prevstate,N,time):
        Resampled_particles = [0 for i in range(N)]
        #For each particle in our filter

        #Draw a sample from our 'proposal distribution' i.e. for computing predictive step
        for i in range(0,N):
            states_of_i_particle_over_time = [ParticleStates[j][i] for j in range(0,time)]
        
        new_states = np.random.choice(states_of_i_particle_over_time.append(self.StateSpace_iplus1(prevstate)))
        new_states_est = np.mean(new_states)

        return(new_states,new_states_est)    

    def smoothing(self,obs,N):
        X0 = [np.mean(obs) for i in range(N)]
        W0 = [self.hiddenmodel_iplus1(obs[0],X0[i]) for i in range(N)]
        sumW = sum(W0)
        for i in range(N):
            W0[i] = W0[i]/sumW
        
        # for t in range(1,len(obs)):
    
    def saveStates(self):
        # load csv module
        import csv

        # define a dictionary with key value pairs
        dict = self.SMCstates

        # open file for writing, "w" is writing
        w = csv.writer(open("States.csv", "w"))

        # loop over dictionary keys and values
        for key, val in dict.items():

            # write every key and value to file
            w.writerow([key, val])

    def smc_smoothing(self, num_particles, observations, transition_kernel, observation_kernel, num_steps):
        # Initialize the particles and their weights
        particles = np.zeros((num_steps, num_particles))
        particles[0, :] = np.random.rand(num_particles)
        weights = np.ones((num_steps, num_particles)) / num_particles

        for step in range(1, num_steps):
            # Propagate the particles through the transition kernel
            for i in range(num_particles):
                particles[step, i] = transition_kernel(particles[step-1, i])

            # Calculate the weights based on the observations and observation kernel
            weights[step, :] = observation_kernel(observations[step], particles[step, :])
            weights[step, :] = weights[step, :] / sum(weights[step, :])

            # Resample the particles according to their weights
            indices = np.random.choice(range(num_particles), size=num_particles, p=weights[step, :])
            particles[step, :] = particles[step, indices]
            weights[step, :] = np.ones(num_particles) / num_particles

        # Calculate the smoothed particles and weights
        smoothed_particles = np.zeros((num_steps, num_particles))
        smoothed_weights = np.zeros((num_steps, num_particles))
        for step in range(num_steps-1, -1, -1):
            # Calculate the smoothed weights
            print(step+1)
            smoothed_weights[step, :] = (weights[step, :] *
                                        np.dot(transition_kernel(particles[step, :]), smoothed_weights[step+1, :]))
            smoothed_weights[step, :] = smoothed_weights[step, :] / sum(smoothed_weights[step, :])

            # Calculate the smoothed particles
            smoothed_particles[step, :] = np.dot(smoothed_weights[step, :], particles[step, :])
        return smoothed_particles, smoothed_weights

    def smc_smoother(self,num_particles, observations, transition_kernel, observation_kernel, num_steps):
        # Initialize the particles and their weights
        particles = np.zeros((num_steps, num_particles))
        particles[0, :] = np.random.rand(num_particles)
        weights = np.ones((num_steps, num_particles)) / num_particles

        for step in range(1, num_steps):
            # Propagate the particles through the transition kernel
            for i in range(num_particles):
                particles[step, i] = transition_kernel(particles[step-1, i])

            # Calculate the weights based on the observations and observation kernel
            for i in range(0, num_particles):
                weights[step, i] = observation_kernel(observations[step], particles[step, i])
            weights[step, :] = weights[step, :] / sum(weights[step, :])

            # Resample the particles according to their weights
            indices = np.random.choice(range(num_particles), size=num_particles, p=weights[step, :])
            particles[step, :] = particles[step, indices]
            weights[step, :] = np.ones(num_particles) / num_particles

        # The smoother estimate at each time step is calculated as a weighted average of particles at all future time steps
        smoothed_estimate = np.zeros((num_steps, num_particles))
        smoothed_estimate[-1, :] = particles[-1, :]
        for step in range(num_steps-2, -1, -1):
            # Calculate the backward kernel, which is the conditional probability of a particle at time t given a particle at time t+1
            backward_kernel = np.zeros((num_particles, num_particles))
            for i in range(num_particles):
                for j in range(num_particles):
                    backward_kernel[i, j] = self.backwardkernelfunc(particles[step+1, j], particles[step, i])

            # Calculate the smoothed weights as the product of the forward and backward kernels
            smoothed_weights = np.zeros((num_particles, num_particles))
            for i in range(num_particles):
                for j in range(num_particles):
                    smoothed_weights[i, j] = weights[step, i] * backward_kernel[i, j] / np.dot(backward_kernel[:, j], weights[step, :])

            # The smoothed estimate at time t is a weighted average of the particles at time t+1
            smoothed_estimate[step, :] = np.dot(smoothed_weights, smoothed_estimate[step+1, :])

        return smoothed_estimate

global glob
def montecarlo(X0,W0, obs):
    X1 = np.multiply(np.random.uniform(low = 0.9, high = 1.1,size = len(X0)),X0) # Evolve according to our state model with uniform
    W1 = [scipy.stats.poisson.pmf(k = obs, mu = x) for x in X1]
    # print("Obs: ", obs)
    # plt.hist(W1)
    # plt.show()
    CDF = []
    CDF[0] = 0
    for i in range(1,len(X1)):
        CDF[i] = CDF[i-1] + W1[i]
    u0 = np.random.uniform(low = 0, high = 1/len(X1))
    for j in range(len(X1)):
        u_j = u0 + (1/len(X1))

    #X_resampled = np.random.choice(a = X1,p = W1, replace = True,size = len(X1))
    
    # X1 = np.array(X1)
    # idx = np.random.randint(0,len(X1),size=(len(X1)))
    # X_resampled = [X1[i] for i in idx]
    # W = [1/len(X1) for i in range(len(X1))]
    return(X1_resampled,W1)


def main():
    #Apply SMC:
    numpart = 3 # Number of Particles
    X0 = np.random.uniform(low = 0,high = 1,size=numpart)
    W0 = [scipy.stats.poisson.pmf(k = 0.5, mu = particle) for particle in X0 ]
    analysis3 = Model()
    obs_arr = list(analysis3.df['Homophobia'])

    HistoryofParticles = [0 for i in range(len(obs_arr))]
    Historyofrateestimates = [0 for i in range(len(obs_arr))]

    num_steps = len(obs_arr)
    for i in range(num_steps):
        [X1,W1] = montecarlo(X0,W0,obs_arr[i]) #Aply SMC
        HistoryofParticles[i] = X1
        #Compute estimate from posterior
        Historyofrateestimates[i] = sum(X1)/numpart

        X0=X1
        W0=W1
    figure(1)
    plt.plot(Historyofrateestimates)
    plt.plot(obs_arr)
    plt.show()
    figure(2)
    
    plt.show()
main()