# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 19:08:34 2023
@author: Adam Turner

Code modified from ​Wörner, T.P. et al. (2020) 
    ‘Resolving heterogeneous macromolecular assemblies by Orbitrap-based single-particle charge detection mass spectrometry’,
    Nature Methods,
    17(4),
    pp. 395–398.
    Available at: https://doi.org/10.1038/s41592-020-0770-7. 
    
Any mention of "original" or "default" in the code or comments refers to the data supplied by the above paper for use with the unedited code
    
"""

#Importing of required packages
import pyteomics.mzxml
import numpy as np
import pandas as pd
import math

#Initialisation of preprocess class
class Preprocess():
    '''
    Class for pre-processing single particle data. 
    
    Instantiation of an object of class Preprocess allows a file name, injection time, resolution and number of full width at half
    maximum to be used for filtering of adjacent peaks. The only changes made to the class were to allow user defined values to be
    used to instantiate objects of class Preprocess,
    all other code was commented but left as it was in the source code
    '''
    def __init__(self, file, it=1, res=220000, times_FWHM=5):
        self.file = file
        self.it = it
        self.res = res
        self.times_FWHM = times_FWHM
        self.__read_peaks()

    def __read_peaks(self):
        '''
        Function reads mzXML file and normalizes the intensities by their injection time and saves centroids
        with the scan number in a DataFrame.
        '''
        data = pyteomics.mzxml.read(self.file) #Loads data from predefined results file into a dataFrame, allowing data to be accessed
        scans = [i for i in data.iterfind("*") if "basePeakIntensity" in i] #Extracts scan information from dataframe containing results file contents
        
        
        mz = [s["m/z array"] for s in scans]#Extracts mz values from results
        intensity = [s["intensity array"] for s in scans]#Extracts intensity from results
        scans = [[int(s["num"])] * s["peaksCount"] for s in scans] #Assigns an index value to each scan in a list

        intensity = np.concatenate(intensity) * self.it #Intensity is multiplied by injection time so it can be used to assign charge
        
        scans = np.concatenate(scans) #Don't think this does anything either, delete
        self.normalized = pd.DataFrame({"m/z" : mz, "Intensity" : intensity, "Scan" : scans}) #Produces a dataframe of each of the extracted arrays combined
        self.filtered = self.normalized
        
       
        
    def filter_data(self, int_threshold, mz_threshold="auto"):
        '''
        Function filtering centroids data for noise and dephased peaks.
        '''
        self.filtered = self.filtered[self.filtered["Intensity"] >= int_threshold] #Removes data points below a user defined intensity threshold
        select = [self.__remove_adjacent(df["m/z"], mz_threshold) for s, df in self.filtered.groupby(["Scan"])] #Removes adjacent datapoints above a user defined m/z threshold using __remove_adjacent
        self.filtered = self.filtered[np.concatenate(select)] #Combines filtered data
        
    def select_mz(self, mi, ma):
        '''
        Function filtering centroids based on m/z values.
        '''
        self.filtered = self.filtered[self.filtered["m/z"].between(mi, ma)] #Selects only data points with m/z values between two user defined points
    
    def select_int(self, mi, ma):
        '''
        Function filtering centroids based on intensity values.
        '''
        self.filtered = self.filtered[self.filtered["Intensity"].between(mi, ma)]#Selects only data points with intensity values between two user defined points
            
    def __remove_adjacent(self, mz, mz_threshold):
        '''
        Function reads list of m/z centroids and filters for adjacent centroids
        below a certain m/z threshold. The m/z threshold can either be set to a fixed value or
        determined automatically based on the FWHM at the instruments resolution limit.
        General procedure is:
        1. Input MZ_centroid_list of length n, MZ_threshold
            [1,2,3,3.5,4], 1
        2. Calculate difference between items yielding list with n-1 items
            [1,2,3,3.5,4] -> [1,1,0.5,0.5] 
        3. Apply threshold (T=True, F=False)
            [1,1,0.5,0.5] >= 1 -> [T, T, F, F]
        4. In order to get a Boolean list for excluding both peaks which distance is below the threshold we 
        generate two Boolean lists with the length n by duplicating the first and the last item, respectively, and
        perform a Boolean conjunction.
            [T, T, F, F] + [F] -> [T, T, F, F, F]
            [T] + [T, T, F, F] -> [T, T, T, F, F]
            [T, T, F, F, F] AND [T, T, T, F, F] -> [T, T, F, F, F]
        5. This Boolean list will be used as selector for the provided list of centroids.
            [T, T, F, F, F] -> [1,2,3,3.5,4] -> [1,2]
        
        If the list of centroids contains only one item the function will raise an indexing error in step 4 
        and the function will return a [False] 
        
        '''
        if mz_threshold == "auto": #Code below runs if the value of mz_threshold is "auto"
            try:
                mz_array = np.array(mz)                
                diff_b = np.diff(mz_array) >= (self.__get_FWHM(mz_array[:-1], self.res) * self.times_FWHM)
                diff_b = np.append(diff_b[0], diff_b) & np.append(diff_b, diff_b[-1])
            except:
                diff_b = np.array([False] * len(mz_array))
                
        else: #Code below runs if the value of mz_threshold is not "auto"
            try:
                mz_array = np.array(mz)
                diff_b = np.diff(mz_array) >= mz_threshold
                diff_b = np.append(diff_b[0], diff_b) & np.append(diff_b, diff_b[-1])
            except:
                diff_b = np.array([False] * len(mz_array))
                
        return diff_b
    
    def __get_FWHM(self, mz, res):
        '''
        Calculates FWHM of a peak based on instrument resolution at 400 m/z, assuming R ~ m/z^-0.5
        '''
        A = res / 400**-.5 
        R = A*mz**-0.5
        return mz/R
    
    def write_peaks(self, label="_processed"):
        '''
        Writes filtered centroids as CSV list.
        This allows processing to be done from the filtered list, rather than having to filter the data every time analysis is required
        '''
        self.filtered.to_csv(self.file + label + ".csv", index=False)
        
'''
Python script for the processing of single-ion data, using the class Preprocess
The majority of this code has been modified or rewritten to allow processing of user data
Original script was only able to process the original author's data
'''  

#Imports necessary packages
import matplotlib.pyplot as plt
import matplotlib.colors as color



custom_map = color.LinearSegmentedColormap.from_list("map", ["white", "red", "darkred"]) #Sets up custom colouring for the figures produced later in the script
filename = input("Input 1 for default settings or filename to input custom settings"+ "\n" +">>>") #Allows the original data and settings to be used. 
                                                                                                   #Primarily used to ensure original results could be replicated using new code


if filename == "1": #If default settings are selected, original values are used for processing
        default = True
        filename = "AaLS.mzXML"
        injection_time = 1
        resolution = 220000
        FWHM = 5
        min_intensity = 50
        
else: #If anything other than 1 is selected, user is prompted to input required settings used in experiment
    default = False
    injection_time = float(input("input max injection time used (in seconds)"+ "\n" +">>>"))
    resolution = int(input("input resolution used (default is 220000)"+ "\n" +">>>"))
    FWHM = int(input("input times FWHM to be used for removing adjacent peaks (default 5)"+ "\n" +">>>"))
    min_intensity = int(input("input minimum intensity for peak filtering (default 50)"+ "\n" +">>>"))


preprocessed_data = Preprocess("C:/MyFiles/University/Year 4/Sem 2/Diss/Code/data/"+filename, injection_time, resolution, FWHM) #Preprocess object instantiated using user defined settings


unfiltered = preprocessed_data.filtered.copy() #Copy of unfiltered data made for comparison to filtered data

preprocessed_data.filter_data(min_intensity) #Datapoints below minimum intensity for peak filtering, filtered out

print(preprocessed_data.filtered.agg( #Prints summary values of intensity and m/z to inform user, and help them decide which values should be filtered
    {
         "Intensity": ["min", "max", "median", "mean", "std"],
         "m/z": ["min", "max", "median", "std"]
    }))

'''
Plots filtered and unfiltered data points of intensity vs m/z to visualize the effect filtering has on the overall data
'''
#Finds minimum and maximum values to be used for bin sizes of filtered vs unfiltered figure
bin_mz_min = math.floor(preprocessed_data.filtered.min()[0]) 
bin_mz_max = math.ceil(preprocessed_data.filtered.max()[0])
bin_intens_min = math.floor(preprocessed_data.filtered.min()[1])
bin_intens_max = math.ceil(preprocessed_data.filtered.max()[1])


bins = np.arange(bin_mz_min, bin_mz_max, 20), np.arange(bin_intens_min, bin_intens_max, 10) #Creates bins using above defined min and max values

fix, ax = plt.subplots(ncols=2, figsize=(8,3)) #Sets layout of figures

#Adds data to the two 2d histograms
ax[0].hist2d(unfiltered["m/z"], unfiltered["Intensity"], bins=bins, cmin=1)
ax[1].hist2d(preprocessed_data.filtered["m/z"], preprocessed_data.filtered["Intensity"], bins=bins, cmin=1)

#Adds labels to histograms
ax[0].set_title("before filtering")
ax[0].set_ylabel("Intensity")
ax[0].set_xlabel("m/z")
ax[1].set_title("after filtering")
ax[1].set_xlabel("m/z")
plt.tight_layout()

plt.savefig("C:/MyFiles/University/Year 4/Sem 2/Diss/Code/Figures/Filtering_effect_"+filename+".png") #Saves filtered vs unfiltered histogram for future use

'''
Allows a user to define the regions of interest or uses default values if original data is being processed
'''
print("Defining regions of interest")
if default == True:
    min_int = 1500
    max_int = 2500
    min_mz = 19000
    max_mz = 21500
else:
    min_int = int(input("input minimum intensity"+ "\n" +">>>"))
    max_int = int(input("input maximum intensity"+ "\n" +">>>"))
    min_mz = int(input("input minimum m/z"+ "\n" +">>>"))
    max_mz = int(input("input maximum m/z"+ "\n" +">>>"))

#Saving regions of interest to .csv file
preprocessed_data.select_int(min_int, max_int)
preprocessed_data.select_mz(min_mz, max_mz)
preprocessed_data.write_peaks()


#Reading filtered data from .csv
#This allows the script to be run from this point if the data has already been filtered
filtered_data = pd.read_csv("C:/MyFiles/University/Year 4/Sem 2/Diss/Code/data/"+filename+"_processed-bce.csv")
filtered_data["z"] = filtered_data["Intensity"] / 12.521 #Regression relationship between charge and intensity used to determine charge (or 'z') of each data pointn
filtered_data["mass"] = filtered_data["m/z"] * filtered_data["z"] - filtered_data["z"] #Calculated charge values used to assign mass to each data point from m/z reading


fig, ax = plt.subplots(figsize=(6,4)) #Layout for density histogram of mass
bins = np.arange(3e5, 4e5, 1e4) #Bins of size 1e4 between 3e5 and 4e5 produced. These values may require adjusting for use with other data

filtered_data["mass"].plot(kind="hist", alpha=0.75,density=True,bins=bins, color="lightgray" ) #Plots a mass density histogram
filtered_data["mass"].plot(kind="kde", color="black") #Plots a kernel density estimate curve of plotted mass density bins

#Formatting and text of axes
ax.set_xlabel("Mass (100kDa)")
ax.set_ylabel("Density")
ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
ax.ticklabel_format(axis='y', style='plain', scilimits=(0,0))
ax.set_title("Mass histogram for GDH\n 10 kDa bins")


x = ax.lines[0].get_xdata() # Get the x-axis data of the distribution
y = ax.lines[0].get_ydata() # Get the y-axis data of the distribution
maxid = np.argmax(y) # The id of the peak (maximum of y data)
plt.plot(x[maxid],y[maxid],'x', ms=5, color="black") #Plots a black cross of size 5 at the highest point of the kernel density estimate curve
plt.text(x[maxid],y[maxid], "%.2f kDa"%(x[maxid]/1e3)) #Addition of label of mass assignment
plt.savefig("C:/MyFiles/University/Year 4/Sem 2/Diss/Code/Figures/MassHistogram_"+filename+"1.png", bbox_inches='tight') #Saves mass density histogram to allow future use
plt.show() #Shows figure