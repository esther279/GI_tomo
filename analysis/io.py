import os, glob, time, sys
import numpy as np
import matplotlib.pyplot as plt


def read_peak_list(filename):
    '''
    ===== Example input file, ./BTBT_peaks.txt        
    #center, size, peak
    [[575, 252], [60, 10], 'sum002'],
    # 11L
    [[525, 548], [180, 10], 'sum11L'],    
    
    ===== Output peak_list1     
    [[[575, 252], [60, 10], 'sum002'],
     [[525, 548], [180, 10], 'sum11L'],
     [[608, 607], [20, 10], 'sum02L'],
     [[589, 684], [58, 6], 'sum12L'],
     [[300, 744], [40, 15], 'sum20L'],
     [[250, 782], [40, 15], 'sum21L'],
     [[588, 858], [40, 15], 'sum22L'],
     [[385, 631], [12, 12], 'sumSi'],
     [[560, 200], [30, 30], 'sumBKG0']]
    '''
    
    peak_list1 = []
    with open(filename, "r") as peaks:
        lines = peaks.readlines()
        for l in lines:
            temp = [] # each line as a list
            if l[0][0] is not "#": 
                l = l.replace("\n", "")
                x = l[l.find('[[')+2:l.find(']')].split(" ")
                x = [int(a.replace(",","")) for a in x]
                l = l[l.find('],')+2:]
                temp.append(x)            
                
                roi = l[l.find('[')+1:l.find(']')].split(" ")
                roi = [int(a.replace(",","")) for a in roi]
                l = l[l.find(']'):]
                temp.append(roi)
                
                index = l[l.find('\'')+1:l.find('\']')]
                temp.append(index)
                peak_list1.append(temp)

    return peak_list1
 
    
    
    
