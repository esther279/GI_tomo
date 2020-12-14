import numpy as np
import pandas as pd

def get_angles_BTBT():
    x = {}; 
    jj=0
    x[jj] = pd.DataFrame([[-53.87, 'sum11L']], columns=['angle','peak']); jj = jj+1
    x[jj] = pd.DataFrame([[-53.87+107.75, 'sum11L']], columns=['angle','peak']); jj = jj+1
    x[jj] = pd.DataFrame([[-53.87+180, 'sum11L']], columns=['angle','peak']); jj = jj+1
    x[jj] = pd.DataFrame([[-53.87+107.75+180, 'sum11L']], columns=['angle','peak']); jj = jj+1
    
    x[jj] = pd.DataFrame([[-53.87-12.8, 'sum11Lb']], columns=['angle','peak']); jj = jj+1        
    x[jj] = pd.DataFrame([[-53.87-12.8+107.75, 'sum11Lb']], columns=['angle','peak']); jj = jj+1
    x[jj] = pd.DataFrame([[-53.87-12.8+180, 'sum11Lb']], columns=['angle','peak']); jj = jj+1        
    x[jj] = pd.DataFrame([[-53.87-12.8+107.75+180, 'sum11Lb']], columns=['angle','peak']); jj = jj+1
    
    
    x[jj] = pd.DataFrame([[0, 'sum02L']], columns=['angle','peak']); jj = jj+1
    x[jj] = pd.DataFrame([[180, 'sum02L']], columns=['angle','peak']); jj = jj+1
    x[jj] = pd.DataFrame([[0-13.35, 'sum02Lb']], columns=['angle','peak']); jj = jj+1
    x[jj] = pd.DataFrame([[180-13.35, 'sum02Lb']], columns=['angle','peak']); jj = jj+1                
    
    offset = 21
    x[jj] = pd.DataFrame([[-53.87+offset, 'sum12L']], columns=['angle','peak']); jj = jj+1     
    x[jj] = pd.DataFrame([[-53.87+offset+68.8, 'sum12L']], columns=['angle','peak']); jj = jj+1
    x[jj] = pd.DataFrame([[-53.87+offset+180, 'sum12L']], columns=['angle','peak']); jj = jj+1     
    x[jj] = pd.DataFrame([[-53.87+offset+68.8+180, 'sum12L']], columns=['angle','peak']); jj = jj+1
    
    alpha2 = -16.31
    x[jj] = pd.DataFrame([[-53.87+offset+alpha2, 'sum12Lb']], columns=['angle','peak']); jj = jj+1     
    x[jj] = pd.DataFrame([[-53.87+offset+68.8+alpha2, 'sum12Lb']], columns=['angle','peak']); jj = jj+1
    x[jj] = pd.DataFrame([[-53.87+offset+180+alpha2, 'sum12Lb']], columns=['angle','peak']); jj = jj+1     
    x[jj] = pd.DataFrame([[-53.87+19.5+68.8+180+alpha2, 'sum12Lb']], columns=['angle','peak']); jj = jj+1
    
    x[jj] = pd.DataFrame([[-53.87-29, 'sum20L']], columns=['angle','peak']); jj = jj+1        
    x[jj] = pd.DataFrame([[-53.87-29+180, 'sum20L']], columns=['angle','peak']); jj = jj+1
    alpha2 = -27.5
    x[jj] = pd.DataFrame([[-53.87-29+alpha2, 'sum20Lb']], columns=['angle','peak']); jj = jj+1        
    x[jj] = pd.DataFrame([[-53.87-29+180+alpha2, 'sum20Lb']], columns=['angle','peak']); jj = jj+1
    
    x[jj] = pd.DataFrame([[-53.87-7.5, 'sum21L']], columns=['angle','peak']); jj = jj+1        
    x[jj] = pd.DataFrame([[-53.87-7.5+139, 'sum21L']], columns=['angle','peak']); jj = jj+1
    x[jj] = pd.DataFrame([[-53.87-7.5+180, 'sum21L']], columns=['angle','peak']); jj = jj+1        
    x[jj] = pd.DataFrame([[-53.87-7.5+139+180, 'sum21L']], columns=['angle','peak']); jj = jj+1
     
    alpha2 = -30.5
    x[jj] = pd.DataFrame([[-53.87-7.5+alpha2, 'sum21Lb']], columns=['angle','peak']); jj = jj+1        
    x[jj] = pd.DataFrame([[-53.87-7.5+139+alpha2, 'sum21Lb']], columns=['angle','peak']); jj = jj+1
    x[jj] = pd.DataFrame([[-53.87-7.5+180+alpha2, 'sum21Lb']], columns=['angle','peak']); jj = jj+1        
    x[jj] = pd.DataFrame([[-53.87-7.5+139+180+alpha2, 'sum21Lb']], columns=['angle','peak']); jj = jj+1 
    
    return x



