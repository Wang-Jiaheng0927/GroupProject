import time

def some_long_running_task():
    total_iterations = 1000
    for i in range(total_iterations):
        # 模拟长时间运行的任务
        time.sleep(0.01)
        
        # 计算进度
        progress = (i + 1) / total_iterations * 100
        
        # 打印进度条
        print(f'\rProgress: [{"=" * int(progress / 2):50}] {progress:.2f}%', end='', flush=True)
    
    print("\nTask completed!")

# 运行示例任务
some_long_running_task()

# import modules
import numpy as np # import scientific library
from get_regression_coefs import get_regression_coefs # import our function to get GDP trend
import pandas as pd # import library for data analysis

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns # for nicer plots (not essential)
sns.set_theme('talk', style = 'white')

# Load dataset
Data = pd.read_excel('pwt100.xlsx', sheet_name = 'Data', header = 0)
Data = Data.loc[:, ("country", "year", "rgdpna", "pop")]
Data["rgdpna_pc"] = Data["rgdpna"] / Data["pop"]
print(Data)

# Create a class and process data
class GDP_data:
    def __init__(self, Data, country, GDP_type, year_min, year_max):
        # Initialize attributes
        self.country = country
        self.GDP_type = GDP_type
        self.year_min = year_min
        self.year_max = year_max
        
        # Clean data
        self.data = Data.loc[Data["country"] == country, ("year", GDP_type)]
        self.data.reset_index(drop=True)#inplace=True
        
        self.Y = self.data.loc[np.logical_and(self.data["year"] <= year_max, self.data["year"] >= year_min), GDP_type]
        self.y = np.log(self.Y)
        self.data = self.data[self.data["year"] >= year_min]
        
        self.T = len(self.Y) # sample size used for regression
        self.T_all = self.data["year"].max() - (year_min - 1) # number of all years in the data after ymin

    # Compute different trend specifications
    ## Additive, Linear Model
    def gen_add_lin(self):
        x1 = np.empty(self.T)
        x2 = np.empty(self.T)
        for t in range(self.T):
            x1[t] = 1.
            x2[t] = t + 1 # recall that Python starts indexing at 0
        
        a_add_lin, b_add_lin = get_regression_coefs(self.Y, x1, x2)

        Yhat_add_lin = np.empty(self.T_all)
        
        for t in range(self.T_all):
            Yhat_add_lin[t] = a_add_lin + b_add_lin * (t + 1)
        
        print(Yhat_add_lin, len(Yhat_add_lin))
        # Convert into log-units
        yhat_add_lin = np.log(Yhat_add_lin)
        print(yhat_add_lin, len(yhat_add_lin))

        return yhat_add_lin

    ## Additive, Quadratic Model
    def gen_add_quad(self):
        x1, x2, x3 = np.empty(self.T), np.empty(self.T), np.empty(self.T)
        for t in range(self.T):
            x1[t] = 1.
            x2[t] = t + 1
            x3[t] = (t + 1)**2
        
        a_add_quad, b1_add_quad, b2_add_quad = get_regression_coefs(self.Y, x1, x2, x3)
        # Initialise predicted values yhat
        Yhat_add_quad = np.empty(self.T_all)
        # Create loop to compute trend for all years
        for t in range(self.T_all):
            Yhat_add_quad[t] = a_add_quad + b1_add_quad * (t + 1) + b2_add_quad * (t + 1)**2 
        # Convert into log-units
        yhat_add_quad = np.log(Yhat_add_quad)

        return yhat_add_quad
    
    # Exponential Model
    ## Exponential, Linear Model
    def gen_exp_lin(self):
        # 3.2.1) Exponential, Linear Model
        x1, x2 = np.empty(self.T), np.empty(self.T) # initialise an empty vector for the first regressor

        for t in range(self.T):
            x1[t] = 1.
            x2[t] = t + 1 # recall that Python starts indexing at 0

        a_exp_lin, b_exp_lin = get_regression_coefs(self.y, x1, x2)

        # Initialise predicted values yhat
        yhat_exp_lin = np.empty(self.T_all)

        # Create loop to compute trend for all years
        for t in range(self.T_all):
            yhat_exp_lin[t] = a_exp_lin + b_exp_lin * (t + 1)
        
        return yhat_exp_lin
    
    ## Exponential, Quadratic model
    def gen_exp_quad(self):
        # 3.2.2) Exponential, Quadratic model
        # Repeat above for quadratic specification
        x1, x2, x3 = np.empty(self.T), np.empty(self.T), np.empty(self.T) # initialise an empty vector for the first regressor

        for t in range(self.T):
            x1[t] = 1.
            x2[t] = t + 1
            x3[t] = (t + 1)**2

        a_exp_quad, b1_exp_quad, b2_exp_quad = get_regression_coefs(self.y, x1, x2, x3)

        # Initialise predicted values yhat
        yhat_exp_quad = np.empty(self.T_all)

        # Create loop to compute trend for all years
        for t in range(self.T_all):
            yhat_exp_quad[t] = a_exp_quad + b1_exp_quad * (t + 1) + b2_exp_quad * (t + 1)**2
        
        return yhat_exp_quad
    
    # Plot by additive and exponential
    def gen_plot(self, yhat_add_lin, yhat_add_quad, yhat_exp_lin, yhat_exp_quad):
        lw = 4

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
        ax1.plot(self.data['year'], np.log(self.data[self.GDP_type]), linewidth = lw)
        ax1.plot(self.data['year'], yhat_add_lin, linewidth = lw, linestyle = 'dashed', label = 'Linear fit')
        ax1.plot(self.data['year'], yhat_add_quad, linewidth = lw, linestyle = 'dotted', label = 'Quadratic fit')
        ax1.set_ylabel(r'$\log(Y_t)$')
        ax1.set_title(f"{self.country} - Additive Model")
        ax1.legend()

        ax2.plot(self.data['year'], np.log(self.data[self.GDP_type]), linewidth = lw)
        ax2.plot(self.data['year'], yhat_exp_lin, linewidth = lw, linestyle = 'dashed', label = 'Linear fit')
        ax2.plot(self.data['year'], yhat_exp_quad, linewidth = lw, linestyle = 'dotted', label = 'Quadratic fit')
        ax2.set_ylabel(r'$\log(Y_t)$')
        ax2.set_title(f'{self.country} - Exponential Model')
        ax2.legend()

        plt.show()
        return fig


# Regression of real GDP of India between 1950 - 2019 (all years in the file)
India_GDPna = GDP_data(Data,'India','rgdpna', 1990, 2019) 
yhat_add_lin = India_GDPna.gen_add_lin()
yhat_add_quad = India_GDPna.gen_add_quad()
yhat_exp_lin = India_GDPna.gen_exp_lin()
yhat_exp_quad = India_GDPna.gen_exp_quad()
India_GDPna.gen_plot(yhat_add_lin, yhat_add_quad, yhat_exp_lin, yhat_exp_quad)

# Regression of real GDP per capita of India between 1950 - 2019 (all years in the file)
India_GDPpc = GDP_data(Data,'India','rgdpna_pc', 1990, 2019)
yhat_add_lin = India_GDPpc.gen_add_lin()
yhat_add_quad = India_GDPpc.gen_add_quad()
yhat_exp_lin = India_GDPpc.gen_exp_lin()
yhat_exp_quad = India_GDPpc.gen_exp_quad()
India_GDPpc.gen_plot(yhat_add_lin, yhat_add_quad, yhat_exp_lin, yhat_exp_quad)

# Regression of real GDP of Singapore between 1960 - 2019 (all years in the file)
Singapore_GDPna = GDP_data(Data,'Singapore','rgdpna', 1980, 2019)
yhat_add_lin = Singapore_GDPna.gen_add_lin()
yhat_add_quad = Singapore_GDPna.gen_add_quad()
yhat_exp_lin = Singapore_GDPna.gen_exp_lin()
yhat_exp_quad = Singapore_GDPna.gen_exp_quad()
Singapore_GDPna.gen_plot(yhat_add_lin, yhat_add_quad, yhat_exp_lin, yhat_exp_quad)

# Regression of real GDP per capita of Singapore between 1960 - 2019 (all years in the file)
Singapore_GDPpc = GDP_data(Data,'Singapore','rgdpna_pc', 1980, 2019)
yhat_add_lin = Singapore_GDPpc.gen_add_lin()
yhat_add_quad = Singapore_GDPpc.gen_add_quad()
yhat_exp_lin = Singapore_GDPpc.gen_exp_lin()
yhat_exp_quad = Singapore_GDPpc.gen_exp_quad()
Singapore_GDPpc.gen_plot(yhat_add_lin, yhat_add_quad, yhat_exp_lin, yhat_exp_quad)

# Example used in assignment 8
"""
# Regression of real GDP of Belgium between 1955 - 2006
Belgium_GDPna = GDP_data(Data,'Belgium','rgdpna', 1955, 2006)
yhat_add_lin = Belgium_GDPna.gen_add_lin()
yhat_add_quad = Belgium_GDPna.gen_add_quad()
yhat_exp_lin = Belgium_GDPna.gen_exp_lin()
yhat_exp_quad = Belgium_GDPna.gen_exp_quad()
Belgium_GDPna.gen_plot(yhat_add_lin, yhat_add_quad, yhat_exp_lin, yhat_exp_quad)
"""