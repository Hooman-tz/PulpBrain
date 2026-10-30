"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Raman spectra (CSV) input script
%
% Ashton Christy
% 25 Apr 2018
%
% Hooman Tavakolizadeh
% May 2023
% 
% Ported to Python 2.7 on 9 Nov 2018
% Ported to Python 3.7 on 18 Mar 2019
% 
% Version 1.20B
% Edited 6 Jun 2022 - Enlighen 2.2.7 update, raw Raman for SQL
% Updated features -
%
% Version 1.21 -- Jun 2023
% Timestamp error resolved - Replications resolved - MySQL.Connector 
% Version for Windows
% 
% July 4 2023
% GUI: Append new data and select samples by date and create new table for prediction only
%
% 
%
% Version 6
% Feb 2023
% 
% Data pan in the main window
% Smaller Message Window
% Report Metrics
%
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

# Load packages
import os, csv, numpy as np, pywt, fnmatch, datetime as dt, time, re
import glob

# Resources (Models/, sql/, assets/) are located relative to this file, so the app
# can be started from any directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def asset(name):
    """Path of a logo/icon file in assets/."""
    return os.path.join(BASE_DIR, 'assets', name)


def _strtobool(value):
    """Replacement for distutils.util.strtobool (distutils was removed in Python 3.12)."""
    value = str(value).strip().lower()
    if value in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    if value in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    raise ValueError("invalid truth value %r" % (value,))


import pandas as pd

from sys import exit
import mysql.connector
from mysql.connector import Error
from datetime import date
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from joblib import load
from scipy.signal import savgol_filter
import configparser

# Database credentials come from config.ini (never committed), with optional
# PULPBRAIN_DB_* environment variables taking precedence.
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
_cfg = configparser.ConfigParser()
_cfg.read(_cfg_path)
_db = _cfg['database'] if _cfg.has_section('database') else {}


def _db_setting(key):
    value = os.environ.get('PULPBRAIN_DB_' + key.upper(), _db.get(key))
    if value is None:
        exit("Missing database setting '%s'. Copy config.example.ini to config.ini "
             "and fill it in (expected at %s), or set PULPBRAIN_DB_%s." % (key, _cfg_path, key.upper()))
    return value


DB_HOST = _db_setting('host')
DB_USER = _db_setting('user')
DB_PASSWORD = _db_setting('password')
DB_NAME = _db_setting('database')

plt.rcParams['figure.dpi'] = 300


targets_list = ['Caliper','Bulk','Density','Air_resistance','Smoothness','Brightness','Yellowness','Opacity','Scattering','Absorption','L*','a*','b*','CSF','Tear','Burst','BL','Stretch','Stiffness','TEA','TensileIndex', 'ZSBLdry', 'ZSBLwet']#, 'Viscosity','WRV']



#%%
    
# Differentiate between spectrometers
def spectrometer(name):
    if "WP-" in name:
        spect = 1
        sn = "WP785"
    if "S785LC" in name:
        spect = 2
        sn = "INNO"
    return spect, sn
    
# Convert excel dates to useable format
def convert_xldate_to_datetime(xldate, datemode):
    # datemode: 0 for 1900-based, 1 for 1904-based
    if xldate == 0:
        xldate = 2
    return (
        dt.datetime(1899, 12, 30)
        + dt.timedelta(days=xldate + 1462 * datemode)
        )
    
"""
######################################################

                Loading Raman Files

######################################################
"""  

    
def append_data(datadir, pe_csvf):
        
     while True:   
        try:
            t0 = time.time()
            # Load data
            files_loaded = []
            raw_data = []
            raw_data_line = []
            spec_stats = []
            valid_time_format = "%Y-%m-%d %H:%M:%S"
            barrelwise_file_time_stamp_format = r'\d{6}-\d{6}'
            
            
            folderlist = [dp for dp, dn, fn in os.walk(os.path.expanduser(datadir))]
            folderlist.sort()
            for folder in folderlist:
                files = [ii for ii in os.listdir(folder)]
                files.sort()
                for kk,file in enumerate(files):
                    if file.endswith(".csv"):
                        file_path = os.path.join(folder, file)
                        with open(file_path, 'r') as f:
                            csv_input = csv.reader(f, delimiter=",")
                            csv_rows = [row for row in csv_input]
                        
                            try: # Raman from CSV
                            
                                version = [ii for ii in csv_rows[0] if "ENLIGHTEN" in ii]

                            
                                if bool(re.search(barrelwise_file_time_stamp_format, file)) and file.startswith('LR'):
                                
                                   """
                                   ######################################
                                           Barrelwise Python API
                                   ######################################
                                   """
                                   spectrometer_name = 'PulpBot(BarrelWise) WP785'
                                   raman_time = re.search(barrelwise_file_time_stamp_format, file).group()
                                   raman_time = dt.datetime.strptime(raman_time, '%y%m%d-%H%M%S')
                                   raman_time = raman_time.strftime('%Y-%m-%d %H:%M:%S')
                         
                                   
                                   raman_raw_barrelwise_df = pd.read_csv(file_path,header= None,names = ['Spectrum'])
                                   raw_data.append(raman_raw_barrelwise_df['Spectrum'].values.reshape(-1,))
                                   spec_stats.append(['0', '1000', raman_time, spectrometer_name, '', '10', '801.451477051', '0.142020002007', '-5.66349990549e-06', '-8.55110027231e-09', '0', 'NA', '785.0', '0', '100', '0.00', '1024'])
                                   files_loaded.append(file) 
                                
                                
                                elif version[0].split(" ")[-1] == "1.1.12":
                                    """
                                    #############################
                                            Version 1.1.12
                                    #############################
                                    """
                #                    print(version)
                                    raw_data.append(np.array(csv_rows[3][17:1041]))
                                    spec_stats.append(csv_rows[3][0:17])
                                    wavelength = csv_rows[2][17:1041]

                                    spec_titles = csv_rows[1]
                                    files_loaded.append(file)
                                    version = version[0].split(" ")[-1]

                                    raw_data.append(np.array(csv_rows[3][17:1041]))
                                    spec_stats.append(csv_rows[3][0:17])
                                    wavelength = csv_rows[2][17:1041]

                                    spec_titles = csv_rows[1]
                                    files_loaded.append(file)
                                    
                                    
                                elif version[0].split(" ")[-1] == "Version":

                                    if csv_rows[0][1].split('.')[0] == "1": # 1.6.13
                                        """
                                        #############################
                                                Version 1.6.13
                                        #############################
                                        """
                                        version = csv_rows[0][1]
                                        # print(version)

                                        if len(csv_rows[1]) == 2:
                                            raw_data.append(np.array(np.array([ii[3] \
                                                for ii in csv_rows[21:]]).astype(float)))
                                            stat_row = [ii[1] for ii in csv_rows[2:19]]
                                            stat_order = [1,2,3,0,4,5,6,7,8,9,10,11,12,13,14,15,16]
                                            stat_row[13] = str(_strtobool(stat_row[13]))
                                            spec_stats.append(['0']+[stat_row[ii] for ii in stat_order if ii!=1])
                                            
                                            try:
                                                spec_stats[-1][2] = spec_stats[-1][2].split('.')[0]
                                                dt.datetime.strptime(spec_stats[-1][2],valid_time_format)
                                            except:
                                                time_split = csv_rows[1][3].split('-')
                                                time_str = time_split[0] + '-' + time_split[1]
                                                formatted_time = f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]} {time_str[9:11]}:{time_str[11:13]}:{time_str[13:]}"
                                                spec_stats[-1][2] = formatted_time
                                                print(f'corrected time format for sample {file}')
                                            
                                            wavelength = [ii[1] for ii in csv_rows[21:]]
                                            files_loaded.append(file)
                                        else:
                                            cols = csv_rows[3]
                                            csv_arr = np.array([ii for ii in csv_rows[22:]]).astype(float)
                                            for col in range(3,len(csv_rows[1])):
                                                raw_data.append(np.array(csv_arr[:,col]))
                                                stat_row = [ii[col] for ii in csv_rows[2:19]]
                                                stat_order = [1,2,3,0,4,5,6,7,8,9,10,11,12,13,14,15,16]
                                                stat_row[13] = str(_strtobool(stat_row[13]))
                                                spec_stats.append(['0']+[stat_row[ii] for ii in stat_order if ii!=1])
                                                
                                                try:
                                                    spec_stats[-1][2] = spec_stats[-1][2].split('.')[0]
                                                    dt.datetime.strptime(spec_stats[-1][2],valid_time_format)
                                                except:
                                                    time_split = csv_rows[1][3].split('-')
                                                    time_str = time_split[0] + '-' + time_split[1]
                                                    formatted_time = f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]} {time_str[9:11]}:{time_str[11:13]}:{time_str[13:]}"
                                                    spec_stats[-1][2] = formatted_time
                                                    print(f'corrected time format for sample {file}')
                                                
                                                wavelength = [ii[1] for ii in csv_rows[22:]]
                                                files_loaded.append(cols[col]+".csv")
                            
                                    elif csv_rows[0][1].split('.')[0] == "2": # 2.x
                                        """
                                        #############################
                                                Version 2.0.1
                                        #############################
                                        """
                                        version = csv_rows[0][1]
                #                        print(version)
                                        if version.split('.')[1] == "0": #2.0.1
                                            if len(csv_rows[1]) == 2:
                                                raw_data.append(np.array([ii[-1] \
                                                    for ii in csv_rows[31:]]).astype(float))
                                                stat_row = [6,14,15,2]            
                                                stat_row.extend(range(16,29))                       
                                                spec_stats.append([csv_rows[ii][1] for ii in stat_row])
                                                
                                                try:
                                                    spec_stats[-1][2] = spec_stats[-1][2].split('.')[0]
                                                    dt.datetime.strptime(spec_stats[-1][2],valid_time_format)
                                                except:
                                                    time_split = csv_rows[1][3].split('-')
                                                    time_str = time_split[0] + '-' + time_split[1]
                                                    formatted_time = f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]} {time_str[9:11]}:{time_str[11:13]}:{time_str[13:]}"
                                                    spec_stats[-1][2] = formatted_time
                                                    print(f'corrected time format for sample {file}')
                                                
                                                wavelength = [ii[1] for ii in csv_rows[31:]]
                                                files_loaded.append(file)
                                            else:
                                                cols = csv_rows[3]
                                                csv_arr = np.array([ii for ii in csv_rows[32:]]).astype(float)
                                                for col in range(3,len(csv_rows[1])):
                                                    raw_data.append(np.array(csv_arr[:,col]))
                                                    stat_row = [6,14,15,2]            
                                                    stat_row.extend(range(16,29)) 
                                                    spec_stats.append([csv_rows[ii][col] for ii in stat_row])
                                                    
                                                    try:
                                                        spec_stats[-1][2] = spec_stats[-1][2].split('.')[0]
                                                        dt.datetime.strptime(spec_stats[-1][2],valid_time_format)
                                                    except:
                                                        time_split = csv_rows[1][3].split('-')
                                                        time_str = time_split[0] + '-' + time_split[1]
                                                        formatted_time = f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]} {time_str[9:11]}:{time_str[11:13]}:{time_str[13:]}"
                                                        spec_stats[-1][2] = formatted_time
                                                        print(f'corrected time format for sample {file}')
                                                    
                                                    wavelength = [ii[1] for ii in csv_rows[31:]]
                                                    files_loaded.append(cols[col]+".csv")
                                        elif csv_rows[0][1].split('.')[1] == "2": #2.2.x
                                            """
                                            #############################
                                                    Version 2.2.7
                                            #############################
                                            """
                                            if len(csv_rows[1]) == 2:
                                                raw_data.append(np.array([ii[-1] \
                                                    for ii in csv_rows[35:]]).astype(float))
                                                stat_row = [7,18,19,2]            
                                                stat_row.extend(range(21,32))
                                                spec_stats.append([csv_rows[ii][1] for ii in stat_row])
                                                
                                                try:
                                                    spec_stats[-1][2] = spec_stats[-1][2].split('.')[0]
                                                    dt.datetime.strptime(spec_stats[-1][2],valid_time_format)
                                                except:
                                                    time_split = csv_rows[1][3].split('-')
                                                    time_str = time_split[0] + '-' + time_split[1]
                                                    formatted_time = f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]} {time_str[9:11]}:{time_str[11:13]}:{time_str[13:]}"
                                                    spec_stats[-1][2] = formatted_time
                                                    print(f'corrected time format for sample {file}')
                                                    try:
                                                        spec_stats[-1][2] = spec_stats[-1][2].split('.')[0]
                                                        dt.datetime.strptime(spec_stats[-1][2],valid_time_format)
                                                    except:
                                                        time_split = csv_rows[1][4].split('-')
                                                        time_str = time_split[0] + '-' + time_split[1]
                                                        formatted_time = f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]} {time_str[9:11]}:{time_str[11:13]}:{time_str[13:]}"
                                                        spec_stats[-1][2] = formatted_time
                                                        print(f'corrected time format for sample {file}')
                                                
                                                wavelength = [ii[1] for ii in csv_rows[35:]]
                                                files_loaded.append(file)
                                            else:
                                                cols = csv_rows[4]
                                                csv_arr = np.array([ii for ii in csv_rows[36:]]).astype(float)
                                                for col in range(3,len(csv_rows[1])):
                                                    raw_data.append(np.array(csv_arr[:,col]))
                                                    stat_row = [7,18,19,2]            
                                                    stat_row.extend(range(21,32)) 
                                                    spec_stats.append([csv_rows[ii][col] for ii in stat_row])
                                                    
                                                    try:
                                                        dt.datetime.strptime(spec_stats[-1][2],valid_time_format)
                                                    except:
                                                        time_split = csv_rows[1][3].split('-')
                                                        time_str = time_split[0] + '-' + time_split[1]
                                                        formatted_time = f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]} {time_str[9:11]}:{time_str[11:13]}:{time_str[13:]}"
                                                        spec_stats[-1][2] = formatted_time
                                                        print(f'corrected time format for sample {file}')
                                                        
                                                        try:
                                                            spec_stats[-1][2] = spec_stats[-1][2].split('.')[0]
                                                            dt.datetime.strptime(spec_stats[-1][2],valid_time_format)
                                                        except:
                                                            time_split = csv_rows[1][4].split('-')
                                                            time_str = time_split[0] + '-' + time_split[1]
                                                            formatted_time = f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]} {time_str[9:11]}:{time_str[11:13]}:{time_str[13:]}"
                                                            spec_stats[-1][2] = formatted_time
                                                            print(f'corrected time format for sample {file}')
                                                    
                                                    wavelength = [ii[1] for ii in csv_rows[35:]]
                                                    files_loaded.append(cols[col]+".csv") 
                                                    


                                  
                            
                            except Exception as e:
                                print(e)
                            f.close()

            
            
            
            raw_data = np.array(raw_data).astype(float)
            print("Data loaded from:\n" + datadir + "\n")

            # Average data
            ii = 0
            jj = 0
            avg_data = []
            files_avg = []
            spec_stats_avg = []
            spec_type = []

            if raw_data.size:

                for ii,file in enumerate(files_loaded):
                    if file.endswith(".csv"):
                        try:
                            # if not "1." in files_loaded[ii+1]:
                            file = file.replace(' ', '-')
                            if files_loaded[ii+1].startswith(file.split('-')[0]+'-'+file.split('-')[1]):

                                # print(file)
                                jj += 1
                            else:   
                                # print("***"+file)
        #                        print(ii-jj,ii+1)
        #                        print(files_loaded[ii-jj]+" — "+files_loaded[ii])
                                avg_data.append(np.mean(raw_data[ii-jj:ii+1,:], axis=0))
                                files_avg.append(files_loaded[ii-jj]) # Issue here if spectra are in multiple subfolders
                                spec_type.append(0)
                                if spec_stats[ii-jj][4] != '': # Fix for 2.2.7
                                    spec_stats_row = spec_stats[ii-jj][:4]
                                    spec_stats_row += ['']
                                    spec_stats_row += spec_stats[ii-jj][4:]
                                    spec_stats_row += ['1024']
                                    spec_stats_row[-4] = str(int(spec_stats_row[-4]=='True'))

                                else:
                                    spec_stats_row = spec_stats[ii-jj]                            
                                spec_stats_avg.append(spec_stats_row)
                                jj = 0
                        except: # Last file in folder
                            print(files_loaded[ii-jj]+" — "+files_loaded[ii])
                            avg_data.append(np.mean(raw_data[ii-jj:ii+1,:], axis=0))  
                            files_avg.append(files_loaded[ii-jj])
                            spec_type.append(0)
                            if spec_stats[ii-jj][4] != '': # Fix for 2.2.7
                                spec_stats_row = spec_stats[ii-jj][:4]
                                spec_stats_row += ['']
                                spec_stats_row += spec_stats[ii-jj][4:]
                                spec_stats_row += ['1024']
                                spec_stats_row[-4] = str(int(spec_stats_row[-4]=='True'))
                            else:
                                spec_stats_row = spec_stats[ii-jj]
                            spec_stats_avg.append(spec_stats_row)
                            break
        #                ii += 1
                avg_data = np.array(avg_data).astype(float)


            # Remove dead pixels
            avg_data = np.delete(avg_data, np.s_[0:49], axis=1)
          

            # Apply DWT
            filt_data = []
            coeffs = []
            sym5 = pywt.Wavelet("sym5")
            discard = [0,6,7]
            discard = [0,5,6,7]

            for ii in range(len(avg_data)):
                coeffs_ii = pywt.wavedec(avg_data[ii,:], sym5, mode="sym", level=7)
                for jj in list(discard):
                    coeffs_ii[jj] = np.zeros_like(coeffs_ii[jj])
                filt_data_1d = pywt.waverec(coeffs_ii, sym5, mode="sym")
                filt_data.append(filt_data_1d)     
                coeffs.append(pywt.ravel_coeffs(coeffs_ii)[0])
                coeffs_length = np.sum([ii.size for ii in coeffs_ii])
                
            filt_data = np.array(filt_data).astype(float)

            # Autoscale data
            autoscaled_data = []
            autoscaled_coeffs = []
            for ii in range(len(filt_data)):
                autoscaled_row = filt_data[ii]/filt_data[ii].std()
                coeffs_row = coeffs[ii]/coeffs[ii][coeffs[ii]!=0].std()
                autoscaled_data.append(autoscaled_row)
                autoscaled_coeffs.append(coeffs_row)
            autoscaled_data = np.array(autoscaled_data).astype(float)
            autoscaled_coeffs = np.array(autoscaled_coeffs).astype(float)
            coeffs = np.array(coeffs).astype(float)

            #%% Load target data from CSV file

            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" + \
                  "%                                                      %\n" + \
                  "%       Please use dialog window to select .CSV        %\n" + \
                  "%             file containing PulpEye data.            %\n" + \
                  "%                                                      %\n" + \
                  "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            try:
                # print("PulpEye Data Loaded from: "+pe_csvf)
                f = open(pe_csvf)
                pe_csv_input = csv.reader(f, delimiter=",")
                pe_csv = [ii for ii in pe_csv_input]
                f.close()
            except:
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" + \
                      "%                                                      %\n" + \
                      "%               Invalid file selected!                 %\n" + \
                      "%                                                      %\n" + \
                      "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
                input("Press Enter to exit...")
                exit(0)
            #%%

            # Extract data from CSV
            # pe_titles_csv = pe_csv[11][1:]
            pe_samples_csv = [ii[1] for ii in pe_csv[12:]]
            # pe_fileidx = []
            pe_missing = []
            pe_missing_fileidx = []
            for ii,pe_csvr in enumerate(pe_csv):
                for jj,kk in enumerate(pe_csvr):
                    if pe_csvr[jj] == "" or pe_csvr[jj] == "-" or \
                        pe_csvr[jj] == "#REF!" or pe_csvr[jj] == "#N/A":
                        pe_csv[ii][jj] = "-32768"#"0"#-32768

            matched_samples = []

            pe_missing=[]
            pe_missing_fileidx = []

            # Define a regular expression pattern for valid names
            pattern_energy_level = re.compile(r"LR\d{2}-\d{3}[-]\d{1,3}")
            pattern_sampleID = re.compile(r"LR\d{2}-\d{3}")


            # Match spectral data with CSV data
            for ii, file in enumerate(files_avg):
                temp_name = file.replace(".", "-")
                temp_name = temp_name.replace("_", "-")
                
                

                if "csv" in file:
                    temp_name = temp_name[:8]
                    if pattern_sampleID.match(temp_name):  # Check if the name matches the valid pattern
                        temp_name = temp_name.split("-")
                        temp_name = f"{temp_name[0]}-{temp_name[1].rjust(3, '0')}"
                        temp_match = fnmatch.filter(pe_samples_csv, f"{temp_name}*")

                for c, sample_name in enumerate(temp_match):
                    sample_name = sample_name.replace(' ', '-')
                    if pattern_energy_level.match(sample_name):  # Check if the name matches the valid pattern
                        try:
                            energy_level = sample_name.split("-")[-1]  # Extract the energy level
                            SampleID = sample_name[2:8]
                
                            record = {
                                "Index": ii,
                                "SampleID": SampleID,
                                "Refining_Level": energy_level, # Add the refining level key and value
                                "row_number":pe_samples_csv.index(sample_name) + 12,
                            }
                            matched_samples.append(record)
                
                        except:
                            print(f"Ignoring irregular sample: {sample_name}")
                else:
                    pe_missing.append(temp_name)
                    pe_missing_fileidx.append(ii)
            # # Adjust the indices by adding 12 for each energy level
            # for energy_level, indices in pe_samplerows.items():
            #     pe_samplerows[energy_level] = [ii + 12 for ii in indices]
            spec_type = np.array(spec_type)
            #%% Connect to MySQL server

            print("Connecting to SQL server...")
            import mysql.connector

            # Set up the connection parameters
            config = {
                'user': DB_USER,
                'password': DB_PASSWORD,
                'host': DB_HOST,
                'database': DB_NAME
            }

            # Establish a connection to the database
            try:
                session = mysql.connector.connect(**config)
                print("Connection established successfully")
            except mysql.connector.Error as err:
                print(f"Something went wrong: {err}")

            cursor = session.cursor()

            # Execute SHOW TABLES query
            cursor.execute("SHOW TABLES")

            # Fetch all tables
            tables = [table[0] for table in cursor.fetchall() if table[0] != 'selected_variables']

            # Dictionary to store DataFrames
            dataframes = {}

            # Iterate over each table and retrieve column names
            for table_name in tables:
                cursor.execute(f"DESCRIBE {table_name}")
                column_names = [column[0] for column in cursor.fetchall()]
                
                # Create a DataFrame for each table with the columns
                dataframes[table_name] = pd.DataFrame(columns=column_names)
                
            row_lists = {table_name: [] for table_name in tables}


            #%%

            """
            write_table function is used to write a row (v) into tables (t)

            """
            def write_table(t,v):
                cursor.execute('USE standalone_validation')
                cursor.execute("DESCRIBE "+str(t))
                columns = [f"`{row[0]}`" for row in cursor.fetchall()]
                query = "INSERT INTO {} ({}) VALUES ({})".format(str(t),
                   ", ".join(columns), ", ".join(["%s"] * len(columns))
               )
                try:
                    cursor.execute(query, v)
                except mysql.connector.IntegrityError as e:
                    if e.errno == mysql.connector.errorcode.ER_DUP_ENTRY:
                         print("Skipping duplicate entry for SampleID: {}".format(sampleid))
                    else:
                        
                        print("Error: {}".format(e))
                        print(e)
                 
            """
            ######################################################

                            Writing Raman Data

            ######################################################
            """   
            for dic in matched_samples:
                ii = dic['Index']
                row = dic['row_number']
                sampleid = dic['SampleID']
                energy_level = dic['Refining_Level']
               
            # Define dates (ramantime - spectra collected; resulttime - PE collected)
                if spec_type[ii] == 0: # Raman
                    ramantime = spec_stats_avg[ii][2]
                    ramantime = ramantime.split('.')[0]

                # if float(pe_csv[row][71]) == 0:
                #     resulttime = convert_xldate_to_datetime(float(pe_csv[row+2][71])-0.01,0).strftime("%Y-%m-%d %H:%M:%S") # Correct date
                # else:
                #     resulttime = convert_xldate_to_datetime(float(pe_csv[row][71]),0).strftime("%Y-%m-%d %H:%M:%S")        



                # Raman Data
                if spec_type[ii] == 0:      
                    
                    sn = "WP785"

                    raman_row = [sampleid,sn]
                    raman_row.extend(spec_stats_avg[ii][1:])
                    raman_row += [float(kk) for kk in list(np.zeros(49))]
                    raman_row += [float(kk) for kk in list(avg_data[ii])] 
                    try:
                        write_table('raman', raman_row)
                    except Exception as e:           
                        print(f"An error occurred: {e}")

                    # row_lists['raman'].append(raman_row)
                        
                
                    # # Autoscaled Raman data
                    # raman_row_scaled = [sampleid, ramantime,sn]
                    # raman_row_scaled.extend(spec_stats_avg[ii][1:])
                    # raman_row_scaled += [float(kk) for kk in list(np.zeros(48))]
                    # raman_row_scaled += [float(kk) for kk in list(autoscaled_data[ii])]
                    # try:    
                    #     write_table('raman_scaled', raman_row_scaled)
                    # except Exception as e:           
                    #     print(f"An error occurred: {e}")
                    
                    # # row_lists['raman_scaled'].append(raman_row_scaled)

                    
                    # # Raman coefficient data
                    # raman_coeffs_row = [sampleid, ramantime,sn]
                    # raman_coeffs_row.extend(spec_stats_avg[ii][1:])
                    # raman_coeffs_row += [float(kk) for kk in list(coeffs[ii,:250])]
                    # raman_coeffs_row += [float(kk) for kk in list(np.zeros(1078-coeffs_length))]
                    # try:
                    #     write_table('raman_coeffs', raman_coeffs_row)
                    # except Exception as e:           
                    #     print(f"An error occurred: {e}")
                        
                    # # row_lists['raman_coeffs'].append(raman_coeffs_row)

                    
                    # # Autoscaled Raman coefficient data
                    # raman_coeffs_s_row = [sampleid, ramantime,sn]
                    # raman_coeffs_s_row.extend(spec_stats_avg[ii][1:])
                    # raman_coeffs_s_row += [float(kk) for kk in list(autoscaled_coeffs[ii,:250])]
                    # try:
                    #     write_table('raman_coeffs_scaled', raman_coeffs_s_row)
                    # except Exception as e:           
                    #     print(f"An error occurred: {e}")
                        
                    # # row_lists['raman_coeffs_scaled'].append(raman_coeffs_row)

                        
            #%%

                       
            for dic in matched_samples:
                try:
                    ii = dic['Index']
                    row = dic['row_number']
                    sampleid = dic['SampleID']
                    energy_level = dic['Refining_Level']
                    
                    
                      
                      
                # Define dates (ramantime - spectra collected; resulttime - PE collected)
                    if spec_type[ii] == 0: # Raman
                        raman_time = spec_stats_avg[ii][2]
                        raman_time = raman_time.split('.')[0]
                    
                    if float(pe_csv[row][71]) == 0:
                        sampletime = convert_xldate_to_datetime(float(pe_csv[row+2][71])-0.01,0).strftime("%Y-%m-%d %H:%M:%S") # Correct date
                    else:
                        sampletime = convert_xldate_to_datetime(float(pe_csv[row][71]),0).strftime("%Y-%m-%d %H:%M:%S")        
                        
                        
                        
                    #Batch Info    
                    # define the query
                    table = 'batchinfo'
                    batch_info_row = (sampleid, sampleid.split('-')[0], pe_csv[row][50], pe_csv[row][2], raman_time, sampletime)
                    batch_info_row = [None if ii=='-32768' else ii for row,ii in enumerate(batch_info_row)]
                    write_table(table,batch_info_row)
                    
                    if batch_info_row not in row_lists['batchinfo']:
                        row_lists['batchinfo'].append(batch_info_row)

                            
                    # Targets (from Refiner)
                    table= 'targets'
                    target_row = [sampleid, energy_level]
                    target_row.extend(pe_csv[row][6:45])
                    target_row = [0 if ii=='-32768' and row != 11 else ii for row,ii in enumerate(target_row)]
                    target_row = [0 if ii=='0' and row>36 else ii for row,ii in enumerate(target_row)]
                    # targets.insert().values(target_row).execute()
                    
                    write_table(table, target_row)
                    
                    # row_lists['targets'].append(target_row)

                
                      
                    # PulpEye data summary
                    table= 'pulpeye'
                    pulpeye_row = [sampleid, energy_level]
                    pulpeye_row.append(pe_csv[row][6])
                    pulpeye_row.extend([ramantime, 1, ii+1,sampletime])
                    pulpeye_row.extend(pe_csv[row][48:53])
                    pulpeye_row.extend(pe_csv[row][61:66])
                    pulpeye_row = [None if ii=='-32768' and row>7 else ii for row,ii in enumerate(pulpeye_row)]
                    # pulpeye_row = [None if ii=='0' and row>7 and row not in range(16,17) else ii for row,ii in enumerate(pulpeye_row)]
                    if pulpeye_row[12:15] == ['0','0','0']:
                        pulpeye_row[12:15] = [None,None,None]
                        
                    write_table(table, pulpeye_row)

                    # row_lists['pulpeye'].append(pulpeye_row)
                    
                
                    # Fiber distributions
                    table= 'fiberdistributions'
                    fiberdistributions_row = [sampleid, energy_level]
                    fiberdistributions_row.extend([sampletime, ii+1])
                    fiberdistributions_row.extend(pe_csv[row][268:468])
                    fiberdistributions_row = [None if ii=='-32768' and row>5 else ii for row,ii in enumerate(fiberdistributions_row)]
                    write_table(table, fiberdistributions_row)
                    
                    # row_lists['fiberdistributions'].append(fiberdistributions_row)

                    # Fiber kinks    
                    table= 'fiberkinks'
                    fiberkinks_row = [sampleid, energy_level, sampletime, ii]
                    fiberkinks_row.extend(pe_csv[row][175:179])
                
                    write_table(table, fiberkinks_row)
                    
                    # row_lists['fiberkinks'].append(fiberkinks_row)

                    
                    # FWT results
                    table= 'fwtdist'
                    fwtdist_row = [sampleid, energy_level]
                    fwtdist_row.extend([sampletime, ii+1])  
                    fwtdist_row.extend(pe_csv[row][199:239])
                    fwtdist_row = [None if ii=='-32768' else ii for row,ii in enumerate(fwtdist_row)]    
                    fwtdist_row = ['0' if ii=='-32768' and row in range(6,11) else ii for row,ii in enumerate(fwtdist_row)]   
                    # fusion1.insert().values(fwtr_row).execute()
                    
                    write_table(table, fwtdist_row)
                    
                    # row_lists['fwtdist'].append(fwtdist_row)

                      
                    # FWT width results
                    table= 'fwtwidthdist'
                    fwtwidthdist_row = [sampleid, energy_level]
                    fwtwidthdist_row.extend([sampletime, ii+1])
                    fwtwidthdist_row.extend(pe_csv[row][179:199])
                    fwtwidthdist_row = [None if ii=='-32768' and row>3 else ii for row,ii in enumerate(fwtwidthdist_row)]    
                    # fusion0.insert().values(fwtw_row).execute()
                    
                    write_table(table, fwtwidthdist_row)
                    
                    # row_lists['fwtwidthdist'].append(fwtwidthdist_row)
                
                
                    
                    # Fiber rummary
                    table= 'fibersummary'
                    fibersummary_row = [sampleid, energy_level]
                    fibersummary_row.extend([sampletime, ii+1])
                    fibersummary_row.extend(pe_csv[row][239:268])
                    fibersummary_row.extend([0])
                    fibersummary_row = [None if ii=='-32768' else ii for row,ii in enumerate(fibersummary_row)]    
                    fibersummary_row = ['0' if ii=='-32768' and row in range(7,11) else ii for row,ii in enumerate(fibersummary_row)]    
                    # fusion2.insert().values(fusion2_row).execute()
                    
                    
                    write_table(table, fibersummary_row)
                    
                    # row_lists['fibersummary'].append(fibersummary_row)

                except Exception as e:           
                    print(f"An error occurred: {e}")       
             
                
              
            session.commit()
            session.close()
            #%%
            t1 = time.time()
            total = t1-t0
            
            print("\nDone.")
            print("Total time elapsed :", total, "\n")
            break
        except Exception as e:
            print(e)
            time.sleep(5)
            break
        
        
        
#%% CREATE EE_PREDICTION TABLES IN SQL

def build_pred_table(start,end):
    
    # Provide the path to the SQL script file
    script_path = os.path.join(BASE_DIR, 'sql', 'build_selected_variables.sql')
    
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
    )

    try:
        # Create a cursor object to execute SQL queries
        cursor = conn.cursor()

        # Read the SQL script file
        with open(script_path) as script_file:
            sql_statements = script_file.read().split(";")  # Split statements by semicolon

        # Execute each SQL statement
        for statement in sql_statements:
            if statement.strip():  # Skip empty statements
                cursor.execute(statement)

        # Commit the changes to the database
        conn.commit()
        
        print("Connected to SQL...")

    except mysql.connector.Error as error:
        print("Error executing SQL script:", error)

    finally:
        # Close the cursor and connection
        if conn.is_connected():
            cursor.close()
            # conn.close()
            # print("MySQL connection is closed")

    try:
        # Create a cursor object to execute SQL queries
        cursor = conn.cursor()
        
        # Delete rows from the table with only batchid based on the condition
        delete_query = """
        DELETE FROM selected_variables
        WHERE SampleID NOT IN (
            SELECT SampleID
            FROM batchinfo
            WHERE `SampleTime` BETWEEN %s AND %s
        )
        """

        # Execute the delete query with the condition date as a parameter
        cursor.execute(delete_query, (start,end))
        cursor.reset()

        cursor.execute('SELECT * FROM standalone_validation.selected_variables;')
        selected_data = cursor.fetchall()
        selected_data = pd.DataFrame(selected_data, columns=[i[0] for i in cursor.description])
        for coeff in range(1,201):
            selected_data = selected_data.drop([f"Coeff_{coeff:03d}"], axis = 1)
        
        cursor.execute('SELECT * FROM standalone_validation.raman WHERE SampleID in (SELECT SampleID from standalone_validation.selected_variables);')
        raman_data = cursor.fetchall()
        raman_data = pd.DataFrame(raman_data, columns=[i[0] for i in cursor.description])


        
        print(f"Samples between {start} and {end} are exported to table Selected_Variables")
        # Commit the changes to the database
        conn.commit()

        # print("Rows deleted successfully.")

    except mysql.connector.Error as error:
        print("Error deleting rows from MySQL table:", error)
        
    return selected_data, raman_data
    
#%%  
    
def gui_prediction():

    models_dir = os.path.join(BASE_DIR, 'Models')

    data_df_list = []
    sampleID = []
    # targets_df_list = []
    #features_df_list = []

    RCoeff_df = pd.DataFrame()

    table = "selected_variables"
    data_df = pd.DataFrame()
    # target_df = pd.DataFrame()
    
    """
    ######################################################
    
            Import Selected_varibales Table
    
    ######################################################
    """  
    
    try:
        with mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD) as conn:
            with conn.cursor() as cursor:
                cursor.execute('USE standalone_validation;')
                cursor.execute(f'SELECT * FROM {table};')
                data = cursor.fetchall()
                # DATAFRAME THE DATA
                data_df = pd.DataFrame(data, columns=[i[0] for i in cursor.description])
                # data_df = data_df.drop(['ZSBLwet', 'ZSBLdry'], axis=1)
                cursor.close()
                data_df.insert(2, 'LABELTEXT', data_df['SampleID'] + '-' + data_df['Refining_Level'])

                
                
                if RCoeff_df.empty :
                    # Prepare a list to hold the Series you want to add to RCoeff_df
                    series_to_add = []
                    
                    # Loop through the range of column indices you're interested in
                    for coeff in range(17, 165):
                        # Instead of inserting into RCoeff_df, append the Series to the list
                        series_to_add.append(data_df[f"Coeff_{coeff:03d}"].copy())
                        
                    RCoeff_df = pd.concat(series_to_add, axis=1)
                    RCoeff_df.columns = [f"Coeff_{coeff:03d}" for coeff in range(17, 165)]


                        
                for coeff in range(1,17):
                    data_df = data_df.drop([f"Coeff_{coeff:03d}"], axis = 1)
                for coeff in range(165,201):
                    data_df = data_df.drop([f"Coeff_{coeff:03d}"], axis = 1)
                    
                    

            """
            ######################################################

                            Import Raman Table

            ######################################################
            """  
                    
            wavenumbers = np.arange(50,700)
            with mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD) as conn:
                with conn.cursor() as cursor:
                    cursor.execute('USE standalone_validation;')
                    cursor.execute('SELECT * FROM raman ;')
                    data = cursor.fetchall()
                    raman_df = pd.DataFrame(data, columns=[i[0] for i in cursor.description])
                    
                    data_frames = [raman_df[['SampleID']]]
                    # Loop through the spectrum columns and add them to the list
                    for sp in range(50, 700):
                        spectrum_column = raman_df[[f'Spectrum_{sp:03d}']]
                        data_frames.append(spectrum_column)
                    # Concatenate all the data frames in the list
                    raman_spec_df = pd.concat(data_frames, axis=1)
                                            
                sampleID.append(data_df['SampleID']) #EXTRACT LABELS
                
   
                
                
                # def multiplicative_scattering_correction(df):
                #     """
                #     Apply Multiplicative Scattering Correction to the spectra in the DataFrame.
                #     Assumes the first column contains non-spectral data (e.g., identifiers).
                    
                #     :param df: DataFrame with the first column as identifiers and remaining columns as spectra.
                #     :return: DataFrame with MSC applied to the spectra.
                #     """
                #     # Calculate the mean spectrum
                #     # mean_spectrum = df.iloc[:, 1:].mean()
                #     mean_spectrum = df.iloc[520,1:]
                
                #     # Prepare an array for corrected spectra
                #     corrected_spectra = np.zeros_like(df.iloc[:, 1:])
                
                #     # Vectorized computation of regression coefficients
                #     x = mean_spectrum.values.reshape(-1, 1)
                #     y = df.iloc[:, 1:].values
                #     xTx = np.dot(x.T, x)
                #     xTy = np.dot(x.T, y.T)
                #     coeffs = xTy / xTx
                
                #     # Applying the correction
                #     corrected_spectra = np.dot(x, coeffs).T
                
                #     # Combine corrected spectra with the identifier column
                #     corrected_df = pd.DataFrame(corrected_spectra, columns=df.columns[1:])
                #     corrected_df.insert(0, df.columns[0], df.iloc[:, 0].values)
                
                #     return corrected_df
                            
    
                def polynomial_background_correction(spectrum, degree):
                    """
                    Remove background from a spectrum using polynomial fitting.
                    
                    :param spectrum: Pandas Series representing the spectrum.
                    :param degree: Degree of the polynomial to be fitted.
                    :return: Background corrected spectrum.
                    """
                    p = np.polyfit(wavenumbers, spectrum, degree)
                    fitted_bg = np.polyval(p, wavenumbers)
                    return spectrum - fitted_bg
                
                def polynomial_df_bg_correction(df, degree=4):
                    """
                    Apply polynomial background correction to each row of a DataFrame and keep the first column.
                
                    :param df: DataFrame where the first column is identifiers and the rest are spectra.
                    :param degree: Degree of the polynomial for background correction.
                    :return: DataFrame with the first column and corrected spectra.
                    """
                    # Extract the first column (e.g., SampleID or similar identifiers)
                    first_column = df.iloc[:, 0]
                
                    # Perform background correction on the spectral data (excluding the first column)
                    transposed_spectra = df.iloc[:, 1:].T
                    corrected_spectra = transposed_spectra.apply(polynomial_background_correction, degree=degree)
                    corrected_df = corrected_spectra.T
                    
                    # Add the first column back to the corrected DataFrame
                    corrected_df.insert(0, first_column.name, first_column)
                
                    return corrected_df
                
                def snv(input_df):
                  
                    # Define a new array and populate it with the corrected data  
                    output_df = input_df.copy()

                    # Apply SNV to each row
                    for i in range(input_df.shape[0]):
                        output_df.iloc[i, 1:] = (input_df.iloc[i, 1:] - np.mean(input_df.iloc[i, 1:])) / np.std(input_df.iloc[i, 1:])

                    return output_df
    
                
                # Applying Scattering Correction to your DataFrame
                
                corrected_df = snv(raman_spec_df)

                corrected_df = polynomial_df_bg_correction(corrected_df)  
       
                raman_bg_norm_smooth_df = corrected_df.copy()
                raman_bg_norm_smooth_df.iloc[:, 1:] = savgol_filter(raman_bg_norm_smooth_df.iloc[:, 1:],
                                                                    window_length=4, polyorder=3, axis=1, deriv=0, mode = 'nearest')
    
    
        
                # # Plot the original spectrum, the fitted background, and the corrected spectrum
                plt.figure(figsize=(10, 6))
                for spect in range(1,len(raman_df)):

                    
                    plt.plot(wavenumbers, raman_spec_df.iloc[spect, 1:], label = 'Raw Spectrum')
    
                    plt.title('Raw Raman Spectra')
                    plt.xlabel('Pixel')
                    plt.ylabel('Intensity')
                    # plt.legend()
                plt.show()
                    
                plt.figure(figsize=(10, 6))
                for spect in range(1,len(raman_df)):
                    
                    plt.plot(wavenumbers, raman_bg_norm_smooth_df.iloc[spect, 1:], label = 'Preprocessed Spectrum')
    
                    plt.title('Detrend_SNV_Smoothing Raman Spectra')
                    plt.xlabel('Pixel')
                    plt.ylabel('Intensity')
                plt.show()
                    
                raman_df = raman_bg_norm_smooth_df.copy()

                
                data_df_list.append(data_df)


    except Error as e:
        print(f"An error occurred: {e}")


    """
    ######################################################
    
                    DATA FUSION
    
    ######################################################
    """     

    data_df = pd.merge(data_df, raman_df, on='SampleID', how='left')
    for coeff in range(17,165):
        data_df = data_df.drop([f"Coeff_{coeff:03d}"], axis = 1)







        
    # EXTRACT FEATURES (VARIBALES)
    targets = data_df[targets_list]
    targets = targets.add_suffix('_Measured')
    x_df = data_df.drop(columns=targets_list)
    # imputer = SimpleImputer(strategy='mean')
    # x_df = imputer.fit_transform(x_df)
    labeltext = x_df[['LABELTEXT']]
    targets.insert(0,'LABELTEXT', labeltext)
    # x_df_sample_ID = x_df["SampleID"]
    x_df = x_df.drop(["SampleID",'LABELTEXT','Refining_Level'], axis = 1)
    s = load(os.path.join(models_dir,'Scaler.joblib'))
    x_std = s.fit_transform(x_df)
    x_df = pd.DataFrame(data =x_std, columns= x_df.columns)
    
    
    models_joblib_files = glob.glob(os.path.join(models_dir, '*.joblib'))
    models_joblib_files = sorted(models_joblib_files)


    # results = pd.DataFrame(columns = ['LabelText'], data = labeltext)
    results = labeltext.copy()
    for model_path in models_joblib_files:
        
        head, tail = os.path.split(model_path)
        model_name = tail.split('_')[0]
        print(model_name)
        if not model_name.startswith('Scaler'):
            model = load(model_path)
            y_prediction = model.predict(x_df)
            
            results[model_name] = y_prediction
            # results.insert(0,'Sample',labeltext)

    results = pd.merge(results, targets, on = 'LABELTEXT', how = 'left')
    # Sort the DataFrame by the selected columns
    results = results.drop(['LABELTEXT'], axis = 1)
    results = results.sort_index(axis=1)
    results.insert(0,'LABELTEXT',labeltext)

    return results

#%%GUI
    
#%% GUI
    
import sys
import os
from PyQt5.QtWidgets import (QApplication, QCalendarWidget, QDateEdit, QDialog, QFileDialog, QFrame, QGridLayout,
                             QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
                             QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
                             QTextEdit, QVBoxLayout, QWidget)
from PyQt5.QtCore import QDate, Qt, QCoreApplication, QThread, pyqtSignal, QRunnable, QThreadPool, QObject
from PyQt5.QtGui import QPixmap, QFont, QIcon, QColor, QBrush

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class DataProcessingWorkerSignals(QObject):
    finished = pyqtSignal()

class DataProcessingWorker(QRunnable):
    def __init__(self, datadir, pe_csvf):
        super(DataProcessingWorker, self).__init__()
        self.datadir = datadir
        self.pe_csvf = pe_csvf
        self.signals = DataProcessingWorkerSignals()

    def run(self):
        # Placeholder for the actual append_data function
        append_data(self.datadir, self.pe_csvf)
        self.signals.finished.emit()


class RemoveDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Remove Sample")
        self.init_ui()
        
    def init_ui(self):    
        self.setGeometry(300, 300, 300, 120)

        remove_dialoge_layout = QVBoxLayout(self)
        
        input_layout = QHBoxLayout()
        input_label = QLabel("LR")
        input_label.setFont(QFont("Arial", 24))

        
        self.remove_button = QPushButton('Remove')
        self.remove_button.setFixedSize(100,40)
        self.remove_button.setStyleSheet("""
            QPushButton {
                background-color: #6E82AA; 
                color: white; 
                border-radius: 20px;  /* Half of the button's size to make it round */
            }
            QPushButton:hover {
                background-color: #8a9bbd;  /* Slightly lighter color for hover effect */
            }

        """)
        self.remove_button.clicked.connect(self.remove_query)
        
        self.input_textbox = QLineEdit()
        self.input_textbox.setFixedSize(200,40)
        self.input_textbox.setFont(QFont("Arial", 24))
        # self.input_textbox.returnPressed.connect(self.remove_query)  # Connect Enter key to button click

        
        
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_textbox)
        input_layout.addStretch()
        input_layout.addWidget(self.remove_button)
        
        remove_dialoge_layout.addLayout(input_layout)


    def remove_query(self):
        sampleID = self.input_textbox.text()
        # SampleID = 'LR' + SampleID
        self.input_textbox.clear()
        
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        
        
        try:
            print(f'Removing sample LR{sampleID}')
            # Create a cursor object to execute SQL queries
            cursor = conn.cursor()
            # Fetch all table names
            cursor.execute("USE standalone_validation")
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            cursor.execute('SELECT COUNT(*) FROM pulpeye WHERE SampleID = %s', (sampleID,))
            count = cursor.fetchall()
            print(count[0])
            if count[0][0] == 0:
                print('count 0')
                QMessageBox.information(self, "Sample Not Found", f"404! Sample LR{sampleID} not found!")

            else:
                # Iterate over all tables and delete the sample if it exists
                for (table_name,) in tables:
                    delete_query = f"DELETE FROM {table_name} WHERE SampleID = %s"
                    try:
                        cursor.execute(delete_query, (sampleID,))
                        conn.commit()
                        print(f"Removed from {table_name}")
                    except mysql.connector.Error as err:
                        print(f"Error removing from {table_name}: {err}")
                
                cursor.close()
                conn.close()
                
                QMessageBox.information(self, "Success", f"Sample LR{sampleID} successfully removed!")

        except:
            QMessageBox.information(self, "Error", f"Something went wrong, failed to remove LR{sampleID}!")


class AddDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Import Actions")
        self.datadir = None
        self.pe_csvf = None
        self.init_ui()

    def init_ui(self):
        self.setGeometry(300, 300, 500, 200)

        layoutV = QVBoxLayout(self)

        # First action
        action1_layout = QHBoxLayout()
        action1_label = QLabel("1) Choose the Raman folder:")
        action1_button = QPushButton("Browse...")
        action1_button.setFixedWidth(100)
        action1_button.setStyleSheet("""
            QPushButton {
                background-color: #6E82AA; 
                color: white; 
                padding: 10px;
                border-radius: 10px;  /* Half of the button's size to make it round */
            }
            QPushButton:hover {
                background-color: #8a9bbd;  /* Slightly lighter color for hover effect */
            }

        """)
        action1_button.clicked.connect(self.get_directory_path)
        self.output_text_1 = QLabel("")
        action1_layout.addWidget(action1_label)
        action1_layout.addWidget(action1_button)
        action1_layout.addStretch()
        action1_layout.addWidget(self.output_text_1)
        action1_layout.addStretch()

        layoutV.addLayout(action1_layout)

        # Second action
        action2_layout = QHBoxLayout()
        action2_label = QLabel("2) Choose the master PulpEye file (.csv):")
        action2_button = QPushButton("Browse...")
        action2_button.setFixedWidth(100)
        action2_button.setStyleSheet("""
            QPushButton {
                background-color: #6E82AA; 
                color: white; 
                padding: 10px;
                border-radius: 10px;  /* Half of the button's size to make it round */
            }
            QPushButton:hover {
                background-color: #8a9bbd;  /* Slightly lighter color for hover effect */
            }

        """)
        action2_button.clicked.connect(self.get_file_path)
        self.output_text_2 = QLabel("")
        action2_layout.addWidget(action2_label)
        action2_layout.addWidget(action2_button)
        action2_layout.addStretch()
        action2_layout.addWidget(self.output_text_2)
        action2_layout.addStretch()
        layoutV.addLayout(action2_layout)

        # Third Action
        action3_layout = QHBoxLayout()
        action3_label = QLabel("3) Write new samples in SQL database:")
        action3_button = QPushButton("Run")
        action3_button.setFixedWidth(100)
        action3_button.setStyleSheet("""
    QPushButton {
        background-color: #6E82AA; 
        color: white; 
        padding: 10px;
        border-radius: 10px;  /* Half of the button's size to make it round */
    }
    QPushButton:hover {
        background-color: #8a9bbd;  /* Slightly lighter color for hover effect */
    }

""")
        action3_button.clicked.connect(self.append_data)
        action3_layout.addWidget(action3_label)
        action3_layout.addWidget(action3_button)
        action3_layout.addStretch()
        layoutV.addLayout(action3_layout)

    # def get_directory_path(self):
    #     self.datadir = QFileDialog.getExistingDirectory(self, "Select Directory")
    #     self.output_text_1.setText(os.path.split(self.datadir)[1])

    # def get_file_path(self):
    #     self.pe_csvf, _ = QFileDialog.getOpenFileName(self, "Select file", "", "CSV Files (*.csv)")
    #     self.output_text_2.setText(os.path.split(self.pe_csvf)[1])

    def get_directory_path(self):
        self.datadir = QFileDialog.getExistingDirectory(self, "Select Directory")
        # self.datadir = directory_path
        self.output_text_1.setText(os.path.split(self.datadir)[1])
        # if self.datadir: 
        #     self.final_output_text.append("*** \n 1) Selected raman folder: " + os.path.split(self.datadir)[1] +"\n\nProceed to step 2...")


    def get_file_path(self):
        self.pe_csvf, _ = QFileDialog.getOpenFileName(self, "Select file", "", "CSV Files (*.csv)")
        # self.pe_csvf = file_path
        self.output_text_2.setText(os.path.split(self.pe_csvf)[1])
        # self.final_output_text.append("*** \n 2) PulpEye data loaded from:\n "+ os.path.split(self.pe_csvf)[1])
        # if self.datadir is not None and self.pe_csvf is not None:
        #     self.final_output_text.append("\nProceed to step 3...")

    def append_data(self):
        if self.datadir and self.pe_csvf:
            worker = DataProcessingWorker(self.datadir, self.pe_csvf)
            worker.signals.finished.connect(self.data_processing_finished)
            QThreadPool.globalInstance().start(worker)
        else:
            QMessageBox.warning(self, "Warning", "Please select a valid Raman Folder or PulpEye File.")

    def data_processing_finished(self):
        QMessageBox.information(self, "Done", "Data processing finished.")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Pulp Brain")

        self.init_ui()
        self.threadpool = QThreadPool()

    def init_ui(self):
        self.setGeometry(100, 100, 900, 600)

        self.welcome_widget = QWidget()
        welcome_layout = QVBoxLayout()
        self.welcome_widget.setStyleSheet("background-color: #FFFFFF;")

        # Add logos to the welcome page
        logo_layout = QHBoxLayout()
        logo1_label = QLabel(self)
        logo2_label = QLabel(self)
        pixmap_logo1 = QPixmap(asset('UBC_logo.png')).scaledToWidth(370, mode=Qt.SmoothTransformation)
        pixmap_logo2 = QPixmap(asset('Canfor_logo.png')).scaledToWidth(400, mode=Qt.SmoothTransformation)
        logo1_label.setPixmap(pixmap_logo1)
        logo2_label.setPixmap(pixmap_logo2)
        logo_layout.addStretch()
        logo_layout.addWidget(logo1_label, alignment=Qt.AlignCenter)
        logo_layout.addWidget(logo2_label, alignment=Qt.AlignCenter)
        logo_layout.addStretch()
        welcome_layout.addLayout(logo_layout)

        # Add button to proceed to the main page
        proceed_button = QPushButton("Get Started")
        proceed_button.setFixedSize(300,80)
        proceed_button.setFont(QFont("Arial", 28))
        proceed_button.setStyleSheet("""
            QPushButton {
                background-color: #6E82AA; 
                color: white; 
                border-radius: 20px;  /* Half of the button's size to make it round */
                }
            QPushButton:hover {
                background-color: #8a9bbd;  /* Slightly lighter color for hover effect */
            }
            

        """)
        proceed_button.clicked.connect(self.show_main_page)
        welcome_layout.addStretch()
        welcome_layout.addWidget(proceed_button, alignment=Qt.AlignCenter)
        welcome_layout.addStretch()

        self.welcome_widget.setLayout(welcome_layout)
        self.setCentralWidget(self.welcome_widget)

        # Create main widget for the main page
        self.tabWidget = QTabWidget()
        tab1 = self.init_data_import_tab()
        tab2 = self.init_prediction_tab()

        self.tabWidget.addTab(tab1, "Data Import")
        self.tabWidget.addTab(tab2, "Prediction")

    def show_main_page(self):
        self.setCentralWidget(self.tabWidget)

    def init_data_import_tab(self):
        tab1 = QWidget()
        
        tab1.setStyleSheet("background-color: #FFFFFF;")

        layoutV = QVBoxLayout()



        ''' Logo '''
        # Logo layout
        logo_layout = QHBoxLayout()
        logo1_label = QLabel(self)
        logo2_label = QLabel(self)
        pixmap_logo1 = QPixmap(asset('UBC_logo.png')).scaledToWidth(75, mode=Qt.SmoothTransformation)
        pixmap_logo2 = QPixmap(asset('Canfor_logo.png')).scaledToWidth(100, mode=Qt.SmoothTransformation)
        logo1_label.setPixmap(pixmap_logo1)
        logo2_label.setPixmap(pixmap_logo2)
        
        # Help Button with Icon and Text
        help_button_widget = QWidget()
        help_vlayout = QVBoxLayout(help_button_widget)
        help_button = QPushButton()
        help_button.setStyleSheet("""
            QPushButton {
                background-color: #6E82AA; 
                color: white; 
                border-radius: 20px;  /* Half of the button's size to make it round */
            }

        """)
        help_button.setIcon(QIcon(QPixmap(asset('help_2.png'))))
        help_button.setFixedSize(40, 40)
        help_button.setIconSize(help_button.size())
        help_label = QLabel("Help")
        help_label.setAlignment(Qt.AlignCenter)

        help_vlayout.addWidget(help_button)
        # help_vlayout.addWidget(help_label)
        help_vlayout.setAlignment(Qt.AlignHCenter)

        
        # logo_layout.addStretch()
        logo_layout.addWidget(logo2_label)
        logo_layout.addWidget(logo1_label)
        logo_layout.addStretch()
        logo_layout.addWidget(help_button_widget)
        layoutV.addLayout(logo_layout)



        '''Edit'''
        # Button to open Actions Dialog
        import_hlayout = QHBoxLayout()
        import_label = QLabel("1) Edit Samples")
        import_button = QPushButton("  Add")
        import_button.setStyleSheet("""
            QPushButton {
                background-color: #6E82AA; 
                color: white; 
                padding: 10px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #8a9bbd;  /* Slightly lighter color for hover effect */
            }
        """)        
        import_button.setIcon(QIcon(QPixmap(asset('add.png'))))
        import_button.setFixedWidth(150)
        import_button.clicked.connect(self.open_actions_dialog)
        
        remove_button = QPushButton("  Remove")
        remove_button.setStyleSheet("""
            QPushButton {
                background-color: #6E82AA; 
                color: white; 
                padding: 10px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #8a9bbd;  /* Slightly lighter color for hover effect */
            }
        """)    
        remove_button.setIcon(QIcon(QPixmap(asset('remove.png'))))
        remove_button.setFixedWidth(150)
        remove_button.clicked.connect(self.open_remove_dialog)
        
        import_hlayout.addStretch()
        import_hlayout.addWidget(import_label)
        # import_hlayout.addStretch()
        import_hlayout.addWidget(import_button)
        import_hlayout.addWidget(remove_button)
        import_hlayout.addStretch()

        layoutV.addLayout(import_hlayout)


        '''Date'''
        # Date pickers
        # action4_layout = QHBoxLayout()
        # action4_label = QLabel("4) Select samples within the following date range:")
        # action4_layout.addWidget(action4_label)
        # layoutV.addLayout(action4_layout)

        action4_2_layout = QHBoxLayout()
        self.start_date_picker = QDateEdit()
        self.start_date_picker.setDate(QDate.currentDate())
        self.end_date_picker = QDateEdit()
        self.end_date_picker.setDate(QDate.currentDate())
        action4_2_layout.addStretch()
        action4_2_layout.addWidget(QLabel("2) Select Samples between      "))
        action4_2_layout.addWidget(self.start_date_picker)
        action4_2_layout.addStretch()
        action4_2_layout.addWidget(QLabel("and"))
        action4_2_layout.addStretch()
        action4_2_layout.addWidget(self.end_date_picker)
        action4_2_layout.addStretch()

        
        action4_button = QPushButton("  Select")
        action4_button.setFixedWidth(150)
        action4_button.setStyleSheet("""
            QPushButton {
                background-color: #6E82AA; 
                color: white; 
                padding: 10px;
                border-radius: 10px;
                }
            QPushButton:hover {
                background-color: #8a9bbd;  /* Slightly lighter color for hover effect */
            }
        """)   
        action4_button.setIcon(QIcon(QPixmap(asset('select_date.png'))))

        action4_button.clicked.connect(self.pred_table)
        
        action4_2_layout.addWidget(action4_button)
        action4_2_layout.addStretch()
        layoutV.addLayout(action4_2_layout)


        """ Message / Data Pane"""
        # Status Output
        output_label = QLabel("Messages:")
        self.final_output_text = QTextEdit()
        self.final_output_text.setReadOnly(True)
        self.final_output_text.setFixedHeight(100)
        layoutV.addWidget(output_label)
        layoutV.addWidget(self.final_output_text)

        # DataPan table
        datapan_layout = QHBoxLayout()
        self.datapan = QTabWidget()
        datapan_layout.addWidget(self.datapan)
        layoutV.addLayout(datapan_layout)

        # Plot Button
        
        
        """Last Row"""
        last_row_layout = QHBoxLayout()
        plotButton = QPushButton('Plot Selected Rows')
        plotButton.setStyleSheet("""
            QPushButton {
                background-color: #6E82AA; 
                color: white; 
                padding: 10px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #8a9bbd;  /* Slightly lighter color for hover effect */
            }
        """) 
        plotButton.clicked.connect(lambda: self.plot_selected_rows(self.datapan))
        
        export_button = QPushButton('  Export Data')
        export_button.setStyleSheet("""
            QPushButton {
                background-color: #6E82AA; 
                color: white; 
                padding: 10px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #8a9bbd;  /* Slightly lighter color for hover effect */
            }
        """) 
        export_button.setIcon(QIcon(QPixmap(asset('save.png'))))

        export_button.clicked.connect(self.export_data)
        
        next_button = QPushButton('Next')
        next_button.setStyleSheet("""
            QPushButton {
                background-color: #6E82AA; 
                color: white; 
                padding: 10px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #8a9bbd;  /* Slightly lighter color for hover effect */
            }
        """) 
        next_button.clicked.connect(lambda: self.tabWidget.setCurrentIndex(1))  # Switch to "Data" tab
        
        last_row_layout.addWidget(export_button)
        last_row_layout.addWidget(plotButton)
        last_row_layout.addStretch()
        last_row_layout.addWidget(next_button)

        layoutV.addLayout(last_row_layout)
        
        
        tab1.setLayout(layoutV)
        return tab1

    def init_prediction_tab(self):
        tab2 = QWidget()
        tab2.setStyleSheet("background-color: #FFFFFF;")

        layoutGrid = QVBoxLayout()
        prediction_button_layout = QHBoxLayout()
        
        pred_label = QLabel("Prediction:")
        pred_button = QPushButton("Run Prediction")
        pred_button.setStyleSheet("""
            QPushButton {
                background-color: #6E82AA; 
                color: white; 
                padding: 10px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #8a9bbd;  /* Slightly lighter color for hover effect */
            }
        """) 
        pred_button.setIcon(QIcon(QPixmap(asset('run.png'))))
        pred_button.clicked.connect(self.perform_prediction)

        prediction_button_layout.addWidget(pred_label)
        prediction_button_layout.addStretch()
        prediction_button_layout.addWidget(pred_button)
        
        layoutGrid.addLayout(prediction_button_layout)

        self.prediction_table = QTabWidget()

        layoutGrid.addWidget(self.prediction_table)

        save_button_layout = QHBoxLayout()
        
        back_button = QPushButton('Back')
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #6E82AA; 
                color: white; 
                padding: 10px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #8a9bbd;  /* Slightly lighter color for hover effect */
            }
        """) 
        back_button.clicked.connect(lambda: self.tabWidget.setCurrentIndex(0))  # Switch to "Data" tab
        
        
        export_button = QPushButton('  Save Predictions')
        export_button.setStyleSheet("""
            QPushButton {
                background-color: #6E82AA; 
                color: white; 
                padding: 10px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #8a9bbd;  /* Slightly lighter color for hover effect */
            }
        """) 
        export_button.setIcon(QIcon(QPixmap(asset('save.png'))))
        export_button.clicked.connect(self.save)
        
        
        save_button_layout.addWidget(back_button)
        save_button_layout.addStretch()
        save_button_layout.addWidget(export_button)
        

        
        layoutGrid.addLayout(save_button_layout)


        tab2.setLayout(layoutGrid)
        return tab2

    def open_actions_dialog(self):
        self.actions_dialog = AddDialog()
        self.actions_dialog.exec_()

    def open_remove_dialog(self):
        self.remove_dialog = RemoveDialog()
        self.remove_dialog.exec()
    # def perform_prediction(self):
    #     self.final_output_text.append("Performing prediction...")

    # def update_table(self,tablename, df, tab_name):
    #     # Check if the tab already exists
    #     for index in range(tablename.count()):
    #         if tablename.tabText(index) == tab_name:
    #             # If the tab exists, update its content
    
    #             # Get the table widget from the existing tab
    #             table = tablename.widget(index)
    
    #             # Clear existing content
    #             table.setRowCount(0)
    #             table.setColumnCount(0)
    
    #             # Set column headers
    #             table.setColumnCount(len(df.columns))
    #             table.setHorizontalHeaderLabels(df.columns)
    
    #             # Populate the table with data
    #             for row_idx, (_, row) in enumerate(df.iterrows()):
    #                 table.insertRow(row_idx)  # Add a new row
    #                 for col_idx, value in enumerate(row):
    #                     item = QTableWidgetItem()
    #                     item.setData(Qt.EditRole, value)  # Store the original numeric value
    #                     table.setItem(row_idx, col_idx, item)
    
    #             # Exit the function since we found and updated the tab
    #             return
    
    #     # If the tab doesn't exist, create a new tab
    #     table = QTableWidget()
    #     table.setSortingEnabled(True)
    #     table.setRowCount(len(df))
    #     table.setColumnCount(len(df.columns))
    #     table.setHorizontalHeaderLabels(df.columns)
    
    #     for row_idx, (_, row) in enumerate(df.iterrows()):
    #         table.insertRow(row_idx)  # Add a new row
    #         for col_idx, value in enumerate(row):
    #             item = QTableWidgetItem()
    #             item.setData(Qt.EditRole, value)  # Store the original numeric value
    #             table.setItem(row_idx, col_idx, item)
    
    #     tablename.addTab(table, tab_name)

    # def pred_table(self):
    #     start_date = self.start_date_picker.date().toString("yyyy-MM-dd")
    #     start_date = start_date + ' 00:00:00'
    #     print(start_date)
    #     end_date = self.end_date_picker.date().toString("yyyy-MM-dd")
    #     end_date = end_date + ' 23:59:59'
    #     self.selected_pulpeye_data, self.selected_raman_data = build_pred_table(start_date, end_date)
    #     self.final_output_text.append(f"\nSamples between {start_date} and {end_date} are exported\n\n****DONE****")
        
    #     self.update_table(self.datapan, self.selected_pulpeye_data, 'PulpEye Data')
    #     self.update_table(self.datapan, self.selected_raman_data, 'Raman Data')


    # def plot_selected_rows(self, datapan):
    #     current_widget = datapan.currentWidget()
    #     if isinstance(current_widget, QTableWidget):
    #         selected_items = current_widget.selectedItems()
    #         selected_rows = set()
    #         for item in selected_items:
    #             selected_rows.add(item.row())
    #         if not selected_rows:
    #             QMessageBox.warning(self, "No Selection", "Please select some rows to plot.")
    #             return

    #         dates = []
    #         samples = []
    #         for row in selected_rows:
    #             dates.append(pd.to_datetime(current_widget.item(row, 0).text()))
    #             samples.append(int(current_widget.item(row, 1).text()))

    #         if dates and samples:
    #             self.plot(dates, samples)
    #         else:
    #             QMessageBox.warning(self, "No Data", "No data available to plot.")

    def plot(self, dates, samples):
        fig, ax = plt.subplots()
        ax.plot(dates, samples, label='Samples')
        ax.set(xlabel='Date', ylabel='Sample Value', title='Sample Data Over Time')
        ax.legend()
        plt.show()

    def pred_table(self):
        start_date = self.start_date_picker.date().toString("yyyy-MM-dd")
        start_date = start_date + ' 00:00:00'
        print(start_date)
        end_date = self.end_date_picker.date().toString("yyyy-MM-dd")
        end_date = end_date + ' 23:59:59'
        self.selected_pulpeye_data, self.selected_raman_data = build_pred_table(start_date, end_date)
        sample_size = len(self.selected_raman_data)
        self.final_output_text.append(f"\n{sample_size} samples between {start_date} and {end_date} are selected\n\n\n****DONE****")
        
        self.update_table(self.datapan, self.selected_pulpeye_data, 'PulpEye Data')
        self.update_table(self.datapan, self.selected_raman_data, 'Raman Data')


    
    def append_data(self):
        if self.datadir is not None and self.pe_csvf is not None:
            self.final_output_text.append('*** \n Please wait!\n Writing data to the SQL database may take a few minutes for large samples.\n')
            QCoreApplication.processEvents()

            # Create an instance of DataProcessingWorker
            worker = DataProcessingWorker(self.datadir, self.pe_csvf)

            # Connect the worker signals to update the UI when finished
            worker.signals.finished.connect(self.data_processing_finished)

            # Start the worker
            QThreadPool.globalInstance().start(worker)
        else:
            QMessageBox.warning(None, "Warning", "Please select a valid Raman Folder or PulpEye File.")
    def data_processing_finished(self):
        self.final_output_text.append("\n****DONE****")
    

    def perform_prediction(self):
        self.results = gui_prediction()
        # self.results = (self.results)
        self.update_table(self.prediction_table,self.results, 'Results')
        
    
    
    def update_table(self,tablename, df, tab_name):
        
        # Check if the tab already exists
        for index in range(tablename.count()):
            if tablename.tabText(index) == tab_name:
                # If the tab exists, update its content
    
                # Get the table widget from the existing tab
                table = tablename.widget(index)
    
                # Clear existing content
                table.setRowCount(0)
                table.setColumnCount(0)
    
                # Set column headers
                table.setColumnCount(len(df.columns))
                table.setHorizontalHeaderLabels(df.columns)
    
                # Populate the table with data
                for row_idx, (_, row) in enumerate(df.iterrows()):
                    table.insertRow(row_idx)  # Add a new row
                    for col_idx, value in enumerate(row):
                        item = QTableWidgetItem()
                        item.setData(Qt.EditRole, value)  # Store the original numeric value
                        table.setItem(row_idx, col_idx, item)
    
                # Exit the function since we found and updated the tab

                return
    
        # If the tab doesn't exist, create a new tab
        table = QTableWidget()
        table.setSortingEnabled(True)
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(df.columns)
    
        for row_idx, (_, row) in enumerate(df.iterrows()):
            table.insertRow(row_idx)  # Add a new row
            for col_idx, value in enumerate(row):
                item = QTableWidgetItem()
                item.setData(Qt.EditRole, value)  # Store the original numeric value
                table.setItem(row_idx, col_idx, item)
    

        tablename.addTab(table, tab_name)




    def plot_selected_rows(self, tablename):
        # Create new dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Line Chart")
        dialog.resize(800, 600)
        
        layout = QVBoxLayout()

        # Create the Matplotlib widget
        figure = Figure(figsize=(10, 8), dpi=100)
        canvas = FigureCanvas(figure)
        self.ax = figure.add_subplot(111)
        # figure.canvas.mpl_connect('motion_notify_event', self.on_hover)

        layout.addWidget(canvas)
        # Add toolbar
        toolbar = NavigationToolbar(canvas, dialog)
        layout.addWidget(toolbar)
        current_tab_index = tablename.currentIndex()
        if current_tab_index == -1:  # No tabs present
            return
        
        table = tablename.widget(current_tab_index)
        
        # Identify selected rows and collect data
        selected_rows = list(set(index.row() for index in table.selectedIndexes()))

        # Initialize a list to hold all the values
        all_values = []
        labels = []

        # Populate the list with values
        for row in selected_rows:
            values = []
            prop = []
            labels.append(table.item(row,0).text())

            for col in range(1, table.columnCount()):  # Skip the first column (sample name)
                item = table.item(row, col)
                if item:
                    prop.append(table.horizontalHeaderItem(col).text())
                    try:
                        values.append(float(item.text()))
                    except:
                        prop.pop()
            all_values.append(values)
        
        # Convert the list to a NumPy array
        all_values_array = np.array(all_values)
        
        # Normalize each column
        max_values = np.max(all_values_array, axis=0)
        min_values = np.min(all_values_array, axis=0)
        range_values = max_values - min_values
        range_values[range_values == 0] = 1  # Prevent division by zero
        
        # normalized_values_array = (all_values_array - min_values) / range_values
        normalized_values_array = all_values_array

        # Plot the normalized values
        for i, values in enumerate(normalized_values_array):
            # self.ax.plot(prop, values, marker='o', label=f'{labels[i]}')
            line, = self.ax.plot(prop, values, label=f'{labels[i]}')  # Capture the line object
            line.set_gid(labels[i])  # Set the gid to the label
        # Add legend and labels
        self.ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        figure.subplots_adjust(right=0.8)  # Adjust the right edge to make room for the legend
        # ax.set_xlabel('Properties')
        self.ax.set_ylabel(' Intesity ')
        self.ax.set_title('Line Plot')
        self.ax.set_xticklabels(prop,rotation = 90)
        self.ax.set_xticks(np.arange(len(prop)))
        figure.tight_layout()


    

        dialog.setLayout(layout)
        dialog.exec_()
        


    def save(self):
        current_datetime = date.today()
        date_str = current_datetime.strftime('%Y-%m-%d')
        date_str = date_str.replace('-','_')
        default_file_name = f"Prediction_{date_str}.xlsx"
    
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
    
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", default_file_name, "Excel Files (*.xlsx);;All Files (*)", options=options)
    
        if file_path:
            if not file_path.endswith('.xlsx'):
                file_path += '.xlsx'
    
            with pd.ExcelWriter(file_path) as writer:
                # for df in self.results:
                sheet = 'UBC_Predictions'
                self.results.to_excel(writer, sheet_name=sheet, index=False)
    
            QMessageBox.information(self, "Saved", f"File saved as {file_path} successfully")
            
    def export_data(self):
        current_datetime = date.today()
        date_str = current_datetime.strftime('%Y-%m-%d')
        date_str = date_str.replace('-','_')
        default_file_name = f"Database_{date_str}.xlsx"
    
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
    
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Data", default_file_name, "Excel Files (*.xlsx);;All Files (*)", options=options)
    
        if file_path:
            if not file_path.endswith('.xlsx'):
                file_path += '.xlsx'
    
            with pd.ExcelWriter(file_path) as writer:
                # for df in self.results:
                sheet = 'PulpEye'
                self.selected_pulpeye_data.to_excel(writer, sheet_name=sheet, index=False)
                sheet = 'Raman'
                self.selected_raman_data.to_excel(writer, sheet_name=sheet, index=False)
    
            QMessageBox.information(self, "Saved", f"File saved as {file_path} successfully")
            

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

    
    

