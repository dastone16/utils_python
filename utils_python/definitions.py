def print_test():
  print("Hello World!")
  
def data_description(data,label):
  description = data.describe().transpose()
  #print(description)
  description.to_csv(save_path + label + '.csv')
  
line_list = [] #dummy list,
def Timeplot(data1,label,savepath=save_path,show=False):
  """
  Outputs timeplot of each ? in data
  Automatically separates each into separate folders.
  """
  data1.columns = data1.columns.str.replace(r"[/]","_",regex=True) #remove issue causing '/' to column label.
  for each in data1:
    ax = sns.lineplot(data=data1,x=data1.index,y=data1[each])
    ax.tick_params(axis='x',which='major',length=5)
    # loc=plticker.MultipleLocator(base=3)
    # ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_locator(plticker.MaxNLocator(4))
    plt.title(label +" "+ each)
    plt.xticks(rotation=45, ha='right')
    fig1 = plt.gcf()
    #plt.show() #not necessary slows down remaining 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig1.savefig(save_path+"/"+str(each)+"/"  + ' time plot.jpg',bbox_inches='tight')
    if show==False:
      plt.close()
  data1.columns = data1.columns.str.replace(r"[_]","/",regex=True)#returns to original format of Date/Time


def data_description(data,label,savepath=save_path):
  """
  Outputs python .describe() info 
  Takes exact save_path unaltered.
  """
  description = data.describe().transpose()
  print(label + " data description saved ...")
  os.makedirs(os.path.dirname(savepath), exist_ok=True)
  description.to_csv(savepath + label + '.csv')

#Save but likely not necessary
def mfile_import(path,columns,date_name,dateformat="%d/%m/%Y %H:%M:%S"):
  """
  Imports all files from set path.
  Not utilized anymore due to typically importing larger weekly data files.
  """
  #Date formating help: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
  filepaths = [f for f in listdir(path) if f.endswith('.csv')]
  output_data = pd.DataFrame()
  for file in filepaths:
    temp_data= pd.read_csv(path + file,header=0,usecols=columns,parse_dates=[date_name]) 
    temp_data[date_name]= pd.to_datetime(temp_data[date_name].dt.strftime(dateformat)) #needed to shift unique format
    output_data = pd.concat([output_data,temp_data])
   
  return output_data

def gun_corr(data,spraygun,savepath=save_path,show=False):
  """
  HOW DOES THIS DIFFER from correlation_plot?
  Check for deletion
  """
  Spray_NN = pd.pivot_table(data[data['Machine']==spraygun],columns='Variable',values='Value by Gun',index=date_label)
  Spray_NN.dropna()
  Spray_NN_Day = Spray_NN.resample('D').mean()
  Spray_NN_table = Spray_NN_Day.corr()
  Spray_NN_table.to_csv(savepath + spraygun + 'gun.correlation.csv')
  x_len,y_len=Spray_NN_table.shape

  f = plt.figure(figsize=(10, 10))
  plt.matshow(Spray_NN_table, fignum=f.number)
  plt.xticks(range(Spray_NN_table.shape[1]), Spray_NN_table.columns, fontsize=12, rotation=90)
  plt.yticks(range(Spray_NN_table.shape[1]), Spray_NN_table.columns, fontsize=12)
  cb = plt.clim(-1,1) #forced limits to -1,1.
  cb = plt.colorbar()
  for i in range(x_len):
    for j in range(y_len):
        text = plt.text(j, i, round(Spray_NN_table.iloc[i, j],2), ha='center', va='center', color='r')
  cb.ax.tick_params(labelsize=16)
  plt.title(spraygun, fontsize=18,horizontalalignment = 'center',y=1.20);
  if show==False:
    plt.close()
  os.makedirs(os.path.dirname(savepath ), exist_ok=True)
  f.savefig(savepath + spraygun + 'correlation.plot.jpg')

#creates scatter plot 
def scatter_plot(data,plotx, ploty,Line=None, savepath=save_path, data_fit=False):
  """
  Simply scatter plot from matplotlib.

  data: dataframe with x, y in columns
  plotx: str, column name of x-axis
  ploty: str, column name of y-axis

  save_path: str, path of save target.  Default global save_path.

  """
  f= plt.figure()
  #ax = fig.gca(autoscalex_on=True)
  plt.scatter(data[plotx],data[ploty])
  #plt.yscale('log')
  plt.ylabel(ploty)
  plt.xlabel(plotx)
  #z = np.polyfit(data[plotx], data[ploty], 1)
  #p = np.poly1d(z)
  #plt.plot(data[plotx],p(data[plotx]),"r--",)
  os.makedirs(os.path.dirname(savepath+'scatter_plot/'), exist_ok=True)
  f.savefig(savepath+ Line + ploty + ' vs.' + plotx + '.scatter.jpg')
  if data_fit:
    X = data[plotx]
    Y = data[ploty]

    X = sm.add_constant(X) # adding a constant

    model = sm.OLS(Y, X, missing='drop').fit()
    predictions = model.predict(X) 

    print_model = model.summary()
  return f

def correlation_plot(data,resample,output_name,size=(18,12),font=18,plot=True,show=False,all_factors=True,corr_variable='Metal Exposure (mA)',labels=False,savepath=save_path):
  """
  data: df, data frame with data
  resample: str, resampling rate for correlation
  output_name: str, outplut label for picture and .csv
  size: 2 value list, Size of output figure
  plot: boolean, Create plot and save
  show: boolean, show plot in output
  all_factors: boolean, make correlation for all factors and save.
  correlation_variable: str, primary correlation variable
  labels: boolean, include labels in graph
  savepath: path to save pictures and files.  *orrelation_plots/ added to savepath
  """
  if corr_variable not in data.columns:
    print("{} missing from {} data".format(corr_variable,output_name))
    return
  
  
  data_corr = data.resample(resample).mean().corr()
  data_corr = data_corr.dropna(how='all')
  data_corr = data_corr.dropna(how='all',axis='columns')
  try:
    if plot:
      f = plt.figure(figsize=size)
      plt.matshow(data_corr, fignum=f.number)
      plt.xticks(range(data_corr.shape[1]), data_corr.columns, fontsize=font, rotation=90)
      plt.yticks(range(data_corr.shape[1]), data_corr.columns, fontsize=font)
      cb = plt.clim(-1,1) 
      cb = plt.colorbar()
      cb.ax.tick_params(labelsize=14)
      cb.ax.set_title(output_name)
      #plt.title('Correlation Matrix L12', fontsize=16);
      if labels:
        for i in range(data_corr.shape[1]):
          for j in range(data_corr.shape[0]):
            text = plt.text(j, i, round(data_corr.iloc[i, j],2), ha='center', va='center', color='k')
      if show==False:
        plt.close()      
      os.makedirs(os.path.dirname(savepath ), exist_ok=True)
      f.savefig(savepath + output_name +'.jpg')
      #print(output_name +" correlation plot saved...")
  except:
    pass  
  try:
    if all_factors:
      # correlation between all factors.
      data_corr_table =  data_corr.stack().reset_index(level=0)
      data_corr_table.rename_axis(index={'Variable':'Factor1'},inplace=True)
      data_corr_table.columns = ['Factor2','Correlation']
      data_corr_table.reset_index(level=0,inplace=True)
      data_corr_table.drop(data_corr_table[data_corr_table['Factor1']==data_corr_table['Factor2']].index,inplace=True)
      
      data_corr_table = data_corr_table.iloc[(-np.abs(data_corr_table.Correlation.values)).argsort()]
      data_corr_table = data_corr_table.drop_duplicates('Correlation')
      data_corr_table.to_csv(savepath + output_name + 'Full.correlation.Table.csv')
      #Defined correlation
      data_corr_ma = data_corr.loc[[corr_variable],:]
      data_corr_ma = data_corr_ma.transpose()
      #data_corr_ma
      data_mA_table = data_corr_ma.iloc[(-np.abs(data_corr_ma[corr_variable].values)).argsort()]
      data_mA_table.rename(columns={corr_variable:'Correlation to '+ corr_variable}, inplace=True)
      #data_mA_table.head()
      os.makedirs(os.path.dirname(savepath), exist_ok=True)
      data_mA_table.to_csv(savepath + output_name + 'correlation.to.'+corr_variable+'.csv')
      #print(output_name +" correlation plot saved to csv...")
  except:
    return

def date_compare(data,date_label,start_date,trans_date,fin_date,
                 output_name,resample='H',label1="Before",label2="After",corr="Metal Exposure (mA)",
                 raw=False,savepath=save_path):
  """
  Data must be dataframe indexed to date.
  resample is time of resampling typically daily 'D', rarely hourly 'H'
  data parsed from start to finish with transition date in middle
  corr: correlation to.  If none correlate to all.
  raw=False will not prettify output columns.
  """
  Summary=pd.DataFrame()
  
  #reindex by date
  data2 = data.resample(resample).mean() #resample by second to get full statistics 
  
  #section by date
  before = data2[start_date:trans_date].copy()
  after = data2[trans_date:fin_date].copy()
  
  correlation_plot(before,'D',output_name+label1,plot=False,all_factors=False)
  correlation_plot(after,'D',output_name+label2,plot=False,all_factors=False)
  before_T = before.transpose()
  #+ " " + str(trans_date)+
  Summary[label1 + ' Mean'] = np.round(before_T.mean(axis=1),4)
  Summary[label1 + ' Std1'] = before_T.std(axis=1)
  Summary[label1 + ' Count1'] = before_T.count(axis=1)
  Summary[label1 + 'StdError'] = sp.sem(before_T,nan_policy='omit', axis=1)
  after_T = after.transpose()
  
  Summary[label2 +' Mean'] = np.round(after_T.mean(axis=1),4)
  Summary[label2 + ' Std2'] = after_T.std(axis=1)
  Summary[label2 + ' Count2'] = after_T.count(axis=1)
  Summary[label2 + 'StdError'] = sp.sem(after_T,nan_policy='omit', axis=1)

  Summary['Difference']=np.round((Summary[label1 + ' Mean'] - Summary[label2 +' Mean']),4)
  Summary['Std Error Diff'] = np.sqrt((Summary[label1 +'StdError']**2) + (Summary[label2 +'StdError']**2))
  Summary['df']=Summary[label1 + ' Count1']+Summary[label2 + ' Count2']-2
  Summary['T-Stat'] = Summary['Difference']/Summary['Std Error Diff']
  Summary['P-value'] = np.round((1.0 - sp.t.cdf(abs(Summary['T-Stat']),Summary['df']))*2.0,3)
  #Summary[label1 + ' Correlation to mA']=before_D_corr.loc[['Metal Exposure (mA)'],:]
  #Summary[label2 + ' Correlation to mA']=after_D_corr.loc[['Metal Exposure (mA)'],:]
  Summary = Summary.iloc[(-np.abs(Summary['T-Stat'].values)).argsort()]
  if raw == False:
    Summary.drop(columns=[label1 + ' Std1',label1 + ' Count1',label1 + 'StdError',
                          label2 + ' Std2',label2 + ' Count2',label2 + 'StdError',
                          'Std Error Diff','df'], inplace=True)
    Summary.to_csv(savepath + output_name+label1+label2+'.diff.csv')
  else:
    Summary.to_csv(savepath + output_name+label1+label2+'.diff.csv')
    pass
  return Summary

  
  
def high_low_days(data,resampler,Output_name,variable,days=10,savepath=save_path):
  """
  #input dataframe with index by day, preferrably resampled by hour. Columns of all variables.

  #originally used for only mA, switched to any variable, but temporary variables inside function still refer to mA.
  """
  data_Day = data.resample(resampler).mean()
  data_Day = data_Day.dropna(how='all')
  data_Day = data_Day.dropna(how='all',axis='columns')
  #data_Day[variable] = data_Day[data_Day[variable]!=0]
  data_Day = data_Day.iloc[(-np.abs(data_Day[variable].values)).argsort()]
  data_High20_mA_Days = data_Day.iloc[0:days,:]
  data_Day2 = data_Day.loc[data_Day[variable]>0,:]
  data_Low20_mA_Days = data_Day2.iloc[-days-1:-1,:] #at 1 day this is empty, so subtract 1 because non-inclusive.
  # dp =0 #non-zero data point count
  # i=-1
  # while dp <days: 
  #   if data_Day.iloc[i-1:i,:][variable]!=0: #remove zero data points from OMS data
  #     data_Low20_mA_Days = data_Day.iloc[i-1:i,:]
  #     dp+=1
  #   else:
  #     i-=1

  #labeled mA because originally only written for mA.  Added ability to input vs any variable.

  data_High_mean = data_High20_mA_Days.resample('10AS').mean() #resample by year to average
  data_High_mean = data_High_mean.reset_index(drop=True).transpose()
  data_Low_mean = data_Low20_mA_Days.resample('10AS').mean()
  data_Low_mean = data_Low_mean.reset_index(drop=True).transpose()
  data_Range_max = data_Day.resample('10AS').max()  # by year to get entire range
  data_Range_max = data_Range_max.reset_index(drop=True).transpose()
  data_Range_min = data_Day.resample('10AS').min()
  data_Range_min = data_Range_min.reset_index(drop=True).transpose()
  data_High_mean.rename(columns={0:'Avg: '+str(days)+' Highest Days'}, inplace=True)
  data_Low_mean.rename(columns={0:'Avg: '+str(days)+' Lowest Days'}, inplace=True)
  data_Range_max.rename(columns={0:'Max'}, inplace=True)
  data_Range_min.rename(columns={0:'Min'}, inplace=True)

  data_Summary = data_Low_mean
  data_Summary['Avg: '+str(days)+' Highest Days'] = data_High_mean['Avg: '+str(days)+' Highest Days']
  data_Summary['Difference'] = data_High_mean['Avg: '+str(days)+' Highest Days']-data_Low_mean['Avg: '+str(days)+' Lowest Days']
  data_Summary['Max'] = data_Range_max['Max']
  data_Summary['Min'] = data_Range_min['Min']
  data_Summary['Range'] = data_Summary['Max'] - data_Summary['Min']
  data_Summary['% of Range'] = data_Summary['Difference']/data_Summary['Range']*100
  data_Summary = data_Summary.iloc[(-np.abs(data_Summary['% of Range'].values)).argsort()]
  os.makedirs(os.path.dirname(savepath), exist_ok=True)
  data_Summary.to_csv(savepath + Output_name + variable + str(days)+'days.High.Low.diff.csv')
  return data_Summary


#Elapsed color definitions
def Elapse_cell_color(val):
  if val < 22:
    color= 'rgb(146,208,80)'#green
  elif val <27:
    color = 'yellow'
  elif math.isnan(val):
    color = 'white' 
  else:
    color = 'rgb(225,45,45)'

  return 'background-color: %s' % color
def Elapse_text_color(val):
  if math.isnan(val):
    color = 'white'
  else:
    color = 'black'
  return 'color: %s' % color

def mA_cell_color(val):
  if val < 0.2:
    color= 'rgb(146,208,80)'#green
  elif val <0.5:
    color = 'yellow'
  elif math.isnan(val):
    color = 'white' 
  else:
    color = 'rgb(225,45,45)'
  return 'background-color: %s' % color
def mA_text_color(val):
  if math.isnan(val):
    color = 'white'
  else:
    color = 'black'
  return 'color: %s' % color

def OOS_cell_color(val):
  if val < 0.1:
    color= 'rgb(146,208,80)'#green
  elif val <0.2:
    color = 'yellow'
  elif val <0.3:
    color = 'rgb(225,45,45)'#red
  elif math.isnan(val):
    color = 'white' 
  else:
    color = 'rgb(0,0,0)'#black
  return 'background-color: %s' % color
def OOS_text_color(val):
  if math.isnan(val):
    color = 'white'
  elif val>=.3:
    color = 'white'
  else:
    color = 'black'
  return 'color: %s' % color

def elapsed(data,date_start,group=['Plant','Line'],variable_name='Variable',variable="Metal Exposure (mA)",multilevel=True):
  """
  data=dataframe not pivot
  group = how to group data in output
  Returns: elapsed value by group,  
  """
  data_L1 = data[data[date_label]>date_start].copy() #greater means closer to today
  data_L1 = data_L1.sort_values(by=date_label)
  data_L1 = data_L1[data_L1[variable_name]==variable].copy() #variable_name makes this generic.
  data_L1_S = data_L1[data_L1['Sample No']==1].copy()#S= first measurement

  data_L1_S.sort_values(group,inplace=True,ascending=True)
  data_L1_S['Lagged_Time']=data_L1_S[date_label].shift(1)
  data_L1_S.dropna(inplace=True)
  data_L1_S['Elapsed Time'] = (data_L1_S[date_label] - data_L1_S['Lagged_Time']).astype("timedelta64[m]")
  #for multilevel
  if multilevel:
    elapsed_by = data_L1_S.groupby(group)['Elapsed Time'].median().unstack().sort_index(axis=1)
  else:
    elapsed_by = data_L1_S.groupby(group)['Elapsed Time'].median()
    elapsed_by = pd.DataFrame(elapsed_by)
  #elapsed_by = data_L1_S.groupby(group)['Elapsed Time'].median()
  #elapsed_by

  elapsed_by_s = elapsed_by.style.applymap(Elapse_cell_color)\
                  .format("{:3.1f}")\
                  .set_properties(**{'width':'100px', 'text-align':'center'})\
                  .applymap(Elapse_text_color)\
                  .set_table_styles([{'selector': 'th','props': [('background-color', 'rgb(255,242,204)'),('border-style','solid'),('border-width','1px'),('color', 'black')]}])
  elapsed_by_s.index.name = None
    
  return elapsed_by, elapsed_by_s

def subplotdata (data,yaxis,yaxislabels,limit,title,save_label,after_date=None,font_size=12,title_size=14,savepath=save_path,lines=None,show=False):
  #still need to figure out how to increase tick size.
  """
  #data is a list of data frames to plot
  #yaxis is a list of columns used for plotting
  #yaxislabels is a list of the y-axis in the plot
  #limited to 11 lines.  
  #limit is boolean for limits adjusted for plots
  #title is title of all plot
  #save label is name of file to be saved.
  #after date
  """
  for each in data:#checks data for empty dataframes
    if len(each)==0:
      print("Data input is empty")
      return
  data = [each for each in data if not len(each) == 0] #Removes empty data

  #x_len=len(yaxis)
  x_len=len(data) #number of data files for y-axis count (miss labelled)
  y_len=len(data[0].groupby(level=0,observed=True)) #length of first dataframe index, level 0.  Chart will be split by 0 level of multi-index. 
  if lines==None:
    lines = list(data[0].index.get_level_values(0).unique())

  #print("x_len "+ str(x_len)) #troubleshooting
  #print("y_len "+ str(y_len))
  colors = ['orange','blue','green','red','purple','gray','brown','pink','olive','cyan','black']
  fig, axs = plt.subplots(x_len,y_len,sharey='row',sharex='col',figsize=(12,12))

  if y_len>1:
    for j in range(x_len): #setting reasonable limits
      axs[j,0].set_ylabel(yaxislabels[j],fontsize=font_size)
      if limit[j]:
        if data[j][yaxis[j]].max()<np.inf:
          axs[j,0].set_ylim(data[j][yaxis[j]].min()-(0.2*data[j][yaxis[j]].min()),data[j][yaxis[j]].max()+(0.1*data[j][yaxis[j]].max()))
      for i in range(y_len):
        axs[j,i].set_xlabel('',fontsize=font_size)
        axs[j,i].tick_params(axis='both', which='major',labelsize=font_size)
        #axs[j,i].yticks(fontsize=font_size)
        inames = [None]*len(data[j].index.names)
        data[j].index.set_names(inames,inplace=True)
        try:
          #print(j, i, lines[i])#Troubleshooting
          data[j].loc[[lines[i]],[yaxis[j]]].plot(kind='bar', ax=axs[j,i],legend=None,color=colors[i],label='')
          #plt.xticks(fontsize=font_size)
          #plt.yticks(fontsize=font_size)
        except KeyError:
          continue
        except IndexError:
          print("Warning IndexError:")
          continue
    axs[0,y_len//2].set_title(title,fontsize=title_size)
  if y_len==1:
    for j in range(x_len): #setting reasonable limits
      axs[j].set_ylabel(yaxislabels[j],fontsize=font_size)
      if limit[j]:
        if data[j][yaxis[j]].max()<np.inf:
          axs[j].set_ylim(data[j][yaxis[j]].min()-(0.2*data[j][yaxis[j]].min()),data[j][yaxis[j]].max()+(0.1*data[j][yaxis[j]].max()))
      i=0
      axs[j].set_xlabel('',fontsize=font_size)
      axs[j].tick_params(axis='both', which='major',labelsize=font_size)
      #axs[j,i].yticks(fontsize=font_size)
      inames = [None]*len(data[j].index.names)
      data[j].index.set_names(inames,inplace=True)
      try:
        #print(j, i, lines[i])#Troubleshooting
        data[j].loc[[lines[i]],[yaxis[j]]].plot(kind='bar', ax=axs[j],legend=None,color=colors[i],label='')
        #plt.xticks(fontsize=font_size)
        #plt.yticks(fontsize=font_size)
      except KeyError:
        continue
      except IndexError:
        print("Warning IndexError:")
        continue
    axs[0].set_title(title,fontsize=title_size)
  os.makedirs(os.path.dirname(savepath), exist_ok=True)
  fig.savefig(savepath + save_label)
  if show==False:
    plt.close() #prevents plot from outputting to output
  #return print(save_label + " subplot complete...")

def sizeof_fmt(num, suffix='B'):#needed for memory check
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def memorycheck():#can't get to output list of high memory variables.
  result ="\n"

  for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                          key= lambda x: -x[1])[:20]:
      result += ("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
      result += "\n"
  return result

#Control chart functions are pulled from control_chart)reference pickle file
#Located:/content/drive/MyDrive/Programming/Ardagh_Data_Analysis/control_function_reference.pkl'
# Each function checks if control limit function has been calculated prior and at what iteration level.
# If calcuation is available at required iteration level then it will pull from reference file. 
# If request is a new sample size or higher iteration a new calculation will be completed and saved to file.

cf_location='/content/drive/MyDrive/Programming/Ardagh_Data_Analysis/control_function_reference.pkl'

def clf_d2(subgroup_size, iterations=100000):
  """
  Shewart Control Limit Function d2
  defined as the expected value (d2) of the range of a sample from a normal population with unit standard deviation

  subgroup_size: integer of subgroup size of measurements
  """
  control_function_reference = pd.read_pickle(cf_location)
  if subgroup_size<1:
    return print("d2 Sugroup size must be greater than zero")
  try:
    if ((subgroup_size in set(control_function_reference.index)) &\
    (control_function_reference.loc[subgroup_size,'d2_it']>=iterations)):
      return control_function_reference.loc[subgroup_size,'d2']
    else:
      a2cl = []
      i=0
      while i <= iterations:
        sample = np.random.normal(0,1,subgroup_size)
        a2cl.append(np.max(sample)-np.min(sample))
        i+=1
      control_function_reference.loc[subgroup_size,'d2'] = np.mean(a2cl)
      control_function_reference.loc[subgroup_size,'d2_it'] = iterations
      control_function_reference.to_pickle(cf_location)
      return np.mean(a2cl)
  except KeyError:#If subgroup_size doesn't exist it will make new entr
    a2cl = []
    i=0
    while i <= iterations:
      sample = np.random.normal(0,1,subgroup_size)
      a2cl.append(np.max(sample)-np.min(sample))
      i+=1
    control_function_reference.loc[subgroup_size,'d2'] = np.mean(a2cl)
    control_function_reference.loc[subgroup_size,'d2_it'] = iterations
    control_function_reference.to_pickle(cf_location)
    return np.mean(a2cl)

def clf_d3(subgroup_size, iterations=100000):
  """
  Shewart Control Limit Function d3
  defined as the standard deviation (d3) of the range of a sample from a normal population with unit standard deviation
  
  subgroup_size: integer of subgroup size of measurements

  iterations: random simulations to run to get constant 10k good
  """
  control_function_reference = pd.read_pickle(cf_location)

  if subgroup_size<2:
    return print("d3 Sugroup size must be greater than 1")
  try:
    if ((subgroup_size in set(control_function_reference.index)) &\
      (control_function_reference.loc[subgroup_size,'d3_it']>=iterations)):
      return control_function_reference.loc[subgroup_size,'d3']
    else:
      a2cl = []
      i=0
      while i < iterations:
        sample = np.random.normal(0,1,subgroup_size)
        a2cl.append(np.max(sample)-np.min(sample))
        i+=1
      control_function_reference.loc[subgroup_size,'d3'] = np.std(a2cl)
      control_function_reference.loc[subgroup_size,'d3_it'] = iterations
      control_function_reference.to_pickle(cf_location)
      return np.std(a2cl)
  except KeyError:#If subgroup_size doesn't exist it will make new entr
    a2cl = []
    i=0
    while i < iterations:
      sample = np.random.normal(0,1,subgroup_size)
      a2cl.append(np.max(sample)-np.min(sample))
      i+=1
    control_function_reference.loc[subgroup_size,'d3'] = np.std(a2cl)
    control_function_reference.loc[subgroup_size,'d3_it'] = iterations
    control_function_reference.to_pickle(cf_location)
    return np.std(a2cl)


def clf_c4(subgroup_size):
  """
  Shewart Control Limit Function c4
  defined as the expected value (c4) of the standard deviation of a sample from a normal population with unit standard deviation
  
  subgroup_size: integer of subgroup size of measurements

  c4 = gamma (n/2)*sqrt(2/(n-1))/(gamma((n-1)/2))
  where gamma = (n-1!)
  """
  #could not simulate using monte carlo.  Used formula above
  control_function_reference = pd.read_pickle(cf_location)

  if subgroup_size<2:
    return print("Subgroup size must be greater than 1.")
  try:
    if ((subgroup_size in set(control_function_reference.index)) & (pd.notna(control_function_reference.loc[subgroup_size,'C4']))):
      return control_function_reference.loc[subgroup_size,'c4']
    else:
      s = subgroup_size #for clarity below
      control_function_reference.loc[subgroup_size,'c4'] = (math.gamma(s/2)*np.sqrt(2/(s-1))/(math.gamma((s-1)/2)))
      control_function_reference.loc[subgroup_size,'c4_it'] = np.inf
      control_function_reference.to_pickle(cf_location)
      return (math.gamma(s/2)*np.sqrt(2/(s-1))/(math.gamma((s-1)/2)))
  except KeyError:#If subgroup_size doesn't exist it will make new entr)
    s = subgroup_size #for clarity below
    control_function_reference.loc[subgroup_size,'c4'] = (math.gamma(s/2)*np.sqrt(2/(s-1))/(math.gamma((s-1)/2)))
    control_function_reference.loc[subgroup_size,'c4_it'] = np.inf
    control_function_reference.to_pickle(cf_location)
    return (math.gamma(s/2)*np.sqrt(2/(s-1))/(math.gamma((s-1)/2)))  



def clf_A2(subgroup_size,iterations=100000):
  """
  Shewart Control Limit Function A2
  
  Formula: 3/ (d2n * sqrt(subgroup_size))

  subgroup_size: integer of subgroup size of measurements
  """  
  control_function_reference = pd.read_pickle(cf_location)

  if subgroup_size<1:
    return print("A2 Sugroup size must be greater than zero")
  try:
    if ((subgroup_size in set(control_function_reference.index)) &\
      (control_function_reference.loc[subgroup_size,'A2_it']>=iterations)):
      return control_function_reference.loc[subgroup_size,'A2']
    else:

      control_function_reference.loc[subgroup_size,'A2'] = (3/(clf_d2(subgroup_size,iterations)*np.sqrt(subgroup_size)))
      control_function_reference.loc[subgroup_size,'A2_it'] = iterations
      control_function_reference.to_pickle(cf_location)
      return (3/(clf_d2(subgroup_size,iterations)*np.sqrt(subgroup_size)))
  except KeyError:#If subgroup_size doesn't exist it will make new entr)

    control_function_reference.loc[subgroup_size,'A2'] = (3/(clf_d2(subgroup_size,iterations)*np.sqrt(subgroup_size)))
    control_function_reference.loc[subgroup_size,'A2_it'] = iterations
    control_function_reference.to_pickle(cf_location)
    return (3/(clf_d2(subgroup_size,iterations)*np.sqrt(subgroup_size)))  
  

def clf_D3(subgroup_size,iterations=100000, sigma_limit=3):
  """
  Shewart Control Limit Function D3
  
  Formula: max(0,1-kd3/d2)
  k = sigma level (almost alway 3)
  d3 = control limit function for d3
  d2 = control limit function for d2

  subgroup_size: integer of subgroup size of measurements
  """ 
  control_function_reference = pd.read_pickle(cf_location)

  if subgroup_size<1:
    return print("D3 Sugroup size must be greater than zero")
  try:
    if ((subgroup_size in set(control_function_reference.index)) &\
      (control_function_reference.loc[subgroup_size,'D3_it']>=iterations)):
      return control_function_reference.loc[subgroup_size,'D3']
    else:

      control_function_reference.loc[subgroup_size,'D3'] = max(0,(1-(sigma_limit*clf_d3(subgroup_size,iterations)/clf_d2(subgroup_size,iterations))))
      control_function_reference.loc[subgroup_size,'D3_it'] = iterations
      control_function_reference.to_pickle(cf_location)
      return max(0,(1-(sigma_limit*clf_d3(subgroup_size,iterations)/clf_d2(subgroup_size,iterations))))
  except KeyError:#If subgroup_size doesn't exist it will make new entr)

    control_function_reference.loc[subgroup_size,'D3'] = max(0,(1-(sigma_limit*clf_d3(subgroup_size,iterations)/clf_d2(subgroup_size,iterations))))
    control_function_reference.loc[subgroup_size,'D3_it'] = iterations
    control_function_reference.to_pickle(cf_location)
    return max(0,(1-(sigma_limit*clf_d3(subgroup_size,iterations)/clf_d2(subgroup_size,iterations))))    
  


def clf_D4(subgroup_size,iterations=100000, sigma_limit=3):
  """
  Shewart Control Limit Function D4
  
  Formula: 1+kd3/d2)
  k = sigma level (almost alway 3)
  d3 = control limit function for d3
  d2 = control limit function for d2

  subgroup_size: integer of subgroup size of measurements
  """ 
  control_function_reference = pd.read_pickle(cf_location)

  if subgroup_size<1:
    return print("D4 Sugroup size must be greater than zero")
  try:
    if ((subgroup_size in set(control_function_reference.index)) &\
      (control_function_reference.loc[subgroup_size,'D4_it']>=iterations)):
      return control_function_reference.loc[subgroup_size,'D4']
    else:

      control_function_reference.loc[subgroup_size,'D4'] = 1+(sigma_limit*clf_d3(subgroup_size,iterations)/clf_d2(subgroup_size,iterations))
      control_function_reference.loc[subgroup_size,'D4_it'] = iterations
      control_function_reference.to_pickle(cf_location)
      return 1+(sigma_limit*clf_d3(subgroup_size,iterations)/clf_d2(subgroup_size,iterations))
  except KeyError:#If subgroup_size doesn't exist it will make new entr)

    control_function_reference.loc[subgroup_size,'D4'] = 1+(sigma_limit*clf_d3(subgroup_size,iterations)/clf_d2(subgroup_size,iterations))
    control_function_reference.loc[subgroup_size,'D4_it'] = iterations
    control_function_reference.to_pickle(cf_location)
    return 1+(sigma_limit*clf_d3(subgroup_size,iterations)/clf_d2(subgroup_size,iterations))  


def clf_A3(subgroup_size):
  """
  Shewart Control Limit Function A3
  Used for Xbar-s chart
  Formula: k/ (c4n * sqrt(subgroup_size))
  k = sigma level (almost always 3)
  subgroup_size: integer of subgroup size of measurements
  """  
  control_function_reference = pd.read_pickle(cf_location)

  if subgroup_size<2:
    return print("Subgroup size must be greater than 1.")
  try:
    if ((subgroup_size in set(control_function_reference.index))&(pd.notna(control_function_reference.loc[subgroup_size,'A3']))):
      return control_function_reference.loc[subgroup_size,'A3']
    else:
      control_function_reference.loc[subgroup_size,'A3'] = (3/(clf_c4(subgroup_size)*np.sqrt(subgroup_size)))
      control_function_reference.loc[subgroup_size,'A3_it'] = np.inf
      control_function_reference.to_pickle(cf_location)
      return (3/(clf_c4(subgroup_size)*np.sqrt(subgroup_size)))
  except KeyError:#If subgroup_size doesn't exist it will make new entr)
    control_function_reference.loc[subgroup_size,'A3'] = (3/(clf_c4(subgroup_size)*np.sqrt(subgroup_size)))
    control_function_reference.loc[subgroup_size,'A3_it'] = np.inf
    control_function_reference.to_pickle(cf_location)
    return (3/(clf_c4(subgroup_size)*np.sqrt(subgroup_size)))



def clf_B3(subgroup_size):
  """
  Shewart Control Limit Function B3
  Used for Xbar-s chart
  Formula: max(0,1-(k/c4)*sqrt(1-c4^2))

  k = sigma level (almost always 3)
  subgroup_size: integer of subgroup size of measurements
  """ 
  control_function_reference = pd.read_pickle(cf_location)

  if subgroup_size<2:
    return print("Subgroup size must be greater than 1.")
  try:
    if ((subgroup_size in set(control_function_reference.index))&(pd.notna(control_function_reference.loc[subgroup_size,'B3']))):
      return control_function_reference.loc[subgroup_size,'B3']
    else:
      s = subgroup_size
      control_function_reference.loc[subgroup_size,'B3'] = max(0,(1-(3/clf_c4(s))*np.sqrt(1-np.power(clf_c4(s),2))))
      control_function_reference.loc[subgroup_size,'B3_it'] = np.inf
      control_function_reference.to_pickle(cf_location)
      return max(0,(1-(3/clf_c4(s))*np.sqrt(1-np.power(clf_c4(s),2))))
  except KeyError:#If subgroup_size doesn't exist it will make new entr)
    s = subgroup_size
    control_function_reference.loc[subgroup_size,'B3'] = max(0,(1-(3/clf_c4(s))*np.sqrt(1-np.power(clf_c4(s),2))))
    control_function_reference.loc[subgroup_size,'B3_it'] = np.inf
    control_function_reference.to_pickle(cf_location)
    return max(0,(1-(3/clf_c4(s))*np.sqrt(1-np.power(clf_c4(s),2))))


def clf_B4(subgroup_size):
  """
  Shewart Control Limit Function B3
  Used for Xbar-s chart
  Formula: 1+(k/c4)*sqrt(1-c4^2))

  k = sigma level (almost always 3)
  subgroup_size: integer of subgroup size of measurements
  """
  control_function_reference = pd.read_pickle(cf_location)

  if subgroup_size<2:
    return print("Subgroup size must be greater than 1.")
  try:
    if ((subgroup_size in set(control_function_reference.index))&(pd.notna(control_function_reference.loc[subgroup_size,'B4']))):
      return control_function_reference.loc[subgroup_size,'B4']
    else:
      s = subgroup_size
      control_function_reference.loc[subgroup_size,'B4'] = 1+(3/clf_c4(s))*np.sqrt(1-np.power(clf_c4(s),2))
      control_function_reference.loc[subgroup_size,'B4_it'] = np.inf
      control_function_reference.to_pickle(cf_location)
      return 1+(3/clf_c4(s))*np.sqrt(1-np.power(clf_c4(s),2))
  except KeyError:#If subgroup_size doesn't exist it will make new entr)
    s = subgroup_size
    control_function_reference.loc[subgroup_size,'B4'] = 1+(3/clf_c4(s))*np.sqrt(1-np.power(clf_c4(s),2))
    control_function_reference.loc[subgroup_size,'B4_it'] = np.inf
    control_function_reference.to_pickle(cf_location)
    return 1+(3/clf_c4(s))*np.sqrt(1-np.power(clf_c4(s),2))


def dev_from_std_line(data,sigma=3):
  """
  Function searches data for points > K standard deviations from standard line 
  data = series of raw data
  mean = float of mean
  stdev = float of std dev
  sigma = standard deviations away from center line to target

  Return: List that contains int location of data points.  Can be summed to give total OOC points
  output = Above location, Above count, Below location, Below count
  Above location = list of location of data >sigma X above centerline
  Above count = integer of count X Sigma above centerline
  Below location = list of location of data <sigma X below centerline
  Below count = interger of count X sigma below centerline
  """
  data_avg = data.groupby(date_label).mean()
  
  calc = control_chart_calc(data)
  mean = calc['Average of Averages']
  stdev = calc['Three Sigma xbar']/3
  upper=[]
  lower=[]
  upper_count=0
  lower_count=0
  for i in range(len(data_avg)):
    if data_avg.iloc[i,0]>(mean+(sigma*stdev)):
      upper.append(1)
      upper_count+=1
    else:
      upper.append(0)
    if data_avg.iloc[i,0]<(mean-(sigma*stdev)):
      lower.append(1)
      lower_count+=1
    else:
      lower.append(0)
  return upper, upper_count, lower,lower_count

def print_failed_points(list):
  failed=[]
  for i,each in enumerate(list):
    if each==1:
      failed.append(i+1)
  return failed

def control_chart_calc(data,value_column='Value',resample_period='T'):
  """
  Input raw dataframe with time index
  Output dictionary with control charting info:
  
  """
  data = data[[value_column]].copy()
  #Control chart calculations
  grouper = data.resample(resample_period)
  mean_data = grouper.agg('mean').dropna(how='all')
  sum_of_averages = grouper.agg('mean').sum()
  subgroups = grouper.agg('mean').count()
  subgr_size = grouper.agg('count')
  subgr_size = subgr_size[subgr_size['Value']>0]
  subgroup_size = subgr_size['Value'].mean().astype('int')
  average_of_averages = sum_of_averages/subgroups
  cmax = grouper.agg('max').dropna(how='any')
  cmin = grouper.agg('min').dropna(how='any')
  range_data = cmax.Value-cmin.Value
  sum_of_ranges = (range_data).sum()
  average_of_ranges = sum_of_ranges/subgroups
  three_sigma_xbar = clf_A2(subgroup_size)*average_of_ranges
  UCL_xbar = average_of_averages+three_sigma_xbar
  LCL_xbar = average_of_averages-three_sigma_xbar
  UCL_Rchart = clf_D4(subgroup_size)*average_of_ranges
  LCL_Rchart = clf_D3(subgroup_size)*average_of_ranges

  output = {   #.at['Value] needed to convert from Series to int.
      'Subgroup Size':subgroup_size,
      'Subgroups':subgroups.at['Value'],
      'Average of Averages':average_of_averages.at['Value'],
      'Sum of Ranges':sum_of_ranges,
      'Average of Ranges':average_of_ranges.at['Value'],
      'Three Sigma xbar':three_sigma_xbar.at['Value'],
      'UCL Xbar':UCL_xbar.at['Value'],
      'LCL Xbar':LCL_xbar.at['Value'],
      'UCL Rchart':UCL_Rchart.at['Value'],
      'LCL Rchart':LCL_Rchart.at['Value'],
      'Mean Data':mean_data      
  }
  return output  

###Control chart setup.
def xbar_range(data,title,ylabel, value_column='Value',resample_period='T',savepath=save_path, show=False):
  """
  data = raw dataframe with time as index
  variable = response variable (ie. 'mA' or 'Final mA')
  subset = how control charts should be subset (ie. by spray machine = 'Machine')
  save_path: adds 'control_charts/' 

  utilize subplotting?
  would like to automatically calculate subgroup size
  """
  grouper = data.resample(resample_period)
  mean_data = grouper.agg('mean').dropna(how='all')
  
  cmax = grouper.agg('max').dropna(how='any')
  cmin = grouper.agg('min').dropna(how='any')
  range_data = cmax.Value-cmin.Value

  range_chart = pd.DataFrame(data=range_data,index=range_data.index)

  #Control chart calculations
  calc = control_chart_calc(data,value_column=value_column,resample_period=resample_period)
  subgroup_size = calc.get('Subgroup Size')
  average_of_averages = calc.get('Average of Averages')
  sum_of_ranges = calc.get('Sum of Ranges')
  average_of_ranges = calc.get('Average of Ranges')
  three_sigma_xbar = calc.get('Three Sigma xbar')
  UCL_xbar = calc.get('UCL Xbar')
  LCL_xbar = calc.get('LCL Xbar')
  UCL_Rchart = calc.get('UCL Rchart')
  LCL_Rchart = calc.get('LCL Rchart')
  if sum_of_ranges==0:
    return

  # #Creating plot
  if subgroup_size>1:
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(12,12))
    #Top axis
    ax1.scatter(data.index,data['Value'],s=8)
    ax1.plot(mean_data.index,mean_data.values,'r-')
    #x = len(mean_data)
    ax1.axhline(y=UCL_xbar,c='red',dashes=(1,2)) #UCL Mean
    ax1.axhline(y=LCL_xbar,c='red',dashes=(1,2)) #LCL Mean
    #ax1.plot(mean_data.index,LCL_xbar*x,'r--')
    ax1.set_ylim(mean_data.values.min()*0.7,mean_data.values.max()*1.3)
    ax1.set_xlabel('Date')
    ax1.set_ylabel(ylabel+ "\nMean")
    #ax1.grid(True)
    # loc=plticker.MultipleLocator(base=10)#high base reduces number of ticks
    # ax1.xaxis.set_major_locator(loc)
    ax1.xaxis.set_major_locator(plticker.MaxNLocator(4))
    ax1.get_xaxis().set_visible(False)
    ax1.minorticks_on()
    ax1.set_title(ylabel + " Xbar-R for " + title,loc="center")
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # Info box Mean Plot
    textstr1 = '\n'.join((
      r'$\mu=%.2f$' % (average_of_averages, ),
      r'$\mathrm{UCL}=%.2f$' %(UCL_xbar, ),
      r'$\mathrm{LCL}=%.2f$' %(LCL_xbar, ),
      r'$\mathrm{Range}=%.2f$' %(data['Value'].max()-data['Value'].min()),))
    ax1.text(0.88, 0.95, textstr1,transform=ax1.transAxes,fontsize=10,verticalalignment='top',bbox=props )
  else:
    fig, ax = plt.subplots(figsize=(12,12))
    #Top axis
    ax.scatter(data.index,data['Value'],s=8)
    ax.plot(mean_data.index,mean_data.values,'r-')
    #x = len(mean_data)
    ax.axhline(y=UCL_xbar,c='red',dashes=(1,2)) #UCL Mean
    ax.axhline(y=LCL_xbar,c='red',dashes=(1,2)) #LCL Mean
    ax.minorticks_on()
    #ax1.plot(mean_data.index,LCL_xbar*x,'r--')
    if mean_data.values.min()<0:
      ax.set_ylim(mean_data.values.min()*1.3,mean_data.values.max()*0.7) #catches negative y-axis control chart
    else:
      ax.set_ylim(mean_data.values.min()*0.7,mean_data.values.max()*1.3)
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel+ "\nMean")
    
    #ax1.grid(True)
    # loc=plticker.MultipleLocator(base=10)#high base reduces number of ticks
    # ax1.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_locator(plticker.MaxNLocator(4))
    ax.get_xaxis().set_visible(True)
    ax.set_title(ylabel + " Xbar-R for " + title,loc="center")
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # Info box Mean Plot
    textstr1 = '\n'.join((
      r'$\mu=%.2f$' % (average_of_averages, ),
      r'$\mathrm{UCL}=%.2f$' %(UCL_xbar, ),
      r'$\mathrm{LCL}=%.2f$' %(LCL_xbar, ),
      r'$\mathrm{Range}=%.2f$' %(data['Value'].max()-data['Value'].min()),))
    plt.text(0.88, 0.95, textstr1,transform=ax.transAxes,fontsize=10,verticalalignment='top',bbox=props )
  if subgroup_size>1: #Range Chart needs subgroup >1
    #ax2.scatter(range_chart.index,range_chart['Value'],s=1)
    ax2.plot(range_data.index,range_data.values,'b--')  
    ax2.axhline(y=UCL_Rchart,c='red',dashes=(2,1)) #UCL Range
    ax2.axhline(y=LCL_Rchart,c='red',dashes=(2,1)) #LCL Range
    ax2.set_ylim(range_data.values.min()*0.8,max(range_data.values.max()*1.2,UCL_Rchart*1.2))
    ax2.set_ylabel(ylabel+ "\nRange")
    ax2.minorticks_on()
    # ax2.xaxis.set_major_locator(loc)
    ax2.xaxis.set_major_locator(plticker.MaxNLocator(4))
    ax2.get_xaxis().set_visible(True)
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #Info box Range Plot
    textstr2 = '\n'.join((
      r'$\overline{Range}=%.2f$' % (average_of_ranges, ),
      r'$\mathrm{UCL}=%.2f$' %(UCL_Rchart, ),
      r'$\mathrm{LCL}=%.2f$' %(LCL_Rchart, ),))
    ax2.text(0.88, 0.95, textstr2,transform=ax2.transAxes,fontsize=10,verticalalignment='top',bbox=props )
  if show==False:
    plt.close()
  else:
    plt.show()

  os.makedirs(os.path.dirname(savepath+'control_charts/'), exist_ok=True)
  title = title.replace('/','_')
  fig.savefig(savepath+"control_charts/"+title+"."+ylabel+".jpg")
  #return range_data

#used for sorting alphanumeric lists like line_list
def tryint(s):
  try:
    return int(s)
  except ValueError:
    return s

def alphanum_key(s):
    """
    Turn a string into a list of string and number chunks.

    >>> alphanum_key("z23a")
    ["z", 23, "a"]

    """
    return [ tryint(s) for s in re.split('([0-9]+)', s) ]


def subset_data(data, variable,group,drop=False, how='mean',plant_sub=None,output=False,save_path=save_path,rename=True,name='',value='Value'):
  """
  data: dataframe of raw data
  variable: list, variable that will be isolated from dataframe Variable  
  group: list, how to group final groupby
  drop: drop group in return dataframe. Use for doing multiple of same dataframe
  how: str, accepts mean, sum, std, min, max, count, range, Oos
  plant_sub: str, subset data to a single plant
  output: bool, will output to save_path
  save_path: str, save_path
  rename: bool, will rename value column with how variable.
  name: str, name for renaming column 
  value: str, column name to pull value from. Default value but can be OOS, etc.

  returns dataframe with Group and Variable of choice.

  Use cases:  Build summary charts for data sets in increasingly granular info.

  """
  #critical_variables = ['Spray Weight Net']

  #group = ['Plant','Line','Spray Type','Spray Spec']

  temp_data = data.copy()
  try:
    average_data_df = temp_data[temp_data.Variable.isin(variable)].copy()
  except:
    return
  average_data_df = average_data_df[average_data_df['Plant']!='Huron Ends'].copy()
  if plant_sub!=None:
    average_data_df = average_data_df[average_data_df['Plant']==plant_sub].copy()
  average_data_df.sort_values(by=group,inplace=True,axis=0)

  for each in average_data_df.columns:
    if each in group:
      continue
    elif each==value:
      continue
    average_data_df.drop(each,axis=1,inplace=True)
  # 
  if how=='mean':
    average_data_chart = average_data_df.groupby(group).mean()#.unstack([len(group)-1])
  if how=='sum':
    average_data_chart = average_data_df.groupby(group).sum()
  if how=='std':
    average_data_chart = average_data_df.groupby(group).std()
  if how=='min':
    average_data_chart = average_data_df.groupby(group).min()
  if how=='max':
    average_data_chart = average_data_df.groupby(group).max()
  if how=='count':
    average_data_chart = average_data_df.groupby(group).count().astype(int)
  if how=='range':
    average_data_chart = average_data_df.groupby(group).max()-average_data_df.groupby(group).min()
  if how=='oos':
    average_data_chart = average_data_df.groupby(group).sum()
  if how=='median':
    average_data_chart = average_data_df.groupby(group).median()
  average_data_chart.reset_index(inplace=True,drop=drop)
  if rename:
    average_data_chart.rename(columns={value: name},inplace=True)
  if output:
    average_data_chart.to_excel(save_path + 'Summary Data.xls')
  return average_data_chart

def multi_subset_data(data, variable,group,how=['mean','mean'],plant_sub=None,output=False,save_path=save_path,rename=True, name='',value='Value'):
  """
  data: dataframe of raw data
  variable: list, variable that will be isolated from dataframe Variable  
  group: list, how to group final groupby
  how: list of str, accepts mean, sum, std, min, max, or count; matches variable list
  plant_sub: str, subset data to a single plant
  output: bool, will output to save_path
  save_path: str, save_path
  rename: bool, will rename value column with how variable.
  name: list, name for renaming column 

  returns dataframe with Group and Variable of choice.
  """
  for i,each in enumerate(variable):
    if i==0:
      temp_df = subset_data(data, [each],group,drop=False, how=how[i],plant_sub=plant_sub,output=False,save_path=save_path,rename=True,name=str(name[i]),value=value[i])
    else:
      merge_df = subset_data(data, [each],group,drop=False, how=how[i],plant_sub=plant_sub,output=False,save_path=save_path,rename=True,name=str(name[i]),value=value[i])
      temp_df = pd.merge(temp_df,merge_df,how='outer', on=group)
  if output:
    name_list = ",".join(name)
    variable_list = ",".join(variable)
    group_list = ",".join(group)
    temp_df.to_excel(save_path +group_list+ name_list +'Multi Summary Data.xls')    
  return temp_df

def get_date_col_label(data):
  for col in data.columns:#couldn't get list comprehension to work.
    if 'Date' in col:
      date_label = col
  return date_label

