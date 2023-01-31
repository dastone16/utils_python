def print_test():
  print("Hello World!")
  
def data_description(data,label):
  description = data.describe().transpose()
  #print(description)
  description.to_csv(save_path + label + '.csv')
  
def date_compare(data,date_label,start_date,trans_date,fin_date,
                 output_name,resample='H',label1="Before",label2="After",corr="Metal Exposure (mA)",
                 raw=False):
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
  Summary[label1 + 'Mean'] = np.round(before_T.mean(axis=1),4)
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
    Summary.to_csv(save_path + output_name+label1+label2+'.diff.csv')
  else:
    Summary.to_csv(save_path + output_name+label1+label2+'.diff.csv')
    pass
  return Summary  


