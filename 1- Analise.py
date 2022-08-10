#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/berna/Downloads/ds_salaries.csv')
pd.set_option('display.max_rows',500)
#print(df.columns)
#display(df)
#print(df['salary_currency'].unique())

y = df.loc[:,['salary_in_usd']]
X = df.drop(columns=['Unnamed: 0','salary','salary_currency','salary_in_usd'])


#Objetivo: Queremos prever o valor do salário atual do funcionário.
#print(X)

#variavel Categoricas:
var_cat = ['employment_type','experience_level','employee_residence','company_location','company_size','job_title']

#Variavel Numericas:
var_num = ['remote_ratio']

#Analisando como esta distribuido os dados
#Para dados com alta cardinalidade podemos usar o labelencoder ou value_counts(substituir os valores pelas suas frequências)


#SESSAO DE ANÁLISE DAS VARIAVEIS INDEPENDENTES
#print(X['job_title'].value_counts(100)) #Bem centrado os dados em FT
#print(X['employment_type'].value_counts(100)) #Labelencoder
#print(X['experience_level'].value_counts(100))
#print(X['employee_residence'].value_counts(100))
#print(X['company_location'].value_counts(100))
#print(X['company_size'].value_counts(100))
#print(X['remote_ratio'].value_counts(100))

#SESSAO DE ANÁLISE DAS VARIAVEL TARGET:

#print(y['salary_in_usd'].describe())

#Tabela de correlação entre as variáveis:


#Transformando variáveis categóricas em numéricas:
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
#display(df.dtypes)
def le():
    for i in df.columns:
        if df.loc[:,i].dtypes != 'int64':
            df.loc[:,i] = le.fit_transform(df.loc[:,i])
    return

#print(df)
corr = df.drop(columns=['Unnamed: 0','salary','salary_currency']).corr()
#print(corr['salary_in_usd'].sort_values(ascending=False))



from matplotlib.pyplot import figure
#figure(figsize=(10,10))
#sns.heatmap(corr,vmin=0, vmax=1)



#Verificando significância das variáveis por Chi-Squared.
#Falta verificar se a significância foi sorte ou não.
from scipy.stats import chi2_contingency
df = df.drop(columns=['Unnamed: 0','salary','salary_currency'])

def chi2():
    for i in df.columns:
        df_aux = df[['salary_in_usd',i]]
        chi2, p, dof, ex = chi2_contingency(df_aux, correction=False)
        print('\n',i,'\n',chi2, '{:.10f}'.format(p))

#print(df['employment_type'].value_counts())
#Verificando distribuições das variáveis com data visualization
#company_location
def barplots():
    for i in df.columns:
        a = np.histogram_bin_edges(df[i], bins='sqrt') #Escolher melhor bins para nosso histograma
        print(a)
        #print(len(df[i].unique()))
        plt.hist(df[i])
        plt.title('Histograma: {i}'.format(i=i))
        #plt.xlabel(df[i].unique())
        plt.legend()
        plt.show()
    #else:
            
    return
    

#consulta
test = df['job_title'].value_counts().sort_values(ascending=False)[:11]
from matplotlib.pyplot import figure
#figure(figsize=(25,15))


#barplots()
#sns.barplot(data = df,x=test.index,y =test)


#Pre processamento de dados:
#Ao fazer o tratamento dos dados, melhora a visualização ou não... faz sentidio? tem significância?
#Será que o Cientista ganha mais que o Engenheiro, esse ganho é significativo ou não?
#Mesma perguntas podem ser feitas para os paises/continentes.




#Job_title

#print(df['job_title'].value_counts())
#Analyst,Engineer and Scientist
df_job = df[['job_title','salary_in_usd']]
#print(df_job)
#print(len(df_job['job_title'].str.extract(r'(Scientist)').dropna())) #Isso ja da len dos job titles Scientist
#print(len(df_job.str.extract(r'(Engineer|Architect)').dropna())) #Isso ja da len dos job titles Scientist
#print(len(df_job.str.extract(r'(Analyst|Analytics)').dropna())) #Isso ja da len dos job titles Scientist

#Scientist
df_s = df_job[df_job['job_title'].str.contains(r'(Scientist)')]
#print(df_s['job_title'].value_counts().sort_values(ascending=False))
#print(df_s['job_title'].value_counts()[0])
#print(df_s['job_title'].value_counts().sort_values(ascending=False).index.tolist())
#print(df_s['salary_in_usd'].groupby(df_s['job_title'].str.contains(r'Scientist')).mean())
#print(df_s['salary_in_usd'].groupby(df_s['job_title']).mean().sort_values(ascending=False).index.tolist())
#print(df_s['salary_in_usd'].groupby(df_s['job_title']).mean()['Data Scientist'])









def grafico_bar(df_s):
    lista = df_s['salary_in_usd'].groupby(df_s['job_title']).mean().sort_values(ascending=False).index.tolist()
    ax = plt.axes()
    for i,job in enumerate(lista,1):
        #print(i)
        salary_mean = df_s['salary_in_usd'].groupby(df_s['job_title']).mean()[job]
        ax.bar(x=i,height=salary_mean,width=0.5,align='center',label='{job},({salary:.0f}K)'.format(job=job,salary=salary_mean/1000))
    #ax.set_xticks(range(1,len(lista)+1))
    #ax.set_xticklabels(lista)
    ax.legend(title='Salary mean',fontsize=20)
    return


#Precisa consertar o REGEX....#Precisa consertar o REGEX....#Precisa consertar o REGEX....#Precisa consertar o REGEX....

df_s = df_job[df_job['job_title'].str.contains(r'(Scientist|Science|Data Science)')]
print(len(df_s['job_title'].value_counts()))
df_s2 = df_job[df_job['job_title'].str.contains(r'[\s]+(Scientist)')]
print('\ndf_2',len(df_s2['job_title'].value_counts()))
merge = pd.merge(df_s2['job_title'],df_s['job_title'],on='job_title',how='outer')
#print('\nMerge\n',merge.value_counts())
#print('\nMerge\n',df_s2.value_counts())


df_s = df_job[df_job['job_title'].str.contains(r'(Scientist|Science|Data Science)')]
df_en = df_job[df_job['job_title'].str.contains(r'(Engineer|Architect)')]
df_ana = df_job[df_job['job_title'].str.contains(r'(Analyst|Analytics)')]
df_rest = df_job[~df_job['job_title'].str.contains(r'(Analyst|Analytics|Engineer|Architect|Scientist)')]
#print(df_rest)
df_list = [df_s,df_en,df_ana,df_rest]
#print(df_list)

#Automatização de gráfico
#def graficos(df_list):
#    plt.subplot(len(df_list),1,len(df_list)) #row,col,index
#    figure(figsize=(35,15))
#   for i,df in enumerate(df_list,1):
#        #print(i,df,'\n')
#        plt.subplot(i,1,i)
#        grafico_bar(df_s = df)
#        plt.show()
#    return
#graficos(df_list)
figure(figsize=(20,20))
plt.subplot(1,1,1)
grafico_bar(df_s)
plt.subplot(2,1,2)
grafico_bar(df_rest)
plt.show()

    

    
    


    
#Microuniversso


#['Data Scientist', 'Research Scientist']
def dadsada():
    ax = plt.axes()
    lista = ['Data Scientist','Research Scientist','AI Scientist']
    ax.set
    for i,job in enumerate(lista,1):
        print(i)
        ax.bar(x=i,height=df_s['salary_in_usd'].groupby(df_s['job_title']).mean()[job],width=0.3,align='center')
    plt.ylim(ymax=190000)
    ax.set_xticks(range(1,4))
    ax.set_xticklabels(['Data Scientist', 'Research Scientist','AI Scientist'])
    plt.show()
    return
#plt.xlim(xmax=2)
#opção botar legenda


# In[ ]:





# In[2]:


from pycaret.classification import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/berna/Downloads/ds_salaries.csv')
target = df.drop(columns=[''])


# In[ ]:


#from sklearn.feature_selection import SelectKBest, chi2
#X_new = SelectKBest(chi2, k=20).fit_transform(X, y)


# In[ ]:


import matplotlib.pyplot as plt
x=[[1,2,3,40],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
y=[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
for i in range(len(x)):
    plt.figure()
    print(x[i],y[i])
    plt.plot(x[i],y[i])
    # Show/save figure as desired.
    plt.show()


# In[ ]:


df_s = df_job[df_job['job_title'].str.contains(r'(Scientist|Science|Data Science)')]
print(len(df_s['job_title'].value_counts()))
lista = list(df['job_title'].value_counts().index)
df_s2 = df_job[df_job['job_title'].str.contains(r'[\s]?(Scient|Science)+[\s]?')]
print('df_2\n',len(df_s2['job_title'].value_counts()))


# In[ ]:




