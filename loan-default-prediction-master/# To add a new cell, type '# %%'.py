# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# here i am importing the packages onto running Environment.

# %%


from sklearn import preprocessing

# %% [markdown]
# here we are importing the preprocessing module

# %%
import scipy.stats as stats


# %%

df=pd.read_excel(r"Sample1.xlsx")


# %%
print(df.columns)
df.shape


# %%
df.describe(include='all')


# %%



# %%
feat_t1= ["A", "B","C","D","E","F"]
feat_t2 = ['M1', 'N1', 'P1', 'Q', 'M2', 'N2', 'P2','M3', 'N3', 'P3']


# %%
df = df[['A', 'B', 'C', 'D', 'E', 'F','M1', 'M2', 'M3','N1', 'N2', 'N3','P1', 'P2', 'P3','Q']]


# %%
df_test1 = df[feat_t1]
df_test2 = df[feat_t2]


# %%
df["N1"].max()


# %%
df_test1.columns


# %%
df_test2= df_test2[['M1', 'M2', 'M3','N1', 'N2', 'N3','P1', 'P2', 'P3','Q']]
df_test2


# %%
df_test1.describe(include='all')


# %%
df_test2.describe(include='all')

# %% [markdown]
# ##this are the columns   *next
# (1000, 17) are the total number of the rows and columns

# %%
df_test2


# %%
df.isnull().sum()


# %%
df.reset_index(drop=True, inplace=True)

# %% [markdown]
# Now, we are removing the index of the dataframe, but the default index will be there to the dataframe.
# %% [markdown]
# df.drop(["ID"],axis=1,inplace=True)

# %%
df_test1[df_test1=="-"] = np.nan


# %%
df_test1[df_test1=="-"] = np.nan
df_test1

# %% [markdown]
# i1= df_test1.index.values
# i1.tolist()
# print(i1)
# 
# %% [markdown]
# i2 = df_test2.index.values.tolist()
# print(i2)
# %% [markdown]
# i1=set(i1)
# i2=set(i2)
# %% [markdown]
# s=i1.intersection(i2)
# len(s)
# #this is count of the intersection of the total stud who both test1 and test2 ..i.e 741 in number

# %%
df_test1.dropna(how="all",inplace=True)
df_test1

# %% [markdown]
# newdf = df_test.drop(df_test.join(df_test.set_index(df["ID"]).index))

# %%
df_test2[df_test2=="-"] = np.nan


# %%
df_test2.dropna(how="all",inplace=True)


# %%
df_test1.shape


# %%
df_test2.shape


# %%
df.replace('-', np.nan, inplace = True)
df = df.dropna()

# %% [markdown]
# df_test1[~df_test1[feat_t1].str.contains(np.nan)]
# %% [markdown]
# #df_new =df.drop("ID",axis=True)
# df_new.dropna(inplace=True)
# df_new[df_new=="-"] = np.nan
# df_new = df_new.apply(pd.to_numeric)
# df_new.dropna(inplace=True)
# %% [markdown]
# df_new.quantile()

# %%
df.dropna(how='all',inplace=True)
df_test1.dropna(how='all',inplace=True)
df_test2.dropna(how='all',inplace=True)


# %%
df_test1

# %% [markdown]
# above code perform the dropping of the rows with all the column value as the NULL
# in df, df1, df2 respectively...

# %%
#df = df.apply(pd.to_numeric)
df_test1 =  df_test1.apply(pd.to_numeric)
df_test2 = df_test2.apply(pd.to_numeric)


# %%
df_test1.rank(ascending=False)


# %%
df_test1.corr(method='spearman')


# %%
d=df_test1.corr(method='spearman')
sns.heatmap(d)


# %%
d=df_test2.corr(method='spearman')
sns.heatmap(d)


# %%
df_test1.mad()


# %%
df_test2.mad()


# %%
def count_unqiue_in_df(d, col):
    for i in col:
        print(d[i].value_counts().hist())


# %%
count_unqiue_in_df(df_test1, feat_t1)

# %% [markdown]
# this shows the amount of the count of occuernce of the each features differnet values which are unqiue in nature...

# %%
count_unqiue_in_df(df_test2, feat_t2)


# %%
print('df_test1')
for i in range(len(feat_t1)):
    print(df_test1[feat_t1[i]].isnull().sum())
print('df_test2')
for i in range(len(feat_t1)):
    print(df_test1[feat_t1[i]].isnull().sum())


# %%
df_test1.shape


# %%
df_test2.shape


# %%
df_count=pd.read_excel(r"Sample1.xlsx")


# %%
df_count


# %%
s1=df_count.shape[0]-df_test1.shape[0] 
s2= df_count.shape[0] -df_test2.shape[0]
print(s2-s1)

# %% [markdown]
# this  is the count of the total students who wrote both test1 and test2

# %%
s1=df_count.shape[0]-df_test1.shape[0] 
s2= df_count.shape[0] -df_test2.shape[0]
print(s1,"this is the count of the total no.of students who atteppemted the test1")
print(s2,"this is the count of the total no.of students who atteppemted the test2")


# %%



# %%


# %% [markdown]
# total number of student who didnt appear in any exam of the above test2
# %% [markdown]
# from above result we can 898 is the count of total stud who attempted atleast one exam in both test1 and test2

# %%
df_test1

# %% [markdown]
# from above result we can 898 is the count of total stud who attempted atleast one exam in test1 only

# %%
df_test2

# %% [markdown]
# from above result we can 741 is the count of total stud who attempted atleast one exam in test2 only.
# %% [markdown]
# corr_mat_df = df_test2.corr()
# corr_mat_df.head()
# %% [markdown]
# sns.heatmap(corr_mat_df,cmap='Blues', fmt='g')

# %%
df_test2


# %%
df_test2.corr()

# %% [markdown]
# df[df=='NaN']=-1

# %%



# %%
df_test2.dtypes


# %%
corr_mat = df_test2.corr().round(2)
#print(corr_mat)
sns.heatmap(data=corr_mat, annot=True)


# %%
df


# %%
df.shape


# %%
df.isnull().sum()


# %%
df.dtypes


# %%
df.describe()


# %%
df["A"].value_counts().plot.bar()


# %%
df["B"].value_counts().plot.bar()


# %%
df["C"].value_counts().plot.bar()


# %%
df["D"].value_counts().plot.bar()


# %%
df["E"].value_counts().plot.bar()


# %%
for i in feat_t1:
    print(df_test1[i].value_counts(),end="\n")


# %%
# convert all columns of DataFrame
#https://www.linkedin.com/pulse/change-data-type-columns-pandas-mohit-sharma#:~:text=The%20best%20way%20to%20convert,floating%2Dpoint%20numbers%20as%20appropriate.
df = df.apply(pd.to_numeric) # convert all columns of DataFrame

# convert just columns "a" and "b"
df = df.apply(pd.to_numeric)


# %%
df.dtypes


# %%
# Get unique count for each variable
df.nunique()

# %% [markdown]
# this is the count of unique number in each col

# %%
df.info()

# %% [markdown]
# all feat are converted into int and float
# 

# %%
#df["A"].plot.hist()
def plot_hist(d, col):
    plt.figure(figsize=(6,6))
    for i in range(len(col)):
        plt.subplot(1,1,1)
        print(i)
        d[col[i]].plot.hist()
        plt.title(col[i])
        #ax.hist(i)
        plt.show()
plot_hist(df,feat_t1)

# %% [markdown]
# here we have the histogram plots of test1:
# >> all the variables are right skewed in nature 
# >> and also the distibution of the variables are somewhat same but not as close as possible...
# >>from the correlation of the test1 i can say A,B,D are highly corelated.
# 
# %% [markdown]
# df.replace(np.nan,-1,inplace=True)
# df

# %%
def plot_hist(dataframe, column ):
    #fig, axs = plt.subplots(2,3)
    plt.figure(figsize=(10,10))
    for i in enumerate(column):
        plt.subplot(5,5,i[0]+1)
        plt.hist(dataframe[i[1]])
    plt.show()


# %%
plot_hist(df,feat_t2)


# %%
plot_hist(df,feat_t1)


# %%



# %%
def plot_bar(dataframe, column ):
    plt.figure(figsize=(10,10))
    for i in enumerate(column):
        plt.subplot(3,2,i[0]+1)
        plt.bar(dataframe[i[1]],height=20,width=2)
    plt.show()


# %%
plot_bar(df,feat_t1)

# %% [markdown]
# as bar plot are not in case of numeric values

# %%
#scatter plot
def scatter_plotq (df, column):
    plt.figure(figsize=(10,10))
    for i in enumerate(column):
        for j in enumerate(column):
            plt.subplot(3,2,i[0]+1)
            sns.scatterplot(data = df, x = i[1], y = j[1])    
    plt.show()


# %%
scatter_plotq(df,feat_t1)

# %% [markdown]
# ##look into this plot afterwards
# %% [markdown]
# def plot_box(dataframe, column ):
#     plt.figure(figsize=(10,20))
#     for i in enumerate(column):
#         plt.subplot(3,2,i[0]+1)
#         plt.boxplot(dataframe[i[1]])
#     plt.show()
# 
# def plot_box1(dataframe, col):
#     dataframe.plot().boxplot(col)
# 
# %% [markdown]
# plot_box1(df,feat_t1)

# %%
df.boxplot(column=feat_t1)


# %%
df.boxplot(column=feat_t2)

# %% [markdown]
# outlier are there but can they give an info :
# >>the IQR values of the A,B,C,D,E,F are all diff and we can see the difference in the above boxplot
# %% [markdown]
# 
# duplicate = df.duplicated()
# duplicate.sum()
# df[duplicate]
# df.drop_duplicates(inplace=True)
# duplicate = df.duplicated()
# duplicate.sum()
# #df[duplicate]
# %% [markdown]
# def remove_outlier1(col):
#     sorted(col)
#     q1,q3 = col.quantile([0.25, 0.75])
#     IQR =q3-q1
#     lower_range = q1-(1.5*IQR)
#     upper_range = q3+(1.5*IQR)
#     #print(lower_range, upper_range,IQR)
#     return [lower_range, upper_range,IQR]
# %% [markdown]
# def dict_low_high(d, col):
#     dict_l_h=dict()
#     for i in range(len(col)):
#         dict_l_h[col[i]]= remove_outlier1(d[col[i]])
#     return dict_l_h
# d=dict_low_high(df_test1,feat_t1)
# d
# 

# %%
df_test1.dtypes

# %% [markdown]
# these are the q1,q3,IQR values of test1

# %%
# Create a function to return index of outliers
def indicies_of_outliers(x):
 q1, q3 = np.percentile(x, [25, 75])
 iqr = q3 - q1
 lower_bound = q1 - (iqr * 1.5)
 upper_bound = q3 + (iqr * 1.5)
 return np.where((x > upper_bound) | (x < lower_bound))
 


# %%
def indicies_of_outliers_fun(df_, col):
    dict_out_col = dict()
    for i in range(len(col)):
        dict_out_col[col[i]] = indicies_of_outliers(df_[col[i]])
    return dict_out_col


# %%
a=[ 31,  33,  42,  61, 147, 168, 183, 189, 296, 299, 314, 387,460, 495, 505, 583, 621, 712, 731, 818, 827]
for i in a:
    print(df["A"][i])


# %%
d1 = indicies_of_outliers_fun(df_test1,feat_t1)
d1
a=d1["A"]


# %%
a= [list(l[0]) for l in a]


# %%
print(a)

# %% [markdown]
# ##outlier_in_col test1 this are the index of the so called outliers  in test1

# %%
d2 = indicies_of_outliers_fun(df_test2,feat_t2)
d2


# %%
print(df_test2['M1'].where(df_test2['M1'] ==306))

# %% [markdown]
# def outlier_in_col(d,col):
#     dict_c=dict()
#     for i in col:
#         dict_c[i]= remove_outlier1(d,i)
#         #print(d[[any([a, b]) for a, b in zip(d[i] < dict_c[i][0], d[i] >  dict_c[i][1])]])
#         print(d[any(d[i] >dict_c[i][0] and d[i] >dict_c[i][1]  )])
# outlier_in_col(df_test1,feat_t1)

# %%
df_test2[df_test2["Q"].isnull()].head()
df_test2


# %%
def max_min_mean(df,col):
    for i in col:
        print(" {3}   min {0} \t max {1}\t std {2}".format(min(df[i]),max(df[i]),np.std(df[i]),i))

# %% [markdown]
# def outlier_fix(df, colu):
#     for i in colu:
#         low, upp = remove_outlier(df[i])
#         df[i] = np.where(df[i]>upp , upp, df[i])
#         df[i] = np.where(df[i]<low , low, df[i])
# %% [markdown]
# outlier_fix(df,feat_t1)
# box_plot_without_outlier(df,feat_t1)
# 
# %% [markdown]
# 
# def box_plot_without_outlier(df,colu):
#     df.boxplot(column=colu)
# %% [markdown]
# low, upp ,iqr= remove_outlier(df["A"])
# df["A"] = np.where(df["A"]>upp , upp, df['A'])
# df["A"] = np.where(df["A"]<low , low, df['A'])
# %% [markdown]
# df.boxplot(column=["A"])

# %%
max_min_mean(df,feat_t1)


# %%
print(df.var(),"\nvar")
print(df.std(),"\nstd")


# %%
plt.figure(figsize=(4,4))
sns.distplot(df['A'], color='g', bins=100, hist_kws={'alpha': 0.4});


# %%
plt.figure(figsize=(4,4))
sns.distplot(df['B'], color='g', bins=100, hist_kws={'alpha': 0.4});


# %%
df.corr()


# %%
df["A"].max()


# %%
df.shape


# %%
from sklearn.preprocessing import StandardScaler
std = StandardScaler()


# %%
df_scaled= std.fit_transform(df.values.reshape(-1,1))
df_scaled.shape

# %% [markdown]
# plt.figure(figsize=(10,10))
# for i in enumerate(feat):
#     plt.subplot(3,3,i[0]+1)
#     #sns.displot()
#     df[i[1]].value_counts().plot.bar()
#     

# %%
corr_mat = df[feat_t1].corr()
sns.heatmap(data=corr_mat, annot=True)


# %%
corr_mat = df[feat_t2].corr()
sns.heatmap(data=corr_mat, annot=True)
#we can see the test Q is not 


# %%
plt.figure(figsize=(9,8))
sns.displot(df[feat_t2],color="g")


# %%
df[feat_t1].hist(figsize=(5,5), bins=10,xlabelsize=8,ylabelsize=8)


# %%
df_num = df.corr()["A"][:-1]
g_feat = df_num[abs(df_num)>0.5].sort_values(ascending =False)
g_feat


# %%
plt.hist(df["A"])
plt.show() 
#here the more marks obtained by stud lie in bw 28-34.
#medium in bw 22-24
#that shows about 0.68 are above avg and bright stud in the test1 of test A
#simillary we can check for other and interput the distribution of the class avg 


# %%
619/899

# %% [markdown]
# df["A"].plot().hist()

# %%
sns.distplot(df['A'])


# %%
fig, axs = plt.subplots(ncols=3)
sns.displot(x='A', y='B', data=df, ax=axs[0])
sns.displot(x='A', y='C', data=df, ax=axs[1])
sns.displot(x='A',y='D', data=df, ax=axs[2])

# %% [markdown]
# for j in enumerate(feat):
#     plt.subplot(3,3,j[0]+1)
#     fig, ax = plt.subplots(figsize=(8,8))
#     ax.scatter(df[j[1]] )
#     ax.set_xlabel(j[1])
#     ax.set_ylabel(j[1])
#     plt.show()

# %%
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing


# %%
df_new


# %%
scaler = preprocessing.StandardScaler()
# Transform the feature
standardized1 = scaler.fit_transform(df_test1)
# Show feature
standardized1


# %%
scaler = preprocessing.StandardScaler()
# Transform the feature
standardized2 = scaler.fit_transform(df_test2)
# Show feature
standardized2


# %%
scaler = preprocessing.MinMaxScaler()
# Transform the feature
standardized_new = scaler.fit_transform(df_new)
# Show feature
standardized_new
scaled_df_new = pd.DataFrame(standardized_new, columns=df_new.columns)
scaled_df_new.head()


# %%
sns.set(font_scale=1.8)


# %%
sns.heatmap(scaled_df_new[feat_t1].corr(),cmap="YlGnBu",annot=True)


# %%
f, ax = plt.subplots(figsize=(16,16))
ax = sns.heatmap(scaled_df_new[feat_t2].corr().round(2),cmap="YlGnBu",annot=True)


# %%
f, ax = plt.subplots(figsize=(16,16))
ax=sns.heatmap(scaled_df_new[feat_t1+feat_t2].corr().round(2),cmap="YlGnBu",annot=True)


# %%



# %%
scaled_df_new[feat_t1].corr().round(2)


# %%
df_norm_col=(df_new-df_new.mean())/df_new.std()
df_norm_col


# %%
min_max_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
minmax = min_max_scale.fit_transform(df_new)


# %%
minmax


# %%
scaled_df_new_minmax = pd.DataFrame(minmax, columns=df_new.columns)
scaled_df_new_minmax.head()


# %%
f, ax1 = plt.subplots(figsize=(16,16))
ax1=sns.heatmap(scaled_df_new_minmax[feat_t1+feat_t2].corr().round(2),cmap="YlGnBu",annot=True)


# %%


# %% [markdown]
# fig, ax = plt.subplots(figsize=(100,200)) 
# sns.heatmap(df_norm_col, cmap='viridis')
# #ax.set_aspect("equal")
# #.show()
# %% [markdown]
# # Standardize or Normalize every column in the figure
# # Standardize:
# sns.clustermap(df_norm_col, standard_scale=1)
# # Normalize
# sns.clustermap(df_norm_col, z_score=1)
# 

# %%
scaler = preprocessing.StandardScaler()
# Transform the feature
standardized = scaler.fit_transform(df_new)
# Show feature
standardized


# %%
scaled_df_new_stand = pd.DataFrame(standardized, columns=df_new.columns)
scaled_df_new_stand.head()


# %%
f, ax1 = plt.subplots(figsize=(16,16))
ax1=sns.heatmap(scaled_df_new_stand[feat_t1+feat_t2].corr().round(2),cmap="YlGnBu",annot=True)


# %%
def heatmap(x, y, size):
    fig, ax = plt.subplots()
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)


# %%
corr = df_new[feat_t1].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)

# %% [markdown]
# https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec

# %%
corr = df_new[feat_t2].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)


# %%
# Print mean and standard deviation
print("Mean:", round(standardized.mean()))
print("Standard deviation:", standardized.std())

# %% [markdown]
# standardized.corr()
# %% [markdown]
# 
# import matplotlib.colors as colors
# 
# norm = colors.Normalize(0,100)
# for pix in feat_t1:
#     val = (norm(pix))
#     print (val)
# 
# img = norm(df_test1)
# print(img)

# %%
from scipy.cluster.hierarchy import dendrogram, linkage


# %%
# Calculate the distance between each sample
# You have to think about the metric you use (how to measure similarity) + about the method of clusterization you use (How to group cars)
Z = linkage(df_norm_col, 'ward')


# %%

# Make the dendrogram
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance (Ward)')
dendrogram(Z, labels=df_norm_col.index, leaf_rotation=90)


# %%
from scipy.cluster import hierarchy

# Calculate the distance between each sample
Z = hierarchy.linkage(df, 'ward')
 
 
# Plot with Custom leaves
hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=8, labels=df.index)


# %%



# %%



# %%



# %%



# %%



# %%


# %% [markdown]
# newdf=pd.concat([df_test1,df_test2]).drop_duplicates(keep=False)
# 
# %% [markdown]
# newdf.shape
# %% [markdown]
# df_test1.set_index(inplace=True)
# df_test2.set_index(,inplace=True)
# newdf=df1.drop(df2.index)

# %%


# %% [markdown]
# pip install sweetviz
# %% [markdown]
# import sweetviz 
# %% [markdown]
# my_report_test1 = sweetviz.analyze([df_test1,"df_test1"])
# my_report_test2 = sweetviz.analyze([df_test2,"df_test2"])
# %% [markdown]
# my_report_df = sweetviz.analyze([df,"df"])
# %% [markdown]
# my_report_cmp = sweetviz.compare([df_test1,"df_test1"],[df_test2,"df_test2"])
# %% [markdown]
# my_report_test1.show_html("report_1.html")
# my_report_test2.show_html("report_2.html")
# my_report_cmp.show_html("report_cmp.html")
# my_report_df.show_html("report_df.html")
# 
# %% [markdown]
# 
# df_with_dash = pd.read_excel(path)
# %% [markdown]
# df_test1_= df_with_dash[feat_t1]
# df_test2_ = df_with_dash[feat_t2]
# 
# my_report_test1_ = sweetviz.analyze([df_test1_,"df_test1_"])
# my_report_test2_ = sweetviz.analyze([df_test2_,"df_test2_"])
# my_report_cmp_= sweetviz.compare([df_test1,"df_test1"],[df_test2,"df_test2"])
# my_report_test1_.show_html("report_1_.html")
# my_report_test2_.show_html("report_2_.html")
# my_report_cmp_.show_html("report_cmp_.html")

# %%



