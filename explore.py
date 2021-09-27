import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import sklearn.feature_selection

def zillow_heatmap(df):
    '''
    returns a heatmap and correlation for our target of tax_value
    '''
    plt.figure(figsize = (8,12))
    heatmap = sns.heatmap(df.corr()[["target"]].sort_values(by='target', ascending=False), vmin = -.5, vmax = .5, annot = True)
    heatmap.set_title(f"Features Correlating with Log error")
    
    return heatmap


def plot_categorical_and_continuous_vars(df,categorical,continuous):
    for cont_col in continuous:
        for cat in categorical:
            categorical_label = cat
            continuous_label = cont_col
            
            fig, axes = plt.subplots(figsize=(12,36), nrows=4,ncols=1)
            fig.suptitle(f'{continuous_label} by {categorical_label}', fontsize=18, y=1.02)
            sns.lineplot(ax=axes[0], x=cat, y=cont_col, data=df)
            axes[0].set_title('Line Plot', fontsize=14)
            axes[0].set_xlabel(categorical_label, fontsize=12)
            axes[0].set_ylabel(continuous_label, fontsize=12)
            
            sns.boxplot(ax=axes[1], x=cat, y=cont_col, data=df,\
                        color='blue')
            axes[1].set_title('Box-and-Whiskers Plot', fontsize=14)
            axes[1].set_xlabel(categorical_label, fontsize=12)
            axes[1].set_ylabel(continuous_label, fontsize=12)
            
            sns.swarmplot(ax=axes[2], x=cat, y=cont_col, data=df,\
                        palette='Blues')
            axes[2].set_title('Swarm Plot', fontsize=14)
            axes[2].set_xlabel(categorical_label, fontsize=12)
            axes[2].set_ylabel(continuous_label, fontsize=12)
            
            sns.barplot(ax=axes[3], x=cat, y=cont_col, data=df,\
                        palette='Purples')
            axes[3].set_title('Bar Plot', fontsize=14)
            axes[3].set_xlabel(categorical_label, fontsize=12)
            axes[3].set_ylabel(continuous_label, fontsize=12)
            
            plt.tight_layout()
            
            plt.show()


def plot_variable_pairs(df, cols, descriptive=None, hue=None):
    '''
    This function takes in a df, a list of cols to plot, and default hue=None 
    and displays a pairplot with a red regression line. If passed a descriptive
    dictionary, converts axis titles to the corresponding names.
    '''
    # sets line-plot options and scatter-plot options
    keyword_arguments={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}}
    
    # creates pairplot object
    pairplot = sns.pairplot(df[cols], hue=hue, kind="reg",\
            plot_kws=keyword_arguments)
    
    # if passed a descriptive dictionary, iterates through matplotlib axes
    # in our pairplot object and sets their axis labels to the corresponding 
    # strings.
    if descriptive:
        for ax in pairplot.axes.flat:
            ax.set_xlabel(descriptive[ax.get_xlabel()])
            ax.set_ylabel(descriptive[ax.get_ylabel()])
    
    # Adds a super-title
    pairplot.fig.suptitle('Correlation of Continuous Variables', y=1.08)
    plt.show()


def plot_pairplot(df, cols, descriptive=None, hue=None):
    '''
    Take in train df, list of columns to plot, and hue=None
    and display scatter plots and hists.
    '''
    pairplot = sns.pairplot(df[cols],hue = df[hue], corner=True)
    pairplot.axes.flat[0].set_ylabel(cols[0])
    if descriptive:
        for ax in pairplot.axes.flat:
            if ax:
                ax.set_xlabel(descriptive[ax.get_xlabel()])
                ax.set_ylabel(descriptive[ax.get_ylabel()])
    pairplot.fig.suptitle('Correlation of Continuous Variables', y=1.08)
    plt.show()

def corr_two_vars(df,x,y):
    r, p = stats.pearsonr(df[x],df[y])
    print(f"p-value:{round(p,5)}")
    print(f"R: {round(r,4)}")
    scatter_plot = df.plot.scatter(x,y)
    scatter_plot.figure.set_dpi(300)
    plt.title(f"{x}'s relationship with {y}")
    if p < .05:
        print("This correlation is statistically significant")

    return r,p

def explore_univariate_quant(train, quant_var):
    '''
    takes in a dataframe and a quantitative variable and returns
    descriptive stats table, histogram, and boxplot of the distributions. 
    '''
    plt.figure(figsize = (30,10))
    sns.set(font_scale = 2)
    descriptive_stats = train[quant_var].describe()

    p = plt.subplot(1, 2, 1)
    p = plt.hist(train[quant_var], color='lightseagreen')
    p = plt.title(quant_var)

    # second plot: box plot
    p = plt.subplot(1, 2, 2)
    p = plt.boxplot(train[quant_var])
    p = plt.title(quant_var)
    plt.tight_layout()
    return p, descriptive_stats

def logerror_county(df, x = "county", y = "target"):
    plt.figure(figsize = (30,10))
    sns.set(font_scale = 2)

    plt.subplot(1, 2, 1)
    sns.barplot(data = df, y = df[y], x = df[x])
    plt.title("bar plot of logerror for each county")

    plt.subplot(1, 2, 2)
    sns.boxplot(df[x], df[y])
    plt.ylim(-.2,.2)
    plt.title("Box plot of logerror for each county")
    plt.tight_layout()
    plt.show()

def logerror_bathrooms(df, x = "bathrooms", y = "target"):
    plt.figure(figsize = (30,10))
    sns.set(font_scale = 2)

    plt.subplot(1, 2, 1)
    sns.barplot(data = df, y = df[y], x = df[x])
    plt.title("bar plot of logerror for different number of bathrooms")

    plt.subplot(1, 2, 2)
    sns.boxplot(df[x], df[y])
    plt.ylim(-.2,.2)
    plt.title("Box plot of logerror for different number of bathrooms")
    plt.tight_layout()
    plt.show()

def logerror_bedrooms(df, x = "bedrooms", y = "target"):
    plt.figure(figsize = (30,10))
    sns.set(font_scale = 2)

    plt.subplot(1, 2, 1)
    sns.barplot(data = df, y = df[y], x = df[x])
    plt.title("bar plot of logerror for different number of bedrooms")

    plt.subplot(1, 2, 2)
    sns.boxplot(df[x], df[y])
    plt.ylim(-.2,.2)
    plt.title("Box plot of logerror for different number of bedrooms")
    plt.tight_layout()
    plt.show()


def logerror_var_relationship(df, x = "", y = "target"):
    plt.figure(figsize = (30,10))
    sns.set(font_scale = 2)

    plt.subplot(1, 2, 1)
    sns.barplot(data = df, y = df[y], x = df[x])

    plt.subplot(1, 2, 2)
    sns.boxplot(df[x], df[y])
    plt.ylim(-.2,.2)
    plt.suptitle(f"graphs of logerror and {x}")
    plt.tight_layout()
    plt.show()

def create_cluster(df, X, k):
    
    """ Takes in df, X (dataframe with variables you want to cluster on) and k
    # It scales the X, calcuates the clusters and return train (with clusters), the Scaled dataframe,
    #the scaler and kmeans object and unscaled centroids as a dataframe"""
    'Credit to annah vu'
    
    scaler = MinMaxScaler(copy=True).fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])
    kmeans = KMeans(n_clusters = k, random_state = 174)
    kmeans.fit(X_scaled)
    kmeans.predict(X_scaled)
    df['cluster'] = kmeans.predict(X_scaled)
    df['cluster'] = 'cluster_' + df.cluster.astype(str)
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    return df, X_scaled, scaler, kmeans, centroids

def create_cluster_elkan(df, X, k):
    
    """ Takes in df, X (dataframe with variables you want to cluster on) and k
    # It scales the X, calcuates the clusters and return train (with clusters), the Scaled dataframe,
    #the scaler and kmeans object and unscaled centroids as a dataframe"""
    'Credit to annah vu'
    scaler = MinMaxScaler(copy=True).fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])
    kmeans = KMeans(n_clusters = k, algorithm = "elkan", random_state = 174)
    kmeans.fit(X_scaled)
    kmeans.predict(X_scaled)
    df['cluster'] = kmeans.predict(X_scaled)
    df['cluster'] = 'cluster_' + df.cluster.astype(str)
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    return df, X_scaled, scaler, kmeans, centroids

def select_kbest(X, y, k):
    # make the object
    kbest = sklearn.feature_selection.SelectKBest(
        sklearn.feature_selection.f_regression,
        k=k)

    # fit the object
    kbest.fit(X, y)
    
    # use the object (.get_support() is that array of booleans to filter the list of column names)
    return X.columns[kbest.get_support()].tolist()

def select_rfe(X, y, k):
    # make the thing
    lm = sklearn.linear_model.LinearRegression()
    rfe = sklearn.feature_selection.RFE(lm, n_features_to_select=k)

    # Fit the thing
    rfe.fit(X, y)
    
    # use the thing
    features_to_use = X.columns[rfe.support_].tolist()
    
    # we need to send show_feature_rankings a trained/fit RFE object
    all_rankings = show_features_rankings(X, rfe)
    
    return features_to_use, all_rankings