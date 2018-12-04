
# coding: utf-8

# In data science the importance of great story telling is commonly known. Through the use of better visualizations, data scientists can create more compelling stories on the projects they work on. Because of that, I'm going to practice using knew visualizations on an NBA Season Stats data set to work on this skill. I am using the same data set as I have for my first EECS 731 project. Now I am going to do very simialir analysis, but will use new, more compelling visualizations to tell the story.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


# In[3]:


df = pd.read_csv('Seasons_Stats.csv')


# For this analysis, I am only going to look at player data for the 2017 season. Because of that I will be dropping the rows for any player before that.

# In[4]:


df.drop(df.index[:24096], inplace=True)


# In[5]:


df.head


# In[6]:


df.drop('Unnamed: 0', axis=1)


# When I first created a visualization at the start of the semester, I did so using matplotlib. I am going to replicate those below to compare with the visualizations I will be creating in seaborn. The first data set looks at the basic relationship between games played and points scored.

# In[7]:


df.plot(x = 'G', y = 'PTS', kind = 'scatter')


# Our intuition would tell us that the more games a player plays, the higher is total points scored in a season would be. We clearly see that in this first plot. 

# In[9]:


df.plot(x = 'PTS', y = 'AST', kind = 'scatter')


# Something very interesting to me personally is the correlation between total points scored and total assists. Among basketball analyst, it is often believed that players who score a lot of points, but fail to also produce a high volume of assists are one-dimensional and selfish players. That being said, I wanted to visualize what this relationship looked like for the 2017 season. It is also important to note, Big men who play in the front court (powerforward or center) may score a lot of points, but don't necessarily dish out a lot of assists, as that doesn't follow their role on the team. Because of that, I am going to do another plot looking at PTS and rebounds.

# In[57]:


df.plot(x = 'PTS', y = 'DRB', kind = 'scatter')


# Interestingly enough, the graph looking at points vs rebounds is also very simialir to the one with assists. I am going to also look at the scatterplot comparing points to TS%. TS% attempts to see how efficient a player is, so I doubt it would be correlated to total points scored, as there are many inefficient players who score a lot of points simply because they take so many shots

# In[10]:


df.plot(x = 'PTS', y = 'TS%', kind = 'scatter')


# Now that I have made some basic visualizations in matplotlib, I'm going to use seaborn along with some of its more advanced features. Similairly to the first plot, I'm going to create a scatter between points and games played. This time I'm going to color code the data by position, so we can have another attribute being assessed in our graphs

# In[14]:


sns.lmplot(x='G', y='PTS', data=df, fit_reg=False, hue='Pos')


# In seaborn, we can also make histograms to get better information on how different stats are distributed.

# In[15]:


sns.boxplot(data=df)


# The above histogram is virtually unreadable. I'm going to create a histogram on major attributes like points, assists, and rebounds to see how these variables are distributed.

# In[20]:


df2 = df[['PTS','AST','ORB', 'DRB']]


# In[21]:


sns.boxplot(data=df2)


# Using Seaborn, I can also plot distributions of positions compared to these various attributes. To do this I am going to create violinplots of different positions and the attributes used above.

# In[24]:


sns.set_style('whitegrid')
sns.violinplot(x= 'Pos', y='PTS', data=df)


# I wasn't sure what to expect with points compared to position. In the modern NBA, no position really dominates scoring, but point guard scoring has been increasing. This can be seen specifically in the 2017 season using the plot above.

# In[25]:


sns.violinplot(x= 'Pos', y='AST', data=df)


# As expected, point guards lead the league in assists, and Centers are at the bottom of the league in assists.

# In[26]:


sns.violinplot(x= 'Pos', y='DRB', data=df)


# Centers lead this distribution as expected, powerforwards are surprisingly low for this distribution.

# In[30]:


stats = df[['Player', 'Pos', 'PTS','AST','ORB', 'DRB', 'TS%', 'FT%', 'STL', 'BLK']]


# I am going to create a new data frame called stats that only have the attributes I personally find important when evaulating a player. Using that, I am going to use the melt function in pandas to create an overall value combining these different attributes.

# In[36]:


melted_df = pd.melt(stats, 
                    id_vars=["Player", 'Pos'], 
                    var_name="Overall") 


# In[37]:


melted_df.tail()


# With this new df, we can make a swarm plot to visualize the different stats, also looking at position.

# In[38]:


sns.swarmplot(x='Overall', y='value', data=melted_df, 
              hue='Pos')


# I'm going to adjust the swarm plot so that it is a little easier to view.

# In[41]:


plt.figure(figsize=(10,6))
sns.swarmplot(x='Overall', 
              y='value', 
              data=melted_df, 
              hue='Pos', 
              dodge=True,)

plt.legend(bbox_to_anchor=(1, 1), loc=2)


# We plotted different relationships between attributes, but we can also create a heatmap to view the specific correlations between each attribute.

# In[43]:


corr = stats.corr()
sns.heatmap(corr)


# From the heatmap, we can see how variables such as points and assists have high correlations, but FTpercentage and blocks have virtually no correlation. The latter makes sense, given players who often get Blocks are centers or powerforward with a lot of height. These players tend not to be the best free throw shooters.

# We can also create histograms to plot the distribution of individual attributes.

# In[46]:


sns.distplot(df.PTS)


# In[48]:


sns.distplot(df.AST)


# We can also use Seeabor to visualize how many players there are at each position in the dataset. 

# In[50]:


sns.countplot(x='Pos', data=df,)


# If we want to look more closely at distributions between two variables, we can use density plots. I'm going to plot two density plots of attributes we know are correlated from the heatmap to see what they look like.

# In[52]:


sns.kdeplot(df.PTS, df.AST)


# In[53]:


sns.kdeplot(df.ORB, df.DRB)


# If we want, we can also plot joint plots that combine the information of histograms and scatterplots to give us further detail of a bi-variate distribution.

# In[54]:


sns.jointplot(x='PTS', y='AST', data=df)


# In[55]:


sns.jointplot(x='ORB', y='DRB', data=df)

