#!/usr/bin/env python
# coding: utf-8

# # Problem Statement:
# 
# ##How good is player X against CSK in Chepauk when they’re playing 3 spinners? 

# In[1]:


import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[19]:


deliveries = pd.read_csv("deliveries_updated_mens_ipl.csv")


# In[20]:


matches=pd.read_csv("matches_updated_mens_ipl.csv")


# In[21]:


df=deliveries.copy()


# In[22]:


df.head()


# ### 1.Player Statistics

# In[25]:


def player_statistics(df):
    df["Dot"]=df["batsman_runs"].apply(lambda x:1 if x==0 else 0)
    df["1s"]=df["batsman_runs"].apply(lambda x:1 if x==1 else 0)
    df["2s"]=df["batsman_runs"].apply(lambda x:1 if x==2 else 0)
    df["3s"]=df["batsman_runs"].apply(lambda x:1 if x==3 else 0)
    df["4s"]=df["batsman_runs"].apply(lambda x:1 if x==4 else 0)
    df["6s"]=df["batsman_runs"].apply(lambda x:1 if x==6 else 0)
    
    
    runs=pd.DataFrame(df.groupby("batsman")["batsman_runs"].sum()).reset_index().rename(columns={"batsman_runs":"runs"})
    innings=pd.DataFrame(df.groupby("batsman")["matchId"].apply(lambda x: len(list(np.unique(x))))).reset_index().rename(columns={"matchId":"innings"})
    balls=pd.DataFrame(df.groupby("batsman")["matchId"].count()).reset_index().rename(columns={"matchId":"balls"})
    dismissals=pd.DataFrame(df.groupby("batsman")["player_dismissed"].count()).reset_index().rename(columns={"player_dismissed":"dismissals"})
    
    dots=pd.DataFrame(df.groupby("batsman")["Dot"].sum()).reset_index()
    ones=pd.DataFrame(df.groupby("batsman")["1s"].sum()).reset_index()
    twos=pd.DataFrame(df.groupby("batsman")["2s"].sum()).reset_index()
    threes=pd.DataFrame(df.groupby("batsman")["3s"].sum()).reset_index()
    fours=pd.DataFrame(df.groupby("batsman")["4s"].sum()).reset_index()
    sixes=pd.DataFrame(df.groupby("batsman")["6s"].sum()).reset_index()
    
    df=pd.merge(innings,runs,on='batsman').merge(balls,on="batsman").merge(dismissals,on="batsman").merge(dots,on="batsman").merge(ones,on="batsman").merge(twos,on="batsman").merge(threes,on="batsman").merge(fours,on="batsman").merge(sixes,on="batsman")
    
    df["SR"]=df.apply(lambda x: 100*(x["runs"]/x["balls"]),axis=1)
    df["RPI"]=df.apply(lambda x: (x["runs"]/x["innings"]),axis=1)
    
    
    return df


# In[26]:


df=player_statistics(df)


# In[27]:


df


# ### 2. Performances in different phases 

# In[30]:


def phase(over):
    if over<=6:
        return "Powerplay_Overs"
    elif over<=15:
        return "Middle_Overs"
    else:
        return "Death_overs"


# In[31]:


deliveries["phase"]=deliveries["over"].apply(lambda x: phase(x))


# In[35]:


def phaseofplay(df,current_phase):
    df=df[df.phase == current_phase]
    df.reset_index(inplace=True, drop=True)
    
    df["Dot"]=df["batsman_runs"].apply(lambda x:1 if x==0 else 0)
    df["1s"]=df["batsman_runs"].apply(lambda x:1 if x==1 else 0)
    df["2s"]=df["batsman_runs"].apply(lambda x:1 if x==2 else 0)
    df["3s"]=df["batsman_runs"].apply(lambda x:1 if x==3 else 0)
    df["4s"]=df["batsman_runs"].apply(lambda x:1 if x==4 else 0)
    df["6s"]=df["batsman_runs"].apply(lambda x:1 if x==6 else 0)
    
    
    runs=pd.DataFrame(df.groupby("batsman")["batsman_runs"].sum()).reset_index().rename(columns={"batsman_runs":"runs"})
    innings=pd.DataFrame(df.groupby("batsman")["matchId"].apply(lambda x: len(list(np.unique(x))))).reset_index().rename(columns={"matchId":"innings"})
    balls=pd.DataFrame(df.groupby("batsman")["matchId"].count()).reset_index().rename(columns={"matchId":"balls"})
    dismissals=pd.DataFrame(df.groupby("batsman")["player_dismissed"].count()).reset_index().rename(columns={"player_dismissed":"dismissals"})
    
    dots=pd.DataFrame(df.groupby("batsman")["Dot"].sum()).reset_index()
    ones=pd.DataFrame(df.groupby("batsman")["1s"].sum()).reset_index()
    twos=pd.DataFrame(df.groupby("batsman")["2s"].sum()).reset_index()
    threes=pd.DataFrame(df.groupby("batsman")["3s"].sum()).reset_index()
    fours=pd.DataFrame(df.groupby("batsman")["4s"].sum()).reset_index()
    sixes=pd.DataFrame(df.groupby("batsman")["6s"].sum()).reset_index()
    
    df=pd.merge(innings,runs,on='batsman').merge(balls,on="batsman").merge(dismissals,on="batsman").merge(dots,on="batsman").merge(ones,on="batsman").merge(twos,on="batsman").merge(threes,on="batsman").merge(fours,on="batsman").merge(sixes,on="batsman")
    
    df["SR"]=df.apply(lambda x: 100*(x["runs"]/x["balls"]),axis=1)
    df["RPI"]=df.apply(lambda x: (x["runs"]/x["innings"]),axis=1)
    
    
    return df


# In[36]:


pp_df=phaseofplay(deliveries,"Powerplay_Overs")
mo_df=phaseofplay(deliveries,"Middle_Overs")
do_df=phaseofplay(deliveries,"Death_overs")


# In[40]:


pp_df.head(1)


# In[41]:


mo_df.head(3)


# In[42]:


do_df.head(3)


# ### 3. Performance by bat in 1st and 2nd innings

# In[43]:


def ByInning(df,current_inning):
    df=df[df.inning == current_inning]
    df.reset_index(inplace=True, drop=True)
    
    df["Dot"]=df["batsman_runs"].apply(lambda x:1 if x==0 else 0)
    df["1s"]=df["batsman_runs"].apply(lambda x:1 if x==1 else 0)
    df["2s"]=df["batsman_runs"].apply(lambda x:1 if x==2 else 0)
    df["3s"]=df["batsman_runs"].apply(lambda x:1 if x==3 else 0)
    df["4s"]=df["batsman_runs"].apply(lambda x:1 if x==4 else 0)
    df["6s"]=df["batsman_runs"].apply(lambda x:1 if x==6 else 0)
    
    
    runs=pd.DataFrame(df.groupby("batsman")["batsman_runs"].sum()).reset_index().rename(columns={"batsman_runs":"runs"})
    innings=pd.DataFrame(df.groupby("batsman")["matchId"].apply(lambda x: len(list(np.unique(x))))).reset_index().rename(columns={"matchId":"innings"})
    balls=pd.DataFrame(df.groupby("batsman")["matchId"].count()).reset_index().rename(columns={"matchId":"balls"})
    dismissals=pd.DataFrame(df.groupby("batsman")["player_dismissed"].count()).reset_index().rename(columns={"player_dismissed":"dismissals"})
    
    dots=pd.DataFrame(df.groupby("batsman")["Dot"].sum()).reset_index()
    ones=pd.DataFrame(df.groupby("batsman")["1s"].sum()).reset_index()
    twos=pd.DataFrame(df.groupby("batsman")["2s"].sum()).reset_index()
    threes=pd.DataFrame(df.groupby("batsman")["3s"].sum()).reset_index()
    fours=pd.DataFrame(df.groupby("batsman")["4s"].sum()).reset_index()
    sixes=pd.DataFrame(df.groupby("batsman")["6s"].sum()).reset_index()
    
    df=pd.merge(innings,runs,on='batsman').merge(balls,on="batsman").merge(dismissals,on="batsman").merge(dots,on="batsman").merge(ones,on="batsman").merge(twos,on="batsman").merge(threes,on="batsman").merge(fours,on="batsman").merge(sixes,on="batsman")
    
    df["SR"]=df.apply(lambda x: 100*(x["runs"]/x["balls"]),axis=1)
    df["RPI"]=df.apply(lambda x: (x["runs"]/x["innings"]),axis=1)
    
    
    return df


# In[44]:


ing1_df = ByInning(deliveries, 1)
ing2_df = ByInning(deliveries, 2)


# In[47]:


ing1_df.head(1)


# In[48]:


ing2_df.head(1)


# In[53]:


full_df=ing1_df[["batsman","RPI"]].merge(ing2_df[["batsman","RPI"]],on="batsman",how="left").rename(columns={"RPI_x":"1st_RPI","RPI_y":"2nd_RPI"})


# In[54]:


full_df.head(3)


# In[60]:


plt.scatter(full_df["1st_RPI"],full_df["2nd_RPI"])
plt.xlabel("1st Innings RPI")
plt.ylabel("2nd Innings RPI")
plt.title("Batsman 1st and 2nd innings RPI comparison")
annotations=list(full_df['batsman'])
players=["V Kohli","MS Dhoni","A Badoni","DA Warner","AB de Villiers","A Ashish Reddy"]
for i,j in enumerate(annotations):
    if j in players:
        plt.annotate(j,(full_df["1st_RPI"][i],full_df["2nd_RPI"][i]))
plt.show()        
    


# ### 4. Performance by Opposition: 

# In[61]:


def ByOpposition(df,current_opposition):
    df=df[df.bowling_team ==current_opposition]
    df.reset_index(inplace=True, drop=True)
    
    df["Dot"]=df["batsman_runs"].apply(lambda x:1 if x==0 else 0)
    df["1s"]=df["batsman_runs"].apply(lambda x:1 if x==1 else 0)
    df["2s"]=df["batsman_runs"].apply(lambda x:1 if x==2 else 0)
    df["3s"]=df["batsman_runs"].apply(lambda x:1 if x==3 else 0)
    df["4s"]=df["batsman_runs"].apply(lambda x:1 if x==4 else 0)
    df["6s"]=df["batsman_runs"].apply(lambda x:1 if x==6 else 0)
    
    
    runs=pd.DataFrame(df.groupby("batsman")["batsman_runs"].sum()).reset_index().rename(columns={"batsman_runs":"runs"})
    innings=pd.DataFrame(df.groupby("batsman")["matchId"].apply(lambda x: len(list(np.unique(x))))).reset_index().rename(columns={"matchId":"innings"})
    balls=pd.DataFrame(df.groupby("batsman")["matchId"].count()).reset_index().rename(columns={"matchId":"balls"})
    dismissals=pd.DataFrame(df.groupby("batsman")["player_dismissed"].count()).reset_index().rename(columns={"player_dismissed":"dismissals"})
    
    dots=pd.DataFrame(df.groupby("batsman")["Dot"].sum()).reset_index()
    ones=pd.DataFrame(df.groupby("batsman")["1s"].sum()).reset_index()
    twos=pd.DataFrame(df.groupby("batsman")["2s"].sum()).reset_index()
    threes=pd.DataFrame(df.groupby("batsman")["3s"].sum()).reset_index()
    fours=pd.DataFrame(df.groupby("batsman")["4s"].sum()).reset_index()
    sixes=pd.DataFrame(df.groupby("batsman")["6s"].sum()).reset_index()
    
    df=pd.merge(innings,runs,on='batsman').merge(balls,on="batsman").merge(dismissals,on="batsman").merge(dots,on="batsman").merge(ones,on="batsman").merge(twos,on="batsman").merge(threes,on="batsman").merge(fours,on="batsman").merge(sixes,on="batsman")
    
    df["SR"]=df.apply(lambda x: 100*(x["runs"]/x["balls"]),axis=1)
    df["RPI"]=df.apply(lambda x: (x["runs"]/x["innings"]),axis=1)
    
    
    return df


# In[62]:


deliveries.bowling_team.unique()


# In[64]:


ByOpposition(deliveries,"Chennai Super Kings").head()


# ### 5. Performances by Venue 

# In[69]:


matches.head()


# In[81]:


df=deliveries.merge(matches[["matchId","venue"]],on="matchId",how="left")


# In[82]:


def ByVenue(df,current_venue):
    df=df[df.venue==current_venue]
    df.reset_index(inplace=True, drop=True)
    
    df["Dot"]=df["batsman_runs"].apply(lambda x:1 if x==0 else 0)
    df["1s"]=df["batsman_runs"].apply(lambda x:1 if x==1 else 0)
    df["2s"]=df["batsman_runs"].apply(lambda x:1 if x==2 else 0)
    df["3s"]=df["batsman_runs"].apply(lambda x:1 if x==3 else 0)
    df["4s"]=df["batsman_runs"].apply(lambda x:1 if x==4 else 0)
    df["6s"]=df["batsman_runs"].apply(lambda x:1 if x==6 else 0)
    
    
    runs=pd.DataFrame(df.groupby("batsman")["batsman_runs"].sum()).reset_index().rename(columns={"batsman_runs":"runs"})
    innings=pd.DataFrame(df.groupby("batsman")["matchId"].apply(lambda x: len(list(np.unique(x))))).reset_index().rename(columns={"matchId":"innings"})
    balls=pd.DataFrame(df.groupby("batsman")["matchId"].count()).reset_index().rename(columns={"matchId":"balls"})
    dismissals=pd.DataFrame(df.groupby("batsman")["player_dismissed"].count()).reset_index().rename(columns={"player_dismissed":"dismissals"})
    
    dots=pd.DataFrame(df.groupby("batsman")["Dot"].sum()).reset_index()
    ones=pd.DataFrame(df.groupby("batsman")["1s"].sum()).reset_index()
    twos=pd.DataFrame(df.groupby("batsman")["2s"].sum()).reset_index()
    threes=pd.DataFrame(df.groupby("batsman")["3s"].sum()).reset_index()
    fours=pd.DataFrame(df.groupby("batsman")["4s"].sum()).reset_index()
    sixes=pd.DataFrame(df.groupby("batsman")["6s"].sum()).reset_index()
    
    df=pd.merge(innings,runs,on='batsman').merge(balls,on="batsman").merge(dismissals,on="batsman").merge(dots,on="batsman").merge(ones,on="batsman").merge(twos,on="batsman").merge(threes,on="batsman").merge(fours,on="batsman").merge(sixes,on="batsman")
    
    df["SR"]=df.apply(lambda x: 100*(x["runs"]/x["balls"]),axis=1)
    df["RPI"]=df.apply(lambda x: (x["runs"]/x["innings"]),axis=1)
    
    
    return df


# In[84]:


df.head()


# ### Filters:
# * current_venue= MA Chidambaram Stadium, Chepauk
# * current_phase= Middle overs
# * current_opposition= Chennai Super Kings

# In[85]:


df.venue.unique()


# In[88]:


def ByCustom(df, current_venue, current_phase, current_opposition):
    
    df = df[df.venue == current_venue]
    df = df[df.phase == current_phase]
    df = df[df.bowling_team == current_opposition]
    
    df.reset_index(inplace=True, drop=True)
    df["Dot"]=df["batsman_runs"].apply(lambda x:1 if x==0 else 0)
    df["1s"]=df["batsman_runs"].apply(lambda x:1 if x==1 else 0)
    df["2s"]=df["batsman_runs"].apply(lambda x:1 if x==2 else 0)
    df["3s"]=df["batsman_runs"].apply(lambda x:1 if x==3 else 0)
    df["4s"]=df["batsman_runs"].apply(lambda x:1 if x==4 else 0)
    df["6s"]=df["batsman_runs"].apply(lambda x:1 if x==6 else 0)
    
    
    runs=pd.DataFrame(df.groupby("batsman")["batsman_runs"].sum()).reset_index().rename(columns={"batsman_runs":"runs"})
    innings=pd.DataFrame(df.groupby("batsman")["matchId"].apply(lambda x: len(list(np.unique(x))))).reset_index().rename(columns={"matchId":"innings"})
    balls=pd.DataFrame(df.groupby("batsman")["matchId"].count()).reset_index().rename(columns={"matchId":"balls"})
    dismissals=pd.DataFrame(df.groupby("batsman")["player_dismissed"].count()).reset_index().rename(columns={"player_dismissed":"dismissals"})
    
    dots=pd.DataFrame(df.groupby("batsman")["Dot"].sum()).reset_index()
    ones=pd.DataFrame(df.groupby("batsman")["1s"].sum()).reset_index()
    twos=pd.DataFrame(df.groupby("batsman")["2s"].sum()).reset_index()
    threes=pd.DataFrame(df.groupby("batsman")["3s"].sum()).reset_index()
    fours=pd.DataFrame(df.groupby("batsman")["4s"].sum()).reset_index()
    sixes=pd.DataFrame(df.groupby("batsman")["6s"].sum()).reset_index()
    
    df=pd.merge(innings,runs,on='batsman').merge(balls,on="batsman").merge(dismissals,on="batsman").merge(dots,on="batsman").merge(ones,on="batsman").merge(twos,on="batsman").merge(threes,on="batsman").merge(fours,on="batsman").merge(sixes,on="batsman")
    
    df["SR"]=df.apply(lambda x: 100*(x["runs"]/x["balls"]),axis=1)
    df["RPI"]=df.apply(lambda x: (x["runs"]/x["innings"]),axis=1)
    
    
    return df


# In[89]:


df = ByCustom(df, "MA Chidambaram Stadium, Chepauk", "Middle_Overs", "Chennai Super Kings")


# In[90]:


df


# In[96]:


wt_sr,wt_rpi=0.13,0.27


# In[97]:


df=df[df.innings>2]


# ### Calculations 

# In[101]:


df["calc_SR"] = df["SR"].apply(lambda x: x*x) 
df["calc_RPI"] = df["RPI"].apply(lambda x: x*x)

sq_sr, sq_rpi=np.sqrt(df[["calc_SR","calc_RPI"]].sum(axis=0))

df["calc_SR"] = df["calc_SR"].apply(lambda x: x/sq_sr) 
df["calc_RPI"] = df["calc_RPI"].apply(lambda x: x/sq_rpi) 

df["calc_SR"] = df["calc_SR"].apply(lambda x: x*wt_sr) 
df["calc_RPI"] = df["calc_RPI"].apply(lambda x: x*wt_rpi) 

best_sr, worst_sr = max(df["calc_SR"]), min(df["calc_SR"])
best_rpi, worst_rpi = max(df["calc_RPI"]), min(df["calc_RPI"])


# ### Calculations and Comparisons

# In[102]:


df["dev_best_SR"] = df["calc_SR"].apply(lambda x: (x-best_sr)*(x-best_sr)) 
df["dev_best_RPI"] = df["calc_RPI"].apply(lambda x: (x-best_rpi)*(x-best_rpi)) 
df["dev_best_sqrt"] = df.apply(lambda x: x["dev_best_SR"] + x["dev_best_RPI"],axis=1)

df["dev_worst_SR"] = df["calc_SR"].apply(lambda x: (x-worst_sr)*(x-worst_sr)) 
df["dev_worst_RPI"] = df["calc_RPI"].apply(lambda x: (x-worst_rpi)*(x-worst_rpi)) 
df["dev_worst_sqrt"] = df.apply(lambda x: x["dev_worst_SR"] + x["dev_worst_RPI"],axis=1) 


# In[103]:


df["score"] = df.apply(lambda x: x["dev_worst_sqrt"]/(x["dev_worst_sqrt"] + x["dev_best_sqrt"]), axis = 1)


# In[104]:


df[["batsman","score"]].head()


# ### End Results 

# In[108]:


df[[ "batsman","innings", "runs", "balls", "dismissals","score"]].sort_values("score",ascending=False)


# In[ ]:




