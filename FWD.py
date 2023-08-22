import requests
import pandas as pd
import numpy as np
import matches_played as mp
from sklearn.linear_model import LinearRegression
import pickle
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
r = requests.get(url)
json = r.json()
elements_df = pd.DataFrame(json['elements'])
elements_types_df = pd.DataFrame(json['element_types'])
teams_df = pd.DataFrame(json['teams'])
elements_df=elements_df[
    (elements_df['chance_of_playing_next_round'] != 50.00) &
    (elements_df['chance_of_playing_next_round'] != 25.00) &
    (elements_df['chance_of_playing_next_round'] != 0.00)
    & (elements_df['points_per_game']!=0)
]
elements_df['total_points'] = elements_df.total_points.astype(float)
elements_df['points_per_game'] = elements_df.points_per_game.astype(float)
elements_df['matches_played']=mp.matches_played(elements_df['total_points'],elements_df['points_per_game'])
elements_df['matches_played']=elements_df['matches_played'].fillna(0)
elements_df['matches_played'] = elements_df.matches_played.astype(int)
elements_df['expected_goals'] = elements_df.expected_goals.astype(float)
elements_df['expected_assists'] = elements_df.expected_assists.astype(float)
elements_df['expected_goals_conceded'] = elements_df.expected_goals_conceded.astype(float)
elements_df['expected_goals']=elements_df['expected_goals']/elements_df['matches_played']
elements_df['expected_assists']=elements_df['expected_assists']/elements_df['matches_played']
elements_df['expected_goals_conceded']=elements_df['expected_goals_conceded']/elements_df['matches_played']
elements_df['expected_goals']=elements_df['expected_goals'].fillna(0)
elements_df['expected_assists']=elements_df['expected_assists'].fillna(0)
elements_df['expected_goals_conceded']=elements_df['expected_goals_conceded'].fillna(0)
elements_df['expected_goals']=round(elements_df['expected_goals'],2)
elements_df['expected_assists']=round(elements_df['expected_assists'],2)
elements_df['expected_goals_conceded']=round(elements_df['expected_goals_conceded'],2)
elements_df.replace([np.inf, -np.inf], 0, inplace=True)
slim_elements_df = elements_df[['second_name','element_type','selected_by_percent','expected_goals','expected_assists','transfers_in','transfers_out','expected_goals_conceded']]
slim_elements_df['position'] = slim_elements_df.element_type.map(elements_types_df.set_index('id').singular_name)
slim_elements_df['selected_by_percent'] = slim_elements_df.selected_by_percent.astype(float)
slim_elements_df['expected_goals'] = slim_elements_df.expected_goals.astype(float)
slim_elements_df['expected_assists'] = slim_elements_df.expected_assists.astype(float)
slim_elements_df['expected_goals_conceded'] = slim_elements_df.expected_goals_conceded.astype(float)
fwd_df = slim_elements_df.loc[slim_elements_df.position == 'Forward']
gws_dataset=pd.read_csv("merged_gw.csv")
gws_dataset=gws_dataset[['name','position','expected_goals','expected_assists','transfers_in','transfers_out','expected_goals_conceded','total_points']]
gws_dataset['expected_goals'] = gws_dataset.expected_goals.astype(float)
gws_dataset['expected_assists'] = gws_dataset.expected_assists.astype(float)
gws_dataset['expected_goals_conceded'] = gws_dataset.expected_goals_conceded.astype(float)
fwd_gws_dataset= gws_dataset.loc[gws_dataset.position == 'FWD']
fw_X=fwd_gws_dataset.iloc[:,2:6]
fw_Y=fwd_gws_dataset.iloc[:,-1].values
regressor_fw=LinearRegression()
regressor_fw.fit(fw_X,fw_Y)
fw_X_this_season=fwd_df.iloc[:,3:7]
fw_predicted_pts=regressor_fw.predict(fw_X_this_season)
fw_names=fwd_df.iloc[:,0]
fw_predicted_points_dataset=pd.DataFrame({'Name':fw_names
                 ,'Predicted_points':fw_predicted_pts})
fw_predicted_points_dataset['Predicted_points'] = fw_predicted_points_dataset.Predicted_points.astype(float)
pickle.dump(fw_predicted_points_dataset,open('FWD.pkl','wb'))
df=pickle.load(open('FWD.pkl','rb'))