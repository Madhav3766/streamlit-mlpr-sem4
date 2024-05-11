import streamlit as st
import pandas as pd
import json

# Load the datasets
matches_df = pd.read_csv('updated-matches-dataset.csv')
dream11_teams_df = pd.read_csv('dream11_teams_streamlit.csv')
all_players_df = pd.read_csv('dream11_all_players_streamlit.csv')
player_roles_df = pd.read_csv('coded-roles-output (1).csv')
model_teams_df = pd.read_csv('output_file-maddy.csv')  # Load the model-predicted Dream11 teams

def get_matches(season, team1, team2):
    return matches_df[(matches_df['Season'] == season) &
                      ((matches_df['Team1'] == team1) & (matches_df['Team2'] == team2) |
                       (matches_df['Team1'] == team2) & (matches_df['Team2'] == team1))]

def get_dream11_team(match_id):
    team_data = dream11_teams_df[dream11_teams_df['match_id'] == match_id]
    if not team_data.empty:
        team_str = team_data.iloc[0]['dream11_team']
        team_json = json.loads(team_str.replace("\'", "\""))
        team_df = pd.DataFrame(team_json)
        team_df = pd.merge(team_df, player_roles_df, left_on='player', right_on='name', how='left')
        team_df['Index'] = range(1, len(team_df) + 1)
        team_df = team_df.set_index('Index')
        return team_df, team_data.iloc[0]['total_team_points']
    return pd.DataFrame(), 0

def get_model_team(match_id):
    team_data = model_teams_df[model_teams_df['match_id'] == match_id]
    if not team_data.empty:
        # Assuming the 'Player' column contains serialized JSON of players, similar to other data manipulations in your code
        team_str = team_data.iloc[0]['Player']
        team_list = json.loads(team_str.replace("\'", "\""))
        team_df = pd.DataFrame(team_list, columns=['player'])
        team_df = pd.merge(team_df, player_roles_df, left_on='player', right_on='name', how='left')
        team_df['Index'] = range(1, len(team_df) + 1)
        team_df = team_df.set_index('Index')
        return team_df
    return pd.DataFrame()

def get_players(match_id):
    players_data = all_players_df[all_players_df['match_id'] == match_id].iloc[0]['players']
    players_json = json.loads(players_data.replace("\'", "\""))
    players_df = pd.DataFrame(players_json)
    players_df = pd.merge(players_df, player_roles_df, left_on='player', right_on='name', how='left')
    return players_df
    
def validate_team(selected_players):
    if len(selected_players) != 11:
        return False, "You must select exactly 11 players."
    
    roles_count = selected_players['role'].value_counts()
    required_roles = ['BAT', 'BWL', 'AR', 'WK']
    for role in required_roles:
        if role not in roles_count or roles_count[role] < 1:
            return False, f"You must select at least one player from each category (BAT, BWL, AR, WK). Missing: {role}"
    
    return True, ""

def calculate_user_team_score(selected_players):
    points = selected_players['points']
    points_sorted = points.sort_values(ascending=False)
    adjusted_points = points_sorted.copy()
    if len(points_sorted) > 1:
        adjusted_points.iloc[0] *= 2
        adjusted_points.iloc[1] *= 1.5
    return adjusted_points, sum(adjusted_points)

def calculate_accuracy(real_team_df, predicted_team_df):
    real_players = set(real_team_df['player'])
    predicted_players = set(predicted_team_df['player'])
    intersection = real_players.intersection(predicted_players)
    accuracy = len(intersection) / 11  # Assuming 11 players in a Dream11 team
    return accuracy, len(intersection)

st.title('Fantasy Best Team Predictor (T20 IPL)')
st.sidebar.header('User Input Features')

selected_season = st.sidebar.selectbox("Select Season", matches_df['Season'].unique())
season_teams = set(matches_df.loc[matches_df['Season'] == selected_season, ['Team1', 'Team2']].values.flatten())
selected_team1 = st.sidebar.selectbox("Select Team 1", sorted(season_teams))
team2_options = sorted(season_teams - {selected_team1})
selected_team2 = st.sidebar.selectbox("Select Team 2", team2_options)

matches = get_matches(selected_season, selected_team1, selected_team2)
if not matches.empty:
    match_options = dict(zip(matches['Date'], matches['ID']))
    selected_date = st.selectbox('Select Match Date', list(match_options.keys()))
    selected_match = match_options[selected_date]
    match_details = matches[matches['ID'] == selected_match]
    st.write('Match Details:')
    st.write(match_details[['City', 'Date', 'Venue']])

    players_df = get_players(selected_match)
    if not players_df.empty:
        player_options = {f"{row['player']} ({row['role']})": row['player'] for index, row in players_df.iterrows()}
        user_team = st.multiselect("Select Your Dream11 Team (Pick 11 Players):", options=player_options.keys(), format_func=lambda x: x)
        if user_team:
            user_team_data = players_df[players_df['player'].isin([player_options[name] for name in user_team])]
            is_valid, message = validate_team(user_team_data)
            if is_valid:
                user_team_data['Index'] = range(1, len(user_team_data) + 1)
                user_team_data = user_team_data.set_index('Index')
                adjusted_points, user_team_score = calculate_user_team_score(user_team_data)
                user_team_data['adjusted_points'] = adjusted_points.values
                st.write('Your Dream11 Team Configuration:')
                st.dataframe(user_team_data[['player', 'role', 'adjusted_points']])
                st.write(f'Your Total Team Points: {user_team_score}')
            else:
                st.error(message)

    if st.button("Show Best Dream11 Team and Model Predicted Dream11 Team"):
        dream11_team_df, total_points = get_dream11_team(selected_match)
        model_team_df = get_model_team(selected_match)
        col1, col2 = st.columns(2)
        with col1:
            st.write('Best Dream11 Team Configuration:')
            if not dream11_team_df.empty:
                st.dataframe(dream11_team_df[['player', 'role', 'points']])
                st.write(f'Total Team Points: {total_points}')
        with col2:
            st.write('Model Predicted Dream11 Team:')
            if not model_team_df.empty:
                st.dataframe(model_team_df[['player', 'role']])
        
            if not dream11_team_df.empty and not model_team_df.empty:
                accuracy, correct_predictions = calculate_accuracy(dream11_team_df, model_team_df)
                st.write(f"Model Accuracy: {accuracy:.2%} ({correct_predictions}/11)")
else:
    st.write('No matches found for the selected criteria.')


st.empty()
st.markdown('---')
st.empty()

roles_df = pd.read_csv('coded-roles-output (1).csv')

# Define the teams with player names
team_1 = {'Mumbai Indians': ['Ishan Kishan', 'RG Sharma', 'Naman Dhir', 'SA Yadav', 'Tilak Varma', 'HH Pandya', 'TH David', 'PP Chawla', 'JJ Bumrah', 'N Thushara', 'N Wadhera', 'SZ Mulani']}
team_2 = {'Kolkata Knight Riders': ['PD Salt', 'SP Narine', 'A Raghuvanshi', 'SS Iyer', 'VR Iyer', 'RK Singh', 'AD Russell', 'Ramandeep Singh', 'MA Starc', 'CV Varun', 'Harshit Rana', 'VG Arora']}

# Convert team lists to DataFrame
team_1_df = pd.DataFrame({'player': team_1['Mumbai Indians']})
team_2_df = pd.DataFrame({'player': team_2['Kolkata Knight Riders']})

# Merge the team dataframes with the roles dataframe
team_1_df = team_1_df.merge(roles_df, left_on='player', right_on='name', how='left').drop(columns=['name'])
team_2_df = team_2_df.merge(roles_df, left_on='player', right_on='name', how='left').drop(columns=['name'])

# Load the predicted team data
predicted_team_df = pd.read_excel('mi-vs-kkr-predicted-team.xlsx')
predicted_team_df = predicted_team_df.merge(roles_df, left_on='Player Name', right_on='name', how='left').drop(columns=['name'])

# Apply the team finding function from combined teams dictionary
teams = {**team_1, **team_2}  # Combining both dictionaries
predicted_team_df['Team'] = predicted_team_df['Player Name'].apply(lambda player: next((team for team, players in teams.items() if player in players), "Team not found"))

# Limit to the top 11 players
predicted_team_df = predicted_team_df.head(11)

# Setting a new index starting from 1
predicted_team_df['Index'] = range(1, len(predicted_team_df) + 1)
predicted_team_df.set_index('Index', inplace=True)

# Display in Streamlit
st.title('Today\'s IPL Match Team')

# Display static teams
col1, col2 = st.columns(2)
with col1:
    st.header('Mumbai Indians')
    st.dataframe(team_1_df.set_index(pd.Index(range(1, len(team_1_df) + 1))))

with col2:
    st.header('Kolkata Knight Riders')
    st.dataframe(team_2_df.set_index(pd.Index(range(1, len(team_2_df) + 1))))

# Display predicted team
st.header('LSTM Predicted Top 11 Players for Today\'s Match')
st.dataframe(predicted_team_df[['Player Name', 'role', 'Team']])

st.markdown("---")  # Visual separator
st.empty()  # Additional spacing

