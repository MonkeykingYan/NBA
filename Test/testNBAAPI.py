import os
import json

playerids = [2738, 202691, 101106, 2760, 2571, 203949, 203546,
             203110, 201939, 203105, 2733, 1626172, 203084]
# Calling API and store the results as JSON

import requests

url = "http://stats.nba.com/stats/playerdashptpass?"

querystring = {"DateFrom": "", "DateTo": "", "GameSegment": "", "LastNGames": "0", "LeagueID": "00", "Location": "",
               "Month": "0", "OpponentTeamID": "0", "Outcome": "", "PORound": "0", "PerMode": "PerGame", "Period": "0",
               "PlayerID": "203110", "Season": "2017-18", "SeasonSegment": "", "SeasonType": "Regular Season",
               "TeamID": "0", "VsConference": "", "VsDivision": ""}

headers = {
    'authorization': "Basic bW9ua2V5a2luZ3lhbkBnbWFpbC5jb206bWF5YW4xOTky",
    'cache-control': "no-cache",
    'postman-token': "6e9ffffd-c7dc-7b57-5716-197f17cfaac2"
}

response = requests.request("GET", url, headers=headers, params=querystring)

print(response.text)
# for playerid in playerids:
#     if playerid == 2733:
#         os.system('curl "http://stats.nba.com/stats/playerdashptpass?'
#             'DateFrom=&'
#             'DateTo=&'
#             'GameSegment=&'
#             'LastNGames=0&'
#             'LeagueID=00&'
#             'Location=&'
#             'Month=0&'
#             'OpponentTeamID=0&'
#             'Outcome=&'
#             'PerMode=Totals&'
#             'Period=0&'
#             'PlayerID={playerid}&'
#             'Season=2015-16&'
#             'SeasonSegment=&'
#             'SeasonType=Regular+Season&'
#             'TeamID=0&'
#             'VsConference=&'
#             'VsDivision=" > {playerid}.json'.format(playerid=playerid))
