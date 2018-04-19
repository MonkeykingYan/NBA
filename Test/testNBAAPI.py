import os
import requests

# # GSW player IDs
playerids = [201575, 201578, 2738, 202691, 101106, 2760, 2571, 203949, 203546,
             203110, 201939, 203105, 2733, 1626172, 203084]
#
# # Calling API and store the results as JSON
# for id in playerids:
#     os.system('curl "http://stats.nba.com/stats/playerdashptpass?'
#               'DateFrom=&'
#               'DateTo=&'
#               'GameSegment=&'
#               'LastNGames=0&'
#               'LeagueID=00&'
#               'Location=&'
#               'Month=0&'
#               'OpponentTeamID=0&'
#               'Outcome=&'
#               'PerMode=Totals&'
#               'Period=0&'
#               'PlayerID={id}&'
#               'Season=2015-16&'
#               'SeasonSegment=&'
#               'SeasonType=Regular+Season&'
#               'TeamID=0&'
#               'VsConference=&'
#               'VsDivision=" > {id}.json'.format(id=id))
import requests

url = "http://stats.nba.com/stats/playerdashptpass"

querystring = {"DateFrom":"","DateTo":"","GameSegment":"","LastNGames":"0","LeagueID":"00","Location":"","Month":"0","OpponentTeamID":"0","Outcome":"","PORound":"0","PerMode":"PerGame","Period":"0","PlayerID":"201942","Season":"2017-18","SeasonSegment":"","SeasonType":"Playoffs","TeamID":"0","VsConference":"","VsDivision":""}

headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'User-Agent' : 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2226.0 Safari/537.36'
    }

response = requests.request("GET", url, headers=headers, params=querystring)

print(response.text)
