#!/usr/bin/env python3
"""Fetch FIFA World Cup 2026 data and compute championship probabilities via Monte Carlo."""

import requests
import json
import os
import math
import random
from datetime import datetime, timezone
from collections import defaultdict

FRIENDS_TEAMS = {
    "Dhruv":    "France",
    "Aaditya":  "Portugal",
    "Varun":    "Portugal",
    "Mishika":  "Spain",
    "Saumya":   "Argentina",
    "Rohith":   "Argentina",
    "Sharanya": "Argentina",
}

TEAM_INFO = {
    "France":    {"flag": "🇫🇷", "color": "#002395", "secondary": "#ED2939",  "abbreviation": "FRA"},
    "Portugal":  {"flag": "🇵🇹", "color": "#006600", "secondary": "#FF0000",  "abbreviation": "POR"},
    "Spain":     {"flag": "🇪🇸", "color": "#c60b1e", "secondary": "#ffc400",  "abbreviation": "ESP"},
    "Argentina": {"flag": "🇦🇷", "color": "#74ACDF", "secondary": "#FFFFFF",  "abbreviation": "ARG"},
}

# Pre-tournament strength ratings (FIFA-ranking-based, scale 0-100)
BASE_STRENGTH = {
    "Argentina": 92, "France": 88, "England": 85, "Spain": 84,
    "Brazil": 83, "Portugal": 81, "Germany": 80, "Netherlands": 79,
    "Belgium": 76, "Croatia": 75, "Italy": 75, "Uruguay": 74,
    "Morocco": 72, "Colombia": 72, "Switzerland": 73, "Denmark": 72,
    "Turkey": 67, "USA": 68, "Mexico": 67, "Japan": 67,
    "Ecuador": 64, "Czech Republic": 64, "Scotland": 64, "Norway": 66,
    "Austria": 68, "Poland": 66, "Ukraine": 68, "Sweden": 67,
    "Serbia": 68, "Senegal": 70, "South Korea": 65, "Chile": 65,
    "Nigeria": 65, "Colombia": 72, "Venezuela": 60, "Paraguay": 62,
    "Peru": 63, "Bolivia": 52, "Costa Rica": 59, "Panama": 58,
    "Honduras": 55, "Jamaica": 54, "Canada": 63, "Saudi Arabia": 58,
    "Iran": 58, "Ghana": 60, "Côte d'Ivoire": 62, "Tunisia": 60,
    "Egypt": 63, "Algeria": 61, "Mali": 60, "DR Congo": 58,
    "South Africa": 57, "Australia": 62, "New Zealand": 55,
    "Slovenia": 63, "Slovakia": 62, "Hungary": 60, "Romania": 62,
    "Georgia": 58, "Israel": 58, "Greece": 60, "Finland": 58,
    "Iceland": 58, "Wales": 61, "Iraq": 57, "Jordan": 55,
    "Bahrain": 52, "UAE": 53, "Uzbekistan": 57, "Qatar": 55,
}

TRACKED_TEAMS = set(FRIENDS_TEAMS.values())


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_football_data_org():
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    if not api_key:
        return None, None
    headers = {"X-Auth-Token": api_key}
    try:
        s = requests.get("https://api.football-data.org/v4/competitions/WC/standings",
                         headers=headers, timeout=15)
        m = requests.get("https://api.football-data.org/v4/competitions/WC/matches?status=FINISHED",
                         headers=headers, timeout=15)
        if s.status_code == 200:
            return s.json(), m.json() if m.status_code == 200 else None
    except Exception as e:
        print(f"football-data.org error: {e}")
    return None, None


def fetch_espn_data():
    slugs = ["fifa.world", "fifa.world.2026", "soccer.world"]
    for slug in slugs:
        try:
            r = requests.get(
                f"https://site.api.espn.com/apis/v2/sports/soccer/{slug}/standings",
                timeout=15)
            if r.status_code == 200:
                data = r.json()
                if data.get("standings") or data.get("children"):
                    print(f"ESPN OK: {slug}")
                    return data, slug
        except Exception as e:
            print(f"ESPN {slug} error: {e}")
    return None, None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_fdo_standings(data):
    groups = {}
    for s in data.get("standings", []):
        if s.get("type") != "TOTAL":
            continue
        gname = s.get("group", "Group ?")
        teams = []
        for e in s.get("table", []):
            t = e.get("team", {})
            teams.append({
                "team":     t.get("name", "Unknown"),
                "position": e.get("position", 0),
                "played":   e.get("playedGames", 0),
                "won":      e.get("won", 0),
                "drawn":    e.get("draw", 0),
                "lost":     e.get("lost", 0),
                "gf":       e.get("goalsFor", 0),
                "ga":       e.get("goalsAgainst", 0),
                "gd":       e.get("goalDifference", 0),
                "points":   e.get("points", 0),
            })
        groups[gname] = sorted(teams, key=lambda x: (-x["points"], -x["gd"], -x["gf"]))
    return groups


def parse_fdo_matches(data):
    if not data:
        return []
    matches = []
    for m in data.get("matches", []):
        home = m.get("homeTeam", {}).get("name", "TBD")
        away = m.get("awayTeam", {}).get("name", "TBD")
        ft   = m.get("score", {}).get("fullTime", {})
        matches.append({
            "home":       home,
            "away":       away,
            "home_score": ft.get("home"),
            "away_score": ft.get("away"),
            "stage":      m.get("stage", "GROUP_STAGE"),
            "date":       m.get("utcDate", "")[:10],
            "status":     m.get("status", "SCHEDULED"),
        })
    return matches


# ---------------------------------------------------------------------------
# Probability engine
# ---------------------------------------------------------------------------

def win_prob(strength_a, strength_b):
    """Logistic win probability for team A vs B."""
    return 1 / (1 + math.exp(-(strength_a - strength_b) / 15))


def sim_group_match(a, b, ratings):
    """Returns (winner, loser) or (None, None) for draw."""
    pa = win_prob(ratings.get(a, 60), ratings.get(b, 60))
    draw = 0.28 * (1 - abs(pa - 0.5) * 1.6)
    pa  *= (1 - draw)
    pb   = 1 - pa - draw
    r = random.random()
    if r < pa:
        return a, b
    elif r < pa + draw:
        return None, None
    return b, a


def sim_ko_match(a, b, ratings):
    """Knockout match — no draws, slight upset adjustment."""
    pa = win_prob(ratings.get(a, 60), ratings.get(b, 60))
    pa = 0.65 * pa + 0.35 * 0.5  # compress towards 50/50 under pressure
    return a if random.random() < pa else b


def adjust_ratings(base, groups):
    adj = dict(base)
    for teams in groups.values():
        for t in teams:
            if t["played"] == 0:
                continue
            delta = (t["points"] / t["played"] - 1.4) * 5 + t["gd"] * 0.4
            adj[t["team"]] = adj.get(t["team"], 60) + delta
    return adj


def simulate_tournament(groups, adjusted_ratings, n=8000):
    wins = defaultdict(int)
    finals = defaultdict(int)
    semis = defaultdict(int)
    quarters = defaultdict(int)

    for _ in range(n):
        # ── Group stage ──────────────────────────────────────────────────
        first, second, thirds = [], [], []
        for g_teams in groups.values():
            cur = {t["team"]: {"pts": t["points"], "gd": t["gd"], "gf": t["gf"],
                               "played": t["played"]}
                   for t in g_teams}
            team_list = [t["team"] for t in g_teams]
            games_each = 3
            for i, ta in enumerate(team_list):
                for tb in team_list[i+1:]:
                    if cur[ta]["played"] < games_each or cur[tb]["played"] < games_each:
                        w, l = sim_group_match(ta, tb, adjusted_ratings)
                        if w:
                            cur[w]["pts"] += 3; cur[w]["gd"] += 1; cur[w]["gf"] += 1
                            cur[l]["gd"] -= 1
                        else:
                            cur[ta]["pts"] += 1; cur[tb]["pts"] += 1
            ranked = sorted(team_list, key=lambda t: (-cur[t]["pts"], -cur[t]["gd"], -cur[t]["gf"]))
            first.append(ranked[0])
            second.append(ranked[1])
            thirds.append({"team": ranked[2], "pts": cur[ranked[2]]["pts"],
                           "gd": cur[ranked[2]]["gd"], "gf": cur[ranked[2]]["gf"]})

        # 8 best 3rd-place teams advance (WC 2026 format)
        best_thirds = [t["team"] for t in sorted(thirds, key=lambda x: (-x["pts"], -x["gd"], -x["gf"]))[:8]]
        r32 = first + second + best_thirds
        random.shuffle(r32)

        # ── Knockout rounds ──────────────────────────────────────────────
        rnames = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]
        current = r32
        for rname in rnames:
            nxt = []
            for i in range(0, len(current) - 1, 2):
                a, b = current[i], current[i+1]
                if rname == "Quarterfinals":
                    quarters[a] += 1; quarters[b] += 1
                elif rname == "Semifinals":
                    semis[a] += 1; semis[b] += 1
                elif rname == "Final":
                    finals[a] += 1; finals[b] += 1
                w = sim_ko_match(a, b, adjusted_ratings)
                nxt.append(w)
            current = nxt
            if len(current) == 1:
                wins[current[0]] += 1
                break

    pct = lambda d: {k: round(v / n * 100, 1) for k, v in d.items()}
    return {"win": pct(wins), "final": pct(finals), "semis": pct(semis), "quarters": pct(quarters)}


# ---------------------------------------------------------------------------
# Build output JSON
# ---------------------------------------------------------------------------

def get_form(matches, team):
    form = []
    for m in reversed(matches):
        if m.get("status") != "FINISHED" or len(form) >= 5:
            continue
        hs, as_ = m.get("home_score"), m.get("away_score")
        if hs is None or as_ is None:
            continue
        if m["home"] == team:
            form.append("W" if hs > as_ else ("D" if hs == as_ else "L"))
        elif m["away"] == team:
            form.append("W" if as_ > hs else ("D" if as_ == hs else "L"))
    return form


def build_output(groups, matches, probs):
    teams_data = {}
    for team, info in TEAM_INFO.items():
        grp_name, grp_entry = None, None
        for gn, gt in groups.items():
            for t in gt:
                if t["team"].lower() == team.lower():
                    grp_name, grp_entry = gn, t
                    break
        teams_data[team] = {
            **info,
            "group":                grp_name,
            "played":               grp_entry["played"]  if grp_entry else 0,
            "won":                  grp_entry["won"]     if grp_entry else 0,
            "drawn":                grp_entry["drawn"]   if grp_entry else 0,
            "lost":                 grp_entry["lost"]    if grp_entry else 0,
            "gf":                   grp_entry["gf"]      if grp_entry else 0,
            "ga":                   grp_entry["ga"]      if grp_entry else 0,
            "gd":                   grp_entry["gd"]      if grp_entry else 0,
            "points":               grp_entry["points"]  if grp_entry else 0,
            "position":             grp_entry.get("position") if grp_entry else None,
            "form":                 get_form(matches, team),
            "eliminated":           False,
            "win_probability":      probs["win"].get(team, 0),
            "final_probability":    probs["final"].get(team, 0),
            "semis_probability":    probs["semis"].get(team, 0),
            "quarters_probability": probs["quarters"].get(team, 0),
        }

    tracked_matches = [m for m in matches
                       if m.get("home") in TRACKED_TEAMS or m.get("away") in TRACKED_TEAMS]

    total_played = sum(t["played"] for gt in groups.values() for t in gt)
    stage = "Knockout Stage" if total_played >= 72 else "Group Stage"

    return {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "tournament_stage": stage,
        "data_source": "live",
        "friends": {
            name: {
                "team":  team,
                "flag":  TEAM_INFO[team]["flag"]  if team in TEAM_INFO else "❓",
                "color": TEAM_INFO[team]["color"] if team in TEAM_INFO else "#555",
            }
            for name, team in FRIENDS_TEAMS.items()
        },
        "teams": teams_data,
        "recent_matches": tracked_matches[-10:],
        "all_groups": {
            gn: [{"team": t["team"], "played": t["played"], "won": t["won"],
                  "drawn": t["drawn"], "lost": t["lost"],
                  "gf": t["gf"], "ga": t["ga"], "gd": t["gd"], "points": t["points"]}
                 for t in gt]
            for gn, gt in groups.items()
        },
    }


# ---------------------------------------------------------------------------
# Demo / fallback data
# ---------------------------------------------------------------------------

DEMO_GROUPS = {
    "Group A": [
        {"team": "Argentina",    "position": 1, "played": 2, "won": 2, "drawn": 0, "lost": 0, "gf": 5, "ga": 1, "gd":  4, "points": 6},
        {"team": "Ecuador",      "position": 2, "played": 2, "won": 1, "drawn": 0, "lost": 1, "gf": 2, "ga": 3, "gd": -1, "points": 3},
        {"team": "Saudi Arabia", "position": 3, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 1, "ga": 2, "gd": -1, "points": 1},
        {"team": "Iceland",      "position": 4, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 1, "ga": 3, "gd": -2, "points": 1},
    ],
    "Group B": [
        {"team": "France",       "position": 1, "played": 2, "won": 2, "drawn": 0, "lost": 0, "gf": 4, "ga": 1, "gd":  3, "points": 6},
        {"team": "Austria",      "position": 2, "played": 2, "won": 1, "drawn": 0, "lost": 1, "gf": 2, "ga": 3, "gd": -1, "points": 3},
        {"team": "Morocco",      "position": 3, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 1, "ga": 2, "gd": -1, "points": 1},
        {"team": "Iraq",         "position": 4, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 1, "ga": 2, "gd": -1, "points": 1},
    ],
    "Group C": [
        {"team": "Spain",        "position": 1, "played": 2, "won": 1, "drawn": 1, "lost": 0, "gf": 3, "ga": 1, "gd":  2, "points": 4},
        {"team": "Japan",        "position": 2, "played": 2, "won": 1, "drawn": 0, "lost": 1, "gf": 2, "ga": 2, "gd":  0, "points": 3},
        {"team": "Colombia",     "position": 3, "played": 2, "won": 0, "drawn": 2, "lost": 0, "gf": 2, "ga": 2, "gd":  0, "points": 2},
        {"team": "Jordan",       "position": 4, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 0, "ga": 2, "gd": -2, "points": 1},
    ],
    "Group D": [
        {"team": "Portugal",     "position": 1, "played": 2, "won": 2, "drawn": 0, "lost": 0, "gf": 4, "ga": 0, "gd":  4, "points": 6},
        {"team": "Uruguay",      "position": 2, "played": 2, "won": 1, "drawn": 0, "lost": 1, "gf": 2, "ga": 2, "gd":  0, "points": 3},
        {"team": "Ghana",        "position": 3, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 1, "ga": 3, "gd": -2, "points": 1},
        {"team": "Bahrain",      "position": 4, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 0, "ga": 2, "gd": -2, "points": 1},
    ],
    "Group E": [
        {"team": "Germany",      "position": 1, "played": 2, "won": 2, "drawn": 0, "lost": 0, "gf": 5, "ga": 1, "gd":  4, "points": 6},
        {"team": "USA",          "position": 2, "played": 2, "won": 1, "drawn": 0, "lost": 1, "gf": 2, "ga": 3, "gd": -1, "points": 3},
        {"team": "Ghana",        "position": 3, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 1, "ga": 2, "gd": -1, "points": 1},
        {"team": "Bolivia",      "position": 4, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 0, "ga": 2, "gd": -2, "points": 1},
    ],
    "Group F": [
        {"team": "Brazil",       "position": 1, "played": 2, "won": 2, "drawn": 0, "lost": 0, "gf": 6, "ga": 1, "gd":  5, "points": 6},
        {"team": "Netherlands",  "position": 2, "played": 2, "won": 1, "drawn": 0, "lost": 1, "gf": 2, "ga": 3, "gd": -1, "points": 3},
        {"team": "Senegal",      "position": 3, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 1, "ga": 2, "gd": -1, "points": 1},
        {"team": "Iran",         "position": 4, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 0, "ga": 3, "gd": -3, "points": 1},
    ],
    "Group G": [
        {"team": "England",      "position": 1, "played": 2, "won": 2, "drawn": 0, "lost": 0, "gf": 4, "ga": 0, "gd":  4, "points": 6},
        {"team": "Nigeria",      "position": 2, "played": 2, "won": 1, "drawn": 0, "lost": 1, "gf": 2, "ga": 2, "gd":  0, "points": 3},
        {"team": "Venezuela",    "position": 3, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 1, "ga": 2, "gd": -1, "points": 1},
        {"team": "Thailand",     "position": 4, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 0, "ga": 3, "gd": -3, "points": 1},
    ],
    "Group H": [
        {"team": "Croatia",      "position": 1, "played": 2, "won": 1, "drawn": 1, "lost": 0, "gf": 3, "ga": 1, "gd":  2, "points": 4},
        {"team": "Italy",        "position": 2, "played": 2, "won": 1, "drawn": 0, "lost": 1, "gf": 2, "ga": 2, "gd":  0, "points": 3},
        {"team": "Turkey",       "position": 3, "played": 2, "won": 0, "drawn": 2, "lost": 0, "gf": 2, "ga": 2, "gd":  0, "points": 2},
        {"team": "Jamaica",      "position": 4, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 0, "ga": 2, "gd": -2, "points": 1},
    ],
    "Group I": [
        {"team": "Belgium",      "position": 1, "played": 2, "won": 2, "drawn": 0, "lost": 0, "gf": 4, "ga": 0, "gd":  4, "points": 6},
        {"team": "South Korea",  "position": 2, "played": 2, "won": 1, "drawn": 0, "lost": 1, "gf": 2, "ga": 2, "gd":  0, "points": 3},
        {"team": "Egypt",        "position": 3, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 1, "ga": 2, "gd": -1, "points": 1},
        {"team": "Canada",       "position": 4, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 0, "ga": 3, "gd": -3, "points": 1},
    ],
    "Group J": [
        {"team": "Morocco",      "position": 1, "played": 2, "won": 1, "drawn": 1, "lost": 0, "gf": 3, "ga": 1, "gd":  2, "points": 4},
        {"team": "Switzerland",  "position": 2, "played": 2, "won": 1, "drawn": 0, "lost": 1, "gf": 2, "ga": 2, "gd":  0, "points": 3},
        {"team": "Chile",        "position": 3, "played": 2, "won": 0, "drawn": 2, "lost": 0, "gf": 1, "ga": 1, "gd":  0, "points": 2},
        {"team": "Honduras",     "position": 4, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 0, "ga": 2, "gd": -2, "points": 1},
    ],
    "Group K": [
        {"team": "Mexico",       "position": 1, "played": 2, "won": 1, "drawn": 1, "lost": 0, "gf": 3, "ga": 1, "gd":  2, "points": 4},
        {"team": "Serbia",       "position": 2, "played": 2, "won": 1, "drawn": 0, "lost": 1, "gf": 2, "ga": 2, "gd":  0, "points": 3},
        {"team": "Côte d'Ivoire","position": 3, "played": 2, "won": 0, "drawn": 2, "lost": 0, "gf": 2, "ga": 2, "gd":  0, "points": 2},
        {"team": "Panama",       "position": 4, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 0, "ga": 2, "gd": -2, "points": 1},
    ],
    "Group L": [
        {"team": "Denmark",      "position": 1, "played": 2, "won": 2, "drawn": 0, "lost": 0, "gf": 4, "ga": 1, "gd":  3, "points": 6},
        {"team": "Australia",    "position": 2, "played": 2, "won": 1, "drawn": 0, "lost": 1, "gf": 2, "ga": 2, "gd":  0, "points": 3},
        {"team": "Ukraine",      "position": 3, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 1, "ga": 2, "gd": -1, "points": 1},
        {"team": "Algeria",      "position": 4, "played": 2, "won": 0, "drawn": 1, "lost": 1, "gf": 0, "ga": 2, "gd": -2, "points": 1},
    ],
}

DEMO_MATCHES = [
    {"home": "France",    "away": "Morocco",   "home_score": 2, "away_score": 0, "stage": "GROUP_STAGE", "date": "2026-06-12", "status": "FINISHED"},
    {"home": "Argentina", "away": "Ecuador",   "home_score": 3, "away_score": 0, "stage": "GROUP_STAGE", "date": "2026-06-12", "status": "FINISHED"},
    {"home": "Portugal",  "away": "Ghana",     "home_score": 2, "away_score": 0, "stage": "GROUP_STAGE", "date": "2026-06-12", "status": "FINISHED"},
    {"home": "Spain",     "away": "Colombia",  "home_score": 2, "away_score": 2, "stage": "GROUP_STAGE", "date": "2026-06-13", "status": "FINISHED"},
    {"home": "Austria",   "away": "France",    "home_score": 1, "away_score": 2, "stage": "GROUP_STAGE", "date": "2026-06-14", "status": "FINISHED"},
    {"home": "Saudi Arabia","away":"Argentina","home_score": 1, "away_score": 2, "stage": "GROUP_STAGE", "date": "2026-06-14", "status": "FINISHED"},
    {"home": "Uruguay",   "away": "Portugal",  "home_score": 0, "away_score": 2, "stage": "GROUP_STAGE", "date": "2026-06-14", "status": "FINISHED"},
    {"home": "Spain",     "away": "Japan",     "home_score": 1, "away_score": 0, "stage": "GROUP_STAGE", "date": "2026-06-15", "status": "FINISHED"},
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("⚽ FIFA WC 2026 Tracker — fetching data...")

    groups, matches = {}, []

    # Try football-data.org (requires API key in env)
    fdo_standings, fdo_matches = fetch_football_data_org()
    if fdo_standings:
        groups  = parse_fdo_standings(fdo_standings)
        matches = parse_fdo_matches(fdo_matches) if fdo_matches else []
        print(f"✅ football-data.org: {len(groups)} groups, {len(matches)} matches")

    # Fall back to ESPN (no auth)
    if not groups:
        espn_data, slug = fetch_espn_data()
        if espn_data:
            print(f"✅ ESPN ({slug}) — note: manual parsing may be incomplete")
            # ESPN standings format is complex and varies — extend here if needed

    # Fall back to demo data
    if not groups:
        print("⚠️  No live API available — using placeholder data")
        groups  = DEMO_GROUPS
        matches = DEMO_MATCHES

    all_teams     = [t["team"] for gt in groups.values() for t in gt]
    adj_ratings   = adjust_ratings(BASE_STRENGTH, groups)

    print(f"🎲 Running Monte Carlo simulation (8 000 iterations)…")
    probs = simulate_tournament(groups, adj_ratings)

    output = build_output(groups, matches, probs)

    os.makedirs("data", exist_ok=True)
    with open("data/tournament.json", "w") as f:
        json.dump(output, f, indent=2)

    print("✅ Saved → data/tournament.json")
    print("\n🏆 Tracked team win probabilities:")
    for team in sorted(TRACKED_TEAMS):
        p = probs["win"].get(team, 0)
        print(f"   {team:12s}  {p:5.1f}%")


if __name__ == "__main__":
    random.seed()
    main()
