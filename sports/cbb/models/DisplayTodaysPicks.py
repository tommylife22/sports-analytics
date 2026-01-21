import pandas as pd

def display_todays_picks(df_pred, LOWER_THRESHOLD: float, UPPER_THRESHOLD: float, engine):

    LOWER_TAIL = True
    UPPER_TAIL = True

    conditions = []

    if UPPER_TAIL:
        conditions.append(df_pred["predProb"] >= UPPER_THRESHOLD)

    if LOWER_TAIL:
        conditions.append(df_pred["predProb"] <= LOWER_THRESHOLD)

    if not conditions:
        raise ValueError("At least one of LOWER_TAIL or UPPER_TAIL must be True")

    df_picks = df_pred[pd.concat(conditions, axis=1).any(axis=1)].copy()

    if df_picks.empty:
        print("No games met tail criteria.")
        df_display = pd.DataFrame()
        
    df_picks["pickSide"] = None
    df_picks.loc[df_picks["predProb"] >= UPPER_THRESHOLD, "pickSide"] = "HOME"
    df_picks.loc[df_picks["predProb"] <= LOWER_THRESHOLD, "pickSide"] = "AWAY"

    game_ids = (
        df_picks["gameId"]
        .dropna()
        .unique()
        .tolist()
    )

    if not game_ids:
        return

    from sqlalchemy import text, bindparam

    sql = text("""
        SELECT
            gameId,
            homeTeam,
            awayTeam,
            venue
        FROM CBB.GameInfo
        WHERE startDate >= DATEADD(HOUR,-10,GETDATE())
    """)

    df_game_info = pd.read_sql(
        sql,
        con=engine,
        params={"game_ids": game_ids}
    )

    df_display = df_game_info.merge(
        df_picks,
        on="gameId",
        how="inner"
    )

    df_display["edge"] = (
        df_display["predProb"] - 0.5238
    ).where(
        df_display["pickSide"] == "HOME",
        (1 - df_display["predProb"]) - 0.5238
    )

    df_display["predictionTimestamp"] = pd.Timestamp.utcnow()

    df_display = df_display.sort_values(
        by="edge",
        ascending=False
    ).reset_index(drop=True)

    assert df_display["gameId"].is_unique
    assert df_display["pickSide"].isin(["HOME", "AWAY"]).all()

    final_df = df_display[['gameId','homeTeam','awayTeam','startDate','venue','homeSpread','predProb','predClass','pickSide','edge']].copy()

    return final_df


def display_overunders_picks(df_pred, LOWER_THRESHOLD: float, UPPER_THRESHOLD: float, engine):
    """
    Display Over/Under model picks for games meeting threshold criteria.

    Parameters:
    -----------
    df_pred : DataFrame
        Predictions dataframe with columns: gameId, predProb, predClass, overUnder, startDate
    LOWER_THRESHOLD : float
        Probability threshold for UNDER bets (e.g., 0.28 = bet UNDER if prob < 0.28)
    UPPER_THRESHOLD : float
        Probability threshold for OVER bets (e.g., 0.54 = bet OVER if prob > 0.54)
    engine : sqlalchemy.engine
        Database connection

    Returns:
    --------
    DataFrame with columns: gameId, homeTeam, awayTeam, startDate, venue,
                           overUnder, predProb, predClass, pickSide, edge
    """
    LOWER_TAIL = True  # Bet UNDER (predicting game goes under)
    UPPER_TAIL = True  # Bet OVER (predicting game goes over)

    conditions = []

    if UPPER_TAIL:
        conditions.append(df_pred["predProb"] >= UPPER_THRESHOLD)

    if LOWER_TAIL:
        conditions.append(df_pred["predProb"] <= LOWER_THRESHOLD)

    if not conditions:
        raise ValueError("At least one of LOWER_TAIL or UPPER_TAIL must be True")

    df_picks = df_pred[pd.concat(conditions, axis=1).any(axis=1)].copy()

    if df_picks.empty:
        print("No games met tail criteria for Over/Under picks.")
        return pd.DataFrame()

    # Set pick side: OVER if high probability, UNDER if low probability
    df_picks["pickSide"] = None
    df_picks.loc[df_picks["predProb"] >= UPPER_THRESHOLD, "pickSide"] = "OVER"
    df_picks.loc[df_picks["predProb"] <= LOWER_THRESHOLD, "pickSide"] = "UNDER"

    game_ids = (
        df_picks["gameId"]
        .dropna()
        .unique()
        .tolist()
    )

    if not game_ids:
        raise ValueError("No valid gameIds found after filtering.")

    from sqlalchemy import text

    sql = text("""
        SELECT
            gameId,
            homeTeam,
            awayTeam,
            venue
        FROM CBB.GameInfo
        WHERE startDate >= DATEADD(HOUR,-10,GETDATE())
    """)

    df_game_info = pd.read_sql(
        sql,
        con=engine,
        params={"game_ids": game_ids}
    )

    df_display = df_game_info.merge(
        df_picks,
        on="gameId",
        how="inner"
    )

    # Calculate edge for Over/Under
    # For OVER bets: edge = predProb - breakeven (52.38% to beat -110)
    # For UNDER bets: edge = (1 - predProb) - breakeven
    df_display["edge"] = (
        df_display["predProb"] - 0.5238
    ).where(
        df_display["pickSide"] == "OVER",
        (1 - df_display["predProb"]) - 0.5238
    )

    df_display["predictionTimestamp"] = pd.Timestamp.utcnow()

    # Sort by edge (highest edge first)
    df_display = df_display.sort_values(
        by="edge",
        ascending=False
    ).reset_index(drop=True)

    # Assertions for data quality
    assert df_display["gameId"].is_unique, "Duplicate gameIds found"
    assert df_display["pickSide"].isin(["OVER", "UNDER"]).all(), "Invalid pickSide values"

    # Select final columns
    final_df = df_display[[
        'gameId', 'homeTeam', 'awayTeam', 'startDate', 'venue',
        'overUnder', 'predProb', 'predClass', 'pickSide', 'edge'
    ]].copy()

    # Print summary
    print(f"\n{'='*80}")
    print(f"OVER/UNDER PICKS - {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*80}\n")

    for _, row in final_df.iterrows():
        print(f"{row['homeTeam']} vs {row['awayTeam']}")
        print(f"  Venue: {row['venue']}")
        print(f"  Line: {row['overUnder']}")
        print(f"  Pick: {row['pickSide']} (Confidence: {row['predProb']:.1%})")
        print(f"  Edge: {row['edge']:.2%}")
        print()

    return final_df