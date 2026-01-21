import numpy as np
import pandas as pd

def _calc_payout(odds, stake):
    """Calculate payout for winning bet given American odds."""
    if odds > 0:
        return stake * (odds / 100)
    else:
        return stake * (100 / abs(odds))


def backtest_betting_strategy(
    df,
    lower_threshold=0.42,
    upper_threshold=0.55,
    stake=1.0,
    home_odds_col="homeMoneyline",
    away_odds_col="awayMoneyline",
):
    data = df.copy()

    # ----------------------------
    # 1. Betting rules
    # ----------------------------
    bet_mask = (
        ((data["proba"] >= upper_threshold) |
         (data["proba"] <= lower_threshold))
    )

    bets = data.loc[bet_mask].copy()

    if bets.empty or len(bets) < 10:
        return None, None

    # ----------------------------
    # 2. Side selection
    # ----------------------------
    bets["bet_side"] = np.where(
        bets["proba"] >= upper_threshold, 1, 0
    )

    bets["win"] = (bets["bet_side"] == bets["actual"]).astype(int)

    # ----------------------------
    # 3. Payouts (using actual odds per game)
    # ----------------------------
    # Select the odds based on which side we bet
    bets["bet_odds"] = np.where(
        bets["bet_side"] == 1,
        bets[home_odds_col],
        bets[away_odds_col]
    )

    # Calculate payout for each bet
    bets["win_payout"] = bets["bet_odds"].apply(lambda o: _calc_payout(o, stake))

    bets["units"] = np.where(
        bets["win"] == 1,
        bets["win_payout"],
        -stake
    )

    # ----------------------------
    # 4. Risk metrics
    # ----------------------------
    mean_return = bets["units"].mean()
    std_return = bets["units"].std(ddof=1)

    sharpe = mean_return / std_return if std_return > 0 else np.nan

    bets["cum_units"] = bets["units"].cumsum()
    drawdown = bets["cum_units"] - bets["cum_units"].cummax()

    summary = {
        "bets": len(bets),
        "win_rate": round(bets["win"].mean(), 4),
        "avg_units": round(mean_return, 4),
        "total_units": round(bets["units"].sum(), 2),
        "roi_per_bet": round(mean_return / stake, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(drawdown.min(), 2),
        "avg_proba": round(bets["proba"].mean(), 4)
    }

    return summary, bets

def optimize_threshold_range(
    df,
    lower_range=(0.0, 0.45),
    upper_range=(0.55, 1.0),
    step=0.01,
    stake=1.0,
    min_bets=10,
    optimize_metric="roi_per_bet",
    home_odds_col="homeMoneyline",
    away_odds_col="awayMoneyline",
):
    """
    Find the optimal threshold range that maximizes a given metric.

    Parameters:
    -----------
    df : DataFrame
        Test data with 'actual', 'pred', 'proba', and odds columns
    lower_range : tuple
        (min, max) values to search for lower threshold
    upper_range : tuple
        (min, max) values to search for upper threshold
    step : float
        Step size for grid search
    stake : float
        Stake per bet
    min_bets : int
        Minimum number of bets required
    optimize_metric : str
        Metric to optimize: 'roi_per_bet', 'sharpe', 'total_units', 'win_rate'
    home_odds_col : str
        Column name for home team moneyline odds
    away_odds_col : str
        Column name for away team moneyline odds

    Returns:
    --------
    dict : Best configuration and results DataFrame
    """
    results = []

    # Generate threshold combinations
    lower_thresholds = np.arange(lower_range[0], lower_range[1], step)
    upper_thresholds = np.arange(upper_range[0], upper_range[1], step)

    for lower in lower_thresholds:
        for upper in upper_thresholds:
            # Skip invalid combinations where lower >= upper
            if lower >= upper:
                continue

            summary, bets = backtest_betting_strategy(
                df,
                lower_threshold=lower,
                upper_threshold=upper,
                stake=stake,
                home_odds_col=home_odds_col,
                away_odds_col=away_odds_col,
            )

            # Skip if not enough bets
            if summary is None:
                continue

            if summary['bets'] < min_bets:
                continue

            results.append({
                'lower_threshold': round(lower, 3),
                'upper_threshold': round(upper, 3),
                **summary
            })

    if not results:
        print("No valid threshold combinations found!")
        return None

    results_df = pd.DataFrame(results)

    # Find best configuration based on optimization metric
    best_idx = results_df[optimize_metric].idxmax()
    best_config = results_df.loc[best_idx].to_dict()

    # Sort results by the optimization metric
    results_df = results_df.sort_values(by=optimize_metric, ascending=False)

    return {
        'best_config': best_config,
        'all_results': results_df
    }

def test_betting_model_performance(test_df):

    correct_preds_0 = len(test_df[(test_df['pred'] == 0) & (test_df['actual'] == 0)])
    correct_preds_1 = len(test_df[(test_df['pred'] == 1) & (test_df['actual'] == 1)])
    num_preds_0 = len(test_df[test_df['pred'] == 0])
    num_preds_1 = len(test_df[test_df['pred'] == 1])

    print(f"Class 0 Correct: {correct_preds_0}")
    print(f"Class 0 Total: {num_preds_0}")
    print(f"Class 1 Correct: {correct_preds_1}")
    print(f"Class 1 Total: {num_preds_1}")
    
    if num_preds_0 > 0:
        print(f"Class 0 Precision: {round(correct_preds_0/num_preds_0,3)*100}%")
    if num_preds_1 > 0:    
        print(f"Class 1 Precision: {round(correct_preds_1/num_preds_1,3)*100}%")