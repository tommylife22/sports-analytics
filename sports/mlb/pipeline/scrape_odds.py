"""
MLB Odds Scraper Script
Extract MLB betting odds (moneylines, spreads, totals) from SportsbookReview
"""
import argparse
from datetime import datetime, timedelta
from src.extractors.odds_scraper import scrape_mlb_odds, scrape_mlb_odds_range


def main():
    parser = argparse.ArgumentParser(
        description='Scrape MLB betting odds from SportsbookReview',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape all bet types for a single date
  python scrape_odds.py --date 2021-04-04

  # Scrape only moneylines for a date
  python scrape_odds.py --date 2021-04-04 --bet-types moneyline

  # Scrape spreads and totals for a date range
  python scrape_odds.py --start-date 2021-04-01 --end-date 2021-04-07 --bet-types spread total

  # Save results to CSV
  python scrape_odds.py --date 2021-04-04 --output odds_2021-04-04.csv
        """
    )

    # Date arguments
    parser.add_argument('--date', type=str, help='Single date to scrape (YYYY-MM-DD)')
    parser.add_argument('--start-date', type=str, help='Start date for range (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for range (YYYY-MM-DD)')

    # Bet type arguments
    parser.add_argument(
        '--bet-types',
        nargs='+',
        choices=['moneyline', 'spread', 'total'],
        default=['moneyline', 'spread', 'total'],
        help='Bet types to scrape (default: all)'
    )

    # Output arguments
    parser.add_argument('--output', type=str, help='Output file prefix (will create separate files for each bet type)')
    parser.add_argument('--format', choices=['csv', 'json'], default='csv', help='Output format (default: csv)')

    args = parser.parse_args()

    # Validate arguments
    if args.date and (args.start_date or args.end_date):
        parser.error("Cannot use --date with --start-date or --end-date")

    if (args.start_date and not args.end_date) or (args.end_date and not args.start_date):
        parser.error("Must specify both --start-date and --end-date for range scraping")

    if not args.date and not args.start_date:
        parser.error("Must specify either --date or --start-date/--end-date")

    # Scrape odds
    if args.date:
        # Single date
        results = scrape_mlb_odds(args.date, args.bet_types)
        date_label = args.date
    else:
        # Date range
        results = scrape_mlb_odds_range(args.start_date, args.end_date, args.bet_types)
        date_label = f"{args.start_date}_to_{args.end_date}"

    # Display results
    print("\n" + "="*60)
    print("PREVIEW OF RESULTS")
    print("="*60)

    for bet_type, df in results.items():
        if len(df) > 0:
            print(f"\n{bet_type.upper()} (showing first 5 rows):")
            print(df.head())
        else:
            print(f"\n{bet_type.upper()}: No data found")

    # Save results
    if args.output:
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)

        for bet_type, df in results.items():
            if len(df) > 0:
                if args.format == 'csv':
                    filename = f"{args.output}_{bet_type}.csv"
                    df.to_csv(filename, index=False)
                    print(f"✓ Saved {bet_type}: {filename}")
                elif args.format == 'json':
                    filename = f"{args.output}_{bet_type}.json"
                    df.to_json(filename, orient='records', indent=2)
                    print(f"✓ Saved {bet_type}: {filename}")

        print("="*60)

    # Statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)

    for bet_type, df in results.items():
        if len(df) > 0:
            print(f"\n{bet_type.upper()}:")
            print(f"  Total records:      {len(df)}")
            print(f"  Unique games:       {df['game_id'].nunique() if 'game_id' in df.columns else 'N/A'}")
            print(f"  Unique sportsbooks: {df['sportsbook'].nunique() if 'sportsbook' in df.columns else 'N/A'}")

            if 'sportsbook' in df.columns:
                print(f"  Sportsbooks: {', '.join(df['sportsbook'].unique())}")

    print("="*60)

    print("\n✓ Scraping complete!")


if __name__ == "__main__":
    main()
