"""
CBB Pipeline Utils
Common utility functions and helpers
"""


def to_string_id(value):
    """
    Convert numeric ID to string
    
    Args:
        value: Value to convert
        
    Returns:
        str: String representation of ID
    """
    return str(int(float(value)))


def est_date_range_to_utc(start_date, end_date):
    """
    Convert EST date range to UTC ISO format
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        tuple: (start_utc, end_utc) in ISO format
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    
    est = ZoneInfo("America/New_York")
    
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=est)
    end_dt = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=est)
    
    return start_dt.isoformat(), end_dt.isoformat()
