import re
from datetime import datetime

def extract_dates_and_years(text):
    if not text:
        return {"years": [], "dates": []}

    found_years = set()
    found_dates = set()

    regex_iso = r'\b(\d{4})-(\d{2})-(\d{2})\b'
    for match in re.finditer(regex_iso, text):
        found_dates.add(match.group(0))
        found_years.add(int(match.group(1)))

    regex_pl = r'\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b'
    for match in re.finditer(regex_pl, text):
        found_dates.add(match.group(0))
        found_years.add(int(match.group(3)))

    regex_year = r'\b(19|20)\d{2}\b'
    matches = re.findall(regex_year, text)
    
    for match in re.finditer(r'\b(19\d{2}|20\d{2})\b', text):
        year = int(match.group(0))
        found_years.add(year)

    regex_context = r'w (\d{4}) roku'
    for match in re.finditer(regex_context, text):
        found_years.add(int(match.group(1)))

    return {
        "years": sorted(list(found_years)),
        "dates": sorted(list(found_dates))
    }