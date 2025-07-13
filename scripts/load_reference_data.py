# src/scripts/load_reference_data.py

import csv
from pathlib import Path
from src.database.models import NeedCategory, LifeSegment
from src.database.database import get_session

def load_needs_data():
    """Load universal and SDT needs into database"""
    db = next(get_session())

    # Load universal needs
    with open('universal_needs.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            need = NeedCategory(
                need=row['need'],
                origin_or_core_issue=row['origin_or_core_issue'],
                solution_or_resolution=row['solution_or_resolution'],
                category_type='universal'
            )
            db.add(need)

    # Load SDT needs
    with open('sdt_needs.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            need = NeedCategory(
                need=row['need'],
                origin_or_core_issue=row['origin_or_core_issue'],
                solution_or_resolution=row['solution_or_resolution'],
                category_type='sdt'
            )
            db.add(need)

    db.commit()

def load_life_segments():
    """Load life segments into database"""
    db = next(get_session())

    with open('life_segments.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            segment = LifeSegment(
                life_area=row['life_area'],
                segment=row['segment'],
                description=row['description'],
                what_belongs_here=row['what_belongs_here']
            )
            db.add(segment)

    db.commit()
