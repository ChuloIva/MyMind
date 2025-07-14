#!/usr/bin/env python3
"""Load reference data for needs profiling system."""

import csv
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.models import NeedCategory, LifeSegment
from src.database.database import get_session
from sqlmodel import select

def load_needs_data():
    """Load universal and SDT needs into database"""
    db = next(get_session())
    data_dir = Path(__file__).parent.parent / "src" / "profiling" / "needs_assessment"
    
    # Load universal needs
    universal_path = data_dir / 'universal_needs.csv'
    print(f"Loading universal needs from {universal_path}")
    with open(universal_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Check if need already exists
            existing = db.exec(select(NeedCategory).where(NeedCategory.need == row['need'])).first()
            if not existing:
                need = NeedCategory(
                    need=row['need'],
                    origin_or_core_issue=row['origin_or_core_issue'],
                    solution_or_resolution=row['solution_or_resolution'],
                    category_type='universal'
                )
                db.add(need)

    # Load SDT needs
    sdt_path = data_dir / 'sdt_needs.csv'
    print(f"Loading SDT needs from {sdt_path}")
    with open(sdt_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Check if need already exists
            existing = db.exec(select(NeedCategory).where(NeedCategory.need == row['need'])).first()
            if not existing:
                need = NeedCategory(
                    need=row['need'],
                    origin_or_core_issue=row['origin_or_core_issue'],
                    solution_or_resolution=row['solution_or_resolution'],
                    category_type='sdt'
                )
                db.add(need)

    db.commit()
    print("✅ Needs data loaded successfully!")

def load_life_segments():
    """Load life segments into database"""
    db = next(get_session())
    data_dir = Path(__file__).parent.parent / "src" / "profiling" / "needs_assessment"
    
    segments_path = data_dir / 'life_segments.csv'
    print(f"Loading life segments from {segments_path}")
    with open(segments_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Check if segment already exists
            existing = db.exec(select(LifeSegment).where(
                LifeSegment.life_area == row['life_area'],
                LifeSegment.segment == row['segment']
            )).first()
            if not existing:
                segment = LifeSegment(
                    life_area=row['life_area'],
                    segment=row['segment'],
                    description=row['description'],
                    what_belongs_here=row['what_belongs_here']
                )
                db.add(segment)

    db.commit()
    print("✅ Life segments data loaded successfully!")

def main():
    """Load all reference data."""
    print("Loading reference data for needs profiling...")
    try:
        load_needs_data()
        load_life_segments()
        print("✅ All reference data loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load reference data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
