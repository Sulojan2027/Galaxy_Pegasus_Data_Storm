"""End-to-end pipeline orchestrator.

Usage:
    python run_pipeline.py [--skip-poi] [--poi-limit N] [--team galaxy_pegasus]

Runs:
    Bronze ingestion → Silver cleaning → POI scraping → Gold enrichment → Modeling

POI scraping is the only slow / network-dependent stage; everything is cached
so you can run it once and skip thereafter with `--skip-poi`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src import config
from src.features.gold_enrichment import run_gold_enrichment
from src.features.poi_scraper import build_poi_features
from src.ingestion.bronze_ingestion import run_bronze_ingestion
from src.modeling.latent_potential_model import run_modeling
from src.processing.silver_cleaning import run_silver_cleaning
from src.utils.io import setup_logging

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Data Storm 7.0 — end-to-end pipeline")
    parser.add_argument("--skip-poi", action="store_true",
                        help="Skip POI scraping (use cached features if present)")
    parser.add_argument("--poi-limit", type=int, default=None,
                        help="Cap number of outlets to scrape (dev only)")
    parser.add_argument("--refresh-poi", action="store_true",
                        help="Force refresh POI cache")
    parser.add_argument("--team", type=str, default="galaxy_pegasus",
                        help="Team name (used in predictions CSV filename)")
    args = parser.parse_args()

    setup_logging()
    logger.info("=" * 60)
    logger.info("Data Storm 7.0 — pipeline start")
    logger.info("project root: %s", config.PROJECT_ROOT)
    logger.info("=" * 60)

    logger.info("[1/5] Bronze ingestion")
    run_bronze_ingestion()

    logger.info("[2/5] Silver cleaning + DQ")
    run_silver_cleaning()

    if args.skip_poi:
        cache = config.POI_CACHE_DIR / "poi_features.parquet"
        if cache.exists():
            logger.info("[3/5] POI scraping skipped — using cache %s", cache)
        else:
            logger.warning("[3/5] POI scraping skipped but no cache exists — POI features will be empty")
    else:
        logger.info("[3/5] POI scraping")
        build_poi_features(refresh=args.refresh_poi, limit=args.poi_limit)

    logger.info("[4/5] Gold enrichment")
    run_gold_enrichment()

    logger.info("[5/5] Modeling")
    result = run_modeling(team_name=args.team)

    logger.info("=" * 60)
    logger.info("Pipeline complete | predictions -> %s", result["predictions_csv"])
    logger.info("Method coverage: %s", result["coverage"])
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
