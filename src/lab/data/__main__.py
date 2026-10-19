"""`python -m lab.data [--refresh]`: download all raw data (Delphi ILINet + PubMed abstracts)."""
import argparse
import logging

from lab.data.delphi import fetch_fluview, tidy_ili
from lab.data.pubmed import fetch_abstracts
from lab.logging_setup import setup_logging

log = logging.getLogger("lab.data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download even if cached")
    args = parser.parse_args()
    setup_logging()

    wide, coverage = tidy_ili(fetch_fluview(refresh=args.refresh))
    log.info("ILI table: %d weeks x %d locations (%s to %s)",
             len(wide), wide.shape[1], wide.index.min().date(), wide.index.max().date())
    log.info("Lowest coverage:\n%s", coverage.head(5).to_string())

    abstracts = fetch_abstracts(refresh=args.refresh)
    log.info("PubMed: %d abstracts; by year: %s", len(abstracts),
             abstracts["year"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
