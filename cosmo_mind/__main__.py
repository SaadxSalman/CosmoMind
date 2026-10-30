"""Allow ``python -m cosmo_mind`` to run the CLI."""
from cosmo_mind.main import app

if __name__ == "__main__":
    raise SystemExit(app())
