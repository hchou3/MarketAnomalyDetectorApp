import os

from sqlalchemy import engine_from_config, pool

from alembic import context
from market_anomaly_detector.db.models import Base

config = context.config

# Priority: programmatic caller (config.attributes) > env var > default.
# Tests set config.attributes["database_url"] so they can target mad_test without
# having MAD_DATABASE_URL in the environment override them.
url = (
    config.attributes.get("database_url")
    or os.environ.get("MAD_DATABASE_URL")
    or "postgresql+psycopg://mad:mad@localhost:5432/mad"
)
config.set_main_option("sqlalchemy.url", url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
