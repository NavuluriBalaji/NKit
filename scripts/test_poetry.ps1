python -m pip install --upgrade pip poetry
poetry config virtualenvs.create false --local
poetry install --no-interaction --no-ansi
poetry run pytest -q
