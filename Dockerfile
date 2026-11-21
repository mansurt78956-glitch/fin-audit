FROM python:3.11-slim
# Исходный каталог приложения внутри контейнера.
WORKDIR /app
COPY . /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN python -m unittest discover -s fin_audit_rl/tests -v
# По умолчанию создаём эпизоды, обучающие траектории и диагностический отчёт.
CMD ["python", "-m", "fin_audit_rl.cli", "--out", "/app/artifacts"]
