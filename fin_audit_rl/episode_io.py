"""Чтение и запись файлов эпизода с сохранением порядка и типов данных."""

import csv
import json


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(file, rows):
    for row in rows:
        file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_episode(episode, public_dir, evaluator_dir):
    for key in ("manifest", "policy"):
        write_json(public_dir / f"{key}.json", episode[key])
    for key in ("truth", "provenance"):
        write_json(evaluator_dir / f"{key}.json", episode[key])

    csv_null = episode["manifest"]["csv_null"]
    for table, rows in episode["tables"].items():
        if table in ("invoices", "payments"):
            with (public_dir / f"{table}.jsonl").open("w", encoding="utf-8") as file:
                write_jsonl(file, rows)
        else:
            with (public_dir / f"{table}.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows({key: csv_null if value is None else value for key, value in row.items()}
                                 for row in rows)


def read_episode(directory):
    def read_json(name):
        return json.loads((directory / f"{name}.json").read_text(encoding="utf-8-sig"))

    manifest = read_json("manifest")
    csv_null = manifest.get("csv_null", "null")
    tables = {}
    for table in ("ledger", "fx"):
        with (directory / f"{table}.csv").open(encoding="utf-8-sig", newline="") as file:
            tables[table] = list(csv.DictReader(file))
    for row in tables["ledger"]:
        for key in ("debit_minor", "credit_minor"):
            row[key] = int(row[key])
        for key in ("entity_id", "invoice_id", "payment_id"):
            if row.get(key) == csv_null:
                row[key] = None
    for table in ("invoices", "payments"):
        with (directory / f"{table}.jsonl").open(encoding="utf-8-sig") as file:
            tables[table] = [json.loads(line) for line in file if line.strip()]

    return {"manifest": manifest, "policy": read_json("policy"), "tables": tables,
            "truth": read_json("truth")}
