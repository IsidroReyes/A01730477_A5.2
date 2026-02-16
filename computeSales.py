# pylint: disable=invalid-name
"""Compute sales totals from a price catalogue and records.

This script reads two JSON files: a price catalogue and a sales record.
It computes per-sale totals and a grand total. It prints human-readable
results to stdout and writes them to ``SalesResults.txt``. The program is
resilient to malformed data: errors are reported and processing continues.

Usage:
    python computeSales.py priceCatalogue.json salesRecord.json

Note:
    The module file name is mandated by the assignment as ``computeSales.py``.
    We disable the C0103 (invalid-name) warning accordingly.

The script follows PEP 8 and is friendly to pylint/flake8.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, getcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Configure Decimal context for currency computations
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN


@dataclass(frozen=True)
class Item:
    """Represents a sale item with a product name and quantity."""

    name: str
    quantity: int


@dataclass(frozen=True)
class ComputedItem:
    """Represents a computed line item with pricing information."""

    name: str
    quantity: int
    unit_price: Decimal
    subtotal: Decimal


@dataclass(frozen=True)
class SaleResult:
    """Represents the calculation result for a single sale."""

    items: List[ComputedItem]
    total: Decimal


class SalesComputationError(Exception):
    """Domain-specific error for sales computation."""


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute sales totals given a price catalogue and a sales "
            "record. Results are printed and saved to SalesResults.txt."
        )
    )
    parser.add_argument(
        "price_catalogue",
        type=Path,
        help="Path to JSON file with product prices.",
    )
    parser.add_argument(
        "sales_record",
        type=Path,
        help="Path to JSON file with sales records.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> object:
    """Load a JSON document from ``path``."""

    try:
        with path.open("r", encoding="utf-8") as fobj:
            return json.load(fobj)
    except FileNotFoundError as exc:
        raise SalesComputationError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        msg = (
            f"Invalid JSON in file: {path} "
            f"(line {exc.lineno}, column {exc.colno})"
        )
        raise SalesComputationError(msg) from exc
    except OSError as exc:
        raise SalesComputationError(f"Unable to read file: {path}") from exc


# -------- Normalization helpers (case-insensitive keys) -------------------

def _get_value_ci(entry: dict, candidates: Iterable[str]) -> Optional[object]:
    """Return the first matching value using case-insensitive key lookup."""

    lower_map = {str(k).lower(): v for k, v in entry.items()}
    for key in candidates:
        k = key.lower()
        if k in lower_map:
            return lower_map[k]
    return None


def to_decimal(value: object, *, context: str) -> Optional[Decimal]:
    """Convert a value to ``Decimal`` safely."""

    if isinstance(value, (int, float, str)):
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            logging.error(
                "Invalid numeric value for %s: %r", context, value
            )
            return None
    logging.error(
        "Unsupported type for %s: %r", context, type(value).__name__
    )
    return None


def normalize_name(entry: dict) -> Optional[str]:
    """Extract a product name using common keys (case-insensitive)."""

    value = _get_value_ci(entry, ("name", "product", "title"))
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def normalize_price(entry: dict) -> Optional[Decimal]:
    """Extract a unit price using common keys (case-insensitive)."""

    raw = _get_value_ci(entry, ("price", "unit_price", "unitPrice"))
    if raw is None:
        return None
    return to_decimal(raw, context="price")


def normalize_quantity(entry: dict) -> Optional[int]:
    """Extract an integer quantity using common keys (case-insensitive)."""

    raw = _get_value_ci(entry, ("quantity", "qty", "count", "amount"))
    if raw is None:
        return None
    if isinstance(raw, bool):
        logging.error("Invalid boolean quantity under supported key")
        return None
    try:
        qty = int(raw)
    except (TypeError, ValueError):
        logging.error("Invalid quantity value: %r", raw)
        return None
    if qty < 0:
        logging.error("Negative quantity is not allowed: %d", qty)
        return None
    return qty


# -------- Price catalogue loader -----------------------------------------

def load_price_catalogue(data: object) -> Dict[str, Decimal]:
    """Convert JSON data into a mapping of product name to unit price.

    Accepts either a mapping ``{name: price}`` or a list of product entries.
    Compatible with product entries like:
        {"title": "Brown eggs", ..., "price": 28.1}
    """

    prices: Dict[str, Decimal] = {}

    if isinstance(data, dict):
        for name, value in data.items():
            price = to_decimal(value, context=f"price for product '{name}'")
            if price is None:
                continue
            prices[str(name).strip()] = price
        return prices

    if isinstance(data, list):
        for idx, entry in enumerate(data, start=1):
            if not isinstance(entry, dict):
                logging.error(
                    "Skipping non-dict price entry at index %d: %r",
                    idx,
                    entry,
                )
                continue
            name = normalize_name(entry)
            price = normalize_price(entry)
            if not name or price is None:
                logging.error(
                    (
                        "Skipping invalid price entry at index %d "
                        "(name=%r, price=%r)"
                    ),
                    idx,
                    name,
                    price,
                )
                continue
            prices[name] = price
        return prices

    raise SalesComputationError(
        "Unsupported price catalogue: expected object or array at root."
    )


# -------- Sales loaders (split to reduce branches/nesting) ----------------

def _item_from_record(entry: dict) -> Optional[Item]:
    """Create an ``Item`` from a flat sales record dict.

    Supports keys like Product/Quantity (any case)."""

    name = normalize_name(entry)
    qty = normalize_quantity(entry)
    if not name or qty is None:
        logging.error(
            "Skipping invalid sales record (name=%r, quantity=%r)", name, qty
        )
        return None
    return Item(name=name, quantity=qty)


def _items_from_sequence(seq: Iterable[dict]) -> List[Item]:
    """Build items from a sequence of dicts; skip invalid entries."""

    items: List[Item] = []
    for elem in seq:
        if not isinstance(elem, dict):
            logging.error("Skipping non-dict item: %r", elem)
            continue
        item = _item_from_record(elem)
        if item:
            items.append(item)
    return items


def _group_flat_records(records: List[dict]) -> List[List[Item]]:
    """Group flat sales records by SALE_ID (if present)."""

    sales_map: Dict[str, List[Item]] = {}
    sales_list: List[List[Item]] = []

    for idx, rec in enumerate(records, start=1):
        item = _item_from_record(rec)
        if item is None:
            logging.error("Skipping invalid sales record at index %d", idx)
            continue
        raw_id = _get_value_ci(
            rec, ("sale_id", "saleid", "sale id", "id", "SALE_ID")
        )
        if raw_id is None:
            sales_list.append([item])
        else:
            sid = str(raw_id)
            sales_map.setdefault(sid, []).append(item)

    sales_list.extend(sales_map.values())
    return sales_list


def _load_sales_from_dict(data: dict) -> Optional[List[List[Item]]]:
    """Try to load sales when root is a dict."""

    # Single flat record
    item = _item_from_record(data)
    if item is not None:
        return [[item]]

    # Possibly {"items": [...]}
    seq = data.get("items")
    if isinstance(seq, list):
        return [_items_from_sequence(seq)]

    return None


def _load_sales_from_list(data: List[object]) -> Optional[List[List[Item]]]:
    """Try to load sales when root is a list."""

    if all(isinstance(e, dict) for e in data):
        return _group_flat_records(data)  # flat records

    if all(isinstance(e, list) for e in data):
        return [_items_from_sequence(seq) for seq in data]

    if all(isinstance(e, dict) and "items" in e for e in data):
        sales: List[List[Item]] = []
        for sale in data:
            seq = sale.get("items")
            if not isinstance(seq, list):
                logging.error(
                    "Skipping sale due to invalid 'items' type: %r",
                    type(seq).__name__,
                )
                continue
            sales.append(_items_from_sequence(seq))
        return sales

    return None


def load_sales(data: object) -> List[List[Item]]:
    """Convert JSON data into a list of sales (each is a list of items).

    Supports *flat sales records* like:
        {"SALE_ID": 1, "SALE_Date": "01/12/23", "Product": "X",
         "Quantity": 1}

    Records are grouped by ``SALE_ID`` when present; otherwise, each record
    becomes an independent sale. Legacy shapes are also supported.
    """

    if isinstance(data, dict):
        result = _load_sales_from_dict(data)
        if result is not None:
            return result
        raise SalesComputationError(
            "Unsupported sales record object structure."
        )

    if isinstance(data, list):
        result = _load_sales_from_list(data)
        if result is not None:
            return result

    raise SalesComputationError(
        "Unsupported sales record: expected array or flat-object records."
    )


# -------- Computation and rendering ---------------------------------------

def compute_sale(items: List[Item],
                 prices: Dict[str, Decimal]) -> SaleResult:
    """Compute totals for a single sale."""

    computed: List[ComputedItem] = []
    total = Decimal("0")

    for item in items:
        if item.name not in prices:
            logging.error("Price not found for product: %s", item.name)
            continue
        unit_price = prices[item.name]
        subtotal = (
            unit_price * Decimal(item.quantity)
        ).quantize(Decimal("0.01"))
        computed.append(
            ComputedItem(
                name=item.name,
                quantity=item.quantity,
                unit_price=unit_price.quantize(Decimal("0.01")),
                subtotal=subtotal,
            )
        )
        total += subtotal

    total = total.quantize(Decimal("0.01"))
    return SaleResult(items=computed, total=total)


def format_currency(value: Decimal) -> str:
    """Format a currency amount with two decimals."""

    return f"${value:,.2f}"


def render_results(sale_results: List[SaleResult],
                   elapsed_seconds: float,
                   *,
                   include_details: bool = True) -> str:
    """Render results as a human-readable multi-line string."""

    lines: List[str] = []
    lines.append("Sales Results")
    lines.append("=" * 60)

    grand_total = Decimal("0")

    for idx, sale in enumerate(sale_results, start=1):
        lines.append(f"Sale #{idx}")
        lines.append("-" * 60)
        if include_details:
            if not sale.items:
                lines.append("  (No valid items)")
            for item in sale.items:
                line = (
                    f"  • {item.name} — qty: {item.quantity}, "
                    f"unit: {format_currency(item.unit_price)}, "
                    f"subtotal: {format_currency(item.subtotal)}"
                )
                lines.append(line)
        lines.append(f"  Total: {format_currency(sale.total)}")
        lines.append("")
        grand_total += sale.total

    lines.append("=" * 60)
    lines.append(f"Grand Total: {format_currency(grand_total)}")
    lines.append(f"Sales Count: {len(sale_results)}")
    lines.append(f"Elapsed Time: {elapsed_seconds:.3f} seconds")

    return "\n".join(lines)


def setup_logging() -> None:
    """Configure logging for console output only."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Entry point for command-line execution."""

    setup_logging()
    args = parse_args(argv)

    start = time.perf_counter()

    try:
        price_data = load_json(args.price_catalogue)
        sales_data = load_json(args.sales_record)

        prices = load_price_catalogue(price_data)
        sales = load_sales(sales_data)

        # Compute results for each sale
        results: List[SaleResult] = [
            compute_sale(sale, prices) for sale in sales
        ]

        elapsed = time.perf_counter() - start

        output = render_results(
            results,
            elapsed_seconds=elapsed,
            include_details=True,
        )

        # Print to console
        print(output)

        # Write to file as required
        out_path = Path("SalesResults.txt")
        out_path.write_text(output + "\n", encoding="utf-8")

        logging.info("Results written to %s", out_path.resolve())
        return 0

    except SalesComputationError as exc:
        logging.error("%s", exc)
        return 2
    except Exception as exc:  # pylint: disable=broad-except
        # Catch-all to report failures gracefully.
        logging.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
