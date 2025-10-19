"""
Product → integer label mapping for 4 classes:
0: Credit reporting, repair, or other
1: Debt collection
2: Consumer Loan
3: Mortgage
We match by lowercase startswith/contains to be robust to minor wording.
"""
from __future__ import annotations
from typing import Optional

RULES = [
    (0, ["credit reporting"]),   # Credit reporting, repair, or other
    (1, ["debt collection"]),    # Debt collection
    (2, ["consumer loan"]),      # Consumer Loan
    (3, ["mortgage"]),           # Mortgage
]

def product_to_label(product: str) -> Optional[int]:
    if not isinstance(product, str):
        return None
    p = product.strip().lower()
    for lbl, keys in RULES:
        for k in keys:
            if p == k or p.startswith(k) or k in p:
                return lbl
    return None  # drop rows not in these 4 classes
