# SPDX-License-Identifier: Apache-2.0
"""Request/response schemas for the API layer."""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class Txn(BaseModel):
    """One transaction as posted by the UI.

    Fields use the space-separated blueprint column names as aliases (e.g.
    "Merchant Name"); `populate_by_name` lets either the alias or the Python
    attribute name be supplied.
    """

    Amount: str = Field("$120.00")
    Merchant_Name: str = Field("AMAZON", alias="Merchant Name")
    Merchant_City: str = Field("ONLINE", alias="Merchant City")
    Merchant_State: str = Field("ONLINE", alias="Merchant State")
    Use_Chip: str = Field("Online Transaction", alias="Use Chip")
    MCC: int = 5942
    Zip: str = "00000"
    Time: str = "03:14"
    Year: int = 2019
    Month: int = 11
    Day: int = 22
    Card: int = 0
    User: int = 0

    model_config = {"populate_by_name": True}

    def to_txn(self) -> Dict:
        """Plain dict keyed by the blueprint column names the engine expects."""
        return self.model_dump(by_alias=True)
