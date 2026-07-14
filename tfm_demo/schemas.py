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


class ImpalaConfig(BaseModel):
    """Impala target as entered in the UI's Data dialog."""

    connection: str = Field("", description="CML data connection name")
    database: str = Field("", description="Impala database holding the splits")


class VastConfig(BaseModel):
    """VAST S3 target as entered in the UI's Data dialog. An empty secret_key
    on save means "keep the stored one" (the API never echoes it back)."""

    endpoint: str = Field("", description="S3 endpoint URL, e.g. https://s3.previewhub.dev")
    bucket: str = Field("", description="Bucket name")
    prefix: str = Field("", description="Folder path inside the bucket for the split objects")
    access_key: str = Field("", description="S3 access key")
    secret_key: str = Field("", description="S3 secret key ('' = keep stored)")


class DataConfig(BaseModel):
    """Storage target for the training splits (the UI's Data dialog)."""

    backend: str = Field("impala", description="'impala' or 'vast'")
    impala: ImpalaConfig = Field(default_factory=ImpalaConfig)
    vast: VastConfig = Field(default_factory=VastConfig)


class NexusConfig(BaseModel):
    """NEXUS head mode as set from the UI (Build-artifacts dialog)."""

    mode: str = Field("off", description="'off', 'stub' or 'live'")
