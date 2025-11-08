# api/resources/StockDataResource.py

import json
from django import http
from tastypie.resources import ModelResource
from tastypie.authorization import Authorization
from tastypie.exceptions import ImmediateHttpResponse

from api.authentication import CustomApiKeyAuthentication
from shop.models import StockData, Study


class StockDataResource(ModelResource):
    """
    Read-friendly resource for study-level candles (OHLCV).
    Supports filtering by study and timestamp ranges.
    """

    class Meta:
        queryset = StockData.objects.all().select_related("study")
        resource_name = "stockdata"
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()
        allowed_methods = ["get"]
        always_return_data = True
        filtering = {
            "id": ["exact", "in"],
            "study": ["exact"],
            "timestamp": ["exact", "range", "gte", "lte"],
        }
        ordering = ["timestamp", "id"]

    # Ensure consistent shape (epoch ms, not strings) in responses
    def dehydrate(self, bundle):
        # timestamp
        ts = bundle.data.get("timestamp")
        if isinstance(ts, str) and ts.isdigit():
            ts = int(ts)
        bundle.data["timestamp"] = ts

        # Optionally expose study id (flat) for convenience in the UI
        study = getattr(bundle.obj, "study_id", None)
        bundle.data["study_id"] = study

        # Normalize numeric fields to native Python types (no Decimals in JSON)
        for fld in ("open", "high", "low", "close", "volume"):
            v = bundle.data.get(fld)
            if v is not None:
                try:
                    bundle.data[fld] = float(v)
                except Exception:
                    pass

        return bundle

    # Guard: disallow POST/PUT/DELETE here to keep this resource read-only API-side.
    # (If you need server-side uploads later, we can add a dedicated bulk endpoint.)
    def obj_create(self, *args, **kwargs):
        raise ImmediateHttpResponse(http.HttpResponseNotAllowed(["GET"]))

    def obj_update(self, *args, **kwargs):
        raise ImmediateHttpResponse(http.HttpResponseNotAllowed(["GET"]))

    def obj_delete(self, *args, **kwargs):
        raise ImmediateHttpResponse(http.HttpResponseNotAllowed(["GET"]))

    def obj_delete_list(self, *args, **kwargs):
        raise ImmediateHttpResponse(http.HttpResponseNotAllowed(["GET"]))
