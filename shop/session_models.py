from decimal import Decimal
from django.conf import settings
from django.db import models
from django.utils import timezone


class TradingSession(models.Model):
    # Human-readable external identifier, auto-generated on first save
    session_id = models.CharField(max_length=32, unique=True, editable=False, blank=True)

    # Enums (inline, like StudyOrder)
    TYPE_CHOICES = [
        ('BACKTEST', 'Backtest'),
        ('PAPER', 'Paper'),
        ('LIVE', 'Live'),
    ]

    STATE_CHOICES = [
        ('DRAFT', 'Draft'),
        ('READY', 'Ready'),
        ('RUNNING', 'Running'),
        ('PAUSED', 'Paused'),
        ('COMPLETED', 'Completed'),
        ('ABORTED', 'Aborted'),
    ]

    STATUS_CHOICES = [
        ('OK', 'OK'),
        ('WARNING', 'Warning'),
        ('ERROR', 'Error'),
    ]

    # Core
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True, default="")

    # Relations (string refs to avoid circular imports)
    study = models.ForeignKey(
        "shop.Study",
        null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name="trading_sessions",
    )
    trained_model = models.ForeignKey(
        "shop.TrainedNnModel",
        null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name="trading_sessions",
    )

    broker_id = models.CharField(max_length=64, null=True, blank=True)
    modelInputSchemaHash = models.CharField(max_length=128, null=True, blank=True)

    # Time window
    sessionStart = models.DateTimeField(null=True, blank=True)
    sessionEnd   = models.DateTimeField(null=True, blank=True)

    # Enums usage
    type = models.CharField(max_length=16, choices=TYPE_CHOICES, default='BACKTEST')
    state = models.CharField(max_length=16, choices=STATE_CHOICES, default='DRAFT')
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default='OK')

    # JSON “lists” (can normalize later)
    sessionStockData        = models.JSONField(default=dict, blank=True, null=True)
    sessionIndicatorsValues = models.JSONField(default=dict, blank=True, null=True)
    sessionOrders           = models.JSONField(default=dict, blank=True, null=True)
    sessionTransactions     = models.JSONField(default=dict, blank=True, null=True)

    # Settings placeholder (FK later if/when you add the model)
    sessionSettings_id = models.CharField(max_length=64, null=True, blank=True)

    # Audit
    createdAt = models.DateTimeField(default=timezone.now)
    updatedAt = models.DateTimeField(auto_now=True)
    createdBy = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name="created_trading_sessions",
    )

    # Balance monitor
    InitialBalance   = models.DecimalField(max_digits=20, decimal_places=6, default=Decimal("0"))
    EquityNow        = models.DecimalField(max_digits=20, decimal_places=6, default=Decimal("0"))
    CashNow          = models.DecimalField(max_digits=20, decimal_places=6, default=Decimal("0"))
    BuyingPowerNow   = models.DecimalField(max_digits=20, decimal_places=6, default=Decimal("0"))
    UnrealizedPnlNow = models.DecimalField(max_digits=20, decimal_places=6, default=Decimal("0"))
    RealizedPnlTotal = models.DecimalField(max_digits=20, decimal_places=6, default=Decimal("0"))
    MarginUsedNow    = models.DecimalField(max_digits=20, decimal_places=6, default=Decimal("0"))
    Currency         = models.CharField(max_length=12, default="USD")

    class Meta:
        db_table = "trading_session"
        ordering = ("-createdAt",)
        indexes = [
            models.Index(fields=("type", "state")),
            models.Index(fields=("status",)),
        ]

    def __str__(self) -> str:
        return f"{self.name} [{self.type}]"

    def save(self, *args, **kwargs):
        # Auto-generate readable session_id like TS-2025-00001 on first save
        if not self.session_id:
            prefix = "TS"
            year = timezone.now().year
            count_for_year = TradingSession.objects.filter(createdAt__year=year).count() + 1
            self.session_id = f"{prefix}-{year}-{count_for_year:05d}"
        super().save(*args, **kwargs)
# --- add this in shop/session_models.py ---

from django.db import models
from decimal import Decimal
from django.utils import timezone

# ... keep TradingSession above ...

class SessionPotentialOrder(models.Model):
    # Enums (inline, like StudyOrder)
    DIRECTION_CHOICES = [
        ('BUY', 'Buy'),
        ('SELL', 'Sell'),
    ]
    DECISION_CHOICES = [
        ('NONE', 'None'),
        ('APPROVED', 'Approved'),
        ('REJECTED', 'Rejected'),
        ('QUEUED', 'Queued'),
    ]

    # Relations
    session = models.ForeignKey(
        "shop.TradingSession",
        on_delete=models.CASCADE,
        related_name="potentialOrders",
    )
    sessionStockDataItem = models.ForeignKey(
        "shop.StockData",  # if you later introduce SessionCandle, switch FK target here
        on_delete=models.CASCADE,
        related_name="sessionPotentialOrders",
        null=True, blank=True,
    )
    trainedModel = models.ForeignKey(
        "shop.TrainedNnModel",
        on_delete=models.SET_NULL,
        related_name="sessionPotentialOrders",
        null=True, blank=True,
    )

    # Intent / parameters
    direction = models.CharField(max_length=4, choices=DIRECTION_CHOICES)
    limitPrice = models.DecimalField(max_digits=16, decimal_places=8, null=True, blank=True)
    takeProfitPrice = models.DecimalField(max_digits=16, decimal_places=8, null=True, blank=True)
    stopPrice = models.DecimalField(max_digits=16, decimal_places=8, null=True, blank=True)

    # TradingPlan-driven fields
    lpOffset = models.DecimalField(max_digits=16, decimal_places=8, null=True, blank=True)
    slATR = models.DecimalField(max_digits=16, decimal_places=8, null=True, blank=True)
    tp = models.IntegerField(null=True, blank=True)  # e.g., 3/4/5 × SL

    # Model output / decision
    prediction = models.JSONField(null=True, blank=True)  # e.g., {"prob":0.78,"dir":"BUY","meta":{...}}
    decision = models.CharField(max_length=9, choices=DECISION_CHOICES, default='NONE')  # noqa: keep 'NONE' as default

    # Audit
    createdAt = models.DateTimeField(default=timezone.now)
    updatedAt = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "session_potential_order"
        ordering = ("-createdAt",)
        indexes = [
            models.Index(fields=("session", "direction")),
            models.Index(fields=("session", "decision")),
            models.Index(fields=("createdAt",)),
        ]

    def __str__(self) -> str:
        return f"SPO for session {self.session_id} [{self.direction}]"
