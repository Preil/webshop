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
class SessionOrder(models.Model):
    # ---- Enums (inline, like StudyOrder) ----
    SIDE_CHOICES = [
        ('BUY', 'Buy'),
        ('SELL', 'Sell'),
    ]
    ORDER_TYPE_CHOICES = [
        ('MARKET', 'Market'),
        ('LIMIT', 'Limit'),
        ('STOP', 'Stop'),
        ('STOP_LIMIT', 'Stop Limit'),
    ]
    TIME_IN_FORCE_CHOICES = [
        ('GTC', 'Good Till Cancelled'),
        ('IOC', 'Immediate Or Cancel'),
        ('FOK', 'Fill Or Kill'),
        ('DAY', 'Day'),
    ]
    STATUS_CHOICES = [
        ('PLACED', 'Placed'),            # created locally, sent to broker
        ('PARTIAL', 'Partial'),          # partially filled
        ('FILLED', 'Filled'),            # fully filled
        ('CANCELLED', 'Cancelled'),
        ('EXPIRED', 'Expired'),
        ('REJECTED', 'Rejected'),
    ]
    EXIT_ROLE_CHOICES = [
        ('TP', 'Take Profit'),
        ('SL', 'Stop Loss'),
        ('TRAIL', 'Trailing Stop'),
        ('NONE', 'None'),
    ]

    # ---- Relations ----
    session = models.ForeignKey(
        "shop.TradingSession", on_delete=models.CASCADE, related_name="orders"
    )
    sessionPotentialOrder = models.ForeignKey(
        "shop.SessionPotentialOrder",
        on_delete=models.SET_NULL,
        related_name="orders",
        null=True, blank=True,
    )
    parentOrder = models.ForeignKey(
        "self", on_delete=models.SET_NULL, related_name="childOrders", null=True, blank=True
    )

    # ---- Instrument / routing ----
    ticker = models.CharField(max_length=32)
    timeframe = models.CharField(max_length=16, null=True, blank=True)

    venue = models.CharField(max_length=32, null=True, blank=True)              # e.g., BINANCE-SPOT, IBKR-SMART
    clientOrderId = models.CharField(max_length=64)                              # idempotency key (unique per session)
    brokerOrderId = models.CharField(max_length=64, null=True, blank=True)       # set after broker ack
    ocoGroupId = models.CharField(max_length=64, null=True, blank=True)          # OCO grouping for exits
    exitRole = models.CharField(max_length=8, choices=EXIT_ROLE_CHOICES, default='NONE')

    # ---- Order intent ----
    side = models.CharField(max_length=4, choices=SIDE_CHOICES)
    orderType = models.CharField(max_length=12, choices=ORDER_TYPE_CHOICES)
    timeInForce = models.CharField(max_length=8, choices=TIME_IN_FORCE_CHOICES, default='GTC')

    quantity = models.DecimalField(max_digits=16, decimal_places=8)              # float/int → keep Decimal
    limitPrice = models.DecimalField(max_digits=16, decimal_places=8, null=True, blank=True)
    stopPrice = models.DecimalField(max_digits=16, decimal_places=8, null=True, blank=True)
    reduceOnly = models.BooleanField(default=False)
    postOnly = models.BooleanField(default=False)
    expireAt = models.DateTimeField(null=True, blank=True)

    # ---- Status & timeline ----
    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default='PLACED')
    createdAt = models.DateTimeField(default=timezone.now)
    placedAt = models.DateTimeField(null=True, blank=True)                       # after broker ack
    updatedAt = models.DateTimeField(auto_now=True)
    filledAt = models.DateTimeField(null=True, blank=True)
    cancelledAt = models.DateTimeField(null=True, blank=True)
    expiredAt = models.DateTimeField(null=True, blank=True)
    rejectReason = models.CharField(max_length=255, null=True, blank=True)

    # ---- Execution roll-ups ----
    filledQty = models.DecimalField(max_digits=16, decimal_places=8, default=Decimal("0"))
    avgFillPrice = models.DecimalField(max_digits=16, decimal_places=8, null=True, blank=True)
    fees = models.DecimalField(max_digits=16, decimal_places=8, default=Decimal("0"))
    slippage = models.DecimalField(max_digits=16, decimal_places=8, null=True, blank=True)

    # ---- Extras ----
    metadata = models.JSONField(null=True, blank=True)

    class Meta:
        db_table = "session_order"
        ordering = ("-createdAt",)
        indexes = [
            models.Index(fields=("session", "status")),
            models.Index(fields=("session", "ticker")),
            models.Index(fields=("-createdAt",)),
            models.Index(fields=("clientOrderId",)),
            models.Index(fields=("brokerOrderId",)),
        ]
        constraints = [
            # idempotency: unique per session
            models.UniqueConstraint(
                fields=["session", "clientOrderId"], name="uq_session_client_order"
            ),
        ]

    def __str__(self) -> str:
        return f"SO#{self.id} {self.side} {self.ticker} {self.orderType} [{self.status}]"
class SessionFill(models.Model):
    LIQUIDITY_CHOICES = [
        ('MAKER', 'Maker'),
        ('TAKER', 'Taker'),
    ]

    # Relations
    session = models.ForeignKey(
        "shop.TradingSession",
        on_delete=models.CASCADE,
        related_name="fills",
    )
    sessionOrder = models.ForeignKey(
        "shop.SessionOrder",
        on_delete=models.CASCADE,
        related_name="fills",
    )

    # Core trade data
    ts = models.DateTimeField(default=timezone.now)                 # fill timestamp
    qty = models.DecimalField(max_digits=16, decimal_places=8)      # executed quantity
    price = models.DecimalField(max_digits=16, decimal_places=8)    # execution price
    fee = models.DecimalField(max_digits=16, decimal_places=8, default=Decimal("0"))
    liquidityType = models.CharField(max_length=6, choices=LIQUIDITY_CHOICES, null=True, blank=True)
    brokerTradeId = models.CharField(max_length=64, null=True, blank=True)

    # Optional metadata (raw broker payload, flags, etc.)
    metadata = models.JSONField(null=True, blank=True)

    class Meta:
        db_table = "session_fill"
        ordering = ("-ts", "-id")
        indexes = [
            models.Index(fields=("session", "sessionOrder")),
            models.Index(fields=("sessionOrder", "ts")),
            models.Index(fields=("session", "ts")),
            models.Index(fields=("brokerTradeId",)),
        ]
        constraints = [
            # Prevent duplicate ingestion of the same broker trade for the same order
            models.UniqueConstraint(
                fields=["sessionOrder", "brokerTradeId"],
                name="uq_session_order_broker_trade"
            ),
        ]

    def __str__(self) -> str:
        return f"Fill#{self.id} SO#{self.sessionOrder_id} {self.qty}@{self.price}"
class SessionStockData(models.Model):
    ticker = models.CharField(max_length=10, db_index=True)
    volume = models.BigIntegerField()
    vw = models.FloatField()  # Volume Weighted Average Price
    open = models.FloatField()
    close = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    timestamp = models.BigIntegerField(db_index=True)  # UNIX timestamp in milliseconds
    transactions = models.IntegerField()
    timeframe = models.CharField(max_length=6)
    trading_session = models.ForeignKey(TradingSession, on_delete=models.CASCADE, related_name="session_stock_data")

    class Meta:
        indexes = [
            models.Index(fields=["ticker", "timestamp"]),
        ]
        ordering = ["timestamp"]
        unique_together = ("trading_session", "ticker", "timestamp")

    def __str__(self):
        return f"{self.ticker} @ {self.timestamp} (Session: {self.trading_session_id})"
class SessionStockDataIndicatorValue(models.Model):
    sessionStockDataItem = models.ForeignKey('SessionStockData', on_delete=models.CASCADE)
    studyIndicator = models.ForeignKey('StudyIndicator', on_delete=models.CASCADE)
    value = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.sessionStockDataItem.trading_session} {self.sessionStockDataItem.pk} {self.value}"
