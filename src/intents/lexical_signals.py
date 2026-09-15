import re
from typing import Dict, Optional, Tuple


LEXICAL_SIGNALS = {
    "delivery_issue": [
        "delivery",
        "delivered",
        "deliver",
        "package",
        "packages",
        "parcel",
        "parcels",
        "arrive",
        "arrived",
        "arrival",
        "late",
        "delayed",
        "driver",
        "tracking",
        "tracked",
        "shipment",
        "shipping",
        "shipped",
        "dispatch",
        "dispatched",
        "delivery slot",
        "click collect",
        "click and collect",
        "c c",
        "pickup",
        "pick up",
        "collected",
        "shopping didn't arrive",
    ],

    "payment_issue": [
        "payment",
        "paid",
        "charged",
        "charge",
        "card",
        "billing",
        "invoice",
        "charged twice",
        "charged more",
        "charged extra",
        "double charged",
    ],

    "account_issue": [
        "account",
        "login",
        "log in",
        "sign in",
        "password",
        "locked",
        "access account",
        "can't access account",
    ],

    "vouchers_promotions": [
        "clubcard",
        "voucher",
        "discount",
        "coupon",
        "promotion",
        "offer",
        "points",
        "promo code",
        "promotional code",
        "raincheck",
        "rain check",
    ],

    "product_quality": [
        "bought",
        "purchased",
        "quality",
        "produce",
        "tastes",
        "taste",
        "smells",
        "smell",
        "hair",
        "fly",
        "maggot",
        "floating",
        "faulty",
        "wonky",
        "rusty",
        "leaked",
        "leaking",
        "black bits",
        "gone off",
        "damaged",
        "damage",
        "broken",
        "broke",
        "mould",
        "mouldy",
        "mold",
        "moldy",
        "expired",
        "out of date",
        "defective",
        "spoiled",
        "spoilt",
        "rotten",
        "contaminated",
        "foreign object",
        "use by",
        "best before",
        "date",
    ],

    "pricing_complaint": [
        "price",
        "expensive",
        "cost",
        "overpriced",
        "shelf price",
        "charged more",
        "price increase",
        "price went up",
    ],

    "technical_problem": [
        "website",
        "web site",
        "app",
        "application",
        "error",
        "crash",
        "crashed",
        "bug",
        "not working",
        "doesn't work",
        "does not work",
        "broken website",
        "broken app",
        "checkout",
        "checkout page",
        "payment page",
        "login page",
        "sign in page",
        "technical",
        "machine",
        "scanner",
        "scan as you shop",
        "site down",
        "page",
        "won't load",
        "cant load",
        "can't load",
    ],

    "customer_service_complaint": [
        "customer service",
        "staff",
        "employee",
        "member of staff",
        "rude",
        "rudely",
        "poor service",
        "bad service",
        "terrible service",
        "refused to help",
        "refused help",
        "ignored my complaint",
        "ignored me",
        "treated me badly",
        "treated me terribly",
        "treated me rudely",
        "unhelpful staff",
        "manager",
        "duty manager",
        "till",
        "tills",
        "checkout",
        "self checkout",
        "queue",
        "queues",
        "long lines",
        "training",
        "customer service training",
        "store experience",
        "refurbishment",
        "local residents",
    ],

    "stock_availability": [
        "stock",
        "in stock",
        "out of stock",
        "available",
        "availability",
        "shelf",
        "shelves",
        "stocked",
        "restock",
        "restocked",
        "discontinued",
        "do you still do",
        "do you stock",
        "why do you not stock",
        "which stores",
        "find",
        "request items",
        "local store",
        "pre order",
        "pre-order",
    ],

    "uncategorized__dm": [
        "dm",
        "dmd",
        "dm'd",
        "direct message",
        "private message",
        "sent dm",
        "dm sent",
    ],

    "uncategorized__meal": [
        "meal deal",
        "meal deals",
        "healthy living salads",
        "fruit bags",
        "crisps",
        "3 for 2",
    ],

    "uncategorized__free": [
        "gluten free",
        "dairy free",
        "free from",
        "gf",
        "wheat free",
        "lactose free",
        "allergen",
        "allergy",
    ],

    "uncategorized__tesco": [
        "opening",
        "opened",
        "diesel",
        "fuel",
        "forecourt",
        "halal",
        "labelling",
        "labeling",
        "clubcard and online tesco account",
    ],
}


COMMON_TYPOS = {
    "pakage": "package",
    "pakages": "packages",
    "parcle": "parcel",
    "parcels": "parcels",
    "arived": "arrived",
    "arive": "arrive",
    "delivry": "delivery",
    "delivred": "delivered",
    "delivary": "delivery",
    "logn": "login",
    "lgin": "login",
    "accunt": "account",
    "acount": "account",
    "pasword": "password",
    "passwrd": "password",
    "shippment": "shipment",
    "shipmnt": "shipment",
    "trackng": "tracking",
    "chargd": "charged",
    "chargedd": "charged",
    "paymnt": "payment",
    "paymnet": "payment",
    "dmd": "dm",
    "dm'd": "dm",
    "clickcollect": "click collect",
}


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()

    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [
        COMMON_TYPOS.get(word, word)
        for word in words
    ]

    return " ".join(words)


def _contains_signal(text: str, signal: str) -> bool:
    pattern = r"\b" + re.escape(signal.lower()) + r"\b"
    return re.search(pattern, text) is not None


def score_categories(text: str) -> Dict[str, float]:
    normalized = normalize_text(text)
    scores: Dict[str, float] = {}

    for category, signals in LEXICAL_SIGNALS.items():
        score = 0.0

        for signal in signals:
            if _contains_signal(normalized, signal):
                score += 1.0

        scores[category] = score

    return scores


def rule_based_category(text: str) -> Optional[str]:
    t = normalize_text(text)

    if not t:
        return None

    if re.search(
        r"\b(dm|direct message|private message)\b",
        t,
    ):
        if not re.search(
            r"\b(machine|scanner|website|site|app|fixed|working|"
            r"order online|cannot order|can't order)\b",
            t,
        ):
            return "uncategorized__dm"

    if re.search(
        r"\b(gluten free|dairy free|free from|gf|wheat free|"
        r"lactose free)\b",
        t,
    ):
        if not re.search(
            r"\b(hair|stone|gravel|bone|mould|mold|rotten|gone off|"
            r"wasp|slug|plastic|hurt|sick|cracked|broken|leaked|"
            r"use by|out of date|expired|date)\b",
            t,
        ):
            return "uncategorized__free"

    if re.search(
        r"\b(meal deal|meal deals|3 for 2|healthy living salads|"
        r"fruit bags)\b",
        t,
    ):
        if not re.search(
            r"\b(hair|stone|gravel|bone|mould|mold|rotten|gone off|"
            r"wasp|slug|plastic|hurt|sick|no duck|no bacon|tiny bits)\b",
            t,
        ):
            return "uncategorized__meal"

    if re.search(
        r"\b(clubcard|voucher|discount|coupon|promotion|promo code|"
        r"raincheck|rain check)\b",
        t,
    ):
        return "vouchers_promotions"

    if re.search(
        r"\b(card|payment|charged)\b",
        t,
    ) and re.search(
        r"\b(app|error)\b",
        t,
    ):
        return None

    if re.search(
        r"\b("
        r"hair|fly|maggot|wasp|slug|stone|gravel|bone|plastic|mystery|"
        r"mould|mouldy|mold|moldy|rotten|rotton|gone off|out of date|"
        r"use by|best before|expired|leaked|leaking|cracked|wonky|rusty|"
        r"defrosted|black bits|foreign object|poor produce|dreadful|"
        r"terrible recipe|new recipe|no duck|no bacon|tiny bits|"
        r"not oven proof|plastic melted|hurt my gum|floating|bakery|"
        r"cheese twist|danish|danishes|tastes like|dry soggy"
        r")\b",
        t,
    ):
        return "product_quality"

    if re.search(
        r"\b("
        r"website|web site|site down|error|page|won't load|cant load|"
        r"can't load|machine|scanner|touch screen|scan as you shop|"
        r"fixed|not working|keeps going down|can't order online|"
        r"cannot order online|order online|subscribe|email suggests"
        r")\b",
        t,
    ):
        return "technical_problem"

    if re.search(
        r"\b("
        r"delivery|delivered|driver|shopping did not arrive|"
        r"shopping didn't arrive|didn't arrive|did not arrive|late|"
        r"no show|cancel an order|cancelled order|wrong items|"
        r"substitute|click and collect|click collect|collected|"
        r"pick up|pickup|postage"
        r")\b",
        t,
    ):
        return "delivery_issue"

    if re.search(
        r"\b("
        r"pre order|preorder|which stores|which store|in stock|"
        r"out of stock|available at|availability|stocking|"
        r"not showing on your website|local store does not have|"
        r"local store doesn't have|do you stock|why do you not stock|"
        r"can you help us find|looking for|request items|still do"
        r")\b",
        t,
    ):
        return "stock_availability"

    if re.search(
        r"\b("
        r"staff|member of staff|manager|customer service|till|tills|"
        r"checkout|checkouts|queue|queues|long lines|baskets|"
        r"changing rooms|assistant button|local residents|"
        r"refurbishment|accused|lazy manager|store experience|"
        r"security guard|community champion|wonderful lady|"
        r"asset to tesco|smells like"
        r")\b",
        t,
    ):
        return "customer_service_complaint"

    if re.search(
        r"\b(halal|labelling|labeling|diesel|fuel|forecourt|"
        r"opening|opened|clubcard and online)\b",
        t,
    ):
        return "uncategorized__tesco"

    return None


def best_category(
    text: str,
) -> Tuple[Optional[str], float, Dict[str, float]]:
    rule_category = rule_based_category(text)
    scores = score_categories(text)

    if rule_category is not None:
        return (
            rule_category,
            max(3.0, scores.get(rule_category, 0.0)),
            scores,
        )

    if not scores:
        return None, 0.0, scores

    max_score = max(scores.values())

    if max_score <= 0:
        return None, 0.0, scores

    winners = [
        category
        for category, score in scores.items()
        if score == max_score
    ]

    if len(winners) != 1:
        return None, 0.0, scores

    return winners[0], max_score, scores


def confidence_from_score(score: float) -> float:
    if score <= 0:
        return 0.0

    if score == 1:
        return 0.5

    if score == 2:
        return 0.6667

    if score == 3:
        return 0.75

    return 0.8333


_DISAMBIGUATION_SUFFIX = re.compile(r"__\d+$")


def base_taxonomy_category(name: str) -> Optional[str]:
    if not name:
        return None

    base = _DISAMBIGUATION_SUFFIX.sub("", name)

    TAXONOMY_BASE_CATEGORIES = {
        "delivery_issue",
        "payment_issue",
        "account_issue",
        "vouchers_promotions",
        "product_quality",
        "pricing_complaint",
        "technical_problem",
        "customer_service_complaint",
        "stock_availability",
        "uncategorized__dm",
        "uncategorized__free",
        "uncategorized__meal",
        "uncategorized__tesco",
        "OVERSIZED_MIXED_CLUSTER__5",
    }

    return (
        base
        if base in TAXONOMY_BASE_CATEGORIES
        else None
    )