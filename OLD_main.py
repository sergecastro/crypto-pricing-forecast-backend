from fastapi import FastAPI
from pydantic import BaseModel
import os, httpx
from dotenv import load_dotenv

# Load environment
load_dotenv()

print(">>> RUNNING main.py BUILD 0.1.6 <<<")


app = FastAPI(title="Crypto Pricing Forecast — DEV v0.1.4")


@app.get("/__whereami")
def __whereami():
    return {"cwd": os.getcwd(), "file": __file__, "routes": [r.path for r in app.routes]}



@app.get("/version")
def version():
    return {"app": "crypto-pricing-forecast", "build": "v0.1.3"}

@app.get("/__whereami")
def __whereami():
    return {
        "cwd": os.getcwd(),
        "file": __file__,
        "routes": [r.path for r in app.routes],
    }



# ---------- Models ----------
class RouteRequest(BaseModel):
    coin: str
    fiat: str = "USD"
    amount: float
    withdraw_chain: str | None = None
    fee_tier: str = "default"

# ---------- Health ----------
@app.get("/health")
def health():
    return {"ok": True}

# ---------- Example quote stub (unchanged) ----------
@app.post("/route/quote")
async def route_quote(req: RouteRequest):
    return {
        "inputs": req.model_dump(),
        "venues": [
            {
                "venue": "Kraken",
                "raw_price": 2431.80,
                "slippage_est": 1.70,
                "trading_fee": 1.95,
                "withdraw_fee": 4.20,
                "network_fee": 0.80,
                "all_in_total": 2440.45,
                "eta_minutes": 8,
                "liquidity": "High",
                "breakdown_links": {
                    "fees": "https://www.kraken.com/fees",
                    "gas": "https://etherscan.io/gastracker"
                }
            }
        ]
    }

# ---------- Env check ----------
@app.get("/env-check")
def env_check():
    return {
        "COINGECKO_API_KEY_set": bool(os.getenv("COINGECKO_API_KEY")),
        "ETHERSCAN_API_KEY_set": bool(os.getenv("ETHERSCAN_API_KEY")),
        "ONEINCH_API_KEY_set": bool(os.getenv("ONEINCH_API_KEY")),  # optional
        "ZEROX_API_KEY_set": bool(os.getenv("ZEROX_API_KEY")),
    }

# ---------- BTC fee (no key) ----------
@app.get("/fees/btc")
async def fees_btc():
    url = "https://mempool.space/api/v1/fees/recommended"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()

# ---------- ETH gas (Etherscan) ----------
@app.get("/fees/eth")
async def fees_eth():
    key = os.getenv("ETHERSCAN_API_KEY")
    if not key:
        return {"error": "ETHERSCAN_API_KEY is missing in .env"}
    url = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={key}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()

# ---------- CoinGecko spot price ----------
SYMBOL_TO_CGID = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "sol": "solana",
    "usdt": "tether",
    "usdc": "usd-coin",
}
@app.get("/price/spot")
async def price_spot(symbol: str = "btc", fiat: str = "usd"):
    key = os.getenv("COINGECKO_API_KEY", "").strip()
    cg_id = SYMBOL_TO_CGID.get(symbol.lower(), symbol.lower())
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies={fiat.lower()}"
    headers = {}
    if key:
        headers = {"x-cg-demo-api-key": key, "x-cg-pro-api-key": key}
    async with httpx.AsyncClient(timeout=10, headers=headers) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    return {"coin": cg_id, "fiat": fiat.lower(), "price": data.get(cg_id, {}).get(fiat.lower())}

# ---------- Binance order book (public) ----------
@app.get("/orderbook/binance")
async def orderbook_binance(symbol: str = "ETHUSDT", limit: int = 20):
    """
    symbol: e.g., ETHUSDT, BTCUSDT, SOLUSDT
    limit: 5, 10, 20, 50, 100, 500, 1000
    """
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol.upper()}&limit={limit}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    return {
        "symbol": symbol.upper(),
        "lastUpdateId": data.get("lastUpdateId"),
        "bids": data.get("bids", []),
        "asks": data.get("asks", []),
    }

# ---------- Coinbase order book (public) ----------
@app.get("/orderbook/coinbase")
async def orderbook_coinbase(product_id: str = "ETH-USD", level: int = 2):
    url = f"https://api.exchange.coinbase.com/products/{product_id}/book?level={level}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    return {"product_id": product_id, "bids": data.get("bids", []), "asks": data.get("asks", [])}

# ---------- Coinbase buy slippage sim ----------
@app.get("/simulate/coinbase_buy")
async def simulate_coinbase_buy(product_id: str = "ETH-USD", amount: float = 0.8, level: int = 2):
    if amount <= 0:
        return {"error": "amount must be > 0"}
    url = f"https://api.exchange.coinbase.com/products/{product_id}/book?level={level}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    asks = data.get("asks", [])
    if not asks:
        return {"error": "no asks returned for this product/level"}
    remaining = amount
    cost = 0.0
    best_ask = float(asks[0][0])
    levels_used = 0
    for price_str, size_str, *_ in asks:
        price = float(price_str)
        size = float(size_str)
        if size <= 0:
            continue
        take = min(remaining, size)
        cost += take * price
        remaining -= take
        levels_used += 1
        if remaining <= 1e-12:
            break
    if remaining > 0:
        return {"error": f"not enough liquidity at level={level} to fill {amount} units"}
    avg_fill = cost / amount
    slippage_abs = avg_fill - best_ask
    slippage_bps = (slippage_abs / best_ask) * 10_000
    return {
        "product_id": product_id,
        "amount": amount,
        "best_ask": best_ask,
        "avg_fill_price": round(avg_fill, 6),
        "slippage_abs": round(slippage_abs, 6),
        "slippage_bps": round(slippage_bps, 2),
        "levels_used": levels_used,
    }

# ---------- ETH network fee in USD ----------
@app.get("/fees/eth_usd")
async def fees_eth_usd(speed: str = "propose", gas_limit: int = 21000):
    key = os.getenv("ETHERSCAN_API_KEY", "").strip()
    if not key:
        return {"error": "ETHERSCAN_API_KEY is missing in .env"}
    # 1) gas (gwei)
    url = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={key}"
    async with httpx.AsyncClient(timeout=10) as client:
        rg = await client.get(url)
        rg.raise_for_status()
        g = rg.json().get("result", {})
    field = {"safe": "SafeGasPrice", "propose": "ProposeGasPrice", "fast": "FastGasPrice"}.get(speed.lower(), "ProposeGasPrice")
    gas_price_gwei = float(g.get(field, g.get("ProposeGasPrice", 0.0)))
    # 2) ETH/USD
    cg_key = os.getenv("COINGECKO_API_KEY", "").strip()
    headers = {"x-cg-demo-api-key": cg_key, "x-cg-pro-api-key": cg_key} if cg_key else {}
    cg_url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
    async with httpx.AsyncClient(timeout=10, headers=headers) as client:
        rp = await client.get(cg_url)
        rp.raise_for_status()
        eth_usd = float(rp.json().get("ethereum", {}).get("usd", 0.0))
    # 3) compute
    fee_eth = gas_price_gwei * 1e-9 * gas_limit
    fee_usd = fee_eth * eth_usd
    return {
        "speed": speed,
        "gas_limit": gas_limit,
        "gas_price_gwei": round(gas_price_gwei, 9),
        "fee_eth": round(fee_eth, 10),
        "eth_usd": eth_usd,
        "fee_usd": round(fee_usd, 6),
        "note": "Estimate for simple ETH transfer; withdrawals may vary."
    }

# ---------- Coinbase ALL-IN ----------
@app.get("/route/coinbase_allin")
async def route_coinbase_allin(
    product_id: str = "ETH-USD",
    amount: float = 0.8,
    trading_fee_bps: float = 25.0,
    withdrawal_fee_usd: float = 0.0,
    speed: str = "propose",
    gas_limit: int = 21000
):
    if amount <= 0:
        return {"error": "amount must be > 0"}
    # 1) book → avg fill
    ob_url = f"https://api.exchange.coinbase.com/products/{product_id}/book?level=2"
    async with httpx.AsyncClient(timeout=10) as client:
        ob = await client.get(ob_url)
        ob.raise_for_status()
        data = ob.json()
    asks = data.get("asks", [])
    if not asks:
        return {"error": "no asks returned for this product/level"}
    remaining = amount
    cost = 0.0
    for price_str, size_str, *_ in asks:
        price = float(price_str); size = float(size_str)
        if size <= 0: continue
        take = min(remaining, size)
        cost += take * price
        remaining -= take
        if remaining <= 1e-12: break
    if remaining > 0:
        return {"error": f"not enough liquidity to fill {amount} units at level=2"}
    avg_fill = cost / amount
    notional_usd = avg_fill * amount
    trading_fee_usd = notional_usd * (trading_fee_bps / 10_000.0)
    # 2) gas in USD
    key = os.getenv("ETHERSCAN_API_KEY", "").strip()
    url_gas = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={key}"
    async with httpx.AsyncClient(timeout=10) as client:
        rg = await client.get(url_gas)
        rg.raise_for_status()
        g = rg.json().get("result", {})
    field = {"safe": "SafeGasPrice", "propose": "ProposeGasPrice", "fast": "FastGasPrice"}.get(speed.lower(), "ProposeGasPrice")
    gas_price_gwei = float(g.get(field, g.get("ProposeGasPrice", 0.0)))
    cg_key = os.getenv("COINGECKO_API_KEY", "").strip()
    headers = {"x-cg-demo-api-key": cg_key, "x-cg-pro-api-key": cg_key} if cg_key else {}
    cg_url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
    async with httpx.AsyncClient(timeout=10, headers=headers) as client:
        rp = await client.get(cg_url)
        rp.raise_for_status()
        eth_usd = float(rp.json().get("ethereum", {}).get("usd", 0.0))
    fee_eth = gas_price_gwei * 1e-9 * gas_limit
    network_fee_usd = fee_eth * eth_usd
    all_in_total_usd = notional_usd + trading_fee_usd + withdrawal_fee_usd + network_fee_usd
    return {
        "product_id": product_id,
        "amount": amount,
        "avg_fill_price": round(avg_fill, 6),
        "notional_usd": round(notional_usd, 6),
        "trading_fee_bps": trading_fee_bps,
        "trading_fee_usd": round(trading_fee_usd, 6),
        "withdrawal_fee_usd": round(withdrawal_fee_usd, 6),
        "network_fee_usd": round(network_fee_usd, 6),
        "all_in_total_usd": round(all_in_total_usd, 6),
    }

# =========================
# DEX: 0x v2 (reconciled)
# =========================

# Token decimals (mainnet)
TOK_DECIMALS = {"ETH": 18, "WETH": 18, "USDC": 6, "USDT": 6, "DAI": 18}

# Canonical mainnet addresses (lowercase preferred), ETH must keep exact pseudo-address casing
TOKEN_ADDR = {
    "WETH": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    "USDC": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "USDT": "0xdac17f958d2ee523a2206206994597c13d831ec7",
    "DAI":  "0x6b175474e89094c44da98b954eedeac495271d0f",
    "ETH":  "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE",
}

def to_base_units(amount: float, symbol: str) -> str:
    d = TOK_DECIMALS.get(symbol.upper())
    if d is None:
        raise ValueError(f"Unknown decimals for token {symbol}")
    return str(int(round(amount * (10 ** d))))

def norm_token(t: str) -> str:
    """Return address or symbol mapping; preserve address case; preserve ETH pseudo-address case."""
    t = t.strip()
    if t.startswith("0x") or t.startswith("0X"):
        return t
    if t.upper() == "ETH":
        return TOKEN_ADDR["ETH"]
    mapped = TOKEN_ADDR.get(t.upper())
    return mapped if mapped else t.upper()

# 0x chains sanity-check
@app.get("/dex/zerox_chains")
async def zerox_chains():
    key = os.getenv("ZEROX_API_KEY", "").strip()
    if not key:
        return {"error": "ZEROX_API_KEY is missing in .env"}
    url = "https://api.0x.org/swap/chains"
    headers = {"0x-api-key": key, "0x-version": "v2", "accept": "application/json"}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, headers=headers)
        return {"status_code": r.status_code, "body": r.text[:400]}

# Primary 0x v2 indicative price
@app.get("/dex/zerox_quote")
async def zerox_quote(
    sell_token: str = "USDC",
    buy_token: str = "ETH",
    amount: float = 100.0,  # human units of sell_token
    taker: str = "0x000000000000000000000000000000000000dead",  # lower-case, avoids checksum issues
    chain_id: int = 1
):
    """
    0x v2 indicative price via /swap/price. If it fails, we try /swap/allowance-holder/price.
    """
    key = os.getenv("ZEROX_API_KEY", "").strip()
    if not key:
        return {"error": "ZEROX_API_KEY is missing in .env"}

    sell_amount = to_base_units(amount, sell_token)
    sell = norm_token(sell_token)
    buy  = norm_token(buy_token)

    attempts = [
        (
            "https://api.0x.org/swap/price",
            {"sellToken": sell, "buyToken": buy, "sellAmount": sell_amount, "taker": taker},
            {"0x-api-key": key, "0x-version": "v2", "0x-chain-id": str(chain_id), "accept": "application/json"},
            "v2_price"
        ),
        (
            "https://api.0x.org/swap/allowance-holder/price",
            {"chainId": chain_id, "sellToken": sell, "buyToken": buy, "sellAmount": sell_amount, "taker": taker},
            {"0x-api-key": key, "0x-version": "v2", "accept": "application/json"},
            "v2_allowance_holder"
        )
    ]

    last_err = None
    async with httpx.AsyncClient(timeout=20) as client:
        for url, params, headers, label in attempts:
            r = await client.get(url, params=params, headers=headers)
            if r.status_code < 400:
                data = r.json()
                fields = {k: data.get(k) for k in [
                    "price", "gasPrice", "gas", "buyToken", "sellToken",
                    "buyAmount", "sellAmount", "issues", "liquidityAvailable", "sources"
                ]}
                return {
                    "route": label,
                    "request": {
                        "sellToken": sell_token, "buyToken": buy_token,
                        "amount_sell": amount, "taker": taker, "chain_id": chain_id
                    },
                    "price": fields
                }
            last_err = {"status_code": r.status_code, "body": r.text, "tried": label}

    return {"error": "0x quote failed", "last_error": last_err}

# ============== Optional: ParaSwap fallback (no key) ==============
DECIMALS = TOK_DECIMALS
ADDR = {
    "ETH":  "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE",
    "WETH": "0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2",
    "USDC": "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "DAI":  "0x6B175474E89094C44Da98b954EedeAC495271d0F",
}
def map_addr(symbol_or_addr: str) -> str:
    t = symbol_or_addr.strip()
    if t.startswith("0x") or t.startswith("0X"):
        return t
    return ADDR.get(t.upper(), t)

@app.get("/dex/paraswap_price")
async def paraswap_price(sell_token: str = "USDC", buy_token: str = "ETH", amount: float = 100.0):
    """Indicative DEX price via ParaSwap /prices (no API key)."""
    try:
        sell_amount = to_base_units(amount, sell_token)
    except Exception as e:
        return {"error": str(e)}
    src = map_addr(sell_token)
    dest = map_addr(buy_token)
    src_dec = DECIMALS.get(sell_token.upper())
    dest_dec = DECIMALS.get(buy_token.upper())
    if src_dec is None or dest_dec is None:
        return {"error": "unsupported token symbol"}
    url = "https://apiv5.paraswap.io/prices"
    params = {
        "network": 1,
        "srcToken": src,
        "destToken": dest,
        "amount": sell_amount,
        "side": "SELL",
        "srcDecimals": src_dec,
        "destDecimals": dest_dec,
    }
    headers = {"accept": "application/json"}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params, headers=headers)
        if r.status_code >= 400:
            return {"status_code": r.status_code, "body": r.text}
        data = r.json()
    pr = data.get("priceRoute", {})
    try:
        src_amt = int(pr.get("srcAmount", "0") or "0")
        dest_amt = int(pr.get("destAmount", "0") or "0")
        unit_price = (dest_amt / (10 ** dest_dec)) / (src_amt / (10 ** src_dec)) if src_amt > 0 else None
    except Exception:
        unit_price = None
    return {
        "request": {"sellToken": sell_token, "buyToken": buy_token, "amount_sell": amount},
        "unit_price": unit_price,
        "srcAmount": pr.get("srcAmount"),
        "destAmount": pr.get("destAmount"),
        "bestRoute": pr.get("bestRoute"),
        "gasCostUSD": pr.get("gasCostUSD"),
        "others": {k: pr.get(k) for k in ("blockNumber", "tokenTransferProxy", "contractAddress")}
    }
