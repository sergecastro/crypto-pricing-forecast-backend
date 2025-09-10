from fastapi import FastAPI, Query
import os
import httpx
from starlette.responses import Response
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import sys
import tenacity
import requests

load_dotenv()

# CREATE THE APP FIRST!
app = FastAPI(title="Crypto Pricing Forecast — CLEAN v1")

# THEN ADD CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Debug CORS
@app.middleware("http")
async def debug_cors(request, call_next):
    print(f"Received request: {request.method} {request.url}", file=sys.stderr)
    response = await call_next(request)
    print(f"Response headers: {response.headers}", file=sys.stderr)
    return response

@app.on_event("startup")
async def _print_routes_on_startup():
    try:
        print("🔎 Registered routes:", [r.path for r in app.routes])
    except Exception as e:
        print("Route print error:", e)
        
        

SYMBOL_TO_CGID = {
    # core
    "btc": "bitcoin",
    "eth": "ethereum",
    "sol": "solana",
    "usdt": "tether",
    "usdc": "usd-coin",
    "ada": "cardano",
    "dai": "dai",

    # NEW coins + common aliases
    "matic": "polygon-ecosystem-token",   # MATIC
    "matic-network": "polygon-ecosystem-token",

    "avax": "avalanche-2",                # AVAX
    "avalanche-2": "avalanche-2",

    "dot": "polkadot",                    # DOT
    "polkadot": "polkadot",

    "link": "chainlink",                  # LINK
    "chainlink": "chainlink",

    "uni": "uniswap",                     # UNI
    "uniswap": "uniswap",
}




@app.get("/")
def root():
    return {"message": "Crypto Pricing API is running"}

@app.get("/__routes__")
def list_routes():
    return [r.path for r in app.routes]

@app.get("/debug/coinbase/top")
def debug_top_of_book(product_id: str = Query("ETH-USD")):
    url = f"https://api.exchange.coinbase.com/products/{product_id}/book?level=1"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    data = r.json()
    best_ask = data["asks"][0]
    best_ask_price = float(best_ask[0])
    best_ask_size = float(best_ask[1])
    return {
        "product_id": product_id,
        "best_ask_price": best_ask_price,
        "best_ask_size": best_ask_size
    }

@app.get("/debug/coinbase/levels")
def debug_coinbase_levels(product_id: str = "ETH-USD", levels: int = 5):
    url = f"https://api.exchange.coinbase.com/products/{product_id}/book?level=2"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    book = r.json()
    asks = book.get("asks", [])
    out = []
    cum_size = 0.0
    for row in asks[:max(1, levels)]:
        price = float(row[0])
        size = float(row[1])
        cum_size += size
        out.append({"price": price, "size": size, "cum_size": cum_size})
    return {"product_id": product_id, "asks": out}

TOKEN_MAPPINGS = {
    "ETH": {"symbol": "ETH", "address": "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE", "decimals": 18},
    "WETH": {"symbol": "WETH", "address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "decimals": 18},
    "USDC": {"symbol": "USDC", "address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "decimals": 6},
    "USDT": {"symbol": "USDT", "address": "0xdAC17F958D2ee523a2206206994597C13D831ec7", "decimals": 6},
    "BTC": {"symbol": "WBTC", "address": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", "decimals": 8},
    "SOL": {"symbol": "SOL", "address": "0xD31a59c85aE9D8edEFeC411D448f90841571b89c", "decimals": 9},
    "ADA": {"symbol": "DAI", "address": "0x6B175474E89094C44Da98b954EedeAC495271d0F", "decimals": 18},
    "DAI": {"symbol": "DAI", "address": "0x6B175474E89094C44Da98b954EedeAC495271d0F", "decimals": 18}
}

@app.get("/dex/paraswap_quote")
async def paraswap_quote(
    sell_token: str = "USDC",
    buy_token: str = "ETH",
    amount: float = 10000.0,
    side: str = "SELL",
    user_address: str = "0x0000000000000000000000000000000000000000"
):
    print(f"Received /dex/paraswap_quote request: sell_token={sell_token}, buy_token={buy_token}, amount={amount}", file=sys.stderr)
    
    src_map = TOKEN_MAPPINGS.get(sell_token.upper(), None)
    dest_map = TOKEN_MAPPINGS.get(buy_token.upper(), None)
    if not src_map or not dest_map:
        print(f"Unsupported token: sell={sell_token}, buy={buy_token}", file=sys.stderr)
        return {"error": "Unsupported token pair"}
    
    src_token = src_map["address"]
    dest_token = dest_map["address"]
    src_decimals = src_map["decimals"]
    dest_decimals = dest_map["decimals"]
    
    def to_wei(amount: float, decimals: int) -> str:
        return str(int(round(amount * (10 ** decimals))))
    
    url = "https://api.paraswap.io/prices"
    params = {
        "srcToken": src_token,
        "destToken": dest_token,
        "srcDecimals": src_decimals,
        "destDecimals": dest_decimals,
        "amount": to_wei(amount, src_decimals if side.upper() == "SELL" else dest_decimals),
        "side": side.upper(),
        "network": "1",
        "userAddress": user_address,
    }
    
    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(2))
    async def fetch_paraswap():
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, params=params)
            print(f"ParaSwap API call: URL={url}, Params={params}", file=sys.stderr)
            print(f"API response status: {r.status_code}, text: {r.text}", file=sys.stderr)
            if r.status_code != 200:
                raise ValueError(f"ParaSwap error: {r.text}")
            return r.json()
    
    try:
        data = await fetch_paraswap()
    except Exception as e:
        print(f"ParaSwap retry failed: {str(e)}", file=sys.stderr)
        return {"error": str(e)}
    
    price_route = data.get("priceRoute", {})
    return {
        "request": {"sell_token": sell_token, "buy_token": buy_token, "amount": amount, "side": side},
        "price": {
            "srcAmount": price_route.get("srcAmount"),
            "destAmount": price_route.get("destAmount"),
            "srcDecimals": src_decimals,
            "destDecimals": dest_decimals,
            "bestRoute": price_route.get("bestRoute", []),
            "gasCostUSD": price_route.get("gasCostUSD"),
        }
    }

@app.get("/__whereami")
def __whereami():
    return {"cwd": os.getcwd(), "file": __file__, "routes": [r.path for r in app.routes]}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/env-check")
def env_check():
    return {
        "COINGECKO_API_KEY_set": bool(os.getenv("COINGECKO_API_KEY")),
        "ETHERSCAN_API_KEY_set": bool(os.getenv("ETHERSCAN_API_KEY")),
        "ZEROX_API_KEY_set": bool(os.getenv("ZEROX_API_KEY")),
    }

@app.get("/fees/btc")
async def fees_btc():
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get("https://mempool.space/api/v1/fees/recommended")
        r.raise_for_status()
        return r.json()

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

@app.get("/price/spot")
async def get_spot_price(symbol: str = Query(None), coin: str = Query(None), fiat: str = "usd"):
    # Accept both 'symbol' and 'coin' parameters
    crypto = symbol or coin or 'ethereum'
    crypto = crypto.lower()
    
    cg_id = SYMBOL_TO_CGID.get(crypto, crypto)
    if cg_id not in SYMBOL_TO_CGID.values():
        return {"error": f"Unsupported symbol: {crypto}"}
    
    key = os.getenv("COINGECKO_API_KEY", "").strip()
    headers = {"x-cg-demo-api-key": key, "x-cg-pro-api-key": key} if key else {}
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies={fiat.lower()}"
    
    async with httpx.AsyncClient(timeout=10, headers=headers) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        price = data.get(cg_id, {}).get(fiat.lower())
        if price is None:
            return {"error": f"No price data for {crypto} in {fiat}"}
        return {"coin": cg_id, "fiat": fiat.lower(), "price": price}

@app.get("/orderbook/coinbase")
async def orderbook_coinbase(product_id: str = "ETH-USD", level: int = 2):
    url = f"https://api.exchange.coinbase.com/products/{product_id}/book?level={level}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    return {"product_id": product_id, "bids": data.get("bids", []), "asks": data.get("asks", [])}

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
        return {"error": f"no asks returned for {product_id} at level={level}"}
    remaining = amount
    cost = 0.0
    best_ask = float(asks[0][0])
    levels_used = 0
    for row in asks:
        price = float(row[0])
        size = float(row[1])
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
        "venue": "coinbase",
        "product_id": product_id,
        "amount": amount,
        "best_quote_or_ask": round(best_ask, 2),
        "avg_fill_price": round(avg_fill, 5),
        "slippage_abs": round(slippage_abs, 5),
        "slippage_bps": round(slippage_bps, 2),
        "levels_used": levels_used,
    }

@app.get("/best_price")
async def best_price(symbol: str = "eth"):
    print("Processing /best_price request for symbol:", symbol, file=sys.stderr)
    try:
        spot_response = await get_spot_price(symbol=symbol, fiat="usd")
        if "error" in spot_response:
            print("Error in spot_response:", spot_response, file=sys.stderr)
            return {"error": "Unable to fetch best price"}
        spot_price = spot_response["price"]
        print("Successfully fetched spot price:", spot_price, file=sys.stderr)
        
        dex_response = await paraswap_quote(sell_token="USDC", buy_token=symbol.upper(), amount=10000.0)
        if "error" in dex_response:
            print("Error in dex_response:", dex_response, file=sys.stderr)
            dex_price = None
        else:
            dest_amount = int(dex_response["price"]["destAmount"] or 0)
            dest_decimals = dex_response["price"]["destDecimals"]
            if dest_amount > 0:
                dex_price = 10000 / (dest_amount / (10 ** dest_decimals))
            else:
                dex_price = None
                print("Invalid destAmount in dex_response", file=sys.stderr)
        
        if dex_price:
            best = min(spot_price, dex_price)
            venue = "CoinGecko" if spot_price < dex_price else "ParaSwap"
        else:
            best = spot_price
            venue = "CoinGecko"
        
        return {
            "best_price": {
                "venue": venue,
                "price_usd": best,
                "spot_price_usd": spot_price,
                "dex_price_usd": dex_price if dex_price else "N/A"
            }
        }
    except Exception as e:
        print(f"Error in /best_price: {str(e)}", file=sys.stderr)
        return {"error": f"Server error: {str(e)}"}
    
@app.get("/history/{symbol}")
async def get_price_history(
    symbol: str,
    days: int = Query(7, description="Allowed: 1, 7, 30, 90"),
    fiat: str = "usd",
):
    # keep requests predictable and lighter
    allowed = {1, 7, 30, 90}
    if days not in allowed:
        days = 7

    cg_id = SYMBOL_TO_CGID.get(symbol.lower(), symbol.lower())
    if cg_id not in SYMBOL_TO_CGID.values():
        return {"error": f"Unsupported symbol: {symbol}"}

    key = os.getenv("COINGECKO_API_KEY", "").strip()
    params = {"vs_currency": fiat.lower(), "days": str(days)}

    base_pro = "https://pro-api.coingecko.com/api/v3"
    base_pub = "https://api.coingecko.com/api/v3"

    async def fetch_once(base_url: str, use_pro_header: bool):
        url = f"{base_url}/coins/{cg_id}/market_chart"
        # IMPORTANT: send exactly one header, not both
        headers = {}
        if key:
            headers = {"x-cg-pro-api-key": key} if use_pro_header else {"x-cg-demo-api-key": key}
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, params=params, headers=headers)
        return r, url

    try:
        # If a key exists, try PRO first with pro header
        if key:
            r, url_used = await fetch_once(base_pro, use_pro_header=True)
            if r.status_code == 400:
                txt = r.text or ""
                # Demo key must use public base
                if 'Demo API key' in txt or '"error_code":10011' in txt:
                    r, url_used = await fetch_once(base_pub, use_pro_header=False)
        else:
            # No key: use public with no headers
            r, url_used = await fetch_once(base_pub, use_pro_header=False)

        if r.status_code != 200:
            return {
                "error": "coingecko_error",
                "status": r.status_code,
                "body": r.text,
                "requested": {"url": url_used, "params": params},
            }

        data = r.json()

    except Exception as e:
        return {
            "error": "request_failed",
            "detail": str(e),
            "requested": {"url": "unknown", "params": params},
        }

    return {
        "coin": cg_id,
        "fiat": fiat.lower(),
        "prices": data.get("prices", []),  # [ [timestamp_ms, price], ... ]
    }








