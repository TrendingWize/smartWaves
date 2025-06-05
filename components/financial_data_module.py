import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Union
from datetime import datetime as dt_class, timedelta
from datetime import date
from openai import OpenAI
import requests
import os

# Config dataclass (if you want to keep config inside the module)
from dataclasses import dataclass, field

current_period_back =0
DEFAULT_MAX_WORKERS = os.cpu_count() or 4
logger = logging.getLogger("financial_data_module")

def _redact(url: str, key: str) -> str:
    """Redacts API key or sensitive value from a URL for safe logging."""
    if not key:
        return url
    return url.replace(key, "***REDACTED***")

@dataclass(slots=True, frozen=True)
class Config:
    fmp_key: str
    openai_key: str = ""
    openai_model: str = "gpt-4o" 
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""
    user_agent: str = "TrendingWizeAnalysis/2.0"
    price_look_forward: int = 7
    retry_attempts: int = 3
    retry_delay: int = 5
    retry_backoff: float = 1.5

def retry(max_attempts: int = 3, delay: int = 2, backoff: float = 1.4):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_attempts:
                        raise
                    pause = delay * (backoff ** (attempt - 1)) + random.random()
                    time.sleep(pause)
            return None
        return wrapper
    return decorator


class FinancialDataModule:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.cfg = config or Config()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.cfg.user_agent})
        self._openai: Optional[OpenAI] = None
        if self.cfg.openai_key and OpenAI is not None:
            self._openai = OpenAI(api_key=self.cfg.openai_key)
            logger.debug("OpenAI client initialised")
        elif self.cfg.openai_key:
            logger.warning("OPENAI_API_KEY set but `openai` package not installed")

    @retry()
    def _get(self, url: str) -> Any | None:
        redacted_url = _redact(url, self.cfg.fmp_key)
        logger.debug("GET %s", redacted_url)
        resp = self.session.get(url, timeout=20)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            logger.error("HTTPError for %s: %s, Response: %s", redacted_url, e, resp.text)
            if 400 <= resp.status_code < 500 and resp.status_code != 429 : # Do not retry client errors other than 429
                 raise # Propagate immediately
            raise # Let retry handler deal with 429 or 5xx

        if not resp.content:
            logger.warning("Empty response from %s", redacted_url)
            return None
        try:
            return resp.json()
        except json.JSONDecodeError:
            logger.error("Non‑JSON body returned from %s: %s", redacted_url, resp.text[:500])
            return None

    @staticmethod
    def _pct_change(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
        if curr is None or prev is None or prev == 0:
            return None
        return curr / prev - 1

    @staticmethod
    def _cagr(first: Optional[float], last: Optional[float], years: int) -> Optional[float]:
        if first is None or first == 0 or last is None or years <= 0:
            return None
        # Handle potential negative numbers if they are not meaningful for CAGR
        if first < 0 and last > 0 or first > 0 and last < 0: # Sign change, CAGR not meaningful
             return None
        if first < 0 and last < 0: # Both negative, take absolute for calculation logic
            return ((abs(last) / abs(first)) ** (1 / years) - 1) * (-1 if last < first else 1)
        return (last / first) ** (1 / years) - 1


    @staticmethod
    def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
        if num is None or den is None or den == 0:
            return None
        try:
            return num / den
        except ZeroDivisionError: # Should be caught by den == 0
            return None

    @retry()
    def get_financial_statements(
        self,
        symbol: str,
        statement: str,
        period_param: str, # Renamed to avoid conflict with global 'period'
        limit: int,
    ) -> Optional[List[Dict[str, Any]]]:
        url = (
            f"https://financialmodelingprep.com/api/v3/{statement}/{symbol}"
            f"?period={period_param}&limit={limit}&apikey={self.cfg.fmp_key}"
        )
        return self._get(url)

    @retry()
    def get_first_historical_close_price(
        self,
        symbol: str,
        target_date_str: str,
        look_forward_days: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if not target_date_str:
            return None
        look_forward_days = (
            look_forward_days if look_forward_days is not None else self.cfg.price_look_forward
        )
        # Use aliased dt_class for strptime
        start = dt_class.strptime(target_date_str[:10], "%Y-%m-%d").date()
        end = start + timedelta(days=look_forward_days)
        url = (
            "https://financialmodelingprep.com/api/v3/historical-price-full/"
            f"{symbol}?from={start}&to={end}&apikey={self.cfg.fmp_key}"
        )
        data = self._get(url)
        if not data or "historical" not in data or not data["historical"]: # Check if historical is empty
            return None
        # FMP historical data is typically sorted ascending by date.
        # We want the first available price ON or AFTER the target_date.
        # So, we iterate normally (oldest to newest in response) and take the first one.
        # If the API returns descending, then reversed() would be correct.
        # Based on typical API behavior, non-reversed is usually what's needed for "first after date".
        # Let's assume the API returns ascending.
        for rec in data["historical"]: # Iterate from earliest to latest in the response
            if rec.get("close") is not None:
                return {"date": rec["date"], "close": rec["close"]}
        return None


    @retry()
    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={self.cfg.fmp_key}"
        data = self._get(url)
        if not data or not isinstance(data, list) or len(data) == 0:
            return None
        row = data[0]
        ts = row.get("timestamp")
        # Use aliased dt_class for fromtimestamp
        date_str = dt_class.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else "N/A"
        return {"date": date_str, "close": row.get("price")}

    @lru_cache(maxsize=128)
    def get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={self.cfg.fmp_key}"
        data = self._get(url)
        if not data or not isinstance(data, list) or len(data) == 0:
            return None
        r = data[0]
        return {
            "company_name": r.get("companyName"),
            "exchange": r.get("exchangeShortName"),
            "industry": r.get("industry"),
            "sector": r.get("sector"),
            "beta": r.get("beta"),
            "currency": r.get("currency"),
            # Add other fields if needed by OpenAI prompt that are available in profile
            "description": r.get("description"),
            "ipoDate": r.get("ipoDate"),
            "website": r.get("website"),
            "ceo": r.get("ceo"),
            "fullTimeEmployees": r.get("fullTimeEmployees"),
            "image": r.get("image"),
            "marketCap": r.get("mktCap"), # Note: mktCap is often more current than in income statement
        }

    def process_batch(self, symbol: str) -> Optional[Dict[str, List[str]]]:
        if not self._openai:
            logger.info("OpenAI disabled – skipping competitor discovery")
            return {symbol: []}
        prompt = (
            "List five competitor tickers ONLY in JSON:\n"
            f'{{ "{symbol}": ["TICK1", "TICK2", "TICK3", "TICK4", "TICK5"] }}'
        )
        try:
            chat = self._openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a US equity research assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = chat.choices[0].message.content
            if content:
                 return json.loads(content)
            return {symbol: []} # Should not happen with json_object
        except (OpenAIError, json.JSONDecodeError, AttributeError) as exc:
            logger.warning("OpenAI peer lookup failed for %s: %s", symbol, exc)
            return {symbol: []}

    def get_peer_metrics(self, symbol: str, peers: List[str], current_period_back: int = 0) -> List[Dict[str, Any]]:
        tickers = [symbol, *peers]
        out: List[Dict[str, Any]] = []
        # Ensure DEFAULT_MAX_WORKERS is at least 1
        max_workers_for_pool = max(1, DEFAULT_MAX_WORKERS)

        with ThreadPoolExecutor(max_workers=max_workers_for_pool) as pool:
            fut_quotes = {
                t: pool.submit(
                    self._get, f"https://financialmodelingprep.com/api/v3/quote/{t}?apikey={self.cfg.fmp_key}"
                )
                for t in tickers
            }
            fut_income = {
                t: pool.submit(
                    self.get_financial_statements, t, "income-statement", "annual", 4
                )
                for t in tickers
            }
            fut_bs = {
                t: pool.submit(
                    self.get_financial_statements, t, "balance-sheet-statement", "annual", 4
                )
                for t in tickers
            }

        for tkr in tickers:
            try:
                quote_data = fut_quotes[tkr].result()
                income_data = fut_income[tkr].result()
                if not income_data or not isinstance(income_data, list) or len(income_data) <= current_period_back:
                     raise RuntimeError(
                    f"No/Insufficient income statements for {symbol_param} (need {current_period_back+1}, got {len(income_data) if income_data else 0})"
                    )
                target_income_statement = income_data[current_period_back]


                bs_data = fut_bs[tkr].result() # Changed variable name
                
                row = {}
                if quote_data and isinstance(quote_data, list) and len(quote_data) > 0:
                    row = quote_data[0]

                def _nth(field: str, n: int, default=None):
                    try:
                        return income_data[n][field] if income_data and len(income_data) > n else default
                    except (IndexError, KeyError, TypeError):
                        return default

                def _bs(field: str, n: int = 0, default=None): # Added n for bs_data
                    try:
                        return bs_data[n][field] if bs_data and len(bs_data) > n else default
                    except (IndexError, KeyError, TypeError):
                        return default

                rev_0, rev_1, rev_3 = _nth("revenue", 0), _nth("revenue", 1), _nth("revenue", 3)
                ebitda_0, ebitda_1 = _nth("ebitda", 0), _nth("ebitda", 1)
                eps_0, eps_1 = _nth("epsdiluted", 0), _nth("epsdiluted", 1)

                eps_growth_yoy = self._pct_change(eps_0, eps_1)
                peg_ratio = (
                    self._safe_div(row.get("pe"), (eps_growth_yoy * 100))
                    if row.get("pe") is not None and eps_growth_yoy and eps_growth_yoy > 0
                    else None
                )
                
                shareholder_equity_0 = _bs("totalStockholdersEquity", 0)
                total_assets_0 = _bs("totalAssets", 0)
                cash_0 = _bs("cashAndCashEquivalents", 0, 0.0) or 0.0
                total_debt_0 = _bs("totalDebt", 0) or ((_bs("longTermDebt", 0) or 0) + (_bs("shortTermDebt", 0) or 0))

                gross_margin = self._safe_div(_nth("grossProfit", 0), rev_0)
                ebitda_margin = self._safe_div(ebitda_0, rev_0)
                net_income_0 = _nth("netIncome", 0)
                net_margin = self._safe_div(net_income_0, rev_0)

                roe = self._safe_div(net_income_0, shareholder_equity_0)

                ebit_0 = _nth("operatingIncome", 0) # operatingIncome is typically EBIT
                if ebit_0 is None: ebit_0 = ebitda_0 # Fallback to EBITDA if op income is missing
                
                income_before_tax_0 = _nth("incomeBeforeTax", 0)
                income_tax_exp_0 = _nth("incomeTaxExpense", 0)
                tax_rate = (
                    self._safe_div(income_tax_exp_0, income_before_tax_0) if income_before_tax_0 else None
                )
                
                nopat = (
                    ebit_0 * (1 - tax_rate)
                    if (ebit_0 is not None and tax_rate is not None)
                    else None
                )
                invested_capital = (
                    (total_debt_0 or 0) + (shareholder_equity_0 or 0) - cash_0
                    if shareholder_equity_0 is not None
                    else None
                )
                roic = self._safe_div(nopat, invested_capital)

                asset_turnover = self._safe_div(rev_0, total_assets_0)
                debt_to_ebitda = self._safe_div(total_debt_0, ebitda_0)
                interest_exp_0 = _nth("interestExpense", 0)
                interest_coverage = self._safe_div(
                    ebit_0, abs(interest_exp_0) if interest_exp_0 else None # Use abs for interest expense
                )
                revenue_cagr_3y = self._cagr(rev_3, rev_0, 3)

                out.append({
                    "symbol": tkr, "price": row.get("price"), "pe": row.get("pe"),
                    "revenue_growth_yoy": self._pct_change(rev_0, rev_1),
                    "revenue_cagr_3y": revenue_cagr_3y,
                    "ebitda_growth_yoy": self._pct_change(ebitda_0, ebitda_1),
                    "eps_diluted_growth_yoy": eps_growth_yoy,
                    "peg_ratio": peg_ratio, "gross_margin": gross_margin,
                    "ebitda_margin": ebitda_margin, "net_margin": net_margin,
                    "roe": roe, "roic": roic, "asset_turnover": asset_turnover,
                    "debt_to_ebitda": debt_to_ebitda, "interest_coverage": interest_coverage,
                })
            except Exception as e:
                logger.error(f"Error processing peer metrics for {tkr}: {e}", exc_info=True)
                out.append({"symbol": tkr, "error": str(e)}) # Add placeholder for errored ticker
        return out

    @retry()
    def get_revenue_product_segmentation(
        self, symbol: str, period_param: str # Renamed
    ) -> Optional[List[Dict[str, Any]]]:
        url = (
            "https://financialmodelingprep.com/api/v4/revenue-product-segmentation"
            f"?symbol={symbol}&period={period_param}&apikey={self.cfg.fmp_key}"
        )
        return self._get(url)

    @retry()
    def get_revenue_geographic_segmentation(
        self, symbol: str, period_param: str # Renamed
    ) -> Optional[List[Dict[str, Any]]]:
        url = (
            "https://financialmodelingprep.com/api/v4/revenue-geographic-segmentation"
            f"?symbol={symbol}&period={period_param}&apikey={self.cfg.fmp_key}"
        )
        return self._get(url)

    @retry()
    def get_outstanding_shares(
        self, symbol: str
    ) -> Optional[List[Dict[str, Any]]]:
        # This endpoint seems to be deprecated or changed.
        # Using /api/v3/historical-price-full/{symbol}/shares for historical shares
        # For latest, it's in company profile or financial statements.
        # The prompt uses income_data[period_back].get("weightedAverageShsOut")
        # This function might not be strictly needed if that's sufficient.
        # Let's try to find a V4 equivalent or adjust.
        # FMP docs suggest /v3/enterprise-values/{symbol}?limit=1 for latest sharesOutstanding
        # Or stick to weightedAverageShsOut from income statement as primary.
        # For historical trend, if needed:
        url = (
            f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}/shares"
            f"?apikey={self.cfg.fmp_key}"
        )
        data = self._get(url)
        if data and "historicalStockSplits" in data: # This is not shares, it's splits
            # This endpoint is incorrect for shares.
            # Let's return an empty list and rely on income statement for shares.
            logger.warning(f"get_outstanding_shares endpoint for {symbol} might be incorrect, relying on income statement.")
            return []
        # A different endpoint might be needed for historical outstanding shares trend
        # /api/v4/shares_float?symbol=AAPL
        # This gives shares float, not total outstanding directly
        return [] # Placeholder, as the original endpoint might be problematic.

    @retry()
    def get_treasury_rate(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        # Use aliased dt_class for utcnow
        today = dt_class.utcnow().date()
        if end_date is None:
            end_date = today.strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        url = (
            "https://financialmodelingprep.com/api/v4/treasury"
            f"?from={start_date}&to={end_date}&apikey={self.cfg.fmp_key}"
        )
        data = self._get(url)
        if not data or not isinstance(data, list) or len(data) == 0:
            return None
        
        latest = None
        latest_dt = dt_class.min.date() # Initialize with a very old date

        for row in data:
            if "year10" in row and row.get("date"):
                try:
                    current_dt = dt_class.strptime(row["date"], "%Y-%m-%d").date()
                    if current_dt >= latest_dt: # Use >= to prefer more recent if dates are same
                        latest_dt = current_dt
                        latest = row
                except ValueError:
                    continue # Skip malformed dates
        
        if not latest:
            return None
        return {"date": latest["date"], "year10": latest["year10"]}

    @retry()
    def _get_sector_ratio_data( # Renamed from get_sector_pe_ratio
        self,
        exchange: str,
        sector_name: str, # Renamed from sector to avoid conflict
        anchor_date: Optional[str | date] = None
    ) -> Optional[List[Dict[str, Any]]]:
        target_date: Optional[dt_class.date] = None
        if anchor_date is None:
            target_date = dt_class.utcnow().date()
        elif isinstance(anchor_date, str):
            try:
                target_date = dt_class.strptime(anchor_date[:10], "%Y-%m-%d").date()
            except ValueError:
                logger.error("Invalid anchor_date string format for sector P/E: %s", anchor_date)
                return None
        elif isinstance(anchor_date, dt_class.date): # Use aliased dt_class
            target_date = anchor_date
        else:
            logger.error("Invalid type for anchor_date for sector P/E: %s (%s)", type(anchor_date), anchor_date)
            return None

        if not target_date: return None # Should not happen with above logic
        date_str = target_date.strftime("%Y-%m-%d")

        if not sector_name:
            logger.warning("Sector name is None or empty for sector P/E request.")
            return None
        if not exchange:
            logger.warning("Exchange is None or empty for sector P/E request.")
            return None
        
        # FMP endpoint for historical sector P/E seems to be /api/v4/sector_pe?date=YYYY-MM-DD
        # The URL in original code might be outdated or for a different version.
        # Let's try the documented v4 one. It might require iterating if no data for exact date.
        # For now, stick to the provided URL structure and adapt if it fails.
        # The URL /stable/historical-sector-pe seems like a bulk data endpoint.
        # Let's assume it works as in the original code.
        url = (
            f"https://financialmodelingprep.com/stable/historical-sector-pe" # Changed to v3 as /stable/ might be internal
            f"?sector={requests.utils.quote(sector_name)}" # URL Encode sector name
            f"&date={date_str}" # FMP often uses 'date' for single day, or from/to for range
            # f"&from={date_str}&to={date_str}" # Using from/to as in original
            f"&exchange={exchange}"
            f"&apikey={self.cfg.fmp_key}"
        )
        # If the above URL doesn't work, try with from/to
        # url = (
        #     f"https://financialmodelingprep.com/api/v3/historical-sector-pe"
        #     f"?sector={requests.utils.quote(sector_name)}"
        #     f"&from={date_str}&to={date_str}" 
        #     f"&exchange={exchange}&apikey={self.cfg.fmp_key}"
        # )
        logger.info(f"Fetching Sector P/E URL: {_redact(url, self.cfg.fmp_key)}")
        data = self._get(url)
        if data and isinstance(data, list) and len(data) > 0:
            logger.info("Got Sector P/E for Sector=%s Exchange=%s Date=%s", sector_name, exchange, date_str)
            return data
        else:
            logger.warning("No Sector P/E data found for Sector=%s Exchange=%s Date=%s. Data: %s",
                        sector_name, exchange, date_str, str(data)[:200])
            return None

    @retry()
    def get_industry_pe_ratio( # Renamed from industry_pe_ratio
        self,
        exchange: str,
        industry_name: str, # Renamed from industry
        anchor_date: Optional[str | date] = None
    ) -> Optional[List[Dict[str, Any]]]:
        target_date: Optional[dt_class.date] = None # Use aliased dt_class
        if anchor_date is None:
            target_date = dt_class.utcnow().date() # Use aliased dt_class
        elif isinstance(anchor_date, str):
            try:
                target_date = dt_class.strptime(anchor_date[:10], "%Y-%m-%d").date() # Use aliased dt_class
            except ValueError:
                logger.error("Invalid anchor_date string format for industry P/E: %s", anchor_date)
                return None
        elif isinstance(anchor_date, dt_class.date): # Use aliased dt_class
            target_date = anchor_date
        else:
            logger.error("Invalid type for anchor_date for industry P/E: %s (%s)", type(anchor_date), anchor_date)
            return None
        
        if not target_date: return None
        date_str = target_date.strftime("%Y-%m-%d")

        if not industry_name:
            logger.warning("Industry name is None or empty for industry P/E request.")
            return None
        if not exchange:
            logger.warning("Exchange is None or empty for industry P/E request.")
            return None
        
        # Similar to sector, using v3 and date parameter, or from/to
        url = (
            f"https://financialmodelingprep.com/stable/historical-industry-pe"
            f"?industry={requests.utils.quote(industry_name)}" # URL Encode industry name
            f"&date={date_str}"
            # f"&from={date_str}&to={date_str}"
            f"&exchange={exchange}"
            f"&apikey={self.cfg.fmp_key}"
        )
        logger.info(f"Fetching Industry P/E URL: {_redact(url, self.cfg.fmp_key)}")
        data = self._get(url)
        if data and isinstance(data, list) and len(data) > 0:
            logger.info("Got Industry P/E for Industry=%s Exchange=%s Date=%s", industry_name, exchange, date_str)
            return data
        else:
            logger.warning("No Industry P/E data found for Industry=%s Exchange=%s Date=%s. Data: %s",
                        industry_name, exchange, date_str, str(data)[:200])
            return None

    def execute_all(self, symbol_param: str, current_period_back: int = 0, current_period_type: str = 'annual', current_period_back=current_period_back) -> Dict[str, Any]: # Renamed params
        t0 = time.perf_counter()
        logger.info("execute_all(%s, period_back=%s, period_type=%s) – start", symbol_param, current_period_back, current_period_type)
    
        profile = self.get_company_profile(symbol_param)
        if not profile:
            raise RuntimeError(f"Could not retrieve profile for {symbol_param}")
    
        exchange = profile.get("exchange")
        company_sector = profile.get("sector") # Renamed to avoid conflict
        company_industry = profile.get("industry") # Renamed to avoid conflict
    
        #max_workers_for_pool = max(1, DEFAULT_MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=max_workers_for_pool) as pool:
            fut_income = pool.submit(
                self.get_financial_statements,
                symbol_param, "income-statement", current_period_type, 5 + current_period_back,
            )
            fut_balance_sheet = pool.submit(
                self.get_financial_statements,
                symbol_param, "balance-sheet-statement", current_period_type, 5 + current_period_back,
            )
            fut_cash_flow = pool.submit(
                self.get_financial_statements,
                symbol_param, "cash-flow-statement", current_period_type, 5 + current_period_back,
            )
            fut_product_seg = pool.submit(
                self.get_revenue_product_segmentation, symbol_param, current_period_type
            )
            fut_geo_seg = pool.submit(
                self.get_revenue_geographic_segmentation, symbol_param, current_period_type
            )
            # fut_shares = pool.submit(self.get_outstanding_shares, symbol_param) # Potentially remove or fix
            fut_peers = pool.submit(self.process_batch, symbol_param)
    
            income_data = fut_income.result()
            if not income_data or not isinstance(income_data, list) or len(income_data) <= current_period_back:
                raise RuntimeError(f"No/Insufficient income statements for {symbol_param} (need {current_period_back+1}, got {len(income_data) if income_data else 0})")
    
            target_income_statement = income_data[current_period_back]
            filing_date = target_income_statement.get("fillingDate")
            if not filing_date: # Also check 'date' as a fallback
                filing_date = target_income_statement.get("date")
            if not filing_date:
                 raise RuntimeError(f"Missing fillingDate/date in income statement for {symbol_param} at period_back {current_period_back}")


            fut_price_now = pool.submit(
                self.get_first_historical_close_price,
                symbol_param, filing_date, self.cfg.price_look_forward,
            )
            # FMP treasury takes from/to, so using filing_date for both gets the rate for that specific day or nearest available
            fut_treasury = pool.submit(self.get_treasury_rate, filing_date, filing_date)

            fut_sector_pe = None
            if exchange and company_sector:
                fut_sector_pe = pool.submit(self._get_sector_ratio_data,
                                exchange=exchange, anchor_date=filing_date, sector_name=company_sector)
            else:
                logger.warning(f"Skipping sector PE for {symbol_param} due to missing exchange ({exchange}) or sector ({company_sector})")

            fut_industry_pe = None
            if exchange and company_industry:
                fut_industry_pe = pool.submit(self.get_industry_pe_ratio,
                                exchange=exchange, anchor_date=filing_date, industry_name=company_industry)
            else:
                logger.warning(f"Skipping industry PE for {symbol_param} due to missing exchange ({exchange}) or industry ({company_industry})")

        balance_sheet_data = fut_balance_sheet.result()
        cash_flow_data = fut_cash_flow.result()
        peers_dict = fut_peers.result() or {symbol_param: []} # Ensure it's a dict
        peers_list = peers_dict.get(symbol_param, []) # Get list of peers for the symbol

        fillingDate_price = fut_price_now.result()
        peer_metrics = self.get_peer_metrics(symbol_param, peers_list)
        
        product_seg = fut_product_seg.result() or []
        # Filter product_seg for the relevant year if possible (matching calendarYear)
        # This is complex as product_seg might not align perfectly with period_back logic directly
        # For now, take the latest available set or filter if 'date' or 'calendarYear' is in product_seg items
        # Example: if product_seg items have 'date', filter by target_income_statement['calendarYear']
        # Assuming product_seg is a list of dicts, each for a year/quarter
        
        geo_seg = fut_geo_seg.result() or []
        # shares_hist = fut_shares.result() # Potentially remove

        # Update profile with potentially more accurate/recent shares outstanding from income statement
        profile['sharesOutstanding'] = target_income_statement.get("weightedAverageShsOutDil") or \
                                       target_income_statement.get("weightedAverageShsOut")
        profile['marketCap'] = (fillingDate_price.get('close') * profile['sharesOutstanding']
                                if fillingDate_price and fillingDate_price.get('close') and profile.get('sharesOutstanding')
                                else profile.get('marketCap')) # Use calculated if possible

        treasury_rate = fut_treasury.result()
    
        sector_pe_data = fut_sector_pe.result() if fut_sector_pe else None
        industry_pe_data = fut_industry_pe.result() if fut_industry_pe else None
    
        def safe_get_first(pe_data_list): # Renamed var
            return pe_data_list[0] if pe_data_list and isinstance(pe_data_list, list) and len(pe_data_list) > 0 else None
    
        payload: Dict[str, Any] = {
            "symbol": symbol_param,
            "profile": profile, # Profile now includes more details
            "treasury_rate_10y": treasury_rate,
            "sector_pe_ratio": safe_get_first(sector_pe_data),
            "industry_pe_ratio": safe_get_first(industry_pe_data),
            "fillingDate_price": fillingDate_price, # Price at/after filling date
            "current_price_info": self.get_latest_price(symbol_param), # Add truly current price for context
            "income_statement": income_data[current_period_back:] if income_data else [],
            "balance_sheet": balance_sheet_data[current_period_back:] if balance_sheet_data else [],
            "cash_flow_statement": cash_flow_data[current_period_back:] if cash_flow_data else [], # Renamed key
            "peers": peers_list,
            "peer_metrics": peer_metrics, # Includes the main symbol as the first item
            # Filter segmentation data if possible (e.g., by calendarYear of target_income_statement)
            "product_segmentation": product_seg, # Renamed key & provide all years for now
            "geographic_segmentation": geo_seg, # Renamed key & provide all years
            # "shares_history": shares_hist, # Potentially remove
            "metadata_package": { # Add some context about this data package
                "fmp_filling_date": filing_date,
                "fmp_calendar_year": target_income_statement.get("calendarYear"),
                "fmp_period": target_income_statement.get("period"),
                "period_back_offset": current_period_back,
                "data_fetch_timestamp": dt_class.utcnow().isoformat()
            }
        }
        logger.info("execute_all(%s) – done in %.2fs", symbol_param, time.perf_counter() - t0)
        return payload

    @staticmethod
    def _cli() -> None: # pragma: no cover
        import argparse, pprint, sys
        p = argparse.ArgumentParser(description="Annual data fetcher")
        p.add_argument("symbol", help="Ticker symbol, e.g. AAPL")
        p.add_argument("--period-back", type=int, default=0, help="Years/Quarters back from most recent filing")
        p.add_argument("--period-type", type=str, default="annual", choices=["annual", "quarter"], help="Period type for financials")
        args = p.parse_args()
        try:
            fmp = FinancialDataModule()
            data = fmp.execute_all(args.symbol, args.period_back, args.period_type)
            # To save to file:
            # with open(f"{args.symbol}_fmp_data.json", "w") as f:
            #    json.dump(data, f, indent=4)
            # logger.info(f"Data saved to {args.symbol}_fmp_data.json")
            pprint.pprint(data)
        except Exception as exc:
            logger.error("CLI Error: %s", exc, exc_info=True)
            sys.exit(1)
