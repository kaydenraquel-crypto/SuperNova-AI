from __future__ import annotations
import httpx
import asyncio
import logging
from typing import List, Dict, Optional
from .config import settings
from .strategy_engine import make_df, eval_rsi_breakout

logger = logging.getLogger(__name__)

async def evaluate_alerts(watch_items: list[dict], bars_by_symbol: dict[str, list[dict]]):
    """Evaluate simple RSI breakout alerts and POST to webhooks if configured."""
    triggered = []
    for item in watch_items:
        symbol = item.get("symbol")
        bars = bars_by_symbol.get(symbol)
        if not bars:
            continue
        df = make_df(bars)
        action, conf, details = eval_rsi_breakout(df)
        if action in ("buy","sell"):
            msg = f"{symbol} potential {action.upper()} (RSI={details['rsi']:.1f})"
            alert_data = { 
                "symbol": symbol, 
                "message": msg,
                "action": action,
                "confidence": conf,
                "indicators": details,
                "profile_id": item.get("profile_id")
            }
            triggered.append(alert_data)
            
            # Send to traditional webhook if configured
            if settings.ALERT_WEBHOOK_URL:
                async with httpx.AsyncClient(timeout=10) as client:
                    try:
                        await client.post(settings.ALERT_WEBHOOK_URL, json=alert_data)
                    except Exception as e:
                        logger.warning(f"Failed to send webhook alert: {e}")
            
            # Send to NovaSignal integration
            await _send_novasignal_alert(alert_data)
            
    return triggered


async def _send_novasignal_alert(alert_data: Dict[str, any]):
    """Send alert to NovaSignal platform via connector"""
    try:
        # Import here to avoid circular imports
        from ..connectors.novasignal import get_connector, AlertPriority
        
        connector = await get_connector()
        
        # Determine alert priority based on confidence
        confidence = alert_data.get("confidence", 0.5)
        if confidence >= 0.8:
            priority = AlertPriority.HIGH
        elif confidence >= 0.6:
            priority = AlertPriority.MEDIUM
        else:
            priority = AlertPriority.LOW
        
        await connector.push_alert_to_novasignal(
            symbol=alert_data["symbol"],
            message=alert_data["message"],
            priority=priority,
            profile_id=alert_data.get("profile_id"),
            confidence=confidence,
            action=alert_data.get("action"),
            strategy="rsi_breakout",
            indicators=alert_data.get("indicators")
        )
        
        logger.info(f"Alert sent to NovaSignal: {alert_data['symbol']}")
        
    except Exception as e:
        logger.error(f"Failed to send NovaSignal alert: {e}")


async def send_custom_alert(
    symbol: str,
    message: str,
    profile_id: Optional[int] = None,
    action: Optional[str] = None,
    confidence: Optional[float] = None,
    strategy: Optional[str] = None,
    indicators: Optional[Dict[str, float]] = None,
    priority: str = "medium"
) -> bool:
    """Send custom alert to both webhook and NovaSignal"""
    alert_data = {
        "symbol": symbol,
        "message": message,
        "profile_id": profile_id,
        "action": action,
        "confidence": confidence,
        "strategy": strategy,
        "indicators": indicators
    }
    
    success = True
    
    # Send to webhook if configured
    if settings.ALERT_WEBHOOK_URL:
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                await client.post(settings.ALERT_WEBHOOK_URL, json=alert_data)
            except Exception as e:
                logger.warning(f"Failed to send webhook alert: {e}")
                success = False
    
    # Send to NovaSignal
    try:
        await _send_novasignal_alert(alert_data)
    except Exception as e:
        logger.error(f"Failed to send NovaSignal alert: {e}")
        success = False
    
    return success


async def batch_send_alerts(alerts: List[Dict[str, any]]) -> int:
    """Send multiple alerts efficiently"""
    tasks = []
    for alert in alerts:
        task = send_custom_alert(**alert)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    success_count = sum(1 for result in results if result is True)
    
    logger.info(f"Sent {success_count}/{len(alerts)} alerts successfully")
    return success_count
